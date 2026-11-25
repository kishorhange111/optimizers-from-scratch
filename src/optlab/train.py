"""Learning-rate sweep per optimizer on a byte-level GPT.

    python -m optlab.train --out results --device cuda

Every run uses the same model, data order, step count, schedule (warmup then cosine) and
weight decay. Only the optimizer and its learning rate change. Each run records its val
bits/byte curve, its optimizer-state bytes and its mean step time. Finished runs are kept in
``<out>/runs.jsonl``, so a restart resumes the sweep.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sysconfig
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .optimizers import build, cosine_lr, state_bytes

GRID = {
    "adamw": [3e-4, 1e-3, 3e-3, 1e-2],
    "lion": [3e-5, 1e-4, 3e-4, 1e-3],
    "adafactor": [3e-4, 1e-3, 3e-3, 1e-2],
    "muon": [5e-3, 1e-2, 2e-2, 4e-2],
}


class Block(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc, self.out = nn.Linear(d, 4 * d, bias=False), nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        b, l, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, l, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, l, d))
        return x + self.out(F.gelu(self.fc(self.n2(x))))


class GPT(nn.Module):
    def __init__(self, d=256, layers=4, heads=4, ctx=256, vocab=256) -> None:
        super().__init__()
        self.emb, self.pos_emb = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.emb(idx) + self.pos_emb(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.emb.weight.t()


def load_corpus():
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .py files under {root}")
    tr, va = [], []
    for f in files:
        rel = os.path.relpath(f, root).replace("\\", "/")
        with open(f, "rb") as fh:
            (va if zlib.crc32(rel.encode()) % 1000 < 50 else tr).append(fh.read())
    return (
        np.frombuffer(b"\n".join(tr), np.uint8).copy(),
        np.frombuffer(b"\n".join(va), np.uint8).copy(),
    )


def _bf16(device: str) -> bool:
    """bf16 autocast on Ampere and newer; plain fp32 on older GPUs (a T4 has no bf16 units)."""
    return device == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def batch(data, bs, ctx, rng, device):
    ix = rng.integers(0, len(data) - ctx - 1, bs)
    x = torch.from_numpy(np.stack([data[i : i + ctx + 1] for i in ix]).astype(np.int64))
    return x[:, :-1].to(device), x[:, 1:].to(device)


@torch.no_grad()
def val_bpb(model, val, ctx, device, n=16, bs=16):
    rng = np.random.default_rng(1234)
    model.eval()
    losses = []
    for _ in range(n // bs or 1):
        x, y = batch(val, bs, ctx, rng, device)
        with torch.autocast("cuda", torch.bfloat16, enabled=_bf16(device)):
            losses.append(F.cross_entropy(model(x).float().flatten(0, 1), y.flatten()).item())
    model.train()
    return float(np.mean(losses)) / math.log(2)


def run(
    name: str,
    lr: float,
    data,
    steps: int,
    device: str,
    bs=32,
    ctx=256,
    wd=0.1,
    model_kw=None,
    seed=0,
    evals=10,
) -> dict:
    train, val = data
    torch.manual_seed(seed)
    model = GPT(ctx=ctx, **(model_kw or {})).to(device)
    opt = build(name, model, lr, weight_decay=wd)
    groups = opt.param_groups + (opt.adam.param_groups if hasattr(opt, "adam") else [])
    for g in groups:
        g["base_lr"] = g["lr"]
    rng = np.random.default_rng(seed)
    hist, times = [], []
    diverged = False
    for step in range(1, steps + 1):
        f = cosine_lr(step, steps, max(1, steps // 20))
        for g in groups:
            g["lr"] = g["base_lr"] * f
        x, y = batch(train, bs, ctx, rng, device)
        t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16, enabled=_bf16(device)):
            loss = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        if step % max(1, steps // evals) == 0 or step == steps:
            hist.append(
                {"step": step, "train": loss.item(), "val_bpb": val_bpb(model, val, ctx, device)}
            )
    n_par = sum(p.numel() for p in model.parameters())
    return {
        "optimizer": name,
        "lr": lr,
        "steps": steps,
        "diverged": diverged,
        "final_val_bpb": hist[-1]["val_bpb"] if hist and not diverged else None,
        "params": n_par,
        "state_bytes": state_bytes(opt),
        "state_per_param_bytes": state_bytes(opt) / n_par,
        "ms_per_step": 1000 * float(np.median(times[5:] or times)),
        "history": hist,
    }


def report(recs, out, gpu):
    lines = [
        f"GPU **{gpu}**, {recs[0]['params']:,} params, {recs[0]['steps']} steps, "
        "same data order, schedule and weight decay (0.1) for every run.\n",
        "Final val bits/byte by learning rate (lower is better; ✗ = diverged):\n",
    ]
    for name in GRID:
        rs = sorted((r for r in recs if r["optimizer"] == name), key=lambda r: r["lr"])
        if not rs:
            continue
        cells = [
            f"{r['lr']:g}: " + ("✗" if r["final_val_bpb"] is None else f"{r['final_val_bpb']:.3f}")
            for r in rs
        ]
        lines.append(f"- **{name}**: " + ", ".join(cells))
    lines += [
        "",
        "| optimizer | best lr | best val bpb | lr range within 0.05 of best "
        "| state bytes / param | ms / step |",
        "|---|---|---|---|---|---|",
    ]
    for name in GRID:
        rs = [r for r in recs if r["optimizer"] == name and r["final_val_bpb"] is not None]
        if not rs:
            continue
        best = min(rs, key=lambda r: r["final_val_bpb"])
        near = [r["lr"] for r in rs if r["final_val_bpb"] <= best["final_val_bpb"] + 0.05]
        lines.append(
            f"| {name} | {best['lr']:g} | {best['final_val_bpb']:.3f} | "
            f"{min(near):g}–{max(near):g} ({len(near)}/{len(rs)}) | "
            f"{best['state_per_param_bytes']:.2f} | {best['ms_per_step']:.1f} |"
        )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--optimizers", nargs="+", default=list(GRID))
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "runs.jsonl")
    recs = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    done = {(r["optimizer"], r["lr"]) for r in recs}
    data = load_corpus()
    kw = dict(model_kw={"d": 32, "layers": 1, "heads": 2}, bs=2, ctx=32) if a.tiny else {}
    for name in a.optimizers:
        for lr in GRID[name][: 1 if a.tiny else None]:
            if (name, lr) in done:
                continue
            r = run(name, lr, data, a.steps, a.device, **kw)
            with open(path, "a") as f:
                f.write(json.dumps(r) + "\n")
            recs.append(r)
            print(
                f"[{time.strftime('%H:%M:%S')}] {name} lr={lr:g} "
                f"val_bpb={r['final_val_bpb']} {r['ms_per_step']:.1f} ms/step",
                flush=True,
            )
    gpu = torch.cuda.get_device_name() if a.device == "cuda" else "cpu"
    print(report(recs, a.out, gpu))


if __name__ == "__main__":
    main()
