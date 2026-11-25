"""Four optimizers from their papers, written as plain ``torch.optim.Optimizer`` subclasses.

| optimizer | state per (m×n) matrix | update |
|---|---|---|
| AdamW (Loshchilov & Hutter 2019) | 2·m·n | ``m̂ / (√v̂ + ε)`` with decoupled weight decay |
| Lion (Chen et al. 2023) | m·n | ``sign(β₁m + (1−β₁)g)``: every coordinate moves by exactly ±lr |
| Adafactor (Shazeer & Stern 2018) | m + n | ``g / √V̂``, rank-1 ``V̂ = R Cᵀ / mean(R)``, RMS clip |
| Muon (Jordan et al. 2024) | m·n | momentum, orthogonalised by 5 Newton–Schulz steps (≈ U Vᵀ) |

Muon only makes sense for 2-D hidden weights. Embeddings, the output head and gains go to an
inner AdamW, as in the reference implementation.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["t"] += 1
                t, m, v = st["t"], st["m"], st["v"]
                m.lerp_(p.grad, 1 - b1)
                v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                p.mul_(1 - g["lr"] * g["weight_decay"])
                denom = (v / (1 - b2**t)).sqrt_().add_(g["eps"])
                p.addcdiv_(m, denom, value=-g["lr"] / (1 - b1**t))


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(torch.sign(m * b1 + p.grad * (1 - b1)), alpha=-g["lr"])
                m.lerp_(p.grad, 1 - b2)


class Adafactor(Optimizer):
    """Without momentum (β₁ = 0) and with an explicit lr, the setting used for T5 fine-tuning
    at a fixed rate. β₂ follows the paper's schedule ``1 − t^(−0.8)``."""

    def __init__(self, params, lr=1e-3, eps=1e-30, clip=1.0, decay=0.8, weight_decay=0.0):
        super().__init__(
            params, dict(lr=lr, eps=eps, clip=clip, decay=decay, weight_decay=weight_decay)
        )

    @staticmethod
    def factored_v(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return torch.outer(r, c) / r.mean()

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad, st = p.grad, self.state[p]
                factored = grad.ndim == 2
                if not st:
                    st["t"] = 0
                    if factored:
                        st["r"] = grad.new_zeros(grad.shape[0])
                        st["c"] = grad.new_zeros(grad.shape[1])
                    else:
                        st["v"] = torch.zeros_like(grad)
                st["t"] += 1
                b2 = 1.0 - st["t"] ** (-g["decay"])
                sq = grad * grad + g["eps"]
                if factored:
                    st["r"].mul_(b2).add_(sq.mean(1), alpha=1 - b2)
                    st["c"].mul_(b2).add_(sq.mean(0), alpha=1 - b2)
                    v = self.factored_v(st["r"], st["c"])
                else:
                    st["v"].mul_(b2).add_(sq, alpha=1 - b2)
                    v = st["v"]
                u = grad / v.sqrt()
                u.div_((u.pow(2).mean().sqrt() / g["clip"]).clamp_(min=1.0))  # RMS clipping
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(u, alpha=-g["lr"])


NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise G (≈ U Vᵀ of G = U S Vᵀ) with a quintic iteration.

    The coefficients are tuned for speed of convergence rather than exactness: singular
    values end up in roughly [0.7, 1.2] instead of exactly 1, which works just as well for
    training (Jordan et al. 2024).
    """
    a, b, c = NS_COEFFS
    X = G / (G.norm() + eps)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.t()
    for _ in range(steps):
        A = X @ X.t()
        X = a * X + (b * A + c * A @ A) @ X
    return X.t() if tall else X


class Muon(Optimizer):
    """Muon for 2-D hidden weights plus an inner AdamW for everything else."""

    def __init__(
        self,
        muon_params,
        adam_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
        adam_lr=3e-3,
        adam_betas=(0.9, 0.95),
        adam_wd=0.0,
    ):
        muon_params, adam_params = list(muon_params), list(adam_params)
        if any(p.ndim != 2 for p in muon_params):
            raise ValueError("Muon parameters must be 2-D matrices")
        super().__init__(
            muon_params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                weight_decay=weight_decay,
            ),
        )
        self.adam = AdamW(adam_params, lr=adam_lr, betas=adam_betas, weight_decay=adam_wd)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            mu = g["momentum"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"].mul_(mu).add_(p.grad)
                d = p.grad.add(buf, alpha=mu) if g["nesterov"] else buf
                o = newton_schulz(d.float(), g["ns_steps"]).to(p.dtype)
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(o, alpha=-g["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)
        self.adam.step()

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none)
        self.adam.zero_grad(set_to_none)


def state_bytes(opt: Optimizer) -> int:
    """Bytes of tensor state held by an optimizer (and Muon's inner AdamW)."""
    total = sum(
        t.numel() * t.element_size()
        for st in opt.state.values()
        for t in st.values()
        if torch.is_tensor(t)
    )
    return total + (state_bytes(opt.adam) if isinstance(opt, Muon) else 0)


def build(name: str, model: torch.nn.Module, lr: float, weight_decay: float = 0.0) -> Optimizer:
    """Optimizer by name. For Muon, ``lr`` is Muon's rate; embeddings/head/gains use AdamW."""
    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    if name == "lion":
        return Lion(model.parameters(), lr=lr, weight_decay=weight_decay * 10)
    if name == "adafactor":
        return Adafactor(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "muon":
        hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and "emb" not in n]
        ids = {id(p) for p in hidden}
        rest = [p for p in model.parameters() if id(p) not in ids]
        return Muon(hidden, rest, lr=lr, weight_decay=weight_decay, adam_wd=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def cosine_lr(step: int, total: int, warmup: int) -> float:
    return min(1.0, step / warmup) * max(0.02, 0.5 * (1 + math.cos(math.pi * step / total)))
