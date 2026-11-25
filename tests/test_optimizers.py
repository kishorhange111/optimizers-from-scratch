import math

import pytest
import torch

from optlab.optimizers import (
    Adafactor,
    AdamW,
    Lion,
    Muon,
    build,
    cosine_lr,
    newton_schulz,
    state_bytes,
)
from optlab.train import GPT, main


def _problem(seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(12, 8, generator=g, dtype=torch.float64)
    y = torch.randn(12, 5, generator=g, dtype=torch.float64)
    W = torch.randn(8, 5, generator=g, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    return A, y, W, b


def _loss(A, y, W, b):
    return ((A @ W + b - y) ** 2).mean()


def _steps(opt, A, y, W, b, n):
    for _ in range(n):
        opt.zero_grad()
        _loss(A, y, W, b).backward()
        opt.step()


def test_adamw_matches_torch():
    A, y, W, b = _problem()
    W2, b2 = W.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    _steps(AdamW([W, b], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1), A, y, W, b, 20)
    ref = torch.optim.AdamW([W2, b2], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    _steps(ref, A, y, W2, b2, 20)
    torch.testing.assert_close(W, W2)
    torch.testing.assert_close(b, b2)


def test_lion_moves_every_coordinate_by_exactly_lr():
    A, y, W, b = _problem()
    W0 = W.detach().clone()
    opt = Lion([W], lr=1e-3, weight_decay=0.0)
    _loss(A, y, W, b).backward()
    g = W.grad.clone()
    opt.step()
    torch.testing.assert_close(W - W0, -1e-3 * torch.sign(g))
    torch.testing.assert_close(opt.state[W]["m"], 0.01 * g)  # m ← β₂m + (1−β₂)g from zero


def test_lion_uses_interpolated_momentum_for_the_sign():
    W = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = Lion([W], lr=1.0, betas=(0.9, 0.99))
    opt.state[W]["m"] = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
    W.grad = torch.tensor([-5.0, -5.0, 20.0], dtype=torch.float64)
    opt.step()
    # 0.9·m + 0.1·g = [0.4, −1.4, 2.9]: the first sign disagrees with the gradient
    torch.testing.assert_close(W.detach(), torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float64))


def test_adafactor_factorisation_is_exact_for_rank_one_second_moments():
    a, c = torch.rand(6, dtype=torch.float64) + 0.5, torch.rand(4, dtype=torch.float64) + 0.5
    G = torch.outer(a, c).sqrt()  # G² = a cᵀ, rank one
    W = torch.zeros(6, 4, dtype=torch.float64, requires_grad=True)
    opt = Adafactor([W], lr=1.0, eps=0.0, clip=1e9)
    W.grad = G
    opt.step()  # first step: β₂ = 0, so the state is exactly this gradient's statistics
    st = opt.state[W]
    assert st["r"].shape == (6,) and st["c"].shape == (4,)
    torch.testing.assert_close(Adafactor.factored_v(st["r"], st["c"]), G**2)
    torch.testing.assert_close(W.detach(), -torch.ones(6, 4, dtype=torch.float64))  # g/√g²


def test_adafactor_clips_update_rms():
    W = torch.zeros(10, dtype=torch.float64, requires_grad=True)
    opt = Adafactor([W], lr=1.0, clip=0.5)
    W.grad = torch.randn(10, dtype=torch.float64)
    opt.step()
    assert W.detach().pow(2).mean().sqrt() == pytest.approx(0.5)


@pytest.mark.parametrize("shape", [(32, 32), (64, 16), (16, 64)])
def test_newton_schulz_approximately_orthogonalises(shape):
    G = torch.randn(*shape, dtype=torch.float64)
    O = newton_schulz(G)
    s = torch.linalg.svdvals(O)
    assert O.shape == G.shape and s.min() > 0.6 and s.max() < 1.25
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    cos = (O * (U @ Vh)).sum() / (O.norm() * (U @ Vh).norm())
    assert cos > 0.95


def test_muon_routes_parameters():
    m = GPT(d=32, layers=1, heads=2, ctx=16)
    opt = build("muon", m, 0.02)
    muon_ids = {id(p) for p in opt.param_groups[0]["params"]}
    assert id(m.blocks[0].qkv.weight) in muon_ids
    assert id(m.emb.weight) not in muon_ids and id(m.norm.weight) not in muon_ids
    with pytest.raises(ValueError):
        Muon([torch.zeros(3, requires_grad=True)], [])


@pytest.mark.parametrize("name", ["adamw", "lion", "adafactor", "muon"])
def test_all_reduce_loss_and_report_state(name):
    torch.manual_seed(0)
    m = GPT(d=32, layers=1, heads=2, ctx=16)
    lr = {"adamw": 3e-3, "lion": 1e-3, "adafactor": 3e-3, "muon": 0.02}[name]
    opt = build(name, m, lr)
    x = torch.randint(0, 256, (8, 16))
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.cross_entropy(m(x).flatten(0, 1), x.roll(-1, 1).flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5
    n = sum(p.numel() for p in m.parameters())
    per = state_bytes(opt) / n
    expected = {
        "adamw": (7.9, 8.1),
        "lion": (3.9, 4.1),
        "adafactor": (0.0, 1.0),
        "muon": (3.9, 8.1),
    }
    lo, hi = expected[name]
    assert lo <= per <= hi


def test_cosine_schedule():
    assert cosine_lr(1, 100, 10) == pytest.approx(0.1, rel=1e-3)
    assert cosine_lr(10, 100, 10) == pytest.approx(0.5 * (1 + math.cos(0.1 * math.pi)), rel=1e-9)
    assert cosine_lr(100, 100, 10) == 0.02


def test_tiny_sweep_runs_and_resumes(tmp_path):
    main(["--tiny", "--device", "cpu", "--out", str(tmp_path), "--steps", "4"])
    lines = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert "best lr" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    main(["--tiny", "--device", "cpu", "--out", str(tmp_path), "--steps", "4"])
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 4
