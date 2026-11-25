# optimizers-from-scratch

[![CI](https://github.com/kishorhange111/optimizers-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/optimizers-from-scratch/actions/workflows/ci.yml)

AdamW, Lion, Adafactor and Muon, implemented from their papers as plain
`torch.optim.Optimizer` subclasses. Each one is checked against the property that defines
it, then all four are compared on a small GPT under the same conditions.

## The optimizers ([`optimizers.py`](src/optlab/optimizers.py))

| optimizer | state per m×n matrix | update | test that pins it down |
|---|---|---|---|
| **AdamW** | 2·mn | `m̂ / (√v̂ + ε)`, decoupled weight decay | matches `torch.optim.AdamW` after 20 steps (float64) |
| **Lion** | mn | `sign(β₁m + (1−β₁)g)`: every coordinate moves by exactly ±lr | step size is exactly lr; the sign comes from the interpolated momentum, not the raw gradient |
| **Adafactor** | m + n | `g / √V̂`, with `V̂ = R Cᵀ / mean(R)` from row and column means of g², plus RMS clipping | the factorisation is exact when g² is rank one; the update RMS is clipped to the threshold |
| **Muon** | mn | Nesterov momentum, then orthogonalised by 5 quintic Newton–Schulz steps (≈ U Vᵀ of its SVD), scaled by √max(1, m/n) | singular values land in [0.6, 1.25] and the result points along U Vᵀ (cos > 0.95), for square, tall and wide matrices |

Muon applies only to 2-D hidden weights. Embeddings, positions and LayerNorm gains go to an
inner AdamW, as in the reference implementation; a test checks the routing.

### Why Muon works

The momentum matrix of a transformer weight is typically dominated by a few directions.
Orthogonalising it sets all its singular values to about 1. This boosts the rare directions
that plain SGD or Adam would barely move, so every direction gets an update of the same
scale. Newton–Schulz does this with five matmuls instead of an SVD. The coefficients
`(3.4445, −4.7750, 2.0315)` trade exactness for speed: singular values only reach about
[0.7, 1.2].

## Experiment ([`train.py`](src/optlab/train.py))

The model is a 3.3M-parameter byte-level GPT (4 layers, d = 256, context 256) on Python
standard-library source. Every run shares the same:
- data order;
- 1,500 steps of batch 32;
- 5% warmup then cosine schedule;
- weight decay 0.1 (×10 for Lion, since its lr is about 10× smaller);
- gradient clipping at 1.

Each optimizer gets a 4-point learning-rate grid around its usual range. For each run the
script reports final val bits/byte, how wide the near-best lr range is (sensitivity), the
optimizer-state bytes per parameter, and the median step time.

```
pytest -q                                           # 15 tests, CPU
python -m optlab.train --out results --device cuda  # resumable sweep (runs.jsonl)
```

## Results (Tesla T4, fp32, one seed)

GPU **Tesla T4**, 3,281,408 params, 1500 steps, same data order, schedule and weight decay (0.1) for every run.

Final val bits/byte by learning rate (lower is better; ✗ = diverged):

- **adamw**: 0.0003: 2.405, 0.001: 1.746, 0.003: 1.503, 0.01: 3.126
- **lion**: 3e-05: 3.251, 0.0001: 2.522, 0.0003: 2.245, 0.001: 3.033
- **adafactor**: 0.0003: 2.550, 0.001: 2.067, 0.003: 2.671, 0.01: 3.524
- **muon**: 0.005: 1.372, 0.01: 1.302, 0.02: 1.248

| optimizer | best lr | best val bpb | lr range within 0.05 of best | state bytes / param | ms / step |
|---|---|---|---|---|---|
| adamw | 0.003 | 1.503 | 0.003–0.003 (1/4) | 8.00 | 74.7 |
| lion | 0.0003 | 2.245 | 0.0003–0.0003 (1/4) | 4.00 | 75.0 |
| adafactor | 0.001 | 2.067 | 0.001–0.001 (1/4) | 0.03 | 75.5 |
| muon | 0.02 | 1.248 | 0.02–0.02 (1/3) | 4.17 | 84.5 |

The Muon lr = 0.04 run did not fit in the Colab session and is still pending. Muon's best lr
is at the top of its grid, so its optimum may be higher still.

What it shows:
- **Muon reaches the lowest loss, at every lr tried.** Its *worst* setting (lr 0.005,
  1.37 bits/byte) beats AdamW's best (1.50). At its best it is 0.25 bits/byte (17%) ahead,
  for 4.2 bytes of state per parameter instead of 8, and about 13% more time per step (five
  Newton–Schulz iterations per matrix). This matches the Muon reports: orthogonalised
  updates give every direction of each weight matrix a useful step, where Adam's per-entry
  scaling lets a few dominant directions take most of it.
- **AdamW, Lion and Adafactor are sharply lr-sensitive here.** Each has exactly one grid
  point within 0.05 of its best, and a 3× step either way costs 0.24–1.6 bits/byte.
- **Lion and Adafactor trail AdamW at this scale and budget** (2.25 and 2.07 against 1.50).
  Lion's fixed-magnitude sign update suits large batches; at batch 32 it is noisy. Adafactor
  ran without momentum here, as in its memory-saving T5 setting. That makes it nearly
  stateless (0.03 bytes per parameter, 270× less than AdamW), and the cost shows in the
  loss.

**Caveats:**
- one seed;
- one 3.3M-parameter model;
- 1,500 steps;
- a 4-point grid per optimizer.

Lion is usually tuned with a larger batch and Adafactor with momentum, so these numbers
compare the optimizers as configured, not at their best possible settings.
