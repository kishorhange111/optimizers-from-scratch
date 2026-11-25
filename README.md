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

**Results are pending** from the Colab run.
