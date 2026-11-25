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
