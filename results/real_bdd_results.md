# Real glare (BDD100K): final pipeline

72 real-glare frames + 72 matched clean frames; nothing trained on BDD. Device mps.

| Arm | mAP@50 stream | mAP@50 glare frames | mAP@50 clean frames | Glare loss recovered | Restores | ms/frame mean / p95 | gate / restoration / detection ms | Real-time (30 fps, p95) |
|---|---|---|---|---|---|---|---|---|
| C1 · never restore | 0.076 | 0.035 | 0.124 | – | 0% | 15 / 20 | 0.0 / 0.0 / 14.6 | yes |
| C2 · always restore | 0.082 | 0.044 | 0.129 | 10% | 100% | 45 / 58 | 0.0 / 30.7 / 14.6 | 10 Hz only |
| C3 · gated (CNN gate) | 0.079 | 0.041 | 0.124 | 7% | 47% | 41 / 66 | 11.7 / 14.5 / 14.6 | 10 Hz only |

Glare drop (C1, clean-set minus glare-set mAP@50, different scenes): 0.088.

Gate on real frames: accuracy 0.972, precision 1.000, recall 0.944, F1 0.971 (TP 68, FN 4, FP 0, TN 72).

Latency per stage (mean): gate 11.7 ms, restoration 30.7 ms, detection 14.6 ms.
