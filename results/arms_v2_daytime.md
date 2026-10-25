# C1 / C2 / C3 on v2 val (DEMO)

**Not final:** v2 failed its shortcut test (`results/shortcut_test.json`). Gate rows are the most affected.

Stream: 1491 val frames, each clean and glared (2982 images, 50/50). Detector `models/yolo_c1_train.pt`. Restorer: daytime classical veil subtraction (src/restore_daytime.py), 20 ms/frame on CPU.

| Arm | mAP50 stream | mAP50 clean | mAP50 glared | mAP50 glared severe | Trigger rate | Latency ms/frame (CPU) |
|---|---|---|---|---|---|---|
| C1 | 0.638 | 0.850 | 0.434 | 0.248 | 0.0% | 16 |
| C2 | 0.662 | 0.724 | 0.602 | 0.513 | 100.0% | 36 |
| C3_brightness | 0.688 | 0.792 | 0.586 | 0.494 | 56.7% | 27 |
| C3_features | 0.722 | 0.840 | 0.602 | 0.510 | 47.3% | 26 |

| Gate | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| C3_brightness | 0.616 | 0.717 | 0.663 | 0.643 |
| C3_features | 0.916 | 0.882 | 0.899 | 0.903 |
