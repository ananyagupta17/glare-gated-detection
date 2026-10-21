# C1 / C2 / C3 on v2 val (DEMO)

**Not final:** v2 failed its shortcut test (`results/shortcut_test.json`). Gate rows are the most affected.

Stream: 1491 val frames, each clean and glared (2982 images, 50/50). Detector `models/yolo_c1_train.pt`. Restorer: whole-frame Flare7K++, 1850 ms/frame on CPU.

| Arm | mAP50 stream | mAP50 clean | mAP50 glared | mAP50 glared severe | Trigger rate | Latency ms/frame (CPU) |
|---|---|---|---|---|---|---|
| C1 | 0.638 | 0.850 | 0.434 | 0.248 | 0.0% | 17 |
| C2 | 0.788 | 0.848 | 0.734 | 0.667 | 100.0% | 1867 |
| C3_brightness | 0.765 | 0.850 | 0.687 | 0.635 | 56.7% | 1066 |
| C3_features | 0.782 | 0.849 | 0.722 | 0.663 | 47.3% | 894 |

| Gate | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| C3_brightness | 0.616 | 0.717 | 0.663 | 0.643 |
| C3_features | 0.916 | 0.882 | 0.899 | 0.903 |
