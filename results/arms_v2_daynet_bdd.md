# C1 / C2 / C3 on v2 val (DEMO)

**Not final:** v2 failed its shortcut test (`results/shortcut_test.json`). Gate rows are the most affected.

Stream: 1491 val frames, each clean and glared (2982 images, 50/50). Detector `models/yolo_c1_train.pt`. Restorer: daytime U-Net trained on KITTI + BDD pairs (models/restore_daytime_net_kitti_bdd.pt), 70 ms/frame on CPU.

| Arm | mAP50 stream | mAP50 clean | mAP50 glared | mAP50 glared severe | Trigger rate | Latency ms/frame (CPU) |
|---|---|---|---|---|---|---|
| C1 | 0.638 | 0.850 | 0.434 | 0.248 | 0.0% | 15 |
| C2 | 0.781 | 0.846 | 0.725 | 0.663 | 100.0% | 85 |
| C3_brightness | 0.758 | 0.847 | 0.675 | 0.632 | 56.7% | 54 |
| C3_features | 0.777 | 0.849 | 0.713 | 0.659 | 47.3% | 49 |

| Gate | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| C3_brightness | 0.616 | 0.717 | 0.663 | 0.643 |
| C3_features | 0.916 | 0.882 | 0.899 | 0.903 |
