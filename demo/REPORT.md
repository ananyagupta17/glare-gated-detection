# Glare-gated detection: demo results

> **DEMO, not final.** These numbers come from v2 synthetic data that **failed its
> shortcut test** (darkest-pixel p1 AUC 0.998; min AUC 0.999; noise-level AUC 0.724; artifact-only classifier 83.7%). The gate rows are the most affected: a gate can
> use those artifacts. Final numbers wait for a synthesis that passes.

## Setup
- **Data:** KITTI val, 1,491 frames, each shown clean **and** glared
  (2,982 images, exactly 50/50). Glare: v2 camera-model synthesis,
  four types, continuous severity. 5 frames used to tune the restorer are excluded.
- **Detector:** YOLOv8n trained on clean KITTI train only (`models/yolo_c1_train.pt`, 50 epochs).
- **Restorer:** daytime trained U-Net (src/restore_daytime_net.py). **70 ms/frame on CPU.**
- **Gates (C3),** fit on v2 *train* only: an upper-third brightness threshold, and a
  logistic regression on Harshitha's 9 gate features (standing in for her v2 gate).

## Headline
| Arm | mAP@50 stream | clean frames | glared frames | glared, severe | Restores | Latency ms/frame (CPU) |
|---|---|---|---|---|---|---|
| C1 · never restore | 0.638 | 0.850 | 0.434 | 0.248 | 0.0% | 17 |
| C2 · always restore | 0.778 | 0.849 | 0.714 | 0.635 | 100.0% | 87 |
| C3 · gate: brightness | 0.754 | 0.850 | 0.666 | 0.605 | 56.7% | 57 |
| C3 · gate: features | 0.772 | 0.849 | 0.703 | 0.632 | 47.3% | 52 |

- **Glare costs the detector 0.415 mAP@50**
  (0.850 → 0.434); on severe frames
  0.865 → 0.248.
- **Gating:** the best C3 (C3 · gate: features) runs at 52 ms/frame against C2's
  87 ms/frame (1.7× cheaper), restoring 47%
  of frames.

![accuracy vs latency](08_chart_accuracy_vs_latency.png)
![mAP per arm](07_chart_map_per_arm.png)

## Restoration recovers most of the glare loss
| Glared frames | n | Clean | Glared | Restored | Share of loss recovered |
|---|---|---|---|---|---|
| All | 1491 | 0.850 | 0.434 | 0.714 | 67% |
| Severe | 640 | 0.865 | 0.248 | 0.634 | 63% |
| Non-severe | 851 | 0.841 | 0.581 | 0.776 | 75% |

**Restoring frames without glare:** over all 1,491 clean val frames, always
restoring changed mAP@50 from 0.850 to 0.849
(-0.001). That is close to free, so here the gate mainly saves time.

## Restorers compared (same frames, same detector, same gates)
| Restorer | ms/frame (CPU) | Glared mAP@50 restored | Share of loss recovered | Clean-frame change (C2−C1) | C2 stream mAP@50 | C3 features stream mAP@50 | C3 features ms/frame |
|---|---|---|---|---|---|---|---|
| Flare7K++ night model, whole-frame (src/restore_seamless.py) | 1,850 | 0.734 | 72% | -0.002 | 0.788 | 0.782 | 894 |
| daytime classical veil subtraction (src/restore_daytime.py) | 20 | 0.602 | 40% | -0.126 | 0.662 | 0.722 | 26 |
| daytime trained U-Net (src/restore_daytime_net.py) | 70 | 0.714 | 67% | -0.001 | 0.778 | 0.772 | 52 |

![recovery by severity](09_chart_recovery_by_severity.png)

## Latency breakdown (mean ms per frame, CPU, warm-up excluded)
| Arm | Gate | Detection | Restoration × trigger rate | Total |
|---|---|---|---|---|
| C1 · never restore | 0.00 | 17.4 | 0 | **17** |
| C2 · always restore | 0.00 | 17.4 | 70 | **87** |
| C3 · gate: brightness | 0.12 | 17.4 | 40 | **57** |
| C3 · gate: features | 1.14 | 17.4 | 33 | **52** |

Restoration is ~4× the cost of detection,
which is what makes skipping it worthwhile.

## Gate quality (val, gate protocol: glared veiling_wash frames with s < 0.1752 not scored)
| Gate | Precision | Recall | F1 | Accuracy | Trigger rate |
|---|---|---|---|---|---|
| C3 · gate: brightness | 0.616 | 0.717 | 0.663 | 0.643 | 56.7% |
| C3 · gate: features | 0.916 | 0.882 | 0.899 | 0.903 | 47.3% |

The features gate is likely inflated by the synthesis artifacts.

## Figures in this folder
| File | Shows |
|---|---|
| `01_pipeline_walkthrough_007004.png` | One frame through the pipeline: clean → glared (with both gate decisions) → restored, with detections |
| `02_glare_types_glared_vs_restored.png` | A typical severe frame of each glare type, before and after restoration |
| `03_severity_ramp_*.png` | Each glare type at lowest, median and highest severity |
| `04_gate_decisions.png` | The features gate: a correct fire, a correct skip, a false alarm and a miss |
| `05_restoring_a_clean_frame.png` | The cost side: restoration applied to a frame with no glare |
| `06_failure_case.png` | Where restoration cannot help (blown-out glare) |
| `07–09_chart_*.png` | The charts above |

**How frames were chosen (no cherry-picking):** per glare type, the severe frame with the
*median* detection gain from restoration; the walkthrough is the median over all types;
the failure case is the lowest F1 after restoration; gate examples are the median gate
score within each outcome. Over the 640 severe frames, the median per-frame F1
gain from restoration was +0.214.
Box colours: green = correct detection, red = false detection, yellow = missed object.

## Caveats
1. **Synthetic glare that failed its shortcut test.** Restoration may suit this additive,
   smooth glare better than real glare; confirm on real glare (e.g. BDD100K) before quoting.
2. **The C3 gates are stand-ins,** fit here on v2 train; Harshitha's v2 gate is not trained yet.
3. **The daytime restorer was tuned or trained on our synthetic glare only** (train split); real-glare performance is untested.
4. KITTI's train/val split shares drives, so absolute mAP is slightly optimistic; the
   comparisons between arms use identical frames and are unaffected.
