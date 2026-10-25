# Values for the manuscript (CGR_MLWA_manuscript.pdf)

Every number comes from a committed results file (named in brackets). KITTI = v2 synthetic
glare, 1,491 held-out pairs; stream = 2,982 images (50/50). Detector `models/yolo_c1_train.pt`,
restorer `models/restore_daytime_net.pt`, CNN gate `models/gate_cnn_mixed.pt`.

## ⚠️ Read first: four things in the draft that the results contradict

1. **The CNN gate does not save time on our CPU.** It costs ~34 ms per frame on the Apple M3
   CPU (4 threads), against 70 ms for the restorer, so C3-CNN averages **98.6 ms vs 88.3 ms
   for always-on (C2)**. The draft's "the gate's job is to save time" holds on CPU only for the
   feature gate (54.9 ms). On the M3 GPU (live demo) the CNN gate takes ~10 ms.
   Harshitha measured 11.7 ms on her CPU (`results/gate_cnn_mixed_metrics.json`). [arms_cnn_latency.json]
2. **§4.1 "reviewed by a person" is not accurate.** The BDD glare labels are Claude's first-pass
   review, accepted by Ananya without an independent re-check
   (`data/bdd_review/review_meta.json`, `label_decision`).
3. **§5.2 classical restorer:** measured **20 ms** (not 35 ms) and clean-frame ΔmAP@50
   **−0.126** (not −0.136). [arms_v2_daytime.json]
4. **The CNN gate was not retrained on KITTI train.** Instead, C3-CNN is also reported on the
   stream with the 521 overlapping val IDs removed (970 pairs). Results barely change. Table 6
   (artifact probe) and the counter-shortcut gate come from Harshitha; they are not in this repo.

## Abstract / Conclusion
- [share]: **91%** of C2's gain kept by the CNN gate (90% excluding its training scenes);
  feature gate 95%, brightness gate 83%. [arms_cnn_latency.json]
- [mean / p95 ms]: CNN gate **98.6 / 147.6 ms** (CPU). For comparison C2 88.3 / 94.0, C1 16.3 / 17.8.

## §4.1 Data, Real
- [n] matched clean frames: **72**
- [matching rule]: daytime/dawn-dusk BDD100K val frames from the lower half of the glare-score
  ranking, matched to the glare set on BDD's weather × scene × time-of-day attributes, seeded (42).
- [search procedure]: 6,036 daytime and dawn/dusk val frames were ranked three ways: a
  saturated-region/veil heuristic (top 100), saturated-blob size (top 60) and the CNN gate
  (top 200), giving 338 pooled candidates, reviewed blind on shuffled contact sheets.
- [n] reviewed: **338** (by Claude, first pass; see point 2 above). 72 confirmed glare, 14 of them marked uncertain.
- "68 of the 72 came from gate-ranked candidates": correct.

## §3.4 Restorer
- [architecture, loss, input size]: 4-level U-Net, 16 base channels, **0.48 M parameters**;
  input resized to **624 × 192**; predicts an additive correction that is upsampled and added to
  the full-resolution frame; **L1 loss**; AdamW, lr 2×10⁻³, OneCycle, batch 8, 15 epochs; best
  epoch on a 5% train holdout. [restore_daytime_net_training.json]

## §4.2 Setup
- [CPU model, threads, timed frames]: **Apple M3 CPU, 4 PyTorch threads, 300 timed frames after
  20 warm-up, image pre-decoded.** mAP scored on the M3 GPU (MPS).
- [bootstrap 95% intervals]: **not computed.** Either remove the sentence or ask for them to be run.

## §5.1 / Table 2 (restorer on held-out pairs) [restorer_map_daynet.json]
| Metric | Subset | Clean | Glared | Restored | ρ |
|---|---|---|---|---|---|
| mAP@50 | All (1,491) | 0.850 | **0.434** | 0.714 | 67% |
| mAP@50 | Non-severe (851) | 0.841 | 0.581 | 0.776 | 75% |
| mAP@50 | Severe (640) | 0.865 | 0.248 | 0.634 | 63% |
| mAP@50:95 | All | 0.596 | 0.269 | 0.477 | 64% |
Restoring clean frames (C2 vs C1, all 1,491): ΔmAP@50 **−0.001** (the draft's "−0.001 to −0.004" can become "−0.001").
Text [glared] = **0.434**.

## Table 3 (restorers) [arms_v2_daytime.json, arms_v2.json, arms_v2_daynet.json]
| Restorer | Latency (CPU) | ρ | Clean ΔmAP@50 |
|---|---|---|---|
| Classical (dark-channel veil subtraction, tuned on train) | **20 ms** | **40%** | **−0.126** |
| Flare7K++ | 1.85 s | 72% | −0.002 (all 1,491; −0.003 on the 200 subset) |
| Daytime U-Net | 70 ms | 67% | −0.001 |
Text "26 times faster": 1,850 / 70 = **26×** ✓.

## §5.3 / Table 4 (arms, 50/50 stream) [arms_cnn_latency.json]
| Arm | Gate | Stream | Glared | Severe | Restored | Mean ms | p95 ms | Gain kept |
|---|---|---|---|---|---|---|---|---|
| C1 | – | 0.638 | 0.434 | 0.248 | 0% | 16.3 | 17.8 | 0% |
| C2 | – | 0.778 | 0.714 | 0.635 | 100% | 88.3 | 94.0 | 100% |
| C3 | CNN | 0.765 | 0.691 | 0.611 | 39% | 98.6 | 147.6 | 91% |
| C3 | Feature | 0.772 | 0.703 | 0.632 | 47% | 54.9 | 91.8 | 95% |
| C3 | Brightness | 0.754 | 0.666 | 0.605 | 57% | 59.9 | 97.6 | 83% |
Stream mAP@50:95: C1 0.430, C2 0.534, C3-CNN 0.524, C3-feature 0.528, C3-brightness 0.517.
Excluding the CNN gate's 521 training scenes (970 pairs): C1 0.650, C2 0.782, C3-CNN 0.770 (90% kept, 39% restored).
Frames per second (mean): C1 61, C2 11.3, C3-CNN 10.1, C3-feature 18.2, C3-brightness 16.7.

## Table 7 (gates)
| Gate | Synthetic P | R | F1 | Real Esfahani P | Real BDD hard P | Cost (CPU) |
|---|---|---|---|---|---|---|
| Brightness | 0.62 | 0.72 | 0.66 | not measured | not measured | 0.12 ms |
| Feature | 0.92 | 0.88 | 0.90 | not measured | not measured | 1.14 ms |
| CNN | **1.00** | **0.82** | **0.90** | **0.88** (R 0.87, crops) | **33%** (68 of 204 flagged) | **34 ms** (our CPU), 11.7 ms (Harshitha's) |
CNN synthetic numbers are on v2 glare; the gate trained on v1 glare.

## §5.5 / Table 8 (real BDD100K, 72 glare + 72 clean) [real_bdd_results.json]
| Arm | Car AP, clean | Car AP, glare | Loss recovered (car AP) |
|---|---|---|---|
| C1 never restore | 0.365 | 0.112 | 0% |
| C2 always restore | 0.368 | 0.131 | 7.5% |
| C3 CNN gate | 0.365 | 0.132 | 7.9% |
By all-class mAP@50: C2 recovers 10%, C3 7% (the "7–10%" range ✓).
- [KITTI car AP] = **0.955** (clean v2 val).
- [BDD training pairs] = **3,000 BDD100K train frames with v2 synthetic glare** (no gain), and
  separately **v3 synthetic glare fitted to real-glare statistics** (also no gain: car AP 0.110;
  pre-registered rule failed). [synth_v3_plan.json, synth_v3_decision.json]
- Gate on the real test set: P 1.00, R 0.94. Don't headline this; the set was built with the gate's help.

## Data availability
Code and frame lists: github.com/ananyagupta17/glare-gated-detection (`data/bdd_real_manifest.csv`,
`data/glare_manifest_v2.csv`; images regenerate from seed 42). BDD100K obtained via the Kaggle
mirror `solesensei/solesensei_bdd100k` under the BDD100K licence.

## Figures (paper_figures/, demo/)
- **Fig. 3** → `demo/01_pipeline_walkthrough_007004.png` (U-Net; median-gain severe frame; per-frame counts in captions).
- **New, real glare** → `paper_figures/fig_real_glare.png` (median frame: no change; 13 of 72 improved, 4 worse, 55 unchanged).
- **New, the leak** → `paper_figures/fig_dark_pixel_leak.png` (p1 luma, clean median 7 vs glared 36).
- **Optional, glare families** → `demo/03_severity_ramp_*.png` (pick 2 rows for one panel).
