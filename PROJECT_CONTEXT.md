# Glare-Gated Object Detection — Project Context

> Repo root as `CLAUDE.md`. Full context for every Claude Code session.
> Reflects Review-1 slide content + subsequent scope decisions.

---

## 1. Project

**VIT capstone, BCSE497J Project-I.** Team of 3. Review 2 requires 60–75% implementation
+ live demo.

### Problem

Sun glare occurs when low-angle sunlight enters the camera's field of view. It causes
overexposure that washes out road-scene detail — vehicles, pedestrians, traffic signs and
lane markings become hard to detect, degrading object-detection reliability in autonomous
vehicles. **~9,000 glare-related crashes occur annually.** Unlike a human driver, a fully
autonomous system has no fallback.

### Abstract (as presented at Review 1)

A lightweight glare gating framework for autonomous-driving perception. The framework first
classifies glare and invokes image restoration **only when glare is likely to affect
perception**. Object detection then runs on either the original or the restored image
depending on the gate's decision. Evaluated on general driving datasets with existing
bounding-box annotations (KITTI, BDD100K), plus real-glare images for realism validation.
Experiments compare detection on clean, glare-affected, restored, and conditionally
restored images. **Objective: determine whether conditional restoration recovers detection
accuracy while reducing unnecessary computational overhead.**

Unlike prior glare-robustness work that mainly measured detector *degradation* under
synthetic glare, this evaluates a complete **classification → restoration → detection**
pipeline.

### Admin
- **SDG:** 9 — Industry, Innovation & Infrastructure
- **TRL:** currently 2 → target 4 (validated in lab on public datasets)
- **Expected outcome:** working 3-stage pipeline + accuracy–latency characterisation
  across C1/C2/C3; Scopus conference paper + prototype

---

## 2. Pipeline

```
Input frame
    │
    ▼
┌──────────────────────┐
│ GLARE GATE           │   binary: glare / no-glare
│ (lightweight)        │
└─────────┬────────────┘
    glare?│
     ┌────┴────┐
    YES        NO (skip)
     │          │
     ▼          │
┌──────────┐    │
│RESTORATION│   │
└─────┬────┘    │
      └────┬────┘
           ▼
   ┌───────────────┐
   │ YOLOv8 (COCO- │
   │  pretrained)  │
   └───────┬───────┘
           ▼
   detections → mAP, latency, trigger rate
```

| Arm | Restoration policy | Purpose |
|-----|-------------------|---------|
| **C1** | never | baseline — how much does glare hurt? |
| **C2** | always | conventional restore-then-detect; upper bound on cost |
| **C3** | gated (proposed) | restore only when gate fires |

---

## 3. ⚠️ Scope decisions — READ FIRST

**Three deviations from the Review-1 slides. Reconcile before Review 2.**

1. **Binary, not severity.** Slides say "glare-*severity* gate" and "classifies the
   severity". Current decision is **binary glare / no-glare**. Either update the slide
   wording or present binary as a deliberate first stage with severity as future work.
   Don't let a panelist catch the mismatch.

2. **KITTI first, BDD100K later.** Slides list both. For the Review-2 timeline use
   **KITTI only** (smaller, faster, and the Huawei glare paper used it as its synthesis
   base — defensible). BDD100K is a scale-up if time allows.

3. **Real-glare validation is optional for Review 2.** Slides mention TT100K (which the
   slides themselves note is primarily traffic-sign). If real-glare validation is needed,
   better options are the Esfahani 200-image set or the GLARE dataset (2,157 real glare
   images). Treat as P2.

**Also critical, given binary scope:**

> **The evaluation set MUST contain both clean and glared frames (~50/50).**
> If every test image is glared, the gate always fires and C3 becomes byte-identical to
> C2 — the experiment produces nothing. **Trigger rate is a first-class metric.**

**No detector training.** COCO-pretrained YOLOv8, inference only. This is what makes the
timeline feasible.

---

## 4. This repo's scope

Owns **data + glare synthesis + detection + evaluation**. Teammates own the gate classifier
(Harshitha) and restoration; both consume this repo's output.

**In scope:** KITTI prep & label conversion · Gaussian glare synthesis · YOLOv8 inference
wrapper · evaluation harness (mAP, latency, trigger rate) · **C1 arm end-to-end**

**Out of scope:** gate classifier · restoration module

---

## 5. Dataset — KITTI

- Download **left colour images (`image_2`)** + **training labels (`label_2`)** only.
  Skip stereo/LiDAR/calibration.
- Official test labels are withheld → use the training set (~7,481 labelled images).
- **Fixed subset of 300–500 images.** Write IDs to `data/subset_ids.txt` and **commit it** —
  teammates must use the identical subset.

**KITTI → YOLO label conversion:**
```
x_center = (left + right)  / 2 / img_width
y_center = (top  + bottom) / 2 / img_height
width    = (right - left)      / img_width
height   = (bottom - top)      / img_height
```

**⚠️ Class mapping gotcha.** COCO-pretrained YOLOv8 outputs COCO classes, not KITTI classes.
Collapse to two:

| KITTI | → ours | ← COCO |
|-------|--------|--------|
| Car, Van | `car` | `car` (id 2) |
| Pedestrian, Person_sitting | `person` | `person` (id 0) |
| Truck, Tram, Misc, Cyclist, DontCare | ignore | — |

Filter both ground truth and predictions before computing mAP, or scores look near-zero and
you'll waste hours debugging.

---

## 6. Glare synthesis

**Gaussian daytime sun-glare model** — a published, accepted approach for *daytime* glare.
Distinct from nighttime lens-flare methods (Flare7K++ etc.), which model the wrong
phenomenon.

**Algorithm:**
1. Flare centre `(cx, cy)` at random position in the **upper image region** (sun above horizon)
2. Normalised Gaussian mask: `mask = exp(-((x-cx)² + (y-cy)²) / (2σ²))`
3. Scale by intensity factor + optional warm tint
4. **Composite additively in linear space** — gamma-decode sRGB → add → gamma-encode.
   Light adds linearly; compositing in gamma space is physically wrong.
5. **Clip only AFTER combining.** Post-combination clipping produces the blown-out white
   saturation characteristic of real glare. Clip early and you lose the effect.

Randomise per image (centre, σ, intensity, tint) so the model can't learn one fixed artifact.

**Why synthesis, not collection** — the review answer:
- Real paired clean/glare images of the same scene are impossible to capture
- We placed the glare, so we get **free ground-truth labels** — no manual annotation.
  Valeo (2020) showed manual glare annotation is unreliable (no crisp boundary); their CNN
  scored only IoU 51 partly because of it.
- The clean original is **restoration ground truth** — paired data for the restoration teammate

**Outputs:**
```
data/clean/            # original KITTI subset
data/glared/           # glared versions
data/labels/           # YOLO boxes (identical for both — glare doesn't move objects)
data/glare_labels.csv  # image_id, is_glared (0/1) — for the gate classifier
data/subset_ids.txt    # committed
```

---

## 7. Detection & evaluation

**Detection:** Ultralytics YOLOv8, COCO-pretrained, no training. Wrap inference to run over a
directory emitting predictions + per-image latency.

**Latency measurement:** time only the inference call (exclude disk I/O); **warm up** with a
few throwaway inferences first — the first call includes lazy init and skews results. Record
the device (MPS/CUDA/CPU) with results; latency is meaningless without it.

**Metrics:** mAP@0.5 · per-frame latency (mean + median) · trigger rate

| Arm | mAP@0.5 | mean latency (ms) | trigger rate |
|-----|---------|-------------------|--------------|
| C1 (never) | | | 0% |
| C2 (always) | | | 100% |
| C3 (gated) | | | ? |

Break results down by **clean vs glared** subsets — that's where the gate's behaviour shows.

**Per-component criteria (team-agreed):** gate → precision/recall/F1 · restoration →
PSNR/SSIM vs clean originals · detection → mAP + latency (this repo)

---

## 8. Repo structure

```
├── CLAUDE.md
├── requirements.txt
├── data/          (see §6)
├── src/
│   ├── kitti_prep.py       # download, subset, label conversion
│   ├── glare_synth.py      # Gaussian glare generation
│   ├── detect.py           # YOLOv8 wrapper + latency timing
│   ├── evaluate.py         # mAP, arm comparison
│   └── run_experiment.py   # orchestrates C1/C2/C3
├── results/
└── notebooks/
```

---

## 9. Build order

**P0 — must exist for the demo**
1. KITTI subset prepared, labels converted, **verified by visualising boxes**
2. Glare synthesis working — visually inspect (does it look like real washout?)
3. YOLOv8 running, sanity-checked on clean images
4. **C1 arm complete:** mAP clean vs glared → *"glare drops detection from X to Y"*

That alone is a genuine demonstrable result and motivates the whole project.

**P1 — if time allows**
5. Evaluation harness generalised to accept any `restore(image)` function
6. C2 arm (once restoration exists)
7. C3 arm (once gate exists — **a brightness-threshold gate is an acceptable v1** if the
   trained classifier isn't ready)

**P2 — later**
8. Severity grading · BDD100K scale-up · real-glare validation · FID/CMMD realism check

---

## 10. Teammate interfaces

**To the gate (Harshitha):** `data/glared/` + `data/clean/` + `data/glare_labels.csv`.
Gate exposes `predict(image) -> bool`.

**To restoration:** `data/glared/` as input, `data/clean/` as ground truth.
Restorer exposes `restore(image) -> image`.

**Critical:** all three components run on the **same image IDs** — not "we all use KITTI",
the same `subset_ids.txt`. Teammates must **not** generate their own glare. If the gate
trains on a different glare distribution than the detector is evaluated on, the three-arm
comparison silently breaks.

---

## 11. Constraints & gotchas

- **Apple Silicon (M3)** for dev. YOLOv8 inference usually works on MPS; may need
  `device='mps'` explicitly, or fall back to CPU. Team GPU exists for training-heavy work.
- **Daytime sun glare** (large-area overexposure), *not* nighttime lens flare
  (point-light ghosts/starbursts). Don't reach for Flare7K++.
- KITTI test labels withheld → use training split.
- COCO ≠ KITTI classes (§5).
- Timing must exclude I/O and include warm-up.

---

## 12. Objections to prepare for (from our own lit survey)

The strongest counter-arguments a reviewer can raise. Have answers ready.

**"IA-YOLO shows enhancement costs only +13 ms with no clean-domain accuracy tax.
Why gate something that cheap?"**
→ Our gate targets **expensive** restoration, not cheap differentiable filters. Deep Trident
(T-ITS 2022) takes **~0.98 s per small crop**, always-on. TogetherNet measured restoration
adding **6.7×–69×** latency. That's the regime where skipping pays.

**"DSNet gets +2–3 mAP at zero inference cost via training-only restoration.
Why not just do that?"**
→ Training-only restoration (DSNet, TogetherNet) resolves the tension by **amputation** —
restoration never runs at inference, so it cannot handle severe localised glare and the
restored image is never available. We resolve it by **selection**: restoration stays
available and fires only when warranted. Also: no joint retraining, works with off-the-shelf
components, per-frame adaptivity.

**"Degradation Type-Aware Restoration (Sensors 2024) already has a degradation classifier
that protects clean images."**
→ Closest prior work. But it targets weather (no glare) and has **no explicit runtime gate** —
it adapts *how* to restore, not *whether* to run restoration at all, and reports no
accuracy–latency trade-off. Our contribution is the skip decision and its measurement.

**"Isn't this just the Huawei self-supervised glare CNN?"**
→ That work routes the glare signal only to a **self-awareness alert** — it never triggers
restoration. Same holds for Valeo (2020), Esfahani (2021) and GCBAM-UNet (2024): four papers
connect glare detection to a downstream action, **none routes it to conditional restoration,
and none measures the cost–benefit.**

---

## 13. Literature (20 papers, with the numbers that matter)

**2026** · *Enhanced Canny-Based Vision System for Vanishing Point Detection…* (SAE WCX) —
quaternion colour filtering, adaptive Canny, Hough/DBSCAN/RANSAC. No restoration; no
end-to-end latency. PSNR 27.01, Entropy 6.94, EPI 0.9814, Canny overlap 0.9880.

**2025** · *FlareGS: 4D Flare Removal Using Gaussian Splatting* (ICCVW) — depth-guided
Uformer + 4D Gaussian splatting; temporally consistent. Compute-heavy; needs depth/multi-view.

**2025** · *Traffic Sign Detection in Glare Environments* (SIViP) — YOLO-SFM (YOLOv8 +
SPD-Conv, FasterNet, MLCA, 4 detection heads). Real glare data. **78.1% mAP@50, 55.5%
mAP@50–95, 76.0% F1, ~2.77M params.** Signs only; no gate or restoration.
← *architecture-modification alternative to our approach*

**2024** · *Glare Reduction for AV Perception* (IEEE T-IV) — saturation-aware deconvolution
across five AV tasks. **Detection +5.15%, recognition +18.16%.** Needs calibration; no
selective execution. ← *proves restoration CAN help*

**2024** · *Lens Flare-Aware Detector in Autonomous Driving* (VISAPP) — synthetic flare on
BDD100K, severity-based confidence correction. **AP 0.599 → 0.614** under severe flare.
Lightweight, severity-aware; needs clean reference; no restoration.

**2024** · *GCBAM-UNet: Sun Glare Segmentation* (Applied System Innovation) — VGG19-UNet +
CBAM. **92.78% accuracy, 86.09% Dice.** Small dataset; no detection/restoration test.

**2024** · *Degradation Type-Aware Image Restoration* (Sensors 24(19):6330) — shared encoder,
restoration/detection decoders + **degradation classifier**. **79.38% mAP, 99.27% degradation
accuracy.** No glare, no runtime gate. ← *closest prior work*

**2023** · *GLARE: A Dataset for Traffic Sign Detection in Sun Glare* (T-ITS 24(11),
pp. 12323–12330) — 2,157 real glare images, seven detectors trained on LISA / GLARE / both.
Mixed training best: **42.3 mAP.**

**2023** · *Improving Lens Flare Removal with General-Purpose Pipeline and Multiple Light
Sources Recovery* (ICCV, pp. 12923–12933) — ISP-aware synthesis + U-Net/Uformer.
**Up to 26.3 dB PSNR.** Limited detection/latency analysis.

**2022** · *Deep Trident Decomposition Network for Single License Plate Image Glare Removal*
(T-ITS 23(7), pp. 6596–6607) — glare-formation decomposition + residual refinement,
OCR-evaluated. **~0.98 s per tiny crop, always-on, plate-only.**
← *expensive-restoration anchor that makes a skip-it gate worthwhile*

**2022** · *Image-Adaptive YOLO* (AAAI 36(2), pp. 1792–1800) — CNN-PP predicts differentiable
filter parameters, trained end-to-end on detection loss. **+13 ms only; no clean-domain
accuracy tax.** Enhancement always-on; no when-to-restore decision.
← *challenge to our motivation — see §12*

**2022** · *TogetherNet* (CGF 41(7), pp. 465–476) — multi-task joint learning; restoration
active during training only. **Restore-then-detect 47.51–50.81 vs bare detector 51.23;
+6.7×–69× latency.** ← *our key motivation*

**2022** · *Rethinking Image Restoration for Object Detection* (NeurIPS 35) — adapts the
restorer (not the detector) toward a detector-friendly pseudo-GT via targeted attack.
Plug-and-play; frozen detector keeps clean accuracy. Small real-set gains; unconditional;
no latency study. ← *"restoration for the eye ≠ for the detector"; orthogonal & composable
with our gate*

**2021** · *Self-Supervised Glare CNN* (NeurIPS ML4AD Workshop) — self-supervised BiSeNet-v2
on physical + GAN glare; **0–5 severity benchmark**. Real precision/recall **98.7 / 91.1**.
Glare routed only to a self-awareness alert, never restoration.
← *closest prior art; quantifies detection drop vs severity*

**2021** · *Enhanced Road Lane Detection Facing Sun Glare* (J. Mobile Multimedia 17(4),
pp. 773–788) — neural network imitating Mertens exposure fusion before Hough/RANSAC.
**Auto-exposed TPR below 8%**; ~150 ms/frame; needs three aligned exposures; always-on.

**2021** · *DSNet: Joint Semantic Learning for Object Detection in Inclement Weather*
(IEEE TPAMI) — RetinaNet + restoration subnet sharing a block; restoration training-only.
**+2–3 mAP at zero inference cost.** Fog-only; needs paired clean GT; no severity notion.
← *key counter-argument our gate must beat on severe, localised glare*

**2021** · *Real-Time Automotive Night-Vision System…* (J. Real-Time Image Processing) —
region-specific conditional processing (headlight core / glare halo / dark road) with masking,
gamma correction, dynamic-patch enhancement. Night glare, not solar; no downstream metric.
← *precedent for conditional, region-specific processing*

**2021** · *Robust Glare Detection: Review, Analysis, and Dataset Release* (arXiv:2110.06006) —
200-image glare-mask dataset; multi-branch U-Net over RGB / HSV / contrast-map /
photometric-map. **Recall 0.7616, F1 0.4927, Accuracy 0.8950.** No severity, restoration or
downstream test.

**2020** · *Valeo — Sun Glare Detection on Automotive Surround-view Cameras* (IS&T EI,
pp. 80-1–80-9) — first public glare dataset (WoodScape); classical thresholding + CNN
segmentation baselines. **Glare IoU 51.** Names "enhance this region" as an option but builds
none. ← *foundational; our motivating citation*

---

## 14. Status (from Review 1)

**Completed:** problem scoping · literature survey (20 papers) · gap identification ·
system design finalised · dataset selection · experimental comparison defined

**Remaining:** synthetic glare generation · train & validate gate classifier · develop
restoration model · integrate gated pipeline · benchmark C1/C2/C3 · draft paper
