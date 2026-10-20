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
   │  YOLOv8n      │
   │ (KITTI-trained)│
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

**Four deviations from the Review-1 slides. Reconcile before Review 2.**

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

4. **Trained detector, not COCO-pretrained.** YOLOv8n trained on the clean KITTI val set
   (1,496 images, 50 epochs, clean-only). KITTI-native 8-class output removes the
   COCO≠KITTI mapping problem entirely. Required by §2's definition of C1 — *"how much
   does glare hurt?"* only has an answer if the detector was never taught to cope with
   glare. Training on glared frames instead makes the model glare-invariant, collapses the
   measured drop toward zero, and hands a reviewer the §12 counter-argument the gate must
   beat ("just augment your training data"). Supersedes the earlier "no detector training,
   inference only" decision.

**Also critical, given binary scope:**

> **The evaluation set MUST contain both clean and glared frames (~50/50).**
> If every test image is glared, the gate always fires and C3 becomes byte-identical to
> C2 — the experiment produces nothing. **Trigger rate is a first-class metric.**

---

## 4. This repo's scope

Owns **data + glare synthesis + detection + evaluation**. Teammates own the gate classifier
(Harshitha) and restoration; both consume this repo's output.

**In scope:** KITTI subset prep · 4-type camera glare synthesis · YOLOv8 inference
wrapper · evaluation harness (mAP, latency, trigger rate) · **C1 arm end-to-end**

**Out of scope:** gate classifier · restoration module

---

## 5. Dataset — KITTI (Ultralytics distribution)

We use the **Ultralytics KITTI** redistribution, already at `datasets/kitti/` and
**already in YOLO format** — no manual `label_2` parsing or coordinate conversion needed.
(`datasets/` is gitignored; the download URL is in `datasets/kitti/kitti.yaml`.)

```
datasets/kitti/
├── kitti.yaml
├── images/train/  5985 × .png      images/val/  1496 × .png
└── labels/train/  5985 × .txt      labels/val/  1496 × .txt
```

- Files are ID-matched: `000001.png` ↔ `000001.txt`. IDs are 6-digit zero-padded.
- Labels are normalised YOLO `class cx cy w h` — one object per line, no empty files in val.
- Image size varies slightly per frame (~1242×375, some 1224×370). Read dimensions per
  image; never hardcode.
- **We draw our subset from `images/val/`** (1,496 frames) — an Ultralytics split of the
  official KITTI *training* set, so labels are present. The official KITTI test labels are
  withheld and unused here.
- **We use all 1,496 val frames** (§10). `src/prepare_subset.py` and its 400-image
  `data/subset_ids.txt` are superseded but kept for provenance.

**Class IDs in this distribution** (from `kitti.yaml`):

```
0 car   1 van   2 truck   3 pedestrian   4 Person_sitting   5 cyclist   6 tram   7 misc
```

Note there is no `DontCare` class — Ultralytics dropped it during conversion.

**Classes: all 8 native KITTI classes** — `car`, `van`, `truck`, `pedestrian`,
`Person_sitting`, `cyclist`, `tram`, `misc`. Because the detector is trained on KITTI
(§3), its output classes *are* KITTI's, so **no COCO mapping and no car/person collapse is
needed**. Per-class AP is reported alongside the mean, so rare classes show their noise
rather than hiding inside it — measured on held-out data, `Person_sitting` AP50 goes
0.8879 → 0.0000 under glare while `car` goes 0.9105 → 0.4904.

---

## 6. Glare synthesis

> **v2 rebuild (2026-10, code written, data NOT yet generated).** `src/glare_synth.py` is
> now v2: every KITTI train + val frame comes out clean AND glared through one camera
> pipeline (highlight expansion → shared exposure jitter → glare × m(s) + spatially
> varying floor → auto-exposure drop → Poisson-Gaussian noise → JPEG q 75–95). Severity s
> is continuous; mild/moderate/severe cutpoints come from `src/calibrate_severity.py`
> (KITTI train frames only, `models/yolo_baseline.pt`). `src/shortcut_test.py` holds
> pass thresholds registered before generation (`results/shortcut_test_thresholds.json`).
> Outputs: `data/v2/{clean,glared}/images/{train,val}/`, manifest
> `data/glare_manifest_v2.csv`. Run order: calibrate → generate → shortcut test.
> The v1 description below (PNG output, constant floor, half glared) is reachable at
> commit `a4d2b69`.

**Physically-motivated camera glare model**, implemented in `src/glare_synth.py`. Four
distinct glare types, 50 images each, so the gate cannot memorise one fixed artifact.
Still strictly *daytime* glare — no ghosts or starbursts, so Flare7K++ and the nighttime
lens-flare literature remain the wrong tool (§11).

### Why not a plain Gaussian

A Gaussian falls off as `exp(-r²)`: four radii out it is at ~1e-4 and gone, which renders
as a hard-edged white ball pasted on the frame. Real veiling glare — light scattered by
lens elements, cover glass and the windscreen — falls off as a **power law** (~1/r² to
1/r³) and is still at ~1e-2 at the same distance, which is why a real sun bleeds across
the whole image. Raising a Gaussian's peak to compensate only makes a brighter ball.

Every lobe is therefore a **Moffat profile**, the standard optical point-spread function:

```
I(r) = amp · (1 + (r/α)²) ^ −β        β = tail weight; lower reaches further
```

### What makes it read as photographed

1. **Multi-scale.** Each type stacks a narrow core, a mid glow and a very wide veil. One
   profile cannot be both a tight disc and a frame-wide wash — and neither is a real lens
   PSF one scale.
2. **The disc is never drawn.** Core amp is **60–200× full scale** (the sun genuinely is
   ~1e5× brighter than a road surface). The saturated disc is just the contour where that
   profile crosses 1.0 after clipping — emergent size, wrapped in its own bloom.
3. **Chromatic halo.** Each channel gets its own α (red widest, blue narrowest), since
   scattering is wavelength-dependent. The core saturates white in all three channels
   while the surround reddens with radius **on its own** — a mechanism, not a set tint.
4. **Black lift.** Scattered light reaches the whole sensor, so a small frame-wide floor
   crushes the shadows. This is the tell of a genuinely glared frame; a localised blob
   cannot reproduce it. Measured: shadows lift 50–125 levels, contrast RMS drops 10–26.
5. **CCD smear.** KITTI was shot on Point Grey Flea2 **CCD** sensors, which bleed charge
   vertically out of a saturated well into the readout column. `direct_sun` gets a subtle
   vertical band.
6. **The sun goes where the sky is.** Position is sampled weighted by clean-frame
   luminance³ within each type's vertical band, so the source lands in open sky rather
   than halfway down a dark treeline. No depth map exists, so this approximates occlusion
   rather than computing it — the one deliberate physical shortcut in the model.
7. **Composite additively in linear space** — gamma-decode sRGB → add → gamma-encode.
   Light adds linearly; compositing in gamma space gives a flat grey decal.
8. **Clip only AFTER combining**, once, at the end. Clipping early throws away the
   blown-out core that makes glare look like glare.

### The four types

| type | physical situation | signature |
|------|-------------------|-----------|
| `diffuse_haze` | sun high behind haze | broad soft near-white glow, no disc |
| `direct_sun` | sun low and in frame | saturated core + chromatic halo + CCD smear |
| `veiling_wash` | scatter, sun near/off frame | heavy tails, low peak, strongest black lift |
| `horizon_streak` | low sun smeared by windscreen | anisotropic, strongly warm |

Verified: worst frame 48% saturated (median 12.6%) so none are degenerate; glare is
purely additive (no pixel ever darkens); reruns are byte-identical.

**Why synthesis, not collection** — the review answer:
- Real paired clean/glare images of the same scene are impossible to capture
- We placed the glare, so we get **free ground-truth labels** — no manual annotation.
  Valeo (2020) showed manual glare annotation is unreliable (no crisp boundary); their CNN
  scored only IoU 51 partly because of it.
- The clean original is **restoration ground truth** — paired data for the restoration teammate

**Determinism.** Everything derives from `--seed` (default 42); per-image sub-seeds mean
an image's glare never depends on processing order. Teammates re-running get pixel-identical
output — which is what makes §10's shared-subset requirement enforceable.

**Outputs:**
```
data/clean/            # original KITTI subset
data/glared/           # glared versions (200)
data/labels/           # YOLO boxes (identical for both — glare doesn't move objects)
data/glare_labels.csv  # image_id, is_glared (0/1), glare_type — for the gate classifier
data/subset_ids.txt    # committed
```

⚠️ `glare_labels.csv` gained a third column **`glare_type`** (empty for clean rows).
`image_id` and `is_glared` are unchanged, so name-keyed parsers still work. It enables a
per-type breakdown of detection loss — a stronger paper result than one averaged number.

---

## 7. Detection & evaluation

**Detection:** Ultralytics YOLOv8n trained on clean KITTI (§3), 8 native classes; see
`models/yolo_baseline.pt`. Wrap inference to run over a directory emitting predictions +
per-image latency.

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
├── datasets/kitti/         # Ultralytics KITTI, YOLO format (gitignored, see §5)
├── data/                   # our derived subset (see §6)
├── src/
│   ├── prepare_subset.py   # 400-image subset — SUPERSEDED, kept for provenance (§10)
│   ├── glare_synth.py      # Moffat-PSF glare synthesis, 4 types (--full = 1,496)
│   ├── train_baseline.py   # C1: train on clean, measure the glare drop
│   ├── detect.py           # YOLOv8 wrapper + latency timing
│   ├── evaluate.py         # mAP, arm comparison
│   └── run_experiment.py   # orchestrates C1/C2/C3
├── results/
└── notebooks/
```

`data/clean*/`, `data/glared*/` and `datasets/` are gitignored (regenerable / 390 MB —
glared PNGs average 469 KB against 51 KB clean, so the 748 `_full` frames alone are
340 MB). **`data/glare_labels_full.csv` is committed** — it is the manifest teammates
regenerate against (§10). The 400-subset files (`data/labels/`, `data/subset_ids.txt`,
`data/glare_labels.csv`) remain committed for provenance.

---

## 9. Build order

**P0 — must exist for the demo**
1. KITTI subset prepared (`prepare_subset.py`), **verified by visualising boxes**
2. Glare synthesis working — visually inspect (does it look photographed, not pasted?)
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

> **Changing with v2 (§6).** Once v2 is generated and the shortcut test passes, the
> contract becomes `data/glare_manifest_v2.csv` (`image_id, split, is_glared, glare_type,
> severity, sat_frac`; two rows per frame) plus `data/v2/`. Gate and restorer train on
> the `train` split; all 1,496 `val` pairs are the test set for every arm. Cross-machine
> check: regenerate, then `git diff data/glare_params_v2.csv data/glare_v2_sha256.txt` —
> params differ = real divergence; only hashes differ = library build mismatch
> (numpy/pillow are pinned in requirements.txt). Until then, the v1 contract below holds.

**The dataset is the full 1,496-image KITTI val set**, not the earlier 400-image subset.
748 are glared, 748 stay clean.

**To the gate (Harshitha):** `data/glared_full/` (748) + `datasets/kitti/images/val/`
(clean originals) + `data/glare_labels_full.csv` (`image_id`, `is_glared`, `glare_type`).
Gate exposes `predict(image) -> bool`.

**To restoration:** `data/glared_full/` as input, the matching
`datasets/kitti/images/val/<id>.png` as ground truth.
Restorer exposes `restore(image) -> image`.

**Alignment is by seed, not by file sharing.** Everyone runs
`python src/glare_synth.py --full` with the default `--seed 42` and gets **byte-identical
pixels** (verified: two consecutive runs produce identical hashes for all 748 PNGs).
`data/glare_labels_full.csv` is committed as the manifest; the images regenerate locally.

**Why 1,496 and not 400:** better statistics (~5× the training data for the C1 baseline,
187 frames per glare type instead of 37), lower overfitting risk, and it matches standard
KITTI evaluation practice of reporting on the whole val split.

**⚠️ Critical — do not mix the two realisations.** The old 400-image subset
(`data/glared/`, `data/glare_labels.csv`, `data/subset_ids.txt`) is a *different* draw:
of the 400 IDs present in both, only 50 agree — 200 flip glared/clean and 150 keep the
glare but change type. Selection and type assignment depend on the population, so they are
not interchangeable. If the gate trains on one and the detector is evaluated on the other,
the three-arm comparison breaks **silently, with no error**. The 400-subset files are kept
only for provenance; new work uses the `_full` set.

---

## 11. Constraints & gotchas

- **Apple Silicon (M3)** for dev. YOLOv8 inference usually works on MPS; may need
  `device='mps'` explicitly, or fall back to CPU. Team GPU exists for training-heavy work.
- **Daytime sun glare** (large-area overexposure), *not* nighttime lens flare
  (point-light ghosts/starbursts). Don't reach for Flare7K++.
- KITTI test labels withheld → use training split.
- Detector is KITTI-trained, 8 native classes — no COCO mapping (§5).
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
