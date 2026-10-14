# Restoration handover

This note transfers the completed restoration work to the teammate responsible
for integration and to any coding agent continuing the project.

## What is implemented

`src/restore.py` wraps the pretrained Flare7K++ Uformer generator behind the
interface expected by the C1/C2/C3 harness:

```python
restore(image: np.ndarray) -> np.ndarray
```

The function accepts a three-channel OpenCV BGR image and returns a same-sized
`uint8` BGR image. Model loading is lazy and cached for the process. The adapter:

1. selects CUDA automatically when available and otherwise uses CPU;
2. imports only the upstream Uformer architecture, avoiding Flare7K's GPU-only
   evaluation utilities;
3. loads `models/restoration/net_g_last.pth` as a safe PyTorch state dictionary;
4. resizes the shorter image side to 512 pixels;
5. runs overlapping 512 by 512 tiles with feathered blending;
6. estimates the flare layer and returns a result at the original resolution.

`src/demo_restoration.py` provides a one-command demonstration, comparison
image, JSON/CSV metrics, PSNR and SSIM. `run_restoration_demo.bat` selects a
local virtual environment when available.

## Required checkout layout

The 82.6 MB checkpoint is shared separately and intentionally ignored by Git.
The upstream Flare7K architecture is also not copied into this repository.

```text
glare-gated-detection/
  external/
    Flare7K/
  models/
    restoration/
      net_g_last.pth
```

Setup from the repository root:

```powershell
git clone https://github.com/ykdai/Flare7K.git external/Flare7K
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy the separately shared checkpoint to
`models/restoration/net_g_last.pth`, then verify its SHA-256 against
`models/restoration/README.md`.

If Flare7K is elsewhere, set `FLARE7K_REPO`. The checkpoint and device may also
be overridden with `FLARE7K_CHECKPOINT` and `FLARE7K_DEVICE`.

## Verified demonstration

The demo was verified on the paired example embedded in
`results/glare_full_example_0.png` using CPU inference. The first measured run
produced:

| Measurement | Value |
|---|---:|
| Model loading | 6.534 s |
| Restoration inference | 37.092 s |
| Input PSNR | 7.7907 dB |
| Restored PSNR | 14.4584 dB |
| PSNR change | +6.6677 dB |
| Input SSIM | 0.439071 |
| Restored SSIM | 0.723428 |
| SSIM change | +0.284357 |

These are single-example demonstration values. Do not describe them as
full-dataset averages or as proof of recovered YOLO accuracy.

A clean re-run on 14 October 2026 reproduced the image metrics exactly. That
run took 45.933 s for inference and 42.634 s for cold model loading. CPU timing
therefore varies substantially with machine state and file cache; record the
device and fresh timing in every reported experiment.

## Commands

Prepared example:

```powershell
.\run_restoration_demo.bat
```

One external image:

```powershell
.\run_restoration_demo.bat --input "C:\path\to\glared.png"
```

Paired folder evaluation:

```powershell
python src/demo_restoration.py `
  --input data/glared_full `
  --clean datasets/kitti/images/val `
  --max-images 5
```

Full integration, once all data/models are present:

```powershell
python src/run_experiment.py --restore auto --gate auto
```

## What remains

1. Restore the complete held-out glared subset and calculate aggregate
   PSNR/SSIM instead of relying on one example.
2. Run C2 and C3 with the same YOLO checkpoint, labels and image IDs as C1.
3. Report mAP@50, mAP@50–95, restoration latency and end-to-end latency.
4. Profile on CUDA; CPU inference is too slow for the full experiment.
5. Test real daytime glare to measure the Flare7K++ night-to-day domain gap.
6. Inspect failures where clipping has destroyed information or restoration has
   introduced artifacts.

## Rules for the next agent

- Read `README.md`, this file and `src/run_experiment.py` before changing the
  restoration interface.
- Do not regenerate a different glare dataset. Use the seed-42 full manifest.
- Keep C1, C2 and C3 identical except for restoration policy.
- Do not claim object-detection recovery until restored images have been passed
  through YOLO and scored.
- Keep generated images, metrics, reports and presentation files out of Git.
- Keep `.pth` checkpoints out of Git; exchange them through the team's shared
  file channel and retain the documented checksum.
- Preserve checkpoint provenance and the upstream Flare7K licence notice.

## Known design decision

The upstream `test_large.py` is CUDA-only and imports unrelated BasicSR modules.
The adapter deliberately loads `uformer_arch.py` in isolation so CPU demo and
harness integration work without installing the full upstream training stack.
