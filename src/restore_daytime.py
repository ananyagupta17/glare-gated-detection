"""Classical daytime sun-glare restorer: estimate the veil, subtract it, restore contrast.

    restore(image: np.ndarray) -> np.ndarray      uint8 BGR in, same-shape uint8 BGR out

WHY CLASSICAL, AND WHY THIS MODEL OF GLARE

Daytime sun glare is extra light: scatter from lens, cover glass and windscreen adds a
smooth, bright, slightly warm veil on top of the scene (src/glare_synth.py composites it
exactly that way, in linear light). So the restorer works in linear light too:

    observed = scene + veil            veil: smooth, low-frequency, >= 0

The veil is estimated from the DARK parts of the frame. Wherever the true scene is dark
(shadows, tyres, tree trunks), whatever brightness remains must be veil, so a local
minimum per channel, smoothed heavily, tracks it. This is the idea behind the dark-
channel prior (He et al., TPAMI 2011), used here additively rather than with a haze
transmission model, because glare adds light instead of blending towards an airlight.

Steps:
  1. linearise (gamma 2.2)
     (the veil map is reduced to the part above the frame's own low percentile, so a
     clean frame's ordinary dark level is not mistaken for glare)
  2. veil = blur(per-channel local minimum), computed on a 1/SCALE-size copy (the veil
     is low-frequency, so this is cheap and loses nothing), capped at a fraction of the
     local brightness so bright regions such as sky are not driven to black
  3. subtract OMEGA x veil, clamp at 0
  4. stretch: rescale so the bright end of the frame is where it was before subtraction
     (undoes the contrast the veil and the camera's auto-exposure took away)
  5. back to sRGB, then mild CLAHE on lightness

On a clean frame the dark-pixel minimum is near zero, so the veil estimate is small and
the frame is left almost unchanged.

Every step is OpenCV/numpy on CPU, about 10 ms per KITTI frame. No training, no weights,
no licence restrictions. Parameters were chosen on the v2 TRAIN split only, by detection
F1 (src/tune_restore_daytime.py, results/restore_daytime_tuning.json); defaults below.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

GAMMA = 2.2
REPO_ROOT = Path(__file__).resolve().parent.parent
TUNING = REPO_ROOT / "results" / "restore_daytime_tuning.json"

DEFAULTS = {
    "omega": 0.9,        # fraction of the estimated veil to remove
    "scale": 4,          # veil estimated on a 1/scale-size copy
    "min_size": 9,       # local-minimum window, in downscaled pixels
    "sigma": 12.0,       # Gaussian blur of the minimum map, in downscaled pixels
    "cap": 0.85,         # veil never exceeds this fraction of the local brightness
    "baseline_pct": 5,   # subtract only veil above this percentile of the veil map (0 = off)
    "stretch_pct": 99.5, # percentile kept fixed by the contrast stretch (0 disables)
    "clahe_clip": 1.5,   # 0 disables CLAHE
    "clahe_tile": 8,
}

_LUT_TO_LIN = ((np.arange(256, dtype=np.float32) / 255.0) ** GAMMA).astype(np.float32)


def load_params() -> dict:
    """Tuned parameters if the tuning file exists, else the defaults."""
    params = dict(DEFAULTS)
    if TUNING.is_file():
        params.update(json.loads(TUNING.read_text())["chosen"])
    return params


def estimate_veil(lin: np.ndarray, p: dict) -> np.ndarray:
    h, w = lin.shape[:2]
    s = max(1, int(p["scale"]))
    small = cv2.resize(lin, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_AREA)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(p["min_size"]), int(p["min_size"])))
    minimum = cv2.erode(small, k)                                    # per-channel local min
    veil = cv2.GaussianBlur(minimum, (0, 0), float(p["sigma"]))
    local = cv2.GaussianBlur(small, (0, 0), float(p["sigma"]))
    veil = np.minimum(veil, float(p["cap"]) * local)                 # protect bright regions
    # A clean frame's dark level is fairly even; real glare is concentrated around the
    # sun. Removing only the part above the frame's own low percentile leaves clean
    # frames close to untouched.
    if p.get("baseline_pct"):
        base = np.percentile(veil.reshape(-1, 3), float(p["baseline_pct"]), axis=0)
        veil = np.clip(veil - base[None, None, :], 0.0, None)
    return cv2.resize(veil, (w, h), interpolation=cv2.INTER_LINEAR)


def restore(image: np.ndarray, params: dict | None = None) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an H x W x 3 uint8 array")
    p = params if params is not None else _PARAMS
    img = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)

    lin = _LUT_TO_LIN[img]                                           # float32, linear light
    veil = estimate_veil(lin, p)
    out = np.clip(lin - float(p["omega"]) * veil, 0.0, None)

    # Contrast stretch: keep the bright end where it was before the veil was removed.
    # Percentiles are taken on a 1/8 copy; the bright end of a frame is not a fine detail.
    if p["stretch_pct"]:
        thumb_in = cv2.resize(lin, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
        thumb_out = cv2.resize(out, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
        hi_in = float(np.percentile(thumb_in.max(axis=2), p["stretch_pct"]))
        hi_out = float(np.percentile(thumb_out.max(axis=2), p["stretch_pct"]))
        if hi_out > 1e-4:
            out *= min(hi_in / hi_out, 3.0)

    srgb = np.clip(out, 0.0, 1.0) ** (1.0 / GAMMA)
    result = np.rint(srgb * 255.0).astype(np.uint8)

    if p["clahe_clip"] and p["clahe_clip"] > 0:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        t = int(p["clahe_tile"])
        clahe = cv2.createCLAHE(clipLimit=float(p["clahe_clip"]), tileGridSize=(t, t))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


_PARAMS = load_params()
