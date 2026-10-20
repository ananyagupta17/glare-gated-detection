"""Paired sun-glare synthesis through a camera model (v2), over KITTI train and val.

Every KITTI frame comes out twice -- once clean, once glared -- through the SAME camera
pipeline. The only difference between the two versions is the glare and the camera's
reaction to it. That pairing is what v1 lacked:

  * v1 glared half the val frames and left the other half untouched, so "clean vs glared"
    compared different scenes.
  * v1 drew glare straight onto the decoded KITTI JPEG and saved a lossless PNG, with a
    constant black-lift floor. The floor alone was a 100%-accurate classifier (min pixel
    0-2 clean vs 31-121 glared), and the clean/glared file formats differed.

v1 is reachable in git history at commit a4d2b69 (src/glare_synth.py, --full mode).

THE CAMERA PIPELINE, PER FRAME

    KITTI JPEG -> decode -> linear light -> highlight expansion          (shared by the pair)
      -> exposure jitter e0                                             (shared by the pair)
      -> [glared] + Moffat glare lobes x m(s) + spatially varying floor
      -> [glared] auto-exposure: gain = (M_clean / M_glared)^k, clamped to [0.35, 1]
      -> Poisson-Gaussian sensor noise                     (independent draw per version)
      -> clip once -> sRGB -> JPEG, quality ~ U(75, 95)    (independent draw per version)

Kept from v1 (the physics, see CLAUDE.md section 6): Moffat power-law lobes stacked at
several scales, a per-channel chromatic halo, CCD smear on direct_sun, compositing in
linear light, a single clip at the end, and sky-weighted source placement.

New in v2:

  * Highlight expansion. KITTI skies are already clipped at 255. Without expansion, the
    auto-exposure drop would turn clipped sky into flat grey. Pixels above a linear knee
    are boosted by a per-frame factor so they stay near white after the drop, an
    approximation of the radiance the original sensor clipped.
  * Spatially varying floor. Scattered light still lifts the blacks, which is real, but
    by a low-frequency random field with a mild ramp toward the sun instead of one
    constant, and sensor noise lands on top of it.
  * Auto-exposure is a CONSEQUENCE of the glare, not an independent draw. The camera
    meters centre-weighted brightness with and without the glare and partly compensates
    (k is how aggressive its AE is), so strong glare darkens the whole scene.
  * Continuous severity s in [0, 1]. Only the amplitudes and the floor scale with s, on
    a log scale between a mild and a severe multiplier per type. Shape parameters stay
    random and independent of s.
  * Mild / moderate / severe bands are cut by measured detection damage, not guessed.
    src/calibrate_severity.py fits per-type cutpoints on KITTI TRAIN frames only and
    writes results/severity_calibration.json; this script reads it and draws s so each
    type gets thirds of each band.

DETERMINISM

Every random draw comes from a per-frame seed, sha256("seed:split:image_id:stream"), not
Python's hash() (salted per process). Output therefore does not depend on processing
order, and frames can be generated in parallel. On one machine reruns are byte-identical.
Across machines, JPEG bytes and float rounding depend on the exact Pillow, libjpeg and
numpy builds, which requirements.txt pins. Every run rewrites the committed
data/glare_params_v2.csv and data/glare_v2_sha256.txt, so `git diff` on those two files
is the cross-machine check: params differ -> a real divergence; params identical but
hashes differ -> a library build difference.

OUTPUTS (Ultralytics layout, so each version is directly evaluable)

    data/v2/clean/images/{train,val}/<id>.jpg     labels/{split} -> datasets/kitti/labels
    data/v2/glared/images/{train,val}/<id>.jpg    labels/{split} -> datasets/kitti/labels
    data/v2/clean.yaml, data/v2/glared.yaml
    data/glare_manifest_v2.csv   image_id, split, is_glared, glare_type, severity, sat_frac
    data/glare_params_v2.csv     every sampled parameter, one row per frame
    data/glare_v2_sha256.txt     checksums of every written JPEG
    results/glare_v2_example_<type>.png

Usage:
    python src/glare_synth.py --dry-run              # plan only, writes nothing
    python src/glare_synth.py                        # train + val, needs calibration
    python src/glare_synth.py --splits val --workers 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
KITTI = REPO_ROOT / "datasets" / "kitti"
OUT_ROOT = REPO_ROOT / "data" / "v2"
MANIFEST = REPO_ROOT / "data" / "glare_manifest_v2.csv"
PARAMS = REPO_ROOT / "data" / "glare_params_v2.csv"
CHECKSUMS = REPO_ROOT / "data" / "glare_v2_sha256.txt"
CALIBRATION = REPO_ROOT / "results" / "severity_calibration.json"
SPLITS = ("train", "val")
VERSIONS = ("clean", "glared")
MANIFEST_COLUMNS = ["image_id", "split", "is_glared", "glare_type", "severity", "sat_frac"]
BANDS = ("mild", "moderate", "severe")

GAMMA = 2.2                       # sRGB <-> linear approximation, as in v1

# --- camera model constants -----------------------------------------------------------
EXPOSURE_JITTER_SIGMA = 0.08      # log-normal base exposure, shared by the pair
HIGHLIGHT_KNEE = 0.80             # linear; above this, highlights are expanded
HIGHLIGHT_BOOST = (1.0, 3.0)      # how far a clipped highlight is pushed past 1.0
AE_K = (0.5, 0.9)                 # AE aggressiveness: 0 = none, 1 = full compensation
AE_MIN_GAIN = 0.35                # the camera never stops down further than this
SHOT_NOISE_A = (2e-5, 2e-4)       # log-uniform; variance = a * signal + b, linear light
READ_NOISE_B = (1e-8, 1e-6)       # log-uniform
JPEG_QUALITY = (75, 95)           # inclusive
FLOOR_GRID = (4, 8)               # rows, cols of the low-frequency floor field
FLOOR_FIELD_MIX = 0.4             # floor = f0 * (1 - mix + mix * field) * ramp
FLOOR_RAMP = 0.3                  # floor falls by this fraction across the frame diagonal
SAT_LEVEL = 250                   # sat_frac: all three channels at or above this

# Amplitude multiplier m(s) = lo * (hi / lo) ** s applied to every lobe and the floor.
# m = 1 reproduces v1's amplitudes. lo is chosen so s = 0 adds only a few code values;
# hi sits at or slightly past v1's worst case. These are starting values: calibration
# measures whether the range spans harmless to severe and refuses to proceed if not.
SEVERITY_SCALE = {
    "diffuse_haze": (0.05, 1.2),
    "direct_sun": (0.08, 1.5),
    "veiling_wash": (0.05, 1.2),
    "horizon_streak": (0.04, 1.0),
}


@dataclass(frozen=True)
class Lobe:
    """One Moffat light lobe: I(r) = amp * (1 + (r/alpha)^2) ^ -beta, in linear RGB."""
    cx: float
    cy: float
    alpha: float                       # radius scale in px
    beta: float                        # tail weight; lower = reaches further
    amp: float                         # linear-light peak at m = 1; >1 saturates on purpose
    tint: tuple[float, float, float]   # per-channel amplitude
    chroma: tuple[float, float, float] = (1.0, 1.0, 1.0)   # per-channel alpha scale
    aspect: float = 1.0                # sy/sx; <1 flattens into a streak
    theta: float = 0.0                 # rotation, radians
    vertical: bool = False             # ignore dy -> full-height band (CCD smear)


@dataclass(frozen=True)
class FrameSpec:
    """Everything sampled for one frame. Both versions render from this alone."""
    split: str
    image_id: str
    glare_type: str
    severity: float
    band: str
    mult: float                        # m(s)
    lobes: tuple[Lobe, ...]
    floor_f0: float                    # floor level at m = 1
    floor_grid: tuple[tuple[float, ...], ...]
    exposure: float                    # e0, shared
    highlight_boost: float             # shared
    ae_k: float
    noise_clean: tuple[float, float]   # (a, b)
    noise_glared: tuple[float, float]
    jpeg_q_clean: int
    jpeg_q_glared: int
    seed: int


# --- per-type samplers (shape at m = 1) -----------------------------------------------
# Ranges are v1's, widened slightly so the types overlap rather than sit in four
# disjoint boxes the gate could memorise.

def _warm(rng: random.Random, g: tuple[float, float], b: tuple[float, float]):
    return (1.0, rng.uniform(*g), rng.uniform(*b))


def _diffuse_haze(rng: random.Random) -> tuple[list[Lobe], float]:
    """Sun high behind haze: no disc at all, just a broad soft near-white glow."""
    a = rng.uniform(130.0, 360.0)
    return [
        Lobe(0, 0, alpha=a, beta=rng.uniform(1.0, 1.5),
             amp=rng.uniform(0.40, 1.00),
             tint=_warm(rng, (0.92, 0.99), (0.85, 0.97)),
             chroma=(1.0, 0.95, 0.90)),
        Lobe(0, 0, alpha=a * rng.uniform(2.3, 4.2), beta=rng.uniform(0.82, 1.12),
             amp=rng.uniform(0.08, 0.24),
             tint=_warm(rng, (0.88, 0.97), (0.80, 0.93)),
             chroma=(1.0, 0.92, 0.84)),
    ], rng.uniform(0.002, 0.009)


def _direct_sun(rng: random.Random) -> tuple[list[Lobe], float]:
    """Sun low and in frame: saturated core, chromatic halo, wide veil, CCD smear."""
    core_a = rng.uniform(3.5, 10.5)
    return [
        # Core: amp far past full scale. The visible disc is where this crosses 1.0.
        Lobe(0, 0, alpha=core_a, beta=rng.uniform(1.6, 2.5),
             amp=rng.uniform(60.0, 200.0),
             tint=_warm(rng, (0.95, 1.00), (0.88, 0.98)),
             chroma=(1.0, 0.96, 0.92)),
        # Glow: bridges core to veil so there is no shoulder between them.
        Lobe(0, 0, alpha=rng.uniform(22.0, 70.0), beta=rng.uniform(1.25, 1.8),
             amp=rng.uniform(1.2, 5.5),
             tint=_warm(rng, (0.80, 0.93), (0.58, 0.78)),
             chroma=(1.0, 0.90, 0.79)),
        # Veil: heavy tail, carries the glare across the frame.
        Lobe(0, 0, alpha=rng.uniform(120.0, 330.0), beta=rng.uniform(0.82, 1.22),
             amp=rng.uniform(0.14, 0.44),
             tint=_warm(rng, (0.66, 0.84), (0.40, 0.62)),
             chroma=(1.0, 0.87, 0.74)),
        # CCD charge bleed up and down the readout column. Subtle by design.
        Lobe(0, 0, alpha=core_a * rng.uniform(1.1, 2.0), beta=rng.uniform(1.0, 1.5),
             amp=rng.uniform(0.05, 0.16),
             tint=_warm(rng, (0.92, 0.99), (0.86, 0.96)),
             vertical=True),
    ], rng.uniform(0.004, 0.015)


def _veiling_wash(rng: random.Random) -> tuple[list[Lobe], float]:
    """Scatter with the sun near or off frame: contrast drains out of everything."""
    a = rng.uniform(280.0, 650.0)
    return [
        Lobe(0, 0, alpha=a, beta=rng.uniform(0.68, 0.98),
             amp=rng.uniform(0.20, 0.55),
             tint=_warm(rng, (0.80, 0.95), (0.66, 0.88)),
             chroma=(1.0, 0.91, 0.82),
             aspect=rng.uniform(0.7, 1.3), theta=rng.uniform(-0.4, 0.4)),
    ], rng.uniform(0.009, 0.028)          # the strongest black lift of the four


def _horizon_streak(rng: random.Random) -> tuple[list[Lobe], float]:
    """Low sun smeared horizontally by the windscreen: anisotropic and strongly warm."""
    a = rng.uniform(180.0, 450.0)
    th = math.radians(rng.uniform(-10.0, 10.0))
    asp = rng.uniform(0.09, 0.30)
    return [
        Lobe(0, 0, alpha=a, beta=rng.uniform(1.1, 1.75),
             amp=rng.uniform(1.8, 9.5),                    # saturates along the band
             tint=_warm(rng, (0.76, 0.91), (0.52, 0.74)),
             chroma=(1.0, 0.90, 0.80), aspect=asp, theta=th),
        Lobe(0, 0, alpha=a * rng.uniform(1.5, 2.7), beta=rng.uniform(0.82, 1.18),
             amp=rng.uniform(0.12, 0.36),
             tint=_warm(rng, (0.68, 0.85), (0.44, 0.66)),
             chroma=(1.0, 0.88, 0.76), aspect=asp * rng.uniform(1.5, 2.5), theta=th),
    ], rng.uniform(0.006, 0.018)


GLARE_TYPES = {
    "diffuse_haze": _diffuse_haze,
    "direct_sun": _direct_sun,
    "veiling_wash": _veiling_wash,
    "horizon_streak": _horizon_streak,
}

# Where each type's source may sit, as a fraction of image height, and whether the
# position is drawn toward bright pixels (unchanged from v1).
TYPE_PLACEMENT = {
    "diffuse_haze": dict(y_range=(0.00, 0.34), sky_weighted=True, off_frame=False),
    "direct_sun": dict(y_range=(0.08, 0.46), sky_weighted=True, off_frame=False),
    "veiling_wash": dict(y_range=(-0.10, 0.42), sky_weighted=False, off_frame=True),
    "horizon_streak": dict(y_range=(0.18, 0.50), sky_weighted=True, off_frame=False),
}


# --- seeding --------------------------------------------------------------------------

def frame_seed(seed: int, split: str, image_id: str, stream: str) -> int:
    """Stable 64-bit seed per (frame, stream). Independent of order and of the process."""
    digest = hashlib.sha256(f"{seed}:{split}:{image_id}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def rng_for(seed: int, split: str, image_id: str, stream: str) -> random.Random:
    return random.Random(frame_seed(seed, split, image_id, stream))


def np_rng_for(seed: int, split: str, image_id: str, stream: str) -> np.random.Generator:
    return np.random.default_rng(frame_seed(seed, split, image_id, stream))


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def severity_mult(glare_type: str, s: float) -> float:
    lo, hi = SEVERITY_SCALE[glare_type]
    return lo * (hi / lo) ** s


# --- sampling -------------------------------------------------------------------------

def place_source(rng: random.Random, clean: np.ndarray, type_name: str) -> tuple[float, float]:
    """Pick the source position; sky-weighted types are drawn toward bright pixels.

    Sampling proportional to luminance^3 within the type's vertical band puts the sun in
    open sky rather than against a dark treeline. No depth map exists, so this
    approximates occlusion rather than computing it.
    """
    h, w = clean.shape[:2]
    place = TYPE_PLACEMENT[type_name]
    y_lo, y_hi = (f * h for f in place["y_range"])

    if place["off_frame"]:
        return rng.uniform(-0.25 * w, 1.25 * w), rng.uniform(y_lo, y_hi)

    band_lo, band_hi = max(0, int(y_lo)), max(1, min(h, int(y_hi)))
    step = 4                                   # subsample; we only need a coarse position
    band = clean[band_lo:band_hi:step, ::step].astype(np.float32).mean(axis=2)
    weights = np.power(band / 255.0, 3.0).ravel()   # ^3 strongly favours sky over road

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 1e-6:     # tunnel frame: nothing bright
        return rng.uniform(0, w - 1), rng.uniform(y_lo, y_hi)

    idx = int(np.searchsorted(np.cumsum(weights), rng.random() * total))
    idx = min(idx, weights.size - 1)
    row, col = divmod(idx, band.shape[1])
    return (col * step + rng.uniform(0, step),
            band_lo + row * step + rng.uniform(0, step))


def sample_frame(clean: np.ndarray, split: str, image_id: str, glare_type: str,
                 severity: float, band: str, seed: int) -> FrameSpec:
    """Draw every parameter for one frame. Each stream has its own RNG, so changing how
    one stream is used never shifts the draws of another."""
    pair = rng_for(seed, split, image_id, "pair")
    glare = rng_for(seed, split, image_id, "glare")
    cam_c = rng_for(seed, split, image_id, "camera-clean")
    cam_g = rng_for(seed, split, image_id, "camera-glared")

    lobes, floor_f0 = GLARE_TYPES[glare_type](glare)
    cx, cy = place_source(glare, clean, glare_type)
    lobes = tuple(replace(lobe, cx=cx, cy=cy) for lobe in lobes)
    rows, cols = FLOOR_GRID
    grid = tuple(tuple(glare.random() for _ in range(cols)) for _ in range(rows))
    ae_k = glare.uniform(*AE_K)

    def camera(rng: random.Random) -> tuple[tuple[float, float], int]:
        noise = (log_uniform(rng, *SHOT_NOISE_A), log_uniform(rng, *READ_NOISE_B))
        return noise, rng.randint(*JPEG_QUALITY)

    noise_c, q_c = camera(cam_c)
    noise_g, q_g = camera(cam_g)

    return FrameSpec(
        split=split, image_id=image_id, glare_type=glare_type,
        severity=float(severity), band=band, mult=severity_mult(glare_type, severity),
        lobes=lobes, floor_f0=floor_f0, floor_grid=grid,
        exposure=math.exp(pair.gauss(0.0, EXPOSURE_JITTER_SIGMA)),
        highlight_boost=pair.uniform(*HIGHLIGHT_BOOST),
        ae_k=ae_k, noise_clean=noise_c, noise_glared=noise_g,
        jpeg_q_clean=q_c, jpeg_q_glared=q_g, seed=seed,
    )


# --- rendering ------------------------------------------------------------------------

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return (rgb.astype(np.float32) / 255.0) ** GAMMA


def linear_to_srgb_u8(linear: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(linear, 0.0, 1.0) ** (1.0 / GAMMA) * 255.0).astype(np.uint8)


def expand_highlights(linear: np.ndarray, boost: float) -> np.ndarray:
    """Push near-clipped pixels past 1.0 so an exposure drop leaves them bright."""
    peak = linear.max(axis=2, keepdims=True)
    t = np.clip((peak - HIGHLIGHT_KNEE) / (1.0 - HIGHLIGHT_KNEE), 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)                     # smoothstep, no visible edge
    return linear * (1.0 + (boost - 1.0) * weight)


def glare_lobes(spec: FrameSpec, height: int, width: int) -> np.ndarray:
    """Sum of all lobes at m = 1, linear RGB, unclipped."""
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    out = np.zeros((height, width, 3), dtype=np.float32)
    for lobe in spec.lobes:
        dx = xs - lobe.cx
        dy = ys - lobe.cy
        if lobe.theta:
            cos_t, sin_t = math.cos(lobe.theta), math.sin(lobe.theta)
            dx, dy = dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t
        # Per-channel alpha: red scatters widest, so the halo reddens outward by itself
        # while the saturated core stays white.
        for ch in range(3):
            a = lobe.alpha * lobe.chroma[ch]
            r2 = (dx / a) ** 2
            if not lobe.vertical:
                r2 = r2 + (dy / (a * lobe.aspect)) ** 2
            out[:, :, ch] += lobe.amp * lobe.tint[ch] * np.power(1.0 + r2, -lobe.beta)
    return out


def floor_map(spec: FrameSpec, height: int, width: int) -> np.ndarray:
    """Frame-wide scattered light at m = 1: a smooth random field with a ramp toward the
    sun, tinted like the widest lobe. Real veiling glare lifts the blacks, but not by one
    constant across the frame."""
    grid = np.asarray(spec.floor_grid, dtype=np.float32)
    field = np.asarray(Image.fromarray(grid).resize((width, height),
                                                              Image.Resampling.BICUBIC))
    field = np.clip(field, 0.0, 1.0)
    shape = (1.0 - FLOOR_FIELD_MIX) + FLOOR_FIELD_MIX * field

    cx, cy = spec.lobes[0].cx, spec.lobes[0].cy
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / math.hypot(width, height)
    ramp = 1.0 - FLOOR_RAMP * np.clip(dist, 0.0, 1.0)

    tint = np.asarray(max(spec.lobes, key=lambda lb: lb.alpha).tint, dtype=np.float32)
    return (spec.floor_f0 * shape * ramp)[:, :, None] * tint[None, None, :]


_METER_CACHE: dict[tuple[int, int], np.ndarray] = {}


def meter(linear: np.ndarray) -> float:
    """Centre-weighted mean luminance of what the sensor records (clipped at 1)."""
    h, w = linear.shape[:2]
    weights = _METER_CACHE.get((h, w))
    if weights is None:
        ys = (np.arange(h, dtype=np.float32) - (h - 1) / 2) / (0.5 * h)
        xs = (np.arange(w, dtype=np.float32) - (w - 1) / 2) / (0.5 * w)
        weights = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2 * 0.6 ** 2))
        weights /= weights.sum()
        _METER_CACHE[(h, w)] = weights
    clipped = np.clip(linear, 0.0, 1.0)
    luma = 0.2126 * clipped[:, :, 0] + 0.7152 * clipped[:, :, 1] + 0.0722 * clipped[:, :, 2]
    return float((luma * weights).sum())


def auto_exposure_gain(scene: np.ndarray, glared: np.ndarray, k: float) -> float:
    m_scene, m_glared = meter(scene), meter(glared)
    if m_glared <= 1e-6:
        return 1.0
    return float(np.clip((m_scene / m_glared) ** k, AE_MIN_GAIN, 1.0))


def add_noise(linear: np.ndarray, noise: tuple[float, float],
              rng: np.random.Generator) -> np.ndarray:
    a, b = noise
    var = a * np.clip(linear, 0.0, None) + b
    return linear + rng.standard_normal(linear.shape, dtype=np.float32) * np.sqrt(var)


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, subsampling=2,
                              optimize=False, progressive=False)
    return buf.getvalue()


def decode_jpeg(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def sat_frac(rgb: np.ndarray) -> float:
    return float((rgb.min(axis=2) >= SAT_LEVEL).mean())


def glare_light(spec: FrameSpec, height: int, width: int) -> np.ndarray:
    """Total added light before auto-exposure: lobes plus floor, both scaled by m(s)."""
    return spec.mult * (glare_lobes(spec, height, width) + floor_map(spec, height, width))


def render_pair(clean_rgb: np.ndarray, spec: FrameSpec) -> tuple[bytes, bytes, dict]:
    """uint8 sRGB in, two JPEGs out (clean, glared) plus measured quantities."""
    h, w = clean_rgb.shape[:2]
    scene = expand_highlights(srgb_to_linear(clean_rgb), spec.highlight_boost) * spec.exposure

    clean_lin = add_noise(scene, spec.noise_clean,
                          np_rng_for(spec.seed, spec.split, spec.image_id, "noise-clean"))
    clean_bytes = encode_jpeg(linear_to_srgb_u8(clean_lin), spec.jpeg_q_clean)

    total = scene + glare_light(spec, h, w)
    gain = auto_exposure_gain(scene, total, spec.ae_k)
    glared_lin = add_noise(total * gain, spec.noise_glared,
                           np_rng_for(spec.seed, spec.split, spec.image_id, "noise-glared"))
    glared_bytes = encode_jpeg(linear_to_srgb_u8(glared_lin), spec.jpeg_q_glared)

    info = {
        "ae_gain": gain,
        "sat_frac_clean": sat_frac(decode_jpeg(clean_bytes)),
        "sat_frac_glared": sat_frac(decode_jpeg(glared_bytes)),
    }
    return clean_bytes, glared_bytes, info


# --- assignment -----------------------------------------------------------------------

def assign_types(ids: list[str], split: str, seed: int) -> dict[str, str]:
    """Balanced: every type gets ~n/4 frames of this split."""
    ids = sorted(ids)
    types = list(itertools.islice(itertools.cycle(GLARE_TYPES), len(ids)))
    random.Random(f"{seed}:{split}:types").shuffle(types)
    return dict(zip(ids, types))


def assign_bands(types_by_id: dict[str, str], split: str, seed: int) -> dict[str, str]:
    """Within each type, a third of the frames go to each band."""
    bands: dict[str, str] = {}
    for gtype in GLARE_TYPES:
        members = sorted(i for i, t in types_by_id.items() if t == gtype)
        cycle = list(itertools.islice(itertools.cycle(BANDS), len(members)))
        random.Random(f"{seed}:{split}:bands:{gtype}").shuffle(cycle)
        bands.update(zip(members, cycle))
    return bands


def load_cutpoints(path: Path) -> dict[str, tuple[float, float]]:
    data = json.loads(path.read_text())
    return {t: (float(c["mild_max_s"]), float(c["severe_min_s"]))
            for t, c in data["cutpoints"].items()}


def band_range(band: str, cuts: tuple[float, float] | None) -> tuple[float, float]:
    """The s interval a band occupies. Without calibration, thirds of [0, 1]."""
    c1, c2 = cuts if cuts else (1 / 3, 2 / 3)
    return {"mild": (0.0, c1), "moderate": (c1, c2), "severe": (c2, 1.0)}[band]


def draw_severity(split: str, image_id: str, band: str, cuts, seed: int) -> float:
    lo, hi = band_range(band, cuts)
    return rng_for(seed, split, image_id, "severity").uniform(lo, hi)


# --- I/O ------------------------------------------------------------------------------

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def list_split(split: str) -> list[tuple[str, Path]]:
    img_dir, lbl_dir = KITTI / "images" / split, KITTI / "labels" / split
    if not img_dir.is_dir():
        sys.exit(f"error: {rel(img_dir)} not found (see CLAUDE.md section 5)")
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".png")
    missing = [p.stem for p in images if not (lbl_dir / f"{p.stem}.txt").is_file()]
    if missing:
        sys.exit(f"error: {len(missing)} {split} image(s) have no label, e.g. {missing[:5]}")
    return [(p.stem, p) for p in images]


def read_rgb(path: Path) -> np.ndarray:
    # Ultralytics' KITTI ".png" files are JPEGs; PIL sniffs the content, not the suffix.
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def prepare_layout(out_root: Path, splits: list[str]) -> None:
    """Ultralytics finds labels by swapping /images/ for /labels/, so each version gets
    a labels dir that symlinks to the untouched KITTI labels (glare moves no object)."""
    for version in VERSIONS:
        for split in splits:
            (out_root / version / "images" / split).mkdir(parents=True, exist_ok=True)
            link = out_root / version / "labels" / split
            link.parent.mkdir(parents=True, exist_ok=True)
            target = Path(os.path.relpath(KITTI / "labels" / split, link.parent))
            if link.is_symlink() and Path(os.readlink(link)) == target:
                continue
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target, target_is_directory=True)
        names = "\n".join(f"  {k}: {v}" for k, v in enumerate(
            ["car", "van", "truck", "pedestrian", "Person_sitting", "cyclist", "tram", "misc"]))
        (out_root / f"{version}.yaml").write_text(
            f"# generated by src/glare_synth.py -- do not edit by hand\n"
            f"path: {(out_root / version).resolve()}\n"
            f"train: images/train\nval: images/val\n\nnames:\n{names}\n")


def clear_stale(directory: Path, keep: set[str]) -> int:
    stale = [f for f in directory.glob("*.jpg") if f.stem not in keep]
    for f in stale:
        f.unlink()
    return len(stale)


def params_row(spec: FrameSpec, info: dict) -> dict:
    core = spec.lobes[0]
    return {
        "split": spec.split, "image_id": spec.image_id, "glare_type": spec.glare_type,
        "severity": f"{spec.severity:.6f}", "band": spec.band, "mult": f"{spec.mult:.6f}",
        "cx": f"{core.cx:.2f}", "cy": f"{core.cy:.2f}",
        "floor_f0": f"{spec.floor_f0:.6f}", "exposure": f"{spec.exposure:.6f}",
        "highlight_boost": f"{spec.highlight_boost:.6f}", "ae_k": f"{spec.ae_k:.6f}",
        "ae_gain": f"{info['ae_gain']:.6f}",
        "noise_a_clean": f"{spec.noise_clean[0]:.3e}", "noise_b_clean": f"{spec.noise_clean[1]:.3e}",
        "noise_a_glared": f"{spec.noise_glared[0]:.3e}",
        "noise_b_glared": f"{spec.noise_glared[1]:.3e}",
        "jpeg_q_clean": spec.jpeg_q_clean, "jpeg_q_glared": spec.jpeg_q_glared,
        "lobes": json.dumps([asdict(lb) for lb in spec.lobes], separators=(",", ":")),
    }


PARAMS_COLUMNS = [
    "split", "image_id", "glare_type", "severity", "band", "mult", "cx", "cy", "floor_f0",
    "exposure", "highlight_boost", "ae_k", "ae_gain", "noise_a_clean", "noise_b_clean",
    "noise_a_glared", "noise_b_glared", "jpeg_q_clean", "jpeg_q_glared", "lobes"]


@dataclass
class Task:
    split: str
    image_id: str
    src: Path
    glare_type: str
    severity: float
    band: str
    seed: int
    out_root: Path


def process(task: Task) -> dict:
    """Worker: render one pair, write both JPEGs, return manifest/params/checksum rows."""
    rgb = read_rgb(task.src)
    spec = sample_frame(rgb, task.split, task.image_id, task.glare_type,
                        task.severity, task.band, task.seed)
    clean_bytes, glared_bytes, info = render_pair(rgb, spec)

    out = {}
    for version, data in (("clean", clean_bytes), ("glared", glared_bytes)):
        path = task.out_root / version / "images" / task.split / f"{task.image_id}.jpg"
        path.write_bytes(data)
        out[version] = (rel(path), hashlib.sha256(data).hexdigest())

    return {
        "manifest": [
            [task.image_id, task.split, "0", "", "0.000000", f"{info['sat_frac_clean']:.6f}"],
            [task.image_id, task.split, "1", task.glare_type, f"{task.severity:.6f}",
             f"{info['sat_frac_glared']:.6f}"],
        ],
        "params": params_row(spec, info),
        "checksums": [out["clean"], out["glared"]],
    }


def save_examples(rows: list[dict], out_root: Path, results_dir: Path) -> list[Path]:
    """Per type: the val frames with the lowest, median and highest severity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for gtype in GLARE_TYPES:
        members = sorted((float(r["severity"]), r["image_id"]) for r in rows
                         if r["glare_type"] == gtype and r["split"] == "val")
        if not members:
            continue
        picks = [members[0], members[len(members) // 2], members[-1]]
        fig, axes = plt.subplots(3, 2, figsize=(14, 6.4))
        for row_axes, (s, image_id) in zip(axes, picks):
            for ax, version in zip(row_axes, VERSIONS):
                path = out_root / version / "images" / "val" / f"{image_id}.jpg"
                ax.imshow(read_rgb(path))
                ax.set_title(f"{image_id}  {version}" + (f"  s={s:.2f}" if version == "glared"
                                                         else ""), fontsize=9)
                ax.axis("off")
        fig.suptitle(f"{gtype}: lowest, median and highest severity in val", fontsize=11)
        fig.tight_layout()
        path = results_dir / f"glare_v2_example_{gtype}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", default="train,val",
                   help="comma-separated subset of train,val (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42, help="master seed (default: %(default)s)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--calibration", type=Path, default=CALIBRATION)
    p.add_argument("--uncalibrated", action="store_true",
                   help="draw s from thirds of [0,1] instead of calibrated bands. For "
                        "tuning only: the manifest it writes must not be committed")
    p.add_argument("--no-examples", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    args = p.parse_args(argv)
    args.splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    bad = [s for s in args.splits if s not in SPLITS]
    if bad:
        p.error(f"unknown split(s): {bad}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cuts = None
    if args.calibration.is_file():
        cuts = load_cutpoints(args.calibration)
        print(f"calibrated bands from {rel(args.calibration)}")
        for gtype, (c1, c2) in cuts.items():
            print(f"  {gtype:<15} mild s<{c1:.3f}  moderate <{c2:.3f}  severe <=1")
    elif args.uncalibrated or args.dry_run:
        print(f"WARNING: no {rel(args.calibration)}; bands are thirds of s (uncalibrated)")
    else:
        sys.exit(f"error: {rel(args.calibration)} not found. Run src/calibrate_severity.py "
                 f"first, or pass --uncalibrated for a tuning run.")

    tasks: list[Task] = []
    for split in args.splits:
        frames = list_split(split)
        types = assign_types([i for i, _ in frames], split, args.seed)
        bands = assign_bands(types, split, args.seed)
        for image_id, src in frames:
            gtype = types[image_id]
            s = draw_severity(split, image_id, bands[image_id],
                              cuts.get(gtype) if cuts else None, args.seed)
            tasks.append(Task(split, image_id, src, gtype, s, bands[image_id],
                              args.seed, args.out_root))
        print(f"{split}: {len(frames)} frames -> {len(frames)} clean + {len(frames)} glared")
        for gtype in GLARE_TYPES:
            counts = [sum(1 for i in types if types[i] == gtype and bands[i] == b)
                      for b in BANDS]
            print(f"    {gtype:<15} " + "  ".join(f"{b} {n:>4}" for b, n in zip(BANDS, counts)))

    if args.dry_run:
        print("\n--dry-run, nothing written. One sample per type:")
        for gtype in GLARE_TYPES:
            t = next(t for t in tasks if t.glare_type == gtype)
            spec = sample_frame(read_rgb(t.src), t.split, t.image_id, gtype,
                                t.severity, t.band, t.seed)
            print(f"  {t.split}/{t.image_id} {gtype:<15} s={spec.severity:.3f} ({spec.band}) "
                  f"m={spec.mult:.3f} e0={spec.exposure:.3f} ae_k={spec.ae_k:.2f} "
                  f"q={spec.jpeg_q_clean}/{spec.jpeg_q_glared}")
        return 0

    prepare_layout(args.out_root, args.splits)
    for split in args.splits:
        keep = {t.image_id for t in tasks if t.split == split}
        for version in VERSIONS:
            removed = clear_stale(args.out_root / version / "images" / split, keep)
            if removed:
                print(f"  removed {removed} stale file(s) from {version}/{split}")

    print(f"\nrendering {len(tasks)} pairs with {args.workers} worker(s)...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, res in enumerate(pool.map(process, tasks, chunksize=8), 1):
            results.append(res)
            if n % 500 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)}")

    manifest = sorted((r for res in results for r in res["manifest"]),
                      key=lambda r: (r[1], r[0], r[2]))
    params = sorted((res["params"] for res in results),
                    key=lambda r: (r["split"], r["image_id"]))
    checksums = sorted(c for res in results for c in res["checksums"])

    # A partial run (one split) must not drop the other split's committed rows.
    if set(args.splits) != set(SPLITS):
        manifest, params, checksums = merge_existing(args.splits, manifest, params, checksums)

    with MANIFEST.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(manifest)
    with PARAMS.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PARAMS_COLUMNS)
        writer.writeheader()
        writer.writerows(params)
    CHECKSUMS.write_text("".join(f"{h}  {p}\n" for p, h in checksums))
    print(f"\n  wrote {rel(MANIFEST)} ({len(manifest)} rows)  <-- commit")
    print(f"  wrote {rel(PARAMS)} ({len(params)} rows)  <-- commit")
    print(f"  wrote {rel(CHECKSUMS)} ({len(checksums)} files)  <-- commit")
    if cuts is None:
        print("  WARNING: uncalibrated run -- do not commit these three files")

    if not args.no_examples and "val" in args.splits:
        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        for path in save_examples(params, args.out_root, results_dir):
            print(f"  wrote {rel(path)}")

    print("\nNext: python src/shortcut_test.py")
    return 0


def merge_existing(splits, manifest, params, checksums):
    keep = lambda split: split not in splits  # noqa: E731
    if MANIFEST.is_file():
        old = [r for r in csv.reader(MANIFEST.open()) if r and r[0] != "image_id"]
        manifest = sorted(manifest + [r for r in old if keep(r[1])],
                          key=lambda r: (r[1], r[0], r[2]))
    if PARAMS.is_file():
        old = [r for r in csv.DictReader(PARAMS.open()) if keep(r["split"])]
        params = sorted(params + old, key=lambda r: (r["split"], r["image_id"]))
    if CHECKSUMS.is_file():
        old = []
        for line in CHECKSUMS.read_text().splitlines():
            h, p = line.split("  ", 1)
            parts = Path(p).parts               # data/v2/<version>/images/<split>/<id>.jpg
            if len(parts) >= 5 and keep(parts[4]):
                old.append((p, h))
        checksums = sorted(checksums + old)
    return manifest, params, checksums


if __name__ == "__main__":
    sys.exit(main())
