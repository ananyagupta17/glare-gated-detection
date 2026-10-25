"""v3 glare rendering: local, backlit glare with starbursts and lens-flare ghosts.

WHY. Measured on 40 real-glare frames from BDD100K TRAIN (results/real_glare_tuning_set.json),
real daytime sun glare is a bright, LOCAL region around the sun on a frame that stays dark
and contrasty: the camera exposes for the sun, so the rest of the scene is backlit
(darkest-pixel luma ~12, mean ~87, contrast ~68). v2 glare lifts and flattens the whole
frame (~50 / ~133 / ~54). A restorer trained on v2 learned to remove a frame-wide veil that
real glare does not have, and recovered only 7-10% of the real-glare detection loss.
Pre-registered in results/synth_v3_plan.json.

WHAT CHANGES (on top of v2's lobes, placement and camera model, all reused):
  * all glare light is multiplied by a localisation envelope around the sun,
        E(r) = (1 + (r / (LOC * diagonal))^2) ^ -LOC_POWER
  * the frame-wide floor (black lift) is scaled by FLOOR_SCALE and also localised
  * backlit exposure: on top of v2's auto-exposure, the SCENE (not the glare light) is
    stopped down by exp(-BACKLIT * s), so stronger glare darkens the scene while the sun
    and its bloom stay blown out, as in real backlit frames; GLARE_GAIN sets bloom strength
  * sun starburst: thin radial rays from the sun (6-12 rays), when the sun is in frame
  * lens-flare ghosts: 2-5 soft coloured discs on the line from the sun through the
    frame centre, when the sun is in frame
The four v3 globals are fitted to the tuning-set statistics (src/fit_synth_v3.py,
results/synth_v3_fit.json). The clean version is rendered exactly as in v2.

DETERMINISM. Every v3 draw comes from its own per-frame stream ("glare-v3"), so v2's
draws (lobes, placement, camera) are identical for the same frame.

USE. Set GLARE_SYNTH_VERSION=v3 before running src/glare_synth.py: glare_synth then
installs this renderer and writes to data/v3/ and the *_v3 manifest/params/checksums,
leaving v2 untouched. Worker processes inherit the variable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIT = REPO_ROOT / "results" / "synth_v3_fit.json"

PARAMS = {                       # defaults; replaced by the fitted values if present
    "LOC": 0.25,                 # envelope radius, fraction of the frame diagonal
    "LOC_POWER": 1.0,
    "FLOOR_SCALE": 0.15,
    "BACKLIT": 0.5,
    "STARBURST_AMP": 0.25,       # linear light, at s = 1
    "GHOST_AMP": 0.05,
    "GLARE_GAIN": 1.0,           # bloom strength of the local glare light
}
P_STARBURST, P_GHOSTS = 0.7, 0.6
if FIT.is_file():
    PARAMS.update(json.loads(FIT.read_text())["chosen"])

gs = None                        # the glare_synth module, set by install()


def _extras(spec, height: int, width: int) -> dict:
    """Starburst and ghost parameters for one frame, from the frame's own v3 stream."""
    rng = gs.rng_for(spec.seed, spec.split, spec.image_id, "glare-v3")
    cx, cy = spec.lobes[0].cx, spec.lobes[0].cy
    in_frame = 0 <= cx < width and 0 <= cy < height
    out = {"starburst": None, "ghosts": []}
    u_star, u_ghost = rng.random(), rng.random()
    n_rays = rng.choice((6, 8, 10, 12))
    rot = rng.uniform(0, 2 * math.pi)
    ray_w = rng.uniform(1.0, 2.5)
    ray_len = rng.uniform(0.06, 0.18)   # shortened after a visual check (rays crossed the frame)
    if in_frame and u_star < P_STARBURST:
        out["starburst"] = (n_rays, rot, ray_w, ray_len)
    n_ghost = rng.randint(2, 5)
    palette = [(0.55, 1.0, 0.6), (0.8, 0.55, 1.0), (1.0, 0.75, 0.45), (0.5, 0.85, 1.0)]
    ghosts = []
    for _ in range(n_ghost):
        ghosts.append((rng.uniform(0.4, 2.2), rng.uniform(5.0, 28.0), rng.uniform(0.3, 1.0),
                       palette[rng.randrange(len(palette))]))
    if in_frame and u_ghost < P_GHOSTS:
        out["ghosts"] = ghosts
    return out


def glare_light_v3(spec, height: int, width: int) -> np.ndarray:
    p = PARAMS
    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    cx, cy = spec.lobes[0].cx, spec.lobes[0].cy
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    diag = math.hypot(width, height)
    env = np.power(1.0 + (r / (p["LOC"] * diag)) ** 2, -p["LOC_POWER"])[:, :, None]

    light = spec.mult * (gs.glare_lobes(spec, height, width)
                         + p["FLOOR_SCALE"] * gs.floor_map(spec, height, width)) * env

    ex = _extras(spec, height, width)
    strength = spec.severity                     # starbursts/ghosts scale with severity
    if ex["starburst"]:
        n, rot, w, length = ex["starburst"]
        theta = np.arctan2(ys - cy, xs - cx) - rot
        step = 2 * math.pi / n
        d = np.abs((theta + step / 2) % step - step / 2)
        perp = r * np.sin(d)
        rays = np.exp(-(perp / w) ** 2) * np.power(1.0 + r / (length * diag), -2.0)
        light += (p["STARBURST_AMP"] * strength * rays)[:, :, None] * np.float32([1.0, 0.97, 0.9])
    if ex["ghosts"]:
        mx, my = width / 2.0, height / 2.0
        for t, rad, a, col in ex["ghosts"]:
            gx, gy = cx + t * (mx - cx), cy + t * (my - cy)
            rr = np.sqrt((xs - gx) ** 2 + (ys - gy) ** 2)
            disc = 1.0 / (1.0 + np.exp((rr - rad) / max(rad * 0.15, 1.0)))
            light += (p["GHOST_AMP"] * strength * a * disc)[:, :, None] * np.float32(col)
    return light


def render_pair_v3(clean_rgb: np.ndarray, spec) -> tuple[bytes, bytes, dict]:
    """As glare_synth.render_pair (clean identical), glared version with v3 light."""
    h, w = clean_rgb.shape[:2]
    scene = gs.expand_highlights(gs.srgb_to_linear(clean_rgb), spec.highlight_boost) * spec.exposure
    clean_lin = gs.add_noise(scene, spec.noise_clean,
                             gs.np_rng_for(spec.seed, spec.split, spec.image_id, "noise-clean"))
    clean_bytes = gs.encode_jpeg(gs.linear_to_srgb_u8(clean_lin), spec.jpeg_q_clean)

    # Backlit exposure: the camera exposes for the sun, so the SCENE is stopped down while
    # the sun and its bloom stay bright (fit round 2; darkening everything, round 1, left
    # the sun unsaturated and the frame flat).
    light = PARAMS["GLARE_GAIN"] * glare_light_v3(spec, h, w)
    backlit = math.exp(-PARAMS["BACKLIT"] * spec.severity)
    total = scene * backlit + light
    gain = gs.auto_exposure_gain(scene, scene + light, spec.ae_k)
    glared_lin = gs.add_noise(total * gain, spec.noise_glared,
                              gs.np_rng_for(spec.seed, spec.split, spec.image_id, "noise-glared"))
    glared_bytes = gs.encode_jpeg(gs.linear_to_srgb_u8(glared_lin), spec.jpeg_q_glared)
    return clean_bytes, glared_bytes, {
        "ae_gain": gain,
        "sat_frac_clean": gs.sat_frac(gs.decode_jpeg(clean_bytes)),
        "sat_frac_glared": gs.sat_frac(gs.decode_jpeg(glared_bytes)),
    }


def install(module) -> None:
    """Point a glare_synth module at the v3 renderer and the v3 output paths."""
    global gs
    gs = module
    module.render_pair = render_pair_v3
    module.OUT_ROOT = REPO_ROOT / "data" / "v3"
    module.MANIFEST = REPO_ROOT / "data" / "glare_manifest_v3.csv"
    module.PARAMS = REPO_ROOT / "data" / "glare_params_v3.csv"
    module.CHECKSUMS = REPO_ROOT / "data" / "glare_v3_sha256.txt"
    module.SYNTH_VERSION = "v3"
