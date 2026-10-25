"""Fit the v3 glare globals to real-glare statistics (rule in results/synth_v3_plan.json).

For each setting on a small grid, render v3 glare on 60 clean BDD100K TRAIN frames (not in
the tuning set, resized to KITTI's 375-px height, severity s ~ U(0.3, 1), types balanced),
measure {p1, p5, mean, std, sat, spread} of the glared frames, and score
    sum over stats of |median_synthetic - median_real| / IQR_real
against the 40 real-glare tuning frames (results/real_glare_tuning_set.json). The lowest
score wins. Writes results/synth_v3_fit.json.

    python src/fit_synth_v3.py
"""

from __future__ import annotations

import itertools
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

os.environ["GLARE_SYNTH_VERSION"] = "v3"
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import glare_synth as gs  # noqa: E402
import glare_synth_v3 as v3  # noqa: E402

TUNE = json.loads((REPO_ROOT / "results" / "real_glare_tuning_set.json").read_text())
MIRROR = Path.home() / ".cache/kagglehub/datasets/solesensei/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images"
# Round 1 (documented in results/synth_v3_fit_round1.json) searched LOC x FLOOR_SCALE x
# BACKLIT with backlit darkening applied to the whole frame; it matched brightness but left
# contrast and blown-out area far too low. Round 2: darkening applies to the scene only,
# FLOOR_SCALE fixed at 0 (it had no measurable effect), bloom strength added.
GRID = {"LOC": [0.08, 0.12, 0.2], "BACKLIT": [0.6, 1.2, 1.8], "GLARE_GAIN": [1.0, 2.5, 6.0]}
FIXED = {"FLOOR_SCALE": 0.0}
KEYS = ["p1", "p5", "mean", "std", "sat", "spread"]
N_FRAMES, SEED = 60, 42


def stats(img: np.ndarray) -> dict:
    """Same statistics as the tuning set (640x360, luma 0-255)."""
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(lum, (0, 0), 25)
    ys, xs = np.nonzero(blur > np.percentile(blur, 90))
    spread = float(np.hypot(xs.std(), ys.std())) / np.hypot(640, 360) if len(xs) else 0.0
    return {"p1": float(np.percentile(lum, 1)), "p5": float(np.percentile(lum, 5)),
            "mean": float(lum.mean()), "std": float(lum.std()),
            "sat": float((img.min(2) >= 245).mean()), "spread": spread}


def render(task):
    path, name, gtype, s, params = task
    v3.PARAMS.update(params)
    rgb = gs.read_rgb(Path(path))
    h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(cv2.resize(rgb, (round(w * 375 / h), 375), interpolation=cv2.INTER_AREA))
    spec = gs.sample_frame(rgb, "fit-v3", name, gtype, s, "fit", SEED)
    _, glared, _ = v3.render_pair_v3(rgb, spec)
    return stats(cv2.cvtColor(gs.decode_jpeg(glared), cv2.COLOR_RGB2BGR))


def main() -> int:
    tune = set(TUNE["frames"])
    val = {p.name for p in (REPO_ROOT / "datasets/bdd100k/images/100k/val").iterdir()}
    pool = sorted({p for d in (MIRROR / "100k" / "train", MIRROR / "10k" / "train")
                   for p in d.glob("*.jpg") if p.name not in tune and p.name not in val},
                  key=lambda p: p.name)
    day = []
    for p in pool:
        g = cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (320, 180)), cv2.COLOR_BGR2GRAY)
        if g.mean() >= 70 and g[:60].mean() >= 90:
            day.append(p)
        if len(day) >= 1500:
            break
    frames = sorted(random.Random(f"{SEED}:fit-v3").sample(day, N_FRAMES), key=lambda p: p.name)
    types = gs.assign_types([p.stem for p in frames], "fit-v3", SEED)
    sev = {p.stem: random.Random(f"{SEED}:fit-v3:{p.stem}").uniform(0.3, 1.0) for p in frames}
    target = TUNE["target_stats"]

    rows = []
    with ProcessPoolExecutor(7) as ex:
        for values in itertools.product(*GRID.values()):
            params = {**{k: v3.PARAMS[k] for k in ("LOC_POWER", "STARBURST_AMP", "GHOST_AMP")},
                      **FIXED, **dict(zip(GRID, values))}
            st = list(ex.map(render, [(str(p), p.stem, types[p.stem], sev[p.stem], params) for p in frames]))
            med = {k: float(np.median([s[k] for s in st])) for k in KEYS}
            score = sum(abs(med[k] - target[k]["median"]) / max(target[k]["q75"] - target[k]["q25"], 1e-6)
                        for k in KEYS)
            rows.append({**dict(zip(GRID, values)), "score": round(score, 3),
                         **{f"median_{k}": round(v, 4) for k, v in med.items()}})
            print(f"  {dict(zip(GRID, values))}  score {score:.2f}  " +
                  " ".join(f"{k} {med[k]:.3g}" for k in KEYS), flush=True)
    best = min(rows, key=lambda r: r["score"])
    chosen = {**{k: v3.PARAMS[k] for k in ("LOC_POWER", "STARBURST_AMP", "GHOST_AMP")},
              **FIXED, **{k: best[k] for k in GRID}}
    (REPO_ROOT / "results" / "synth_v3_fit.json").write_text(json.dumps({
        "rule": "results/synth_v3_plan.json fitting_rule", "n_frames": N_FRAMES,
        "frames": [p.name for p in frames], "target": target, "grid": GRID,
        "chosen": chosen, "best": best, "results": rows,
        "v2_reference": "v2 synthetic glare on BDD: p1 ~50, mean ~133, std ~54 (results/real_glare_tuning_set.json context)",
    }, indent=2) + "\n")
    print(f"chosen {chosen}  score {best['score']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
