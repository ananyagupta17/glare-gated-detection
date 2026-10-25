"""Paired training data on real BDD100K driving frames: clean BDD frame + v2 synthetic glare.

    python src/make_bdd_pairs.py                 # 3,000 pairs -> data/bdd_pairs/

WHY. The trained daytime restorer learned only KITTI frames and recovered just ~7-10% of
the real-glare loss on BDD100K (results/real_bdd_results.md). Real glare frames cannot be
training targets (no glare-free twin exists), but real BDD frames can be, with our
synthetic glare added, so the restorer learns BDD's camera, resolution and scenes.

NO OVERLAP WITH THE TEST SET. Frames come from the mirror's BDD TRAIN folders
(images/100k/train and images/10k/train); any frame whose name also appears in BDD VAL
(where the real-glare test set lives) is dropped.

CLEAN TARGETS MUST BE CLEAN. Night frames are dropped by brightness (mean luma >= 70 and
upper-third mean >= 90, since the 2018 labels file for train is 1.4 GB), and any frame
the CNN gate scores p(glare) >= 0.3 is dropped, so real glare does not end up as a
"clean" target. Then 3,000 frames are sampled (seed 42).

Frames are resized to KITTI's height (375 px, 667 x 375) first, so the glare covers the
same share of the frame as on KITTI. Glare: the same v2 camera pipeline as KITTI
(src/glare_synth.py): types balanced, severity
stratified uniform on [0, 1], per-frame seeds from ("bdd-train", file name). Both versions
go through the same camera steps, as for KITTI.

Output: data/bdd_pairs/{clean,glared}/<name>.jpg, data/bdd_pairs/pairs.csv
"""

from __future__ import annotations

import csv
import hashlib
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import glare_synth as gs  # noqa: E402

MIRROR = Path.home() / ".cache/kagglehub/datasets/solesensei/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images"
SOURCES = [MIRROR / "100k" / "train", MIRROR / "10k" / "train"]
VAL = REPO_ROOT / "datasets" / "bdd100k" / "images" / "100k" / "val"
OUT = REPO_ROOT / "data" / ("bdd_pairs_v3" if gs.SYNTH_VERSION == "v3" else "bdd_pairs")
N_PAIRS, SEED, SPLIT_TAG = 3000, 42, "bdd-train"
LUMA_MIN, UPPER_MIN, GATE_MAX = 70.0, 90.0, 0.3
KITTI_H = 375                      # frames are resized to this height before adding glare


def stats(path: str) -> tuple[str, float, float]:
    img = cv2.imread(path)
    if img is None:
        return path, -1.0, -1.0
    g = cv2.cvtColor(cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return path, float(g.mean()), float(g[:60].mean())


def render(task: tuple) -> list:
    path, name, gtype, s, k = task
    rgb = gs.read_rgb(Path(path))
    # Glare sizes in glare_synth.py are in pixels for KITTI's 375-px-tall frames; BDD is
    # 720 tall, so without this the glare covers proportionally far less of the frame.
    h, w = rgb.shape[:2]
    rgb = np.ascontiguousarray(cv2.resize(rgb, (round(w * KITTI_H / h), KITTI_H),
                                          interpolation=cv2.INTER_AREA))
    spec = gs.sample_frame(rgb, SPLIT_TAG, name, gtype, s, f"stratum {k}", SEED)
    clean, glared, info = gs.render_pair(rgb, spec)
    (OUT / "clean" / f"{name}.jpg").write_bytes(clean)
    (OUT / "glared" / f"{name}.jpg").write_bytes(glared)
    return [name, Path(path).parent.parent.name + "/" + Path(path).parent.name, gtype, f"{s:.6f}",
            f"{info['ae_gain']:.4f}", hashlib.sha256(glared).hexdigest()[:16]]


def main() -> int:
    import torch
    import train_gate_cnn
    val_names = {p.name for p in VAL.iterdir()}
    pool = sorted({p for d in SOURCES for p in d.glob("*.jpg") if p.name not in val_names},
                  key=lambda p: p.name)
    seen, uniq = set(), []
    for p in pool:                                     # a name in both folders: keep one
        if p.name not in seen:
            seen.add(p.name); uniq.append(p)
    print(f"{len(uniq)} BDD train frames not in BDD val")

    with ProcessPoolExecutor(7) as ex:
        st = list(ex.map(stats, [str(p) for p in uniq], chunksize=64))
    day = [Path(p) for p, m, u in st if m >= LUMA_MIN and u >= UPPER_MIN]
    print(f"{len(day)} pass the daytime brightness filter")

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    gate = torch.load(REPO_ROOT / "models" / "gate_cnn_mixed.pt", weights_only=False, map_location=dev).eval()
    keep = []
    for i in range(0, len(day), 64):
        batch = day[i:i + 64]
        x = torch.from_numpy(np.stack([train_gate_cnn.to_input(cv2.imread(str(p))) for p in batch]))
        with torch.no_grad():
            p = torch.sigmoid(gate(x.permute(0, 3, 1, 2).to(dev))).cpu().numpy()
        keep += [q for q, pp in zip(batch, p) if pp < GATE_MAX]
    print(f"{len(keep)} have gate p(glare) < {GATE_MAX}")
    if len(keep) < N_PAIRS:
        sys.exit(f"error: only {len(keep)} usable frames, need {N_PAIRS}")

    pick = sorted(random.Random(f"{SEED}:bdd-pairs").sample(keep, N_PAIRS), key=lambda p: p.name)
    names = [p.stem for p in pick]
    types = gs.assign_types(names, SPLIT_TAG, SEED)
    strata = gs.assign_strata(types, SPLIT_TAG, SEED)
    tasks = []
    for p in pick:
        k, n = strata[p.stem]
        tasks.append((str(p), p.stem, types[p.stem], gs.draw_severity(SPLIT_TAG, p.stem, k, n, SEED), k))
    for v in ("clean", "glared"):
        (OUT / v).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(7) as ex:
        rows = list(ex.map(render, tasks, chunksize=8))
    with (OUT / "pairs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "source", "glare_type", "severity", "ae_gain", "glared_sha256_16"])
        w.writerows(sorted(rows))
    print(f"wrote {len(rows)} pairs to {OUT.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
