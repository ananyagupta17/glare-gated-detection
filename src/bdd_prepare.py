"""Build a real-glare test set from BDD100K: find sun-glare frames, review, convert labels.

BDD100K is used for TESTING ONLY; nothing here trains on it. Expected layout:

    datasets/bdd100k/images/100k/val/*.jpg            (10,000 images, 1280 x 720)
    datasets/bdd100k/labels/det_20/det_val.json       (Detection 2020 labels)

STEP 1 -- candidates. Ranks the daytime val frames by how glare-like they are and writes
contact sheets for human review:

    python src/bdd_prepare.py candidates              # top 400 by default
      -> data/bdd_review/candidates.csv   (rank, name, score, weather, scene, verdict)
      -> data/bdd_review/sheet_NN.jpg     (12 numbered thumbnails per sheet)

The glare score looks for what sun glare does to a frame: a large blown-out region in
the upper part of the image (the sun or its bloom) and lifted shadows (veiling light).
It only RANKS frames; every frame is confirmed by a person. Fill the `verdict` column
with `glare` or `no`. Split the sheets between team members and cross-check ~20 frames
so inter-rater agreement can be reported.

STEP 2 -- build. After review:

    python src/bdd_prepare.py build
      -> datasets/bdd_real/{glare,clean}/images/val/  (symlinks)
      -> datasets/bdd_real/{glare,clean}/labels/val/  (YOLO labels, KITTI class ids)
      -> data/bdd_real_manifest.csv

The clean set is the same size as the confirmed glare set, drawn (seeded) from daytime
frames with low glare scores, matched on BDD's weather x scene attributes so the two
sets differ as little as possible apart from the glare. They are still different
scenes; report that confound.

CLASS MAPPING -- fixed here before any evaluation (also written to
results/bdd_class_map.json). KITTI ids: 0 car, 2 truck, 3 pedestrian, 5 cyclist, 6 tram.
At evaluation the detector's 'van' (1) predictions are counted as car, because BDD
labels vans as cars. Bus, bicycle, motorcycle, traffic light and traffic sign have no
KITTI counterpart and are dropped. BDD 'rider' includes motorcyclists; noted.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BDD = REPO_ROOT / "datasets" / "bdd100k"
IMAGES = BDD / "images" / "100k" / "val"
LABELS_JSON = BDD / "labels" / "det_20" / "det_val.json"
REVIEW = REPO_ROOT / "data" / "bdd_review"
OUT = REPO_ROOT / "datasets" / "bdd_real"
MANIFEST = REPO_ROOT / "data" / "bdd_real_manifest.csv"
CLASS_MAP_JSON = REPO_ROOT / "results" / "bdd_class_map.json"
SEED = 42

CLASS_MAP = {"car": 0, "truck": 2, "pedestrian": 3, "rider": 5, "train": 6}
DROPPED = ["bus", "bicycle", "motorcycle", "traffic light", "traffic sign",
           "other person", "other vehicle", "trailer"]
EVAL_MERGE = {1: 0}            # detector 'van' predictions count as 'car' on BDD


def fail(msg: str) -> None:
    sys.exit(f"error: {msg}")


def load_labels() -> list[dict]:
    if not IMAGES.is_dir() or not LABELS_JSON.is_file():
        fail(f"BDD100K not found. Expected {IMAGES} and {LABELS_JSON} (see the docstring).")
    frames = json.loads(LABELS_JSON.read_text())
    return [f for f in frames if (IMAGES / f["name"]).is_file()]


def glare_score(path: str) -> float:
    """Higher = more glare-like. Ranking only; a person confirms every frame."""
    img = cv2.imread(path)
    if img is None:
        return -1.0
    small = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
    upper = small[: 120]                                         # top two thirds
    sat = (upper.min(axis=2) >= 245).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(sat, connectivity=8)
    blob = stats[1:, cv2.CC_STAT_AREA].max() / sat.size if n > 1 else 0.0
    luma = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lift = float(np.percentile(luma, 5)) / 255.0                 # veiling light lifts shadows
    contrast = float(luma.std()) / 128.0
    return float(3.0 * blob + 1.0 * lift - 0.5 * contrast)


def to_yolo(frame: dict, width: int = 1280, height: int = 720) -> list[str]:
    rows = []
    for lab in frame.get("labels") or []:
        cls = CLASS_MAP.get(lab.get("category"))
        box = lab.get("box2d")
        if cls is None or not box:
            continue
        x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w < 1 or h < 1:
            continue
        rows.append(f"{cls} {(x1 + w / 2) / width:.6f} {(y1 + h / 2) / height:.6f} "
                    f"{w / width:.6f} {h / height:.6f}")
    return rows


def contact_sheets(rows: list[dict], per_sheet: int = 12) -> int:
    REVIEW.mkdir(parents=True, exist_ok=True)
    for old in REVIEW.glob("sheet_*.jpg"):
        old.unlink()
    tw, th = 480, 270
    n_sheets = 0
    for s in range(0, len(rows), per_sheet):
        tiles = []
        for r in rows[s: s + per_sheet]:
            img = cv2.resize(cv2.imread(str(IMAGES / r["name"])), (tw, th))
            cv2.rectangle(img, (0, 0), (tw, 30), (0, 0, 0), -1)
            cv2.putText(img, f"#{r['rank']}  {r['name']}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) < per_sheet:
            tiles.append(np.zeros((th, tw, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[i: i + 3]) for i in range(0, per_sheet, 3)])
        cv2.imwrite(str(REVIEW / f"sheet_{s // per_sheet + 1:02d}.jpg"), grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        n_sheets += 1
    return n_sheets


def cmd_candidates(top: int, workers: int) -> int:
    frames = [f for f in load_labels() if (f.get("attributes") or {}).get("timeofday") == "daytime"]
    print(f"{len(frames)} daytime val frames; scoring...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(glare_score, [str(IMAGES / f["name"]) for f in frames], chunksize=32))
    ranked = sorted(zip(scores, frames), key=lambda t: (-t[0], t[1]["name"]))
    REVIEW.mkdir(parents=True, exist_ok=True)
    rows = [{"rank": k + 1, "name": f["name"], "score": f"{sc:.4f}",
             "weather": f["attributes"].get("weather", ""), "scene": f["attributes"].get("scene", ""),
             "verdict": ""} for k, (sc, f) in enumerate(ranked[:top])]
    with (REVIEW / "candidates.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (REVIEW / "all_daytime_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "score", "weather", "scene"])
        w.writerows([[f["name"], f"{sc:.4f}", f["attributes"].get("weather", ""),
                      f["attributes"].get("scene", "")] for sc, f in ranked])
    n = contact_sheets(rows)
    print(f"wrote {REVIEW / 'candidates.csv'} ({len(rows)} candidates) and {n} contact sheets")
    print("next: fill the verdict column with 'glare' or 'no', then run: bdd_prepare.py build")
    return 0


def cmd_build() -> int:
    cand = REVIEW / "candidates.csv"
    if not cand.is_file():
        fail("run `bdd_prepare.py candidates` and review first")
    reviewed = list(csv.DictReader(cand.open()))
    bad = [r["name"] for r in reviewed if r["verdict"].strip().lower() not in ("glare", "no", "")]
    if bad:
        fail(f"verdict must be 'glare' or 'no'; check {bad[:5]}")
    glare = [r for r in reviewed if r["verdict"].strip().lower() == "glare"]
    if not glare:
        fail("no frames marked 'glare' yet")
    unreviewed = sum(1 for r in reviewed if not r["verdict"].strip())
    frames = {f["name"]: f for f in load_labels()}

    # matched clean set: low glare score, same weather x scene mix, seeded
    scores = list(csv.DictReader((REVIEW / "all_daytime_scores.csv").open()))
    cand_names = {r["name"] for r in reviewed}
    pool = [r for r in scores if r["name"] not in cand_names]
    pool = pool[len(pool) // 2:]                                  # lower half of the ranking
    by_stratum = defaultdict(list)
    for r in pool:
        by_stratum[(r["weather"], r["scene"])].append(r["name"])
    rng = random.Random(f"{SEED}:bdd-clean")
    need = Counter((r["weather"], r["scene"]) for r in glare)
    clean, short = [], 0
    for stratum, k in sorted(need.items()):
        names = sorted(by_stratum.get(stratum, []))
        rng.shuffle(names)
        clean += names[:k]
        short += max(0, k - len(names))
    if short:                                                    # fill from any stratum
        rest = sorted(set(r["name"] for r in pool) - set(clean))
        rng.shuffle(rest)
        clean += rest[:short]

    rows = []
    for split_name, names in (("glare", [r["name"] for r in glare]), ("clean", clean)):
        img_dir = OUT / split_name / "images" / "val"
        lbl_dir = OUT / split_name / "labels" / "val"
        for d in (img_dir, lbl_dir):
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                f.unlink()
        for name in names:
            stem = Path(name).stem
            (img_dir / name).symlink_to((IMAGES / name).resolve())
            lines = to_yolo(frames[name])
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            a = frames[name].get("attributes") or {}
            rows.append([name, int(split_name == "glare"), a.get("weather", ""), a.get("scene", ""),
                         len(lines)])
    with MANIFEST.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "is_glared", "weather", "scene", "n_objects"])
        w.writerows(rows)
    CLASS_MAP_JSON.write_text(json.dumps({
        "bdd_to_kitti_id": CLASS_MAP, "kitti_names": {0: "car", 2: "truck", 3: "pedestrian",
                                                       5: "cyclist", 6: "tram"},
        "dropped_bdd_classes": DROPPED, "eval_prediction_merge": {"van (1)": "car (0)"},
        "note": "fixed before any BDD evaluation; BDD 'rider' includes motorcyclists",
    }, indent=2) + "\n")
    print(f"glare frames {len(glare)} (unreviewed candidates left: {unreviewed}), "
          f"matched clean frames {len(clean)}\nwrote {MANIFEST} and {OUT}/")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--top", type=int, default=400)
    c.add_argument("--workers", type=int, default=7)
    sub.add_parser("build")
    a = p.parse_args(argv)
    return cmd_candidates(a.top, a.workers) if a.cmd == "candidates" else cmd_build()


if __name__ == "__main__":
    sys.exit(main())
