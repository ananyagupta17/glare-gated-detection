"""Cut glare severity into mild / moderate / severe by measured detection damage.

Severity s in [0, 1] is an input to the synthesis. Whether a given s "barely hurts" or
"wrecks" detection is an empirical question, and the answer differs by glare type, so the
band edges are measured instead of guessed:

  1. Sample N KITTI TRAIN frames (seeded), balanced across the four glare types.
  2. Render each frame clean and at every s on a grid, through glare_synth's pipeline.
  3. Score each rendering with a detector: per-frame F1 at IoU 0.5, conf 0.25,
     class-aware greedy matching against the KITTI labels.
  4. Per type, the mean F1 drop (clean minus glared) as a function of s, made monotone
     with a running max, then interpolated:
         mild     drop < MILD_MAX_DROP
         severe   drop >= SEVERE_MIN_DROP
         moderate in between

WHY TRAIN FRAMES, AND WHY THIS DETECTOR

The val split is the test set for every arm, so nothing about the bands may be fitted on
it. The calibration frames are KITTI train frames, and their IDs are recorded in the
output.

The detector must also not have trained on the calibration frames: a detector that saw a
frame during training scores it unrealistically well when clean, which distorts the drop.
The new C1 detector (models/yolo_c1_train.pt) trained on all 5,985 train frames, so it is
disqualified. The default is models/yolo_baseline.pt, which trained on 1,198 frames of
KITTI VAL and never saw a train frame. It never sees a val frame here either: only its
weights are used, and only train frames are scored. Override with --detector, and the
choice and the weights' sha256 are recorded either way.

The two thresholds are fixed here, before any run, so the bands cannot be tuned after
seeing results:
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glare_synth as gs  # noqa: E402

REPO_ROOT = gs.REPO_ROOT
MILD_MAX_DROP = 0.05       # mean per-frame F1 drop below which glare counts as mild
SEVERE_MIN_DROP = 0.25     # mean per-frame F1 drop at or above which it counts as severe
IOU_MATCH = 0.5
CONF = 0.25
S_GRID = tuple(round(0.1 * i, 1) for i in range(11))     # 0.0, 0.1, ..., 1.0
SPLIT_TAG = "calib-train"  # seeds differ from the generation run's train draws

__doc__ += f"""
    MILD_MAX_DROP   = {MILD_MAX_DROP}
    SEVERE_MIN_DROP = {SEVERE_MIN_DROP}

Usage:
    python src/calibrate_severity.py                 # 480 train frames, yolo_baseline.pt
    python src/calibrate_severity.py --dry-run       # print the plan only
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-frames", type=int, default=480, help="divisible by 4 for balance")
    p.add_argument("--detector", type=Path, default=REPO_ROOT / "models" / "yolo_baseline.pt")
    p.add_argument("--detector-note", type=str,
                   default="trained on 1,198 KITTI val frames (train_baseline.py, val-split); "
                           "never saw a KITTI train frame")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=gs.parse_args([]).workers)
    p.add_argument("--out", type=Path, default=gs.CALIBRATION)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def read_labels(path: Path, width: int, height: int) -> np.ndarray:
    """YOLO normalised labels -> (n, 5) [cls, x1, y1, x2, y2] in pixels."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c, cx, cy, w, h = map(float, line.split()[:5])
        rows.append([c, (cx - w / 2) * width, (cy - h / 2) * height,
                     (cx + w / 2) * width, (cy + h / 2) * height])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def frame_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred (n, 6) [cls, x1, y1, x2, y2, conf]; gt (m, 5). Greedy by confidence."""
    if len(gt) == 0 and len(pred) == 0:
        return 1.0
    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    for p in pred[np.argsort(-pred[:, 5])]:
        cand = np.where((gt[:, 0] == p[0]) & ~used)[0]
        if len(cand) == 0:
            continue
        overlaps = iou(p[1:5], gt[cand, 1:5])
        best = int(np.argmax(overlaps))
        if overlaps[best] >= IOU_MATCH:
            used[cand[best]] = True
            tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    return 2 * tp / max(2 * tp + fp + fn, 1)


def render_levels(task: tuple) -> tuple[str, str, list[bytes]]:
    """Worker: one frame rendered clean and at every s on the grid."""
    image_id, src, gtype, seed = task
    rgb = gs.read_rgb(Path(src))
    out = []
    clean_bytes = None
    for s in S_GRID:
        spec = gs.sample_frame(rgb, SPLIT_TAG, image_id, gtype, s, "calibration", seed)
        c, g, _ = gs.render_pair(rgb, spec)
        clean_bytes = clean_bytes or c       # the clean version does not depend on s
        out.append(g)
    return image_id, gtype, [clean_bytes] + out


def interpolate_crossing(curve: list[float], level: float) -> float:
    """Smallest s where the monotone curve reaches level, linear between grid points."""
    for i, d in enumerate(curve):
        if d >= level:
            if i == 0:
                return 0.0
            d0 = curve[i - 1]
            return S_GRID[i - 1] + (level - d0) / (d - d0) * (S_GRID[i] - S_GRID[i - 1])
    return float("nan")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_frames % len(gs.GLARE_TYPES):
        sys.exit("error: --n-frames must be divisible by 4 so each type gets equal frames")

    train = gs.list_split("train")              # never "val": val is the test set
    ids = sorted(i for i, _ in train)
    src = dict(train)
    chosen = sorted(random.Random(f"{args.seed}:calibration").sample(ids, args.n_frames))
    types = gs.assign_types(chosen, SPLIT_TAG, args.seed)

    print(f"calibration: {len(chosen)} KITTI train frames x {len(S_GRID)} severities "
          f"(+ clean) = {len(chosen) * (len(S_GRID) + 1)} renders")
    print(f"  detector {gs.rel(args.detector)}: {args.detector_note}")
    print(f"  thresholds: mild drop < {MILD_MAX_DROP}, severe drop >= {SEVERE_MIN_DROP}")
    if args.dry_run:
        print("--dry-run, nothing rendered or written")
        return 0
    if not args.detector.is_file():
        sys.exit(f"error: {gs.rel(args.detector)} not found")

    from ultralytics import YOLO
    from train_baseline import pick_device
    device = pick_device(args.device)
    model = YOLO(str(args.detector))

    f1 = {gtype: [] for gtype in gs.GLARE_TYPES}   # per type: list of per-frame rows
    t0 = time.time()
    tasks = [(i, str(src[i]), types[i], args.seed) for i in chosen]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (image_id, gtype, renders) in enumerate(pool.map(render_levels, tasks), 1):
            imgs = [gs.decode_jpeg(b)[:, :, ::-1].copy() for b in renders]   # RGB -> BGR
            h, w = imgs[0].shape[:2]
            gt = read_labels(gs.KITTI / "labels" / "train" / f"{image_id}.txt", w, h)
            preds = model.predict(imgs, conf=CONF, device=device, verbose=False)
            scores = []
            for r in preds:
                b = r.boxes
                pred = np.concatenate([b.cls.cpu().numpy()[:, None], b.xyxy.cpu().numpy(),
                                       b.conf.cpu().numpy()[:, None]], axis=1) \
                    if len(b) else np.zeros((0, 6), dtype=np.float32)
                scores.append(frame_f1(pred, gt))
            f1[gtype].append([image_id] + scores)
            if n % 40 == 0 or n == len(tasks):
                print(f"  {n}/{len(tasks)}  ({time.time() - t0:.0f}s)")

    cutpoints, curves, problems = {}, {}, []
    for gtype, rows in f1.items():
        arr = np.asarray([r[1:] for r in rows], dtype=np.float64)
        drop = (arr[:, :1] - arr[:, 1:]).mean(axis=0)            # per s
        monotone = np.maximum.accumulate(drop).tolist()
        c1 = interpolate_crossing(monotone, MILD_MAX_DROP)
        c2 = interpolate_crossing(monotone, SEVERE_MIN_DROP)
        curves[gtype] = {"mean_f1_clean": round(float(arr[:, 0].mean()), 4),
                         "mean_drop": [round(float(d), 4) for d in drop],
                         "monotone_drop": [round(float(d), 4) for d in monotone]}
        if monotone[0] >= MILD_MAX_DROP:
            problems.append(f"{gtype}: drop at s=0 is {monotone[0]:.3f}, already past mild; "
                            f"lower SEVERITY_SCALE[{gtype!r}][0]")
        if not np.isfinite(c2):
            problems.append(f"{gtype}: drop at s=1 is {monotone[-1]:.3f}, never severe; "
                            f"raise SEVERITY_SCALE[{gtype!r}][1]")
        cutpoints[gtype] = {"mild_max_s": round(c1, 4), "severe_min_s": round(c2, 4)}

    sha = hashlib.sha256(args.detector.read_bytes()).hexdigest()
    result = {
        "status": "ok" if not problems else "failed",
        "problems": problems,
        "thresholds": {"mild_max_drop": MILD_MAX_DROP, "severe_min_drop": SEVERE_MIN_DROP},
        "metric": f"per-frame F1, IoU {IOU_MATCH}, conf {CONF}, class-aware greedy matching; "
                  f"drop = F1(clean) - F1(glared), mean over frames, running max over s",
        "s_grid": list(S_GRID),
        "severity_scale": {k: list(v) for k, v in gs.SEVERITY_SCALE.items()},
        "cutpoints": cutpoints,
        "curves": curves,
        "detector": {"path": gs.rel(args.detector), "sha256": sha, "note": args.detector_note,
                     "device": device},
        "data_use": {
            "detector_training_data": "1,198 frames of the KITTI val split "
                                      "(train_baseline.py --train-source val-split)",
            "frames_scored": "KITTI train split only (listed under frames.ids_by_type)",
            "val_test_frames_used_for_calibration": False,
            "statement": "The calibration detector was trained on part of val. It scored "
                         "only KITTI train frames here; no val (test) frame was rendered, "
                         "scored or otherwise used to set the severity bands.",
        },
        "frames": {"split": "train", "n": len(chosen), "seed": args.seed,
                   "seed_split_tag": SPLIT_TAG,
                   "ids_by_type": {t: sorted(r[0] for r in rows) for t, rows in f1.items()}},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n{'type':<16}{'F1 clean':>9}{'mild <':>9}{'severe >=':>11}")
    for gtype, c in cutpoints.items():
        print(f"{gtype:<16}{curves[gtype]['mean_f1_clean']:>9.3f}"
              f"{c['mild_max_s']:>9.3f}{c['severe_min_s']:>11.3f}")
    print(f"wrote {gs.rel(args.out)}")
    if problems:
        # A failed calibration must not be picked up by glare_synth.py.
        failed = args.out.with_name(args.out.stem + "_failed.json")
        args.out.rename(failed)
        print(f"\nFAILED -- moved to {gs.rel(failed)}:\n  " + "\n  ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
