"""Draw detections on clean vs glared versions of the same frame, side by side.

C1 says glare costs 41.8 mAP points. That is a number a panel has to take on faith. This
turns it into something visible: the same scene twice, boxes drawn, with the objects the
detector loses marked in red.

WHY THERE IS NO PER-FRAME mAP

mAP is an average of average precision over a whole dataset. On a single image it is
noise -- one missed car swings it wildly, and a frame with three objects cannot produce a
number comparable to one with twenty. Labelling each panel "mAP 0.64" / "mAP 0.22" would
be quoting a dataset statistic as if it were a property of that frame.

So each panel reports what is actually true of that frame -- how many ground-truth objects
were found at IoU >= 0.5 -- and the dataset-level figures appear once, in the caption,
labelled as dataset-level. The per-frame count is also the honest thing to point at during
a demo, because it is what the viewer can verify by looking.

WHAT THE COLOURS MEAN

  green solid    a ground-truth object the detector found
  red dashed     a ground-truth object it MISSED -- this is the glare damage
  amber dotted   a false positive: something detected that is not really there

Red boxes appearing in the lower panel and not the upper one is the entire story of C1.

FRAME SELECTION

Frames are drawn from the 298 held-out IDs the detector never trained on, restricted to
those that were glared (so the same scene exists in both conditions -- a paired
comparison). Selection is seeded and stratified to cover all four glare types rather than
picking the most damaged frames, because cherry-picking the worst cases is exactly what a
reviewer will suspect. Pass --seed to redraw a different sample.

Detection uses conf 0.25, the threshold a deployed system would run at -- not the 0.001
used for mAP computation, which would bury the image under hundreds of junk boxes.

Usage:
    python src/visualize_detections.py
    python src/visualize_detections.py --layout side    # left/right instead of stacked
    python src/visualize_detections.py --seed 7         # a different 5 frames
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

KITTI_NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian",
               4: "Person_sitting", 5: "cyclist", 6: "tram", 7: "misc"}
GLARE_TYPES = ["direct_sun", "horizon_streak", "diffuse_haze", "veiling_wash"]

FOUND = "#1FA35C"
MISSED = "#E03131"
GHOST = "#D08700"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "yolo_baseline.pt")
    p.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "glare_labels_full.csv")
    p.add_argument("--glared-dir", type=Path, default=REPO_ROOT / "data" / "glared_full")
    p.add_argument("--clean-dir", type=Path, default=None)
    p.add_argument("--labels-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "kitti" / "labels" / "val")
    p.add_argument("--baseline", type=Path,
                   default=REPO_ROOT / "results" / "baseline_c1_metrics.json")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--conf", type=float, default=0.25,
                   help="deployment threshold, not the 0.001 used for mAP")
    p.add_argument("--iou", type=float, default=0.5, help="IoU for calling a box a match")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--layout", choices=["stacked", "side"], default="stacked",
                   help="KITTI frames are ~1242x375, so stacked is far more legible")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    return p.parse_args(argv)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_gt(label_path: Path, width: int, height: int) -> list[dict]:
    """YOLO normalised `class cx cy w h` -> pixel xyxy."""
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:5])
        boxes.append({
            "cls": cls,
            "xyxy": ((cx - bw / 2) * width, (cy - bh / 2) * height,
                     (cx + bw / 2) * width, (cy + bh / 2) * height),
        })
    return boxes


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(preds: list[dict], gts: list[dict], thr: float) -> dict:
    """Greedy IoU matching, highest confidence first, same class required."""
    unmatched = list(range(len(gts)))
    hits, ghosts = [], []
    for pred in sorted(preds, key=lambda p: -p["conf"]):
        best, best_iou = None, thr
        for gi in unmatched:
            if gts[gi]["cls"] != pred["cls"]:
                continue
            score = iou(pred["xyxy"], gts[gi]["xyxy"])
            if score >= best_iou:
                best, best_iou = gi, score
        if best is None:
            ghosts.append(pred)
        else:
            unmatched.remove(best)
            hits.append(pred)
    return {"hits": hits, "ghosts": ghosts, "missed": [gts[i] for i in unmatched]}


def draw_panel(ax, image: np.ndarray, result: dict, title: str, subtitle: str) -> None:
    ax.imshow(image)
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    def box(xyxy, colour, style, width, label=None):
        x1, y1, x2, y2 = xyxy
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour,
            linewidth=width, linestyle=style))
        if label:
            ax.text(x1, y1 - 3, label, color="white", fontsize=6.5, weight="bold",
                    va="bottom", ha="left",
                    bbox=dict(facecolor=colour, edgecolor="none", pad=1.1))

    for pred in result["ghosts"]:
        box(pred["xyxy"], GHOST, (0, (1, 1.6)), 1.1)
    for pred in result["hits"]:
        box(pred["xyxy"], FOUND, "solid", 1.5,
            f"{KITTI_NAMES.get(pred['cls'], pred['cls'])} {pred['conf']:.2f}")
    for gt in result["missed"]:
        box(gt["xyxy"], MISSED, (0, (3.5, 2)), 1.9,
            f"MISSED {KITTI_NAMES.get(gt['cls'], gt['cls'])}")

    ax.set_title(title, fontsize=11, weight="bold", loc="left", pad=6)
    ax.text(0.0, -0.045, subtitle, transform=ax.transAxes, fontsize=8.6,
            va="top", ha="left", color="#333333")
    ax.axis("off")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.model.is_file():
        sys.exit(f"error: {rel(args.model)} not found. Run src/train_baseline.py first.")
    if args.clean_dir is None:
        candidate = REPO_ROOT / "data" / "clean_full"
        args.clean_dir = candidate if candidate.is_dir() else \
            REPO_ROOT / "datasets" / "kitti" / "images" / "val"

    from train_baseline import load_rows, stratified_split
    from ultralytics import YOLO

    records = load_rows(args.csv, args.clean_dir, args.glared_dir, args.labels_dir)
    by_id = {r["id"]: r for r in records}
    _, heldout = stratified_split(records, args.val_frac, args.seed)

    # Paired comparison needs both versions of the same scene, so only glared IDs qualify.
    pool = sorted(i for i in heldout if by_id[i]["is_glared"])
    if not pool:
        sys.exit("error: no glared frames in the held-out split")

    # Cover every glare type before filling remaining slots -- not the worst frames.
    rng = random.Random(args.seed)
    by_type: dict[str, list[str]] = {}
    for image_id in pool:
        by_type.setdefault(by_id[image_id]["glare_type"], []).append(image_id)
    chosen = [rng.choice(by_type[t]) for t in GLARE_TYPES if t in by_type][:args.n]
    if len(chosen) < args.n:
        rest = [i for i in pool if i not in chosen]
        chosen += rng.sample(rest, min(args.n - len(chosen), len(rest)))
    chosen.sort(key=lambda i: (by_id[i]["glare_type"], i))

    dataset_line = ""
    if args.baseline.is_file():
        import json
        paired = json.loads(args.baseline.read_text())["heldout"]["paired"]
        dataset_line = (f"Dataset-level (298 held-out frames, paired):  "
                        f"clean mAP@50 {paired['clean']['mAP50']:.4f}   "
                        f"glared {paired['glared']['mAP50']:.4f}   "
                        f"drop {paired['mAP_drop']:.4f}")

    model = YOLO(str(args.model))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"visualising {len(chosen)} held-out frames  |  conf {args.conf}  "
          f"IoU {args.iou}  |  device {args.device}")

    for slot, image_id in enumerate(chosen):
        rec = by_id[image_id]
        panels = []
        for label, path in (("clean", rec["clean"]), ("glared", rec["glared"])):
            with Image.open(path) as im:
                array = np.asarray(im.convert("RGB"))
            out = model.predict(array[:, :, ::-1], device=args.device, imgsz=args.imgsz,
                                conf=args.conf, verbose=False)[0]
            preds = [{"cls": int(c), "conf": float(f), "xyxy": tuple(map(float, b))}
                     for c, f, b in zip(out.boxes.cls.tolist(), out.boxes.conf.tolist(),
                                        out.boxes.xyxy.tolist())]
            gts = load_gt(rec["label"], array.shape[1], array.shape[0])
            panels.append((label, array, match(preds, gts, args.iou), len(gts)))

        stacked = args.layout == "stacked"
        fig, axes = plt.subplots(2 if stacked else 1, 1 if stacked else 2,
                                 figsize=(13, 8.2) if stacked else (19, 4.4))
        for ax, (label, array, res, n_gt) in zip(np.ravel(axes), panels):
            found = len(res["hits"])
            head = (f"{image_id} — CLEAN" if label == "clean"
                    else f"{image_id} — GLARED ({rec['glare_type']})")
            sub = (f"{found} of {n_gt} ground-truth objects found  "
                   f"·  {len(res['missed'])} missed  "
                   f"·  {len(res['ghosts'])} false positives")
            draw_panel(ax, array, res, head, sub)

        clean_found = len(panels[0][2]["hits"])
        glared_found = len(panels[1][2]["hits"])
        lost = clean_found - glared_found
        fig.suptitle(f"{rec['glare_type']}  —  detector loses {lost} of "
                     f"{clean_found} objects it found on the clean frame",
                     fontsize=13, weight="bold", y=0.985)
        legend = ("green = found   ·   red dashed = MISSED (glare damage)   "
                  "·   amber dotted = false positive")
        fig.text(0.5, 0.045, legend, ha="center", fontsize=8.8, color="#444444")
        if dataset_line:
            fig.text(0.5, 0.018, dataset_line, ha="center", fontsize=8.2, color="#666666")

        fig.tight_layout(rect=(0, 0.065, 1, 0.965))
        out_path = args.out_dir / f"detection_visual_{slot}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  {rel(out_path)}   {rec['glare_type']:<15} "
              f"clean {clean_found}/{panels[0][3]}  ->  glared {glared_found}/{panels[1][3]}"
              f"   (lost {lost})")

    print(f"\nwrote {len(chosen)} figure(s) to {rel(args.out_dir)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
