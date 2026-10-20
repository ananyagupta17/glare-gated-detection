"""C1 detector, retrained on the proper split: all 5,985 clean KITTI train frames.

train_baseline.py trained on 1,198 frames carved out of KITTI val and left the 5,985-frame
train split unused. This script trains on the train split instead, so every one of the
1,496 val frames stays held out for evaluation.

WHAT IS GUARANTEED BEFORE TRAINING STARTS

  * Every training image resolves under datasets/kitti/images/train/ and has a label.
  * No training ID appears in KITTI val, and none appears in the glare manifest or in
    data/glared_full/ -- the detector never sees a glared frame (CLAUDE.md section 3: C1
    only answers "how much does glare hurt?" if the detector was never taught glare).
  * The dataset counts are printed and written to the JSON before the first epoch.

WHY VAL IS NOT USED FOR CHECKPOINT SELECTION

Ultralytics normally validates every epoch and keeps best.pt by val mAP. Picking the best
epoch on the evaluation set is a mild form of fitting to it, so per-epoch validation is
off (val=False) and the saved weights are last.pt, the epoch-50 model. Ultralytics still
runs one validation pass after the final epoch; it is printed but not used for anything.

This script only trains. Evaluation (clean vs glared, paired) is a separate step.

    python src/train_c1_kitti.py                 # 50 epochs, seed 42, auto device
    python src/train_c1_kitti.py --device 0      # team GPU
    python src/train_c1_kitti.py --dry-run       # checks and counts only
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import KITTI_NAMES, pick_device, rel, write_yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
KITTI = REPO_ROOT / "datasets" / "kitti"
RUN_NAME = "c1_train_split"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--weights", type=str, default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None, help="mps / cpu / 0; auto-detected")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-out", type=Path, default=REPO_ROOT / "models" / "yolo_c1_train.pt")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "results" / "c1_train_split.json")
    p.add_argument("--manifest", type=Path,
                   default=REPO_ROOT / "data" / "glare_labels_full.csv")
    p.add_argument("--dry-run", action="store_true", help="run the checks, do not train")
    return p.parse_args(argv)


def ids_in(directory: Path, suffix: str) -> set[str]:
    return {f.stem for f in directory.glob(f"*{suffix}")} if directory.is_dir() else set()


def check_split() -> dict:
    """Verify the train split is clean, labelled and disjoint from everything we evaluate."""
    train_img = KITTI / "images" / "train"
    train_ids = ids_in(train_img, ".png")
    label_ids = ids_in(KITTI / "labels" / "train", ".txt")
    val_ids = ids_in(KITTI / "images" / "val", ".png")
    val_label_ids = ids_in(KITTI / "labels" / "val", ".txt")

    problems = []
    if not train_ids:
        problems.append(f"no images in {rel(train_img)}")
    if train_ids - label_ids:
        problems.append(f"{len(train_ids - label_ids)} train images without labels")
    if train_ids & val_ids:
        problems.append(f"{len(train_ids & val_ids)} IDs in both train and val")

    glared_ids = ids_in(REPO_ROOT / "data" / "glared_full", ".png") \
        | ids_in(REPO_ROOT / "data" / "glared", ".png")
    if train_ids & glared_ids:
        problems.append(f"{len(train_ids & glared_ids)} train IDs also exist as glared frames")
    # Symlinks or copies smuggled into the train folder would resolve elsewhere.
    foreign = [f.name for f in train_img.glob("*.png")
               if f.resolve().parent != train_img.resolve()]
    if foreign:
        problems.append(f"{len(foreign)} train images resolve outside {rel(train_img)}")
    if problems:
        sys.exit("error: train split failed checks:\n  " + "\n  ".join(problems))

    boxes = Counter()
    for f in (KITTI / "labels" / "train").glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                boxes[KITTI_NAMES[int(line.split()[0])]] += 1

    manifest_ids = set()
    manifest = REPO_ROOT / "data" / "glare_labels_full.csv"
    if manifest.is_file():
        manifest_ids = {r["image_id"] for r in csv.DictReader(manifest.open())}

    return {
        "n_train_images": len(train_ids),
        "n_val_images": len(val_ids),
        "n_val_labels": len(val_label_ids),
        "train_boxes_per_class": {n: boxes.get(n, 0) for n in KITTI_NAMES.values()},
        "n_train_boxes": sum(boxes.values()),
        "glared_images_in_train": 0,
        "train_val_overlap": 0,
        "manifest_ids_in_train": len(train_ids & manifest_ids),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.device = pick_device(args.device)

    counts = check_split()
    if counts["manifest_ids_in_train"]:
        sys.exit(f"error: {counts['manifest_ids_in_train']} glare-manifest IDs are in train")

    print(f"C1 retrain on KITTI train  |  device {args.device}")
    print(f"  train images  {counts['n_train_images']}  (clean only, 0 glared)")
    print(f"  train boxes   {counts['n_train_boxes']}")
    for name, n in counts["train_boxes_per_class"].items():
        print(f"    {name:<15}{n:>7}")
    print(f"  val images    {counts['n_val_images']}  (held out, not trained on)")
    print(f"  train/val ID overlap 0, glared frames in train 0")

    data_yaml = write_yaml(REPO_ROOT / "datasets" / "kitti_train_c1.yaml", KITTI,
                           "images/train", "images/val")
    print(f"  dataset.yaml -> {rel(data_yaml)}")

    result = {
        "status": "training",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "train_source": "kitti-train", "train_on": "clean",
            "weights": args.weights, "epochs": args.epochs, "batch": args.batch,
            "imgsz": args.imgsz, "seed": args.seed, "device": args.device,
            "n_train_images": counts["n_train_images"],
            "n_val_images": counts["n_val_images"],
            "classes": list(KITTI_NAMES.values()),
            "checkpoint": "last.pt (val not used for epoch selection)",
            "data_yaml": rel(data_yaml),
        },
        "dataset_checks": counts,
        "model_out": rel(args.model_out),
    }

    if args.dry_run:
        print("\n--dry-run: checks passed, nothing trained or written")
        return 0
    write_json(args.json_out, result)
    print(f"  wrote {rel(args.json_out)}")

    from ultralytics import YOLO

    print(f"\ntraining {args.weights} for {args.epochs} epochs "
          f"(imgsz {args.imgsz}, batch {args.batch})...")
    t0 = time.time()
    model = YOLO(args.weights)
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, seed=args.seed, deterministic=True,
                val=False, project=str(REPO_ROOT / "runs" / "train"), name=RUN_NAME,
                exist_ok=True, verbose=True)

    last = Path(model.trainer.last)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(last, args.model_out)
    print(f"  saved {rel(last)} -> {rel(args.model_out)}")

    result.update(status="complete",
                  finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  train_seconds=round(time.time() - t0, 1))
    write_json(args.json_out, result)
    print(f"wrote {rel(args.json_out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
