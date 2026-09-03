"""Pick the fixed KITTI subset that every arm of the experiment runs on.

Source is the Ultralytics KITTI redistribution (`datasets/kitti/`), which is already in
YOLO format — no label conversion, see CLAUDE.md §5. We draw from the `val` split
(1,496 frames of the official KITTI *training* set, so labels exist).

Outputs (CLAUDE.md §6):
    data/clean/         subset images, original filenames
    data/labels/        matching YOLO labels, copied verbatim (all 8 KITTI classes)
    data/subset_ids.txt committed — the contract with the gate and restoration teammates

The ID list is the authoritative artifact: once `data/subset_ids.txt` exists this script
reuses it instead of resampling, so re-running can never silently shift the subset out
from under a teammate. Pass --force to deliberately draw a new one.

Usage:
    python src/prepare_subset.py                 # 400 images, seed 42
    python src/prepare_subset.py -n 500 --force  # deliberately redraw
    python src/prepare_subset.py --stats-only    # inspect without copying
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# datasets/kitti/kitti.yaml. No DontCare — Ultralytics dropped it during conversion.
KITTI_NAMES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "Person_sitting",
    5: "cyclist",
    6: "tram",
    7: "misc",
}
# Collapse to the two COCO-comparable classes (CLAUDE.md §5). Applied at eval time, not
# here — reported below only so we can see the subset is worth evaluating on.
CAR_IDS = {0, 1}
PERSON_IDS = {3, 4}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "datasets" / "kitti",
                   help="Ultralytics KITTI root (default: %(default)s)")
    p.add_argument("--split", default="val", help="split to draw from (default: %(default)s)")
    p.add_argument("-n", "--count", type=int, default=400, help="subset size (default: %(default)s)")
    p.add_argument("--seed", type=int, default=42, help="sampling seed (default: %(default)s)")
    p.add_argument("--out-images", type=Path, default=REPO_ROOT / "data" / "clean")
    p.add_argument("--out-labels", type=Path, default=REPO_ROOT / "data" / "labels")
    p.add_argument("--ids-file", type=Path, default=REPO_ROOT / "data" / "subset_ids.txt")
    p.add_argument("--force", action="store_true",
                   help="resample even if --ids-file exists (changes the team-wide subset)")
    p.add_argument("--stats-only", action="store_true", help="report the subset, copy nothing")
    return p.parse_args(argv)


def index_split(dataset_root: Path, split: str) -> dict[str, tuple[Path, Path]]:
    """Map image ID -> (image path, label path) for every ID that has both."""
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    for d in (image_dir, label_dir):
        if not d.is_dir():
            sys.exit(f"error: {d} not found. Download the dataset first "
                     f"(see the download URL in {dataset_root / 'kitti.yaml'}).")

    pairs: dict[str, tuple[Path, Path]] = {}
    unlabelled: list[str] = []
    for img in sorted(image_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = label_dir / f"{img.stem}.txt"
        if label.is_file():
            pairs[img.stem] = (img, label)
        else:
            unlabelled.append(img.stem)

    if unlabelled:
        print(f"  note: skipped {len(unlabelled)} image(s) with no label file "
              f"(e.g. {', '.join(unlabelled[:3])})")
    if not pairs:
        sys.exit(f"error: no image/label pairs found under {image_dir}")
    return pairs


def choose_ids(available: dict[str, tuple[Path, Path]], args: argparse.Namespace) -> list[str]:
    """Reuse a committed ID list if present, else draw a fresh reproducible sample."""
    if args.ids_file.is_file() and not args.force:
        ids = [ln.strip() for ln in args.ids_file.read_text().splitlines() if ln.strip()]
        missing = [i for i in ids if i not in available]
        if missing:
            sys.exit(f"error: {len(missing)} ID(s) in {rel(args.ids_file)} are not in the "
                     f"{args.split} split (e.g. {', '.join(missing[:5])}). "
                     f"Check --split, or pass --force to redraw.")
        print(f"  reusing existing subset: {len(ids)} IDs from {rel(args.ids_file)}")
        if len(ids) != args.count:
            print(f"  note: that list holds {len(ids)} IDs, not the requested {args.count}. "
                  f"Pass --force to redraw at n={args.count}.")
        return ids

    if args.count > len(available):
        sys.exit(f"error: asked for {args.count} images but only {len(available)} are available")
    ids = sorted(random.Random(args.seed).sample(sorted(available), args.count))
    print(f"  sampled {len(ids)} IDs from {len(available)} {args.split} frames (seed {args.seed})")
    return ids


def summarise(ids: list[str], available: dict[str, tuple[Path, Path]]) -> None:
    per_class: Counter[int] = Counter()
    boxes_total = 0
    no_target = []
    for image_id in ids:
        classes = [int(float(ln.split()[0]))
                   for ln in available[image_id][1].read_text().splitlines() if ln.strip()]
        per_class.update(classes)
        boxes_total += len(classes)
        if not any(c in CAR_IDS or c in PERSON_IDS for c in classes):
            no_target.append(image_id)

    print(f"\n  {boxes_total} boxes across {len(ids)} images "
          f"({boxes_total / len(ids):.1f} per image)")
    for cid, name in KITTI_NAMES.items():
        print(f"    {cid} {name:<15} {per_class.get(cid, 0):>6}")

    cars = sum(per_class.get(c, 0) for c in CAR_IDS)
    people = sum(per_class.get(c, 0) for c in PERSON_IDS)
    print(f"  collapsed for eval (§5): car {cars}, person {people}, "
          f"ignored {boxes_total - cars - people}")
    if no_target:
        print(f"  warning: {len(no_target)} image(s) contain no car/person GT — they "
              f"contribute nothing to mAP (e.g. {', '.join(no_target[:5])})")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def copy_subset(ids: list[str], available: dict[str, tuple[Path, Path]],
                out_images: Path, out_labels: Path) -> None:
    """Copy the subset, and clear any files a previous, different subset left behind."""
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    keep = set(ids)
    stale = [p for d in (out_images, out_labels) for p in d.iterdir()
             if p.is_file() and p.name != ".gitkeep" and p.stem not in keep]
    for p in stale:
        p.unlink()
    if stale:
        print(f"  removed {len(stale)} stale file(s) from a previous subset")

    for image_id in ids:
        img, label = available[image_id]
        shutil.copy2(img, out_images / img.name)
        shutil.copy2(label, out_labels / label.name)
    print(f"  copied {len(ids)} images -> {rel(out_images)}/")
    print(f"  copied {len(ids)} labels -> {rel(out_labels)}/  (verbatim, all 8 classes)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"source: {rel(args.dataset_root / 'images' / args.split)}")
    available = index_split(args.dataset_root, args.split)
    ids = choose_ids(available, args)
    summarise(ids, available)

    if args.stats_only:
        print("\n--stats-only: nothing written.")
        return 0

    print()
    copy_subset(ids, available, args.out_images, args.out_labels)
    args.ids_file.parent.mkdir(parents=True, exist_ok=True)
    args.ids_file.write_text("\n".join(ids) + "\n")
    print(f"  wrote {rel(args.ids_file)}  <-- commit this; teammates must use the same IDs")

    print("\nNext (CLAUDE.md §9): src/glare_synth.py -> data/glared/ + data/glare_labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
