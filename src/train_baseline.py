"""C1 baseline: train a detector on CLEAN frames, then measure what glare costs it.

C1 is the arm that answers "how much does glare hurt?" (CLAUDE.md section 2). That
question only has an answer if the detector has never been taught to cope with glare, so
training here is clean-only by default. Training on glared frames instead makes the model
glare-invariant, collapses the measured drop toward zero, and turns the result into the
section 12 counter-argument the gate is supposed to beat ("just augment your training
data"). --train-on mixed runs that variant deliberately; it is an augmentation baseline,
not C1.

WHAT IS MEASURED, AND WHAT IS TRUSTWORTHY

The detector trains on 80% of the 1496 val IDs, so evaluating on all 1496 scores the
model partly on its own training data. Both figures are reported and labelled:

  heldout   the 20% (300 IDs) never seen in training. THIS IS THE DEFENSIBLE NUMBER.
  full      all 1496, as a convenience. Inflated by construction -- ~80% of it is
            training data. Never quote it without the caveat.

Even 'full' glared images were not literally seen (training used clean frames), but the
underlying scenes were, so the model has memorised their layout and object placement.
Contamination is reduced, not removed.

PAIRED VS UNPAIRED

Splitting by the is_glared column compares two DIFFERENT sets of scenes, so scene
difficulty is confounded with glare. Because every glared frame also has its clean
original, this script additionally reports a PAIRED comparison: the same held-out frames
scored clean, then glared. Same scenes, same objects, glare the only variable. That is
the cleanest estimate of glare damage and is reported as `paired`.

A stronger setup exists and is one flag away: --train-source kitti-train trains on the
untouched 5985-image KITTI train split, which leaves ALL 1496 val frames genuinely held
out (187 per glare type instead of ~37). It costs ~5x the training time.

Classes are KITTI's native 8. Training on KITTI removes the COCO class-mapping problem in
section 5 entirely, so no collapse to car/person is needed. Rare classes
(Person_sitting, tram, misc) have few instances and noisy AP; per-class output is
reported so that is visible rather than hidden inside the mean.

Usage:
    python src/train_baseline.py                      # train + evaluate
    python src/train_baseline.py --epochs 1           # smoke test
    python src/train_baseline.py --skip-train         # re-evaluate an existing model
    python src/train_baseline.py --train-on mixed     # augmentation baseline, not C1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

KITTI_NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian",
               4: "Person_sitting", 5: "cyclist", 6: "tram", 7: "misc"}

GLARE_TYPES = ["direct_sun", "horizon_streak", "diffuse_haze", "veiling_wash"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "glare_labels_full.csv")
    p.add_argument("--glared-dir", type=Path, default=REPO_ROOT / "data" / "glared_full")
    p.add_argument("--clean-dir", type=Path, default=None,
                   help="clean images; defaults to data/clean_full if present, else "
                        "datasets/kitti/images/val (they are byte-identical)")
    p.add_argument("--labels-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "kitti" / "labels" / "val")
    p.add_argument("--work-dir", type=Path, default=REPO_ROOT / "datasets" / "full_dataset")
    p.add_argument("--eval-dir", type=Path, default=REPO_ROOT / "datasets" / "eval_subsets")
    p.add_argument("--model-out", type=Path, default=REPO_ROOT / "models" / "yolo_baseline.pt")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "results" / "baseline_c1_metrics.json")
    p.add_argument("--weights", type=str, default="yolov8n.pt")
    p.add_argument("--train-on", choices=["clean", "mixed"], default="clean",
                   help="clean = C1 baseline (default). mixed = augmentation baseline.")
    p.add_argument("--train-source", choices=["val-split", "kitti-train"], default="val-split",
                   help="val-split = 80/20 of the 1496 (default). kitti-train = the "
                        "untouched 5985-image train split, leaving all 1496 held out.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None, help="mps / cpu / 0; auto-detected")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-train", action="store_true",
                   help="evaluate an existing --model-out instead of training")
    return p.parse_args(argv)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_rows(csv_path: Path, clean_dir: Path, glared_dir: Path, labels_dir: Path):
    """Read the glare CSV and resolve every image ID to its clean and glared files."""
    if not csv_path.is_file():
        sys.exit(f"error: {rel(csv_path)} not found. Run src/glare_synth.py --full first.")

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        sys.exit(f"error: {rel(csv_path)} is empty")

    missing_clean, missing_glared, missing_label = [], [], []
    records = []
    for r in rows:
        image_id = r["image_id"]
        is_glared = r["is_glared"] == "1"
        clean = clean_dir / f"{image_id}.png"
        glared = glared_dir / f"{image_id}.png"
        label = labels_dir / f"{image_id}.txt"
        if not clean.is_file():
            missing_clean.append(image_id)
        if is_glared and not glared.is_file():
            missing_glared.append(image_id)
        if not label.is_file():
            missing_label.append(image_id)
        records.append(dict(id=image_id, is_glared=is_glared,
                            glare_type=r.get("glare_type", ""),
                            clean=clean, glared=glared, label=label))

    for name, miss in (("clean image", missing_clean), ("glared image", missing_glared),
                       ("label", missing_label)):
        if miss:
            sys.exit(f"error: {len(miss)} {name}(s) missing, e.g. {', '.join(miss[:5])}")

    return records


def stratified_split(records, val_frac: float, seed: int):
    """Split by (is_glared, glare_type) so both sides keep the same composition."""
    buckets = defaultdict(list)
    for rec in records:
        buckets[(rec["is_glared"], rec["glare_type"])].append(rec["id"])

    rng = random.Random(seed)
    train_ids, val_ids = set(), set()
    for key in sorted(buckets, key=str):
        ids = sorted(buckets[key])
        rng.shuffle(ids)
        cut = round(len(ids) * val_frac)
        val_ids.update(ids[:cut])
        train_ids.update(ids[cut:])
    return train_ids, val_ids


def link_set(dest_images: Path, dest_labels: Path, chosen, by_id, use_glared_for) -> int:
    """Populate an Ultralytics images/labels pair with symlinks (no image is duplicated)."""
    for d in (dest_images, dest_labels):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for image_id in sorted(chosen):
        rec = by_id[image_id]
        src = rec["glared"] if image_id in use_glared_for else rec["clean"]
        (dest_images / f"{image_id}.png").symlink_to(src.resolve())
        (dest_labels / f"{image_id}.txt").symlink_to(rec["label"].resolve())
    return len(chosen)


def write_yaml(path: Path, root: Path, train_rel: str, val_rel: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {k}: {v}" for k, v in KITTI_NAMES.items())
    path.write_text(
        f"# generated by src/train_baseline.py -- do not edit by hand\n"
        f"path: {root.resolve()}\n"
        f"train: {train_rel}\n"
        f"val: {val_rel}\n\n"
        f"names:\n{names}\n"
    )
    return path


def build_train_dataset(args, records, by_id, train_ids, val_ids) -> Path:
    """Lay out datasets/full_dataset/{images,labels}/{train,val} and its dataset.yaml."""
    root = args.work_dir

    if args.train_on == "mixed":
        # Use each ID's assigned condition: glared where the CSV says so, else clean.
        use_glared = {r["id"] for r in records if r["is_glared"]}
    else:
        use_glared = set()          # C1: the detector never sees glare during training

    n_train = link_set(root / "images" / "train", root / "labels" / "train",
                       train_ids, by_id, use_glared)
    # Training-time val is clean-only too, so the monitored curve matches the train
    # distribution. The real clean-vs-glared comparison is done separately below.
    n_val = link_set(root / "images" / "val", root / "labels" / "val",
                     val_ids, by_id, use_glared)

    glared_in_train = len(train_ids & use_glared)
    print(f"  train {n_train} images ({n_train - glared_in_train} clean, "
          f"{glared_in_train} glared) | val {n_val}")

    return write_yaml(root / "dataset.yaml", root, "images/train", "images/val")


def build_eval_subset(args, name: str, ids, by_id, glared: bool) -> Path | None:
    """One evaluation subset as its own tiny dataset, so Ultralytics computes the mAP."""
    if not ids:
        return None
    root = args.eval_dir / name
    use_glared = set(ids) if glared else set()
    link_set(root / "images" / "val", root / "labels" / "val", ids, by_id, use_glared)
    return write_yaml(root / "dataset.yaml", root, "images/val", "images/val")


def evaluate(model, yaml_path: Path, args) -> dict:
    """Run Ultralytics validation and pull out overall + per-class numbers."""
    metrics = model.val(data=str(yaml_path), split="val", device=args.device,
                        imgsz=args.imgsz, batch=args.batch, verbose=False,
                        plots=False, save_json=False,
                        project=str(REPO_ROOT / "runs" / "val"), name=yaml_path.parent.name,
                        exist_ok=True)
    box = metrics.box
    per_class = {}
    # ap_class_index lists which classes actually had ground truth in this subset; the
    # per-class arrays are indexed positionally against it, not by class id.
    for pos, cls_id in enumerate(getattr(box, "ap_class_index", [])):
        try:
            per_class[KITTI_NAMES[int(cls_id)]] = {
                "AP50": round(float(box.ap50[pos]), 4),
                "AP50_95": round(float(box.ap[pos]), 4),
            }
        except (IndexError, TypeError):
            continue
    return {
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "per_class": per_class,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.clean_dir is None:
        candidate = REPO_ROOT / "data" / "clean_full"
        args.clean_dir = candidate if candidate.is_dir() else \
            REPO_ROOT / "datasets" / "kitti" / "images" / "val"
    args.device = pick_device(args.device)

    records = load_rows(args.csv, args.clean_dir, args.glared_dir, args.labels_dir)
    by_id = {r["id"]: r for r in records}
    print(f"C1 baseline  |  {len(records)} images  |  device {args.device}")
    print(f"  clean from  {rel(args.clean_dir)}")
    print(f"  glared from {rel(args.glared_dir)}")
    print(f"  train-on={args.train_on}  train-source={args.train_source}")

    train_ids, val_ids = stratified_split(records, args.val_frac, args.seed)
    print(f"\nsplit (seed {args.seed}, stratified by is_glared x glare_type):")
    data_yaml = build_train_dataset(args, records, by_id, train_ids, val_ids)

    if args.train_source == "kitti-train":
        kitti = REPO_ROOT / "datasets" / "kitti"
        data_yaml = write_yaml(args.work_dir / "dataset_kitti_train.yaml", kitti,
                               "images/train", "images/val")
        train_ids = set()                        # nothing from val was trained on
        val_ids = {r["id"] for r in records}
        print("  --train-source kitti-train: training on the 5985-image split; "
              "all 1496 val frames are held out")
    print(f"  dataset.yaml -> {rel(data_yaml)}")

    from ultralytics import YOLO

    if args.skip_train:
        if not args.model_out.is_file():
            sys.exit(f"error: --skip-train but {rel(args.model_out)} does not exist")
        print(f"\n--skip-train: loading {rel(args.model_out)}")
        model = YOLO(str(args.model_out))
    else:
        print(f"\ntraining {args.weights} for {args.epochs} epochs "
              f"(imgsz {args.imgsz}, batch {args.batch})...")
        model = YOLO(args.weights)
        model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, seed=args.seed,
                    project=str(REPO_ROOT / "runs" / "train"), name="c1_baseline",
                    exist_ok=True, verbose=True)
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        best = Path(model.trainer.best)
        shutil.copy2(best, args.model_out)
        print(f"  saved {rel(best)} -> {rel(args.model_out)}")
        model = YOLO(str(args.model_out))

    # ---- evaluation -------------------------------------------------------------
    glared_ids = {r["id"] for r in records if r["is_glared"]}
    clean_ids = {r["id"] for r in records if not r["is_glared"]}
    type_of = {r["id"]: r["glare_type"] for r in records}

    scopes = {
        "heldout": val_ids,
        "full": {r["id"] for r in records},
    }

    results: dict = {"config": {
        "train_on": args.train_on, "train_source": args.train_source,
        "epochs": args.epochs, "weights": args.weights, "device": args.device,
        "seed": args.seed, "n_images": len(records),
        "n_train": len(train_ids), "n_heldout": len(val_ids),
        "classes": list(KITTI_NAMES.values()),
    }}

    for scope, scope_ids in scopes.items():
        s_clean = sorted(clean_ids & scope_ids)
        s_glared = sorted(glared_ids & scope_ids)
        if not s_clean or not s_glared:
            continue
        print(f"\n=== {scope}: {len(s_clean)} clean vs {len(s_glared)} glared ===")

        block: dict = {}
        y = build_eval_subset(args, f"{scope}_clean", s_clean, by_id, glared=False)
        block["clean"] = {**evaluate(model, y, args), "count": len(s_clean)}
        y = build_eval_subset(args, f"{scope}_glared", s_glared, by_id, glared=True)
        block["glared"] = {**evaluate(model, y, args), "count": len(s_glared)}
        block["mAP_drop"] = round(block["clean"]["mAP50"] - block["glared"]["mAP50"], 4)

        # Paired: the same frames scored clean, then glared. Glare is the only variable.
        y = build_eval_subset(args, f"{scope}_paired_clean", s_glared, by_id, glared=False)
        paired_clean = evaluate(model, y, args)
        block["paired"] = {
            "clean": {**paired_clean, "count": len(s_glared)},
            "glared": block["glared"],
            "mAP_drop": round(paired_clean["mAP50"] - block["glared"]["mAP50"], 4),
            "note": "same frames scored clean then glared; scene content controlled",
        }

        per_type = {}
        for gtype in GLARE_TYPES:
            t_ids = sorted(i for i in s_glared if type_of[i] == gtype)
            y = build_eval_subset(args, f"{scope}_{gtype}", t_ids, by_id, glared=True)
            if y is None:
                continue
            m = evaluate(model, y, args)
            y_c = build_eval_subset(args, f"{scope}_{gtype}_clean", t_ids, by_id, glared=False)
            m_clean = evaluate(model, y_c, args)
            per_type[gtype] = {**m, "count": len(t_ids),
                               "paired_clean_mAP50": m_clean["mAP50"],
                               "mAP_drop": round(m_clean["mAP50"] - m["mAP50"], 4)}
        block["per_type"] = per_type
        results[scope] = block

    # Top-level keys in the requested shape, taken from the held-out (defensible) scope.
    head = results.get("heldout") or results.get("full") or {}
    if head:
        results["clean_mAP50"] = head["clean"]["mAP50"]
        results["glared_mAP50"] = head["glared"]["mAP50"]
        results["mAP_drop"] = head["mAP_drop"]
        results["per_type"] = {k: {"mAP50": v["mAP50"], "count": v["count"]}
                               for k, v in head["per_type"].items()}
        results["scope_of_top_level_keys"] = "heldout" if "heldout" in results else "full"

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n")

    # ---- summary ----------------------------------------------------------------
    print("\n" + "=" * 68)
    for scope in ("heldout", "full"):
        if scope not in results:
            continue
        b = results[scope]
        tag = "DEFENSIBLE (never trained on)" if scope == "heldout" \
            else "INFLATED (~80% was training data)"
        print(f"\n{scope.upper()}  -- {tag}")
        print(f"  Clean mAP = {b['clean']['mAP50']:.4f}, "
              f"Glared mAP = {b['glared']['mAP50']:.4f}, "
              f"Drop = {b['mAP_drop'] * 100:.2f}%")
        p = b["paired"]
        print(f"  paired (same frames): clean {p['clean']['mAP50']:.4f} -> "
              f"glared {p['glared']['mAP50']:.4f}, drop {p['mAP_drop'] * 100:.2f}%")
        print(f"  {'type':<16}{'n':>5}{'clean':>9}{'glared':>9}{'drop':>9}")
        for gtype, m in sorted(b["per_type"].items(), key=lambda kv: -kv[1]["mAP_drop"]):
            print(f"  {gtype:<16}{m['count']:>5}{m['paired_clean_mAP50']:>9.4f}"
                  f"{m['mAP50']:>9.4f}{m['mAP_drop']:>9.4f}")
    print("\n" + "=" * 68)
    print(f"wrote {rel(args.json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
