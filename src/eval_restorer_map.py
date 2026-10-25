"""Detection mAP before and after restoration, on the v2 val pairs: all, severe, non-severe.

    python src/eval_restorer_map.py none          # detector on glared input only (C1)
    python src/eval_restorer_map.py tiling128     # Vihaan's handover tiling (src/restore.py)
    python src/eval_restorer_map.py overlap384    # src/restore_seamless.py, mode overlap
    python src/eval_restorer_map.py daytime       # src/restore_daytime.py (classical, daytime)
    python src/eval_restorer_map.py daynet        # src/restore_daytime_net.py (trained, daytime)
    python src/eval_restorer_map.py whole         # src/restore_seamless.py, mode whole

For every run the detector scores three versions of the same val frames:

  clean      the v2 clean versions               (reference, the upper bound)
  input      the v2 glared versions              (what the camera delivers)
  output     the restored glared versions        (skipped for `none`)

each as mAP@50 and mAP@50:95, over all frames, severe frames and non-severe frames (the
`severe` flag in data/glare_params_v2.csv; there is no mild band, see
results/severity_scale_decision.json). The metric code is train_baseline.evaluate(), the
Ultralytics validator call behind results/baseline_c1_metrics.json.

WHICH FRAMES

Any val frame used for severity calibration or restorer tuning is excluded before the
detector runs. The IDs come from results/severity_calibration.json and
results/restore_tuning.json; both must exist. Calibration uses KITTI train frames only,
so it normally excludes nothing from val. The tuning file lists the val frames that were
inspected to choose between restorer modes.

WHICH DETECTOR

models/yolo_c1_train.pt by default: trained on the KITTI train split, never on val.
models/yolo_baseline.pt is refused because it trained on 1,198 val frames.

COST

Restoration runs on CPU at roughly 6 s (whole) to 20-45 s (tiled) per frame. Outputs
are cached as lossless PNGs under data/restored/v2_<name>/, so an interrupted run
resumes. --limit N evaluates the first N eligible frames only, and the JSON says so.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESTORERS = ("none", "tiling128", "overlap384", "whole", "daytime", "daynet", "daynet_bdd")
GROUPS = ("all", "severe", "non_severe")
V2_ROOT = REPO_ROOT / "data" / "v2"
PARAMS = REPO_ROOT / "data" / "glare_params_v2.csv"
CALIBRATION = REPO_ROOT / "results" / "severity_calibration.json"
TUNING = REPO_ROOT / "results" / "restore_tuning.json"
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "yolo_c1_train.pt"
REFUSED_WEIGHTS = {"yolo_baseline.pt": "trained on 1,198 KITTI val frames"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("restorer", choices=RESTORERS)
    p.add_argument("--v2-root", type=Path, default=V2_ROOT)
    p.add_argument("--params", type=Path, default=PARAMS)
    p.add_argument("--calibration", type=Path, default=CALIBRATION)
    p.add_argument("--tuning", type=Path, default=TUNING)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--device", type=str, default=None, help="detector device; auto-detected")
    p.add_argument("--restore-threads", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--work-dir", type=Path, default=REPO_ROOT / "datasets" / "eval_restorer")
    p.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "restored")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args(argv)


def fail(message: str) -> None:
    sys.exit(f"error: {message}")


def v2_val_dirs(v2_root: Path) -> dict[str, Path]:
    """The v2 val folders this script reads. Missing -> refuse."""
    dirs = {v: v2_root / v / "images" / "val" for v in ("clean", "glared")}
    missing = [str(d) for d in dirs.values() if not d.is_dir()]
    if missing:
        fail(f"v2 val data not found: {', '.join(missing)}. Generate it with "
             f"src/glare_synth.py first.")
    return dirs


def excluded_ids(calibration: Path, tuning: Path) -> dict[str, set[str]]:
    """IDs used for calibration or tuning, by source. Both files are required."""
    for path in (calibration, tuning):
        if not path.is_file():
            fail(f"{path} not found. It lists frames that must not be evaluated, so the "
                 f"evaluation does not run without it.")
    cal = json.loads(calibration.read_text())
    tun = json.loads(tuning.read_text())
    cal_ids = {i for ids in cal["frames"]["ids_by_type"].values() for i in ids}
    return {
        "calibration": cal_ids if cal["frames"].get("split") == "val" else set(),
        "calibration_split": {cal["frames"].get("split", "?")},
        "tuning": set(tun["val_ids_used"]),
    }


def eligible(val_ids: list[str], excluded: dict[str, set[str]]) -> tuple[list[str], list[str]]:
    """Split val IDs into (evaluated, removed). Removed = used for calibration or tuning."""
    banned = excluded["calibration"] | excluded["tuning"]
    keep = [i for i in val_ids if i not in banned]
    return keep, sorted(set(val_ids) & banned)


def load_severe(params: Path) -> dict[str, bool]:
    """val image_id -> severe flag. Refuses an uncalibrated params file (empty flags)."""
    if not params.is_file():
        fail(f"{params} not found; it holds each frame's severe flag")
    rows = [r for r in csv.DictReader(params.open()) if r["split"] == "val"]
    if any(r.get("severe", "") not in ("0", "1") for r in rows):
        fail(f"{params} has missing severe flags: it was written without a calibration "
             f"file and cannot be used for evaluation")
    return {r["image_id"]: r["severe"] == "1" for r in rows}


def severity_groups(ids: list[str], severe: dict[str, bool]) -> dict[str, list[str]]:
    return {"all": ids,
            "severe": [i for i in ids if severe[i]],
            "non_severe": [i for i in ids if not severe[i]]}


def check_weights(weights: Path) -> None:
    if weights.name in REFUSED_WEIGHTS:
        fail(f"{weights.name} was {REFUSED_WEIGHTS[weights.name]}; evaluating it on val "
             f"would be contaminated. Use models/yolo_c1_train.pt.")
    if not weights.is_file():
        fail(f"{weights} not found. Wait for src/train_c1_kitti.py to finish.")


def make_restorer(name: str, threads: int):
    import os
    import torch
    torch.set_num_threads(threads)
    os.environ.setdefault("FLARE7K_DEVICE", "cpu")
    if name == "daytime":
        from types import SimpleNamespace
        import restore_daytime
        return SimpleNamespace(restore=restore_daytime.restore)
    if name == "daynet_bdd":
        os.environ.setdefault("DAYTIME_NET_DEVICE", "cpu")
        import restore_daytime_net
        return restore_daytime_net.DaytimeNetRestorer(
            REPO_ROOT / "models" / "restore_daytime_net_kitti_bdd.pt", device="cpu")
    if name == "daynet":
        os.environ.setdefault("DAYTIME_NET_DEVICE", "cpu")  # CPU, comparable with the others
        import restore_daytime_net
        return restore_daytime_net.get_restorer()
    if name == "tiling128":
        import restore
        return restore.get_restorer()                       # handover default, overlap 128
    from restore_seamless import SeamlessFlare7KRestorer
    from restore import DEFAULT_CHECKPOINT, DEFAULT_FLARE7K_REPO
    mode = "overlap" if name == "overlap384" else "whole"
    return SeamlessFlare7KRestorer(mode=mode, overlap=384, device=os.environ["FLARE7K_DEVICE"],
                                   checkpoint=os.getenv("FLARE7K_CHECKPOINT", str(DEFAULT_CHECKPOINT)),
                                   flare7k_repo=os.getenv("FLARE7K_REPO", str(DEFAULT_FLARE7K_REPO)))


def restore_all(name: str, ids: list[str], glared_dir: Path, cache: Path, threads: int) -> dict:
    """Restore every frame once; reuse cached outputs. Times only the restore call."""
    import cv2
    cache.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (cache / f"{i}.png").is_file()]
    times = []
    if todo:
        restorer = make_restorer(name, threads)
        for n, image_id in enumerate(todo, 1):
            image = cv2.imread(str(glared_dir / f"{image_id}.jpg"))
            t = time.perf_counter()
            out = restorer.restore(image)
            times.append(time.perf_counter() - t)
            cv2.imwrite(str(cache / f"{image_id}.png"), out)
            if n % 25 == 0 or n == len(todo):
                print(f"  restored {n}/{len(todo)}  (mean {sum(times) / len(times):.1f} s/frame)")
    return {"restored_now": len(todo), "from_cache": len(ids) - len(todo),
            "mean_restore_seconds": round(sum(times) / len(times), 2) if times else None}


def subset(work: Path, name: str, ids: list[str], image_dir: Path, suffix: str) -> Path:
    """An Ultralytics dataset of symlinks; labels are the untouched KITTI val labels."""
    from train_baseline import KITTI_NAMES, write_yaml
    root = work / name
    images, labels = root / "images" / "val", root / "labels" / "val"
    for d in (images, labels):
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)
    kitti_labels = REPO_ROOT / "datasets" / "kitti" / "labels" / "val"
    for image_id in ids:
        (images / f"{image_id}{suffix}").symlink_to((image_dir / f"{image_id}{suffix}").resolve())
        (labels / f"{image_id}.txt").symlink_to((kitti_labels / f"{image_id}.txt").resolve())
    assert len(KITTI_NAMES) == 8
    return write_yaml(root / "dataset.yaml", root, "images/val", "images/val")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dirs = v2_val_dirs(args.v2_root)
    excluded = excluded_ids(args.calibration, args.tuning)
    check_weights(args.weights)
    severe = load_severe(args.params)

    val_ids = sorted(p.stem for p in dirs["glared"].glob("*.jpg"))
    missing_clean = [i for i in val_ids if not (dirs["clean"] / f"{i}.jpg").is_file()]
    if missing_clean:
        fail(f"{len(missing_clean)} glared val frames have no clean version, e.g. {missing_clean[:5]}")
    ids, removed = eligible(val_ids, excluded)
    if args.limit:
        ids = ids[: args.limit]
    print(f"{args.restorer}: {len(ids)} val pairs evaluated, {len(removed)} excluded "
          f"(calibration/tuning): {removed}")

    from train_baseline import evaluate, pick_device
    from ultralytics import YOLO
    device = pick_device(args.device)
    model = YOLO(str(args.weights))
    ev = SimpleNamespace(device=device, imgsz=args.imgsz, batch=args.batch)

    versions = {"clean": (dirs["clean"], ".jpg"), "input": (dirs["glared"], ".jpg")}
    restore_info = None
    if args.restorer != "none":
        cache = args.cache_dir / f"v2_{args.restorer}" / "val"
        restore_info = restore_all(args.restorer, ids, dirs["glared"], cache, args.restore_threads)
        versions["output"] = (cache, ".png")

    results: dict = {}
    for version, (image_dir, suffix) in versions.items():
        block = {}
        groups = severity_groups(ids, severe)
        for group, group_ids in groups.items():
            if not group_ids:
                continue
            y = subset(args.work_dir / args.restorer, f"{version}_{group}", group_ids,
                       image_dir, suffix)
            m = evaluate(model, y, ev)
            block[group] = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"],
                            "per_class": m["per_class"], "count": len(group_ids)}
            print(f"  {version:<7}{group:<11} n={len(group_ids):>5}  mAP50 {m['mAP50']:.4f}"
                  f"  mAP50-95 {m['mAP50_95']:.4f}")
        results[version] = block

    out = args.out or REPO_ROOT / "results" / f"restorer_map_{args.restorer}.json"
    out.write_text(json.dumps({
        "restorer": args.restorer,
        "n_frames": len(ids),
        "limit": args.limit,
        "excluded_ids": removed,
        "excluded_sources": {"calibration_split": sorted(excluded["calibration_split"]),
                             "tuning_file": str(args.tuning.relative_to(REPO_ROOT))
                             if args.tuning.is_relative_to(REPO_ROOT) else str(args.tuning)},
        "detector_weights": str(args.weights.relative_to(REPO_ROOT))
        if args.weights.is_relative_to(REPO_ROOT) else str(args.weights),
        "device": device,
        "restore_device": "cpu" if args.restorer != "none" else None,
        "restore": restore_info,
        "metric": "train_baseline.evaluate(): Ultralytics model.val, imgsz "
                  f"{args.imgsz}, as in results/baseline_c1_metrics.json",
        "severe_flag_from": "data/glare_params_v2.csv (s >= severe_min_s per glare type)",
        "results": results,
    }, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
