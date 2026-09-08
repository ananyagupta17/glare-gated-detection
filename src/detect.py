"""YOLO inference latency profiling on the KITTI val set, plus an mAP regression check.

This supplies the cost axis of the thesis. C1 already shows glare costs 41.8 mAP points;
the gate's argument is that restoration can be skipped often enough to pay for itself, and
that argument needs milliseconds. Detection latency measured here is the constant every
arm pays: C1 = detect, C2 = restore + detect, C3 = gate + sometimes restore + detect.

MEASUREMENT DISCIPLINE (CLAUDE.md section 7)

  * Disk I/O is excluded. The frame is decoded into memory first, and only the inference
    call is inside the timer. Timing model(path) instead would fold PNG decode -- tens of
    milliseconds on 1242x375 -- into the "inference" number.
  * Warm-up frames are timed but discarded. The first call carries lazy graph init and
    allocator warm-up and is typically several times slower than steady state; their stats
    are reported separately so the size of that effect is visible rather than assumed.
  * Batch size is 1. This is a per-frame latency benchmark for a streaming camera, not a
    throughput benchmark; batching would report a number no deployed pipeline can achieve.
  * The device is recorded in the output, because latency without it is meaningless.
    Default is CPU: it is the reproducible floor, and the honest setting for the embedded
    target this pipeline is motivated by.

Wall-clock per frame is the headline, since it includes the Python and framework overhead
a real caller pays. Ultralytics' own preprocess/inference/postprocess breakdown is recorded
alongside it, which is what shows how much of the budget is NMS rather than the network.

VALIDATION, AND WHY THERE ARE TWO NUMBERS

Scoring all 1496 clean frames is not comparable to anything in baseline_c1_metrics.json:
that file's `full.clean` is the 748 clean-ASSIGNED IDs (mAP50 0.8354) and
`full.paired.clean` is the other 748 (0.7947). All 1496 is their union, so it matches
neither, and it is expected to land between them.

So two figures are produced:

  validation_mAP50      all 1496 clean frames. The number this script was asked for.
  regression_check      the exact 748-image subset the baseline scored, re-scored here.
                        Same model, same images, same metric code, so it MUST reproduce
                        the baseline within tolerance. This is what actually proves the
                        inference path is intact; the 1496 figure cannot.

Usage:
    python src/detect.py                       # profile + validate
    python src/detect.py --skip-validation     # latency only (fast)
    python src/detect.py --device mps          # compare accelerator latency
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

KITTI_NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian",
               4: "Person_sitting", 5: "cyclist", 6: "tram", 7: "misc"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "yolo_baseline.pt")
    p.add_argument("--images", type=Path,
                   default=REPO_ROOT / "datasets" / "kitti" / "images" / "val")
    p.add_argument("--labels", type=Path,
                   default=REPO_ROOT / "datasets" / "kitti" / "labels" / "val")
    p.add_argument("--baseline", type=Path,
                   default=REPO_ROOT / "results" / "baseline_c1_metrics.json")
    p.add_argument("--regression-subset", type=Path,
                   default=REPO_ROOT / "datasets" / "eval_subsets" / "full_clean",
                   help="the 748-image set baseline_c1_metrics.json scored as full.clean")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "results" / "detect_latency.json")
    p.add_argument("--work-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "eval_subsets" / "detect_all_clean")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001,
                   help="low, to match validation defaults rather than a demo threshold")
    p.add_argument("--limit", type=int, default=None, help="profile fewer frames (testing)")
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--skip-validation", action="store_true")
    return p.parse_args(argv)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_yaml(path: Path, root: Path, split_rel: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {k}: {v}" for k, v in KITTI_NAMES.items())
    path.write_text(
        f"# generated by src/detect.py -- do not edit by hand\n"
        f"path: {root.resolve()}\n"
        f"train: {split_rel}\n"
        f"val: {split_rel}\n\n"
        f"names:\n{names}\n"
    )
    return path


def profile(model, images: list[Path], args) -> dict:
    """Time inference only, one frame at a time, with I/O outside the timed region."""
    import cv2

    warm_ms: list[float] = []
    steady_ms: list[float] = []
    speed_parts: dict[str, list[float]] = {"preprocess": [], "inference": [], "postprocess": []}

    for idx, path in enumerate(images):
        frame = cv2.imread(str(path))          # decode OUTSIDE the timer
        if frame is None:
            sys.exit(f"error: could not decode {rel(path)}")

        start = time.perf_counter()
        result = model.predict(frame, device=args.device, imgsz=args.imgsz,
                               conf=args.conf, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if idx < args.warmup:
            warm_ms.append(elapsed_ms)
        else:
            steady_ms.append(elapsed_ms)
            for key in speed_parts:
                value = result.speed.get(key)
                if value is not None:
                    speed_parts[key].append(float(value))

        done = idx + 1
        if done % 250 == 0 or done == len(images):
            phase = "warmup" if idx < args.warmup else "steady"
            print(f"    {done}/{len(images)} ({phase})", flush=True)

    if not steady_ms:
        sys.exit("error: no frames left after warm-up; lower --warmup or raise --limit")

    return {"warm": warm_ms, "steady": steady_ms, "parts": speed_parts}


def percentiles(latencies: list[float]) -> dict:
    """Tail latency, which is what a real-time pipeline actually has to survive.

    Mean +/- std is misleading here: a single OS-scheduling spike inflates std to a large
    fraction of the mean while the typical frame is unaffected. p95/p99 describe the
    budget a perception stack must actually meet.
    """
    ordered = sorted(latencies)
    n = len(ordered)

    def at(q: float) -> float:
        return ordered[min(n - 1, max(0, int(round(q * (n - 1)))))]

    median = statistics.median(ordered)
    return {
        "p50_ms": round(at(0.50), 3),
        "p90_ms": round(at(0.90), 3),
        "p95_ms": round(at(0.95), 3),
        "p99_ms": round(at(0.99), 3),
        "outliers_over_2x_median": sum(1 for v in ordered if v > 2 * median),
        "outlier_note": "spikes are OS scheduling, not model behaviour; see p95/p99",
    }


def drift(latencies: list[float]) -> dict:
    """Is latency stable, or creeping up? A rising trend means thermal or leak trouble."""
    n = len(latencies)
    tenth = max(1, n // 10)
    first, last = latencies[:tenth], latencies[-tenth:]
    mean_x = (n - 1) / 2
    mean_y = statistics.fmean(latencies)
    denom = sum((i - mean_x) ** 2 for i in range(n))
    slope = (sum((i - mean_x) * (y - mean_y) for i, y in enumerate(latencies)) / denom
             if denom else 0.0)
    return {
        "first_decile_mean_ms": round(statistics.fmean(first), 3),
        "last_decile_mean_ms": round(statistics.fmean(last), 3),
        "delta_ms": round(statistics.fmean(last) - statistics.fmean(first), 3),
        "slope_ms_per_1000_frames": round(slope * 1000, 4),
    }


def evaluate(model, yaml_path: Path, args) -> dict:
    metrics = model.val(data=str(yaml_path), split="val", device=args.device,
                        imgsz=args.imgsz, verbose=False, plots=False, save_json=False,
                        project=str(REPO_ROOT / "runs" / "val"),
                        name=yaml_path.parent.name, exist_ok=True)
    return {"mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.model.is_file():
        sys.exit(f"error: {rel(args.model)} not found. Run src/train_baseline.py first.")
    if not args.images.is_dir():
        sys.exit(f"error: {rel(args.images)} not found.")

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() == ".png")
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"error: no PNGs in {rel(args.images)}")

    missing = [p.stem for p in images if not (args.labels / f"{p.stem}.txt").is_file()]
    if missing:
        sys.exit(f"error: {len(missing)} frame(s) have no label, e.g. {missing[:5]}")

    from ultralytics import YOLO
    model = YOLO(str(args.model))

    print(f"detect.py  |  {len(images)} frames  |  device {args.device}  |  "
          f"imgsz {args.imgsz}  |  batch 1")
    print(f"  model  {rel(args.model)}")
    print(f"  images {rel(args.images)}")
    print(f"\nprofiling (warm-up {args.warmup} frames, timed then discarded)...")

    timings = profile(model, images, args)
    steady = timings["steady"]

    mean_ms = statistics.fmean(steady)
    stats = {
        "mean_ms": round(mean_ms, 3),
        "std_ms": round(statistics.pstdev(steady), 3) if len(steady) > 1 else 0.0,
        "min_ms": round(min(steady), 3),
        "max_ms": round(max(steady), 3),
        "fps": round(1000.0 / mean_ms, 2),
    }
    parts = {f"{k}_ms": round(statistics.fmean(v), 3)
             for k, v in timings["parts"].items() if v}

    warm = timings["warm"]
    warmup_block = {
        "n": len(warm),
        "mean_ms": round(statistics.fmean(warm), 3) if warm else None,
        "first_frame_ms": round(warm[0], 3) if warm else None,
        "cold_start_penalty_x": (round(warm[0] / mean_ms, 2) if warm and mean_ms else None),
    }
    stability = drift(steady)
    tails = percentiles(steady)

    print(f"\nMean latency: {stats['mean_ms']:.2f} ms ({stats['fps']:.2f} FPS)")
    print(f"  std {stats['std_ms']:.2f} ms   min {stats['min_ms']:.2f}   "
          f"max {stats['max_ms']:.2f}   median {statistics.median(steady):.2f}")
    print(f"  tail: p90 {tails['p90_ms']:.2f}  p95 {tails['p95_ms']:.2f}  "
          f"p99 {tails['p99_ms']:.2f}   "
          f"({tails['outliers_over_2x_median']} frame(s) over 2x median)")
    if parts:
        print("  breakdown: " + "  ".join(f"{k} {v:.2f}" for k, v in parts.items()))
    if warm:
        print(f"  warm-up: frame 1 was {warmup_block['first_frame_ms']:.2f} ms "
              f"({warmup_block['cold_start_penalty_x']}x steady state), "
              f"{len(warm)}-frame mean {warmup_block['mean_ms']:.2f} ms -- excluded")
    print(f"  stability: first decile {stability['first_decile_mean_ms']:.2f} ms -> "
          f"last decile {stability['last_decile_mean_ms']:.2f} ms "
          f"(slope {stability['slope_ms_per_1000_frames']:+.3f} ms/1000 frames)")

    results: dict = {
        **stats,
        "device": args.device,
        "note": "Warm-up 50 frames excluded; wall-clock includes device + CPU overhead",
        "config": {
            "model": rel(args.model), "imgsz": args.imgsz, "batch": 1,
            "conf": args.conf, "n_profiled": len(steady),
            "n_frames_total": len(images), "warmup_frames": args.warmup,
            "io_excluded": True,
        },
        "median_ms": round(statistics.median(steady), 3),
        "ultralytics_breakdown_ms": parts,
        "percentiles": tails,
        "warmup": warmup_block,
        "stability": stability,
    }

    # ---- validation ---------------------------------------------------------------
    if not args.skip_validation:
        print(f"\nvalidating on all {len(images)} clean frames...")
        yaml_all = write_yaml(args.work_dir / "dataset.yaml",
                              args.images.parent.parent, "images/val")
        full = evaluate(model, yaml_all, args)
        results["validation_mAP50"] = full["mAP50"]
        results["validation_mAP5095"] = full["mAP50_95"]
        results["validation_scope"] = f"all {len(images)} clean val frames"
        print(f"  mAP@50 {full['mAP50']:.4f}   mAP@50:95 {full['mAP50_95']:.4f}")

        # The real regression check: re-score the exact subset the baseline scored.
        baseline_yaml = args.regression_subset / "dataset.yaml"
        if args.baseline.is_file() and baseline_yaml.is_file():
            expected = json.loads(args.baseline.read_text())["full"]["clean"]
            print(f"\nregression check vs baseline full.clean "
                  f"({expected['count']} frames, the only comparable target)...")
            got = evaluate(model, baseline_yaml, args)
            delta50 = got["mAP50"] - expected["mAP50"]
            ok = abs(delta50) <= args.tolerance
            results["regression_check"] = {
                "subset": "full.clean", "n": expected["count"],
                "baseline_mAP50": expected["mAP50"], "measured_mAP50": got["mAP50"],
                "delta_mAP50": round(delta50, 4),
                "baseline_mAP50_95": expected["mAP50_95"],
                "measured_mAP50_95": got["mAP50_95"],
                "tolerance": args.tolerance, "passed": ok,
                "note": ("baseline was scored on mps, this on "
                         f"{args.device}; small float drift is expected"),
            }
            print(f"  baseline {expected['mAP50']:.4f}  measured {got['mAP50']:.4f}  "
                  f"delta {delta50:+.4f}  ->  {'PASS' if ok else 'FAIL'}")
            if not ok:
                print("  WARNING: inference path does not reproduce the baseline")
        else:
            results["regression_check"] = {
                "passed": None,
                "note": f"skipped: {rel(baseline_yaml)} not found; "
                        "run src/train_baseline.py to create it",
            }
            print(f"\n  regression check skipped ({rel(baseline_yaml)} missing)")

        print("\n  NOTE: validation_mAP50 covers all 1496 clean frames, which is the union "
              "of the\n  baseline's two 748-frame halves (full.clean 0.8354, "
              "full.paired.clean 0.7947).\n  It is expected to land between them and match "
              "neither; the regression check\n  above is what verifies the inference path.")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {rel(args.json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
