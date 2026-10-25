"""C3 v1: a brightness-threshold glare gate, no training required.

CLAUDE.md section 9 (P1 item 7) explicitly allows a brightness-threshold gate as a v1 so
the C3 arm can be measured before the trained classifier lands. This is that gate. It
decides, per frame, whether restoration should run:

    mean luminance of the upper third  >  threshold   ->  "glared", trigger restoration
                                       otherwise      ->  "clean",  skip restoration

The upper third is where the sun is, and it is mostly sky, so it carries the glare signal
with the least road clutter. Luminance is Rec.601 luma on sRGB, normalised to [0, 1] --
deliberately the cheapest thing that could work, since a gate that costs as much as the
restoration it skips is pointless.

WHY THE THRESHOLD IS TUNED ON A SPLIT, NOT ON EVERYTHING

Picking the threshold that maximises accuracy over all 1496 frames and then reporting that
accuracy is fitting and scoring on the same data. It is only one scalar, so the optimism is
small, but it is free to avoid: the threshold is chosen on the training split and reported
on the held-out split.

The split is IMPORTED from train_baseline.py rather than reimplemented, so the gate's
held-out frames are byte-for-byte the same frames the detector held out. That matters more
than it looks: C3's numbers are only comparable with C1's if both were measured on the same
images. A copied-and-drifted split would break the comparison silently, which is exactly
the failure mode CLAUDE.md section 10 warns about.

WHAT THE TWO ERRORS COST

The gate's errors are not symmetric, and accuracy hides that:

  false negative   glare missed, restoration skipped, detection eats the full glare
                   penalty. C1 measured that at 41.8 mAP points.
  false positive   clean frame restored for nothing. Costs latency only.

So a miss is far more expensive than a needless restore, and the accuracy-optimal
threshold is not the deployment-optimal one. Both are reported, along with a per-glare-type
recall breakdown -- which matters because C1 found horizon_streak to be the most damaging
type by a wide margin, so a gate that misses horizon_streak fails precisely where it hurts.

Gate latency is measured too, with the same discipline as detect.py (I/O outside the timer,
warm-up discarded). C3's whole argument is
    gate + trigger_rate x restore  <  always restore
and that inequality needs the gate's own cost on the left.

Usage:
    python src/gate_brightness.py
    python src/gate_brightness.py --region 0.5     # use the upper half instead
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

GLARE_TYPES = ["direct_sun", "horizon_streak", "diffuse_haze", "veiling_wash"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "glare_labels_full.csv")
    p.add_argument("--glared-dir", type=Path, default=REPO_ROOT / "data" / "glared_full")
    p.add_argument("--clean-dir", type=Path, default=None,
                   help="defaults to data/clean_full if present, else datasets/kitti/images/val")
    p.add_argument("--labels-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "kitti" / "labels" / "val")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "results" / "gate_brightness_metrics.json")
    p.add_argument("--region", type=float, default=1 / 3,
                   help="fraction of image height to average, from the top (default: 1/3)")
    p.add_argument("--sweep-steps", type=int, default=400)
    p.add_argument("--fn-cost", type=float, default=10.0,
                   help="cost of a miss relative to a needless restore, for the "
                        "deployment-optimal threshold (default: %(default)s)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--latency-warmup", type=int, default=50)
    p.add_argument("--skip-latency", action="store_true")
    return p.parse_args(argv)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def upper_region_luma(frame, region: float) -> float:
    """Rec.601 luma, averaged over the top `region` of the frame, normalised to [0, 1]."""
    import cv2
    rows = max(1, int(frame.shape[0] * region))
    return float(cv2.cvtColor(frame[:rows], cv2.COLOR_BGR2GRAY).mean()) / 255.0


def score_at(threshold: float, samples) -> dict:
    """Confusion matrix and derived rates for one threshold. Positive class = glared."""
    tp = fp = tn = fn = 0
    for luma, is_glared in samples:
        predicted = luma > threshold
        if predicted and is_glared:
            tp += 1
        elif predicted and not is_glared:
            fp += 1
        elif not predicted and is_glared:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn or 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": round(threshold, 4),
        "accuracy": round(100.0 * (tp + tn) / total, 2),
        "precision": round(100.0 * precision, 2),
        "recall": round(100.0 * recall, 2),
        "f1": round(100.0 * 2 * precision * recall / (precision + recall), 2)
        if (precision + recall) else 0.0,
        "trigger_rate": round(100.0 * (tp + fp) / total, 2),
        "confusion_matrix": {
            "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn,
        },
        "n": total,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.clean_dir is None:
        candidate = REPO_ROOT / "data" / "clean_full"
        args.clean_dir = candidate if candidate.is_dir() else \
            REPO_ROOT / "datasets" / "kitti" / "images" / "val"

    # Imported, not reimplemented: the gate must hold out the same frames the detector did.
    from train_baseline import load_rows, stratified_split
    import cv2

    records = load_rows(args.csv, args.clean_dir, args.glared_dir, args.labels_dir)
    train_ids, heldout_ids = stratified_split(records, args.val_frac, args.seed)
    print(f"gate_brightness  |  {len(records)} frames  |  upper {args.region:.3f} of height")
    print(f"  split imported from train_baseline.py (seed {args.seed}): "
          f"{len(train_ids)} tune / {len(heldout_ids)} held out")

    print("\ncomputing upper-region luminance...")
    luma_by_id: dict[str, float] = {}
    for n, rec in enumerate(records, 1):
        # The gate sees the frame the pipeline would feed it: glared if glared, else clean.
        path = rec["glared"] if rec["is_glared"] else rec["clean"]
        frame = cv2.imread(str(path))
        if frame is None:
            sys.exit(f"error: could not decode {rel(path)}")
        luma_by_id[rec["id"]] = upper_region_luma(frame, args.region)
        if n % 500 == 0 or n == len(records):
            print(f"    {n}/{len(records)}", flush=True)

    truth = {r["id"]: r["is_glared"] for r in records}
    type_of = {r["id"]: r["glare_type"] for r in records}
    as_samples = lambda ids: [(luma_by_id[i], truth[i]) for i in sorted(ids)]

    clean_vals = [luma_by_id[i] for i in luma_by_id if not truth[i]]
    glared_vals = [luma_by_id[i] for i in luma_by_id if truth[i]]
    print(f"\n  clean  mean {statistics.fmean(clean_vals):.3f}   "
          f"glared mean {statistics.fmean(glared_vals):.3f}   "
          f"separation {statistics.fmean(glared_vals) - statistics.fmean(clean_vals):+.3f}")

    # ---- tune on the training split only ------------------------------------------
    lo, hi = min(luma_by_id.values()), max(luma_by_id.values())
    step = (hi - lo) / args.sweep_steps
    candidates = [lo + i * step for i in range(args.sweep_steps + 1)]
    train_samples = as_samples(train_ids)

    sweep = [score_at(t, train_samples) for t in candidates]
    best_acc = max(sweep, key=lambda s: (s["accuracy"], s["f1"]))
    best_f1 = max(sweep, key=lambda s: (s["f1"], s["accuracy"]))
    # Deployment view: a miss costs mAP, a needless restore costs only latency.
    best_cost = min(sweep, key=lambda s: (
        args.fn_cost * s["confusion_matrix"]["false_negative"]
        + s["confusion_matrix"]["false_positive"]))

    print(f"\ntuned on {len(train_ids)} training frames:")
    print(f"  best accuracy   threshold {best_acc['threshold']:.4f}  "
          f"acc {best_acc['accuracy']:.2f}%")
    print(f"  best F1         threshold {best_f1['threshold']:.4f}  f1 {best_f1['f1']:.2f}%")
    print(f"  best cost (FN={args.fn_cost:g}x FP)  threshold {best_cost['threshold']:.4f}")

    threshold = best_acc["threshold"]
    if not (0.65 <= threshold <= 0.75):
        print(f"\n  NOTE: the optimum ({threshold:.4f}) falls outside the suggested "
              f"0.65-0.75 band.\n  At 0.65 recall would be "
              f"{score_at(0.65, train_samples)['recall']:.1f}% and at 0.75 "
              f"{score_at(0.75, train_samples)['recall']:.1f}%, versus "
              f"{best_acc['recall']:.1f}% here -- that band sits above the crossover "
              f"and\n  trades away real recall.")

    # ---- report on held out (headline) and full (for reference) --------------------
    heldout = score_at(threshold, as_samples(heldout_ids))
    full = score_at(threshold, as_samples(set(luma_by_id)))

    per_type = {}
    for gtype in GLARE_TYPES:
        ids = [i for i in heldout_ids if truth[i] and type_of[i] == gtype]
        caught = sum(1 for i in ids if luma_by_id[i] > threshold)
        if ids:
            per_type[gtype] = {"n": len(ids), "caught": caught,
                               "recall": round(100.0 * caught / len(ids), 2)}

    for name, block in (("HELD OUT (defensible)", heldout), ("FULL 1496 (tuned on 80%)", full)):
        cm = block["confusion_matrix"]
        print(f"\n{name}  n={block['n']}  threshold {threshold:.4f}")
        print(f"  accuracy {block['accuracy']:.2f}%   precision {block['precision']:.2f}%   "
              f"recall {block['recall']:.2f}%   F1 {block['f1']:.2f}%")
        print(f"  trigger rate {block['trigger_rate']:.2f}%  (ground truth 50%)")
        print(f"  TP {cm['true_positive']}  FP {cm['false_positive']}  "
              f"TN {cm['true_negative']}  FN {cm['false_negative']}")

    print("\nper-type recall (held out) -- C1 ranked horizon_streak the most damaging:")
    for gtype, m in sorted(per_type.items(), key=lambda kv: kv[1]["recall"]):
        print(f"  {gtype:<16}{m['caught']:>4}/{m['n']:<4} {m['recall']:>7.2f}%")

    results = {
        **{k: heldout[k] for k in
           ("threshold", "accuracy", "precision", "recall", "trigger_rate",
            "confusion_matrix")},
        "units": "accuracy/precision/recall/trigger_rate are percentages",
        "scope_of_top_level_keys": "heldout",
        "config": {
            "region": args.region, "seed": args.seed, "val_frac": args.val_frac,
            "n_total": len(records), "n_tune": len(train_ids),
            "n_heldout": len(heldout_ids),
            "split_source": "train_baseline.stratified_split (identical to C1 held-out set)",
            "luminance": "Rec.601 luma on sRGB, normalised [0,1]",
        },
        "f1": heldout["f1"],
        "heldout": heldout,
        "full": full,
        "per_type_recall_heldout": per_type,
        "tuning": {
            "tuned_on": "training split only",
            "best_accuracy": best_acc,
            "best_f1": best_f1,
            f"best_cost_fn_{args.fn_cost:g}x": best_cost,
            "suggested_band_0.65_0.75": {
                "at_0.65": score_at(0.65, train_samples),
                "at_0.75": score_at(0.75, train_samples),
            },
        },
        "separation": {
            "clean_mean_luma": round(statistics.fmean(clean_vals), 4),
            "glared_mean_luma": round(statistics.fmean(glared_vals), 4),
        },
    }

    # ---- gate latency, needed for the C3 break-even ---------------------------------
    if not args.skip_latency:
        print("\nmeasuring gate latency (I/O excluded, warm-up discarded)...")
        paths = [(r["glared"] if r["is_glared"] else r["clean"]) for r in records]
        timings: list[float] = []
        for idx, path in enumerate(paths):
            frame = cv2.imread(str(path))            # decode OUTSIDE the timer
            start = time.perf_counter()
            upper_region_luma(frame, args.region)
            elapsed = (time.perf_counter() - start) * 1000.0
            if idx >= args.latency_warmup:
                timings.append(elapsed)
        ordered = sorted(timings)
        mean_ms = statistics.fmean(timings)
        results["latency"] = {
            "mean_ms": round(mean_ms, 4),
            "median_ms": round(statistics.median(timings), 4),
            "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
            "n_profiled": len(timings),
            "warmup_excluded": args.latency_warmup,
            "note": "luminance only; I/O outside the timed region",
        }
        print(f"  gate: {mean_ms:.4f} ms/frame (median "
              f"{results['latency']['median_ms']:.4f})")

        detect_json = REPO_ROOT / "results" / "detect_latency.json"
        if detect_json.is_file():
            detect_ms = json.loads(detect_json.read_text())["mean_ms"]
            overhead = 100.0 * mean_ms / detect_ms
            results["latency"]["detect_mean_ms"] = detect_ms
            results["latency"]["gate_overhead_pct_of_detect"] = round(overhead, 3)
            print(f"  detection is {detect_ms:.2f} ms, so the gate adds "
                  f"{overhead:.2f}% on top")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {rel(args.json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
