"""C1/C2/C3 evaluation harness, with restoration and gate as pluggable interfaces.

Neither teammate's module exists yet, so both are behind interfaces this harness probes at
run time. The point is that when they land, integration is a drop-in rather than a rewrite:

    restoration   src/restore.py must expose   restore(image: np.ndarray) -> np.ndarray
    gate          models/gate_classifier.pt, loaded through a `predict` that returns
                  either (is_glared: bool, confidence: float) or a bare bool

Run it today with --restore identity to prove the plumbing end to end: with an identity
restorer C2 must come out byte-identical to C1, and if it does not, the harness is wrong
rather than the restoration is. That check is worth more than it sounds -- integration is
where this kind of project usually fails, and it fails the night before the review.

WHAT "C3 BEATS C2" ACTUALLY MEANS

C3 restores a SUBSET of what C2 restores, so the two arms differ only on frames where the
gate does not fire:

    gate fires            C2 restores, C3 restores      identical
    gate misses (FN)      C2 restores, C3 does not      C3 is WORSE
    gate skips clean (TN) C2 restores, C3 does not      C3 better ONLY if restoring a
                          clean frame hurts it

So the expected ordering is C1 < C3 <= C2 on mAP, and C1 < C3 < C2 on latency. Asserting
C3 > C2 on mAP would fail on correct code. What C3 claims is efficiency, so that is what
gets checked:

    recovery = (C3 - C1) / (C2 - C1)      share of C2's accuracy gain that C3 keeps
    cost     = (C3_ms - C1_ms) / (C2_ms - C1_ms)   share of C2's added cost that C3 pays

C3 wins when recovery is high and cost is low. C3 > C2 is possible -- it happens when
restoration degrades clean frames, which is a real effect and the motivation behind the
Sensors 2024 degradation classifier -- but it is an observation to report, not an
invariant to assert.

Every arm is scored with the same Ultralytics metric code as train_baseline.py, and the
split is imported from it, so C1 here is directly comparable to the C1 already published in
baseline_c1_metrics.json rather than a lookalike.

Usage:
    python src/run_experiment.py                      # C1 now, C2/C3 as available
    python src/run_experiment.py --restore identity   # plumbing check: C2 must equal C1
    python src/run_experiment.py --gate brightness    # C3 via the v1 gate (section 9 P1)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

GLARE_TYPES = ["direct_sun", "horizon_streak", "diffuse_haze", "veiling_wash"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "yolo_baseline.pt")
    p.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "glare_labels_full.csv")
    p.add_argument("--glared-dir", type=Path, default=REPO_ROOT / "data" / "glared_full")
    p.add_argument("--clean-dir", type=Path, default=None)
    p.add_argument("--labels-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "kitti" / "labels" / "val")
    p.add_argument("--restore-module", type=Path, default=REPO_ROOT / "src" / "restore.py")
    p.add_argument("--gate-model", type=Path,
                   default=REPO_ROOT / "models" / "gate_classifier.pt")
    p.add_argument("--gate-metrics", type=Path,
                   default=REPO_ROOT / "results" / "gate_brightness_metrics.json")
    p.add_argument("--work-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "eval_subsets" / "arms")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "results" / "experiment_c123_results.json")
    p.add_argument("--restore", choices=["auto", "identity", "none"], default="auto",
                   help="auto = src/restore.py if present; identity = plumbing check")
    p.add_argument("--gate", choices=["auto", "trained", "brightness", "none"], default="auto",
                   help="auto = trained gate if present, else the v1 brightness gate")
    p.add_argument("--scope", choices=["heldout", "full"], default="heldout",
                   help="heldout = the 298 frames never trained on (defensible)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---- pluggable interfaces -----------------------------------------------------------

def load_restore(args):
    """Return (fn, label, is_identity). fn(image: np.ndarray) -> np.ndarray, or None."""
    if args.restore == "none":
        return None, "disabled", False
    if args.restore == "identity":
        return (lambda image: image), "identity (plumbing check)", True

    if not args.restore_module.is_file():
        print(f"  Restore module not found (Vihaan's job)   -- expected "
              f"{rel(args.restore_module)} exposing restore(image) -> image")
        return None, "missing", False

    spec = importlib.util.spec_from_file_location("restore_mod", args.restore_module)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                       # a teammate's WIP must not kill the run
        print(f"  Restore module failed to import: {exc}")
        return None, f"import error: {exc}", False
    fn = getattr(module, "restore", None)
    if not callable(fn):
        print(f"  {rel(args.restore_module)} has no callable restore(image)")
        return None, "no restore() found", False
    print(f"  restoration loaded from {rel(args.restore_module)}")
    return fn, f"module:{rel(args.restore_module)}", False


def load_gate(args):
    """Return (fn, label). fn(image) -> bool. Accepts (bool, confidence) or bare bool."""
    if args.gate == "none":
        return None, "disabled"

    if args.gate in ("auto", "trained") and args.gate_model.is_file():
        try:
            import torch
            model = torch.load(args.gate_model, weights_only=False)
            if hasattr(model, "eval"):
                model.eval()

            def predict(image):
                out = model.predict(image) if hasattr(model, "predict") else model(image)
                # Interface allows (is_glared, confidence) or a bare bool.
                return bool(out[0]) if isinstance(out, (tuple, list)) else bool(out)

            print(f"  gate loaded from {rel(args.gate_model)}")
            return predict, f"trained:{rel(args.gate_model)}"
        except Exception as exc:
            print(f"  gate at {rel(args.gate_model)} failed to load: {exc}")
            if args.gate == "trained":
                return None, f"load error: {exc}"

    if args.gate == "trained":
        print(f"  Gate classifier not found (Harshitha's job) -- expected "
              f"{rel(args.gate_model)}")
        return None, "missing"

    # Fall back to the v1 brightness gate. CLAUDE.md section 9 P1 item 7 explicitly allows
    # this so C3 is measurable before the trained classifier exists.
    if not args.gate_metrics.is_file():
        print(f"  Gate classifier not found (Harshitha's job), and no "
              f"{rel(args.gate_metrics)} to fall back on")
        return None, "missing"

    from gate_brightness import upper_region_luma
    meta = json.loads(args.gate_metrics.read_text())
    threshold = meta["threshold"]
    region = meta["config"]["region"]
    print(f"  Gate classifier not found (Harshitha's job) -- falling back to the v1 "
          f"brightness gate (threshold {threshold:.4f})")
    return (lambda image: upper_region_luma(image, region) > threshold), \
           f"brightness_v1:threshold={threshold}"


# ---- arm construction ---------------------------------------------------------------

def materialise(dest: Path, ids, by_id, restore_fn, gate_fn, labels_dir: Path) -> dict:
    """Build one arm's image set; symlink when untouched, write a PNG when restored."""
    import cv2

    images, labels = dest / "images" / "val", dest / "labels" / "val"
    for d in (images, labels):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    fired = 0
    gate_ms = restore_ms = 0.0
    for image_id in sorted(ids):
        rec = by_id[image_id]
        # The pipeline receives the glared frame when glared, else the clean one.
        source = rec["glared"] if rec["is_glared"] else rec["clean"]
        (labels / f"{image_id}.txt").symlink_to(rec["label"].resolve())

        should_restore = restore_fn is not None
        frame = None
        if gate_fn is not None:
            frame = cv2.imread(str(source))
            start = time.perf_counter()
            decision = bool(gate_fn(frame))
            gate_ms += (time.perf_counter() - start) * 1000.0
            should_restore = should_restore and decision
            fired += decision

        if not should_restore:
            (images / f"{image_id}.png").symlink_to(source.resolve())
            continue

        if frame is None:
            frame = cv2.imread(str(source))
        start = time.perf_counter()
        restored = restore_fn(frame)
        restore_ms += (time.perf_counter() - start) * 1000.0
        cv2.imwrite(str(images / f"{image_id}.png"), restored)

    n = len(ids) or 1
    return {"n": len(ids), "restored": fired if gate_fn else (n if restore_fn else 0),
            "gate_ms_per_frame": round(gate_ms / n, 4),
            "restore_ms_per_frame_all": round(restore_ms / n, 4),
            "trigger_rate": round(100.0 * fired / n, 2) if gate_fn else None}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.model.is_file():
        sys.exit(f"error: {rel(args.model)} not found. Run src/train_baseline.py first.")
    if args.clean_dir is None:
        candidate = REPO_ROOT / "data" / "clean_full"
        args.clean_dir = candidate if candidate.is_dir() else \
            REPO_ROOT / "datasets" / "kitti" / "images" / "val"

    from train_baseline import load_rows, stratified_split, write_yaml, evaluate
    from ultralytics import YOLO

    records = load_rows(args.csv, args.clean_dir, args.glared_dir, args.labels_dir)
    by_id = {r["id"]: r for r in records}
    train_ids, heldout_ids = stratified_split(records, args.val_frac, args.seed)
    scope_ids = heldout_ids if args.scope == "heldout" else {r["id"] for r in records}

    print(f"run_experiment  |  scope={args.scope} ({len(scope_ids)} frames)  |  "
          f"device {args.device}")
    print(f"  split imported from train_baseline.py (seed {args.seed})\n")

    restore_fn, restore_label, restore_identity = load_restore(args)
    gate_fn, gate_label = load_gate(args)

    model = YOLO(str(args.model))
    ev_args = Namespace(device=args.device, imgsz=args.imgsz, batch=args.batch)
    detect_ms = 0.0
    detect_json = REPO_ROOT / "results" / "detect_latency.json"
    if detect_json.is_file():
        detect_ms = json.loads(detect_json.read_text())["mean_ms"]

    clean_ids = sorted(i for i in scope_ids if not by_id[i]["is_glared"])
    glared_ids = sorted(i for i in scope_ids if by_id[i]["is_glared"])
    results: dict = {"config": {
        "scope": args.scope, "n": len(scope_ids), "device": args.device,
        "seed": args.seed, "model": rel(args.model),
        "restore": restore_label, "gate": gate_label,
        "detect_ms_per_frame": detect_ms,
    }}

    def run_arm(name: str, ids, use_restore, use_gate):
        stats = materialise(args.work_dir / name, ids, by_id,
                            restore_fn if use_restore else None,
                            gate_fn if use_gate else None, args.labels_dir)
        yaml_path = write_yaml(args.work_dir / name / "dataset.yaml",
                               args.work_dir / name, "images/val", "images/val")
        return {**evaluate(model, yaml_path, ev_args), **stats}

    # ---- C1: never restore ---------------------------------------------------------
    print("\n=== C1 (never restore) ===")
    c1_clean = run_arm("c1_clean", clean_ids, False, False)
    c1_glared = run_arm("c1_glared", glared_ids, False, False)
    per_type = {}
    for gtype in GLARE_TYPES:
        t_ids = [i for i in glared_ids if by_id[i]["glare_type"] == gtype]
        if t_ids:
            m = run_arm(f"c1_{gtype}", t_ids, False, False)
            per_type[gtype] = {"mAP50": m["mAP50"], "count": len(t_ids)}
    c1_all = run_arm("c1_all", sorted(scope_ids), False, False)
    results["c1"] = {
        "status": "complete",
        "mAP50_clean": c1_clean["mAP50"], "mAP50_glared": c1_glared["mAP50"],
        "mAP50_all": c1_all["mAP50"], "mAP50_95_all": c1_all["mAP50_95"],
        "mAP_drop": round(c1_clean["mAP50"] - c1_glared["mAP50"], 4),
        "per_type": per_type,
        "latency_ms": round(detect_ms, 3), "trigger_rate": 0.0,
    }
    print(f"  clean {c1_clean['mAP50']:.4f}  glared {c1_glared['mAP50']:.4f}  "
          f"drop {results['c1']['mAP_drop']:.4f}  |  all {c1_all['mAP50']:.4f}")

    # ---- C2: always restore --------------------------------------------------------
    print("\n=== C2 (always restore) ===")
    if restore_fn is None:
        results["c2"] = {"status": "waiting", "reason": restore_label,
                         "expected_interface": "src/restore.py :: restore(image) -> image"}
        print("  skipped -- waiting on the restoration module")
    else:
        c2 = run_arm("c2_all", sorted(scope_ids), True, False)
        results["c2"] = {
            "status": "complete", "mAP50": c2["mAP50"], "mAP50_95": c2["mAP50_95"],
            "latency_ms": round(detect_ms + c2["restore_ms_per_frame_all"], 3),
            "restore_ms_per_frame": c2["restore_ms_per_frame_all"],
            "trigger_rate": 100.0, "restored": c2["restored"],
        }
        print(f"  mAP50 {c2['mAP50']:.4f}  restore {c2['restore_ms_per_frame_all']:.3f} "
              f"ms/frame  latency {results['c2']['latency_ms']:.3f} ms")
        if restore_identity:
            same = abs(c2["mAP50"] - c1_all["mAP50"]) < 1e-9
            results["c2"]["plumbing_check"] = {
                "identical_to_c1": same, "c1_mAP50": c1_all["mAP50"],
                "note": "identity restoration must reproduce C1 exactly",
            }
            print(f"  plumbing check: C2 {'==' if same else '!='} C1 "
                  f"({c1_all['mAP50']:.4f})  ->  {'PASS' if same else 'FAIL'}")

    # ---- C3: gated -----------------------------------------------------------------
    print("\n=== C3 (gated restore) ===")
    if restore_fn is None or gate_fn is None:
        missing = [n for n, v in (("restoration", restore_fn), ("gate", gate_fn)) if v is None]
        results["c3"] = {"status": "waiting", "reason": f"missing: {', '.join(missing)}",
                         "expected_interface": "gate.predict(image) -> (bool, float) | bool"}
        print(f"  skipped -- waiting on {', '.join(missing)}")
    else:
        c3 = run_arm("c3_all", sorted(scope_ids), True, True)
        latency = detect_ms + c3["gate_ms_per_frame"] + c3["restore_ms_per_frame_all"]
        results["c3"] = {
            "status": "complete", "mAP50": c3["mAP50"], "mAP50_95": c3["mAP50_95"],
            "trigger_rate": c3["trigger_rate"], "latency_ms": round(latency, 3),
            "gate_ms_per_frame": c3["gate_ms_per_frame"],
            "restore_ms_per_frame": c3["restore_ms_per_frame_all"],
            "restored": c3["restored"],
        }
        print(f"  mAP50 {c3['mAP50']:.4f}  trigger {c3['trigger_rate']:.2f}%  "
              f"latency {latency:.3f} ms")

    # ---- comparison ----------------------------------------------------------------
    if results.get("c2", {}).get("status") == "complete" and \
            results.get("c3", {}).get("status") == "complete":
        c1m, c2m, c3m = c1_all["mAP50"], results["c2"]["mAP50"], results["c3"]["mAP50"]
        c1l, c2l, c3l = detect_ms, results["c2"]["latency_ms"], results["c3"]["latency_ms"]
        gain, cost = c2m - c1m, c2l - c1l
        verdict = {
            "recovery_of_c2_gain": round((c3m - c1m) / gain, 4) if abs(gain) > 1e-9 else None,
            "share_of_c2_added_cost": round((c3l - c1l) / cost, 4) if abs(cost) > 1e-9 else None,
            "c3_beats_c1": c3m >= c1m,
            "c3_cheaper_than_c2": c3l < c2l,
            "c3_above_c2": c3m > c2m,
            "note": ("C3 restores a subset of C2, so C3 <= C2 on mAP is EXPECTED, not a "
                     "bug. C3 above C2 happens only when restoring clean frames hurts "
                     "them. The claim being tested is efficiency: keep most of C2's gain "
                     "for a fraction of its cost."),
        }
        results["comparison"] = verdict
        print("\n=== C1 vs C2 vs C3 ===")
        print(f"  {'arm':<6}{'mAP50':>9}{'latency ms':>13}{'trigger':>10}")
        print(f"  {'C1':<6}{c1m:>9.4f}{c1l:>13.3f}{0.0:>9.1f}%")
        print(f"  {'C2':<6}{c2m:>9.4f}{c2l:>13.3f}{100.0:>9.1f}%")
        print(f"  {'C3':<6}{c3m:>9.4f}{c3l:>13.3f}"
              f"{results['c3']['trigger_rate']:>9.1f}%")
        if verdict["recovery_of_c2_gain"] is not None:
            print(f"\n  C3 keeps {verdict['recovery_of_c2_gain']:.1%} of C2's accuracy gain "
                  f"for {verdict['share_of_c2_added_cost']:.1%} of its added cost")
        if verdict["c3_above_c2"]:
            print("  note: C3 scored ABOVE C2 -- restoration is degrading clean frames")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {rel(args.json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
