"""Gate classifier on v2 data: train on the v2 train split, score on the v2 val split.

Follows results/gate_protocol_v2.md (pre-registered 2026-10-20):
  * Positive: the glared version of a frame, unless excluded.
  * Negative: the clean version of EVERY frame, including frames whose glared version is
    excluded.
  * Excluded: glared veiling_wash frames with s < 0.1753, read from `gate_excluded` in
    data/glare_params_v2.csv. Applied identically in training and scoring. Excluded
    frames are not a gate error either way; scoring still records how often the gate fires
    on them, since at runtime they count towards the trigger rate.

FEATURES
The same 9 hand-engineered features as v1, imported from src/train_gate_classifier.py
rather than copied: src/shortcut_test.py imports that module too, and its threshold is
pre-registered on exactly this feature set. Changing the features here would leave the
shortcut test auditing a different gate from the one being scored.

NO PEEKING AT VAL
v2 val (1,496 pairs) is the test set for every arm (CLAUDE.md section 10). Epoch selection
uses a holdout carved out of TRAIN, split by frame id, so a frame's clean and glared
versions always land on the same side -- a pair split across train and holdout would leak
the scene. The decision threshold is fixed at 0.5 and never tuned.

Data (written by src/glare_synth.py):
    data/glare_manifest_v2.csv               image_id, split, is_glared, glare_type, severity
    data/glare_params_v2.csv                 per frame: severe, gate_excluded, ...
    data/v2/{clean,glared}/images/{split}/<id>.jpg

Outputs:
    models/gate_v2.pt                        GateClassifier, same contract as v1 (not committed)
    runs/gate_v2/train_log.json              per-epoch loss and holdout metrics (not committed)
    results/gate_v2_metrics.json             val precision/recall/F1, per type, per severity
    results/gate_v2_predictions.csv          one row per val image

Integration contract (unchanged from v1):
    model = torch.load("models/gate_v2.pt", weights_only=False)
    is_glared, confidence = model.predict(image)   # image: BGR uint8, cv2.imread output

Usage:
    python src/train_gate_v2.py train                 # MLP (default)
    python src/train_gate_v2.py train --model lr      # logistic regression
    python src/train_gate_v2.py score
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_gate_classifier import FEATURE_NAMES, GateClassifier, extract_features  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GLARE_TYPES = ("diffuse_haze", "direct_sun", "veiling_wash", "horizon_streak")

# Pre-registered in results/gate_protocol_v2.md. The params file is the source of truth for
# which frames are excluded; this constant only guards against a calibration that drifted.
EXCLUDE_TYPE = "veiling_wash"
EXCLUDE_BELOW_S = 0.1753

# Brightness gate v1, old 400/1,496-frame v1 split -- a different realisation, so it is
# context for the reader, not a like-for-like comparison.
BRIGHTNESS_V1 = {"accuracy": 80.87, "precision": 78.26, "recall": 85.14,
                 "trigger_rate": 54.03, "latency_ms": 0.114, "data": "v1, not comparable"}

# Severity bins for the recall-vs-s curve. The protocol's only band is severe / not
# severe (no mild band, results/severity_scale_decision.json); bins add resolution.
S_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --- data -----------------------------------------------------------------------------

def load_frames(manifest: Path, params: Path, calibration: Path, v2_root: Path,
                split: str) -> list[dict]:
    """One dict per frame of `split`, with both image paths and the protocol flags.

    Refuses (exits) rather than guessing whenever the files disagree with each other or
    with the protocol, because a silently wrong exclusion corrupts every gate number.
    """
    for path in (manifest, params):
        if not path.is_file():
            sys.exit(f"error: {rel(path)} not found. v2 data has not been generated yet "
                     f"(src/glare_synth.py, after src/write_severe_calibration.py).")

    glared_rows = {r["image_id"]: r for r in csv.DictReader(manifest.open())
                   if r["split"] == split and r["is_glared"] == "1"}
    param_rows = {r["image_id"]: r for r in csv.DictReader(params.open())
                  if r["split"] == split}
    if not param_rows:
        sys.exit(f"error: no '{split}' rows in {rel(params)}")
    if set(glared_rows) != set(param_rows):
        sys.exit(f"error: {rel(manifest)} and {rel(params)} list different '{split}' "
                 f"frames ({len(glared_rows)} vs {len(param_rows)})")

    # The calibration file may round the cutoff differently from the protocol (it stores
    # 0.1752 against the registered 0.1753). That only matters if a frame's s falls
    # between the two, so compare the excluded sets they produce, not the decimals.
    limit = None
    if calibration.is_file():
        cut = json.loads(calibration.read_text())["cutpoints"].get(EXCLUDE_TYPE, {})
        limit = cut.get("gate_exclude_below_s")
        if limit is None:
            sys.exit(f"error: {rel(calibration)} has no gate_exclude_below_s for {EXCLUDE_TYPE}")
        lo, hi = sorted((float(limit), EXCLUDE_BELOW_S))
        between = [i for i, p in param_rows.items()
                   if p["glare_type"] == EXCLUDE_TYPE and lo <= float(p["severity"]) < hi]
        if between:
            sys.exit(f"error: {rel(calibration)} cutoff {limit} and the pre-registered "
                     f"{EXCLUDE_BELOW_S} exclude different {split} frames, e.g. {between[0]}")

    frames, missing = [], []
    for image_id in sorted(param_rows):
        p = param_rows[image_id]
        if p["gate_excluded"] == "" or p["severe"] == "":
            sys.exit(f"error: {rel(params)} has empty severe/gate_excluded ({split}/"
                     f"{image_id}). That is an --uncalibrated run; the gate cannot use it.")
        gtype, s, excluded = p["glare_type"], float(p["severity"]), p["gate_excluded"] == "1"
        if glared_rows[image_id]["glare_type"] != gtype:
            sys.exit(f"error: {split}/{image_id} glare_type differs between manifest and params")
        if excluded and gtype != EXCLUDE_TYPE:
            sys.exit(f"error: {split}/{image_id} is a {gtype} frame flagged gate_excluded; "
                     f"the protocol excludes {EXCLUDE_TYPE} only")
        if limit is not None and excluded != (gtype == EXCLUDE_TYPE and s < float(limit)):
            sys.exit(f"error: {split}/{image_id} gate_excluded={int(excluded)} disagrees with "
                     f"s={s:.6f} against the calibrated cutoff {limit}")

        paths = {v: v2_root / v / "images" / split / f"{image_id}.jpg"
                 for v in ("clean", "glared")}
        missing += [str(q) for q in paths.values() if not q.is_file()]
        frames.append(dict(id=image_id, glare_type=gtype, severity=s,
                           severe=p["severe"] == "1", excluded=excluded,
                           clean=paths["clean"], glared=paths["glared"]))
    if missing:
        sys.exit(f"error: {len(missing)} image(s) missing, e.g. {missing[0]}. "
                 f"Run src/glare_synth.py --splits {split}.")
    return frames


def _features_of(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    return extract_features(img)


def extract_all(paths: list[Path], workers: int) -> np.ndarray:
    print(f"  extracting {len(FEATURE_NAMES)} features from {len(paths)} images "
          f"({workers} workers)...", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        feats = list(pool.map(_features_of, [str(p) for p in paths], chunksize=32))
    print(f"  done in {time.perf_counter() - t0:.0f} s")
    return np.stack(feats).astype(np.float32)


def labelled_images(frames: list[dict]) -> tuple[list[Path], np.ndarray]:
    """Protocol labels: every clean version is a negative; glared versions are positives
    unless excluded, in which case they are dropped entirely."""
    paths, y = [], []
    for f in frames:
        paths.append(f["clean"])
        y.append(0.0)
        if not f["excluded"]:
            paths.append(f["glared"])
            y.append(1.0)
    return paths, np.array(y, dtype=np.float32)


# --- metrics --------------------------------------------------------------------------

def binary_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    tp = int(((pred == 1) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pct = lambda x: round(100 * x, 2)  # noqa: E731
    return {"precision": pct(precision), "recall": pct(recall), "f1": pct(f1),
            "accuracy": pct((tp + tn) / len(true)) if len(true) else 0.0,
            "specificity": pct(tn / (tn + fp)) if tn + fp else 0.0,
            "trigger_rate": pct((tp + fp) / len(true)) if len(true) else 0.0,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}}


def recall_of(pred: np.ndarray) -> dict:
    return {"n": int(len(pred)), "detected": int(pred.sum()),
            "recall": round(100 * float(pred.mean()), 2) if len(pred) else None}


# --- train ----------------------------------------------------------------------------

def build_head(kind: str) -> nn.Module:
    if kind == "lr":
        return nn.Linear(len(FEATURE_NAMES), 1)
    return nn.Sequential(nn.Linear(len(FEATURE_NAMES), 16), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(16, 1))


def cmd_train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    frames = load_frames(args.manifest, args.params, args.calibration, args.v2_root, "train")

    # Holdout for epoch selection, by frame id so a pair never straddles the split.
    ids = sorted(f["id"] for f in frames)
    random.Random(args.seed).shuffle(ids)
    hold_ids = set(ids[:round(len(ids) * args.holdout_frac)])
    fit = [f for f in frames if f["id"] not in hold_ids]
    hold = [f for f in frames if f["id"] in hold_ids]

    n_excl = sum(f["excluded"] for f in frames)
    print(f"gate v2 train  |  {len(frames)} train frames, {n_excl} glared {EXCLUDE_TYPE} "
          f"excluded from positives  |  model {args.model}  |  device cpu")
    print(f"  fit {len(fit)} frames, holdout {len(hold)} frames (epoch selection only)")

    fit_paths, y_fit = labelled_images(fit)
    hold_paths, y_hold = labelled_images(hold)
    print(f"  fit images: {int(y_fit.sum())} positive, {int((y_fit == 0).sum())} negative")
    x_fit = extract_all(fit_paths, args.workers)
    x_hold = extract_all(hold_paths, args.workers)

    feat_mean = x_fit.mean(axis=0)
    feat_std = x_fit.std(axis=0) + 1e-6
    norm = lambda x: (torch.from_numpy(x) - torch.from_numpy(feat_mean)) / torch.from_numpy(feat_std)  # noqa: E731
    x_fit_t, x_hold_t = norm(x_fit), norm(x_hold)
    y_fit_t, y_hold_t = torch.from_numpy(y_fit), torch.from_numpy(y_hold)

    head = build_head(args.model)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"\ntraining for {args.epochs} epochs...")
    log, best_f1, best_epoch, best_state = [], -1.0, 0, None
    n = x_fit_t.shape[0]
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            opt.zero_grad()
            loss = loss_fn(head(x_fit_t[idx]).squeeze(1), y_fit_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        head.eval()
        with torch.no_grad():
            hold_logits = head(x_hold_t).squeeze(1)
            hold_loss = loss_fn(hold_logits, y_hold_t).item()
        m = binary_metrics((hold_logits > 0).int().numpy(), y_hold.astype(int))
        log.append({"epoch": epoch, "train_loss": round(total / n, 5),
                    "holdout_loss": round(hold_loss, 5), **{k: m[k] for k in
                    ("precision", "recall", "f1", "accuracy")}})
        if m["f1"] > best_f1:
            best_f1, best_epoch = m["f1"], epoch
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}  loss {total / n:.4f}  holdout loss {hold_loss:.4f}  "
                  f"P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}")

    head.load_state_dict(best_state)
    model = GateClassifier(head, feat_mean, feat_std).eval()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.model_out)
    print(f"\n  best holdout F1 {best_f1:.2f} at epoch {best_epoch}")
    print(f"  wrote {rel(args.model_out)}")

    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    args.log_out.write_text(json.dumps({
        "config": {"model": args.model, "features": FEATURE_NAMES, "epochs": args.epochs,
                   "batch": args.batch, "lr": args.lr, "seed": args.seed,
                   "holdout_frac": args.holdout_frac, "selection": "max holdout F1",
                   "threshold": 0.5, "device": "cpu"},
        "data": {"train_frames": len(frames), "gate_excluded": n_excl,
                 "fit_frames": len(fit), "holdout_frames": len(hold),
                 "fit_positives": int(y_fit.sum()), "fit_negatives": int((y_fit == 0).sum())},
        "best_epoch": best_epoch, "best_holdout_f1": best_f1, "epochs_log": log,
    }, indent=2) + "\n")
    print(f"  wrote {rel(args.log_out)}")
    print("\nNext: python src/train_gate_v2.py score")
    return 0


# --- score ----------------------------------------------------------------------------

def measure_latency(model: GateClassifier, sample: Path, n: int = 100) -> float:
    """Warm up, then time predict() only (excludes disk I/O), mirroring detect.py."""
    img = cv2.imread(str(sample))
    for _ in range(5):
        model.predict(img)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(img)
    return (time.perf_counter() - t0) / n * 1000.0


def cmd_score(args: argparse.Namespace) -> int:
    if not args.model_out.is_file():
        sys.exit(f"error: {rel(args.model_out)} not found. Run `train` first.")
    model: GateClassifier = torch.load(args.model_out, weights_only=False).eval()
    frames = load_frames(args.manifest, args.params, args.calibration, args.v2_root, "val")
    print(f"gate v2 score  |  {len(frames)} val pairs  |  {rel(args.model_out)}")

    # Every val image, excluded ones included, so the predictions file covers the runtime
    # trigger rate too. Excluded images are filtered out of the protocol metrics below.
    images = [(f, v) for f in frames for v in ("clean", "glared")]
    x = extract_all([f[v] for f, v in images], args.workers)
    with torch.no_grad():
        feat = (torch.from_numpy(x) - model.feat_mean) / model.feat_std
        prob = torch.sigmoid(model(feat)).numpy()
    pred = (prob > 0.5).astype(int)

    # The batched path above must be exactly what model.predict() does at runtime.
    for i in (0, len(images) // 2, len(images) - 1):
        f, v = images[i]
        is_g, _ = model.predict(cv2.imread(str(f[v])))
        if int(is_g) != pred[i]:
            sys.exit(f"error: batched prediction disagrees with predict() on {f['id']} {v}")

    is_glared = np.array([v == "glared" for _, v in images])
    excluded = np.array([v == "glared" and f["excluded"] for f, v in images])
    scored = ~excluded
    gtype = np.array([f["glare_type"] for f, _ in images])
    sev = np.array([f["severity"] for f, _ in images])
    severe = np.array([f["severe"] for f, _ in images])
    pos = is_glared & scored

    metrics = {"protocol": "results/gate_protocol_v2.md",
               "split": "v2 val", "threshold": 0.5,
               "n_images_scored": int(scored.sum()), "n_positives": int(pos.sum()),
               "n_negatives": int((~is_glared).sum()), "n_excluded": int(excluded.sum())}
    metrics["overall"] = binary_metrics(pred[scored], is_glared[scored].astype(int))

    metrics["recall_per_type"] = {t: recall_of(pred[pos & (gtype == t)]) for t in GLARE_TYPES}
    metrics["recall_per_severity_band"] = {
        "severe": recall_of(pred[pos & severe]),
        "not_severe": recall_of(pred[pos & ~severe]),
        "per_type": {t: {"severe": recall_of(pred[pos & severe & (gtype == t)]),
                         "not_severe": recall_of(pred[pos & ~severe & (gtype == t)])}
                     for t in GLARE_TYPES},
    }
    metrics["recall_per_severity_bin"] = {
        f"[{lo:.1f},{hi:.1f})": recall_of(pred[pos & (sev >= lo) & ((sev < hi) | (hi == 1.0))])
        for lo, hi in zip(S_BINS[:-1], S_BINS[1:])}
    metrics["excluded_frames"] = {
        "note": "not scored as positives or negatives; gate firing on them only adds to "
                "the runtime trigger rate",
        **recall_of(pred[excluded])}
    metrics["runtime_trigger_rate_all_val"] = round(100 * float(pred.mean()), 2)

    latency = measure_latency(model, frames[0]["glared"])
    detect_json = REPO_ROOT / "results" / "detect_latency.json"
    detect_ms = json.loads(detect_json.read_text())["mean_ms"] if detect_json.is_file() else None
    metrics["latency_ms_per_frame"] = round(latency, 4)
    metrics["latency_pct_of_detection"] = round(latency / detect_ms * 100, 2) if detect_ms else None
    metrics["baseline_brightness_v1"] = BRIGHTNESS_V1
    if args.log_out.is_file():
        metrics["train"] = {k: v for k, v in json.loads(args.log_out.read_text()).items()
                            if k != "epochs_log"}

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")
    with args.pred_out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "version", "glare_type", "severity", "severe",
                    "gate_excluded", "label", "predicted_glare", "confidence"])
        for (f, v), p, c in zip(images, pred, prob):
            glared = v == "glared"
            label = "excluded" if glared and f["excluded"] else int(glared)
            w.writerow([f["id"], v, f["glare_type"] if glared else "",
                        f"{f['severity']:.6f}" if glared else "",
                        int(f["severe"]) if glared else "", int(glared and f["excluded"]),
                        label, int(p), f"{c:.4f}"])

    o = metrics["overall"]
    print("\n" + "=" * 68)
    print(f"V2 VAL  ({metrics['n_positives']} positives, {metrics['n_negatives']} negatives, "
          f"{metrics['n_excluded']} excluded)")
    print(f"  precision {o['precision']:6.2f}%   recall {o['recall']:6.2f}%   F1 {o['f1']:6.2f}%")
    print(f"  accuracy  {o['accuracy']:6.2f}%   specificity {o['specificity']:6.2f}%   "
          f"trigger rate {o['trigger_rate']:6.2f}%")
    print("  recall per type:")
    for t, r in metrics["recall_per_type"].items():
        print(f"    {t:<15} {r['recall']:6.2f}%  ({r['detected']}/{r['n']})")
    b = metrics["recall_per_severity_band"]
    print(f"  recall severe {b['severe']['recall']}%  not severe {b['not_severe']['recall']}%")
    for k, r in metrics["recall_per_severity_bin"].items():
        print(f"    s in {k}  {r['recall']}%  ({r['detected']}/{r['n']})")
    e = metrics["excluded_frames"]
    print(f"  fired on {e['detected']}/{e['n']} excluded frames (runtime trigger only)")
    print(f"  latency {latency:.3f} ms/frame")
    print(f"  wrote {rel(args.metrics_out)}\n  wrote {rel(args.pred_out)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("train", "score"))
    p.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "glare_manifest_v2.csv")
    p.add_argument("--params", type=Path, default=REPO_ROOT / "data" / "glare_params_v2.csv")
    p.add_argument("--calibration", type=Path,
                   default=REPO_ROOT / "results" / "severity_calibration.json")
    p.add_argument("--v2-root", type=Path, default=REPO_ROOT / "data" / "v2")
    p.add_argument("--model-out", type=Path, default=REPO_ROOT / "models" / "gate_v2.pt")
    p.add_argument("--log-out", type=Path, default=REPO_ROOT / "runs" / "gate_v2" / "train_log.json")
    p.add_argument("--metrics-out", type=Path, default=REPO_ROOT / "results" / "gate_v2_metrics.json")
    p.add_argument("--pred-out", type=Path, default=REPO_ROOT / "results" / "gate_v2_predictions.csv")
    p.add_argument("--model", choices=("mlp", "lr"), default="mlp")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return cmd_train(args) if args.command == "train" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
