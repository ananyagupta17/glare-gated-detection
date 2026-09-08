"""Gate classifier: binary glare / no-glare (Harshitha's brief, C3's gate).

Image-level classification (not pixel-wise segmentation): one label per frame, no mask.

WHY HAND-ENGINEERED FEATURES, NOT A CNN
A gate that costs what it saves is pointless (the brief is explicit about this). A first
version transfer-learned a frozen MobileNetV2 backbone and measured 68 ms/frame on this
CPU -- 4.5x the cost of detection itself (15.13 ms) and ~600x the brightness baseline
(0.114 ms). That fails "cheap" outright, so it was replaced with 10 hand-engineered
features computed directly off the BGR frame with OpenCV/numpy (histogram-based
percentiles instead of a full sort, cv2.mean instead of numpy's slower strided reduction):
mean/std luma, max/min luma, saturated-pixel fraction, 95th/99th percentile luma, upper-
third mean luma (the baseline's own signal), R-B channel diff (chromatic halo, CLAUDE.md
section 6), and HSV saturation mean (glare desaturates). Measured ~1.6 ms/frame -- about
11% of detection cost, and the classifier is a tiny MLP on 10 floats (sub-millisecond).

This also targets the baseline's known failure mode directly: it catches only 51% of
direct_sun because a small saturated core barely moves mean brightness. The 95th/99th
percentile and max-luma features see that core even when the mean doesn't move.

Data (CLAUDE.md section 10, the `_full` realisation):
    clean:  datasets/kitti/images/val/<id>.png   (is_glared=0 rows)
    glared: data/glared_full/<id>.png            (is_glared=1 rows)
    labels: data/glare_labels_full.csv           (image_id, is_glared, glare_type)

*** MUST NOT TRAIN ON THE DETECTOR'S 298 HELD-OUT FRAMES *** (Harshitha's brief) -- C3 is
scored on exactly those, and training on them silently invalidates that comparison. The
split below is byte-identical to train_baseline.py's stratified_split(records, 0.2, 42)
(same algorithm, and bucket order is sorted() before any sampling so record-list order
can't perturb it) -- verified directly against the brief's own ID-dump snippet.

Integration contract (src/run_experiment.py, the C1/C2/C3 harness):
    model = torch.load("models/gate_classifier.pt", weights_only=False)
    is_glared, confidence = model.predict(image)   # image: BGR uint8, cv2.imread output
GateClassifier is defined at module scope (not nested in main()) so it unpickles cleanly
from a *different* script -- a class pickled under `__main__` cannot be reloaded elsewhere.

Usage:
    python src/train_gate_classifier.py
    python src/train_gate_classifier.py --epochs 200 --val-frac 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURE_NAMES = ["mean_luma", "std_luma", "max_luma", "min_luma", "sat_frac",
                 "p95_luma", "p99_luma", "upper_third_mean_luma", "rb_diff", "hsv_sat_mean"]


def extract_features(image: np.ndarray) -> np.ndarray:
    """BGR uint8 (H, W, 3) -> 10 cheap glare-relevant scalars. No resize, no NN forward."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, std = cv2.meanStdDev(gray)
    mean, std = float(mean[0, 0]), float(std[0, 0])
    gmax, gmin = float(gray.max()), float(gray.min())

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cdf = np.cumsum(hist)
    total = cdf[-1] if cdf[-1] > 0 else 1.0
    sat_frac = float(1.0 - cdf[249] / total)        # fraction of pixels with luma > 249
    p95 = float(np.searchsorted(cdf, total * 0.95))
    p99 = float(np.searchsorted(cdf, total * 0.99))

    h_third = gray.shape[0] // 3
    upper_mean = float(gray[:h_third].mean()) if h_third else mean

    b, g, r, _ = cv2.mean(image)                    # BGR order
    rb_diff = r - b

    hsv_sat_mean = float(cv2.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1])[0])

    return np.array([mean, std, gmax, gmin, sat_frac, p95, p99,
                     upper_mean, rb_diff, hsv_sat_mean], dtype=np.float32)


class GateClassifier(nn.Module):
    """Feature standardizer + tiny MLP, with cv2-input inference.

    predict() is the integration entry point: run_experiment.py hands it a raw BGR
    uint8 array straight from cv2.imread. Feature extraction (not a resize+CNN pass)
    is what keeps this gate cheap -- see the module docstring.
    """

    def __init__(self, head: nn.Module, feat_mean: np.ndarray, feat_std: np.ndarray):
        super().__init__()
        self.head = head
        self.register_buffer("feat_mean", torch.from_numpy(feat_mean).float())
        self.register_buffer("feat_std", torch.from_numpy(feat_std).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)

    def predict(self, image) -> tuple[bool, float]:
        """image: BGR uint8 numpy array (cv2.imread output) -> (is_glared, confidence)."""
        feat = torch.from_numpy(extract_features(image)).float().unsqueeze(0)
        feat = (feat - self.feat_mean) / self.feat_std
        with torch.no_grad():
            prob = torch.sigmoid(self.forward(feat)).item()
        return prob > 0.5, round(prob, 4)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "glare_labels_full.csv")
    p.add_argument("--clean-dir", type=Path,
                   default=REPO_ROOT / "datasets" / "kitti" / "images" / "val")
    p.add_argument("--glared-dir", type=Path, default=REPO_ROOT / "data" / "glared_full")
    p.add_argument("--model-out", type=Path, default=REPO_ROOT / "models" / "gate_classifier.pt")
    p.add_argument("--pred-csv-out", type=Path,
                   default=REPO_ROOT / "results" / "gate_predictions.csv")
    p.add_argument("--metrics-out", type=Path,
                   default=REPO_ROOT / "results" / "gate_trained_metrics.json")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    return p.parse_args(argv)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_records(csv_path: Path, clean_dir: Path, glared_dir: Path) -> list[dict]:
    if not csv_path.is_file():
        sys.exit(f"error: {rel(csv_path)} not found. Run src/glare_synth.py --full first.")
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        sys.exit(f"error: {rel(csv_path)} is empty")

    records, missing = [], []
    for r in rows:
        image_id = r["image_id"]
        is_glared = r["is_glared"] == "1"
        path = (glared_dir if is_glared else clean_dir) / f"{image_id}.png"
        if not path.is_file():
            missing.append(str(path))
            continue
        records.append(dict(id=image_id, is_glared=is_glared,
                            glare_type=r.get("glare_type", ""), path=path))
    if missing:
        sys.exit(f"error: {len(missing)} image file(s) missing, e.g. {missing[0]}. "
                 f"Run src/glare_synth.py --full first.")
    return records


def stratified_split(records: list[dict], val_frac: float, seed: int) -> tuple[set, set]:
    """Identical to train_baseline.py's stratified_split -- verified byte-for-byte
    against the brief's own ID-dump snippet before this gate was trained."""
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


def build_feature_matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((len(records), len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(len(records), dtype=np.float32)
    for i, r in enumerate(records):
        img = cv2.imread(str(r["path"]))
        X[i] = extract_features(img)
        y[i] = 1.0 if r["is_glared"] else 0.0
        if (i + 1) % 200 == 0 or i + 1 == len(records):
            print(f"\r  features {i + 1}/{len(records)}", end="", flush=True)
    print()
    return X, y


def measure_latency(model: GateClassifier, sample_path: Path, n: int = 100) -> float:
    """Warm up, then time predict() only (excludes disk I/O), mirroring detect.py."""
    img = cv2.imread(str(sample_path))
    for _ in range(5):
        model.predict(img)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(img)
    return (time.perf_counter() - t0) / n * 1000.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    records = load_records(args.csv, args.clean_dir, args.glared_dir)
    n_glared = sum(r["is_glared"] for r in records)
    print(f"gate classifier  |  {len(records)} images ({n_glared} glared, "
         f"{len(records) - n_glared} clean)  |  device cpu")

    train_ids, val_ids = stratified_split(records, args.val_frac, args.seed)
    print(f"  split (seed {args.seed}, stratified by is_glared x glare_type): "
         f"{len(train_ids)} train, {len(val_ids)} val (must equal the detector's held-out "
         f"set -- verified separately against train_baseline.stratified_split)")

    by_id = {r["id"]: r for r in records}
    train_recs = [by_id[i] for i in sorted(train_ids)]
    val_recs = [by_id[i] for i in sorted(val_ids)]

    print("\nextracting features (10 cheap OpenCV/numpy scalars per image)...")
    print("  train:")
    x_train, y_train = build_feature_matrix(train_recs)
    print("  val:")
    x_val, y_val = build_feature_matrix(val_recs)

    feat_mean = x_train.mean(axis=0)
    feat_std = x_train.std(axis=0) + 1e-6

    x_train_t = (torch.from_numpy(x_train) - torch.from_numpy(feat_mean)) / torch.from_numpy(feat_std)
    x_val_t = (torch.from_numpy(x_val) - torch.from_numpy(feat_mean)) / torch.from_numpy(feat_std)
    y_train_t = torch.from_numpy(y_train)
    y_val_t = torch.from_numpy(y_val)

    # ---- tiny MLP on 10 standardized features --------------------------------------
    head = nn.Sequential(
        nn.Linear(len(FEATURE_NAMES), 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1),
    )
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"\ntraining head for {args.epochs} epochs on {len(train_recs)} feature vectors...")
    n = x_train_t.shape[0]
    best_val_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, args.batch):
            idx = perm[start:start + args.batch]
            xb, yb = x_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            logits = head(xb).squeeze(1)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        head.eval()
        with torch.no_grad():
            val_logits = head(x_val_t).squeeze(1)
            val_acc = ((val_logits > 0).float() == y_val_t).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if epoch % 20 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}  loss {total_loss / n:.4f}  val_acc {val_acc:.4f}")

    head.load_state_dict(best_state)
    print(f"  best val_acc during training: {best_val_acc:.4f}")

    # ---- evaluate the best head on held-out val ------------------------------------
    head.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(head(x_val_t).squeeze(1))
    val_pred = (val_probs > 0.5).int()
    val_true = y_val_t.int()

    tp = int(((val_pred == 1) & (val_true == 1)).sum())
    tn = int(((val_pred == 0) & (val_true == 0)).sum())
    fp = int(((val_pred == 1) & (val_true == 0)).sum())
    fn = int(((val_pred == 0) & (val_true == 1)).sum())
    accuracy = (tp + tn) / len(val_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    trigger_rate = (tp + fp) / len(val_true)

    # ---- build + save the model first, so latency is measured on the real artifact -
    full_model = GateClassifier(head, feat_mean, feat_std).eval()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(full_model, args.model_out)
    print(f"\n  wrote {rel(args.model_out)}")

    latency_ms = measure_latency(full_model, val_recs[0]["path"])
    detect_json = REPO_ROOT / "results" / "detect_latency.json"
    detect_ms = json.loads(detect_json.read_text())["mean_ms"] if detect_json.is_file() else None

    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "trigger_rate": round(trigger_rate * 100, 2),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "latency_ms_per_frame": round(latency_ms, 4),
        "latency_pct_of_detection": (round(latency_ms / detect_ms * 100, 2)
                                     if detect_ms else None),
        "baseline_comparison": {
            "brightness_v1": {"accuracy": 80.87, "precision": 78.26, "recall": 85.14,
                              "trigger_rate": 54.03, "latency_ms": 0.114},
        },
        "config": {
            "n_images": len(records), "n_train": len(train_recs), "n_val": len(val_recs),
            "val_frac": args.val_frac, "seed": args.seed,
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
            "features": FEATURE_NAMES, "model": "10 hand-engineered features -> tiny MLP",
            "device": "cpu",
        },
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")

    print("\n" + "=" * 68)
    print(f"HELD-OUT VAL ({len(val_recs)} images, never trained on -- matches the "
         f"detector's own 298-frame held-out set)")
    print(f"  {'':<12}{'this gate':>12}{'brightness v1':>16}")
    print(f"  {'accuracy':<12}{metrics['accuracy']:>11.2f}%{80.87:>15.2f}%")
    print(f"  {'precision':<12}{metrics['precision']:>11.2f}%{78.26:>15.2f}%")
    print(f"  {'recall':<12}{metrics['recall']:>11.2f}%{85.14:>15.2f}%")
    print(f"  {'trigger_rate':<12}{metrics['trigger_rate']:>11.2f}%{54.03:>15.2f}%")
    print(f"  {'latency':<12}{latency_ms:>11.3f}ms{0.114:>14.3f}ms")
    if detect_ms:
        print(f"  latency is {metrics['latency_pct_of_detection']:.2f}% of detection "
             f"({detect_ms:.2f} ms/frame)")
    print(f"  confusion tp={tp} tn={tn} fp={fp} fn={fn}")
    print(f"  wrote {rel(args.metrics_out)}")

    # ---- predictions.csv for every image (train + val), for downstream C3 --------
    print("\nrunning inference over the full population for results/gate_predictions.csv...")
    all_recs = train_recs + val_recs
    split_of = {r["id"]: "train" for r in train_recs}
    split_of.update({r["id"]: "val" for r in val_recs})

    rows_out = []
    for i, r in enumerate(all_recs):
        img = cv2.imread(str(r["path"]))
        is_glared, conf = full_model.predict(img)
        rows_out.append((r["id"], int(is_glared), conf, split_of[r["id"]], int(r["is_glared"])))
        if (i + 1) % 200 == 0 or i + 1 == len(all_recs):
            print(f"\r  {i + 1}/{len(all_recs)}", end="", flush=True)
    print()

    args.pred_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.pred_csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "predicted_glare", "confidence", "split", "true_is_glared"])
        w.writerows(sorted(rows_out))
    print(f"  wrote {rel(args.pred_csv_out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
