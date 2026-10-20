"""Does anything other than the glare tell clean and glared v2 frames apart?

v1's black-lift floor was a perfect classifier by itself (min pixel 0-2 clean vs 31-121
glared), and the gate scored 100% because of it. This test runs on the v2 data and fails
loudly if a shortcut like that is still there.

The pass thresholds below were fixed BEFORE any v2 frame was generated, and are written
to results/shortcut_test_thresholds.json so they can be compared against later. Changing
them after seeing results defeats the point; if one has to change, record why in git.

DATA

  * Probes (single features): all 1,496 val frames, clean versions vs glared versions.
  * Classifiers: trained on the train split (5,985 + 5,985), scored on val.
  * AUC is two-sided, max(AUC, 1 - AUC): a feature that is LOWER on glared frames is just
    as much a giveaway as one that is higher.

FEATURES (all on the decoded JPEG, luma = Rec.601 on sRGB)

  p1_luma, min_luma       darkest-pixel probes, where v1 leaked
  upper_third_mean_luma   the brightness-threshold gate's feature; the baseline
  mean_luma               whole-frame mean brightness      } pass/fail on the mild band
  midtone_histogram       16-bin luma histogram over       } only; glare itself moves
                          [64, 192), fraction of pixels    } these at higher severity
  jpeg_quality_est        quality recovered from the luminance quantisation table
  blockiness              8x8 block-edge gradient / interior gradient
  hf_mf_ratio             high- to mid-frequency energy ratio. A ratio, so a pure contrast
                          change cancels; what remains is spectrum shape, which is where
                          a compositing or noise artifact would show
  noise_level             Immerkaer's fast noise estimate
  pixel_count             H x W

Usage:
    python src/shortcut_test.py --write-thresholds   # record thresholds, no data needed
    python src/shortcut_test.py                      # run (needs v2 data)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glare_synth as gs  # noqa: E402
from train_gate_classifier import FEATURE_NAMES as GATE_FEATURES  # noqa: E402
from train_gate_classifier import extract_features as gate_features  # noqa: E402

REPO_ROOT = gs.REPO_ROOT
THRESHOLDS_JSON = REPO_ROOT / "results" / "shortcut_test_thresholds.json"
RESULTS_JSON = REPO_ROOT / "results" / "shortcut_test.json"
SEED = 42                  # glare_synth.py's default; the determinism check re-renders with it

NUISANCE = ["jpeg_quality_est", "blockiness", "hf_mf_ratio", "noise_level", "pixel_count"]

THRESHOLDS = {
    "registered": "2026-10-20, before any v2 frame was generated",
    "auc_definition": "two-sided ROC AUC, max(AUC, 1 - AUC); val clean versions (1,496) "
                      "vs val glared versions (1,496)",
    "checks": {
        "darkest_pixel": {
            "features": ["p1_luma", "min_luma"],
            "rule": "each AUC <= AUC(upper_third_mean_luma)",
            "reason": "v1's constant floor made min pixel a perfect classifier. Veiling "
                      "glare really does lift the blacks, so the darkest pixels may carry "
                      "some signal, but no more than plain brightness does.",
        },
        "single_feature": {
            "max_auc": {
                "jpeg_quality_est": 0.60,
                "blockiness": 0.60,
                "hf_mf_ratio": 0.60,
                "pixel_count": 0.60,
                "noise_level": 0.65,
            },
            "noise_level_reason": "Shot-noise variance grows with signal (variance = a * "
                                  "signal + b). Glared frames carry more light, so they "
                                  "genuinely carry more noise, as a real sensor would. "
                                  "That allows a small excess over 0.60. Above 0.65 the "
                                  "separation comes from our noise model, not physics.",
        },
        "mild_band_brightness": {
            "max_auc": {"mean_luma": 0.60, "midtone_histogram": 0.60},
            "scope": "pass/fail on the mild band only: val glared frames in the mild band "
                     "vs their paired clean versions. The all-frames AUC is reported with "
                     "no pass/fail.",
            "midtone_histogram_scoring": "logistic regression on the 16 bins, trained on "
                                         "the train split (all bands); AUC of its scores "
                                         "on val",
            "reason": "Glare adds light and auto-exposure only partly compensates, and "
                      "veiling glare pushes shadows into the midtones, so across all "
                      "severities these two probes measure the glare itself. Mild glare "
                      "should barely register; if these separate mild frames, something "
                      "other than the glare is doing it.",
            "changed": "2026-10-20, before generation: moved from single_feature "
                       "(all frames, <= 0.60) to the mild band only",

            "status": "not_evaluable",
            "status_reason": "The mild severity band was dropped after calibration "
                             "(results/severity_scale_decision.json, commit ddf2673), so "
                             "there are no mild frames to evaluate. Marked 2026-10-20; the "
                             "registered limit is kept for the record.",
        },
        "brightness_removed_lr": {
            "features": NUISANCE,
            "max_accuracy": 0.65,
            "model": "logistic regression, standardised features, L2 1e-3",
            "train": "train split", "test": "val split",
            "reason": "With every brightness feature removed, only artifacts are left. A "
                      "classifier on artifacts alone must stay near chance.",
        },
        "mild_vs_severe": {
            "features": GATE_FEATURES,
            "model": "logistic regression on the gate classifier's 9 features "
                     "(src/train_gate_classifier.py), trained on the train split",
            "band_accuracy": "accuracy on val glared frames in the band plus their "
                             "paired clean frames (balanced 50/50)",
            "min_gap_points": 10,
            "rule": "accuracy(severe) - accuracy(mild) >= 10 percentage points",
            "reason": "Mild glare barely changes the frame. If mild frames are as easy to "
                      "spot as severe ones, something other than the glare gives them away.",

            "status": "not_evaluable",
            "status_reason": "The mild severity band was dropped after calibration "
                             "(results/severity_scale_decision.json, commit ddf2673), so "
                             "there are no mild frames to evaluate. Marked 2026-10-20; the "
                             "registered limit is kept for the record.",
        },
        "determinism": {
            "n_frames": 5,
            "rule": "re-rendering val frames gives sha256-identical JPEGs to "
                    "data/glare_v2_sha256.txt",
        },
    },
}


# --- features -------------------------------------------------------------------------

_STD_LUMA_Q = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def jpeg_quality(im: Image.Image) -> float:
    """Invert libjpeg's quality scaling from the luminance table."""
    tables = getattr(im, "quantization", None)
    if not tables:
        return float("nan")
    scale = 100.0 * float(np.mean(np.asarray(tables[0], dtype=np.float64) / _STD_LUMA_Q))
    return (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale


def blockiness(luma: np.ndarray) -> float:
    ratios = []
    for d in (np.abs(np.diff(luma, axis=1)), np.abs(np.diff(luma, axis=0)).T):
        edge = d[:, 7::8].mean()
        inner = np.delete(d, np.s_[7::8], axis=1).mean()
        ratios.append(edge / max(inner, 1e-9))
    return float(np.mean(ratios))


def hf_mf_ratio(luma: np.ndarray) -> float:
    hf = np.abs(cv2.Laplacian(luma, cv2.CV_64F)).mean()
    mf = np.abs(cv2.Laplacian(cv2.GaussianBlur(luma, (0, 0), 2.0), cv2.CV_64F)).mean()
    return float(hf / max(mf, 1e-9))


def noise_level(luma: np.ndarray) -> float:
    """Immerkaer (1996): sigma from the image's response to a Laplacian-difference mask."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = luma.shape
    conv = cv2.filter2D(luma, cv2.CV_64F, kernel)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


def features(path: str) -> dict:
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        q = jpeg_quality(im)
        rgb = np.asarray(im.convert("RGB"))
    luma = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float64)
    h = luma.shape[0]
    hist, _ = np.histogram(luma, bins=16, range=(64, 192))
    return {
        "p1_luma": float(np.percentile(luma, 1)),
        "min_luma": float(luma.min()),
        "upper_third_mean_luma": float(luma[: h // 3].mean()),
        "mean_luma": float(luma.mean()),
        "midtone_histogram": (hist / luma.size).tolist(),
        "jpeg_quality_est": q,
        "blockiness": blockiness(luma),
        "hf_mf_ratio": hf_mf_ratio(luma),
        "noise_level": noise_level(luma),
        "pixel_count": float(luma.size),
        "gate": gate_features(np.ascontiguousarray(rgb[:, :, ::-1])).tolist(),
    }


# --- statistics -----------------------------------------------------------------------

def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling; P(score(pos) > score(neg))."""
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def two_sided(a: float) -> float:
    return max(a, 1.0 - a)


class LogReg:
    """Standardised L2 logistic regression by Newton's method. Deterministic."""

    def __init__(self, l2: float = 1e-3, iters: int = 50):
        self.l2, self.iters = l2, iters

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogReg":
        self.mu = x.mean(axis=0)
        self.sd = x.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        z = np.hstack([np.ones((len(x), 1)), (x - self.mu) / self.sd])
        w = np.zeros(z.shape[1])
        reg = self.l2 * np.eye(z.shape[1])
        reg[0, 0] = 0.0
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))
            grad = z.T @ (p - y) / len(y) + reg @ w
            hess = (z * (p * (1 - p))[:, None]).T @ z / len(y) + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = np.hstack([np.ones((len(x), 1)), (x - self.mu) / self.sd])
        return 1.0 / (1.0 + np.exp(-np.clip(z @ self.w, -30, 30)))


# --- data -----------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    if not gs.MANIFEST.is_file():
        sys.exit(f"error: {gs.rel(gs.MANIFEST)} not found. Generate v2 data first.")
    return list(csv.DictReader(gs.MANIFEST.open()))


def load_params() -> dict[tuple[str, str], dict]:
    return {(r["split"], r["image_id"]): r for r in csv.DictReader(gs.PARAMS.open())}


def image_path(split: str, image_id: str, glared: bool) -> Path:
    return gs.OUT_ROOT / ("glared" if glared else "clean") / "images" / split / f"{image_id}.jpg"


def extract_all(paths: list[Path], workers: int) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(features, [str(p) for p in paths], chunksize=16))


def matrix(feats: list[dict], keys: list[str]) -> np.ndarray:
    cols = []
    for k in keys:
        col = np.asarray([f[k] for f in feats], dtype=np.float64)
        cols.append(col if col.ndim == 2 else col[:, None])
    return np.hstack(cols)


# --- checks ---------------------------------------------------------------------------

def check(name: str, value: float, limit: float, passed: bool, **extra) -> dict:
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name:<34} {value:.4f}   limit {limit:.4f}")
    return {"value": round(value, 4), "limit": round(limit, 4), "passed": bool(passed), **extra}


def determinism(params: dict, n: int) -> dict:
    expected = {}
    for line in gs.CHECKSUMS.read_text().splitlines():
        h, p = line.split("  ", 1)
        expected[p] = h
    val_ids = sorted(i for (s, i) in params if s == "val")
    picks = [val_ids[k * (len(val_ids) - 1) // max(n - 1, 1)] for k in range(n)]
    mismatched = []
    n_of_type = {}
    for (split, _), row in params.items():
        if split == "val":
            n_of_type[row["glare_type"]] = n_of_type.get(row["glare_type"], 0) + 1
    for image_id in picks:
        r = params[("val", image_id)]
        rgb = gs.read_rgb(gs.KITTI / "images" / "val" / f"{image_id}.png")
        # Recompute s exactly from the stratum; the params file rounds it to 6 decimals,
        # and the rounded value would not reproduce the generated bytes.
        s = gs.draw_severity("val", image_id, int(r["stratum"]), n_of_type[r["glare_type"]], SEED)
        if abs(s - float(r["severity"])) > 1e-6:
            mismatched.append(f"{image_id} (severity {r['severity']} != recomputed {s:.6f})")
            continue
        spec = gs.sample_frame(rgb, "val", image_id, r["glare_type"], s,
                               f"stratum {r['stratum']}", SEED)
        clean, glared, _ = gs.render_pair(rgb, spec)
        for version, data in (("clean", clean), ("glared", glared)):
            key = gs.rel(image_path("val", image_id, version == "glared"))
            if hashlib.sha256(data).hexdigest() != expected.get(key):
                mismatched.append(key)
    print(f"  [{'PASS' if not mismatched else 'FAIL'}] determinism ({n} val frames)"
          f"{'' if not mismatched else ': ' + ', '.join(mismatched)}")
    return {"frames": picks, "mismatched": mismatched, "passed": not mismatched}


def v1_reference(workers: int) -> dict | None:
    """The same brightness probes on v1 data, so the report shows the leak closing."""
    csv_path = REPO_ROOT / "data" / "glare_labels_full.csv"
    glared_dir = REPO_ROOT / "data" / "glared_full"
    if not csv_path.is_file() or not glared_dir.is_dir():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    g = [glared_dir / f"{r['image_id']}.png" for r in rows if r["is_glared"] == "1"]
    c = [gs.KITTI / "images" / "val" / f"{r['image_id']}.png" for r in rows
         if r["is_glared"] == "0"]
    if not all(p.is_file() for p in g):
        return None
    fg, fc = extract_all(g, workers), extract_all(c, workers)
    out = {k: round(two_sided(auc(matrix(fg, [k]).ravel(), matrix(fc, [k]).ravel())), 4)
           for k in ("p1_luma", "min_luma", "upper_third_mean_luma", "mean_luma")}
    out["note"] = "v1 is unpaired: 748 glared IDs vs 748 different clean IDs"
    return out


def run(workers: int) -> int:
    manifest = load_manifest()
    params = load_params()
    pairs = {s: sorted({r["image_id"] for r in manifest if r["split"] == s}) for s in gs.SPLITS}
    if not pairs["val"] or not pairs["train"]:
        sys.exit("error: the manifest needs both train and val rows")

    print(f"extracting features: {len(pairs['train'])} train + {len(pairs['val'])} val pairs")
    feats = {}
    for split in gs.SPLITS:
        for glared in (False, True):
            paths = [image_path(split, i, glared) for i in pairs[split]]
            feats[(split, glared)] = extract_all(paths, workers)

    vc, vg = feats[("val", False)], feats[("val", True)]
    tc, tg = feats[("train", False)], feats[("train", True)]
    th = THRESHOLDS["checks"]
    results: dict = {"thresholds": THRESHOLDS, "n_val_pairs": len(pairs["val"]),
                     "n_train_pairs": len(pairs["train"]), "checks": {}}
    out = results["checks"]

    def probe(key: str) -> float:
        return two_sided(auc(matrix(vg, [key]).ravel(), matrix(vc, [key]).ravel()))

    print("\ndarkest-pixel probes vs brightness baseline")
    baseline = probe("upper_third_mean_luma")
    print(f"  baseline AUC(upper_third_mean_luma) = {baseline:.4f}")
    out["darkest_pixel"] = {"baseline_upper_third_auc": round(baseline, 4)}
    for key in th["darkest_pixel"]["features"]:
        a = probe(key)
        out["darkest_pixel"][key] = check(f"AUC({key})", a, baseline, a <= baseline)

    print("\nsingle-feature probes")
    out["single_feature"] = {}
    for key, limit in th["single_feature"]["max_auc"].items():
        a = probe(key)
        out["single_feature"][key] = check(f"AUC({key})", a, limit, a <= limit)

    print("\nbrightness probes (all frames, report only; mild-band check not_evaluable)")
    out["mild_band_brightness"] = {"status": th["mild_band_brightness"]["status"]}
    for key in th["mild_band_brightness"]["max_auc"]:
        if key == "midtone_histogram":
            lr = LogReg().fit(np.vstack([matrix(tc, [key]), matrix(tg, [key])]),
                              np.r_[np.zeros(len(tc)), np.ones(len(tg))])
            sg, sc = lr.score(matrix(vg, [key])), lr.score(matrix(vc, [key]))
        else:
            sg, sc = matrix(vg, [key]).ravel(), matrix(vc, [key]).ravel()
        a_all = two_sided(auc(sg, sc))
        print(f"  [info] AUC({key}) all frames {a_all:.4f}")
        out["mild_band_brightness"][key] = {"all_frames_auc": round(a_all, 4)}

    print("\nbrightness-removed logistic regression (train -> val)")
    keys = th["brightness_removed_lr"]["features"]
    lr = LogReg().fit(np.vstack([matrix(tc, keys), matrix(tg, keys)]),
                      np.r_[np.zeros(len(tc)), np.ones(len(tg))])
    pred = np.r_[lr.score(matrix(vc, keys)), lr.score(matrix(vg, keys))] >= 0.5
    truth = np.r_[np.zeros(len(vc)), np.ones(len(vg))].astype(bool)
    acc = float((pred == truth).mean())
    limit = th["brightness_removed_lr"]["max_accuracy"]
    out["brightness_removed_lr"] = check("accuracy", acc, limit, acc <= limit)

    print("\nmild vs severe: not_evaluable (no mild band), skipped")
    out["mild_vs_severe"] = {"status": th["mild_vs_severe"]["status"]}

    print("\ndeterminism")
    out["determinism"] = determinism(params, th["determinism"]["n_frames"])

    print("\nv1 reference (report only)")
    ref = v1_reference(workers)
    results["v1_reference"] = ref
    if ref:
        for k, v in ref.items():
            if k != "note":
                print(f"  v1 AUC({k}) = {v}")
    else:
        print("  v1 data not present, skipped")

    flat = [out["darkest_pixel"][k]["passed"] for k in th["darkest_pixel"]["features"]]
    flat += [v["passed"] for v in out["single_feature"].values()]
    flat += [out["brightness_removed_lr"]["passed"], out["determinism"]["passed"]]
    results["not_evaluable"] = ["mild_band_brightness", "mild_vs_severe"]
    results["passed"] = all(flat)
    RESULTS_JSON.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n{'ALL CHECKS PASSED' if results['passed'] else 'SHORTCUT TEST FAILED'}"
          f"  -> {gs.rel(RESULTS_JSON)}")
    return 0 if results["passed"] else 1


def write_thresholds() -> None:
    THRESHOLDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_JSON.write_text(json.dumps(THRESHOLDS, indent=2) + "\n")
    print(f"wrote {gs.rel(THRESHOLDS_JSON)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write-thresholds", action="store_true",
                   help="write the thresholds file and exit (no data needed)")
    p.add_argument("--workers", type=int, default=gs.parse_args([]).workers)
    args = p.parse_args(argv)
    if args.write_thresholds:
        write_thresholds()
        return 0
    if THRESHOLDS_JSON.is_file() and json.loads(THRESHOLDS_JSON.read_text()) != THRESHOLDS:
        sys.exit(f"error: {gs.rel(THRESHOLDS_JSON)} differs from the thresholds in this "
                 f"script. They were registered before generation; reconcile in git first.")
    write_thresholds()
    return run(args.workers)


if __name__ == "__main__":
    sys.exit(main())
