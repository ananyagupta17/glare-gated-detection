"""C1 / C2 / C3 on the v2 val stream: the demo results table.

DEMO STATUS: v2 failed its shortcut test (results/shortcut_test.json: darkest-pixel AUC
0.998, noise level 0.72, artifact-only classifier 0.84). Every number this script writes
is labelled demo. The gate rows are the most affected: a gate trained on this data can
use the synthesis artifacts. Final numbers wait for a synthesis that passes.

THE STREAM

Each of the 1,491 eligible val frames (calibration/tuning IDs excluded) enters twice:
its clean version and its glared version, so the stream is exactly 50/50 (2,982 images).
Every arm sees the same stream; only what happens before detection differs.

  C1   never restore                     detect(frame)
  C2   always restore                    detect(restore(frame))
  C3   restore when the gate fires       detect(restore(frame) if gate(frame) else frame)

Restorer: the whole-frame Flare7K++ restorer (src/restore_seamless.py, mode whole), read
from the caches written by eval_restorer_map.py (glared) and this script (clean).

TWO GATES FOR C3, both fit on v2 TRAIN only, both under results/gate_protocol_v2.md
(glared veiling_wash frames with gate_excluded = 1 are left out of the positives):
  brightness   upper-third mean luma > t, t chosen for accuracy on train
  features     logistic regression on the gate classifier's 9 features
               (src/train_gate_classifier.py), as in shortcut_test.py
Harshitha's own v2 gate is not trained yet; these stand in for it.

COST

Per-frame latency = gate + detection + trigger_rate x restoration, each measured on CPU
with warm-up excluded and I/O outside the timer. Restoration time is the mean measured
by the full whole run (results/restorer_map_whole.json).

    python src/run_arms_v2.py
Output: results/arms_v2.json and results/arms_v2.md
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import eval_restorer_map as erm  # noqa: E402
import shortcut_test as st  # noqa: E402

RESTORER_DESCR = {
    "whole": "Flare7K++ night model, whole-frame (src/restore_seamless.py)",
    "daytime": "daytime classical veil subtraction (src/restore_daytime.py)",
    "daynet": "daytime trained U-Net (src/restore_daytime_net.py)",
    "daynet_bdd": "daytime U-Net trained on KITTI + BDD pairs (models/restore_daytime_net_kitti_bdd.pt)",
}
RESTORER = "whole"          # set by --restorer; "whole" keeps the original file names
GLARED_CACHE = REPO_ROOT / "data" / "restored" / "v2_whole" / "val"
CLEAN_CACHE = REPO_ROOT / "data" / "restored" / "v2_whole_clean" / "val"
WORK = REPO_ROOT / "datasets" / "eval_arms_v2"
KITTI_LABELS = REPO_ROOT / "datasets" / "kitti" / "labels" / "val"
OUT_JSON = REPO_ROOT / "results" / "arms_v2.json"
OUT_MD = REPO_ROOT / "results" / "arms_v2.md"
WORKERS = 7


def train_gates(params: dict) -> tuple[float, "st.LogReg", dict]:
    """Fit both gates on v2 train. Excluded glared frames are dropped from positives."""
    ids = sorted(p.stem for p in (erm.V2_ROOT / "glared" / "images" / "train").glob("*.jpg"))
    pos_ids = [i for i in ids if params[("train", i)]["gate_excluded"] != "1"]
    neg = st.extract_all([erm.V2_ROOT / "clean" / "images" / "train" / f"{i}.jpg" for i in ids], WORKERS)
    pos = st.extract_all([erm.V2_ROOT / "glared" / "images" / "train" / f"{i}.jpg" for i in pos_ids], WORKERS)

    lum_n = np.array([f["upper_third_mean_luma"] for f in neg])
    lum_p = np.array([f["upper_third_mean_luma"] for f in pos])
    cands = np.unique(np.r_[lum_n, lum_p])
    accs = [((lum_p > t).sum() + (lum_n <= t).sum()) / (len(lum_p) + len(lum_n)) for t in cands]
    t_best = float(cands[int(np.argmax(accs))])

    lr = st.LogReg().fit(np.vstack([st.matrix(neg, ["gate"]), st.matrix(pos, ["gate"])]),
                         np.r_[np.zeros(len(neg)), np.ones(len(pos))])
    info = {"train_negatives": len(neg), "train_positives": len(pos),
            "train_excluded_positives": len(ids) - len(pos_ids),
            "brightness_threshold_luma": round(t_best, 3),
            "brightness_train_accuracy": round(float(max(accs)), 4)}
    return t_best, lr, info


def gate_metrics(fired: dict, stream: list, params: dict) -> dict:
    """Precision/recall per the v2 gate protocol: excluded glared frames are not scored."""
    tp = fp = tn = fn = 0
    for image_id, version in stream:
        if version == "glared" and params[("val", image_id)]["gate_excluded"] == "1":
            continue
        f, positive = fired[(image_id, version)], version == "glared"
        tp += f and positive; fp += f and not positive
        tn += (not f) and not positive; fn += (not f) and positive
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "accuracy": round((tp + tn) / (tp + fp + tn + fn), 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def build(arm: str, group: str, items: list, restored: dict) -> Path:
    """Ultralytics dataset of symlinks: <id>_<version>.<ext>, label = the KITTI val label."""
    from train_baseline import write_yaml
    root = WORK / arm / group
    images, labels = root / "images" / "val", root / "labels" / "val"
    for d in (images, labels):
        if d.exists():
            for f in d.iterdir():
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)
    for image_id, version in items:
        src = restored[(image_id, version)] if arm_restores(arm, image_id, version) else \
            erm.V2_ROOT / version / "images" / "val" / f"{image_id}.jpg"
        name = f"{image_id}_{version}"
        (images / f"{name}{src.suffix}").symlink_to(src.resolve())
        (labels / f"{name}.txt").symlink_to((KITTI_LABELS / f"{image_id}.txt").resolve())
    return write_yaml(root / "dataset.yaml", root, "images/val", "images/val")


ARM_DECISIONS: dict[str, dict] = {}


def arm_restores(arm: str, image_id: str, version: str) -> bool:
    return ARM_DECISIONS[arm][(image_id, version)]


def time_ms(fn, inputs, warmup: int = 10) -> float:
    for x in inputs[:warmup]:
        fn(x)
    times = []
    for x in inputs[warmup:]:
        t = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.mean(times)


def configure(name: str) -> None:
    """Point caches and outputs at one restorer. 'whole' keeps the original names."""
    global RESTORER, GLARED_CACHE, CLEAN_CACHE, OUT_JSON, OUT_MD
    RESTORER = name
    GLARED_CACHE = REPO_ROOT / "data" / "restored" / f"v2_{name}" / "val"
    CLEAN_CACHE = REPO_ROOT / "data" / "restored" / f"v2_{name}_clean" / "val"
    suffix = "" if name == "whole" else f"_{name}"
    OUT_JSON = REPO_ROOT / "results" / f"arms_v2{suffix}.json"
    OUT_MD = REPO_ROOT / "results" / f"arms_v2{suffix}.md"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorer", choices=sorted(RESTORER_DESCR), default="whole")
    configure(ap.parse_args().restorer)
    whole = json.loads((REPO_ROOT / "results" / f"restorer_map_{RESTORER}.json").read_text())
    dirs = erm.v2_val_dirs(erm.V2_ROOT)
    ids, removed = erm.eligible(sorted(p.stem for p in dirs["glared"].glob("*.jpg")),
                                erm.excluded_ids(erm.CALIBRATION, erm.TUNING))
    params = {(r["split"], r["image_id"]): r for r in csv.DictReader(erm.PARAMS.open())}
    stream = [(i, v) for i in ids for v in ("clean", "glared")]
    print(f"stream: {len(ids)} val frames x 2 = {len(stream)} images (50/50); excluded {removed}")

    # restorations: glared from the full whole run, clean restored here (cached)
    missing = [i for i in ids if not (GLARED_CACHE / f"{i}.png").is_file()]
    if missing:
        sys.exit(f"error: {len(missing)} restored glared frames missing; run eval_restorer_map.py whole")
    clean_info = erm.restore_all(RESTORER, ids, dirs["clean"], CLEAN_CACHE, threads=4)
    print(f"clean restorations: {clean_info}")
    restored = {(i, "glared"): GLARED_CACHE / f"{i}.png" for i in ids}
    restored.update({(i, "clean"): CLEAN_CACHE / f"{i}.png" for i in ids})

    # gates, fit on train, applied to the original val frames
    print("fitting gates on v2 train...")
    t_bright, lr, gate_info = train_gates(params)
    val_feats = dict(zip(stream, st.extract_all(
        [erm.V2_ROOT / v / "images" / "val" / f"{i}.jpg" for i, v in stream], WORKERS)))
    fired_bright = {k: f["upper_third_mean_luma"] > t_bright for k, f in val_feats.items()}
    scores = lr.score(st.matrix([val_feats[k] for k in stream], ["gate"]))
    fired_lr = {k: bool(s >= 0.5) for k, s in zip(stream, scores)}

    ARM_DECISIONS.update({
        "C1": {k: False for k in stream},
        "C2": {k: True for k in stream},
        "C3_brightness": fired_bright,
        "C3_features": fired_lr,
    })
    severe = {i: params[("val", i)]["severe"] == "1" for i in ids}
    groups = {
        "stream": stream,
        "clean": [(i, "clean") for i in ids],
        "glared": [(i, "glared") for i in ids],
        "glared_severe": [(i, "glared") for i in ids if severe[i]],
        "glared_non_severe": [(i, "glared") for i in ids if not severe[i]],
    }

    from train_baseline import evaluate, pick_device
    from ultralytics import YOLO
    device = pick_device(None)
    model = YOLO(str(erm.DEFAULT_WEIGHTS))
    ev = SimpleNamespace(device=device, imgsz=640, batch=16)
    results: dict = {}
    for arm, decisions in ARM_DECISIONS.items():
        results[arm] = {"trigger_rate": round(sum(decisions.values()) / len(decisions), 4),
                        "trigger_rate_clean": round(sum(decisions[k] for k in groups["clean"]) / len(ids), 4),
                        "trigger_rate_glared": round(sum(decisions[k] for k in groups["glared"]) / len(ids), 4)}
        for group, items in groups.items():
            m = evaluate(model, build(arm, group, items, restored), ev)
            results[arm][group] = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "n": len(items)}
            print(f"  {arm:<14}{group:<18} n={len(items):>5}  mAP50 {m['mAP50']:.4f}  "
                  f"mAP50-95 {m['mAP50_95']:.4f}")

    # latency (CPU, warm-up excluded, decoded images in memory)
    import cv2
    sample = [cv2.imread(str(erm.V2_ROOT / v / "images" / "val" / f"{i}.jpg")) for i, v in stream[:120]]
    cpu_model = YOLO(str(erm.DEFAULT_WEIGHTS))
    detect_ms = time_ms(lambda im: cpu_model.predict(im, device="cpu", imgsz=640, verbose=False), sample)

    def bright(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return g[: g.shape[0] // 3].mean() > t_bright

    from train_gate_classifier import extract_features as gate_features

    def feat_gate(im):
        return lr.score(gate_features(im)[None, :].astype(np.float64)) >= 0.5

    gate_ms = {"C1": 0.0, "C2": 0.0, "C3_brightness": time_ms(bright, sample),
               "C3_features": time_ms(feat_gate, sample)}
    restore_ms = 1000 * whole["restore"]["mean_restore_seconds"]
    for arm in results:
        lat = gate_ms[arm] + detect_ms + results[arm]["trigger_rate"] * restore_ms
        results[arm]["latency_ms"] = {"gate": round(gate_ms[arm], 3), "detect": round(detect_ms, 2),
                                      "restore_expected": round(results[arm]["trigger_rate"] * restore_ms, 1),
                                      "total_mean": round(lat, 1)}

    gates = {"C3_brightness": gate_metrics(fired_bright, stream, params),
             "C3_features": gate_metrics(fired_lr, stream, params)}
    payload = {
        "status": "DEMO -- v2 failed its shortcut test (results/shortcut_test.json); not final",
        "stream": {"frames": len(ids), "images": len(stream), "clean_fraction": 0.5,
                   "excluded_ids": removed},
        "detector": {"weights": "models/yolo_c1_train.pt", "mAP_device": device,
                     "latency_device": "cpu"},
        "restorer": {"name": RESTORER, "description": RESTORER_DESCR[RESTORER], "device": "cpu",
                     "mean_ms_per_frame": round(restore_ms, 1),
                     "source": f"results/restorer_map_{RESTORER}.json", "clean_restore": clean_info},
        "gates": {**gate_info, "metrics_val": gates,
                  "protocol": "results/gate_protocol_v2.md (excluded glared frames not scored)"},
        "arms": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(payload)
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)} and {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


def write_markdown(p: dict) -> None:
    a, g = p["arms"], p["gates"]["metrics_val"]
    rows = []
    for arm in ("C1", "C2", "C3_brightness", "C3_features"):
        r = a[arm]
        rows.append(f"| {arm} | {r['stream']['mAP50']:.3f} | {r['clean']['mAP50']:.3f} | "
                    f"{r['glared']['mAP50']:.3f} | {r['glared_severe']['mAP50']:.3f} | "
                    f"{r['trigger_rate']:.1%} | {r['latency_ms']['total_mean']:.0f} |")
    OUT_MD.write_text(
        "# C1 / C2 / C3 on v2 val (DEMO)\n\n"
        "**Not final:** v2 failed its shortcut test (`results/shortcut_test.json`). "
        "Gate rows are the most affected.\n\n"
        f"Stream: {p['stream']['frames']} val frames, each clean and glared "
        f"({p['stream']['images']} images, 50/50). Detector `models/yolo_c1_train.pt`. "
        f"Restorer: {p['restorer']['description']}, {p['restorer']['mean_ms_per_frame']:.0f} ms/frame on CPU.\n\n"
        "| Arm | mAP50 stream | mAP50 clean | mAP50 glared | mAP50 glared severe | "
        "Trigger rate | Latency ms/frame (CPU) |\n|---|---|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n\n"
        "| Gate | Precision | Recall | F1 | Accuracy |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | "
                    f"{v['accuracy']:.3f} |" for k, v in g.items()) + "\n")


if __name__ == "__main__":
    sys.exit(main())
