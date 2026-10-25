"""Real-glare test: the final pipeline on BDD100K frames (built by src/bdd_prepare.py).

    python src/eval_real_bdd.py
Output: results/real_bdd_results.json and results/real_bdd_results.md

Frames: the confirmed real-glare BDD100K val frames and an equal number of matched clean
frames (data/bdd_real_manifest.csv), as one stream. Nothing here was trained on BDD.

Pipeline (same components as the live demo):
    gate      Harshitha's CNN gate, models/gate_cnn_mixed.pt (glare if p > 0.5)
    restorer  trained daytime U-Net, models/restore_daytime_net.pt
    detector  YOLOv8n trained on clean KITTI, models/yolo_c1_train.pt

Arms on the same stream: C1 never restore, C2 always restore, C3 restore when the gate
fires. Reported per arm: mAP@50 on the whole stream, on glare frames and on clean frames,
mean per-frame F1, the restoration trigger rate, and latency per stage.

SCORING. mAP@50 is computed here, not by Ultralytics, because of the fixed class mapping
(src/bdd_prepare.py, results/bdd_class_map.json): the detector's 'van' predictions count
as 'car', since BDD labels vans as cars. Classes scored: car, truck, pedestrian, cyclist,
tram (those with any ground truth in the set). AP per class is the all-point
interpolated area under the precision-recall curve (VOC 2010+), predictions at conf
>= 0.001, greedy matching at IoU >= 0.5 within a frame. Per-frame F1 uses conf 0.25.

The KITTI-trained detector has never seen BDD's camera or resolution, so absolute mAP is
lower than on KITTI; the comparison between arms on identical frames is what matters.
Real glare and clean frames are different scenes (matched only on weather and scene type),
so glare-vs-clean differences carry that confound.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import bdd_prepare as bp  # noqa: E402
import calibrate_severity as cal  # noqa: E402
import restore_daytime_net  # noqa: E402
import train_gate_cnn  # noqa: E402  (GateCNN, needed to unpickle the gate)

OUT_JSON = REPO_ROOT / "results" / "real_bdd_results.json"
OUT_MD = REPO_ROOT / "results" / "real_bdd_results.md"
NAMES = {0: "car", 2: "truck", 3: "pedestrian", 5: "cyclist", 6: "tram"}
MAP_CONF = 0.001
REALTIME_MS = {"30 fps": 1000 / 30, "10 Hz (KITTI camera rate)": 100.0}   # fixed before results


def read_yolo(path: Path, w: int, h: int) -> np.ndarray:
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                c, cx, cy, bw, bh = map(float, line.split()[:5])
                rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def remap(pred: np.ndarray) -> np.ndarray:
    pred = pred.copy()
    for src, dst in bp.EVAL_MERGE.items():
        pred[pred[:, 0] == src, 0] = dst
    return pred


def average_precision(preds: list[tuple[str, np.ndarray]], gts: dict[str, np.ndarray]) -> dict:
    """preds: [(frame, (n,6) cls x1 y1 x2 y2 conf)]; gts: frame -> (m,5). mAP@50 over classes."""
    per_class = {}
    for c, name in NAMES.items():
        n_gt = sum(int((g[:, 0] == c).sum()) for g in gts.values())
        if n_gt == 0:
            continue
        dets = [(float(p[5]), f, p[1:5]) for f, ps in preds for p in ps if p[0] == c]
        dets.sort(key=lambda d: -d[0])
        used = {f: np.zeros(int((g[:, 0] == c).sum()), bool) for f, g in gts.items()}
        gt_boxes = {f: g[g[:, 0] == c, 1:5] for f, g in gts.items()}
        tp = np.zeros(len(dets))
        for k, (_, f, box) in enumerate(dets):
            boxes = gt_boxes.get(f)
            if boxes is None or not len(boxes):
                continue
            ov = cal.iou(box, boxes)
            ov[used[f]] = -1
            j = int(np.argmax(ov))
            if ov[j] >= cal.IOU_MATCH:
                used[f][j] = True
                tp[k] = 1
        if not len(dets):
            per_class[name] = {"AP50": 0.0, "n_gt": n_gt}
            continue
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        recall = ctp / n_gt
        precision = ctp / np.maximum(ctp + cfp, 1e-9)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[1.0], precision, [0.0]])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        per_class[name] = {"AP50": round(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), 4),
                           "n_gt": n_gt}
    m = [v["AP50"] for v in per_class.values()]
    return {"mAP50": round(float(np.mean(m)), 4) if m else None, "per_class": per_class}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorer-weights", type=Path,
                    default=REPO_ROOT / "models" / "restore_daytime_net.pt")
    args = ap.parse_args()
    args.restorer_weights = args.restorer_weights.resolve()
    global OUT_JSON, OUT_MD
    if args.restorer_weights.name != "restore_daytime_net.pt":
        OUT_JSON = REPO_ROOT / "results" / f"real_bdd_results_{args.restorer_weights.stem}.json"
        OUT_MD = OUT_JSON.with_suffix(".md")
    if not bp.MANIFEST.is_file():
        sys.exit("error: data/bdd_real_manifest.csv not found; run src/bdd_prepare.py build first")
    rows = list(csv.DictReader(bp.MANIFEST.open()))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from ultralytics import YOLO
    detector = YOLO(str(REPO_ROOT / "models" / "yolo_c1_train.pt"))
    gate = torch.load(REPO_ROOT / "models" / "gate_cnn_mixed.pt", weights_only=False,
                      map_location=device).eval()
    restorer = restore_daytime_net.DaytimeNetRestorer(args.restorer_weights, device=device)

    def detect(img):
        r = detector.predict(img, conf=MAP_CONF, device=device, verbose=False)[0]
        b = r.boxes
        if not len(b):
            return np.zeros((0, 6), np.float32)
        return remap(np.concatenate([b.cls.cpu().numpy()[:, None], b.xyxy.cpu().numpy(),
                                     b.conf.cpu().numpy()[:, None]], 1))

    def gate_p(img):
        x = torch.from_numpy(train_gate_cnn.to_input(img)).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(torch.sigmoid(gate(x))[0])

    warm = cv2.imread(str(bp.OUT / ("glare" if rows[0]["is_glared"] == "1" else "clean")
                          / "images" / "val" / rows[0]["name"]))
    for _ in range(5):
        gate_p(warm); detect(restorer.restore(warm))

    frames, gts = [], {}
    pred = {"C1": [], "C2": [], "C3": []}
    f1 = {a: {"glare": [], "clean": []} for a in pred}
    lat = {"gate": [], "restore": [], "detect": []}
    per_frame = {"C1": [], "C2": [], "C3": []}       # end-to-end ms per frame, per arm
    gate_rec = []
    print(f"evaluating {len(rows)} real BDD frames on {device}...")
    for n, r in enumerate(rows, 1):
        kind = "glare" if r["is_glared"] == "1" else "clean"
        img = cv2.imread(str(bp.OUT / kind / "images" / "val" / r["name"]))
        gt = read_yolo(bp.OUT / kind / "labels" / "val" / f"{Path(r['name']).stem}.txt",
                       img.shape[1], img.shape[0])
        fid = f"{kind}/{r['name']}"
        gts[fid] = gt

        t0 = time.perf_counter(); p = gate_p(img); t1 = time.perf_counter()
        restored = restorer.restore(img); t2 = time.perf_counter()
        d_plain = detect(img); t3 = time.perf_counter()
        d_rest = detect(restored)
        lat["gate"].append(1000 * (t1 - t0)); lat["restore"].append(1000 * (t2 - t1))
        lat["detect"].append(1000 * (t3 - t2))
        fired = p > 0.5
        gate_rec.append((kind, fired, p))
        g, rs, dt = lat["gate"][-1], lat["restore"][-1], lat["detect"][-1]
        per_frame["C1"].append(dt)
        per_frame["C2"].append(rs + dt)
        per_frame["C3"].append(g + (rs if fired else 0.0) + dt)
        for arm, d in (("C1", d_plain), ("C2", d_rest), ("C3", d_rest if fired else d_plain)):
            pred[arm].append((fid, d))
            f1[arm][kind].append(cal.frame_f1(d[d[:, 5] >= cal.CONF], gt))
        frames.append((fid, kind))
        if n % 50 == 0:
            print(f"  {n}/{len(rows)}", flush=True)

    sets = {"stream": [f for f, _ in frames], "glare": [f for f, k in frames if k == "glare"],
            "clean": [f for f, k in frames if k == "clean"]}
    results = {}
    tg, tr, td = (statistics.mean(lat[k]) for k in ("gate", "restore", "detect"))
    trig = sum(1 for _, f, _ in gate_rec if f) / len(gate_rec)
    for arm in pred:
        results[arm] = {s: average_precision([x for x in pred[arm] if x[0] in set(ids)],
                                             {f: gts[f] for f in ids}) for s, ids in sets.items()}
        results[arm]["f1_mean"] = {k: round(float(np.mean(v)), 4) for k, v in f1[arm].items()}
        rate = {"C1": 0.0, "C2": 1.0, "C3": trig}[arm]
        results[arm]["trigger_rate"] = round(rate, 4)
        results[arm]["latency_ms"] = round(statistics.mean(per_frame[arm]), 1)
        results[arm]["latency_p95_ms"] = round(float(np.percentile(per_frame[arm], 95)), 1)
        results[arm]["latency_split_ms"] = {
            "gate": round(tg, 2) if arm == "C3" else 0.0,
            "restoration": round(rate * tr, 2), "detection": round(td, 2)}
        results[arm]["realtime"] = {k: {"mean_ok": results[arm]["latency_ms"] <= v,
                                        "p95_ok": results[arm]["latency_p95_ms"] <= v}
                                    for k, v in REALTIME_MS.items()}
        results[arm]["glare_drop"] = None
    c1 = results["C1"]
    for arm in pred:
        a = results[arm]
        drop = c1["clean"]["mAP50"] - c1["glare"]["mAP50"]
        a["glare_drop_C1"] = round(drop, 4)
        a["recovery_share"] = (round((a["glare"]["mAP50"] - c1["glare"]["mAP50"]) / drop, 4)
                               if drop > 0 else None)
    pos = [f for k, f, _ in gate_rec if k == "glare"]
    neg = [f for k, f, _ in gate_rec if k == "clean"]
    tp, fn, fp, tn = sum(pos), len(pos) - sum(pos), sum(neg), len(neg) - sum(neg)
    gate_m = {"precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
              "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
              "accuracy": round((tp + tn) / len(gate_rec), 4), "tp": tp, "fn": fn, "fp": fp, "tn": tn}

    review = json.loads((bp.REVIEW / "review_meta.json").read_text()) if (bp.REVIEW / "review_meta.json").is_file() else None
    OUT_JSON.write_text(json.dumps({
        "data": {"source": "BDD100K val (100k images), via Kaggle mirror solesensei/solesensei_bdd100k",
                 "frames_glare": len(sets["glare"]), "frames_clean": len(sets["clean"]),
                 "manifest": "data/bdd_real_manifest.csv", "review": review},
        "pipeline": {"gate": "models/gate_cnn_mixed.pt (CNN, p > 0.5)",
                     "restorer": str(args.restorer_weights.relative_to(REPO_ROOT)),
                     "detector": "models/yolo_c1_train.pt"},
        "device": device, "scoring": "own mAP@50 (VOC all-point), van->car merge; F1 at conf 0.25",
        "latency_ms_mean": {"gate": round(tg, 2), "restore": round(tr, 2), "detect": round(td, 2)},
        "latency_ms_p95": {k: round(float(np.percentile(v, 95)), 2) for k, v in lat.items()},
        "realtime_targets_ms": REALTIME_MS,
        "glare_drop_note": "clean-set minus glare-set mAP@50 under C1; real frames are not paired",
        "gate": gate_m, "arms": results,
    }, indent=2, default=lambda o: o.item()) + "\n")

    def row(arm, label):
        a = results[arm]
        rec = "–" if a["recovery_share"] is None or arm == "C1" else f"{a['recovery_share']:.0%}"
        rt = "yes" if a["realtime"]["30 fps"]["p95_ok"] else ("10 Hz only" if a["realtime"]["10 Hz (KITTI camera rate)"]["p95_ok"] else "no")
        sp = a["latency_split_ms"]
        return (f"| {label} | {a['stream']['mAP50']:.3f} | {a['glare']['mAP50']:.3f} | {a['clean']['mAP50']:.3f} | "
                f"{rec} | {a['trigger_rate']:.0%} | {a['latency_ms']:.0f} / {a['latency_p95_ms']:.0f} | "
                f"{sp['gate']:.1f} / {sp['restoration']:.1f} / {sp['detection']:.1f} | {rt} |")
    OUT_MD.write_text(
        "# Real glare (BDD100K): final pipeline\n\n"
        f"{len(sets['glare'])} real-glare frames + {len(sets['clean'])} matched clean frames; nothing trained on BDD. "
        f"Device {device}.\n\n"
        "| Arm | mAP@50 stream | mAP@50 glare frames | mAP@50 clean frames | Glare loss recovered | Restores | ms/frame mean / p95 | gate / restoration / detection ms | Real-time (30 fps, p95) |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join([row("C1", "C1 · never restore"), row("C2", "C2 · always restore"),
                     row("C3", "C3 · gated (CNN gate)")]) + "\n\n"
        f"Glare drop (C1, clean-set minus glare-set mAP@50, different scenes): {c1['clean']['mAP50'] - c1['glare']['mAP50']:.3f}.\n\n"
        f"Gate on real frames: accuracy {gate_m['accuracy']:.3f}, precision {gate_m['precision']:.3f}, recall {gate_m['recall']:.3f}, "
        f"F1 {gate_m['f1']:.3f} (TP {tp}, FN {fn}, FP {fp}, TN {tn}).\n\n"
        f"Latency per stage (mean): gate {tg:.1f} ms, restoration {tr:.1f} ms, detection {td:.1f} ms.\n")
    print(OUT_MD.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
