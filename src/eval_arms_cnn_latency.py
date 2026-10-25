"""Paper numbers not covered by src/run_arms_v2.py: the C3 arm with the CNN gate, and
end-to-end CPU latency (mean and p95) for every arm.

    python src/eval_arms_cnn_latency.py
Output: results/arms_cnn_latency.json

Same 50/50 stream as run_arms_v2.py (1,491 KITTI val frames x clean/glared, v2 glare),
final restorer models/restore_daytime_net.pt (cached restorations from the daynet run),
detector models/yolo_c1_train.pt.

CNN GATE OVERLAP. Harshitha's CNN gate (models/gate_cnn_mixed.pt, src/train_gate_cnn.py)
took its KITTI training frames from KITTI VAL IDs (v1 glare, data/glare_labels_full.csv,
70% of the 748 v1-glared IDs). Those scenes are in this test stream. So the CNN-gate arm is
reported twice: on the full stream, and on the stream with those IDs removed (the clean
estimate). The other arms are also re-scored on the reduced stream for a fair comparison.

LATENCY. End to end per frame on the CPU, image decoded beforehand (I/O excluded),
warm-up excluded: C1 = detect; C2 = restore + detect; C3 = gate + (restore if fired) +
detect. Measured on a fixed seeded sample of the stream.
"""

from __future__ import annotations

import csv
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import eval_restorer_map as erm  # noqa: E402
import run_arms_v2 as ra  # noqa: E402
import shortcut_test as st  # noqa: E402
import train_gate_cnn  # noqa: E402

N_LATENCY, WARMUP, SEED = 300, 20, 42
OUT = REPO_ROOT / "results" / "arms_cnn_latency.json"


def cnn_gate_train_ids() -> set[str]:
    ids = [r["image_id"] for r in csv.DictReader((REPO_ROOT / "data/glare_labels_full.csv").open())
           if r["is_glared"] == "1"]
    tr, _ = train_gate_cnn.split(ids, 0.3, 42)        # her defaults: --test-frac 0.3 --seed 42
    return set(tr)


def main() -> int:
    ra.configure("daynet")
    params = {(r["split"], r["image_id"]): r for r in csv.DictReader(erm.PARAMS.open())}
    dirs = erm.v2_val_dirs(erm.V2_ROOT)
    ids, removed = erm.eligible(sorted(p.stem for p in dirs["glared"].glob("*.jpg")),
                                erm.excluded_ids(erm.CALIBRATION, erm.TUNING))
    overlap = cnn_gate_train_ids()
    streams = {"full": ids, "no_cnn_train_ids": [i for i in ids if i not in overlap]}
    restored = {(i, "glared"): ra.GLARED_CACHE / f"{i}.png" for i in ids}
    restored.update({(i, "clean"): ra.CLEAN_CACHE / f"{i}.png" for i in ids})
    assert all(p.is_file() for p in restored.values()), "run run_arms_v2.py --restorer daynet first"

    # ---- gates ----------------------------------------------------------------------
    t_bright, lr, _ = ra.train_gates(params)
    gate = torch.load(REPO_ROOT / "models" / "gate_cnn_mixed.pt", weights_only=False,
                      map_location="cpu").eval()
    full_stream = [(i, v) for i in ids for v in ("clean", "glared")]
    p_cnn = {}
    for k in range(0, len(full_stream), 64):
        b = full_stream[k:k + 64]
        x = torch.from_numpy(np.stack([train_gate_cnn.to_input(cv2.imread(
            str(erm.V2_ROOT / v / "images" / "val" / f"{i}.jpg"))) for i, v in b])).permute(0, 3, 1, 2)
        with torch.no_grad():
            p_cnn.update(zip(b, torch.sigmoid(gate(x)).numpy().tolist()))
    feats = dict(zip(full_stream, st.extract_all(
        [erm.V2_ROOT / v / "images" / "val" / f"{i}.jpg" for i, v in full_stream], ra.WORKERS)))
    lr_p = dict(zip(full_stream, lr.score(st.matrix([feats[k] for k in full_stream], ["gate"]))))
    fired_all = {
        "C1": {k: False for k in full_stream}, "C2": {k: True for k in full_stream},
        "C3_cnn": {k: p_cnn[k] > 0.5 for k in full_stream},
        "C3_features": {k: bool(lr_p[k] >= 0.5) for k in full_stream},
        "C3_brightness": {k: feats[k]["upper_third_mean_luma"] > t_bright for k in full_stream},
    }

    # ---- accuracy per stream ---------------------------------------------------------
    from train_baseline import evaluate, pick_device
    from ultralytics import YOLO
    device = pick_device(None)
    model = YOLO(str(erm.DEFAULT_WEIGHTS))
    ev = SimpleNamespace(device=device, imgsz=640, batch=16)
    results = {}
    for sname, sids in streams.items():
        stream = [(i, v) for i in sids for v in ("clean", "glared")]
        severe = {i: params[("val", i)]["severe"] == "1" for i in sids}
        groups = {"stream": stream, "clean": [(i, "clean") for i in sids],
                  "glared": [(i, "glared") for i in sids],
                  "glared_severe": [(i, "glared") for i in sids if severe[i]]}
        results[sname] = {"n_frames": len(sids)}
        for arm, fired in fired_all.items():
            ra.ARM_DECISIONS[f"{arm}_{sname}"] = fired
            row = {"trigger_rate": round(sum(fired[k] for k in stream) / len(stream), 4)}
            for g, items in groups.items():
                m = evaluate(model, ra.build(f"{arm}_{sname}", g, items, restored), ev)
                row[g] = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "n": len(items)}
            if arm.startswith("C3"):
                row["gate"] = ra.gate_metrics(fired, stream, params)
            results[sname][arm] = row
            print(f"  {sname:<18}{arm:<14} stream {row['stream']['mAP50']:.4f}  glared "
                  f"{row['glared']['mAP50']:.4f}  restores {row['trigger_rate']:.3f}", flush=True)
        c1, c2 = results[sname]["C1"]["stream"]["mAP50"], results[sname]["C2"]["stream"]["mAP50"]
        for arm in fired_all:
            v = results[sname][arm]["stream"]["mAP50"]
            results[sname][arm]["gain_kept"] = round((v - c1) / (c2 - c1), 4) if c2 > c1 else None

    # ---- end-to-end CPU latency ------------------------------------------------------
    torch.set_num_threads(4)
    import restore_daytime_net
    restorer = restore_daytime_net.DaytimeNetRestorer(device="cpu")
    det = YOLO(str(erm.DEFAULT_WEIGHTS))
    sample = random.Random(f"{SEED}:latency").sample(full_stream, N_LATENCY + WARMUP)
    imgs = [cv2.imread(str(erm.V2_ROOT / v / "images" / "val" / f"{i}.jpg")) for i, v in sample]

    def gate_cnn(im):
        x = torch.from_numpy(train_gate_cnn.to_input(im)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(gate(x))[0]) > 0.5

    from train_gate_classifier import extract_features as gf
    gate_fns = {"C3_cnn": gate_cnn,
                "C3_features": lambda im: bool(lr.score(gf(im)[None, :].astype(np.float64))[0] >= 0.5),
                "C3_brightness": lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[: im.shape[0] // 3].mean() > t_bright}
    latency = {}
    for arm in ("C1", "C2", "C3_cnn", "C3_features", "C3_brightness"):
        times, n_restored = [], 0
        for k, im in enumerate(imgs):
            t0 = time.perf_counter()
            fire = arm == "C2" or (arm.startswith("C3") and gate_fns[arm](im))
            x = restorer.restore(im) if fire else im
            det.predict(x, device="cpu", imgsz=640, verbose=False)
            dt = (time.perf_counter() - t0) * 1000
            if k >= WARMUP:
                times.append(dt)
                n_restored += fire
        latency[arm] = {"mean_ms": round(statistics.mean(times), 1),
                        "p95_ms": round(float(np.percentile(times, 95)), 1),
                        "fps_mean": round(1000 / statistics.mean(times), 1),
                        "restored_in_sample": round(n_restored / len(times), 3)}
        print(f"  latency {arm:<14} mean {latency[arm]['mean_ms']} ms  p95 {latency[arm]['p95_ms']} ms", flush=True)

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip() or platform.processor()
    OUT.write_text(json.dumps({
        "stream": "v2 KITTI val, 50/50 clean/glared; tuning IDs excluded",
        "restorer": "models/restore_daytime_net.pt", "detector": "models/yolo_c1_train.pt",
        "cnn_gate": "models/gate_cnn_mixed.pt (p > 0.5)",
        "cnn_gate_train_overlap": {"ids_in_stream": len(set(ids) & overlap),
                                   "note": "v1-glared KITTI val IDs the CNN gate trained on (her split: test_frac 0.3, seed 42)"},
        "map_device": device, "accuracy": results,
        "latency": {"device": f"CPU ({chip}), torch threads 4", "timed_frames": N_LATENCY,
                    "warmup_frames": WARMUP, "definition": "end to end per frame, image pre-decoded",
                    "per_arm": latency},
    }, indent=2, default=lambda o: o.item()) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
