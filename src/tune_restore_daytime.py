"""Tune src/restore_daytime.py on v2 TRAIN pairs only, by detection; write
results/restore_daytime_tuning.json.

Objective: mean per-frame F1 of the detector on restore(glared), over 200 seeded train
pairs, subject to restore(clean) costing at most 0.01 mean F1 against the clean frame, so
the restorer is safe on frames without glare. PSNR is recorded for information only: the
synthetic glare also makes the camera stop down, and PSNR punishes that global brightness
change even when objects become more visible.

Detector: models/yolo_baseline.pt, trained on 1,198 KITTI VAL frames, so it never saw a
train frame (the same reasoning as src/calibrate_severity.py). Per-frame F1 as in
calibration: IoU 0.5, conf 0.25, class-aware greedy matching. No val frame is opened.

    python src/tune_restore_daytime.py
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import calibrate_severity as cal  # noqa: E402
import restore_daytime as rd  # noqa: E402

V2 = REPO_ROOT / "data" / "v2"
N_PAIRS, SEED, MAX_CLEAN_F1_LOSS = 200, 42, 0.01
DETECTOR = REPO_ROOT / "models" / "yolo_baseline.pt"


def main() -> int:
    from train_baseline import pick_device
    from ultralytics import YOLO
    device = pick_device(None)
    model = YOLO(str(DETECTOR))

    ids = sorted(p.stem for p in (V2 / "glared" / "images" / "train").glob("*.jpg"))
    pick = sorted(random.Random(f"{SEED}:daytime-tuning").sample(ids, N_PAIRS))
    glared = [cv2.imread(str(V2 / "glared" / "images" / "train" / f"{i}.jpg")) for i in pick]
    clean = [cv2.imread(str(V2 / "clean" / "images" / "train" / f"{i}.jpg")) for i in pick]
    gts = [cal.read_labels(REPO_ROOT / "datasets" / "kitti" / "labels" / "train" / f"{i}.txt",
                           img.shape[1], img.shape[0]) for i, img in zip(pick, clean)]

    def f1s(images):
        out = []
        for k in range(0, len(images), 32):
            for r, gt in zip(model.predict(images[k:k + 32], conf=cal.CONF, device=device,
                                           verbose=False), gts[k:k + 32]):
                b = r.boxes
                pred = np.concatenate([b.cls.cpu().numpy()[:, None], b.xyxy.cpu().numpy(),
                                       b.conf.cpu().numpy()[:, None]], 1) if len(b) \
                    else np.zeros((0, 6), np.float32)
                out.append(cal.frame_f1(pred, gt))
        return np.asarray(out)

    f1_clean, f1_glared = f1s(clean), f1s(glared)
    print(f"{N_PAIRS} train pairs: F1 clean {f1_clean.mean():.4f}, glared {f1_glared.mean():.4f}")

    grid = {"omega": [0.5, 0.8, 1.0], "sigma": [6.0, 12.0], "baseline_pct": [0, 5, 20],
            "stretch_pct": [0, 99.0], "clahe_clip": [0, 1.5]}
    rows, t0 = [], time.time()
    for values in itertools.product(*grid.values()):
        o = dict(zip(grid, values))
        params = {**rd.DEFAULTS, **o}
        fr = f1s([rd.restore(g, params) for g in glared])
        fc = f1s([rd.restore(c, params) for c in clean])
        rows.append({**o, "f1_restored_glared": round(float(fr.mean()), 4),
                     "f1_restored_clean": round(float(fc.mean()), 4),
                     "clean_f1_loss": round(float(f1_clean.mean() - fc.mean()), 4),
                     "clean_ok": bool(f1_clean.mean() - fc.mean() <= MAX_CLEAN_F1_LOSS)})
        print(f"  {o}  glared {fr.mean():.4f}  clean {fc.mean():.4f}"
              f"  {'ok' if rows[-1]['clean_ok'] else 'CLEAN LOSS'}  ({time.time() - t0:.0f}s)",
              flush=True)

    ok = [r for r in rows if r["clean_ok"]]
    if not ok:
        print("no setting keeps clean frames within the allowed F1 loss; nothing chosen")
        best = None
    else:
        best = max(ok, key=lambda r: r["f1_restored_glared"])
    chosen = {**rd.DEFAULTS, **{k: best[k] for k in grid}} if best else None

    psnr = None
    if chosen:
        psnr = {"glared": round(float(np.mean([cv2.PSNR(c, g) for g, c in zip(glared, clean)])), 3),
                "restored": round(float(np.mean([cv2.PSNR(c, rd.restore(g, chosen))
                                                 for g, c in zip(glared, clean)])), 3)}
        ts = []
        for _ in range(25):
            t = time.perf_counter()
            rd.restore(glared[0], chosen)
            ts.append((time.perf_counter() - t) * 1000)
    out = REPO_ROOT / "results" / "restore_daytime_tuning.json"
    out.write_text(json.dumps({
        "split": "train only", "n_pairs": N_PAIRS, "seed": SEED, "ids": pick,
        "detector": "models/yolo_baseline.pt (never saw a KITTI train frame)",
        "objective": f"max mean per-frame F1 on restore(glared) s.t. clean F1 loss <= {MAX_CLEAN_F1_LOSS}",
        "baseline_f1": {"clean": round(float(f1_clean.mean()), 4), "glared": round(float(f1_glared.mean()), 4)},
        "chosen": chosen, "chosen_scores": best, "psnr_info_only": psnr,
        "latency_ms_median_cpu": round(float(np.median(ts[5:])), 2) if chosen else None,
        "grid": grid, "results": rows,
    }, indent=2) + "\n")
    print(f"chosen {chosen}\n  scores {best}\n  psnr {psnr}\nwrote {out.relative_to(REPO_ROOT)}")
    if not chosen:
        out.rename(out.with_name("restore_daytime_tuning_failed.json"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
