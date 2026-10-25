"""Figures for the manuscript, written to paper_figures/.

  fig_real_glare.png   real BDD100K glare: camera vs always-restored, detections drawn.
                       Frame choice is stated, not picked: the real-glare test frame whose
                       per-frame F1 change from restoration is the MEDIAN over the 72 frames.
  fig_dark_pixel_leak.png   the v2 shortcut: distribution of 1st-percentile luma for clean vs
                       glared KITTI val frames (v2), with the brightness baseline for scale.

    python src/make_paper_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import bdd_prepare as bp  # noqa: E402
import calibrate_severity as cal  # noqa: E402
import eval_real_bdd as erb  # noqa: E402
import restore_daytime_net  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO_ROOT / "paper_figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_CLEAN, C_GLARE = "#2a78d6", "#eb6834"


def draw(img, pred, gt):
    out = img.copy()
    used = np.zeros(len(gt), bool); tp = np.zeros(len(pred), bool)
    for j in (np.argsort(-pred[:, 5]) if len(pred) else []):
        cand = np.where((gt[:, 0] == pred[j, 0]) & ~used)[0]
        if len(cand):
            ov = cal.iou(pred[j, 1:5], gt[cand, 1:5]); k = int(np.argmax(ov))
            if ov[k] >= cal.IOU_MATCH:
                used[cand[k]] = True; tp[j] = True
    for g, h in zip(gt, used):
        if not h: cv2.rectangle(out, tuple(map(int, g[1:3])), tuple(map(int, g[3:5])), (0, 215, 255), 3)
    for p, ok in zip(pred, tp):
        cv2.rectangle(out, tuple(map(int, p[1:3])), tuple(map(int, p[3:5])), (60, 200, 60) if ok else (40, 40, 230), 3)
    return out[:, :, ::-1], int(tp.sum()), int((~tp).sum()), int((~used).sum())


def real_glare_figure():
    import torch
    from ultralytics import YOLO
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    det = YOLO(str(REPO_ROOT / "models" / "yolo_c1_train.pt"))
    rest = restore_daytime_net.DaytimeNetRestorer(device=dev)
    rows = [r for r in csv.DictReader(bp.MANIFEST.open()) if r["is_glared"] == "1"]
    recs = []
    for r in rows:
        img = cv2.imread(str(bp.OUT / "glare" / "images" / "val" / r["name"]))
        gt = erb.read_yolo(bp.OUT / "glare" / "labels" / "val" / f"{Path(r['name']).stem}.txt", img.shape[1], img.shape[0])
        res = rest.restore(img)
        d0, d1 = [erb.remap(x) for x in (det_predict(det, img, dev), det_predict(det, res, dev))]
        f0, f1 = cal.frame_f1(d0, gt), cal.frame_f1(d1, gt)
        recs.append((f1 - f0, r["name"], img, res, d0, d1, gt))
    recs.sort(key=lambda t: (t[0], t[1]))
    gain, name, img, res, d0, d1, gt = recs[len(recs) // 2]
    a = draw(img, d0, gt); b = draw(res, d1, gt)
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.9))
    for axi, (im, tp, fp, miss), t in zip(ax, (a, b), ("Camera frame (C1)", "Restored (C2)")):
        axi.imshow(im); axi.axis("off")
        axi.set_title(f"{t}: {tp} found, {fp} false, {miss} missed (of {len(gt)})", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_real_glare.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    gains = [t[0] for t in recs]
    return {"frame": name, "f1_change": round(gain, 4), "selection": "median per-frame F1 change over 72 real-glare frames",
            "median_f1_change": round(float(np.median(gains)), 4),
            "frames_improved": sum(g > 0 for g in gains), "frames_worse": sum(g < 0 for g in gains),
            "frames_unchanged": sum(g == 0 for g in gains)}


def det_predict(det, img, dev):
    r = det.predict(img, conf=cal.CONF, device=dev, verbose=False)[0]; b = r.boxes
    if not len(b):
        return np.zeros((0, 6), np.float32)
    return np.concatenate([b.cls.cpu().numpy()[:, None], b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)


def leak_figure():
    import eval_restorer_map as erm
    dirs = erm.v2_val_dirs(erm.V2_ROOT)
    ids = sorted(p.stem for p in dirs["glared"].glob("*.jpg"))
    p1 = {v: [float(np.percentile(cv2.cvtColor(cv2.imread(str(dirs[v] / f"{i}.jpg")), cv2.COLOR_BGR2GRAY), 1)) for i in ids]
          for v in ("clean", "glared")}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.arange(0, 141, 4)
    ax.hist(p1["clean"], bins=bins, color=C_CLEAN, alpha=0.85, label="Clean", edgecolor="white", linewidth=0.6)
    ax.hist(p1["glared"], bins=bins, color=C_GLARE, alpha=0.7, label="Glared (v2)", edgecolor="white", linewidth=0.6)
    ax.set_xlabel("1st-percentile luma of the frame (0-255)", color=INK2)
    ax.set_ylabel("Frames", color=INK2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    sc = json.loads((REPO_ROOT / "results" / "shortcut_test.json").read_text())["checks"]["darkest_pixel"]
    ax.set_title(f"Darkest pixels alone separate the classes (AUC {sc['p1_luma']['value']:.3f}; "
                 f"brightness baseline {sc['baseline_upper_third_auc']:.3f})", fontsize=9.5, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_dark_pixel_leak.png", dpi=200); plt.close(fig)
    return {"n_pairs": len(ids), "median_p1": {k: float(np.median(v)) for k, v in p1.items()}}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    meta = {"real_glare": real_glare_figure(), "dark_pixel_leak": leak_figure()}
    (OUT / "figures.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=1))
