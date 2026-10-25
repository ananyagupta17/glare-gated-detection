"""Bootstrap 95% confidence intervals for the paper's main numbers.

    python src/bootstrap_ci.py
Output: results/bootstrap_ci.json

PROTOCOL. Resampling unit = the frame: on KITTI a clean/glared pair is drawn together (it is
one scene); on BDD100K each test frame is drawn independently, glare and clean sets
separately. 1,000 resamples, seed 42, percentile intervals (2.5th, 97.5th).

mAP here is computed by this script (VOC all-point interpolated AP at IoU 0.5, class-aware
greedy matching within an image, detections at conf >= 0.001, up to 300 per image),
averaged over the classes that have ground truth. Ultralytics interpolates differently, so
point estimates differ slightly from the tables; each interval is reported with its own
point estimate. Matching is done once per image, so resampling only re-weights images.

KITTI: v2 val pairs (tuning IDs excluded), detector models/yolo_c1_train.pt, restorer
models/restore_daytime_net.pt (cached outputs), gates as in src/eval_arms_cnn_latency.py.
BDD100K: data/bdd_real_manifest.csv, van counted as car.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import bdd_prepare as bp  # noqa: E402
import calibrate_severity as cal  # noqa: E402
import eval_real_bdd as erb  # noqa: E402
import eval_restorer_map as erm  # noqa: E402
import run_arms_v2 as ra  # noqa: E402
import shortcut_test as st  # noqa: E402
import train_gate_cnn  # noqa: E402

B, SEED = 1000, 42
OUT = REPO_ROOT / "results" / "bootstrap_ci.json"


def detect_all(model, paths, device):
    out = []
    for k in range(0, len(paths), 32):
        for r in model.predict([cv2.imread(str(p)) for p in paths[k:k + 32]], conf=0.001,
                               max_det=300, device=device, verbose=False):
            b = r.boxes
            out.append(np.concatenate([b.cls.cpu().numpy()[:, None], b.xyxy.cpu().numpy(),
                                       b.conf.cpu().numpy()[:, None]], 1) if len(b)
                       else np.zeros((0, 6), np.float32))
    return out


def match_flags(pred, gt):
    """Per detection: is it a TP (IoU>=0.5, class match, greedy by confidence)."""
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(pred), bool)
    for j in (np.argsort(-pred[:, 5]) if len(pred) else []):
        cand = np.where((gt[:, 0] == pred[j, 0]) & ~used)[0]
        if len(cand):
            ov = cal.iou(pred[j, 1:5], gt[cand, 1:5])
            k = int(np.argmax(ov))
            if ov[k] >= cal.IOU_MATCH:
                used[cand[k]] = True
                tp[j] = True
    return tp


class APTable:
    """Detections of a set of images with fixed TP flags; AP under per-image weights."""

    def __init__(self, preds, gts, classes):
        self.classes = [c for c in classes if sum(int((g[:, 0] == c).sum()) for g in gts) > 0]
        self.n_img = len(preds)
        self.gt_count = np.array([[int((g[:, 0] == c).sum()) for c in self.classes] for g in gts])
        self.per_class = {}
        for ci, c in enumerate(self.classes):
            conf, tp, img = [], [], []
            for i, (p, g) in enumerate(zip(preds, gts)):
                m = p[:, 0] == c
                if m.any():
                    flags = match_flags(p[m], g)
                    conf.append(p[m, 5]); tp.append(flags); img.append(np.full(int(m.sum()), i))
            if conf:
                conf, tp, img = np.concatenate(conf), np.concatenate(tp), np.concatenate(img)
                o = np.argsort(-conf, kind="mergesort")
                self.per_class[ci] = (tp[o].astype(float), img[o])
            else:
                self.per_class[ci] = (np.zeros(0), np.zeros(0, int))

    def mAP(self, w=None):
        w = np.ones(self.n_img) if w is None else w
        aps = []
        for ci in range(len(self.classes)):
            n_gt = float((self.gt_count[:, ci] * w).sum())
            if n_gt <= 0:
                continue
            tp, img = self.per_class[ci]
            if not len(tp):
                aps.append(0.0); continue
            wt = w[img]
            ctp, cfp = np.cumsum(tp * wt), np.cumsum((1 - tp) * wt)
            rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
            mrec = np.concatenate([[0.0], rec, [1.0]]); mpre = np.concatenate([[1.0], prec, [0.0]])
            mpre = np.maximum.accumulate(mpre[::-1])[::-1]
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
        return float(np.mean(aps)) if aps else float("nan")


def ci(vals):
    v = np.asarray([x for x in vals if x == x])
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def gate_prf(fired, truth, keep):
    tp = int(((fired & truth) * keep).sum()); fp = int(((fired & ~truth) * keep).sum())
    fn = int(((~fired & truth) * keep).sum()); tn = int(((~fired & ~truth) * keep).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), (tp + tn) / max(tp + fp + fn + tn, 1)


def main() -> int:
    from ultralytics import YOLO
    from train_baseline import pick_device
    device = pick_device(None)
    det = YOLO(str(erm.DEFAULT_WEIGHTS))
    rng = np.random.default_rng(SEED)
    out = {"protocol": __doc__.split("PROTOCOL.")[1].split("KITTI:")[0].strip(), "B": B, "seed": SEED}

    # ---------------- KITTI ----------------
    ra.configure("daynet")
    params = {(r["split"], r["image_id"]): r for r in csv.DictReader(erm.PARAMS.open())}
    dirs = erm.v2_val_dirs(erm.V2_ROOT)
    ids, _ = erm.eligible(sorted(p.stem for p in dirs["glared"].glob("*.jpg")),
                          erm.excluded_ids(erm.CALIBRATION, erm.TUNING))
    n = len(ids)
    lab = REPO_ROOT / "datasets" / "kitti" / "labels" / "val"
    gts = []
    for i in ids:
        h, w = cv2.imread(str(dirs["clean"] / f"{i}.jpg")).shape[:2]
        gts.append(cal.read_labels(lab / f"{i}.txt", w, h))
    print("detecting KITTI versions...", flush=True)
    P = {"clean": detect_all(det, [dirs["clean"] / f"{i}.jpg" for i in ids], device),
         "glared": detect_all(det, [dirs["glared"] / f"{i}.jpg" for i in ids], device),
         "clean_r": detect_all(det, [ra.CLEAN_CACHE / f"{i}.png" for i in ids], device),
         "glared_r": detect_all(det, [ra.GLARED_CACHE / f"{i}.png" for i in ids], device)}

    # gate decisions (same models and thresholds as eval_arms_cnn_latency.py)
    t_bright, lr, _ = ra.train_gates(params)
    gate = torch.load(REPO_ROOT / "models" / "gate_cnn_mixed.pt", weights_only=False, map_location="cpu").eval()
    stream = [(i, v) for i in ids for v in ("clean", "glared")]
    feats = dict(zip(stream, st.extract_all([erm.V2_ROOT / v / "images" / "val" / f"{i}.jpg" for i, v in stream], ra.WORKERS)))
    lrp = dict(zip(stream, lr.score(st.matrix([feats[k] for k in stream], ["gate"]))))
    pc = {}
    for k in range(0, len(stream), 64):
        b = stream[k:k + 64]
        x = torch.from_numpy(np.stack([train_gate_cnn.to_input(cv2.imread(str(erm.V2_ROOT / v / "images" / "val" / f"{i}.jpg"))) for i, v in b])).permute(0, 3, 1, 2)
        with torch.no_grad():
            pc.update(zip(b, torch.sigmoid(gate(x)).numpy().tolist()))
    fire = {"C3_cnn": {k: pc[k] > 0.5 for k in stream},
            "C3_features": {k: bool(lrp[k] >= 0.5) for k in stream},
            "C3_brightness": {k: feats[k]["upper_third_mean_luma"] > t_bright for k in stream}}

    # per-arm stream tables: images ordered [clean_0..n-1, glared_0..n-1]
    def arm_preds(arm):
        pcl, pgl = [], []
        for j, i in enumerate(ids):
            fc = arm == "C2" or (arm.startswith("C3") and fire[arm][(i, "clean")])
            fg = arm == "C2" or (arm.startswith("C3") and fire[arm][(i, "glared")])
            pcl.append(P["clean_r"][j] if fc else P["clean"][j])
            pgl.append(P["glared_r"][j] if fg else P["glared"][j])
        return pcl + pgl
    classes = list(range(8))
    tables = {arm: APTable(arm_preds(arm), gts + gts, classes)
              for arm in ("C1", "C2", "C3_cnn", "C3_features", "C3_brightness")}
    t_clean = APTable(P["clean"], gts, classes)
    t_glared = APTable(P["glared"], gts, classes)
    t_rest = APTable(P["glared_r"], gts, classes)

    excl = {i for i in ids if params[("val", i)]["gate_excluded"] == "1"}
    truth = np.array([v == "glared" for i, v in stream]); keep = np.array([not (v == "glared" and i in excl) for i, v in stream])
    firearr = {a: np.array([fire[a][k] for k in stream]) for a in fire}

    def kitti_stats(wp):
        ws = np.concatenate([wp, wp])
        m = {a: t.mAP(ws) for a, t in tables.items()}
        mc, mg, mr = t_clean.mAP(wp), t_glared.mAP(wp), t_rest.mAP(wp)
        s = {f"stream_mAP50_{a}": v for a, v in m.items()}
        s.update(clean=mc, glared=mg, restored=mr, glare_drop=mc - mg,
                 recovery=(mr - mg) / (mc - mg) if mc > mg else float("nan"))
        for a in fire:
            s[f"gain_kept_{a}"] = (m[a] - m["C1"]) / (m["C2"] - m["C1"]) if m["C2"] > m["C1"] else float("nan")
            wk = np.concatenate([wp, wp]) * keep
            p, r, f, acc = gate_prf(firearr[a], truth, wk)
            s.update({f"gate_{a}_precision": p, f"gate_{a}_recall": r, f"gate_{a}_f1": f, f"gate_{a}_accuracy": acc})
        return s

    point = kitti_stats(np.ones(n))
    print("KITTI point estimates:", {k: round(v, 4) for k, v in point.items()}, flush=True)
    boot = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot.append(kitti_stats(np.bincount(idx, minlength=n).astype(float)))
        if (b + 1) % 200 == 0:
            print(f"  KITTI bootstrap {b + 1}/{B}", flush=True)
    out["kitti"] = {"n_pairs": n, "metrics": {k: {"point": round(point[k], 4), "ci95": ci([s[k] for s in boot])} for k in point}}

    # ---------------- BDD100K real glare ----------------
    import restore_daytime_net
    rest = restore_daytime_net.DaytimeNetRestorer(device=device)
    rows = list(csv.DictReader(bp.MANIFEST.open()))
    sets = {}
    for kind in ("glare", "clean"):
        rs = [r for r in rows if (r["is_glared"] == "1") == (kind == "glare")]
        imgs = [cv2.imread(str(bp.OUT / kind / "images" / "val" / r["name"])) for r in rs]
        g = [erb.read_yolo(bp.OUT / kind / "labels" / "val" / f"{Path(r['name']).stem}.txt", im.shape[1], im.shape[0]) for r, im in zip(rs, imgs)]
        x = torch.from_numpy(np.stack([train_gate_cnn.to_input(im) for im in imgs])).permute(0, 3, 1, 2)
        with torch.no_grad():
            fired = (torch.sigmoid(gate(x)).numpy() > 0.5)
        raw = [erb.remap(p) for p in detect_det(det, imgs, device)]
        restd = [erb.remap(p) for p in detect_det(det, [rest.restore(im) for im in imgs], device)]
        sets[kind] = {"gt": g, "C1": raw, "C2": restd, "C3": [rd if f else rw for rw, rd, f in zip(raw, restd, fired)], "fired": fired}
    car = [0]
    bt = {kind: {arm: APTable(sets[kind][arm], sets[kind]["gt"], car) for arm in ("C1", "C2", "C3")} for kind in sets}
    ng, nc = len(sets["glare"]["gt"]), len(sets["clean"]["gt"])

    def bdd_stats(wg, wc):
        c1c = bt["clean"]["C1"].mAP(wc); s = {"car_AP_clean_C1": c1c}
        g1 = bt["glare"]["C1"].mAP(wg)
        for arm in ("C1", "C2", "C3"):
            ga = bt["glare"][arm].mAP(wg); s[f"car_AP_glare_{arm}"] = ga
            if arm != "C1":
                s[f"car_recovery_{arm}"] = (ga - g1) / (c1c - g1) if c1c > g1 else float("nan")
        s["car_glare_drop"] = c1c - g1
        return s

    pb = bdd_stats(np.ones(ng), np.ones(nc))
    bb = [bdd_stats(np.bincount(rng.integers(0, ng, ng), minlength=ng).astype(float),
                    np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float)) for _ in range(B)]
    out["bdd100k"] = {"n_glare": ng, "n_clean": nc, "metrics": {k: {"point": round(pb[k], 4), "ci95": ci([s[k] for s in bb])} for k in pb}}
    # gate precision on hard real frames: 68 of 204 flagged reviewed candidates (Wilson interval)
    k_, n_ = 68, 204
    z = 1.96; ph = k_ / n_; den = 1 + z * z / n_
    c0 = (ph + z * z / (2 * n_)) / den; half = z * np.sqrt(ph * (1 - ph) / n_ + z * z / (4 * n_ * n_)) / den
    out["bdd100k"]["gate_precision_hard_frames"] = {"point": round(ph, 4), "ci95_wilson": [round(c0 - half, 4), round(c0 + half, 4)], "k": k_, "n": n_}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


def detect_det(det, imgs, device):
    out = []
    for k in range(0, len(imgs), 32):
        for r in det.predict(imgs[k:k + 32], conf=0.001, max_det=300, device=device, verbose=False):
            b = r.boxes
            out.append(np.concatenate([b.cls.cpu().numpy()[:, None], b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]], 1)
                       if len(b) else np.zeros((0, 6), np.float32))
    return out


if __name__ == "__main__":
    sys.exit(main())
