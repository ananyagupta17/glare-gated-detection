"""Build the demo folder: example images of each pipeline stage, charts, and REPORT.md.

Runs after src/run_arms_v2.py. Reads its results plus the restorer and analysis JSONs,
re-fits the two C3 gates exactly as run_arms_v2.py does (and refuses to draw gate
examples unless the re-fit reproduces the recorded gate metrics), runs the detector on
the selected frames, and writes everything to demo/.

FRAME SELECTION (stated in the report, so nothing is cherry-picked silently)
  * per glare type: the severe val frame whose detection gain from restoration is the
    MEDIAN for that type (a typical case, not the best)
  * pipeline walk-through: the median-gain severe frame over all types
  * failure case: the severe frame with the lowest F1 after restoration
  * gate examples: for each outcome (fired on glare, skipped clean, false alarm, miss),
    the frame whose gate score is the median within that outcome

DEMO STATUS: v2 failed its shortcut test; every page says so.

    python src/make_demo.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import calibrate_severity as cal  # noqa: E402
import eval_restorer_map as erm  # noqa: E402
import run_arms_v2 as ra  # noqa: E402
import shortcut_test as st  # noqa: E402

OUT = REPO_ROOT / "demo"
R = REPO_ROOT / "results"
TYPES = ("diffuse_haze", "direct_sun", "veiling_wash", "horizon_streak")
NAMES = {0: "car", 1: "van", 2: "truck", 3: "pedestrian", 4: "Person_sitting",
         5: "cyclist", 6: "tram", 7: "misc"}

# chart tokens (dataviz reference palette, light mode; validated: 3 slots pass,
# aqua below 3:1 so every bar carries a visible value label)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_CLEAN, C_GLARED, C_RESTORED = "#2a78d6", "#eb6834", "#1baf7a"
# box colours on images (not a data palette): matched, false alarm, missed
BOX_TP, BOX_FP, BOX_MISS = (60, 200, 60), (40, 40, 230), (0, 215, 255)   # BGR

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 10})


# --- detection helpers -----------------------------------------------------------------

def predict(model, images: list, device: str) -> list[np.ndarray]:
    out = []
    for i in range(0, len(images), 32):
        for r in model.predict(images[i:i + 32], conf=cal.CONF, device=device, verbose=False):
            b = r.boxes
            out.append(np.concatenate([b.cls.cpu().numpy()[:, None], b.xyxy.cpu().numpy(),
                                       b.conf.cpu().numpy()[:, None]], 1) if len(b)
                       else np.zeros((0, 6), np.float32))
    return out


def match(pred: np.ndarray, gt: np.ndarray):
    """Greedy class-aware IoU>=0.5 matching, as in calibrate_severity.frame_f1."""
    used = np.zeros(len(gt), bool)
    tp_pred = np.zeros(len(pred), bool)
    for j in np.argsort(-pred[:, 5]) if len(pred) else []:
        p = pred[j]
        cand = np.where((gt[:, 0] == p[0]) & ~used)[0]
        if len(cand):
            ov = cal.iou(p[1:5], gt[cand, 1:5])
            k = int(np.argmax(ov))
            if ov[k] >= cal.IOU_MATCH:
                used[cand[k]] = True
                tp_pred[j] = True
    return tp_pred, used


def gt_for(image_id: str, shape) -> np.ndarray:
    h, w = shape[:2]
    return cal.read_labels(REPO_ROOT / "datasets" / "kitti" / "labels" / "val" / f"{image_id}.txt", w, h)


def draw(img: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, str]:
    img = img.copy()
    tp_pred, gt_hit = match(pred, gt)
    for g, hit in zip(gt, gt_hit):
        if not hit:
            x1, y1, x2, y2 = map(int, g[1:5])
            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_MISS, 2)
    for p, ok in zip(pred, tp_pred):
        x1, y1, x2, y2 = map(int, p[1:5])
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_TP if ok else BOX_FP, 2)
    tp, fp, miss = int(tp_pred.sum()), int((~tp_pred).sum()), int((~gt_hit).sum())
    f1 = 2 * tp / max(2 * tp + fp + miss, 1)
    return img[:, :, ::-1], f"F1 {f1:.2f} · {tp} found, {fp} false, {miss} missed (of {len(gt)})"


def panel_figure(rows: list[tuple[np.ndarray, str]], title: str, path: Path, note: str = "") -> None:
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 3.7 * len(rows) + 0.6))
    axes = np.atleast_1d(axes)
    for ax, (img, caption) in zip(axes, rows):
        ax.imshow(img)
        ax.set_title(caption, fontsize=10, loc="left", color=INK)
        ax.axis("off")
    fig.suptitle(title, fontsize=13, x=0.01, ha="left", color=INK)
    footer = ("Boxes: green = detected correctly · red = false detection · yellow = missed object.  "
              "DEMO: v2 failed its shortcut test; not final numbers.")
    fig.text(0.01, 0.005, (note + "\n" if note else "") + footer, fontsize=8, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- charts ----------------------------------------------------------------------------

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, groups, series, values, colors):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for k, (name, color) in enumerate(zip(series, colors)):
        vals = [values[g][k] for g in range(len(groups))]
        pos = x - 0.4 + width * (k + 0.5)
        ax.bar(pos, vals, width * 0.92, color=color, label=name, edgecolor=SURFACE, linewidth=2)
        for px, v in zip(pos, vals):
            ax.text(px, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.0)
    style(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.13), ncol=n)


def charts(arms: dict, whole: dict) -> list[Path]:
    out = []
    labels = {"C1": "C1 never restore", "C2": "C2 always restore",
              "C3_brightness": "C3 gate: brightness", "C3_features": "C3 gate: features"}
    order = list(labels)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars(ax, [labels[a].replace(" ", "\n", 1) for a in order], ["Clean frames", "Glared frames"],
         [[arms[a]["clean"]["mAP50"], arms[a]["glared"]["mAP50"]] for a in order],
         [C_CLEAN, C_GLARED])
    ax.set_ylabel("mAP@50")
    ax.set_title("Detection accuracy per arm, on clean and on glared frames", loc="left", pad=34)
    fig.tight_layout()
    p = OUT / "07_chart_map_per_arm.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    xs = [arms[a]["latency_ms"]["total_mean"] for a in order]
    ys = [arms[a]["stream"]["mAP50"] for a in order]
    ax.scatter(xs, ys, s=70, color=C_CLEAN, edgecolor=SURFACE, linewidth=2, zorder=3)
    # per-point label offsets so neighbouring labels never collide
    place = {"C1": ((10, -4), "left"), "C2": ((10, 6), "left"),
             "C3_brightness": ((10, -14), "left"), "C3_features": ((-10, -30), "right")}
    for a, x, y in zip(order, xs, ys):
        (dx, dy), ha = place[a]
        ax.annotate(f"{labels[a]}\n{y:.3f} mAP · {x:,.0f} ms · restores {arms[a]['trigger_rate']:.0%}",
                    (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=8.5, color=INK,
                    ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency per frame, ms (CPU, log scale)")
    ax.set_ylabel("mAP@50, whole 50/50 stream")
    ax.set_xlim(min(xs) * 0.6, max(xs) * 6)
    style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_title("Accuracy vs cost: what gating buys", loc="left")
    fig.tight_layout()
    p = OUT / "08_chart_accuracy_vs_latency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    res = whole["results"]
    groups = [("All glared", "all"), ("Severe", "severe"), ("Non-severe", "non_severe")]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars(ax, [g for g, _ in groups], ["Clean", "Glared", "Restored"],
         [[res["clean"][k]["mAP50"], res["input"][k]["mAP50"], res["output"][k]["mAP50"]]
          for _, k in groups], [C_CLEAN, C_GLARED, C_RESTORED])
    ax.set_ylabel("mAP@50")
    ax.set_title("How much of the glare loss restoration recovers", loc="left", pad=34)
    fig.tight_layout()
    p = OUT / "09_chart_recovery_by_severity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


# --- main ------------------------------------------------------------------------------

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--restorer", choices=sorted(ra.RESTORER_DESCR), default="whole")
    name = ap.parse_args().restorer
    ra.configure(name)
    arms_doc = json.loads(ra.OUT_JSON.read_text())
    arms = arms_doc["arms"]
    whole = json.loads((R / f"restorer_map_{name}.json").read_text())
    analysis = json.loads((R / "restored_analysis.json").read_text()) if name == "whole" else None
    shortcut = json.loads((R / "shortcut_test.json").read_text())
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    dirs = erm.v2_val_dirs(erm.V2_ROOT)
    ids, _ = erm.eligible(sorted(p.stem for p in dirs["glared"].glob("*.jpg")),
                          erm.excluded_ids(erm.CALIBRATION, erm.TUNING))
    params = {(r["split"], r["image_id"]): r for r in csv.DictReader(erm.PARAMS.open())}
    path = {"clean": lambda i: dirs["clean"] / f"{i}.jpg",
            "glared": lambda i: dirs["glared"] / f"{i}.jpg",
            "restored": lambda i: ra.GLARED_CACHE / f"{i}.png",
            "clean_restored": lambda i: ra.CLEAN_CACHE / f"{i}.png"}

    # gates: re-fit exactly as run_arms_v2.py and check they reproduce its metrics
    print("re-fitting gates...")
    t_bright, lr, _ = ra.train_gates(params)
    stream = [(i, v) for i in ids for v in ("clean", "glared")]
    feats = dict(zip(stream, st.extract_all([path[v](i) for i, v in stream], ra.WORKERS)))
    score = dict(zip(stream, lr.score(st.matrix([feats[k] for k in stream], ["gate"]))))
    fired = {k: s >= 0.5 for k, s in score.items()}
    reproduced = ra.gate_metrics(fired, stream, params) == arms_doc["gates"]["metrics_val"]["C3_features"]
    print(f"gate re-fit reproduces run_arms_v2 metrics: {reproduced}")

    # per-frame F1 for glared and restored, to choose typical frames
    from train_baseline import pick_device
    from ultralytics import YOLO
    device = pick_device(None)
    model = YOLO(str(erm.DEFAULT_WEIGHTS))
    severe_ids = [i for i in ids if params[("val", i)]["severe"] == "1"]
    print(f"scoring {len(severe_ids)} severe frames, glared and restored...")
    f1 = {}
    for version in ("glared", "restored"):
        imgs = [cv2.imread(str(path[version](i))) for i in severe_ids]
        for i, img, pr in zip(severe_ids, imgs, predict(model, imgs, device)):
            f1[(i, version)] = cal.frame_f1(pr, gt_for(i, img.shape))
    gain = {i: f1[(i, "restored")] - f1[(i, "glared")] for i in severe_ids}

    def median_by(keys, value):
        keys = sorted(keys, key=lambda k: (value(k), k))
        return keys[len(keys) // 2] if keys else None

    def show(i, version):
        img = cv2.imread(str(path[version](i)))
        return draw(img, predict(model, [img], device)[0], gt_for(i, img.shape))

    def describe(i):
        r = params[("val", i)]
        return f"{i} · {r['glare_type']} · severity s={float(r['severity']):.2f}"

    def gate_line(i, version):
        k = (i, version)
        bright = feats[k]["upper_third_mean_luma"] > t_bright
        return (f"gate (features) score {score[k]:.2f} → {'RESTORE' if fired[k] else 'skip'}; "
                f"gate (brightness) → {'RESTORE' if bright else 'skip'}")

    selected = {}
    # 01 pipeline walk-through
    i = median_by(severe_ids, lambda k: gain[k])
    selected["pipeline"] = i
    rows = [show(i, "clean"), show(i, "glared"), show(i, "restored")]
    panel_figure([(rows[0][0], f"1. Clean camera frame — {rows[0][1]}"),
                  (rows[1][0], f"2. Same frame with synthetic sun glare ({describe(i)}) — {rows[1][1]}\n"
                               f"    {gate_line(i, 'glared')}"),
                  (rows[2][0], f"3. After restoration ({ra.RESTORER_DESCR[name]}) — {rows[2][1]}")],
                 "How the pipeline handles one glared frame", OUT / f"01_pipeline_walkthrough_{i}.png",
                 note=f"Selection: the severe val frame with the median detection gain from restoration.")

    # 02 one typical frame per glare type
    rows = []
    for t in TYPES:
        cand = [k for k in severe_ids if params[("val", k)]["glare_type"] == t]
        i = median_by(cand, lambda k: gain[k])
        selected[f"type_{t}"] = i
        g, r = show(i, "glared"), show(i, "restored")
        rows += [(g[0], f"{t} — glared ({describe(i)}): {g[1]}"),
                 (r[0], f"{t} — restored: {r[1]}")]
    panel_figure(rows, "The four glare types: glared vs restored (typical severe frame of each)",
                 OUT / "02_glare_types_glared_vs_restored.png",
                 note="Selection: per type, the severe val frame with the median detection gain.")

    # 03 severity ramps (generated by glare_synth.py)
    for t in TYPES:
        src = R / f"glare_v2_example_{t}.png"
        if src.is_file():
            shutil.copy2(src, OUT / f"03_severity_ramp_{t}.png")

    # 04 gate decisions
    if reproduced:
        outcomes = {
            "fired on glare (correct)": [k for k in stream if k[1] == "glared" and fired[k]],
            "skipped a clean frame (correct)": [k for k in stream if k[1] == "clean" and not fired[k]],
            "false alarm: restored a clean frame": [k for k in stream if k[1] == "clean" and fired[k]],
            "missed glare: did not restore": [k for k in stream if k[1] == "glared" and not fired[k]
                                              and params[("val", k[0])]["gate_excluded"] != "1"],
        }
        rows = []
        for outcome, keys in outcomes.items():
            k = median_by(keys, lambda kk: score[kk])
            if k is None:
                continue
            img = cv2.imread(str(path[k[1]](k[0])))[:, :, ::-1]
            extra = describe(k[0]) if k[1] == "glared" else f"{k[0]} · clean"
            rows.append((img, f"{outcome} — {extra} · features score {score[k]:.2f} "
                              f"(n = {len(keys)} in this outcome)"))
            selected[f"gate_{outcome}"] = f"{k[0]}_{k[1]}"
        panel_figure(rows, "C3 gate decisions (features gate), one typical frame per outcome",
                     OUT / "04_gate_decisions.png",
                     note="Selection: within each outcome, the frame with the median gate score.")
    else:
        print("WARNING: gate re-fit did not reproduce the recorded metrics; 04 skipped")

    # 05 restoring a clean frame
    i = selected["pipeline"]
    c, cr = show(i, "clean"), show(i, "clean_restored")
    panel_figure([(c[0], f"Clean frame {i} — {c[1]}"),
                  (cr[0], f"Same clean frame after restoration — {cr[1]}")],
                 "What restoration does to a frame with no glare (the cost C2 pays on every clean frame)",
                 OUT / "05_restoring_a_clean_frame.png",
                 note=f"Over all {len(ids):,} clean val frames, always restoring (C2 vs C1) changed "
                      f"mAP@50 by {clean_change(arms):+.3f}.")

    # 06 failure case
    i = min(severe_ids, key=lambda k: (f1[(k, 'restored')], gain[k], k))
    selected["failure"] = i
    rows = [show(i, "clean"), show(i, "glared"), show(i, "restored")]
    panel_figure([(rows[0][0], f"Clean — {rows[0][1]}"),
                  (rows[1][0], f"Glared ({describe(i)}) — {rows[1][1]}"),
                  (rows[2][0], f"Restored — {rows[2][1]}")],
                 "Limitation: where restoration cannot help", OUT / "06_failure_case.png",
                 note="Selection: the severe val frame with the lowest F1 after restoration. "
                      "Clipped (pure white) pixels hold no information to recover.")

    chart_paths = charts(arms, whole)
    write_report(arms_doc, whole, analysis, shortcut, selected, reproduced, gain, f1, severe_ids, name)
    (OUT / "selection.json").write_text(json.dumps(
        {"selected": selected, "gate_refit_reproduces_metrics": reproduced,
         "severe_frames_scored": len(severe_ids),
         "median_f1_gain_severe": round(float(np.median(list(gain.values()))), 4)}, indent=2) + "\n")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))
    return 0


def clean_change(arms: dict) -> float:
    """What always restoring costs on clean frames: C2 vs C1 mAP@50, all clean val frames."""
    return arms["C2"]["clean"]["mAP50"] - arms["C1"]["clean"]["mAP50"]


def comparison_rows() -> str:
    """One row per restorer that has been evaluated, from its own result files."""
    rows = []
    for key in ("whole", "daytime", "daynet"):
        suffix = "" if key == "whole" else f"_{key}"
        arms_p, map_p = R / f"arms_v2{suffix}.json", R / f"restorer_map_{key}.json"
        if not (arms_p.is_file() and map_p.is_file()):
            continue
        a = json.loads(arms_p.read_text())
        m = json.loads(map_p.read_text())["results"]
        rec = (m["output"]["all"]["mAP50"] - m["input"]["all"]["mAP50"]) / \
              (m["clean"]["all"]["mAP50"] - m["input"]["all"]["mAP50"])
        ar = a["arms"]
        rows.append(f"| {ra.RESTORER_DESCR[key]} | {a['restorer']['mean_ms_per_frame']:,.0f} | "
                    f"{m['output']['all']['mAP50']:.3f} | {rec:.0%} | {clean_change(ar):+.3f} | "
                    f"{ar['C2']['stream']['mAP50']:.3f} | {ar['C3_features']['stream']['mAP50']:.3f} | "
                    f"{ar['C3_features']['latency_ms']['total_mean']:,.0f} |")
    return "\n".join(rows)


def write_report(doc, whole, analysis, shortcut, selected, reproduced, gain, f1, severe_ids,
                 restorer) -> None:
    a, g = doc["arms"], doc["gates"]["metrics_val"]
    res = whole["results"]
    lat = {k: v["latency_ms"] for k, v in a.items()}
    name = {"C1": "C1 · never restore", "C2": "C2 · always restore",
            "C3_brightness": "C3 · gate: brightness", "C3_features": "C3 · gate: features"}
    arm_rows = "\n".join(
        f"| {name[k]} | {v['stream']['mAP50']:.3f} | {v['clean']['mAP50']:.3f} | "
        f"{v['glared']['mAP50']:.3f} | {v['glared_severe']['mAP50']:.3f} | "
        f"{v['trigger_rate']:.1%} | {lat[k]['total_mean']:,.0f} |" for k, v in a.items())
    lat_rows = "\n".join(
        f"| {name[k]} | {lat[k]['gate']:.2f} | {lat[k]['detect']:.1f} | "
        f"{lat[k]['restore_expected']:,.0f} | **{lat[k]['total_mean']:,.0f}** |" for k in a)
    sev = "\n".join(
        f"| {lbl} | {res['clean'][k]['count']} | {res['clean'][k]['mAP50']:.3f} | "
        f"{res['input'][k]['mAP50']:.3f} | {res['output'][k]['mAP50']:.3f} | "
        f"{(res['output'][k]['mAP50'] - res['input'][k]['mAP50']) / (res['clean'][k]['mAP50'] - res['input'][k]['mAP50']):.0%} |"
        for lbl, k in (("All", "all"), ("Severe", "severe"), ("Non-severe", "non_severe")))
    gate_rows = "\n".join(f"| {name[k]} | {v['precision']:.3f} | {v['recall']:.3f} | "
                          f"{v['f1']:.3f} | {v['accuracy']:.3f} | {a[k]['trigger_rate']:.1%} |"
                          for k, v in g.items())
    sep = analysis["separation"]["results"] if analysis else None
    cc = clean_change(a)
    descr = ra.RESTORER_DESCR[restorer]
    ch = shortcut["checks"]
    fails = [f"darkest-pixel p1 AUC {ch['darkest_pixel']['p1_luma']['value']:.3f}",
             f"min AUC {ch['darkest_pixel']['min_luma']['value']:.3f}",
             f"noise-level AUC {ch['single_feature']['noise_level']['value']:.3f}",
             f"artifact-only classifier {ch['brightness_removed_lr']['value']:.1%}"]
    best_c3 = max(("C3_brightness", "C3_features"), key=lambda k: a[k]["stream"]["mAP50"])
    c2_ms, c3_ms = lat["C2"]["total_mean"], lat[best_c3]["total_mean"]

    (OUT / "REPORT.md").write_text(f"""# Glare-gated detection: demo results

> **DEMO, not final.** These numbers come from v2 synthetic data that **failed its
> shortcut test** ({"; ".join(fails)}). The gate rows are the most affected: a gate can
> use those artifacts. Final numbers wait for a synthesis that passes.

## Setup
- **Data:** KITTI val, {doc['stream']['frames']:,} frames, each shown clean **and** glared
  ({doc['stream']['images']:,} images, exactly 50/50). Glare: v2 camera-model synthesis,
  four types, continuous severity. 5 frames used to tune the restorer are excluded.
- **Detector:** YOLOv8n trained on clean KITTI train only (`models/yolo_c1_train.pt`, 50 epochs).
- **Restorer:** {descr}. **{doc['restorer']['mean_ms_per_frame']:,.0f} ms/frame on CPU.**
- **Gates (C3),** fit on v2 *train* only: an upper-third brightness threshold, and a
  logistic regression on Harshitha's 9 gate features (standing in for her v2 gate).

## Headline
| Arm | mAP@50 stream | clean frames | glared frames | glared, severe | Restores | Latency ms/frame (CPU) |
|---|---|---|---|---|---|---|
{arm_rows}

- **Glare costs the detector {res['clean']['all']['mAP50'] - res['input']['all']['mAP50']:.3f} mAP@50**
  ({res['clean']['all']['mAP50']:.3f} → {res['input']['all']['mAP50']:.3f}); on severe frames
  {res['clean']['severe']['mAP50']:.3f} → {res['input']['severe']['mAP50']:.3f}.
- **Gating:** the best C3 ({name[best_c3]}) runs at {c3_ms:,.0f} ms/frame against C2's
  {c2_ms:,.0f} ms/frame ({c2_ms / c3_ms:.1f}× cheaper), restoring {a[best_c3]['trigger_rate']:.0%}
  of frames.

![accuracy vs latency](08_chart_accuracy_vs_latency.png)
![mAP per arm](07_chart_map_per_arm.png)

## Restoration recovers most of the glare loss
| Glared frames | n | Clean | Glared | Restored | Share of loss recovered |
|---|---|---|---|---|---|
{sev}

**Restoring frames without glare:** over all {doc['stream']['frames']:,} clean val frames, always
restoring changed mAP@50 from {a['C1']['clean']['mAP50']:.3f} to {a['C2']['clean']['mAP50']:.3f}
({cc:+.3f}). {"That cost is real, so the gate protects accuracy on clean frames as well as saving time." if cc < -0.005 else "That is close to free, so here the gate mainly saves time."}

## Restorers compared (same frames, same detector, same gates)
| Restorer | ms/frame (CPU) | Glared mAP@50 restored | Share of loss recovered | Clean-frame change (C2−C1) | C2 stream mAP@50 | C3 features stream mAP@50 | C3 features ms/frame |
|---|---|---|---|---|---|---|---|
{comparison_rows()}

![recovery by severity](09_chart_recovery_by_severity.png)

## Latency breakdown (mean ms per frame, CPU, warm-up excluded)
| Arm | Gate | Detection | Restoration × trigger rate | Total |
|---|---|---|---|---|
{lat_rows}

Restoration is ~{doc['restorer']['mean_ms_per_frame'] / lat['C1']['detect']:.0f}× the cost of detection,
which is what makes skipping it worthwhile.

## Gate quality (val, gate protocol: glared veiling_wash frames with s < 0.1752 not scored)
| Gate | Precision | Recall | F1 | Accuracy | Trigger rate |
|---|---|---|---|---|---|
{gate_rows}

The features gate is likely inflated by the synthesis artifacts.{(" After Flare7K++ restoration the same features separate clean from glared far less well: accuracy " + format(sep['clean_vs_glared']['accuracy'], '.1%') + " on originals vs " + format(sep['clean_vs_glared_restored']['accuracy'], '.1%') + " clean-vs-restored (200-frame subset).") if sep else ""}

## Figures in this folder
| File | Shows |
|---|---|
| `01_pipeline_walkthrough_{selected['pipeline']}.png` | One frame through the pipeline: clean → glared (with both gate decisions) → restored, with detections |
| `02_glare_types_glared_vs_restored.png` | A typical severe frame of each glare type, before and after restoration |
| `03_severity_ramp_*.png` | Each glare type at lowest, median and highest severity |
| `04_gate_decisions.png` | The features gate: a correct fire, a correct skip, a false alarm and a miss |
| `05_restoring_a_clean_frame.png` | The cost side: restoration applied to a frame with no glare |
| `06_failure_case.png` | Where restoration cannot help (blown-out glare) |
| `07–09_chart_*.png` | The charts above |

**How frames were chosen (no cherry-picking):** per glare type, the severe frame with the
*median* detection gain from restoration; the walkthrough is the median over all types;
the failure case is the lowest F1 after restoration; gate examples are the median gate
score within each outcome. Over the {len(severe_ids)} severe frames, the median per-frame F1
gain from restoration was {float(np.median(list(gain.values()))):+.3f}.
Box colours: green = correct detection, red = false detection, yellow = missed object.

## Caveats
1. **Synthetic glare that failed its shortcut test.** Restoration may suit this additive,
   smooth glare better than real glare; confirm on real glare (e.g. BDD100K) before quoting.
2. **The C3 gates are stand-ins,** fit here on v2 train; Harshitha's v2 gate is not trained yet.
3. {"**Flare7K++ is a nighttime lens-flare model** applied to daytime sun glare." if restorer == "whole" else "**The daytime restorer was tuned or trained on our synthetic glare only** (train split); real-glare performance is untested."}
4. KITTI's train/val split shares drives, so absolute mAP is slightly optimistic; the
   comparisons between arms use identical frames and are unaffected.
""")


if __name__ == "__main__":
    sys.exit(main())
