"""Live demo: glare-gated object detection, in the browser.

    python src/demo_app.py            # then open http://127.0.0.1:7860

The final pipeline only: a glare gate decides per frame whether to run the daytime
restorer, then YOLOv8n detects. The page plays KITTI driving frames with sun glare
(by default a 50/50 mix of clean and glared frames) and shows, live:

  * the same frame without and with the pipeline, with detections
  * the path the frame took (restored or bypassed) and the gate's confidence
  * speed and per-stage latency, how often restoration runs, the gate's hits and false
    triggers, and detection F1 without vs with the pipeline

Components: the CNN glare gate (MobileNetV3-Small trained on synthetic + real glare by
Harshitha, src/train_gate_cnn.py, models/gate_cnn_mixed.pt), the trained daytime
restorer (src/restore_daytime_net.py,
models/restore_daytime_net.pt) and the clean-trained YOLOv8n (models/yolo_c1_train.pt).
Glared frames use the v2 synthetic glare; ground-truth boxes are the KITTI labels.
Timing covers gate + restoration + detection; disk reads and drawing are excluded.
Gate counters follow results/gate_protocol_v2.md (glared veiling_wash frames below the
calibrated severity are not counted as glare).
"""

from __future__ import annotations

import csv
import platform
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import calibrate_severity as cal  # noqa: E402
import eval_restorer_map as erm  # noqa: E402
import train_gate_cnn  # noqa: E402  (defines GateCNN, needed to unpickle the gate)

GATE_WEIGHTS = REPO_ROOT / "models" / "gate_cnn_mixed.pt"
PARAMS = {r["image_id"]: r for r in csv.DictReader(erm.PARAMS.open()) if r["split"] == "val"}
LABELS = REPO_ROOT / "datasets" / "kitti" / "labels" / "val"
BOX_TP, BOX_FP, BOX_MISS = (60, 200, 60), (40, 40, 230), (0, 215, 255)       # BGR
STREAMS = ["Mixed: clean + glare", "Glare only", "Clean only"]
SEED = 42


def chip_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True, timeout=2).stdout.strip() or platform.processor()
    except Exception:
        return platform.processor() or "this machine"


# --- pipeline --------------------------------------------------------------------------

class Pipeline:
    """CNN glare gate -> trained daytime restorer -> YOLOv8n."""

    def __init__(self):
        import torch
        from ultralytics import YOLO
        import restore_daytime_net
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.detector = YOLO(str(erm.DEFAULT_WEIGHTS))
        self.restorer = restore_daytime_net.DaytimeNetRestorer(device=self.device)
        self.torch = torch
        # Harshitha's gate is a pickled module; its file references only torch/torchvision
        # classes and train_gate_cnn.GateCNN (checked), so loading it runs no foreign code.
        self.gate_model = torch.load(GATE_WEIGHTS, weights_only=False,
                                     map_location=self.device).eval()
        warm = cv2.imread(str(erm.V2_ROOT / "glared" / "images" / "val" / "000000.jpg"))
        for _ in range(5):
            self.gate(warm)
            self.detect(self.restore(warm))

    def gate(self, img: np.ndarray) -> tuple[bool, float]:
        """Same decision as GateCNN.predict (p > 0.5), with the input on the gate's device."""
        x = self.torch.from_numpy(train_gate_cnn.to_input(img)).permute(2, 0, 1).unsqueeze(0)
        with self.torch.no_grad():
            p = float(self.torch.sigmoid(self.gate_model(x.to(self.device)))[0])
        return p > 0.5, p

    def restore(self, img: np.ndarray) -> np.ndarray:
        return self.restorer.restore(img)

    def detect(self, img: np.ndarray) -> np.ndarray:
        r = self.detector.predict(img, conf=cal.CONF, device=self.device, verbose=False)[0]
        b = r.boxes
        if not len(b):
            return np.zeros((0, 6), np.float32)
        return np.concatenate([b.cls.cpu().numpy()[:, None], b.xyxy.cpu().numpy(),
                               b.conf.cpu().numpy()[:, None]], 1)


def draw(img: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, dict]:
    """Boxes only (no text on the image); returns the RGB image and the match counts."""
    out = img.copy()
    used = np.zeros(len(gt), bool)
    tp_pred = np.zeros(len(pred), bool)
    for j in (np.argsort(-pred[:, 5]) if len(pred) else []):
        cand = np.where((gt[:, 0] == pred[j, 0]) & ~used)[0]
        if len(cand):
            ov = cal.iou(pred[j, 1:5], gt[cand, 1:5])
            k = int(np.argmax(ov))
            if ov[k] >= cal.IOU_MATCH:
                used[cand[k]] = True
                tp_pred[j] = True
    for g, hit in zip(gt, used):
        if not hit:
            cv2.rectangle(out, tuple(map(int, g[1:3])), tuple(map(int, g[3:5])), BOX_MISS, 2)
    for p, ok in zip(pred, tp_pred):
        cv2.rectangle(out, tuple(map(int, p[1:3])), tuple(map(int, p[3:5])), BOX_TP if ok else BOX_FP, 2)
    tp, fp, miss = int(tp_pred.sum()), int((~tp_pred).sum()), int((~used).sum())
    return out[:, :, ::-1], {"tp": tp, "fp": fp, "miss": miss, "n": len(gt),
                             "f1": 2 * tp / max(2 * tp + fp + miss, 1)}


def stream_frames(which: str) -> list[tuple[str, str]]:
    dirs = erm.v2_val_dirs(erm.V2_ROOT)
    ids, _ = erm.eligible(sorted(p.stem for p in dirs["glared"].glob("*.jpg")),
                          erm.excluded_ids(erm.CALIBRATION, erm.TUNING))
    rng = random.Random(SEED)
    if which == "Glare only":
        frames = [(i, "glared") for i in ids]
    elif which == "Clean only":
        frames = [(i, "clean") for i in ids]
    else:
        frames = [(i, rng.choice(("clean", "glared"))) for i in ids]
    rng.shuffle(frames)
    return frames


# --- page pieces (HTML; colours from the Gradio theme so light and dark both work) ----

CSS = """
footer {display: none !important;}
.gradio-container {max-width: 1440px !important; margin: 0 auto;}
#title h1 {font-size: 1.9rem; margin-bottom: 0.1rem;}
#title p {color: var(--body-text-color-subdued); margin-top: 0;}
.cap {font-size: 0.95rem; padding: 6px 2px 0; color: var(--body-text-color);}
.cap b {font-weight: 600;}
.flow {display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 4px 0 2px;}
.node {padding: 8px 14px; border-radius: 8px; border: 1px solid var(--border-color-primary);
       background: var(--block-background-fill); font-weight: 600; font-size: 0.92rem;}
.node.on {border-color: var(--color-accent); box-shadow: 0 0 0 2px var(--color-accent) inset;}
.node.off {opacity: 0.38;}
.node small {display: block; font-weight: 400; color: var(--body-text-color-subdued);}
.arrow {color: var(--body-text-color-subdued); font-size: 1.2rem;}
.cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px;}
.card {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 14px 16px;
       background: var(--block-background-fill);}
.card .k {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;
          color: var(--body-text-color-subdued);}
.card .v {font-size: 1.9rem; font-weight: 700; line-height: 1.2; margin-top: 4px;
          font-variant-numeric: tabular-nums;}
.card .s {font-size: 0.86rem; color: var(--body-text-color-subdued); margin-top: 2px;
          font-variant-numeric: tabular-nums;}
.legend {font-size: 0.88rem; color: var(--body-text-color-subdued);}
.sw {display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin: 0 4px 0 10px;
     vertical-align: -1px;}
"""


def flow_html(fired: bool | None, p: float | None, info: str) -> str:
    def node(title, sub, state):
        return f'<div class="node {state}">{title}<small>{sub}</small></div>'
    restore_state = "" if fired is None else ("on" if fired else "off")
    bypass_state = "" if fired is None else ("off" if fired else "on")
    gate_sub = "glare?" if p is None else f"p(glare) = {p:.2f}"
    return (f'<div class="flow">{node("Camera frame", info or "KITTI driving scene", "")}'
            f'<span class="arrow">→</span>{node("Glare gate", gate_sub, "on" if p is not None else "")}'
            f'<span class="arrow">→</span>{node("Daytime restoration", "runs only on glare", restore_state)}'
            f'<span class="arrow">/</span>{node("Bypass", "no glare: skip", bypass_state)}'
            f'<span class="arrow">→</span>{node("YOLOv8n detector", "cars, pedestrians, …", "on" if p is not None else "")}'
            f'</div>')


def caption(title: str, m: dict | None) -> str:
    if not m:
        return f'<div class="cap"><b>{title}</b></div>'
    return (f'<div class="cap"><b>{title}</b> — F1 {m["f1"]:.2f} · {m["tp"]} of {m["n"]} objects found'
            f' · {m["fp"]} false · {m["miss"]} missed</div>')


def cards_html(s: dict | None, chip: str) -> str:
    def card(k, v, sub):
        return f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{sub}</div></div>'
    if not s or not s["n"]:
        dash = "—"
        return ('<div class="cards">' + card("Speed", dash, "frames per second") +
                card("Latency", dash, "per frame") + card("Restoration runs on", dash, "of frames") +
                card("Gate accuracy", dash, "glare caught · false triggers") +
                card("Detection F1 · glare frames", dash, "without → with pipeline") +
                card("Detection F1 · clean frames", dash, "without → with pipeline") + "</div>")
    mean = {k: (float(np.mean(v)) if v else 0.0) for k, v in s["lat"].items()}
    fps = 1000.0 / mean["total"] if mean["total"] else 0.0
    pos, neg = s["tp"] + s["fn"], s["fp"] + s["tn"]
    acc = (s["tp"] + s["tn"]) / (pos + neg) if pos + neg else 0.0
    f = lambda xs: f"{np.mean(xs):.2f}" if xs else "—"  # noqa: E731
    return ('<div class="cards">'
            + card("Speed", f"{fps:.1f} fps", f"{mean['total']:.0f} ms per frame · {chip}")
            + card("Latency per frame", f"{mean['total']:.0f} ms",
                   f"gate {mean['gate']:.1f} · restoration {mean['restore']:.1f} · detection {mean['detect']:.1f}")
            + card("Restoration runs on", f"{s['restored'] / s['n']:.0%}",
                   f"of {s['n']} frames · skipped on the rest")
            + card("Gate accuracy", f"{acc:.0%}",
                   f"glare caught {s['tp']}/{pos} · false triggers {s['fp']}/{neg}")
            + card("Detection F1 · glare frames", f"{f(s['f1']['plain']['glared'])} → {f(s['f1']['pipe']['glared'])}",
                   "without → with pipeline")
            + card("Detection F1 · clean frames", f"{f(s['f1']['plain']['clean'])} → {f(s['f1']['pipe']['clean'])}",
                   "without → with pipeline")
            + "</div>")


# --- app -------------------------------------------------------------------------------

def build_ui(pipe: Pipeline):
    import gradio as gr
    chip = f"{chip_name()} ({pipe.device.upper()})"

    def run(stream, fps_cap):
        s = {"n": 0, "restored": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
             "f1": {"plain": {"clean": [], "glared": []}, "pipe": {"clean": [], "glared": []}},
             "lat": {k: deque(maxlen=30) for k in ("gate", "restore", "detect", "total")}}
        for image_id, version in stream_frames(stream):
            t_frame = time.perf_counter()
            img = cv2.imread(str(erm.V2_ROOT / version / "images" / "val" / f"{image_id}.jpg"))
            gt = cal.read_labels(LABELS / f"{image_id}.txt", img.shape[1], img.shape[0])

            t0 = time.perf_counter()                               # ---- timed ----
            fired, p = pipe.gate(img)
            t1 = time.perf_counter()
            x = pipe.restore(img) if fired else img
            t2 = time.perf_counter()
            pred = pipe.detect(x)
            t3 = time.perf_counter()                               # ---- end ----

            plain = pred if x is img else pipe.detect(img)         # for comparison only
            left, m_plain = draw(img, plain, gt)
            right, m_pipe = draw(x, pred, gt)

            r = PARAMS[image_id]
            glare = version == "glared"
            truth = (f"glare · {r['glare_type'].replace('_', ' ')} · severity {float(r['severity']):.2f}"
                     if glare else "clean")
            s["n"] += 1
            s["restored"] += bool(fired)
            if not (glare and r["gate_excluded"] == "1"):
                key = ("tp" if fired else "fn") if glare else ("fp" if fired else "tn")
                s[key] += 1
            s["f1"]["plain"][version].append(m_plain["f1"])
            s["f1"]["pipe"][version].append(m_pipe["f1"])
            for k, v in (("gate", t1 - t0), ("restore", t2 - t1), ("detect", t3 - t2), ("total", t3 - t0)):
                s["lat"][k].append(1000 * v)

            yield (left, right, caption("Without our pipeline", m_plain),
                   caption("With our pipeline · glare removed" if fired else
                           "With our pipeline · no glare detected, frame passed through unchanged", m_pipe),
                   flow_html(fired, p, f"frame {image_id} · {truth}"), cards_html(s, chip))
            if fps_cap:
                wait = 1.0 / fps_cap - (time.perf_counter() - t_frame)
                if wait > 0:
                    time.sleep(wait)

    theme = gr.themes.Base(primary_hue="orange", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    with gr.Blocks(title="Glare-Gated Object Detection", theme=theme, css=CSS) as ui:
        gr.Markdown("# Glare-Gated Object Detection\n"
                    "Live pipeline on KITTI driving frames with sun glare: a glare gate decides "
                    "whether each frame needs restoration, the daytime restorer runs only when it "
                    "does, and YOLOv8n detects objects.", elem_id="title")
        flow = gr.HTML(flow_html(None, None, ""))
        with gr.Row(equal_height=True):
            stream = gr.Radio(STREAMS, value=STREAMS[0], label="Input", scale=3)
            fps_cap = gr.Slider(0, 30, value=0, step=1, scale=2,
                                label="Playback speed limit (fps, 0 = real speed)")
            start = gr.Button("Start", variant="primary", scale=1)
            stop = gr.Button("Stop", scale=1)
        with gr.Row():
            with gr.Column():
                left = gr.Image(show_label=False, interactive=False, height=320,
                                show_download_button=False, show_fullscreen_button=True)
                cap_l = gr.HTML(caption("Without our pipeline", None))
            with gr.Column():
                right = gr.Image(show_label=False, interactive=False, height=320,
                                 show_download_button=False, show_fullscreen_button=True)
                cap_r = gr.HTML(caption("With our pipeline", None))
        gr.HTML('<div class="legend">Boxes:<span class="sw" style="background:#3cc83c"></span>'
                'correct detection<span class="sw" style="background:#e62828"></span>false detection'
                '<span class="sw" style="background:#ffd700"></span>missed object</div>')
        cards = gr.HTML(cards_html(None, chip))
        event = start.click(run, [stream, fps_cap], [left, right, cap_l, cap_r, flow, cards])
        stop.click(None, None, None, cancels=[event])
    return ui


def main() -> int:
    print("loading gate, restorer and detector...")
    pipe = Pipeline()
    build_ui(pipe).queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
