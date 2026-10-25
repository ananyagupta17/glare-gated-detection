"""Live demo dashboard: glare gate -> daytime restoration -> YOLOv8n, in the browser.

    python src/demo_app.py            # then open http://127.0.0.1:7860

Plays the v2 KITTI val frames as a live stream (by default a 50/50 mix of clean and
glared frames, as in the C1/C2/C3 evaluation) and runs the pipeline on every frame:

  left   what a plain detector sees: the camera frame, detected as-is (C1)
  right  the pipeline under the selected policy: C1 never restore, C2 always restore,
         C3 restore only when the gate fires

and keeps live counters: per-stage latency (gate, restoration, detection), pipeline
frames per second, restoration trigger rate, the gate's hits and misses against the
known glare labels, and per-frame detection F1 for the plain detector vs the pipeline.

Boxes: green = correct detection, red = false detection, yellow = missed object
(ground truth from the KITTI labels). Timing excludes disk I/O and drawing; the plain-
detector panel is for display and is not counted in the pipeline latency.

DEMO STATUS: synthetic glare that failed its shortcut test; the features gate is likely
inflated (see demo/REPORT.md). Every number on screen is live, not pre-recorded.
"""

from __future__ import annotations

import csv
import json
import random
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
from train_gate_classifier import extract_features as gate_features  # noqa: E402

GATES = json.loads((REPO_ROOT / "results" / "gates_v2.json").read_text())
PARAMS = {r["image_id"]: r for r in csv.DictReader(erm.PARAMS.open()) if r["split"] == "val"}
LABELS = REPO_ROOT / "datasets" / "kitti" / "labels" / "val"
BOX_TP, BOX_FP, BOX_MISS = (60, 200, 60), (40, 40, 230), (0, 215, 255)       # BGR

MODES = ["C3 · gated (proposed)", "C2 · always restore", "C1 · never restore"]
GATE_NAMES = ["Features gate (9 image statistics)", "Brightness gate (upper-third luma)"]
RESTORERS = {
    "Daytime classical (veil subtraction, ~10-30 ms)": "classical",
    "Daytime learned (U-Net on our paired frames)": "learned",
    "Flare7K++ night model (old, ~1.9 s)": "flare7k",
}
STREAMS = ["Mixed 50/50 clean + glared", "Glared only", "Clean only"]


# --- models ----------------------------------------------------------------------------

class Pipeline:
    def __init__(self):
        import torch
        from ultralytics import YOLO
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.detector = YOLO(str(erm.DEFAULT_WEIGHTS))
        self._restorers = {}
        f = GATES["features"]
        self.mu, self.sd, self.w = (np.asarray(f[k], dtype=np.float64) for k in ("mu", "sd", "w"))
        self.t_bright = float(GATES["brightness"]["threshold"])
        warm = cv2.imread(str(erm.V2_ROOT / "clean" / "images" / "val" / "000000.jpg"))
        for _ in range(5):
            self.detect(warm)

    def restorer(self, key: str):
        if key not in self._restorers:
            if key == "classical":
                import restore_daytime
                self._restorers[key] = restore_daytime.restore
            elif key == "learned":
                import restore_daytime_net
                r = restore_daytime_net.get_restorer()
                self._restorers[key] = r.restore
            else:
                from restore_seamless import SeamlessFlare7KRestorer
                r = SeamlessFlare7KRestorer(mode="whole", device="cpu")
                self._restorers[key] = r.restore
            warm = cv2.imread(str(erm.V2_ROOT / "glared" / "images" / "val" / "000000.jpg"))
            self._restorers[key](warm)
        return self._restorers[key]

    def gate(self, img: np.ndarray, which: str) -> tuple[bool, float]:
        if which.startswith("Brightness"):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            v = float(g[: g.shape[0] // 3].mean())
            return v > self.t_bright, v
        x = (gate_features(img).astype(np.float64) - self.mu) / self.sd
        z = self.w[0] + float(np.dot(self.w[1:], x))
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return p >= 0.5, p

    def detect(self, img: np.ndarray) -> np.ndarray:
        r = self.detector.predict(img, conf=cal.CONF, device=self.device, verbose=False)[0]
        b = r.boxes
        if not len(b):
            return np.zeros((0, 6), np.float32)
        return np.concatenate([b.cls.cpu().numpy()[:, None], b.xyxy.cpu().numpy(),
                               b.conf.cpu().numpy()[:, None]], 1)


def draw(img: np.ndarray, pred: np.ndarray, gt: np.ndarray, banner: str, color) -> tuple[np.ndarray, float]:
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
    f1 = 2 * tp / max(2 * tp + fp + miss, 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(out, f"{banner}   F1 {f1:.2f}  ({tp} found, {fp} false, {miss} missed)",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return out[:, :, ::-1], f1


def stream_frames(which: str, seed: int) -> list[tuple[str, str]]:
    dirs = erm.v2_val_dirs(erm.V2_ROOT)
    ids, _ = erm.eligible(sorted(p.stem for p in dirs["glared"].glob("*.jpg")),
                          erm.excluded_ids(erm.CALIBRATION, erm.TUNING))
    rng = random.Random(seed)
    if which == "Glared only":
        frames = [(i, "glared") for i in ids]
    elif which == "Clean only":
        frames = [(i, "clean") for i in ids]
    else:   # each frame appears once, clean or glared at random: a 50/50 mixed stream
        frames = [(i, rng.choice(("clean", "glared"))) for i in ids]
    rng.shuffle(frames)
    return frames


# --- dashboard -------------------------------------------------------------------------

def build_ui(pipe: Pipeline):
    import gradio as gr

    def run(mode, gate_name, restorer_name, stream, fps_cap, seed):
        frames = stream_frames(stream, int(seed))
        restore = None if mode.startswith("C1") else pipe.restorer(RESTORERS[restorer_name])
        stats = {"n": 0, "restored": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
        f1 = {"plain": {"clean": [], "glared": []}, "pipe": {"clean": [], "glared": []}}
        recent = {k: deque(maxlen=30) for k in ("gate", "restore", "detect", "total")}
        for image_id, version in frames:
            t_frame = time.perf_counter()
            img = cv2.imread(str(erm.V2_ROOT / version / "images" / "val" / f"{image_id}.jpg"))
            gt = cal.read_labels(LABELS / f"{image_id}.txt", img.shape[1], img.shape[0])

            t0 = time.perf_counter()                       # ---- pipeline, timed ----
            if mode.startswith("C3"):
                fire, score = pipe.gate(img, gate_name)
            else:
                fire, score = mode.startswith("C2"), None
            t1 = time.perf_counter()
            x = restore(img) if (fire and restore) else img
            t2 = time.perf_counter()
            pred_pipe = pipe.detect(x)
            t3 = time.perf_counter()                       # ---- end of timed part ----

            pred_plain = pred_pipe if x is img else pipe.detect(img)   # display only
            truth = version == "glared"
            r = PARAMS[image_id]
            label = (f"GLARED · {r['glare_type'].replace('_', ' ')} · s={float(r['severity']):.2f}"
                     f"{' · SEVERE' if r['severe'] == '1' else ''}" if truth else "CLEAN")
            left, f1_plain = draw(img, pred_plain, gt, f"Camera, plain detector · {label}",
                                  (90, 170, 255) if truth else (220, 220, 220))
            action = ("RESTORED" if fire else "skipped restoration")
            right, f1_pipe = draw(x, pred_pipe, gt, f"Pipeline · {action}",
                                  (110, 220, 110) if fire else (220, 220, 220))

            stats["n"] += 1
            stats["restored"] += bool(fire and restore)
            if mode.startswith("C3"):
                key = ("tp" if fire else "fn") if truth else ("fp" if fire else "tn")
                stats[key] += 1
            f1["plain"][version].append(f1_plain)
            f1["pipe"][version].append(f1_pipe)
            for k, v in (("gate", t1 - t0), ("restore", t2 - t1), ("detect", t3 - t2),
                         ("total", t3 - t0)):
                recent[k].append(1000 * v)

            yield left, right, panel(mode, gate_name, restorer_name, fire, score, stats, f1, recent)
            if fps_cap:
                wait = 1.0 / fps_cap - (time.perf_counter() - t_frame)
                if wait > 0:
                    time.sleep(wait)

    def panel(mode, gate_name, restorer_name, fire, score, s, f1, recent):
        mean = {k: (np.mean(v) if v else 0.0) for k, v in recent.items()}
        fps = 1000.0 / mean["total"] if mean["total"] else 0.0
        avg = lambda xs: f"{np.mean(xs):.2f}" if xs else "–"  # noqa: E731
        gate_line = ""
        if mode.startswith("C3"):
            pos, neg = s["tp"] + s["fn"], s["fp"] + s["tn"]
            gate_line = (f"| Gate decision this frame | **{'RESTORE' if fire else 'skip'}** "
                         f"({'p' if gate_name.startswith('Features') else 'luma'} = {score:.2f}) |\n"
                         f"| Gate: glare caught | {s['tp']} / {pos} "
                         f"({s['tp'] / pos:.0%}) |\n" if pos else "") + \
                        (f"| Gate: clean frames restored by mistake | {s['fp']} / {neg} "
                         f"({s['fp'] / neg:.0%}) |\n" if neg else "")
        return (
            f"### Live · {s['n']} frames · **{fps:.1f} fps** pipeline\n"
            f"| | |\n|---|---|\n"
            f"| Policy | {mode} |\n"
            f"| Restorer | {restorer_name if not mode.startswith('C1') else '—'} |\n"
            f"| Restoration trigger rate | **{s['restored'] / s['n']:.0%}** |\n"
            + gate_line +
            f"\n**Latency per frame (ms, mean of last 30, device: detector {pipe.device}, restorer CPU)**\n\n"
            f"| Gate | Restoration | Detection | **Total** |\n|---|---|---|---|\n"
            f"| {mean['gate']:.1f} | {mean['restore']:.1f} | {mean['detect']:.1f} | "
            f"**{mean['total']:.1f}** |\n\n"
            f"**Detection F1 per frame (mean so far)**\n\n"
            f"| | Plain detector | Pipeline |\n|---|---|---|\n"
            f"| Clean frames | {avg(f1['plain']['clean'])} | {avg(f1['pipe']['clean'])} |\n"
            f"| Glared frames | {avg(f1['plain']['glared'])} | {avg(f1['pipe']['glared'])} |\n"
        )

    learned_ok = (REPO_ROOT / "models" / "restore_daytime_net.pt").is_file()
    # Flare7K++ (the old night model) is offered only where its code is present; it is
    # not part of the main pipeline.
    flare_ok = (REPO_ROOT / "src" / "restore_seamless.py").is_file() and \
        (REPO_ROOT / "src" / "restore.py").is_file()
    choices = [k for k, v in RESTORERS.items()
               if (v != "learned" or learned_ok) and (v != "flare7k" or flare_ok)]
    with gr.Blocks(title="Glare-gated detection · live demo") as ui:
        gr.Markdown(
            "## Glare-gated object detection · live pipeline\n"
            "**Gate → daytime restoration (only if the gate fires) → YOLOv8n**, on KITTI frames "
            "with synthetic sun glare. Boxes: 🟩 correct · 🟥 false · 🟨 missed. "
            "_Demo data: the synthetic glare failed its shortcut test, so gate accuracy here is "
            "optimistic._")
        with gr.Row():
            mode = gr.Radio(MODES, value=MODES[0], label="Policy")
            gate_name = gr.Dropdown(GATE_NAMES, value=GATE_NAMES[0], label="Gate (C3)")
            restorer_name = gr.Dropdown(choices, value=choices[1] if learned_ok else choices[0],
                                        label="Restorer")
        with gr.Row():
            stream = gr.Radio(STREAMS, value=STREAMS[0], label="Input stream")
            fps_cap = gr.Slider(0, 30, value=0, step=1, label="Playback cap (fps, 0 = as fast as it runs)")
            seed = gr.Number(value=42, precision=0, label="Stream seed")
        with gr.Row():
            start = gr.Button("▶ Start", variant="primary")
            stop = gr.Button("■ Stop")
        with gr.Row():
            left = gr.Image(label="Camera · plain detector (C1)", type="numpy")
            right = gr.Image(label="Pipeline output", type="numpy")
        stats = gr.Markdown()
        event = start.click(run, [mode, gate_name, restorer_name, stream, fps_cap, seed],
                            [left, right, stats])
        stop.click(None, None, None, cancels=[event])
    return ui


def main() -> int:
    print("loading detector, gates and restorers...")
    pipe = Pipeline()
    pipe.restorer("classical")
    build_ui(pipe).queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
