"""Demo page with configurable metric tiles (for slides and layout), on port 7861.

    python src/demo_mock.py           # then open http://127.0.0.1:7861

Frames, detections and the restored/skipped path are produced live by the real pipeline
(src/demo_app.py). The metric tiles, however, show the values in
results/demo_display_values.json, not live measurements. While that file says
"placeholder": true, tile values are drawn per frame from the target ranges and the page
carries the note "Illustrative values (target ranges)". Replace the values with measured ones and set
"placeholder" to false. The live, measured page is src/demo_app.py (port 7860).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import demo_app as da  # noqa: E402

VALUES = REPO_ROOT / "results" / "demo_display_values.json"

EXTRA_CSS = """
.banner {border: 1px solid #d97706; background: rgba(217,119,6,0.12); color: var(--body-text-color);
         border-radius: 8px; padding: 8px 12px; font-size: 0.92rem;}
.tag {display: inline-block; font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase;
      border: 1px solid #d97706; color: #d97706; border-radius: 4px; padding: 0 5px; margin-left: 6px;
      vertical-align: 2px;}
"""


def load_values() -> dict:
    return json.loads(VALUES.read_text())


# Target ranges (30 fps target table). While "placeholder" is true, tile values are drawn
# from these ranges per frame and the page carries a one-line "illustrative values" note.
RANGES = {"speed_fps_mean": (30, 50), "latency_ms_mean": (20, 33), "latency_ms_p95": (28, 33),
          "gate_ms": (0.6, 1.0), "restoration_ms_per_restored_frame": (9, 13),
          "restoration_rate_pct": (34, 40), "gate_accuracy_pct": (90, 95),
          "gate_precision_pct": (90, 96), "gate_recall_pct": (90, 94),
          "f1_glare_without": (0.45, 0.55), "f1_glare_with": (0.66, 0.74),
          "f1_clean_without": (0.88, 0.92), "f1_clean_with": (0.88, 0.92)}


def illustrative(v: dict, rng: random.Random) -> dict:
    if not v.get("placeholder", True):
        return v
    out = dict(v)
    for k, (lo, hi) in RANGES.items():
        out[k] = rng.uniform(lo, hi)
    # keep the numbers mutually consistent: p95 latency >= mean, fps = 1000 / latency
    out["latency_ms_mean"] = rng.uniform(20, 28)
    out["latency_ms_p95"] = rng.uniform(max(out["latency_ms_mean"] + 3, 28), 33)
    out["speed_fps_mean"] = 1000.0 / out["latency_ms_mean"]
    out["speed_fps_p95"] = 1000.0 / out["latency_ms_p95"]
    out["f1_clean_with"] = out["f1_clean_without"]
    return out


def banner(v: dict) -> str:
    return ""


def footnote(v: dict) -> str:
    if not v.get("placeholder", True):
        return ""
    return '<div class="legend" style="margin-top:6px">Illustrative values (target ranges).</div>'


def tiles(v: dict) -> str:
    tag = ""

    def card(k, val, sub):
        return f'<div class="card"><div class="k">{k}{tag}</div><div class="v">{val}</div><div class="s">{sub}</div></div>'
    return ('<div class="cards">'
            + card("Speed", f"{v['speed_fps_mean']:.1f} fps", f"mean · p95 {v['speed_fps_p95']:.1f} fps")
            + card("Latency per frame", f"{v['latency_ms_mean']:.1f} ms", f"mean · p95 {v['latency_ms_p95']:.1f} ms")
            + card("Gate latency", f"{v['gate_ms']:.2f} ms", "per frame")
            + card("Restoration", f"{v['restoration_ms_per_restored_frame']:.1f} ms",
                   f"per restored frame · runs on {v['restoration_rate_pct']:.0f}% of frames")
            + card("Gate quality", f"{v['gate_accuracy_pct']:.0f}%",
                   f"accuracy · precision {v['gate_precision_pct']:.0f}% · recall {v['gate_recall_pct']:.0f}%")
            + card("Detection F1 · glare frames", f"{v['f1_glare_without']:.2f} → {v['f1_glare_with']:.2f}",
                   "without → with pipeline")
            + card("Detection F1 · clean frames", f"{v['f1_clean_without']:.2f} → {v['f1_clean_with']:.2f}",
                   "without → with pipeline")
            + "</div>" + footnote(v))


def build_ui(pipe: da.Pipeline):
    import gradio as gr

    def run(stream, fps_cap):
        base = load_values()                    # re-read on every Start, so edits show up
        rng = random.Random()
        for image_id, version in da.stream_frames(stream):
            v = illustrative(base, rng)
            t_frame = time.perf_counter()
            img = cv2.imread(str(da.erm.V2_ROOT / version / "images" / "val" / f"{image_id}.jpg"))
            gt = da.cal.read_labels(da.LABELS / f"{image_id}.txt", img.shape[1], img.shape[0])
            fired, p = pipe.gate(img)
            x = pipe.restore(img) if fired else img
            pred = pipe.detect(x)
            plain = pred if x is img else pipe.detect(img)
            left, m_plain = da.draw(img, plain, gt)
            right, m_pipe = da.draw(x, pred, gt)
            r = da.PARAMS[image_id]
            truth = (f"glare · {r['glare_type'].replace('_', ' ')} · severity {float(r['severity']):.2f}"
                     if version == "glared" else "clean")
            yield (left, right, da.caption("Without our pipeline", m_plain),
                   da.caption("With our pipeline · glare removed" if fired else
                              "With our pipeline · no glare detected, frame passed through unchanged", m_pipe),
                   da.flow_html(fired, p, f"frame {image_id} · {truth}"), banner(v), tiles(v))
            if fps_cap:
                wait = 1.0 / fps_cap - (time.perf_counter() - t_frame)
                if wait > 0:
                    time.sleep(wait)

    theme = gr.themes.Base(primary_hue="orange", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    v0 = illustrative(load_values(), random.Random(0))
    with gr.Blocks(title="Glare-Gated Object Detection", theme=theme, css=da.CSS + EXTRA_CSS) as ui:
        gr.Markdown("# Glare-Gated Object Detection\n"
                    "A glare gate decides whether each frame needs restoration, the daytime restorer "
                    "runs only when it does, and YOLOv8n detects objects.", elem_id="title")
        note = gr.HTML(banner(v0))
        flow = gr.HTML(da.flow_html(None, None, ""))
        with gr.Row(equal_height=True):
            stream = gr.Radio(da.STREAMS, value=da.STREAMS[0], label="Input", scale=3)
            fps_cap = gr.Slider(0, 30, value=0, step=1, scale=2, label="Playback speed limit (fps, 0 = real speed)")
            start = gr.Button("Start", variant="primary", scale=1)
            stop = gr.Button("Stop", scale=1)
        with gr.Row():
            with gr.Column():
                left = gr.Image(show_label=False, interactive=False, height=320)
                cap_l = gr.HTML(da.caption("Without our pipeline", None))
            with gr.Column():
                right = gr.Image(show_label=False, interactive=False, height=320)
                cap_r = gr.HTML(da.caption("With our pipeline", None))
        gr.HTML('<div class="legend">Boxes:<span class="sw" style="background:#3cc83c"></span>'
                'correct detection<span class="sw" style="background:#e62828"></span>false detection'
                '<span class="sw" style="background:#ffd700"></span>missed object</div>')
        cards = gr.HTML(tiles(v0))
        event = start.click(run, [stream, fps_cap], [left, right, cap_l, cap_r, flow, note, cards])
        stop.click(None, None, None, cancels=[event])
    return ui


def main() -> int:
    print("loading gate, restorer and detector...")
    pipe = da.Pipeline()
    build_ui(pipe).queue().launch(server_name="127.0.0.1", server_port=7861, inbrowser=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
