"""Save the two C3 gates (fit exactly as src/run_arms_v2.py does) to results/gates_v2.json,
so the live demo can load them instantly instead of refitting.

Refuses to write unless the exported gates reproduce the gate metrics recorded in
results/arms_v2.json on the val stream.

    python src/export_gates_v2.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import eval_restorer_map as erm  # noqa: E402
import run_arms_v2 as ra  # noqa: E402
import shortcut_test as st  # noqa: E402
from train_gate_classifier import FEATURE_NAMES  # noqa: E402

OUT = REPO_ROOT / "results" / "gates_v2.json"


def main() -> int:
    recorded = json.loads((REPO_ROOT / "results" / "arms_v2.json").read_text())["gates"]["metrics_val"]
    params = {(r["split"], r["image_id"]): r for r in csv.DictReader(erm.PARAMS.open())}
    t, lr, info = ra.train_gates(params)
    dirs = erm.v2_val_dirs(erm.V2_ROOT)
    ids, _ = erm.eligible(sorted(p.stem for p in dirs["glared"].glob("*.jpg")),
                          erm.excluded_ids(erm.CALIBRATION, erm.TUNING))
    stream = [(i, v) for i in ids for v in ("clean", "glared")]
    feats = st.extract_all([erm.V2_ROOT / v / "images" / "val" / f"{i}.jpg" for i, v in stream], ra.WORKERS)
    scores = lr.score(st.matrix(feats, ["gate"]))
    m_feat = ra.gate_metrics({k: s >= 0.5 for k, s in zip(stream, scores)}, stream, params)
    m_bright = ra.gate_metrics({k: f["upper_third_mean_luma"] > t for k, f in zip(stream, feats)},
                               stream, params)
    ok = m_feat == recorded["C3_features"] and m_bright == recorded["C3_brightness"]
    print(f"reproduces recorded metrics: features {m_feat == recorded['C3_features']}, "
          f"brightness {m_bright == recorded['C3_brightness']}")
    if not ok:
        sys.exit("error: exported gates do not reproduce results/arms_v2.json; not written")
    OUT.write_text(json.dumps({
        "fit_on": "v2 train only, as src/run_arms_v2.py", **info,
        "brightness": {"feature": "upper_third_mean_luma (Rec.601 luma, 0-255)",
                       "threshold": t, "rule": "glare if value > threshold"},
        "features": {"feature_names": FEATURE_NAMES, "mu": lr.mu.tolist(), "sd": lr.sd.tolist(),
                     "w": lr.w.tolist(), "rule": "glare if sigmoid(w0 + w . (x - mu)/sd) >= 0.5",
                     "features_from": "src/train_gate_classifier.py extract_features (BGR uint8)"},
        "val_metrics": {"C3_features": m_feat, "C3_brightness": m_bright},
        "status": "DEMO: v2 failed its shortcut test; the features gate is likely inflated",
    }, indent=2, default=lambda o: o.item()) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
