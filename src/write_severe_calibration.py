"""Write results/severity_calibration.json with severe cutoffs only (no mild band).

Implements the rule in results/severity_scale_decision.json ("proposed_next_step"), with
one approved change: the output carries severe_min_s per type and gate_exclude_below_s
for veiling_wash, and no mild_max_s at all.

Nothing is rendered and no detector runs. The cutoffs are recomputed from the drop curves
stored by the last calibration run, with the same interpolation calibrate_severity.py
uses, so no number is copied by hand.

REFUSES TO WRITE when:
  * the source file's sha256 differs from the one recorded in the decision file
  * the source's severity_scale differs from glare_synth.SEVERITY_SCALE
  * the source's frame list differs from the first calibration's (v1) frame list
  * any type's drop curve never reaches the severe threshold

    python src/write_severe_calibration.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import calibrate_severity as cal  # noqa: E402
import glare_synth as gs  # noqa: E402

GATE_EXCLUDE_TYPE = "veiling_wash"


class Refused(Exception):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, frames_ref: Path, decision: Path, scale: dict) -> dict:
    """Return the severe-only calibration dict, or raise Refused."""
    dec = json.loads(decision.read_text())
    expected_sha = dec["severe_band"]["current_scale_source"]["sha256"]
    actual_sha = sha256(source)
    if actual_sha != expected_sha:
        raise Refused(f"{source.name} sha256 {actual_sha[:12]}... does not match the decision "
                      f"file's {expected_sha[:12]}...")

    src = json.loads(source.read_text())
    want = {t: list(v) for t, v in scale.items()}
    if src["severity_scale"] != want:
        raise Refused(f"severity_scale in {source.name} {src['severity_scale']} does not "
                      f"match glare_synth.SEVERITY_SCALE {want}")

    ref = json.loads(frames_ref.read_text())
    if src["frames"]["ids_by_type"] != ref["frames"]["ids_by_type"]:
        raise Refused(f"frame list in {source.name} differs from {frames_ref.name}")

    severe = src["thresholds"]["severe_min_drop"]
    mild = src["thresholds"]["mild_max_drop"]
    cutpoints = {}
    for gtype, curve in src["curves"].items():
        s_severe = cal.interpolate_crossing(curve["monotone_drop"], severe)
        if s_severe != s_severe:                      # NaN: never reaches the threshold
            raise Refused(f"{gtype}: drop never reaches {severe}; no severe band")
        cutpoints[gtype] = {"severe_min_s": round(s_severe, 4)}
    exclude = cal.interpolate_crossing(src["curves"][GATE_EXCLUDE_TYPE]["monotone_drop"], mild)
    cutpoints[GATE_EXCLUDE_TYPE]["gate_exclude_below_s"] = round(exclude, 4)

    return {
        "status": "ok-severe-only",
        "cutpoints": cutpoints,
        "thresholds": {"severe_min_drop": severe,
                       "gate_exclusion_drop": f"{mild} (veiling_wash only; not a band)"},
        "severity_scale": src["severity_scale"],
        "source": {"file": str(source.relative_to(REPO_ROOT)) if source.is_relative_to(REPO_ROOT)
                   else source.name, "sha256": actual_sha},
        "decision": "results/severity_scale_decision.json",
        "frames": src["frames"],
        "detector": src["detector"],
        "data_use": src.get("data_use"),
        "note": "No mild band (see decision file). Cutoffs recomputed from the stored monotone "
                "drop curves with calibrate_severity.interpolate_crossing.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path,
                   default=REPO_ROOT / "results" / "severity_calibration_failed.json")
    p.add_argument("--frames-ref", type=Path,
                   default=REPO_ROOT / "results" / "severity_calibration_failed_v1.json")
    p.add_argument("--decision", type=Path,
                   default=REPO_ROOT / "results" / "severity_scale_decision.json")
    p.add_argument("--out", type=Path, default=gs.CALIBRATION)
    args = p.parse_args(argv)
    try:
        result = build(args.source, args.frames_ref, args.decision, gs.SEVERITY_SCALE)
    except Refused as exc:
        sys.exit(f"refused: {exc}")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for gtype, c in result["cutpoints"].items():
        print(f"  {gtype:<15} " + "  ".join(f"{k} {v}" for k, v in c.items()))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
