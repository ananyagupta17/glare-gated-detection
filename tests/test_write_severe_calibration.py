"""Tests for src/write_severe_calibration.py, on fake calibration files only.

    python -m unittest discover tests
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import write_severe_calibration as wsc  # noqa: E402

SCALE = {"diffuse_haze": (0.05, 1.2), "direct_sun": (0.08, 1.5),
         "veiling_wash": (0.05, 1.2), "horizon_streak": (0.02, 1.0)}
FRAMES = {"split": "train", "ids_by_type": {t: [f"{k:06d}"] for k, t in enumerate(SCALE)}}
RISING = [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.24, 0.3, 0.35, 0.4, 0.45]   # crosses 0.25 at 0.6167


def calibration(curves=None, scale=SCALE, frames=FRAMES) -> dict:
    curves = curves or {t: RISING for t in scale}
    return {"severity_scale": {t: list(v) for t, v in scale.items()},
            "frames": frames, "thresholds": {"mild_max_drop": 0.05, "severe_min_drop": 0.25},
            "curves": {t: {"monotone_drop": c} for t, c in curves.items()},
            "detector": {"path": "models/yolo_baseline.pt"}}


class Writer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def files(self, source: dict, ref_frames=FRAMES, sha: str | None = None):
        src = self.dir / "source.json"
        src.write_text(json.dumps(source))
        ref = self.dir / "v1.json"
        ref.write_text(json.dumps({"frames": ref_frames}))
        dec = self.dir / "decision.json"
        digest = sha or hashlib.sha256(src.read_bytes()).hexdigest()
        dec.write_text(json.dumps({"severe_band": {"current_scale_source": {"sha256": digest}}}))
        return src, ref, dec

    def test_happy_path(self):
        out = wsc.build(*self.files(calibration()), SCALE)
        self.assertEqual(out["status"], "ok-severe-only")
        for gtype, c in out["cutpoints"].items():
            self.assertAlmostEqual(c["severe_min_s"], 0.6167, places=4)
            self.assertNotIn("mild_max_s", c)
        self.assertAlmostEqual(out["cutpoints"]["veiling_wash"]["gate_exclude_below_s"], 0.1667,
                               places=4)
        self.assertNotIn("gate_exclude_below_s", out["cutpoints"]["direct_sun"])

    def test_refuses_sha_mismatch(self):
        with self.assertRaisesRegex(wsc.Refused, "sha256"):
            wsc.build(*self.files(calibration(), sha="0" * 64), SCALE)

    def test_refuses_scale_mismatch(self):
        old = {**SCALE, "horizon_streak": (0.04, 1.0)}
        with self.assertRaisesRegex(wsc.Refused, "severity_scale"):
            wsc.build(*self.files(calibration(scale=old)), SCALE)

    def test_refuses_frame_mismatch(self):
        other = {"split": "train", "ids_by_type": {t: ["999999"] for t in SCALE}}
        with self.assertRaisesRegex(wsc.Refused, "frame list"):
            wsc.build(*self.files(calibration(), ref_frames=other), SCALE)

    def test_refuses_type_without_severe_band(self):
        flat = {t: RISING for t in SCALE}
        flat["veiling_wash"] = [0.0] * 6 + [0.1] * 5
        with self.assertRaisesRegex(wsc.Refused, "never reaches"):
            wsc.build(*self.files(calibration(curves=flat)), SCALE)

    def test_cli_refusal_writes_nothing(self):
        src, ref, dec = self.files(calibration(), sha="0" * 64)
        out = self.dir / "severity_calibration.json"
        with self.assertRaises(SystemExit) as ctx:
            wsc.main(["--source", str(src), "--frames-ref", str(ref), "--decision", str(dec),
                      "--out", str(out)])
        self.assertIn("refused", str(ctx.exception))
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
