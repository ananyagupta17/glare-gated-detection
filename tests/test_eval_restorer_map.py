"""Tests for src/eval_restorer_map.py. None of them runs the detector or a restorer.

    python -m unittest discover tests
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "src" / "eval_restorer_map.py"
sys.path.insert(0, str(REPO_ROOT / "src"))
import eval_restorer_map as erm  # noqa: E402


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, timeout=120)


class Refusals(unittest.TestCase):
    def test_refuses_when_v2_val_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = run("whole", "--v2-root", tmp)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("v2 val data not found", res.stderr)

    def test_rejects_unknown_restorer(self):
        res = run("sharpen")
        self.assertEqual(res.returncode, 2)                  # argparse usage error
        self.assertIn("invalid choice", res.stderr)

    def test_refuses_missing_exclusion_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for v in ("clean", "glared"):
                (tmp / v / "images" / "val").mkdir(parents=True)
            res = run("none", "--v2-root", str(tmp),
                      "--calibration", str(tmp / "nope.json"), "--tuning", str(tmp / "nope2.json"))
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("not found", res.stderr)

    def test_refuses_val_trained_detector(self):
        with self.assertRaises(SystemExit) as ctx:
            erm.check_weights(Path("models/yolo_baseline.pt"))
        self.assertIn("contaminated", str(ctx.exception))


class Exclusion(unittest.TestCase):
    def files(self, tmp: Path, cal_split: str, cal_ids: list[str], tuning_ids: list[str]):
        cal, tun = tmp / "cal.json", tmp / "tun.json"
        cal.write_text(json.dumps({"frames": {"split": cal_split,
                                              "ids_by_type": {"direct_sun": cal_ids}}}))
        tun.write_text(json.dumps({"val_ids_used": tuning_ids}))
        return cal, tun

    def test_tuning_ids_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal, tun = self.files(Path(tmp), "train", ["000002", "000003"], ["000013", "000129"])
            excluded = erm.excluded_ids(cal, tun)
        keep, removed = erm.eligible(["000000", "000001", "000013", "000129"], excluded)
        self.assertEqual(keep, ["000000", "000001"])
        self.assertEqual(removed, ["000013", "000129"])

    def test_train_calibration_ids_do_not_touch_val(self):
        # Calibration frames are train frames; an ID number that also exists in val is a
        # different image, so it must not be removed.
        with tempfile.TemporaryDirectory() as tmp:
            cal, tun = self.files(Path(tmp), "train", ["000001"], [])
            keep, removed = erm.eligible(["000000", "000001"], erm.excluded_ids(cal, tun))
        self.assertEqual((keep, removed), (["000000", "000001"], []))

    def test_val_calibration_ids_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal, tun = self.files(Path(tmp), "val", ["000001"], [])
            keep, removed = erm.eligible(["000000", "000001"], erm.excluded_ids(cal, tun))
        self.assertEqual((keep, removed), (["000000"], ["000001"]))

    def test_real_tuning_file_lists_the_sample_frames(self):
        ids = json.loads((REPO_ROOT / "results" / "restore_tuning.json").read_text())["val_ids_used"]
        self.assertEqual(sorted(ids), ["000129", "001071", "003507", "003781", "005796"])


if __name__ == "__main__":
    unittest.main()
