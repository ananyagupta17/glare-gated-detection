"""Tests for src/bdd_prepare.py that need no BDD100K data.

    python -m unittest discover tests
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import bdd_prepare as bp  # noqa: E402


class ClassMapping(unittest.TestCase):
    def test_mapping_and_drops(self):
        frame = {"labels": [
            {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 128, "y2": 72}},
            {"category": "rider", "box2d": {"x1": 640, "y1": 360, "x2": 704, "y2": 432}},
            {"category": "bus", "box2d": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}},
            {"category": "traffic sign", "box2d": {"x1": 10, "y1": 10, "x2": 50, "y2": 50}},
            {"category": "car"},                                   # no box: skipped
        ]}
        rows = bp.to_yolo(frame)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], "0 0.050000 0.050000 0.100000 0.100000")
        self.assertTrue(rows[1].startswith("5 "))

    def test_van_merge_is_declared(self):
        self.assertEqual(bp.EVAL_MERGE, {1: 0})


class GlareScore(unittest.TestCase):
    def test_sun_blob_scores_higher(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(0)
            road = (rng.random((720, 1280, 3)) * 120 + 20).astype(np.uint8)
            sun = road.copy()
            cv2.circle(sun, (640, 150), 120, (255, 255, 255), -1)
            cv2.imwrite(f"{tmp}/road.jpg", road)
            cv2.imwrite(f"{tmp}/sun.jpg", sun)
            self.assertGreater(bp.glare_score(f"{tmp}/sun.jpg"), bp.glare_score(f"{tmp}/road.jpg"))


class Refusal(unittest.TestCase):
    def test_missing_dataset_refuses(self):
        old = bp.IMAGES
        bp.IMAGES = Path("/nonexistent/bdd")
        try:
            with self.assertRaises(SystemExit):
                bp.load_labels()
        finally:
            bp.IMAGES = old


if __name__ == "__main__":
    unittest.main()
