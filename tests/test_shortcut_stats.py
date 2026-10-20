"""Unit tests for the statistics in shortcut_test.py and calibrate_severity.py.

    python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import calibrate_severity as cal  # noqa: E402
import shortcut_test as st  # noqa: E402


class Auc(unittest.TestCase):
    def test_perfect_and_chance(self):
        self.assertEqual(st.auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])), 1.0)
        self.assertEqual(st.auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])), 0.0)
        self.assertEqual(st.auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])), 0.5)

    def test_two_sided(self):
        self.assertEqual(st.two_sided(0.2), 0.8)
        self.assertEqual(st.two_sided(0.7), 0.7)

    def test_matches_pairwise_definition(self):
        rng = np.random.default_rng(0)
        pos, neg = rng.integers(0, 5, 40).astype(float), rng.integers(0, 5, 30).astype(float)
        pairwise = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
        self.assertAlmostEqual(st.auc(pos, neg), pairwise)


class LogisticRegression(unittest.TestCase):
    def test_separable_and_noise(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 2))
        y = (x[:, 0] > 0).astype(float)
        acc = ((st.LogReg().fit(x, y).score(x) >= 0.5) == y).mean()
        self.assertGreater(acc, 0.95)
        noise = rng.normal(size=(400, 3))
        y_rand = rng.integers(0, 2, 400).astype(float)
        test = rng.normal(size=(400, 3))
        acc = ((st.LogReg().fit(noise, y_rand).score(test) >= 0.5)
               == rng.integers(0, 2, 400)).mean()
        self.assertLess(acc, 0.6)


class Features(unittest.TestCase):
    def test_noise_estimate_tracks_sigma(self):
        rng = np.random.default_rng(0)
        base = np.full((200, 300), 128.0)
        for sigma in (2.0, 8.0):
            est = st.noise_level(base + rng.normal(0, sigma, base.shape))
            self.assertAlmostEqual(est, sigma, delta=0.15 * sigma)

    def test_hf_mf_ratio_ignores_contrast_scale(self):
        rng = np.random.default_rng(0)
        img = rng.normal(0, 20, (120, 200)).cumsum(axis=1)
        self.assertAlmostEqual(st.hf_mf_ratio(img), st.hf_mf_ratio(0.5 * img), places=6)


class CalibrationMath(unittest.TestCase):
    def test_frame_f1(self):
        gt = np.array([[0, 0, 0, 10, 10], [3, 20, 20, 30, 30]], np.float32)
        perfect = np.array([[0, 0, 0, 10, 10, .9], [3, 20, 20, 30, 30, .8]], np.float32)
        self.assertEqual(cal.frame_f1(perfect, gt), 1.0)
        wrong_class = np.array([[1, 0, 0, 10, 10, .9]], np.float32)
        self.assertEqual(cal.frame_f1(wrong_class, gt), 0.0)
        self.assertEqual(cal.frame_f1(np.zeros((0, 6), np.float32), np.zeros((0, 5))), 1.0)

    def test_interpolate_crossing(self):
        curve = [0.0, 0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3]
        self.assertAlmostEqual(cal.interpolate_crossing(curve, 0.05), 0.25)
        self.assertAlmostEqual(cal.interpolate_crossing(curve, 0.25), 0.55)
        self.assertTrue(np.isnan(cal.interpolate_crossing(curve, 0.9)))


if __name__ == "__main__":
    unittest.main()
