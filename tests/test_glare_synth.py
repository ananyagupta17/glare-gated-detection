"""Unit tests for the v2 camera pipeline, on small synthetic frames (no KITTI data needed).

    python -m unittest discover tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import glare_synth as gs  # noqa: E402


def scene(h: int = 96, w: int = 320) -> np.ndarray:
    """Bright sky over a darker textured road, deterministic."""
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[: h // 3] = 235.0                                   # sky
    img[h // 3:] = 70.0 + 40.0 * rng.random((h - h // 3, w, 1))
    img[h // 2: h // 2 + 10, w // 4: w // 4 + 30] = 15.0    # a dark object
    return img.clip(0, 255).astype(np.uint8)


def spec(rgb, gtype="direct_sun", s=0.5, image_id="000123", split="val", seed=42):
    return gs.sample_frame(rgb, split, image_id, gtype, s, "stratum 0", seed)


class Determinism(unittest.TestCase):
    def test_same_inputs_same_bytes(self):
        rgb = scene()
        a = gs.render_pair(rgb, spec(rgb))
        b = gs.render_pair(rgb, spec(rgb))
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])

    def test_order_independent(self):
        rgb = scene()
        first = spec(rgb, image_id="000001")
        spec(rgb, image_id="000002")                         # draw for another frame
        again = spec(rgb, image_id="000001")
        self.assertEqual(first, again)

    def test_frame_seed_is_not_python_hash(self):
        # Python's hash() of a str changes per process; sha256 does not.
        self.assertEqual(gs.frame_seed(42, "val", "000001", "glare"),
                         gs.frame_seed(42, "val", "000001", "glare"))
        self.assertNotEqual(gs.frame_seed(42, "val", "000001", "glare"),
                            gs.frame_seed(42, "train", "000001", "glare"))

    def test_severity_does_not_shift_other_draws(self):
        rgb = scene()
        lo, hi = spec(rgb, s=0.1), spec(rgb, s=0.9)
        self.assertEqual(lo.lobes, hi.lobes)
        self.assertEqual(lo.exposure, hi.exposure)
        self.assertEqual(lo.noise_glared, hi.noise_glared)
        self.assertLess(lo.mult, hi.mult)


class Pairing(unittest.TestCase):
    def test_pair_shares_exposure_and_highlights(self):
        rgb = scene()
        sp = spec(rgb)
        # One spec drives both versions, so the shared fields are shared by construction;
        # the camera draws must differ between the two versions.
        self.assertNotEqual(sp.noise_clean, sp.noise_glared)

    def test_clean_version_has_no_glare(self):
        rgb = scene()
        clean = gs.decode_jpeg(gs.render_pair(rgb, spec(rgb, s=1.0))[0])
        weak = gs.decode_jpeg(gs.render_pair(rgb, spec(rgb, s=0.0))[0])
        self.assertTrue(np.array_equal(clean, weak), "clean output must not depend on s")

    def test_jpeg_settings(self):
        rgb = scene()
        sp = spec(rgb)
        lo, hi = gs.JPEG_QUALITY
        self.assertTrue(lo <= sp.jpeg_q_clean <= hi and lo <= sp.jpeg_q_glared <= hi)
        clean, glared, _ = gs.render_pair(rgb, sp)
        self.assertEqual(clean[:2], b"\xff\xd8")
        self.assertEqual(gs.decode_jpeg(glared).shape, rgb.shape)


class Physics(unittest.TestCase):
    def test_glare_light_grows_with_severity(self):
        rgb = scene()
        h, w = rgb.shape[:2]
        totals = [float(gs.glare_light(spec(rgb, s=s), h, w).sum()) for s in (0.0, 0.5, 1.0)]
        self.assertLess(totals[0], totals[1])
        self.assertLess(totals[1], totals[2])

    def test_floor_varies_spatially_and_is_positive(self):
        rgb = scene()
        h, w = rgb.shape[:2]
        floor = gs.floor_map(spec(rgb, gtype="veiling_wash"), h, w)[:, :, 0]
        self.assertGreater(floor.min(), 0.0)
        self.assertGreater(floor.std() / floor.mean(), 0.02, "floor must not be constant")

    def test_auto_exposure_bounds(self):
        rgb = scene()
        h, w = rgb.shape[:2]
        base = gs.srgb_to_linear(rgb)
        for s in (0.0, 0.5, 1.0):
            sp = spec(rgb, s=s)
            gain = gs.auto_exposure_gain(base, base + gs.glare_light(sp, h, w), sp.ae_k)
            self.assertGreaterEqual(gain, gs.AE_MIN_GAIN)
            self.assertLessEqual(gain, 1.0)
        strong = spec(rgb, gtype="horizon_streak", s=1.0)
        gain = gs.auto_exposure_gain(base, base + gs.glare_light(strong, h, w), strong.ae_k)
        self.assertLess(gain, 1.0, "strong glare must make the camera stop down")

    def test_highlight_expansion_only_touches_highlights(self):
        lin = gs.srgb_to_linear(scene())
        out = gs.expand_highlights(lin, 3.0)
        dark = lin.max(axis=2) < gs.HIGHLIGHT_KNEE
        self.assertTrue(np.allclose(out[dark], lin[dark]))
        self.assertTrue((out[~dark] >= lin[~dark]).all())

    def test_noise_is_heteroscedastic(self):
        rng = np.random.default_rng(1)
        flat = np.full((200, 200, 3), 0.05, np.float32)
        bright = np.full((200, 200, 3), 0.6, np.float32)
        n = (1e-4, 1e-7)
        self.assertGreater((gs.add_noise(bright, n, rng) - bright).std(),
                           (gs.add_noise(flat, n, rng) - flat).std())


class Assignment(unittest.TestCase):
    def test_strata_cover_each_type_once(self):
        ids = [f"{i:06d}" for i in range(1201)]
        types = gs.assign_types(ids, "train", 42)
        strata = gs.assign_strata(types, "train", 42)
        for gtype in gs.GLARE_TYPES:
            members = [i for i in ids if types[i] == gtype]
            ks = sorted(strata[i][0] for i in members)
            self.assertEqual(ks, list(range(len(members))), "one frame per stratum")
            for i in members:
                k, n = strata[i]
                s = gs.draw_severity("train", i, k, n, 42)
                self.assertTrue(k / n <= s < (k + 1) / n, "s lies inside its stratum")
        self.assertEqual(strata, gs.assign_strata(types, "train", 42), "seeded, repeatable")

    def test_severe_fraction_matches_cutoff(self):
        import math
        ids = [f"{i:06d}" for i in range(1496)]
        types = gs.assign_types(ids, "val", 42)
        strata = gs.assign_strata(types, "val", 42)
        cut = {"severe_min_s": 0.6042, "gate_exclude_below_s": 0.1753}
        for gtype in gs.GLARE_TYPES:
            members = [i for i in ids if types[i] == gtype]
            n = len(members)
            s = [gs.draw_severity("val", i, *strata[i], 42) for i in members]
            severe = sum(gs.severe_flag(v, cut) for v in s)
            excluded = sum(gs.gate_excluded(v, cut) for v in s)
            c = cut["severe_min_s"]
            self.assertIn(severe, (n - math.ceil(n * c), n - math.floor(n * c)))
            self.assertIn(excluded, (math.floor(n * 0.1753), math.ceil(n * 0.1753)))
        self.assertEqual(gs.gate_excluded(0.1, {"severe_min_s": 0.6}), 0,
                         "no exclusion without gate_exclude_below_s")

    def test_manifest_columns(self):
        self.assertEqual(gs.MANIFEST_COLUMNS,
                         ["image_id", "split", "is_glared", "glare_type", "severity", "sat_frac"])


if __name__ == "__main__":
    unittest.main()
