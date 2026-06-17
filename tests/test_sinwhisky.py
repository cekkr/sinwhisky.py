"""
Unit tests for SinWhisky — ordered from the simplest properties to the most
complex (geometry -> resampler invariants -> analytic exact cases -> behaviour
on real signals -> C/Python cross-validation -> compressor round-trip).

Run:
    python3 -m unittest discover -s tests -v
or:
    python3 tests/test_sinwhisky.py
"""

import os
import sys
import math
import shutil
import struct
import subprocess
import unittest

import numpy as np

# Make ``import v2.whisky`` work no matter the CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v2.whisky import (  # noqa: E402
    find_circle_from_three_points,
    sinwhisky_resample,
    find_optimal_circles,
    encode_circles_for_compression,
    decode_circles_from_compression,
    reconstruct_signal,
)

_C_DIR = os.path.join(_ROOT, "sinwhisky_c")


# ---------------------------------------------------------------------------
# 1. Circle geometry — the most basic building block
# ---------------------------------------------------------------------------

class TestCircleGeometry(unittest.TestCase):
    def test_unit_circle(self):
        # Three points on the unit circle centred at the origin.
        cx, cy, r = find_circle_from_three_points((1, 0), (0, 1), (-1, 0))
        self.assertAlmostEqual(cx, 0.0, places=9)
        self.assertAlmostEqual(cy, 0.0, places=9)
        self.assertAlmostEqual(r, 1.0, places=9)

    def test_offset_circle(self):
        # Circle centred at (2, -3), radius 5.
        cx0, cy0, r0 = 2.0, -3.0, 5.0
        pts = [(cx0 + r0 * math.cos(a), cy0 + r0 * math.sin(a))
               for a in (0.3, 1.7, 4.0)]
        cx, cy, r = find_circle_from_three_points(*pts)
        self.assertAlmostEqual(cx, cx0, places=7)
        self.assertAlmostEqual(cy, cy0, places=7)
        self.assertAlmostEqual(r, r0, places=7)

    def test_collinear_raises(self):
        with self.assertRaises(ValueError):
            find_circle_from_three_points((0, 0), (1, 1), (2, 2))


# ---------------------------------------------------------------------------
# 2. Resampler invariants (independent of the signal content)
# ---------------------------------------------------------------------------

class TestResamplerInvariants(unittest.TestCase):
    def test_zoom_zero_is_identity(self):
        y = [0.0, 1.0, 0.5, -1.0, 2.0]
        out = sinwhisky_resample(y, zoom=0)
        np.testing.assert_allclose(out, y, atol=0.0)

    def test_output_length_formula(self):
        for n in (1, 2, 3, 8, 33):
            for zoom in (0, 1, 3, 7):
                y = np.linspace(0, 1, n)
                out = sinwhisky_resample(y, zoom=zoom)
                expected = (n - 1) * (zoom + 1) + 1 if n >= 1 else 0
                self.assertEqual(len(out), expected, (n, zoom))

    def test_originals_preserved_exactly(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(20)
        zoom = 6
        out = sinwhisky_resample(y, zoom=zoom)
        # Original samples land at indices that are multiples of (zoom+1).
        np.testing.assert_array_equal(out[:: zoom + 1], y)

    def test_single_sample(self):
        out = sinwhisky_resample([3.14], zoom=5)
        np.testing.assert_array_equal(out, [3.14])

    def test_empty(self):
        out = sinwhisky_resample([], zoom=5)
        self.assertEqual(len(out), 0)

    def test_two_samples_linear(self):
        # No interior samples -> no circles -> pure linear interpolation.
        out = sinwhisky_resample([0.0, 4.0], zoom=3)
        np.testing.assert_allclose(out, [0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Analytic exact cases (the algorithm must reproduce these perfectly)
# ---------------------------------------------------------------------------

class TestAnalyticCases(unittest.TestCase):
    def test_constant_signal(self):
        out = sinwhisky_resample([2.5] * 6, zoom=4)
        np.testing.assert_allclose(out, 2.5, atol=1e-9)

    def test_linear_ramp_is_exact(self):
        # Collinear triplets -> circle is rejected -> linear fallback, which
        # reproduces the straight line exactly.
        n, zoom = 7, 5
        y = 2.0 * np.arange(n) - 1.0  # slope 2, intercept -1
        out = sinwhisky_resample(y, zoom=zoom)
        x_out = np.arange(len(out)) / (zoom + 1)
        np.testing.assert_allclose(out, 2.0 * x_out - 1.0, atol=1e-9)

    def test_points_on_circle_are_recovered(self):
        # Samples taken from the upper half of a circle of radius R, spaced by 1
        # on the x-axis. With aperture disabled every fitted circle equals the
        # original one, so the inserted points must land back on the circle.
        R = 5.0
        xs = np.arange(-3, 4)            # -3..3, step 1
        ys = np.sqrt(R * R - xs * xs)    # upper semicircle
        zoom = 4
        out = sinwhisky_resample(ys, zoom=zoom, use_aperture=False, neighbors=1)

        # Map every output index back to its true x position.
        x_out = xs[0] + np.arange(len(out)) / (zoom + 1)
        expected = np.sqrt(R * R - x_out * x_out)
        np.testing.assert_allclose(out, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# 4. Behaviour on real signals (no blow-ups, sane smoothing)
# ---------------------------------------------------------------------------

class TestSignalBehaviour(unittest.TestCase):
    def test_sine_endpoints_and_bounds(self):
        n = 64
        x = np.linspace(0, 4 * math.pi, n)
        y = np.sin(x)
        zoom = 5
        out = sinwhisky_resample(y, zoom=zoom)

        # Endpoints preserved.
        self.assertAlmostEqual(out[0], y[0], places=9)
        self.assertAlmostEqual(out[-1], y[-1], places=9)

        # No wild overshoot: a circle interpolation can overshoot a bit, but
        # must stay well-bounded for a smooth sine.
        self.assertLessEqual(np.max(np.abs(out)), 1.15)

        # The resampled curve must be much smoother than the input: the largest
        # step between consecutive resampled points is far below the input step.
        in_step = np.max(np.abs(np.diff(y)))
        out_step = np.max(np.abs(np.diff(out)))
        self.assertLess(out_step, in_step)

    def test_more_neighbors_stays_bounded(self):
        # neighbors>1 extends circles beyond the 3 circumscribed points
        # (paper, p.7). It must remain finite and bounded.
        n = 48
        y = np.sin(np.linspace(0, 6 * math.pi, n))
        for nb in (1, 2, 3):
            out = sinwhisky_resample(y, zoom=4, neighbors=nb)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertLessEqual(np.max(np.abs(out)), 1.3, f"neighbors={nb}")

    def test_nan_input_does_not_crash(self):
        y = [0.0, 1.0, float("nan"), 1.0, 0.0]
        out = sinwhisky_resample(y, zoom=3)
        self.assertEqual(len(out), (len(y) - 1) * 4 + 1)
        # The NaN original is preserved; surrounding region must not be all-NaN.
        finite = np.isfinite(out)
        self.assertGreater(finite.sum(), len(out) // 2)


# ---------------------------------------------------------------------------
# 5. C <-> Python cross-validation
# ---------------------------------------------------------------------------

def _build_c_cli():
    """Compile sinwhisky_c/sw_cli; return path or None if no compiler."""
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        return None
    out_bin = os.path.join(_C_DIR, "sw_cli")
    src = [os.path.join(_C_DIR, "sinwhisky.c"), os.path.join(_C_DIR, "sw_cli.c")]
    cmd = [cc, "-std=c99", "-O2", *src, "-lm", "-o", out_bin]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    return out_bin


class TestCrossValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _build_c_cli()

    def _run_c(self, samples, zoom, neighbors, use_aperture):
        stdin = "\n".join("%.17g" % v for v in samples)
        res = subprocess.run(
            [self.cli, str(zoom), str(neighbors), str(int(use_aperture))],
            input=stdin, capture_output=True, text=True, check=True,
        )
        return np.array([float(line) for line in res.stdout.split()], dtype=float)

    def test_c_matches_python(self):
        if self.cli is None:
            self.skipTest("no C compiler / build failed")

        rng = np.random.default_rng(42)
        cases = [
            [0.0, 1.0, 0.0, -1.0, 0.0],
            list(np.sin(np.linspace(0, 4 * math.pi, 40))),
            list(2.0 * np.arange(10) - 3.0),                 # ramp
            [5.0] * 8,                                       # constant
            list(rng.standard_normal(50)),                   # noise
        ]
        for samples in cases:
            for zoom in (0, 1, 3, 6):
                for neighbors in (1, 2, 3):
                    for ap in (True, False):
                        c_out = self._run_c(samples, zoom, neighbors, ap)
                        py_out = sinwhisky_resample(
                            samples, zoom=zoom, neighbors=neighbors,
                            use_aperture=ap,
                        )
                        self.assertEqual(len(c_out), len(py_out))
                        np.testing.assert_allclose(
                            c_out, py_out, atol=2e-4, rtol=2e-4,
                            err_msg=f"zoom={zoom} nb={neighbors} ap={ap} "
                                    f"n={len(samples)}",
                        )


# ---------------------------------------------------------------------------
# 6. Circle-based compressor round-trip
# ---------------------------------------------------------------------------

class TestCompressorRoundTrip(unittest.TestCase):
    def test_encode_decode_matches_reconstruct(self):
        x = np.linspace(0, 4 * math.pi, 400)
        y = np.sin(x)
        signal = list(zip(x, y))
        circles = find_optimal_circles(signal, max_error_threshold=0.02)
        self.assertGreater(len(circles), 0)

        direct = reconstruct_signal(circles, x)
        encoded = encode_circles_for_compression(circles)
        roundtrip = decode_circles_from_compression(encoded, x)

        # encode->decode must reproduce the direct reconstruction exactly.
        np.testing.assert_allclose(roundtrip, direct, atol=1e-9)

    def test_reconstruction_is_reasonable(self):
        x = np.linspace(0, 2 * math.pi, 300)
        y = np.sin(x)
        circles = find_optimal_circles(list(zip(x, y)), max_error_threshold=0.01)
        recon = reconstruct_signal(circles, x)
        covered = recon != 0.0
        # The bulk of the signal should be covered and close to the original.
        self.assertGreater(covered.mean(), 0.7)
        err = np.abs(recon[covered] - y[covered])
        self.assertLess(np.mean(err), 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
