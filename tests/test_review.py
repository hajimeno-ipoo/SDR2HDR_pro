from __future__ import annotations

import unittest
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from sdr2hdr.review import (
    compute_clipped_region_luma_mae,
    compute_chroma_preservation_error,
    compute_log_luma_mae,
    compute_temporal_luma_delta,
    save_hdr_exr,
)


class ReviewTests(unittest.TestCase):
    def test_save_hdr_exr_preserves_rgb_primaries_and_pixel_positions(self) -> None:
        # Two different rows catch both channel swaps and interleaved/planar errors.
        primaries = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        ], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "primaries.png"
            output = Path(directory) / "primaries.exr"
            self.assertTrue(cv2.imwrite(str(source), (primaries * 65535)[..., ::-1]))
            save_hdr_exr(str(source), 0.0, str(output), width=3, height=2)
            self.assertGreater(output.stat().st_size, 0)
            decoded = subprocess.run([
                "ffmpeg", "-v", "error", "-i", str(output), "-frames:v", "1",
                # Read the saved half floats without a pixel-format conversion.
                "-pix_fmt", "gbrpf16le", "-f", "rawvideo", "-",
            ], check=True, capture_output=True).stdout
            self.assertEqual(len(decoded), 2 * 3 * 3 * 2)
            planes = np.frombuffer(decoded, dtype="<f2").reshape(3, 2, 3)
            # PQ 1 is 10000 nits, so the unchanged 1000-nit reference yields 10.
            # Both 0 and 10 are exact in EXR half precision: no tolerance is needed.
            for plane, rgb_channel in zip(planes, (1, 2, 0)):
                np.testing.assert_array_equal(plane, primaries[..., rgb_channel] * 10)

    def test_compute_log_luma_mae_identical(self) -> None:
        frame = np.full((16, 16, 3), 0.5, dtype=np.float32)
        mae = compute_log_luma_mae(frame, frame)
        self.assertAlmostEqual(mae, 0.0, places=5)

    def test_compute_log_luma_mae_difference(self) -> None:
        f1 = np.full((16, 16, 3), 0.2, dtype=np.float32)
        f2 = np.full((16, 16, 3), 0.8, dtype=np.float32)
        mae = compute_log_luma_mae(f1, f2)
        self.assertGreater(mae, 0.0)

    def test_compute_clipped_region_luma_mae(self) -> None:
        sdr = np.zeros((10, 10, 3), dtype=np.float32)
        sdr[:5, :5] = 0.95  # clipped region
        
        pred = np.full((10, 10, 3), 0.5, dtype=np.float32)
        target = np.full((10, 10, 3), 1.0, dtype=np.float32)
        mae = compute_clipped_region_luma_mae(pred, target, sdr, clip_threshold=0.9)
        self.assertGreater(mae, 0.0)

    def test_compute_chroma_preservation_error_pure_gain(self) -> None:
        orig = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
        # Scaled by 2.0x gain, chroma ratio is identical
        pred = orig * 2.0
        err = compute_chroma_preservation_error(pred, orig)
        self.assertAlmostEqual(err, 0.0, places=5)

    def test_compute_temporal_luma_delta(self) -> None:
        f1 = np.full((8, 8, 3), 0.2, dtype=np.float32)
        f2 = np.full((8, 8, 3), 0.4, dtype=np.float32)
        f3 = np.full((8, 8, 3), 0.6, dtype=np.float32)
        delta = compute_temporal_luma_delta([f1, f2, f3])
        self.assertAlmostEqual(delta, 0.2, places=4)


if __name__ == "__main__":
    unittest.main()
