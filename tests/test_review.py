from __future__ import annotations

import unittest
import numpy as np

from sdr2hdr.review import (
    compute_clipped_region_luma_mae,
    compute_chroma_preservation_error,
    compute_log_luma_mae,
    compute_temporal_luma_delta,
)


class ReviewTests(unittest.TestCase):
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
