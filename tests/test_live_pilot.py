import unittest

import numpy as np

from scripts.evaluate_model import reference_bt2020
from scripts.prepare_live_pilot import partition_contents
from sdr2hdr.dataset import derive_target_maps


class LivePilotTests(unittest.TestCase):
    def test_partition_is_reproducible_disjoint_and_complete(self):
        names = [f'content_{i}' for i in range(31)]
        split = partition_contents(names)
        self.assertEqual(split, partition_contents(list(reversed(names))))
        self.assertEqual([len(split[k]) for k in ('train', 'validation', 'test')], [21, 5, 5])
        all_names = sum(split.values(), [])
        self.assertEqual(set(all_names), set(names))
        self.assertEqual(len(all_names), len(set(all_names)))

    def test_working_reference_uses_matching_luminance_after_color_conversion(self):
        rgb709 = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
        rgb2020 = reference_bt2020({'hdr_linear': rgb709})
        luminance = rgb2020 @ np.array([.2627, .6780, .0593], dtype=np.float32)
        np.testing.assert_allclose(luminance, [[.2126, .7152, .0722]], atol=1e-4)

    def test_original_wide_gamut_reference_is_preferred(self):
        original = np.array([[[0, 1.2, 0]]], dtype=np.float32)
        actual = reference_bt2020({'hdr_2020_linear': original, 'hdr_linear': np.zeros_like(original)})
        np.testing.assert_array_equal(actual, original)

    def test_training_target_uses_bt709_luminance(self):
        sdr = np.full((16, 16, 3), .2, dtype=np.float32)
        hdr = np.zeros_like(sdr)
        hdr[..., 0] = 1
        target = derive_target_maps(sdr, hdr)
        np.testing.assert_allclose(target.target_luma_rel, .2126)


if __name__ == '__main__':
    unittest.main()
