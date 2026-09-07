import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sdr2hdr.ai import TorchMapEnhancer
from sdr2hdr.dataset import HDRSDRPairDataset, derive_target_maps
from sdr2hdr.model import EnhancementUNet, HDRGuidedEnhancementUNet


@unittest.skipIf(torch is None, "torch not installed")
class ModelDatasetTests(unittest.TestCase):
    def test_model_returns_expected_shape(self) -> None:
        model = EnhancementUNet()
        output = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 3, 64, 64))
        self.assertGreaterEqual(float(output.detach().min()), -1.0)
        self.assertLessEqual(float(output.detach().max()), 1.0)

    def test_model_handles_odd_input_sizes(self) -> None:
        model = EnhancementUNet()
        output = model(torch.rand(1, 3, 611, 917))
        self.assertEqual(tuple(output.shape), (1, 3, 611, 917))

    def test_v2_model_returns_expected_5ch_shape_and_bounds(self) -> None:
        model = HDRGuidedEnhancementUNet()
        output = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 5, 64, 64))
        # ch 0..2 maps (-1..1)
        self.assertGreaterEqual(float(output[:, :3].detach().min()), -1.0)
        self.assertLessEqual(float(output[:, :3].detach().max()), 1.0)
        # ch 3 target_luma_rel (0..1)
        self.assertGreaterEqual(float(output[:, 3].detach().min()), 0.0)
        self.assertLessEqual(float(output[:, 3].detach().max()), 1.0)
        # ch 4 reconstruction_confidence (0..1)
        self.assertGreaterEqual(float(output[:, 4].detach().min()), 0.0)
        self.assertLessEqual(float(output[:, 4].detach().max()), 1.0)

    def test_v2_model_handles_odd_input_sizes(self) -> None:
        model = HDRGuidedEnhancementUNet()
        output = model(torch.rand(1, 3, 611, 917))
        self.assertEqual(tuple(output.shape), (1, 5, 611, 917))

    def test_derive_target_maps_returns_bounded_maps_and_v2_targets(self) -> None:
        sdr = np.full((32, 32, 3), 0.25, dtype=np.float32)
        hdr = np.full((32, 32, 3), 0.45, dtype=np.float32)
        targets = derive_target_maps(sdr, hdr)
        self.assertEqual(targets.expansion.shape, (32, 32))
        self.assertEqual(targets.contrast.shape, (32, 32))
        self.assertEqual(targets.protection.shape, (32, 32))
        self.assertTrue(np.all(targets.expansion >= -1.0))
        self.assertTrue(np.all(targets.expansion <= 1.0))
        self.assertTrue(np.all(targets.protection >= -1.0))
        self.assertTrue(np.all(targets.protection <= 1.0))
        # v2 targets
        self.assertIsNotNone(targets.target_luma_rel)
        self.assertIsNotNone(targets.reconstruction_target)
        self.assertIsNotNone(targets.identity_mask)
        self.assertEqual(targets.target_luma_rel.shape, (32, 32))
        self.assertTrue(np.all(targets.target_luma_rel >= 0.0))
        self.assertTrue(np.all(targets.target_luma_rel <= 1.0))

    def test_derive_target_maps_reduces_expansion_for_vivid_memory_colors(self) -> None:
        sdr = np.full((16, 16, 3), [0.18, 0.55, 0.12], dtype=np.float32)
        hdr = np.full((16, 16, 3), [0.35, 0.75, 0.22], dtype=np.float32)
        targets = derive_target_maps(sdr, hdr)
        self.assertGreater(float(targets.protection.mean()), 0.0)
        self.assertLess(float(targets.expansion.mean()), 0.25)

    def test_dataset_loads_npz_and_returns_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample.npz"
            np.savez_compressed(
                sample_path,
                sdr_linear=np.full((48, 48, 3), 0.2, dtype=np.float32),
                hdr_linear=np.full((48, 48, 3), 0.4, dtype=np.float32),
                peak_nits=np.float32(1000.0),
            )
            dataset = HDRSDRPairDataset(temp_dir, patch_size=32, training=False)
            sample = dataset[0]
            self.assertEqual(tuple(sample["sdr_linear"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["target_maps"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["clip_mask"].shape), (1, 32, 32))
            self.assertEqual(tuple(sample["target_luma_rel"].shape), (1, 32, 32))
            self.assertEqual(tuple(sample["reconstruction_target"].shape), (1, 32, 32))
            self.assertEqual(tuple(sample["identity_mask"].shape), (1, 32, 32))
            self.assertIn("peak_nits", sample)
            self.assertEqual(float(sample["peak_nits"]), 1000.0)
            self.assertEqual(dataset.content_names, ["sample"])

    def test_dataset_reads_content_name_metadata_for_group_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, content_name in enumerate(("scene_a", "scene_b")):
                np.savez_compressed(
                    Path(temp_dir) / f"sample_{index}.npz",
                    sdr_linear=np.full((16, 16, 3), 0.2, dtype=np.float32),
                    hdr_linear=np.full((16, 16, 3), 0.4, dtype=np.float32),
                    content_name=np.asarray(content_name),
                )

            dataset = HDRSDRPairDataset(temp_dir, patch_size=8, training=False)
            self.assertEqual(dataset.content_names, ["scene_a", "scene_b"])

    def test_validation_dataset_center_crops_large_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = Path(temp_dir) / "sample.npz"
            np.savez_compressed(
                sample_path,
                sdr_linear=np.full((96, 128, 3), 0.2, dtype=np.float32),
                hdr_linear=np.full((96, 128, 3), 0.4, dtype=np.float32),
            )
            dataset = HDRSDRPairDataset(temp_dir, patch_size=32, training=False)
            sample = dataset[0]
            self.assertEqual(tuple(sample["sdr_linear"].shape), (3, 32, 32))
            self.assertEqual(tuple(sample["target_maps"].shape), (3, 32, 32))

    def test_torch_map_enhancer_loads_scripted_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "enhancer.pt"
            model = EnhancementUNet().eval()
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, str(model_path))
            enhancer = TorchMapEnhancer(str(model_path), device="cpu")
            frame = np.full((16, 16, 3), 0.25, dtype=np.float32)
            maps = enhancer.estimate(frame)
            self.assertEqual(maps.expansion.shape, (16, 16))
            self.assertEqual(maps.contrast.shape, (16, 16))
            self.assertEqual(maps.protection.shape, (16, 16))

    def test_torch_map_enhancer_preserves_output_shape_with_low_res_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "enhancer.pt"
            model = EnhancementUNet().eval()
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, str(model_path))
            enhancer = TorchMapEnhancer(str(model_path), device="cpu", inference_scale=0.5)
            frame = np.full((73, 121, 3), 0.25, dtype=np.float32)
            maps = enhancer.estimate(frame)
            self.assertEqual(maps.expansion.shape, (73, 121))
            self.assertTrue(np.all(maps.expansion >= 0.0))
            self.assertTrue(np.all(maps.expansion <= 1.0))


if __name__ == "__main__":
    unittest.main()
