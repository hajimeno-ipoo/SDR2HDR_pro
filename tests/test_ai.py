import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sdr2hdr.ai import (
    EnhancementGuidance,
    HeuristicEnhancer,
    TorchMapEnhancer,
    estimate_heuristic_maps,
    parse_enhancement_output,
)
from sdr2hdr.hdr_guidance import (
    blend_target_luma,
    blend_target_luma_torch,
    build_reconstruction_gate,
    build_reconstruction_gate_torch,
    clamp_to_peak,
    clamp_to_peak_torch,
    linear_nits_to_pq,
    linear_nits_to_pq_torch,
    resolve_anchor_nits,
    target_rel_to_linear,
    target_rel_to_linear_torch,
)
from sdr2hdr.model import EnhancementUNet, HDRGuidedEnhancementUNet


@unittest.skipIf(torch is None, "torch not installed")
class AITests(unittest.TestCase):
    def test_heuristics_use_bt709_luminance_and_preserve_flat_regions(self) -> None:
        enhancer = TorchMapEnhancer.__new__(TorchMapEnhancer)
        for rgb in ([0., 1., 0.], [.7, .7, .7]):
            frame = np.full((32, 32, 3), rgb, dtype=np.float32)
            expected_expansion = np.clip((np.dot(rgb, [.2126, .7152, .0722]) - .55) / .45, 0, 1)
            maps = estimate_heuristic_maps(frame)
            expansion, contrast, _ = enhancer._heuristic_maps_torch(torch.from_numpy(frame))
            np.testing.assert_allclose(maps.expansion, expected_expansion, atol=1e-6)
            np.testing.assert_allclose(expansion.numpy(), expected_expansion, atol=1e-6)
            np.testing.assert_allclose(contrast.numpy(), 0, atol=1e-5)

    def test_parse_enhancement_output_3ch(self) -> None:
        tensor = torch.randn(1, 3, 32, 32)
        exp, cont, prot, target_luma, recon_conf = parse_enhancement_output(tensor)
        self.assertEqual(exp.shape, (1, 32, 32))
        self.assertEqual(cont.shape, (1, 32, 32))
        self.assertEqual(prot.shape, (1, 32, 32))
        self.assertIsNone(target_luma)
        self.assertIsNone(recon_conf)

    def test_parse_enhancement_output_5ch(self) -> None:
        tensor = torch.randn(1, 5, 32, 32)
        exp, cont, prot, target_luma, recon_conf = parse_enhancement_output(tensor)
        self.assertEqual(exp.shape, (1, 32, 32))
        self.assertEqual(cont.shape, (1, 32, 32))
        self.assertEqual(prot.shape, (1, 32, 32))
        self.assertIsNotNone(target_luma)
        self.assertIsNotNone(recon_conf)
        self.assertEqual(target_luma.shape, (1, 32, 32))
        self.assertEqual(recon_conf.shape, (1, 32, 32))

    def test_parse_enhancement_output_invalid_channels(self) -> None:
        tensor_4ch = torch.randn(1, 4, 32, 32)
        with self.assertRaises(RuntimeError):
            parse_enhancement_output(tensor_4ch)

        tensor_6ch = torch.randn(1, 6, 32, 32)
        with self.assertRaises(RuntimeError):
            parse_enhancement_output(tensor_6ch)

    def test_heuristic_enhancer_guidance_has_no_hdr(self) -> None:
        enhancer = HeuristicEnhancer()
        frame = np.full((16, 16, 3), 0.5, dtype=np.float32)
        guidance = enhancer.estimate_guidance(frame)
        self.assertFalse(guidance.has_hdr_guidance)
        self.assertIsNone(guidance.target_luma_rel)
        self.assertIsNone(guidance.reconstruction_confidence)

    def test_torch_map_enhancer_v2_model_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "v2_model.pt"
            model = HDRGuidedEnhancementUNet().eval()
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, str(model_path))

            enhancer = TorchMapEnhancer(str(model_path), device="cpu")
            frame = np.full((16, 16, 3), 0.3, dtype=np.float32)
            guidance = enhancer.estimate_guidance(frame)
            self.assertTrue(guidance.has_hdr_guidance)
            self.assertEqual(guidance.expansion.shape, (16, 16))
            self.assertEqual(guidance.target_luma_rel.shape, (16, 16))
            self.assertEqual(guidance.reconstruction_confidence.shape, (16, 16))
            self.assertTrue(np.all(guidance.target_luma_rel >= 0.0))
            self.assertTrue(np.all(guidance.target_luma_rel <= 1.0))
            self.assertTrue(np.all(guidance.reconstruction_confidence >= 0.0))
            self.assertTrue(np.all(guidance.reconstruction_confidence <= 1.0))

    def test_torch_map_enhancer_legacy_model_backward_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "legacy_model.pt"
            model = EnhancementUNet().eval()
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, str(model_path))

            enhancer = TorchMapEnhancer(str(model_path), device="cpu")
            frame = np.full((16, 16, 3), 0.3, dtype=np.float32)
            guidance = enhancer.estimate_guidance(frame)
            self.assertFalse(guidance.has_hdr_guidance)
            self.assertIsNone(guidance.target_luma_rel)
            self.assertIsNone(guidance.reconstruction_confidence)

            # Test legacy estimate() returning EnhancementMaps
            maps = enhancer.estimate(frame)
            self.assertEqual(maps.expansion.shape, (16, 16))
            self.assertEqual(maps.contrast.shape, (16, 16))
            self.assertEqual(maps.protection.shape, (16, 16))


class HDRGuidanceMathTests(unittest.TestCase):
    def test_resolve_anchor_nits(self) -> None:
        self.assertEqual(resolve_anchor_nits("reference", 1000.0), 203.0)
        self.assertEqual(resolve_anchor_nits("vivid", 1000.0), 1000.0)

    def test_target_rel_to_linear(self) -> None:
        rel = np.array([0.203, 1.0], dtype=np.float32)
        linear = target_rel_to_linear(rel, peak_nits=1000.0, anchor_nits=203.0)
        np.testing.assert_allclose(linear[0], 1.0, atol=1e-3)
        np.testing.assert_allclose(linear[1], 1000.0 / 203.0, atol=1e-3)

    def test_reconstruction_gate_bounds_and_protection(self) -> None:
        clipped = np.ones((10, 10), dtype=np.float32)
        confidence = np.ones((10, 10), dtype=np.float32)
        protection = np.zeros((10, 10), dtype=np.float32)
        gate = build_reconstruction_gate(clipped, confidence, 0.6, protection)
        np.testing.assert_allclose(gate, 0.6, atol=1e-4)

        # Full protection turns gate off
        full_prot = np.ones((10, 10), dtype=np.float32)
        gate_protected = build_reconstruction_gate(clipped, confidence, 0.6, full_prot)
        np.testing.assert_allclose(gate_protected, 0.0, atol=1e-4)

    def test_blend_target_luma(self) -> None:
        legacy = np.full((5, 5), 1.0, dtype=np.float32)
        model = np.full((5, 5), 3.0, dtype=np.float32)
        blended = blend_target_luma(legacy, model, 0.5)
        np.testing.assert_allclose(blended, 2.0, atol=1e-4)

    def test_clamp_to_peak(self) -> None:
        nits = np.array([500.0, 1200.0], dtype=np.float32)
        clamped = clamp_to_peak(nits, 1000.0)
        self.assertEqual(clamped[0], 500.0)
        self.assertEqual(clamped[1], 1000.0)

    def test_linear_nits_to_pq(self) -> None:
        # 10000 nit -> 1.0
        pq_max = linear_nits_to_pq(np.array([10000.0], dtype=np.float32))
        np.testing.assert_allclose(pq_max[0], 1.0, atol=1e-3)
        # 0 nit -> 0.0
        pq_min = linear_nits_to_pq(np.array([0.0], dtype=np.float32))
        np.testing.assert_allclose(pq_min[0], 0.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
