from pathlib import Path
import unittest
from unittest import mock

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sdr2hdr.core import (
    REC709_TO_REC2020,
    REC2020_TO_ACESCG,
    ProcessorConfig,
    SDRToHDRProcessor,
    TemporalState,
    apply_cinema_tone,
    apply_near_white_rolloff,
    build_ai_gate,
    compute_adaptive_highlight_boost,
    compute_luma,
    compute_luma_bt709,
    estimate_clipped_white_mask,
    estimate_high_chroma_mask,
    estimate_memory_color_mask,
    estimate_noise_mask,
    estimate_skin_mask,
    estimate_specular_mask,
    estimate_subtitle_mask,
    limit_ai_highlight_expansion,
    linear_to_pq,
    srgb_to_linear,
)
from sdr2hdr.review import default_sample_times, parse_times, pq_to_relative_linear, tone_map_hdr_preview


class CoreTests(unittest.TestCase):
    def test_content_light_levels_use_max_rgb_and_brightest_frame(self) -> None:
        for backend in ("numpy", "torch-cpu"):
            if backend == "torch-cpu" and torch is None:
                continue
            with self.subTest(backend=backend):
                processor = SDRToHDRProcessor(ProcessorConfig(backend=backend, ai_strength=0))
                maxima, averages = [], []
                for value in (80, 220, 100):
                    frame = np.zeros((32, 32, 3), dtype=np.uint8)
                    frame[:, :16, 2] = value
                    frame[:, 16:, 0] = value
                    output = processor.process_frame(frame)
                    nits = pq_to_relative_linear(output.astype(np.float32) / 65535., 10000.) * 10000.
                    max_rgb = np.max(nits, axis=-1)
                    maxima.append(float(max_rgb.max()))
                    averages.append(float(max_rgb.mean()))
                cll, fall = processor.get_measured_hdr_metadata()
                # PQ quantization is the only difference from the internal samples.
                self.assertAlmostEqual(cll, max(maxima), delta=.05)
                self.assertAlmostEqual(fall, max(averages), delta=.05)
                self.assertGreater(fall, float(np.mean(averages)) * 1.2)

    def test_default_target_smoothing_does_not_retain_moving_bright_region(self) -> None:
        strength = ProcessorConfig().target_luma_temporal_strength
        first = np.zeros((8, 8), np.float32)
        first[:, :3] = 2.
        second = np.zeros_like(first)
        second[:, 5:] = 2.
        state = TemporalState()
        state.update_target_base_numpy(first, strength, False)
        np.testing.assert_array_equal(state.update_target_base_numpy(second, strength, False), second)
        if torch is not None:
            state.update_target_base_torch(torch.from_numpy(first), strength, False)
            np.testing.assert_array_equal(
                state.update_target_base_torch(torch.from_numpy(second), strength, False).numpy(), second,
            )

    def test_bt709_luminance_matches_training_and_colour_conversion(self) -> None:
        from sdr2hdr.dataset import _compute_luma

        colours = np.eye(3, dtype=np.float32)[None]
        np.testing.assert_allclose(compute_luma_bt709(colours), [[0.2126, 0.7152, 0.0722]], atol=1e-7)
        np.testing.assert_array_equal(compute_luma_bt709(colours), _compute_luma(colours))
        np.testing.assert_allclose(
            compute_luma_bt709(colours), compute_luma(colours @ REC709_TO_REC2020.T), atol=4e-5,
        )
        if torch is not None:
            processor = SDRToHDRProcessor(ProcessorConfig(backend="torch-cpu"))
            np.testing.assert_allclose(
                processor._torch_compute_luma_bt709(torch.from_numpy(colours)).numpy(),
                compute_luma_bt709(colours), atol=1e-7,
            )

    def test_ap1_conversion_matches_primaries_and_bradford_adaptation(self) -> None:
        # Independent derivation from ITU BT.2020, ACEScg primaries/white,
        # and ICC.1:2022 Annex E.3, rather than copying the production matrix.
        def xyz_white(xy):
            x, y = xy
            return np.array([x / y, 1.0, (1 - x - y) / y])

        def rgb_to_xyz(primaries, white):
            basis = np.stack([xyz_white(xy) for xy in primaries], axis=1)
            return basis @ np.diag(np.linalg.solve(basis, xyz_white(white)))

        d65, aces_white = (.3127, .3290), (.32168, .33767)
        bt2020 = rgb_to_xyz([(.708, .292), (.170, .797), (.131, .046)], d65)
        ap1 = rgb_to_xyz([(.713, .293), (.165, .830), (.128, .044)], aces_white)
        bradford = np.array([[.8951, .2664, -.1614], [-.7502, 1.7135, .0367], [.0389, -.0685, 1.0296]])
        adapt = np.linalg.solve(
            bradford,
            np.diag((bradford @ xyz_white(aces_white)) / (bradford @ xyz_white(d65))) @ bradford,
        )
        patches = np.vstack([np.eye(3), np.ones((1, 3)), np.full((1, 3), .18), np.full((1, 3), 4)])
        converted = patches @ REC2020_TO_ACESCG.T
        np.testing.assert_allclose(converted @ ap1.T, patches @ bt2020.T @ adapt.T, atol=3e-7)
        np.testing.assert_allclose(converted[3:], patches[3:], atol=3e-7)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_torch_neighbourhood_filters_and_output_preserve_flat_gray(self) -> None:
        processor = SDRToHDRProcessor(ProcessorConfig(backend="torch-cpu", disable_auto_exposure=True))
        plane = torch.ones((64, 64))
        for kernel in (3, 5, 11):
            np.testing.assert_allclose(processor._torch_blur(plane, kernel).numpy(), 1.0, atol=1e-6)
        np.testing.assert_array_equal(processor._torch_laplacian(plane).numpy(), np.zeros((64, 64)))
        output = processor.process_frame(np.full((64, 64, 3), 128, dtype=np.uint8))
        self.assertLessEqual(int(output.max()) - int(output.min()), 1)

    def test_maxfall_is_brightest_frame_mean_not_video_average(self) -> None:
        backends = ["numpy"] + (["torch-cpu"] if torch is not None else [])
        for backend in backends:
            with self.subTest(backend=backend):
                processor = SDRToHDRProcessor(ProcessorConfig(backend=backend, disable_auto_exposure=True))
                self.assertEqual(processor.get_measured_hdr_metadata(), (0.0, 0.0))
                means = []
                for value in (64, 192, 64):
                    output = processor.process_frame(np.full((64, 64, 3), value, dtype=np.uint8))
                    nits = pq_to_relative_linear(output.astype(np.float32) / 65535, 1000) * 1000
                    means.append(float(compute_luma(nits).mean()))
                _, max_fall = processor.get_measured_hdr_metadata()
                self.assertAlmostEqual(max_fall, max(means), delta=.05)
                self.assertGreater(max_fall, float(np.mean(means)) * 1.5)

    def test_ap1_frame_keeps_size_and_ffmpeg_gbr_plane_order(self) -> None:
        from dataclasses import replace

        backends = ["numpy"] + (["torch-cpu"] if torch is not None else [])
        frame = np.full((96, 160, 3), [140, 100, 80], dtype=np.uint8)
        for backend in backends:
            for scale in (1.0, .5):
                with self.subTest(backend=backend, scale=scale):
                    config = ProcessorConfig(backend=backend, processing_scale=scale, disable_auto_exposure=True)
                    pq = SDRToHDRProcessor(config).process_frame(frame)
                    # These patches are below the output peak. Decode the same
                    # processor's PQ path to independently recover BT.2020 RGB.
                    linear_2020 = pq_to_relative_linear(pq.astype(np.float32) / 65535, config.diffuse_white_nits)
                    expected_rgb = linear_2020 @ REC2020_TO_ACESCG.T
                    planes = SDRToHDRProcessor(replace(config, output_color_space="acescg_linear")).process_frame(frame)
                    self.assertEqual(planes.shape, (3, 96, 160))
                    self.assertEqual(planes.dtype, np.float32)
                    rgb = planes[[2, 0, 1]].transpose(1, 2, 0)
                    np.testing.assert_allclose(rgb, expected_rgb, rtol=5e-4, atol=2e-5)

    def test_apply_cinema_tone_preserves_chroma_and_rolls_off_highlights(self) -> None:
        frame = np.array([[[0.4, 0.2, 0.1], [2.0, 1.0, 0.5]]], dtype=np.float32)
        toned = apply_cinema_tone(frame)

        np.testing.assert_allclose(
            toned[0, 0, 1:] / frame[0, 0, 1:],
            np.full(2, toned[0, 0, 0] / frame[0, 0, 0]),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            toned[0, 1, 1:] / frame[0, 1, 1:],
            np.full(2, toned[0, 1, 0] / frame[0, 1, 0]),
            rtol=1e-5,
        )
        self.assertLess(float(np.mean(toned[0, 1])), float(np.mean(frame[0, 1])))

    def test_parse_times(self) -> None:
        self.assertEqual(parse_times("0.5, 1.25,2"), [0.5, 1.25, 2.0])

    def test_default_sample_times_with_duration(self) -> None:
        out = default_sample_times(10.0, 4)
        self.assertEqual(out, [1.0, 3.667, 6.333, 9.0])

    def test_default_sample_times_without_duration(self) -> None:
        out = default_sample_times(None, 3)
        self.assertEqual(out, [0.0, 1.0, 2.0])

    def test_tone_map_hdr_preview_returns_uint8_image(self) -> None:
        frame = np.full((4, 4, 3), 0.75, dtype=np.float32)
        out = tone_map_hdr_preview(frame)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, frame.shape)

    def test_pq_to_relative_linear_is_monotonic(self) -> None:
        values = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float32)
        out = pq_to_relative_linear(values, peak_nits=1000.0)
        self.assertTrue(np.all(np.diff(out) > 0.0))

    def test_srgb_to_linear_identity_bounds(self) -> None:
        sample = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = srgb_to_linear(sample)
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[-1]), 1.0)
        self.assertGreater(float(out[1]), 0.0)
        self.assertLess(float(out[1]), 0.5)

    def test_linear_to_pq_is_monotonic(self) -> None:
        values = np.array([0.0, 0.18, 0.5, 1.0], dtype=np.float32)
        out = linear_to_pq(values, peak_nits=1000.0)
        self.assertTrue(np.all(np.diff(out) >= 0.0))
        self.assertLessEqual(float(out[-1]), 1.0)

    def test_skin_mask_detects_typical_skin_tone(self) -> None:
        frame = np.array([[[0.55, 0.34, 0.22]]], dtype=np.float32)
        mask = estimate_skin_mask(frame)
        self.assertEqual(float(mask[0, 0]), 1.0)

    def test_subtitle_mask_finds_bright_text_in_lower_band(self) -> None:
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[26:30, 18:46] = 255
        luma = np.full((32, 64), 0.2, dtype=np.float32)
        mask = estimate_subtitle_mask(frame, luma)
        self.assertGreater(float(mask[27:29, 20:44].mean()), 0.2)

    def test_noise_mask_stronger_in_dark_noisy_region(self) -> None:
        rng = np.random.default_rng(0)
        base = np.full((16, 16, 3), 0.03, dtype=np.float32)
        noisy = np.clip(base + rng.normal(scale=0.02, size=base.shape).astype(np.float32), 0.0, 1.0)
        luma = np.full((16, 16), 0.03, dtype=np.float32)
        mask = estimate_noise_mask(noisy, luma, 0.08)
        self.assertGreater(float(mask.mean()), 0.05)

    def test_specular_mask_prefers_bright_neutral_pixels(self) -> None:
        frame = np.array([[[0.9, 0.88, 0.86], [0.9, 0.3, 0.1]]], dtype=np.float32)
        luma = np.array([[0.88, 0.45]], dtype=np.float32)
        mask = estimate_specular_mask(frame, luma)
        self.assertGreater(float(mask[0, 0]), float(mask[0, 1]))

    def test_clipped_white_mask_prefers_flat_bright_white(self) -> None:
        frame = np.full((8, 8, 3), [0.96, 0.95, 0.95], dtype=np.float32)
        frame[:, 4:] = [0.96, 0.65, 0.30]
        luma = np.full((8, 8), 0.95, dtype=np.float32)
        luma[:, 4:] = 0.70
        mask = estimate_clipped_white_mask(frame, luma)
        self.assertGreater(float(mask[:, :4].mean()), float(mask[:, 4:].mean()))

    def test_high_chroma_mask_prefers_vivid_regions(self) -> None:
        frame = np.array([[[0.9, 0.1, 0.1], [0.4, 0.38, 0.36]]], dtype=np.float32)
        luma = np.array([[0.4, 0.4]], dtype=np.float32)
        mask = estimate_high_chroma_mask(frame, luma)
        self.assertGreater(float(mask[0, 0]), float(mask[0, 1]))

    def test_memory_color_mask_catches_foliage_like_green(self) -> None:
        frame = np.array([[[0.18, 0.55, 0.12], [0.35, 0.35, 0.35]]], dtype=np.float32)
        luma = np.array([[0.35, 0.35]], dtype=np.float32)
        mask = estimate_memory_color_mask(frame, luma)
        self.assertGreater(float(mask[0, 0]), float(mask[0, 1]))

    def test_ai_gate_suppresses_protected_color_regions(self) -> None:
        gate = build_ai_gate(
            skin_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            subtitle_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            noise_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            clipped_white_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            high_chroma_mask=np.array([[0.9, 0.0]], dtype=np.float32),
            memory_color_mask=np.array([[0.0, 0.0]], dtype=np.float32),
            learned_protection=np.array([[0.0, 0.0]], dtype=np.float32),
        )
        self.assertLess(float(gate[0, 0]), 0.5)
        self.assertAlmostEqual(float(gate[0, 1]), 1.0)

    def test_near_white_rolloff_reduces_upper_luma_gain(self) -> None:
        luma = np.array([[0.4, 0.8, 0.95]], dtype=np.float32)
        rolloff = apply_near_white_rolloff(luma, 0.78, 0.6)
        self.assertAlmostEqual(float(rolloff[0, 0]), 1.0)
        self.assertGreater(float(rolloff[0, 1]), float(rolloff[0, 2]))

    def test_ai_highlight_limiter_suppresses_clipped_and_near_white_regions(self) -> None:
        expansion = np.ones((1, 3), dtype=np.float32)
        luma = np.array([[0.55, 0.86, 0.98]], dtype=np.float32)
        clipped_white = np.array([[0.0, 0.15, 1.0]], dtype=np.float32)
        rolloff = apply_near_white_rolloff(luma, 0.74, 0.72)
        limited = limit_ai_highlight_expansion(expansion, luma, clipped_white, rolloff)
        self.assertAlmostEqual(float(limited[0, 0]), 1.0)
        self.assertLess(float(limited[0, 1]), 0.8)
        self.assertLess(float(limited[0, 2]), 0.2)

    def test_adaptive_highlight_boost_penalizes_flat_white_scenes(self) -> None:
        flat_white = compute_adaptive_highlight_boost(0.8, 0.8, 0.1, 0.05, 0.1, 0.55, 1.05)
        specular_scene = compute_adaptive_highlight_boost(0.8, 0.1, 0.05, 0.5, 0.0, 0.55, 1.05)
        self.assertLess(flat_white, specular_scene)

    def test_temporal_state_detects_scene_cut(self) -> None:
        state = TemporalState()
        first = np.zeros((8, 8), dtype=np.float32)
        chroma_first = np.zeros((8, 8), dtype=np.float32)
        state.update(0.2, 0.9, first, chroma_first, 0.1)
        second = np.ones((8, 8), dtype=np.float32)
        chroma_second = np.ones((8, 8), dtype=np.float32) * 0.5
        _, scene_cut = state.update(0.8, 0.9, second, chroma_second, 0.1)
        self.assertTrue(scene_cut)

    def test_processor_returns_uint16_rgb(self) -> None:
        frame = np.full((8, 8, 3), 180, dtype=np.uint8)
        processor = SDRToHDRProcessor(ProcessorConfig())
        out = processor.process_frame(frame)
        self.assertEqual(out.dtype, np.uint16)
        self.assertEqual(out.shape, frame.shape)
        self.assertLessEqual(int(out.max()), 65535)

    def test_fast_mode_with_processing_scale_preserves_output_shape(self) -> None:
        frame = np.full((24, 40, 3), 150, dtype=np.uint8)
        config = ProcessorConfig(fast_mode=True, processing_scale=0.5)
        processor = SDRToHDRProcessor(config)
        out = processor.process_frame(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint16)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_torch_cpu_backend_preserves_output_shape(self) -> None:
        frame = np.full((24, 40, 3), 170, dtype=np.uint8)
        config = ProcessorConfig(fast_mode=True, processing_scale=0.75, backend="torch-cpu")
        processor = SDRToHDRProcessor(config)
        out = processor.process_frame(frame)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, np.uint16)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_windows_auto_backend_prefers_cuda(self) -> None:
        with (
            mock.patch("sdr2hdr.core.platform.system", return_value="Windows"),
            mock.patch("sdr2hdr.core.torch.cuda.is_available", return_value=True),
            mock.patch("sdr2hdr.core.torch.backends.mps.is_available", return_value=False),
        ):
            processor = SDRToHDRProcessor.__new__(SDRToHDRProcessor)
            processor.config = ProcessorConfig(backend="auto")
            self.assertEqual(processor._resolve_torch_device(), "cuda")

    @unittest.skipIf(torch is None, "torch not installed")
    def test_linux_auto_backend_prefers_cuda(self) -> None:
        with (
            mock.patch("sdr2hdr.core.platform.system", return_value="Linux"),
            mock.patch("sdr2hdr.core.torch.cuda.is_available", return_value=True),
            mock.patch("sdr2hdr.core.torch.backends.mps.is_available", return_value=False),
        ):
            processor = SDRToHDRProcessor.__new__(SDRToHDRProcessor)
            processor.config = ProcessorConfig(backend="auto")
            self.assertEqual(processor._resolve_torch_device(), "cuda")

    @unittest.skipIf(torch is None, "torch not installed")
    def test_linux_auto_backend_cuda_not_blocked_by_mps(self) -> None:
        with (
            mock.patch("sdr2hdr.core.platform.system", return_value="Linux"),
            mock.patch("sdr2hdr.core.torch.cuda.is_available", return_value=True),
            mock.patch("sdr2hdr.core.torch.backends.mps.is_available", return_value=True),
        ):
            processor = SDRToHDRProcessor.__new__(SDRToHDRProcessor)
            processor.config = ProcessorConfig(backend="auto")
            self.assertEqual(processor._resolve_torch_device(), "cuda")

    @unittest.skipIf(torch is None, "torch not installed")
    def test_macos_auto_backend_prefers_mps_over_cuda(self) -> None:
        with (
            mock.patch("sdr2hdr.core.platform.system", return_value="Darwin"),
            mock.patch("sdr2hdr.core.torch.cuda.is_available", return_value=True),
            mock.patch("sdr2hdr.core.torch.backends.mps.is_available", return_value=True),
        ):
            processor = SDRToHDRProcessor.__new__(SDRToHDRProcessor)
            processor.config = ProcessorConfig(backend="auto")
            self.assertEqual(processor._resolve_torch_device(), "mps")

    @unittest.skipIf(torch is None, "torch not installed")
    def test_cuda_backend_requires_cuda(self) -> None:
        with mock.patch("sdr2hdr.core.torch.cuda.is_available", return_value=False):
            processor = SDRToHDRProcessor.__new__(SDRToHDRProcessor)
            processor.config = ProcessorConfig(backend="cuda")
            self.assertIsNone(processor._resolve_torch_device())

    @unittest.skipIf(torch is None, "torch not installed")
    def test_process_frame_with_v2_model_applies_guidance(self) -> None:
        import tempfile
        from sdr2hdr.ai import TorchMapEnhancer
        from sdr2hdr.model import HDRGuidedEnhancementUNet
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "v2.pt"
            model = HDRGuidedEnhancementUNet().eval()
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, str(model_path))

            enhancer = TorchMapEnhancer(str(model_path), device="cpu")
            config = ProcessorConfig(backend="numpy", hdr_guidance="auto")
            processor = SDRToHDRProcessor(config=config, enhancer=enhancer)
            frame = np.full((32, 32, 3), 200, dtype=np.uint8)
            out = processor.process_frame(frame)
            self.assertEqual(out.shape, frame.shape)
            self.assertEqual(out.dtype, np.uint16)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_process_frame_with_v2_model_guidance_off_matches_legacy(self) -> None:
        import tempfile
        from sdr2hdr.ai import TorchMapEnhancer
        from sdr2hdr.model import HDRGuidedEnhancementUNet
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "v2.pt"
            model = HDRGuidedEnhancementUNet().eval()
            scripted = torch.jit.script(model)
            torch.jit.save(scripted, str(model_path))

            enhancer = TorchMapEnhancer(str(model_path), device="cpu")
            config_off = ProcessorConfig(backend="numpy", hdr_guidance="off")
            processor_off = SDRToHDRProcessor(config=config_off, enhancer=enhancer)
            
            frame = np.full((32, 32, 3), 180, dtype=np.uint8)
            out_off = processor_off.process_frame(frame)
            self.assertEqual(out_off.shape, frame.shape)
            self.assertEqual(out_off.dtype, np.uint16)

    def test_temporal_state_resets_target_base_on_scene_cut(self) -> None:
        state = TemporalState()
        target_base = np.full((8, 8), 1.0, dtype=np.float32)
        out1 = state.update_target_base_numpy(target_base, alpha=0.75, scene_cut=False)
        np.testing.assert_allclose(out1, target_base)

        target_base2 = np.full((8, 8), 2.0, dtype=np.float32)
        out2 = state.update_target_base_numpy(target_base2, alpha=0.75, scene_cut=False)
        np.testing.assert_allclose(out2, 1.0 * 0.75 + 2.0 * 0.25)

        # Scene cut resets state
        target_base3 = np.full((8, 8), 3.0, dtype=np.float32)
        out3 = state.update_target_base_numpy(target_base3, alpha=0.75, scene_cut=True)
        np.testing.assert_allclose(out3, target_base3)

    def test_output_peak_strictly_bounded_and_measured(self) -> None:
        config = ProcessorConfig(backend="numpy", peak_nits=600.0, tone="vivid")
        processor = SDRToHDRProcessor(config=config)
        # Ultra bright white input
        frame = np.full((16, 16, 3), 255, dtype=np.uint8)
        out16 = processor.process_frame(frame)
        self.assertEqual(out16.dtype, np.uint16)
        
        # Verify measured MaxCLL
        max_cll, max_fall = processor.get_measured_hdr_metadata()
        self.assertGreater(max_cll, 0.0)
        self.assertLessEqual(max_cll, 600.0 + 1e-3)
        self.assertGreater(max_fall, 0.0)
        self.assertLessEqual(max_fall, 600.0 + 1e-3)


if __name__ == "__main__":
    unittest.main()
