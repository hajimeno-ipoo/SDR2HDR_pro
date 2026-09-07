import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from sdr2hdr.app import (
    CancelToken,
    ConversionCallbacks,
    ConversionRequest,
    build_request_config,
    build_output_path,
    default_encoder_for_platform,
    is_hardware_encoder_failure,
    is_videotoolbox_failure,
    resolve_model_device,
    resolve_model_backend,
    validate_request,
    run_conversion,
)
from sdr2hdr.ai import HeuristicEnhancer
from sdr2hdr.io import has_expected_hdr_metadata


class AppTests(unittest.TestCase):
    @mock.patch("sdr2hdr.app.ffprobe_video")
    @mock.patch("sdr2hdr.io.shutil.which", return_value=None)
    def test_ap1_missing_metadata_writer_fails_before_decoding(self, _which, probe) -> None:
        with self.assertRaisesRegex(ValueError, "exrstdattr"):
            run_conversion(ConversionRequest("input.mp4", "out_%06d.exr", encoder="openexr_acescg"))
        probe.assert_not_called()

    @unittest.skipUnless(all(shutil.which(tool) for tool in ("ffmpeg", "exrstdattr", "exrheader")),
                         "FFmpeg and OpenEXR tools are required")
    def test_completed_ap1_sequence_contains_color_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.mkv"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=2",
                "-frames:v", "2", "-c:v", "ffv1", str(source),
            ], check=True)
            request = ConversionRequest(str(source), str(root / "frame_%06d.exr"),
                                        encoder="openexr_acescg", preset="cinema", backend="numpy")
            with mock.patch("sdr2hdr.app.build_enhancer", return_value=HeuristicEnhancer()):
                result = run_conversion(request)
            self.assertEqual(result.processed_frames, 2)
            for path in sorted(root.glob("*.exr")):
                header = subprocess.run(["exrheader", str(path)], check=True,
                                        capture_output=True, text=True).stdout
                self.assertIn("chromaticities", header)
                self.assertIn("whiteLuminance (type float): 203", header)

    def test_build_output_path_adds_hdr_suffix(self) -> None:
        self.assertEqual(build_output_path("/tmp/example.mp4"), str(Path("/tmp/example_hdr.mp4")))

    def test_build_output_path_converts_transport_stream_extensions_to_mp4(self) -> None:
        self.assertEqual(build_output_path("/tmp/example.m2ts"), str(Path("/tmp/example_hdr.mp4")))
        self.assertEqual(build_output_path("/tmp/example.ts"), str(Path("/tmp/example_hdr.mp4")))

    def test_build_output_path_uses_mov_for_prores(self) -> None:
        self.assertEqual(
            build_output_path("/tmp/example.mp4", encoder="prores_422hq"),
            str(Path("/tmp/example_hdr.mov")),
        )

    def test_build_output_path_uses_mov_for_prores_xq(self) -> None:
        self.assertEqual(
            build_output_path("/tmp/example.mp4", encoder="prores_4444_xq"),
            str(Path("/tmp/example_hdr.mov")),
        )

    def test_build_output_path_uses_sequence_pattern_for_openexr(self) -> None:
        self.assertEqual(
            build_output_path("/tmp/example.mp4", encoder="openexr"),
            str(Path("/tmp/example_hdr_%06d.exr")),
        )

    def test_build_output_path_uses_sequence_pattern_for_acescg_openexr(self) -> None:
        self.assertEqual(
            build_output_path("/tmp/example.mp4", encoder="openexr_acescg"),
            str(Path("/tmp/example_hdr_%06d.exr")),
        )

    def test_acescg_encoder_selects_linear_output_color_space(self) -> None:
        request = ConversionRequest(
            input_path="/tmp/input.mp4",
            output_path="/tmp/output_%06d.exr",
            encoder="openexr_acescg",
            preset="cinema",
        )
        config, _, _ = build_request_config(request)
        self.assertEqual(config.output_color_space, "acescg_linear")

    def test_detects_videotoolbox_failure_message(self) -> None:
        self.assertTrue(is_videotoolbox_failure("Error: cannot create compression session: -12908"))
        self.assertTrue(is_videotoolbox_failure("hevc_videotoolbox failed"))
        self.assertFalse(is_videotoolbox_failure("generic libx265 failure"))

    def test_detects_hardware_encoder_failure_message(self) -> None:
        self.assertTrue(is_hardware_encoder_failure("hevc_videotoolbox failed"))
        self.assertTrue(is_hardware_encoder_failure("OpenEncodeSessionEx failed: unsupported device"))
        self.assertTrue(is_hardware_encoder_failure("Cannot load nvcuda.dll"))
        self.assertFalse(is_hardware_encoder_failure("generic libx265 failure"))

    def test_default_encoder_for_platform(self) -> None:
        self.assertEqual(default_encoder_for_platform("Darwin"), "hevc_videotoolbox")
        self.assertEqual(default_encoder_for_platform("Windows"), "hevc_nvenc")
        self.assertEqual(default_encoder_for_platform("Linux"), "libx265")

    def test_portrait_uses_stronger_default_ai_strength_when_model_path_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            model_path = Path(temp_dir) / "model.pt"
            input_path.write_bytes(b"")
            model_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                model_path=str(model_path),
            )
            config, _, _ = build_request_config(request)
            self.assertEqual(config.ai_strength, 0.25)

    def test_natural_uses_natural_default_ai_strength_when_model_path_is_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            model_path = Path(temp_dir) / "model.pt"
            input_path.write_bytes(b"")
            model_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="natural",
                model_path=str(model_path),
            )
            config, _, _ = build_request_config(request)
            self.assertEqual(config.ai_strength, 0.15)

    def test_validate_request_requires_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            input_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                model_path=None,
            )
            with self.assertRaises(ValueError):
                validate_request(request)

    def test_validate_request_rejects_non_pt_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            model_path = Path(temp_dir) / "model.onnx"
            input_path.write_bytes(b"")
            model_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                backend="numpy",
                model_path=str(model_path),
            )
            with self.assertRaises(ValueError):
                validate_request(request)

    def test_validate_request_rejects_missing_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "in.mp4"
            input_path.write_bytes(b"")
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(Path(temp_dir) / "out.mp4"),
                preset="portrait",
                model_path=str(Path(temp_dir) / "missing.pt"),
            )
            with self.assertRaises(ValueError):
                validate_request(request)

    def test_resolve_model_device_uses_backend_resolved_device_for_auto(self) -> None:
        request = ConversionRequest(input_path="/tmp/in.mp4", output_path="/tmp/out.mp4", device="auto")
        self.assertEqual(resolve_model_device(request, "mps"), "mps")
        self.assertEqual(resolve_model_device(request, None), "cpu")

    def test_resolve_model_backend_uses_torch_device_for_auto(self) -> None:
        request = ConversionRequest(
            input_path="/tmp/in.mp4",
            output_path="/tmp/out.mp4",
            backend="auto",
            model_path="model.pt",
        )
        self.assertEqual(resolve_model_backend(request, "mps"), "mps")
        self.assertEqual(resolve_model_backend(request, None), "numpy")

    def test_run_conversion_respects_cancel_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            token = CancelToken()
            token.cancel()
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                model_path=str(model_path),
            )
            with mock.patch("sdr2hdr.app.TorchMapEnhancer") as enhancer_cls:
                enhancer_cls.return_value = HeuristicEnhancer()
                result = run_conversion(request, callbacks=ConversionCallbacks(), cancel_token=token)
            self.assertTrue(result.cancelled)
            self.assertEqual(result.processed_frames, 0)
            self.assertFalse(output_path.exists())

    def test_cancel_keeps_partial_output_after_some_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            token = CancelToken()
            events: list[int] = []

            def on_progress(processed: int, total: int | None, fps: float | None) -> None:
                events.append(processed)
                if processed >= 2:
                    token.cancel()

            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                model_path=str(model_path),
            )
            with mock.patch("sdr2hdr.app.TorchMapEnhancer") as enhancer_cls:
                enhancer_cls.return_value = HeuristicEnhancer()
                result = run_conversion(
                    request,
                    callbacks=ConversionCallbacks(on_progress=on_progress),
                    cancel_token=token,
                )
            self.assertTrue(result.cancelled)
            self.assertGreaterEqual(result.processed_frames, 2)
            self.assertTrue(output_path.exists())
            self.assertTrue(has_expected_hdr_metadata(str(output_path)))

    def test_cancel_can_drop_partial_output_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            token = CancelToken()
            token.cancel()
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                model_path=str(model_path),
                keep_partial_output_on_cancel=False,
            )
            with mock.patch("sdr2hdr.app.TorchMapEnhancer") as enhancer_cls:
                enhancer_cls.return_value = HeuristicEnhancer()
                result = run_conversion(request, callbacks=ConversionCallbacks(), cancel_token=token)
            self.assertTrue(result.cancelled)
            self.assertFalse(output_path.exists())

    def test_completed_output_has_expected_hdr_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            model_path.write_bytes(b"placeholder")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                model_path=str(model_path),
                max_frames=12,
            )
            with mock.patch("sdr2hdr.app.TorchMapEnhancer") as enhancer_cls:
                enhancer_cls.return_value = HeuristicEnhancer()
                result = run_conversion(request, callbacks=ConversionCallbacks())
            self.assertFalse(result.cancelled)
            self.assertTrue(has_expected_hdr_metadata(str(output_path)))

    def test_validate_request_checks_guidance_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"
            input_path.write_bytes(b"dummy")
            model_path.write_bytes(b"dummy")

            # Invalid option
            req_invalid = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                model_path=str(model_path),
                hdr_guidance="invalid_option",
            )
            with self.assertRaises(ValueError):
                validate_request(req_invalid)

            # Invalid strength
            req_invalid_strength = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                model_path=str(model_path),
                luminance_guidance_strength=1.5,
            )
            with self.assertRaises(ValueError):
                validate_request(req_invalid_strength)

    def test_build_request_config_propagates_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            model_path = Path(temp_dir) / "model.pt"

            req = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                model_path=str(model_path),
                hdr_guidance="off",
                luminance_guidance_strength=0.85,
                reconstruction_strength=0.45,
            )
            config, _, _ = build_request_config(req)
            self.assertEqual(config.hdr_guidance, "off")
            self.assertEqual(config.luminance_guidance_strength, 0.85)
            self.assertEqual(config.reconstruction_strength, 0.45)


if __name__ == "__main__":
    unittest.main()
