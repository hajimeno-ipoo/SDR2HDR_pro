import unittest
from unittest import mock
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from sdr2hdr.io import (
    VideoInfo, build_audio_output_args, finalize_process, is_interlaced_video,
    open_decoder, open_encoder, require_exr_metadata_tool, stamp_ap1_exr_sequence, restamp_hdr_metadata,
)


class IoTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
    def test_coloured_hdr_content_light_levels_reach_hevc_header(self) -> None:
        import json
        import math
        from sdr2hdr.core import ProcessorConfig, SDRToHDRProcessor

        processor = SDRToHDRProcessor(ProcessorConfig(backend="numpy", ai_strength=0))
        info = VideoInfo(64, 64, 1., 3, "rgb48le", 3., "progressive")
        with tempfile.TemporaryDirectory() as temporary:
            output = str(Path(temporary) / "colour.mp4")
            source = str(Path(temporary) / "source.mkv")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=size=64x64:rate=1:duration=3", "-c:v", "ffv1", source], check=True)
            encoder = open_encoder(output, source, info, 1000., encoder="libx265", x265_preset="ultrafast")
            for level in (80, 220, 100):
                frame = np.zeros((64, 64, 3), np.uint8)
                frame[:, :32, 2] = level
                frame[:, 32:, 0] = level
                encoder.stdin.write(processor.process_frame(frame).tobytes())
            finalize_process(encoder, "encoder")
            cll, fall = (math.ceil(value) for value in processor.get_measured_hdr_metadata())
            restamp_hdr_metadata(output, cll, fall)
            frames = json.loads(subprocess.run([
                "ffprobe", "-v", "error", "-show_frames", "-of", "json", output,
            ], check=True, capture_output=True, text=True).stdout)["frames"]
            self.assertEqual(len(frames), 3)
            light = next(row for row in frames[0]["side_data_list"]
                         if row["side_data_type"] == "Content light level metadata")
            self.assertEqual((light["max_content"], light["max_average"]), (cll, fall))

    @mock.patch("sdr2hdr.io.shutil.which", return_value=None)
    def test_ap1_metadata_requires_official_writer(self, _: mock.Mock) -> None:
        with self.assertRaisesRegex(ValueError, "exrstdattr"):
            require_exr_metadata_tool()

    @unittest.skipUnless(all(shutil.which(tool) for tool in ("ffmpeg", "exrstdattr", "exrheader")),
                         "FFmpeg and OpenEXR tools are required")
    def test_ap1_metadata_preserves_pixels_and_only_tags_written_frames(self) -> None:
        info = VideoInfo(16, 16, 1., 2, "gbrpf32le", 2., "progressive")
        planes = np.full((3, 16, 16), 4., np.float32)
        planes[0] = .25
        planes[1] = .125
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pattern = str(root / "frame_%06d.exr")
            encoder = open_encoder(pattern, "unused", info, 1000., encoder="openexr_acescg")
            encoder.stdin.write(planes.tobytes() * 2)
            finalize_process(encoder, "encoder")
            sentinel = root / "frame_000099.exr"
            sentinel.write_bytes(b"not part of this conversion")
            def decode(path):
                return subprocess.run([
                    "ffmpeg", "-v", "error", "-i", str(path), "-pix_fmt", "gbrpf32le",
                    "-f", "rawvideo", "-",
                ], check=True, capture_output=True).stdout
            original = [decode(root / f"frame_{i:06d}.exr") for i in (1, 2)]
            stamp_ap1_exr_sequence(pattern, 2, 203.)
            for index in (1, 2):
                path = root / f"frame_{index:06d}.exr"
                self.assertEqual(decode(path), original[index-1])
                header = subprocess.run(["exrheader", str(path)], check=True,
                                        capture_output=True, text=True).stdout
                for expected in ("chromaticities", "0.713", "0.32168", "whiteLuminance", "203", "unknown"):
                    self.assertIn(expected, header)
            self.assertEqual(sentinel.read_bytes(), b"not part of this conversion")

    @mock.patch("sdr2hdr.io.require_exr_metadata_tool", return_value="exrstdattr")
    @mock.patch("sdr2hdr.io.subprocess.run", side_effect=subprocess.CalledProcessError(1, "exrstdattr"))
    def test_ap1_metadata_failure_preserves_original(self, _run: mock.Mock, _tool: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame_000001.exr"
            path.write_bytes(b"original")
            with self.assertRaises(subprocess.CalledProcessError):
                stamp_ap1_exr_sequence(str(Path(temporary) / "frame_%06d.exr"), 1, 203.)
            self.assertEqual(path.read_bytes(), b"original")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for EXR round-trip verification")
    def test_ap1_exr_roundtrip_keeps_channels_and_values_above_one(self) -> None:
        from sdr2hdr.core import ProcessorConfig, SDRToHDRProcessor

        frame = np.full((96, 160, 3), [140, 100, 80], dtype=np.uint8)
        planes = SDRToHDRProcessor(ProcessorConfig(
            backend="numpy", processing_scale=.5, output_color_space="acescg_linear",
        )).process_frame(frame)
        # Include known HDR linear values to check half-float encoding itself.
        high_planes = np.full_like(planes, 4.)
        high_planes[0] = 2.
        high_planes[1] = .5
        info = VideoInfo(160, 96, 1., 2, "gbrpf32le", 2., "progressive")
        with tempfile.TemporaryDirectory() as temporary:
            pattern = str(Path(temporary) / "frame_%06d.exr")
            encoder = open_encoder(pattern, "unused", info, 1000., encoder="openexr_acescg")
            encoder.stdin.write(planes.tobytes())
            encoder.stdin.write(high_planes.tobytes())
            encoder.stdin.close()
            finalize_process(encoder, "encoder")
            paths = sorted(Path(temporary).glob("*.exr"))
            self.assertEqual(len(paths), 2)
            for path, expected in zip(paths, (planes, high_planes)):
                result = subprocess.run([
                    "ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                    "-pix_fmt", "gbrpf32le", "-f", "rawvideo", "-",
                ], check=True, capture_output=True)
                decoded = np.frombuffer(result.stdout, np.float32).reshape(3, 96, 160)
                np.testing.assert_allclose(decoded, expected, rtol=1e-3, atol=1e-6)

    def test_is_interlaced_video_detects_progressive(self) -> None:
        info = VideoInfo(1920, 1080, 29.97, None, "yuv420p", 10.0, "progressive")
        self.assertFalse(is_interlaced_video(info))

    def test_is_interlaced_video_detects_interlaced_field_order(self) -> None:
        info = VideoInfo(1920, 1080, 29.97, None, "yuv420p", 10.0, "tt")
        self.assertTrue(is_interlaced_video(info))

    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_decoder_inserts_bwdif_for_interlaced_input(self, popen_mock: mock.Mock) -> None:
        info = VideoInfo(1440, 1080, 29.97, None, "yuv420p", 10.0, "tb")
        open_decoder("input.m2ts", info)
        cmd = popen_mock.call_args.args[0]
        self.assertIn("-vf", cmd)
        self.assertIn("bwdif=mode=send_frame:parity=auto:deint=all", cmd)

    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_decoder_skips_bwdif_for_progressive_input(self, popen_mock: mock.Mock) -> None:
        info = VideoInfo(1920, 1080, 23.976, None, "yuv420p", 10.0, "progressive")
        open_decoder("input.mp4", info)
        cmd = popen_mock.call_args.args[0]
        self.assertNotIn("-vf", cmd)

    @mock.patch("sdr2hdr.io.ffprobe_first_audio_codec", return_value="pcm_bluray")
    def test_build_audio_output_args_transcodes_pcm_bluray_for_mp4(self, _: mock.Mock) -> None:
        self.assertEqual(build_audio_output_args("output.mp4", "input.m2ts"), ["-c:a", "aac", "-b:a", "192k"])

    @mock.patch("sdr2hdr.io.ffprobe_first_audio_codec", return_value="aac")
    def test_build_audio_output_args_copies_supported_mp4_audio(self, _: mock.Mock) -> None:
        self.assertEqual(build_audio_output_args("output.mp4", "input.mp4"), ["-c:a", "copy"])

    @mock.patch("sdr2hdr.io.ffprobe_first_audio_codec", return_value=None)
    def test_build_audio_output_args_handles_missing_audio(self, _: mock.Mock) -> None:
        self.assertEqual(build_audio_output_args("output.mp4", "input.mp4"), [])

    @mock.patch("sdr2hdr.io.build_audio_output_args", return_value=["-c:a", "copy"])
    @mock.patch("sdr2hdr.io.is_videotoolbox_available", return_value=False)
    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_encoder_builds_software_prores_4444_command(
        self,
        popen_mock: mock.Mock,
        _: mock.Mock,
        __: mock.Mock,
    ) -> None:
        info = VideoInfo(1920, 1080, 24.0, None, "bgr24", 1.0, "progressive")
        open_encoder("output.mov", "input.mp4", info, 1000.0, encoder="prores_4444")
        command = popen_mock.call_args.args[0]
        self.assertIn("prores_ks", command)
        self.assertIn("yuv444p10le", command)
        self.assertIn("-profile:v", command)
        self.assertIn("4", command)
        self.assertIn("+write_colr", command)

    @mock.patch("sdr2hdr.io.is_prores_4444_xq_available", return_value=True)
    @mock.patch("sdr2hdr.io.is_videotoolbox_available", return_value=True)
    @mock.patch("sdr2hdr.io.build_audio_output_args", return_value=[])
    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_encoder_builds_prores_4444_xq_12bit_command(
        self,
        popen_mock: mock.Mock,
        _: mock.Mock,
        __: mock.Mock,
        ___: mock.Mock,
    ) -> None:
        info = VideoInfo(1920, 1080, 24.0, None, "bgr24", 1.0, "progressive")
        open_encoder("output.mov", "input.mp4", info, 1000.0, encoder="prores_4444_xq")
        command = popen_mock.call_args.args[0]
        self.assertIn("prores_videotoolbox", command)
        self.assertIn("p416le", command)
        self.assertIn("5", command)

    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_encoder_builds_openexr_sequence_command(self, popen_mock: mock.Mock) -> None:
        info = VideoInfo(1920, 1080, 24.0, None, "bgr24", 1.0, "progressive")
        open_encoder("output_hdr_%06d.exr", "input.mp4", info, 1000.0, encoder="openexr")
        command = popen_mock.call_args.args[0]
        self.assertIn("-f", command)
        self.assertIn("image2", command)
        self.assertIn("exr", command)
        self.assertNotIn("-map", command)

    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_encoder_builds_acescg_openexr_float_command(self, popen_mock: mock.Mock) -> None:
        info = VideoInfo(1920, 1080, 24.0, None, "bgr24", 1.0, "progressive")
        open_encoder("output_hdr_%06d.exr", "input.mp4", info, 1000.0, encoder="openexr_acescg")
        command = popen_mock.call_args.args[0]
        self.assertEqual(command[command.index("-pix_fmt") + 1], "gbrpf32le")
        self.assertIn("ACEScg", " ".join(command))
        self.assertNotIn("-map", command)


if __name__ == "__main__":
    unittest.main()
