"""Decode still formats missing from libmpv without reducing HDR to SDR.

Only the HDR member of a completed application job may use the producer's
TIFF colour definition or the application's Ultra HDR encoding contract.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .io import require_ultrahdr_tool


def _run(command, *, data=None):
    result = subprocess.run(command, input=data, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace") or
                           result.stdout.decode("utf-8", errors="replace"))
    if not result.stdout and command[0] == "ffmpeg":
        raise RuntimeError("比較用の画像を復号できませんでした")
    return result.stdout


def prepare_image(path: str, info: dict, *, app_hdr_output: bool = False) -> dict:
    """Return original metadata plus a lossless in-memory PNG when necessary."""
    info = dict(info)
    extension = Path(path).suffix.lower()
    if extension == ".jxl" or (app_hdr_output and extension == ".avif"):
        # Use the same FFmpeg/libjxl already used for export. Homebrew libmpv's
        # separate FFmpeg build has no JPEG XL decoder.
        command = ["ffmpeg", "-v", "error", "-i", path, "-map", "0:v:0", "-frames:v", "1"]
        if extension == ".avif":
            # Decode YCbCr to RGB at 16-bit precision before the floating-point
            # display stage; avoid libmpv's lower-precision YCbCr intermediates.
            decode_filter = "zscale=matrix=gbr:range=full,format=gbrp16le"
            if int(info["width"]) % 2 or int(info["height"]) % 2:
                # Expand odd-sized 4:2:0 planes before zscale's RGB conversion.
                decode_filter = (
                    "scale=iw:ih:in_range=limited:out_range=limited,format=yuv444p16le,"
                    + decode_filter
                )
            command += ["-vf", decode_filter]
        info["image_bytes"] = _run(command + [
            "-c:v", "png", "-pix_fmt", "rgb48be", "-f", "image2pipe", "-",
        ])
    elif app_hdr_output and extension in {".tif", ".tiff"}:
        # save_image_hdr stores full-range BT.2020/PQ RGB16 in TIFF, whose
        # FFmpeg encoder cannot signal this. This is producer knowledge, not
        # a guess applied to arbitrary TIFF inputs or other HDR formats.
        info["image_bytes"] = _run([
            "ffmpeg", "-v", "error", "-i", path, "-frames:v", "1",
            "-vf", "setparams=range=full:color_primaries=bt2020:color_trc=smpte2084:colorspace=gbr",
            "-c:v", "png", "-pix_fmt", "rgb48be", "-f", "image2pipe", "-",
        ])
        info.update(color_primaries="bt2020", color_transfer="smpte2084",
                    color_source="変換設定", bit_depth=16)
    elif app_hdr_output and extension in {".jpg", ".jpeg"}:
        # Our encoder uses BT.2100 HDR intent and an alternate-colour-space
        # gain map. Its official decoder restores BT.2100/PQ RGBA1010102.
        with tempfile.TemporaryDirectory(prefix="sdr2hdr-compare-") as directory:
            raw = Path(directory) / "hdr.raw"
            _run([require_ultrahdr_tool(), "-m", "1", "-j", path,
                  "-o", "2", "-O", "5", "-z", str(raw)])
            packed = np.fromfile(raw, dtype="<u4")
        width, height = int(info["width"]), int(info["height"])
        if packed.size != width * height:
            raise RuntimeError("復元したHDR画像の寸法が一致しません")
        rgb = np.stack([(packed >> shift) & 1023 for shift in (0, 10, 20)], axis=-1)
        rgb16 = np.rint(rgb * (65535.0 / 1023.0)).astype("<u2")
        info["image_bytes"] = _run([
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb48le",
            "-video_size", f"{width}x{height}", "-color_primaries", "bt2020",
            "-color_trc", "smpte2084", "-colorspace", "rgb", "-color_range", "pc",
            "-i", "-", "-frames:v", "1", "-c:v", "png", "-pix_fmt", "rgb48be",
            "-f", "image2pipe", "-",
        ], data=rgb16.tobytes())
        info.update(color_primaries="bt2020", color_transfer="smpte2084",
                    color_source="HDR復元", bit_depth=10)
    return info
