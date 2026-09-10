from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
import subprocess
import struct
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import cv2

from .output_color import ACES_OUTPUTS, EXR_ENCODERS, HLG_ENCODER


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int | None
    pix_fmt: str | None
    duration: float | None
    field_order: str | None


def ffprobe_comparison(path: str) -> dict:
    """Read source tags, without replacing absent tags with player guesses."""
    import av

    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=color_primaries,color_transfer,color_space,pix_fmt,bits_per_raw_sample,width,height,avg_frame_rate,duration:format=duration",
        "-of", "json", path,
    ], check=True, capture_output=True, text=True, timeout=20)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError("映像または画像を読み取れません")
    info = streams[0]
    info["duration"] = info.get("duration") or payload.get("format", {}).get("duration")
    info["bit_depth"] = None
    if info.get("pix_fmt"):
        try:
            components = av.VideoFormat(info["pix_fmt"]).components
            info["bit_depth"] = max(c.bits for c in components)
        except (ValueError, TypeError):
            pass
    return info


def ffprobe_audio_codecs(path: str) -> list[str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    return [stream.get("codec_name", "unknown") for stream in streams]


def ffprobe_first_audio_codec(path: str) -> str | None:
    codecs = ffprobe_audio_codecs(path)
    return codecs[0] if codecs else None


def ffprobe_video(path: str) -> VideoInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,pix_fmt,duration,field_order:format=duration",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    fmt = payload.get("format", {})
    num, den = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / max(float(den), 1.0)
    frames = stream.get("nb_frames")
    duration = stream.get("duration") or fmt.get("duration")
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frames=int(frames) if frames and frames != "N/A" else None,
        pix_fmt=stream.get("pix_fmt"),
        duration=float(duration) if duration and duration != "N/A" else None,
        field_order=stream.get("field_order"),
    )


def is_interlaced_video(info: VideoInfo) -> bool:
    field_order = (info.field_order or "").lower()
    return field_order not in {"", "unknown", "progressive"}


def open_decoder(path: str, info: VideoInfo) -> subprocess.Popen[bytes]:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
    ]
    if is_interlaced_video(info):
        cmd += [
            "-vf",
            "bwdif=mode=send_frame:parity=auto:deint=all",
        ]
    cmd += [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-vsync",
        "0",
        "-",
    ]
    return start_logged_process(cmd, stdout=subprocess.PIPE)


def build_audio_output_args(output_path: str, source_path: str) -> list[str]:
    output_suffix = Path(output_path).suffix.lower()
    audio_codec = ffprobe_first_audio_codec(source_path)
    if audio_codec is None:
        return []
    if output_suffix == ".mp4":
        copy_safe_codecs = {"aac", "mp3", "ac3", "eac3", "alac"}
        if audio_codec not in copy_safe_codecs:
            return ["-c:a", "aac", "-b:a", "192k"]
    return ["-c:a", "copy"]


def is_videotoolbox_available() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "prores_videotoolbox" in result.stdout


def is_prores_4444_xq_available() -> bool:
    if not is_videotoolbox_available():
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "encoder=prores_videotoolbox"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "p416le" in result.stdout and "4444 XQ" in result.stdout


def build_encoder_command(
    output_path: str,
    source_path: str,
    info: VideoInfo,
    peak_nits: float,
    encoder: str = "hevc_videotoolbox",
    x265_preset: str = "medium",
    x265_crf: int = 16,
) -> list[str]:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mastering = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    max_cll = f"{int(peak_nits)},{max(int(peak_nits * 0.4), 1)}"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gbrpf32le" if encoder in {"openexr_acescg", *ACES_OUTPUTS} else "rgb48le",
        "-s",
        f"{info.width}x{info.height}",
        "-r",
        f"{info.fps:.06f}",
        "-i",
        "-",
    ]
    if encoder not in EXR_ENCODERS:
        cmd += [
            "-i",
            source_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
        ]
    audio_args = build_audio_output_args(output_path, source_path) if encoder not in EXR_ENCODERS else []
    if encoder == "hevc_videotoolbox":
        cmd += [
            "-pix_fmt",
            "p010le",
            "-vf",
            "scale=in_color_matrix=bt2020:out_color_matrix=bt2020",
            "-c:v",
            "hevc_videotoolbox",
            "-profile:v",
            "main10",
            "-tag:v",
            "hvc1",
            "-bsf:v",
            "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9",
            "-allow_sw",
            "1",
            "-prio_speed",
            "true",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
        ]
        cmd += audio_args + [output_path]
    elif encoder == "hevc_nvenc":
        cmd += [
            "-pix_fmt",
            "p010le",
            "-vf",
            "scale=in_color_matrix=bt2020:out_color_matrix=bt2020",
            "-c:v",
            "hevc_nvenc",
            "-profile:v",
            "main10",
            "-preset",
            "p5",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "18",
            "-b:v",
            "0",
            "-tag:v",
            "hvc1",
            "-bsf:v",
            "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
        ]
        cmd += audio_args + [output_path]
    elif encoder in {"prores_422hq", "prores_4444", "prores_4444_xq"}:
        is_4444 = encoder == "prores_4444"
        is_xq = encoder == "prores_4444_xq"
        if is_xq and not is_prores_4444_xq_available():
            raise RuntimeError("ProRes 4444 XQ (12-bit) requires an FFmpeg VideoToolbox encoder with p416le support.")
        enc_name = "prores_videotoolbox" if is_videotoolbox_available() else "prores_ks"
        profile_val = "5" if is_xq else ("4" if is_4444 else "3")
        pix_fmt_val = "p416le" if is_xq else ("p410le" if is_4444 and enc_name == "prores_videotoolbox" else (
            "yuv444p10le" if is_4444 else "yuv422p10le"
        ))
        cmd += [
            "-c:v",
            enc_name,
            "-vf", "scale=in_color_matrix=bt2020:out_color_matrix=bt2020",
            "-profile:v",
            profile_val,
            "-pix_fmt",
            pix_fmt_val,
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
            "-movflags",
            "+write_colr",
        ]
        if enc_name == "prores_videotoolbox":
            cmd[cmd.index("-movflags"):cmd.index("-movflags")] = ["-allow_sw", "1"]
        cmd += audio_args + [output_path]
    elif encoder in EXR_ENCODERS:
        cmd += [
            "-f",
            "image2",
            "-c:v",
            "exr",
            "-format",
            "half",
            "-compression",
            "zip16",
            "-pix_fmt",
            "gbrpf32le" if encoder in {"openexr_acescg", *ACES_OUTPUTS} else "rgb48le",
            "-metadata",
            "colorspace=ACEScg" if encoder in {"openexr_acescg", *ACES_OUTPUTS} else "colorspace=BT.2020/PQ",
        ]
        if encoder in {"openexr_acescg", *ACES_OUTPUTS}:
            cmd += ["-color_trc", "linear"]
        else:
            cmd += [
                "-color_primaries",
                "bt2020",
                "-color_trc",
                "smpte2084",
                "-colorspace",
                "bt2020nc",
            ]
        cmd.append(output_path)
    elif encoder == HLG_ENCODER:
        cmd += [
            "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
            "-vf", "scale=in_color_matrix=bt2020:out_color_matrix=bt2020",
            "-tag:v", "hvc1", "-preset", x265_preset, "-crf", str(x265_crf),
            "-x265-params", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
            "-color_primaries", "bt2020", "-color_trc", "arib-std-b67",
            "-colorspace", "bt2020nc",
        ]
        cmd += audio_args + [output_path]
    else:
        cmd += [
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-tag:v",
            "hvc1",
            "-preset",
            x265_preset,
            "-crf",
            str(x265_crf),
            "-bsf:v",
            "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9",
            "-x265-params",
            f"hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display={mastering}:max-cll={max_cll}",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
        ]
        cmd += audio_args + [output_path]
    return cmd


def start_logged_process(cmd: list[str], **kwargs) -> subprocess.Popen:
    # A file cannot fill a pipe and block FFmpeg while frame queues are full.
    log = tempfile.TemporaryFile()
    try:
        process = subprocess.Popen(cmd, stderr=log, **kwargs)
    except BaseException:
        log.close()
        raise
    process._stderr_log = log
    return process


def open_encoder(
    output_path: str, source_path: str, info: VideoInfo, peak_nits: float,
    encoder: str = "hevc_videotoolbox", x265_preset: str = "medium", x265_crf: int = 16,
) -> subprocess.Popen[bytes]:
    cmd = build_encoder_command(output_path, source_path, info, peak_nits,
                                encoder, x265_preset, x265_crf)
    return start_logged_process(cmd, stdin=subprocess.PIPE)


def log_output_filter(*, linear_ap1: bool = False, rgb_input: bool = False) -> str:
    # Preserve the existing Log mode's BT.709 input interpretation and 100-nit
    # zscale reference. This mode does not decode camera-specific Log curves.
    matrix_in = "gbr" if rgb_input else "bt709"
    prefix = f"zscale=pin=bt709:tin=bt709:min={matrix_in}:p=bt2020:"
    if linear_ap1:
        from sdr2hdr.core import REC2020_TO_ACESCG
        names = ("rr", "rg", "rb", "gr", "gg", "gb", "br", "bg", "bb")
        matrix = ":".join(f"{name}={float(value):.12g}" for name, value in
                          zip(names, REC2020_TO_ACESCG.flat))
        return prefix + "t=linear:m=gbr:range=pc:npl=100,format=gbrpf32le,colorchannelmixer=" + matrix
    return prefix + "t=smpte2084:m=gbr:range=pc:npl=100,format=gbrp16le,format=rgb48le"


def build_log_encoder_command(output_path: str, source_path: str, info: VideoInfo,
                              encoder: str, preset: str, crf: int) -> list[str]:
    cmd = build_encoder_command(output_path, source_path, info, 1000, encoder, preset, crf)
    # Reuse the selected codec, precision, audio and colour-tag options.
    # Replace only the raw-frame input by the original video and its Log filter.
    end = cmd.index("-map") if encoder not in EXR_ENCODERS else cmd.index("-f", 5)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", source_path] + cmd[end:]
    if encoder not in EXR_ENCODERS:
        cmd[cmd.index("1:a?")] = "0:a?"
    filters = log_output_filter(linear_ap1=encoder == "openexr_acescg",
                                rgb_input=(info.pix_fmt or "").startswith(("rgb", "bgr", "gbr")))
    if "-vf" in cmd:
        index = cmd.index("-vf")
        cmd[index + 1] = filters + "," + cmd[index + 1] + ":out_range=tv"
    else:
        if encoder not in EXR_ENCODERS:
            filters += ",scale=in_color_matrix=bt2020:out_color_matrix=bt2020:out_range=tv"
        cmd[-1:-1] = ["-vf", filters]
    return cmd


def read_frame(process: subprocess.Popen[bytes], width: int, height: int) -> np.ndarray | None:
    frame_size = width * height * 3
    assert process.stdout is not None
    buffer = process.stdout.read(frame_size)
    if len(buffer) != frame_size:
        return None
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)


def finalize_process(process: subprocess.Popen[bytes], name: str, allow_broken_pipe: bool = False) -> None:
    if process.stdin is not None:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass  # Report FFmpeg's diagnostic below instead of losing it.
    if process.stdout is not None:
        process.stdout.close()
    log = getattr(process, "_stderr_log", None)
    if log is not None:
        return_code = process.wait()
        log.seek(0)
        stderr = log.read()
        log.close()
    else:
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.stderr is not None:
            process.stderr.close()
        return_code = process.wait()
    rendered = stderr.decode("utf-8", errors="replace").strip()
    if allow_broken_pipe and return_code != 0 and "Broken pipe" in rendered:
        return
    if return_code != 0:
        raise RuntimeError(f"{name} failed with code {return_code}: {rendered}")


def quote_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def require_exr_metadata_tool() -> str:
    tool = shutil.which("exrstdattr")
    if tool is None:
        raise ValueError("AP1 EXR output requires OpenEXR's exrstdattr command on PATH.")
    return tool


def exr_sequence_paths(pattern: str, frame_count: int) -> list[Path]:
    """Resolve exactly the frame names written by FFmpeg (numbering starts at 1)."""
    if frame_count == 0:
        return []
    output = Path(pattern)
    tokens = list(re.finditer(r"%0?(\d*)d", output.name))
    if len(tokens) != 1:
        if tokens or frame_count != 1:
            raise ValueError("EXR sequence path must contain one frame number, such as %06d.")
        paths = [output]
    else:
        token = tokens[0]
        width = int(token.group(1) or 0)
        paths = [output.with_name(output.name[:token.start()] + str(index).zfill(width)
                                  + output.name[token.end():]) for index in range(1, frame_count + 1)]
    return paths


def stamp_ap1_exr_sequence(
    pattern: str, frame_count: int, white_luminance: float,
    *, aces_output: str | None = None,
) -> None:
    """Tag only this conversion's frames, without changing encoded pixels."""
    tool = require_exr_metadata_tool()
    paths = exr_sequence_paths(pattern, frame_count)
    if aces_output is None:
        attributes = [
            "-whiteLuminance", str(float(white_luminance)),
            "-string", "colorInteropID", "unknown",
            "-comments", "Display-referred linear AP1 / ACES white. "
            "Tone-mapped SDR conversion; not recovered scene exposure. "
            "whiteLuminance specifies nits at RGB (1,1,1).",
        ]
    else:
        config_name, view = ACES_OUTPUTS[aces_output]
        attributes = [
            "-string", "colorInteropID", "lin_ap1_scene",
            "-string", "ocioConfig", config_name,
            "-string", "ocioInverseView", view,
            "-comments", "ACEScg from inverse ACES output transform of graded SDR. "
            "Reference display: Rec.2100-PQ, 1000 nit. Not recovered camera exposure.",
        ]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Missing EXR frame: {path}")
    for path in paths:
        with tempfile.TemporaryDirectory(prefix=".exr-metadata-", dir=path.parent) as temporary:
            tagged = Path(temporary) / path.name
            subprocess.run([
                tool, "-chromaticities", "0.713", "0.293", "0.165", "0.830",
                "0.128", "0.044", "0.32168", "0.33767",
                *attributes,
                str(path), str(tagged),
            ], check=True, capture_output=True)
            tagged.replace(path)


def _stamp_prores_mov_colr(path: Path) -> None:
    """Update colr only inside ProRes visual sample entries, never media payload.

    FFmpeg 7.1 VideoToolbox can supply an unspecified colr atom even when output
    colour options and ProRes frame headers are set. Walk QuickTime atom lengths
    rather than searching arbitrary compressed data for a byte sequence.
    """
    offsets: list[int] = []
    with path.open("r+b") as stream:
        def atoms(start: int, end: int):
            while start < end:
                stream.seek(start)
                header = stream.read(8)
                if len(header) != 8:
                    raise ValueError("Truncated MOV atom header")
                size, kind = struct.unpack(">I4s", header)
                header_size = 8
                if size == 1:
                    extended = stream.read(8)
                    if len(extended) != 8:
                        raise ValueError("Truncated MOV extended size")
                    size = struct.unpack(">Q", extended)[0]
                    header_size = 16
                elif size == 0:
                    size = end - start
                if size < header_size or start + size > end:
                    raise ValueError("Invalid MOV atom size")
                yield kind, start + header_size, start + size
                start += size

        def sample_colours(sample: int, sample_end: int, *, nested: bool = False) -> None:
            if nested:
                stream.seek(sample_end - 4)
                if stream.read(4) == b"\x00\x00\x00\x00":
                    sample_end -= 4  # QuickTime sample-description terminator.
            # VisualSampleEntry has 78 bytes after its atom header.
            for extension, data, data_end in atoms(sample + 78, sample_end):
                if extension == b"colr":
                    stream.seek(data)
                    if data_end - data < 10 or stream.read(4) not in {b"nclc", b"nclx"}:
                        raise ValueError("Unsupported ProRes MOV colour atom")
                    offsets.append(data + 4)
                elif extension == b"glbl" and not nested:
                    # VideoToolbox extradata contains another sample description.
                    for codec, embedded, embedded_end in atoms(data, data_end):
                        if codec in {b"apch", b"ap4h", b"ap4x"}:
                            sample_colours(embedded, embedded_end, nested=True)

        def walk(start: int, end: int) -> None:
            for kind, payload, limit in atoms(start, end):
                if kind in {b"moov", b"trak", b"mdia", b"minf", b"stbl"}:
                    walk(payload, limit)
                elif kind == b"stsd":
                    if payload + 8 > limit:
                        raise ValueError("Invalid MOV sample description")
                    for codec, sample, sample_end in atoms(payload + 8, limit):
                        if codec not in {b"apch", b"ap4h", b"ap4x"}:
                            continue
                        sample_colours(sample, sample_end)

        walk(0, path.stat().st_size)
        if not offsets:
            raise ValueError("ProRes MOV has no colour atom")
        for offset in offsets:
            stream.seek(offset)
            stream.write(struct.pack(">HHH", 9, 16, 9))  # BT.2020 / PQ / BT.2020 NCL


def restamp_prores_metadata(path: str) -> None:
    """Set both ProRes frame headers and MOV colour atoms, with atomic replacement."""
    source = Path(path)
    with tempfile.TemporaryDirectory(prefix=".prores-metadata-", dir=source.parent) as directory:
        output = Path(directory) / "tagged.mov"
        subprocess.run([
            "ffmpeg", "-v", "error", "-i", str(source), "-map", "0", "-c", "copy",
            "-bsf:v", "prores_metadata=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc", "-movflags", "+write_colr", str(output),
        ], check=True, capture_output=True)
        _stamp_prores_mov_colr(output)
        output.replace(source)


def restamp_hdr_metadata(path: str, max_cll: int | None = None, max_fall: int | None = None) -> None:
    source = Path(path)
    if not source.exists():
        return
    with tempfile.NamedTemporaryFile(suffix=source.suffix, dir=source.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v?", "-map", "0:a?"]
    if max_cll is not None and max_fall is not None:
        try:
            # Isolate PyAV's bundled FFmpeg from OpenCV's macOS AVFoundation classes.
            subprocess.run([
                sys.executable, "-m", "sdr2hdr.hevc_metadata", str(source), str(temp_path),
                str(max_cll), str(max_fall),
            ], check=True, capture_output=True)
            temp_path.replace(source)
        finally:
            temp_path.unlink(missing_ok=True)
        return
    else:
        cmd += [
            "-c:v",
            "copy",
            "-tag:v",
            "hvc1",
            "-bsf:v",
            "hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9",
        ]
    cmd += [
        "-c:a", "copy",
        "-movflags",
        "+faststart",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "smpte2084",
        "-colorspace",
        "bt2020nc",
        str(temp_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        temp_path.replace(source)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def has_expected_hdr_metadata(path: str) -> bool:
    source = Path(path)
    if not source.exists():
        return False
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_space,color_transfer,color_primaries",
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        return False
    stream = streams[0]
    return (
        stream.get("color_space") == "bt2020nc"
        and stream.get("color_transfer") == "smpte2084"
        and stream.get("color_primaries") == "bt2020"
    )


def load_image(path: str) -> np.ndarray:
    """Reads an image as BGR8."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def require_ultrahdr_tool() -> str:
    executable = shutil.which("ultrahdr_app")
    if executable is None:
        raise ValueError("HDR JPEGの保存にはlibultrahdrのultrahdr_appが必要です。macOSでは brew install libultrahdr で導入できます。")
    return executable


def _save_jpeg_hdr(output_path, image_rgb48, cancel_check) -> bool:
    executable = require_ultrahdr_tool()
    h, w = image_rgb48.shape[:2]
    # libultrahdr v1.4: full-range PQ, BT.2100, little-endian RGBA1010102.
    rgb = np.rint(image_rgb48.astype(np.float64) * (1023.0 / 65535.0)).astype(np.uint32)
    packed = rgb[..., 0] | (rgb[..., 1] << 10) | (rgb[..., 2] << 20) | np.uint32(3 << 30)
    # ST 2084: describe the actual packed input peak, not PQ's 10,000 nit ceiling.
    # libultrahdr -L sets hdr_capacity_max only; gain-map samples remain unchanged.
    power = (float(rgb.max()) / 1023.0) ** (32.0 / 2523.0)
    peak_nits = 10000.0 * (max(power - 3424.0 / 4096.0, 0.0)
                          / (2413.0 / 128.0 - 2392.0 / 128.0 * power)) ** (16384.0 / 2610.0)
    # The decoder requires capacity_max > capacity_min, even for all-SDR pixels.
    minimum_peak = float(np.nextafter(np.float32(203.0), np.float32(np.inf)))
    with tempfile.TemporaryDirectory(prefix=".ultrahdr-", dir=Path(output_path).parent) as directory:
        raw = Path(directory) / "hdr.raw"
        encoded = Path(directory) / "hdr.jpg"
        packed.astype("<u4").tofile(raw)
        cmd = [executable, "-m", "0", "-p", str(raw), "-w", str(w), "-h", str(h),
               "-a", "5", "-C", "2", "-t", "2", "-R", "1", "-q", "95", "-Q", "95",
               "-s", "1", "-M", "1", "-L", str(max(minimum_peak, min(10000.0, peak_nits))),
               "-z", str(encoded)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                if cancel_check and cancel_check():
                    process.kill()
                    process.communicate()
                    return False
                try:
                    log, _ = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            process.kill()
            process.communicate()
            raise
        if process.returncode or not encoded.is_file() or not encoded.stat().st_size:
            raise RuntimeError("HDR JPEGの保存に失敗しました: " + log.decode("utf-8", errors="replace"))
        if cancel_check and cancel_check():
            return False
        _tag_jpeg_hdr_gainmap(encoded)
        if cancel_check and cancel_check():
            return False
        encoded.replace(output_path)
    return True


def _bt2100_pq_icc() -> bytes:
    if platform.system() == "Darwin":
        import Quartz

        color = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceITUR_2100_PQ)
        return bytes(Quartz.CGColorSpaceCopyICCData(color))
    return (Path(__file__).parent / "assets" / "bt2100-pq.icc").read_bytes()


def _jpeg_header_segments(data: bytes | bytearray, start: int, end: int):
    """Read marker segments before the first scan in an encoder-produced JPEG."""
    if data[start:start + 2] != b"\xff\xd8":
        raise RuntimeError("HDR JPEG encoder returned an invalid image header.")
    pos = start + 2
    while pos + 4 <= end:
        marker = data[pos:pos + 2]
        if marker == b"\xff\xda":
            return
        size = int.from_bytes(data[pos + 2:pos + 4], "big") + 2
        if marker[0] != 255 or size < 4 or pos + size > end:
            break
        yield pos, size, marker
        pos += size
    raise RuntimeError("HDR JPEG encoder returned an incomplete image header.")


def _tag_jpeg_hdr_gainmap(path: Path) -> None:
    """Replace the alternate BT.2020/PQ ICC, retaining both JPEG scans and ISO metadata.

    libultrahdr 1.4's ISO-only writer uses a Lab PCS PQ profile that ColorSync
    rejects. MPF's second image size must follow the replacement ICC's length.
    This is for our encoder output, not a general JPEG metadata editor.
    """
    data = bytearray(path.read_bytes())
    mpf = [(p + 8, p + size) for p, size, marker in _jpeg_header_segments(data, 0, len(data))
           if marker == b"\xff\xe2" and data[p + 4:p + 8] == b"MPF\0"]
    if len(mpf) != 1:
        raise RuntimeError("HDR JPEG encoder returned no unique MPF index.")
    base, end = mpf[0]
    if data[base:base + 4] not in (b"II\x2a\0", b"MM\0\x2a"):
        raise RuntimeError("HDR JPEG encoder returned an invalid MPF index.")
    endian = "<" if data[base:base + 2] == b"II" else ">"
    ifd = base + struct.unpack_from(endian + "I", data, base + 4)[0]
    if not base + 8 <= ifd <= end - 2:
        raise RuntimeError("HDR JPEG encoder returned an invalid MPF directory.")
    count = struct.unpack_from(endian + "H", data, ifd)[0]
    if ifd + 2 + count * 12 + 4 > end:
        raise RuntimeError("HDR JPEG encoder returned a truncated MPF directory.")
    entries = None
    for i in range(count):
        tag, kind, size, offset = struct.unpack_from(endian + "HHII", data, ifd + 2 + i * 12)
        if tag == 0xB002 and kind == 7 and size == 32:
            entries = base + offset
    if entries is None or not ifd + 2 + count * 12 + 4 <= entries <= end - 32:
        raise RuntimeError("HDR JPEG encoder returned unsupported MPF image entries.")
    size, offset = struct.unpack_from(endian + "II", data, entries + 20)
    start = base + offset
    if start < end or start + size != len(data) or data[-2:] != b"\xff\xd9":
        raise RuntimeError("HDR JPEG encoder returned an invalid gain-map extent.")
    profiles = [(p, n) for p, n, marker in _jpeg_header_segments(data, start, start + size)
                if marker == b"\xff\xe2" and data[p + 4:p + 16] == b"ICC_PROFILE\0"]
    if not profiles:
        # Writers using the base gamut (including XMP builds) need no alternate ICC.
        return
    if len(profiles) != 1 or data[profiles[0][0] + 16:profiles[0][0] + 18] != b"\x01\x01":
        raise RuntimeError("HDR JPEG encoder returned an unsupported alternate ICC layout.")
    pos, old_size = profiles[0]
    profile = _bt2100_pq_icc()
    payload = b"ICC_PROFILE\0\x01\x01" + profile
    replacement = b"\xff\xe2" + struct.pack(">H", len(payload) + 2) + payload
    data[pos:pos + old_size] = replacement
    struct.pack_into(endian + "I", data, entries + 20, size + len(replacement) - old_size)
    path.write_bytes(data)


def _tag_hdr_png(output_path: str) -> None:
    """Write PNG Third Edition cICP before IDAT; preserve encoded RGB16 samples."""
    path = Path(output_path)
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("PNG encoder returned an invalid file.")
    parts = [data[:8]]
    offset = 8
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        chunk = data[offset:offset + 12 + length]
        if kind != b"cICP":
            parts.append(chunk)
        if kind == b"IHDR":
            payload = b"cICP" + bytes((9, 16, 0, 1))  # BT.2020, PQ, RGB, full range.
            parts.append(struct.pack(">I", 4) + payload + struct.pack(">I", zlib.crc32(payload)))
        offset += 12 + length
    path.write_bytes(b"".join(parts))


def _tag_hdr_tiff(output_path: str) -> None:
    """Embed full-range BT.2100 PQ ICC without rewriting any pixel strips.

    ICC Profile Embedding technical note: TIFF tag 34675, type UNDEFINED.
    This handles the single-image classic TIFF produced by our FFmpeg encoder.
    """
    profile = _bt2100_pq_icc()
    with Path(output_path).open("r+b") as stream:
        header = stream.read(8)
        if header[:4] not in {b"II\x2a\x00", b"MM\x00\x2a"}:
            raise RuntimeError("TIFF encoder returned an unsupported header.")
        endian = "<" if header[:2] == b"II" else ">"
        stream.seek(struct.unpack_from(endian + "I", header, 4)[0])
        count = struct.unpack(endian + "H", stream.read(2))[0]
        entries = [stream.read(12) for _ in range(count)]
        next_ifd = stream.read(4)
        if any(len(entry) != 12 for entry in entries) or next_ifd != bytes(4):
            raise RuntimeError("TIFF encoder returned an incomplete or multi-image file.")
        entries = [entry for entry in entries if struct.unpack_from(endian + "H", entry)[0] != 34675]
        stream.seek(0, 2)
        stream.write(bytes(stream.tell() % 2))
        profile_offset = stream.tell()
        stream.write(profile)
        stream.write(bytes(stream.tell() % 2))
        ifd_offset = stream.tell()
        entries.append(struct.pack(endian + "HHII", 34675, 7, len(profile), profile_offset))
        entries.sort(key=lambda entry: struct.unpack_from(endian + "H", entry)[0])
        stream.write(struct.pack(endian + "H", len(entries)) + b"".join(entries) + next_ifd)
        # Original strip offsets stay valid. Publish the new directory last.
        stream.seek(4)
        stream.write(struct.pack(endian + "I", ifd_offset))


def save_image_hdr(
    output_path: str,
    image_rgb48: np.ndarray,
    peak_nits: float = 1000.0,
    cancel_check: Callable[[], bool] | None = None,
) -> bool:
    """Save BT.2020/PQ RGB16 as an HDR still image."""
    if Path(output_path).suffix.lower() in {".jpg", ".jpeg"}:
        return _save_jpeg_hdr(output_path, image_rgb48, cancel_check)
    h, w = image_rgb48.shape[:2]
    mastering = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    max_cll = f"{int(peak_nits)},{max(int(peak_nits * 0.4), 1)}"
    
    ext = Path(output_path).suffix.lower()
    
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "-s", f"{w}x{h}",
        # Encoders consume the AVFrame's colour properties. Output-only flags
        # do not supply those properties to raw RGB frames on all FFmpeg builds.
        "-color_primaries", "bt2020", "-color_trc", "smpte2084",
        "-colorspace", "rgb", "-color_range", "pc",
        "-i", "-",
    ]
    
    if ext == ".png":
        cmd += ["-c:v", "png", "-pix_fmt", "rgb48be", "-frames:v", "1",
                "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                "-colorspace", "rgb", "-color_range", "pc"]
    elif ext == ".jxl":
        cmd += [
            "-c:v", "libjxl", "-distance", "0", "-pix_fmt", "rgb48le",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
        ]
    elif ext == ".avif":
        avif_filter = (
            "zscale=matrixin=gbr:primariesin=bt2020:transferin=smpte2084:rangein=full:"
            "matrix=bt2020nc:primaries=bt2020:transfer=smpte2084:range=limited"
        )
        if w % 2 or h % 2:
            # zscale cannot process odd-sized subsampled planes. Convert the
            # matrix at full chroma resolution, then subsample without resizing.
            avif_filter += (
                ",format=yuv444p16le,scale=iw:ih:in_range=limited:out_range=limited,"
                "format=yuv420p10le"
            )
        cmd += [
            "-vf", avif_filter,
            "-c:v", "libaom-av1",
            "-still-picture", "1",
            "-pix_fmt", "yuv420p10le",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
        ]
    elif ext in {".tif", ".tiff"}:
        cmd += [
            "-c:v", "tiff",
            "-pix_fmt", "rgb48le",
            # FFmpeg does not serialize these as a TIFF colour profile.
            # _tag_hdr_tiff embeds the matching ICC after encoding.
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
        ]
    else:
        raise ValueError(f"Unsupported image output extension: {ext}")

    cmd.append(output_path)
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        data = image_rgb48.tobytes()
        while True:
            if cancel_check and cancel_check():
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
                return False
            try:
                _, stderr = process.communicate(input=data, timeout=0.1 if cancel_check else None)
                break
            except subprocess.TimeoutExpired:
                data = None  # communicate resumes the input it already buffered.
    except Exception as exc:
        process.kill()
        process.communicate()
        raise RuntimeError(f"FFmpeg image encoding failed: {exc}")
    
    if process.returncode != 0:
        stderr = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg image encoding failed with code {process.returncode}: {stderr}")
    if ext == ".png":
        _tag_hdr_png(output_path)
    elif ext in {".tif", ".tiff"}:
        _tag_hdr_tiff(output_path)
    return True
