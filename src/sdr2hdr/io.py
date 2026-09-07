from __future__ import annotations

import json
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int | None
    pix_fmt: str | None
    duration: float | None
    field_order: str | None


def ffprobe_first_audio_codec(path: str) -> str | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        return None
    return streams[0].get("codec_name")


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
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


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


def open_encoder(
    output_path: str,
    source_path: str,
    info: VideoInfo,
    peak_nits: float,
    encoder: str = "hevc_videotoolbox",
    x265_preset: str = "medium",
    x265_crf: int = 16,
) -> subprocess.Popen[bytes]:
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
        "gbrpf32le" if encoder == "openexr_acescg" else "rgb48le",
        "-s",
        f"{info.width}x{info.height}",
        "-r",
        f"{info.fps:.06f}",
        "-i",
        "-",
    ]
    if encoder not in {"openexr", "openexr_acescg"}:
        cmd += [
            "-i",
            source_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
        ]
    audio_args = build_audio_output_args(output_path, source_path) if encoder not in {"openexr", "openexr_acescg"} else []
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
    elif encoder in {"openexr", "openexr_acescg"}:
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
            "gbrpf32le" if encoder == "openexr_acescg" else "rgb48le",
            "-metadata",
            "colorspace=ACEScg" if encoder == "openexr_acescg" else "colorspace=BT.2020/PQ",
        ]
        if encoder == "openexr_acescg":
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
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def read_frame(process: subprocess.Popen[bytes], width: int, height: int) -> np.ndarray | None:
    frame_size = width * height * 3
    assert process.stdout is not None
    buffer = process.stdout.read(frame_size)
    if len(buffer) != frame_size:
        return None
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)


def finalize_process(process: subprocess.Popen[bytes], name: str, allow_broken_pipe: bool = False) -> None:
    stderr = b""
    if process.stdin is not None:
        try:
            process.stdin.close()
        except BrokenPipeError:
            if not allow_broken_pipe:
                raise
    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        stderr = process.stderr.read()
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


def stamp_ap1_exr_sequence(pattern: str, frame_count: int, white_luminance: float) -> None:
    """Tag only the frames just written, preserving their encoded pixel values."""
    tool = require_exr_metadata_tool()
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
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Missing EXR frame: {path}")
    for path in paths:
        with tempfile.TemporaryDirectory(prefix=".exr-metadata-", dir=path.parent) as temporary:
            tagged = Path(temporary) / path.name
            subprocess.run([
                tool, "-chromaticities", "0.713", "0.293", "0.165", "0.830",
                "0.128", "0.044", "0.32168", "0.33767",
                "-whiteLuminance", str(float(white_luminance)),
                # AP1 here is display-referred, not the scene-referred lin_ap1_scene ID.
                "-string", "colorInteropID", "unknown",
                "-comments", "Display-referred linear AP1 / ACES white. "
                "Tone-mapped SDR conversion; not recovered scene exposure. "
                "whiteLuminance specifies nits at RGB (1,1,1).",
                str(path), str(tagged),
            ], check=True, capture_output=True)
            tagged.replace(path)


def restamp_hdr_metadata(path: str, max_cll: int | None = None, max_fall: int | None = None) -> None:
    source = Path(path)
    if not source.exists():
        return
    with tempfile.NamedTemporaryFile(suffix=source.suffix, dir=source.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v?", "-map", "0:a?"]
    if max_cll is not None and max_fall is not None:
        mastering = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
        x265_params = (
            "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:"
            f"colormatrix=bt2020nc:master-display={mastering}:"
            f"max-cll={max(int(max_cll), 1)},{max(int(max_fall), 1)}"
        )
        cmd += [
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv420p10le",
            "-tag:v",
            "hvc1",
            "-x265-params",
            x265_params,
        ]
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


def save_image_hdr(
    output_path: str,
    image_rgb48: np.ndarray,
    peak_nits: float = 1000.0,
) -> None:
    """Saves a 16-bit RGB image as a HDR static image (AVIF or JXL or TIFF)."""
    h, w = image_rgb48.shape[:2]
    mastering = "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"
    max_cll = f"{int(peak_nits)},{max(int(peak_nits * 0.4), 1)}"
    
    ext = Path(output_path).suffix.lower()
    
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "-s", f"{w}x{h}",
        "-i", "-",
    ]
    
    if ext == ".jxl":
        cmd += [
            "-c:v", "libjxl",
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
        ]
    elif ext == ".avif":
        cmd += [
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
            # TIFF HDR metadata support in FFmpeg is limited, 
            # but we set the colorspace tags.
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
        ]
    else:
        # Default to JXL if unknown
        cmd += ["-c:v", "libjxl"]

    cmd.append(output_path)
    
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.communicate(input=image_rgb48.tobytes())
    except Exception as exc:
        process.kill()
        raise RuntimeError(f"FFmpeg image encoding failed: {exc}")
    
    if process.returncode != 0:
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg image encoding failed with code {process.returncode}: {stderr}")
