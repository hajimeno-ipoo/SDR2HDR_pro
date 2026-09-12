from __future__ import annotations

import os
import subprocess
import tempfile
import zipfile
from functools import wraps
import platform
import queue
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from sdr2hdr.ai import TorchMapEnhancer
from sdr2hdr.io import (
    ffprobe_video,
    ffprobe_audio_codecs,
    build_log_encoder_command,
    log_output_filter,
    start_logged_process,
    finalize_process,
    open_decoder,
    open_encoder,
    read_frame,
    restamp_hdr_metadata,
    has_expected_hdr_metadata,
    require_exr_metadata_tool,
    stamp_ap1_exr_sequence,
    exr_sequence_paths,
    restamp_prores_metadata,
)

from .output_color import ACES_OUTPUTS, EXR_ENCODERS, HLG_ENCODER, PRORES_ENCODERS, OutputColorTransform, EXRVideoTransform

# AI libraries are imported lazily inside run_conversion to ensure physical separation
# and avoid loading heavy dependencies when only using Log Passthrough mode.

def get_presets():
    from sdr2hdr.core import ProcessorConfig
    return {
        "poc": ProcessorConfig(
            peak_nits=600.0,
            ai_strength=0.15,
            detail_boost=0.12,
            scene_smoothing=0.82,
            processing_scale=0.75,
            fast_mode=True,
            saturation=1.0,
        ),
        "balanced": ProcessorConfig(
            peak_nits=1000.0,
            ai_strength=0.30,
            detail_boost=0.20,
            scene_smoothing=0.88,
            fast_mode=True,
            saturation=1.10,
        ),
        "high": ProcessorConfig(
            peak_nits=1000.0,
            ai_strength=0.42,
            detail_boost=0.28,
            scene_smoothing=0.92,
            saturation=1.18,
            gamut_expansion=0.12,
        ),
        "portrait": ProcessorConfig(
            peak_nits=800.0,
            ai_strength=0.18,
            detail_boost=0.12,
            scene_smoothing=0.93,
            scene_cut_threshold=0.14,
            highlight_boost=0.72,
            subtitle_protection=0.90,
            shadow_noise_floor=0.10,
            skin_protection=0.82,
            shadow_rolloff=0.62,
            processing_scale=0.85,
            fast_mode=True,
            clipped_white_protection=0.78,
            near_white_rolloff_start=0.74,
            near_white_rolloff_strength=0.72,
            saturation=1.03,
        ),
        "natural": ProcessorConfig(
            peak_nits=800.0,
            ai_strength=0.15,
            detail_boost=0.10,
            scene_smoothing=0.93,
            scene_cut_threshold=0.14,
            highlight_boost=0.55,
            subtitle_protection=0.90,
            shadow_noise_floor=0.10,
            skin_protection=0.82,
            shadow_rolloff=0.62,
            processing_scale=0.85,
            fast_mode=True,
            clipped_white_protection=0.78,
            near_white_rolloff_start=0.74,
            near_white_rolloff_strength=0.72,
            saturation=1.05,
        ),
        "anime": ProcessorConfig(
            peak_nits=1000.0,
            ai_strength=0.22,
            detail_boost=0.08,
            scene_smoothing=0.96,
            highlight_boost=0.75,
            shadow_noise_floor=0.05,
            fast_mode=True,
            saturation=1.12,
        ),
        "cinema": ProcessorConfig(
            peak_nits=1000.0,
            tone="cinema",
            diffuse_white_nits=203.0,
            ai_strength=0.20,
            detail_boost=0.10,
            scene_smoothing=0.95,
            scene_cut_threshold=0.12,
            highlight_boost=0.60,
            subtitle_protection=0.92,
            shadow_noise_floor=0.08,
            skin_protection=0.85,
            shadow_rolloff=0.70,
            clipped_white_protection=0.75,
            near_white_rolloff_start=0.70,
            near_white_rolloff_strength=0.80,
            saturation=1.02,
            gamut_expansion=0.06,
        ),
    }

PRESETS = ("poc", "balanced", "high", "portrait", "natural", "anime", "cinema")
HDR10_ENCODERS = {"hevc_videotoolbox", "hevc_nvenc", "libx265"}

def get_preset_descriptions():
    return {
        "natural": "【推奨】自然な色合いと明るさを保つ、最も汎用的な設定です。",
        "balanced": "明るさとディテールのバランスが取れた、標準的な設定です。",
        "high": "輝度の拡張を強調し、よりダイナミックで力強い印象を与えます。",
        "portrait": "人物の肌の質感を保護し、柔らかなHDR表現を行います。",
        "anime": "アニメ特有の鮮やかな色彩と発光を強調し、輪郭ノイズを抑えた設定です。",
        "cinema": "【映画調】フィルムの滑らかな階調とハイライトロールオフを適用します。",
        "poc": "処理速度を優先した軽量な設定です（検証用）。",
    }

X265_PROFILE_DEFAULTS = {
    "preview": {"preset": "veryfast", "crf": 20},
    "balanced": {"preset": "medium", "crf": 16},
    "final": {"preset": "slow", "crf": 14},
}


@dataclass
class ConversionRequest:
    input_path: str
    output_path: str
    preset: str = "natural"
    encoder: str = "hevc_videotoolbox"
    x265_mode: str = "balanced"
    x265_preset: str | None = None
    x265_crf: int | None = None
    peak_nits: float | None = None
    ai_strength: float | None = None
    highlight_boost: float | None = None
    detail_boost: float | None = None
    processing_scale: float | None = None
    fast_mode: bool = False
    backend: str = "auto"
    model_path: str | None = None
    device: str = "cpu"
    max_frames: int | None = None
    fallback_to_x265_on_hardware_error: bool = False
    keep_partial_output_on_cancel: bool = True
    verify_hdr_metadata: bool = True
    saturation: float | None = None
    hdr_guidance: str = "auto"
    luminance_guidance_strength: float = 0.70
    reconstruction_strength: float = 0.60
    exr_delivery: str = "zip"


@dataclass
class LogConversionRequest:
    input_path: str
    output_path: str
    encoder: str = "hevc_videotoolbox"
    x265_mode: str = "balanced"
    x265_preset: str | None = None
    x265_crf: int | None = None
    max_frames: int | None = None
    keep_partial_output_on_cancel: bool = True
    verify_hdr_metadata: bool = True
    exr_delivery: str = "zip"


@dataclass
class ImageConversionRequest:
    input_path: str
    output_path: str
    preset: str = "natural"
    peak_nits: float | None = None
    ai_strength: float | None = None
    highlight_boost: float | None = None
    detail_boost: float | None = None
    processing_scale: float | None = None
    fast_mode: bool = False
    backend: str = "auto"
    model_path: str | None = None
    device: str = "cpu"
    verify_hdr_metadata: bool = True
    saturation: float | None = None
    hdr_guidance: str = "auto"
    luminance_guidance_strength: float = 0.70
    reconstruction_strength: float = 0.60


@dataclass
class ImageLogConversionRequest:
    input_path: str
    output_path: str
    verify_hdr_metadata: bool = True


@dataclass
class ConversionResult:
    output_path: str
    processed_frames: int
    total_frames: int | None
    cancelled: bool = False


@dataclass
class ConversionCallbacks:
    on_status: Callable[[str], None] | None = None
    on_progress: Callable[[int, int | None, float | None], None] | None = None
    on_complete: Callable[[ConversionResult], None] | None = None
    on_error: Callable[[str], None] | None = None


class CancelToken:
    def __init__(self) -> None:
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True


def output_extensions(encoder: str | None = None, exr_delivery: str = "zip") -> tuple[str, ...]:
    if encoder in PRORES_ENCODERS:
        return (".mov",)
    if encoder in EXR_ENCODERS:
        return (".mov",) if exr_delivery == "video" else (".zip",)
    return (".mp4", ".mov", ".mkv")


def build_output_path(input_path: str, extension: Optional[str] = None, encoder: Optional[str] = None, exr_delivery: str = "zip") -> str:
    path = Path(input_path)
    if extension:
        suffix = extension
    elif encoder is None and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".avif", ".tif", ".tiff", ".jxl"}:
        suffix = ".tif"
    else:
        suffix = output_extensions(encoder, exr_delivery)[0]
    return str(path.with_name(f"{path.stem}_hdr{suffix}"))


def validate_export_request(request) -> None:
    source, output = Path(request.input_path), Path(request.output_path)
    if not request.input_path.strip() or not source.is_file():
        raise ValueError(f"Input file does not exist: {source}")
    if not request.output_path.strip():
        raise ValueError("Output path is required.")
    if source.resolve() == output.resolve():
        raise ValueError("Input and output paths must be different.")
    if isinstance(request, (ConversionRequest, LogConversionRequest)):
        valid = HDR10_ENCODERS | PRORES_ENCODERS | EXR_ENCODERS | {HLG_ENCODER}
        if isinstance(request, LogConversionRequest):
            valid -= {*ACES_OUTPUTS, HLG_ENCODER}
        if request.encoder not in valid:
            raise ValueError(f"Unsupported encoder for this mode: {request.encoder}")
        if request.exr_delivery not in {"zip", "video"}:
            raise ValueError(f"Unknown EXR delivery: {request.exr_delivery}")
        extensions = output_extensions(request.encoder, request.exr_delivery)
        if request.x265_mode not in X265_PROFILE_DEFAULTS:
            raise ValueError(f"Unknown x265 mode: {request.x265_mode}")
        if request.max_frames is not None and request.max_frames <= 0:
            raise ValueError("max_frames must be positive.")
        if request.encoder in {"openexr_acescg", *ACES_OUTPUTS}:
            require_exr_metadata_tool()
    else:
        extensions = (".tif", ".tiff", ".jxl", ".avif", ".jpg", ".jpeg", ".png")
        if output.suffix.lower() in {".jpg", ".jpeg"}:
            from sdr2hdr.io import require_ultrahdr_tool
            require_ultrahdr_tool()
    if output.suffix.lower() not in extensions:
        raise ValueError(f"選択した出力形式には {', '.join(extensions)} の拡張子が必要です。")


def _publish_output(function):
    """Work in an owned directory; publish only after processing and packaging."""
    @wraps(function)
    def run(request, callbacks=None, cancel_token=None):
        validate_export_request(request)
        destination = Path(request.output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        exr = getattr(request, "encoder", None) in EXR_ENCODERS
        inner_callbacks = replace(callbacks, on_complete=None) if callbacks else None
        with tempfile.TemporaryDirectory(prefix=".sdr2hdr-", dir=destination.parent) as directory:
            root = Path(directory)
            working = root / ("frame_%06d.exr" if exr else destination.name)
            if cancel_token and cancel_token.cancel_requested:
                result = ConversionResult(str(destination), 0, None, cancelled=True)
            else:
                result = function(replace(request, output_path=str(working)), inner_callbacks, cancel_token)
            if exr and not result.cancelled and request.exr_delivery == "video":
                video = root / destination.name
                result.cancelled = _encode_exr_video(request, str(working), str(video),
                                                     result.processed_frames, callbacks, cancel_token)
                working = video
            elif exr and not result.cancelled:
                _emit_status(callbacks, "EXR連番をZIPにまとめています")
                frames = exr_sequence_paths(str(working), result.processed_frames)
                if not frames or any(not frame.is_file() for frame in frames):
                    raise RuntimeError("EXR frame sequence is incomplete.")
                files = list(frames)
                audio_codecs = ffprobe_audio_codecs(request.input_path)
                if audio_codecs:
                    # Keep MOV for existing compatible tracks; Matroska also
                    # supports Opus/Vorbis/FLAC without changing the audio codec.
                    mov_codecs = {"aac", "mp3", "ac3", "eac3", "alac",
                                  "pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be",
                                  "pcm_s32le", "pcm_s32be", "pcm_f32le", "pcm_f32be",
                                  "pcm_f64le", "pcm_f64be", "pcm_u8"}
                    audio = root / ("audio.mov" if all(codec in mov_codecs for codec in audio_codecs)
                                    else "audio.mka")
                    cmd = ["ffmpeg", "-v", "error", "-i", request.input_path,
                           "-map", "0:a", "-c:a", "copy", "-vn", str(audio)]
                    _, cancelled = _run_ffmpeg(cmd, None, cancel_token, result.total_frames)
                    if cancelled:
                        result.cancelled = True
                    else:
                        files.append(audio)
                archive = root / destination.name
                if not result.cancelled:
                    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
                        for file in files:
                            if cancel_token and cancel_token.cancel_requested:
                                result.cancelled = True
                                break
                            bundle.write(file, file.name)
                    working = archive
            if cancel_token and cancel_token.cancel_requested:
                result.cancelled = True
            # The GUI discards partial output. The existing CLI option can keep a
            # video fragment, but never replaces an existing file on cancellation.
            keep_partial = (not exr and getattr(request, "keep_partial_output_on_cancel", False)
                            and result.processed_frames > 0 and not destination.exists())
            if not result.cancelled or keep_partial:
                if not working.is_file() or working.stat().st_size == 0:
                    raise RuntimeError("No output file was produced.")
                working.replace(destination)
            result = replace(result, output_path=str(destination))
        _emit_status(callbacks, "キャンセル済" if result.cancelled else "完了")
        _emit_complete(callbacks, result)
        return result
    return run


def _encode_exr_video(request, pattern, output_path, frame_count, callbacks, cancel_token):
    import numpy as np
    from .hdr_guidance import resolve_anchor_nits

    if isinstance(request, ConversionRequest):
        config, _, _ = build_request_config(request)
        white = resolve_anchor_nits(config.tone, config.peak_nits, config.diffuse_white_nits)
    else:
        white = 100.0  # The existing Log zscale path's linear reference.
    transform = EXRVideoTransform(request.encoder, white)
    info = ffprobe_video(request.input_path)
    # Log decoding can apply the input's display rotation before saving EXR.
    # Read dimensions from those pixels, while retaining the source frame rate.
    frame_info = ffprobe_video(str(exr_sequence_paths(pattern, 1)[0]))
    info = replace(info, width=frame_info.width, height=frame_info.height)
    _emit_status(callbacks, "EXRからBT.2020/PQのHDR動画へ変換中")
    encoder = open_encoder(output_path, request.input_path, info, 1000, encoder="prores_4444")
    decoder = None
    cancelled = False
    error = None
    start = time.monotonic()
    try:
        decoder = start_logged_process([
            "ffmpeg", "-v", "error", "-framerate", f"{info.fps:.06f}",
            "-start_number", "1", "-i", pattern, "-frames:v", str(frame_count),
            "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-",
        ], stdout=subprocess.PIPE)
        for index in range(frame_count):
            if cancel_token and cancel_token.cancel_requested:
                cancelled = True
                break
            raw = decoder.stdout.read(info.width * info.height * 12)
            if len(raw) != info.width * info.height * 12:
                raise RuntimeError(f"Missing or incomplete EXR frame {index + 1}")
            rgb = np.frombuffer(raw, dtype="<f4").reshape(3, info.height, info.width)
            rgb = rgb[[2, 0, 1]].transpose(1, 2, 0)  # GBR planes -> RGB pixels.
            pq = transform.to_pq(rgb)
            pixels = np.clip(np.round(pq * 65535), 0, 65535).astype("<u2")
            encoder.stdin.write(pixels.tobytes())
            _emit_progress(callbacks, index + 1, frame_count,
                           (index + 1) / max(time.monotonic() - start, 1e-6))
    except Exception as exc:
        error = exc
    finally:
        if cancelled or error is not None:
            _terminate_process(decoder)
            _terminate_process(encoder)
        try:
            finalize_process(encoder, "EXR video encoder")
        except RuntimeError as exc:
            if not cancelled and (error is None or isinstance(error, BrokenPipeError)):
                error = exc
        if decoder is not None:
            try:
                finalize_process(decoder, "EXR video decoder")
            except RuntimeError as exc:
                if not cancelled and error is None:
                    error = exc
    if error is not None:
        raise error
    if not cancelled:
        restamp_prores_metadata(output_path)
    return cancelled


def _run_ffmpeg(cmd, callbacks, cancel_token, total_frames):
    """Consume progress continuously and check cancellation even without output."""
    events = queue.Queue()
    process = start_logged_process(cmd[:-1] + ["-progress", "pipe:1", "-nostats", cmd[-1]],
                                   stdout=subprocess.PIPE)
    def read_progress():
        for line in process.stdout:
            if line.startswith(b"frame="):
                events.put(int(line.split(b"=", 1)[1]))
    reader = threading.Thread(target=read_progress, daemon=True)
    reader.start()
    processed, cancelled = 0, False
    start = time.monotonic()
    try:
        while process.poll() is None or not events.empty():
            if cancel_token and cancel_token.cancel_requested:
                cancelled = True
                _terminate_process(process)
                break
            try:
                processed = events.get(timeout=0.1)
                _emit_progress(callbacks, processed, total_frames, processed / max(time.monotonic()-start, 1e-6))
            except queue.Empty:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=5)
        while not events.empty():
            processed = events.get_nowait()
        try:
            finalize_process(process, "FFmpeg")
        except RuntimeError:
            if not cancelled:
                raise
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        log = getattr(process, "_stderr_log", None)
        if log is not None and not log.closed:
            log.close()
    return processed, cancelled


def build_request_config(request: ConversionRequest | ImageConversionRequest) -> tuple[object, str, int]:
    from dataclasses import replace
    presets = get_presets()
    config = replace(presets[request.preset])
    if getattr(request, "encoder", None) == "openexr_acescg":
        config.output_color_space = "acescg_linear"
    if getattr(request, "encoder", None) in {*ACES_OUTPUTS, HLG_ENCODER}:
        config.output_color_space = request.encoder
    if request.preset in {"portrait", "natural"} and request.model_path and request.ai_strength is None:
        config.ai_strength = 0.25 if request.preset == "portrait" else 0.15
    if request.peak_nits is not None:
        config.peak_nits = request.peak_nits
    if request.ai_strength is not None:
        config.ai_strength = request.ai_strength
    if request.highlight_boost is not None:
        config.highlight_boost = request.highlight_boost
    if request.detail_boost is not None:
        config.detail_boost = request.detail_boost
    if request.processing_scale is not None:
        config.processing_scale = request.processing_scale
    if request.fast_mode:
        config.fast_mode = True
    if getattr(request, "saturation", None) is not None:
        config.saturation = request.saturation
    if getattr(request, "hdr_guidance", None) is not None:
        config.hdr_guidance = request.hdr_guidance
    if getattr(request, "luminance_guidance_strength", None) is not None:
        config.luminance_guidance_strength = request.luminance_guidance_strength
    if getattr(request, "reconstruction_strength", None) is not None:
        config.reconstruction_strength = request.reconstruction_strength
    config.backend = request.backend
    profile = X265_PROFILE_DEFAULTS.get(getattr(request, "x265_mode", "balanced"), X265_PROFILE_DEFAULTS["balanced"])
    x265_preset = getattr(request, "x265_preset", None) or profile["preset"]
    x265_crf = getattr(request, "x265_crf", None) if getattr(request, "x265_crf", None) is not None else profile["crf"]
    return config, x265_preset, x265_crf


def validate_request(request: ConversionRequest | ImageConversionRequest) -> None:
    validate_export_request(request)
    if getattr(request, "hdr_guidance", "auto") not in {"auto", "on", "off"}:
        raise ValueError(f"Unknown hdr_guidance option: {request.hdr_guidance}")
    if (
        getattr(request, "luminance_guidance_strength", 0.7) < 0.0
        or getattr(request, "luminance_guidance_strength", 0.7) > 1.0
    ):
        raise ValueError("luminance_guidance_strength must be between 0.0 and 1.0")
    if (
        getattr(request, "reconstruction_strength", 0.6) < 0.0
        or getattr(request, "reconstruction_strength", 0.6) > 1.0
    ):
        raise ValueError("reconstruction_strength must be between 0.0 and 1.0")
    input_path = Path(request.input_path)
    output_path = Path(request.output_path)
    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")
    if not request.output_path.strip():
        raise ValueError("Output path is required.")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    if not request.model_path or not request.model_path.strip():
        raise ValueError("AI model is required. Select a model from the models folder.")
    model_path = Path(request.model_path)
    suffix = model_path.suffix.lower()
    if suffix != ".pt":
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
    if request.preset not in get_presets():
        raise ValueError(f"Unknown preset: {request.preset}")
    valid_encoders = {
        "hevc_videotoolbox",
        "hevc_nvenc",
        "libx265",
        "prores_422hq",
        "prores_4444",
        "prores_4444_xq",
        "openexr",
        "openexr_acescg",
        *ACES_OUTPUTS,
        HLG_ENCODER,
    }
    if hasattr(request, "encoder") and request.encoder not in valid_encoders:
        raise ValueError(f"Unknown encoder: {request.encoder}")
    if getattr(request, "encoder", None) in {"openexr_acescg", *ACES_OUTPUTS}:
        require_exr_metadata_tool()
    if getattr(request, "encoder", None) in {*ACES_OUTPUTS, HLG_ENCODER}:
        config, _, _ = build_request_config(request)
        OutputColorTransform(request.encoder, config.peak_nits)
    if hasattr(request, "x265_mode") and request.x265_mode not in X265_PROFILE_DEFAULTS:
        raise ValueError(f"Unknown x265 mode: {request.x265_mode}")
    if request.model_path and not model_path.exists():
        raise ValueError(f"Model file does not exist: {request.model_path}")
    if request.backend in {"cuda", "mps", "torch-cpu", "numpy"} and suffix != ".pt":
        raise ValueError(f"Backend '{request.backend}' requires a TorchScript model (.pt).")


def resolve_model_device(request: ConversionRequest, torch_device: str | None) -> str:
    if request.device != "auto":
        return request.device
    return torch_device or "cpu"


def resolve_model_backend(request: ConversionRequest, torch_device: str | None) -> str:
    if request.backend == "auto":
        if torch_device is not None:
            return torch_device
        return "numpy"
    if request.backend == "numpy":
        return "torch-cpu"
    return request.backend


def build_enhancer(request: ConversionRequest, torch_device: str | None) -> object:
    model_backend = resolve_model_backend(request, torch_device)
    return TorchMapEnhancer(
        request.model_path or "",
        device=resolve_model_device(request, torch_device if model_backend != "torch-cpu" else "cpu"),
    )


def _emit_status(callbacks: ConversionCallbacks | None, message: str) -> None:
    if callbacks and callbacks.on_status:
        callbacks.on_status(message)


def _emit_progress(callbacks: ConversionCallbacks | None, processed: int, total: int | None, fps: float | None) -> None:
    if callbacks and callbacks.on_progress:
        callbacks.on_progress(processed, total, fps)


def _emit_complete(callbacks: ConversionCallbacks | None, result: ConversionResult) -> None:
    if callbacks and callbacks.on_complete:
        callbacks.on_complete(result)


def _emit_error(callbacks: ConversionCallbacks | None, message: str) -> None:
    if callbacks and callbacks.on_error:
        callbacks.on_error(message)


def is_hardware_encoder_failure(message: str) -> bool:
    lowered = message.lower()
    return (
        "videotoolbox" in lowered
        or "compression session" in lowered
        or "nvenc" in lowered
        or "nvidia" in lowered
        or "no capable devices found" in lowered
        or "cannot load nvcuda" in lowered
        or "unsupported device" in lowered
    )


def is_videotoolbox_failure(message: str) -> bool:
    lowered = message.lower()
    return "videotoolbox" in lowered or "compression session" in lowered


def default_encoder_for_platform(system_name: str | None = None) -> str:
    system_name = system_name or platform.system()
    if system_name == "Darwin":
        return "hevc_videotoolbox"
    if system_name == "Windows":
        return "hevc_nvenc"
    return "libx265"


def _terminate_process(process: object | None) -> None:
    if process is None:
        return
    try:
        process.terminate()
    except Exception:
        return


def _wait_terminated_process(process: object | None) -> None:
    if process is None:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    for handle_name in ("stdin", "stdout", "stderr", "_stderr_log"):
        handle = getattr(process, handle_name, None)
        if handle is not None:
            try:
                handle.close()
            except (OSError, ValueError):
                pass


@_publish_output
def run_conversion(
    request: ConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    current_request = request
    attempted_fallback = False
    while True:
        try:
            return _run_conversion_once(current_request, callbacks=callbacks, cancel_token=cancel_token)
        except RuntimeError as exc:
            if (
                current_request.fallback_to_x265_on_hardware_error
                and current_request.encoder in {"hevc_videotoolbox", "hevc_nvenc"}
                and not attempted_fallback
                and is_hardware_encoder_failure(str(exc))
            ):
                attempted_fallback = True
                _emit_status(callbacks, f"{current_request.encoder} failed; falling back to libx265")
                try:
                    os.remove(current_request.output_path)
                except FileNotFoundError:
                    pass
                current_request = replace(current_request, encoder="libx265")
                continue
            raise


def _run_conversion_once(
    request: ConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    # Lazy imports to ensure AI engine is only loaded when needed
    from sdr2hdr.ai import HeuristicEnhancer
    from sdr2hdr.core import SDRToHDRProcessor
    import torch

    config, x265_preset, x265_crf = build_request_config(request)
    if request.encoder in {"openexr_acescg", *ACES_OUTPUTS}:
        require_exr_metadata_tool()
    if request.encoder in ACES_OUTPUTS:
        exr_sequence_paths(request.output_path, 2)
    info = ffprobe_video(request.input_path)
    total_frames = request.max_frames if request.max_frames is not None else info.frames
    processor = SDRToHDRProcessor(config, enhancer=HeuristicEnhancer())
    _emit_status(callbacks, "Loading AI model")
    processor.enhancer = build_enhancer(request, processor.torch_device)
    if request.hdr_guidance == "on":
        if isinstance(processor.enhancer, TorchMapEnhancer):
            dummy = torch.rand(1, 3, 64, 64).to(processor.enhancer.device)
            with torch.inference_mode():
                out = processor.enhancer.model(dummy)
            if out.shape[1] != 5:
                raise RuntimeError("HDR Guidance requires a v2 5-channel model.")
        else:
            raise RuntimeError("HDR Guidance requires a v2 5-channel model.")
    encoder = open_encoder(
        request.output_path,
        request.input_path,
        info,
        config.peak_nits,
        encoder=request.encoder,
        x265_preset=x265_preset,
        x265_crf=x265_crf,
    )
    try:
        decoder = open_decoder(request.input_path, info)
    except BaseException:
        _terminate_process(encoder)
        _wait_terminated_process(encoder)
        raise
    processed = 0
    cancelled = False
    start = time.monotonic()
    _emit_status(callbacks, "Preparing conversion")

    _SENTINEL = object()
    decode_q: queue.Queue = queue.Queue(maxsize=3)
    encode_q: queue.Queue = queue.Queue(maxsize=3)
    pipeline_error: list[BaseException] = []
    abort = threading.Event()
    stop_decode = threading.Event()
    decoder_eof = threading.Event()

    def put_item(target, item, *, decoding=False):
        while not abort.is_set():
            if decoding and (stop_decode.is_set() or (cancel_token and cancel_token.cancel_requested)):
                return False
            try:
                target.put(item, timeout=0.1)
                return True
            except queue.Full:
                if not decoding and cancel_token and cancel_token.cancel_requested and item is not _SENTINEL:
                    return False
        return False


    def _decoder_thread() -> None:
        try:
            frame_count = 0
            while True:
                if abort.is_set() or stop_decode.is_set() or (cancel_token and cancel_token.cancel_requested):
                    break
                if request.max_frames is not None and frame_count >= request.max_frames:
                    break
                frame = read_frame(decoder, info.width, info.height)
                if frame is None:
                    decoder_eof.set()
                    break
                if not put_item(decode_q, frame, decoding=True):
                    break
                frame_count += 1
        except Exception as exc:
            pipeline_error.append(exc)
            abort.set()
        finally:
            put_item(decode_q, _SENTINEL, decoding=True)

    def _encoder_thread() -> None:
        try:
            assert encoder.stdin is not None
            while True:
                if abort.is_set():
                    break
                try:
                    item = encode_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    break
                try:
                    encoder.stdin.write(item.tobytes())
                except BrokenPipeError:
                    abort.set()
                    break
        except Exception as exc:
            pipeline_error.append(exc)
            abort.set()

    dec_thread = threading.Thread(target=_decoder_thread, daemon=True)
    enc_thread = threading.Thread(target=_encoder_thread, daemon=True)
    dec_thread.start()
    enc_thread.start()

    try:
        while True:
            if cancel_token and cancel_token.cancel_requested:
                cancelled = True
                if not request.keep_partial_output_on_cancel or request.encoder in EXR_ENCODERS:
                    abort.set()
                    _terminate_process(encoder)
                _emit_status(callbacks, "Cancelling")
                break
            if abort.is_set():
                break
            try:
                item = decode_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                break
            hdr_frame = processor.process_frame(item)
            if not put_item(encode_q, hdr_frame):
                continue
            processed += 1
            if processed == 1:
                _emit_status(callbacks, "Converting")
            elapsed = max(time.monotonic() - start, 1e-6)
            fps = processed / elapsed
            _emit_progress(callbacks, processed, total_frames, fps)
        stop_decode.set()
        if not decoder_eof.is_set():
            _terminate_process(decoder)
        put_item(encode_q, _SENTINEL)
        enc_thread.join(timeout=30)
        dec_thread.join(timeout=5)
        if enc_thread.is_alive():
            abort.set()
            _terminate_process(encoder)
            enc_thread.join(timeout=5)
            raise RuntimeError("Encoder did not finish after the frame pipeline stopped.")
        if pipeline_error:
            raise pipeline_error[0]
    except Exception as exc:
        _emit_error(callbacks, str(exc))
        raise
    finally:
        abort.set()
        stop_decode.set()
        if not decoder_eof.is_set():
            _terminate_process(decoder)
        if enc_thread.is_alive():
            _terminate_process(encoder)
        dec_thread.join(timeout=5)
        enc_thread.join(timeout=5)
        encoder_error = None
        try:
            finalize_process(encoder, "encoder", allow_broken_pipe=False)
        except RuntimeError as exc:
            if not cancelled:
                encoder_error = exc
        # We explicitly stop the decoder on cancellation, frame limit and error.
        try:
            finalize_process(decoder, "decoder", allow_broken_pipe=True)
        except RuntimeError:
            if decoder_eof.is_set() and not (cancelled or request.max_frames or encoder_error or pipeline_error):
                raise
        if encoder_error is not None:
            raise encoder_error
    keep_output = not cancelled or request.keep_partial_output_on_cancel
    if processed > 0 and keep_output and not (cancelled and request.encoder in EXR_ENCODERS):
        if request.encoder in {"openexr_acescg", *ACES_OUTPUTS}:
            from sdr2hdr.hdr_guidance import resolve_anchor_nits
            _emit_status(callbacks, "Writing EXR color metadata")
            stamp_ap1_exr_sequence(
                request.output_path, processed,
                resolve_anchor_nits(config.tone, config.peak_nits, config.diffuse_white_nits),
                aces_output=request.encoder if request.encoder in ACES_OUTPUTS else None,
            )
        elif request.encoder in PRORES_ENCODERS:
            _emit_status(callbacks, "Writing ProRes HDR color metadata")
            restamp_prores_metadata(request.output_path)
    if cancelled:
        if keep_output and processed > 0 and request.encoder in HDR10_ENCODERS:
            restamp_hdr_metadata(request.output_path)
        if not keep_output or processed == 0:
            outputs = (exr_sequence_paths(request.output_path, processed)
                       if request.encoder in EXR_ENCODERS else [Path(request.output_path)])
            for output in outputs:
                output.unlink(missing_ok=True)
        result = ConversionResult(
            output_path=request.output_path,
            processed_frames=processed,
            total_frames=total_frames,
            cancelled=True,
        )
        _emit_complete(callbacks, result)
        return result
    if processed == 0:
        raise RuntimeError("No frames were processed. Check the input path and video stream.")
    measured_cll, measured_fall = processor.get_measured_hdr_metadata()
    if request.encoder in HDR10_ENCODERS and request.verify_hdr_metadata:
        _emit_status(callbacks, "Writing measured HDR metadata without re-encoding")
        restamp_hdr_metadata(request.output_path, int(round(measured_cll)), int(round(measured_fall)))
        if not has_expected_hdr_metadata(request.output_path):
            raise RuntimeError("HDRメタデータの検証に失敗しました。BT.2020 / PQ / BT.2020ncを確認できません。")
    result = ConversionResult(output_path=request.output_path, processed_frames=processed, total_frames=total_frames)
    _emit_complete(callbacks, result)
    return result

@_publish_output
def run_log_conversion(request: LogConversionRequest, callbacks=None, cancel_token=None) -> ConversionResult:
    info = ffprobe_video(request.input_path)
    total = request.max_frames if request.max_frames is not None else info.frames
    profile = X265_PROFILE_DEFAULTS[request.x265_mode]
    cmd = build_log_encoder_command(request.output_path, request.input_path, info, request.encoder,
                                    request.x265_preset or profile["preset"],
                                    request.x265_crf if request.x265_crf is not None else profile["crf"])
    if request.max_frames is not None:
        cmd[-1:-1] = ["-frames:v", str(request.max_frames)]
    _emit_status(callbacks, "動画Logを変換中")
    processed, cancelled = _run_ffmpeg(cmd, callbacks, cancel_token, total)
    if not cancelled:
        if processed == 0:
            raise RuntimeError("No frames were processed.")
        if request.encoder == "openexr_acescg":
            stamp_ap1_exr_sequence(request.output_path, processed, 100.0)
        elif request.encoder in PRORES_ENCODERS:
            restamp_prores_metadata(request.output_path)
        elif request.encoder in HDR10_ENCODERS and request.verify_hdr_metadata:
            restamp_hdr_metadata(request.output_path)
            if not has_expected_hdr_metadata(request.output_path):
                raise RuntimeError("HDRメタデータの検証に失敗しました。BT.2020 / PQ / BT.2020ncを確認できません。")
    return ConversionResult(request.output_path, processed, total, cancelled)


@_publish_output
def run_image_conversion(
    request: ImageConversionRequest,
    callbacks: ConversionCallbacks | None = None,
    cancel_token: CancelToken | None = None,
) -> ConversionResult:
    from sdr2hdr.ai import HeuristicEnhancer
    from sdr2hdr.core import SDRToHDRProcessor
    from sdr2hdr.io import load_image, save_image_hdr

    config, _, _ = build_request_config(request)
    _emit_status(callbacks, "AIモデルの読み込み中")
    processor = SDRToHDRProcessor(config, enhancer=HeuristicEnhancer())
    processor.enhancer = build_enhancer(request, processor.torch_device)
    
    if cancel_token and cancel_token.cancel_requested:
        return ConversionResult(request.output_path, 0, 1, True)
    _emit_status(callbacks, "画像を読み込み中")
    img_bgr8 = load_image(request.input_path)
    
    _emit_status(callbacks, "AI処理中")
    hdr_rgb48 = processor.process_frame(img_bgr8)
    
    if cancel_token and cancel_token.cancel_requested:
        return ConversionResult(request.output_path, 1, 1, True)
    _emit_status(callbacks, "HDR画像を保存中")
    saved = save_image_hdr(request.output_path, hdr_rgb48, peak_nits=config.peak_nits,
                           cancel_check=(lambda: cancel_token.cancel_requested) if cancel_token else None)
    if not saved:
        return ConversionResult(request.output_path, 1, 1, True)
    
    result = ConversionResult(output_path=request.output_path, processed_frames=1, total_frames=1)
    _emit_complete(callbacks, result)
    return result


@_publish_output
def run_image_log_conversion(request: ImageLogConversionRequest, callbacks=None, cancel_token=None) -> ConversionResult:
    from sdr2hdr.io import save_image_hdr
    import numpy as np
    info = ffprobe_video(request.input_path)
    _emit_status(callbacks, "画像Logを変換中")
    # Use the same RGB16 input to the TIFF/JXL/AVIF saver as image AI mode.
    raw = Path(request.output_path).parent / "image.rgb48"
    cmd = ["ffmpeg", "-v", "error", "-i", request.input_path,
           "-vf", log_output_filter(rgb_input=(info.pix_fmt or "").startswith(("rgb", "bgr", "gbr"))), "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb48le", str(raw)]
    _, cancelled = _run_ffmpeg(cmd, callbacks, cancel_token, 1)
    if cancelled:
        return ConversionResult(request.output_path, 0, 1, True)
    expected = info.width * info.height * 6
    if not raw.is_file() or raw.stat().st_size != expected:
        raise RuntimeError("Image conversion returned an incomplete RGB frame.")
    pixels = np.fromfile(raw, dtype="<u2").reshape(info.height, info.width, 3)
    saved = save_image_hdr(request.output_path, pixels,
                           cancel_check=(lambda: cancel_token.cancel_requested) if cancel_token else None)
    return ConversionResult(request.output_path, 1, 1, not saved)
