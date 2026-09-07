from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sdr2hdr.review import REC2020_TO_REC709, linear_to_srgb, pq_to_relative_linear
from sdr2hdr.pairing import ManifestVideoPair, load_manifest_video_pairs


@dataclass(frozen=True)
class PreparedVideoPair:
    """One video pair and the content group used for train/validation splitting."""

    hdr_path: Path
    sdr_path: Path | None
    content_name: str


def extract_raw_frames(input_path: str, output_dir: Path, pix_fmt: str, sample_every: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        input_path,
        "-vf",
        f"select='not(mod(n\\,{sample_every}))'",
        "-vsync",
        "0",
        "-pix_fmt",
        pix_fmt,
        str(output_dir / "frame_%06d.png"),
    ]
    subprocess.run(cmd, check=True)


def srgb_to_linear(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame, 0.0, 1.0)
    return np.where(frame <= 0.04045, frame / 12.92, ((frame + 0.055) / 1.055) ** 2.4)


def tone_map_hdr_linear_to_sdr_linear(
    frame_709_linear: np.ndarray,
    clip_ratio: float = 0.0,
    quantize_8bit: bool = True,
) -> np.ndarray:
    luma = np.clip(
        0.2126 * frame_709_linear[..., 0]
        + 0.7152 * frame_709_linear[..., 1]
        + 0.0722 * frame_709_linear[..., 2],
        0.0,
        None,
    )
    white = max(float(np.percentile(luma, 99.7)), 0.5)
    exposed = frame_709_linear / white
    mapped = exposed / (1.0 + exposed)

    if clip_ratio > 0.0:
        gain = 1.0 + clip_ratio * 0.8
        mapped = np.clip(mapped * gain, 0.0, 1.0)

    srgb = linear_to_srgb(np.clip(mapped, 0.0, 1.0))
    if quantize_8bit:
        srgb = np.round(srgb * 255.0) / 255.0

    return srgb_to_linear(srgb).astype(np.float32)


def convert_frame_to_npz(
    hdr_frame_path: Path,
    output_path: Path,
    peak_nits: float,
    clip_ratio: float = 0.0,
    content_name: str | None = None,
) -> None:
    hdr_bgr16 = cv2.imread(str(hdr_frame_path), cv2.IMREAD_UNCHANGED)
    if hdr_bgr16 is None:
        raise RuntimeError(f"failed to read HDR frame: {hdr_frame_path}")
    hdr_rgb16 = cv2.cvtColor(hdr_bgr16, cv2.COLOR_BGR2RGB)
    hdr_2020_linear = pq_to_relative_linear(hdr_rgb16.astype(np.float32) / 65535.0, peak_nits=peak_nits)
    hdr_709_linear = np.clip(np.tensordot(hdr_2020_linear, REC2020_TO_REC709.T, axes=1), 0.0, 1.5)
    sdr_linear = tone_map_hdr_linear_to_sdr_linear(hdr_709_linear, clip_ratio=clip_ratio)
    payload: dict[str, np.ndarray] = {
        "sdr_linear": sdr_linear.astype(np.float16),
        "hdr_linear": hdr_709_linear.astype(np.float16),
        "peak_nits": np.asarray(peak_nits, dtype=np.float32),
    }
    if content_name is not None:
        payload["content_name"] = np.asarray(content_name)
    np.savez_compressed(output_path, **payload)


def convert_paired_frame_to_npz(
    sdr_frame_path: Path,
    hdr_frame_path: Path,
    output_path: Path,
    peak_nits: float,
    content_name: str | None = None,
) -> None:
    """Write a training pair from aligned real SDR and HDR frames."""
    sdr_bgr8 = cv2.imread(str(sdr_frame_path), cv2.IMREAD_COLOR)
    hdr_bgr16 = cv2.imread(str(hdr_frame_path), cv2.IMREAD_UNCHANGED)
    if sdr_bgr8 is None:
        raise RuntimeError(f"failed to read SDR frame: {sdr_frame_path}")
    if hdr_bgr16 is None:
        raise RuntimeError(f"failed to read HDR frame: {hdr_frame_path}")
    if sdr_bgr8.shape[:2] != hdr_bgr16.shape[:2]:
        raise RuntimeError(
            f"SDR/HDR frame dimensions differ: {sdr_frame_path.name}={sdr_bgr8.shape[:2]}, "
            f"{hdr_frame_path.name}={hdr_bgr16.shape[:2]}"
        )

    sdr_rgb8 = cv2.cvtColor(sdr_bgr8, cv2.COLOR_BGR2RGB)
    hdr_rgb16 = cv2.cvtColor(hdr_bgr16, cv2.COLOR_BGR2RGB)
    sdr_linear = srgb_to_linear(sdr_rgb8.astype(np.float32) / 255.0).astype(np.float32)
    hdr_2020_linear = pq_to_relative_linear(hdr_rgb16.astype(np.float32) / 65535.0, peak_nits=peak_nits)
    hdr_709_linear = np.clip(np.tensordot(hdr_2020_linear, REC2020_TO_REC709.T, axes=1), 0.0, 1.5)
    payload: dict[str, np.ndarray] = {
        "sdr_linear": sdr_linear.astype(np.float16),
        "hdr_linear": hdr_709_linear.astype(np.float16),
        "peak_nits": np.asarray(peak_nits, dtype=np.float32),
        "pair_source": np.asarray("real_sdr_hdr"),
        # Preserve the original wide-gamut reference for output evaluation.
        "hdr_2020_linear": hdr_2020_linear.astype(np.float16),
    }
    if content_name is not None:
        payload["content_name"] = np.asarray(content_name)
    np.savez_compressed(output_path, **payload)


def find_matching_video(directory: Path, source: Path) -> Path:
    exact = directory / source.name
    if exact.is_file():
        return exact
    matches = sorted(path for path in directory.iterdir() if path.is_file() and path.stem == source.stem)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no SDR counterpart found for {source.name} in {directory}")
    raise RuntimeError(f"multiple SDR counterparts found for {source.name}: {matches}")


def collect_video_pairs(
    input_dir: Path,
    sdr_dir: Path | None,
    manifest_path: Path | None = None,
    *,
    dataset_type: str = "Open-source",
    hdr_format: str = "HDR10",
    sdr_format: str = "SDR",
) -> list[tuple[Path, Path | None]]:
    """Collect HDR/SDR paths while preserving the legacy directory mode."""

    return [
        (pair.hdr_path, pair.sdr_path)
        for pair in collect_video_pairs_with_metadata(
            input_dir,
            sdr_dir,
            manifest_path,
            dataset_type=dataset_type,
            hdr_format=hdr_format,
            sdr_format=sdr_format,
        )
    ]


def collect_video_pairs_with_metadata(
    input_dir: Path,
    sdr_dir: Path | None,
    manifest_path: Path | None = None,
    *,
    dataset_type: str = "Open-source",
    hdr_format: str = "HDR10",
    sdr_format: str = "SDR",
) -> list[PreparedVideoPair]:
    """Collect pairs and preserve the content group from a manifest when present."""

    if not input_dir.is_dir():
        raise NotADirectoryError(f"HDR directory does not exist: {input_dir}")
    if manifest_path is not None:
        if sdr_dir is None:
            raise ValueError("--manifest requires --sdr-dir")
        manifest_pairs: list[ManifestVideoPair] = load_manifest_video_pairs(
            manifest_path,
            input_dir,
            sdr_dir,
            dataset_type=dataset_type,
            hdr_format=hdr_format,
            sdr_format=sdr_format,
        )
        return [
            PreparedVideoPair(pair.hdr_path, pair.sdr_path, pair.content_name)
            for pair in manifest_pairs
        ]

    pairs: list[PreparedVideoPair] = []
    for video_path in sorted(input_dir.iterdir()):
        if not video_path.is_file():
            continue
        pairs.append(
            PreparedVideoPair(
                video_path,
                find_matching_video(sdr_dir, video_path) if sdr_dir else None,
                video_path.stem,
            )
        )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare SDR/HDR training pairs from HDR videos.")
    parser.add_argument("--input-dir", required=True, help="Directory containing HDR videos")
    parser.add_argument(
        "--sdr-dir",
        help="Optional directory containing aligned real SDR videos with matching filenames",
    )
    parser.add_argument(
        "--manifest",
        help="CSV manifest containing exact HDR/SDR filenames and pairing metadata",
    )
    parser.add_argument(
        "--manifest-type",
        default="Open-source",
        help="Manifest Type value to include when --manifest is used",
    )
    parser.add_argument(
        "--hdr-format",
        default="HDR10",
        help="Manifest video_format value for HDR rows",
    )
    parser.add_argument(
        "--sdr-format",
        default="SDR",
        help="Manifest video_format value for SDR rows",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to save .npz training samples")
    parser.add_argument("--sample-every", type=int, default=24, help="Sample every N frames")
    parser.add_argument("--peak-nits", type=float, default=1000.0, help="Peak nits used for relative linear decode")
    parser.add_argument("--degradation-profiles", default="natural,clipped", help="Comma-separated degradation profiles")
    parser.add_argument("--variants-per-profile", type=int, default=1, help="Number of variants per profile")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    sdr_dir = Path(args.sdr_dir) if args.sdr_dir else None
    if sdr_dir is not None and not sdr_dir.is_dir():
        parser.error(f"--sdr-dir does not exist or is not a directory: {sdr_dir}")
    manifest_path = Path(args.manifest) if args.manifest else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = [p.strip() for p in args.degradation_profiles.split(",") if p.strip()]

    try:
        video_pairs = collect_video_pairs_with_metadata(
            input_dir,
            sdr_dir,
            manifest_path,
            dataset_type=args.manifest_type,
            hdr_format=args.hdr_format,
            sdr_format=args.sdr_format,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    with tempfile.TemporaryDirectory(prefix="sdr2hdr-train-") as temp_dir:
        temp_root = Path(temp_dir)
        for pair in video_pairs:
            video_path = pair.hdr_path
            sdr_video_path = pair.sdr_path
            hdr_frames_dir = temp_root / f"{video_path.stem}_hdr_frames"
            extract_raw_frames(str(video_path), hdr_frames_dir, "rgb48le", args.sample_every)
            if sdr_video_path is not None:
                sdr_frames_dir = temp_root / f"{video_path.stem}_sdr_frames"
                extract_raw_frames(str(sdr_video_path), sdr_frames_dir, "rgb24", args.sample_every)
                hdr_frames = sorted(hdr_frames_dir.glob("frame_*.png"))
                sdr_frames = sorted(sdr_frames_dir.glob("frame_*.png"))
                if len(hdr_frames) != len(sdr_frames):
                    raise RuntimeError(
                        f"SDR/HDR frame counts differ for {video_path.name}: "
                        f"HDR={len(hdr_frames)}, SDR={len(sdr_frames)}"
                    )
                for hdr_frame_path, sdr_frame_path in zip(hdr_frames, sdr_frames):
                    output_path = out_dir / f"{video_path.stem}_{hdr_frame_path.stem}_real.npz"
                    convert_paired_frame_to_npz(
                        sdr_frame_path,
                        hdr_frame_path,
                        output_path,
                        peak_nits=args.peak_nits,
                        content_name=pair.content_name,
                    )
                continue
            for hdr_frame_path in sorted(hdr_frames_dir.glob("frame_*.png")):
                for profile in profiles:
                    clip_ratio = 0.5 if profile == "clipped" else 0.0
                    output_path = out_dir / f"{video_path.stem}_{hdr_frame_path.stem}_{profile}.npz"
                    convert_frame_to_npz(
                        hdr_frame_path,
                        output_path,
                        peak_nits=args.peak_nits,
                        clip_ratio=clip_ratio,
                        content_name=pair.content_name,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
