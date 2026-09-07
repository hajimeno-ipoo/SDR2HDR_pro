"""Extract real adjacent LIVE pairs, preserving the existing content split."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from sdr2hdr.io import ffprobe_video


def adjacent_indices(frame_count: int, fractions: list[float]) -> list[tuple[int, int]]:
    if frame_count < 2:
        raise ValueError("At least two source frames are required")
    starts = sorted({min(frame_count - 2, max(0, round(f * (frame_count - 1)))) for f in fractions})
    return [(start, start + 1) for start in starts]


def extract(path: str, indices: list[int], width: int, height: int, hdr: bool) -> np.ndarray:
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-i", path, "-an", "-vf",
        f"select={select},scale={width}:{height}:flags=lanczos", "-fps_mode", "passthrough",
        "-pix_fmt", "rgb48le" if hdr else "bgr24", "-f", "rawvideo", "-",
    ], check=True, capture_output=True)
    dtype = np.dtype("<u2") if hdr else np.dtype("u1")
    expected = len(indices) * height * width * 3 * dtype.itemsize
    if len(result.stdout) != expected:
        raise ValueError(f"Decoded frame count mismatch for {path}")
    return np.frombuffer(result.stdout, dtype=dtype).reshape(len(indices), height, width, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("data/live_pilot/split.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source_plan.json").write_text(json.dumps(plan, indent=2))
    for split in ("train", "validation"):
        destination = args.output / split
        destination.mkdir()
        for content in plan["splits"][split]:
            paths = plan["pairs"][content]
            hdr, sdr = ffprobe_video(paths["hdr"]), ffprobe_video(paths["sdr"])
            if (hdr.width, hdr.height, hdr.frames, hdr.fps) != (sdr.width, sdr.height, sdr.frames, sdr.fps):
                raise ValueError(f"Incompatible source pair: {content}")
            if not hdr.frames or not hdr.fps:
                raise ValueError(f"Missing frame metadata: {content}")
            pairs = adjacent_indices(hdr.frames, plan["fractions"])
            indices = sorted({index for pair in pairs for index in pair})
            width = plan["width"]
            height = round(width * hdr.height / hdr.width)
            sdr_frames = extract(paths["sdr"], indices, width, height, False)
            hdr_frames = extract(paths["hdr"], indices, width, height, True)
            positions = {index: position for position, index in enumerate(indices)}
            for first, second in pairs:
                selected = [positions[first], positions[second]]
                np.savez_compressed(
                    destination / f"{content}_frames_{first:06d}_{second:06d}.npz",
                    sdr_bgr8=sdr_frames[selected], hdr_pq16=hdr_frames[selected],
                    frame_indices=np.array([first, second]), timestamps=np.array([first, second]) / hdr.fps,
                    fps=hdr.fps, content_name=content, reference_nits=float(plan["peak_nits"]),
                )
            print(split, content, pairs, flush=True)


if __name__ == "__main__":
    main()
