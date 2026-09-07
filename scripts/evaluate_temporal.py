"""Compare only v2 temporal smoothing, without training or changing app defaults."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from sdr2hdr.ai import TorchMapEnhancer
from sdr2hdr.app import get_presets
from sdr2hdr.core import SDRToHDRProcessor
from sdr2hdr.io import ffprobe_video
from sdr2hdr.review import compute_chroma_preservation_error, pq_to_relative_linear, tone_map_linear_preview


def read_frames(path: str, count: int, start: float, hdr: bool) -> np.ndarray:
    pixel_format, dtype = ('rgb48le', np.uint16) if hdr else ('bgr24', np.uint8)
    result = subprocess.run([
        'ffmpeg', '-v', 'error', '-ss', f'{start:.9f}', '-i', path,
        '-frames:v', str(count), '-vf', 'scale=960:540:flags=lanczos',
        '-pix_fmt', pixel_format, '-f', 'rawvideo', '-',
    ], check=True, capture_output=True)
    frames = np.frombuffer(result.stdout, dtype=dtype).reshape(-1, 540, 960, 3)
    if len(frames) != count:
        raise ValueError(f'Expected {count} frames, got {len(frames)}')
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('data/live_pilot/split.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    enhancer = TorchMapEnhancer(args.model, device='cpu')
    luma = np.array([.2627, .6780, .0593], np.float32)
    rows = []
    for content in plan['splits']['validation']:
        pair = plan['pairs'][content]
        info = ffprobe_video(pair['sdr'])
        count = round(info.fps)
        start_frame = max(0, round((info.frames - count) / 2))
        start = start_frame / info.fps
        sdr = read_frames(pair['sdr'], count, start, hdr=False)
        hdr = read_frames(pair['hdr'], count, start, hdr=True)
        reference = [pq_to_relative_linear(
            cv2.resize(frame, (480, 270)).astype(np.float32) / 65535, 1000,
        ) for frame in hdr]
        del hdr
        previews = [tone_map_linear_preview(reference[count // 2])]
        for strength in (.75, 0.):
            config = replace(get_presets()['natural'], backend='numpy', ai_strength=.25,
                             target_luma_temporal_strength=strength)
            processor = SDRToHDRProcessor(config, enhancer)
            errors, chroma = [], []
            for index, frame in enumerate(sdr):
                pq = cv2.resize(processor.process_frame(frame), (480, 270))
                linear = pq_to_relative_linear(pq.astype(np.float32) / 65535, 1000)
                errors.append(np.log(linear @ luma + 1e-4) - np.log(reference[index] @ luma + 1e-4))
                chroma.append(compute_chroma_preservation_error(linear, reference[index]))
                if index == count // 2:
                    previews.append(tone_map_linear_preview(linear))
            errors = np.asarray(errors)
            row = dict(content=content, frames=count, start_frame=start_frame, temporal_strength=strength,
                       log_luma_mae=float(np.mean(np.abs(errors))), chroma_error=float(np.mean(chroma)),
                       temporal_log_error_delta=float(np.mean(np.abs(np.diff(errors, axis=0)))))
            rows.append(row)
            print(row, flush=True)
        if not cv2.imwrite(str(args.output / f'{content}.png'), np.concatenate(previews, axis=1)[..., ::-1]):
            raise RuntimeError('Could not write comparison preview')
    with (args.output / 'report.json').open('x') as handle:
        json.dump(dict(
            scope='Five validation-content midpoint ~1s clips. Actual processor output before video encoding. '
                  '960x540 processing, 480x270 measurement. Not an unused holdout or perceptual flicker verdict. '
                  'Preview columns: reference / temporal strength 0.75 / 0, independently tone mapped.',
            model=args.model, rows=rows,
        ), handle, indent=2)


if __name__ == '__main__':
    main()
