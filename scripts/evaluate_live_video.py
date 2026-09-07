"""Evaluate fixed midpoint clips after model selection, retaining only reports/previews."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from sdr2hdr.app import ConversionRequest, run_conversion
from sdr2hdr.io import ffprobe_video
from sdr2hdr.review import (
    compute_chroma_preservation_error, compute_log_luma_mae,
    pq_to_relative_linear, tone_map_linear_preview,
)


def decode(path: str, count: int, start: float = 0) -> np.ndarray:
    result = subprocess.run([
        'ffmpeg', '-v', 'error', '-ss', f'{start:.9f}', '-i', path,
        '-frames:v', str(count), '-vf', 'scale=480:270:flags=lanczos',
        '-pix_fmt', 'rgb48le', '-f', 'rawvideo', '-',
    ], check=True, capture_output=True)
    frames = np.frombuffer(result.stdout, dtype=np.uint16).reshape(-1, 270, 480, 3)
    if len(frames) != count:
        raise ValueError(f'Expected {count} frames, got {len(frames)}: {path}')
    return frames.astype(np.float32) / 65535


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('data/live_pilot/split.json'))
    parser.add_argument('--output', type=Path, default=Path('models/live_v2/test_video'))
    parser.add_argument('--models', nargs='+', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    coefficients = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
    for content in plan['splits']['test']:
        pair = plan['pairs'][content]
        info = ffprobe_video(pair['sdr'])
        count = round(info.fps)
        start_frame = max(0, round((info.frames - count) / 2))
        start = start_frame / info.fps
        reference = pq_to_relative_linear(decode(pair['hdr'], count, start), 1000)
        log_reference = np.log(reference @ coefficients + 1e-4)
        previews = [tone_map_linear_preview(reference[count // 2])]
        with tempfile.TemporaryDirectory(prefix='live-video-acceptance-') as temporary:
            proxy = str(Path(temporary) / 'sdr.mkv')
            subprocess.run([
                'ffmpeg', '-v', 'error', '-ss', f'{start:.9f}', '-i', pair['sdr'],
                '-frames:v', str(count), '-an', '-vf', 'scale=960:540:flags=lanczos',
                '-c:v', 'ffv1', '-pix_fmt', 'bgr0', proxy,
            ], check=True)
            for index, model in enumerate(args.models):
                output = str(Path(temporary) / f'{index}.mp4')
                result = run_conversion(ConversionRequest(
                    input_path=proxy, output_path=output, model_path=model,
                    preset='natural', encoder='libx265', x265_mode='preview',
                    backend='numpy', device='cpu', ai_strength=0.25,
                ))
                if result.cancelled or result.processed_frames != count:
                    raise ValueError(f'Incomplete conversion: {result}')
                prediction = pq_to_relative_linear(decode(output, count), 1000)
                error = np.log(prediction @ coefficients + 1e-4) - log_reference
                metadata = json.loads(subprocess.run([
                    'ffprobe', '-v', 'error', '-read_intervals', '%+#1',
                    '-show_streams', '-show_frames', '-of', 'json', output,
                ], check=True, capture_output=True, text=True).stdout)
                row = dict(
                    content=content, model=model, frames=count, start_frame=start_frame,
                    log_luma_mae=compute_log_luma_mae(prediction, reference),
                    chroma_error=compute_chroma_preservation_error(prediction, reference),
                    temporal_log_error_delta=float(np.mean(np.abs(np.diff(error, axis=0)))),
                    near_black_fraction=float(np.mean(prediction @ coefficients < 1e-4)),
                    peak_channel_fraction=float(np.mean(np.max(prediction, axis=-1) >= 0.799)),
                    metadata=metadata,
                )
                rows.append(row)
                previews.append(tone_map_linear_preview(prediction[count // 2]))
                print({k: v for k, v in row.items() if k != 'metadata'}, flush=True)
        if not cv2.imwrite(str(args.output / f'{content}.png'), np.concatenate(previews, axis=1)[..., ::-1]):
            raise RuntimeError('Preview write failed')
    with (args.output / 'report.json').open('x') as handle:
        json.dump(dict(
            scope='Five fixed ~1s midpoint clips; 960x540 conversion, 480x270 decoded measurement. '
                  'Temporal error change is not a perceptual flicker or ghosting verdict. '
                  'Preview columns: reference, then models in argument order; independently tone mapped.',
            models=args.models, rows=rows,
        ), handle, indent=2)


if __name__ == '__main__':
    main()
