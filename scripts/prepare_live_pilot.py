"""Create a fixed content split and a small LIVE dataset without copying videos."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path

from scripts.prepare_data import convert_paired_frame_to_npz
from sdr2hdr.io import ffprobe_video
from sdr2hdr.pairing import load_manifest_video_pairs


def partition_contents(contents: list[str], seed: int = 20260907) -> dict[str, list[str]]:
    names = sorted(set(contents))
    if len(names) != 31:
        raise ValueError(f"Expected the 31 LIVE contents, got {len(names)}")
    random.Random(seed).shuffle(names)
    return dict(train=sorted(names[:21]), validation=sorted(names[21:26]), test=sorted(names[26:]))


def create_plan(root: Path, output: Path) -> None:
    manifest = root / 'JOD_separate.csv'
    pairs = load_manifest_video_pairs(manifest, root / 'open-sourced_HDR10', root / 'open-sourced_SDR')
    chosen = {p.content_name: p for p in pairs if p.resolution == '3840x2160' and p.bitrate == '50000'}
    splits = partition_contents(list(chosen))
    plan = dict(seed=20260907, source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                width=960, peak_nits=1000, fractions=[0.1, 0.3, 0.5, 0.7, 0.9], splits=splits,
                pairs={name: dict(hdr=str(p.hdr_path.resolve()), sdr=str(p.sdr_path.resolve()))
                       for name, p in sorted(chosen.items())})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'split.json').open('x') as handle:
        json.dump(plan, handle, indent=2)
    print({name: len(values) for name, values in splits.items()}, flush=True)


def prepare_split(output: Path, split: str) -> None:
    plan = json.loads((output / 'split.json').read_text())
    destination = output / split
    destination.mkdir(exist_ok=False)
    for content in plan['splits'][split]:
        paths = plan['pairs'][content]
        hdr = ffprobe_video(paths['hdr'])
        sdr = ffprobe_video(paths['sdr'])
        if ((hdr.width, hdr.height, hdr.frames, hdr.fps) != (sdr.width, sdr.height, sdr.frames, sdr.fps)
                or not hdr.duration or not sdr.duration or abs(hdr.duration - sdr.duration) > 1 / hdr.fps):
            raise ValueError(f"Incompatible video pair: {content}")
        width = min(plan['width'], hdr.width)
        height = round(hdr.height * width / hdr.width)
        with tempfile.TemporaryDirectory(prefix='live-pilot-') as temporary:
            for index, fraction in enumerate(plan['fractions']):
                # Seek the identical frame in both streams.
                frame_index = round(fraction * (hdr.frames - 1)) if hdr.frames else round(fraction * hdr.duration * hdr.fps)
                time = frame_index / hdr.fps
                for role, pixel_format in [('hdr', 'rgb48le'), ('sdr', 'rgb24')]:
                    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{time:.9f}', '-i', paths[role],
                                    '-frames:v', '1', '-vf', f'scale={width}:{height}:flags=lanczos',
                                    '-pix_fmt', pixel_format, str(Path(temporary) / f'{role}.png')], check=True)
                convert_paired_frame_to_npz(Path(temporary) / 'sdr.png', Path(temporary) / 'hdr.png',
                                           destination / f'{content}_frame_{index:06d}.npz',
                                           peak_nits=plan['peak_nits'], content_name=content)
        print(split, content, len(plan['fractions']), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['plan', 'prepare'])
    parser.add_argument('--root', type=Path, default=Path('data/LIVE_Paired_Comparison_HDRvsSDR_Database'))
    parser.add_argument('--output', type=Path, default=Path('data/live_pilot'))
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation'])
    args = parser.parse_args()
    if args.operation == 'plan':
        create_plan(args.root, args.output)
    else:
        for split in args.splits:
            prepare_split(args.output, split)


if __name__ == '__main__':
    main()
