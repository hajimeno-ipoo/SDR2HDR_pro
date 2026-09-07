"""Compare conversion outputs on held-out NPZ frames without writing videos."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from scripts.train import split_train_validation_indices_by_group
from sdr2hdr.ai import TorchMapEnhancer
from sdr2hdr.app import get_presets
from sdr2hdr.core import SDRToHDRProcessor, REC709_TO_REC2020
from sdr2hdr.dataset import HDRSDRPairDataset, linear_to_srgb
from sdr2hdr.review import compute_log_luma_mae, compute_chroma_preservation_error, pq_to_relative_linear


def reference_bt2020(sample: Mapping[str, np.ndarray]) -> np.ndarray:
    if 'hdr_2020_linear' in sample:
        return sample['hdr_2020_linear'].astype(np.float32)
    # Older NPZ files only retain the clipped BT.709 working reference.
    return sample['hdr_linear'].astype(np.float32) @ REC709_TO_REC2020.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--preset', choices=get_presets(), default='natural')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--all-samples', action='store_true', help='Evaluate a separately partitioned directory in full')
    parser.add_argument('--output', type=Path, help='Save the evaluation JSON; existing files are not overwritten')
    args = parser.parse_args()
    if args.width < 64:
        parser.error('--width must be at least 64')
    dataset = HDRSDRPairDataset(args.data_dir, training=False, patch_size=0)
    if len(dataset) < 2:
        parser.error('At least two samples are required')
    _, indices = split_train_validation_indices_by_group(
        dataset.content_names, max(1, len(dataset) // 10)
    )
    if args.all_samples:
        indices = list(range(len(dataset)))
    enhancers = [TorchMapEnhancer(path, device='cpu') for path in args.models]
    rows = []
    for index in indices:
        with np.load(dataset.paths[index], allow_pickle=False) as sample:
            sdr = sample['sdr_linear'].astype(np.float32)
            hdr = reference_bt2020(sample)
            peak = float(sample['peak_nits'])
        h, w = sdr.shape[:2]
        size = (min(w, args.width), max(64, round(h * min(w, args.width) / w)))
        # Match inference input encoding; keep the reference in linear light.
        rgb = np.round(linear_to_srgb(sdr) * 255).astype(np.uint8)
        bgr = cv2.resize(rgb[..., ::-1], size, interpolation=cv2.INTER_AREA)
        reference = cv2.resize(hdr, size, interpolation=cv2.INTER_AREA)
        for path, enhancer in zip(args.models, enhancers):
            config = replace(get_presets()[args.preset], backend='numpy', ai_strength=0.25)
            # Each held-out frame is independent, not a continuous video.
            output = SDRToHDRProcessor(config, enhancer).process_frame(bgr)
            prediction = pq_to_relative_linear(output.astype(np.float32) / 65535, peak)
            rows.append(dict(sample=dataset.paths[index].name, model=path,
                             log_luma_mae=compute_log_luma_mae(prediction, reference),
                             chroma_error=compute_chroma_preservation_error(prediction, reference)))
    summary = {path: float(np.mean([r['log_luma_mae'] for r in rows if r['model'] == path]))
               for path in args.models}
    chroma = {path: float(np.mean([r['chroma_error'] for r in rows if r['model'] == path]))
              for path in args.models}
    report = json.dumps(dict(preset=args.preset, width=args.width,
                         scope='held-out independent frames; no temporal or encoded-video assessment',
                         rows=rows, mean_log_luma_mae=summary, mean_chroma_error=chroma), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            handle.write(report + '\n')
        print(json.dumps(dict(mean_log_luma_mae=summary, mean_chroma_error=chroma, output=str(args.output)), indent=2))
    else:
        print(report)


if __name__ == '__main__':
    main()
