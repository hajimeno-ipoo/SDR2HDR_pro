"""Real output regressions for rotated EXR video and ZIP audio containers."""
import json
import subprocess
import zipfile
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from sdr2hdr import app, io


def run(*args):
    return subprocess.check_output(args, stderr=subprocess.PIPE)


def audio_packets(path):
    data = json.loads(run('ffprobe', '-v', 'error', '-select_streams', 'a',
                         '-show_packets', '-show_data_hash', 'sha256', '-of', 'json', str(path)))
    streams = {}
    for packet in data['packets']:
        streams.setdefault(packet['stream_index'], []).append(packet['data_hash'])
    return list(streams.values())


@pytest.mark.parametrize('codecs,filename', [
    (['libopus'], 'audio.mka'),
    (['aac', 'libopus'], 'audio.mka'),
    (['aac', 'pcm_s16le'], 'audio.mov'),
])
def test_zip_preserves_every_audio_track_without_reencoding(tmp_path, codecs, filename):
    source = tmp_path / 'source.mkv'
    cmd = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=64x32:r=2']
    for index in range(len(codecs)):
        cmd += ['-f', 'lavfi', '-i', f'sine=frequency={440+index*100}:sample_rate=48000']
    cmd += ['-t', '1', '-map', '0:v', '-c:v', 'libx264']
    for index, codec in enumerate(codecs):
        cmd += ['-map', f'{index+1}:a', f'-c:a:{index}', codec]
    run(*cmd, str(source))
    output = tmp_path / 'frames.zip'
    result = app.run_log_conversion(app.LogConversionRequest(str(source), str(output), encoder='openexr'))
    assert not result.cancelled and result.processed_frames == 2
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ['frame_000001.exr', 'frame_000002.exr', filename]
        archive.extractall(tmp_path / 'extracted')
    audio = tmp_path / 'extracted' / filename
    assert io.ffprobe_audio_codecs(source) == io.ffprobe_audio_codecs(audio)
    assert audio_packets(source) == audio_packets(audio)
    for index in range(len(codecs)):
        def decoded(path):
            return run('ffmpeg', '-v', 'error', '-i', str(path), '-map', f'0:a:{index}', '-f', 's16le', '-')
        assert decoded(source) == decoded(audio)
    assert not list(tmp_path.glob('.sdr2hdr-*'))


def test_rotated_exr_video_keeps_saved_pixel_layout(tmp_path):
    base = tmp_path / 'base.mp4'
    source = tmp_path / 'rotated.mp4'
    run('ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=64x32:r=2',
        '-t', '1', '-c:v', 'libx264', str(base))
    run('ffmpeg', '-v', 'error', '-display_rotation', '90', '-i', str(base), '-c', 'copy', str(source))
    request = app.LogConversionRequest(str(source), str(tmp_path / 'frames.zip'), encoder='openexr')
    app.run_log_conversion(request)
    with zipfile.ZipFile(request.output_path) as archive:
        archive.extractall(tmp_path / 'frames')
    frames = sorted((tmp_path / 'frames').glob('*.exr'))
    assert len(frames) == 2
    frame_info = io.ffprobe_video(str(frames[0]))
    assert (frame_info.width, frame_info.height) == (32, 64)
    expected = []
    for frame in frames:
        raw = run('ffmpeg', '-v', 'error', '-i', str(frame), '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', '-')
        rgb = np.frombuffer(raw, '<f4').reshape(3, 64, 32)[[2, 0, 1]].transpose(1, 2, 0)
        expected.append(np.clip(np.round(rgb * 65535), 0, 65535).astype('<u2').tobytes())
    captured = []
    def encoder(*args, **kwargs):
        process = io.open_encoder(*args, **kwargs)
        stream = process.stdin
        class Writer:
            def write(self, data):
                captured.append(data)
                return stream.write(data)
            def close(self):
                stream.close()
        process.stdin = Writer()
        return process
    output = tmp_path / 'rotated.mov'
    # Exercise the real ProRes software writer, without requiring Apple hardware.
    with patch.object(io, 'is_videotoolbox_available', return_value=False), \
         patch.object(app, 'open_encoder', side_effect=encoder):
        result = app.run_log_conversion(replace(request, output_path=str(output), exr_delivery='video'))
    assert result.processed_frames == 2 and not result.cancelled
    assert captured == expected  # Layout and order checked before lossy compression.
    info = io.ffprobe_video(str(output))
    assert (info.width, info.height, info.fps, info.frames) == (32, 64, 2, 2)
