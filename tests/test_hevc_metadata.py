"""Verify real compressed output, not just encoder command options."""
import json
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from sdr2hdr import app
from sdr2hdr.io import restamp_hdr_metadata, has_expected_hdr_metadata
from sdr2hdr.ai import HeuristicEnhancer


def _run(*args):
    return subprocess.check_output(args, stderr=subprocess.PIPE)


def _samples(path):
    packets = json.loads(_run('ffprobe', '-v', 'error', '-show_packets', '-of', 'json', str(path)))['packets']
    contents = path.read_bytes()
    result = []
    for packet in packets:
        data = contents[int(packet['pos']):int(packet['pos']) + int(packet['size'])]
        if packet['codec_type'] == 'video':
            pictures = []
            pos = 0
            while pos < len(data):
                size = int.from_bytes(data[pos:pos + 4], 'big')
                assert size >= 2 and pos + 4 + size <= len(data)
                nal = data[pos + 4:pos + 4 + size]
                if (nal[0] >> 1) & 63 <= 31:
                    pictures.append(nal)
                pos += size + 4
            data = pictures
        result.append((packet['codec_type'], packet.get('pts_time'), packet.get('dts_time'),
                       packet.get('duration_time'), data))
    return result


def _decoded(path):
    return _run('ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
                '-pix_fmt', 'yuv420p10le', '-f', 'rawvideo', '-')


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source.mp4'
    _run('ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=64x64:r=12',
         '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '1',
         '-c:v', 'libx264', '-c:a', 'aac', str(path))
    return path


@pytest.mark.parametrize('mode', ['preview', 'balanced', 'final'])
@pytest.mark.parametrize('suffix', ['mp4', 'mov'])
def test_app_preserves_selected_compression_audio_and_timing(source, tmp_path, mode, suffix):
    output = tmp_path / f'output.{suffix}'
    measured = []
    before = tmp_path / f'before.{suffix}'

    def stamp(path, cll, fall):
        shutil.copy2(path, before)
        measured.append((max(cll, 1), max(fall, 1)))
        restamp_hdr_metadata(path, cll, fall)

    request = app.ConversionRequest(str(source), str(output), encoder='libx265', x265_mode=mode,
                                    backend='numpy', model_path=str(Path('models/enhancement_model_20260310.pt').resolve()))
    with mock.patch.object(app, 'restamp_hdr_metadata', side_effect=stamp), \
         mock.patch.object(app, 'build_enhancer', return_value=HeuristicEnhancer()):
        result = app.run_conversion(request)
    assert result.processed_frames == 12
    assert len(measured) == 1
    assert _samples(before) == _samples(output)
    assert _decoded(before) == _decoded(output)
    frames = json.loads(_run('ffprobe', '-v', 'error', '-show_frames', '-select_streams', 'v:0',
                             '-of', 'json', str(output)))['frames']
    assert len(frames) == 12
    for frame in frames:
        light = [row for row in frame['side_data_list'] if row['side_data_type'] == 'Content light level metadata']
        assert len(light) == 1
        assert (light[0]['max_content'], light[0]['max_average']) == measured[0]
        mastering = [row for row in frame['side_data_list'] if row['side_data_type'] == 'Mastering display metadata']
        assert len(mastering) == 1
        assert mastering[0]['max_luminance'] == '10000000/10000'
        assert (frame['color_primaries'], frame['color_transfer'], frame['color_space']) == ('bt2020', 'smpte2084', 'bt2020nc')
    # A second update must replace, rather than accumulate, stale static metadata.
    restamp_hdr_metadata(str(output), 432, 123)
    assert _samples(before) == _samples(output)
    frames = json.loads(_run('ffprobe', '-v', 'error', '-show_frames', '-select_streams', 'v:0',
                             '-of', 'json', str(output)))['frames']
    for frame in frames:
        light = [row for row in frame['side_data_list'] if row['side_data_type'] == 'Content light level metadata']
        assert len(light) == 1
        assert (light[0]['max_content'], light[0]['max_average']) == (432, 123)


def test_failed_metadata_update_preserves_original_and_removes_temporary_file(tmp_path):
    source = tmp_path / 'existing.mp4'
    source.write_bytes(b'existing file')
    with pytest.raises(subprocess.CalledProcessError):
        restamp_hdr_metadata(str(source), 1000, 400)
    assert source.read_bytes() == b'existing file'
    assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize('mode', ['ai', 'log'])
@pytest.mark.parametrize('outcome', ['valid', 'mismatch', 'probe_error', 'disabled'])
def test_hdr_verification_gates_publication_and_completion(source, tmp_path, mode, outcome):
    output = tmp_path / 'output.mp4'
    output.write_bytes(b'previous output')
    events, completed, statuses = [], [], []
    callbacks = app.ConversionCallbacks(on_complete=completed.append, on_status=statuses.append)

    def stamp(path, *levels):
        restamp_hdr_metadata(path, *levels)
        events.append(('write', path))
        if outcome == 'mismatch':
            # Simulate a writer leaving a real video without the required HDR tags.
            shutil.copy2(source, path)

    def verify(path):
        assert events == [('write', path)]
        assert output.read_bytes() == b'previous output'
        assert not completed
        events.append(('read', path))
        if outcome == 'probe_error':
            raise subprocess.CalledProcessError(1, ['ffprobe', path])
        return has_expected_hdr_metadata(path)

    request_type, convert = ((app.ConversionRequest, app.run_conversion) if mode == 'ai'
                             else (app.LogConversionRequest, app.run_log_conversion))
    request = request_type(str(source), str(output), encoder='libx265',
                           x265_preset='ultrafast', max_frames=2,
                           verify_hdr_metadata=outcome != 'disabled',
                           **({'backend': 'numpy'} if mode == 'ai' else {}))
    with mock.patch.object(app, 'restamp_hdr_metadata', side_effect=stamp) as writer, \
         mock.patch.object(app, 'has_expected_hdr_metadata', side_effect=verify) as reader, \
         mock.patch.object(app, 'build_enhancer', return_value=HeuristicEnhancer()):
        if outcome in {'mismatch', 'probe_error'}:
            error = RuntimeError if outcome == 'mismatch' else subprocess.CalledProcessError
            with pytest.raises(error):
                convert(request, callbacks)
            assert not completed and '完了' not in statuses
            assert output.read_bytes() == b'previous output'
        else:
            result = convert(request, callbacks)
            assert result.processed_frames == 2 and not result.cancelled
            assert completed == [result] and statuses[-1] == '完了'
            assert result.output_path == str(output)
            assert has_expected_hdr_metadata(str(output))
        if outcome == 'disabled':
            writer.assert_not_called()
            reader.assert_not_called()
        else:
            assert writer.call_count == reader.call_count == 1
            assert [event[0] for event in events] == ['write', 'read']
    assert not list(tmp_path.glob('.sdr2hdr-*'))
