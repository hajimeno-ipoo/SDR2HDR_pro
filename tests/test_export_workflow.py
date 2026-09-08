"""Regression coverage for selected formats, stalled writers and owned output."""
import json
import platform
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from sdr2hdr import app, io
from sdr2hdr.gui import AppState, SDR2HDRGUI, describe_mode_hint

MODEL = str(Path(__file__).resolve().parents[1] / 'models/enhancement_model_20260310.pt')


def command(*args):
    return subprocess.run(args, check=True, capture_output=True).stdout


def streams(path):
    return json.loads(command('ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)))['streams']


def audio_samples(path):
    return command('ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0', '-f', 's16le', '-')


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'source.mp4'
    command('ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=128x72:rate=12',
            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '1',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(path))
    return path


@pytest.mark.parametrize('encoder', ['libx265', 'hevc_videotoolbox', 'prores_422hq',
                                     'prores_4444', 'prores_4444_xq', 'openexr', 'openexr_acescg'])
def test_log_saves_selected_codec_frames_audio_and_colour(source, tmp_path, encoder):
    if encoder in {'hevc_videotoolbox', 'prores_4444_xq'} and platform.system() != 'Darwin':
        pytest.skip('Mac encoder requires macOS')
    output = tmp_path / ('hdr' + app.output_extensions(encoder)[0])
    complete = []
    result = app.run_log_conversion(app.LogConversionRequest(str(source), str(output), encoder=encoder,
                                   x265_preset='ultrafast'), app.ConversionCallbacks(on_complete=complete.append))
    assert result.processed_frames == 12 and not result.cancelled
    assert complete == [result] and complete[0].output_path == str(output)
    assert not list(tmp_path.glob('.sdr2hdr-*'))
    if encoder in app.EXR_ENCODERS:
        with zipfile.ZipFile(output) as archive:
            assert archive.namelist() == [f'frame_{n:06}.exr' for n in range(1, 13)] + ['audio.mov']
            archive.extractall(tmp_path / 'unpacked')
        assert not list(tmp_path.glob('*.exr'))
        assert audio_samples(tmp_path / 'unpacked/audio.mov') == audio_samples(source)
        if encoder == 'openexr_acescg':
            header = command('exrheader', str(tmp_path / 'unpacked/frame_000001.exr')).decode()
            assert 'whiteLuminance (type float): 100' in header
            assert 'chromaticities' in header
        else:
            # Independent FFmpeg reference for the existing BT.709 interpretation,
            # including the colour conversion that was missing in the old Log path.
            raw = command('ffmpeg', '-v', 'error', '-i', str(source), '-frames:v', '1',
                          '-vf', 'zscale=pin=bt709:tin=bt709:min=bt709:p=bt2020:t=smpte2084:m=gbr:range=pc:npl=100,format=gbrp16le',
                          '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', '-')
            expected = np.frombuffer(raw, '<f4')
            saved = command('ffmpeg', '-v', 'error', '-i', str(tmp_path/'unpacked/frame_000001.exr'),
                            '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', '-')
            # Half-float storage rounding plus one RGB16 transport step.
            np.testing.assert_allclose(np.frombuffer(saved, '<f4'), expected, rtol=0,
                                       atol=float(np.spacing(np.float16(1))) / 2 + 1/65535)
    else:
        video = next(s for s in streams(output) if s['codec_type'] == 'video')
        assert video['codec_name'] == ('prores' if encoder in app.PRORES_ENCODERS else 'hevc')
        assert int(video['nb_frames']) == 12
        assert (video['color_primaries'], video['color_transfer'], video['color_space']) == ('bt2020', 'smpte2084', 'bt2020nc')
        assert audio_samples(output) == audio_samples(source)


@pytest.mark.parametrize('mode', ['ai', 'log'])
@pytest.mark.parametrize('input_suffix', ['.png', '.jpg'])
def test_image_formats_save_real_files_and_lossless_rgb(source, tmp_path, mode, input_suffix):
    image = tmp_path / ('source' + input_suffix)
    command('ffmpeg', '-v', 'error', '-i', str(source), '-frames:v', '1', str(image))
    decoded = {}
    for suffix, codec in [('.tif', 'tiff'), ('.jxl', 'jpegxl'), ('.avif', 'av1')]:
        output = tmp_path / ('image' + suffix)
        if mode == 'ai':
            request = app.ImageConversionRequest(str(image), str(output), model_path=MODEL, backend='numpy')
            result = app.run_image_conversion(request)
        else:
            result = app.run_image_log_conversion(app.ImageLogConversionRequest(str(image), str(output)))
        assert result.processed_frames == 1
        video = streams(output)[0]
        assert video['codec_name'] == codec and (video['width'], video['height']) == (128, 72)
        decoded[suffix] = command('ffmpeg', '-v', 'error', '-i', str(output), '-pix_fmt', 'rgb48le', '-f', 'rawvideo', '-')
    assert decoded['.tif'] == decoded['.jxl']
    assert len(decoded['.jxl']) == 128 * 72 * 6
    assert not list(tmp_path.glob('.sdr2hdr-*'))


def test_jxl_keeps_adjacent_16bit_values_and_reports_encoder_error(tmp_path):
    pixels = np.arange(64*64*3, dtype=np.uint16).reshape(64, 64, 3)
    output = tmp_path / 'ramp.jxl'
    io.save_image_hdr(str(output), pixels)
    saved = command('ffmpeg', '-v', 'error', '-i', str(output), '-pix_fmt', 'rgb48le', '-f', 'rawvideo', '-')
    assert saved == pixels.tobytes()
    process = Mock(returncode=17)
    process.communicate.return_value = (None, b'specific image encoder error')
    with patch('sdr2hdr.io.subprocess.Popen', return_value=process):
        with pytest.raises(RuntimeError, match='specific image encoder error'):
            io.save_image_hdr(str(output), pixels)


def test_failing_encoder_releases_full_queues_and_preserves_existing_file(source, tmp_path):
    output = tmp_path / 'existing.mov'
    output.write_bytes(b'existing output')
    processes, failures = [], []
    def failing_encoder(*args, **kwargs):
        process = io.start_logged_process([sys.executable, '-c',
            "import sys,time; sys.stdin.buffer.read(1024); time.sleep(.2); "
            "sys.stderr.write('x'*262144+'specific encoder failure'); sys.exit(37)"], stdin=subprocess.PIPE)
        processes.append(process)
        return process
    class FastProcessor:
        torch_device = None
        def __init__(self, *args, **kwargs): pass
        def process_frame(self, frame): return np.repeat(frame, 8, axis=0).astype(np.uint16)
    def convert():
        try:
            with patch('sdr2hdr.app.open_encoder', side_effect=failing_encoder), \
                 patch('sdr2hdr.core.SDRToHDRProcessor', FastProcessor), \
                 patch('sdr2hdr.app.build_enhancer', return_value=object()):
                app.run_conversion(app.ConversionRequest(str(source), str(output), encoder='prores_422hq'))
        except Exception as exc:
            failures.append(exc)
    worker = threading.Thread(target=convert, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive(), 'Encoder failure left a full frame queue blocked'
    assert len(failures) == 1 and 'specific encoder failure' in str(failures[0])
    assert all(process.poll() == 37 for process in processes)
    assert output.read_bytes() == b'existing output'
    assert not list(tmp_path.glob('.sdr2hdr-*'))


@pytest.mark.parametrize('kind', ['video_ai', 'video_log', 'image_ai', 'image_log'])
def test_precancel_keeps_existing_output_in_all_modes(source, tmp_path, kind):
    output = tmp_path / ('existing.tif' if kind.startswith('image') else 'existing.mp4')
    output.write_bytes(b'keep')
    types = {'video_ai': (app.ConversionRequest, app.run_conversion),
             'video_log': (app.LogConversionRequest, app.run_log_conversion),
             'image_ai': (app.ImageConversionRequest, app.run_image_conversion),
             'image_log': (app.ImageLogConversionRequest, app.run_image_log_conversion)}
    request_type, run = types[kind]
    token = app.CancelToken(); token.cancel()
    result = run(request_type(str(source), str(output)), cancel_token=token)
    assert result.cancelled and result.processed_frames == 0
    assert output.read_bytes() == b'keep'
    assert not list(tmp_path.glob('.sdr2hdr-*'))


@pytest.mark.parametrize('encoder,path', [('prores_422hq','bad.mp4'), ('openexr','frame_%06d.exr'), ('unknown','bad.mp4')])
def test_invalid_video_format_fails_before_starting_encoder(source, tmp_path, encoder, path):
    with patch('sdr2hdr.app.open_encoder') as start:
        with pytest.raises(ValueError):
            app.run_conversion(app.ConversionRequest(str(source), str(tmp_path/path), encoder=encoder))
        start.assert_not_called()


class Var:
    def __init__(self, value=''): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


def test_save_paths_remain_independent_and_keep_manual_location():
    gui = SDR2HDRGUI.__new__(SDR2HDRGUI)
    gui._output_path_state = {}
    pairs = {key: (Var('/input/first.mp4'), Var()) for key in ['video_ai','video_log','image_ai','image_log']}
    for key, (source, output) in pairs.items():
        gui._sync_path(key, source, output, '.mp4')
    ai_source, ai_output = pairs['video_ai']
    ai_output.set('/chosen/custom.name.mp4')
    gui._sync_path('video_ai', ai_source, ai_output, '.mov')
    assert ai_output.get() == '/chosen/custom.name.mov'
    gui._sync_path('video_ai', ai_source, ai_output, '.zip')
    assert ai_output.get() == '/chosen/custom.name.zip'
    for key in ['video_log','image_ai','image_log']:
        source, output = pairs[key]
        source.set('/input/second.mp4')
        gui._sync_path(key, source, output, '.tif' if key.startswith('image') else '.mp4')
        assert output.get() == '/input/second_hdr' + ('.tif' if key.startswith('image') else '.mp4')
    ai_source.set('/input/third.mp4')
    gui._sync_path('video_ai', ai_source, ai_output, '.zip')
    assert ai_output.get() == '/chosen/custom.name.zip'


@pytest.mark.parametrize('extension', ['.mov','.zip','.mp4','.tif','.jxl','.avif'])
def test_dialog_uses_selected_format_and_current_location(extension):
    gui = SDR2HDRGUI.__new__(SDR2HDRGUI)
    gui.state = AppState.IDLE
    gui._input_batches = {}
    gui._output_path_state = {'video_ai': ('input.mp4','/chosen/name.mp4'), 'video_log': ('other.mp4','other_hdr.mp4')}
    output = Var('/chosen/custom' + extension)
    with patch('sdr2hdr.gui.filedialog.asksaveasfilename', return_value='/chosen/result'+extension) as save:
        gui._browse_selected_output(output, extension, 'video_ai')
    assert save.call_args.kwargs['defaultextension'] == extension
    assert save.call_args.kwargs['filetypes'] == [(extension[1:].upper(), '*'+extension)]
    assert save.call_args.kwargs['initialdir'] == '/chosen'
    assert gui._output_path_state['video_log'] == ('other.mp4','other_hdr.mp4')
    assert output.get() == '/chosen/result'+extension


def test_prores_descriptions_do_not_claim_x265_settings():
    for encoder in app.PRORES_ENCODERS:
        hint = describe_mode_hint(encoder, 'final', 'mps', 'natural', MODEL)
        assert 'ProRes' in hint and 'MOV' in hint and 'x265' not in hint


def test_running_video_cancel_discards_only_owned_output(source, tmp_path):
    output = tmp_path / 'existing.mp4'; output.write_bytes(b'keep')
    token = app.CancelToken()
    def progress(processed, total, fps):
        if processed == 2: token.cancel()
    result = app.run_conversion(app.ConversionRequest(str(source), str(output), encoder='libx265',
                                model_path=MODEL, backend='numpy', keep_partial_output_on_cancel=False),
                                app.ConversionCallbacks(on_progress=progress), token)
    assert result.cancelled and result.processed_frames == 2
    assert output.read_bytes() == b'keep'
    assert not list(tmp_path.glob('.sdr2hdr-*'))


def test_cancel_during_image_encoding_stops_child_and_preserves_destination(source, tmp_path):
    image = tmp_path/'source.png'
    command('ffmpeg','-v','error','-i',str(source),'-frames:v','1',str(image))
    output = tmp_path/'existing.avif'; output.write_bytes(b'keep')
    token = app.CancelToken()
    real_popen = subprocess.Popen
    children = []
    def slow_image_encoder(cmd, **kwargs):
        # Only delay the raw-image saver, not the Log conversion subprocess.
        if 'libaom-av1' in cmd:
            cmd = [sys.executable,'-c','import time; time.sleep(60)']
            threading.Timer(.2, token.cancel).start()
        process = real_popen(cmd, **kwargs)
        children.append(process)
        return process
    start = time.monotonic()
    with patch('sdr2hdr.io.subprocess.Popen', side_effect=slow_image_encoder):
        result = app.run_image_log_conversion(app.ImageLogConversionRequest(str(image),str(output)), cancel_token=token)
    assert result.cancelled and time.monotonic()-start < 5
    assert all(child.poll() is not None for child in children)
    assert output.read_bytes() == b'keep'
    assert not list(tmp_path.glob('.sdr2hdr-*'))


def test_failure_marks_job_and_starts_next_without_blocking_dialog():
    from types import SimpleNamespace
    import queue
    from sdr2hdr.gui import QueueJob, AppState
    gui = SDR2HDRGUI.__new__(SDR2HDRGUI)
    gui.root = Mock()
    gui.queue_jobs = [QueueJob(SimpleNamespace(), status='running'), QueueJob(SimpleNamespace())]
    gui.current_job_index = 0
    gui.event_queue = queue.Queue(); gui.event_queue.put(('failed','specific failure'))
    gui.progress = Mock(); gui.status_var = Var(); gui.progress_var = Var()
    gui._refresh_job_list = Mock(); gui._log = Mock(); gui._set_state = Mock(); gui._start_job = Mock()
    with patch('sdr2hdr.gui.messagebox.showerror') as dialog:
        gui._drain_events()
    assert gui.queue_jobs[0].status == 'failed'
    gui._log.assert_called_with('specific failure')
    gui._start_job.assert_called_once_with(1)
    dialog.assert_not_called()
