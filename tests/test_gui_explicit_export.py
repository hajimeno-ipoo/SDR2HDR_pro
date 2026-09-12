"""The GUI must not touch delivery paths until the user exports."""
import hashlib
import json
import subprocess
import threading
import time
import tkinter as tk
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from sdr2hdr.app import CancelToken, ConversionRequest, ImageConversionRequest, ImageLogConversionRequest, LogConversionRequest
from sdr2hdr.gui import AppState, QueueJob, SDR2HDRGUI
from sdr2hdr.gui_output import PreviewOutputs, export_preview


def until(root, predicate):
    deadline = time.monotonic() + 40
    while not predicate():
        root.update()
        assert time.monotonic() < deadline
        time.sleep(.005)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).digest()


def test_four_conversion_routes_export_only_on_click_and_export_cleans_preview(tmp_path):
    image = tmp_path / 'source.png'
    cv2.imwrite(str(image), np.full((72,128,3), 160, np.uint8))
    video = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=128x72:rate=12',
                    '-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t','0.5',
                    '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(video)],check=True)
    target = tmp_path / 'delivery'
    model = str(Path(__file__).resolve().parents[1] / 'models/enhancement_model_20260310.pt')
    requests = [
        ConversionRequest(str(video),str(target/'ai.mov'),encoder='prores_422hq',backend='numpy',model_path=model),
        LogConversionRequest(str(video),str(target/'log.zip'),encoder='openexr'),
        ImageConversionRequest(str(image),str(target/'ai.png'),backend='numpy',model_path=model),
        ImageLogConversionRequest(str(image),str(target/'log.png')),
    ]
    root = tk.Tk(); app = SDR2HDRGUI(root)
    cache = Path(app.preview_outputs.directory.name)
    try:
        app.queue_jobs = [QueueJob(request) for request in requests]
        app._refresh_job_list()
        app.start_button.invoke()
        until(root, lambda: app.state in {AppState.COMPLETED,AppState.FAILED})
        assert [job.status for job in app.queue_jobs] == ['completed']*4
        assert not target.exists()
        assert app.last_output_path is None
        assert app.open_output_button.instate(['disabled'])
        previews = [Path(job.preview_path) for job in app.queue_jobs]
        assert all(path.is_file() and path.stat().st_size for path in previews)
        assert len(app.compare_view.pairs) == 3  # ZIP is explicitly excluded.
        assert {pair.hdr_path for pair in app.compare_view.pairs} == {str(previews[i]) for i in (0,2,3)}
        until(root, lambda: app.compare_view.controller is not None and app.compare_view.controller.ready)
        saved_hashes = [digest(path) for path in previews]
        app.export_button.invoke()
        until(root, lambda: app.state not in {AppState.EXPORTING,AppState.CANCELLING})
        assert not app.queue_jobs and not app.queue_view.get_children()
        assert not app.compare_view.pairs and not app.compare_view.selection.get()
        assert app.compare_view.controller is None
        assert app.compare_view.clock.get() == '00:00 / 00:00'
        assert app.compare_view.fit_button.instate(['disabled'])
        assert not app.open_output_button.instate(['disabled'])
        assert not app.open_folder_button.instate(['disabled'])
        assert [digest(request.output_path) for request in requests] == saved_hashes
        assert all(not path.parent.exists() for path in previews)
        assert not list(cache.iterdir())
        assert not list(target.glob('.sdr2hdr-*'))
        with zipfile.ZipFile(requests[1].output_path) as archive:
            assert archive.namelist() == [f'frame_{n:06}.exr' for n in range(1,7)] + ['audio.mov']
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',requests[0].output_path]))
        stream = next(s for s in info['streams'] if s['codec_type']=='video')
        assert int(stream['nb_frames']) == 6
        assert (stream['color_primaries'],stream['color_transfer']) == ('bt2020','smpte2084')
        assert any(s['codec_type']=='audio' for s in info['streams'])
        assert app.last_output_path == requests[-1].output_path
        with patch('sdr2hdr.gui.open_path') as open_path:
            app.open_output_button.invoke()
            app.open_folder_button.invoke()
            assert [call.args[0] for call in open_path.call_args_list] == [requests[-1].output_path, str(target)]
    finally:
        app._close()
    assert not cache.exists()
    assert [digest(request.output_path) for request in requests] == saved_hashes


def test_two_sequential_video_inputs_convert_and_export_as_distinct_files(tmp_path):
    sources = [tmp_path / 'first.mp4', tmp_path / 'second.mp4']
    for source, colour in zip(sources, ['red', 'blue']):
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'color=c={colour}:size=128x72:rate=12',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000', '-t', '0.5',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(source)], check=True)
    source_hashes = [digest(path) for path in sources]
    destinations = [tmp_path / 'delivery' / name for name in ('first_hdr.mov', 'second_hdr.mov')]
    root = tk.Tk()
    app = SDR2HDRGUI(root)
    try:
        app.encoder_var.set(app.encoder_options['prores_422hq'])
        app.backend_var.set(app.backend_options['numpy'])
        with patch('sdr2hdr.gui.filedialog.askopenfilenames', return_value=(str(sources[0]),)):
            app._browse_input()
        with patch('sdr2hdr.gui.filedialog.asksaveasfilename', return_value=str(destinations[0])):
            app._browse_output()
        app._enqueue_inputs()
        with patch('sdr2hdr.gui.filedialog.askopenfilenames', return_value=(str(sources[1]),)):
            app._browse_input()
        assert app.output_var.get() == str(destinations[1])
        app._enqueue_inputs()
        app.start_button.invoke()
        until(root, lambda: app.state in {AppState.COMPLETED, AppState.FAILED})
        assert [job.status for job in app.queue_jobs] == ['completed', 'completed']
        previews = [Path(job.preview_path) for job in app.queue_jobs]
        assert [p.name for p in previews] == [p.name for p in destinations]
        assert [(pair.sdr_path, Path(pair.hdr_path).name) for pair in app.compare_view.pairs] == [
            (str(source), destination.name) for source, destination in zip(sources, destinations)]
        hashes = [digest(path) for path in previews]
        assert hashes[0] != hashes[1]
        assert all(not path.exists() for path in destinations)
        app.export_button.invoke()
        until(root, lambda: app.state not in {AppState.EXPORTING, AppState.CANCELLING})
        assert [digest(path) for path in destinations] == hashes
        assert len(list(destinations[0].parent.iterdir())) == 2
        assert not app.queue_jobs and not app.compare_view.pairs
        for destination in destinations:
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(destination)]))
            video = next(s for s in info['streams'] if s['codec_type'] == 'video')
            assert int(video['nb_frames']) == 6
            assert any(s['codec_type'] == 'audio' for s in info['streams'])
        assert [digest(path) for path in sources] == source_hashes
    finally:
        app._close()


@pytest.mark.parametrize('exists', [False, True])
def test_duplicate_export_destinations_stop_before_any_write(tmp_path, exists):
    root = tk.Tk()
    app = SDR2HDRGUI(root)
    destination = tmp_path / 'same.mov'
    if exists:
        destination.write_bytes(b'previous delivery')
    try:
        for index in range(2):
            preview = app.preview_outputs.allocate(str(destination))
            Path(preview).write_bytes(bytes([index + 1]) * 128)
            app.queue_jobs.append(QueueJob(ConversionRequest(str(tmp_path / f'{index}.mp4'), str(destination)),
                                           status='completed', preview_path=preview))
        with patch('sdr2hdr.gui.messagebox.showerror') as error, patch('sdr2hdr.gui.export_preview') as export:
            app._export()
        export.assert_not_called()
        assert '保存先が重複' in error.call_args.args[0]
        assert destination.read_bytes() == b'previous delivery' if exists else not destination.exists()
        assert [job.status for job in app.queue_jobs] == ['completed', 'completed']
        assert all(Path(job.preview_path).is_file() for job in app.queue_jobs)
    finally:
        app._close()


@pytest.mark.parametrize('failure', ['cancel','error'])
def test_interrupted_copy_preserves_existing_destination_and_allows_retry(tmp_path, failure):
    source, destination = tmp_path/'preview.mov', tmp_path/'saved.mov'
    source.write_bytes(b'new output' * 1_000_000)
    destination.write_bytes(b'previous output')
    token = CancelToken()
    def interrupt(copied,size):
        if failure == 'cancel':
            token.cancel()
        else:
            raise OSError('write failed')
    if failure == 'cancel':
        assert not export_preview(str(source),str(destination),token,interrupt)
    else:
        with pytest.raises(OSError,match='write failed'):
            export_preview(str(source),str(destination),token,interrupt)
    assert destination.read_bytes() == b'previous output'
    assert source.stat().st_size == 10_000_000
    assert not list(tmp_path.glob('.sdr2hdr-export-*'))
    assert export_preview(str(source),str(destination),CancelToken())
    assert digest(source) == digest(destination)


def test_close_waits_for_active_writer_before_deleting_owned_directory(tmp_path):
    root = tk.Tk(); app = SDR2HDRGUI(root)
    app.cancel_token = CancelToken()
    cache = Path(app.preview_outputs.directory.name)
    ready = threading.Event(); release = threading.Event(); finished = threading.Event()
    def writer():
        ready.set()
        release.wait(5)
        (cache/'last-frame').write_bytes(b'last write')
        finished.set()
    app.worker = threading.Thread(target=writer)
    app.worker.start(); assert ready.wait(2)
    app._close()
    assert app.cancel_token.cancel_requested
    assert cache.exists()
    release.set()
    until(root, lambda: not cache.exists())
    assert finished.is_set()


def test_overwrite_decline_does_not_start_export(tmp_path):
    root = tk.Tk(); app = SDR2HDRGUI(root)
    try:
        destination = tmp_path/'existing.png'; destination.write_bytes(b'existing')
        preview = Path(app.preview_outputs.allocate(str(destination))); preview.write_bytes(b'preview')
        app.queue_jobs = [QueueJob(ImageLogConversionRequest('input.png',str(destination)), 'completed', preview_path=str(preview))]
        with patch('sdr2hdr.gui.messagebox.askyesno',return_value=False) as confirm:
            app._export()
        confirm.assert_called_once()
        assert app.worker is None
        assert destination.read_bytes() == b'existing'
        assert app.queue_jobs[0].status == 'completed'
    finally:
        app._close()


@pytest.mark.parametrize('failure', ['cancel', 'error'])
def test_partial_export_cleans_only_successful_jobs_and_can_retry(tmp_path, failure):
    root = tk.Tk(); app = SDR2HDRGUI(root)
    try:
        sources, previews, destinations = [], [], []
        for index, status in enumerate(('completed', 'completed', 'queued', 'failed', 'cancelled')):
            source = tmp_path / f'source{index}.png'
            cv2.imwrite(str(source), np.full((16,16,3), 80 + index, np.uint8))
            destination = tmp_path / f'output{index}.png'
            preview = Path(app.preview_outputs.allocate(str(destination)))
            preview.write_bytes(source.read_bytes())
            sources.append(source); previews.append(preview); destinations.append(destination)
            app.queue_jobs.append(QueueJob(ImageLogConversionRequest(str(source), str(destination)), status, preview_path=str(preview)))
        original_hashes = [digest(path) for path in sources]
        retained = app.queue_jobs[1:]
        with patch.object(app.compare_view, '_select'):
            for source, preview in zip(sources[:2], previews[:2]):
                app.compare_view.add_pair(str(source), str(preview), image=True)
        app._refresh_job_list()
        def deliver(source, destination, token, progress):
            if source == str(previews[1]):
                if failure == 'cancel':
                    token.cancel()
                    return False
                raise OSError('test write failure')
            return export_preview(source, destination, token, progress)
        with patch('sdr2hdr.gui.export_preview', side_effect=deliver), patch('sdr2hdr.gui.messagebox.showerror'):
            app.export_button.invoke()
            until(root, lambda: app.state not in {AppState.EXPORTING, AppState.CANCELLING})
        assert app.queue_jobs == retained
        assert [job.status for job in app.queue_jobs] == ['completed', 'queued', 'failed', 'cancelled']
        assert digest(destinations[0]) == original_hashes[0]
        assert not previews[0].parent.exists()
        assert all(path.is_file() for path in previews[1:])
        assert all(not path.exists() for path in destinations[1:])
        assert [pair.hdr_path for pair in app.compare_view.pairs] == [str(previews[1])]
        assert not app.export_button.instate(['disabled'])
        app.export_button.invoke()
        until(root, lambda: app.state not in {AppState.EXPORTING, AppState.CANCELLING})
        assert app.queue_jobs == retained[1:]
        assert not app.compare_view.pairs
        assert not previews[1].parent.exists()
        assert all(path.is_file() for path in previews[2:])
        assert [digest(path) for path in sources] == original_hashes
        assert digest(destinations[1]) == original_hashes[1]
    finally:
        app._close()


def test_discard_rejects_files_outside_owned_job_folder(tmp_path):
    previews = PreviewOutputs()
    try:
        source = tmp_path / 'original.png'; source.write_bytes(b'original')
        destination = tmp_path / 'saved.png'; destination.write_bytes(b'saved')
        owned = Path(previews.allocate('result.png')); owned.write_bytes(b'preview')
        sibling = Path(previews.allocate('other.png')); sibling.write_bytes(b'keep')
        for invalid in (source, destination, Path(previews.directory.name) / 'root-level.png'):
            with pytest.raises(ValueError):
                previews.discard(str(invalid))
        alias = owned.parent / 'outside.png'; alias.symlink_to(source)
        with pytest.raises(ValueError):
            previews.discard(str(alias))
        previews.discard(str(owned))
        assert not owned.parent.exists()
        assert sibling.read_bytes() == b'keep'
        assert source.read_bytes() == b'original' and destination.read_bytes() == b'saved'
    finally:
        previews.close()


def test_cleanup_failure_does_not_undo_export_or_block_event_loop(tmp_path):
    root = tk.Tk(); app = SDR2HDRGUI(root)
    try:
        destination = tmp_path / 'saved.png'
        preview = Path(app.preview_outputs.allocate(str(destination))); preview.write_bytes(b'converted')
        job = QueueJob(ImageLogConversionRequest('input.png', str(destination)), 'completed', preview_path=str(preview))
        app.queue_jobs = [job]
        app._refresh_job_list()
        with patch.object(app.preview_outputs, 'discard', side_effect=PermissionError('denied')):
            app.export_button.invoke()
            until(root, lambda: app.state not in {AppState.EXPORTING, AppState.CANCELLING})
        assert destination.read_bytes() == preview.read_bytes() == b'converted'
        assert job.status == 'exported' and app.queue_jobs == [job]
        assert '削除できませんでした' in app.result_var.get()
        assert not app.open_output_button.instate(['disabled'])
    finally:
        app._close()
