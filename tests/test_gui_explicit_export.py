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
from sdr2hdr.gui_output import export_preview


def until(root, predicate):
    deadline = time.monotonic() + 40
    while not predicate():
        root.update()
        assert time.monotonic() < deadline
        time.sleep(.005)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).digest()


def test_four_conversion_routes_export_only_on_click_and_close_cleans_preview(tmp_path):
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
        assert [job.status for job in app.queue_jobs] == ['exported']*4
        assert [digest(request.output_path) for request in requests] == saved_hashes
        assert all(path.is_file() for path in previews)
        assert not list(target.glob('.sdr2hdr-*'))
        with zipfile.ZipFile(requests[1].output_path) as archive:
            assert archive.namelist() == [f'frame_{n:06}.exr' for n in range(1,7)] + ['audio.mov']
        info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',requests[0].output_path]))
        stream = next(s for s in info['streams'] if s['codec_type']=='video')
        assert int(stream['nb_frames']) == 6
        assert (stream['color_primaries'],stream['color_transfer']) == ('bt2020','smpte2084')
        assert any(s['codec_type']=='audio' for s in info['streams'])
        assert app.last_output_path == requests[-1].output_path
    finally:
        app._close()
    assert not cache.exists()
    assert [digest(request.output_path) for request in requests] == saved_hashes


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
