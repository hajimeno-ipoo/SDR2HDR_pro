"""ZIP remains available; video uses the selected representation's display transform."""
import json
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import PyOpenColorIO as ocio
import pytest

from sdr2hdr import app, io
from sdr2hdr.gui import SDR2HDRGUI, EXR_DELIVERY_OPTIONS, describe_mode_hint
from sdr2hdr.output_color import EXRVideoTransform

MODEL = str(Path(__file__).resolve().parents[1] / 'models/enhancement_model_20260310.pt')
OFFICIAL = {
    'openexr_acescg_1_3': ('cg-config-v2.2.0_aces-v1.3_ocio-v2.4', 'ACES 1.1 - HDR Video (1000 nits & Rec.2020 lim)'),
    'openexr_acescg_2_0': ('cg-config-v4.0.0_aces-v2.0_ocio-v2.5', 'ACES 2.0 - HDR 1000 nits (Rec.2020)'),
}


def command(*args):
    return subprocess.run(args, check=True, capture_output=True).stdout


@pytest.fixture
def source(tmp_path):
    source = tmp_path/'input.mp4'
    command('ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=128x72:rate=3',
            '-f','lavfi','-i','sine=frequency=440:sample_rate=48000','-t','1',
            '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',str(source))
    return source


def read_rgb(path):
    raw = command('ffmpeg','-v','error','-i',str(path),'-pix_fmt','gbrpf32le','-f','rawvideo','-')
    return np.frombuffer(raw,'<f4').reshape(-1,3,72,128)[:,[2,0,1]].transpose(0,2,3,1)


def official_pq(mode, rgb, white):
    pixels = np.array(rgb, dtype=np.float32, order='C', copy=True)
    if mode in OFFICIAL:
        config, view = OFFICIAL[mode]
        transform = ocio.DisplayViewTransform(src='ACEScg',display='Rec.2100-PQ - Display',view=view)
        ocio.Config.CreateFromBuiltinConfig(config).getProcessor(transform).getDefaultCPUProcessor().applyRGB(pixels)
    elif mode == 'openexr_acescg':
        # Invert the published coordinates independently in double precision,
        # then apply ST 2084 in nits; do not call the application's PQ helper.
        matrix = np.array([[.974894977924,.019599108637,.005505913439],
                           [.002179562798,.995535468893,.002284968309],
                           [.004797239684,.024532016635,.970670743682]])
        nits = np.linalg.solve(matrix,pixels.reshape(-1,3).T).T.reshape(pixels.shape)*white
        y = np.maximum(nits,0)/10000
        ym = y**(2610/16384)
        pixels = ((3424/4096+(2413/128)*ym)/(1+(2392/128)*ym))**(2523/32)
    return pixels


class RecordingWriter:
    def __init__(self, stream, frames):
        self.stream, self.frames = stream, frames
    def write(self, data):
        self.frames.append(data)
        return self.stream.write(data)
    def close(self):
        self.stream.close()


def record_video_input(frames):
    def open_encoder(*args, **kwargs):
        process = io.open_encoder(*args, **kwargs)
        if kwargs.get('encoder') == 'prores_4444':
            process.stdin = RecordingWriter(process.stdin, frames)
        return process
    return open_encoder


@pytest.mark.parametrize('mode',['openexr','openexr_acescg',*OFFICIAL])
def test_video_uses_zip_pixels_and_correct_display_transform(source,tmp_path,mode):
    archive = tmp_path/'frames.zip'
    movie = tmp_path/'frames.mov'
    request = app.ConversionRequest(str(source),str(archive),encoder=mode,preset='cinema',
                                    model_path=MODEL,backend='numpy')
    zip_result = app.run_conversion(request)
    original_zip = archive.read_bytes()
    completed = []
    sent_frames = []
    with patch('sdr2hdr.app.open_encoder',side_effect=record_video_input(sent_frames)):
        movie_result = app.run_conversion(replace(request,output_path=str(movie),exr_delivery='video'),
                                          app.ConversionCallbacks(on_complete=completed.append))
    assert movie_result.processed_frames == zip_result.processed_frames == 3
    assert completed == [movie_result] and completed[0].output_path == str(movie)
    assert archive.read_bytes() == original_zip
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ['frame_000001.exr','frame_000002.exr','frame_000003.exr','audio.mov']
        bundle.extractall(tmp_path/'extracted')
    # Encode an independent reference from the actual half-float ZIP frames.
    reference = tmp_path/'reference.mov'
    process = io.open_encoder(str(reference),str(source),io.ffprobe_video(str(source)),1000,encoder='prores_4444')
    assert len(sent_frames) == 3
    for path, raw in zip(sorted((tmp_path/'extracted').glob('*.exr')),sent_frames):
        expected = official_pq(mode,read_rgb(path)[0],203.0)
        actual = np.frombuffer(raw,'<u2').reshape(72,128,3)/65535
        # Isolate colour-transform accuracy before lossy compression: one RGB16
        # step covers float32 PQ arithmetic and rounding to encoder transport.
        np.testing.assert_allclose(actual,expected,atol=1/65535,rtol=0)
        process.stdin.write(raw)
    io.finalize_process(process,'reference encoder')
    # Compare storage using identical, independently checked encoder input.
    # A one-step RGB16 difference can cross a lossy codec quantisation boundary;
    # comparing differently rounded inputs after compression is not a transform test.
    np.testing.assert_array_equal(read_rgb(movie),read_rgb(reference))
    streams = json.loads(command('ffprobe','-v','error','-show_streams','-of','json',str(movie)))['streams']
    video = next(s for s in streams if s['codec_type']=='video')
    assert video['codec_name']=='prores' and video['codec_tag_string']=='ap4h'
    assert int(video['nb_frames']) == 3
    assert (video['color_primaries'],video['color_transfer'],video['color_space']) == ('bt2020','smpte2084','bt2020nc')
    audio = lambda p: command('ffmpeg','-v','error','-i',str(p),'-map','0:a','-f','s16le','-')
    assert audio(movie) == audio(source)
    assert not list(tmp_path.glob('.sdr2hdr-*')) and not list(tmp_path.glob('*.exr'))


@pytest.mark.parametrize('mode',OFFICIAL)
def test_forward_aces_is_official_transform_without_custom_clipping(mode):
    rgb = np.array([[[0,0,0],[.18,.18,.18],[4,.2,.01],[-.01,.1,.3],[16,16,16]]],np.float32)
    expected = official_pq(mode,rgb,203)
    np.testing.assert_array_equal(EXRVideoTransform(mode,203).to_pq(rgb),expected)


@pytest.mark.parametrize('mode',['openexr','openexr_acescg'])
def test_log_supports_both_delivery_choices(source,tmp_path,mode):
    request = app.LogConversionRequest(str(source),str(tmp_path/'log.zip'),encoder=mode)
    app.run_log_conversion(request)
    sent_frames = []
    with patch('sdr2hdr.app.open_encoder',side_effect=record_video_input(sent_frames)):
        result = app.run_log_conversion(replace(request,output_path=str(tmp_path/'log.mov'),exr_delivery='video'))
    assert result.processed_frames == 3
    with zipfile.ZipFile(tmp_path/'log.zip') as bundle:
        bundle.extractall(tmp_path/'extracted')
    assert len(sent_frames) == 3
    reference = tmp_path/'reference.mov'
    process = io.open_encoder(str(reference),str(source),io.ffprobe_video(str(source)),1000,encoder='prores_4444')
    for path, raw in zip(sorted((tmp_path/'extracted').glob('*.exr')),sent_frames):
        expected = official_pq(mode,read_rgb(path)[0],100)
        actual = np.frombuffer(raw,'<u2').reshape(72,128,3)/65535
        np.testing.assert_allclose(actual,expected,atol=1/65535,rtol=0)
        process.stdin.write(raw)
    io.finalize_process(process,'reference encoder')
    np.testing.assert_array_equal(read_rgb(tmp_path/'log.mov'),read_rgb(reference))



@pytest.mark.parametrize('action',['cancel','fail'])
def test_video_stage_cancel_or_failure_preserves_existing_file(source,tmp_path,action):
    output = tmp_path/'existing.mov';output.write_bytes(b'keep-existing')
    token = app.CancelToken()
    video_stage = False
    def status(message):
        nonlocal video_stage
        if message.startswith('EXRから'):
            video_stage = True
    def progress(processed,total,fps):
        if video_stage: token.cancel()
    callbacks = app.ConversionCallbacks(on_status=status,on_progress=progress if action=='cancel' else None)
    request = app.ConversionRequest(str(source),str(output),encoder='openexr_acescg_2_0',
                                    exr_delivery='video',model_path=MODEL,backend='numpy')
    if action=='fail':
        with patch('sdr2hdr.app.EXRVideoTransform.to_pq',side_effect=RuntimeError('output transform failed')):
            with pytest.raises(RuntimeError,match='output transform failed'):
                app.run_conversion(request,callbacks,token)
    else:
        assert app.run_conversion(request,callbacks,token).cancelled
    assert video_stage and output.read_bytes()==b'keep-existing'
    assert not list(tmp_path.glob('.sdr2hdr-*'))


def test_extension_validation_and_cli_choice(source,tmp_path):
    from sdr2hdr.cli import build_parser
    assert app.build_output_path(str(source),encoder='openexr_acescg_2_0').endswith('.zip')
    assert app.build_output_path(str(source),encoder='openexr_acescg_2_0',exr_delivery='video').endswith('.mov')
    args = build_parser().parse_args([str(source),'--model-path',MODEL,'--encoder','openexr_acescg_2_0','--exr-delivery','video'])
    assert args.exr_delivery=='video'
    with pytest.raises(ValueError,match='.mov'):
        app.validate_export_request(app.ConversionRequest(str(source),str(tmp_path/'bad.zip'),
                                    encoder='openexr',exr_delivery='video'))


class Var:
    def __init__(self,value=''): self.value=value
    def get(self): return self.value
    def set(self,value): self.value=value


def test_gui_delivery_preserves_manual_stem_and_separates_tabs():
    gui=SDR2HDRGUI.__new__(SDR2HDRGUI)
    gui._output_path_state={}
    gui.input_var=Var('/input/movie.mp4');gui.output_var=Var('/chosen/custom.zip')
    gui.log_input_var=Var('/input/log.mp4');gui.log_output_var=Var('/other/log.zip')
    gui._selected_encoder=Mock(return_value='openexr_acescg_2_0')
    gui._selected_log_encoder=Mock(return_value='openexr_acescg')
    gui.exr_delivery_var=Var(EXR_DELIVERY_OPTIONS['video'])
    gui.log_exr_delivery_var=Var(EXR_DELIVERY_OPTIONS['zip'])
    gui._sync_output_path();gui._sync_log_output_path()
    assert gui.output_var.get()=='/chosen/custom.mov'
    assert gui.log_output_var.get()=='/other/log.zip'
    gui.exr_delivery_var.set(EXR_DELIVERY_OPTIONS['zip']);gui._sync_output_path()
    assert gui.output_var.get()=='/chosen/custom.zip'
    hint=describe_mode_hint('openexr_acescg_2_0','','','','','video')
    assert 'ACES 2.0の公式出力変換' in hint and 'BT.2020/PQ' in hint and '保持する場合はZIP' in hint
