"""Output acceptance: inspect saved pixels, not just FFmpeg argument strings."""
import json
import zipfile
import platform
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import PyOpenColorIO as ocio

from sdr2hdr.app import ConversionRequest, ConversionCallbacks, CancelToken, run_conversion
from sdr2hdr.core import ProcessorConfig, SDRToHDRProcessor
from sdr2hdr.io import VideoInfo, open_encoder, finalize_process, restamp_prores_metadata
from sdr2hdr.output_color import OutputColorTransform

MODEL = str(Path(__file__).resolve().parents[1] / 'models/enhancement_model_20260310.pt')


def command(*args, data=None):
    return subprocess.run(args, input=data, capture_output=True, check=True).stdout


def source_video(root, frames=3):
    source = root / 'input.mp4'
    command('ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=128x72:rate=3',
            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
            '-t', str(frames / 3), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(source))
    return source


def streams(path):
    return json.loads(command('ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)))['streams']


def decode_rgb(path, width, height):
    raw = command('ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', '1',
                  '-pix_fmt', 'gbrpf32le', '-f', 'rawvideo', '-')
    return np.frombuffer(raw, '<f4').reshape(3, height, width).transpose(1, 2, 0)[..., [2, 0, 1]]


@pytest.mark.parametrize('encoder,depth', [('prores_422hq',10), ('prores_4444',10), ('prores_4444_xq',12)])
def test_prores_saved_headers_pixels_audio_and_real_precision(tmp_path, encoder, depth):
    if depth == 12 and platform.system() != 'Darwin':
        pytest.skip('12-bit VideoToolbox ProRes requires macOS; no 10-bit substitute')
    source = source_video(tmp_path)
    output = tmp_path / 'hdr.mov'
    # A 16-bit neutral ramp includes values between adjacent 10-bit levels.
    ramp = np.tile(np.linspace(.1, .9, 4096, dtype=np.float32), (64, 1))
    pixels = np.repeat(np.round(ramp[...,None]*65535).astype(np.uint16), 3, axis=2)
    process = open_encoder(str(output), str(source), VideoInfo(4096,64,3,3,'rgb48le',1.,'progressive'),
                           1000, encoder=encoder)
    for _ in range(3):
        process.stdin.write(pixels.tobytes())
    finalize_process(process, 'encoder')
    original_pixels = command('ffmpeg','-v','error','-i',str(output),'-pix_fmt','yuv444p12le','-f','rawvideo','-')
    original_audio = command('ffmpeg','-v','error','-i',str(output),'-map','0:a','-f','s16le','-')
    restamp_prores_metadata(str(output))
    result = streams(output)
    video = next(s for s in result if s['codec_type']=='video')
    # ProRes 4444 decodes as 12-bit even when its encoder input is 10-bit.
    assert int(video['bits_per_raw_sample']) == (10 if encoder == 'prores_422hq' else 12)
    assert (video['color_primaries'],video['color_transfer'],video['color_space']) == ('bt2020','smpte2084','bt2020nc')
    assert int(video['nb_frames']) == 3
    # Inspect both MOV sample-description and ProRes frame header independently.
    container = output.read_bytes()
    import re
    colour_atoms = list(re.finditer(b'colrnclc', container))
    assert colour_atoms
    for atom in colour_atoms:
        assert struct.unpack('>HHH',container[atom.start()+8:atom.start()+14]) == (9,16,9)
    packet=command('ffmpeg','-v','error','-i',str(output),'-map','0:v','-c','copy','-frames:v','1','-f','image2pipe','-')
    assert packet[4:8] == b'icpf'
    assert packet[22:25] == bytes([9,16,9])
    decoded=command('ffmpeg','-v','error','-i',str(output),'-pix_fmt','yuv444p12le','-f','rawvideo','-')
    assert decoded == original_pixels
    assert command('ffmpeg','-v','error','-i',str(output),'-map','0:a','-f','s16le','-') == original_audio
    assert len([s for s in result if s['codec_type']=='audio']) == 1
    if encoder == 'prores_4444_xq':
        luma=np.frombuffer(decoded,'<u2').reshape(3,3,64,4096)[0,0]
        assert len(np.unique(luma)) > 1024, 'A 12-bit header alone is not evidence of 12-bit precision'


def test_prores_metadata_failure_keeps_original(tmp_path):
    output=tmp_path/'invalid.mov';output.write_bytes(b'not a movie')
    with pytest.raises(subprocess.CalledProcessError):
        restamp_prores_metadata(str(output))
    assert output.read_bytes() == b'not a movie'
    assert list(tmp_path.iterdir()) == [output]


# ITU-R BT.2100-3 table 5: inverse OOTF at 1000 nit, gamma 1.2, then OETF.
# This oracle does not call OCIO or the application's colour functions.
def reference_hlg(nits):
    rgb=np.asarray(nits,dtype=np.float64)/1000
    luminance=rgb@np.array([.2627,.6780,.0593])
    scene=rgb*np.maximum(luminance,1e-30)[...,None]**(-1/6)
    a=.17883277;b=1-4*a;c=.5-a*np.log(4*a)
    return np.where(scene<=1/12,np.sqrt(3*scene),a*np.log(np.maximum(12*scene-b,1e-30))+c)


def test_hlg_colours_match_itu_and_survive_ten_bit_encoding(tmp_path):
    colours=np.array([[0,0,0],[1,1,1],[10,10,10],[100,100,100],[203,203,203],
                      [500,500,500],[1000,1000,1000],[400,100,50],[50,400,100],[100,50,400]],np.float32)
    nits=np.repeat(np.repeat(colours[None],64,0),64,1)
    rendered=OutputColorTransform('hevc_hlg',1000).render(nits)
    expected=reference_hlg(colours)
    # Half of a 16-bit rounding step plus float32 arithmetic error.
    np.testing.assert_allclose(rendered[32,32::64]/65535,expected,atol=1/65535,rtol=0)
    source=source_video(tmp_path);output=tmp_path/'hlg.mp4'
    encoder=open_encoder(str(output),str(source),VideoInfo(640,64,1,1,'rgb48le',1.,'progressive'),
                         1000,encoder='hevc_hlg',x265_preset='ultrafast',x265_crf=0)
    encoder.stdin.write(rendered.tobytes());finalize_process(encoder,'encoder')
    video=streams(output)[0]
    assert (video['pix_fmt'],video['color_primaries'],video['color_transfer']) == ('yuv420p10le','bt2020','arib-std-b67')
    frame=json.loads(command('ffprobe','-v','error','-show_frames','-of','json',str(output)))['frames'][0]
    assert not any(s['side_data_type'] in {'Mastering display metadata','Content light level metadata'}
                   for s in frame.get('side_data_list',[]))
    # Sample patch interiors, excluding chroma-subsampling boundaries. Two 10-bit
    # code steps cover RGB<->limited-range YCbCr rounding; this is not a visual-quality score.
    decoded=decode_rgb(output,640,64)[32,32::64]
    np.testing.assert_allclose(decoded,expected,atol=2/876,rtol=0)


@pytest.mark.parametrize('mode,config,view,tolerance',[
    ('openexr_acescg_1_3','cg-config-v2.2.0_aces-v1.3_ocio-v2.4','ACES 1.1 - HDR Video (1000 nits & Rec.2020 lim)',2/1023),
    ('openexr_acescg_2_0','cg-config-v4.0.0_aces-v2.0_ocio-v2.5','ACES 2.0 - HDR 1000 nits (Rec.2020)',1/1023),
])
def test_aces_sequence_real_model_forward_view_and_metadata(tmp_path,mode,config,view,tolerance):
    source=source_video(tmp_path)
    request=ConversionRequest(str(source),str(tmp_path/'frames.zip'),encoder=mode,
                              model_path=MODEL,backend='numpy')
    result=run_conversion(request)
    with zipfile.ZipFile(tmp_path/'frames.zip') as archive:
        assert archive.namelist() == ['frame_000001.exr','frame_000002.exr','frame_000003.exr','audio.mov']
        archive.extractall(tmp_path/'extracted')
    assert not list(tmp_path.glob('*.exr'))
    assert command('ffmpeg','-v','error','-i',str(source),'-map','0:a','-f','s16le','-') == command(
        'ffmpeg','-v','error','-i',str(tmp_path/'extracted/audio.mov'),'-map','0:a','-f','s16le','-')
    files=sorted((tmp_path/'extracted').glob('*.exr'))
    assert result.processed_frames == len(files) == 3
    # Render the same decoded SDR frame through the existing PQ path, with the
    # actual model and preset (not a substitute enhancer).
    from sdr2hdr.app import build_request_config,build_enhancer
    cfg,_,_=build_request_config(ConversionRequest(str(source),'unused',model_path=MODEL,backend='numpy'))
    processor=SDRToHDRProcessor(cfg,build_enhancer(request,None))
    raw=command('ffmpeg','-v','error','-i',str(source),'-frames:v','1','-pix_fmt','bgr24','-f','rawvideo','-')
    expected=processor.process_frame(np.frombuffer(raw,np.uint8).reshape(72,128,3))/65535
    aces=decode_rgb(files[0],128,72).copy()
    assert np.isfinite(aces).all()
    transform=ocio.DisplayViewTransform(src='ACEScg',display='Rec.2100-PQ - Display',view=view)
    official = ocio.Config.CreateFromBuiltinConfig(config)
    forward = official.getProcessor(transform).getDefaultCPUProcessor()
    reference = np.ascontiguousarray(expected, dtype=np.float32)
    official.getProcessor(transform, ocio.TRANSFORM_DIR_INVERSE).getDefaultCPUProcessor().applyRGB(reference)
    forward.applyRGB(reference)
    forward.applyRGB(aces)
    # Compare integration against the official round trip, including its range
    # handling. This does not require identity with the pre-inverse display image.
    np.testing.assert_allclose(aces,reference,atol=tolerance,rtol=0)
    for path in files:
        header=command('exrheader',str(path)).decode()
        for text in ['16-bit floating-point','chromaticities','lin_ap1_scene',config,view]:
            assert text in header
        assert 'whiteLuminance' not in header


@pytest.mark.parametrize('mode,config,view',[
    ('openexr_acescg_1_3','cg-config-v2.2.0_aces-v1.3_ocio-v2.4','ACES 1.1 - HDR Video (1000 nits & Rec.2020 lim)'),
    ('openexr_acescg_2_0','cg-config-v4.0.0_aces-v2.0_ocio-v2.5','ACES 2.0 - HDR 1000 nits (Rec.2020)'),
])
def test_saved_aces_values_match_official_inverse(tmp_path,mode,config,view):
    # Include dark saturated colours that exercise the official range handling.
    colours=np.array([[0,0,0],[1,1,1],[203,203,203],[1000,1000,1000],
                      [400,100,50],[50,400,100],[100,50,400],[.25,.06,5]],np.float32)
    nits=np.repeat(np.repeat(colours[None],64,0),64,1)
    # Independent ST 2084 encoding in the application's float32 transport precision;
    # do not reuse the application's PQ function. The official inverse is sensitive
    # to small PQ differences near the ACES 1.3 range boundary.
    nits32=nits.astype(np.float32)
    y=np.clip(nits32/np.float32(10000),0,1)
    m1=np.float32(2610/16384)
    m2=np.float32(2523/4096*128)
    c1=np.float32(3424/4096)
    c2=np.float32(2413/4096*32)
    c3=np.float32(2392/4096*32)
    ym=np.power(y,m1)
    reference=np.ascontiguousarray(np.power((c1+c2*ym)/(1+c3*ym),m2),dtype=np.float32)
    transform=ocio.DisplayViewTransform(src='ACEScg',display='Rec.2100-PQ - Display',view=view)
    ocio.Config.CreateFromBuiltinConfig(config).getProcessor(
        transform,ocio.TRANSFORM_DIR_INVERSE).getDefaultCPUProcessor().applyRGB(reference)
    source=source_video(tmp_path);output=tmp_path/'frame.exr'
    encoder=open_encoder(str(output),str(source),VideoInfo(512,64,1,1,'gbrpf32le',1.,'progressive'),
                         1000,encoder=mode)
    encoder.stdin.write(OutputColorTransform(mode,1000).render(nits).tobytes())
    finalize_process(encoder,'encoder')
    saved=decode_rgb(output,512,64)
    # One half-float spacing covers storage rounding plus float32 PQ arithmetic.
    # Compare ACES values directly: a forward view could hide differences by clipping.
    spacing=np.abs(np.spacing(reference.astype(np.float16)).astype(np.float32))
    assert np.isfinite(saved).all()
    assert np.all(np.abs(saved-reference)<=spacing), 'Saved ACES values differ from the official inverse'


@pytest.mark.parametrize('mode',['hevc_hlg','openexr_acescg_1_3','openexr_acescg_2_0'])
def test_new_outputs_reject_unsupported_peak_before_writing(tmp_path,mode):
    source=source_video(tmp_path)
    with pytest.raises(ValueError,match='1000 nit'):
        run_conversion(ConversionRequest(str(source),str(tmp_path/('output.mp4' if mode=='hevc_hlg' else 'output.zip')),encoder=mode,
                                         model_path=MODEL,backend='numpy',peak_nits=2000))
    assert not list(tmp_path.glob('output*'))


@pytest.mark.parametrize('keep',[False,True])
def test_aces_cancel_only_touches_this_sequence(tmp_path,keep):
    source=source_video(tmp_path); sentinel=tmp_path/'frames.zip';sentinel.write_bytes(b'keep')
    token=CancelToken()
    def progress(processed,total,fps):
        if processed==1: token.cancel()
    result=run_conversion(ConversionRequest(str(source),str(tmp_path/'frames.zip'),
                          encoder='openexr_acescg_2_0',model_path=MODEL,backend='numpy',
                          keep_partial_output_on_cancel=keep),ConversionCallbacks(on_progress=progress),token)
    assert result.cancelled and result.processed_frames==1
    assert sentinel.read_bytes()==b'keep'
    assert not list(tmp_path.glob('*.exr'))
    assert not list(tmp_path.glob('.sdr2hdr-*'))
