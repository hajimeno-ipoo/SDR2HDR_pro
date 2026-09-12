"""Opt-in macOS measurement of the real comparison render targets.

Run with SDR2HDR_TEST_NATIVE_HDR=1 on an EDR display. No screen capture is used
as a photometric reference: pixels are read from the application's actual FBO.
"""
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sdr2hdr.io import save_image_hdr
from sdr2hdr.hdr_guidance import linear_nits_to_pq
from test_compare_images import avif_reference, jpeg_reference

pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin' or os.environ.get('SDR2HDR_TEST_NATIVE_HDR') != '1',
    reason='Requires macOS native GUI permission and an EDR display',
)


def test_mpv_still_decoder_pixels_colour_precision_and_stream_cleanup(tmp_path):
    import cv2
    import tkinter as tk
    from sdr2hdr.compare_controller import CompareController
    from sdr2hdr.compare_images import prepare_image
    from sdr2hdr.compare_view import color_label
    from sdr2hdr.io import ffprobe_comparison
    from sdr2hdr.mpv_player import MpvPlayer
    from sdr2hdr.native_surface import create_surface

    patches=np.array([[.25,.25,.25],[.5,.5,.5],[.58,.58,.58],[.75,.75,.75],
                      [.9,.9,.9],[.7,.5,.4],[.4,.7,.5],[.5,.4,.7]])
    ramp=np.repeat((.5+np.arange(16)/2048)[:,None],3,axis=1)
    colours=np.vstack([patches,ramp])
    pixels=np.repeat(np.repeat(np.rint(colours*65535).astype(np.uint16)[None],320,0),32,1)
    height,width=pixels.shape[:2];xs=np.arange(len(colours))*32+16
    source=tmp_path/'sdr.png';cv2.imwrite(str(source),np.full((height,width,3),128,np.uint8))
    folder=tmp_path/'日本語のフォルダ';folder.mkdir()
    paths={suffix:folder/('てすと.'+suffix) for suffix in ['png','jxl','tif','avif','jpg']}
    references={}
    for suffix,path in paths.items():
        assert save_image_hdr(str(path),pixels)
        if suffix=='avif':
            references[suffix]=avif_reference(path,width,height,xs,height//2)
        elif suffix=='jpg':
            references[suffix]=jpeg_reference(path,width,height,tmp_path/'official.raw')[height//2,xs]
        else:
            references[suffix]=pixels[height//2,xs]/65535

    # libmpv still decoding remains in use on Windows. This opt-in Mac FBO
    # test checks that decoder path, not Mac's new AppKit image display.
    root=tk.Tk();root.geometry('800x800')
    widgets=[tk.Frame(root,width=800,height=400) for _ in range(2)]
    for widget in widgets: widget.pack()
    root.update()
    controller=CompareController(MpvPlayer(create_surface(widgets[0]),hdr=False),
                                 MpvPlayer(create_surface(widgets[1]),hdr=True))
    def until(condition,seconds=25):
        deadline=time.monotonic()+seconds
        while not condition():
            root.update()
            errors=controller.sdr.poll()+controller.hdr.poll()
            assert not errors,errors
            assert time.monotonic()<deadline
            time.sleep(.005)
    gl=ctypes.CDLL('/System/Library/Frameworks/OpenGL.framework/OpenGL')
    gl.glReadPixels.argtypes=[ctypes.c_int]*4+[ctypes.c_uint]*2+[ctypes.c_void_p]
    gl.glBindFramebuffer.argtypes=[ctypes.c_uint,ctypes.c_uint]
    gl.glReadBuffer.argtypes=[ctypes.c_uint];gl.glGetError.restype=ctypes.c_uint
    report={}
    try:
        for index,(suffix,path) in enumerate(paths.items()):
            sdr_info=prepare_image(str(source),ffprobe_comparison(str(source)))
            hdr_info=prepare_image(str(path),ffprobe_comparison(str(path)),app_hdr_output=True)
            controller.load_pair(str(source),str(path),sdr_info,hdr_info,image=True)
            until(lambda: controller.ready)
            surface=controller.hdr.surface
            # Read the shared linear image before macOS applies display mapping.
            # Lossy AVIF/JPEG patches can vary even near their centres. Inspect
            # at 1:1 pixels so display interpolation is not compared to a single
            # source sample. Both dimensions fit this native framebuffer.
            controller.hdr.mpv['video-unscaled']='yes'
            actual=[];errors=[];old=surface.draw_callback
            def capture(fbo,w,h):
                old(fbo,w,h)
                gl.glBindFramebuffer(0x8CA8,fbo);gl.glReadBuffer(0x8CE0)
                assert w>=width and h>=height
                left=(w-width)//2
                bottom=(h-height)//2
                samples=[]
                for i in range(len(colours)):
                    rgba=(ctypes.c_float*4)()
                    gl.glReadPixels(left+int(xs[i]),bottom+height-1-height//2,
                                    1,1,0x1908,0x1406,rgba)
                    samples.append(list(rgba)[:3])
                actual[:]=samples;errors.append(gl.glGetError())
            callback=surface.draw_type(capture)
            surface.bridge.comparison_layer_set_callback(surface.layer_pointer,callback)
            surface.pending.set()
            try:
                until(lambda: bool(actual))
            finally:
                surface.bridge.comparison_layer_set_callback(surface.layer_pointer,old)
            actual=linear_nits_to_pq(np.array(actual)*203.0)
            # Keep the existing PQ-domain precision requirement (2^-10),
            # converting the new linear buffer back to PQ for comparison.
            # No extra allowance for compression:
            # AVIF and JPEG references are the decoded file, not pre-save pixels.
            error=float(np.max(abs(actual-references[suffix])))
            assert error<=2**-10,(suffix,error,actual,references[suffix])
            assert errors and not any(errors)
            params=controller.hdr.mpv.video_params
            assert params['primaries']=='bt.2020' and params['gamma']=='pq'
            # macOS increases available EDR asynchronously after layer activation.
            until(lambda: 'Active' in surface.output_status())
            assert controller.image and controller.sdr.mpv.pause and controller.hdr.mpv.pause
            unique=int(len(np.unique(actual[8:,0])))
            if suffix in {'png','jxl','tif'}:
                # In the tested interval, an 8-bit grid can contain at most
                # this many distinct codes, including the verified error band.
                # A fixed demand for 12/16 distinct steps had no precision basis:
                # transfer-function round trips can merge adjacent half-ULPs.
                lower=references[suffix][8:,0].min()-2**-10
                upper=references[suffix][8:,0].max()+2**-10
                max_8bit_levels=int(np.floor(upper*255)-np.ceil(lower*255)+1)
                assert unique>max_8bit_levels,(suffix,unique,max_8bit_levels)
            assert len(controller.hdr.mpv._python_streams)==(0 if suffix=='png' else 1)
            report[suffix]={'maximum_pq_error':error,'ramp_levels':unique,
                            'colour':color_label(hdr_info),'state':surface.output_status()}
        sdr,hdr=controller.sdr,controller.hdr
    finally:
        controller.close()
        root.destroy()
    assert sdr.closed and hdr.closed
    assert not sdr.mpv._python_streams and not hdr.mpv._python_streams
    (tmp_path/'render-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('render report:',tmp_path/'render-results.json',flush=True)


def test_actual_linear_output_matches_pq_and_hlg_reference(tmp_path):
    import subprocess
    import tkinter as tk
    from sdr2hdr.compare_images import prepare_image
    from sdr2hdr.hdr_guidance import linear_nits_to_pq
    from sdr2hdr.io import ffprobe_comparison
    from sdr2hdr.mpv_player import MpvPlayer
    from sdr2hdr.native_surface import create_surface

    # The shared buffer must retain source luminance before OS display mapping.
    # HLG uses BT.2100's inverse OETF and 1000-nit reference OOTF (gamma 1.2),
    # as used in mpv v0.41.0 video/out/gpu/video_shaders.c.
    codes=np.rint(np.array([.25,.4,.5,.58,.65,.7,.7518271])*65535).astype(np.uint16)
    pixels=np.broadcast_to(np.repeat(codes,64)[None,:,None],(320,448,3)).copy()
    pq_path=tmp_path/'pq.png';assert save_image_hdr(str(pq_path),pixels)
    pq_info=prepare_image(str(pq_path),ffprobe_comparison(str(pq_path)),app_hdr_output=True)
    assert 'source_peak_nits' not in pq_info
    hlg_path=tmp_path/'hlg.mkv'
    subprocess.run(['ffmpeg','-v','error','-f','rawvideo','-pixel_format','rgb48le',
                    '-video_size','448x320','-framerate','1','-i','pipe:0','-frames:v','1',
                    '-vf','setparams=range=full:color_primaries=bt2020:color_trc=arib-std-b67:colorspace=gbr',
                    '-c:v','ffv1','-pix_fmt','gbrp16le','-color_primaries','bt2020',
                    '-color_trc','arib-std-b67','-colorspace','rgb','-color_range','pc',
                    str(hlg_path)],input=pixels.astype('<u2').tobytes(),check=True)
    hlg_info=ffprobe_comparison(str(hlg_path))
    assert hlg_info['color_transfer']=='arib-std-b67'
    hlg=codes.astype(float)/65535
    scene=np.where(hlg<=.5,hlg**2/3,(np.exp((hlg-.55991073)/.17883277)+.28466892)/12)
    hlg_pq=linear_nits_to_pq(1000*scene**1.2)
    root=tk.Tk();root.geometry('720x400')
    widget=tk.Frame(root,width=700,height=360);widget.pack();root.update()
    surface=create_surface(widget);player=MpvPlayer(surface,hdr=True)
    gl=ctypes.CDLL('/System/Library/Frameworks/OpenGL.framework/OpenGL')
    gl.glReadPixels.argtypes=[ctypes.c_int]*4+[ctypes.c_uint]*2+[ctypes.c_void_p]
    gl.glBindFramebuffer.argtypes=[ctypes.c_uint,ctypes.c_uint]
    gl.glReadBuffer.argtypes=[ctypes.c_uint];gl.glGetError.restype=ctypes.c_uint
    actual=[];old=surface.draw_callback
    def capture(fbo,w,h):
        old(fbo,w,h);gl.glBindFramebuffer(0x8CA8,fbo);gl.glReadBuffer(0x8CE0)
        samples=[]
        for x in np.arange(7)*64+32:
            rgba=(ctypes.c_float*4)()
            gl.glReadPixels((w-448)//2+int(x),h//2,1,1,0x1908,0x1406,rgba)
            samples.append(list(rgba)[:3])
        actual.append((np.array(samples),gl.glGetError()))
    callback=surface.draw_type(capture)
    surface.bridge.comparison_layer_set_callback(surface.layer_pointer,callback)
    def until(condition):
        deadline=time.monotonic()+20
        while not condition():
            root.update();assert not player.poll()
            assert time.monotonic()<deadline
            time.sleep(.005)
    report={}
    try:
        for path,info,pq,peak in [(pq_path,pq_info,codes/65535,None),
                                  (hlg_path,hlg_info,hlg_pq,1000)]:
            actual.clear()
            player.load(str(path),info);player.mpv['video-unscaled']='yes'
            until(lambda:player.loaded and bool(actual))
            assert not player.mpv['vf']
            if path==hlg_path:
                assert player.mpv.video_params['gamma']=='hlg'
            else:
                assert surface._metadata_range is None
            assert player.mpv['target-trc']=='linear'
            measured,error=actual[-1];assert error==0
            measured_pq=linear_nits_to_pq(measured*203.0)
            deviation=float(np.max(abs(measured_pq-pq[:,None])))
            # Preserve the earlier PQ-domain accuracy requirement.
            assert deviation<=2**-10,(path.name,deviation,measured,pq)
            assert np.all(np.diff(measured[:,0])>0)
            assert measured.max()>1  # HDR values survive until the OS stage.
            assert surface.layer.EDRMetadata() is not None
            if path==hlg_path:
                assert abs(surface._metadata_range[1]-peak)<.15
            else:
                assert surface._metadata_range is None
            report[path.name]={'max_pq_error':deviation,'linear':measured[:,0].tolist(),
                               'source_luminance_range':surface._metadata_range}
    finally:
        surface.bridge.comparison_layer_set_callback(surface.layer_pointer,old)
        player.close();root.destroy()
    (tmp_path/'linear-hdr.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
