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
from test_compare_images import avif_reference, jpeg_reference

pytestmark = pytest.mark.skipif(
    sys.platform != 'darwin' or os.environ.get('SDR2HDR_TEST_NATIVE_HDR') != '1',
    reason='Requires macOS native GUI permission and an EDR display',
)


def test_actual_still_pixels_colour_precision_and_stream_cleanup(tmp_path):
    import cv2
    import tkinter as tk
    from sdr2hdr.gui import SDR2HDRGUI

    patches=np.array([[.25,.25,.25],[.5,.5,.5],[.58,.58,.58],[.75,.75,.75],
                      [.9,.9,.9],[.7,.5,.4],[.4,.7,.5],[.5,.4,.7]])
    ramp=np.repeat((.5+np.arange(16)/2048)[:,None],3,axis=1)
    colours=np.vstack([patches,ramp])
    pixels=np.repeat(np.repeat(np.rint(colours*65535).astype(np.uint16)[None],320,0),32,1)
    height,width=pixels.shape[:2];xs=np.arange(len(colours))*32+16
    source=tmp_path/'sdr.png';cv2.imwrite(str(source),np.full((height,width,3),128,np.uint8))
    paths={suffix:tmp_path/('hdr.'+suffix) for suffix in ['png','jxl','tif','avif','jpg']}
    references={}
    for suffix,path in paths.items():
        assert save_image_hdr(str(path),pixels)
        if suffix=='avif':
            references[suffix]=avif_reference(path,width,height,xs,height//2)
        elif suffix=='jpg':
            references[suffix]=jpeg_reference(path,width,height,tmp_path/'official.raw')[height//2,xs]
        else:
            references[suffix]=pixels[height//2,xs]/65535

    root=tk.Tk();app=SDR2HDRGUI(root);root.update();view=app.compare_view
    loaded=[];original_load=view._load
    def observe_load(pair,info):
        original_load(pair,info)
        loaded.append(pair.hdr_path)
    view._load=observe_load
    def until(condition,seconds=25):
        deadline=time.monotonic()+seconds
        while not condition():
            root.update()
            assert time.monotonic()<deadline,view.message.get()
            time.sleep(.005)
    gl=ctypes.CDLL('/System/Library/Frameworks/OpenGL.framework/OpenGL')
    gl.glReadPixels.argtypes=[ctypes.c_int]*4+[ctypes.c_uint]*2+[ctypes.c_void_p]
    gl.glBindFramebuffer.argtypes=[ctypes.c_uint,ctypes.c_uint]
    gl.glReadBuffer.argtypes=[ctypes.c_uint];gl.glGetError.restype=ctypes.c_uint
    report={}
    try:
        for index,(suffix,path) in enumerate(paths.items()):
            view.add_pair(str(source),str(path),image=True)
            if index:
                view.selection.current(index);view._select()
            until(lambda: str(path) in loaded and view.controller is not None and
                  view.controller.ready and view._load_started is None)
            controller=view.controller;surface=controller.hdr.surface
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
            actual=np.array(actual)
            # Two half-float ULPs at PQ values [0.5,1), covering the intermediate
            # and final RGBA16Float surfaces. No extra allowance for compression:
            # AVIF and JPEG references are the decoded file, not pre-save pixels.
            error=float(np.max(abs(actual-references[suffix])))
            assert error<=2**-10,(suffix,error,actual,references[suffix])
            assert errors and not any(errors)
            params=controller.hdr.mpv.video_params
            assert params['primaries']=='bt.2020' and params['gamma']=='pq'
            assert 'Active' in surface.output_status()
            until(lambda: not view.message.get())
            assert controller.image and controller.sdr.mpv.pause and controller.hdr.mpv.pause
            assert not view.controls.winfo_ismapped()
            unique=int(len(np.unique(actual[8:,0])))
            if suffix in {'png','jxl','tif'}:
                # Sixteen steps across <2/255 must not collapse to an 8-bit grid.
                assert unique>=12,(suffix,unique)
            assert len(controller.hdr.mpv._python_streams)==(0 if suffix=='png' else 1)
            report[suffix]={'maximum_pq_error':error,'ramp_levels':unique,
                            'colour':view.hdr_info.get(),'state':surface.output_status()}
        sdr,hdr=view.controller.sdr,view.controller.hdr
    finally:
        app._close()
    assert sdr.closed and hdr.closed
    assert not sdr.mpv._python_streams and not hdr.mpv._python_streams
    (tmp_path/'render-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('render report:',tmp_path/'render-results.json',flush=True)
