"""Measure rendered zoom, not just the configured mpv option."""
import ctypes
import os
import sys
import time
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'darwin' or os.environ.get('SDR2HDR_TEST_NATIVE_HDR') != '1', reason='Requires native macOS GUI')


def test_video_and_image_zoom_changes_both_rendered_sizes_and_fit_resets(tmp_path):
    import tkinter as tk
    from sdr2hdr.gui import SDR2HDRGUI
    from sdr2hdr.io import save_image_hdr
    from sdr2hdr.app import run_log_conversion, LogConversionRequest

    pixels=np.zeros((144,256,3),np.uint8);pixels[60:84,108:116]=180;pixels[60:84,140:148]=180
    source=tmp_path/'source.png';cv2.imwrite(str(source),pixels)
    hdr_image=tmp_path/'hdr.png';save_image_hdr(str(hdr_image),(pixels.astype(np.uint16)*257))
    video=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-loop','1','-i',str(source),'-t','1','-r','12',
                    '-c:v','libx264','-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709',
                    '-colorspace','bt709',str(video)],check=True)
    hdr_video=tmp_path/'hdr.mov'
    run_log_conversion(LogConversionRequest(str(video),str(hdr_video),encoder='prores_422hq'))
    root=tk.Tk();app=SDR2HDRGUI(root);root.geometry('1680x900');view=app.compare_view
    def until(predicate):
        deadline=time.monotonic()+25
        while not predicate():
            for surface, _ in originals:
                surface.pending.set()
            root.update()
            assert time.monotonic()<deadline,view.message.get()
            time.sleep(.005)
    gl=ctypes.CDLL('/System/Library/Frameworks/OpenGL.framework/OpenGL')
    gl.glReadPixels.argtypes=[ctypes.c_int]*4+[ctypes.c_uint]*2+[ctypes.c_void_p]
    gl.glBindFramebuffer.argtypes=[ctypes.c_uint,ctypes.c_uint]
    gl.glReadBuffer.argtypes=[ctypes.c_uint]
    captures={};callbacks=[];originals=[]
    centres={};capture_centres=False
    try:
        for index,(sdr,hdr,still) in enumerate([(source,hdr_image,True),(video,hdr_video,False)]):
            view.add_pair(str(sdr),str(hdr),image=still)
            if index:
                view.selection.current(index);view._select()
            until(lambda:view.controller is not None and view.controller.ready and view._load_started is None and not view.zoom_combo.instate(['disabled']))
            assert view.zoom.get()=='100%'
            assert not view.zoom_combo.instate(['disabled'])
            assert bool(view.controls.winfo_ismapped()) is not still
            for widget in (view.zoom_combo,view.fit_button,view.controls if not still else view.zoom_combo,view.message_label):
                assert widget.winfo_rooty()+widget.winfo_height()<=view.winfo_rooty()+view.winfo_height()
            if not callbacks:
                for side,player in [('sdr',view.controller.sdr),('hdr',view.controller.hdr)]:
                    surface=player.surface;original=surface.draw_callback
                    def capture(fbo,w,h,side=side,original=original):
                        original(fbo,w,h)
                        gl.glBindFramebuffer(0x8CA8,fbo);gl.glReadBuffer(0x8CE0)
                        data=(ctypes.c_float*(w*4))()
                        gl.glReadPixels(0,h//2,w,1,0x1908,0x1406,data)
                        row=np.array(data).reshape(w,4)[:,:3].mean(axis=1)
                        if capture_centres:
                            full=(ctypes.c_float*(w*h*4))()
                            gl.glReadPixels(0,0,w,h,0x1908,0x1406,full)
                            luminance=np.array(full).reshape(h,w,4)[:,:,:3].mean(axis=2)
                            ys,xs=np.nonzero(luminance>(luminance.max()+luminance.min())/2)
                            if xs.size:
                                centres.setdefault(side,[]).append((float(xs.mean()),float(ys.mean())))
                        # Distance between symmetric patch centres avoids measuring the
                        # transfer-dependent brightness threshold as a geometric edge.
                        left=np.flatnonzero(row[:w//2] > (row.max()+row.min())/2)
                        right=np.flatnonzero(row[w//2:] > (row.max()+row.min())/2)+w//2
                        if left.size and right.size:
                            captures.setdefault(side,[]).append((float(right.mean()-left.mean()), min(w,h*256/144)))
                    callback=surface.draw_type(capture)
                    callbacks.append(callback);originals.append((surface,original))
                    surface.bridge.comparison_layer_set_callback(surface.layer_pointer,callback)
            widths={}
            for percent in (100,50,200,400,25):
                captures.clear()
                view.zoom.set(f'{percent}%');view.zoom_combo.event_generate('<<ComboboxSelected>>')
                until(lambda: all(len(captures.get(side,[]))>=3 for side in ('sdr','hdr')))
                widths[percent]={side:captures[side][-1][0] for side in ('sdr','hdr')}
                assert view.controller.sdr.mpv.pause and view.controller.hdr.mpv.pause
            # Verify actual enlargement/reduction, without imposing pixel-exact
            # geometry on libmpv's displayed, resampled image.
            for side in ('sdr','hdr'):
                assert 0 < widths[25][side] < widths[50][side] < widths[100][side] < widths[200][side] < widths[400][side]
            captures.clear()
            view.fit_button.invoke()
            until(lambda: all(len(captures.get(side,[]))>=3 for side in ('sdr','hdr')))
            for side in ('sdr','hdr'):
                assert captures[side][-1][0] == widths[100][side]
            assert view.zoom.get()=='100%'
            assert view.controller.sdr.mpv.video_zoom==view.controller.hdr.mpv.video_zoom==0
            # Both surfaces must receive dragging and move the actual render.
            capture_centres=True
            view.zoom.set('200%');view._zoom_changed()
            centres.clear()
            until(lambda: all(len(centres.get(side,[]))>=3 for side in ('sdr','hdr')))
            centred={side:centres[side][-1] for side in ('sdr','hdr')}
            for widget in (view.sdr_surface,view.hdr_surface):
                view.controller.set_pan(0,0)
                root.update()
                x,y=widget.winfo_width()//2,widget.winfo_height()//2
                import AppKit
                player=view.controller.sdr if widget is view.sdr_surface else view.controller.hdr
                native=player.surface.view
                def native_event(kind,px,py):
                    local=AppKit.NSMakePoint(px,py if native.isFlipped() else native.bounds().size.height-py)
                    point=native.convertPoint_toView_(local,None)
                    return AppKit.NSEvent.mouseEventWithType_location_modifierFlags_timestamp_windowNumber_context_eventNumber_clickCount_pressure_(
                        kind,point,0,0,native.window().windowNumber(),None,0,1,1)
                native.mouseDown_(native_event(AppKit.NSEventTypeLeftMouseDown,x,y))
                until(lambda:view._pan_drag is not None)
                centres.clear()
                native.mouseDragged_(native_event(AppKit.NSEventTypeLeftMouseDragged,x+24,y+12))
                native.mouseUp_(native_event(AppKit.NSEventTypeLeftMouseUp,x+24,y+12))
                until(lambda: all(len(centres.get(side,[]))>=3 for side in ('sdr','hdr')))
                assert view._pan_drag is None
                assert view.controller.sdr.mpv.video_pan_x==view.controller.hdr.mpv.video_pan_x>0
                assert view.controller.sdr.mpv.video_pan_y==view.controller.hdr.mpv.video_pan_y>0
                for side in ('sdr','hdr'):
                    assert centres[side][-1][0]>centred[side][0]+10
                    assert centres[side][-1][1]<centred[side][1]-5
            view.fit_button.invoke()
            assert view.controller.pan==(0,0)
            assert view.controller.sdr.mpv.video_pan_x==view.controller.hdr.mpv.video_pan_x==0
            assert view.controller.sdr.mpv.video_pan_y==view.controller.hdr.mpv.video_pan_y==0
            capture_centres=False
            if not still:
                view._toggle()
                until(lambda:(view.controller.sdr.get_position() or 0)>0.1)
                assert view.controller.playing
                view._toggle()
            print('image' if still else 'video',widths)
    finally:
        for surface,original in originals:
            surface.bridge.comparison_layer_set_callback(surface.layer_pointer,original)
        app._close()
