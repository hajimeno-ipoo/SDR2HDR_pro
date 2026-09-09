"""Real still-image decoding and HDR signalling, independent of display settings."""
import struct
import subprocess

import cv2
import numpy as np
import pytest

from sdr2hdr.compare_images import prepare_image
from sdr2hdr.io import ffprobe_comparison, save_image_hdr


def run(*command):
    return subprocess.run(command, capture_output=True, check=True).stdout


def rgb16_from_png(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None and image.dtype == np.uint16
    return image[..., ::-1]


def png_color(data):
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    offset = 8
    while offset < len(data):
        size = struct.unpack_from('>I', data, offset)[0]
        if data[offset+4:offset+8] == b'cICP':
            return data[offset+8:offset+8+size]
        offset += 12+size
    return None


@pytest.mark.parametrize('suffix', ['.png', '.jxl', '.tif'])
def test_rgb16_stills_keep_pixels_and_hdr_meaning(tmp_path, suffix):
    pixels = np.random.default_rng(54).integers(0, 65536, (32, 64, 3), dtype=np.uint16)
    path = tmp_path / ('hdr'+suffix)
    assert save_image_hdr(str(path), pixels)
    metadata = ffprobe_comparison(str(path))
    info = prepare_image(str(path), metadata, app_hdr_output=True)
    if suffix == '.tif':
        assert metadata.get('color_transfer') not in {'smpte2084', 'arib-std-b67'}
        assert info['color_source'] == '変換設定'
    else:
        assert metadata['color_primaries'] == 'bt2020'
        assert metadata['color_transfer'] == 'smpte2084'
    data = info.get('image_bytes', path.read_bytes())
    assert png_color(data) == bytes([9,16,0,1])
    np.testing.assert_array_equal(rgb16_from_png(data), pixels)


def avif_reference(path, width, height, xs, y):
    """BT.2020-2 NCL inverse and 10-bit narrow-range scaling (Tables 4/5).

    Sample centres of constant patches, so chroma interpolation does not change
    the reference. This avoids the implicit RGB conversion in FFmpeg/swscale.
    """
    raw = np.frombuffer(run('ffmpeg','-v','error','-i',str(path),'-frames:v','1',
                            '-pix_fmt','yuv420p10le','-f','rawvideo','-'), '<u2')
    size = width*height
    yy = raw[:size].reshape(height,width)[y,xs].astype(float)
    chroma_width, chroma_height = (width+1)//2, (height+1)//2
    chroma_size = chroma_width * chroma_height
    cb = raw[size:size+chroma_size].reshape(chroma_height,chroma_width)[y//2,xs//2].astype(float)
    cr = raw[size+chroma_size:].reshape(chroma_height,chroma_width)[y//2,xs//2].astype(float)
    yy, cb, cr = (yy-64)/876, (cb-512)/896, (cr-512)/896
    r = yy+1.4746*cr
    b = yy+1.8814*cb
    g = (yy-.2627*r-.0593*b)/.6780
    return np.stack([r,g,b], axis=-1).clip(0,1)


@pytest.mark.parametrize('width,height', [(256,64), (257,64), (256,65), (257,65), (65,33)])
def test_avif_tags_and_decoded_rgb_follow_bt2020_matrix(tmp_path, width, height):
    patches=np.array([[.25,.25,.25],[.75,.75,.75],[.7,.5,.4],[.4,.7,.5]])
    patch_indices = np.minimum(np.arange(width)*4//width, 3)
    pixels=np.repeat(np.rint(patches[patch_indices]*65535).astype(np.uint16)[None],height,0)
    path=tmp_path/'hdr.avif'
    assert save_image_hdr(str(path),pixels)
    metadata=ffprobe_comparison(str(path))
    assert (metadata['width'],metadata['height']) == (width,height)
    assert metadata['pix_fmt'] == 'yuv420p10le'
    assert (metadata['color_primaries'],metadata['color_transfer'],metadata['color_space']) == ('bt2020','smpte2084','bt2020nc')
    info=prepare_image(str(path),metadata,app_hdr_output=True)
    assert png_color(info['image_bytes']) == bytes([9,16,0,1])
    xs=((np.arange(4)+.5)*width/4).astype(int)
    reference=avif_reference(path,width,height,xs,height//2)
    decoded_image=rgb16_from_png(info['image_bytes'])
    assert decoded_image.shape == (height,width,3)
    decoded=decoded_image[height//2,xs]/65535
    # RGB16 rounding, not an allowance for lossy encoding (reference is decoded).
    np.testing.assert_allclose(decoded,reference,rtol=0,atol=2/65535)


def test_input_tiff_is_not_assumed_to_be_an_application_hdr_output(tmp_path):
    path=tmp_path/'input.tif'
    cv2.imwrite(str(path),np.zeros((16,16,3),np.uint16))
    metadata=ffprobe_comparison(str(path))
    assert prepare_image(str(path),metadata) == metadata
    assert metadata.get('color_transfer') != 'smpte2084'


def jpeg_reference(path, width, height, raw_path):
    run('ultrahdr_app','-m','1','-j',str(path),'-o','2','-O','5','-z',str(raw_path))
    packed=np.fromfile(raw_path,'<u4').reshape(height,width)
    return np.stack([(packed>>shift)&1023 for shift in (0,10,20)],axis=-1)/1023


def test_ultra_hdr_preview_matches_official_pq_reconstruction(tmp_path):
    pixels=np.full((64,64,3),49151,dtype=np.uint16)
    path=tmp_path/'hdr.jpg'
    assert save_image_hdr(str(path),pixels)
    metadata=ffprobe_comparison(str(path))
    assert metadata['bit_depth']==8  # JPEG's base rendition is not its HDR result.
    info=prepare_image(str(path),metadata,app_hdr_output=True)
    assert info['bit_depth']==10 and info['color_source']=='HDR復元'
    assert png_color(info['image_bytes']) == bytes([9,16,0,1])
    reference=jpeg_reference(path,64,64,tmp_path/'reference.raw')
    expected=np.rint(reference*65535).astype(np.uint16)
    np.testing.assert_array_equal(rgb16_from_png(info['image_bytes']),expected)
    assert reference.min()>.7  # HDR intent survives; SDR base is not displayed.


def test_corrupt_jxl_does_not_silently_fall_back_to_sdr(tmp_path):
    path=tmp_path/'bad.jxl';path.write_bytes(b'not an image')
    with pytest.raises(RuntimeError):
        prepare_image(str(path),{},app_hdr_output=True)
