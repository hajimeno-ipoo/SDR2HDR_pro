"""Actual HDR image export, signalling, reconstruction and publication checks."""
import struct
import subprocess
import zlib
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from sdr2hdr import app, io
from sdr2hdr.hdr_guidance import linear_nits_to_pq


def run(*args):
    return subprocess.run(args, check=True, capture_output=True).stdout


def chunks(path):
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    pos = 8
    result = []
    while pos < len(data):
        size = struct.unpack_from('>I', data, pos)[0]
        kind, payload = data[pos+4:pos+8], data[pos+8:pos+8+size]
        assert struct.unpack_from('>I', data, pos+8+size)[0] == zlib.crc32(kind+payload)
        result.append((kind, payload))
        pos += size+12
    return result


def test_png_preserves_full_rgb16_and_hdr_signalling(tmp_path):
    pixels = np.random.default_rng(2026).integers(0, 65536, (33, 65, 3), dtype=np.uint16)
    path = tmp_path/'hdr.png'
    assert io.save_image_hdr(str(path), pixels)
    records = chunks(path)
    assert dict(records)[b'IHDR'][8:10] == bytes([16, 2])
    assert [v for k,v in records if k == b'cICP'] == [bytes([9,16,0,1])]
    assert [k for k,v in records].index(b'cICP') < [k for k,v in records].index(b'IDAT')
    np.testing.assert_array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., ::-1], pixels)


def test_jpeg_reconstructs_hdr_with_official_decoder(tmp_path):
    # Constant neutral 1000 nit patch isolates HDR reconstruction from JPEG edge loss.
    pixels = np.full((64,64,3), np.rint(linear_nits_to_pq(np.array(1000.0))*65535), dtype=np.uint16)
    path = tmp_path/'hdr.jpeg'
    assert io.save_image_hdr(str(path), pixels)
    raw = tmp_path/'decoded.raw'
    metadata = tmp_path/'gainmap.cfg'
    run('ultrahdr_app', '-m', '1', '-j', str(path), '-o', '0', '-O', '4', '-z', str(raw), '-f', str(metadata))
    decoded = np.fromfile(raw, '<f2').reshape(64,64,4)[...,:3].astype(float)*203
    assert np.isfinite(decoded).all()
    # HDR must survive (> SDR white); this is not a lossless/visual quality claim.
    assert decoded.min() > 203 and decoded.max() < 10000
    assert '--maxContentBoost' in metadata.read_text()
    # Compare against the official encode of independently packed PQ samples.
    code = int(np.rint(float(pixels[0,0,0])*1023/65535))
    reference_raw = tmp_path/'reference.raw'
    np.full((64,64), code+(code<<10)+(code<<20)+(3<<30), dtype='<u4').tofile(reference_raw)
    reference_jpg = tmp_path/'reference.jpg'
    run('ultrahdr_app','-m','0','-p',str(reference_raw),'-w','64','-h','64','-a','5','-C','2','-t','2','-R','1','-q','95','-Q','95','-s','1','-M','1','-z',str(reference_jpg))
    assert path.read_bytes() == reference_jpg.read_bytes()
    assert cv2.imread(str(path)).shape == (64,64,3)  # Ordinary JPEG SDR fallback.


@pytest.mark.parametrize('mode', ['ai', 'log'])
@pytest.mark.parametrize('suffix', ['.png', '.jpg', '.jpeg'])
def test_both_image_paths_publish_real_files(tmp_path, mode, suffix):
    source = tmp_path/'source.png'
    cv2.imwrite(str(source), np.full((32,64,3), 180, dtype=np.uint8))
    output = tmp_path/('hdr'+suffix)
    if mode == 'ai':
        request = app.ImageConversionRequest(str(source), str(output), backend='numpy', model_path=str(Path(__file__).resolve().parents[1]/'models/enhancement_model_20260310.pt'))
        convert = app.run_image_conversion
    else:
        request = app.ImageLogConversionRequest(str(source), str(output))
        convert = app.run_image_log_conversion
    result = convert(request)
    assert not result.cancelled and result.processed_frames == 1 and output.stat().st_size > 0
    assert not list(tmp_path.glob('.sdr2hdr-*'))
    if suffix == '.png':
        assert dict(chunks(output))[b'cICP'] == bytes([9,16,0,1])
    else:
        raw = tmp_path/'decoded.raw'
        run('ultrahdr_app','-m','1','-j',str(output),'-o','0','-O','4','-z',str(raw))
        assert raw.stat().st_size == 32*64*8


def test_missing_encoder_fails_before_touching_existing_file(tmp_path):
    source = tmp_path/'source.png'
    cv2.imwrite(str(source), np.zeros((32,32,3), dtype=np.uint8))
    output = tmp_path/'existing.jpg'
    output.write_bytes(b'existing')
    with patch.object(io.shutil, 'which', return_value=None), pytest.raises(ValueError, match='ultrahdr_app'):
        app.run_image_log_conversion(app.ImageLogConversionRequest(str(source),str(output)))
    assert output.read_bytes() == b'existing'


def test_jpeg_cancel_preserves_existing_file(tmp_path):
    output = tmp_path/'existing.jpg'
    output.write_bytes(b'existing')
    assert not io.save_image_hdr(str(output), np.zeros((32,32,3), dtype=np.uint16), cancel_check=lambda:True)
    assert output.read_bytes() == b'existing'
    assert not list(tmp_path.glob('.ultrahdr-*'))
