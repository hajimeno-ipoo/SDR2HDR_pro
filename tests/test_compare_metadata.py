"""Recorded output HDR metadata reaches the display without inferred values."""
import json
import struct
import subprocess
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from sdr2hdr.io import ffprobe_comparison, restamp_prores_metadata


@pytest.fixture(scope='module')
def hdr_outputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp('comparison-hdr-metadata')
    common = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
              'color=gray:size=64x64:rate=24', '-frames:v', '1',
              '-color_primaries', 'bt2020', '-colorspace', 'bt2020nc']
    paths = {}
    for name in ('prores', 'hdr10', 'hlg'):
        path = directory / (name + '.mov')
        if name == 'prores':
            args = ['-c:v', 'prores_ks', '-profile:v', '5', '-pix_fmt', 'yuv444p10le',
                    '-color_trc', 'smpte2084', '-movflags', '+write_colr']
        else:
            transfer = 'smpte2084' if name == 'hdr10' else 'arib-std-b67'
            params = f'log-level=error:pools=1:repeat-headers=1:colorprim=bt2020:transfer={transfer}:colormatrix=bt2020nc'
            if name == 'hdr10':
                # A genuine 10,000-nit mastering tag must not be mistaken for
                # mpv's untagged-PQ default. Units follow MDCV / CLLI syntax.
                params += (':master-display=G(13250,34500)B(7500,3000)R(34000,16000)'
                           'WP(15635,16450)L(100000000,1):max-cll=4321,321')
            args = ['-c:v', 'libx265', '-pix_fmt', 'yuv420p10le', '-preset', 'ultrafast',
                    '-x265-params', params, '-color_trc', transfer, '-tag:v', 'hvc1']
        subprocess.run(common + args + [str(path)], check=True, capture_output=True)
        if name == 'prores':
            # Match the application's completed output, including its required
            # ProRes/MOV color tagging pass after FFmpeg encodes the frames.
            restamp_prores_metadata(str(path))
        paths[name] = path
    return paths


def test_probe_reads_recorded_hdr10_frame_metadata(hdr_outputs):
    path = str(hdr_outputs['hdr10'])
    raw = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-show_entries', 'stream_side_data', '-of', 'json', path],
                         check=True, capture_output=True, text=True)
    # This fixture needs frame-side inspection; stream-only probing loses it.
    assert not json.loads(raw.stdout)['streams'][0].get('side_data_list')
    info = ffprobe_comparison(path, hdr_metadata=True)
    assert info['color_transfer'] == 'smpte2084'
    assert struct.unpack('>8H2I', info['hdr10_display_info']) == (
        13250, 34500, 7500, 3000, 34000, 16000, 15635, 16450, 100000000, 1)
    assert struct.unpack('>HH', info['hdr10_content_info']) == (4321, 321)


@pytest.mark.parametrize('name,transfer', [('prores', 'smpte2084'), ('hlg', 'arib-std-b67')])
def test_probe_does_not_invent_hdr10_metadata(hdr_outputs, name, transfer):
    info = ffprobe_comparison(str(hdr_outputs[name]), hdr_metadata=True)
    assert info['color_transfer'] == transfer
    assert info['color_primaries'] == 'bt2020'
    assert 'hdr10_display_info' not in info
    assert 'hdr10_content_info' not in info


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS HDR metadata API')
def test_display_uses_file_metadata_and_resets_on_each_output(hdr_outputs):
    import Quartz
    from sdr2hdr.native_surface import MacSurface

    hdr10 = ffprobe_comparison(str(hdr_outputs['hdr10']), hdr_metadata=True)
    prores = ffprobe_comparison(str(hdr_outputs['prores']), hdr_metadata=True)
    # Exercise the real bridge with the bytes read from the encoded file.
    metadata = Quartz.CAEDRMetadata.HDR10MetadataWithDisplayInfo_contentInfo_opticalOutputScale_(
        hdr10['hdr10_display_info'], hdr10['hdr10_content_info'], 203.0)
    assert metadata is not None

    surface = MacSurface.__new__(MacSurface)
    surface.layer = Mock()
    surface.pending = Mock()
    player = MagicMock()
    surface.player = player
    player.video_out_params = {'min-luma': 0, 'max-luma': 10000}
    with patch.object(Quartz, 'CAEDRMetadata') as api:
        factory = api.HDR10MetadataWithDisplayInfo_contentInfo_opticalOutputScale_
        surface.configure_color(player, hdr10, True)
        factory.assert_called_once_with(hdr10['hdr10_display_info'], hdr10['hdr10_content_info'], 203.0)
        surface.layer.setEDRMetadata_.reset_mock()
        surface._update_hdr_metadata()
        surface.layer.setEDRMetadata_.assert_not_called()

        # Neither a preceding HDR10 video nor an inferred mpv peak may supply
        # metadata to this application's ProRes output.
        surface.configure_color(player, prores, True)
        factory.assert_called_with(None, None, 203.0)
        surface.layer.setEDRMetadata_.reset_mock()
        surface._update_hdr_metadata()
        surface.layer.setEDRMetadata_.assert_not_called()
        assert surface._metadata_range is None

        # HLG does not consume PQ file tags; its existing linear conversion stays.
        surface.configure_color(player, {**hdr10, 'color_transfer': 'arib-std-b67'}, True)
        factory.assert_called_with(None, None, 203.0)
        player.video_out_params = {'min-luma': 0, 'max-luma': 1000}
        surface._update_hdr_metadata()
        api.HDR10MetadataWithMinLuminance_maxLuminance_opticalOutputScale_.assert_called_once_with(0, 1000, 203.0)

        surface.configure_color(player, prores, False)
        surface.layer.setEDRMetadata_.assert_called_with(None)
