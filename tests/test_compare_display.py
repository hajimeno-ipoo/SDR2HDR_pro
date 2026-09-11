"""Display-capability transitions and per-file HDR metadata isolation."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

import pytest

from sdr2hdr.mpv_player import MpvPlayer
from sdr2hdr.native_surface import MacSurface, WindowsSurface


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS surface')
def test_still_and_hlg_metadata_are_preserved_and_cleared_for_sdr():
    import Quartz

    surface=MacSurface.__new__(MacSurface)
    surface.layer=Mock();surface.pending=Mock()
    player=MagicMock();surface.player=player
    surface.configure_color(player,{'color_transfer':'arib-std-b67'},True)
    initial=surface.layer.setEDRMetadata_.call_args.args[0]
    assert initial is not None
    player.video_out_params={'min-luma':.005,'max-luma':1000}
    surface._update_hdr_metadata()
    first=surface.layer.setEDRMetadata_.call_args.args[0]
    assert first is not None and first is not initial
    surface.configure_color(player,{'color_transfer':'smpte2084','source_peak_nits':406},True)
    player.video_out_params={'min-luma':0,'max-luma':406}
    surface._update_hdr_metadata()
    assert surface._metadata_range==(0,406)
    # A different file starts without the preceding file's brightness metadata.
    surface.configure_color(player,{'color_transfer':'smpte2084'},True)
    assert surface._metadata_range is None
    assert surface.layer.setEDRMetadata_.call_args.args[0] is not first
    surface.configure_color(player,{},False)
    surface.layer.setEDRMetadata_.assert_called_with(None)
    surface.layer.setToneMapMode_.assert_called_with(Quartz.CAToneMapModeNever)
    player.__setitem__.assert_any_call('target-trc','srgb')


def test_windows_delegates_hdr_target_to_display_capabilities():
    # Configuration regression only; this is not a Windows hardware test.
    surface=WindowsSurface.__new__(WindowsSurface);surface.widget=Mock()
    options=surface.player_options(True)
    assert options['target_colorspace_hint']=='auto'
    assert options['target_colorspace_hint_mode']=='target'
    player={};surface.configure_color(player,{},True)
    assert player=={'target-prim':'auto','target-trc':'auto','target-peak':'auto'}


def test_still_peak_is_cleared_when_loading_video():
    surface=Mock();surface.player_options.return_value={}
    player=MagicMock()
    module=SimpleNamespace(MPV=Mock(return_value=player),strict_decoder=None)
    with patch.dict('sys.modules',mpv=module):
        wrapper=MpvPlayer(surface,hdr=True)
        wrapper.load('still.png',{'source_peak_nits':406.})
        player.__setitem__.assert_any_call('vf','format=sig-peak=2')
        player.__setitem__.reset_mock()
        wrapper.load('video.mov',{'color_transfer':'smpte2084'})
        player.__setitem__.assert_called_once_with('vf','')
        assert module.MPV.call_args.kwargs['tone_mapping']=='bt.2390'
        assert module.MPV.call_args.kwargs['tone_mapping_param']==.5
