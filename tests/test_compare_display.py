"""Display-capability transitions and per-file HDR metadata isolation."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

import pytest

from sdr2hdr.mpv_player import MpvPlayer
from sdr2hdr.native_surface import MacSurface, WindowsSurface


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS surface')
def test_display_peak_follows_current_screen_and_sdr_state():
    surface=MacSurface.__new__(MacSurface)
    surface.player={};surface._target_peak=None;surface._configured_hdr=True
    screen=Mock();window=Mock();surface.view=Mock()
    surface.view.window.return_value=window;window.screen.return_value=screen
    for headroom,expected in [(1,203),(2.5,507),(4.93,1000),(float('nan'),203),(100,10000)]:
        screen.maximumExtendedDynamicRangeColorComponentValue.return_value=headroom
        assert surface._update_display_peak()
        assert surface.player['target-peak']==expected
        assert isinstance(surface.player['target-peak'],int)
        assert not surface._update_display_peak()
    surface._configured_hdr=False
    assert surface._update_display_peak()
    assert surface.player['target-peak']==203
    surface._configured_hdr=True;window.screen.return_value=None
    assert not surface._update_display_peak()


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
