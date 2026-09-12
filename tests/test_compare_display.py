"""Display-capability transitions and per-file HDR metadata isolation."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

import pytest

from sdr2hdr.mpv_player import MpvPlayer
from sdr2hdr.native_surface import MacSurface, WindowsSurface


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS surface')
def test_pq_stills_use_video_metadata_policy_and_preserve_hlg_and_sdr():
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
    # A PQ image uses the same metadata-free policy as ProRes PQ, even when
    # mpv reports its inferred 10,000-nit maximum after an HLG input.
    with patch.object(Quartz, 'CAEDRMetadata') as api:
        factory=api.HDR10MetadataWithDisplayInfo_contentInfo_opticalOutputScale_
        surface.configure_color(player,{'color_transfer':'smpte2084'},True)
        factory.assert_called_once_with(None,None,203.0)
        assert surface._metadata_range is None
        surface.layer.setEDRMetadata_.reset_mock()
        player.video_out_params={'min-luma':0,'max-luma':10000}
        surface._update_hdr_metadata()
        surface.layer.setEDRMetadata_.assert_not_called()
        api.HDR10MetadataWithMinLuminance_maxLuminance_opticalOutputScale_.assert_not_called()
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


def test_stills_and_videos_load_without_a_peak_override():
    surface=Mock();surface.player_options.return_value={}
    player=MagicMock()
    module=SimpleNamespace(MPV=Mock(return_value=player),strict_decoder=None)
    with patch.dict('sys.modules',mpv=module):
        wrapper=MpvPlayer(surface,hdr=True)
        for path in ('日本語/てすと.png','video.mov'):
            info={'color_transfer':'smpte2084'}
            wrapper.load(path,info)
            player.__setitem__.assert_not_called()
            surface.configure_color.assert_called_with(player,info,True)
            player.command.assert_called_with('loadfile',path,'replace')
        assert module.MPV.call_args.kwargs['tone_mapping']=='bt.2390'
        assert module.MPV.call_args.kwargs['tone_mapping_param']==.5
