"""Tests for ForcingConfig."""

import pytest
from nos_utils.config import ForcingConfig


class TestForcingConfig:
    def test_basic_creation(self):
        cfg = ForcingConfig(
            lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
            pdy="20260401", cyc=12,
        )
        assert cfg.domain == (-80, -70, 25, 35)
        assert cfg.pdy == "20260401"
        assert cfg.cyc == 12
        assert cfg.nowcast_hours == 6
        assert cfg.forecast_hours == 48

    def test_secofs_factory(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.lon_min == -88.0
        assert cfg.lon_max == -63.0
        assert cfg.lat_min == 17.0
        assert cfg.lat_max == 40.0
        assert cfg.met_num == 2
        assert cfg.nowcast_hours == 6
        assert cfg.forecast_hours == 48

    def test_stofs_factory(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.lon_min == pytest.approx(-98.5035)
        assert cfg.nowcast_hours == 24
        assert cfg.forecast_hours == 108

    def test_factory_overrides(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12, forecast_hours=120)
        assert cfg.forecast_hours == 120

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="lon_min"):
            ForcingConfig(lon_min=-70, lon_max=-80, lat_min=25, lat_max=35,
                         pdy="20260401", cyc=12)

    def test_invalid_lat_raises(self):
        with pytest.raises(ValueError, match="lat_min"):
            ForcingConfig(lon_min=-80, lon_max=-70, lat_min=35, lat_max=25,
                         pdy="20260401", cyc=12)

    def test_igrd_requires_grid_file(self):
        with pytest.raises(ValueError, match="grid_file"):
            ForcingConfig(lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
                         pdy="20260401", cyc=12, igrd_met=1)

    def test_defaults(self):
        cfg = ForcingConfig(lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
                           pdy="20260401", cyc=12)
        assert cfg.igrd_met == 0
        assert cfg.met_num == 1
        assert cfg.scale_hflux == 1.0
        assert cfg.nws == 2
        assert cfg.grid_file is None
        assert cfg.variables == []
