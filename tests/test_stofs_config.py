"""Tests for STOFS-3D-ATL ForcingConfig extensions."""

import json
import pytest
from pathlib import Path

from nos_utils.config import ForcingConfig


class TestSTOFSConfigFactory:
    def test_domain_bounds(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.lon_min == pytest.approx(-98.5035)
        assert cfg.lon_max == pytest.approx(-52.4867)
        assert cfg.lat_min == pytest.approx(7.347)
        assert cfg.lat_max == pytest.approx(52.5904)

    def test_run_window(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.nowcast_hours == 24
        assert cfg.forecast_hours == 108

    def test_gfs_resolution(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.gfs_resolution == "0p25"

    def test_hrrr_domain(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.hrrr_lon_min == -98.5
        assert cfg.hrrr_lon_max == -49.5
        assert cfg.hrrr_lat_min == 5.5
        assert cfg.hrrr_lat_max == 50.0
        assert cfg.hrrr_domain == (-98.5, -49.5, 5.5, 50.0)

    def test_hrrr_domain_fallback(self):
        """SECOFS has no HRRR-specific domain — falls back to main domain."""
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.hrrr_domain == cfg.domain

    def test_obc_roi_indices(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.obc_roi_2d == {"x1": 2805, "x2": 2923, "y1": 1598, "y2": 2325}
        assert cfg.obc_roi_3d == {"x1": 482, "x2": 600, "y1": 94, "y2": 821}

    def test_nudge_roi_indices(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.nudge_roi_3d == {"x1": 422, "x2": 600, "y1": 94, "y2": 835}

    def test_adt_enabled(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.adt_enabled is True

    def test_ssh_offset(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.obc_ssh_offset == 0.04

    def test_nwm_product(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.nwm_product == "medium_range_mem1"
        assert cfg.nwm_n_list_target == 121
        assert cfg.nwm_n_list_min == 97

    def test_nudging_defaults(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.nudging_enabled is True
        assert cfg.nudging_timescale_seconds == 86400.0

    def test_override_works(self):
        cfg = ForcingConfig.for_stofs_3d_atl(
            pdy="20260401", cyc=12,
            forecast_hours=72,
            adt_enabled=False,
        )
        assert cfg.forecast_hours == 72
        assert cfg.adt_enabled is False

    def test_ufs_variant(self):
        cfg = ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260401", cyc=12)
        assert cfg.nws == 4
        assert cfg.adt_enabled is True
        assert cfg.obc_roi_2d is not None


class TestSTOFSConfigSecofsBwdCompat:
    """Verify SECOFS defaults are unchanged by STOFS additions."""

    def test_secofs_no_roi(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.obc_roi_2d is None
        assert cfg.obc_roi_3d is None
        assert cfg.nudge_roi_3d is None

    def test_secofs_no_adt(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.adt_enabled is False

    def test_secofs_nwm_product(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.nwm_product == "analysis_assim"

    def test_secofs_no_hrrr_domain(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.hrrr_lon_min is None


class TestSTOFSFromYAML:
    """Test from_yaml() parsing of STOFS-specific fields."""

    def test_parse_stofs_yaml(self, tmp_path):
        yaml_content = {
            "grid": {
                "domain": {
                    "lon_min": -98.5035, "lon_max": -52.4867,
                    "lat_min": 7.347, "lat_max": 52.5904,
                },
                "n_levels": 51,
            },
            "model": {
                "physics": {"nws": 2},
                "run": {"nowcast_hours": 24, "forecast_hours": 108},
            },
            "forcing": {
                "atmospheric": {
                    "gfs": {"resolution": "0.25"},
                    "hrrr_blend": {
                        "enabled": True,
                        "lon_min": -98.5, "lon_max": -49.5,
                        "lat_min": 5.5, "lat_max": 50.0,
                    },
                },
                "ocean": {
                    "obc": {
                        "ssh_offset": 0.04,
                        "roi_2ds": {"x1": 2805, "x2": 2923, "y1": 1598, "y2": 2325},
                        "roi_3dz": {"x1": 482, "x2": 600, "y1": 94, "y2": 821},
                    },
                    "nudging": {
                        "enabled": True,
                        "timescale_days": 1.0,
                        "roi_3dz": {"x1": 422, "x2": 600, "y1": 94, "y2": 835},
                    },
                    "adt": {"enabled": True},
                },
                "river": {
                    "primary": "nwm",
                    "n_list_target": 121,
                    "n_list_min": 97,
                },
            },
        }
        import yaml
        yaml_file = tmp_path / "stofs_3d_atl.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))

        cfg = ForcingConfig.from_yaml(yaml_file, pdy="20260401", cyc=12)

        assert cfg.gfs_resolution == "0p25"
        assert cfg.hrrr_lon_min == -98.5
        assert cfg.hrrr_domain == (-98.5, -49.5, 5.5, 50.0)
        assert cfg.obc_roi_2d == {"x1": 2805, "x2": 2923, "y1": 1598, "y2": 2325}
        assert cfg.obc_roi_3d == {"x1": 482, "x2": 600, "y1": 94, "y2": 821}
        assert cfg.nudge_roi_3d == {"x1": 422, "x2": 600, "y1": 94, "y2": 835}
        assert cfg.adt_enabled is True
        assert cfg.obc_ssh_offset == 0.04
        assert cfg.nudging_enabled is True
        assert cfg.nwm_n_list_target == 121
