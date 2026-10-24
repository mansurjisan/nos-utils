"""Tests for STOFS-3D-ATL and STOFS-3D-PAC ForcingConfig extensions."""

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
        assert cfg.forecast_hours == 96

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

    def test_model_dt_is_150(self):
        """STOFS-3D-ATL runs at dt=150 (param.nml dt=150.). The elev2D.th.nc
        time_step must match or SCHISM aborts (misc_subs.F90:563
        'MISC: elev2D.th dt wrong')."""
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.model_dt == 150.0

    def test_ufs_model_dt_is_150(self):
        cfg = ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260401", cyc=12)
        assert cfg.model_dt == 150.0


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

    def test_secofs_model_dt_is_120(self):
        """SECOFS runs at dt=120, so the elev2D writer keeps time_step=120
        (byte-identical to prior output)."""
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert cfg.model_dt == 120.0

    def test_secofs_ufs_model_dt_is_120(self):
        cfg = ForcingConfig.for_secofs_ufs(pdy="20260401", cyc=12)
        assert cfg.model_dt == 120.0

    def test_default_model_dt_is_120(self):
        """Bare ForcingConfig (no factory) defaults to 120.0."""
        cfg = ForcingConfig(
            lon_min=-80.0, lon_max=-70.0,
            lat_min=25.0, lat_max=35.0,
            pdy="20260401", cyc=12,
        )
        assert cfg.model_dt == 120.0


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


class TestModelDtFromYAML:
    """from_yaml() parses model.physics.dt into ForcingConfig.model_dt."""

    def _yaml(self, tmp_path, physics_block):
        content = {
            "grid": {
                "domain": {
                    "lon_min": -98.5035, "lon_max": -52.4867,
                    "lat_min": 7.347, "lat_max": 52.5904,
                },
                "n_levels": 51,
            },
            "model": {
                "physics": physics_block,
                "run": {"nowcast_hours": 24, "forecast_hours": 108},
            },
        }
        import yaml
        f = tmp_path / "stofs_3d_atl.yaml"
        f.write_text(yaml.dump(content))
        return f

    def test_dt_150_parsed(self, tmp_path):
        f = self._yaml(tmp_path, {"nws": 2, "dt": 150.0})
        cfg = ForcingConfig.from_yaml(f, pdy="20260401", cyc=12)
        assert cfg.model_dt == 150.0

    def test_dt_120_parsed(self, tmp_path):
        f = self._yaml(tmp_path, {"nws": 4, "dt": 120.0})
        cfg = ForcingConfig.from_yaml(f, pdy="20260401", cyc=12)
        assert cfg.model_dt == 120.0

    def test_dt_defaults_when_absent(self, tmp_path):
        """No model.physics.dt and no DELT_MODEL env -> defaults to 120.0."""
        import os
        f = self._yaml(tmp_path, {"nws": 2})
        old = os.environ.pop("DELT_MODEL", None)
        try:
            cfg = ForcingConfig.from_yaml(f, pdy="20260401", cyc=12)
            assert cfg.model_dt == 120.0
        finally:
            if old is not None:
                os.environ["DELT_MODEL"] = old

    def test_delt_model_env_fallback(self, tmp_path):
        """When model.physics.dt is absent, honor the prep's DELT_MODEL export."""
        import os
        f = self._yaml(tmp_path, {"nws": 2})
        old = os.environ.get("DELT_MODEL")
        try:
            os.environ["DELT_MODEL"] = "150.0"
            cfg = ForcingConfig.from_yaml(f, pdy="20260401", cyc=12)
            assert cfg.model_dt == 150.0
        finally:
            if old is not None:
                os.environ["DELT_MODEL"] = old
            else:
                os.environ.pop("DELT_MODEL", None)


class TestStLawrenceSubdirFromYAML:
    """from_yaml() parses river.st_lawrence.subdir / csv_name."""

    def _yaml(self, tmp_path, stl_block):
        content = {
            "grid": {
                "domain": {
                    "lon_min": -98.5035, "lon_max": -52.4867,
                    "lat_min": 7.347, "lat_max": 52.5904,
                },
                "n_levels": 51,
            },
            "model": {"run": {"nowcast_hours": 24, "forecast_hours": 108}},
            "forcing": {"river": {"primary": "nwm",
                                  "st_lawrence": stl_block}},
        }
        import yaml
        f = tmp_path / "stofs_3d_atl.yaml"
        f.write_text(yaml.dump(content))
        return f

    def test_subdir_parsed(self, tmp_path):
        f = self._yaml(tmp_path, {
            "enabled": True,
            "subdir": "canadian_water",
            "csv_name": "QC_02OA016_hourly_hydrometric.csv",
        })
        cfg = ForcingConfig.from_yaml(f, pdy="20260401", cyc=12)
        assert cfg.st_lawrence_enabled is True
        assert cfg.st_lawrence_subdir == "canadian_water"
        assert cfg.st_lawrence_csv_name == "QC_02OA016_hourly_hydrometric.csv"

    def test_subdir_defaults_when_absent(self, tmp_path):
        # Backward-compatible default when subdir not specified in YAML.
        f = self._yaml(tmp_path, {"enabled": True})
        cfg = ForcingConfig.from_yaml(f, pdy="20260401", cyc=12)
        assert cfg.st_lawrence_subdir == "can_streamgauge"


class TestSTOFSPACConfigFactory:
    """Tests for STOFS-3D-PAC ForcingConfig factory methods.

    Domain values verified from operational fix files in tmp/opencode/:
    - hgrid.ll: 2,961,412 node rows, lon 93.1098-289.6750, lat -33.0388-66.3254
    - weight_gfs.nc: GFS src grid 92.5-290.0, -33.5-67.0 (791x403 at 0.25 deg)
    """

    def test_domain_bounds_0_360(self):
        """Pacific domain is in native 0-360 (mesh spans antimeridian)."""
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.lon_min == pytest.approx(93.1098)
        assert cfg.lon_max == pytest.approx(289.6750)
        assert cfg.lat_min == pytest.approx(-33.0388)
        assert cfg.lat_max == pytest.approx(66.3254)
        # Must satisfy ForcingConfig.__post_init__ lon_min < lon_max check
        assert cfg.lon_min < cfg.lon_max

    def test_run_window(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.nowcast_hours == 24
        assert cfg.forecast_hours == 48

    def test_model_dt_is_90(self):
        """PAC model runs at dt=90s (verified from param.nml_6globaloutput)."""
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.model_dt == 90.0

    def test_n_levels_84(self):
        """PAC mesh has 84 vertical levels (verified from vgrid.in line 2)."""
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.n_levels == 84

    def test_no_st_lawrence(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.st_lawrence_enabled is False

    def test_no_dynamic_adjust(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.dynamic_adjust_enabled is False

    def test_adt_enabled_with_own_offset(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.adt_enabled is True
        assert cfg.adt_datum_offset == pytest.approx(-0.4715517)

    def test_nudging_enabled(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.nudging_enabled is True
        assert cfg.nudging_timescale_seconds == 86400.0

    def test_obc_elev_segments(self):
        """3 elev-forced segments: Indian Ocean=0, Pacific=1, Arctic=9."""
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.obc_elev_segments == [0, 1, 9]

    def test_nwm_product(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.nwm_product == "medium_range_mem1"

    def test_obc_min_timesteps(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.obc_min_timesteps == 4
        assert cfg.obc_min_slots == 4

    def test_gfs_resolution(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.gfs_resolution == "0p25"

    def test_override_works(self):
        cfg = ForcingConfig.for_stofs_3d_pac(
            pdy="20260401", cyc=12,
            forecast_hours=72,
            obc_ssh_offset=0.05,
        )
        assert cfg.forecast_hours == 72
        assert cfg.obc_ssh_offset == pytest.approx(0.05)

    def test_ufs_variant_nws4(self):
        cfg = ForcingConfig.for_stofs_3d_pac_ufs(pdy="20260401", cyc=12)
        assert cfg.nws == 4

    def test_ufs_datm_domain(self):
        """DATM grid verified from weight_gfs.nc: 92.5-290.0, -33.5-67.0."""
        cfg = ForcingConfig.for_stofs_3d_pac_ufs(pdy="20260401", cyc=12)
        assert cfg.datm_lon_min == pytest.approx(92.5)
        assert cfg.datm_lon_max == pytest.approx(290.0)
        assert cfg.datm_lat_min == pytest.approx(-33.5)
        assert cfg.datm_lat_max == pytest.approx(67.0)
        assert cfg.datm_dx == pytest.approx(0.025)

    def test_ufs_task_layout(self):
        """3952 SCHISM ranks (operational partition.prop) + 120 DATM = 4072."""
        cfg = ForcingConfig.for_stofs_3d_pac_ufs(pdy="20260401", cyc=12)
        assert cfg.ufs_schism_tasks == 3952
        assert cfg.ufs_datm_tasks == 120
        assert cfg.ufs_total_tasks == 4072

    def test_ufs_model_dt_is_90(self):
        cfg = ForcingConfig.for_stofs_3d_pac_ufs(pdy="20260401", cyc=12)
        assert cfg.model_dt == 90.0

    def test_ufs_no_st_lawrence(self):
        cfg = ForcingConfig.for_stofs_3d_pac_ufs(pdy="20260401", cyc=12)
        assert cfg.st_lawrence_enabled is False

    def test_ufs_no_dynamic_adjust(self):
        cfg = ForcingConfig.for_stofs_3d_pac_ufs(pdy="20260401", cyc=12)
        assert cfg.dynamic_adjust_enabled is False

    def test_forcing_domain_uses_datm_when_nws4(self):
        """nws=4: forcing_domain returns datm_domain (wider DATM grid)."""
        cfg = ForcingConfig.for_stofs_3d_pac_ufs(pdy="20260401", cyc=12)
        assert cfg.forcing_domain == cfg.datm_domain
        assert cfg.forcing_domain == (92.5, 290.0, -33.5, 67.0)

    def test_forcing_domain_uses_mesh_when_nws2(self):
        """nws=2 (standalone): forcing_domain returns mesh domain."""
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.forcing_domain == cfg.domain
        assert cfg.forcing_domain[0] == pytest.approx(93.1098)


class TestSTOFSPACFromYAML:
    """Test from_yaml() parsing of STOFS-3D-PAC YAML fields."""

    def _yaml(self, tmp_path, extra_forcing=None):
        content = {
            "system": {"name": "stofs_3d_pac_ufs", "framework": "stofs_ufs"},
            "execution": {"mode": "ufs"},
            "prep": {"extras": {"st_lawrence": False, "obc_dynamic_adjust": False}},
            "grid": {
                "domain": {
                    "lon_min": 93.1098, "lon_max": 289.6750,
                    "lat_min": -33.0388, "lat_max": 66.3254,
                },
                "n_levels": 84,
            },
            "model": {
                "physics": {"nws": 4, "dt": 90.0},
                "run": {"nowcast_hours": 24, "forecast_hours": 108},
            },
            "forcing": {
                "atmospheric": {"gfs": {"resolution": "0.25"}, "met_num": 2},
                "ocean": {
                    "obc": {
                        "elev_segments": [0, 1, 9],
                        "ssh_offset": 0.0,
                        "obc_mode": "none",
                    },
                    "nudging": {"enabled": True, "timescale_days": 1.0},
                    "adt": {"enabled": False},
                },
                "river": {"primary": "nwm", "st_lawrence": {"enabled": False}},
            },
            "ufs_coastal": {
                "enabled": True,
                "datm_domain": "STOFS3D_PAC",
                "blend_resolution": 0.025,
                "datm_tasks": 120,
                "schism_tasks": 3952,
                "total_tasks": 4072,
                "nhours_fcst": 108,
                "dt_atmos": 720,
            },
        }
        if extra_forcing:
            import copy
            content = copy.deepcopy(content)
            content["forcing"].update(extra_forcing)
        import yaml
        f = tmp_path / "stofs_3d_pac_ufs.yaml"
        f.write_text(yaml.dump(content))
        return f

    def test_domain_parsed(self, tmp_path):
        cfg = ForcingConfig.from_yaml(self._yaml(tmp_path), pdy="20260401", cyc=12)
        assert cfg.lon_min == pytest.approx(93.1098)
        assert cfg.lon_max == pytest.approx(289.6750)
        assert cfg.lon_min < cfg.lon_max

    def test_nws4_parsed(self, tmp_path):
        cfg = ForcingConfig.from_yaml(self._yaml(tmp_path), pdy="20260401", cyc=12)
        assert cfg.nws == 4

    def test_model_dt_90_parsed(self, tmp_path):
        cfg = ForcingConfig.from_yaml(self._yaml(tmp_path), pdy="20260401", cyc=12)
        assert cfg.model_dt == 90.0

    def test_obc_elev_segments_parsed(self, tmp_path):
        cfg = ForcingConfig.from_yaml(self._yaml(tmp_path), pdy="20260401", cyc=12)
        assert cfg.obc_elev_segments == [0, 1, 9]

    def test_no_st_lawrence_from_yaml(self, tmp_path):
        cfg = ForcingConfig.from_yaml(self._yaml(tmp_path), pdy="20260401", cyc=12)
        assert cfg.st_lawrence_enabled is False

    def test_no_dynamic_adjust_from_yaml(self, tmp_path):
        cfg = ForcingConfig.from_yaml(self._yaml(tmp_path), pdy="20260401", cyc=12)
        assert cfg.dynamic_adjust_enabled is False

    def test_datm_preset_stofs3d_pac(self, tmp_path):
        """STOFS3D_PAC preset resolves to verified bounds from weight_gfs.nc."""
        cfg = ForcingConfig.from_yaml(self._yaml(tmp_path), pdy="20260401", cyc=12)
        assert cfg.datm_lon_min == pytest.approx(92.5)
        assert cfg.datm_lon_max == pytest.approx(290.0)
        assert cfg.datm_lat_min == pytest.approx(-33.5)
        assert cfg.datm_lat_max == pytest.approx(67.0)

    def test_ufs_tasks_parsed(self, tmp_path):
        cfg = ForcingConfig.from_yaml(self._yaml(tmp_path), pdy="20260401", cyc=12)
        assert cfg.ufs_schism_tasks == 3952
        assert cfg.ufs_datm_tasks == 120
        assert cfg.ufs_total_tasks == 4072



class TestPacificBranchKeepsExistingSystems:
    """Pacific (#74) options must not change ATL / SECOFS / AK. MJ (10/24/26)"""

    @staticmethod
    def _from_yaml(tmp_path, ocean, run=None):
        import yaml
        doc = {
            "grid": {"domain": {"lon_min": -98.5, "lon_max": -52.5,
                                "lat_min": 7.3, "lat_max": 52.6}, "n_levels": 51},
            "model": {"physics": {"nws": 2}, "run": run or {"nowcast_hours": 24, "forecast_hours": 96}},
            "forcing": {"ocean": ocean},
        }
        f = tmp_path / "x.yaml"
        f.write_text(yaml.dump(doc))
        return ForcingConfig.from_yaml(f, pdy="20260401", cyc=12)

    def test_atl_adt_offset_with_and_without_yaml_key(self, tmp_path):
        stofs = {"obc": {"roi_2ds": {"x1": 1, "x2": 2, "y1": 1, "y2": 2},
                         "roi_3dz": {"x1": 1, "x2": 2, "y1": 1, "y2": 2}},
                 "adt": {"enabled": True}}
        assert self._from_yaml(tmp_path, stofs).adt_datum_offset == -0.45
        assert ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12).adt_datum_offset == -0.45
        stofs["adt"]["datum_offset"] = -0.4715517
        assert self._from_yaml(tmp_path, stofs).adt_datum_offset == pytest.approx(-0.4715517)

    def test_atl_keeps_ops_rtofs_slots_and_qc(self, tmp_path):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert cfg.obc_min_slots == 23
        assert cfg.obc_min_timesteps == 21
        assert cfg.obc_dt == 21600.0
        assert cfg.obc_archive_extended is False
        assert cfg.surface_restore_enabled is False

    def test_secofs_and_unspecified_yaml_keep_3h_nudging(self, tmp_path):
        assert ForcingConfig.for_secofs(pdy="20260401", cyc=12).nudging_dt == 10800.0
        assert ForcingConfig.for_secofs_ufs(pdy="20260401", cyc=12).nudging_dt == 10800.0
        cfg = self._from_yaml(tmp_path, {"nudging": {"enabled": True}})
        assert cfg.nudging_dt == 10800.0
        assert cfg.obc_archive_extended is False

    def test_pacific_sets_its_own_values(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert cfg.nudging_dt == 86400.0
        assert cfg.obc_dt == 86400.0
        assert cfg.obc_archive_extended is True
        assert cfg.surface_restore_enabled is False

    def test_surface_restore_refuses_without_experimental_env(self, tmp_path, monkeypatch):
        from nos_utils.forcing.surface_restore import SurfaceRestoreProcessor
        monkeypatch.delenv("NOS_SURFACE_RESTORE_EXPERIMENTAL", raising=False)
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12, surface_restore_enabled=True)
        res = SurfaceRestoreProcessor(cfg, tmp_path, tmp_path / "out").process()
        assert not res.success
        assert "untested" in res.errors[0]

    def test_from_yaml_parses_min_slots_and_archive_extended(self, tmp_path):
        base = {"nudging": {"enabled": True}}
        cfg = self._from_yaml(tmp_path, base)
        assert cfg.obc_min_slots == 23 and cfg.obc_archive_extended is False
        cfg = self._from_yaml(tmp_path, {"obc": {"min_slots": 4, "archive_extended": True}})
        assert cfg.obc_min_slots == 4 and cfg.obc_archive_extended is True

    def test_from_yaml_tolerates_null_ocean_blocks(self, tmp_path):
        cfg = self._from_yaml(tmp_path, {"nudging": None, "surface_restore": None})
        assert cfg.nudging_dt == 10800.0
        assert self._from_yaml(tmp_path, None).nudging_dt == 10800.0

    def test_atl_rerun_archive_map_is_the_four_base_entries(self):
        from nos_utils.orchestrator import PrepOrchestrator
        orch = PrepOrchestrator(ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12), {})
        assert orch._OBC_ARCHIVE_NAMES == {
            "elev2D.th.nc": "elev2dth_non_adj", "TEM_3D.th.nc": "tem3dth",
            "SAL_3D.th.nc": "sal3dth", "uv3D.th.nc": "uv3dth"}

    def test_tsuv_keeps_depth_and_logs_no_failure(self, tmp_path, caplog):
        import logging
        import numpy as np
        from netCDF4 import Dataset
        from nos_utils.forcing.rtofs import RTOFSProcessor
        rng = np.random.default_rng(0)
        files = []
        for tag in ("n000", "f006"):
            p = tmp_path / f"rtofs_glo_3dz_{tag}_6hrly_hvr_US_east.nc"
            with Dataset(p, "w") as d:
                d.createDimension("MT", 1); d.createDimension("Depth", 3)
                d.createDimension("Y", 5); d.createDimension("X", 6)
                d.createVariable("Depth", "f4", ("Depth",))[:] = [0, 10, 20]
                d.createVariable("Longitude", "f4", ("Y", "X"))[:] = np.tile(np.arange(6.), (5, 1)) - 70
                d.createVariable("Latitude", "f4", ("Y", "X"))[:] = np.tile(np.arange(5.)[:, None], (1, 6)) + 30
                for v in ("temperature", "salinity", "u", "v"):
                    d.createVariable(v, "f4", ("MT", "Depth", "Y", "X"))[:] = rng.random((1, 3, 5, 6))
            files.append(p)
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        cfg.obc_roi_3d = {"x1": 0, "x2": 4, "y1": 0, "y2": 3}
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path / "out")
        with caplog.at_level(logging.WARNING):
            out = proc._stofs_prepare_tsuv(files, tmp_path)
        assert "Failed to read 3D group" not in caplog.text
        with Dataset(out) as d:
            assert "depth" in d.variables
