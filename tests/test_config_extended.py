"""Tests for extended ForcingConfig: STOFS, UFS, ensemble, from_yaml."""

import pytest
from nos_utils.config import ForcingConfig


class TestSTOFSConfig:
    def test_stofs_defaults(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260324", cyc=12)
        assert cfg.lon_min == pytest.approx(-98.5035)
        assert cfg.lon_max == pytest.approx(-52.4867)
        assert cfg.nowcast_hours == 24
        assert cfg.forecast_hours == 108
        assert cfg.met_num == 2
        assert cfg.n_levels == 51
        assert cfg.nudging_enabled is True
        assert cfg.obc_ssh_offset == 0.04

    def test_stofs_override(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260324", cyc=12, forecast_hours=72)
        assert cfg.forecast_hours == 72


class TestUFSConfig:
    def test_secofs_ufs(self):
        cfg = ForcingConfig.for_secofs_ufs(pdy="20260324", cyc=12)
        assert cfg.nws == 4
        assert cfg.met_num == 2
        assert cfg.lon_min == -88.0


class TestEnsembleConfig:
    def test_control_member(self):
        cfg = ForcingConfig.for_ensemble(pdy="20260324", cyc=12, member=0)
        assert cfg.met_num == 2  # Control uses GFS+HRRR

    def test_perturbation_member(self):
        cfg = ForcingConfig.for_ensemble(pdy="20260324", cyc=12, member=3)
        assert cfg.met_num == 1  # Perturbation uses GEFS only

    def test_secofs_base(self):
        cfg = ForcingConfig.for_ensemble(pdy="20260324", cyc=12, base_ofs="secofs")
        assert cfg.nowcast_hours == 6
        assert cfg.forecast_hours == 48


class TestFromYAML:
    def test_secofs_yaml(self):
        yaml_path = "/mnt/d/NOS-Workflow-Project/nos_ofs_complete_package/nos_ofs/parm/systems/secofs.yaml"
        try:
            cfg = ForcingConfig.from_yaml(yaml_path, pdy="20260324", cyc=12)
        except (ImportError, FileNotFoundError):
            pytest.skip("YAML file or PyYAML not available")

        assert cfg.lon_min == -88.0
        assert cfg.lon_max == -63.0
        assert cfg.nowcast_hours == 6
        assert cfg.forecast_hours == 48  # from base schism.yaml: forecast_days=5 (but secofs override missing)
        assert cfg.met_num == 2  # inferred from secondary: HRRR

    def test_stofs_yaml(self):
        yaml_path = "/mnt/d/NOS-Workflow-Project/nos_ofs_complete_package/nos_ofs/parm/systems/stofs_3d_atl.yaml"
        try:
            cfg = ForcingConfig.from_yaml(yaml_path, pdy="20260324", cyc=12)
        except (ImportError, FileNotFoundError):
            pytest.skip("YAML file or PyYAML not available")

        assert cfg.lon_min == pytest.approx(-98.5035)
        assert cfg.nowcast_hours == 24
        assert cfg.forecast_hours == 108

    def test_yaml_override(self):
        yaml_path = "/mnt/d/NOS-Workflow-Project/nos_ofs_complete_package/nos_ofs/parm/systems/secofs.yaml"
        try:
            cfg = ForcingConfig.from_yaml(yaml_path, pdy="20260401", cyc=18, nws=4)
        except (ImportError, FileNotFoundError):
            pytest.skip("YAML file or PyYAML not available")

        assert cfg.pdy == "20260401"
        assert cfg.cyc == 18
        assert cfg.nws == 4
