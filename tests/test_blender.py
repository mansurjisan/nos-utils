"""Tests for BlenderProcessor."""

from pathlib import Path
import numpy as np
import pytest
from nos_utils.forcing.blender import BlenderProcessor, rotate_winds_lcc, SFLUX_TO_DATM


class TestRotateWinds:
    def test_zero_rotation_at_lov(self):
        """At LoV longitude, rotation angle should be 0."""
        u = np.array([[10.0]])
        v = np.array([[0.0]])
        lon = np.array([[-97.5]])  # LoV
        u_e, v_e = rotate_winds_lcc(u, v, lon)
        assert u_e[0, 0] == pytest.approx(10.0, abs=0.001)
        assert v_e[0, 0] == pytest.approx(0.0, abs=0.001)

    def test_rotation_away_from_lov(self):
        """Away from LoV, winds should be rotated."""
        u = np.array([[10.0]])
        v = np.array([[0.0]])
        lon = np.array([[-80.0]])  # 17.5° east of LoV
        u_e, v_e = rotate_winds_lcc(u, v, lon)
        # Should have some v component after rotation
        assert abs(v_e[0, 0]) > 0.1

    def test_preserves_wind_speed(self):
        """Rotation should preserve wind magnitude."""
        u = np.array([[3.0]])
        v = np.array([[4.0]])
        lon = np.array([[-75.0]])
        u_e, v_e = rotate_winds_lcc(u, v, lon)
        speed_orig = np.sqrt(u**2 + v**2)
        speed_rot = np.sqrt(u_e**2 + v_e**2)
        assert speed_rot[0, 0] == pytest.approx(speed_orig[0, 0], rel=1e-5)


class TestBlenderProcessor:
    def test_no_gfs_fails(self, mock_config, tmp_path):
        proc = BlenderProcessor(mock_config, tmp_path / "empty", tmp_path / "out")
        result = proc.process()
        assert not result.success
        assert "No GFS" in result.errors[0]

    def test_variable_mapping_complete(self):
        """All 8 sflux variables should have DATM mappings."""
        expected = ["uwind", "vwind", "stmp", "spfh", "prmsl", "prate", "dswrf", "dlwrf"]
        for var in expected:
            assert var in SFLUX_TO_DATM, f"Missing DATM mapping for {var}"


class TestNetCDFUtils:
    def test_validate_monotonic_ok(self):
        from nos_utils.io.netcdf_utils import validate_monotonic
        assert validate_monotonic([1.0, 2.0, 3.0])

    def test_validate_monotonic_fail(self):
        from nos_utils.io.netcdf_utils import validate_monotonic
        with pytest.raises(ValueError, match="Non-monotonic"):
            validate_monotonic([1.0, 3.0, 2.0])

    def test_replace_fill_values(self):
        from nos_utils.io.netcdf_utils import replace_fill_values
        data = np.array([1.0, 99999.0, -30000.0, 5.0])
        result = replace_fill_values(data, threshold=10000.0, fill_value=-9999.0)
        assert result[0] == 1.0
        assert result[1] == -9999.0
        assert result[2] == -9999.0
        assert result[3] == 5.0
