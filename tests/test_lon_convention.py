"""
Tests for the longitude convention helpers in nos_utils.coords.

Covers:
  - normalize_lon() for both pm180 and 0360 conventions
  - lon_convention() detection from ForcingConfig
  - Integration: rtofs._interpolate_2d_to_boundary uses correct convention
  - Integration: StructuredGridInterpolator constructor respects convention
  - Integration: BlenderProcessor subsets GFS correctly for 0-360 domain
  - Regression: Atlantic configs (pm180) produce same results as before
"""

import numpy as np
import pytest

from nos_utils.coords import normalize_lon, lon_convention, PM180, LON360
from nos_utils.config import ForcingConfig


# ---------------------------------------------------------------------------
# Unit tests: normalize_lon
# ---------------------------------------------------------------------------

class TestNormalizeLon:

    # ---- to pm180 (Atlantic) ----

    def test_0360_to_pm180_basic(self):
        lon = np.array([270.0, 0.0, 90.0, 180.0])
        out = normalize_lon(lon, "pm180")
        np.testing.assert_array_almost_equal(out, [-90.0, 0.0, 90.0, 180.0])

    def test_already_pm180_unchanged(self):
        lon = np.array([-90.0, 0.0, 90.0])
        out = normalize_lon(lon, "pm180")
        np.testing.assert_array_almost_equal(out, lon)

    def test_pm180_wraps_above_180(self):
        lon = np.array([181.0, 270.0, 359.0, 360.0])
        out = normalize_lon(lon, "pm180")
        expected = np.array([-179.0, -90.0, -1.0, 0.0])
        np.testing.assert_array_almost_equal(out, expected)

    def test_pm180_wraps_below_minus180(self):
        lon = np.array([-181.0, -270.0])
        out = normalize_lon(lon, "pm180")
        expected = np.array([179.0, 90.0])
        np.testing.assert_array_almost_equal(out, expected)

    def test_pm180_rtofs_typical(self):
        """RTOFS global grid goes 0..360; after normalization everything < 180."""
        rtofs_lon = np.array([0.0, 92.5, 180.0, 181.0, 270.0, 289.675, 359.9])
        out = normalize_lon(rtofs_lon, "pm180")
        assert np.all(out >= -180.0)
        assert np.all(out <= 180.0)

    # ---- to 0360 (Pacific) ----

    def test_pm180_to_0360_basic(self):
        lon = np.array([-90.0, 0.0, 90.0, -180.0])
        out = normalize_lon(lon, "0360")
        np.testing.assert_array_almost_equal(out, [270.0, 0.0, 90.0, 180.0])

    def test_already_0360_unchanged(self):
        lon = np.array([93.0, 180.0, 270.0, 289.675])
        out = normalize_lon(lon, "0360")
        np.testing.assert_array_almost_equal(out, lon)

    def test_0360_range(self):
        lon = np.array([-180.0, -90.0, -1.0, 0.0, 1.0, 90.0, 180.0, 270.0, 359.0])
        out = normalize_lon(lon, "0360")
        assert np.all(out >= 0.0)
        assert np.all(out < 360.0)

    def test_0360_rtofs_typical(self):
        """RTOFS global grid 0..360 should be unchanged under 0360 convention."""
        rtofs_lon = np.array([88.4, 92.5, 180.0, 270.0, 289.675, 290.3])
        out = normalize_lon(rtofs_lon, "0360")
        np.testing.assert_array_almost_equal(out, rtofs_lon)

    def test_0360_negative_inputs_wrapped(self):
        """Negative lons from HRRR after LCC regrid get wrapped to 0-360."""
        lcc_lon = np.array([-134.1, -101.2, -120.0])  # US West Coast HRRR
        out = normalize_lon(lcc_lon, "0360")
        expected = np.array([225.9, 258.8, 240.0])
        np.testing.assert_array_almost_equal(out, expected)

    # ---- scalar inputs ----

    def test_scalar_pm180(self):
        out = normalize_lon(270.0, "pm180")
        assert float(out) == pytest.approx(-90.0)

    def test_scalar_0360(self):
        out = normalize_lon(-90.0, "0360")
        assert float(out) == pytest.approx(270.0)

    # ---- edge cases ----

    def test_exact_180_pm180(self):
        """180.0 should stay 180.0 under pm180 (not fold to -180)."""
        out = normalize_lon(np.array([180.0]), "pm180")
        assert float(out[0]) == pytest.approx(180.0)

    def test_exact_0_0360(self):
        """0.0 should stay 0.0 under 0360."""
        out = normalize_lon(np.array([0.0]), "0360")
        assert float(out[0]) == pytest.approx(0.0)

    def test_roundtrip_pm180_then_0360(self):
        """pm180 → 0360 → pm180 should be identity (up to float precision)."""
        original = np.array([-90.0, 0.0, 90.0, -45.0, 45.0])
        roundtripped = normalize_lon(normalize_lon(original, "0360"), "pm180")
        np.testing.assert_array_almost_equal(roundtripped, original)

    def test_roundtrip_0360_then_pm180(self):
        """0360 → pm180 → 0360 should be identity (up to float precision)."""
        original = np.array([93.0, 180.0, 270.0, 289.675, 0.0])
        roundtripped = normalize_lon(normalize_lon(original, "pm180"), "0360")
        np.testing.assert_array_almost_equal(roundtripped, original)

    def test_bad_convention_raises(self):
        with pytest.raises(ValueError, match="Unknown lon convention"):
            normalize_lon(np.array([0.0]), "degrees_east")

    def test_preserves_shape(self):
        lon_2d = np.arange(12.0).reshape(3, 4) - 6.0
        out = normalize_lon(lon_2d, "pm180")
        assert out.shape == (3, 4)

    def test_does_not_modify_input(self):
        """normalize_lon must return a copy, not mutate the input."""
        lon = np.array([270.0, 280.0])
        original = lon.copy()
        normalize_lon(lon, "pm180")
        np.testing.assert_array_equal(lon, original)


# ---------------------------------------------------------------------------
# Unit tests: lon_convention detection
# ---------------------------------------------------------------------------

class TestLonConvention:

    def test_atlantic_secofs(self):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        assert lon_convention(cfg) == PM180

    def test_atlantic_stofs_atl(self):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        assert lon_convention(cfg) == PM180

    def test_pacific_standalone(self):
        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        assert lon_convention(cfg) == LON360

    def test_pacific_ufs(self):
        cfg = ForcingConfig.for_stofs_3d_pac_ufs(pdy="20260401", cyc=12)
        assert lon_convention(cfg) == LON360

    def test_generic_pacific_0360(self):
        """Any config with lon_min>=0 and lon_max>180 is 0360."""
        cfg = ForcingConfig(
            lon_min=93.0, lon_max=290.0,
            lat_min=-34.0, lat_max=67.0,
            pdy="20260401", cyc=12,
        )
        assert lon_convention(cfg) == LON360

    def test_generic_atlantic_pm180(self):
        cfg = ForcingConfig(
            lon_min=-90.0, lon_max=-60.0,
            lat_min=10.0, lat_max=45.0,
            pdy="20260401", cyc=12,
        )
        assert lon_convention(cfg) == PM180

    def test_no_config_attributes_raises(self):
        class Bad:
            pass
        with pytest.raises(TypeError):
            lon_convention(Bad())

    def test_pm180_constants(self):
        assert PM180 == "pm180"
        assert LON360 == "0360"


# ---------------------------------------------------------------------------
# Integration: StructuredGridInterpolator respects convention
# ---------------------------------------------------------------------------

class TestStructuredGridInterpolatorConvention:

    def test_atlantic_pm180(self):
        """Atlantic grid in 0-360 is normalized to pm180 internally."""
        from nos_utils.interp.structured_interp import StructuredGridInterpolator
        pytest.importorskip("scipy")

        # Small 4x4 RTOFS-like grid with lons 350-10 (straddles 0 meridian)
        lons_raw = np.array([[350.0, 355.0, 0.0, 5.0],
                              [350.0, 355.0, 0.0, 5.0],
                              [350.0, 355.0, 0.0, 5.0],
                              [350.0, 355.0, 0.0, 5.0]], dtype=np.float32)
        lats_raw = np.array([[10.0, 10.0, 10.0, 10.0],
                              [15.0, 15.0, 15.0, 15.0],
                              [20.0, 20.0, 20.0, 20.0],
                              [25.0, 25.0, 25.0, 25.0]], dtype=np.float32)

        interp = StructuredGridInterpolator(lons_raw, lats_raw, convention="pm180")
        # Internal lon should be in [-180, 180]
        assert interp.lon.min() >= -180.0
        assert interp.lon.max() <= 180.0

    def test_pacific_0360(self):
        """Pacific grid in -180..180 (HRRR output) normalised to 0360."""
        from nos_utils.interp.structured_interp import StructuredGridInterpolator
        pytest.importorskip("scipy")

        lons_raw = np.array([[-134.0, -120.0, -101.0, -80.0],
                              [-134.0, -120.0, -101.0, -80.0],
                              [-134.0, -120.0, -101.0, -80.0],
                              [-134.0, -120.0, -101.0, -80.0]], dtype=np.float32)
        lats_raw = np.array([[21.0, 21.0, 21.0, 21.0],
                              [35.0, 35.0, 35.0, 35.0],
                              [50.0, 50.0, 50.0, 50.0],
                              [65.0, 65.0, 65.0, 65.0]], dtype=np.float32)

        interp = StructuredGridInterpolator(lons_raw, lats_raw, convention="0360")
        # Negative lons should be folded to 0-360
        assert interp.lon.min() >= 0.0
        assert interp.lon.max() < 360.0


# ---------------------------------------------------------------------------
# Integration: RTOFS processor uses correct convention on boundary lons
# ---------------------------------------------------------------------------

class TestRTOFSConventionIntegration:

    def test_interpolate_2d_uses_pm180_for_atlantic(self, tmp_path):
        """Atlantic boundary nodes are in pm180; RTOFS data normalised to pm180."""
        from nos_utils.forcing.rtofs import RTOFSProcessor

        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path / "out")

        # Boundary nodes in pm180 range
        proc._bnd_lons = np.array([-80.0, -75.0, -70.0])
        proc._bnd_lats = np.array([30.0, 32.0, 34.0])

        # RTOFS data in 0-360
        rtofs_lon = np.linspace(260.0, 295.0, 10)   # covers -100 to -65 in pm180
        rtofs_lat = np.linspace(28.0, 36.0, 10)
        lon2d, lat2d = np.meshgrid(rtofs_lon, rtofs_lat)
        data = np.full((10, 10), 0.5, dtype=np.float32)

        # Should not raise; result should be finite (interpolation finds data)
        result = proc._interpolate_2d_to_boundary(lon2d, lat2d, data)
        assert result is not None
        assert len(result) == 3
        assert np.all(np.isfinite(result))

    def test_interpolate_2d_uses_0360_for_pacific(self, tmp_path):
        """Pacific boundary nodes are in 0-360; RTOFS data also 0-360."""
        from nos_utils.forcing.rtofs import RTOFSProcessor

        cfg = ForcingConfig.for_stofs_3d_pac(pdy="20260401", cyc=12)
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path / "out")

        # Boundary nodes in 0-360 (Indian Ocean segment near 93°E)
        proc._bnd_lons = np.array([93.5, 94.0, 94.5])
        proc._bnd_lats = np.array([-10.0, -8.0, -6.0])

        # RTOFS data in 0-360 covering the Indian Ocean
        rtofs_lon = np.linspace(88.0, 100.0, 20)
        rtofs_lat = np.linspace(-15.0, 0.0, 20)
        lon2d, lat2d = np.meshgrid(rtofs_lon, rtofs_lat)
        data = np.full((20, 20), 0.2, dtype=np.float32)

        result = proc._interpolate_2d_to_boundary(lon2d, lat2d, data)
        assert result is not None
        assert len(result) == 3
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Integration: BlenderProcessor GFS subsetting for 0-360 domain
# ---------------------------------------------------------------------------

class TestBlenderConventionIntegration:
    """Verify BlenderProcessor correctly subsets GFS data for Pacific 0-360 domain."""

    @pytest.fixture
    def pac_config(self):
        return ForcingConfig.for_stofs_3d_pac_ufs(pdy="20260401", cyc=12)

    def test_lon_convention_is_0360(self, pac_config):
        assert lon_convention(pac_config) == LON360

    def test_gfs_subset_mask_selects_correct_region(self, pac_config):
        """GFS lons in 0-360 are correctly used for Pacific domain subsetting."""
        from nos_utils.coords import normalize_lon

        # Simulate GFS 0.25-degree global grid (0-360 native)
        gfs_lon_full = np.linspace(0.0, 359.75, 1440, dtype=np.float32)

        # After normalization to 0-360 convention, the Pacific domain
        # datm_lon_min=92.5 to datm_lon_max=290.0 should select the right lons
        _conv = lon_convention(pac_config)
        assert _conv == LON360
        gfs_lon_norm = normalize_lon(gfs_lon_full, _conv)

        lon_min, lon_max, _, _ = pac_config.datm_domain
        BUFFER = 1.0
        lon_mask = (gfs_lon_norm >= lon_min - BUFFER) & (gfs_lon_norm <= lon_max + BUFFER)

        selected = gfs_lon_norm[lon_mask]
        # Should select lons 91.5..291.0 (Pacific domain + 1° buffer)
        assert selected.min() >= 91.0
        assert selected.max() <= 291.5
        # Should NOT include lons near 0 or 360 (outside Pacific domain)
        assert not np.any((selected < 91.0) | (selected > 292.0))

    def test_gfs_subset_atlantic_pm180_unchanged(self):
        """Atlantic GFS subsetting is unchanged after refactor."""
        from nos_utils.coords import normalize_lon

        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        gfs_lon_full = np.linspace(0.0, 359.75, 1440, dtype=np.float32)

        _conv = lon_convention(cfg)
        assert _conv == PM180
        gfs_lon_norm = normalize_lon(gfs_lon_full, _conv)

        lon_min, lon_max, _, _ = cfg.datm_domain if cfg.nws == 4 else cfg.domain
        BUFFER = 1.0
        lon_mask = (gfs_lon_norm >= lon_min - BUFFER) & (gfs_lon_norm <= lon_max + BUFFER)

        selected = gfs_lon_norm[lon_mask]
        # SECOFS domain ~-88 to -63; with buffer should be roughly -89 to -62
        assert selected.min() >= -90.0
        assert selected.max() <= -61.0


# ---------------------------------------------------------------------------
# Regression: Atlantic results unchanged
# ---------------------------------------------------------------------------

class TestAtlanticRegression:
    """Ensure the refactored normalizations produce the same results for
    Atlantic configs as the original np.where(lon>180, lon-360, lon)."""

    def test_normalize_lon_pm180_matches_old_formula(self):
        """normalize_lon('pm180') must be identical to the old inline formula."""
        rtofs_lon = np.linspace(0.0, 360.0, 721, dtype=np.float64)
        old = np.where(rtofs_lon > 180, rtofs_lon - 360, rtofs_lon)
        new = normalize_lon(rtofs_lon, "pm180")
        np.testing.assert_array_almost_equal(new, old, decimal=10)

    def test_rtofs_domain_subset_atlantic_unchanged(self, tmp_path):
        """After normalisation, Atlantic boundary lon subset gives same coverage."""
        from nos_utils.forcing.rtofs import RTOFSProcessor

        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path / "out")

        # Representative Atlantic boundary nodes
        proc._bnd_lons = np.array([-88.0, -80.0, -70.0, -63.0])
        proc._bnd_lats = np.array([20.0, 25.0, 30.0, 35.0])

        # RTOFS 0-360 grid covering the Atlantic
        rtofs_lon = np.linspace(260.0, 300.0, 30)  # -100 to -60 in pm180
        rtofs_lat = np.linspace(18.0, 38.0, 30)
        lon2d, lat2d = np.meshgrid(rtofs_lon, rtofs_lat)
        data = np.full((30, 30), 1.0, dtype=np.float32)

        result = proc._interpolate_2d_to_boundary(lon2d, lat2d, data)
        assert result is not None
        assert np.all(np.isfinite(result))
        # All boundary nodes should get value ≈ 1.0 (full coverage)
        np.testing.assert_array_almost_equal(result, np.ones(4), decimal=5)
