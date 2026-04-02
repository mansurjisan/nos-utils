"""Shared test fixtures for nos-utils."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig


@pytest.fixture
def mock_config():
    """Minimal ForcingConfig for testing."""
    return ForcingConfig(
        lon_min=-80.0, lon_max=-70.0,
        lat_min=25.0, lat_max=35.0,
        pdy="20260401", cyc=12,
        nowcast_hours=6, forecast_hours=48,
    )


@pytest.fixture
def secofs_config():
    """SECOFS ForcingConfig."""
    return ForcingConfig.for_secofs(pdy="20260401", cyc=12)


@pytest.fixture
def stofs_config():
    """STOFS-3D-ATL ForcingConfig."""
    return ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)


@pytest.fixture
def synthetic_grid():
    """Small 5x5 lat/lon grid for writer tests."""
    lons = np.linspace(-80.0, -70.0, 5)
    lats = np.linspace(25.0, 35.0, 5)
    return lons, lats


@pytest.fixture
def synthetic_data(synthetic_grid):
    """Synthetic forcing data matching the 5x5 grid with 3 time steps."""
    lons, lats = synthetic_grid
    ny, nx = len(lats), len(lons)
    n_times = 3

    data = {}
    for var in ["uwind", "vwind", "prmsl", "stmp", "spfh", "dlwrf", "dswrf", "prate"]:
        data[var] = [np.random.rand(ny, nx).astype(np.float32) for _ in range(n_times)]

    base = datetime(2026, 3, 31, 6, 0, 0)
    times = [base + timedelta(hours=i * 3) for i in range(n_times)]

    return data, times


@pytest.fixture
def tmp_output_dir():
    """Temporary output directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory(prefix="nos_utils_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_gfs_dir(tmp_path):
    """Create mock GFS directory structure with empty GRIB2 files."""
    gfs_root = tmp_path / "gfs_data"

    # Create files for today's 12z cycle: f000-f054
    date_str = "20260401"
    atmos_dir = gfs_root / f"gfs.{date_str}" / "12" / "atmos"
    atmos_dir.mkdir(parents=True)

    for fhr in range(0, 55):
        grib_file = atmos_dir / f"gfs.t12z.pgrb2.0p25.f{fhr:03d}"
        # Write enough bytes to pass size check (or not, for QC tests)
        grib_file.write_bytes(b"\x00" * 1024)

    # Also create yesterday's 12z for backup
    prev_dir = gfs_root / f"gfs.20260331" / "12" / "atmos"
    prev_dir.mkdir(parents=True)
    for fhr in range(0, 55):
        grib_file = prev_dir / f"gfs.t12z.pgrb2.0p25.f{fhr:03d}"
        grib_file.write_bytes(b"\x00" * 1024)

    return gfs_root


@pytest.fixture
def mock_hrrr_dir(tmp_path):
    """Create mock HRRR directory structure."""
    hrrr_root = tmp_path / "hrrr_data"

    for date_str in ["20260331", "20260401"]:
        conus_dir = hrrr_root / f"hrrr.{date_str}" / "conus"
        conus_dir.mkdir(parents=True)

        for hr in range(0, 24):
            grib_file = conus_dir / f"hrrr.t{hr:02d}z.wrfsfcf01.grib2"
            grib_file.write_bytes(b"\x00" * 512)

    # Forecast files from 12z
    conus_dir = hrrr_root / "hrrr.20260401" / "conus"
    for fhr in range(1, 49):
        grib_file = conus_dir / f"hrrr.t12z.wrfsfcf{fhr:02d}.grib2"
        grib_file.write_bytes(b"\x00" * 512)

    return hrrr_root


@pytest.fixture
def mock_gefs_dir(tmp_path):
    """Create mock GEFS directory structure for member gep01."""
    gefs_root = tmp_path / "gefs_data"

    for date_str in ["20260331", "20260401"]:
        for cyc in [0, 6, 12, 18]:
            product_dir = (gefs_root / f"gefs.{date_str}" / f"{cyc:02d}"
                          / "atmos" / "pgrb2sp25")
            product_dir.mkdir(parents=True)

            max_fhr = 132 if cyc == 12 else 6
            for fhr in range(3, max_fhr + 1, 3):
                for prefix in ["gep01", "gec00"]:
                    grib_file = product_dir / f"{prefix}.t{cyc:02d}z.pgrb2s.0p25.f{fhr:03d}"
                    grib_file.write_bytes(b"\x00" * 1024)

    return gefs_root
