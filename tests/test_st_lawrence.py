"""Tests for the St. Lawrence River forcing processor."""

from pathlib import Path
from datetime import datetime

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
nc = pytest.importorskip("netCDF4")

from nos_utils.config import ForcingConfig  # noqa: E402
from nos_utils.forcing.st_lawrence import (  # noqa: E402
    StLawrenceProcessor,
    AIR_TO_WATER_SLOPE,
    AIR_TO_WATER_INTERCEPT,
    DEFAULT_CSV_NAME,
)


def _write_hydrometric_csv(path: Path, start_date: str = "2026-04-01",
                            n_days: int = 7, flow_cms: float = 8000.0,
                            temp_c: float = 5.0) -> None:
    """Write a minimal CSV matching EC hydrometric format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "STATION,date_local,parameter,value,col4,col5,col6,col7,col8",
    ]
    base = pd.Timestamp(f"{start_date} 12:00:00", tz="UTC")
    for i in range(n_days):
        dt = base + pd.Timedelta(days=i)
        lines.append(f"02OA016,{dt.isoformat()},47,{flow_cms + i*10},-,-,-,-,-")
        lines.append(f"02OA016,{dt.isoformat()},5,{temp_c + i*0.1},-,-,-,-,-")
    path.write_text("\n".join(lines) + "\n")


def _write_fake_sflux_rad(path: Path, start_time: datetime,
                          n_times: int = 48, grid_nx: int = 20,
                          grid_ny: int = 15) -> None:
    """Write a minimal sflux rad file covering the St. Lawrence mouth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(str(path), "w") as ds:
        ds.createDimension("ny_grid", grid_ny)
        ds.createDimension("nx_grid", grid_nx)
        ds.createDimension("time", n_times)

        lon_v = ds.createVariable("lon", "f4", ("ny_grid", "nx_grid"))
        lat_v = ds.createVariable("lat", "f4", ("ny_grid", "nx_grid"))
        time_v = ds.createVariable("time", "f8", ("time",))
        stmp_v = ds.createVariable("stmp", "f4", ("time", "ny_grid", "nx_grid"))

        # River mouth is (45.415, -73.623) — center grid on that point.
        lons_1d = np.linspace(-74.0, -73.0, grid_nx)
        lats_1d = np.linspace(45.0, 45.8, grid_ny)
        lon_v[:], lat_v[:] = np.meshgrid(lons_1d, lats_1d)

        time_v.units = f"days since {start_time:%Y-%m-%d %H:%M:%S}"
        time_v[:] = np.arange(n_times) / 24.0  # hourly in days

        # Temperature field: 10°C constant above freezing.
        stmp_v[:] = 283.15  # 10°C in Kelvin


class TestStLawrenceFluxTh:
    """flux.th generation from the hydrometric CSV."""

    def test_writes_7_day_discharge(self, tmp_path):
        pdy = "20260401"
        input_dir = tmp_path / "comin"
        csv = input_dir / pdy / "can_streamgauge" / DEFAULT_CSV_NAME
        _write_hydrometric_csv(csv, start_date="2026-04-01", n_days=7,
                               flow_cms=8000.0)

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        # 24h nowcast + 108h forecast = 5.5 days, rounded up to 6 -> 7 rows.
        proc = StLawrenceProcessor(cfg, input_dir, out_dir)

        result = proc.process()

        assert result.success
        flux_path = out_dir / "flux.th"
        assert flux_path.exists()
        data = np.loadtxt(flux_path)
        # 6 days forecast + 1 -> 7 rows (132h/24 = 5.5 -> ceil=6, +1 = 7)
        assert data.shape == (7, 2)
        # Time column starts at 0 and steps by 86400 seconds.
        assert data[0, 0] == 0
        assert data[1, 0] == 86400
        # Negative sign = inflow in SCHISM.
        assert (data[:, 1] < 0).all()
        assert np.isclose(abs(data[0, 1]), 8000.0, atol=1e-3)

    def test_missing_csv_returns_failure(self, tmp_path):
        pdy = "20260401"
        input_dir = tmp_path / "comin"
        (input_dir / pdy / "can_streamgauge").mkdir(parents=True)

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        proc = StLawrenceProcessor(cfg, input_dir, out_dir)

        result = proc.process()
        assert not result.success
        assert any("no previous-cycle archive" in e.lower() or
                   "csv missing" in e.lower() for e in result.errors)

    def test_previous_day_fallback(self, tmp_path):
        pdy = "20260402"
        input_dir = tmp_path / "comin"
        # Only yesterday's CSV exists.
        csv = input_dir / "20260401" / "can_streamgauge" / DEFAULT_CSV_NAME
        _write_hydrometric_csv(csv, start_date="2026-04-01", n_days=7,
                               flow_cms=7500.0)
        (input_dir / pdy / "can_streamgauge").mkdir(parents=True)

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        proc = StLawrenceProcessor(cfg, input_dir, out_dir)

        result = proc.process()
        # The yesterday CSV starts at 2026-04-01 12:00, today's start is
        # 2026-04-02 12:00. Yesterday's CSV has those rows too, so it
        # should succeed.
        assert result.success
        assert (out_dir / "flux.th").exists()


class TestStLawrenceTempFromSflux:
    """TEM_1.th derivation from GFS radiation sflux file."""

    def test_applies_regression(self, tmp_path):
        pdy = "20260401"
        input_dir = tmp_path / "comin"
        csv = input_dir / pdy / "can_streamgauge" / DEFAULT_CSV_NAME
        _write_hydrometric_csv(csv, start_date="2026-04-01", n_days=7)

        sflux_file = tmp_path / "sflux" / "sflux_rad_1.1.nc"
        _write_fake_sflux_rad(
            sflux_file, datetime(2026, 4, 1, 12, 0, 0), n_times=168,
        )

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        proc = StLawrenceProcessor(
            cfg, input_dir, out_dir, sflux_rad_file=sflux_file,
        )

        result = proc.process()
        assert result.success
        temp_path = out_dir / "TEM_1.th"
        assert temp_path.exists()

        data = np.loadtxt(temp_path)
        # Constant 10°C air temp → 0.83*10 + 2.817 = 11.147°C water.
        expected = AIR_TO_WATER_SLOPE * 10.0 + AIR_TO_WATER_INTERCEPT
        assert np.allclose(data[:, 1], expected, atol=0.01)


class TestArchiveFallback:
    """Previous-cycle archive fallback when CSV is missing entirely."""

    def test_uses_archive(self, tmp_path):
        pdy = "20260401"
        input_dir = tmp_path / "comin"
        (input_dir / pdy / "can_streamgauge").mkdir(parents=True)
        # No CSV for today or yesterday.

        archive_dir = tmp_path / "prev_rerun"
        archive_dir.mkdir()
        archive_flux = archive_dir / "stofs_3d_atl.t12z.riv.obs.flux.th"
        archive_tem = archive_dir / "stofs_3d_atl.t12z.riv.obs.tem_1.th"
        archive_flux.write_text("\n".join([
            f"{i*86400} {-7000.0 - i*10:.3f}" for i in range(7)
        ]) + "\n")
        archive_tem.write_text("\n".join([
            f"{i*86400} {5.0 + i*0.1:.3f}" for i in range(7)
        ]) + "\n")

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        proc = StLawrenceProcessor(
            cfg, input_dir, out_dir,
            prev_rerun_dir=archive_dir,
            archive_prefix="stofs_3d_atl.t12z",
        )

        result = proc.process()
        assert result.success, result.errors
        # flux.th should exist in out_dir and contain shifted values.
        assert (out_dir / "flux.th").exists()
        assert (out_dir / "TEM_1.th").exists()
