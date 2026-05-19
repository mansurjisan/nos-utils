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
        # CSV lives under the cycle's PDY (operational $COMINlaw/<PDY>/...)
        # but must cover dates from model_t0 = 2026-03-31 12:00 onward.
        csv = input_dir / pdy / "can_streamgauge" / DEFAULT_CSV_NAME
        _write_hydrometric_csv(csv, start_date="2026-03-31", n_days=8,
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
        # Time column starts at 0 (= model_t0 = cycle - nowcast_hours) and
        # steps by 86400 seconds.
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
        # Only yesterday's CSV exists. With nowcast_hours=24 the model_t0
        # for today's cyc=12 run is 2026-04-01 12:00, so the CSV must
        # include that anchor day.
        csv = input_dir / "20260401" / "can_streamgauge" / DEFAULT_CSV_NAME
        _write_hydrometric_csv(csv, start_date="2026-04-01", n_days=7,
                               flow_cms=7500.0)
        (input_dir / pdy / "can_streamgauge").mkdir(parents=True)

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        proc = StLawrenceProcessor(cfg, input_dir, out_dir)

        result = proc.process()
        # model_t0 = 2026-04-02 12:00 - 24h = 2026-04-01 12:00, which is the
        # first row of yesterday's CSV — should succeed.
        assert result.success
        assert (out_dir / "flux.th").exists()


class TestStLawrenceTempFromSflux:
    """TEM_1.th derivation from GFS radiation sflux file."""

    def test_applies_regression(self, tmp_path):
        pdy = "20260401"
        input_dir = tmp_path / "comin"
        csv = input_dir / pdy / "can_streamgauge" / DEFAULT_CSV_NAME
        # model_t0 = cycle - 24h = 2026-03-31 12:00; CSV must cover that.
        _write_hydrometric_csv(csv, start_date="2026-03-31", n_days=8)

        sflux_file = tmp_path / "sflux" / "sflux_rad_1.1.nc"
        # sflux must also extend back to model_t0 = 2026-03-31 12:00.
        _write_fake_sflux_rad(
            sflux_file, datetime(2026, 3, 31, 12, 0, 0), n_times=192,
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


class TestStLawrenceCustomSubdir:
    """StLawrenceProcessor honors a custom CSV subdirectory.

    Operational WCOSS2 uses $COMINlaw/<pdy>/canadian_water/ with a
    QC_..._hourly_hydrometric.csv filename, not the legacy
    can_streamgauge/02OA016_hydrometric.csv layout.
    """

    def test_custom_subdir_and_csv_name(self, tmp_path):
        pdy = "20260401"
        input_dir = tmp_path / "comin"
        csv_name = "QC_02OA016_hourly_hydrometric.csv"
        csv = input_dir / pdy / "canadian_water" / csv_name
        _write_hydrometric_csv(csv, start_date="2026-03-31", n_days=8,
                               flow_cms=8000.0)

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        proc = StLawrenceProcessor(
            cfg, input_dir, out_dir,
            csv_name=csv_name,
            subdir="canadian_water",
        )

        result = proc.process()
        assert result.success, result.errors
        assert (out_dir / "flux.th").exists()
        assert proc._csv_path_for(datetime(2026, 4, 1)).name == csv_name
        assert "canadian_water" in str(proc._csv_path_for(datetime(2026, 4, 1)))

    def test_legacy_subdir_still_default(self, tmp_path):
        # Without an explicit subdir the processor keeps the legacy layout
        # so existing callers (and direct construction) are unaffected.
        pdy = "20260401"
        input_dir = tmp_path / "comin"
        csv = input_dir / pdy / "can_streamgauge" / DEFAULT_CSV_NAME
        _write_hydrometric_csv(csv, start_date="2026-03-31", n_days=8)

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        proc = StLawrenceProcessor(cfg, input_dir, out_dir)

        result = proc.process()
        assert result.success, result.errors


def _write_csv_rows(path: Path, rows) -> None:
    """Write an EC-format CSV from explicit (timestamp, param, value) rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["STATION,date_local,parameter,value,col4,col5,col6,col7,col8"]
    for ts, param, val in rows:
        lines.append(f"02OA016,{ts},{param},{val},-,-,-,-,-")
    path.write_text("\n".join(lines) + "\n")


class TestCsvLookupRobustness:
    """Regression: live CSV cadence/timezone/coverage need not land
    exactly on the model time axis.

    The 20260518 t00z STOFS-UFS prep cold-started St. Lawrence because
    ``_read_hydrometric_csv`` did an exact-timestamp ``df_flow.loc[dt]``
    that raised on day 0 for model_t0 = cycle - nowcast_hours
    (2026-05-17 00:00 UTC) whenever the dcom CSV's discharge rows didn't
    fall on that exact instant. The asof + UTC-normalise fix must
    discover the value anyway.
    """

    def _cfg_dir(self, tmp_path):
        # Mirror production: pdy=20260518 cyc=0 -> model_t0 2026-05-17 00:00.
        input_dir = tmp_path / "comin"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260518", cyc=0)
        return cfg, input_dir, tmp_path / "out"

    def test_subdaily_rows_not_on_model_hour(self, tmp_path):
        """Discharge reported at 06:00/18:00 (never 00:00) — old code
        raised at day 0; asof picks the nearest prior reading."""
        cfg, input_dir, out_dir = self._cfg_dir(tmp_path)
        csv = input_dir / "20260518" / "can_streamgauge" / DEFAULT_CSV_NAME
        rows = []
        for day in (16, 17, 18):
            for hh in (6, 18):
                ts = f"2026-05-{day:02d}T{hh:02d}:00:00+00:00"
                rows.append((ts, 47, 8000.0 + day))
                rows.append((ts, 5, 5.0))
        _write_csv_rows(csv, rows)

        result = StLawrenceProcessor(cfg, input_dir, out_dir).process()
        assert result.success, result.errors
        data = np.loadtxt(out_dir / "flux.th")
        assert data.shape == (7, 2)
        assert (data[:, 1] < 0).all()

    def test_csv_starts_after_model_t0_backfills(self, tmp_path):
        """CSV only covers the cycle day, not model_t0 the day before —
        the earliest sample is carried backward instead of failing."""
        cfg, input_dir, out_dir = self._cfg_dir(tmp_path)
        csv = input_dir / "20260518" / "can_streamgauge" / DEFAULT_CSV_NAME
        rows = [
            (f"2026-05-18T{hh:02d}:00:00+00:00", 47, 9000.0)
            for hh in range(0, 19, 6)
        ]
        _write_csv_rows(csv, rows)

        result = StLawrenceProcessor(cfg, input_dir, out_dir).process()
        assert result.success, result.errors
        data = np.loadtxt(out_dir / "flux.th")
        assert data.shape == (7, 2)
        assert np.isclose(abs(data[0, 1]), 9000.0, atol=1e-3)

    def test_tz_naive_local_timestamps(self, tmp_path):
        """Naive (no offset) timestamps must be treated as UTC, not
        rejected by a tz-aware exact lookup."""
        cfg, input_dir, out_dir = self._cfg_dir(tmp_path)
        csv = input_dir / "20260518" / "can_streamgauge" / DEFAULT_CSV_NAME
        rows = []
        for day in (16, 17, 18, 19):
            rows.append((f"2026-05-{day:02d} 12:00:00", 47, 7000.0 + day))
        _write_csv_rows(csv, rows)

        result = StLawrenceProcessor(cfg, input_dir, out_dir).process()
        assert result.success, result.errors
        assert (out_dir / "flux.th").exists()

    def test_empty_discharge_still_triggers_archive_fallback(self, tmp_path):
        """A CSV with no discharge (param 47) rows must still raise so
        the previous-cycle archive fallback is used (semantics preserved
        — we never fabricate flow from an empty series)."""
        cfg, input_dir, out_dir = self._cfg_dir(tmp_path)
        csv = input_dir / "20260518" / "can_streamgauge" / DEFAULT_CSV_NAME
        # Temperature rows only — no parameter 47.
        _write_csv_rows(csv, [
            (f"2026-05-{d:02d}T00:00:00+00:00", 5, 5.0) for d in (16, 17, 18)
        ])
        archive_dir = tmp_path / "prev_rerun"
        archive_dir.mkdir()
        (archive_dir / "stofs_3d_atl.t00z.riv.obs.flux.th").write_text(
            "\n".join(f"{i*86400} {-7000.0:.3f}" for i in range(7)) + "\n"
        )
        (archive_dir / "stofs_3d_atl.t00z.riv.obs.tem_1.th").write_text(
            "\n".join(f"{i*86400} {5.0:.3f}" for i in range(7)) + "\n"
        )

        result = StLawrenceProcessor(
            cfg, input_dir, out_dir,
            prev_rerun_dir=archive_dir,
            archive_prefix="stofs_3d_atl.t00z",
        ).process()
        assert result.success, result.errors
        assert (out_dir / "flux.th").exists()
        assert (out_dir / "TEM_1.th").exists()


class TestADTDcomrootFallback:
    """adt.py falls back to $DCOMROOT when COMINadt is unset.

    Operational WCOSS2 J-jobs don't always export COMINadt; the CMEMS
    ADT files live under $DCOMROOT, so _find_adt_data() must resolve
    them there. Regression guard for the STOFS-3D-ATL "ADT missing" bug.
    """

    def _write_adt(self, root: Path, date_str: str) -> Path:
        adt = (root / date_str / "validation_data" / "marine" /
               "cmems" / "ssh" /
               f"nrt_global_allsat_phy_l4_{date_str}_{date_str}.nc")
        adt.parent.mkdir(parents=True, exist_ok=True)
        adt.write_bytes(b"")  # presence is all _find_adt_data checks
        return adt

    def test_falls_back_to_dcomroot(self, tmp_path, monkeypatch):
        from nos_utils.forcing.adt import ADTBlender

        dcom = tmp_path / "dcom"
        expected = self._write_adt(dcom, "20260401")

        monkeypatch.delenv("COMINadt", raising=False)
        monkeypatch.setenv("DCOMROOT", str(dcom))

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        blender = ADTBlender(cfg, tmp_path / "rtofs")
        found = blender._find_adt_data()
        assert found == expected

    def test_cominadt_takes_precedence(self, tmp_path, monkeypatch):
        from nos_utils.forcing.adt import ADTBlender

        adt_root = tmp_path / "adt"
        dcom = tmp_path / "dcom"
        expected = self._write_adt(adt_root, "20260401")
        # A different (wrong) file under DCOMROOT must be ignored.
        self._write_adt(dcom, "20260401")

        monkeypatch.setenv("COMINadt", str(adt_root))
        monkeypatch.setenv("DCOMROOT", str(dcom))

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        blender = ADTBlender(cfg, tmp_path / "rtofs")
        found = blender._find_adt_data()
        assert found == expected

    def test_missing_everywhere_returns_none(self, tmp_path, monkeypatch):
        from nos_utils.forcing.adt import ADTBlender

        monkeypatch.delenv("COMINadt", raising=False)
        monkeypatch.delenv("DCOMROOT", raising=False)

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        blender = ADTBlender(cfg, tmp_path / "rtofs")
        assert blender._find_adt_data() is None
