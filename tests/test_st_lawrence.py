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
                            temp_c: float = 5.0, freq: str = "h",
                            tz_offset: str = "-05:00") -> None:
    """Write a CSV in the real operational *wide* ECCC hydrometric layout.

    This mirrors the NCO dcom ``QC_02OA016_hourly_hydrometric.csv`` that
    ``gen_fluxth_st_lawrence_riv.py`` reads: 10 columns, with the local
    timezone-aware Date at column 1 and Discharge at column 6 (water level
    + bilingual grade/symbol/approval qualifier columns in between). It is
    NOT the parameter-coded "long" table (no ``parameter`` column).

    Cadence defaults to hourly over a ``-05:00`` local span so the row at
    each daily UTC anchor (12:00Z == 07:00 local) exists; ``temp_c`` is
    unused (the wide export has no temperature column) and kept only for
    signature compatibility with existing callers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ID,Date,"
        "Water Level / Niveau d'eau (m),Grade,Symbol / Symbole,Approval / Approbation,"
        "Discharge / Débit (m³/s),Grade,Symbol / Symbole,Approval / Approbation"
    )
    lines = [header]
    # Anchor at 12:00 UTC == 07:00 local for a -05:00 offset, then step in
    # local time so each daily UTC anchor lands on a written row.
    base_utc = pd.Timestamp(f"{start_date} 12:00:00", tz="UTC")
    base_local = base_utc.tz_convert(tz_offset)
    periods = int(round(n_days * 24)) if freq == "h" else int(round(n_days * 24 * 12))
    step = pd.Timedelta(hours=1) if freq == "h" else pd.Timedelta(minutes=5)
    for i in range(periods):
        dt_local = base_local + i * step
        # ISO string carrying the literal local offset (e.g. "...-05:00").
        date_str = dt_local.isoformat()
        # Discharge ramps with the elapsed *day* so the daily-anchor value is
        # deterministic regardless of cadence.
        day = (dt_local - base_local).total_seconds() / 86400.0
        flow = flow_cms + day * 10.0
        wl = 5.0 + day * 0.01
        lines.append(
            f"02OA016,{date_str},{wl:.3f},-1,,Final / Finale,"
            f"{flow:.3f},-1,,Final / Finale"
        )
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


class TestStLawrenceWideCSVFormat:
    """Regression: the parser must read the real *wide* ECCC dcom layout.

    The operational ``QC_02OA016_hourly_hydrometric.csv`` is a wide table
    (discharge at column 6, local tz-aware Date at column 1) — NOT the
    parameter-coded "long" table. The previous parser filtered
    ``parameter == 47``, matched nothing on the real file, and raised
    "No discharge data" at day 0, producing zero St. Lawrence files and a
    downstream SCHISM ``other_hot_init`` abort on the missing ``flux.th``.
    """

    def _build(self, tmp_path, **csv_kwargs):
        pdy = "20260401"
        input_dir = tmp_path / "comin"
        csv = input_dir / pdy / "canadian_water" / "QC_02OA016_hourly_hydrometric.csv"
        # model_t0 = cyc 12 - 24h nowcast = 2026-03-31 12:00Z; CSV must
        # cover from that anchor forward.
        _write_hydrometric_csv(csv, start_date="2026-03-31", n_days=8,
                               flow_cms=8000.0, **csv_kwargs)
        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        proc = StLawrenceProcessor(
            cfg, input_dir, out_dir,
            csv_name="QC_02OA016_hourly_hydrometric.csv",
            subdir="canadian_water",
        )
        return proc, out_dir

    def test_reads_discharge_from_wide_column_hourly(self, tmp_path):
        proc, out_dir = self._build(tmp_path, freq="h")
        # Exercise the parser directly so a successful read can't be masked
        # by the archive fallback in process().
        start = proc._cycle_datetime()
        n_days_total = proc._n_days_total()
        dv_hind = proc._daily_range(start, days=1)
        dv_full = proc._daily_range(start, days=n_days_total)

        # Must NOT raise "No discharge data" on the real wide layout.
        series = proc._read_hydrometric_csv(
            proc._find_csv(proc._pdy_datetime()), dv_hind, dv_full
        )
        # Day-0 anchor (model_t0 = 2026-03-31 12:00Z == 07:00 -05:00) reads
        # the discharge column directly = 8000.0 (ramp offset 0).
        assert np.isclose(series.flow_cms[0], 8000.0, atol=1e-3)
        # Day 1 anchor steps the ramp by one day -> 8010.0.
        assert np.isclose(series.flow_cms[1], 8010.0, atol=1e-3)
        # Series length = nowcast_days + forecast_days + 1 rows.
        assert len(series.flow_cms) == len(series.seconds_from_start)
        # No CSV temperature column -> sentinel (overwritten by sflux later).
        assert series.temp_c[0] == -9999.0

    def test_reads_discharge_5min_cadence(self, tmp_path):
        # Real dcom can be sub-hourly; a 5-min cadence must still align the
        # exact 12:00Z daily anchors.
        proc, out_dir = self._build(tmp_path, freq="5min")
        start = proc._cycle_datetime()
        dv_hind = proc._daily_range(start, days=1)
        dv_full = proc._daily_range(start, days=proc._n_days_total())
        series = proc._read_hydrometric_csv(
            proc._find_csv(proc._pdy_datetime()), dv_hind, dv_full
        )
        assert np.isclose(series.flow_cms[0], 8000.0, atol=1e-3)

    def test_process_writes_flux_th_from_wide_csv(self, tmp_path):
        proc, out_dir = self._build(tmp_path, freq="h")
        result = proc.process()
        assert result.success, result.errors
        flux_path = out_dir / "flux.th"
        assert flux_path.exists()
        data = np.loadtxt(flux_path)
        # Inflow is written negative; day-0 magnitude == observed discharge.
        assert (data[:, 1] < 0).all()
        assert np.isclose(abs(data[0, 1]), 8000.0, atol=1e-3)

    def test_local_timezone_converted_to_utc(self, tmp_path):
        """A non-UTC local offset (-05:00) must be tz_convert'd to UTC.

        Without ``tz_convert('UTC')`` the local 07:00-05:00 row would index
        at 07:00 *naive*, never matching the 12:00Z daily lookup key, and the
        day-0 raise would fire even with the correct discharge column.
        """
        proc, out_dir = self._build(tmp_path, freq="h", tz_offset="-05:00")
        start = proc._cycle_datetime()
        dv_hind = proc._daily_range(start, days=1)
        dv_full = proc._daily_range(start, days=proc._n_days_total())
        # If tz handling were wrong this would raise KeyError("No discharge…").
        series = proc._read_hydrometric_csv(
            proc._find_csv(proc._pdy_datetime()), dv_hind, dv_full
        )
        assert np.isclose(series.flow_cms[0], 8000.0, atol=1e-3)

    def test_parameter_coded_long_file_no_longer_required(self, tmp_path):
        """Regression guard: a parameter==47 "long" file is NOT what the
        parser expects anymore. The real fix reads a fixed discharge column;
        a 4-column long-format table (ID,date,parameter,value) lacks a
        column 6 and must be rejected with a clear error rather than silently
        matching ``parameter == 47``.
        """
        pdy = "20260401"
        input_dir = tmp_path / "comin"
        csv = input_dir / pdy / "canadian_water" / "QC_02OA016_hourly_hydrometric.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        base = pd.Timestamp("2026-03-31 12:00:00", tz="UTC")
        long_lines = ["STATION,date_local,parameter,value"]
        for i in range(8):
            dt = base + pd.Timedelta(days=i)
            long_lines.append(f"02OA016,{dt.isoformat()},47,{8000.0 + i*10}")
            long_lines.append(f"02OA016,{dt.isoformat()},5,{5.0 + i*0.1}")
        csv.write_text("\n".join(long_lines) + "\n")

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=12)
        proc = StLawrenceProcessor(
            cfg, input_dir, out_dir,
            csv_name="QC_02OA016_hourly_hydrometric.csv",
            subdir="canadian_water",
        )
        start = proc._cycle_datetime()
        dv_hind = proc._daily_range(start, days=1)
        dv_full = proc._daily_range(start, days=proc._n_days_total())
        # The old long layout has only 4 columns -> rejected (no col 6).
        with pytest.raises(ValueError, match="columns"):
            proc._read_hydrometric_csv(
                proc._find_csv(proc._pdy_datetime()), dv_hind, dv_full
            )


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

    def test_glob_fallback_operational_prefix(self, tmp_path):
        """Bootstrapping from an operational STOFS rerun dir: the files
        carry the production prefix (stofs_3d_atl.t12z.*) but the UFS
        run's archive_prefix is stofs_3d_atl_ufs.t00z. The exact match
        misses; the *.riv.obs.<kind>.th glob must still find them."""
        pdy = "20260518"
        input_dir = tmp_path / "comin"
        (input_dir / pdy / "can_streamgauge").mkdir(parents=True)
        # No CSV anywhere -> exact-lookup raises -> archive fallback.

        rerun = tmp_path / "ops_rerun"
        rerun.mkdir()
        (rerun / "stofs_3d_atl.t12z.riv.obs.flux.th").write_text(
            "\n".join(f"{i*86400} {-8000.0 - i*5:.3f}" for i in range(7)) + "\n"
        )
        (rerun / "stofs_3d_atl.t12z.riv.obs.tem_1.th").write_text(
            "\n".join(f"{i*86400} {4.0:.3f}" for i in range(7)) + "\n"
        )

        out_dir = tmp_path / "out"
        cfg = ForcingConfig.for_stofs_3d_atl(pdy=pdy, cyc=0)
        proc = StLawrenceProcessor(
            cfg, input_dir, out_dir,
            prev_rerun_dir=rerun,
            archive_prefix="stofs_3d_atl_ufs.t00z",  # deliberately mismatched
        )

        result = proc.process()
        assert result.success, result.errors
        assert (out_dir / "flux.th").exists()
        assert (out_dir / "TEM_1.th").exists()
        # Shifted operational discharge made it through.
        data = np.loadtxt(out_dir / "flux.th")
        assert data.shape == (7, 2)
        assert (data[:, 1] < 0).all()


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
