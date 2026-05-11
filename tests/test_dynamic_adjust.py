"""Tests for the STOFS dynamic SSH adjustment processor."""

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
scipy = pytest.importorskip("scipy")
nc_mod = pytest.importorskip("netCDF4")

from nos_utils.config import ForcingConfig  # noqa: E402
from nos_utils.forcing.dynamic_adjust import (  # noqa: E402
    DynamicAdjustProcessor,
    apply_ssh_time_varying_adjust,
    compute_bias,
    load_observations,
    parse_noaa_xml,
    read_bp_stations,
    read_diff_bp,
    read_model_start,
    DEFAULT_STATIONS,
    DEFAULT_STATION_LONS,
    DEFAULT_STATION_LATS,
    ObsBundle,
)


def _write_fake_elev(path: Path, nt: int = 5, n_bnd: int = 4,
                     init_val: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with nc_mod.Dataset(str(path), "w") as ds:
        ds.createDimension("time", nt)
        ds.createDimension("nOpenBndNodes", n_bnd)
        ds.createDimension("nLevels", 1)
        ds.createDimension("nComponents", 1)
        v = ds.createVariable(
            "time_series", "f4",
            ("time", "nOpenBndNodes", "nLevels", "nComponents"),
        )
        v[:] = init_val


class TestApplyAdjust:
    def test_time_varying_pattern(self, tmp_path):
        nc = tmp_path / "elev2D.th.nc"
        _write_fake_elev(nc, nt=5, n_bnd=3, init_val=1.0)
        adj0 = 0.10
        adj1 = 0.20

        ok = apply_ssh_time_varying_adjust(nc, adj0=adj0, adj1=adj1)
        assert ok
        with nc_mod.Dataset(str(nc)) as ds:
            vals = ds.variables["time_series"][:]
        # t=0 -> 1.0 - 0.10 = 0.90
        assert np.allclose(vals[0], 0.90, atol=1e-5)
        # t=1 -> 1.0 - (0.10+0.20)/2 = 0.85
        assert np.allclose(vals[1], 0.85, atol=1e-5)
        # t>=2 -> 1.0 - 0.20 = 0.80
        assert np.allclose(vals[2:], 0.80, atol=1e-5)

    def test_nan_treated_as_zero(self, tmp_path):
        nc = tmp_path / "elev2D.th.nc"
        _write_fake_elev(nc, nt=3, n_bnd=2, init_val=0.5)
        ok = apply_ssh_time_varying_adjust(nc, adj0=float("nan"), adj1=0.1)
        assert ok
        with nc_mod.Dataset(str(nc)) as ds:
            vals = ds.variables["time_series"][:]
        # t=0: 0.5 - 0 = 0.5
        assert np.allclose(vals[0], 0.5, atol=1e-5)
        # t=1: 0.5 - (0+0.1)/2 = 0.45
        assert np.allclose(vals[1], 0.45, atol=1e-5)
        # t=2: 0.5 - 0.1 = 0.4
        assert np.allclose(vals[2], 0.4, atol=1e-5)


class TestXmlParser:
    def test_parses_observation_rows(self, tmp_path):
        xml = tmp_path / "8670870.xml"
        xml.write_text(
            '<obs t="2026-04-01 00:00" v="0.12"/>\n'
            '<obs t="2026-04-01 00:06" v="0.15"/>\n'
            '<obs t="2026-04-01 00:12" v="0.11"/>\n'
        )
        times, vals = parse_noaa_xml(xml)
        assert len(times) == 3
        assert np.allclose(vals, [0.12, 0.15, 0.11])

    def test_skips_bad_values(self, tmp_path):
        xml = tmp_path / "bad.xml"
        xml.write_text(
            '<obs t="2026-04-01 00:00" v="0.12"/>\n'
            '<obs t="2026-04-01 00:06" v="NaN"/>\n'
            '<obs t="2026-04-01 00:12" v="0.11"/>\n'
        )
        times, vals = parse_noaa_xml(xml)
        assert len(times) == 2


class TestBpParsers:
    def test_read_station_bp(self, tmp_path):
        bp = tmp_path / "station.bp"
        bp.write_text(
            "station.bp\n"
            "2\n"
            "1 -80.9030 32.0347 0.0 ! 8670870\n"
            "2 -79.9236 32.7808 0.0 ! 8665530\n"
        )
        ids, lons, lats = read_bp_stations(bp)
        assert ids == ["8670870", "8665530"]
        assert np.allclose(lons, [-80.903, -79.9236])

    def test_read_diff_bp(self, tmp_path):
        bp = tmp_path / "diff.bp"
        bp.write_text(
            "diff.bp\n"
            "2\n"
            "1 -80.9030 32.0347 0.111 ! 8670870\n"
            "2 -79.9236 32.7808 0.222 ! 8665530\n"
        )
        offsets = read_diff_bp(bp)
        assert offsets["8670870"] == pytest.approx(0.111)
        assert offsets["8665530"] == pytest.approx(0.222)


class TestModelStart:
    def test_parses_start_datetime(self, tmp_path):
        nml = tmp_path / "param.nml"
        nml.write_text("&CORE\n start_year = 2026\n start_month = 4\n "
                       "start_day = 1\n start_hour = 12\n/\n")
        dt = read_model_start(nml)
        assert dt == datetime(2026, 4, 1, 12)

    def test_missing_file_returns_none(self, tmp_path):
        assert read_model_start(tmp_path / "nonexistent.nml") is None


class TestComputeBias:
    def test_returns_nan_when_no_data(self):
        obs = ObsBundle(
            station_ids=np.array([], dtype=object),
            times=np.array([], dtype="datetime64[ns]"),
            elev=np.array([], dtype=float),
            station_lons={}, station_lats={},
        )
        model = np.array([[0.0, 0.0], [3600.0, 0.1]], dtype=float)
        avg, per = compute_bias(
            obs, model, datetime(2026, 4, 1),
            ["8670870"], {}, datetime(2026, 4, 1), datetime(2026, 4, 3),
        )
        assert np.isnan(avg)
        assert per == {}

    def test_positive_bias_when_model_above_obs(self):
        # Synthesize obs at -0.1m and model at 0.2m → bias = +0.3m.
        start = datetime(2026, 4, 1, 0, 0)
        end = datetime(2026, 4, 2, 0, 0)
        obs_times = pd.date_range(start, end, freq="6min", tz="UTC")
        obs_times_np = np.array(
            [pd.Timestamp(t).tz_convert(None).to_datetime64()
             for t in obs_times]
        )
        sid = "8670870"
        obs = ObsBundle(
            station_ids=np.array([sid] * len(obs_times), dtype=object),
            times=obs_times_np,
            elev=np.full(len(obs_times), -0.1, dtype=float),
            station_lons={sid: -80.9}, station_lats={sid: 32.0},
        )
        # Model: 1 column per station, hourly.
        model_times = pd.date_range(start, end, freq="h", tz="UTC")
        secs = np.array(
            [(t - pd.Timestamp(start, tz="UTC")).total_seconds()
             for t in model_times]
        )
        model_vals = np.full(len(secs), 0.2)
        model_staout = np.column_stack([secs, model_vals])
        avg, per = compute_bias(
            obs, model_staout, start,
            [sid], {}, start, end,
        )
        assert np.isclose(avg, 0.3, atol=0.05)
        assert sid in per


class TestProcessorIntegration:
    def test_degrades_gracefully_without_inputs(self, tmp_path):
        """Without any inputs, processor should report warnings and still
        touch elev2D.th.nc with a zero-bias (noop) adjustment."""
        elev = tmp_path / "elev2D.th.nc"
        _write_fake_elev(elev, nt=4, n_bnd=3, init_val=2.0)

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = DynamicAdjustProcessor(
            cfg, input_path=tmp_path, output_path=tmp_path,
            elev2d_th_nc=elev,
        )
        result = proc.process()

        # Without obs/prev cycle data, today's bias is NaN and adj0=0.0,
        # so the result is a no-op adjust → success.
        assert result.success, result.errors
        assert any("today's bias = NaN" in w.lower() or
                   "adj0 defaulting" in w.lower() for w in result.warnings)
        with nc_mod.Dataset(str(elev)) as ds:
            vals = ds.variables["time_series"][:]
        assert np.allclose(vals, 2.0, atol=1e-5)  # unchanged

    def test_missing_elev_file_is_error(self, tmp_path):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = DynamicAdjustProcessor(
            cfg, input_path=tmp_path, output_path=tmp_path,
            elev2d_th_nc=tmp_path / "nope.nc",
        )
        result = proc.process()
        assert not result.success
        assert any("elev2d.th.nc" in e.lower() for e in result.errors)

    def test_writes_avg_bias_scalar(self, tmp_path):
        elev = tmp_path / "elev2D.th.nc"
        _write_fake_elev(elev, nt=3, n_bnd=2)

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = DynamicAdjustProcessor(
            cfg, input_path=tmp_path, output_path=tmp_path,
            elev2d_th_nc=elev,
        )
        result = proc.process()

        bias_file = tmp_path / "average_bias_today"
        assert bias_file.exists()
        content = bias_file.read_text().strip()
        # With no obs, bias is NaN.
        assert content.lower() == "nan"
