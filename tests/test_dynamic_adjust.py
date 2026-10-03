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
    densify_hourly,
    compute_bias,
    load_observations,
    parse_noaa_xml,
    read_bp_stations,
    read_diff_bp,
    read_model_start,
    read_station_in,
    _bc_average,
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

    def test_forecast_offset_is_adj1_everywhere(self, tmp_path):
        nc = tmp_path / "elev2D.th.nc"
        _write_fake_elev(nc, nt=5, n_bnd=3, init_val=1.0)
        assert apply_ssh_time_varying_adjust(nc, adj0=0.10, adj1=0.20, start_offset_hours=24)
        with nc_mod.Dataset(str(nc)) as ds:
            vals = ds.variables["time_series"][:]
        assert np.allclose(vals, 0.80, atol=1e-5)

    def test_offset_one_starts_at_average(self, tmp_path):
        nc = tmp_path / "elev2D.th.nc"
        _write_fake_elev(nc, nt=3, n_bnd=2, init_val=1.0)
        assert apply_ssh_time_varying_adjust(nc, adj0=0.10, adj1=0.20, start_offset_hours=1)
        with nc_mod.Dataset(str(nc)) as ds:
            vals = ds.variables["time_series"][:]
        assert np.allclose(vals[0], 0.85, atol=1e-5)
        assert np.allclose(vals[1:], 0.80, atol=1e-5)

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


def _dcom_row(sid, ts, value, flag=1):
    return f"{sid} 1 WL {ts:%Y-%m-%d %H:%M} {value:7.3f}   0.003   {flag} 0 0 0"


class TestDcomTextParser:
    """Ops reads the .xml files as whitespace text: fields 4/5/6 = date/time/value."""

    def _rows(self, n, sid="8670870"):
        t0 = datetime(2026, 9, 24)
        return [_dcom_row(sid, t0 + timedelta(minutes=6 * i), 1.0 + i / 100) for i in range(n)]

    def test_real_format_rows_and_flag0_kept(self, tmp_path):
        rows = self._rows(12)
        rows[3] = _dcom_row("8670870", datetime(2026, 9, 24, 0, 30), 1.153, flag=0)
        f = tmp_path / "8670870.xml"
        f.write_text("\n".join(rows) + "\n")
        times, vals = parse_noaa_xml(f)
        assert len(times) == 12
        assert times[0] == pd.Timestamp("2026-09-24 00:00", tz="UTC")
        assert times[1] == pd.Timestamp("2026-09-24 00:06", tz="UTC")
        assert vals[0] == pytest.approx(1.0) and vals[3] == pytest.approx(1.153)

    def test_negative_value_parsed(self, tmp_path):
        rows = self._rows(10)
        rows[9] = "8670870 1 WL 2026-10-01 23:24  -0.618   0.007   1 0 0 0"
        f = tmp_path / "x.xml"
        f.write_text("\n".join(rows) + "\n")
        times, vals = parse_noaa_xml(f)
        assert vals[-1] == pytest.approx(-0.618)
        assert times[-1] == pd.Timestamp("2026-10-01 23:24", tz="UTC")

    def test_file_under_ten_lines_is_skipped(self, tmp_path):
        f = tmp_path / "short.xml"
        f.write_text("\n".join(self._rows(9)) + "\n")
        assert parse_noaa_xml(f) == ([], [])

    def test_exactly_ten_lines_kept_and_blank_line_counts_but_is_dropped(self, tmp_path):
        f = tmp_path / "ten.xml"
        f.write_text("\n".join(self._rows(10)) + "\n")
        assert len(parse_noaa_xml(f)[0]) == 10
        f.write_text("\n".join(self._rows(9)) + "\n\n")
        assert len(parse_noaa_xml(f)[0]) == 9

    def test_rows_without_field_six_are_dropped(self, tmp_path):
        rows = self._rows(12)
        rows[2] = "8670870 1 WL 2026-09-24 00:12"
        rows[5] = "garbage"
        f = tmp_path / "gaps.xml"
        f.write_text("\n".join(rows) + "\n")
        assert len(parse_noaa_xml(f)[0]) == 10

    def test_xml_attribute_fallback_when_no_text_rows(self, tmp_path):
        f = tmp_path / "attr.xml"
        f.write_text('<obs t="2026-04-01 00:00" v="0.12"/>\n<obs t="2026-04-01 00:06" v="0.15"/>\n')
        times, vals = parse_noaa_xml(f)
        assert len(times) == 2 and vals == [0.12, 0.15]


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


class TestStationIn:
    HEADER = "1 1 1 1 1 1 1 1 0 !on (1)|off(0) flags for elev air pressure windx windy T S u v w\n"

    def test_bp_comment_without_space_and_flags_header(self, tmp_path):
        f = tmp_path / "bias.bp"
        f.write_text(
            self.HEADER + "2\n"
            "49 -80.901700 32.036700 0 !8670870\n"
            "48 -79.923600 32.780800 0 !8665530\n"
        )
        ids, lons, lats = read_bp_stations(f)
        assert ids == ["8670870", "8665530"]
        assert lons == [-80.9017, -79.9236] and lats == [32.0367, 32.7808]

    def test_read_station_in_row_order(self, tmp_path):
        f = tmp_path / "station.in"
        f.write_text(
            self.HEADER + "3\n"
            "1 -81.871 26.648 0 !8725520\n"
            "2 -81.808 24.550 0 !8724580\n"
            "3 -81.106 24.711 0 !8723970\n"
        )
        assert read_station_in(f) == ["8725520", "8724580", "8723970"]

    def test_read_station_in_keeps_row_index_for_unlabelled_row(self, tmp_path):
        f = tmp_path / "station.in"
        f.write_text(self.HEADER + "3\n1 0 0 0 !a\n2 0 0 0\n3 0 0 0 !c\n")
        assert read_station_in(f) == ["a", "", "c"]


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
            model_station_ids=["8670870"],
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
            model_station_ids=[sid],
        )
        assert np.isclose(avg, 0.3, atol=0.05)
        assert sid in per


def _flat_bundle(levels, start, end):
    """Constant observation level per station at 6-min spacing."""
    times = pd.date_range(start, end, freq="6min")
    ids, tt, el = [], [], []
    for sid, lev in levels.items():
        ids += [sid] * len(times)
        tt += list(times.to_numpy())
        el += [lev] * len(times)
    return ObsBundle(
        station_ids=np.array(ids, dtype=object), times=np.array(tt),
        elev=np.array(el, dtype=float),
        station_lons={sid: 0.0 for sid in levels}, station_lats={sid: 0.0 for sid in levels},
    )


def _hourly_staout(columns, start, end):
    n = int((end - start).total_seconds() // 3600) + 1
    secs = np.arange(n) * 3600.0
    return np.column_stack([secs] + [np.full(n, c) for c in columns])


class TestBiasColumnMapping:
    START = datetime(2026, 9, 29, 12)
    END = datetime(2026, 10, 1, 12)

    def test_columns_follow_station_in_rows_not_bias_list_position(self):
        # bias list order A, B; station.in rows: x, B (row 2), y, A (row 4)
        obs = _flat_bundle({"A": 0.1, "B": 0.2}, self.START - timedelta(hours=1), self.END + timedelta(hours=1))
        staout = _hourly_staout([9.0, 0.5, 9.0, 0.9], self.START, self.END + timedelta(hours=1))
        avg, per = compute_bias(
            obs, staout, self.START, ["A", "B"], {}, self.START, self.END,
            model_station_ids=["x", "B", "y", "A"],
        )
        assert per["A"] == pytest.approx(0.8) and per["B"] == pytest.approx(0.3)
        assert avg == pytest.approx(0.55)

    def test_first_match_wins_for_duplicate_ids(self):
        obs = _flat_bundle({"A": 0.0}, self.START - timedelta(hours=1), self.END + timedelta(hours=1))
        staout = _hourly_staout([0.2, 0.7], self.START, self.END + timedelta(hours=1))
        avg, _ = compute_bias(
            obs, staout, self.START, ["A"], {}, self.START, self.END,
            model_station_ids=["A", "A"],
        )
        assert avg == pytest.approx(0.2)

    def test_station_absent_from_station_in_is_skipped(self):
        obs = _flat_bundle({"A": 0.1, "B": 0.2}, self.START - timedelta(hours=1), self.END + timedelta(hours=1))
        staout = _hourly_staout([0.5, 9.0], self.START, self.END + timedelta(hours=1))
        avg, per = compute_bias(
            obs, staout, self.START, ["A", "B"], {}, self.START, self.END,
            model_station_ids=["A", "other"],
        )
        assert list(per) == ["A"] and avg == pytest.approx(0.4)

    def test_no_station_in_means_nan_not_positional_columns(self):
        obs = _flat_bundle({"A": 0.1}, self.START - timedelta(hours=1), self.END + timedelta(hours=1))
        staout = _hourly_staout([0.5], self.START, self.END + timedelta(hours=1))
        for ids in (None, [], ["unrelated"]):
            avg, per = compute_bias(
                obs, staout, self.START, ["A"], {}, self.START, self.END,
                model_station_ids=ids,
            )
            assert np.isnan(avg) and per == {}

    def test_column_beyond_staout_width_is_skipped(self):
        obs = _flat_bundle({"A": 0.1}, self.START - timedelta(hours=1), self.END + timedelta(hours=1))
        staout = _hourly_staout([0.5], self.START, self.END + timedelta(hours=1))
        avg, _ = compute_bias(
            obs, staout, self.START, ["A"], {}, self.START, self.END,
            model_station_ids=["p", "q", "A"],
        )
        assert np.isnan(avg)

    def test_average_rounded_to_three_decimals_like_ops(self):
        obs = _flat_bundle({"A": 0.0}, self.START - timedelta(hours=1), self.END + timedelta(hours=1))
        staout = _hourly_staout([0.03849], self.START, self.END + timedelta(hours=1))
        avg, _ = compute_bias(
            obs, staout, self.START, ["A"], {}, self.START, self.END,
            model_station_ids=["A"],
        )
        assert avg == 0.038


class TestOpsApplyArithmetic:
    def test_bc_average_is_exact_decimal_truncated_at_five_places(self):
        assert _bc_average(-0.037, -0.039) == -0.038
        assert _bc_average(-0.037, -0.038) == -0.0375
        assert _bc_average(0.00001, 0.0) == 0.0
        assert _bc_average(0.0, 0.0) == 0.0

    def test_float32_subtraction_matches_ncap2(self, tmp_path):
        nc = tmp_path / "elev2D.th.nc"
        _write_fake_elev(nc, nt=4, n_bnd=2, init_val=-0.618)
        assert apply_ssh_time_varying_adjust(nc, adj0=-0.037, adj1=-0.039)
        base = np.float32(-0.618)
        with nc_mod.Dataset(str(nc)) as ds:
            got = ds.variables["time_series"][:, 0, 0, 0].data
        assert got.dtype == np.float32
        np.testing.assert_array_equal(
            got,
            np.array([base - np.float32(-0.037), base - np.float32(-0.038),
                      base - np.float32(-0.039), base - np.float32(-0.039)], dtype=np.float32),
        )


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


class TestProcessorOpsObsEndToEnd:
    """PDY 20261001 12z: dcom text rows, model station.in with the bias stations off the first rows."""

    START = datetime(2026, 9, 29, 12)
    END = datetime(2026, 10, 1, 12)
    # staout_1 columns follow these station.in rows; filler rows hold a decoy 9.0.
    ROWS = ["f1", "8665530", "f2", "f3", "8670870", "f4"]

    def _scene(self, tmp_path, *, short_for=None, station_in=True, adj0="-0.037\n"):
        obs = tmp_path / "dcom" / "coops_waterlvlobs"
        obs.mkdir(parents=True)
        times = pd.date_range(self.START - timedelta(hours=1), self.END + timedelta(hours=1), freq="6min")
        for sid, lev in (("8670870", 0.2), ("8665530", 0.1)):
            rows = [_dcom_row(sid, t.to_pydatetime(), lev, flag=0 if i % 7 == 0 else 1)
                    for i, t in enumerate(times)]
            if sid == short_for:
                rows = rows[:5]
            (obs / f"{sid}.xml").write_text("\n".join(rows) + "\n")
        # model: 8670870 = 0.5004 (bias 0.3004), 8665530 = 0.3002 (bias 0.2002)
        cols = {"f1": 9.0, "8665530": 0.3002, "f2": 9.0, "f3": 9.0, "8670870": 0.5004, "f4": 9.0}
        staout = tmp_path / "staout_1"
        np.savetxt(staout, _hourly_staout([cols[r] for r in self.ROWS], self.START, self.END + timedelta(hours=24)))
        nml = tmp_path / "param.nml"
        nml.write_text("&CORE\n start_year = 2026\n start_month = 9\n start_day = 29\n start_hour = 12\n/\n")
        sin = tmp_path / "station.in"
        if station_in:
            sin.write_text(
                "1 1 1 1 1 1 1 1 0 !flags\n%d\n" % len(self.ROWS)
                + "".join(f"{i + 1} 0 0 0 !{r}\n" for i, r in enumerate(self.ROWS))
            )
        prev = tmp_path / "avg_bias"
        prev.write_text(adj0)
        out = tmp_path / "out"
        out.mkdir()
        elev = out / "elev2D.th.nc"
        _write_fake_elev(elev, nt=5, n_bnd=2, init_val=1.0)
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20261001", cyc=12)
        proc = DynamicAdjustProcessor(
            cfg, input_path=out, output_path=out, obs_dir=obs,
            prev_staout_1=staout, prev_param_nml=nml,
            station_in=sin if station_in else None,
            prev_avg_bias_file=prev, elev2d_th_nc=elev,
            stations=["8670870", "8665530"], station_lons=[0.0, 0.0], station_lats=[0.0, 0.0],
        )
        return proc, elev, out

    def _series(self, elev):
        with nc_mod.Dataset(str(elev)) as ds:
            return ds.variables["time_series"][:, 0, 0, 0].data

    def test_bias_from_station_in_columns_rounded_and_applied(self, tmp_path):
        proc, elev, out = self._scene(tmp_path)
        res = proc.process()
        assert res.success, res.errors
        assert (out / "average_bias_today").read_text() == "0.250\n"
        assert res.metadata["adj1"] == 0.25
        one = np.float32(1.0)
        np.testing.assert_array_equal(
            self._series(elev),
            np.array([one - np.float32(-0.037), one - np.float32(0.1065),
                      one - np.float32(0.25), one - np.float32(0.25), one - np.float32(0.25)],
                     dtype=np.float32),
        )

    def test_short_obs_file_station_is_skipped(self, tmp_path):
        proc, _, out = self._scene(tmp_path, short_for="8665530")
        assert proc.process().success
        assert (out / "average_bias_today").read_text() == "0.300\n"

    def test_missing_station_in_gives_nan_bias_and_zero_adj1(self, tmp_path):
        proc, elev, out = self._scene(tmp_path, station_in=False)
        res = proc.process()
        assert res.success, res.errors
        assert any("station.in" in w for w in res.warnings)
        assert (out / "average_bias_today").read_text().strip() == "nan"
        np.testing.assert_allclose(self._series(elev), [1.037, 1.0185, 1.0, 1.0, 1.0], atol=1e-6)


class TestObsDirResolution:
    """obs_dir auto-resolves from COMINwl/DCOMROOT, mirroring the
    operational dynamic-adjust script: $DCOMROOT/<PDY>/coops_waterlvlobs,
    cycle day preferred, previous day as fallback. Regression guard for
    the STOFS-3D-ATL "obs_dir not provided -> zero bias" gap.
    """

    def _mkobs(self, root: Path, date_str: str) -> Path:
        d = root / date_str / "coops_waterlvlobs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_cominwl_today(self, tmp_path, monkeypatch):
        wl = tmp_path / "wl"
        expected = self._mkobs(wl, "20260401")
        monkeypatch.setenv("COMINwl", str(wl))
        monkeypatch.delenv("DCOMROOT", raising=False)
        assert DynamicAdjustProcessor._resolve_obs_dir("20260401") == expected

    def test_dcomroot_previous_day_fallback(self, tmp_path, monkeypatch):
        dcom = tmp_path / "dcom"
        # Only the previous day's dir exists.
        expected = self._mkobs(dcom, "20260331")
        monkeypatch.delenv("COMINwl", raising=False)
        monkeypatch.setenv("DCOMROOT", str(dcom))
        assert DynamicAdjustProcessor._resolve_obs_dir("20260401") == expected

    def test_cominwl_takes_precedence(self, tmp_path, monkeypatch):
        wl = tmp_path / "wl"
        dcom = tmp_path / "dcom"
        expected = self._mkobs(wl, "20260401")
        self._mkobs(dcom, "20260401")  # wrong one, must be ignored
        monkeypatch.setenv("COMINwl", str(wl))
        monkeypatch.setenv("DCOMROOT", str(dcom))
        assert DynamicAdjustProcessor._resolve_obs_dir("20260401") == expected

    def test_missing_everywhere_returns_none(self, monkeypatch):
        monkeypatch.delenv("COMINwl", raising=False)
        monkeypatch.delenv("DCOMROOT", raising=False)
        assert DynamicAdjustProcessor._resolve_obs_dir("20260401") is None

    def test_init_auto_resolves_when_obs_dir_omitted(self, tmp_path, monkeypatch):
        wl = tmp_path / "wl"
        expected = self._mkobs(wl, "20260401")
        monkeypatch.setenv("COMINwl", str(wl))
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = DynamicAdjustProcessor(
            cfg, input_path=tmp_path, output_path=tmp_path,
        )
        assert proc.obs_dir == expected

    def test_init_explicit_obs_dir_wins(self, tmp_path, monkeypatch):
        wl = tmp_path / "wl"
        self._mkobs(wl, "20260401")
        monkeypatch.setenv("COMINwl", str(wl))
        explicit = tmp_path / "explicit_obs"
        explicit.mkdir()
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = DynamicAdjustProcessor(
            cfg, input_path=tmp_path, output_path=tmp_path,
            obs_dir=explicit,
        )
        assert proc.obs_dir == explicit


class TestDensifyHourly:
    def _six_hourly(self, path, vals):
        with nc_mod.Dataset(str(path), "w", format="NETCDF4_CLASSIC") as ds:
            ds.createDimension("time", len(vals))
            ds.createDimension("nOpenBndNodes", 2)
            ds.createDimension("nLevels", 1)
            ds.createDimension("one", 1)
            ds.createVariable("time", "f4", ("time",))[:] = np.arange(len(vals)) * 21600.0
            ds.createVariable("time_step", "f4", ("one",))[0] = 21600.0
            ds.createVariable("time_series", "f4", ("time", "nOpenBndNodes", "nLevels", "one"))[:] = \
                np.asarray(vals, np.float32)[:, None, None, None]

    def test_six_hourly_file_is_resampled_then_ramped_as_ops(self, tmp_path):
        nc = tmp_path / "elev2D.th.nc"
        self._six_hourly(nc, [1.0, 1.0, 1.0])
        assert apply_ssh_time_varying_adjust(nc, adj0=0.10, adj1=0.20)
        with nc_mod.Dataset(str(nc)) as ds:
            t, ts, dt = ds["time"][:], ds["time_series"][:, 0, 0, 0], ds["time_step"][0]
        assert len(t) == 13 and dt == 3600.0 and t[1] - t[0] == 3600.0
        np.testing.assert_allclose(ts[:3], [0.90, 0.85, 0.80], atol=1e-5)
        np.testing.assert_allclose(ts[3:], 0.80, atol=1e-5)

    def test_linear_in_time_and_hourly_files_untouched(self, tmp_path):
        nc = tmp_path / "e.nc"
        self._six_hourly(nc, [0.0, 6.0])
        assert densify_hourly(nc)
        with nc_mod.Dataset(str(nc)) as ds:
            np.testing.assert_allclose(ds["time_series"][:, 0, 0, 0], np.arange(7), atol=1e-6)
        assert not densify_hourly(nc)
