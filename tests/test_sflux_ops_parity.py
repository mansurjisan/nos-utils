"""STOFS-3D-ATL standalone sflux parity with ops v3.1.

Covers the GFS file chain of stofs_3d_atl_create_surface_forcing_gfs.sh, the
true GFS lon/lat of the extracted points, the last-PRATE-record rule, the HRRR
wind-rotation flag and its config gating. No real wgrib2 is needed. MJ (10/02/26)
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.gfs import GFSProcessor
from nos_utils.forcing.hrrr import HRRRProcessor
from nos_utils.io.grib_extract import Wgrib2Extractor

PDY = "20260927"


def _ext() -> Wgrib2Extractor:
    """Extractor with a placeholder binary; the tests replace subprocess.run. MJ (10/02/26)"""
    ext = Wgrib2Extractor.__new__(Wgrib2Extractor)
    ext.wgrib2 = "wgrib2"
    return ext


def _touch(root: Path, date: str, cyc: int, fhrs, size: int = 1024) -> None:
    d = root / f"gfs.{date}" / f"{cyc:02d}" / "atmos"
    d.mkdir(parents=True, exist_ok=True)
    for fhr in fhrs:
        (d / f"gfs.t{cyc:02d}z.pgrb2.0p25.f{fhr:03d}").write_bytes(b"\x00" * size)


def _proc(root: Path, tmp_path: Path, phase="nowcast", **cfg_kw) -> GFSProcessor:
    cfg = ForcingConfig.for_stofs_3d_atl(PDY, 12, **cfg_kw)
    proc = GFSProcessor(cfg, root, tmp_path / "out", phase=phase)
    proc.MIN_FILE_SIZE = 0
    return proc


def _key(proc: GFSProcessor, f: Path):
    """(cycle datetime, lead) of a selected file. MJ (10/02/26)"""
    date = datetime.strptime(f.parents[2].name.split(".")[1], "%Y%m%d")
    return date + timedelta(hours=int(f.name.split(".t")[1][:2])), proc._parse_fhr(f)


def _full_tree(root: Path) -> None:
    """Every ops-chain file, plus long leads and f000 analyses the chain must ignore. MJ (10/02/26)"""
    _touch(root, "20260926", 6, range(0, 31))
    _touch(root, "20260926", 12, range(0, 61))
    _touch(root, "20260926", 18, range(0, 31))
    _touch(root, "20260927", 0, range(0, 31))
    _touch(root, "20260927", 6, range(0, 31))
    _touch(root, "20260927", 12, range(0, 100))


class TestOpsChainSelection:
    @pytest.mark.parametrize("valid, cycle, lead", [
        (datetime(2026, 9, 26, 12), datetime(2026, 9, 26, 6), 6),
        (datetime(2026, 9, 26, 13), datetime(2026, 9, 26, 12), 1),
        (datetime(2026, 9, 26, 18), datetime(2026, 9, 26, 12), 6),
        (datetime(2026, 9, 26, 19), datetime(2026, 9, 26, 18), 1),
        (datetime(2026, 9, 27, 0), datetime(2026, 9, 26, 18), 6),
        (datetime(2026, 9, 27, 1), datetime(2026, 9, 27, 0), 1),
        (datetime(2026, 9, 27, 12), datetime(2026, 9, 27, 6), 6),
        (datetime(2026, 9, 27, 13), datetime(2026, 9, 27, 12), 1),
        (datetime(2026, 10, 1, 15), datetime(2026, 9, 27, 12), 99),
    ])
    def test_cycle_and_lead_per_valid_hour(self, valid, cycle, lead):
        got = GFSProcessor._ops_chain_cycle(valid, datetime(2026, 9, 27, 12))
        assert got == (cycle, lead)

    def test_list_is_the_ops_chain_with_the_pre_window_buffer(self, tmp_path):
        root = tmp_path / "gfs"
        _full_tree(root)
        proc = _proc(root, tmp_path)
        got = [_key(proc, f) for f in proc.find_input_files()]
        expected = (
            [(datetime(2026, 9, 26, 6), h) for h in range(3, 7)]  # 3 buffer hours + ops f006 MJ (10/02/26)
            + [(datetime(2026, 9, 26, 12), h) for h in range(1, 7)]
            + [(datetime(2026, 9, 26, 18), h) for h in range(1, 7)]
            + [(datetime(2026, 9, 27, 0), h) for h in range(1, 7)]
            + [(datetime(2026, 9, 27, 6), h) for h in range(1, 7)]
            + [(datetime(2026, 9, 27, 12), h) for h in range(1, 100)]
        )
        assert got == expected

    @pytest.mark.parametrize("phase, n, last", [
        ("nowcast", 31, datetime(2026, 9, 27, 15)),
        ("forecast", 103, datetime(2026, 10, 1, 15)),
    ])
    def test_phase_windows_hold_unique_contiguous_hours(self, tmp_path, phase, n, last):
        root = tmp_path / "gfs"
        _full_tree(root)
        proc = _proc(root, tmp_path, phase=phase)
        files = proc._select_files_for_window(proc.find_input_files())
        times = [proc._parse_valid_time(f) for f in files]
        assert len(files) == n and times[-1] == last
        assert all(b - a == timedelta(hours=1) for a, b in zip(times, times[1:]))
        assert all(proc._parse_fhr(f) > 0 for f in files)

    def test_undersized_file_is_not_used_and_its_hour_is_filled(self, tmp_path, caplog):
        root = tmp_path / "gfs"
        _full_tree(root)
        small = root / "gfs.20260927" / "12" / "atmos" / "gfs.t12z.pgrb2.0p25.f010"
        small.write_bytes(b"\x00" * 10)
        proc = _proc(root, tmp_path, phase="forecast")
        proc.MIN_FILE_SIZE = 500
        files = proc.find_input_files()
        assert small not in files and len(files) == 127  # 22z comes from the 06z cycle MJ (10/02/26)
        assert proc._ops_state["filled"] == [datetime(2026, 9, 27, 22)]

    def test_ops_size_floor_applies_to_0p25_only_under_the_flag(self, tmp_path):
        on = GFSProcessor(ForcingConfig.for_stofs_3d_atl(PDY, 12), tmp_path, tmp_path)
        off = GFSProcessor(ForcingConfig.for_stofs_3d_atl(PDY, 12, gfs_ops_timeline=False),
                           tmp_path, tmp_path)
        assert on.MIN_FILE_SIZE == 500_000_000
        assert off.MIN_FILE_SIZE == 400_000_000

    def test_default_search_is_untouched_without_the_flag(self, tmp_path):
        root = tmp_path / "gfs"
        _full_tree(root)
        proc = _proc(root, tmp_path, gfs_ops_timeline=False)
        assert proc.find_input_files() == proc._build_file_list()


def _all_leads_tree(root: Path, first=datetime(2026, 9, 25, 0), last=datetime(2026, 9, 27, 12),
                    max_lead=120) -> None:
    c = first
    while c <= last:
        _touch(root, c.strftime("%Y%m%d"), c.hour, range(0, max_lead + 1))
        c += timedelta(hours=6)


def _remove(root: Path, cycle: datetime, leads) -> None:
    for fhr in leads:
        (root / f"gfs.{cycle:%Y%m%d}" / f"{cycle:%H}" / "atmos"
         / f"gfs.t{cycle:%H}z.pgrb2.0p25.f{fhr:03d}").unlink(missing_ok=True)


class _SourceExtractor:
    """Grid 2x3; every field holds the valid time of its source file, in hours since 2026-01-01. MJ (10/02/26)"""

    def __init__(self, proc_ref):
        self.proc_ref = proc_ref

    def get_grid(self, f, domain):
        return np.arange(3.0), np.arange(2.0)

    def extract_many(self, f, var_levels, domain, **kw):
        h = (self.proc_ref[0]._parse_valid_time(f) - datetime(2026, 1, 1)).total_seconds() / 3600
        return {vl: np.full((2, 3), h, dtype=np.float32) for vl in var_levels}


def _process(root: Path, tmp_path: Path, phase: str):
    from netCDF4 import Dataset
    proc = _proc(root, tmp_path, phase=phase)
    ref = [proc]
    proc._extractor = _SourceExtractor(ref)
    res = proc.process()
    with Dataset(str(tmp_path / "out" / "sflux_air_1.0001.nc")) as ds:
        base = datetime(2026, 9, 26)
        times = [base + timedelta(days=float(t)) for t in ds["time"][:]]
        values = np.asarray(ds["uwind"][:, 0, 0])
    return res, times, values


class TestDegradedInput:
    CYCLE = datetime(2026, 9, 27, 12)

    @staticmethod
    def _hourly(times):
        return all(b - a == timedelta(hours=1) for a, b in zip(times, times[1:]))

    def test_complete_chain_reports_complete_and_warns_nothing(self, tmp_path, caplog):
        root = tmp_path / "gfs"
        _all_leads_tree(root)
        res, times, _ = _process(root, tmp_path, "forecast")
        assert times[-1] == self.CYCLE + timedelta(hours=99) and len(times) == 103
        md = res.metadata
        assert md["gfs_ops_chain_complete"] and md["gfs_ops_qc_pass"]
        assert md["gfs_filled_hours"] == [] and md["gfs_held_hours"] == []
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_gap_inside_the_chain_is_filled_from_the_next_older_cycle(self, tmp_path, caplog):
        root = tmp_path / "gfs"
        _all_leads_tree(root)
        _remove(root, datetime(2026, 9, 26, 18), range(1, 7))  # valid 19z-00z gone MJ (10/02/26)
        res, times, values = _process(root, tmp_path, "nowcast")
        assert self._hourly(times) and times[0] == datetime(2026, 9, 26, 9)
        assert times[-1] == self.CYCLE + timedelta(hours=3)
        md = res.metadata
        assert not md["gfs_ops_chain_complete"] and md["gfs_ops_qc_pass"]  # 118 of 124 records MJ (10/02/26)
        assert md["gfs_filled_hours"] == [f"2026092619", "2026092620", "2026092621",
                                          "2026092622", "2026092623", "2026092700"]
        assert any("filled from older cycles" in r.message and "0926 19z" in r.message
                   for r in caplog.records if r.levelname == "WARNING")
        assert any("GFS ops chain incomplete" in w for w in res.warnings)
        i = times.index(datetime(2026, 9, 26, 19))
        assert values[i] == (datetime(2026, 9, 26, 19) - datetime(2026, 1, 1)).total_seconds() / 3600

    @pytest.mark.parametrize("last_lead, n_records", [(70, 95), (50, 75)])
    def test_truncated_12z_runs_to_the_phase_end_from_the_06z_cycle(
            self, tmp_path, caplog, last_lead, n_records):
        root = tmp_path / "gfs"
        _all_leads_tree(root)
        _remove(root, self.CYCLE, range(last_lead + 1, 100))
        proc = _proc(root, tmp_path, phase="forecast")
        files = proc.find_input_files()
        assert len([f for f in files if proc._parse_valid_time(f) >= datetime(2026, 9, 26, 12)
                    and _key(proc, f)[0] == self.CYCLE]) == last_lead
        res, times, _ = _process(root, tmp_path / "run", "forecast")
        assert self._hourly(times) and times[-1] == self.CYCLE + timedelta(hours=99)
        md = res.metadata
        assert not md["gfs_ops_qc_pass"] and not md["gfs_ops_chain_complete"]
        assert len(md["gfs_filled_hours"]) == 99 - last_lead and md["gfs_held_hours"] == []
        filled_from = {_key(proc, f)[0] for f in proc.find_input_files()
                       if proc._parse_valid_time(f) > self.CYCLE + timedelta(hours=last_lead)}
        assert filled_from == {datetime(2026, 9, 27, 6)}
        assert any("would ship the previous day's file" in r.message for r in caplog.records)

    def test_list_under_the_threshold_with_two_cycles_missing_is_still_full(self, tmp_path, caplog):
        root = tmp_path / "gfs"
        _all_leads_tree(root)
        for cyc in (datetime(2026, 9, 27, 6), self.CYCLE):
            _remove(root, cyc, range(0, 121))
        res, times, _ = _process(root, tmp_path, "forecast")
        assert self._hourly(times) and times[0] == datetime(2026, 9, 27, 9)
        assert times[-1] == self.CYCLE + timedelta(hours=99)
        assert not res.metadata["gfs_ops_qc_pass"] and res.metadata["gfs_held_hours"] == []
        assert len(res.metadata["gfs_filled_hours"]) == 103

    def test_data_that_ends_early_is_held_to_the_phase_end(self, tmp_path, caplog):
        root = tmp_path / "gfs"
        _all_leads_tree(root, max_lead=60)
        res, times, values = _process(root, tmp_path, "forecast")
        end = self.CYCLE + timedelta(hours=99)
        assert self._hourly(times) and times[-1] == end
        held = res.metadata["gfs_held_hours"]
        assert held[0] == "2026093001" and held[-1] == "2026100115" and len(held) == 39
        last_real = times.index(datetime(2026, 9, 30, 0))
        assert (values[last_real + 1:] == values[last_real]).all()
        assert any("holding the last record for 0930 01z..1001 15z (39 h)" in r.message
                   for r in caplog.records)
        assert any("GFS data ends early" in w for w in res.warnings)
        assert not res.metadata["gfs_ops_chain_complete"]

    def test_empty_chain_is_filled_from_yesterday(self, tmp_path):
        root = tmp_path / "gfs"
        _all_leads_tree(root, last=datetime(2026, 9, 26, 12))
        res, times, _ = _process(root, tmp_path, "forecast")
        assert self._hourly(times) and times[-1] == self.CYCLE + timedelta(hours=99)
        # hourly leads stop at f120 = 10-01 12z; the last 3 hours are held MJ (10/02/26)
        assert res.metadata["gfs_held_hours"] == ["2026100113", "2026100114", "2026100115"]
        assert res.metadata["gfs_unfilled_hours"] == res.metadata["gfs_held_hours"]

    def test_nowcast_window_is_complete_when_only_the_forecast_is_short(self, tmp_path):
        root = tmp_path / "gfs"
        _all_leads_tree(root)
        _remove(root, self.CYCLE, range(4, 100))
        res, times, _ = _process(root, tmp_path, "nowcast")
        md = res.metadata
        assert md["gfs_ops_chain_complete"] and not md["gfs_ops_qc_pass"]
        assert md["gfs_filled_hours"] == [] and times[-1] == self.CYCLE + timedelta(hours=3)


class TestLastPrateRecord:
    def _extract(self, tmp_path, **cfg_kw):
        cfg = ForcingConfig.for_stofs_3d_atl(PDY, 12, **cfg_kw)
        ext = MagicMock()
        ext.get_grid.return_value = (np.arange(3.0), np.arange(2.0))
        ext.extract_many.return_value = {}
        proc = GFSProcessor(cfg, tmp_path, tmp_path / "out", extractor=ext)
        f = tmp_path / "gfs.20260927" / "12" / "atmos" / "gfs.t12z.pgrb2.0p25.f003"
        proc._extract_all([f])
        return ext.extract_many.call_args

    def test_flag_requests_the_last_record(self, tmp_path):
        assert self._extract(tmp_path).kwargs == {"last_record": True}

    def test_default_call_is_unchanged(self, tmp_path):
        assert self._extract(tmp_path, gfs_ops_timeline=False).kwargs == {}

    def test_last_record_numbers_and_direct_decode(self, tmp_path, monkeypatch):
        inv = ("8:344478:d=2026092712:PRATE:surface:3 hour fcst:\n"
               "9:403256:d=2026092712:PRATE:surface:0-3 hour ave fcst:\n"
               "10:7237346:d=2026092712:DSWRF:surface:0-3 hour ave fcst:\n")
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if "-s" in cmd:
                return subprocess.CompletedProcess(cmd, 0, inv, "")
            if "-bin" in cmd:
                np.arange(6, dtype=np.float32).tofile(cmd[-1])
            if "-nxny" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "1:0:(3 x 2)\n", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        ext = _ext()
        nums = ext._last_record_numbers(tmp_path / "c.grb2")
        assert nums[("PRATE", "surface")] == 9 and nums[("DSWRF", "surface")] == 10
        out = ext._extract_record_from(tmp_path / "c.grb2", "PRATE", "surface", tmp_path, 9)
        assert out.shape == (2, 3)
        decode = [c for c in calls if "-bin" in c][0]
        assert decode[decode.index("-d") + 1] == "9"
        assert not any("-match" in c for c in calls)


class TestTrueGridCoordinates:
    def _fake_wgrib2(self, monkeypatch, lon_row, lat_col):
        lon2d, lat2d = np.meshgrid(lon_row, lat_col)

        def fake_run(cmd, **kw):
            if "-small_grib" in cmd:
                Path(cmd[-1]).write_bytes(b"x")
            elif "-nxny" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0, f"1:0:({len(lon_row)} x {len(lat_col)})\n", "")
            elif "rcl_lon" in cmd:
                lon2d.astype("<f4").tofile(cmd[-1])
            elif "rcl_lat" in cmd:
                lat2d.astype("<f4").tofile(cmd[-1])
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)

    def test_points_sit_on_the_gfs_grid_not_on_the_bounds(self, tmp_path, monkeypatch):
        self._fake_wgrib2(monkeypatch, np.arange(261.5, 307.5 + 0.1, 0.25),
                          np.arange(7.5, 52.5 + 0.1, 0.25))
        lons, lats = _ext().get_grid(
            tmp_path / "f.grib2", (-98.5035, -52.4867, 7.347, 52.5904))
        assert lons.shape == (185,) and lats.shape == (181,)
        assert lons[0] == -98.5 and lons[-1] == -52.5
        assert lats[0] == 7.5 and lats[-1] == 52.5
        assert np.all(np.diff(lons) == 0.25)

    def test_zero_360_domain_keeps_its_convention(self, tmp_path, monkeypatch):
        self._fake_wgrib2(monkeypatch, np.arange(93.0, 290.0 + 0.1, 0.5), np.arange(-10.0, 30.1, 0.5))
        lons, _ = _ext().get_grid(tmp_path / "f.grib2", (93.0, 290.0, -10.0, 30.0))
        assert lons[0] == 93.0 and lons[-1] == 290.0

    def test_irregular_axes_fall_back_to_the_bounds(self, tmp_path, monkeypatch):
        lon2d = np.add.outer(np.arange(3.0), np.arange(4.0))

        def fake_run(cmd, **kw):
            if "-small_grib" in cmd:
                Path(cmd[-1]).write_bytes(b"x")
            elif "-nxny" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "1:0:(4 x 3)\n", "")
            elif "rcl_lon" in cmd or "rcl_lat" in cmd:
                lon2d.astype("<f4").tofile(cmd[-1])
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        lons, lats = _ext().get_grid(tmp_path / "f.grib2", (-80.0, -70.0, 25.0, 35.0))
        assert lons[0] == -80.0 and lons[-1] == -70.0 and len(lats) == 3


class TestHrrrWindRotation:
    """Rotation flag and the ops -small_grib subset of the native extraction. MJ (10/02/26)"""
    NX, NY = 5, 4

    def _native(self, tmp_path, monkeypatch, rotate, small_grib=False, cmds=None):
        lon2d, lat2d = np.meshgrid(np.linspace(-80.0, -76.0, self.NX), np.linspace(30.0, 33.0, self.NY))
        u = np.full((self.NY, self.NX), 6.0, dtype=np.float32)
        v = np.full((self.NY, self.NX), 2.0, dtype=np.float32)

        def fake_run(cmd, **kw):
            out = Path(cmd[-1])
            if cmds is not None:
                cmds.append(list(cmd))
            if "-match" in cmd:
                out.write_bytes(b"x")
            elif "rcl_lon" in cmd:
                lon2d.astype("<f4").tofile(out)
            elif "rcl_lat" in cmd:
                lat2d.astype("<f4").tofile(out)
            elif "-bin" in cmd:
                name = Path(cmd[1]).name
                data = u if "_uwind" in name else v if "_vwind" in name else u * 0 + 1
                data.tofile(out)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        cfg = ForcingConfig.for_stofs_3d_atl(PDY, 12)
        cfg.hrrr_rotate_winds = rotate
        cfg.hrrr_small_grib = small_grib
        proc = HRRRProcessor(cfg, tmp_path, tmp_path / "out", variables=["uwind", "vwind", "stmp"])
        proc._extractor = SimpleNamespace(wgrib2="wgrib2", _get_nxny=lambda f: (self.NX, self.NY))
        f = tmp_path / "hrrr.20260927" / "conus" / "hrrr.t12z.wrfsfcf01.grib2"
        res = proc._extract_native([f])
        return res, lon2d, u, v

    def test_off_keeps_grid_relative_winds(self, tmp_path, monkeypatch):
        res, _, u, v = self._native(tmp_path, monkeypatch, rotate=False)
        assert np.array_equal(res["data"]["uwind"][0], u)
        assert np.array_equal(res["data"]["vwind"][0], v)

    def test_on_rotates_to_earth_relative(self, tmp_path, monkeypatch):
        res, lon2d, u, v = self._native(tmp_path, monkeypatch, rotate=True)
        u_e, v_e = HRRRProcessor._rotate_winds_lcc(
            u, v, lon2d.astype(np.float32).astype(np.float64))
        assert np.array_equal(res["data"]["uwind"][0], u_e)
        assert np.array_equal(res["data"]["vwind"][0], v_e)
        assert not np.array_equal(res["data"]["uwind"][0], u)


    def test_small_grib_subsets_over_the_hrrr_domain_only_when_asked(self, tmp_path, monkeypatch):
        cmds = []
        self._native(tmp_path, monkeypatch, rotate=False, small_grib=True, cmds=cmds)
        first = [c for c in cmds if "-match" in c][0]
        assert "-grib" not in first
        i = first.index("-small_grib")
        assert first[i + 1:i + 3] == ["-98.5:-49.5", "5.5:50.0"]
        cmds.clear()
        self._native(tmp_path, monkeypatch, rotate=False, small_grib=False, cmds=cmds)
        first = [c for c in cmds if "-match" in c][0]
        assert "-small_grib" not in first and "-grib" in first


def _yaml_cfg(tmp_path, name, mode=None, atm=""):
    yaml = pytest.importorskip("yaml")
    body = {"system": {"name": name},
            "grid": {"domain": {"lon_min": -98.5, "lon_max": -52.5, "lat_min": 7.3, "lat_max": 52.6}},
            "model": {"physics": {"nws": 4}}}
    if mode:
        body["execution"] = {"mode": mode}
    f = tmp_path / f"{name}_{mode}.yaml"
    f.write_text(yaml.dump(body) + atm)
    return ForcingConfig.from_yaml(f, pdy=PDY, cyc=12)


class TestConfigGating:
    def test_defaults_keep_the_legacy_behaviour(self):
        cfg = ForcingConfig(lon_min=-80, lon_max=-70, lat_min=25, lat_max=35, pdy=PDY, cyc=12)
        assert cfg.gfs_ops_timeline is False and cfg.hrrr_rotate_winds is True
        assert cfg.hrrr_small_grib is False

    def test_factories(self):
        for atl in (ForcingConfig.for_stofs_3d_atl(PDY, 12), ForcingConfig.for_stofs_3d_atl_ufs(PDY, 12)):
            assert atl.gfs_ops_timeline is True and atl.hrrr_rotate_winds is False
            assert atl.hrrr_small_grib is True and atl.datm_rotate_hrrr_winds is False
        for other in (ForcingConfig.for_stofs_3d_pac(PDY, 12), ForcingConfig.for_secofs(PDY, 12)):
            assert other.gfs_ops_timeline is False and other.hrrr_rotate_winds is True
            assert other.hrrr_small_grib is False and other.datm_rotate_hrrr_winds is True

    def test_yaml_standalone_atl_follows_ops(self, tmp_path):
        cfg = _yaml_cfg(tmp_path, "stofs_3d_atl_ufs", mode="standalone")
        assert cfg.nws == 2 and cfg.gfs_ops_timeline is True and cfg.hrrr_rotate_winds is False
        assert cfg.hrrr_small_grib is True

    def test_yaml_coupled_atl_follows_ops_no_rotation(self, tmp_path):
        cfg = _yaml_cfg(tmp_path, "stofs_3d_atl_ufs")
        assert cfg.nws == 4 and cfg.gfs_ops_timeline is True and cfg.hrrr_rotate_winds is False
        assert cfg.hrrr_small_grib is True and cfg.datm_rotate_hrrr_winds is False

    @pytest.mark.parametrize("name,mode", [
        ("secofs_ufs", None), ("secofs_ufs", "standalone"), ("secofs_ufs_ww3", None),
        ("stofs_3d_ak_ufs", None), ("stofs_3d_pac_ufs", None)])
    def test_yaml_other_systems_keep_the_defaults(self, tmp_path, name, mode):
        cfg = _yaml_cfg(tmp_path, name, mode=mode)
        assert cfg.gfs_ops_timeline is False and cfg.hrrr_rotate_winds is True
        assert cfg.hrrr_small_grib is False and cfg.datm_rotate_hrrr_winds is True

    def test_yaml_blend_rotate_key_overrides(self, tmp_path):
        atm = "forcing:\n  atmospheric:\n    hrrr:\n      blend_rotate_winds: true\n"
        assert _yaml_cfg(tmp_path, "stofs_3d_atl_ufs", atm=atm).datm_rotate_hrrr_winds is True
        atm = "forcing:\n  atmospheric:\n    hrrr:\n      blend_rotate_winds: false\n"
        assert _yaml_cfg(tmp_path, "secofs_ufs", atm=atm).datm_rotate_hrrr_winds is False

    def test_yaml_keys_override_the_name_rule(self, tmp_path):
        atm = ("forcing:\n  atmospheric:\n    gfs:\n      ops_timeline: false\n"
               "    hrrr:\n      rotate_winds: true\n      small_grib: false\n")
        cfg = _yaml_cfg(tmp_path, "stofs_3d_atl_ufs", mode="standalone", atm=atm)
        assert cfg.gfs_ops_timeline is False and cfg.hrrr_rotate_winds is True
        assert cfg.hrrr_small_grib is False
        atm = ("forcing:\n  atmospheric:\n    gfs:\n      ops_timeline: true\n"
               "    hrrr:\n      rotate_winds: false\n      small_grib: true\n")
        cfg = _yaml_cfg(tmp_path, "secofs_ufs", atm=atm)
        assert cfg.gfs_ops_timeline is True and cfg.hrrr_rotate_winds is False
        assert cfg.hrrr_small_grib is True

    def test_yaml_bad_value_is_rejected(self, tmp_path):
        atm = "forcing:\n  atmospheric:\n    hrrr:\n      rotate_winds: maybe\n"
        with pytest.raises(ValueError):
            _yaml_cfg(tmp_path, "stofs_3d_atl_ufs", mode="standalone", atm=atm)


class TestSilentFallbacks:
    def test_inventory_failure_raises_instead_of_decoding_the_first_prate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 1, "", "boom"))
        with pytest.raises(RuntimeError, match="wgrib2 -s failed"):
            _ext()._last_record_numbers(tmp_path / "c.grb2")

    def test_base_extract_many_warns_when_last_record_is_unsupported(self, caplog):
        from nos_utils.io.grib_extract import CfgribExtractor, GRIBExtractor

        class Plain(GRIBExtractor):
            def extract(self, *a, **k):
                return None

            def get_grid(self, *a, **k):
                return None

        Plain().extract_many(Path("f"), [("PRATE", "surface")], (0, 1, 0, 1), last_record=True)
        assert any("last matching record" in r.message for r in caplog.records)
        assert CfgribExtractor.extract_many is GRIBExtractor.extract_many  # cfgrib inherits it. MJ (10/02/26)

    def test_get_grid_warns_when_the_axes_cannot_be_read(self, tmp_path, monkeypatch, caplog):
        def fake_run(cmd, **kw):
            if "-small_grib" in cmd:
                Path(cmd[-1]).write_bytes(b"x")
            elif "-nxny" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "1:0:(4 x 3)\n", "")
            return subprocess.CompletedProcess(cmd, 1, "", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        lons, _ = _ext().get_grid(tmp_path / "f.grib2", (-80.0, -70.0, 25.0, 35.0))
        assert lons[0] == -80.0
        assert any("evenly spaced" in r.message for r in caplog.records)


def test_factory_flags_follow_the_final_nws():
    atl = ForcingConfig.for_stofs_3d_atl(PDY, 12, nws=4)
    assert atl.gfs_ops_timeline is True and atl.hrrr_rotate_winds is False
    assert atl.hrrr_small_grib is True and atl.datm_rotate_hrrr_winds is False
    kept = ForcingConfig.for_stofs_3d_atl(PDY, 12, nws=4, gfs_ops_timeline=False, hrrr_rotate_winds=True)
    assert kept.gfs_ops_timeline is False and kept.hrrr_rotate_winds is True
    assert ForcingConfig.for_stofs_3d_atl(PDY, 12).gfs_ops_timeline is True


def test_coupled_atl_datm_inputs_use_the_ops_gfs_chain(tmp_path, monkeypatch):
    """The DATM path reuses GFSProcessor, so the ops chain applies to nws=4 too. MJ (10/03/26)"""
    from nos_utils.forcing.gfs import GFSProcessor
    cfg = ForcingConfig.for_stofs_3d_atl_ufs(PDY, 12)
    proc = GFSProcessor(cfg, tmp_path, tmp_path / "o", resolution="0p25")
    assert cfg.nws == 4 and proc.MIN_FILE_SIZE == GFSProcessor.OPS_MIN_FILE_SIZE
    monkeypatch.setattr(GFSProcessor, "_build_ops_file_list", lambda self: ["ops"])
    assert proc.find_input_files() == ["ops"]


def test_hrrr_blend_weight_gating(tmp_path):
    """0.99 for ATL (factories and name rule), 1.0 elsewhere; yaml key overrides. MJ (10/03/26)"""
    assert ForcingConfig.for_stofs_3d_atl(PDY, 12).datm_hrrr_weight == 0.99
    assert ForcingConfig.for_stofs_3d_atl_ufs(PDY, 12).datm_hrrr_weight == 0.99
    for other in (ForcingConfig.for_secofs(PDY, 12), ForcingConfig.for_stofs_3d_pac(PDY, 12)):
        assert other.datm_hrrr_weight == 1.0
    assert _yaml_cfg(tmp_path, "stofs_3d_atl_ufs").datm_hrrr_weight == 0.99
    for name in ("secofs_ufs", "stofs_3d_ak_ufs", "stofs_3d_pac_ufs"):
        assert _yaml_cfg(tmp_path, name).datm_hrrr_weight == 1.0
    atm = "forcing:\n  atmospheric:\n    hrrr:\n      blend_weight: 1.0\n"
    assert _yaml_cfg(tmp_path, "stofs_3d_atl_ufs", atm=atm).datm_hrrr_weight == 1.0
    with pytest.raises(ValueError):
        _yaml_cfg(tmp_path, "stofs_3d_atl_ufs",
                  atm="forcing:\n  atmospheric:\n    hrrr:\n      blend_weight: 0\n")
