"""Tests for STOFS-3D-ATL NWM extensions."""

import json
import pytest
from datetime import datetime
from pathlib import Path

import numpy as np

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nwm import NWMProcessor, RiverConfig, _nwm_valid_time


class TestSourcesJSON:
    """Test RiverConfig.from_sources_json() for STOFS gen_sourcesink format."""

    def test_basic_parse(self, tmp_path):
        sources = {
            "100": [20104159, 9643431],
            "200": [12345678],
            "300": [111, 222, 333],
        }
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))

        rc = RiverConfig.from_sources_json(path)
        assert rc.n_rivers == 3
        assert rc.node_indices == [100, 200, 300]
        assert rc.feature_id_groups == [[20104159, 9643431], [12345678], [111, 222, 333]]
        assert rc.feature_ids == [20104159, 12345678, 111]  # first of each group

    def test_empty_groups(self, tmp_path):
        sources = {"100": [], "200": [12345]}
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))

        rc = RiverConfig.from_sources_json(path)
        assert rc.n_rivers == 2
        assert rc.feature_id_groups[0] == []
        assert rc.feature_id_groups[1] == [12345]

    def test_sorted_by_element_id(self, tmp_path):
        sources = {"300": [3], "100": [1], "200": [2]}
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))

        rc = RiverConfig.from_sources_json(path)
        assert rc.node_indices == [100, 200, 300]


class TestSTOFSMode:
    """Test is_stofs_mode detection."""

    def test_stofs_config_with_sources_json(self, tmp_path):
        sources = {"100": [123], "200": [456]}
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        rc = RiverConfig.from_sources_json(path)
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        assert proc.is_stofs_mode is True

    def test_secofs_mode(self, tmp_path):
        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12)
        rc = RiverConfig([1], [1], [50.0])
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        assert proc.is_stofs_mode is False


class TestSTOFSFileDiscovery:
    """Test STOFS NWM file discovery patterns."""

    def test_empty_directory(self, tmp_path):
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        rc = RiverConfig.from_sources_json(self._write_sources(tmp_path))
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        files = proc.find_input_files()
        assert files == []

    def test_finds_medium_range_files(self, tmp_path):
        """Create a minimal NWM directory structure with medium_range_mem1 files."""
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260402", cyc=12)
        rc = RiverConfig.from_sources_json(self._write_sources(tmp_path))

        # Create yesterday t12z f001 file
        nwm_dir = tmp_path / "nwm.20260401" / "medium_range_mem1"
        nwm_dir.mkdir(parents=True)
        fname = "nwm.t12z.medium_range.channel_rt_1.f001.conus.nc"
        (nwm_dir / fname).write_bytes(b"\x00" * 15_000_000)  # 15MB > 10MB threshold

        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        files = proc.find_input_files()
        assert len(files) == 1
        assert "t12z" in files[0].name

    @staticmethod
    def _write_sources(tmp_path):
        sources = {"100": [123]}
        path = tmp_path / "sources.json"
        path.write_text(json.dumps(sources))
        return path


class TestDispatchByProduct:
    """Dispatch must follow nwm_product, not is_stofs_mode.

    SECOFS-UFS loads STOFS-style sources.json (so is_stofs_mode=True) but
    its nwm_product is "analysis_assim". Routing it through the
    medium_range_mem1 multi-cycle assembler returns 0 files and silently
    falls back to climatology (1 m³/s × 3522 rivers).
    """

    @staticmethod
    def _write_sources(tmp_path):
        path = tmp_path / "sources.json"
        path.write_text(json.dumps({"100": [123]}))
        return path

    def test_secofs_ufs_uses_secofs_finder(self, tmp_path):
        """STOFS-mode + analysis_assim → SECOFS file finder, not STOFS."""
        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        assert cfg.nwm_product == "analysis_assim"
        rc = RiverConfig.from_sources_json(self._write_sources(tmp_path))
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        assert proc.is_stofs_mode is True

        called = {"secofs": 0, "stofs": 0}
        proc._find_secofs_nwm_files = lambda: called.__setitem__(
            "secofs", called["secofs"] + 1) or []
        proc._find_stofs_nwm_files = lambda: called.__setitem__(
            "stofs", called["stofs"] + 1) or []

        proc.find_input_files()
        assert called == {"secofs": 1, "stofs": 0}

    def test_stofs_3d_atl_uses_stofs_finder(self, tmp_path):
        """medium_range_mem1 product → STOFS multi-cycle assembler."""
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260507", cyc=12)
        assert cfg.nwm_product == "medium_range_mem1"
        rc = RiverConfig.from_sources_json(self._write_sources(tmp_path))
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)

        called = {"secofs": 0, "stofs": 0}
        proc._find_secofs_nwm_files = lambda: called.__setitem__(
            "secofs", called["secofs"] + 1) or []
        proc._find_stofs_nwm_files = lambda: called.__setitem__(
            "stofs", called["stofs"] + 1) or []

        proc.find_input_files()
        assert called == {"secofs": 0, "stofs": 1}

    def test_finds_analysis_assim_v3_layout(self, tmp_path):
        """Production NWM v3.0 layout: nwm.YYYYMMDD/<product>/nwm.tHHz.<product>.channel_rt.tm00.conus.nc"""
        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        rc = RiverConfig.from_sources_json(self._write_sources(tmp_path))
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)

        # Lay down a production-shaped tree: per-product subdir + tm00 lead tag
        nwm_dir = tmp_path / "nwm.20260507" / "analysis_assim"
        nwm_dir.mkdir(parents=True)
        (nwm_dir / "nwm.t00z.analysis_assim.channel_rt.tm00.conus.nc").write_bytes(b"\x00")
        (nwm_dir / "nwm.t01z.analysis_assim.channel_rt.tm00.conus.nc").write_bytes(b"\x00")

        files = proc.find_input_files()
        assert len(files) == 2
        assert all("analysis_assim" in f.name for f in files)

    def test_finds_analysis_assim_flat_layout(self, tmp_path):
        """Older caches without the per-product subdir still resolve."""
        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        rc = RiverConfig.from_sources_json(self._write_sources(tmp_path))
        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)

        nwm_dir = tmp_path / "nwm.20260507"
        nwm_dir.mkdir()
        (nwm_dir / "nwm.t00z.analysis_assim.channel_rt.tm00.conus.nc").write_bytes(b"\x00")

        files = proc.find_input_files()
        assert len(files) == 1


class TestValidTimeSort:
    """Files must be ordered by valid time, not by filename.

    Within an analysis_assim cycle the ``tmHH`` lookback runs *backwards*
    in valid time (tm00 = cycle hour, tm01 = 1h earlier, …). Lexicographic
    sort produces a non-monotonic time axis once multiple cycles are
    staged, which SCHISM rejects.
    """

    def test_parse_analysis_assim_lookback(self, tmp_path):
        d = tmp_path / "nwm.20260506" / "analysis_assim"
        d.mkdir(parents=True)
        # t18z lookback: tm00 = 18z, tm01 = 17z, tm02 = 16z
        f00 = d / "nwm.t18z.analysis_assim.channel_rt.tm00.conus.nc"
        f01 = d / "nwm.t18z.analysis_assim.channel_rt.tm01.conus.nc"
        f02 = d / "nwm.t18z.analysis_assim.channel_rt.tm02.conus.nc"
        for f in (f00, f01, f02):
            f.write_bytes(b"")
        assert _nwm_valid_time(f00) == datetime(2026, 5, 6, 18)
        assert _nwm_valid_time(f01) == datetime(2026, 5, 6, 17)
        assert _nwm_valid_time(f02) == datetime(2026, 5, 6, 16)

    def test_parse_forecast_lead(self, tmp_path):
        d = tmp_path / "nwm.20260506" / "short_range"
        d.mkdir(parents=True)
        f01 = d / "nwm.t12z.short_range.channel_rt.f001.conus.nc"
        f18 = d / "nwm.t12z.short_range.channel_rt.f018.conus.nc"
        for f in (f01, f18):
            f.write_bytes(b"")
        assert _nwm_valid_time(f01) == datetime(2026, 5, 6, 13)
        assert _nwm_valid_time(f18) == datetime(2026, 5, 7, 6)

    def test_parse_medium_range_member(self, tmp_path):
        d = tmp_path / "nwm.20260506" / "medium_range_mem1"
        d.mkdir(parents=True)
        f = d / "nwm.t06z.medium_range.channel_rt_1.f120.conus.nc"
        f.write_bytes(b"")
        assert _nwm_valid_time(f) == datetime(2026, 5, 11, 6)

    def test_parse_unparseable(self, tmp_path):
        f = tmp_path / "garbage.nc"
        f.write_bytes(b"")
        # Sentinel — sorts to start, doesn't crash.
        assert _nwm_valid_time(f) == datetime.min

    def test_find_input_files_sorted_by_valid_time(self, tmp_path):
        """Multi-cycle analysis_assim must come out monotonic in valid time."""
        cfg = ForcingConfig.for_secofs(pdy="20260506", cyc=18)
        sources = tmp_path / "sources.json"
        sources.write_text(json.dumps({"100": [123]}))
        rc = RiverConfig.from_sources_json(sources)

        # Stage 4 cycles × 3 lookbacks = 12 files spanning a 14h window.
        d = tmp_path / "nwm.20260506" / "analysis_assim"
        d.mkdir(parents=True)
        for cyc in (0, 6, 12, 18):
            for tm in (0, 1, 2):
                fname = (f"nwm.t{cyc:02d}z.analysis_assim.channel_rt."
                         f"tm{tm:02d}.conus.nc")
                (d / fname).write_bytes(b"")

        proc = NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)
        files = proc.find_input_files()
        assert len(files) == 12

        # Valid times must be strictly non-decreasing.
        times = [_nwm_valid_time(f) for f in files]
        assert times == sorted(times), \
            f"Time grid not monotonic: {[t.strftime('%H') for t in times]}"

        # Spot-check the contract: first file is the earliest valid time
        # (t00z/tm02 = day-2h = 22z prior day in this case is unreachable,
        # earliest valid in this fixture is t00z/tm02 = 2026-05-05 22z).
        assert files[0].name.endswith("t00z.analysis_assim.channel_rt.tm02.conus.nc")
        # Last is t18z/tm00 = 2026-05-06 18z (the cycle hour itself).
        assert files[-1].name.endswith("t18z.analysis_assim.channel_rt.tm00.conus.nc")


class TestNormalizeToSimulationGrid:
    """The output time axis must be monotonic, hourly, and start at 0.

    NWM extract returns ``times`` in *hours from start_time*
    (= cycle - nowcast_hours). Analysis_assim cycles routinely include
    pre-nowcast lookback, and consecutive cycles leave 4h gaps between
    in-cycle 1h steps — both produce a non-monotonic / sparse axis that
    SCHISM rejects.
    """

    @staticmethod
    def _proc(tmp_path):
        cfg = ForcingConfig.for_secofs(pdy="20260506", cyc=18)
        sources = tmp_path / "sources.json"
        sources.write_text(json.dumps({"100": [123]}))
        rc = RiverConfig.from_sources_json(sources)
        return NWMProcessor(cfg, tmp_path, tmp_path, river_config=rc)

    def test_drops_pre_nowcast_lookback(self, tmp_path):
        proc = self._proc(tmp_path)
        # Pre-nowcast hours (-14..-1) must be dropped; cycle hour onward kept.
        flows = np.arange(20).reshape(20, 1).astype(float)  # 20 rivers col, but we'll treat each row as a timestep
        flows = np.repeat(flows, 1, axis=1)  # (20, 1)
        times = list(range(-14, 6))  # -14, -13, ..., +5
        out_flows, out_times = proc._normalize_to_simulation_grid(flows, times, n_target=73)
        assert all(t >= 0 for t in out_times)
        assert out_times[0] == 0.0  # starts at 0 (forward-filled if needed)
        # First 6 hours (0..5) come from raw input rows at indices 14..19
        # (rows 14..19 had times 0..5).
        assert out_flows[0, 0] == 14.0  # row index 14 was time=0
        assert out_flows[5, 0] == 19.0  # row index 19 was time=5

    def test_forward_fills_gaps(self, tmp_path):
        proc = self._proc(tmp_path)
        # Sparse hours (0, 4, 5) must dense-fill to 0,1,2,3,4,5 with the
        # most-recent prior flow.
        flows = np.array([[10.0], [40.0], [50.0]])  # 3 rivers (cols), 3 timesteps (rows)
        times = [0.0, 4.0, 5.0]
        out_flows, out_times = proc._normalize_to_simulation_grid(flows, times, n_target=6)
        assert out_times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        # h=0,1,2,3 all forward-fill from h=0 (value 10.0)
        # h=4 has its own value (40.0), h=5 its own (50.0)
        assert out_flows[:, 0].tolist() == [10.0, 10.0, 10.0, 10.0, 40.0, 50.0]

    def test_dedup_keeps_first(self, tmp_path):
        proc = self._proc(tmp_path)
        # If two files map to the same integer hour (e.g. analysis_assim
        # and short_range overlap), keep the first (analysis sort-wins
        # before forecast products).
        flows = np.array([[100.0], [200.0]])
        times = [3.0, 3.4]  # both round to 3
        out_flows, out_times = proc._normalize_to_simulation_grid(flows, times, n_target=10)
        # Hour 3 entry comes from the FIRST occurrence (100.0)
        assert out_flows[out_times.index(3.0), 0] == 100.0

    def test_monotonic_after_normalize(self, tmp_path):
        proc = self._proc(tmp_path)
        # Construct a deliberately pathological input: out-of-order
        # hours from multiple cycles. Output must be strictly monotonic.
        flows = np.arange(24).reshape(24, 1).astype(float)
        times = [-14, -13, -12, -8, -7, -6, -2, -1, 0, 4, 5, 6,
                 16, 17, 18, 22, 23, 24, 28, 29, 30, 34, 35, 36]
        out_flows, out_times = proc._normalize_to_simulation_grid(flows, times, n_target=73)
        diffs = np.diff(out_times)
        assert (diffs > 0).all(), f"Non-monotonic: {out_times[:10]}"
        assert all(d == 1.0 for d in diffs), f"Not hourly: unique diffs = {set(diffs)}"

    def test_empty_when_all_outside_window(self, tmp_path):
        proc = self._proc(tmp_path)
        # All times before nowcast or after end → empty result, climatology
        # path takes over via _pad_to_target.
        flows = np.array([[1.0], [2.0]])
        times = [-50.0, -40.0]
        out_flows, out_times = proc._normalize_to_simulation_grid(flows, times, n_target=10)
        assert out_times == []
        assert out_flows.shape == (0, 1)


class TestAutoDetectSourcesJSON:
    """Test that NWMProcessor auto-detects sources.json vs regular JSON."""

    def test_auto_detect_stofs_format(self, tmp_path):
        sources = {"100": [123, 456], "200": [789]}
        path = tmp_path / "river.json"
        path.write_text(json.dumps(sources))

        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12,
                                              river_config_file=path)
        proc = NWMProcessor(cfg, tmp_path, tmp_path)
        rc = proc.river_config
        assert rc is not None
        assert rc.feature_id_groups is not None
        assert rc.n_rivers == 2

    def test_auto_detect_regular_format(self, tmp_path):
        data = [{"feature_id": 123, "node_index": 1, "clim_flow": 50.0}]
        path = tmp_path / "river.json"
        path.write_text(json.dumps(data))

        cfg = ForcingConfig.for_secofs(pdy="20260401", cyc=12,
                                       river_config_file=path)
        proc = NWMProcessor(cfg, tmp_path, tmp_path)
        rc = proc.river_config
        assert rc is not None
        assert rc.feature_id_groups is None  # regular format
