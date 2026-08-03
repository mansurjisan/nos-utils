"""Tests for GFSProcessor."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.gfs import GFSProcessor


class TestGFSFileDiscovery:
    def test_find_files_in_standard_path(self, mock_config, mock_gfs_dir):
        """Find GFS files in gfs.YYYYMMDD/HH/atmos/ structure."""
        # Disable file size check for mock files
        proc = GFSProcessor(mock_config, mock_gfs_dir, Path("/tmp/out"))
        proc.MIN_FILE_SIZE = 0  # Mock files are tiny

        files = proc.find_input_files()
        assert len(files) > 0

    def test_compute_search_cycles(self, mock_config):
        """Verify cycle computation covers nowcast window."""
        proc = GFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"))
        cycles = proc._compute_search_cycles()

        # For 12z with 6h nowcast, should include at least 06z and 12z
        cycle_hours = [c[1] for c in cycles]
        assert 12 in cycle_hours
        assert 6 in cycle_hours

    def test_compute_search_cycles_24h_nowcast(self):
        """STOFS-style 24h nowcast should search more cycles."""
        cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)
        proc = GFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"))
        cycles = proc._compute_search_cycles()

        # 24h nowcast from 12z goes back to yesterday 12z
        assert len(cycles) >= 5  # Should span ~24h of 6h cycles

    def test_compute_search_cycles_forecast_24h_nowcast_covers_window_start(self):
        """Regression: forecast-phase lookback must scale with nowcast_hours.

        The datm_forcing.nc written by the forecast-phase build is also read
        by the nowcast SCHISM execution (shared file, see _get_time_window),
        so its earliest searched cycle must reach back to cycle - nowcast_hours.
        STOFS-3D-ATL-UFS runs nowcast_hours=24; a fixed 6h-back fallback
        previously left 18h of the window uncovered.
        """
        cfg = ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260802", cyc=12)
        proc = GFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"), phase="forecast")
        cycles = proc._compute_search_cycles()

        cycle_dt = datetime(2026, 8, 2, 12)
        window_start = cycle_dt - timedelta(hours=cfg.nowcast_hours)  # 2026-08-01 12z

        earliest = min(date + timedelta(hours=hour) for date, hour in cycles)
        assert earliest <= window_start, (
            f"earliest searched cycle {earliest} does not cover window start {window_start}"
        )
        # Minimum lookback -- not ballooned past what's needed to cover it.
        assert earliest == window_start
        # Newest-first: the current cycle must lead the list so downstream
        # keep-first dedup (_select_files_for_window / _extract_all) prefers
        # it over the older fallback cycles for every valid time it covers.
        assert cycles == [
            (datetime(2026, 8, 2), 12),
            (datetime(2026, 8, 2), 6),
            (datetime(2026, 8, 2), 0),
            (datetime(2026, 8, 1), 18),
            (datetime(2026, 8, 1), 12),
        ]

    def test_compute_search_cycles_forecast_6h_nowcast_same_set_new_order(self):
        """nowcast_hours=6 (SECOFS-UFS) forecast-phase search still covers
        just one fallback cycle back (same set as the pre-fix fixed
        "one cycle back" fallback), but newest-first instead of oldest-first.

        This is a deliberate behavior change, not a regression: the old
        oldest-first order meant the 06z (older) cycle won the keep-first
        dedup for every valid time it shared with 12z, so SECOFS-UFS's
        forecast forcing was silently sourced from 06z. Newest-first makes
        12z (the current, freshest cycle) win instead -- see
        test_forecast_freshest_cycle_wins_at_and_after_cycle_time in
        TestGFSFreshestCycleWins for the downstream file-selection proof.
        """
        cfg = ForcingConfig.for_secofs_ufs(pdy="20260401", cyc=12)
        proc = GFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"), phase="forecast")
        cycles = proc._compute_search_cycles()

        assert cycles == [
            (datetime(2026, 4, 1), 12),
            (datetime(2026, 4, 1), 6),
        ]

    def test_compute_search_cycles_forecast_24h_nowcast_date_boundary(self):
        """A 24h lookback from a t00z cycle must cross into the previous day."""
        cfg = ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260802", cyc=0)
        proc = GFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"), phase="forecast")
        cycles = proc._compute_search_cycles()

        assert cycles == [
            (datetime(2026, 8, 2), 0),
            (datetime(2026, 8, 1), 18),
            (datetime(2026, 8, 1), 12),
            (datetime(2026, 8, 1), 6),
            (datetime(2026, 8, 1), 0),
        ]
        # Earliest (last, since newest-first) cycle's date rolled back to
        # the previous day.
        assert cycles[-1][0] == datetime(2026, 8, 1)

    def test_compute_search_cycles_forecast_nws2_unaffected_by_nowcast_hours(self):
        """Standalone sflux (nws=2) forecast search doesn't scale with
        nowcast_hours -- only nws=4 (DATM) shares the file with the nowcast leg.
        """
        cfg = ForcingConfig(
            lon_min=-80.0, lon_max=-70.0, lat_min=25.0, lat_max=35.0,
            pdy="20260401", cyc=12, nowcast_hours=24, forecast_hours=108, nws=2,
        )
        proc = GFSProcessor(cfg, Path("/tmp"), Path("/tmp/out"), phase="forecast")
        cycles = proc._compute_search_cycles()

        assert cycles == [
            (datetime(2026, 4, 1), 12),
            (datetime(2026, 4, 1), 6),
        ]

    def test_backup_list(self, mock_config, mock_gfs_dir):
        proc = GFSProcessor(mock_config, mock_gfs_dir, Path("/tmp/out"))
        proc.MIN_FILE_SIZE = 0

        backup = proc._build_backup_list()
        assert len(backup) > 0

    def test_base_date_computation(self, mock_config):
        proc = GFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"))
        base = proc._compute_base_date()

        # base_date is always day-start (00Z), matching Fortran convention
        # 12z - 6h = 06z → truncated to 00Z on same day
        expected = datetime(2026, 4, 1, 0, 0, 0)
        assert base == expected


class TestGFSVariables:
    def test_default_variables(self, mock_config):
        proc = GFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"))
        assert len(proc.variables) == 8
        assert "uwind" in proc.variables
        assert "prate" in proc.variables

    def test_custom_variables(self, mock_config):
        proc = GFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"),
                           variables=["uwind", "vwind"])
        assert proc.variables == ["uwind", "vwind"]

    def test_all_18_variables_mapped(self):
        """Ensure all 18 GRIB2 variables have valid mappings."""
        assert len(GFSProcessor.GRIB2_VARIABLES) == 18
        for name, (grib_var, level) in GFSProcessor.GRIB2_VARIABLES.items():
            assert isinstance(grib_var, str)
            assert isinstance(level, str)


class TestGFSProcess:
    def test_process_no_input_returns_failure(self, mock_config, tmp_path):
        """Empty input path -> failure result."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        proc = GFSProcessor(mock_config, empty_dir, tmp_path / "out")
        proc.MIN_FILE_SIZE = 0
        result = proc.process()

        assert not result.success
        assert "No GFS" in result.errors[0]

    def test_process_with_mock_extractor(self, mock_config, mock_gfs_dir, tmp_path):
        """Full pipeline with mocked GRIB extraction."""
        mock_extractor = MagicMock()
        mock_extractor.get_grid.return_value = (
            np.linspace(-80, -70, 5),
            np.linspace(25, 35, 5),
        )
        mock_extractor.extract.return_value = np.random.rand(5, 5).astype(np.float32)

        out_dir = tmp_path / "out"
        proc = GFSProcessor(mock_config, mock_gfs_dir, out_dir,
                           extractor=mock_extractor)
        proc.MIN_FILE_SIZE = 0

        result = proc.process()

        assert result.success
        assert len(result.output_files) > 0
        assert result.metadata["num_input_files"] > 0

    def test_min_file_size(self):
        # Class-level fallback is 40 MB; resolution-specific set in __init__
        assert GFSProcessor.MIN_FILE_SIZE == 40_000_000
        assert GFSProcessor.MIN_FILE_SIZE_BY_RES["0p25"] == 400_000_000
        assert GFSProcessor.MIN_FILE_SIZE_BY_RES["0p50"] == 40_000_000


class TestGFSFreshestCycleWins:
    """File-selection-level regression test for the forecast-phase
    newest-first cycle ordering.

    _compute_search_cycles's forecast-phase output order feeds
    _build_file_list (which walks `cycles` to assemble the discovered file
    list), whose result is then deduped keep-first-per-valid-time by
    _select_files_for_window / _extract_all. Getting that order wrong
    doesn't just reorder a list -- because every searched cycle gets leads
    reaching out toward the same forecast end (see _build_file_list's
    max_fhr), an older cycle's files cover the same valid times as the
    current cycle's. Oldest-first order let a stale cycle's long lead win
    the keep-first dedup for every valid time it shared with the current
    cycle, including the entire forecast window at and after cycle time.
    """

    @staticmethod
    def _make_forecast_multicycle_dir(root, cycles, max_lead_hours):
        """Build a GFS tree with one cycle dir per (date, hour) in
        `cycles`, each carrying hourly leads out to `max_lead_hours` --
        long enough for every cycle's files to overlap every other
        cycle's valid times, including times at and after the newest
        (current) cycle. Mirrors the realistic overlap
        test_gfs_prune_parity.py's `_make_multicycle_gfs_dir` fabricates,
        scoped to a specific cycle set instead of a fixed day range.
        """
        gfs_root = root / "gfs_data"
        for date, cyc in cycles:
            ds = date.strftime("%Y%m%d")
            atmos = gfs_root / f"gfs.{ds}" / f"{cyc:02d}" / "atmos"
            atmos.mkdir(parents=True, exist_ok=True)
            for fhr in range(0, max_lead_hours + 1):
                f = atmos / f"gfs.t{cyc:02d}z.pgrb2.0p25.f{fhr:03d}"
                f.write_bytes(b"\x00" * 1024)
        return gfs_root

    @staticmethod
    def _source_cycle(f):
        """Recover (cyc_date, cyc_hour) for a discovered GFS file from its
        own filename and parent path -- independent of any processor
        bookkeeping, so this is a check on the file that was actually
        selected, not on internal state.
        """
        cyc_hour = int(f.name.split(".t")[1][:2])
        for parent in [f.parent, f.parent.parent, f.parent.parent.parent]:
            if parent.name.startswith("gfs."):
                cyc_date = datetime.strptime(
                    parent.name.split("gfs.")[1][:8], "%Y%m%d"
                )
                return cyc_date, cyc_hour
        raise AssertionError(f"could not recover source cycle from {f}")

    def test_forecast_freshest_cycle_wins_at_and_after_cycle_time(self, tmp_path):
        """nowcast_hours=24 forecast search (STOFS-3D-ATL-UFS shape):
        every selected record at or after cycle time must come from the
        CURRENT cycle; only pre-cycle nowcast-overlap hours may come from
        older fallback cycles, and they must not all come from the single
        oldest cycle.

        Regression proof: with the forecast branch's cycle order reverted
        to oldest-first (`cycles.insert(0, ...)` instead of
        `cycles.append(...)`), this test fails on the first assertion --
        the oldest searched cycle's long lead wins the keep-first dedup
        for every valid time it covers, including the entire forecast
        window at and after cycle time.
        """
        pdy, cyc = "20260802", 12
        cfg = ForcingConfig.for_stofs_3d_atl_ufs(pdy=pdy, cyc=cyc)
        cycle_dt = datetime(2026, 8, 2, 12)
        current_cycle = (datetime(2026, 8, 2), 12)
        oldest_cycle = (datetime(2026, 8, 1), 12)

        # Determine the cycle set independently of file discovery (this
        # only needs input_path to exist for _build_file_list's path
        # probing later, not for computing the cycle list itself).
        probe = GFSProcessor(cfg, tmp_path, tmp_path / "out", phase="forecast")
        cycles = probe._compute_search_cycles()
        assert set(cycles) == {
            (datetime(2026, 8, 1), 12), (datetime(2026, 8, 1), 18),
            (datetime(2026, 8, 2), 0), (datetime(2026, 8, 2), 6),
            (datetime(2026, 8, 2), 12),
        }

        # Every cycle gets long leads (40h) so its files overlap valid
        # times well past cycle_dt -- the exact multi-cycle overlap
        # _build_file_list produces in production (every searched cycle's
        # max_fhr reaches out toward the same forecast end).
        gfs_root = self._make_forecast_multicycle_dir(tmp_path, cycles, 40)

        proc = GFSProcessor(cfg, gfs_root, tmp_path / "out", phase="forecast")
        proc.MIN_FILE_SIZE = 0  # mock files are tiny

        discovered = proc.find_input_files()
        selected = proc._select_files_for_window(discovered)
        assert selected, "no files survived selection"

        at_or_after = [f for f in selected if proc._parse_valid_time(f) >= cycle_dt]
        assert at_or_after, "expected some records at/after cycle time"
        for f in at_or_after:
            source_cycle = self._source_cycle(f)
            vt = proc._parse_valid_time(f)
            assert source_cycle == current_cycle, (
                f"valid time {vt} (>= cycle time {cycle_dt}) sourced from "
                f"{source_cycle} instead of the current cycle {current_cycle}"
            )

        pre_cycle = [
            (proc._parse_valid_time(f), self._source_cycle(f))
            for f in selected if proc._parse_valid_time(f) < cycle_dt
        ]
        assert pre_cycle, "expected some pre-cycle nowcast-overlap records"
        assert not all(src == oldest_cycle for _, src in pre_cycle), (
            "all pre-cycle records sourced from the single oldest cycle -- "
            "freshest-wins dedup is not taking effect"
        )
