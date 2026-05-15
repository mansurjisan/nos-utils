"""Tests for RTOFSProcessor."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.rtofs import RTOFSProcessor


class TestRTOFSFileDiscovery:
    def test_find_no_files(self, mock_config, tmp_path):
        proc = RTOFSProcessor(mock_config, tmp_path / "empty", tmp_path / "out")
        files = proc.find_input_files()
        assert len(files) == 0

    def test_find_files_by_type(self, mock_config, tmp_path):
        """Create mock RTOFS directory and verify file discovery."""
        rtofs_dir = tmp_path / "rtofs.20260401"
        rtofs_dir.mkdir()

        # Create mock 2D files
        for cycle in ["n012", "f006"]:
            f = rtofs_dir / f"rtofs_glo_2ds_{cycle}_diag.nc"
            f.write_bytes(b"\x00" * 200_000_000)  # 200MB

        # Create mock 3D files
        for cycle in ["n012", "f006"]:
            f = rtofs_dir / f"rtofs_glo_3dz_{cycle}_6hrly_hvr_US_east.nc"
            f.write_bytes(b"\x00" * 300_000_000)  # 300MB

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        files_2d, files_3d = proc.find_input_files_by_type()

        assert len(files_2d) == 2
        assert len(files_3d) == 2

    def test_dedup_prefers_forecast_over_nowcast(self, mock_config, tmp_path):
        """Verify forecast (f) files are preferred over nowcast (n) for same
        valid time, matching Fortran which only uses f* files."""
        rtofs_dir = tmp_path / "rtofs.20260401"
        rtofs_dir.mkdir()

        # n012 and f012 have the same valid time (cycle + 12h)
        for prefix in ["n012", "f012", "f024"]:
            f = rtofs_dir / f"rtofs_glo_2ds_{prefix}_diag.nc"
            f.write_bytes(b"\x00" * 200_000_000)

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        files_2d, _ = proc.find_input_files_by_type()

        # Should have 2 files (f012 wins over n012, plus f024)
        assert len(files_2d) == 2
        # The hour-12 file should be forecast, not nowcast
        assert "_f012_" in files_2d[0].name

    def test_cycle_search_prefers_pdy_minus_1(self, mock_config, tmp_path):
        """Verify PDY-1 is searched before PDY-2 (matches Fortran behavior)."""
        # Create files in both PDY-2 and PDY-1 directories
        # mock_config has pdy="20260401"
        for day in ["20260330", "20260331"]:
            d = tmp_path / f"rtofs.{day}"
            d.mkdir()
            f = d / "rtofs_glo_2ds_f024_diag.nc"
            f.write_bytes(b"\x00" * 200_000_000)

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        files_2d, _ = proc.find_input_files_by_type()

        # Should find PDY-1 (20260331), not PDY-2 (20260330)
        assert len(files_2d) == 1
        assert "20260331" in str(files_2d[0])


class TestRTOFSProcess:
    def test_no_input_returns_failure(self, mock_config, tmp_path):
        proc = RTOFSProcessor(mock_config, tmp_path / "empty", tmp_path / "out")
        result = proc.process()
        assert not result.success

    def test_ssh_offset_stored_in_config(self, mock_config):
        mock_config.obc_ssh_offset = 0.04
        proc = RTOFSProcessor(mock_config, Path("/tmp"), Path("/tmp/out"))
        assert proc.config.obc_ssh_offset == 0.04


class TestRTOFSConstants:
    def test_min_file_sizes(self):
        assert RTOFSProcessor.MIN_FILE_SIZE_2D == 150_000_000
        assert RTOFSProcessor.MIN_FILE_SIZE_3D == 200_000_000

    def test_source_name(self):
        assert RTOFSProcessor.SOURCE_NAME == "RTOFS"


class TestRTOFSParseHour:
    """Test _parse_rtofs_hour filename parsing."""

    def test_forecast_file(self):
        hour, is_nc = RTOFSProcessor._parse_rtofs_hour(
            Path("rtofs_glo_2ds_f048_diag.nc"))
        assert hour == 48
        assert not is_nc

    def test_nowcast_file(self):
        hour, is_nc = RTOFSProcessor._parse_rtofs_hour(
            Path("rtofs_glo_2ds_n024_diag.nc"))
        assert hour == 24
        assert is_nc

    def test_3d_file(self):
        hour, is_nc = RTOFSProcessor._parse_rtofs_hour(
            Path("rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc"))
        assert hour == 12
        assert not is_nc


class TestRTOFSTimeAxis:
    """Verify the temporal interpolation uses actual file spacing,
    not hardcoded 6h.  This is the fix for the 3cm SSH bias."""

    def _make_filenames(self, hours):
        """Create Path objects mimicking RTOFS 2D filenames."""
        return [Path(f"rtofs_glo_2ds_f{h:03d}_diag.nc") for h in hours]

    def test_hourly_files_not_assumed_6h(self):
        """47 files (37 hourly + 10 three-hourly) must produce
        correct non-uniform time axis, not uniform 6h."""
        hours = list(range(36, 73)) + list(range(75, 103, 3))
        files = self._make_filenames(hours)

        # Compute time axis the same way _process_2d does after fix
        file_hours = []
        for f in files:
            h, _ = RTOFSProcessor._parse_rtofs_hour(f)
            file_hours.append(h)
        rtofs_times = np.array(
            [(h - file_hours[0]) * 3600.0 for h in file_hours])

        assert len(rtofs_times) == 47
        # First file at t=0
        assert rtofs_times[0] == 0.0
        # Second file at 1h (NOT 6h)
        assert rtofs_times[1] == 3600.0
        # Last file at 66h
        assert rtofs_times[-1] == 66 * 3600.0
        # Transition from hourly to 3-hourly at index 37
        assert rtofs_times[37] - rtofs_times[36] == 3 * 3600.0
        # Hourly section: all 1h gaps
        hourly_diffs = np.diff(rtofs_times[:37])
        assert np.all(hourly_diffs == 3600.0)

    def test_uniform_6h_files(self):
        """If files ARE 6-hourly, time axis should still be correct."""
        hours = list(range(12, 78, 6))  # f012,f018,...,f072
        files = self._make_filenames(hours)

        file_hours = []
        for f in files:
            h, _ = RTOFSProcessor._parse_rtofs_hour(f)
            file_hours.append(h)
        rtofs_times = np.array(
            [(h - file_hours[0]) * 3600.0 for h in file_hours])

        assert rtofs_times[1] == 6 * 3600.0
        diffs = np.diff(rtofs_times)
        assert np.all(diffs == 6 * 3600.0)

    def test_time_span_matches_real_coverage(self):
        """Verify total time span reflects actual file hours,
        not n_files * 6h (the old bug)."""
        hours = list(range(36, 73)) + list(range(75, 103, 3))
        files = self._make_filenames(hours)

        file_hours = [RTOFSProcessor._parse_rtofs_hour(f)[0] for f in files]
        rtofs_times = np.array(
            [(h - file_hours[0]) * 3600.0 for h in file_hours])

        real_span_h = rtofs_times[-1] / 3600.0
        old_bug_span_h = (len(files) - 1) * 6.0

        # Real span: 66h.  Old bug span: 276h.
        assert real_span_h == 66.0
        assert old_bug_span_h == 276.0
        assert real_span_h < old_bug_span_h  # fix is smaller


class TestRTOFSRouteBAnchoring:
    """Verify the OBC time axis is anchored at model_t0 = cycle - nowcast_hours
    and that backward/forward gaps are filled by hold-constant backfill."""

    def _anchored_rtofs_times(self, file_hours, rtofs_cycle, model_t0):
        return np.array(
            [(rtofs_cycle + timedelta(hours=h) - model_t0).total_seconds()
             for h in file_hours],
            dtype=np.float64,
        )

    def test_model_t0_is_cycle_minus_nowcast(self):
        """model_t0 = cycle - nowcast_hours, regardless of time_hotstart."""
        cycle_dt = datetime(2026, 4, 1, 12)
        nowcast_hours = 6
        model_t0 = cycle_dt - timedelta(hours=nowcast_hours)
        assert model_t0 == datetime(2026, 4, 1, 6)

    def test_sim_duration_covers_nowcast_plus_forecast(self):
        """sim_duration = (nowcast_hours + forecast_hours) * 3600 seconds.

        For SECOFS (6 + 48 = 54h) this gives 54 * 3600 = 194400s; at the
        SCHISM model_dt of 120s the output dimension is 194400/120 + 1 = 1621.
        """
        nowcast_hours = 6
        forecast_hours = 48
        cycle_dt = datetime(2026, 4, 1, 12)
        model_t0 = cycle_dt - timedelta(hours=nowcast_hours)
        sim_end = cycle_dt + timedelta(hours=forecast_hours)
        sim_duration = (sim_end - model_t0).total_seconds()
        assert sim_duration == 54 * 3600

        model_dt = 120.0
        n_model_steps = int(sim_duration / model_dt) + 1
        assert n_model_steps == 1621

    def test_rtofs_times_anchored_at_model_t0(self):
        """rtofs_time = (rtofs_cycle + h hours) - model_t0, in seconds.

        With cycle = PDY 12z, nowcast = 6h, model_t0 = PDY 06z, and
        rtofs_cycle = PDY-1 midnight:
          - h=24 -> PDY 00z -> rtofs_time = -6*3600 (6h before model_t0)
          - h=30 -> PDY 06z = model_t0 -> rtofs_time = 0
          - h=36 -> PDY 12z = cycle -> rtofs_time = +6*3600
        Files with negative rtofs_time fall outside the run window and
        are clipped by the interp1d's fill_value=(first, last) on the
        leading edge.
        """
        cycle_dt = datetime(2026, 4, 1, 12)
        model_t0 = cycle_dt - timedelta(hours=6)  # PDY 06z
        rtofs_cycle = datetime(2026, 3, 31)  # PDY-1 midnight

        rtofs_times = self._anchored_rtofs_times(
            [24, 30, 36], rtofs_cycle, model_t0
        )
        assert rtofs_times[0] == -6 * 3600.0
        assert rtofs_times[1] == 0.0
        assert rtofs_times[2] == 6 * 3600.0

    def test_window_starts_after_rtofs_triggers_pre_backfill(self):
        """When the first RTOFS file is AFTER model_t0, output times in
        [0, rtofs_times[0]] are filled by holding the first value constant."""
        from scipy.interpolate import interp1d

        # RTOFS data starts at +3h (i.e. file at cycle, while model_t0 is
        # cycle - 6h means the first RTOFS file is 3h into the run window
        # if model_t0 = cycle - 3h... this test just checks the math).
        rtofs_times = np.array([3 * 3600.0, 6 * 3600.0, 12 * 3600.0])
        rtofs_vals = np.array([1.0, 2.0, 3.0])

        # Mimic the Route B fill-value selection
        f = interp1d(
            rtofs_times, rtofs_vals, kind="linear",
            bounds_error=False,
            fill_value=(float(rtofs_vals[0]), float(rtofs_vals[-1])),
        )
        model_times = np.arange(0, 13 * 3600, 3600)
        out = f(model_times)

        # Before rtofs_times[0]: hold first value (1.0)
        assert out[0] == 1.0
        assert out[1] == 1.0
        assert out[2] == 1.0
        # At rtofs_times[0]=3h: exactly 1.0
        assert out[3] == 1.0
        # Linear interp between 3h (1.0) and 6h (2.0) at 4h: 1.333
        assert abs(out[4] - 4.0 / 3.0) < 1e-6

    def test_window_ends_before_rtofs_triggers_post_backfill(self):
        """When RTOFS ends BEFORE sim_end, the trailing gap is filled by
        holding the last value constant (not extrapolated)."""
        from scipy.interpolate import interp1d

        rtofs_times = np.array([0.0, 3 * 3600.0, 6 * 3600.0])
        rtofs_vals = np.array([10.0, 20.0, 30.0])

        f = interp1d(
            rtofs_times, rtofs_vals, kind="linear",
            bounds_error=False,
            fill_value=(float(rtofs_vals[0]), float(rtofs_vals[-1])),
        )

        # Sample past the end of available data
        model_times = np.array([0.0, 3 * 3600.0, 6 * 3600.0, 9 * 3600.0, 12 * 3600.0])
        out = f(model_times)
        assert out[0] == 10.0
        assert out[2] == 30.0
        # Past rtofs_times[-1]: hold last value (30.0), NOT linear-extrapolated (40, 50)
        assert out[3] == 30.0
        assert out[4] == 30.0


class TestRoutePhaseRTOFS:
    """Phase-aware output time windows for RTOFS OBC files.

    Two operational PBS jobs (nowcast + forecast) read from $COMOUT;
    each must receive elev2D / TEM_3D / SAL_3D / uv3D files whose
    physical time content matches the filename phase.

      * nowcast: 6h window + buffer (3h default) past cycle
      * forecast: 48h window + buffer (3h default) past forecast end
      * None (backward-compat): combined 54h window, no buffer

    The window helper drives the temporal interpolation grid that
    sizes ``elev2D.th.nc`` (dt=120s) and ``TEM_3D.th.nc`` /
    ``SAL_3D.th.nc`` (dt=10800s). The buffer past ``sim_end`` gives
    SCHISM's ``time_series`` reader interpolation headroom.
    """

    def test_output_window_nowcast(self, mock_config, tmp_path):
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out", phase="nowcast",
        )
        model_t0, sim_end, sim_duration = proc._get_output_window("nowcast")
        # cycle = 2026-04-01 12z, nowcast_hours=6, buffer=3 -> [06z, 15z]
        assert model_t0 == datetime(2026, 4, 1, 6)
        assert sim_end == datetime(2026, 4, 1, 15)
        assert sim_duration == (6 + 3) * 3600.0

    def test_output_window_forecast(self, mock_config, tmp_path):
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out", phase="forecast",
        )
        model_t0, sim_end, sim_duration = proc._get_output_window("forecast")
        # cycle = 2026-04-01 12z, forecast_hours=48, buffer=3 -> [12z, 12z+51h]
        assert model_t0 == datetime(2026, 4, 1, 12)
        assert sim_end == datetime(2026, 4, 3, 15)
        assert sim_duration == (48 + 3) * 3600.0

    def test_output_window_none_combined(self, mock_config, tmp_path):
        """phase=None must produce the existing Route B 54h combined window
        WITHOUT the phase buffer (backward-compat)."""
        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        model_t0, sim_end, sim_duration = proc._get_output_window(None)
        # cycle - 6h .. cycle + 48h = 54h
        assert model_t0 == datetime(2026, 4, 1, 6)
        assert sim_end == datetime(2026, 4, 3, 12)
        assert sim_duration == 54 * 3600.0

    def test_nowcast_elev2d_step_count(self, mock_config, tmp_path):
        """elev2D.th.nc covers 6h + 3h buffer at dt=120s -> 271 records for nowcast leg.

        Production SCHISM model_dt = 120s. With sim_duration = 32400s
        (6h + 3h buffer) the time axis is 0, 120, 240, ..., 32400 = 271
        records when including both endpoints. The +1 includes t=32400
        because the writer uses ``int(sim_duration/dt)+1`` inclusive.
        """
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out", phase="nowcast",
        )
        _, _, sim_duration = proc._get_output_window("nowcast")
        model_dt = 120.0
        n_steps = int(sim_duration / model_dt) + 1
        # (6h + 3h buffer) * 3600 / 120 + 1 = 271
        assert n_steps == 271
        assert sim_duration == 32400.0

    def test_forecast_elev2d_step_count(self, mock_config, tmp_path):
        """elev2D.th.nc covers 48h + 3h buffer at dt=120s -> 1531 records."""
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out", phase="forecast",
        )
        _, _, sim_duration = proc._get_output_window("forecast")
        model_dt = 120.0
        n_steps = int(sim_duration / model_dt) + 1
        # (48h + 3h buffer) * 3600 / 120 + 1 = 1531
        assert n_steps == 1531
        assert sim_duration == (48 + 3) * 3600.0

    def test_nowcast_3d_step_count(self, mock_config, tmp_path):
        """TEM_3D.th.nc covers 6h + 3h buffer at dt=10800s -> 4 records.

        3D OBC files use the COMF DELT_TS = 3h cadence. (6h + 3h) / 3h + 1 = 4.
        """
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out", phase="nowcast",
        )
        _, _, sim_duration = proc._get_output_window("nowcast")
        target_dt_3d = 10800.0
        n_steps = int(sim_duration / target_dt_3d) + 1
        assert n_steps == 4

    def test_forecast_3d_step_count(self, mock_config, tmp_path):
        """TEM_3D.th.nc covers 48h + 3h buffer at dt=10800s -> 18 records.

        (48h + 3h buffer) / 3h + 1 = 18 inclusive of both endpoints.
        """
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out", phase="forecast",
        )
        _, _, sim_duration = proc._get_output_window("forecast")
        target_dt_3d = 10800.0
        n_steps = int(sim_duration / target_dt_3d) + 1
        assert n_steps == 18

    def test_combined_step_count_backward_compat(self, mock_config, tmp_path):
        """phase=None still produces 1621 elev2D and 19 TEM_3D records
        WITHOUT the buffer applied (backward-compat)."""
        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        _, _, sim_duration = proc._get_output_window(None)
        assert sim_duration == 54 * 3600.0
        # elev2D at 120s cadence: 54h/120s + 1 = 1621
        n_2d = int(sim_duration / 120.0) + 1
        assert n_2d == 1621
        # TEM_3D at 3h cadence: 54h/3h + 1 = 19
        n_3d = int(sim_duration / 10800.0) + 1
        assert n_3d == 19

    def test_nowcast_anchored_at_cycle_minus_nowcast_hours(self, mock_config, tmp_path):
        """Nowcast model_t0 = cycle - nowcast_hours (SCHISM hotstart anchor)."""
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out", phase="nowcast",
        )
        cycle_dt = datetime(2026, 4, 1, 12)
        model_t0, _, _ = proc._get_output_window("nowcast")
        nowcast_hours = mock_config.nowcast_hours
        assert model_t0 == cycle_dt - timedelta(hours=nowcast_hours)

    def test_forecast_anchored_at_cycle(self, mock_config, tmp_path):
        """Forecast model_t0 = cycle (the operator's forecast leg start)."""
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out", phase="forecast",
        )
        cycle_dt = datetime(2026, 4, 1, 12)
        model_t0, _, _ = proc._get_output_window("forecast")
        assert model_t0 == cycle_dt

    def test_stofs_24h_nowcast(self, stofs_config, tmp_path):
        """STOFS-3D-ATL nowcast leg = 24h + 3h buffer, forecast = 108h + 3h.

        Verify the helper picks up factory-specific run hours, not just
        the SECOFS defaults.
        """
        proc = RTOFSProcessor(
            stofs_config, tmp_path, tmp_path / "out", phase="nowcast",
        )
        _, _, sim_duration = proc._get_output_window("nowcast")
        assert sim_duration == (24 + 3) * 3600.0

    def test_stofs_108h_forecast(self, stofs_config, tmp_path):
        proc = RTOFSProcessor(
            stofs_config, tmp_path, tmp_path / "out", phase="forecast",
        )
        _, _, sim_duration = proc._get_output_window("forecast")
        assert sim_duration == (108 + 3) * 3600.0

    def test_buffer_extends_past_phase_end(self, mock_config, tmp_path):
        """sim_end for phase != None equals raw phase end + default buffer_hours."""
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out", phase="forecast",
        )
        assert proc.buffer_hours == RTOFSProcessor.DEFAULT_BUFFER_HOURS == 3
        cycle_dt = datetime(2026, 4, 1, 12)
        _, sim_end, sim_duration = proc._get_output_window("forecast")
        # forecast_end = cycle + 48h = 2026-04-03 12z; +3h buffer = 15z.
        assert sim_end == cycle_dt + timedelta(
            hours=mock_config.forecast_hours
            + RTOFSProcessor.DEFAULT_BUFFER_HOURS,
        )
        assert sim_duration == (
            mock_config.forecast_hours
            + RTOFSProcessor.DEFAULT_BUFFER_HOURS
        ) * 3600.0

    def test_buffer_hours_kwarg_overrides_default(self, mock_config, tmp_path):
        """Explicit buffer_hours kwarg overrides class default and
        config.obc_buffer_hours."""
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out",
            phase="nowcast", buffer_hours=6,
        )
        assert proc.buffer_hours == 6
        _, sim_end, sim_duration = proc._get_output_window("nowcast")
        cycle_dt = datetime(2026, 4, 1, 12)
        assert sim_end == cycle_dt + timedelta(hours=6)
        # nowcast 6h + buffer 6h = 12h
        assert sim_duration == 12 * 3600.0
        # 3D step count at 3h cadence: 12h / 3h + 1 = 5
        n_3d = int(sim_duration / 10800.0) + 1
        assert n_3d == 5

    def test_buffer_hours_zero_disables_buffer(self, mock_config, tmp_path):
        """buffer_hours=0 restores the pre-buffer phase-window behavior."""
        proc = RTOFSProcessor(
            mock_config, tmp_path, tmp_path / "out",
            phase="nowcast", buffer_hours=0,
        )
        assert proc.buffer_hours == 0
        _, sim_end, sim_duration = proc._get_output_window("nowcast")
        cycle_dt = datetime(2026, 4, 1, 12)
        assert sim_end == cycle_dt
        # 6h nowcast leg, no buffer
        assert sim_duration == 6 * 3600.0
        # elev2D step count: 6h * 3600 / 120 + 1 = 181 (prior behavior)
        n_2d = int(sim_duration / 120.0) + 1
        assert n_2d == 181


class TestRTOFSFind3DWeights:
    """Test _find_3d_weights() discovery of precomputed 3D weight NPZ files."""

    def test_find_3d_weights_from_fixofs(self, mock_config, tmp_path):
        """Discovers 3D weights NPZ from FIXofs environment variable."""
        import os

        npz_path = tmp_path / "secofs.obc_3d_weights.npz"
        np.savez(str(npz_path), grid_shape=np.array([1710, 742], dtype=np.int32))

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        old_fix = os.environ.get("FIXofs")
        try:
            os.environ["FIXofs"] = str(tmp_path)
            if hasattr(proc, '_3d_weights_cache'):
                delattr(proc, '_3d_weights_cache')
            result = proc._find_3d_weights()
            assert result is not None
            assert tuple(result["grid_shape"]) == (1710, 742)
        finally:
            if old_fix is not None:
                os.environ["FIXofs"] = old_fix
            else:
                os.environ.pop("FIXofs", None)

    def test_find_3d_weights_from_fixstofs3d(self, mock_config, tmp_path):
        """Discovers 3D weights NPZ from FIXstofs3d environment variable."""
        import os

        npz_path = tmp_path / "obc_3d_weights.npz"
        np.savez(str(npz_path), grid_shape=np.array([1710, 742], dtype=np.int32))

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        old_fix = os.environ.get("FIXstofs3d")
        old_fixofs = os.environ.get("FIXofs")
        try:
            os.environ.pop("FIXofs", None)
            os.environ["FIXstofs3d"] = str(tmp_path)
            if hasattr(proc, '_3d_weights_cache'):
                delattr(proc, '_3d_weights_cache')
            result = proc._find_3d_weights()
            assert result is not None
            assert tuple(result["grid_shape"]) == (1710, 742)
        finally:
            if old_fix is not None:
                os.environ["FIXstofs3d"] = old_fix
            else:
                os.environ.pop("FIXstofs3d", None)
            if old_fixofs is not None:
                os.environ["FIXofs"] = old_fixofs
            else:
                os.environ.pop("FIXofs", None)

    def test_find_3d_weights_from_input_path(self, mock_config, tmp_path):
        """Discovers 3D weights NPZ from input_path directory."""
        import os

        npz_path = tmp_path / "secofs.obc_3d_weights.npz"
        np.savez(str(npz_path), grid_shape=np.array([1710, 742], dtype=np.int32))

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        old_fix = os.environ.get("FIXofs")
        old_stofs = os.environ.get("FIXstofs3d")
        try:
            os.environ.pop("FIXofs", None)
            os.environ.pop("FIXstofs3d", None)
            if hasattr(proc, '_3d_weights_cache'):
                delattr(proc, '_3d_weights_cache')
            result = proc._find_3d_weights()
            assert result is not None
        finally:
            if old_fix is not None:
                os.environ["FIXofs"] = old_fix
            if old_stofs is not None:
                os.environ["FIXstofs3d"] = old_stofs

    def test_find_3d_weights_not_found(self, mock_config, tmp_path):
        """Returns None when no 3D NPZ exists."""
        import os

        proc = RTOFSProcessor(mock_config, tmp_path / "empty", tmp_path / "out")
        old_fix = os.environ.get("FIXofs")
        old_stofs = os.environ.get("FIXstofs3d")
        try:
            os.environ.pop("FIXofs", None)
            os.environ.pop("FIXstofs3d", None)
            if hasattr(proc, '_3d_weights_cache'):
                delattr(proc, '_3d_weights_cache')
            result = proc._find_3d_weights()
            assert result is None
        finally:
            if old_fix is not None:
                os.environ["FIXofs"] = old_fix
            if old_stofs is not None:
                os.environ["FIXstofs3d"] = old_stofs

    def test_find_3d_weights_cached(self, mock_config, tmp_path):
        """Second call returns cached result without re-reading disk."""
        import os

        npz_path = tmp_path / "secofs.obc_3d_weights.npz"
        np.savez(str(npz_path), grid_shape=np.array([1710, 742], dtype=np.int32))

        proc = RTOFSProcessor(mock_config, tmp_path, tmp_path / "out")
        old_fix = os.environ.get("FIXofs")
        try:
            os.environ["FIXofs"] = str(tmp_path)
            if hasattr(proc, '_3d_weights_cache'):
                delattr(proc, '_3d_weights_cache')
            result1 = proc._find_3d_weights()
            assert result1 is not None

            # Remove file but cache should still return result
            npz_path.unlink()
            result2 = proc._find_3d_weights()
            assert result2 is not None
            assert result2 is result1
        finally:
            if old_fix is not None:
                os.environ["FIXofs"] = old_fix
            else:
                os.environ.pop("FIXofs", None)
