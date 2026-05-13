"""Tests for NWMProcessor."""

import json
from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nwm import NWMProcessor, RiverConfig, MONTHLY_FLOW_FACTOR


class TestRiverConfig:
    def test_from_text(self, tmp_path):
        cfg_file = tmp_path / "rivers.txt"
        cfg_file.write_text(
            "# feature_id node_index name clim_flow\n"
            "12345 100 TestRiver1 50.0\n"
            "67890 200 TestRiver2 30.0\n"
        )
        rc = RiverConfig.from_text(cfg_file)
        assert rc.n_rivers == 2
        assert rc.feature_ids == [12345, 67890]
        assert rc.node_indices == [100, 200]
        assert rc.clim_flows == [50.0, 30.0]

    def test_from_json_list(self, tmp_path):
        cfg_file = tmp_path / "rivers.json"
        data = [
            {"feature_id": 111, "node_index": 10, "name": "R1", "clim_flow": 25.0},
            {"feature_id": 222, "node_index": 20, "name": "R2", "clim_flow": 15.0},
        ]
        cfg_file.write_text(json.dumps(data))
        rc = RiverConfig.from_json(cfg_file)
        assert rc.n_rivers == 2
        assert rc.names == ["R1", "R2"]


class TestNWMClimatology:
    def test_generate_climatology(self, tmp_path):
        """Verify climatology fallback produces reasonable output."""
        rc = RiverConfig(
            feature_ids=[111, 222],
            node_indices=[10, 20],
            clim_flows=[100.0, 50.0],
        )
        cfg = ForcingConfig(
            lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
            pdy="20260401", cyc=12,  # April
        )
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)
        flows, times = proc._generate_climatology(55)

        assert flows.shape == (55, 2)
        assert len(times) == 55

        # April factor = 1.5
        expected_r1 = 100.0 * 1.5
        assert flows[0, 0] == pytest.approx(expected_r1)

    def test_monthly_factors_sum(self):
        """Factors should average ~1.0 over the year."""
        avg = sum(MONTHLY_FLOW_FACTOR.values()) / 12
        assert 0.9 < avg < 1.1

    def test_stofs_mode_climatology_non_zero(self, tmp_path):
        """STOFS sources.json carries no clim_flows — fallback must still
        produce non-zero flows so SCHISM doesn't run with dead rivers when
        NWM file discovery fails (V18 SECOFS-UFS regression)."""
        rc = RiverConfig(
            feature_ids=[111, 222, 333],
            node_indices=[10, 20, 30],
            clim_flows=[0.0, 0.0, 0.0],  # STOFS-mode loads zeros
            feature_id_groups=[[111], [222], [333]],
        )
        cfg = ForcingConfig(
            lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
            pdy="20260507", cyc=0,
            river_default_flow=1.5,
        )
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)
        flows, times = proc._generate_climatology(55)

        assert flows.shape == (55, 3)
        # May factor = 1.30, so each river gets 1.5 * 1.30 = 1.95 m^3/s
        assert flows.min() > 0.0, "STOFS climatology must not be all zeros"
        assert flows[0, 0] == pytest.approx(1.5 * 1.30)

    def test_clim_flows_preserved_when_positive(self, tmp_path):
        """COMF rivers with non-zero clim_flows should keep their value;
        the default fallback applies only to zero/missing entries."""
        rc = RiverConfig(
            feature_ids=[111, 222],
            node_indices=[10, 20],
            clim_flows=[100.0, 0.0],  # second is missing
        )
        cfg = ForcingConfig(
            lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
            pdy="20260401", cyc=12,
            river_default_flow=2.0,
        )
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)
        flows, _ = proc._generate_climatology(10)

        assert flows[0, 0] == pytest.approx(100.0 * 1.5)  # April factor
        assert flows[0, 1] == pytest.approx(2.0 * 1.5)


class TestNWMProcess:
    def test_no_config_returns_failure(self, mock_config, tmp_path):
        proc = NWMProcessor(mock_config, tmp_path, tmp_path / "out")
        result = proc.process()
        assert not result.success
        assert "river configuration" in result.errors[0].lower()

    def test_climatology_fallback(self, mock_config, tmp_path):
        """With no NWM files, should fall back to climatology and succeed."""
        rc = RiverConfig(
            feature_ids=[111],
            node_indices=[10],
            clim_flows=[75.0],
        )
        out_dir = tmp_path / "out"
        proc = NWMProcessor(mock_config, tmp_path / "empty_nwm", out_dir, river_config=rc)
        result = proc.process()

        assert result.success
        assert result.metadata["used_climatology"] is True

        # Check output files exist
        assert (out_dir / "vsource.th").exists()
        assert (out_dir / "msource.th").exists()
        assert (out_dir / "source_sink.in").exists()

    def test_vsource_format(self, mock_config, tmp_path):
        """Verify vsource.th has correct format: time flow1 flow2 ..."""
        rc = RiverConfig(
            feature_ids=[111, 222],
            node_indices=[10, 20],
            clim_flows=[100.0, 50.0],
        )
        out_dir = tmp_path / "out"
        proc = NWMProcessor(mock_config, tmp_path, out_dir, river_config=rc)
        proc.process()

        vsource = (out_dir / "vsource.th").read_text().strip().split("\n")
        # Each line: time_seconds flow1 flow2
        parts = vsource[0].split()
        assert len(parts) == 3  # time + 2 rivers
        assert float(parts[0]) == 0.0  # First time = 0

    def test_source_sink_format(self, mock_config, tmp_path):
        """Verify source_sink.in lists all river nodes."""
        rc = RiverConfig(
            feature_ids=[111, 222, 333],
            node_indices=[10, 20, 30],
            clim_flows=[50.0, 30.0, 20.0],
        )
        out_dir = tmp_path / "out"
        proc = NWMProcessor(mock_config, tmp_path, out_dir, river_config=rc)
        proc.process()

        lines = (out_dir / "source_sink.in").read_text().strip().split("\n")
        assert lines[0] == "3"  # 3 sources
        assert lines[-1] == "0"  # 0 sinks


class TestNormalizeToSimulationGrid:
    """Regression tests for _normalize_to_simulation_grid.

    SCHISM source readers require the time axis to start at 0. When the
    NWM cycle's earliest-available product is at hour first_hour > 0
    (e.g., analysis_assim didn't include tm00), hours [0, first_hour)
    must be back-filled with the flow from first_hour so the resulting
    vsource.th doesn't start at 14400s and trigger
    'ABORT: MISC: vsource.th start time wrong' at SCHISM init.
    """

    def _proc(self, tmp_path):
        rc = RiverConfig(
            feature_ids=[111], node_indices=[10], clim_flows=[50.0],
        )
        cfg = ForcingConfig(
            lon_min=-80, lon_max=-70, lat_min=25, lat_max=35,
            pdy="20260507", cyc=0, river_default_flow=2.0,
        )
        return NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)

    def test_first_hour_zero_passes_through(self, tmp_path):
        proc = self._proc(tmp_path)
        flows = np.array([[1.0], [2.0], [3.0]])
        times = [0.0, 1.0, 2.0]
        out_flows, out_times = proc._normalize_to_simulation_grid(flows, times, n_target=3)
        assert out_times == [0.0, 1.0, 2.0]
        assert out_flows.shape == (3, 1)
        assert out_flows[0, 0] == 1.0

    def test_first_hour_gt_zero_backfills_to_zero(self, tmp_path):
        """The bug-fix regression: when earliest data is at hour 4, the output
        time axis must still start at 0 (back-fill [0, 4) with flow at h=4).
        """
        proc = self._proc(tmp_path)
        flows = np.array([[10.0], [20.0], [30.0]])
        times = [4.0, 5.0, 6.0]
        out_flows, out_times = proc._normalize_to_simulation_grid(flows, times, n_target=8)
        # Output must start at 0, not at 4
        assert out_times[0] == 0.0
        assert out_times == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert out_flows.shape == (8, 1)
        # Hours [0, 4) back-filled with flow at first available hour (=10.0)
        for h in range(4):
            assert out_flows[h, 0] == 10.0, f"hour {h} should back-fill to 10.0"
        # Hours [4, 7) match input
        assert out_flows[4, 0] == 10.0
        assert out_flows[5, 0] == 20.0
        assert out_flows[6, 0] == 30.0
        # Hour 7 forward-filled from hour 6
        assert out_flows[7, 0] == 30.0

    def test_empty_input_returns_empty(self, tmp_path):
        proc = self._proc(tmp_path)
        out_flows, out_times = proc._normalize_to_simulation_grid(
            np.array([]).reshape(0, 1), [], n_target=10
        )
        assert out_flows.size == 0
        assert out_times == []

    def test_all_times_outside_window_returns_empty(self, tmp_path):
        """All input hours >= n_target → climatology fallback signal."""
        proc = self._proc(tmp_path)
        flows = np.array([[1.0], [2.0]])
        times = [10.0, 11.0]  # both beyond n_target=5
        out_flows, out_times = proc._normalize_to_simulation_grid(flows, times, n_target=5)
        assert out_flows.size == 0
        assert out_times == []


class TestExtractStreamflowTimeAnchor:
    """``_extract_streamflow`` must report ``times`` as hours-from-model_t0,
    not as file-index, so the downstream normalize/write stages anchor the
    output ``vsource.th`` at SCHISM's ``model_t0 = cycle - nowcast_hours``.

    Earlier code labelled file 0 as ``t=0`` regardless of its actual valid
    time. For analysis_assim with pre-cycle lookback (tmHH > 0) this shifted
    the cycle-hour flow to a positive offset.
    """

    def _write_nwm_file(self, dir_path, cyc, tm, feature_ids, flows,
                        valid_time_attr=True):
        """Write a minimal NWM channel_rt netCDF the processor can read."""
        from netCDF4 import Dataset
        from datetime import datetime, timedelta
        fname = f"nwm.t{cyc:02d}z.analysis_assim.channel_rt.tm{tm:02d}.conus.nc"
        path = dir_path / fname
        ds = Dataset(str(path), "w", format="NETCDF4")
        ds.createDimension("feature_id", len(feature_ids))
        v_fid = ds.createVariable("feature_id", "i8", ("feature_id",))
        v_q = ds.createVariable("streamflow", "f4", ("feature_id",))
        v_fid[:] = np.array(feature_ids, dtype=np.int64)
        v_q[:] = np.array(flows, dtype=np.float32)
        if valid_time_attr:
            cycle = datetime(2026, 5, 7, cyc) - timedelta(hours=tm)
            ds.model_output_valid_time = cycle.strftime("%Y-%m-%d_%H:%M:%S")
        ds.close()
        return path

    def test_time_anchored_at_cycle_minus_nowcast(self, tmp_path):
        """Files at valid_time = cycle should map to t_hours = nowcast_hours.

        SECOFS-UFS: nowcast_hours=6, cycle=2026-05-07 00z, model_t0 = 2026-05-06
        18z. An analysis_assim tm00 file (valid at cycle) should report
        ``t_hours = 6``, not ``t_hours = 0``.
        """
        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        rc = RiverConfig(feature_ids=[111], node_indices=[10], clim_flows=[50.0])
        nwm_dir = tmp_path / "nwm.20260507" / "analysis_assim"
        nwm_dir.mkdir(parents=True)
        # Three lookback hours: tm02 (valid 22z prior day), tm01 (23z), tm00 (00z cycle)
        for tm in (2, 1, 0):
            self._write_nwm_file(nwm_dir, cyc=0, tm=tm,
                                 feature_ids=[111], flows=[10.0 + tm])
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)
        files = sorted(nwm_dir.glob("*.nc"))
        # Hand the files in valid-time order (tm02, tm01, tm00).
        from nos_utils.forcing.nwm import _nwm_valid_time
        files = sorted(files, key=_nwm_valid_time)
        flows, times = proc._extract_streamflow(files)
        # model_t0 = cycle - 6h = 2026-05-06 18z
        # tm02 valid 22z prior day = 2026-05-06 22z → t=4
        # tm01 valid 23z prior day = 2026-05-06 23z → t=5
        # tm00 valid 00z cycle    = 2026-05-07 00z → t=6
        assert times == [4.0, 5.0, 6.0], (
            f"Expected times [4,5,6] = hours from model_t0; got {times}"
        )

    def test_time_axis_falls_back_to_filename_without_attribute(self, tmp_path):
        """When NetCDF lacks ``model_output_valid_time``, derive from filename."""
        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        rc = RiverConfig(feature_ids=[111], node_indices=[10], clim_flows=[50.0])
        nwm_dir = tmp_path / "nwm.20260507" / "analysis_assim"
        nwm_dir.mkdir(parents=True)
        # Build files without model_output_valid_time attribute.
        for tm in (1, 0):
            self._write_nwm_file(nwm_dir, cyc=0, tm=tm,
                                 feature_ids=[111], flows=[10.0],
                                 valid_time_attr=False)
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)
        files = sorted(nwm_dir.glob("*.nc"))
        from nos_utils.forcing.nwm import _nwm_valid_time
        files = sorted(files, key=_nwm_valid_time)
        flows, times = proc._extract_streamflow(files)
        # tm01 → 23z prior day = t=5; tm00 → 00z cycle = t=6
        assert times == [5.0, 6.0]
