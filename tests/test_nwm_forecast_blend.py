"""Regression tests for the SECOFS NWM analysis+forecast blend.

The bug: ``_find_secofs_nwm_files`` searched only ``analysis_assim`` and
stopped at the first non-empty product. analysis_assim carries only past
snapshots, so ``_pad_to_target`` repeated the last snapshot across the whole
forecast horizon and ``vsource.th`` came out FLAT in time. The operational
COMF generator (``schism_cp_nwm_files_local.sh``) stages analysis +
short_range + medium_range_mem1, giving a time-varying series.

These tests verify the blend stages forecast files, collapses overlapping
forecasts to one per hour (short_range preferred over medium_range), and
that ``process()`` yields a time-varying vsource.th.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nwm import NWMProcessor, RiverConfig, _nwm_valid_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _touch_analysis(root: Path, pdy: str, cyc: int, tm: int = 0) -> Path:
    """Create an empty analysis_assim channel_rt file (filename carries the
    valid time; content not needed for discovery-only tests)."""
    d = root / f"nwm.{pdy}" / "analysis_assim"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"nwm.t{cyc:02d}z.analysis_assim.channel_rt.tm{tm:02d}.conus.nc"
    p.write_bytes(b"")
    return p


def _touch_short(root: Path, pdy: str, cyc: int, fhr: int) -> Path:
    d = root / f"nwm.{pdy}" / "short_range"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"nwm.t{cyc:02d}z.short_range.channel_rt.f{fhr:03d}.conus.nc"
    p.write_bytes(b"")
    return p


def _touch_medium(root: Path, pdy: str, cyc: int, fhr: int) -> Path:
    d = root / f"nwm.{pdy}" / "medium_range_mem1"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"nwm.t{cyc:02d}z.medium_range.channel_rt_1.f{fhr:03d}.conus.nc"
    p.write_bytes(b"")
    return p


def _sources(tmp_path: Path) -> Path:
    p = tmp_path / "sources.json"
    p.write_text(json.dumps({"100": [123]}))
    return p


def _proc(tmp_path: Path, pdy="20260507", cyc=0, **cfg_kw) -> NWMProcessor:
    cfg = ForcingConfig.for_secofs(pdy=pdy, cyc=cyc, **cfg_kw)
    rc = RiverConfig.from_sources_json(_sources(tmp_path))
    return NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)


# ---------------------------------------------------------------------------
# Discovery / selection logic (no netCDF needed)
# ---------------------------------------------------------------------------

class TestForecastBlendSelection:
    def test_analysis_only_unchanged(self, tmp_path):
        """No forecast dirs → blend returns exactly the analysis files
        (preserves the existing analysis-only contract)."""
        _touch_analysis(tmp_path, "20260507", cyc=0)
        _touch_analysis(tmp_path, "20260507", cyc=1)
        proc = _proc(tmp_path)
        files = proc.find_input_files()
        assert len(files) == 2
        assert all("analysis_assim" in f.name for f in files)

    def test_blend_adds_forecast_files(self, tmp_path):
        """Analysis present AND forecast present → both are returned."""
        # Analysis frontier at cycle 00z (model start = 06-05-06 18z).
        for cyc in (18, 19, 20, 21, 22, 23):
            _touch_analysis(tmp_path, "20260506", cyc=cyc)
        _touch_analysis(tmp_path, "20260507", cyc=0)
        # Forecast from the 00z cycle: short f001..f018, medium f003..f060.
        for fhr in range(1, 19):
            _touch_short(tmp_path, "20260507", cyc=0, fhr=fhr)
        for fhr in range(3, 61):
            _touch_medium(tmp_path, "20260507", cyc=0, fhr=fhr)

        proc = _proc(tmp_path)
        files = proc.find_input_files()
        names = [f.name for f in files]
        assert any("analysis_assim" in n for n in names)
        assert any("short_range" in n for n in names)
        assert any("medium_range" in n for n in names)

    def test_forecast_one_file_per_hour(self, tmp_path):
        """Overlapping short+medium forecasts collapse to one file per hour."""
        _touch_analysis(tmp_path, "20260507", cyc=0)  # frontier = 00z
        for fhr in range(1, 19):
            _touch_short(tmp_path, "20260507", cyc=0, fhr=fhr)
        for fhr in range(3, 61):
            _touch_medium(tmp_path, "20260507", cyc=0, fhr=fhr)

        proc = _proc(tmp_path)
        fc = proc._select_forecast_files(
            [datetime(2026, 5, 7), datetime(2026, 5, 6)],
            proc._glob_product_files(
                "analysis_assim", [datetime(2026, 5, 7), datetime(2026, 5, 6)]),
        )
        # Valid times must be unique (one per hour) and strictly increasing.
        vts = [_nwm_valid_time(f) for f in fc]
        assert vts == sorted(vts)
        assert len(vts) == len(set(vts)), "duplicate forecast hours not collapsed"

    def test_short_range_preferred_over_medium(self, tmp_path):
        """On an hour both products cover, short_range wins (fresher cycle)."""
        _touch_analysis(tmp_path, "20260507", cyc=0)  # frontier = 00z
        # Both products produce a file valid at 05-07 06z:
        #   short f006 from t00z, medium f006 from t00z.
        _touch_short(tmp_path, "20260507", cyc=0, fhr=6)
        _touch_medium(tmp_path, "20260507", cyc=0, fhr=6)

        proc = _proc(tmp_path)
        fc = proc._select_forecast_files(
            [datetime(2026, 5, 7), datetime(2026, 5, 6)],
            proc._glob_product_files(
                "analysis_assim", [datetime(2026, 5, 7), datetime(2026, 5, 6)]),
        )
        by_vt = {_nwm_valid_time(f): f for f in fc}
        chosen = by_vt[datetime(2026, 5, 7, 6)]
        assert "short_range" in chosen.name
        assert "channel_rt_1" not in chosen.name

    def test_forecast_excluded_before_frontier(self, tmp_path):
        """Forecast files valid at/under the analysis frontier are dropped
        (analysis owns the observed past)."""
        for cyc in (0, 1, 2, 3):
            _touch_analysis(tmp_path, "20260507", cyc=cyc)  # frontier = 03z
        # A forecast valid at 02z (<= frontier) must NOT be selected.
        _touch_short(tmp_path, "20260507", cyc=0, fhr=2)   # valid 02z
        _touch_short(tmp_path, "20260507", cyc=0, fhr=10)  # valid 10z (kept)

        proc = _proc(tmp_path)
        fc = proc._select_forecast_files(
            [datetime(2026, 5, 7), datetime(2026, 5, 6)],
            proc._glob_product_files(
                "analysis_assim", [datetime(2026, 5, 7), datetime(2026, 5, 6)]),
        )
        vts = [_nwm_valid_time(f) for f in fc]
        assert datetime(2026, 5, 7, 2) not in vts
        assert datetime(2026, 5, 7, 10) in vts

    def test_forecast_excluded_past_window_end(self, tmp_path):
        """Forecast files valid past the phase window end are dropped, and the
        window-end hour itself is the inclusive cap.

        for_secofs phase=None: start = 05-06 18z, total_hours = 6+48+18 = 72,
        so window_end = 05-09 18z (hour 72). A file valid exactly at 18z is
        kept; one at 19z (hour 73) is dropped. f-leads are from the 05-07 00z
        cycle, so valid = 05-07 00z + lead.
        """
        _touch_analysis(tmp_path, "20260507", cyc=0)
        _touch_medium(tmp_path, "20260507", cyc=0, fhr=50)   # 05-09 02z  (kept)
        _touch_medium(tmp_path, "20260507", cyc=0, fhr=66)   # 05-09 18z  (kept, == window_end)
        _touch_medium(tmp_path, "20260507", cyc=0, fhr=67)   # 05-09 19z  (drop, past end)
        _touch_medium(tmp_path, "20260507", cyc=0, fhr=120)  # 05-12 00z  (drop)

        proc = _proc(tmp_path)
        fc = proc._select_forecast_files(
            [datetime(2026, 5, 7), datetime(2026, 5, 6)],
            proc._glob_product_files(
                "analysis_assim", [datetime(2026, 5, 7), datetime(2026, 5, 6)]),
        )
        vts = [_nwm_valid_time(f) for f in fc]
        assert datetime(2026, 5, 9, 18) in vts            # window-end hour kept
        assert datetime(2026, 5, 9, 19) not in vts        # one hour past -> dropped
        assert datetime(2026, 5, 12, 0) not in vts        # far tail dropped
        assert max(vts) == datetime(2026, 5, 9, 18)       # window_end is the cap

    def test_explicit_short_range_product_single_glob(self, tmp_path):
        """nwm_product='short_range' is an explicit single-product override
        (no analysis blend)."""
        _touch_analysis(tmp_path, "20260507", cyc=0)
        _touch_short(tmp_path, "20260507", cyc=0, fhr=1)
        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        cfg.nwm_product = "short_range"
        rc = RiverConfig.from_sources_json(_sources(tmp_path))
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)
        files = proc.find_input_files()
        assert len(files) == 1
        assert "short_range" in files[0].name


# ---------------------------------------------------------------------------
# End-to-end: time-varying vsource.th
# ---------------------------------------------------------------------------

netCDF4 = pytest.importorskip("netCDF4")


def _write_channel_rt(path: Path, feature_ids, flows, valid_time: datetime):
    """Write a minimal NWM channel_rt netCDF the extractor can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("feature_id", len(feature_ids))
    v_fid = ds.createVariable("feature_id", "i8", ("feature_id",))
    v_q = ds.createVariable("streamflow", "f4", ("feature_id",))
    v_fid[:] = np.array(feature_ids, dtype=np.int64)
    v_q[:] = np.array(flows, dtype=np.float32)
    ds.model_output_valid_time = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
    ds.close()
    return path


class TestForecastBlendEndToEnd:
    def test_vsource_is_time_varying(self, tmp_path):
        """The flat-vsource regression: with analysis + forecast staged,
        vsource.th must VARY in time instead of repeating a single snapshot."""
        root = tmp_path
        cycle = datetime(2026, 5, 7, 0)
        start = cycle - timedelta(hours=6)  # model_t0 = 05-06 18z
        fid = [123, 456]

        # Analysis: hourly snapshots 18z..00z, each from its own cycle, with a
        # ramping discharge so even the nowcast portion varies.
        for h in range(0, 7):
            vt = start + timedelta(hours=h)
            d = root / f"nwm.{vt.strftime('%Y%m%d')}" / "analysis_assim"
            f = d / f"nwm.t{vt.hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc"
            _write_channel_rt(f, fid, [100.0 + 10 * h, 0.0], vt)

        # short_range f001..f018 from the 00z cycle (valid 01z..18z 05-07),
        # discharge keeps ramping so the forecast leg is clearly time-varying.
        for fhr in range(1, 19):
            vt = cycle + timedelta(hours=fhr)
            f = (root / "nwm.20260507" / "short_range"
                 / f"nwm.t00z.short_range.channel_rt.f{fhr:03d}.conus.nc")
            _write_channel_rt(f, fid, [200.0 + 5 * fhr, 0.0], vt)

        # medium_range_mem1 f003..f072 from the 00z cycle to fill the tail.
        for fhr in range(3, 73):
            vt = cycle + timedelta(hours=fhr)
            f = (root / "nwm.20260507" / "medium_range_mem1"
                 / f"nwm.t00z.medium_range.channel_rt_1.f{fhr:03d}.conus.nc")
            _write_channel_rt(f, fid, [400.0 + 2 * fhr, 0.0], vt)

        proc = NWMProcessor(
            ForcingConfig.for_secofs(pdy="20260507", cyc=0),
            root, tmp_path / "out",
            river_config=RiverConfig.from_sources_json(_sources(tmp_path)),
        )
        result = proc.process()
        assert result.success
        assert result.metadata["used_climatology"] is False
        # Many forecast files contributed, not just the handful of analysis.
        assert result.metadata["nwm_files_used"] > 10

        vsource = (tmp_path / "out" / "vsource.th").read_text().strip().splitlines()
        q = np.array([float(line.split()[1]) for line in vsource])
        # The core assertion: the source series is NOT flat.
        assert q.std() > 1.0, f"vsource.th is flat (std={q.std():.3g})"
        # And it spans the full window (>= ~54h of hourly rows).
        assert len(vsource) >= 55

    def test_aggregated_valid_time_filename_fallback(self, tmp_path):
        """Aggregated extractor (SECOFS-UFS path) must anchor on the filename
        valid time when a file lacks ``model_output_valid_time`` — otherwise a
        mixed analysis+forecast series would be mis-ordered by index."""
        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        rc = RiverConfig.from_sources_json(_sources(tmp_path))
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out", river_config=rc)
        # Two analysis files WITHOUT the valid-time attribute; filenames carry
        # tm01 (valid 23z prior day) and tm00 (valid 00z cycle). model_t0 =
        # cycle - 6h = 05-06 18z -> hours 5 and 6.
        d = tmp_path / "nwm.20260507" / "analysis_assim"
        d.mkdir(parents=True, exist_ok=True)
        for tm in (1, 0):
            f = d / f"nwm.t00z.analysis_assim.channel_rt.tm{tm:02d}.conus.nc"
            ds = netCDF4.Dataset(str(f), "w", format="NETCDF4")
            ds.createDimension("feature_id", 2)
            ds.createVariable("feature_id", "i8", ("feature_id",))[:] = [123, 456]
            ds.createVariable("streamflow", "f4", ("feature_id",))[:] = [50.0, 0.0]
            ds.close()  # no model_output_valid_time attribute
        files = sorted(d.glob("*.nc"), key=_nwm_valid_time)
        flows, times = proc._extract_streamflow_aggregated(files)
        assert times == [5.0, 6.0], f"expected filename-derived hours; got {times}"

    def test_pad_guard_warns_when_forecast_missing(self, tmp_path, caplog):
        """Only analysis staged (no forecast) → large pad → loud WARNING."""
        import logging
        cycle = datetime(2026, 5, 7, 0)
        start = cycle - timedelta(hours=6)
        for h in range(0, 7):
            vt = start + timedelta(hours=h)
            d = tmp_path / f"nwm.{vt.strftime('%Y%m%d')}" / "analysis_assim"
            f = d / f"nwm.t{vt.hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc"
            _write_channel_rt(f, [123, 456], [100.0, 0.0], vt)

        proc = NWMProcessor(
            ForcingConfig.for_secofs(pdy="20260507", cyc=0),
            tmp_path, tmp_path / "out",
            river_config=RiverConfig.from_sources_json(_sources(tmp_path)),
        )
        with caplog.at_level(logging.WARNING):
            result = proc.process()
        assert result.success
        assert any("FLAT" in r.message and "COMINnwm" in r.message
                   for r in caplog.records), "expected the flat-tail WARNING"
