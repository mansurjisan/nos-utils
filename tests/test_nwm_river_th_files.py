"""Tests for the SECOFS-UFS boundary-river (river.ctl) real-time NWM overlay.

Bug: ``NWMProcessor._write_river_th_files`` wrote schism_flux.th (the
open-boundary river discharge for Savannah/Cooper) from USGS *annual/daily
climatology*, held CONSTANT for the whole cycle. The pre-operational Fortran
system uses real-time river flow, so in low-flow seasons the UFS climatology
over-supplies the Savannah River 2.4-2.8x, causing a persistent up-river
water-level bias.

The fix samples per-station discharge from the same NWM channel_rt files
already opened for the interior vsource sources (mapped via the
``river.files.nwm_reach`` FIX file, e.g. ``secofs_ufs.nwm.reach.dat``), and
keeps the climatology formula as a graceful fallback when a reach isn't
mapped or isn't present in the staged NWM files.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nwm import NWMProcessor, RiverConfig

netCDF4 = pytest.importorskip("netCDF4")

# Two boundary stations mirroring SECOFS-UFS: Savannah River at Clyo
# (USGS 02198500, 6 grid-point column) and Cooper River near Moultrie Dam
# (6 different USGS id, 4 grid-point column). Q_Scale = 0.1667 on every
# column, matching the production river.ctl convention.
SAVANNAH_FEATURE_ID = 20104159
COOPER_FEATURE_ID = 9643431
Q_SCALE = 0.1667

RIVER_CTL_TEXT = """\
Section 1: USGS river stations
2 2 1.0    !! NIJ NRIVERS DELT
RiverID StationID NWS_ID Agency Q_min Q_max Q_mean T_min T_max T_mean Name
1 02198500 SAVGA1 USGS 10.0 500.0 65.0 5.0 25.0 15.0 "Savannah River at Clyo"
2 02172035 COOSC1 USGS 5.0 300.0 40.0 5.0 25.0 16.0 "Cooper River near Moultrie Dam"
Section 2: Grid node mappings
GRID_ID NODE_ID ELE_ID DIR FLAG RiverID_Q Q_Scale RiverID_T T_Scale Name
1 1001 1 1 3 1 0.1667 1 1.0 "Savannah 1"
2 1002 1 1 3 1 0.1667 1 1.0 "Savannah 2"
3 1003 1 1 3 1 0.1667 1 1.0 "Savannah 3"
4 1004 1 1 3 1 0.1667 1 1.0 "Savannah 4"
5 1005 1 1 3 1 0.1667 1 1.0 "Savannah 5"
6 1006 1 1 3 1 0.1667 1 1.0 "Savannah 6"
7 2001 1 1 3 2 0.1667 2 1.0 "Cooper 1"
8 2002 1 1 3 2 0.1667 2 1.0 "Cooper 2"
9 2003 1 1 3 2 0.1667 2 1.0 "Cooper 3"
10 2004 1 1 3 2 0.1667 2 1.0 "Cooper 4"
"""

REACH_DAT_TEXT = f"""\
REACH_ID FLAG
2
{SAVANNAH_FEATURE_ID} 1
{COOPER_FEATURE_ID} 1
"""

# The actual ops SECOFS shape: a single Section-1 station (Savannah) with
# its 6-node Section-2 column, mapped 1-1 against a reach.dat trimmed to
# one reach -- unlike RIVER_CTL_TEXT/REACH_DAT_TEXT above, which are a
# synthetic 2-station fixture that never exercises this shipped pairing.
RIVER_CTL_TEXT_ONE_STATION = """\
Section 1: USGS river stations
6 1 1.0    !! NIJ NRIVERS DELT
RiverID StationID NWS_ID Agency Q_min Q_max Q_mean T_min T_max T_mean Name
1 02198500 SAVGA1 USGS 10.0 500.0 65.0 5.0 25.0 15.0 "Savannah River at Clyo"
Section 2: Grid node mappings
GRID_ID NODE_ID ELE_ID DIR FLAG RiverID_Q Q_Scale RiverID_T T_Scale Name
1 1001 1 1 3 1 0.1667 1 1.0 "Savannah 1"
2 1002 1 1 3 1 0.1667 1 1.0 "Savannah 2"
3 1003 1 1 3 1 0.1667 1 1.0 "Savannah 3"
4 1004 1 1 3 1 0.1667 1 1.0 "Savannah 4"
5 1005 1 1 3 1 0.1667 1 1.0 "Savannah 5"
6 1006 1 1 3 1 0.1667 1 1.0 "Savannah 6"
"""

REACH_DAT_TEXT_ONE = f"""\
REACH_ID FLAG
1
{SAVANNAH_FEATURE_ID} 1
"""


def _write_channel_rt(path: Path, feature_ids, flows, valid_time: datetime) -> Path:
    """Write a minimal NWM channel_rt netCDF the extractor can read.

    A ``None`` entry in *flows* is written masked (fill value), mimicking a
    missing/masked NWM streamflow value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("feature_id", len(feature_ids))
    v_fid = ds.createVariable("feature_id", "i8", ("feature_id",))
    v_q = ds.createVariable("streamflow", "f4", ("feature_id",),
                            fill_value=-999900.0)
    v_fid[:] = np.array(feature_ids, dtype=np.int64)
    v_q[:] = np.ma.masked_invalid(np.array(
        [np.nan if f is None else f for f in flows], dtype=np.float32))
    ds.model_output_valid_time = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
    ds.close()
    return path


def _make_proc(tmp_path, phase="nowcast", cyc=12, pdy="20260401"):
    """NWMProcessor with cfg.pdy/cyc chosen so the nowcast window (cycle -
    6h .. cycle - 6h + 7h) stays inside a single calendar day, and the NWM
    root at ``tmp_path`` (also the FIX search-dir fallback used by
    ``_find_nwm_reach_file``)."""
    cfg = ForcingConfig(
        lon_min=-88.0, lon_max=-63.0, lat_min=17.0, lat_max=40.0,
        pdy=pdy, cyc=cyc, nowcast_hours=6, forecast_hours=48,
    )
    proc = NWMProcessor(cfg, tmp_path, tmp_path / "out",
                        river_config=None, phase=phase)
    proc.create_output_dir()
    return proc


def _ctl_cfg() -> RiverConfig:
    return RiverConfig._parse_river_ctl(RIVER_CTL_TEXT)


def _stage_analysis_files(root: Path, start: datetime, n_hours: int,
                          savannah_flows, cooper_flows) -> None:
    """Stage one analysis_assim tm00 file per hour [0, n_hours] with the
    two boundary feature_ids, following the pattern used elsewhere in this
    suite (test_nwm_forecast_blend.py) of keying each hourly snapshot as
    its own tm00 cycle so filename-derived dates stay simple."""
    for h in range(n_hours + 1):
        vt = start + timedelta(hours=h)
        d = root / f"nwm.{vt.strftime('%Y%m%d')}" / "analysis_assim"
        f = d / f"nwm.t{vt.hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc"
        _write_channel_rt(
            f, [SAVANNAH_FEATURE_ID, COOPER_FEATURE_ID],
            [savannah_flows[h], cooper_flows[h]], vt,
        )


class TestBoundaryRiverNWMOverlay:
    def test_flow_is_time_varying_when_reach_mapped_and_covered(self, tmp_path):
        """Both stations mapped + present in NWM files -> schism_flux.th
        must vary in time (not the old constant-climatology behavior)."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)

        start = proc._phase_start_time()  # cycle - nowcast_hours
        n_hours = 7  # nowcast(6) + river_th_extra_hours(1)
        savannah_flows = [50.0 + 10 * h for h in range(n_hours + 1)]  # ramps
        cooper_flows = [40.0 + 2 * h for h in range(n_hours + 1)]
        _stage_analysis_files(tmp_path, start, n_hours, savannah_flows, cooper_flows)

        ctl_cfg = _ctl_cfg()
        nwm_files = proc.find_input_files()
        assert nwm_files, "expected staged analysis_assim files to be found"

        out_files = proc._write_river_th_files([], ctl_cfg, nwm_files)
        assert len(out_files) == 3

        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = [[float(v) for v in line.split()] for line in flux_lines]
        rows = np.array(rows)  # (n_steps, 1 + n_riv)
        q_cols = rows[:, 1:]  # drop the time column

        # Savannah columns (0-5) and Cooper columns (6-9) must both vary.
        savannah_cols = q_cols[:, 0:6]
        cooper_cols = q_cols[:, 6:10]
        assert savannah_cols.std() > 0.0, "Savannah flux is flat (still climatology)"
        assert cooper_cols.std() > 0.0, "Cooper flux is flat (still climatology)"

        # All 6 Savannah columns must be identical to each other (single
        # station, Q_Scale applied uniformly) -- same for Cooper.
        for c in range(1, 6):
            assert np.allclose(savannah_cols[:, 0], savannah_cols[:, c])
        for c in range(1, 4):
            assert np.allclose(cooper_cols[:, 0], cooper_cols[:, c])

        # First row (t=0, hour 0) must reflect the NWM value at hour 0,
        # not the annual Q_mean (65.0) climatology.
        expected_t0 = -(savannah_flows[0] * Q_SCALE)
        assert rows[0, 1] == pytest.approx(expected_t0, abs=0.02)
        old_climatology_value = -(65.0 * Q_SCALE)
        assert rows[0, 1] != pytest.approx(old_climatology_value, abs=0.02)

    def test_fallback_when_reach_map_file_missing(self, tmp_path, caplog):
        """No nwm_reach FIX file on disk -> climatology fallback (constant
        Q for the whole run), with a clear warning."""
        proc = _make_proc(tmp_path)
        # Intentionally do NOT write nwm.reach.dat.

        start = proc._phase_start_time()
        n_hours = 7
        savannah_flows = [50.0 + 10 * h for h in range(n_hours + 1)]
        cooper_flows = [40.0 + 2 * h for h in range(n_hours + 1)]
        _stage_analysis_files(tmp_path, start, n_hours, savannah_flows, cooper_flows)

        ctl_cfg = _ctl_cfg()
        nwm_files = proc.find_input_files()
        assert nwm_files

        with caplog.at_level(logging.WARNING):
            out_files = proc._write_river_th_files([], ctl_cfg, nwm_files)
        assert len(out_files) == 3
        assert any("reach map FIX file not found" in r.message for r in caplog.records)

        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in flux_lines])
        q_cols = rows[:, 1:]
        # Climatology Q is constant across the whole run -> zero std per column.
        assert q_cols.std(axis=0).max() == pytest.approx(0.0, abs=1e-9)
        expected = -(65.0 * Q_SCALE)
        assert rows[0, 1] == pytest.approx(expected, abs=0.02)

    def test_fallback_when_reach_absent_from_nwm_files(self, tmp_path, caplog):
        """Reach map exists, but neither mapped feature_id shows up in the
        staged NWM files -> climatology fallback, with a clear warning."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)

        start = proc._phase_start_time()
        n_hours = 7
        # Stage NWM files carrying a DIFFERENT feature_id (not Savannah/Cooper).
        for h in range(n_hours + 1):
            vt = start + timedelta(hours=h)
            d = tmp_path / f"nwm.{vt.strftime('%Y%m%d')}" / "analysis_assim"
            f = d / f"nwm.t{vt.hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc"
            _write_channel_rt(f, [999999], [123.0], vt)

        ctl_cfg = _ctl_cfg()
        nwm_files = proc.find_input_files()
        assert nwm_files

        with caplog.at_level(logging.WARNING):
            out_files = proc._write_river_th_files([], ctl_cfg, nwm_files)
        assert len(out_files) == 3
        assert any("were found in the staged NWM" in r.message for r in caplog.records)

        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in flux_lines])
        q_cols = rows[:, 1:]
        assert q_cols.std(axis=0).max() == pytest.approx(0.0, abs=1e-9)

    def test_partial_coverage_warns_for_uncovered_station_only(self, tmp_path, caplog):
        """Savannah's reach is in the NWM files, Cooper's is not -> Savannah
        columns vary, Cooper columns stay at climatology, and the warning
        names only the uncovered station."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)

        start = proc._phase_start_time()
        n_hours = 7
        savannah_flows = [50.0 + 10 * h for h in range(n_hours + 1)]
        for h in range(n_hours + 1):
            vt = start + timedelta(hours=h)
            d = tmp_path / f"nwm.{vt.strftime('%Y%m%d')}" / "analysis_assim"
            f = d / f"nwm.t{vt.hour:02d}z.analysis_assim.channel_rt.tm00.conus.nc"
            # Only Savannah's feature_id is present in these files.
            _write_channel_rt(f, [SAVANNAH_FEATURE_ID], [savannah_flows[h]], vt)

        ctl_cfg = _ctl_cfg()
        nwm_files = proc.find_input_files()

        with caplog.at_level(logging.WARNING):
            out_files = proc._write_river_th_files([], ctl_cfg, nwm_files)
        assert len(out_files) == 3
        assert any("Cooper River near Moultrie Dam" in r.message
                   for r in caplog.records)

        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in flux_lines])
        savannah_cols = rows[:, 1:7]
        cooper_cols = rows[:, 7:11]
        assert savannah_cols.std() > 0.0, "Savannah should be NWM-driven (time-varying)"
        assert cooper_cols.std(axis=0).max() == pytest.approx(0.0, abs=1e-9), \
            "Cooper should still be constant climatology"

    def test_column_count_and_q_scale_distribution_preserved(self, tmp_path):
        """6-node Savannah / 4-node Cooper column layout and the 0.1667
        Q_Scale distribution must survive the NWM overlay unchanged."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)

        start = proc._phase_start_time()
        n_hours = 7
        savannah_flows = [100.0] * (n_hours + 1)
        cooper_flows = [80.0] * (n_hours + 1)
        _stage_analysis_files(tmp_path, start, n_hours, savannah_flows, cooper_flows)

        ctl_cfg = _ctl_cfg()
        assert ctl_cfg.n_rivers == 10
        assert ctl_cfg.q_scale == [Q_SCALE] * 10

        nwm_files = proc.find_input_files()
        proc._write_river_th_files([], ctl_cfg, nwm_files)

        out_dir = tmp_path / "out"
        for name in ("schism_flux.th", "schism_temp.th", "schism_salt.th"):
            lines = (out_dir / name).read_text().strip().splitlines()
            for line in lines:
                assert len(line.split()) == 1 + 10, f"{name}: expected 10 river columns"

        salt_lines = (out_dir / "schism_salt.th").read_text().strip().splitlines()
        salt_vals = [float(v) for v in salt_lines[0].split()[1:]]
        assert salt_vals == pytest.approx([0.005] * 10)

        flux_lines = (out_dir / "schism_flux.th").read_text().strip().splitlines()
        flux_row0 = [float(v) for v in flux_lines[0].split()[1:]]
        expected_savannah = -(100.0 * Q_SCALE)
        expected_cooper = -(80.0 * Q_SCALE)
        assert flux_row0[0:6] == pytest.approx([expected_savannah] * 6, abs=0.02)
        assert flux_row0[6:10] == pytest.approx([expected_cooper] * 4, abs=0.02)

    def test_legacy_ad_hoc_river_config_unaffected(self, tmp_path):
        """When river.ctl-style per-grid scaling isn't populated (ad-hoc
        RiverConfig, as used in older/unit tests), the NWM overlay must not
        engage -- output stays on the pre-existing clim_flows path."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)
        ctl_cfg = RiverConfig(
            feature_ids=[1, 2], node_indices=[10, 20],
            clim_flows=[42.0, 24.0],
        )
        out_files = proc._write_river_th_files([], ctl_cfg, nwm_files=[])
        assert len(out_files) == 3
        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        first_row = [float(v) for v in flux_lines[0].split()]
        assert first_row[1] == pytest.approx(-42.0, abs=0.001)
        assert first_row[2] == pytest.approx(-24.0, abs=0.001)

    def test_no_nwm_files_uses_climatology(self, tmp_path):
        """nwm_files=None/[] (NWM discovery failed upstream) -> climatology
        for every column, same as before this change."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)
        ctl_cfg = _ctl_cfg()

        out_files = proc._write_river_th_files([], ctl_cfg, nwm_files=[])
        assert len(out_files) == 3
        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in flux_lines])
        assert rows[:, 1:].std(axis=0).max() == pytest.approx(0.0, abs=1e-9)

    def test_fallback_when_station_count_mismatches_reach_file(self, tmp_path, caplog):
        """river.ctl declares 1 station but the reach file lists 2 reaches.
        The real deploy is a 1-1 pair (see
        test_flow_is_time_varying_for_real_secofs_one_station_one_reach_pair
        below) -- this only guards against accidental FIX drift (e.g. an
        un-trimmed reach.dat shipped alongside a trimmed river.ctl): mapping
        must be distrusted, constant climatology for every column plus a
        mismatch warning, never a positional guess."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)  # 2 reaches

        start = proc._phase_start_time()
        n_hours = 7
        flows = [50.0 + 10 * h for h in range(n_hours + 1)]
        _stage_analysis_files(tmp_path, start, n_hours, flows, flows)

        ctl_cfg = RiverConfig._parse_river_ctl(RIVER_CTL_TEXT_ONE_STATION)
        assert ctl_cfg.n_rivers == 6
        nwm_files = proc.find_input_files()

        with caplog.at_level(logging.WARNING):
            out_files = proc._write_river_th_files([], ctl_cfg, nwm_files)
        assert len(out_files) == 3
        assert any("station-count mismatch" in r.message for r in caplog.records)

        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in flux_lines])
        assert rows[:, 1:].std(axis=0).max() == pytest.approx(0.0, abs=1e-9)
        assert rows[0, 1] == pytest.approx(-(65.0 * Q_SCALE), abs=0.02)

    def test_flow_is_time_varying_for_real_secofs_one_station_one_reach_pair(self, tmp_path):
        """The actual ops SECOFS shape: 1 Section-1 station (Savannah)
        mapped 1-1 against a reach.dat trimmed to 1 reach -- unlike
        test_flow_is_time_varying_when_reach_mapped_and_covered above
        (synthetic 2-station ctl), this is the pair that is actually
        shipped in production. The overlay must be active: schism_flux.th
        time-varying on all 6 Savannah columns, tracking the staged
        real-time NWM values."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT_ONE)

        start = proc._phase_start_time()
        n_hours = 7
        savannah_flows = [50.0 + 10 * h for h in range(n_hours + 1)]
        cooper_flows = [40.0 + 2 * h for h in range(n_hours + 1)]  # unmapped, ignored
        _stage_analysis_files(tmp_path, start, n_hours, savannah_flows, cooper_flows)

        ctl_cfg = RiverConfig._parse_river_ctl(RIVER_CTL_TEXT_ONE_STATION)
        assert ctl_cfg.n_rivers == 6
        nwm_files = proc.find_input_files()
        assert nwm_files, "expected staged analysis_assim files to be found"

        out_files = proc._write_river_th_files([], ctl_cfg, nwm_files)
        assert len(out_files) == 3

        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in flux_lines])
        q_cols = rows[:, 1:]
        assert q_cols.shape[1] == 6

        assert q_cols.std(axis=0).min() > 1e-3, \
            "one or more Savannah columns are flat (still climatology)"
        old_climatology_value = -(65.0 * Q_SCALE)
        assert q_cols[0, 0] != pytest.approx(old_climatology_value, abs=0.02)

        dt = 120.0  # ForcingConfig.schism_dt default
        for h in (0, 3, 7):
            r = int(h * 3600 / dt)
            expected = -(savannah_flows[h] * Q_SCALE)
            assert q_cols[r, :] == pytest.approx([expected] * 6, abs=0.02)

    def test_flag_zero_first_reach_keeps_later_station_aligned(self, tmp_path, caplog):
        """A flag=0 first row must keep its positional slot (Fortran Ius
        semantics): Savannah (row 1, flag 0) falls back to climatology while
        Cooper (row 2, flag 1) still maps to ITS OWN reach, not Savannah's."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(
            f"REACH_ID FLAG\n2\n{SAVANNAH_FEATURE_ID} 0\n{COOPER_FEATURE_ID} 1\n"
        )

        start = proc._phase_start_time()
        n_hours = 7
        savannah_flows = [500.0] * (n_hours + 1)  # constant, distinctive
        cooper_flows = [45.0 + 2 * h for h in range(n_hours + 1)]  # ramps
        _stage_analysis_files(tmp_path, start, n_hours, savannah_flows, cooper_flows)

        ctl_cfg = _ctl_cfg()
        nwm_files = proc.find_input_files()
        with caplog.at_level(logging.INFO):
            proc._write_river_th_files([], ctl_cfg, nwm_files)
        assert any("flagged outside NWM domain" in r.message for r in caplog.records)

        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in flux_lines])
        savannah_cols = rows[:, 1:7]
        cooper_cols = rows[:, 7:11]
        # Savannah: climatology (annual Q_mean 65), NOT the staged 500 m3/s.
        assert savannah_cols.std(axis=0).max() == pytest.approx(0.0, abs=1e-9)
        assert rows[0, 1] == pytest.approx(-(65.0 * Q_SCALE), abs=0.02)
        # Cooper: time-varying from its own reach (a positional shift would
        # put Savannah's 500 m3/s here instead).
        assert cooper_cols.std() > 0.0
        assert rows[0, 7] == pytest.approx(-(cooper_flows[0] * Q_SCALE), abs=0.02)

    def test_masked_streamflow_value_falls_back_not_nan(self, tmp_path):
        """A masked NWM streamflow value (missing data) must not write NaN
        into schism_flux.th -- that hour falls back to the station's
        climatology Q while neighboring hours keep their NWM values."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)

        start = proc._phase_start_time()
        n_hours = 7
        savannah_flows = [50.0 + 10 * h for h in range(n_hours + 1)]
        cooper_flows = [45.0 + 2 * h for h in range(n_hours + 1)]
        savannah_flows[3] = None  # masked at hour 3
        _stage_analysis_files(tmp_path, start, n_hours, savannah_flows, cooper_flows)

        ctl_cfg = _ctl_cfg()
        nwm_files = proc.find_input_files()
        proc._write_river_th_files([], ctl_cfg, nwm_files)

        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in flux_lines])
        assert np.isfinite(rows).all(), "NaN leaked into schism_flux.th"

        dt = 120.0  # ForcingConfig.schism_dt default
        r3 = int(3 * 3600 / dt)
        r2 = int(2 * 3600 / dt)
        # Masked hour -> Savannah annual Q_mean (65.0), not NaN, not a
        # neighboring hour's NWM value.
        assert rows[r3, 1] == pytest.approx(-(65.0 * Q_SCALE), abs=0.02)
        # Neighboring hour still carries its NWM value (also pins the
        # hour->row anchoring mid-series, not just at t=0).
        assert rows[r2, 1] == pytest.approx(-(savannah_flows[2] * Q_SCALE), abs=0.02)
        # Cooper unaffected at the masked hour.
        assert rows[r3, 7] == pytest.approx(-(cooper_flows[3] * Q_SCALE), abs=0.02)

    def test_negative_streamflow_clamped_to_zero_flux(self, tmp_path):
        """A negative NWM streamflow value (NWM occasionally emits small
        negative streamflow) must clamp to zero Q -- schism_flux.th must
        never flip to an outflow (positive) sign at that hour."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)

        start = proc._phase_start_time()
        n_hours = 7
        savannah_flows = [50.0 + 10 * h for h in range(n_hours + 1)]
        cooper_flows = [45.0 + 2 * h for h in range(n_hours + 1)]
        savannah_flows[3] = -25.0  # negative at hour 3
        _stage_analysis_files(tmp_path, start, n_hours, savannah_flows, cooper_flows)

        ctl_cfg = _ctl_cfg()
        nwm_files = proc.find_input_files()
        proc._write_river_th_files([], ctl_cfg, nwm_files)

        flux_lines = (tmp_path / "out" / "schism_flux.th").read_text().strip().splitlines()
        rows = np.array([[float(v) for v in line.split()] for line in flux_lines])

        dt = 120.0  # ForcingConfig.schism_dt default
        r3 = int(3 * 3600 / dt)
        r2 = int(2 * 3600 / dt)
        # Negative Q clamps to 0 -> flux is exactly 0.0, never positive
        # (an unclamped negative Q would write a positive/outflow flux).
        assert rows[r3, 1] == 0.0
        # Neighboring hour still carries its (negated, inflow) NWM value.
        assert rows[r2, 1] == pytest.approx(-(savannah_flows[2] * Q_SCALE), abs=0.02)
        # Cooper unaffected at that hour.
        assert rows[r3, 7] == pytest.approx(-(cooper_flows[3] * Q_SCALE), abs=0.02)


class TestFindNwmReachFileConfiguredPath:
    """ForcingConfig.nwm_reach_file (river.files.nwm_reach) wiring into
    _find_nwm_reach_file."""

    def test_configured_absolute_path_is_used_not_basename_search(self, tmp_path):
        """A configured path must win over the hardcoded-basename search,
        even when a basename candidate also exists on disk."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)  # decoy basename match
        alt_dir = tmp_path / "alt_fix"
        alt_dir.mkdir()
        configured = alt_dir / "custom_reach_map.dat"
        configured.write_text(REACH_DAT_TEXT_ONE)
        proc.config.nwm_reach_file = configured

        assert proc._find_nwm_reach_file() == configured

    def test_unset_field_falls_back_to_basename_search(self, tmp_path):
        """No nwm_reach_file configured -> existing hardcoded-basename
        search behavior is preserved."""
        proc = _make_proc(tmp_path)
        assert proc.config.nwm_reach_file is None
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)

        assert proc._find_nwm_reach_file() == tmp_path / "nwm.reach.dat"

    def test_configured_path_missing_does_not_fall_back(self, tmp_path, caplog):
        """A configured path that doesn't exist must not silently fall back
        to the basename search -- an operator's typo should surface as a
        missing reach map (climatology fallback upstream), not accidentally
        resolve to an unrelated FIX file."""
        proc = _make_proc(tmp_path)
        (tmp_path / "nwm.reach.dat").write_text(REACH_DAT_TEXT)  # would match basename search
        proc.config.nwm_reach_file = tmp_path / "does_not_exist.dat"

        with caplog.at_level(logging.WARNING):
            found = proc._find_nwm_reach_file()
        assert found is None
        assert any("river.files.nwm_reach" in r.message for r in caplog.records)
