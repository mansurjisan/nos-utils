"""RTOFS 3dz tile selection when ops stages multiple regions side by side.

Ops publishes "alaska", "US_east" and "US_west" 3dz tiles for every valid
time, with no region in the discovery glob
(rtofs_glo_3dz_*_6hrly_hvr_*.nc matches all three) and no region in the
valid-time dedup's sort key. The dedup keeps only the first file per valid
time, and ties break on the original glob order -- plain alphabetical
filename sort, since Python's sort is stable.

"US_east" < "alaska" as plain strings (ord('U')=85 < ord('a')=97), so the
dedup silently prefers US_east regardless of which tile the domain actually
needs. That happens to be correct for SECOFS/STOFS-3D-ATL and wrong for
STOFS-3D-AK, whose domain has zero overlap with the US_east tile.

Reproduced here with the exact three-region file set WCOSS2 actually stages
(confirmed via `ls` on /lfs/h1/ops/prod/com/rtofs/v2.5/rtofs.<date>).
"""
from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.rtofs import RTOFSProcessor

_REGIONS = ("alaska", "US_east", "US_west")
_FHRS = ("f006", "f012", "f018", "f024", "n006", "n012")


@pytest.fixture(autouse=True)
def _no_size_floor(monkeypatch):
    """Every test in this file only exercises file DISCOVERY (globbing,
    dedup, region filtering) -- never the netCDF content. Writing a real
    250 MB / 200 MB file per fixture instance just to clear
    validate_file_size's threshold check costs ~5.7 GB per test and ~34 GB
    across the file for no behavioural difference: threshold <= 0 makes
    validate_file_size (forcing/base.py) return True unconditionally, so a
    0-byte file clears it exactly the same as a 250 MB one.
    """
    monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_2D", 0)
    monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_3D", 0)


def _stage_three_region_rtofs(tmp_path, pdy="20260729"):
    """The real WCOSS2 layout: three 3dz tiles per valid time, one 2ds set."""
    rtofs_dir = tmp_path / f"rtofs.{pdy}"
    rtofs_dir.mkdir()
    for fhr in _FHRS:
        for region in _REGIONS:
            (rtofs_dir / f"rtofs_glo_3dz_{fhr}_6hrly_hvr_{region}.nc").touch()
        (rtofs_dir / f"rtofs_glo_2ds_{fhr}_diag.nc").touch()
    return tmp_path


def _tiles_selected(files_3d):
    return sorted({f.name.rsplit("hvr_", 1)[-1].split(".")[0] for f in files_3d})


class TestRegionSelection:
    def test_unfiltered_reproduces_the_bug(self, tmp_path):
        """The documented current behaviour: unset region means US_east wins
        by alphabetical accident, silently discarding the other two tiles as
        if they were duplicates."""
        root = _stage_three_region_rtofs(tmp_path)
        cfg = ForcingConfig(
            lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
            pdy="20260730", cyc=0,
        )
        assert cfg.rtofs_3d_region is None
        proc = RTOFSProcessor(cfg, root, root)
        _, files_3d = proc.find_input_files_by_type()
        assert _tiles_selected(files_3d) == ["US_east"]

    def test_alaska_region_selects_the_alaska_tile(self, tmp_path):
        root = _stage_three_region_rtofs(tmp_path)
        cfg = ForcingConfig(
            lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
            pdy="20260730", cyc=0, rtofs_3d_region="alaska",
        )
        proc = RTOFSProcessor(cfg, root, root)
        _, files_3d = proc.find_input_files_by_type()
        assert _tiles_selected(files_3d) == ["alaska"]
        assert len(files_3d) > 0

    def test_explicit_us_east_matches_the_unfiltered_default(self, tmp_path):
        """SECOFS/STOFS-3D-ATL declaring rtofs_3d_region="US_east" must pick
        exactly the files the unfiltered (buggy-by-luck) path already picks
        -- this is what makes their correctness explicit rather than
        accidental, with zero behaviour change."""
        root = _stage_three_region_rtofs(tmp_path)
        unfiltered = ForcingConfig(
            lon_min=-88.0, lon_max=-63.0, lat_min=17.0, lat_max=40.0,
            pdy="20260730", cyc=0,
        )
        explicit = ForcingConfig(
            lon_min=-88.0, lon_max=-63.0, lat_min=17.0, lat_max=40.0,
            pdy="20260730", cyc=0, rtofs_3d_region="US_east",
        )
        files_a = RTOFSProcessor(unfiltered, root, root).find_input_files_by_type()[1]
        files_b = RTOFSProcessor(explicit, root, root).find_input_files_by_type()[1]
        assert sorted(f.name for f in files_a) == sorted(f.name for f in files_b)

    def test_2d_discovery_is_unaffected(self, tmp_path):
        """SSH/elevation reads the genuinely global 2ds product, which has
        no region suffix at all -- the region field must not touch it.

        Compares against the unfiltered path rather than a fixed count:
        n006/f006 (etc.) land on the same valid time here, so the existing
        nowcast/forecast dedup legitimately drops half of them -- that is
        unrelated to the region fix and not what this test is checking.
        """
        root = _stage_three_region_rtofs(tmp_path)
        unfiltered = ForcingConfig(
            lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
            pdy="20260730", cyc=0,
        )
        filtered = ForcingConfig(
            lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
            pdy="20260730", cyc=0, rtofs_3d_region="alaska",
        )
        files_2d_a, _ = RTOFSProcessor(unfiltered, root, root).find_input_files_by_type()
        files_2d_b, _ = RTOFSProcessor(filtered, root, root).find_input_files_by_type()
        assert len(files_2d_a) > 0
        assert sorted(f.name for f in files_2d_a) == sorted(f.name for f in files_2d_b)

    def test_wrong_region_name_finds_nothing_rather_than_falling_back(self, tmp_path):
        """A typo'd or unstaged region must not silently degrade to the
        unfiltered (US_east-by-accident) behaviour -- discovery must return
        zero files. See TestPartialFailurePropagates below for whether that
        actually stops process() from reporting success."""
        root = _stage_three_region_rtofs(tmp_path)
        cfg = ForcingConfig(
            lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
            pdy="20260730", cyc=0, rtofs_3d_region="not_a_real_tile",
        )
        proc = RTOFSProcessor(cfg, root, root)
        _, files_3d = proc.find_input_files_by_type()
        assert files_3d == []

    def test_stofs_mode_path_also_respects_region(self, tmp_path):
        """_process_stofs calls the same find_input_files_by_type before
        branching on is_stofs_mode, so a ROI-configured system (STOFS-3D-ATL)
        is exposed to the same tile-selection bug whenever it falls through
        to the Python path -- confirm the region filter reaches it too."""
        root = _stage_three_region_rtofs(tmp_path)
        cfg = ForcingConfig(
            lon_min=-98.5035, lon_max=-52.4867, lat_min=7.347, lat_max=52.5904,
            pdy="20260730", cyc=0,
            obc_roi_2d={"x1": 0, "x2": 1, "y1": 0, "y2": 1},
            rtofs_3d_region="US_east",
        )
        assert RTOFSProcessor(cfg, root, root).is_stofs_mode
        _, files_3d = RTOFSProcessor(cfg, root, root).find_input_files_by_type()
        assert _tiles_selected(files_3d) == ["US_east"]


def _with_fake_grid(proc):
    """Bypass hgrid.gr3 parsing: _load_grid only needs to populate boundary
    node arrays, and every test here is about RTOFS file selection, not
    grid I/O. Real coordinates so a real interpolation attempt (if reached)
    would not itself explode.
    """
    import numpy as np
    proc._bnd_lons = np.array([160.0, 165.0])
    proc._bnd_lats = np.array([55.0, 56.0])
    proc._bnd_depths = np.array([50.0, 50.0])
    proc._bnd_ids = np.array([1, 2])
    return proc


class TestPartialFailurePropagates:
    """discovery-level correctness (above) is necessary but not sufficient:
    process() has its own success rule -- `len(output_files) > 0` -- that a
    missing 3D tile does not trip on its own, because a 2D-only result
    (elev2D.th.nc, no TEM_3D/SAL_3D/uv3D) already satisfies it. Confirmed by
    reading rtofs.py directly: `if files_3d:` guards the ONLY place 3D
    output is added to output_files, and nothing downstream checks
    metadata["n_3d_files"]. These test process() itself, not discovery, so
    they only pass if the explicit region + empty files_3d + non-empty
    files_2d combination hard-fails before reaching that success
    computation. Uses touch()-only files: this path returns before
    attempting to open any of them.
    """

    def test_secofs_mode_fails_when_region_set_and_3d_missing(self, tmp_path):
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()
        # a real 3D file exists, but not for the requested region -- this is
        # the exact shape of a typo'd or not-yet-staged tile name
        (rtofs_dir / "rtofs_glo_3dz_f006_6hrly_hvr_alaska.nc").touch()

        cfg = ForcingConfig(
            lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
            pdy="20260730", cyc=0, rtofs_3d_region="not_a_real_tile",
        )
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))
        result = proc.process()
        assert result.success is False
        assert result.output_files == []
        assert any("not_a_real_tile" in e for e in result.errors)

    def test_secofs_mode_still_succeeds_2d_only_when_region_unset(self, tmp_path):
        """The pre-existing, unrelated leniency this fix must NOT remove:
        a system that never declared rtofs_3d_region keeps today's
        behaviour exactly, including a legitimate 2D-only cycle."""
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()

        cfg = ForcingConfig(
            lon_min=-88.0, lon_max=-63.0, lat_min=17.0, lat_max=40.0,
            pdy="20260730", cyc=0,
        )
        assert cfg.rtofs_3d_region is None
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))
        result = proc.process()
        # _process_2d will fail on the empty touch()'d file (not real
        # netCDF) -- that is a SEPARATE, pre-existing failure mode. The
        # point here is only that the function reaches that attempt at all,
        # i.e. is not short-circuited by the new region guard.
        assert "Cannot load boundary nodes" not in " ".join(result.errors)
        assert not any("rtofs_3d_region" in e for e in result.errors)

    def test_stofs_mode_fails_when_region_set_and_3d_missing(self, tmp_path):
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()
        (rtofs_dir / "rtofs_glo_3dz_f006_6hrly_hvr_alaska.nc").touch()

        cfg = ForcingConfig(
            lon_min=-98.5035, lon_max=-52.4867, lat_min=7.347, lat_max=52.5904,
            pdy="20260730", cyc=0,
            obc_roi_2d={"x1": 0, "x2": 1, "y1": 0, "y2": 1},
            rtofs_3d_region="not_a_real_tile",
        )
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path)
        assert proc.is_stofs_mode
        result = proc.process()
        assert result.success is False
        assert result.output_files == []
        assert any("not_a_real_tile" in e for e in result.errors)


class TestSplitDatePartialSuccess:
    """The scenario a second review round caught in the first version of
    the M1 fix: PDY-1 has ONLY 2-D data, PDY-2 has ONLY the requested 3-D
    tile. The date-fallback loop's job is to keep searching for a region
    match -- it must NOT reward that search by pairing the 3-D it finds on
    PDY-2 with nothing, while discarding the perfectly good 2-D data that
    was sitting on PDY-1.

    That pairing is fatal for a domain like STOFS-3D-AK: both open
    boundaries are elevation-forced (parm/systems/stofs_3d_ak_ufs.yaml),
    so elev2D.th.nc is mandatory, and `if files_3d:` runs independently of
    `files_2d` in _process_secofs/_process_stofs -- a 3-D-only pick would
    have produced TEM_3D.th.nc/SAL_3D.th.nc/uv3D.th.nc, no elev2D.th.nc,
    and success=True.
    """

    def test_discovery_keeps_2d_and_drops_the_mismatched_3d(self, tmp_path, monkeypatch):
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_2D", 0)
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_3D", 0)
        d1 = tmp_path / "rtofs.20260729"
        d1.mkdir()
        (d1 / "rtofs_glo_2ds_f006_diag.nc").touch()
        d2 = tmp_path / "rtofs.20260728"
        d2.mkdir()
        (d2 / "rtofs_glo_3dz_f006_6hrly_hvr_alaska.nc").touch()

        cfg = ForcingConfig(lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
                            pdy="20260730", cyc=0, rtofs_3d_region="alaska")
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path)
        files_2d, files_3d = proc.find_input_files_by_type()

        assert len(files_2d) == 1, "PDY-1's 2D data must survive"
        assert files_3d == [], (
            "PDY-2's 3D must NOT be paired with PDY-1's 2D across dates -- "
            "discard it rather than mixing cycle dates"
        )
        assert proc._rtofs_cycle_date.strftime("%Y%m%d") == "20260729"

    def test_process_does_not_report_success_from_3d_alone(self, tmp_path, monkeypatch):
        """End to end: even when 3D processing would itself SUCCEED (not
        just when discovery finds nothing), the step overall must not,
        because no elev2D.th.nc was ever produced. _process_3d is
        monkeypatched to return a fake path -- simulating a real, valid
        RTOFS 3dz file -- so this proves the fix acts at discovery, not by
        coincidentally failing on the dummy files' unparsable content."""
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_2D", 0)
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_3D", 0)
        d1 = tmp_path / "rtofs.20260729"
        d1.mkdir()
        (d1 / "rtofs_glo_2ds_f006_diag.nc").touch()
        d2 = tmp_path / "rtofs.20260728"
        d2.mkdir()
        (d2 / "rtofs_glo_3dz_f006_6hrly_hvr_alaska.nc").touch()

        cfg = ForcingConfig(lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
                            pdy="20260730", cyc=0, rtofs_3d_region="alaska")
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))
        fake_3d_outputs = [tmp_path / "TEM_3D.th.nc", tmp_path / "SAL_3D.th.nc",
                          tmp_path / "uv3D.th.nc"]
        monkeypatch.setattr(RTOFSProcessor, "_process_3d",
                            lambda self, files: fake_3d_outputs)

        result = proc.process()
        assert result.success is False, (
            "must not report success from 3D output alone when the "
            "matching 2D (elev2D.th.nc) was never produced"
        )
        assert result.output_files == []
        assert not any("TEM_3D" in str(f) for f in result.output_files)


class TestPartialProcessingSuccessBlocked:
    """A fourth review round found the deeper version of the bug
    TestSplitDatePartialSuccess covers above: even when BOTH 2-D and 3-D
    files survive discovery paired to the same cycle date, they are still
    processed independently -- `if files_2d: ...` and `if files_3d: ...`
    run one after another with no shared outcome, so
    `len(output_files) > 0` alone cannot tell "both halves produced" from
    "exactly one half produced, the other silently empty or failed".

    Two distinct ways that split happens even after date-pairing fixed the
    discovery-level version:
      1. _sort_and_dedup applies the phase time-window filter separately
         per type (rtofs.py, in _sort_and_dedup), AFTER the "both present"
         check at discovery. A file set that had both types when picked
         can still end up with one emptied by the window.
      2. _process_2d/_process_3d (or the STOFS Fortran exe) can each fail
         independently on bad content, regardless of how many files were
         found.

    These monkeypatch the processing methods themselves (not the file
    content) to isolate the success-condition bug from the unrelated,
    pre-existing question of whether bogus touch()'d files parse -- the
    same pattern TestSplitDatePartialSuccess uses above.
    """

    def test_phase_window_leaves_3d_only_still_fails(self, tmp_path, monkeypatch):
        """Reproduces the reviewer's Case 1: a single date has both types
        at discovery, but the phase window (self.phase is not None) keeps
        only the 3-D file because the 2-D file's valid time falls before
        the window starts."""
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        # n006 -> valid 2026-07-29 06:00, before t_start=2026-07-29 12:00
        (rtofs_dir / "rtofs_glo_2ds_n006_diag.nc").touch()
        # f024 -> valid 2026-07-30 00:00, inside the window
        (rtofs_dir / "rtofs_glo_3dz_f024_6hrly_hvr_alaska.nc").touch()

        cfg = ForcingConfig(lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
                            pdy="20260730", cyc=0, rtofs_3d_region="alaska",
                            nowcast_hours=6, forecast_hours=48)
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path, phase="nowcast"))

        files_2d, files_3d = proc.find_input_files_by_type()
        assert files_2d == [], "the 2D file must be filtered out by the phase window"
        assert len(files_3d) == 1, "the 3D file's valid time is inside the window"

        fake_3d = [tmp_path / "TEM_3D.th.nc", tmp_path / "SAL_3D.th.nc", tmp_path / "uv3D.th.nc"]
        monkeypatch.setattr(RTOFSProcessor, "_process_3d", lambda self, files: fake_3d)

        result = proc.process()
        assert result.success is False, (
            "phase-window filtering emptied the 2D side after discovery "
            "found both -- must not report success from 3D alone"
        )
        assert result.output_files == []

    def test_2d_processing_failure_with_3d_success_still_fails(self, tmp_path, monkeypatch):
        """_process_2d returning None (a real processing failure, e.g. no
        ocean points in the interpolation domain) must not be papered over
        by a successful _process_3d."""
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()
        (rtofs_dir / "rtofs_glo_3dz_f006_6hrly_hvr_alaska.nc").touch()

        cfg = ForcingConfig(lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
                            pdy="20260730", cyc=0, rtofs_3d_region="alaska")
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))

        monkeypatch.setattr(RTOFSProcessor, "_process_2d", lambda self, files: None)
        fake_3d = [tmp_path / "TEM_3D.th.nc"]
        monkeypatch.setattr(RTOFSProcessor, "_process_3d", lambda self, files: fake_3d)

        result = proc.process()
        assert result.success is False, (
            "2D processing failed outright -- must not report success "
            "because 3D alone produced output"
        )
        assert result.output_files == []

    def test_3d_processing_failure_with_2d_success_still_fails(self, tmp_path, monkeypatch):
        """_process_3d returning [] (a real processing failure) must not
        be papered over by a successful _process_2d."""
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()
        (rtofs_dir / "rtofs_glo_3dz_f006_6hrly_hvr_alaska.nc").touch()

        cfg = ForcingConfig(lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
                            pdy="20260730", cyc=0, rtofs_3d_region="alaska")
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))

        fake_2d = tmp_path / "elev2D.th.nc"
        monkeypatch.setattr(RTOFSProcessor, "_process_2d", lambda self, files: fake_2d)
        monkeypatch.setattr(RTOFSProcessor, "_process_3d", lambda self, files: [])

        result = proc.process()
        assert result.success is False, (
            "3D processing failed outright -- must not report success "
            "because 2D alone produced output"
        )
        assert result.output_files == []

    def test_stofs_mode_python_fallback_requires_both(self, tmp_path, monkeypatch):
        """The STOFS branch's Python fallback (_call_fortran_gen_3dth
        unavailable) mirrors _process_secofs's own if-files_2d/if-files_3d
        structure -- confirm it is held to the same both-required rule."""
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()
        (rtofs_dir / "rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc").touch()

        cfg = ForcingConfig(lon_min=-98.5035, lon_max=-52.4867, lat_min=7.347, lat_max=52.5904,
                            pdy="20260730", cyc=0,
                            obc_roi_2d={"x1": 0, "x2": 1, "y1": 0, "y2": 1},
                            rtofs_3d_region="US_east")
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))
        assert proc.is_stofs_mode

        monkeypatch.setattr(RTOFSProcessor, "_stofs_prepare_ssh",
                            lambda self, files, work_dir: work_dir / "SSH_1.nc")
        monkeypatch.setattr(RTOFSProcessor, "_stofs_prepare_tsuv",
                            lambda self, files, work_dir: work_dir / "TSUV_1.nc")
        monkeypatch.setattr(RTOFSProcessor, "_call_fortran_gen_3dth",
                            lambda self, work_dir, ssh_path, tsuv_path: False)
        monkeypatch.setattr(RTOFSProcessor, "_process_2d",
                            lambda self, files: tmp_path / "elev2D.th.nc")
        monkeypatch.setattr(RTOFSProcessor, "_process_3d", lambda self, files: [])

        result = proc.process()
        assert result.success is False, (
            "STOFS-mode Python fallback: 3D failed outright -- must not "
            "report success because 2D alone produced output"
        )
        assert result.output_files == []

    def test_stofs_mode_fortran_partial_output_still_fails(self, tmp_path, monkeypatch):
        """The Fortran path copies whichever of the four filenames exist in
        work_dir independently of each other (see the `for fname in [...]:
        if src.exists()` loop in _process_stofs) -- confirm a Fortran run
        that only wrote elev2D.th.nc (e.g. it crashed partway through the
        3D interpolation but had already written the SSH-only output) is
        still held to the both-required rule."""
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()
        (rtofs_dir / "rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc").touch()

        cfg = ForcingConfig(lon_min=-98.5035, lon_max=-52.4867, lat_min=7.347, lat_max=52.5904,
                            pdy="20260730", cyc=0,
                            obc_roi_2d={"x1": 0, "x2": 1, "y1": 0, "y2": 1},
                            rtofs_3d_region="US_east")
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))

        monkeypatch.setattr(RTOFSProcessor, "_stofs_prepare_ssh",
                            lambda self, files, work_dir: work_dir / "SSH_1.nc")
        monkeypatch.setattr(RTOFSProcessor, "_stofs_prepare_tsuv",
                            lambda self, files, work_dir: work_dir / "TSUV_1.nc")

        def _fake_fortran(self, work_dir, ssh_path, tsuv_path):
            (work_dir / "elev2D.th.nc").touch()
            return True

        monkeypatch.setattr(RTOFSProcessor, "_call_fortran_gen_3dth", _fake_fortran)

        result = proc.process()
        assert result.success is False, (
            "Fortran wrote only elev2D.th.nc -- must not report success "
            "without the 3D boundary files"
        )
        assert result.output_files == []


class TestIncomplete3DArtifactBlocked:
    """A fifth review round found a narrower version of the same class of
    bug: TestPartialProcessingSuccessBlocked above checks "both types
    present at all", but _process_3d (SECOFS Python path, and the STOFS
    Python fallback that reuses it) writes TEM_3D.th.nc and SAL_3D.th.nc on
    two INDEPENDENT conditions (`if all_temp:` / `if all_salt:`) -- one
    RTOFS 3dz file missing the "salinity" variable produces temperature
    with no salinity, and the old `obc3d_ok = bool(obc_files)` treated that
    nonempty-but-incomplete list as a complete 3D boundary. Alaska declares
    isatype=4, so a missing SAL_3D.th.nc is not a valid boundary set.

    The Fortran wrapper had the same shape of bug one level up: its own
    completeness check was `len(found) >= 3` -- any three of the four
    expected files, which is satisfied by elev2D+TEM_3D+uv3D just as
    happily as by the intended "uv3D missing" case.
    """

    def test_secofs_temperature_without_salinity_still_fails(self, tmp_path, monkeypatch):
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()
        (rtofs_dir / "rtofs_glo_3dz_f006_6hrly_hvr_alaska.nc").touch()

        cfg = ForcingConfig(lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
                            pdy="20260730", cyc=0, rtofs_3d_region="alaska")
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))

        fake_2d = tmp_path / "elev2D.th.nc"
        monkeypatch.setattr(RTOFSProcessor, "_process_2d", lambda self, files: fake_2d)
        # Simulates a source RTOFS file with a temperature variable but no
        # salinity variable: _process_3d would return [TEM_3D.th.nc] only.
        monkeypatch.setattr(RTOFSProcessor, "_process_3d",
                            lambda self, files: [tmp_path / "TEM_3D.th.nc"])

        result = proc.process()
        assert result.success is False, (
            "TEM_3D.th.nc alone is not a complete 3D boundary -- "
            "SAL_3D.th.nc is missing"
        )
        assert result.output_files == []
        assert any("SAL_3D.th.nc" in e for e in result.errors)

    def test_stofs_python_fallback_temperature_without_salinity_still_fails(self, tmp_path, monkeypatch):
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()
        (rtofs_dir / "rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc").touch()

        cfg = ForcingConfig(lon_min=-98.5035, lon_max=-52.4867, lat_min=7.347, lat_max=52.5904,
                            pdy="20260730", cyc=0,
                            obc_roi_2d={"x1": 0, "x2": 1, "y1": 0, "y2": 1},
                            rtofs_3d_region="US_east")
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))

        monkeypatch.setattr(RTOFSProcessor, "_stofs_prepare_ssh",
                            lambda self, files, work_dir: work_dir / "SSH_1.nc")
        monkeypatch.setattr(RTOFSProcessor, "_stofs_prepare_tsuv",
                            lambda self, files, work_dir: work_dir / "TSUV_1.nc")
        monkeypatch.setattr(RTOFSProcessor, "_call_fortran_gen_3dth",
                            lambda self, work_dir, ssh_path, tsuv_path: False)
        monkeypatch.setattr(RTOFSProcessor, "_process_2d",
                            lambda self, files: tmp_path / "elev2D.th.nc")
        monkeypatch.setattr(RTOFSProcessor, "_process_3d",
                            lambda self, files: [tmp_path / "TEM_3D.th.nc"])

        result = proc.process()
        assert result.success is False, (
            "STOFS-mode Python fallback: TEM_3D.th.nc alone is not a "
            "complete 3D boundary -- SAL_3D.th.nc is missing"
        )
        assert result.output_files == []
        assert any("SAL_3D.th.nc" in e for e in result.errors)

    def test_stofs_fortran_temperature_without_salinity_still_fails(self, tmp_path, monkeypatch):
        """Reproduces the reviewer's exact STOFS case: elev2D.th.nc,
        TEM_3D.th.nc and uv3D.th.nc all exist in work_dir, SAL_3D.th.nc
        does not -- three of the four expected files, the case the old
        `len(found) >= 3` inside _call_fortran_gen_3dth would have called
        complete."""
        rtofs_dir = tmp_path / "rtofs.20260729"
        rtofs_dir.mkdir()
        (rtofs_dir / "rtofs_glo_2ds_f006_diag.nc").touch()
        (rtofs_dir / "rtofs_glo_3dz_f006_6hrly_hvr_US_east.nc").touch()

        cfg = ForcingConfig(lon_min=-98.5035, lon_max=-52.4867, lat_min=7.347, lat_max=52.5904,
                            pdy="20260730", cyc=0,
                            obc_roi_2d={"x1": 0, "x2": 1, "y1": 0, "y2": 1},
                            rtofs_3d_region="US_east")
        proc = _with_fake_grid(RTOFSProcessor(cfg, tmp_path, tmp_path))

        monkeypatch.setattr(RTOFSProcessor, "_stofs_prepare_ssh",
                            lambda self, files, work_dir: work_dir / "SSH_1.nc")
        monkeypatch.setattr(RTOFSProcessor, "_stofs_prepare_tsuv",
                            lambda self, files, work_dir: work_dir / "TSUV_1.nc")

        def _fake_fortran(self, work_dir, ssh_path, tsuv_path):
            (work_dir / "elev2D.th.nc").touch()
            (work_dir / "TEM_3D.th.nc").touch()
            (work_dir / "uv3D.th.nc").touch()
            # SAL_3D.th.nc deliberately not written
            return True

        monkeypatch.setattr(RTOFSProcessor, "_call_fortran_gen_3dth", _fake_fortran)

        result = proc.process()
        assert result.success is False, (
            "elev2D+TEM_3D+uv3D (3 of 4 expected files) must not count as "
            "complete when the missing one is SAL_3D.th.nc"
        )
        assert result.output_files == []
        assert any("SAL_3D.th.nc" in e for e in result.errors)

    def test_fortran_helper_rejects_missing_salinity_even_with_three_files(self, tmp_path, monkeypatch):
        """Direct unit coverage of _call_fortran_gen_3dth's own output
        verification (not mocked away, unlike the process()-level tests
        above): a real subprocess run that writes exactly 3 of the 4
        expected files, omitting SAL_3D.th.nc, must be rejected."""
        exec_dir = tmp_path / "exec"
        exec_dir.mkdir()
        script = exec_dir / "stofs_3d_atl_gen_3Dth_from_hycom"
        script.write_text("#!/bin/sh\ntouch elev2D.th.nc TEM_3D.th.nc uv3D.th.nc\n")
        script.chmod(0o755)
        monkeypatch.setenv("EXECstofs3d", str(exec_dir))

        cfg = ForcingConfig(lon_min=-98.5035, lon_max=-52.4867, lat_min=7.347, lat_max=52.5904,
                            pdy="20260730", cyc=0,
                            obc_roi_2d={"x1": 0, "x2": 1, "y1": 0, "y2": 1})
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path)
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        assert proc._call_fortran_gen_3dth(work_dir, None, None) is False

    def test_fortran_helper_still_allows_missing_uv3d(self, tmp_path, monkeypatch):
        """The complement of the test above: uv3D.th.nc remains the one
        genuinely optional file -- elev2D + TEM_3D + SAL_3D must still be
        accepted as complete."""
        exec_dir = tmp_path / "exec"
        exec_dir.mkdir()
        script = exec_dir / "stofs_3d_atl_gen_3Dth_from_hycom"
        script.write_text("#!/bin/sh\ntouch elev2D.th.nc TEM_3D.th.nc SAL_3D.th.nc\n")
        script.chmod(0o755)
        monkeypatch.setenv("EXECstofs3d", str(exec_dir))

        cfg = ForcingConfig(lon_min=-98.5035, lon_max=-52.4867, lat_min=7.347, lat_max=52.5904,
                            pdy="20260730", cyc=0,
                            obc_roi_2d={"x1": 0, "x2": 1, "y1": 0, "y2": 1})
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path)
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        assert proc._call_fortran_gen_3dth(work_dir, None, None) is True


class TestFactoryDefaults:
    """The production factories no longer rely on alphabetical luck."""

    def test_secofs_pins_us_east(self):
        assert ForcingConfig.for_secofs(pdy="20260730", cyc=0).rtofs_3d_region == "US_east"

    def test_secofs_ufs_pins_us_east(self):
        assert ForcingConfig.for_secofs_ufs(pdy="20260730", cyc=0).rtofs_3d_region == "US_east"

    def test_stofs_3d_atl_pins_us_east(self):
        assert ForcingConfig.for_stofs_3d_atl(pdy="20260730", cyc=0).rtofs_3d_region == "US_east"

    def test_stofs_3d_atl_ufs_pins_us_east(self):
        assert ForcingConfig.for_stofs_3d_atl_ufs(pdy="20260730", cyc=0).rtofs_3d_region == "US_east"


class TestYamlWiring:
    def test_rtofs_3d_region_read_from_yaml(self, tmp_path):
        yml = tmp_path / "cfg.yaml"
        yml.write_text(
            "system:\n  name: x\n  prefix: x\n"
            "grid:\n  domain: {lon_min: 156.0, lon_max: 204.0, lat_min: 48.5, lat_max: 67.0}\n"
            "  files: {horizontal: x.hgrid.gr3}\n"
            "model:\n  physics: {dt: 45.0}\n  run: {nowcast_hours: 6, forecast_hours: 48}\n"
            "forcing:\n  ocean:\n    rtofs_3d_region: alaska\n"
        )
        cfg = ForcingConfig.from_yaml(yml)
        assert cfg.rtofs_3d_region == "alaska"

    def test_unset_in_yaml_stays_none(self, tmp_path):
        yml = tmp_path / "cfg.yaml"
        yml.write_text(
            "system:\n  name: x\n  prefix: x\n"
            "grid:\n  domain: {lon_min: -88.0, lon_max: -63.0, lat_min: 17.0, lat_max: 40.0}\n"
            "  files: {horizontal: x.hgrid.gr3}\n"
            "model:\n  physics: {dt: 120.0}\n  run: {nowcast_hours: 6, forecast_hours: 48}\n"
        )
        cfg = ForcingConfig.from_yaml(yml)
        assert cfg.rtofs_3d_region is None


class TestDateFallbackWithRegion:
    """The pre-existing date-fallback loop accumulates files_2d/files_3d
    across dates and stops at the first date where EITHER is non-empty.
    Filtering 3D by region makes that reachable in a new, damaging way: a
    region-specific tile can legitimately lag the others by a day even
    when 2D (global, unfiltered) is already staged for the newer cycle --
    the loop must not give up on 3D after that first date.

    But 2D and 3D always come from the SAME cycle date once found:
    _process_2d/_process_3d/nudging.py all compute each file's valid time
    as ``_rtofs_cycle_date + hours_from_filename``, one shared date for
    both variables, so returning 2D from one date and 3D from another
    would silently misdate whichever type didn't match -- a corrupted
    time axis, not a missing-file error. Every test here checks
    _rtofs_cycle_date alongside the file lists to confirm that never
    happens.
    """

    def _stage(self, root, date_str, two_d=False, three_d_region=None):
        d = root / f"rtofs.{date_str}"
        d.mkdir()
        if two_d:
            (d / "rtofs_glo_2ds_f006_diag.nc").touch()
        if three_d_region:
            (d / f"rtofs_glo_3dz_f006_6hrly_hvr_{three_d_region}.nc").touch()
        return d

    def test_region_tile_lagging_a_day_is_still_found(self, tmp_path, monkeypatch):
        """pdy-1: 2D only. pdy-2: 2D AND the requested 3D tile. Old code
        stopped at pdy-1 and returned files_3d=[] permanently -- the exact
        shape of the AK prep's original failure, just one day further back
        than the alphabetical-tile bug. New code must keep searching and
        settle on pdy-2, where BOTH types are available together."""
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_2D", 0)
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_3D", 0)
        self._stage(tmp_path, "20260729", two_d=True)
        self._stage(tmp_path, "20260728", two_d=True, three_d_region="alaska")

        cfg = ForcingConfig(lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
                            pdy="20260730", cyc=0, rtofs_3d_region="alaska")
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path)
        files_2d, files_3d = proc.find_input_files_by_type()

        assert len(files_2d) == 1
        assert len(files_3d) == 1
        assert proc._rtofs_cycle_date.strftime("%Y%m%d") == "20260728", (
            "2D and 3D must come from the SAME date -- pdy-2, where 3D was "
            "actually found -- not pdy-1, which only satisfied 2D"
        )

    def test_region_never_found_falls_back_to_first_hit_like_before(self, tmp_path, monkeypatch):
        """If the requested tile genuinely never shows up across all three
        candidate dates, the outcome is the same empty files_3d as before
        -- this only changes HOW THOROUGHLY it searches, not the worst case."""
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_2D", 0)
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_3D", 0)
        self._stage(tmp_path, "20260729", two_d=True)

        cfg = ForcingConfig(lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
                            pdy="20260730", cyc=0, rtofs_3d_region="alaska")
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path)
        files_2d, files_3d = proc.find_input_files_by_type()
        assert len(files_2d) == 1
        assert files_3d == []
        assert proc._rtofs_cycle_date.strftime("%Y%m%d") == "20260729"

    def test_no_region_keeps_the_original_first_hit_behaviour_exactly(self, tmp_path, monkeypatch):
        """Systems that never set rtofs_3d_region (today: everything in
        production) must see byte-identical search behaviour: stop at the
        FIRST date with any hit, even if a later date could offer more.
        This is deliberately not "improved" as an uncontrolled side effect
        of the region feature -- SECOFS/ATL only get the new behaviour if
        they explicitly opt in."""
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_2D", 0)
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_3D", 0)
        self._stage(tmp_path, "20260729", two_d=True)                      # 2D only
        self._stage(tmp_path, "20260728", two_d=True, three_d_region="US_east")  # both

        cfg = ForcingConfig(lon_min=-88.0, lon_max=-63.0, lat_min=17.0, lat_max=40.0,
                            pdy="20260730", cyc=0)
        assert cfg.rtofs_3d_region is None
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path)
        files_2d, files_3d = proc.find_input_files_by_type()
        assert len(files_2d) == 1
        assert files_3d == [], "unset region must stop at the first hit, same as main"
        assert proc._rtofs_cycle_date.strftime("%Y%m%d") == "20260729"

    def test_both_types_on_the_first_date_stops_immediately(self, tmp_path, monkeypatch):
        """The common case shouldn't pay for the fallback machinery: when
        the first date already has both types, no further dates are
        touched at all."""
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_2D", 0)
        monkeypatch.setattr(RTOFSProcessor, "MIN_FILE_SIZE_3D", 0)
        self._stage(tmp_path, "20260729", two_d=True, three_d_region="alaska")
        # A second, older date exists too -- it must never be consulted.
        older = self._stage(tmp_path, "20260728", two_d=True, three_d_region="alaska")

        cfg = ForcingConfig(lon_min=156.0, lon_max=204.0, lat_min=48.5, lat_max=67.0,
                            pdy="20260730", cyc=0, rtofs_3d_region="alaska")
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path)
        files_2d, files_3d = proc.find_input_files_by_type()
        assert proc._rtofs_cycle_date.strftime("%Y%m%d") == "20260729"
        assert not any(str(older) in str(f) for f in files_2d + files_3d)
