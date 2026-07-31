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
