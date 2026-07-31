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


def _stage_three_region_rtofs(tmp_path, pdy="20260729"):
    """The real WCOSS2 layout: three 3dz tiles per valid time, one 2ds set."""
    rtofs_dir = tmp_path / f"rtofs.{pdy}"
    rtofs_dir.mkdir()
    for fhr in _FHRS:
        for region in _REGIONS:
            (rtofs_dir / f"rtofs_glo_3dz_{fhr}_6hrly_hvr_{region}.nc").write_bytes(
                b"0" * 250_000_000  # over MIN_FILE_SIZE_3D
            )
        (rtofs_dir / f"rtofs_glo_2ds_{fhr}_diag.nc").write_bytes(
            b"0" * 200_000_000  # over MIN_FILE_SIZE_2D
        )
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
        unfiltered (US_east-by-accident) behaviour -- it should surface as
        zero files found, which the caller's critical_sources check turns
        into a loud prep failure instead of quietly wrong data."""
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
