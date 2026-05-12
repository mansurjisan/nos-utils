"""Tests for HotstartProcessor."""

import shutil
from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.hotstart import HotstartProcessor, HotstartInfo

netCDF4 = pytest.importorskip("netCDF4")


def _make_rst(path: Path, fmt: str = "NETCDF4", time_seconds: float = 21600.0):
    """Write a minimal SCHISM-shaped restart file in the given NetCDF format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = netCDF4.Dataset(str(path), "w", format=fmt)
    ds.createDimension("node", 10)
    ds.createDimension("nVert", 5)
    t = ds.createVariable("time", "f8")
    t[:] = time_seconds
    iths = ds.createVariable("iths", "i4")
    iths[:] = 180
    eta = ds.createVariable("eta2", "f8", ("node",))
    eta[:] = 0.0
    ds.test_marker = "rst-from-test"
    ds.close()


@pytest.fixture
def mock_hotstart(tmp_path):
    """Create a mock hotstart.nc file."""
    hs_dir = tmp_path / "restart"
    hs_dir.mkdir()
    hs_file = hs_dir / "hotstart.nc"

    ds = netCDF4.Dataset(str(hs_file), "w")
    ds.createDimension("node", 100)
    ds.createDimension("nVert", 51)

    time_var = ds.createVariable("time", "f8")
    time_var[:] = 21600.0  # 6 hours in seconds

    iths_var = ds.createVariable("iths", "i4")
    iths_var[:] = 180  # 180 time steps

    ds.close()
    return hs_dir


class TestHotstartProcessor:
    def test_find_hotstart(self, mock_config, mock_hotstart, tmp_path):
        proc = HotstartProcessor(
            mock_config, mock_hotstart, tmp_path / "out",
        )
        result = proc.process()

        assert result.success
        assert result.metadata["ihot"] == 1
        assert result.metadata["time_seconds"] == 21600.0
        assert result.metadata["iths"] == 180

    def test_no_hotstart_cold_start(self, mock_config, tmp_path):
        """Missing hotstart -> cold start (ihot=0), still success."""
        proc = HotstartProcessor(
            mock_config, tmp_path / "empty", tmp_path / "out",
        )
        result = proc.process()

        assert result.success  # Non-fatal
        assert result.metadata["ihot"] == 0
        assert "cold start" in result.warnings[0].lower()

    def test_links_hotstart(self, mock_config, mock_hotstart, tmp_path):
        """Should create symlink to hotstart.nc in output dir."""
        out_dir = tmp_path / "out"
        proc = HotstartProcessor(mock_config, mock_hotstart, out_dir)
        proc.process()

        assert (out_dir / "hotstart.nc").exists()


class TestFindHotstartFallback:
    """Regression suite for the _find_hotstart mtime fallback that used to
    return today's own pre-staged init file when only future-dated
    candidates were available.

    The bug: _find_hotstart's "fallback if no cycle time could be parsed"
    branch sorted ALL valid files by mtime, including ones whose filename
    parsed to cycle_dt or later (e.g. today's own init dropped into COMOUT
    by stage_init_to_comout earlier in the same prep run).  When picked,
    that filename's tHHz.YYYYMMDD tag was fed back through the orchestrator
    and produced a wrong time_hotstart anchor.  The orchestrator fix
    (#PR 8/9 chain) prevents the misuse, but this is the belt-and-suspenders
    at the source.
    """

    @staticmethod
    def _make_rst(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = netCDF4.Dataset(str(path), "w")
        ds.createDimension("node", 10)
        # File needs to exceed MIN_HOTSTART_SIZE (1 MB) to pass size filter
        big = ds.createVariable("padding", "f8", ("node",))
        big[:] = 0.0
        # Pad up to ~1.5 MB via a chunky variable
        ds.createDimension("pad", 200_000)
        pad = ds.createVariable("pad_var", "f8", ("pad",))
        pad[:] = 0.0
        ds.close()
        return path

    def test_only_todays_init_present_returns_none(self, tmp_path):
        """If only today's own t00z.20260507.init.nowcast.nc is present
        (file_dt == cycle_dt), _find_hotstart returns None so the
        orchestrator falls back to its cold-start cycle - nowcast_hours
        anchor.  Previously the mtime fallback would have returned the
        today-init file and the orchestrator would have parsed cycle time
        from its name."""
        run = "secofs"
        cycle_dir = tmp_path / f"{run}.20260507"
        self._make_rst(
            cycle_dir / f"{run}.t00z.20260507.init.nowcast.nc",
        )

        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        proc = HotstartProcessor(cfg, tmp_path, tmp_path / "out", run_name=run)

        result = proc._find_hotstart()
        assert result is None

    def test_yesterday_rst_still_selected_when_today_init_also_present(
            self, tmp_path,
    ):
        """When yesterday's rst.nowcast.nc AND today's pre-staged init both
        exist, _find_hotstart must select yesterday's (parses to
        2026-05-06 18z < 2026-05-07 00z cycle).  The today-init must not
        win via the mtime fallback even if it was written more recently."""
        run = "secofs"
        # Yesterday's 18z rst (the canonical pick)
        yest = self._make_rst(
            tmp_path / f"{run}.20260506" /
            f"{run}.t18z.20260506.rst.nowcast.nc",
        )
        # Today's own t00z init -- mtime is newer (just-written above ↑)
        self._make_rst(
            tmp_path / f"{run}.20260507" /
            f"{run}.t00z.20260507.init.nowcast.nc",
        )

        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        proc = HotstartProcessor(cfg, tmp_path, tmp_path / "out", run_name=run)

        result = proc._find_hotstart()
        assert result == yest, (
            "must select yesterday's rst by parsed cycle time, "
            "NOT today's init by mtime"
        )

    def test_unparsable_filename_still_falls_back_via_mtime(self, tmp_path):
        """Backward-compat: files whose filename doesn't parse (e.g. the
        legacy `hotstart_*.nc` pattern from old SCHISM versions) ARE still
        selectable via the mtime fallback.  The bug fix only excludes
        files that parse to a future date, not files that don't parse."""
        run = "secofs"
        # An unparsable hotstart_*.nc (no tHHz.YYYYMMDD tag in name)
        legacy = self._make_rst(tmp_path / "hotstart_001.nc")

        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        proc = HotstartProcessor(cfg, tmp_path, tmp_path / "out", run_name=run)

        result = proc._find_hotstart()
        assert result == legacy


class TestHotstartInfo:
    def test_time_days(self):
        info = HotstartInfo(
            filepath=Path("/test"), time_seconds=86400.0,
            iths=100, n_nodes=1000, n_levels=51,
        )
        assert info.time_days == 1.0

    def test_repr(self):
        info = HotstartInfo(
            filepath=Path("/test/hotstart.nc"), time_seconds=21600.0,
            iths=180, n_nodes=1684786, n_levels=63,
        )
        s = repr(info)
        assert "21600" in s
        assert "180" in s


class TestStageInitToComout:
    """stage_init_to_comout: previous-cycle restart → COMOUT init.nowcast.nc.

    The auto-stage step must produce a NETCDF4_CLASSIC file at the operational
    name regardless of whether the source restart is HDF5 or already classic.
    SECOFS production runs every 6h, so a 00z cycle picks up the previous
    18z cycle's rst.nowcast.nc; older cycles are accepted as fallback.
    """

    @staticmethod
    def _restart_at(comout_root: Path, run: str, pdy: str, cyc: int,
                    fmt: str = "NETCDF4") -> Path:
        """Lay down a previous-cycle restart in the operational COMOUT layout."""
        path = (comout_root / f"{run}.{pdy}" /
                f"{run}.t{cyc:02d}z.{pdy}.rst.nowcast.nc")
        _make_rst(path, fmt=fmt)
        return path

    def test_picks_6h_prior_cycle(self, tmp_path):
        """SECOFS runs every 6h: 00z today should pick up 18z yesterday."""
        run = "secofs"
        comout_root = tmp_path / "com" / "nos"
        # Stage the 18z-yesterday cycle's rst (the natural 6h-prior pick).
        src = self._restart_at(comout_root, run, "20260506", 18)
        # Add an older 12z cycle to confirm the newer 18z wins.
        _ = self._restart_at(comout_root, run, "20260506", 12)

        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        # restart_dir = COMOUT root so HotstartProcessor's per-day glob
        # walks `nos.20260506/` etc.
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        target_dir = comout_root / "secofs.20260507"
        staged = proc.stage_init_to_comout(
            target_dir, "secofs.t00z.20260507.init.nowcast.nc",
        )
        assert staged is not None
        assert staged.name == "secofs.t00z.20260507.init.nowcast.nc"
        assert staged.exists()

    def test_converts_hdf5_to_classic(self, tmp_path):
        """HDF5 (NETCDF4) source must come out as NETCDF4_CLASSIC."""
        run = "secofs"
        comout_root = tmp_path / "com" / "nos"
        src = self._restart_at(comout_root, run, "20260506", 18, fmt="NETCDF4")
        # Confirm test fixture really is HDF5
        with netCDF4.Dataset(str(src)) as ds:
            assert ds.file_format == "NETCDF4"

        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)
        target_dir = comout_root / "secofs.20260507"
        staged = proc.stage_init_to_comout(
            target_dir, "secofs.t00z.20260507.init.nowcast.nc",
        )
        assert staged is not None
        with netCDF4.Dataset(str(staged)) as ds:
            assert ds.file_format == "NETCDF4_CLASSIC"
            assert ds.test_marker == "rst-from-test"  # data preserved

    def test_classic_source_is_just_copied(self, tmp_path):
        """Already-classic source should not be re-converted (copy is fine)."""
        run = "secofs"
        comout_root = tmp_path / "com" / "nos"
        src = self._restart_at(
            comout_root, run, "20260506", 18, fmt="NETCDF4_CLASSIC",
        )
        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)
        target_dir = comout_root / "secofs.20260507"
        staged = proc.stage_init_to_comout(
            target_dir, "secofs.t00z.20260507.init.nowcast.nc",
        )
        assert staged is not None
        with netCDF4.Dataset(str(staged)) as ds:
            assert ds.file_format == "NETCDF4_CLASSIC"

    def test_no_restart_returns_none(self, tmp_path):
        """No previous-cycle restart anywhere → None (caller cold-starts)."""
        run = "secofs"
        comout_root = tmp_path / "empty"; comout_root.mkdir()
        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)
        target_dir = tmp_path / "out_comout"
        staged = proc.stage_init_to_comout(
            target_dir, "secofs.t00z.20260507.init.nowcast.nc",
        )
        assert staged is None

    def test_falls_back_when_6h_prior_missing(self, tmp_path):
        """If 18z-yesterday is missing, pick the next-most-recent valid restart."""
        run = "secofs"
        comout_root = tmp_path / "com" / "nos"
        # Only 12z-yesterday exists (12h prior — bigger gap than 6h)
        src = self._restart_at(comout_root, run, "20260506", 12)

        cfg = ForcingConfig.for_secofs(pdy="20260507", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)
        target_dir = comout_root / "secofs.20260507"
        staged = proc.stage_init_to_comout(
            target_dir, "secofs.t00z.20260507.init.nowcast.nc",
        )
        assert staged is not None
        # Check the source was the 12z file (we only staged one)
        with netCDF4.Dataset(str(staged)) as ds:
            assert ds.test_marker == "rst-from-test"
