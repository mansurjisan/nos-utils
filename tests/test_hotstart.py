"""Tests for HotstartProcessor."""

import os
import shutil
from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.hotstart import (
    HotstartProcessor, HotstartInfo, HotstartStagingError,
)

netCDF4 = pytest.importorskip("netCDF4")


def _make_rst(path: Path, fmt: str = "NETCDF4", time_seconds: float = 21600.0,
              eta2: float = 0.0):
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
    eta[:] = eta2
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


class TestStalenessWarning:
    """Regression coverage for the silent hotstart/anchor mismatch: SCHISM's
    ihot=1 relabels whatever restart is selected to the time_hotstart anchor
    (cycle - nowcast_hours) regardless of the restart's own valid time, so a
    mismatch must be flagged loudly instead of silently skipping or
    re-simulating ocean evolution.
    """

    @staticmethod
    def _restart_at(root: Path, run: str, pdy: str, cyc: int) -> Path:
        path = root / f"{run}.{pdy}" / f"{run}.t{cyc:02d}z.{pdy}.rst.nowcast.nc"
        _make_rst(path)
        return path

    @staticmethod
    def _init_at(root: Path, run: str, dir_pdy: str, tag_pdy: str, cyc: int) -> Path:
        """Lay down an init-staged file (stage_init_to_comout's output
        naming): lives under ``{run}.{dir_pdy}`` but its own filename tag
        is ``tag_pdy``/``cyc`` — normally the CONSUMING cycle, which may
        differ from the content's true valid time.
        """
        path = root / f"{run}.{dir_pdy}" / f"{run}.t{cyc:02d}z.{tag_pdy}.init.nowcast.nc"
        _make_rst(path)
        return path

    def test_daily_cadence_warns_evolution_skipped(self, tmp_path):
        """Daily 00z cadence with nowcast_hours=6: only yesterday's 00z
        restart is available, 18h short of the 18z anchor a 6h nowcast
        window expects. This is the stofs_3d_ak_ufs bring-up scenario."""
        run = "secofs"
        root = tmp_path / "com" / "nos"
        self._restart_at(root, run, "20260730", 0)

        cfg = ForcingConfig.for_secofs(pdy="20260731", cyc=0)
        proc = HotstartProcessor(cfg, root, tmp_path / "out", run_name=run)
        result = proc.process()

        assert result.success
        assert result.warnings, "staleness warning must land in ForcingResult.warnings"
        msg = " ".join(result.warnings)
        assert "18h of ocean evolution will be skipped" in msg

    def test_normal_6hourly_chain_is_silent(self, tmp_path):
        """The routine warm chain (file_dt == anchor == previous cycle)
        must not trip the staleness warning."""
        run = "secofs"
        root = tmp_path / "com" / "nos"
        self._restart_at(root, run, "20260730", 18)

        cfg = ForcingConfig.for_secofs(pdy="20260731", cyc=0)
        proc = HotstartProcessor(cfg, root, tmp_path / "out", run_name=run)
        result = proc.process()

        assert result.success
        assert result.warnings == []

    def test_restart_newer_than_anchor_warns_re_simulated(self, tmp_path):
        """A restart valid AFTER the anchor (but still before the cycle)
        means SCHISM will re-simulate the gap, not skip it."""
        run = "secofs"
        root = tmp_path / "com" / "nos"
        self._restart_at(root, run, "20260730", 21)

        cfg = ForcingConfig.for_secofs(pdy="20260731", cyc=0)
        proc = HotstartProcessor(cfg, root, tmp_path / "out", run_name=run)
        result = proc.process()

        assert result.success
        assert result.warnings
        msg = " ".join(result.warnings)
        assert "3h will be re-simulated" in msg

    def test_init_staged_restart_is_unverifiable(self, tmp_path):
        """Reproduces the false negative: stage_init_to_comout copies
        whatever restart it found for cycle C to a name tagged with C
        itself, so the name encodes the CONSUMING cycle, not the
        content's valid time. Here cycle 20260731 06z (nowcast_hours=6)
        finds only secofs.t00z.20260731.init.nowcast.nc in COMOUT -- its
        tag happens to parse to 00z, which equals the anchor
        (06z - 6h), but the underlying content could genuinely be from
        any earlier cycle (e.g. Jul 30 18z). The old code trusted the
        name and went silent; it must instead flag the file as
        unverifiable.
        """
        run = "secofs"
        root = tmp_path / "com" / "nos"
        self._init_at(root, run, dir_pdy="20260731", tag_pdy="20260731", cyc=0)

        cfg = ForcingConfig.for_secofs(pdy="20260731", cyc=6)
        proc = HotstartProcessor(cfg, root, tmp_path / "out", run_name=run)
        result = proc.process()

        assert result.success
        assert result.warnings, "init-staged restart must be flagged, not silent"
        msg = " ".join(result.warnings)
        assert "cannot be verified" in msg
        assert "init-staged" in msg
        assert "skipped" not in msg

    def test_init_staged_and_genuine_rst_tie_prefers_rst(self, tmp_path):
        """When an init-staged copy and a genuine rst.nowcast.nc both
        parse to the same valid time and both precede the cycle, the
        rst must win the tie: find_input_files globs the
        `*.rst.nowcast.nc` pattern before `*.init.nowcast.nc` (COMF vs.
        init naming, in that literal list order), and _find_hotstart's
        `scored.sort(..., reverse=True)` is stable, so equal-valid-time
        candidates keep that discovery order and the rst is scored[0].
        A genuine restart's name is trustworthy, so selecting it must
        stay silent (warnings == []).
        """
        run = "secofs"
        root = tmp_path / "com" / "nos"
        rst = self._restart_at(root, run, "20260731", 0)
        self._init_at(root, run, dir_pdy="20260731", tag_pdy="20260731", cyc=0)

        cfg = ForcingConfig.for_secofs(pdy="20260731", cyc=6)
        proc = HotstartProcessor(cfg, root, tmp_path / "out", run_name=run)

        selected = proc._find_hotstart()
        assert selected == rst, "genuine rst must win the tie over the init copy"

        result = proc.process()
        assert result.success
        assert result.warnings == []


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


class TestStageInitPreservesSeededInit:
    """Regression coverage for a WCOSS2 20260802/12z incident: an operator
    hand-seeded a cold-start rest-state init at the documented target name
    (``$COMOUT/stofs_3d_ak_ufs.t12z.20260802.init.nowcast.nc``), and
    ``stage_init_to_comout``'s lookback found an unrelated, 60h-stale
    restart from earlier bring-up testing and silently converted/copied it
    ONTO the seeded file, destroying it. ``stage_init_to_comout`` must never
    overwrite an existing target.
    """

    def test_seeded_init_is_not_overwritten_by_stale_restart(
        self, tmp_path, caplog,
    ):
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        target = target_dir / f"{run}.t12z.20260802.init.nowcast.nc"

        # Operator hand-seeds the cold-start rest-state init. time_seconds
        # is the distinguishing marker: the seed's content must survive.
        # NETCDF4_CLASSIC so this test isolates the PRESERVE/overwrite
        # behavior from the separate format-failure path covered by
        # test_preserved_seed_non_classic_format_raises below.
        _make_rst(target, fmt="NETCDF4_CLASSIC", time_seconds=0.0)
        seeded_mtime = target.stat().st_mtime
        seeded_size = target.stat().st_size

        # A stale, unrelated restart from earlier bring-up testing sitting
        # in the lookback path, 60h before the t12z anchor.
        _make_rst(
            comout_root / f"{run}.20260730" /
            f"{run}.t00z.20260730.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=12)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        with caplog.at_level("WARNING", logger="nos_utils.forcing.hotstart"):
            staged = proc.stage_init_to_comout(
                target_dir, f"{run}.t12z.20260802.init.nowcast.nc",
            )

        assert staged == target

        # Content unchanged: still the seeded file, not the stale restart.
        with netCDF4.Dataset(str(target)) as ds:
            assert float(ds.variables["time"][:].flat[0]) == 0.0
        assert target.stat().st_mtime == seeded_mtime
        assert target.stat().st_size == seeded_size

        assert any(
            "PRESERVED" in r.message and "t00z.20260730" in r.message
            for r in caplog.records
        ), "expected a prominent preserve warning naming the stale restart"

    def test_seeded_init_preserved_warning_reaches_forcingresult_warnings(
        self, tmp_path,
    ):
        """Same clobber scenario, exercised through the ``warnings=``
        plumbing the orchestrator uses so the message lands in
        ``ForcingResult.warnings`` rather than only the log."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        target = target_dir / f"{run}.t12z.20260802.init.nowcast.nc"
        # NETCDF4_CLASSIC -- isolates the preserve/warnings-plumbing
        # behavior from the separate format-failure path.
        _make_rst(target, fmt="NETCDF4_CLASSIC", time_seconds=0.0)
        _make_rst(
            comout_root / f"{run}.20260730" /
            f"{run}.t00z.20260730.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=12)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        warnings: list = []
        staged = proc.stage_init_to_comout(
            target_dir, f"{run}.t12z.20260802.init.nowcast.nc",
            warnings=warnings,
        )
        assert staged == target
        assert warnings, "preserve warning must be appended to the caller's list"
        assert "PRESERVED" in warnings[0]

    def test_no_pre_existing_init_still_stages_from_restart(self, tmp_path):
        """Pin the pre-fix behavior: with no existing target, staging from
        the found restart proceeds exactly as before the guard was added."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        _make_rst(
            comout_root / f"{run}.20260801" /
            f"{run}.t18z.20260801.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        target = target_dir / f"{run}.t00z.20260802.init.nowcast.nc"
        assert not target.exists()

        staged = proc.stage_init_to_comout(
            target_dir, f"{run}.t00z.20260802.init.nowcast.nc",
        )
        assert staged == target
        assert target.exists()
        with netCDF4.Dataset(str(target)) as ds:
            assert float(ds.variables["time"][:].flat[0]) == 21600.0

    def test_double_stage_same_cycle_is_silent(self, tmp_path, caplog):
        """Prep runs two phases (nowcast, forecast) and BOTH call
        stage_init_to_comout with the same target and the same source
        restart: phase 1 creates the file, phase 2 finds it already there.
        This is the ordinary path on EVERY warm cycle, so it must be
        completely silent -- no PRESERVED warning, nothing appended to the
        caller's warnings list -- only an informational "already staged"
        log record. A naive "preserve on any pre-existing target" fix would
        fire the operator-seed warning here forever; that's the flaw this
        test guards against.
        """
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        _make_rst(
            comout_root / f"{run}.20260801" /
            f"{run}.t18z.20260801.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)
        init_filename = f"{run}.t00z.20260802.init.nowcast.nc"

        warnings: list = []
        first = proc.stage_init_to_comout(
            target_dir, init_filename, warnings=warnings,
        )
        assert first is not None

        caplog.clear()
        with caplog.at_level("INFO", logger="nos_utils.forcing.hotstart"):
            second = proc.stage_init_to_comout(
                target_dir, init_filename, warnings=warnings,
            )

        assert second == first
        assert warnings == [], (
            "a same-state re-stage (normal phase-2-of-a-cycle case) must "
            "not append anything to the caller's warnings list"
        )
        assert not any(
            r.levelname == "WARNING" and "PRESERVED" in r.message
            for r in caplog.records
        ), "same-state re-stage must not fire the PRESERVED warning"
        assert any(
            r.levelname == "INFO" and "already staged" in r.message
            for r in caplog.records
        ), "expected an INFO 'already staged' record on the second call"

    def test_unreadable_unprovenanced_target_is_preserved_and_raises(self, tmp_path, caplog):
        """Round-3 review, finding 2 (blocker): a corrupt/truncated leftover
        at the target path has no provenance sidecar proving this method
        wrote it, so it can no longer be distinguished here from a healthy
        seed sitting behind a transient probe failure. The old self-heal-
        by-restage behavior (silently overwriting it from a found restart)
        is exactly the failure mode that can destroy a valid operator seed
        on a flaky probe -- so this now PRESERVES the file and raises
        HotstartStagingError instead of restaging over it. A human must
        inspect and clear it manually."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        target = target_dir / f"{run}.t12z.20260802.init.nowcast.nc"

        target_dir.mkdir(parents=True, exist_ok=True)
        junk = b"not a valid netcdf file -- truncated mid-write"
        target.write_bytes(junk)

        _make_rst(
            comout_root / f"{run}.20260730" /
            f"{run}.t00z.20260730.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=12)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        warnings: list = []
        with caplog.at_level("WARNING", logger="nos_utils.forcing.hotstart"):
            with pytest.raises(HotstartStagingError) as excinfo:
                proc.stage_init_to_comout(
                    target_dir, f"{run}.t12z.20260802.init.nowcast.nc",
                    warnings=warnings,
                )

        assert "could not be read" in str(excinfo.value)

        # Junk content must be untouched -- NOT overwritten by the stale
        # restart that was sitting in the lookback path.
        assert target.read_bytes() == junk

        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any(
            "could not be read" in r.message for r in error_records
        ), "expected an ERROR record about the unreadable target"
        assert any(
            "could not be read" in w for w in warnings
        ), "the error message must also land in the warnings list"

    def test_preserved_seed_non_classic_format_raises(self, tmp_path, caplog):
        """A preserved operator seed that is HDF5 (NETCDF4) rather than
        NETCDF4_CLASSIC must FAIL PREP, not merely log an ERROR and
        continue (round-2 review finding 2): SCHISM's parallel-IO staging
        requires classic and segfaults at rank scale on HDF5
        (ush/nos_run.sh archives rst.nowcast.nc as HDF5, so `cp rst ->
        init` is a realistic operator mistake). Letting prep report green
        here meant SCHISM launched at rank scale with a seed the code
        already knew would fail. The seed is still preserved -- never
        destroyed -- but stage_init_to_comout must now raise
        HotstartStagingError instead of returning normally, and the ERROR
        log record plus the warnings-list entry (both consumed by the
        orchestrator's ForcingResult.warnings) must still be produced
        before the raise."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        target = target_dir / f"{run}.t12z.20260802.init.nowcast.nc"

        _make_rst(target, fmt="NETCDF4", time_seconds=0.0)
        _make_rst(
            comout_root / f"{run}.20260730" /
            f"{run}.t00z.20260730.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=12)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        warnings: list = []
        with caplog.at_level("WARNING", logger="nos_utils.forcing.hotstart"):
            with pytest.raises(HotstartStagingError) as excinfo:
                proc.stage_init_to_comout(
                    target_dir, f"{run}.t12z.20260802.init.nowcast.nc",
                    warnings=warnings,
                )

        assert "NETCDF4" in str(excinfo.value)
        assert "NETCDF4_CLASSIC" in str(excinfo.value)

        # Seed itself must be untouched -- still HDF5, not converted, not
        # deleted, not overwritten.
        with netCDF4.Dataset(str(target)) as ds:
            assert ds.file_format == "NETCDF4"

        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any(
            "NETCDF4" in r.message and "NETCDF4_CLASSIC" in r.message
            for r in error_records
        ), "expected an ERROR record naming the seed's actual format"
        assert any(
            "NETCDF4_CLASSIC" in w for w in warnings
        ), "the format ERROR message must also land in the warnings list"


class TestStageInitValidatesTargetBeforeSourceDiscovery:
    """Round-3 review, finding 1 (blocker): the target's format must be
    validated BEFORE ``_find_hotstart()`` is ever called, not after.

    The old code called ``source = self._find_hotstart()`` first and
    returned ``None`` immediately when no source existed -- before the
    existing target was inspected at all. On a first-cycle bring-up
    (operator seed present, NO previous restart anywhere), a non-CLASSIC
    seed sailed through unexamined: prep stayed green and the runner
    handed SCHISM a format guaranteed to fail at rank scale. The fix moves
    the format gate ahead of source discovery so it fires unconditionally.
    """

    def test_non_classic_seed_raises_even_with_no_source(self, tmp_path, caplog):
        """First-cycle bring-up shape: a non-CLASSIC (HDF5) seed and NO
        previous restart anywhere. Must still raise HotstartStagingError
        -- the absence of a source must not suppress the format check."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        target = target_dir / f"{run}.t12z.20260802.init.nowcast.nc"

        _make_rst(target, fmt="NETCDF4", time_seconds=0.0)
        # No restart anywhere else in comout_root -- _find_hotstart() would
        # return None if it were ever consulted.

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=12)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        warnings: list = []
        with caplog.at_level("WARNING", logger="nos_utils.forcing.hotstart"):
            with pytest.raises(HotstartStagingError) as excinfo:
                proc.stage_init_to_comout(
                    target_dir, f"{run}.t12z.20260802.init.nowcast.nc",
                    warnings=warnings,
                )

        assert "NETCDF4" in str(excinfo.value)
        assert "NETCDF4_CLASSIC" in str(excinfo.value)

        # Seed itself untouched -- still HDF5.
        with netCDF4.Dataset(str(target)) as ds:
            assert ds.file_format == "NETCDF4"

        error_records = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any(
            "NETCDF4" in r.message and "NETCDF4_CLASSIC" in r.message
            for r in error_records
        ), "expected an ERROR record naming the seed's actual format"
        assert any("NETCDF4_CLASSIC" in w for w in warnings)

    def test_classic_seed_with_no_source_is_used_as_is(self, tmp_path, caplog):
        """First-cycle bring-up, the EXPECTED shape: a valid
        NETCDF4_CLASSIC operator seed with NO previous restart anywhere to
        compare against must be recognized and kept as the returned init
        file, not treated as an anomaly and not blocked by the absence of
        a source."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        target = target_dir / f"{run}.t12z.20260802.init.nowcast.nc"

        _make_rst(target, fmt="NETCDF4_CLASSIC", time_seconds=0.0)
        seeded_bytes = target.read_bytes()
        # No restart anywhere else in comout_root.

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=12)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        warnings: list = []
        with caplog.at_level("WARNING", logger="nos_utils.forcing.hotstart"):
            staged = proc.stage_init_to_comout(
                target_dir, f"{run}.t12z.20260802.init.nowcast.nc",
                warnings=warnings,
            )

        assert staged == target
        assert target.read_bytes() == seeded_bytes, "seed content must be untouched"
        assert not any(
            r.levelname == "WARNING" and "PRESERVED" in r.message
            for r in caplog.records
        ), "a valid first-cycle seed must not trip the anomaly/PRESERVED warning"


class TestStageInitTransientProbeFailure:
    """Round-3 review, finding 2 (blocker): a failed ``_netcdf_format``
    probe on an UNPROVENANCED existing seed must never be treated as proof
    of corruption. The probe can fail transiently (not only on genuine
    corruption); silently re-staging over it can destroy a perfectly
    healthy operator seed. The fix: only a target whose provenance sidecar
    proves THIS method wrote it -- and whose stat is unchanged since --
    may be treated as fine despite a failed probe (writes go through
    temp+os.replace so they cannot be torn). Anything else must preserve
    and raise, never silently restage.
    """

    def test_probe_failure_on_unprovenanced_seed_with_stale_restart_raises(
        self, tmp_path, monkeypatch,
    ):
        """A genuinely valid CLASSIC seed with no provenance sidecar (an
        operator-placed file, never written by this method), sitting next
        to a stale restart the lookback would otherwise find, whose
        format probe FAILS transiently (forced via monkeypatch, not real
        corruption). Must be PRESERVED (content untouched) and must raise
        HotstartStagingError -- never silently re-staged from the stale
        restart."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        target = target_dir / f"{run}.t12z.20260802.init.nowcast.nc"

        _make_rst(target, fmt="NETCDF4_CLASSIC", time_seconds=0.0)
        seeded_bytes = target.read_bytes()

        _make_rst(
            comout_root / f"{run}.20260730" /
            f"{run}.t00z.20260730.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=12)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        # _netcdf_format is a @staticmethod -- patch it on the class.
        monkeypatch.setattr(
            HotstartProcessor, "_netcdf_format", staticmethod(lambda path: None),
        )

        warnings: list = []
        with pytest.raises(HotstartStagingError):
            proc.stage_init_to_comout(
                target_dir, f"{run}.t12z.20260802.init.nowcast.nc",
                warnings=warnings,
            )

        assert target.read_bytes() == seeded_bytes, (
            "a healthy seed behind a flaky probe must never be silently "
            "overwritten by a stale restart"
        )
        assert warnings

    def test_probe_failure_on_provenanced_target_is_preserved(
        self, tmp_path, monkeypatch, caplog,
    ):
        """A target THIS method staged (sidecar's recorded target stamp
        matches its current stat) whose format probe fails transiently
        must be treated as fine: no raise, no WARNING, an INFO log, and
        the previously-staged path is returned untouched."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        _make_rst(
            comout_root / f"{run}.20260801" /
            f"{run}.t18z.20260801.rst.nowcast.nc",
            time_seconds=21600.0, fmt="NETCDF4_CLASSIC",
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)
        init_filename = f"{run}.t00z.20260802.init.nowcast.nc"

        first = proc.stage_init_to_comout(target_dir, init_filename)
        assert first is not None
        staged_bytes = first.read_bytes()

        monkeypatch.setattr(
            HotstartProcessor, "_netcdf_format", staticmethod(lambda path: None),
        )

        warnings: list = []
        with caplog.at_level("INFO", logger="nos_utils.forcing.hotstart"):
            second = proc.stage_init_to_comout(
                target_dir, init_filename, warnings=warnings,
            )

        assert second == first
        assert second.read_bytes() == staged_bytes, "provenanced seed must be untouched"
        assert warnings == [], "a proven-transient probe failure must not warn"
        assert not any(r.levelname == "WARNING" for r in caplog.records)
        assert any(
            r.levelname == "INFO" and "transient" in r.message.lower()
            for r in caplog.records
        ), "expected an INFO record naming the transient probe failure"


class TestStageInitProvenanceIdentity:
    """Round-2 review, finding 1: replace the weak scalar time/iths
    identity test in stage_init_to_comout with a provenance sidecar.

    Every daily ihot=1 restart carries IDENTICAL time/iths scalars --
    SCHISM always relabels the restart to the same time_hotstart anchor
    regardless of the restart's actual content -- so comparing only those
    two scalars misclassifies a genuinely different ocean state as an
    "identical restage". Worse, the old `elif same:` early return
    bypassed the NETCDF4_CLASSIC format check entirely, so a non-CLASSIC
    target with matching scalars returned silently.

    The fix: `stage_init_to_comout` writes a `<target>.provenance.json`
    sidecar (temp+os.replace) whenever it is the one that wrote `target`,
    recording the source restart's (path, size, mtime_ns) AND target's
    own post-write (size, mtime_ns). A LATER call is only an "identical
    restage" when the sidecar exists, parses, and both stamps match the
    CURRENT source/target exactly.
    """

    def test_matching_scalars_different_eta2_no_sidecar_is_preserved(
        self, tmp_path, caplog,
    ):
        """Reviewer repro: target time=21600/iths=180/eta2=0 vs a
        candidate source restart with the SAME time=21600/iths=180 but
        eta2=1 -- a genuinely different ocean state that the old scalar
        compare would have called "identical" (INFO, no warning). With no
        provenance sidecar for this target, it must NOT be treated as an
        identical restage: WARNING fired, seed preserved untouched.
        NETCDF4_CLASSIC is used for the target so this isolates the
        identity logic from the separate format-failure path.
        """
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        target = target_dir / f"{run}.t12z.20260802.init.nowcast.nc"

        _make_rst(target, fmt="NETCDF4_CLASSIC", time_seconds=21600.0, eta2=0.0)
        _make_rst(
            comout_root / f"{run}.20260730" /
            f"{run}.t00z.20260730.rst.nowcast.nc",
            time_seconds=21600.0, eta2=1.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=12)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)

        warnings: list = []
        with caplog.at_level("WARNING", logger="nos_utils.forcing.hotstart"):
            staged = proc.stage_init_to_comout(
                target_dir, f"{run}.t12z.20260802.init.nowcast.nc",
                warnings=warnings,
            )

        assert staged == target
        assert warnings, "matching scalars with no sidecar must still warn, not go silent"
        assert any("PRESERVED" in w for w in warnings)
        with netCDF4.Dataset(str(target)) as ds:
            assert float(ds.variables["eta2"][:][0]) == 0.0, "target must be untouched"

    def test_normal_two_phase_restage_uses_sidecar(self, tmp_path, caplog):
        """The ordinary nowcast-then-forecast double call: phase 1 writes
        target + sidecar from `source`; phase 2 finds the same source
        (untouched) and the same target (untouched) and must classify
        that as identical via the sidecar -- INFO, no warning, no
        exception."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        _make_rst(
            comout_root / f"{run}.20260801" /
            f"{run}.t18z.20260801.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)
        init_filename = f"{run}.t00z.20260802.init.nowcast.nc"

        warnings: list = []
        first = proc.stage_init_to_comout(
            target_dir, init_filename, warnings=warnings,
        )
        assert first is not None
        sidecar = first.with_name(first.name + ".provenance.json")
        assert sidecar.exists(), "stage must write a provenance sidecar next to the target"

        caplog.clear()
        with caplog.at_level("INFO", logger="nos_utils.forcing.hotstart"):
            second = proc.stage_init_to_comout(
                target_dir, init_filename, warnings=warnings,
            )

        assert second == first
        assert warnings == [], (
            "a sidecar-matched restage must not append anything to the "
            "caller's warnings list"
        )
        assert not any(
            r.levelname == "WARNING" and "PRESERVED" in r.message
            for r in caplog.records
        )
        assert any(
            r.levelname == "INFO" and "already staged" in r.message
            for r in caplog.records
        )

    def test_operator_overwrite_after_stage_breaks_sidecar_match(
        self, tmp_path, caplog,
    ):
        """An operator touching/replacing the staged target after phase 1
        must NOT be silently treated as identical just because the
        source restart is unchanged: the sidecar's recorded TARGET stamp
        no longer matches the target's current stat, so the preserve/
        WARNING path must fire instead of the silent INFO path."""
        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        _make_rst(
            comout_root / f"{run}.20260801" /
            f"{run}.t18z.20260801.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)
        init_filename = f"{run}.t00z.20260802.init.nowcast.nc"

        first = proc.stage_init_to_comout(target_dir, init_filename)
        assert first is not None
        target = first

        # Operator touches the staged target well after staging -- content
        # stays valid NETCDF4_CLASSIC, only the mtime changes, but that's
        # enough to break the sidecar's recorded target stamp.
        st = target.stat()
        os.utime(target, (st.st_atime, st.st_mtime + 3600))

        warnings: list = []
        with caplog.at_level("WARNING", logger="nos_utils.forcing.hotstart"):
            second = proc.stage_init_to_comout(
                target_dir, init_filename, warnings=warnings,
            )

        assert second == target
        assert warnings, "operator-touched target must not silently match the sidecar"
        assert any("PRESERVED" in w for w in warnings)

    def test_sidecar_not_matched_by_hotstart_discovery_globs(self, tmp_path):
        """The provenance sidecar must never be discoverable as a
        candidate hotstart/restart file by find_input_files -- verified
        against the actual glob patterns, not assumed."""
        import fnmatch

        run = "stofs_3d_ak_ufs"
        comout_root = tmp_path / "com" / "nos"
        target_dir = comout_root / f"{run}.20260802"
        _make_rst(
            comout_root / f"{run}.20260801" /
            f"{run}.t18z.20260801.rst.nowcast.nc",
            time_seconds=21600.0,
        )

        cfg = ForcingConfig.for_secofs(pdy="20260802", cyc=0)
        proc = HotstartProcessor(cfg, comout_root, tmp_path / "out", run_name=run)
        init_filename = f"{run}.t00z.20260802.init.nowcast.nc"
        staged = proc.stage_init_to_comout(target_dir, init_filename)
        assert staged is not None
        sidecar = staged.with_name(staged.name + ".provenance.json")
        assert sidecar.exists()

        for pattern in [
            "hotstart*.nc",
            f"{run}*hotstart*.nc",
            f"{run}*.rst.nowcast.nc",
            f"{run}*.init.nowcast.nc",
            f"{run}*restart*.nc",
        ]:
            assert not fnmatch.fnmatch(sidecar.name, pattern), (
                f"sidecar {sidecar.name} must not match discovery pattern {pattern}"
            )

        # Belt-and-suspenders: it must not actually turn up as a
        # candidate from the real discovery walk either.
        candidates = proc.find_input_files()
        assert sidecar not in candidates


class TestSearchRootFromCycleLeaf:
    """Regression: cold-start despite a valid prior-day restart.

    nco_bridge sets ``paths["restart"]`` from ``$COMIN``, which in the
    SECOFS/STOFS-UFS J-jobs is the *current* cycle's dated leaf
    (``$COMROOT/$NET/$RUN.$PDY``) — created empty by prep itself.  A
    t00z cycle cold-started because the per-day walk, anchored at that
    empty leaf, never crossed into the sibling prior-day dir holding
    ``$RUN.t18z.$(PDY-1).rst.nowcast.nc``.  HotstartProcessor must
    additionally anchor the walk at the parent of a dated cycle leaf so
    the prior-day restart is discovered.
    """

    def test_prior_day_restart_found_from_current_cycle_leaf(self, tmp_path):
        run = "secofs_ufs"
        com_root = tmp_path / "com" / "nos"
        prior = (com_root / f"{run}.20260518" /
                 f"{run}.t18z.20260518.rst.nowcast.nc")
        _make_rst(prior)
        # Current-cycle dir exists but is empty (prep just created it).
        cur = com_root / f"{run}.20260519"
        cur.mkdir(parents=True, exist_ok=True)

        cfg = ForcingConfig.for_secofs(pdy="20260519", cyc=0)
        # nco_bridge hands the CURRENT-CYCLE LEAF, not the COMOUT root.
        proc = HotstartProcessor(cfg, cur, tmp_path / "out", run_name=run)

        found = proc._find_hotstart()
        assert found is not None, (
            "must cross the day boundary into the sibling prior-day dir"
        )
        assert found == prior

    def test_stage_init_from_current_cycle_leaf(self, tmp_path):
        run = "secofs_ufs"
        com_root = tmp_path / "com" / "nos"
        _make_rst(
            com_root / f"{run}.20260518" /
            f"{run}.t18z.20260518.rst.nowcast.nc",
            fmt="NETCDF4_CLASSIC",
        )
        cur = com_root / f"{run}.20260519"
        cur.mkdir(parents=True, exist_ok=True)

        cfg = ForcingConfig.for_secofs(pdy="20260519", cyc=0)
        proc = HotstartProcessor(cfg, cur, tmp_path / "out", run_name=run)
        staged = proc.stage_init_to_comout(
            cur, f"{run}.t00z.20260519.init.nowcast.nc",
        )
        assert staged is not None
        assert staged.name == f"{run}.t00z.20260519.init.nowcast.nc"
        assert staged.exists()

    def test_comout_root_input_still_works(self, tmp_path):
        """Non-dated root input (the documented contract) is unchanged:
        the parent is NOT added and discovery still works."""
        run = "secofs_ufs"
        com_root = tmp_path / "com" / "nos"
        prior = (com_root / f"{run}.20260518" /
                 f"{run}.t18z.20260518.rst.nowcast.nc")
        _make_rst(prior)

        cfg = ForcingConfig.for_secofs(pdy="20260519", cyc=0)
        proc = HotstartProcessor(cfg, com_root, tmp_path / "out", run_name=run)
        assert proc._search_roots() == [com_root]
        assert proc._find_hotstart() == prior
