"""Tests for PrepOrchestrator."""

from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.orchestrator import PrepOrchestrator, PrepResult


@pytest.fixture
def orch_paths(tmp_path):
    """Create minimal directory structure for orchestrator."""
    paths = {
        "output": str(tmp_path / "work"),
        "fix": str(tmp_path / "fix"),
    }
    # Create fix dir with a param.nml template
    fix_dir = tmp_path / "fix"
    fix_dir.mkdir()
    (fix_dir / "param.nml").write_text(
        "&CORE\n"
        "  rnday = rnday_value\n"
        "  dt = 120.\n"
        "/\n"
        "&OPT\n"
        "  start_year = start_year_value\n"
        "  start_month = start_month_value\n"
        "  start_day = start_day_value\n"
        "  start_hour = start_hour_value\n"
        "/\n"
    )
    return paths


class TestPrepOrchestrator:
    def test_minimal_run(self, mock_config, orch_paths):
        """Orchestrator with only fix dir should produce param.nml + tidal."""
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")

        assert isinstance(result, PrepResult)
        assert result.phase == "nowcast"
        assert result.elapsed_seconds > 0

        # Should have at least hotstart + tidal + param_nml results
        sources = [r.source for r in result.results]
        assert "HOTSTART" in sources
        assert "TIDAL" in sources
        assert "PARAM_NML" in sources

    def test_param_nml_created(self, mock_config, orch_paths, tmp_path):
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")

        param_file = Path(orch_paths["output"]) / "param.nml"
        assert param_file.exists()
        content = param_file.read_text()
        assert "rnday_value" not in content  # should be substituted

    def test_summary(self, mock_config, orch_paths):
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")

        summary = result.summary()
        assert "PrepResult" in summary
        assert "nowcast" in summary


class TestTimeHotstartAnchor:
    """time_hotstart must be derived as cycle - nowcast_hours, regardless of
    whether _run_hotstart selected a restart file or not.

    Regression suite for the bug where PrepOrchestrator.run() was parsing
    time_hotstart from the selected hotstart filename's ``tHHz.YYYYMMDD``
    tag.  That tag encodes the cycle that produced the restart, not the
    restart's time origin; when today's own pre-staged init file gets
    selected, the parse returns cycle time and the launcher's sim_start
    misaligns with the OBC time axis by LEN_NOWCAST hours -- surfacing as
    SCHISM partition_hgrid:534 ParMETIS heap corruption at 2914-rank scale.
    """

    def test_time_hotstart_equals_cycle_minus_nowcast_warm(self, mock_config,
                                                            orch_paths):
        """Even if _run_hotstart selected today's pre-staged init file,
        the marker must anchor to cycle - nowcast_hours.

        mock_config has pdy=20260401, cyc=12, nowcast_hours=6 ->
        expected time_hotstart = 2026040106.
        """
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")
        assert result.success

        marker = Path(orch_paths["output"]) / "time_hotstart.t12z"
        assert marker.is_file(), "_write_time_markers must emit time_hotstart"
        assert marker.read_text().strip() == "2026040106"

    def test_base_date_matches_time_hotstart(self, mock_config, orch_paths):
        """base_date.${cycle} must byte-match time_hotstart.${cycle} -- the
        existing _write_time_markers contract."""
        orch = PrepOrchestrator(mock_config, orch_paths)
        orch.run(phase="nowcast")

        th = (Path(orch_paths["output"]) / "time_hotstart.t12z").read_text()
        bd = (Path(orch_paths["output"]) / "base_date.t12z").read_text()
        assert th == bd

    def test_time_nowcastend_is_cycle(self, mock_config, orch_paths):
        """Sanity-check the other marker: time_nowcastend == cycle time.
        mock_config pdy=20260401, cyc=12 -> 2026040112.
        """
        orch = PrepOrchestrator(mock_config, orch_paths)
        orch.run(phase="nowcast")

        marker = Path(orch_paths["output"]) / "time_nowcastend.t12z"
        assert marker.read_text().strip() == "2026040112"

    def test_env_time_hotstart_ignored(self, mock_config, orch_paths,
                                       monkeypatch):
        """An environment override of $time_hotstart must NOT change the
        marker value -- the YAML nowcast_hours anchor is authoritative.

        Upstream J-jobs sometimes export a 24h-back convention; tolerating
        that here would re-introduce the OBC/sim_start misalignment.
        """
        monkeypatch.setenv("time_hotstart", "2026033112")  # 24h-back ALL wrong
        orch = PrepOrchestrator(mock_config, orch_paths)
        orch.run(phase="nowcast")

        marker = Path(orch_paths["output"]) / "time_hotstart.t12z"
        # Still cycle - nowcast_hours, not the env value.
        assert marker.read_text().strip() == "2026040106"


class TestRoutePhaseOrchestrator:
    """Verify phase is threaded through to the phase-aware processors.

    The orchestrator's ``run(phase=...)`` API must forward the phase
    string to NWMProcessor / NudgingProcessor / RTOFSProcessor /
    UFSConfigProcessor so each emits its phase-specific output window.
    """

    def test_orchestrator_forwards_phase_to_param_nml(self, mock_config, orch_paths):
        """Smoke check: phase reaches param.nml processor (already wired)."""
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="forecast")
        assert result.phase == "forecast"

    def test_run_ufs_config_signature_accepts_phase(self, mock_config, tmp_path):
        """``_run_ufs_config`` must accept the phase kwarg (regression for
        the orchestrator extension to pass phase to UFSConfigProcessor)."""
        import inspect
        orch = PrepOrchestrator(mock_config, {"output": str(tmp_path)})
        sig = inspect.signature(orch._run_ufs_config)
        assert "phase" in sig.parameters

    def test_run_nudging_passes_phase(self, mock_config, tmp_path,
                                       monkeypatch):
        """NudgingProcessor instance receives the orchestrator's phase."""
        from nos_utils.forcing.nudging import NudgingProcessor

        captured = {}

        original_init = NudgingProcessor.__init__

        def spy_init(self, *args, **kwargs):
            captured["phase"] = kwargs.get("phase")
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(NudgingProcessor, "__init__", spy_init)

        mock_config.nudging_enabled = True
        orch = PrepOrchestrator(
            mock_config, {"output": str(tmp_path / "out"),
                          "fix": str(tmp_path / "fix"),
                          "rtofs": str(tmp_path / "rtofs")},
        )
        (tmp_path / "fix").mkdir()
        (tmp_path / "rtofs").mkdir()
        # nudging gate requires rtofs path + nudging_enabled — both set.
        orch._run_nudging(tmp_path / "out", phase="forecast")
        assert captured["phase"] == "forecast"

    def test_run_ufs_config_passes_phase(self, mock_config, tmp_path,
                                          monkeypatch):
        """UFSConfigProcessor instance receives the orchestrator's phase."""
        from nos_utils.forcing.ufs_config import UFSConfigProcessor

        captured = {}
        original_init = UFSConfigProcessor.__init__

        def spy_init(self, *args, **kwargs):
            captured["phase"] = kwargs.get("phase")
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(UFSConfigProcessor, "__init__", spy_init)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        fix_dir = tmp_path / "fix"
        fix_dir.mkdir()
        orch = PrepOrchestrator(
            mock_config, {"output": str(out_dir), "fix": str(fix_dir)},
        )
        orch._run_ufs_config(out_dir, phase="nowcast")
        assert captured["phase"] == "nowcast"


class TestArchiveManifest:
    """archive_to_comout manifest refactor + flag gating (P2).

    Invariants under test:

      * Flag OFF (default) runs the legacy hardcoded blocks; the $COMOUT
        file set is unchanged.
      * Flag ON + SECOFS-shaped config (st_lawrence_enabled=False,
        obc_min_timesteps=0) produces *exactly* the same $COMOUT file set
        as flag OFF — the byte-identical guarantee.
      * Flag ON + STOFS-shaped config additionally emits the St. Lawrence
        individual files (riv.obs.flux.th / riv.obs.tem_1.th) and the
        four OBC-QC archive artifacts, without altering the COMMON tars.
    """

    # The COMMON tar names every flag/config combination must produce
    # when all payloads exist. {prefix}=stofs_3d_atl / secofs_ufs,
    # cycle=t12z, pdy=20260401.
    _OBC_PAYLOAD = ["elev2D.th.nc", "TEM_3D.th.nc", "SAL_3D.th.nc",
                    "uv3D.th.nc", "TEM_nu.nc", "SAL_nu.nc"]
    _RIVER_PAYLOAD = ["schism_flux.th", "schism_temp.th", "schism_salt.th"]
    _NWM_PAYLOAD = ["vsource.th", "msource.th", "source_sink.in",
                    "vsink.th"]

    def _seed_workdir(self, work: Path) -> None:
        """Drop every payload file the COMMON + EXTRA entries reference."""
        work.mkdir(parents=True, exist_ok=True)
        sflux = work / "sflux"
        sflux.mkdir(exist_ok=True)
        # Met (unchanged path) — GFS stack 1 + HRRR stack 2.
        (sflux / "sflux_air_1.0001.nc").write_bytes(b"gfs")
        (sflux / "sflux_air_2.0001.nc").write_bytes(b"hrrr")
        for name in (self._OBC_PAYLOAD + self._RIVER_PAYLOAD
                     + self._NWM_PAYLOAD):
            (work / name).write_bytes(b"x")
        # St. Lawrence outputs (only consumed when st_lawrence_enabled).
        (work / "flux.th").write_text("0 -1.0\n")
        (work / "TEM_1.th").write_text("0 4.0\n")
        # Misc copy_map / marker files (identical in both paths).
        (work / "param.nml").write_text("&CORE\n/\n")
        (work / "bctides.in").write_text("tides\n")

    def _archive(self, cfg, run_name, tmp_path, monkeypatch,
                 manifest):
        """Run archive_to_comout once and return the set of basenames
        written under $COMOUT (recursively, so datm_input/* is included).
        """
        work = tmp_path / "work"
        comout = tmp_path / "comout"
        self._seed_workdir(work)
        if manifest is None:
            monkeypatch.delenv("NOS_ARCHIVE_MANIFEST", raising=False)
        else:
            monkeypatch.setenv("NOS_ARCHIVE_MANIFEST", manifest)
        orch = PrepOrchestrator(
            cfg, {"output": str(work)}, run_name=run_name,
            skip_legacy=True,
        )
        result = PrepResult(success=True, phase="nowcast")
        orch.archive_to_comout(result, comout)
        return {
            p.relative_to(comout).as_posix()
            for p in comout.rglob("*") if p.is_file()
        }

    def _secofs_cfg(self):
        return ForcingConfig.for_secofs_ufs(pdy="20260401", cyc=12)

    def _stofs_cfg(self):
        return ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)

    def test_flag_off_secofs_baseline_fileset(self, tmp_path,
                                              monkeypatch):
        """Flag OFF + SECOFS: the canonical legacy $COMOUT file set."""
        files = self._archive(self._secofs_cfg(), "secofs_ufs",
                              tmp_path, monkeypatch, manifest=None)
        # COMMON tars + met tars + copied files. No St. Lawrence /
        # OBC-QC artifacts (SECOFS config disables both).
        expected = {
            "secofs_ufs.t12z.20260401.met.nowcast.nc.tar",
            "secofs_ufs.t12z.20260401.met.nowcast.nc.2.tar",
            "secofs_ufs.t12z.20260401.obc.nowcast.tar",
            "secofs_ufs.t12z.20260401.obc.tar",
            "secofs_ufs.t12z.20260401.river.th.tar",
            "secofs_ufs.t12z.20260401.nwm.source.sink.now.tar",
            "secofs_ufs.t12z.20260401.nowcast.in",
            "secofs_ufs.t12z.20260401.bctides.in.nowcast",
            "secofs_ufs.source_sink.in",
            "secofs_ufs.t12z.20260401.river.vsource.th",
            "secofs_ufs.t12z.20260401.river.msource.th",
            "secofs_ufs.t12z.20260401.inputs.prep.json",
        }
        assert files == expected
        # Negative: no St. Lawrence / OBC-QC artifacts under OFF.
        assert not any("riv.obs" in f for f in files)
        assert not any(f.endswith(".elev2dth_non_adj.nc")
                       for f in files)

    def test_flag_on_secofs_identical_to_flag_off(self, tmp_path,
                                                  monkeypatch):
        """Byte-identical guarantee: flag ON + SECOFS-shaped config must
        yield *exactly* the same $COMOUT file set as flag OFF."""
        off = self._archive(self._secofs_cfg(), "secofs_ufs",
                            tmp_path / "off", monkeypatch,
                            manifest=None)
        on = self._archive(self._secofs_cfg(), "secofs_ufs",
                           tmp_path / "on", monkeypatch,
                           manifest="YES")
        assert on == off, (
            "manifest ON with SECOFS config diverged from OFF: "
            f"only-ON={on - off} only-OFF={off - on}"
        )

    def test_flag_on_secofs_tar_payloads_match_off(self, tmp_path,
                                                   monkeypatch):
        """Not just names — the COMMON tar payloads must match too."""
        import tarfile

        def _payloads(root: Path):
            work = root / "work"
            comout = root / "comout"
            self._seed_workdir(work)
            orch = PrepOrchestrator(
                self._secofs_cfg(), {"output": str(work)},
                run_name="secofs_ufs", skip_legacy=True,
            )
            orch.archive_to_comout(
                PrepResult(success=True, phase="nowcast"), comout,
            )
            out = {}
            for tar in sorted(comout.glob("*.tar")):
                with tarfile.open(tar) as tf:
                    out[tar.name] = sorted(tf.getnames())
            return out

        monkeypatch.delenv("NOS_ARCHIVE_MANIFEST", raising=False)
        off = _payloads(tmp_path / "off")
        monkeypatch.setenv("NOS_ARCHIVE_MANIFEST", "1")
        on = _payloads(tmp_path / "on")
        assert on == off, f"tar payloads diverged: OFF={off} ON={on}"

    def test_flag_on_stofs_adds_st_lawrence_and_qc(self, tmp_path,
                                                   monkeypatch):
        """Flag ON + STOFS-shaped config: COMMON set is the SECOFS set
        (under the stofs_3d_atl prefix) PLUS the St. Lawrence individual
        files and the four OBC-QC archive artifacts — additive only."""
        files = self._archive(self._stofs_cfg(), "stofs_3d_atl",
                              tmp_path, monkeypatch, manifest="TRUE")

        common = {
            "stofs_3d_atl.t12z.20260401.met.nowcast.nc.tar",
            "stofs_3d_atl.t12z.20260401.met.nowcast.nc.2.tar",
            "stofs_3d_atl.t12z.20260401.obc.nowcast.tar",
            "stofs_3d_atl.t12z.20260401.obc.tar",
            "stofs_3d_atl.t12z.20260401.river.th.tar",
            "stofs_3d_atl.t12z.20260401.nwm.source.sink.now.tar",
            "stofs_3d_atl.t12z.20260401.nowcast.in",
            "stofs_3d_atl.t12z.20260401.bctides.in.nowcast",
            "stofs_3d_atl.source_sink.in",
            "stofs_3d_atl.t12z.20260401.river.vsource.th",
            "stofs_3d_atl.t12z.20260401.river.msource.th",
            "stofs_3d_atl.t12z.20260401.inputs.prep.json",
        }
        st_lawrence = {
            "stofs_3d_atl.t12z.riv.obs.flux.th",
            "stofs_3d_atl.t12z.riv.obs.tem_1.th",
        }
        obc_qc = {
            "stofs_3d_atl.t12z.elev2dth_non_adj.nc",
            "stofs_3d_atl.t12z.tem3dth.nc",
            "stofs_3d_atl.t12z.sal3dth.nc",
            "stofs_3d_atl.t12z.uv3dth.nc",
        }
        assert common.issubset(files), (
            f"missing COMMON entries: {common - files}"
        )
        assert st_lawrence.issubset(files), (
            f"missing St. Lawrence files: {st_lawrence - files}"
        )
        assert obc_qc.issubset(files), (
            f"missing OBC-QC artifacts: {obc_qc - files}"
        )
        # Exactly COMMON + extras, nothing stray.
        assert files == common | st_lawrence | obc_qc

    def test_flag_on_stofs_river_tar_not_clobbered_by_st_lawrence(
            self, tmp_path, monkeypatch):
        """St. Lawrence flux.th/TEM_1.th must NOT be folded into
        river.th.tar (whose schism_*.th payload is renamed to
        flux.th/TEM_1.th run-side and would clobber them)."""
        import tarfile

        files = self._archive(self._stofs_cfg(), "stofs_3d_atl",
                              tmp_path, monkeypatch, manifest="YES")
        assert ("stofs_3d_atl.t12z.20260401.river.th.tar" in files)
        river_tar = (tmp_path / "comout"
                     / "stofs_3d_atl.t12z.20260401.river.th.tar")
        with tarfile.open(river_tar) as tf:
            names = set(tf.getnames())
        assert names == set(self._RIVER_PAYLOAD), (
            f"river.th.tar payload polluted: {names}"
        )
        assert "flux.th" not in names and "TEM_1.th" not in names

    def test_invalid_flag_values_treated_as_off(self, tmp_path,
                                                monkeypatch):
        """Only YES/1/TRUE (case-insensitive) enable the manifest;
        anything else (NO, 0, garbage) stays OFF."""
        for val in ("NO", "0", "false", "off", "maybe", ""):
            monkeypatch.setenv("NOS_ARCHIVE_MANIFEST", val)
            assert PrepOrchestrator._archive_manifest_enabled() is False
        for val in ("YES", "yes", "1", "TRUE", "true", "True"):
            monkeypatch.setenv("NOS_ARCHIVE_MANIFEST", val)
            assert PrepOrchestrator._archive_manifest_enabled() is True


class TestPrepResult:
    def test_all_output_files(self):
        from nos_utils.forcing.base import ForcingResult
        r = PrepResult(
            success=True, phase="nowcast",
            results=[
                ForcingResult(success=True, source="GFS",
                             output_files=[Path("/a.nc"), Path("/b.nc")]),
                ForcingResult(success=True, source="TIDAL",
                             output_files=[Path("/c.in")]),
            ],
        )
        assert len(r.all_output_files) == 3

    def test_all_errors(self):
        from nos_utils.forcing.base import ForcingResult
        r = PrepResult(
            success=False, phase="nowcast",
            results=[
                ForcingResult(success=False, source="GFS", errors=["no files"]),
                ForcingResult(success=True, source="TIDAL"),
            ],
        )
        assert r.all_errors == ["no files"]
