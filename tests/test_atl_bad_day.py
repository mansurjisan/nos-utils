"""STOFS-3D-ATL bad-day checks (production v3.1.5): restart gate, OBC inputs, tide_fac,
pre-adjust archive, COMOUT_PREV reuse, NWM vsource backup. SECOFS must not change."""

import os
from datetime import datetime
from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nwm import NWMProcessor
from nos_utils.forcing.tidal import TidalProcessor
from nos_utils.orchestrator import PrepOrchestrator, PrepResult

from .test_stofs_obc_qc import _write_obc_file
from .test_stofs_river_ops import _cfg as _river_cfg, _fix_dir, _stage_nwm

netCDF4 = pytest.importorskip("netCDF4")

RUN = "stofs_3d_atl_ufs"


def _atl(**kw):
    kw.setdefault("restart_min_bytes", 100)
    return ForcingConfig.for_stofs_3d_atl(pdy="20261001", cyc=12, **kw)


def _restart(root, day, name=None, size_pad=0, fmt="NETCDF4"):
    d = root / f"{RUN}.{day}"
    d.mkdir(parents=True, exist_ok=True)
    f = d / (name or f"{RUN}.t12z.{day}.rst.nowcast.nc")
    with netCDF4.Dataset(str(f), "w", format=fmt) as ds:
        ds.createDimension("one", 1)
        t = ds.createVariable("time", "f8", ("one",))
        t[:] = 0.0
        if size_pad:
            ds.createDimension("pad", size_pad)
            ds.createVariable("pad", "f8", ("pad",))[:] = 0.0
    return f


def _orch(tmp_path, cfg, **paths):
    p = {"output": str(tmp_path / "work"), "comout": str(tmp_path / "com" / f"{RUN}.20261001"),
         "restart": str(tmp_path / "com")}
    p.update({k: str(v) for k, v in paths.items()})
    return PrepOrchestrator(cfg, p, run_name=RUN, skip_legacy=True)


class TestHotstartGate:
    def test_no_restart_fails_prep_stage(self, tmp_path):
        res = _orch(tmp_path, _atl())._run_hotstart(tmp_path / "work")
        assert res.success is False
        assert "RESTART FILE NOT FOUND" in res.errors[0]

    def test_restart_inside_window_passes(self, tmp_path):
        _restart(tmp_path / "com", "20260927")  # PDY-4
        res = _orch(tmp_path, _atl())._run_hotstart(tmp_path / "work")
        assert res.success and res.metadata["ihot"] == 1

    def test_restart_in_pdy_minus_5_passes_and_minus_6_fails(self, tmp_path):
        _restart(tmp_path / "com", "20260926")
        assert _orch(tmp_path, _atl())._run_hotstart(tmp_path / "w1").success
        other = tmp_path / "other"
        _restart(other, "20260925")
        res = _orch(tmp_path, _atl(), restart=other, comout=tmp_path / "c2" / "x")._run_hotstart(
            tmp_path / "w2")
        assert res.success is False

    def test_too_small_restart_fails(self, tmp_path):
        _restart(tmp_path / "com", "20260930")
        res = _orch(tmp_path, _atl(restart_min_bytes=10 ** 9))._run_hotstart(tmp_path / "work")
        assert res.success is False

    def test_seeded_init_file_passes(self, tmp_path):
        seed = _restart(tmp_path / "com", "20261001", name=f"{RUN}.t12z.20261001.init.nowcast.nc",
                        fmt="NETCDF4_CLASSIC")
        orch = _orch(tmp_path, _atl())
        res = orch._run_hotstart(tmp_path / "work")
        assert res.success, res.errors
        assert seed.exists()

    def test_small_seed_alone_fails(self, tmp_path):
        _restart(tmp_path / "com", "20261001", name=f"{RUN}.t12z.20261001.init.nowcast.nc",
                        fmt="NETCDF4_CLASSIC")
        res = _orch(tmp_path, _atl(restart_min_bytes=10 ** 9))._run_hotstart(tmp_path / "work")
        assert res.success is False

    def test_coldstart_yes_refused(self, tmp_path, monkeypatch):
        _restart(tmp_path / "com", "20260930")
        monkeypatch.setenv("COLDSTART", "YES")
        res = _orch(tmp_path, _atl())._run_hotstart(tmp_path / "work")
        assert res.success is False and "COLDSTART=YES" in res.errors[0]

    def test_coldstart_no_is_fine(self, tmp_path, monkeypatch):
        _restart(tmp_path / "com", "20260930")
        monkeypatch.setenv("COLDSTART", "NO")
        assert _orch(tmp_path, _atl())._run_hotstart(tmp_path / "work").success

    def test_secofs_unchanged_without_restart_or_with_coldstart(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COLDSTART", "YES")
        cfg = ForcingConfig.for_secofs_ufs(pdy="20261001", cyc=12)
        assert cfg.ops_bad_day_checks is False
        res = _orch(tmp_path, cfg)._run_hotstart(tmp_path / "work")
        assert res.success and res.metadata["ihot"] == 0

    def test_restart_root_follows_overridden_comout(self, tmp_path):
        # COMOUT=<root>/<run>.PDY: the processor also searches the parent, so an override
        # of COMOUT alone reaches the restart in the sibling PDY-1 dir.
        root = tmp_path / "alt"
        _restart(root, "20260930")
        res = _orch(tmp_path, _atl(), restart=root / f"{RUN}.20261001",
                    comout=root / f"{RUN}.20261001")._run_hotstart(tmp_path / "work")
        assert res.success and res.metadata["ihot"] == 1


def _obc_dir(tmp_path, names=None):
    out = tmp_path / "work"
    full = list(PrepOrchestrator._OPS_RUN_INPUTS) + ["sflux/sflux_inputs.txt"] + [
        f"sflux/sflux_{v}_{k}.0001.nc" for v in ("air", "prc", "rad") for k in (1, 2)]
    for n in names or full:
        (out / n).parent.mkdir(parents=True, exist_ok=True)
        (out / n).write_bytes(b"x")
    return out


class TestObcInputsGate:
    def test_all_present_passes(self, tmp_path):
        out = _obc_dir(tmp_path)
        assert _orch(tmp_path, _atl())._check_ops_obc_inputs(out).success

    @pytest.mark.parametrize("gone", [
        "elev2D.th.nc", "TEM_3D.th.nc", "SAL_3D.th.nc", "uv3D.th.nc", "TEM_nu.nc", "SAL_nu.nc",
        "param.nml", "bctides.in", "msource.th", "vsink.th", "vsource.th", "flux.th", "TEM_1.th",
        "sflux/sflux_inputs.txt", "sflux/sflux_air_1.0001.nc", "sflux/sflux_prc_1.0001.nc",
        "sflux/sflux_rad_1.0001.nc", "sflux/sflux_air_2.0001.nc", "sflux/sflux_prc_2.0001.nc",
        "sflux/sflux_rad_2.0001.nc"])
    def test_missing_file_fails(self, tmp_path, gone):
        out = _obc_dir(tmp_path)
        (out / gone).unlink()
        res = _orch(tmp_path, _atl())._check_ops_obc_inputs(out)
        assert not res.success
        assert gone.split("/")[-1].replace("0001", "*") in res.errors[0]

    def test_empty_file_fails(self, tmp_path):
        out = _obc_dir(tmp_path)
        (out / "TEM_nu.nc").write_bytes(b"")
        assert not _orch(tmp_path, _atl())._check_ops_obc_inputs(out).success

    def _run(self, tmp_path, monkeypatch, cfg, with_obc):
        from nos_utils.forcing.base import ForcingResult
        orch = _orch(tmp_path, cfg)
        ok = lambda src: (lambda *a, **k: ForcingResult(success=True, source=src))  # noqa: E731
        monkeypatch.setattr(orch, "_run_hotstart", ok("HOTSTART"))
        monkeypatch.setattr(orch, "_run_tidal", ok("TIDAL"))
        monkeypatch.setattr(orch, "_run_param_nml", ok("PARAM_NML"))
        monkeypatch.setattr(orch, "_run_datm", ok("DATM"))
        monkeypatch.setattr(orch, "_run_ufs_config", ok("UFS_CONFIG"))
        (tmp_path / "work").mkdir(exist_ok=True)
        if with_obc:
            _obc_dir(tmp_path)
        return orch.run(phase="nowcast")

    def test_prep_fails_without_obc_files(self, tmp_path, monkeypatch):
        cfg = _atl(st_lawrence_enabled=False)
        assert self._run(tmp_path, monkeypatch, cfg, with_obc=False).success is False

    def test_prep_passes_with_obc_files(self, tmp_path, monkeypatch):
        cfg = _atl(st_lawrence_enabled=False)
        assert self._run(tmp_path, monkeypatch, cfg, with_obc=True).success is True

    def test_non_atl_prep_ignores_missing_obc(self, tmp_path, monkeypatch):
        cfg = ForcingConfig.for_secofs_ufs(pdy="20261001", cyc=12)
        assert self._run(tmp_path, monkeypatch, cfg, with_obc=False).success is True


class TestTideFacRequired:
    def _fix(self, tmp_path):
        fix = tmp_path / "fix"
        fix.mkdir(exist_ok=True)
        (fix / "sys.bctides.in_template").write_text("T\n")
        return fix

    def _clear_env(self, monkeypatch):
        for v in ("EXECnos", "EXECofs", "EXECstofs3d"):
            monkeypatch.delenv(v, raising=False)

    def test_missing_exe_fails_for_atl(self, tmp_path, monkeypatch):
        self._clear_env(monkeypatch)
        res = TidalProcessor(_atl(), self._fix(tmp_path), tmp_path / "w").process()
        assert res.success is False and "Fortran tide_fac" in res.errors[0]
        assert not (tmp_path / "w" / "bctides.in").exists()

    def test_failing_exe_fails_for_atl(self, tmp_path, monkeypatch):
        exe_dir = tmp_path / "exec"
        exe_dir.mkdir()
        exe = exe_dir / "stofs_3d_atl_tide_fac"
        exe.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
        exe.chmod(0o755)
        self._clear_env(monkeypatch)
        monkeypatch.setenv("EXECstofs3d", str(exe_dir))
        res = TidalProcessor(_atl(), self._fix(tmp_path), tmp_path / "w").process()
        assert res.success is False and "returned 3" in res.errors[0]

    def test_working_exe_passes_for_atl(self, tmp_path, monkeypatch):
        exe_dir = tmp_path / "exec"
        exe_dir.mkdir()
        exe = exe_dir / "stofs_3d_atl_tide_fac"
        exe.write_text("#!/bin/sh\ncat > /dev/null\ncp bctides.in_template bctides.in\n")
        exe.chmod(0o755)
        self._clear_env(monkeypatch)
        monkeypatch.setenv("EXECstofs3d", str(exe_dir))
        res = TidalProcessor(_atl(), self._fix(tmp_path), tmp_path / "w").process()
        assert res.success and res.metadata["mode"] == "fortran_tide_fac"

    def test_other_systems_keep_the_python_fallback(self, tmp_path, monkeypatch):
        self._clear_env(monkeypatch)
        cfg = _atl(ops_bad_day_checks=False)
        res = TidalProcessor(cfg, self._fix(tmp_path), tmp_path / "w").process()
        assert res.success and res.metadata["mode"] == "template"


class TestPreAdjustArchive:
    def _work(self, tmp_path, elev_nt=25):
        work = tmp_path / "work"
        _write_obc_file(work / "elev2D.th.nc", nt=elev_nt)
        for n in ("TEM_3D.th.nc", "SAL_3D.th.nc", "uv3D.th.nc"):
            _write_obc_file(work / n, nt=25, is_3d=True)
        return work

    def _elev_mark(self, path, val):
        with netCDF4.Dataset(str(path), "r+") as ds:
            ds.variables["time_series"][:] = val

    def _elev_val(self, path):
        with netCDF4.Dataset(str(path)) as ds:
            return float(ds.variables["time_series"][:].flat[0])

    def test_archive_is_pre_adjust_and_in_rerun(self, tmp_path):
        work = self._work(tmp_path)
        self._elev_mark(work / "elev2D.th.nc", 1.0)
        orch = _orch(tmp_path, _atl())
        orch._snapshot_pre_adjust(work)
        self._elev_mark(work / "elev2D.th.nc", 9.0)  # the in-place dynamic adjust
        comout = tmp_path / "com" / f"{RUN}.20261001"
        arch = []
        orch._archive_bad_day_rerun(work, comout, "nowcast", arch)
        rerun = comout / "rerun"
        assert self._elev_val(rerun / f"{RUN}.t12z.elev2dth_non_adj.nc") == 1.0
        for base in ("tem3dth", "sal3dth", "uv3dth"):
            assert (rerun / f"{RUN}.t12z.{base}.nc").exists()
        assert not list(comout.glob("*.elev2dth_non_adj.nc"))

    def test_forecast_phase_does_not_overwrite_nowcast_archive(self, tmp_path):
        work = self._work(tmp_path)
        orch = _orch(tmp_path, _atl())
        comout = tmp_path / "com" / f"{RUN}.20261001"
        self._elev_mark(work / "elev2D.th.nc", 1.0)
        orch._archive_bad_day_rerun(work, comout, "nowcast", [])
        self._elev_mark(work / "elev2D.th.nc", 2.0)
        orch._archive_bad_day_rerun(work, comout, "forecast", [])
        rerun = comout / "rerun"
        assert self._elev_val(rerun / f"{RUN}.t12z.elev2dth_non_adj.nc") == 1.0
        assert self._elev_val(rerun / f"{RUN}.t12z.elev2dth_non_adj.fcst.nc") == 2.0

    def test_full_archive_leaves_no_adjusted_copy_at_comout_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("NOS_ARCHIVE_MANIFEST", "YES")
        work = self._work(tmp_path)
        orch = _orch(tmp_path, _atl())
        comout = tmp_path / "com" / f"{RUN}.20261001"
        orch.archive_to_comout(PrepResult(success=True, phase="nowcast"), comout)
        assert not (comout / f"{RUN}.t12z.elev2dth_non_adj.nc").exists()
        assert (comout / "rerun" / f"{RUN}.t12z.elev2dth_non_adj.nc").exists()

    def test_qc_reuse_reads_pre_adjust_from_comout_prev(self, tmp_path):
        work = self._work(tmp_path, elev_nt=5)
        prev = tmp_path / "com" / f"{RUN}.20260930"
        _write_obc_file(prev / "rerun" / f"{RUN}.t12z.elev2dth_non_adj.nc", nt=25)
        # An adjusted copy under the active name must not be picked up.
        _write_obc_file(prev / "rerun" / "elev2D.th.nc", nt=25)
        self._elev_mark(prev / "rerun" / "elev2D.th.nc", 7.0)
        self._elev_mark(prev / "rerun" / f"{RUN}.t12z.elev2dth_non_adj.nc", 1.0)
        orch = _orch(tmp_path, _atl(), prev_comout=prev)
        res = orch._qc_obc_dimensions(work)
        assert res is not None and res.success
        assert self._elev_val(work / "elev2D.th.nc") == 1.0

    def test_qc_does_not_fall_back_to_adjusted_active_name(self, tmp_path):
        work = self._work(tmp_path, elev_nt=5)
        prev = tmp_path / "com" / f"{RUN}.20260930"
        _write_obc_file(prev / "rerun" / "elev2D.th.nc", nt=25)
        res = _orch(tmp_path, _atl(), prev_comout=prev)._qc_obc_dimensions(work)
        assert res is not None and not res.success

    def test_qc_forecast_phase_reads_fcst_archive(self, tmp_path):
        work = self._work(tmp_path, elev_nt=5)
        prev = tmp_path / "com" / f"{RUN}.20260930"
        _write_obc_file(prev / "rerun" / f"{RUN}.t12z.elev2dth_non_adj.fcst.nc", nt=25)
        res = _orch(tmp_path, _atl(), prev_comout=prev)._qc_obc_dimensions(work, phase="forecast")
        assert res is not None and res.success

    def test_secofs_writes_no_rerun_archives(self, tmp_path):
        work = self._work(tmp_path)
        cfg = ForcingConfig.for_secofs_ufs(pdy="20261001", cyc=12)
        comout = tmp_path / "com" / "secofs_ufs.20261001"
        arch = []
        _orch(tmp_path, cfg)._archive_bad_day_rerun(work, comout, "nowcast", arch)
        assert arch == [] and not (comout / "rerun").exists()


class TestPrevCycleResolution:
    def test_comin_rerun_override_wins(self, tmp_path):
        orch = _orch(tmp_path, _atl(), prev_rerun=tmp_path / "hand", prev_comout=tmp_path / "prev")
        assert orch._prev_rerun() == tmp_path / "hand"

    def test_comout_prev_rerun_is_default_for_atl(self, tmp_path):
        orch = _orch(tmp_path, _atl(), prev_comout=tmp_path / "prev")
        assert orch._prev_rerun() == tmp_path / "prev" / "rerun"

    def test_unset_gives_none(self, tmp_path):
        assert _orch(tmp_path, _atl())._prev_rerun() is None

    def test_secofs_does_not_use_comout_prev(self, tmp_path):
        cfg = ForcingConfig.for_secofs_ufs(pdy="20261001", cyc=12)
        assert _orch(tmp_path, cfg, prev_comout=tmp_path / "prev")._prev_rerun() is None

    def test_st_lawrence_gets_comout_prev_rerun(self, tmp_path, monkeypatch):
        import nos_utils.forcing.st_lawrence as sl
        seen = {}

        class Fake:
            def __init__(self, *a, **k):
                seen.update(k)

            def process(self):
                return None

        monkeypatch.setattr(sl, "StLawrenceProcessor", Fake)
        _orch(tmp_path, _atl(), prev_comout=tmp_path / "prev")._run_st_lawrence(tmp_path / "work")
        assert seen["prev_rerun_dir"] == tmp_path / "prev" / "rerun"

    def test_st_lawrence_archive_reaches_next_cycle(self, tmp_path):
        from nos_utils.forcing.st_lawrence import StLawrenceProcessor
        work = tmp_path / "work"
        work.mkdir()
        flux = "".join(f"{i * 86400} -1000.000\n" for i in range(6))
        (work / "flux.th").write_text(flux)
        (work / "TEM_1.th").write_text("".join(f"{i * 86400} 5.000\n" for i in range(6)))
        comout = tmp_path / "com" / f"{RUN}.20261001"
        _orch(tmp_path, _atl())._archive_bad_day_rerun(work, comout, "nowcast", [])
        assert (comout / "rerun" / f"{RUN}.t12z.riv.obs.flux.th").read_text() == flux
        (tmp_path / "o2").mkdir()
        proc = StLawrenceProcessor(_atl(), tmp_path / "in", tmp_path / "o2",
                                   prev_rerun_dir=comout / "rerun", archive_prefix=f"{RUN}.t12z")
        assert proc._fallback_from_archive("flux.th") is not None


def _prev_vsource(rerun, name, n):
    rerun.mkdir(parents=True, exist_ok=True)
    (rerun / name).write_text("".join(f"{i * 3600} {i}.0 {i}.5 {i}.9\n" for i in range(n)))


class TestNwmMinimum:
    def _proc(self, tmp_path, phase, prev=None, bad_day=True, **kw):
        fix = _fix_dir(tmp_path)
        _stage_nwm(tmp_path, n_hours=6)
        return NWMProcessor(_river_cfg(fix, ops_bad_day_checks=bad_day, **kw), tmp_path / "nwm",
                            tmp_path / "out", phase=phase, prev_rerun_dir=prev,
                            archive_prefix=f"{RUN}.t12z")

    def test_short_list_without_backup_fails(self, tmp_path):
        res = self._proc(tmp_path, "nowcast", prev=tmp_path / "rerun").process()
        assert res.success is False and "nwm_n_list_min" in res.errors[0]

    def test_nowcast_uses_yesterdays_forecast_file_unshifted(self, tmp_path):
        _prev_vsource(tmp_path / "rerun", f"{RUN}.t12z.vsource.fcst.th", 60)
        res = self._proc(tmp_path, "nowcast", prev=tmp_path / "rerun").process()
        assert res.success and res.metadata["vsource_from_previous_cycle"] is True
        rows = (tmp_path / "out" / "vsource.th").read_text().splitlines()
        assert rows[0].split()[:2] == ["0", "0.0"]
        assert rows[2].split()[:2] == ["7200", "2.0"]
        assert len(rows) == 2 + 3 + 1  # nowcast_hours 2 + buffer + 1

    def test_forecast_drops_24_rows_and_pads_with_last(self, tmp_path):
        _prev_vsource(tmp_path / "rerun", f"{RUN}.t12z.vsource.fcst.th", 30)
        res = self._proc(tmp_path, "forecast", prev=tmp_path / "rerun", forecast_hours=8).process()
        assert res.success
        rows = (tmp_path / "out" / "vsource.th").read_text().splitlines()
        assert len(rows) == 12
        assert rows[0].split() == ["0", "24.0", "24.5", "24.9"]
        assert rows[5].split() == ["18000", "29.0", "29.5", "29.9"]
        assert rows[6].split()[1:] == ["29.0", "29.5", "29.9"]  # padded with the last row
        assert [int(r.split()[0]) for r in rows] == [i * 3600 for i in range(len(rows))]

    def test_nowcast_falls_back_to_nowcast_file_shifted(self, tmp_path):
        _prev_vsource(tmp_path / "rerun", f"{RUN}.t12z.vsource.th", 40)
        res = self._proc(tmp_path, "nowcast", prev=tmp_path / "rerun").process()
        rows = (tmp_path / "out" / "vsource.th").read_text().splitlines()
        assert rows[0].split()[1] == "24.0" and res.success

    def test_gate_off_keeps_legacy_behaviour(self, tmp_path):
        res = self._proc(tmp_path, "nowcast", bad_day=False).process()
        assert res.success and "vsource_from_previous_cycle" not in res.metadata or \
            res.metadata["vsource_from_previous_cycle"] is False


class TestConfigFlag:
    def test_factories(self):
        assert ForcingConfig.for_stofs_3d_atl("20261001", 12).ops_bad_day_checks is True
        assert ForcingConfig.for_stofs_3d_atl_ufs("20261001", 12).ops_bad_day_checks is True
        assert ForcingConfig.for_stofs_3d_atl("20261001", 12).restart_min_bytes == 20 * 1024 ** 3
        for f in ("for_secofs_ufs", "for_stofs_3d_pac", "for_stofs_3d_pac_ufs"):
            assert getattr(ForcingConfig, f)("20261001", 12).ops_bad_day_checks is False

    def _yaml(self, tmp_path, name, extra=""):
        pytest.importorskip("yaml")
        y = tmp_path / f"{name}.yaml"
        y.write_text(f"system:\n  name: {name}\n{extra}")
        return ForcingConfig.from_yaml(y, pdy="20261001", cyc=12)

    def test_yaml_by_name(self, tmp_path):
        assert self._yaml(tmp_path, "stofs_3d_atl_ufs").ops_bad_day_checks is True
        assert self._yaml(tmp_path, "secofs_ufs").ops_bad_day_checks is False
        assert self._yaml(tmp_path, "stofs_3d_ak_ufs").ops_bad_day_checks is False

    def test_yaml_override(self, tmp_path):
        cfg = self._yaml(tmp_path, "stofs_3d_atl_ufs",
                         "prep:\n  ops_bad_day_checks: false\n  restart_min_bytes: 5\n")
        assert cfg.ops_bad_day_checks is False and cfg.restart_min_bytes == 5
