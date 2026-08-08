"""Tests for WaveBoundaryProcessor (GFS-Wave boundary spectra -> nest.ww3).

Fixtures under tests/data/wave_boundary/ are trimmed real-product samples:
  - buoys_subset.txt: ~40 data lines cut from the operational
    wave_gfs.buoys points file (Gulf-of-Alaska + North Pacific/Bering DAT
    buoys, including the 3 AK extras 46035/46070/46071, plus a Bermuda
    IBP block as non-AK noise). Note: the real file carries NO TYPE==IBP
    points in the Alaska region at all -- that's why 46070/46071/46035
    must be pulled in via extra_points rather than the window/TYPE filter.
  - gfswave.BER51.spec: header + first 2 complete time records cut from a
    live gfswave.<NAME>.spec sample.
"""

import io
import json
import shutil
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from nos_utils.config import ForcingConfig
from nos_utils.orchestrator import PrepOrchestrator
from nos_utils.forcing.wave_boundary import (
    WaveBoundaryProcessor,
    parse_ww3_points_file,
    _point_in_window,
)

FIXTURE_DIR = Path(__file__).parent / "data" / "wave_boundary"
BUOYS_SUBSET = FIXTURE_DIR / "buoys_subset.txt"
SPEC_SAMPLE = FIXTURE_DIR / "gfswave.BER51.spec"

HAS_TAR = shutil.which("tar") is not None


@pytest.fixture
def orch_paths(tmp_path):
    """Minimal orchestrator directory structure (mirrors test_orchestrator.py)."""
    paths = {
        "output": str(tmp_path / "work"),
        "fix": str(tmp_path / "fix"),
    }
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


def _touch(path: Path, content: bytes = b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _make_tar(tar_path: Path, members: dict, gz: bool = False):
    """Build a tiny synthetic tar. Values are either a Path (file added by
    content) or a str (written as literal member content)."""
    mode = "w:gz" if gz else "w"
    with tarfile.open(tar_path, mode) as tf:
        for arcname, src in members.items():
            if isinstance(src, Path):
                tf.add(src, arcname=arcname)
            else:
                data = src.encode()
                info = tarfile.TarInfo(name=arcname)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


# --------------------------------------------------------------- parsing

class TestParsePointsFile:
    def test_parses_real_subset(self):
        points = parse_ww3_points_file(BUOYS_SUBSET)
        assert len(points) == 43
        names = {p.name for p in points}
        assert {"46035", "46070", "46071"}.issubset(names)

    def test_ak_dat_buoys_are_type_dat_not_ibp(self):
        """The 3 AK extras are real NDBC DAT buoys, not IBP boundary points --
        this is exactly why they need extra_points instead of the window/TYPE
        filter (see module docstring)."""
        points = {p.name: p for p in parse_ww3_points_file(BUOYS_SUBSET)}
        for name in ("46035", "46070", "46071"):
            assert points[name].point_type == "DAT"

    def test_bermuda_points_are_ibp(self):
        points = parse_ww3_points_file(BUOYS_SUBSET)
        ber = [p for p in points if p.name.startswith("BER")]
        assert len(ber) == 24
        assert all(p.point_type == "IBP" for p in ber)

    def test_section_marker_comments_are_skipped(self):
        points = parse_ww3_points_file(BUOYS_SUBSET)
        names = {p.name for p in points}
        assert "AGGA48" not in names
        assert "AGPN48" not in names

    def test_lon_lat_values_read_verbatim(self):
        points = {p.name: p for p in parse_ww3_points_file(BUOYS_SUBSET)}
        # 46070 is recorded as +175.28 in the real product file (not the
        # -175.3 an NDBC lookup would suggest) -- the parser must not try
        # to "correct" this, only read it.
        assert points["46070"].lon == pytest.approx(175.28)
        assert points["46070"].lat == pytest.approx(55.00)
        assert points["46035"].lon == pytest.approx(-177.58)


# ---------------------------------------------------------- window select

class TestWindowSelection:
    def test_point_inside_simple_window(self):
        assert _point_in_window(-150.0, 55.0, -160.0, -140.0, 45.0, 65.0)

    def test_point_outside_simple_window(self):
        assert not _point_in_window(-65.0, 31.5, -160.0, -140.0, 45.0, 65.0)

    def test_dateline_crossing_window_authored_pm180(self):
        # AK-style window crossing the dateline, authored -180/180: 170E to
        # 155W.
        assert _point_in_window(175.28, 55.0, 170.0, -155.0, 45.0, 65.0)
        assert _point_in_window(-177.58, 57.05, 170.0, -155.0, 45.0, 65.0)
        assert not _point_in_window(-65.5, 31.5, 170.0, -155.0, 45.0, 65.0)

    def test_dateline_crossing_window_authored_0360(self):
        # Same physical window, authored directly in 0-360.
        assert _point_in_window(175.28, 55.0, 170.0, 205.0, 45.0, 65.0)
        assert _point_in_window(-177.58, 57.05, 170.0, 205.0, 45.0, 65.0)
        assert not _point_in_window(-65.5, 31.5, 170.0, 205.0, 45.0, 65.0)

    def test_lat_bounds_respected(self):
        assert not _point_in_window(175.0, 30.0, 170.0, -155.0, 45.0, 65.0)


# -------------------------------------------------------- cycle discovery

class TestCycleDiscovery:
    def test_finds_newest_cycle_at_or_before_nowcast_start(self, tmp_path, mock_config):
        # mock_config: pdy=20260401 cyc=12 nowcast_hours=6 -> nowcast_start
        # = 2026-04-01 06Z, itself a cycle boundary.
        root = tmp_path / "gfswave"
        cyc_dir = root / "gfs.20260401" / "06" / "wave" / "station"
        _touch(cyc_dir / "gfswave.t06z.ibp_tar")
        _touch(cyc_dir / "gfswave.t06z.spec_tar.gz")

        proc = WaveBoundaryProcessor(mock_config, root, tmp_path / "work")
        found = proc._find_cycle()
        assert found is not None
        date, cyc, ibp, spec = found
        assert date.strftime("%Y%m%d") == "20260401"
        assert cyc == 6
        assert proc._cycles_checked == 1

    def test_walks_back_when_newest_cycle_missing(self, tmp_path, mock_config):
        root = tmp_path / "gfswave"
        # Only 20260331/18 (2 cycles back) has both tars.
        cyc_dir = root / "gfs.20260331" / "18" / "wave" / "station"
        _touch(cyc_dir / "gfswave.t18z.ibp_tar")
        _touch(cyc_dir / "gfswave.t18z.spec_tar.gz")

        proc = WaveBoundaryProcessor(
            mock_config, root, tmp_path / "work", max_cycle_fallback=4,
        )
        found = proc._find_cycle()
        assert found is not None
        date, cyc, ibp, spec = found
        assert date.strftime("%Y%m%d") == "20260331"
        assert cyc == 18
        assert proc._cycles_checked == 3  # 06Z (missing), 00Z (missing), 18Z prev day (hit)

    def test_gives_up_beyond_max_cycle_fallback(self, tmp_path, mock_config):
        root = tmp_path / "gfswave"  # empty tree
        proc = WaveBoundaryProcessor(
            mock_config, root, tmp_path / "work", max_cycle_fallback=1,
        )
        assert proc._find_cycle() is None
        assert proc._cycles_checked == 2

    def test_requires_both_tars_present(self, tmp_path, mock_config):
        root = tmp_path / "gfswave"
        cyc_dir = root / "gfs.20260401" / "06" / "wave" / "station"
        _touch(cyc_dir / "gfswave.t06z.ibp_tar")
        # spec_tar.gz deliberately missing
        proc = WaveBoundaryProcessor(
            mock_config, root, tmp_path / "work", max_cycle_fallback=0,
        )
        assert proc._find_cycle() is None

    def test_cycle_result_cached(self, tmp_path, mock_config):
        root = tmp_path / "gfswave"
        cyc_dir = root / "gfs.20260401" / "06" / "wave" / "station"
        _touch(cyc_dir / "gfswave.t06z.ibp_tar")
        _touch(cyc_dir / "gfswave.t06z.spec_tar.gz")
        proc = WaveBoundaryProcessor(mock_config, root, tmp_path / "work")
        first = proc._find_cycle()
        assert proc._cycles_checked == 1
        # find_input_files() reuses the cached result rather than re-walking.
        files = proc.find_input_files()
        assert len(files) == 2
        assert proc._cycles_checked == 1
        assert proc._find_cycle() == first


# ------------------------------------------------------------ tar extract

@pytest.mark.skipif(not HAS_TAR, reason="tar binary not available")
class TestTarExtraction:
    def test_extracts_dot_slash_prefixed_member(self, tmp_path):
        ibp_tar = tmp_path / "gfswave.t06z.ibp_tar"
        _make_tar(ibp_tar, {
            "./gfswave.BER51.spec": SPEC_SAMPLE,
            "./gfswave.OTHER99.spec": "dummy\n",
        })
        members = WaveBoundaryProcessor._list_tar_members(ibp_tar)
        assert "./gfswave.BER51.spec" in members

        m = WaveBoundaryProcessor._match_member(members, "BER51")
        assert m == "./gfswave.BER51.spec"

        dest = tmp_path / "work"
        dest.mkdir()
        extracted = WaveBoundaryProcessor._extract_members(ibp_tar, [m], dest)
        assert len(extracted) == 1
        assert extracted[0].name == "gfswave.BER51.spec"
        assert extracted[0].read_text() == SPEC_SAMPLE.read_text()

    def test_extracts_bare_named_member_from_gzip_tar(self, tmp_path):
        spec_targz = tmp_path / "gfswave.t06z.spec_tar.gz"
        _make_tar(spec_targz, {"gfswave.46070.spec": "dummy spec content\n"}, gz=True)

        members = WaveBoundaryProcessor._list_tar_members(spec_targz, gz=True)
        assert "gfswave.46070.spec" in members

        m = WaveBoundaryProcessor._match_member(members, "46070")
        assert m == "gfswave.46070.spec"

        dest = tmp_path / "work"
        dest.mkdir()
        extracted = WaveBoundaryProcessor._extract_members(
            spec_targz, [m], dest, gz=True,
        )
        assert len(extracted) == 1
        assert extracted[0].read_text() == "dummy spec content\n"

    def test_match_member_returns_none_for_missing_point(self):
        assert WaveBoundaryProcessor._match_member(
            ["./gfswave.BER51.spec"], "NOPE",
        ) is None

    def test_only_the_requested_member_is_extracted(self, tmp_path):
        ibp_tar = tmp_path / "gfswave.t06z.ibp_tar"
        _make_tar(ibp_tar, {
            "./gfswave.BER51.spec": "a\n",
            "./gfswave.BER52.spec": "b\n",
            "./gfswave.HNL51.spec": "c\n",
        })
        members = WaveBoundaryProcessor._list_tar_members(ibp_tar)
        m = WaveBoundaryProcessor._match_member(members, "BER52")
        dest = tmp_path / "work"
        dest.mkdir()
        extracted = WaveBoundaryProcessor._extract_members(ibp_tar, [m], dest)
        assert [p.name for p in extracted] == ["gfswave.BER52.spec"]
        assert sorted(p.name for p in dest.iterdir()) == ["gfswave.BER52.spec"]


# ------------------------------------------------------------- inp emit

class TestWW3BoundInpEmission:
    def test_writes_read_mode_and_bare_filenames(self, tmp_path, mock_config):
        work = tmp_path / "work"
        work.mkdir()
        proc = WaveBoundaryProcessor(mock_config, tmp_path, work)
        spec_files = [work / "gfswave.BER51.spec", work / "gfswave.46070.spec"]
        for f in spec_files:
            f.write_text("x\n")

        inp = proc._write_ww3_bound_inp(spec_files)

        assert inp == work / "ww3_bound.inp"
        text = inp.read_text()
        assert "'READ'" in text
        assert "'gfswave.BER51.spec'" in text
        assert "'gfswave.46070.spec'" in text
        # terminated by a '$' line after the file list
        lines = [ln for ln in text.splitlines() if ln.strip()]
        list_idx = lines.index("'gfswave.46070.spec'")
        assert lines[list_idx + 1].strip() == "$"


# --------------------------------------------------------- end-to-end

class TestProcessEndToEnd:
    def _build_tree(self, tmp_path):
        root = tmp_path / "gfswave"
        cyc_dir = root / "gfs.20260401" / "06" / "wave" / "station"
        cyc_dir.mkdir(parents=True)
        _make_tar(cyc_dir / "gfswave.t06z.ibp_tar", {
            "./gfswave.BER51.spec": SPEC_SAMPLE,
        })
        _make_tar(cyc_dir / "gfswave.t06z.spec_tar.gz", {
            "gfswave.46070.spec": "dummy\n",
            "gfswave.46071.spec": "dummy\n",
            "gfswave.46035.spec": "dummy\n",
        }, gz=True)
        return root

    @pytest.mark.skipif(not HAS_TAR, reason="tar binary not available")
    def test_missing_executable_still_emits_inputs(
        self, tmp_path, mock_config, monkeypatch,
    ):
        for var in ("EXECnos", "EXECofs", "EXECstofs3d"):
            monkeypatch.delenv(var, raising=False)

        root = self._build_tree(tmp_path)
        work = tmp_path / "work"

        proc = WaveBoundaryProcessor(
            mock_config, root, work,
            points_file=BUOYS_SUBSET,
            window={"lon_min": -66.0, "lon_max": -64.0, "lat_min": 31.0, "lat_max": 34.0},
            extra_points=["46070", "46071", "46035"],
        )
        result = proc.process()

        assert result.success is False
        assert any("ww3_bound" in e for e in result.errors)
        assert (work / "ww3_bound.inp").exists()

        spec_names = {p.name for p in result.output_files}
        assert "gfswave.46070.spec" in spec_names
        assert "gfswave.46071.spec" in spec_names
        assert "gfswave.46035.spec" in spec_names

        assert result.metadata["mode"] == "missing_executable"
        assert result.metadata["n_points_extra"] == 3
        # All 24 Bermuda IBP points fall inside the -66..-64/31..34 window;
        # only BER51 is actually present in the synthetic ibp_tar, so the
        # rest surface as a "not found" warning rather than aborting.
        assert result.metadata["n_points_ibp"] == 24
        assert any("not found in tar listings" in w for w in result.warnings)

    @pytest.mark.skipif(not HAS_TAR, reason="tar binary not available")
    def test_success_with_fake_executable(self, tmp_path, mock_config, monkeypatch):
        root = self._build_tree(tmp_path)
        work = tmp_path / "work"

        exec_dir = tmp_path / "exec"
        exec_dir.mkdir()
        exe = exec_dir / "ww3_bound"
        exe.write_text("#!/bin/sh\ntouch nest.ww3\n")
        exe.chmod(0o755)
        monkeypatch.setenv("EXECnos", str(exec_dir))
        for var in ("EXECofs", "EXECstofs3d"):
            monkeypatch.delenv(var, raising=False)

        proc = WaveBoundaryProcessor(
            mock_config, root, work, extra_points=["46070"],
        )
        result = proc.process()

        assert result.success is True
        assert (work / "nest.ww3").exists()
        assert result.metadata["mode"] == "ww3_bound"
        assert any(p.name == "nest.ww3" for p in result.output_files)

    def test_no_points_selected_fails_cleanly(self, tmp_path, mock_config):
        root = self._build_tree(tmp_path)
        work = tmp_path / "work"
        proc = WaveBoundaryProcessor(mock_config, root, work)  # no points, no extras
        result = proc.process()
        assert result.success is False
        assert "No wave boundary points selected" in result.errors[0]

    def test_no_cycle_found_reports_actionable_error(self, tmp_path, mock_config):
        root = tmp_path / "empty_gfswave"
        work = tmp_path / "work"
        proc = WaveBoundaryProcessor(mock_config, root, work, max_cycle_fallback=0)
        result = proc.process()
        assert result.success is False
        assert "No GFS-Wave cycle" in result.errors[0]


# ------------------------------------------------------- config threading

class TestWaveConfigThreading:
    _BASE_YAML = (
        "system:\n"
        "  name: stofs_3d_ak_ufs\n"
        "grid:\n"
        "  domain: {lon_min: 156.0, lon_max: 235.0, lat_min: 45.0, lat_max: 75.0}\n"
        "model:\n"
        "  physics: {dt: 90.0}\n"
        "  run: {nowcast_hours: 12, forecast_hours: 180}\n"
    )

    def _cfg(self, tmp_path, extra):
        path = tmp_path / "cfg.yaml"
        path.write_text(self._BASE_YAML + extra)
        return ForcingConfig.from_yaml(path, pdy="20260801", cyc=12)

    def test_defaults_off_when_block_absent(self, tmp_path):
        cfg = self._cfg(tmp_path, "")
        assert cfg.waves_enabled is False
        assert cfg.wave_points_file is None
        assert cfg.wave_extra_points == []
        assert cfg.wave_lon_min is None
        assert cfg.wave_max_cycle_fallback == 4

    def test_full_block_threads_through(self, tmp_path):
        extra = (
            "forcing:\n"
            "  waves:\n"
            "    enabled: true\n"
            "    points_file: /fix/wave_gfs.buoys\n"
            "    extra_points: [\"46070\", \"46071\", \"46035\"]\n"
            "    window: {lon_min: 170.0, lon_max: 235.0, lat_min: 45.0, lat_max: 75.0}\n"
            "    max_cycle_fallback: 6\n"
        )
        cfg = self._cfg(tmp_path, extra)
        assert cfg.waves_enabled is True
        assert cfg.wave_points_file == Path("/fix/wave_gfs.buoys")
        assert cfg.wave_extra_points == ["46070", "46071", "46035"]
        assert cfg.wave_lon_min == 170.0
        assert cfg.wave_lon_max == 235.0
        assert cfg.wave_lat_min == 45.0
        assert cfg.wave_lat_max == 75.0
        assert cfg.wave_max_cycle_fallback == 6

    def test_partial_window_leaves_missing_keys_none(self, tmp_path):
        extra = (
            "forcing:\n  waves:\n    enabled: true\n"
            "    window: {lon_min: 170.0, lat_min: 45.0}\n"
        )
        cfg = self._cfg(tmp_path, extra)
        assert cfg.wave_lon_min == 170.0
        assert cfg.wave_lon_max is None
        assert cfg.wave_lat_max is None

    def test_null_waves_block_does_not_crash(self, tmp_path):
        cfg = self._cfg(tmp_path, "forcing:\n  waves: null\n")
        assert cfg.waves_enabled is False

    def test_null_extra_points_does_not_crash(self, tmp_path):
        cfg = self._cfg(
            tmp_path,
            "forcing:\n  waves:\n    enabled: true\n    extra_points: null\n",
        )
        assert cfg.wave_extra_points == []

    def test_null_window_does_not_crash(self, tmp_path):
        cfg = self._cfg(
            tmp_path,
            "forcing:\n  waves:\n    enabled: true\n    window: null\n",
        )
        assert cfg.wave_lon_min is None


class TestConfigDefaultsAreIsolated:
    def test_wave_extra_points_default_not_shared_across_instances(self):
        a = ForcingConfig(
            lon_min=-1, lon_max=1, lat_min=-1, lat_max=1, pdy="20260101", cyc=0,
        )
        b = ForcingConfig(
            lon_min=-1, lon_max=1, lat_min=-1, lat_max=1, pdy="20260101", cyc=0,
        )
        a.wave_extra_points.append("X")
        assert b.wave_extra_points == []


# --------------------------------------------------------- orchestrator

class TestOrchestratorWaveGating:
    def test_wave_step_skipped_when_disabled(self, mock_config, orch_paths):
        orch = PrepOrchestrator(mock_config, orch_paths)
        result = orch.run(phase="nowcast")
        assert "WAVE_BC" not in [r.source for r in result.results]

    def test_wave_step_skipped_when_no_gfswave_path(self, mock_config, orch_paths):
        cfg = replace(mock_config, waves_enabled=True)
        orch = PrepOrchestrator(cfg, orch_paths)  # no "gfswave" key in paths
        result = orch.run(phase="nowcast")
        assert "WAVE_BC" not in [r.source for r in result.results]

    def test_wave_step_runs_when_enabled_and_path_present(
        self, mock_config, orch_paths, tmp_path,
    ):
        cfg = replace(mock_config, waves_enabled=True, wave_max_cycle_fallback=0)
        paths = dict(orch_paths)
        paths["gfswave"] = str(tmp_path / "gfswave")  # empty tree -> WAVE_BC fails, but runs
        orch = PrepOrchestrator(cfg, paths)
        result = orch.run(phase="nowcast")
        assert "WAVE_BC" in [r.source for r in result.results]


@pytest.mark.skipif(not HAS_TAR, reason="tar binary not available")
class TestOrchestratorWaveArchive:
    def test_nest_ww3_and_metadata_archived(
        self, mock_config, orch_paths, tmp_path, monkeypatch,
    ):
        root = tmp_path / "gfswave"
        cyc_dir = root / "gfs.20260401" / "06" / "wave" / "station"
        cyc_dir.mkdir(parents=True)
        _make_tar(cyc_dir / "gfswave.t06z.ibp_tar", {})
        _make_tar(
            cyc_dir / "gfswave.t06z.spec_tar.gz",
            {"gfswave.46070.spec": "x\n"}, gz=True,
        )

        exec_dir = tmp_path / "exec"
        exec_dir.mkdir()
        exe = exec_dir / "ww3_bound"
        exe.write_text("#!/bin/sh\ntouch nest.ww3\n")
        exe.chmod(0o755)
        monkeypatch.setenv("EXECnos", str(exec_dir))
        for var in ("EXECofs", "EXECstofs3d"):
            monkeypatch.delenv(var, raising=False)

        cfg = replace(mock_config, waves_enabled=True, wave_extra_points=["46070"])
        paths = dict(orch_paths)
        paths["gfswave"] = str(root)
        orch = PrepOrchestrator(cfg, paths, run_name="stofs_3d_ak_ufs")
        result = orch.run(phase="nowcast")

        comout = tmp_path / "comout"
        orch.archive_to_comout(result, comout)

        assert (comout / "stofs_3d_ak_ufs.t12z.nest.ww3").exists()
        assert (comout / "stofs_3d_ak_ufs.t12z.ww3_bound.inp").exists()

        meta_files = list(comout.glob("*.wave_bc.json"))
        assert len(meta_files) == 1
        payload = json.loads(meta_files[0].read_text())
        assert payload["source"] == "WAVE_BC"
        assert payload["success"] is True
        assert payload["mode"] == "ww3_bound"
