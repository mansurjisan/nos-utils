"""STOFS-3D-ATL river inputs follow ops v3.1.5: json source order, static FIX
source_sink.in / msource.th / vsink.th, gen_sourcesink.py vsource.th format."""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from nos_utils.config import ForcingConfig
from nos_utils.forcing.nwm import NWMProcessor, RiverConfig

netCDF4 = pytest.importorskip("netCDF4")

# json order (= source_sink.in order) deliberately not sorted
SOURCES = {"300": [11, 12], "100": [21], "200": [31]}
SOURCE_SINK_IN = "3\n300\n100\n200\n\n4\n1\n2\n3\n4\n"
MSOURCE = "0 " + " ".join(["-9999"] * 3 + ["0"] * 3) + "\n864000 " + " ".join(["-9999"] * 3 + ["0"] * 3) + "\n"
VSINK = "0 -0.1 -0.2 -0.3 -0.4\n3153600000 -0.1 -0.2 -0.3 -0.4\n"


def _fix_dir(tmp_path, **override):
    fix = tmp_path / "fix"
    fix.mkdir(exist_ok=True)
    files = {
        "stofs_3d_atl_river_sources_conus.json": json.dumps(SOURCES),
        "stofs_3d_atl_river_source_sink.in": SOURCE_SINK_IN,
        "stofs_3d_atl_river_msource.th": MSOURCE,
        "stofs_3d_atl_river_vsink.th": VSINK,
    }
    files.update(override)
    for name, text in files.items():
        if text is None:
            continue
        (fix / name).write_text(text)
    return fix


def _cfg(fix, **kw):
    kw.setdefault("nowcast_hours", 2)
    kw.setdefault("forecast_hours", 2)
    kw.setdefault("ops_bad_day_checks", False)  # these fixtures carry a handful of NWM files MJ (10/05/26)
    return ForcingConfig.for_stofs_3d_atl(
        pdy="20261001", cyc=12,
        river_config_file=fix / "stofs_3d_atl_river_sources_conus.json", **kw)


def _proc(tmp_path, fix, phase="nowcast", **kw):
    return NWMProcessor(_cfg(fix, **kw), tmp_path / "nwm", tmp_path / "out", phase=phase)


def _write_nwm(path, valid, flows):
    """NWM channel_rt stand-in; float64 streamflow like the unpacked NWM variable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    ds.createDimension("feature_id", 4)
    fid = ds.createVariable("feature_id", "i8", ("feature_id",))
    q = ds.createVariable("streamflow", "f8", ("feature_id",))
    fid[:] = np.array([11, 12, 21, 31], dtype=np.int64)
    q[:] = np.array(flows, dtype=np.float64)
    ds.model_output_valid_time = valid.strftime("%Y-%m-%d_%H:%M:%S")
    ds.close()
    return path


def _stage_nwm(tmp_path, n_hours=6):
    """Hourly files from the nowcast start (cycle - 2 h); flows chosen so float32 sums differ."""
    start = datetime(2026, 10, 1, 10)
    files = []
    for h in range(n_hours):
        vt = start + timedelta(hours=h)
        f = tmp_path / "nwm" / ("nwm.t%02dz.medium_range.channel_rt_1.f001.conus.nc" % vt.hour)
        files.append(_write_nwm(f, vt, [6000.00991 + h, 360.0, 2.5 + h, 0.25]))
    return files


class TestConfigFlag:
    def test_atl_factories(self):
        assert ForcingConfig.for_stofs_3d_atl("20261001", 12).river_ops_static is True
        assert ForcingConfig.for_stofs_3d_atl_ufs("20261001", 12).river_ops_static is True

    def test_atl_factory_on_for_standalone_and_coupled(self):
        assert ForcingConfig.for_stofs_3d_atl("20261001", 12, nws=2).river_ops_static is True
        assert ForcingConfig.for_stofs_3d_atl("20261001", 12, nws=4).river_ops_static is True
        assert ForcingConfig.for_stofs_3d_atl_ufs("20261001", 12, river_ops_static=False).river_ops_static is False

    @pytest.mark.parametrize("factory", [
        "for_secofs", "for_secofs_ufs", "for_stofs_3d_pac", "for_stofs_3d_pac_ufs"])
    def test_other_systems_off(self, factory):
        assert getattr(ForcingConfig, factory)("20261001", 12).river_ops_static is False

    def test_default_off(self):
        assert ForcingConfig(-90.0, -60.0, 20.0, 40.0, pdy="20261001", cyc=12).river_ops_static is False

    def test_coupled_nws4_on_by_name(self, tmp_path):
        pytest.importorskip("yaml")
        p = tmp_path / "x.yaml"
        p.write_text("system:\n  name: stofs_3d_atl_ufs\nmodel:\n  physics:\n    nws: 4\n")
        assert ForcingConfig.from_yaml(p, pdy="20261001", cyc=12).river_ops_static is True

    @pytest.mark.parametrize("name,river,expected", [
        ("stofs_3d_atl_ufs", "", True),
        ("stofs_3d_atl_ufs", "ops_static_files: true", True),
        ("stofs_3d_atl_ufs_standalone", "", True),
        ("secofs_ufs", "", False),
        ("stofs_3d_ak_ufs", "", False),
        ("stofs_3d_atl_ufs", "ops_static_files: false", False),
        ("secofs_ufs", "ops_static_files: true", True),
    ])
    def test_from_yaml(self, tmp_path, name, river, expected):
        pytest.importorskip("yaml")
        p = tmp_path / "x.yaml"
        p.write_text("system:\n  name: %s\nforcing:\n  river:\n    primary: nwm\n    %s\n" % (name, river))
        assert ForcingConfig.from_yaml(p, pdy="20261001", cyc=12).river_ops_static is expected

    def test_from_yaml_rejects_garbage(self, tmp_path):
        pytest.importorskip("yaml")
        p = tmp_path / "x.yaml"
        p.write_text("system:\n  name: stofs_3d_atl_ufs\nforcing:\n  river:\n    ops_static_files: maybe\n")
        with pytest.raises(ValueError, match="ops_static_files"):
            ForcingConfig.from_yaml(p, pdy="20261001", cyc=12)


class TestSourceOrder:
    def test_keep_order(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text(json.dumps(SOURCES))
        assert RiverConfig.from_sources_json(p, keep_order=True).node_indices == [300, 100, 200]
        assert RiverConfig.from_sources_json(p).node_indices == [100, 200, 300]

    def test_atl_processor_keeps_json_order(self, tmp_path):
        fix = _fix_dir(tmp_path)
        assert _proc(tmp_path, fix).river_config.node_indices == [300, 100, 200]

    def test_secofs_style_config_still_sorted(self, tmp_path):
        fix = _fix_dir(tmp_path)
        cfg = ForcingConfig.for_secofs(
            pdy="20261001", cyc=12, river_config_file=fix / "stofs_3d_atl_river_sources_conus.json")
        proc = NWMProcessor(cfg, tmp_path, tmp_path / "out")
        assert proc.river_config.node_indices == [100, 200, 300]


class TestStageOpsFiles:
    def test_copies_byte_exact(self, tmp_path):
        fix = _fix_dir(tmp_path)
        proc = _proc(tmp_path, fix)
        proc.create_output_dir()
        staged = proc._stage_ops_river_files()
        assert sorted(p.name for p in staged) == ["msource.th", "source_sink.in", "vsink.th"]
        out = tmp_path / "out"
        assert (out / "source_sink.in").read_text() == SOURCE_SINK_IN
        assert (out / "msource.th").read_text() == MSOURCE
        assert (out / "vsink.th").read_text() == VSINK

    def test_missing_files_named(self, tmp_path):
        fix = _fix_dir(tmp_path, **{"stofs_3d_atl_river_vsink.th": None,
                                    "stofs_3d_atl_river_msource.th": None})
        proc = _proc(tmp_path, fix)
        proc.create_output_dir()
        with pytest.raises(FileNotFoundError) as exc:
            proc._stage_ops_river_files()
        assert "stofs_3d_atl_river_vsink.th" in str(exc.value)
        assert "stofs_3d_atl_river_msource.th" in str(exc.value)

    def test_other_systems_vsink_names_not_accepted(self, tmp_path):
        fix = _fix_dir(tmp_path, **{"stofs_3d_atl_river_vsink.th": None})
        (fix / "vsink.th").write_text(VSINK)
        (fix / "secofs_ufs.vsink.th").write_text(VSINK)
        proc = _proc(tmp_path, fix)
        proc.create_output_dir()
        with pytest.raises(FileNotFoundError):
            proc._stage_ops_river_files()

    def test_source_order_mismatch(self, tmp_path):
        fix = _fix_dir(tmp_path, **{"stofs_3d_atl_river_source_sink.in": "3\n100\n200\n300\n\n0\n"})
        proc = _proc(tmp_path, fix)
        proc.create_output_dir()
        with pytest.raises(ValueError, match="misordered"):
            proc._stage_ops_river_files()

    def test_source_count_mismatch(self, tmp_path):
        fix = _fix_dir(tmp_path, **{"stofs_3d_atl_river_source_sink.in": "2\n300\n100\n\n0\n"})
        proc = _proc(tmp_path, fix)
        proc.create_output_dir()
        with pytest.raises(ValueError, match="declares 2 sources"):
            proc._stage_ops_river_files()

    def test_msource_columns_mismatch(self, tmp_path):
        fix = _fix_dir(tmp_path, **{"stofs_3d_atl_river_msource.th": "0 -9999 -9999 -9999\n864000 0 0 0\n"})
        proc = _proc(tmp_path, fix)
        proc.create_output_dir()
        with pytest.raises(ValueError, match="msource.th has 4 columns"):
            proc._stage_ops_river_files()

    def test_vsink_columns_mismatch(self, tmp_path):
        fix = _fix_dir(tmp_path, **{"stofs_3d_atl_river_vsink.th": "0 -0.1 -0.2\n3600 -0.1 -0.2\n"})
        proc = _proc(tmp_path, fix)
        proc.create_output_dir()
        with pytest.raises(ValueError, match="vsink.th has 3 columns"):
            proc._stage_ops_river_files()


class TestProcessOps:
    def test_outputs_match_ops_layout(self, tmp_path):
        fix = _fix_dir(tmp_path)
        _stage_nwm(tmp_path)
        proc = _proc(tmp_path, fix)
        proc.find_input_files = lambda: sorted((tmp_path / "nwm").glob("nwm.t*.nc"))
        result = proc.process()
        assert result.success, result.errors
        assert result.metadata["ops_static"] is True
        out = tmp_path / "out"
        assert sorted(p.name for p in out.iterdir()) == [
            "msource.th", "source_sink.in", "vsink.th", "vsource.th"]
        assert (out / "source_sink.in").read_text() == SOURCE_SINK_IN
        assert (out / "msource.th").read_text() == MSOURCE
        assert (out / "vsink.th").read_text() == VSINK

        lines = (out / "vsource.th").read_text().split("\n")
        assert lines[-1] == "" and len(lines) == 7  # nowcast 2 h + 3 h buffer -> 6 rows

        def vals(h):  # column order 300, 100, 200 (json order): fid sums 11+12, 21, 31
            return [6000.00991 + h + 360.0, 2.5 + h, 0.25]

        assert lines[0] == "0 " + " ".join("%.4f" % v for v in vals(0))
        assert lines[1] == "3600 " + " ".join("%.4f" % v for v in vals(1))
        for h in range(2, 6):
            assert lines[h] == ("%.4f" % (h * 3600.0)) + "".join(" %10.4f" % v for v in vals(h))
        # float64 sum keeps the 4th decimal that float32 would move (6360.0099 vs 6360.0098)
        assert lines[0].split()[1] == "6360.0099"

    def test_forecast_phase_same_layout(self, tmp_path):
        fix = _fix_dir(tmp_path)
        start = datetime(2026, 10, 1, 12)
        for h in range(6):
            vt = start + timedelta(hours=h)
            _write_nwm(tmp_path / "nwm" / ("nwm.t%02dz.f.nc" % vt.hour), vt,
                       [10.0 + h, 20.0, 30.0 + h, 40.0])
        proc = _proc(tmp_path, fix, phase="forecast")
        proc.find_input_files = lambda: sorted((tmp_path / "nwm").glob("nwm.t*.nc"))
        assert proc.process().success
        lines = (tmp_path / "out" / "vsource.th").read_text().split("\n")
        assert lines[0] == "0 30.0000 30.0000 40.0000"
        assert lines[1] == "3600 31.0000 31.0000 40.0000"

    def test_missing_fix_fails_before_reading_nwm(self, tmp_path):
        fix = _fix_dir(tmp_path, **{"stofs_3d_atl_river_source_sink.in": None})
        proc = _proc(tmp_path, fix)

        def boom():
            raise AssertionError("NWM discovery must not run")
        proc.find_input_files = boom
        result = proc.process()
        assert result.success is False
        assert "stofs_3d_atl_river_source_sink.in" in result.errors[0]
        assert not (tmp_path / "out" / "vsource.th").exists()

    def test_non_ops_mode_unchanged(self, tmp_path):
        """river_ops_static=False keeps the sorted order, %.4e rows, all -9999 msource and 0 sinks."""
        fix = _fix_dir(tmp_path)
        _stage_nwm(tmp_path)
        proc = _proc(tmp_path, fix, river_ops_static=False)
        proc.find_input_files = lambda: sorted((tmp_path / "nwm").glob("nwm.t*.nc"))
        result = proc.process()
        assert result.success and result.metadata["ops_static"] is False
        out = tmp_path / "out"
        assert (out / "source_sink.in").read_text() == "3\n100\n200\n300\n\n0\n"
        assert not (out / "vsink.th").exists()
        assert set((out / "msource.th").read_text().split("\n")[0].split()[1:]) == {"-9.9990e+03"}
        first = (out / "vsource.th").read_text().split("\n")[0].split()
        assert first[:2] == ["0", "2.5000e+00"]
