"""ADT archive each cycle and the ops fallback to the previous cycle's field."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("netCDF4")
from netCDF4 import Dataset  # noqa: E402

from nos_utils.config import ForcingConfig  # noqa: E402
from nos_utils.forcing.adt import ADTBlender  # noqa: E402
from nos_utils.orchestrator import PrepOrchestrator  # noqa: E402

from .test_adt_esmf import _adt_file, _cfg, _map, _ssh1  # noqa: E402

NAME = "stofs_3d_atl.t12z.adt_aft_cvtz_cln.nc"


@pytest.fixture(autouse=True)
def _no_dcom(monkeypatch):
    monkeypatch.delenv("COMINadt", raising=False)
    monkeypatch.delenv("DCOMROOT", raising=False)


def _blend(tmp_path, archive=None, prev=(), with_adt=True):
    if with_adt:
        _adt_file(tmp_path / "adt_20260401.nc", {})
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    b = ADTBlender(_cfg(_map(tmp_path / "wt.nc")), tmp_path, archive_path=archive, prev_dirs=prev)
    return b, b.blend_ssh(_ssh1(tmp_path / "SSH_1.nc"), work)


def test_field_is_archived_with_ops_fill(tmp_path):
    arc = tmp_path / "out" / NAME
    arc.parent.mkdir()
    b, out = _blend(tmp_path, archive=arc)
    assert out is not None
    with Dataset(str(arc)) as ds:
        ds["surf_el"].set_auto_maskandscale(False)
        f = ds["surf_el"][0]
        assert ds["surf_el"].dtype == np.float32 and f.shape == (2, 2)
        np.testing.assert_allclose(f, 0.9 - 0.45, atol=1e-6)


def test_missing_cells_are_written_as_ops_fill(tmp_path):
    arc = tmp_path / NAME
    _adt_file(tmp_path / "adt_20260401.nc", {(0, 2): np.nan})
    b = ADTBlender(_cfg(_map(tmp_path / "wt.nc")), tmp_path, archive_path=arc)
    (tmp_path / "work").mkdir()
    b.blend_ssh(_ssh1(tmp_path / "SSH_1.nc"), tmp_path / "work")
    with Dataset(str(arc)) as ds:
        ds["surf_el"].set_auto_maskandscale(False)
        assert ds["surf_el"][0].ravel()[1] == np.float32(-2147483647.0)


def test_no_adt_reuses_previous_archive_with_warning(tmp_path):
    prev = tmp_path / "stofs_3d_atl.20260331"
    prev.mkdir()
    arc_prev = prev / NAME
    b0, _ = _blend(tmp_path, archive=arc_prev)
    for f in tmp_path.glob("adt_*.nc"):
        f.unlink()
    out_dir = tmp_path / "today"
    out_dir.mkdir()
    b, out = _blend(tmp_path, archive=out_dir / NAME, prev=[tmp_path / "missing", prev],
                    with_adt=False)
    assert out is not None and b.regrid == "previous"
    assert any("previous cycle" in w for w in b.warnings)
    with Dataset(str(out)) as ds:
        np.testing.assert_allclose(ds["ssh"][0], 0.9 - 0.45, atol=1e-6)
    assert (out_dir / NAME).is_file()


def test_no_adt_and_no_archive_returns_none(tmp_path):
    b, out = _blend(tmp_path, prev=[tmp_path / "nowhere"], with_adt=False)
    assert out is None


def test_archive_of_wrong_shape_is_skipped(tmp_path):
    prev = tmp_path / "prev"
    prev.mkdir()
    with Dataset(str(prev / NAME), "w") as ds:
        ds.createDimension("time", 1)
        ds.createDimension("y", 3)
        ds.createDimension("x", 3)
        ds.createVariable("surf_el", "f4", ("time", "y", "x"))[:] = 0.1
    b, out = _blend(tmp_path, prev=[prev], with_adt=False)
    assert out is None


def test_orchestrator_prev_dirs_and_archive_copy(tmp_path):
    cfg = ForcingConfig.for_stofs_3d_atl(pdy="20260402", cyc=12)
    comout = tmp_path / "com" / "stofs_3d_atl.20260402"
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "com" / "stofs_3d_atl.20260401").mkdir(parents=True)
    paths = {"output": work, "comout": str(comout), "prev_rerun": str(tmp_path / "rerun"),
             "restart": str(tmp_path / "com")}
    orch = PrepOrchestrator(cfg, paths, run_name="stofs_3d_atl")
    dirs = orch._prev_cycle_dirs()
    assert dirs[0] == tmp_path / "rerun"
    assert tmp_path / "com" / "stofs_3d_atl.20260401" in dirs
    assert orch._adt_archive_name() == NAME
    (work / NAME).write_bytes(b"x")
    done = []
    comout.mkdir(parents=True)
    orch._archive_adt_field(work, comout, done)
    assert done == [comout / NAME] and (comout / NAME).read_bytes() == b"x"
