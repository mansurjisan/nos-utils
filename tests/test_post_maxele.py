"""Tests for nos_utils.post.maxele."""
from __future__ import annotations

import pytest

netCDF4 = pytest.importorskip("netCDF4")
import numpy as np  # noqa: E402

from nos_utils.post.maxele import FILL_VALUE, write_maxele  # noqa: E402
from tests.post_fixtures import write_out2d_stack  # noqa: E402


def test_maxele_reduces_over_stacks(tmp_path):
    # Node value = node_id + hour/100 -> global max at the last hour.
    s1 = write_out2d_stack(tmp_path / "out2d_1.nc", hours=[1, 2, 3])
    s2 = write_out2d_stack(tmp_path / "out2d_2.nc", hours=[4, 5, 6])
    out = tmp_path / "maxele.nc"

    write_maxele([s1, s2], out, base_date="2026-07-10 00:00")

    with netCDF4.Dataset(out) as ds:
        z = ds["zeta_max"][:]
        expected = np.array([n + 1 + 6 / 100.0 for n in range(4)], dtype="f4")
        assert np.allclose(z, expected)
        assert len(ds.dimensions["node"]) == 4
        t = ds["time"][:]
        assert list(t) == [3600.0, 6 * 3600.0]
        assert ds["time"].units == "seconds since 2026-07-10 00:00"
        assert ds["time"].base_date == "2026-07-10 00:00"
        assert ds["zeta_max"]._FillValue == np.float32(FILL_VALUE)
        assert ds["depth"].mesh == "adcirc_mesh"
        assert ds["depth"].coordinates == "time y x"
        gfill = ds.getncattr("_FillValue")
        gfill = gfill.decode() if isinstance(gfill, bytes) else gfill
        assert gfill == "-99999."
        assert np.allclose(ds["x"][:], [0.0, 1.0, 0.0, 1.0])


def test_maxele_window_override_and_empty_stack(tmp_path):
    s1 = write_out2d_stack(tmp_path / "out2d_1.nc", hours=[1, 2])
    s2 = write_out2d_stack(tmp_path / "out2d_2.nc", hours=[])
    out = tmp_path / "maxele.nc"

    write_maxele(
        [s1, s2], out, base_date="2026-07-10 00:00",
        window_seconds=(90000.0, 432000.0),
    )

    with netCDF4.Dataset(out) as ds:
        assert list(ds["time"][:]) == [90000.0, 432000.0]


def test_maxele_no_coords_stack(tmp_path):
    s1 = write_out2d_stack(
        tmp_path / "out2d_1.nc", hours=[1], with_coords=False
    )
    out = tmp_path / "maxele.nc"
    write_maxele([s1], out, base_date="2026-07-10 00:00")
    with netCDF4.Dataset(out) as ds:
        assert "zeta_max" in ds.variables
        assert "x" not in ds.variables


def test_maxele_rejects_empty_inputs(tmp_path):
    with pytest.raises(ValueError):
        write_maxele([], tmp_path / "o.nc", base_date="2026-07-10 00:00")
