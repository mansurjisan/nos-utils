"""SchismVgrid header parsing across the v2.1 and v3.1 STOFS-3D-ATL vgrid.in layouts."""
import numpy as np
import pytest

from nos_utils.io.schism_vgrid import SchismVgrid

_LSC2_BODY = (
    "1 2 3\n"                      # kbp per node (3 nodes)
    "1 -1.0 -9.0 -9.0\n"
    "2 -0.5 -1.0 -9.0\n"
    "3  0.0  0.0  0.0\n"
)


@pytest.mark.parametrize("line0", [
    "1\n",                                              # v2.1 ops
    "1    !average # of layers=10.095866448283013\n",   # v3.1 ops
])
def test_lsc2_detected_with_and_without_trailing_comment(tmp_path, line0):
    path = tmp_path / "vgrid.in"
    path.write_text(line0 + "3  \n" + _LSC2_BODY)

    vg = SchismVgrid.read(path)

    assert vg.nvrt == 3
    assert vg.kz == 0
    vg.load_boundary_sigma([1, 3])
    np.testing.assert_array_equal(vg.node_kbp, [1, 3])
    np.testing.assert_array_equal(vg.node_sigma[:, 0], [-1.0, -0.5, 0.0])
    np.testing.assert_array_equal(vg.node_sigma[:, 1], [-9.0, -9.0, 0.0])


def test_simple_format_header_with_trailing_comment(tmp_path):
    path = tmp_path / "vgrid.in"
    path.write_text(
        "4 2 100.0 !nvrt kz h_s\n"
        "Z levels\n"
        "1 -5000.0\n"
        "2 -100.0\n"
        "S levels\n"
        "3 -1.0\n"
        "4 0.0\n"
    )

    vg = SchismVgrid.read(path)

    assert (vg.nvrt, vg.kz, vg.h_s) == (4, 2, 100.0)
    np.testing.assert_array_equal(vg.z_levels, [-5000.0, -100.0])
    np.testing.assert_array_equal(vg.sigma_levels, [-1.0, 0.0])
