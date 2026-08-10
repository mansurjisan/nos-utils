"""Regression tests for correct int16 packing of surf_el (SSH) and T/S/U/V
in the real RTOFS writer methods.

Background
----------
The Fortran gen_3Dth_from_hycom reads surf_el / temperature / salinity /
water_u / water_v with low-level nf90_get_var (auto-converts i2->f4, does NOT
apply scale_factor/add_offset) then manually unpacks:

    ssh  = ssh  * 1.e-3              (scale=0.001, offset=0)
    temp = temp * 1.e-3 + 20         (scale=0.001, offset=20)
    salt = salt * 1.e-3 + 20
    uvel = uvel * 1.e-3              (scale=0.001, offset=0)
    vvel = vvel * 1.e-3

So the on-disk value MUST be packed:
    stored = round((real - add_offset) / scale_factor)

These tests call the real writer methods and verify:
  - The on-disk dtype is int16 (i2), not float.
  - The raw stored value (auto-scale OFF) is the EXACT expected packed integer.
  - The high-level auto-scaled read (scale_factor/add_offset applied) recovers
    the original real value within quantization tolerance.
  - The Fortran-emulated unpack (raw * scale_factor + add_offset) matches real
    within quantization tolerance.
  - Extreme/masked SSH cells stored as fill (-30000) unpack below the dry-node
    threshold rjunk+0.1 ≈ -9.9 m.

Tolerance rationale
-------------------
int16 packing is lossy: the max quantization error is scale_factor/2 = 0.0005.
Raw-stored tests use EXACT integer equality (the stored int16 must match
round((real - offset) / scale) computed in the same float32 precision as the
writer — not a tolerance, an exact check).
Round-trip tests (auto-scale read, Fortran unpack) use 0.001 = one scale unit,
comfortably above the 0.0005 quantization limit.
"""

import numpy as np
import pytest

nc4 = pytest.importorskip("netCDF4")
from netCDF4 import Dataset

from nos_utils.config import ForcingConfig
from nos_utils.forcing.rtofs import RTOFSProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atl_config():
    return ForcingConfig.for_stofs_3d_atl(pdy="20260401", cyc=12)


def _make_proc(tmp_path, cfg=None):
    cfg = cfg or _atl_config()
    return RTOFSProcessor(cfg, tmp_path, tmp_path / "out")


def _expected_packed(real_f32, offset_f32, scale_f32):
    """Compute the expected int16 stored value in the same float32 precision
    the writer uses, so the raw-packed assertion is exact."""
    return int(np.clip(
        np.round((np.float32(real_f32) - np.float32(offset_f32))
                 / np.float32(scale_f32)),
        np.iinfo(np.int16).min, np.iinfo(np.int16).max,
    ))


# ---------------------------------------------------------------------------
# SSH / surf_el packing
# ---------------------------------------------------------------------------

class TestSSHPacking:
    """surf_el must be stored as int16, explicitly packed, single-packed."""

    def _write_ssh(self, tmp_path):
        """Write a minimal SSH_1.nc using _stofs_prepare_ssh with a synthetic
        fixture containing known real SSH values."""
        proc = _make_proc(tmp_path)

        ny, nx = 3, 4
        # Real SSH values in metres: mix of positive, negative, and an extreme.
        real_ssh = np.array([
            [0.5,  -1.25, 0.0,   2.0],
            [-0.8,  0.3, -2.5,   1.1],
            [0.01, -0.07, 0.9,  15000.0],  # last cell is extreme -> fill
        ], dtype=np.float32)

        fpath = tmp_path / "rtofs.20260331" / "rtofs_glo_2ds_n012_diag.nc"
        fpath.parent.mkdir(parents=True, exist_ok=True)
        ds = Dataset(str(fpath), "w", format="NETCDF4")
        ds.createDimension("MT", 1)
        ds.createDimension("Y", ny)
        ds.createDimension("X", nx)
        mt = ds.createVariable("MT", "f8", ("MT",))
        mt[:] = [0.0]
        lon = ds.createVariable("Longitude", "f4", ("Y", "X"))
        lat = ds.createVariable("Latitude",  "f4", ("Y", "X"))
        lon[:] = np.zeros((ny, nx))
        lat[:] = np.zeros((ny, nx))
        ssh = ds.createVariable("ssh", "f4", ("MT", "Y", "X"),
                                fill_value=-30000.0)
        ssh.missing_value = -30000.0
        ssh[0] = real_ssh
        ds.close()

        roi = {"x1": 0, "x2": nx - 1, "y1": 0, "y2": ny - 1}
        proc.config = proc.config.__class__(
            **{**proc.config.__dict__, "obc_roi_2d": roi}
        )
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        out = proc._stofs_prepare_ssh([fpath], work)
        return out, real_ssh

    def test_dtype_is_int16(self, tmp_path):
        out, _ = self._write_ssh(tmp_path)
        assert out is not None and out.exists()
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        assert ds.variables["surf_el"].dtype == np.dtype("int16"), (
            "surf_el should be stored as int16"
        )
        ds.close()

    def test_scale_offset_attrs(self, tmp_path):
        out, _ = self._write_ssh(tmp_path)
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        v = ds.variables["surf_el"]
        assert abs(float(v.scale_factor) - 0.001) < 1e-9
        assert abs(float(v.add_offset) - 0.0) < 1e-9
        ds.close()

    def test_raw_stored_is_exact_packed_int(self, tmp_path):
        """Raw int16 on disk must be the EXACT integer
        round((real - add_offset) / scale_factor), computed in float32 to
        match the writer.  No tolerance — stored integers are exact."""
        out, real = self._write_ssh(tmp_path)
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        v = ds.variables["surf_el"]
        scale  = float(v.scale_factor)
        offset = float(v.add_offset)
        raw = np.asarray(v[0], dtype=np.int16)
        ds.close()

        for j in range(real.shape[0]):
            for i in range(real.shape[1]):
                r = float(real[j, i])
                if abs(r) >= 10000:
                    assert int(raw[j, i]) == -30000, (
                        f"extreme cell ({j},{i}) raw={raw[j,i]}, want fill -30000"
                    )
                else:
                    expected = _expected_packed(r, offset, scale)
                    assert int(raw[j, i]) == expected, (
                        f"cell ({j},{i}) real={r} raw={raw[j,i]}, want {expected}"
                    )

    def test_autoscale_read_recovers_real(self, tmp_path):
        """High-level auto-scale read must recover real SSH within quantization
        tolerance (scale_factor/2 = 0.0005; use 0.001 = one scale unit)."""
        out, real = self._write_ssh(tmp_path)
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(True)
        got = np.ma.filled(
            np.asarray(ds.variables["surf_el"][0], dtype=np.float64),
            fill_value=np.nan,
        )
        ds.close()
        for j in range(real.shape[0]):
            for i in range(real.shape[1]):
                r = float(real[j, i])
                if abs(r) < 10000:
                    assert abs(got[j, i] - r) < 0.001, (
                        f"cell ({j},{i}) auto-read={got[j,i]}, want real={r}"
                    )

    def test_fortran_unpack_recovers_real(self, tmp_path):
        """Emulate Fortran: raw * scale_factor + add_offset must recover real
        SSH within quantization tolerance (0.001 = one scale unit)."""
        out, real = self._write_ssh(tmp_path)
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        v = ds.variables["surf_el"]
        scale  = float(v.scale_factor)
        offset = float(v.add_offset)
        raw = np.asarray(v[0], dtype=np.float64)
        ds.close()
        for j in range(real.shape[0]):
            for i in range(real.shape[1]):
                r = float(real[j, i])
                if abs(r) < 10000:
                    fortran = raw[j, i] * scale + offset
                    assert abs(fortran - r) < 0.001, (
                        f"cell ({j},{i}) Fortran-unpack={fortran}, want {r}"
                    )

    def test_extreme_cell_triggers_dry_node(self, tmp_path):
        """Fill cell must unpack to < rjunk+0.1 (-9.9 m) -> dry node."""
        out, _ = self._write_ssh(tmp_path)
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        v = ds.variables["surf_el"]
        scale  = float(v.scale_factor)
        offset = float(v.add_offset)
        raw_fill = int(v[0, 2, 3])  # last cell (2,3) was extreme (15000 m)
        ds.close()
        assert raw_fill == -30000, f"fill raw={raw_fill}, want -30000"
        fortran = raw_fill * scale + offset
        rjunk_threshold = -3.e4 * 1.e-3 + 0.1  # rjunk + 0.1 = -29.9
        assert fortran < rjunk_threshold, (
            f"fill cell Fortran unpack={fortran} not < {rjunk_threshold}: "
            f"would not be flagged dry"
        )


# ---------------------------------------------------------------------------
# TSUV packing
# ---------------------------------------------------------------------------

class TestTSUVPacking:
    """T/S/U/V must be stored as int16, explicitly packed."""

    def _write_tsuv(self, tmp_path):
        proc = _make_proc(tmp_path)
        nz, ny, nx = 2, 3, 3

        temp = np.full((nz, ny, nx), 18.5,  dtype=np.float32)
        salt = np.full((nz, ny, nx), 35.0,  dtype=np.float32)
        u    = np.full((nz, ny, nx),  0.25, dtype=np.float32)
        v    = np.full((nz, ny, nx), -0.10, dtype=np.float32)
        lon  = np.zeros((ny, nx), dtype=np.float32)
        lat  = np.zeros((ny, nx), dtype=np.float32)
        dep  = np.array([5.0, 50.0], dtype=np.float32)

        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        out = proc._write_tsuv_nc(
            work, [temp], [salt], [u], [v], lon, lat, dep,
        )
        real = {"temperature": 18.5, "salinity": 35.0,
                "water_u": 0.25, "water_v": -0.10}
        return out, real

    def test_dtype_is_int16(self, tmp_path):
        out, _ = self._write_tsuv(tmp_path)
        assert out is not None and out.exists()
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        for name in ("temperature", "salinity", "water_u", "water_v"):
            assert ds.variables[name].dtype == np.dtype("int16"), (
                f"{name} should be int16"
            )
        ds.close()

    def test_scale_offset_attrs(self, tmp_path):
        out, _ = self._write_tsuv(tmp_path)
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        expected_meta = {
            "temperature": (20.0, 0.001),
            "salinity":    (20.0, 0.001),
            "water_u":     ( 0.0, 0.001),
            "water_v":     ( 0.0, 0.001),
        }
        for name, (offset, scale) in expected_meta.items():
            v = ds.variables[name]
            assert abs(float(v.scale_factor) - scale) < 1e-9, name
            assert abs(float(v.add_offset) - offset) < 1e-9, name
        ds.close()

    def test_raw_stored_is_exact_packed_int(self, tmp_path):
        """Raw int16 on disk must be the EXACT integer
        round((real - add_offset) / scale_factor), computed in float32 to
        match the writer.  No tolerance — stored integers are exact."""
        out, real = self._write_tsuv(tmp_path)
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        for name in real:
            v = ds.variables[name]
            scale  = float(v.scale_factor)
            offset = float(v.add_offset)
            raw = int(v[0, 0, 0, 0])
            expected = _expected_packed(real[name], offset, scale)
            assert raw == expected, (
                f"{name}: raw={raw}, want {expected} "
                f"(real={real[name]}, offset={offset}, scale={scale})"
            )
        ds.close()

    def test_autoscale_read_recovers_real(self, tmp_path):
        """High-level auto-scale read must recover real T/S/U/V within
        quantization tolerance (scale_factor/2 = 0.0005; use 0.001)."""
        out, real = self._write_tsuv(tmp_path)
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(True)
        for name, r in real.items():
            got = float(np.asarray(ds.variables[name][0, 0, 0, 0]))
            assert abs(got - r) < 0.001, (
                f"{name}: auto-read={got}, want {r}"
            )
        ds.close()

    def test_fortran_unpack_recovers_real(self, tmp_path):
        """Emulate Fortran unpack: raw * scale_factor + add_offset must
        recover real T/S/U/V within quantization tolerance (0.001)."""
        out, real = self._write_tsuv(tmp_path)
        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        for name, r in real.items():
            v = ds.variables[name]
            scale  = float(v.scale_factor)
            offset = float(v.add_offset)
            raw = float(v[0, 0, 0, 0])
            fortran = raw * scale + offset
            assert abs(fortran - r) < 0.001, (
                f"{name}: Fortran-unpack={fortran}, want {r}"
            )
        ds.close()

    def test_zero_current_is_not_fill(self, tmp_path):
        """A genuine slack-water current (u=v=0.0 m/s) must pack to 0, NOT be
        misclassified as missing (-30000). This pins the fix for conflating
        real zero currents with the -30000 fill sentinel."""
        proc = _make_proc(tmp_path)
        nz, ny, nx = 1, 2, 2
        temp = np.full((nz, ny, nx), 15.0, dtype=np.float32)
        salt = np.full((nz, ny, nx), 34.0, dtype=np.float32)
        u    = np.zeros((nz, ny, nx), dtype=np.float32)   # real slack water
        v    = np.zeros((nz, ny, nx), dtype=np.float32)
        lon  = np.zeros((ny, nx), dtype=np.float32)
        lat  = np.zeros((ny, nx), dtype=np.float32)
        dep  = np.array([5.0], dtype=np.float32)

        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        out = proc._write_tsuv_nc(work, [temp], [salt], [u], [v], lon, lat, dep)

        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        for name in ("water_u", "water_v"):
            raw = int(ds.variables[name][0, 0, 0, 0])
            assert raw == 0, (
                f"{name}: zero current stored as {raw}, want 0 "
                f"(must not be flagged as -30000 fill)"
            )
        ds.close()

    def test_missing_current_is_fill(self, tmp_path):
        """A missing current cell (-30000, from ma.filled) must be stored as
        the -30000 int16 fill sentinel, matching operational change_miss."""
        proc = _make_proc(tmp_path)
        nz, ny, nx = 1, 2, 2
        temp = np.full((nz, ny, nx), 15.0, dtype=np.float32)
        salt = np.full((nz, ny, nx), 34.0, dtype=np.float32)
        u    = np.full((nz, ny, nx), -30000.0, dtype=np.float32)  # missing
        v    = np.full((nz, ny, nx), -30000.0, dtype=np.float32)
        lon  = np.zeros((ny, nx), dtype=np.float32)
        lat  = np.zeros((ny, nx), dtype=np.float32)
        dep  = np.array([5.0], dtype=np.float32)

        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        out = proc._write_tsuv_nc(work, [temp], [salt], [u], [v], lon, lat, dep)

        ds = Dataset(str(out))
        ds.set_auto_maskandscale(False)
        for name in ("water_u", "water_v"):
            raw = int(ds.variables[name][0, 0, 0, 0])
            assert raw == -30000, (
                f"{name}: missing current stored as {raw}, want -30000 fill"
            )
        ds.close()
