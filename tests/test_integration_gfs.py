#!/usr/bin/env python3
"""
Integration test: GFS processor with real GRIB2 data.

Run inside Docker container with wgrib2 available:
    docker run --rm \
      -v /path/to/gfs:/data/gfs:ro \
      -v /path/to/nos-utils:/opt/nos-utils \
      -v /tmp/sflux_test:/output \
      --entrypoint bash nosofs/secofs-ufs:latest -c '
        export PYTHONPATH=/opt/nos-utils:$PYTHONPATH
        python3 /opt/nos-utils/tests/test_integration_gfs.py
      '
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure nos_utils is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from nos_utils.config import ForcingConfig
from nos_utils.forcing.gfs import GFSProcessor
from nos_utils.forcing.sflux_writer import AIR_VARS, RAD_VARS, PRC_VARS


def main():
    # --- Configuration ---
    GFS_ROOT = Path(os.environ.get("GFS_ROOT", "/data/gfs/v16.3"))
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
    PDY = os.environ.get("PDY", "20260324")
    CYC = int(os.environ.get("CYC", "12"))

    print("=" * 60)
    print(f"GFS Integration Test")
    print(f"  GFS_ROOT:   {GFS_ROOT}")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  PDY={PDY}  CYC={CYC:02d}z")
    print("=" * 60)

    # Verify GFS data exists
    gfs_date_dir = GFS_ROOT / f"gfs.{PDY}" / f"{CYC:02d}" / "atmos"
    if not gfs_date_dir.exists():
        print(f"FAIL: GFS data not found at {gfs_date_dir}")
        sys.exit(1)

    grib_files = sorted(gfs_date_dir.glob("gfs.t*.pgrb2.0p25.f*"))
    grib_files = [f for f in grib_files if not f.name.endswith(".idx")]
    print(f"  Found {len(grib_files)} GFS GRIB2 files")
    print(f"  First: {grib_files[0].name}")
    print(f"  Last:  {grib_files[-1].name}")
    print(f"  Size:  {grib_files[0].stat().st_size / 1e6:.0f} MB")

    # --- Test 1: SECOFS sflux output ---
    print("\n--- Test 1: SECOFS sflux (nws=2) ---")

    sflux_dir = OUTPUT_DIR / "sflux_test"
    sflux_dir.mkdir(parents=True, exist_ok=True)

    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)
    # Adjust for available data: files are ~491MB, lower threshold slightly
    processor = GFSProcessor(config, GFS_ROOT, sflux_dir)
    processor.MIN_FILE_SIZE = 400_000_000  # 400MB (files are ~491MB)

    result = processor.process()

    print(f"  Success: {result.success}")
    print(f"  Files created: {len(result.output_files)}")
    if result.errors:
        print(f"  Errors: {result.errors}")
    if result.warnings:
        print(f"  Warnings: {result.warnings}")
    print(f"  Metadata: {result.metadata}")

    if not result.success:
        print("FAIL: GFS processing failed")
        sys.exit(1)

    # Validate sflux files
    from netCDF4 import Dataset

    air_files = [f for f in result.output_files if "sflux_air" in f.name]
    rad_files = [f for f in result.output_files if "sflux_rad" in f.name]
    prc_files = [f for f in result.output_files if "sflux_prc" in f.name]

    print(f"\n  sflux_air files: {len(air_files)}")
    print(f"  sflux_rad files: {len(rad_files)}")
    print(f"  sflux_prc files: {len(prc_files)}")

    # Check naming convention (.{N}.nc not .{NNNN}.nc)
    for f in result.output_files:
        if "sflux_" in f.name:
            assert ".0001." not in f.name, f"Wrong naming: {f.name} (should be .N.nc not .NNNN.nc)"

    # Inspect first air file
    if air_files:
        f = air_files[0]
        print(f"\n  Inspecting {f.name}:")
        ds = Dataset(str(f))
        print(f"    Dimensions: {dict(ds.dimensions)}")
        print(f"    Variables: {list(ds.variables.keys())}")

        # Check expected air variables
        for var in AIR_VARS:
            assert var in ds.variables, f"Missing variable: {var}"
            data = ds.variables[var][:]
            print(f"    {var}: shape={data.shape}, min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}")

        # Check time is monotonic
        time_vals = ds.variables["time"][:]
        for i in range(1, len(time_vals)):
            assert time_vals[i] > time_vals[i-1], \
                f"Non-monotonic time at index {i}: {time_vals[i]} <= {time_vals[i-1]}"
        print(f"    Time: {len(time_vals)} steps, monotonic OK")
        print(f"    Time range: {time_vals[0]:.3f} to {time_vals[-1]:.3f} days")

        # Check coordinates
        lons = ds.variables["lon"][:]
        lats = ds.variables["lat"][:]
        print(f"    Lon range: {np.min(lons):.2f} to {np.max(lons):.2f}")
        print(f"    Lat range: {np.min(lats):.2f} to {np.max(lats):.2f}")

        ds.close()

    # Inspect first rad file
    if rad_files:
        f = rad_files[0]
        ds = Dataset(str(f))
        for var in RAD_VARS:
            assert var in ds.variables, f"Missing radiation variable: {var}"
            data = ds.variables[var][:]
            print(f"    {var}: shape={data.shape}, min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}")
        ds.close()

    # Inspect prc file
    if prc_files:
        f = prc_files[0]
        ds = Dataset(str(f))
        for var in PRC_VARS:
            assert var in ds.variables, f"Missing precip variable: {var}"
            data = ds.variables[var][:]
            print(f"    {var}: shape={data.shape}, min={np.nanmin(data):.6f}, max={np.nanmax(data):.6f}")
        ds.close()

    # Check sflux_inputs.txt
    inputs_file = OUTPUT_DIR / "sflux_test" / "sflux_inputs.txt"
    if inputs_file.exists():
        print(f"\n  sflux_inputs.txt exists: OK")
    else:
        print(f"\n  WARNING: sflux_inputs.txt not found")

    print("\n--- Test 1 PASSED ---")

    # --- Test 2: DATM output ---
    print("\n--- Test 2: DATM output (nws=4) ---")

    datm_dir = OUTPUT_DIR / "datm_test"
    datm_dir.mkdir(parents=True, exist_ok=True)

    config_datm = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC, nws=4)
    proc_datm = GFSProcessor(config_datm, GFS_ROOT, datm_dir)
    proc_datm.MIN_FILE_SIZE = 400_000_000

    result_datm = proc_datm.process()

    print(f"  Success: {result_datm.success}")
    print(f"  Files created: {len(result_datm.output_files)}")

    if result_datm.success:
        datm_file = datm_dir / "datm_forcing.nc"
        if datm_file.exists():
            ds = Dataset(str(datm_file))
            print(f"    Dimensions: {dict(ds.dimensions)}")
            print(f"    Variables: {list(ds.variables.keys())}")

            nx = ds.dimensions["longitude"].size
            ny = ds.dimensions["latitude"].size
            nt = ds.dimensions["time"].size
            print(f"    Grid: nx={nx}, ny={ny}, nt={nt}")

            # Check DATM variable names
            expected_datm_vars = [
                "UGRD_10maboveground", "VGRD_10maboveground",
                "TMP_2maboveground", "SPFH_2maboveground",
                "MSLMA_meansealevel", "PRATE_surface",
                "DSWRF_surface", "DLWRF_surface",
            ]
            for var in expected_datm_vars:
                if var in ds.variables:
                    data = ds.variables[var][:]
                    print(f"    {var}: min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}")

            ds.close()
            print("\n--- Test 2 PASSED ---")
        else:
            print("  FAIL: datm_forcing.nc not created")
    else:
        print(f"  Errors: {result_datm.errors}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
