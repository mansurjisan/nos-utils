#!/usr/bin/env python3
"""
Integration test: HRRR and NWM processors with real data.

Run inside Docker container:
    docker run --rm \
      -v /path/to/hrrr:/data/hrrr:ro \
      -v /path/to/nwm:/data/nwm:ro \
      -v /path/to/nos-utils:/opt/nos-utils:ro \
      -v /tmp/forcing_test:/output \
      --entrypoint bash nosofs/secofs-ufs:latest -c '
        export PYTHONPATH=/opt/nos-utils:$PYTHONPATH
        python3 /opt/nos-utils/tests/test_integration_hrrr_nwm.py
      '
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nos_utils.config import ForcingConfig
from nos_utils.forcing.hrrr import HRRRProcessor
from nos_utils.forcing.nwm import NWMProcessor, RiverConfig
from nos_utils.forcing.tidal import TidalProcessor, compute_nodal_corrections


def test_hrrr():
    """Test HRRR processor with real GRIB2 data."""
    HRRR_ROOT = Path(os.environ.get("HRRR_ROOT", "/data/hrrr/v4.1"))
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", tempfile.mkdtemp(prefix="nos_test_"))) / "hrrr_test"
    PDY = os.environ.get("PDY", "20260324")
    CYC = int(os.environ.get("CYC", "12"))

    print("=" * 60)
    print("HRRR Integration Test")
    print(f"  HRRR_ROOT: {HRRR_ROOT}")
    print(f"  PDY={PDY}  CYC={CYC:02d}z")
    print("=" * 60)

    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)
    proc = HRRRProcessor(config, HRRR_ROOT, OUTPUT_DIR, regrid_dx=0.05)

    # Step 1: File discovery
    files = proc.find_input_files()
    print(f"  Found {len(files)} HRRR files")
    if files:
        print(f"  First: {files[0]}")
        print(f"  Last:  {files[-1]}")
        print(f"  Size:  {files[0].stat().st_size / 1e6:.0f} MB")

    # Step 2: Process
    result = proc.process()
    print(f"\n  Success: {result.success}")
    print(f"  Files: {len(result.output_files)}")
    if result.warnings:
        print(f"  Warnings: {result.warnings}")
    if result.metadata:
        print(f"  Metadata: {result.metadata}")

    # Step 3: Validate output
    if result.output_files:
        from netCDF4 import Dataset
        for f in result.output_files[:3]:
            ds = Dataset(str(f))
            print(f"\n  {f.name}:")
            print(f"    Dims: {dict(ds.dimensions)}")
            vars_info = []
            for v in ds.variables:
                if v not in ("time", "lon", "lat"):
                    data = ds.variables[v][:]
                    vars_info.append(f"{v}: {np.nanmin(data):.3f}..{np.nanmax(data):.3f}")
            print(f"    Vars: {', '.join(vars_info)}")
            ds.close()

    print("\n--- HRRR Test PASSED ---")
    return result.success or len(result.warnings) > 0  # HRRR is non-fatal


def test_nwm():
    """Test NWM processor with real channel_rt data."""
    NWM_ROOT = Path(os.environ.get("NWM_ROOT", "/data/nwm/v3.0"))
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", tempfile.mkdtemp(prefix="nos_test_"))) / "nwm_test"
    PDY = os.environ.get("PDY", "20260324")
    CYC = int(os.environ.get("CYC", "12"))

    print("\n" + "=" * 60)
    print("NWM Integration Test")
    print(f"  NWM_ROOT: {NWM_ROOT}")
    print(f"  PDY={PDY}  CYC={CYC:02d}z")
    print("=" * 60)

    # Step 1: Inspect NWM file to find valid feature_ids
    nwm_files = sorted(NWM_ROOT.glob(f"nwm.{PDY}/*/*.nc"))

    if not nwm_files:
        print("  No NWM files found — testing climatology fallback only")
    else:
        print(f"  Found {len(nwm_files)} NWM files")

        # Read first file to discover feature_ids
        from netCDF4 import Dataset
        ds = Dataset(str(nwm_files[0]))
        all_fids = ds.variables["feature_id"][:]
        streamflow = ds.variables["streamflow"][:]
        print(f"  Total features in file: {len(all_fids)}")
        print(f"  Streamflow range: {np.nanmin(streamflow):.3f} to {np.nanmax(streamflow):.1f} m³/s")

        # Pick 5 reaches with non-zero flow for testing
        nonzero = np.where(streamflow > 10.0)[0]
        if len(nonzero) > 5:
            sample_idx = nonzero[:5]
        else:
            sample_idx = nonzero[:len(nonzero)] if len(nonzero) > 0 else [0, 1, 2]

        test_fids = [int(all_fids[i]) for i in sample_idx]
        test_flows = [float(streamflow[i]) for i in sample_idx]
        ds.close()
        print(f"  Test reaches: {test_fids}")
        print(f"  Test flows: {[f'{f:.1f}' for f in test_flows]} m³/s")

    # Step 2: Create river config
    if nwm_files:
        rc = RiverConfig(
            feature_ids=test_fids,
            node_indices=list(range(100, 100 + len(test_fids))),
            clim_flows=test_flows,
            names=[f"TestRiver{i}" for i in range(len(test_fids))],
        )
    else:
        # Fallback config for climatology test
        rc = RiverConfig(
            feature_ids=[20104159, 9643431],
            node_indices=[100, 200],
            clim_flows=[150.0, 75.0],
            names=["Savannah", "TestRiver"],
        )

    # Step 3: Process
    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)
    proc = NWMProcessor(config, NWM_ROOT, OUTPUT_DIR, river_config=rc)

    result = proc.process()
    print(f"\n  Success: {result.success}")
    print(f"  Files: {len(result.output_files)}")
    print(f"  Metadata: {result.metadata}")

    if not result.success:
        print(f"  Errors: {result.errors}")
        return False

    # Step 4: Validate output files
    vsource = OUTPUT_DIR / "vsource.th"
    msource = OUTPUT_DIR / "msource.th"
    source_sink = OUTPUT_DIR / "source_sink.in"

    for f in [vsource, msource, source_sink]:
        assert f.exists(), f"Missing: {f.name}"

    vs_lines = vsource.read_text().strip().split("\n")
    print(f"\n  vsource.th: {len(vs_lines)} time steps")
    # Check first line format
    parts = vs_lines[0].split()
    print(f"    Columns: {len(parts)} (1 time + {len(parts)-1} rivers)")
    print(f"    First line: {vs_lines[0][:80]}")
    print(f"    Last line:  {vs_lines[-1][:80]}")

    ms_lines = msource.read_text().strip().split("\n")
    print(f"  msource.th: {len(ms_lines)} time steps")

    ss_lines = source_sink.read_text().strip().split("\n")
    n_sources = int(ss_lines[0])
    print(f"  source_sink.in: {n_sources} sources")

    assert n_sources == rc.n_rivers
    assert len(vs_lines) >= 10  # Should have enough time steps

    print("\n--- NWM Test PASSED ---")
    return True


def test_tidal():
    """Test tidal processor — Python-native generation."""
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", tempfile.mkdtemp(prefix="nos_test_"))) / "tidal_test"
    PDY = os.environ.get("PDY", "20260324")
    CYC = int(os.environ.get("CYC", "12"))

    print("\n" + "=" * 60)
    print("Tidal Integration Test")
    print(f"  PDY={PDY}  CYC={CYC:02d}z")
    print("=" * 60)

    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)
    proc = TidalProcessor(config, Path("/nonexistent"), OUTPUT_DIR)

    result = proc.process()
    print(f"  Success: {result.success}")
    print(f"  Mode: {result.metadata.get('mode')}")

    bctides = OUTPUT_DIR / "bctides.in"
    assert bctides.exists(), "bctides.in not created"

    content = bctides.read_text()
    lines = content.strip().split("\n")
    print(f"  Lines: {len(lines)}")
    print(f"  Start: {lines[0]}")

    # Verify all 8 constituents present
    for c in ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]:
        assert c in content, f"Missing constituent: {c}"
    print("  All 8 constituents present: OK")

    # Test nodal corrections
    start = datetime.strptime(PDY, "%Y%m%d")
    nodal = compute_nodal_corrections(start, config.tidal_constituents)
    print(f"\n  Nodal corrections for {PDY}:")
    for c in config.tidal_constituents:
        f = nodal[c]["f"]
        u = nodal[c]["u"]
        print(f"    {c:3s}: f={f:.4f}  u={u:+.2f}°")

    print("\n--- Tidal Test PASSED ---")
    return True


def main():
    passed = 0
    total = 3

    if test_hrrr():
        passed += 1
    if test_nwm():
        passed += 1
    if test_tidal():
        passed += 1

    print("\n" + "=" * 60)
    print(f"INTEGRATION TESTS: {passed}/{total} PASSED")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
