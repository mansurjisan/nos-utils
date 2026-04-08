#!/usr/bin/env python3
"""
SCHISM compatibility test with real SECOFS fix files.

Validates that nos-utils output is compatible with actual SCHISM model input:
  - bctides.in from real template (~23K lines)
  - param.nml from real SECOFS template (3 namelists)
  - sflux time axis starts from base_date
  - Hotstart file discovery from real COMOUT
  - partition.prop from real hgrid.gr3 (1.68M elements)

Run inside Docker:
    docker run --rm \
      -v /path/to/com:/data/com:ro \
      -v /path/to/fix:/data/fix:ro \
      -v /path/to/nos-utils:/opt/nos-utils \
      -v /tmp/schism_compat_test:/output \
      --entrypoint bash nosofs/secofs-ufs:latest -c '
        export PYTHONPATH=/opt/nos-utils:$PYTHONPATH
        python3 /opt/nos-utils/tests/test_integration_schism_compat.py
      '
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nos_utils.config import ForcingConfig
from nos_utils.forcing.param_nml import ParamNmlProcessor
from nos_utils.forcing.tidal import TidalProcessor
from nos_utils.forcing.hotstart import HotstartProcessor
from nos_utils.forcing.partition import PartitionProcessor
from nos_utils.forcing.gfs import GFSProcessor
from nos_utils.forcing.sflux_writer import SfluxWriter


FIX_DIR = Path(os.environ.get("FIX_DIR", "/data/fix/secofs"))
COM_DIR = Path(os.environ.get("COM_DIR", "/data/com"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", tempfile.mkdtemp(prefix="nos_test_")))
PDY = os.environ.get("PDY", "20260324")
CYC = int(os.environ.get("CYC", "12"))

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))


def test_bctides():
    """Test 1: bctides.in from real SECOFS template."""
    print("\n--- Test 1: bctides.in from real template ---")

    template = FIX_DIR / "secofs.bctides.in_template"
    if not template.exists():
        # Try alternative names
        for alt in ["secofs.bctides.in", "bctides.in_template", "bctides.in"]:
            alt_path = FIX_DIR / alt
            if alt_path.exists():
                template = alt_path
                break

    if not template.exists():
        print(f"  SKIP: No bctides template found in {FIX_DIR}")
        return

    template_lines = template.read_text().strip().split("\n")
    print(f"  Template: {template.name} ({len(template_lines)} lines)")

    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)
    config.bctides_template = template

    out_dir = OUTPUT_DIR / "bctides_test"
    proc = TidalProcessor(config, FIX_DIR, out_dir)
    result = proc.process()

    check("TidalProcessor succeeds", result.success)

    bctides = out_dir / "bctides.in"
    if bctides.exists():
        content = bctides.read_text()
        lines = content.strip().split("\n")

        check("bctides.in > 1000 lines", len(lines) > 1000,
              f"got {len(lines)} lines (expected ~23K)")
        check("bctides.in has M2", "M2" in content)
        check("bctides.in has S2", "S2" in content)
        check("Start date updated", "2026" in lines[0],
              f"line 1: {lines[0][:40]}")

        print(f"  Lines: {len(lines)}")
        print(f"  Line 1 (start): {lines[0][:60]}")
        print(f"  Size: {bctides.stat().st_size:,} bytes")


def test_param_nml():
    """Test 2: param.nml from real SECOFS template."""
    print("\n--- Test 2: param.nml from real template ---")

    # Find the real param.nml
    template = None
    for name in ["secofs.param.nml", "param.nml", "secofs_param.nml"]:
        p = FIX_DIR / name
        if p.exists():
            template = p
            break

    if template is None:
        print(f"  SKIP: No param.nml template found in {FIX_DIR}")
        # List what's in fix dir
        print(f"  Fix dir contents: {[f.name for f in sorted(FIX_DIR.glob('*param*'))]}")
        return

    template_content = template.read_text()
    print(f"  Template: {template.name} ({len(template_content)} bytes)")

    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)
    out_dir = OUTPUT_DIR / "param_test"

    proc = ParamNmlProcessor(config, FIX_DIR, out_dir,
                             template_name=template.name, phase="nowcast")
    result = proc.process()

    check("ParamNmlProcessor succeeds", result.success,
          str(result.errors) if not result.success else "")

    param = out_dir / "param.nml"
    if param.exists():
        content = param.read_text()

        check("No placeholders remain",
              "rnday_value" not in content and "start_year_value" not in content)
        check("Has &CORE namelist", "&CORE" in content or "&core" in content.lower())
        check("Has &OPT namelist", "&OPT" in content or "&opt" in content.lower())
        check("Has &SCHOUT namelist", "&SCHOUT" in content or "&schout" in content.lower())

        # Check rnday for 6h nowcast
        for line in content.split("\n"):
            if "rnday" in line.lower() and "=" in line and "!" not in line.split("=")[0]:
                val = line.split("=")[1].strip().split()[0].rstrip(",")
                try:
                    rnday = float(val)
                    check("rnday = 0.25 (6h nowcast)", abs(rnday - 0.25) < 0.01,
                          f"got rnday={rnday}")
                except ValueError:
                    pass
                print(f"  {line.strip()}")
                break

        # Check start time
        for line in content.split("\n"):
            if "start_year" in line and "=" in line:
                check("start_year = 2026", "2026" in line, line.strip())
                break

        print(f"  Size: {param.stat().st_size:,} bytes")
        print(f"  Lines: {len(content.strip().split(chr(10)))}")


def test_hotstart():
    """Test 3: Hotstart discovery from real COMOUT."""
    print("\n--- Test 3: Hotstart file discovery ---")

    nosofs_dir = COM_DIR / "nosofs" / "v3.7"
    if not nosofs_dir.exists():
        nosofs_dir = COM_DIR / "nosofs"

    print(f"  Searching in: {nosofs_dir}")
    if nosofs_dir.exists():
        subdirs = sorted(nosofs_dir.glob("secofs.*"))
        print(f"  Found {len(subdirs)} secofs.* directories")
        for d in subdirs[:3]:
            hotstart_files = list(d.glob("*hotstart*")) + list(d.glob("*restart*"))
            print(f"    {d.name}: {len(hotstart_files)} hotstart/restart files")

    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)
    out_dir = OUTPUT_DIR / "hotstart_test"

    proc = HotstartProcessor(config, nosofs_dir, out_dir, run_name="secofs")
    result = proc.process()

    print(f"  Success: {result.success}")
    print(f"  ihot: {result.metadata.get('ihot', 'N/A')}")
    if result.warnings:
        print(f"  Warnings: {result.warnings}")
    if result.metadata.get("source_file"):
        print(f"  Source: {result.metadata['source_file']}")

    check("HotstartProcessor succeeds", result.success)


def test_partition():
    """Test 4: partition.prop from real hgrid.gr3."""
    print("\n--- Test 4: partition.prop from real grid ---")

    grid_file = None
    for name in ["secofs.hgrid.gr3", "hgrid.gr3", "secofs.hgrid.ll"]:
        p = FIX_DIR / name
        if p.exists():
            grid_file = p
            break

    if grid_file is None:
        print(f"  SKIP: No hgrid.gr3 found in {FIX_DIR}")
        return

    print(f"  Grid: {grid_file.name} ({grid_file.stat().st_size / 1e6:.0f} MB)")

    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)
    out_dir = OUTPUT_DIR / "partition_test"
    nprocs = 120  # Typical SECOFS

    proc = PartitionProcessor(config, FIX_DIR, out_dir,
                              nprocs=nprocs, grid_file=grid_file)
    result = proc.process()

    check("PartitionProcessor succeeds", result.success,
          str(result.errors) if not result.success else "")

    if result.success:
        n_elem = result.metadata["n_elements"]
        check("n_elements > 1M", n_elem > 1_000_000,
              f"got {n_elem:,}")
        print(f"  Elements: {n_elem:,}")
        print(f"  Ranks: {nprocs}")

        # Verify partition.prop
        partition_file = out_dir / "partition.prop"
        if partition_file.exists():
            lines = partition_file.read_text().strip().split("\n")
            check("partition.prop has correct line count",
                  len(lines) == n_elem,
                  f"got {len(lines)}, expected {n_elem}")
            ranks = set(int(r) for r in lines[:1000])
            check("Uses all ranks 0 to nprocs-1",
                  len(ranks) == nprocs,
                  f"got {len(ranks)} unique ranks")
            print(f"  File size: {partition_file.stat().st_size / 1e6:.1f} MB")


def test_sflux_time_axis():
    """Test 5: sflux time axis starts from base_date (not PDY)."""
    print("\n--- Test 5: sflux time axis validation ---")

    gfs_dir = COM_DIR / "gfs" / "v16.3"
    if not gfs_dir.exists():
        print(f"  SKIP: GFS data not found at {gfs_dir}")
        return

    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)
    out_dir = OUTPUT_DIR / "sflux_time_test"

    proc = GFSProcessor(config, gfs_dir, out_dir)
    proc.MIN_FILE_SIZE = 400_000_000

    result = proc.process()
    check("GFS processing succeeds", result.success)

    if not result.success:
        return

    # Check time axis of first sflux file
    from netCDF4 import Dataset

    air_files = [f for f in result.output_files if "sflux_air" in f.name]
    if not air_files:
        print("  No sflux_air files found")
        return

    ds = Dataset(str(air_files[0]))
    time_var = ds.variables["time"]
    time_vals = time_var[:]
    time_units = time_var.units

    print(f"  File: {air_files[0].name}")
    print(f"  Time units: {time_units}")
    print(f"  Time[0]: {time_vals[0]:.6f}")
    print(f"  Time[-1]: {time_vals[-1]:.6f}")
    print(f"  N timesteps: {len(time_vals)}")

    # base_date should be cycle_time - nowcast_hours
    cycle_dt = datetime.strptime(PDY, "%Y%m%d") + timedelta(hours=CYC)
    expected_base = cycle_dt - timedelta(hours=config.nowcast_hours)
    expected_base_str = expected_base.strftime("%Y-%m-%d")

    check("Time units reference base_date (not PDY)",
          expected_base_str in time_units,
          f"expected '{expected_base_str}' in '{time_units}'")

    check("Time[0] >= 0 (starts at or after base_date)",
          time_vals[0] >= 0.0,
          f"time[0]={time_vals[0]}")

    # Check monotonicity
    monotonic = all(time_vals[i] < time_vals[i+1] for i in range(len(time_vals)-1))
    check("Time axis is monotonically increasing", monotonic)

    # Check for NaN in data
    for var in ["uwind", "vwind", "prmsl", "stmp", "spfh"]:
        if var in ds.variables:
            data = ds.variables[var][:]
            n_valid = np.count_nonzero(~np.isnan(data) & (data != -9999.0))
            n_total = data.size
            pct = 100 * n_valid / n_total
            check(f"{var} has >90% valid data", pct > 90, f"{pct:.1f}% valid")

    ds.close()


def main():
    print("=" * 60)
    print("SCHISM Compatibility Test — Real SECOFS Fix Files")
    print(f"  FIX_DIR: {FIX_DIR}")
    print(f"  COM_DIR: {COM_DIR}")
    print(f"  PDY={PDY}  CYC={CYC:02d}z")
    print("=" * 60)

    # Check fix dir
    if not FIX_DIR.exists():
        print(f"\nFATAL: FIX_DIR not found: {FIX_DIR}")
        sys.exit(1)

    fix_files = list(FIX_DIR.glob("secofs.*"))
    print(f"  Fix files: {len(fix_files)}")
    for f in sorted(fix_files)[:10]:
        print(f"    {f.name} ({f.stat().st_size / 1e6:.1f} MB)")

    test_bctides()
    test_param_nml()
    test_hotstart()
    test_partition()
    test_sflux_time_axis()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
