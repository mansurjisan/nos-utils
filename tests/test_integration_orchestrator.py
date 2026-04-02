#!/usr/bin/env python3
"""
Integration test: Full PrepOrchestrator with real data.

Run inside Docker container:
    docker run --rm \
      -v /path/to/com:/data/com:ro \
      -v /path/to/nos-utils:/opt/nos-utils \
      -v /tmp/prep_test:/output \
      --entrypoint bash nosofs/secofs-ufs:latest -c '
        export PYTHONPATH=/opt/nos-utils:$PYTHONPATH
        python3 /opt/nos-utils/tests/test_integration_orchestrator.py
      '
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nos_utils.config import ForcingConfig
from nos_utils.orchestrator import PrepOrchestrator


def main():
    COM_ROOT = Path(os.environ.get("COM_ROOT", "/data/com"))
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
    PDY = os.environ.get("PDY", "20260324")
    CYC = int(os.environ.get("CYC", "12"))

    print("=" * 60)
    print("PrepOrchestrator Integration Test")
    print(f"  COM_ROOT: {COM_ROOT}")
    print(f"  PDY={PDY}  CYC={CYC:02d}z")
    print("=" * 60)

    # Verify available data
    data_sources = {}
    for name, subpath in [
        ("gfs", "gfs/v16.3"),
        ("hrrr", "hrrr/v4.1"),
        ("nwm", "nwm/v3.0"),
        ("rtofs", "rtofs"),
    ]:
        path = COM_ROOT / subpath
        if path.exists():
            data_sources[name] = str(path)
            print(f"  {name:6s}: {path} (found)")
        else:
            print(f"  {name:6s}: {path} (NOT FOUND)")

    # Create config
    config = ForcingConfig.for_secofs(pdy=PDY, cyc=CYC)

    # Build paths dict
    paths = {
        "output": str(OUTPUT_DIR / "prep_work"),
        "fix": str(OUTPUT_DIR / "fix"),  # No real fix files — param.nml will use fallback
    }
    paths.update(data_sources)

    # Create a minimal param.nml template in fix dir
    fix_dir = Path(paths["fix"])
    fix_dir.mkdir(parents=True, exist_ok=True)
    (fix_dir / "param.nml").write_text(
        "! SECOFS param.nml template for integration test\n"
        "&CORE\n"
        "  rnday = rnday_value  !total run time in days\n"
        "  dt = 120. !Time step in sec\n"
        "  ihot = 1\n"
        "  nws = 2\n"
        "/\n"
        "&OPT\n"
        "  start_year = start_year_value !int\n"
        "  start_month = start_month_value !int\n"
        "  start_day = start_day_value !int\n"
        "  start_hour = start_hour_value !double\n"
        "/\n"
    )

    # Run orchestrator
    print("\n--- Running PrepOrchestrator ---")
    orch = PrepOrchestrator(config, paths, run_name="secofs")

    # Lower GFS file size threshold for our test data (~491MB)
    from nos_utils.forcing.gfs import GFSProcessor
    GFSProcessor.MIN_FILE_SIZE = 400_000_000

    result = orch.run(phase="nowcast")

    # Print results
    print("\n" + result.summary())

    # Check output files
    print(f"\n--- Output Files ({len(result.all_output_files)}) ---")
    for f in result.all_output_files:
        if f.exists():
            size = f.stat().st_size
            print(f"  {f.name}: {size:,} bytes")
        else:
            print(f"  {f.name}: MISSING")

    # Validate key outputs
    work_dir = Path(paths["output"])

    checks = {
        "param.nml": work_dir / "param.nml",
        "bctides.in": work_dir / "bctides.in",
    }

    # Check sflux dir
    sflux_dir = work_dir / "sflux"
    if sflux_dir.exists():
        sflux_files = list(sflux_dir.glob("sflux_*.nc"))
        checks["sflux files"] = sflux_files[0] if sflux_files else None
        print(f"\n  sflux directory: {len(sflux_files)} files")

    print(f"\n--- Validation ---")
    all_ok = True
    for name, path in checks.items():
        if path and Path(path).exists():
            print(f"  [OK] {name}")
        else:
            print(f"  [MISSING] {name}")
            all_ok = False

    # Check param.nml content
    param_file = work_dir / "param.nml"
    if param_file.exists():
        content = param_file.read_text()
        if "rnday_value" in content:
            print("  [FAIL] param.nml still has placeholders!")
            all_ok = False
        else:
            # Extract rnday
            for line in content.split("\n"):
                if "rnday" in line and "=" in line:
                    print(f"  param.nml: {line.strip()}")
                    break

    if result.all_warnings:
        print(f"\n--- Warnings ({len(result.all_warnings)}) ---")
        for w in result.all_warnings[:10]:
            print(f"  {w}")

    if result.all_errors:
        print(f"\n--- Errors ({len(result.all_errors)}) ---")
        for e in result.all_errors:
            print(f"  {e}")

    print(f"\n{'=' * 60}")
    print(f"Orchestrator: {'PASSED' if result.success else 'FAILED'}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")
    print(f"{'=' * 60}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
