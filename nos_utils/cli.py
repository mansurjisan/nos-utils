"""
CLI entry point for nos-utils.

Usage:
    nos-utils prep --ofs secofs --pdy 20260324 --cyc 12 --output /work/prep/ \
                   --gfs /data/gfs --hrrr /data/hrrr --fix /data/fix/secofs
    nos-utils prep --ofs stofs_3d_atl --pdy 20260324 --cyc 12 --ufs --output /work/prep/
    nos-utils prep --yaml /path/to/secofs.yaml --pdy 20260324 --cyc 12 --output /work/prep/
    nos-utils list
"""

import argparse
import logging
import sys
from pathlib import Path


def cmd_prep(args):
    """Run prep orchestrator."""
    from .config import ForcingConfig
    from .orchestrator import PrepOrchestrator
    from .forcing.gfs import GFSProcessor

    # Build config
    if args.yaml:
        config = ForcingConfig.from_yaml(args.yaml, pdy=args.pdy, cyc=args.cyc)
    elif args.ofs == "secofs":
        if args.ufs:
            config = ForcingConfig.for_secofs_ufs(pdy=args.pdy, cyc=args.cyc)
        else:
            config = ForcingConfig.for_secofs(pdy=args.pdy, cyc=args.cyc)
    elif args.ofs == "stofs_3d_atl":
        config = ForcingConfig.for_stofs_3d_atl(pdy=args.pdy, cyc=args.cyc)
        if args.ufs:
            config.nws = 4
    else:
        print(f"Unknown OFS: {args.ofs}. Use --yaml for custom configs.")
        sys.exit(1)

    # Build paths
    paths = {"output": args.output}
    if args.fix:
        paths["fix"] = args.fix
    if args.gfs:
        paths["gfs"] = args.gfs
    if args.hrrr:
        paths["hrrr"] = args.hrrr
    if args.nwm:
        paths["nwm"] = args.nwm
    if args.rtofs:
        paths["rtofs"] = args.rtofs

    # Lower GFS file size for testing
    if args.min_gfs_size:
        GFSProcessor.MIN_FILE_SIZE = args.min_gfs_size

    # Run
    orch = PrepOrchestrator(config, paths, run_name=args.ofs or "secofs")
    result = orch.run(phase=args.phase)

    print(result.summary())
    sys.exit(0 if result.success else 1)


def cmd_list(args):
    """List available OFS configs and processors."""
    print("Available OFS factories:")
    print("  secofs          — SE Coastal (6h nowcast, 48h forecast)")
    print("  secofs_ufs      — SE Coastal UFS-Coastal (nws=4)")
    print("  stofs_3d_atl    — Atlantic Storm Surge (24h nowcast, 108h forecast)")
    print("  ensemble        — Ensemble member forcing")
    print()
    print("Processors:")
    from .forcing import __all__
    for name in sorted(__all__):
        if "Processor" in name:
            print(f"  {name}")


def main():
    parser = argparse.ArgumentParser(
        prog="nos-utils",
        description="NOS-OFS forcing generation utilities",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    # prep command
    prep = subparsers.add_parser("prep", help="Run prep orchestrator")
    prep.add_argument("--ofs", default="secofs", help="OFS name (secofs, stofs_3d_atl)")
    prep.add_argument("--pdy", required=True, help="Production date YYYYMMDD")
    prep.add_argument("--cyc", type=int, required=True, help="Cycle hour (0,6,12,18)")
    prep.add_argument("--phase", default="nowcast", choices=["nowcast", "forecast", "full"])
    prep.add_argument("--output", required=True, help="Output directory")
    prep.add_argument("--yaml", help="YAML config file (overrides --ofs)")
    prep.add_argument("--ufs", action="store_true", help="UFS-Coastal mode (nws=4)")
    prep.add_argument("--fix", help="FIX directory path")
    prep.add_argument("--gfs", help="GFS data root (COMINgfs)")
    prep.add_argument("--hrrr", help="HRRR data root (COMINhrrr)")
    prep.add_argument("--nwm", help="NWM data root (COMINnwm)")
    prep.add_argument("--rtofs", help="RTOFS data root (COMINrtofs)")
    prep.add_argument("--min-gfs-size", type=int, help="Min GFS file size in bytes")

    # list command
    subparsers.add_parser("list", help="List available OFS configs")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "prep":
        cmd_prep(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
