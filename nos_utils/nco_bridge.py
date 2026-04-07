"""
NCO environment variable bridge.

Reads standard NCO/ecFlow environment variables (PDY, cyc, COMINgfs, etc.)
and constructs a ForcingConfig + paths dict for the PrepOrchestrator.

This is the glue between the operational HPC job environment and nos-utils.

Usage from shell::

    export PYTHONPATH=/path/to/nos-utils:$PYTHONPATH
    python3 -c "
        from nos_utils.nco_bridge import config_from_env, run_prep
        run_prep(phase='nowcast')
    "

Usage from Python::

    from nos_utils.nco_bridge import config_from_env
    config, paths = config_from_env()
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import ForcingConfig

log = logging.getLogger(__name__)


def config_from_env(
    ofs_override: Optional[str] = None,
    yaml_override: Optional[str] = None,
) -> Tuple[ForcingConfig, Dict[str, str]]:
    """
    Build ForcingConfig + paths dict from NCO environment variables.

    Environment variables read:
        PDY, cyc          — Production date and cycle
        RUN               — OFS name (secofs, stofs_3d_atl, etc.)
        OFS_CONFIG        — Path to YAML config file (optional)
        COMINgfs, COMINhrrr, COMINrtofs, COMINnwm — Input data paths
        FIXofs            — Fix file directory
        DATA              — Working directory
        COMOUT            — Output archive directory
        COMIN             — Previous cycle output (for hotstart)
        USE_DATM          — "true" for UFS-Coastal mode (nws=4)

    Returns:
        (ForcingConfig, paths_dict)
    """
    pdy = os.environ.get("PDY", "")
    cyc = int(os.environ.get("cyc", "12"))
    ofs = ofs_override or os.environ.get("RUN", "secofs")

    if not pdy:
        raise EnvironmentError("PDY environment variable not set")

    log.info(f"NCO bridge: PDY={pdy} cyc={cyc:02d}z OFS={ofs}")

    # Build config from YAML or factory
    yaml_path = yaml_override or os.environ.get("OFS_CONFIG", "")
    if yaml_path and Path(yaml_path).exists():
        log.info(f"Loading config from YAML: {yaml_path}")
        config = ForcingConfig.from_yaml(yaml_path, pdy=pdy, cyc=cyc)
    else:
        # Use factory method based on OFS name
        use_datm = os.environ.get("USE_DATM", "").lower() == "true"

        factory_map = {
            "secofs": ForcingConfig.for_secofs_ufs if use_datm else ForcingConfig.for_secofs,
            "stofs_3d_atl": ForcingConfig.for_stofs_3d_atl_ufs if use_datm else ForcingConfig.for_stofs_3d_atl,
        }
        factory = factory_map.get(ofs)
        if factory is None:
            log.warning(f"No factory for '{ofs}', using secofs defaults")
            factory = ForcingConfig.for_secofs

        config = factory(pdy=pdy, cyc=cyc)

    # Build paths dict from environment
    paths = {}

    env_to_path = {
        "gfs": "COMINgfs",
        "hrrr": "COMINhrrr",
        "nwm": "COMINnwm",
        "rtofs": "COMINrtofs",
        "fix": "FIXofs",
        "output": "DATA",
        "comout": "COMOUT",
    }

    for key, env_var in env_to_path.items():
        val = os.environ.get(env_var, "")
        if val:
            paths[key] = val

    # Hotstart search directory.
    # Priority: RESTART_DIR (explicit) > COMIN (previous cycle) > COMOUT parent (heuristic)
    restart_dir = os.environ.get("RESTART_DIR", "")
    comout = os.environ.get("COMOUT", "")
    comin = os.environ.get("COMIN", "")
    if restart_dir:
        paths["restart"] = restart_dir
        log.info(f"Hotstart search dir (RESTART_DIR): {restart_dir}")
    elif comin:
        paths["restart"] = comin
        log.info(f"Hotstart search dir (COMIN): {comin}")
    elif comout:
        # Heuristic: COMOUT parent contains previous cycle dirs
        # e.g., COMOUT=/ptmp/com/nosofs/v3.7/secofs.20260402
        #        → parent = /ptmp/com/nosofs/v3.7/ (has secofs.20260401/)
        parent = str(Path(comout).parent)
        paths["restart"] = parent
        if not Path(parent).is_dir():
            log.warning(f"COMOUT parent {parent} does not exist, hotstart search may fail")
        else:
            log.info(f"Hotstart search dir (COMOUT parent): {parent}")

    # Ensure output dir exists
    if "output" not in paths:
        paths["output"] = os.environ.get("DATAROOT", "/tmp") + "/nos_prep"

    # Resolve relative file paths in config against FIXofs
    fix_dir = paths.get("fix", "")
    if fix_dir:
        fix_path = Path(fix_dir)
        # River config file
        if config.river_config_file and not Path(config.river_config_file).is_absolute():
            resolved = fix_path / config.river_config_file
            if resolved.exists():
                config.river_config_file = resolved
                log.info(f"Resolved river_config_file: {resolved}")
        # Bctides template
        if config.bctides_template and not Path(config.bctides_template).is_absolute():
            resolved = fix_path / config.bctides_template
            if resolved.exists():
                config.bctides_template = resolved
                log.info(f"Resolved bctides_template: {resolved}")
        # Grid file
        if config.grid_file and not Path(config.grid_file).is_absolute():
            resolved = fix_path / config.grid_file
            if resolved.exists():
                config.grid_file = resolved

    return config, paths


def run_prep(
    phase: str = "nowcast",
    ofs: Optional[str] = None,
    skip_legacy: bool = True,
) -> bool:
    """
    Run the prep orchestrator using NCO environment variables.

    Convenience function for calling from shell scripts.

    Args:
        phase: "nowcast", "forecast", or "full"
        ofs: OFS name override
        skip_legacy: If True (default), skip OBC/river (handled by legacy shell).
            Set False to run all steps including Python OBC/river.

    Returns:
        True if successful
    """
    from .orchestrator import PrepOrchestrator

    config, paths = config_from_env(ofs_override=ofs)
    run_name = ofs or os.environ.get("RUN", "secofs")

    orch = PrepOrchestrator(config, paths, run_name=run_name,
                           skip_legacy=skip_legacy)
    result = orch.run(phase=phase)

    print(result.summary())

    # Archive to COMOUT
    if result.success and "comout" in paths:
        orch.archive_to_comout(result, Path(paths["comout"]))

    return result.success
