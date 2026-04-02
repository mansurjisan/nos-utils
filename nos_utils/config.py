"""
Forcing configuration dataclass.

Simple, standalone config with no dependency on nos_ofs.
The caller provides values directly or uses factory methods.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class ForcingConfig:
    """
    Configuration for forcing generation.

    All values are caller-provided. No YAML coupling or environment variable magic.

    Attributes:
        lon_min, lon_max, lat_min, lat_max: Domain bounding box (degrees)
        pdy: Production date YYYYMMDD
        cyc: Cycle hour (0, 6, 12, 18)
        nowcast_hours: Length of nowcast/hindcast period
        forecast_hours: Length of forecast period
        igrd_met: Grid interpolation method (0=native, 1+=interpolate to model grid)
        met_num: Number of met sources (1=GFS only, 2=GFS+HRRR)
        scale_hflux: Heat flux scaling factor
        grid_file: Path to model grid (hgrid.ll) for interpolation when igrd_met > 0
        nws: SCHISM forcing mode (2=sflux standalone, 4=DATM UFS-Coastal)
        variables: List of variables to extract (empty = use processor defaults)
    """

    # Domain bounds
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    # Cycle info
    pdy: str
    cyc: int

    # Run window
    nowcast_hours: int = 6
    forecast_hours: int = 48

    # Met settings
    igrd_met: int = 0
    met_num: int = 1
    scale_hflux: float = 1.0

    # Model grid (needed when igrd_met > 0)
    grid_file: Optional[Path] = None

    # Output mode
    nws: int = 2

    # Variable selection (empty = processor defaults)
    variables: List[str] = field(default_factory=list)

    # --- OBC (ocean boundary) settings ---
    # RTOFS ROI indices for 2D (ssh) extraction
    obc_roi_2d: Optional[dict] = None  # {x1, x2, y1, y2}
    # RTOFS ROI indices for 3D (T,S,U,V) extraction
    obc_roi_3d: Optional[dict] = None  # {x1, x2, y1, y2}
    # SSH offset applied to boundary elevation (meters)
    obc_ssh_offset: float = 0.0
    # Nudging enabled and timescale
    nudging_enabled: bool = False
    nudging_timescale_seconds: float = 86400.0
    # Number of vertical levels
    n_levels: int = 51

    # --- River settings ---
    # River config file mapping NWM reach IDs to SCHISM nodes
    river_config_file: Optional[Path] = None
    # River climatology for fallback
    river_clim_file: Optional[Path] = None
    # Default river temperature and salinity
    river_default_temp: float = 15.0
    river_default_salt: float = 0.0

    # --- Tidal settings ---
    # Tidal constituents to use
    tidal_constituents: List[str] = field(
        default_factory=lambda: ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
    )
    # Pre-computed bctides.in template
    bctides_template: Optional[Path] = None

    def __post_init__(self):
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max})")
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max})")
        if self.igrd_met > 0 and self.grid_file is None:
            raise ValueError("grid_file required when igrd_met > 0")

    @property
    def domain(self):
        """Return domain bounds as tuple (lon_min, lon_max, lat_min, lat_max)."""
        return (self.lon_min, self.lon_max, self.lat_min, self.lat_max)

    @classmethod
    def for_secofs(cls, pdy: str, cyc: int, **overrides) -> "ForcingConfig":
        """Factory with SECOFS defaults (SE Coastal Ocean Forecast System)."""
        defaults = dict(
            lon_min=-88.0, lon_max=-63.0,
            lat_min=17.0, lat_max=40.0,
            pdy=pdy, cyc=cyc,
            nowcast_hours=6, forecast_hours=48,
            met_num=2,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_secofs_ufs(cls, pdy: str, cyc: int, **overrides) -> "ForcingConfig":
        """Factory with SECOFS UFS-Coastal defaults (nws=4, DATM coupling)."""
        defaults = dict(
            lon_min=-88.0, lon_max=-63.0,
            lat_min=17.0, lat_max=40.0,
            pdy=pdy, cyc=cyc,
            nowcast_hours=6, forecast_hours=48,
            met_num=2, nws=4,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_stofs_3d_atl(cls, pdy: str, cyc: int, **overrides) -> "ForcingConfig":
        """Factory with STOFS-3D-ATL defaults (Atlantic Storm Surge)."""
        defaults = dict(
            lon_min=-98.5035, lon_max=-52.4867,
            lat_min=7.347, lat_max=52.5904,
            pdy=pdy, cyc=cyc,
            nowcast_hours=24, forecast_hours=108,
            met_num=2, n_levels=51,
            nudging_enabled=True,
            nudging_timescale_seconds=86400.0,
            obc_ssh_offset=0.04,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_stofs_3d_atl_ufs(cls, pdy: str, cyc: int, **overrides) -> "ForcingConfig":
        """Factory with STOFS-3D-ATL UFS-Coastal defaults (nws=4, DATM coupling)."""
        defaults = dict(
            lon_min=-98.5035, lon_max=-52.4867,
            lat_min=7.347, lat_max=52.5904,
            pdy=pdy, cyc=cyc,
            nowcast_hours=24, forecast_hours=108,
            met_num=2, nws=4, n_levels=51,
            nudging_enabled=True,
            nudging_timescale_seconds=86400.0,
            obc_ssh_offset=0.04,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def for_ensemble(
        cls, pdy: str, cyc: int,
        member: int = 0, n_members: int = 6,
        base_ofs: str = "stofs_3d_atl",
        **overrides,
    ) -> "ForcingConfig":
        """
        Factory for ensemble member forcing.

        Args:
            member: Member index (0=control, 1-N=perturbation)
            n_members: Total ensemble size
            base_ofs: Base OFS to inherit domain/timing from
        """
        if base_ofs == "secofs":
            cfg = cls.for_secofs(pdy, cyc)
        else:
            cfg = cls.for_stofs_3d_atl(pdy, cyc)

        # Ensemble uses met_num=1 for perturbation members (GEFS, not GFS+HRRR)
        # Control (member 0) can use GFS+HRRR blend
        defaults = dict(
            lon_min=cfg.lon_min, lon_max=cfg.lon_max,
            lat_min=cfg.lat_min, lat_max=cfg.lat_max,
            pdy=pdy, cyc=cyc,
            nowcast_hours=cfg.nowcast_hours,
            forecast_hours=cfg.forecast_hours,
            met_num=2 if member == 0 else 1,
            n_levels=cfg.n_levels,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def from_yaml(cls, yaml_path, **overrides) -> "ForcingConfig":
        """
        Load from a NOS-OFS YAML config file.

        Extracts grid.domain, model.run, and forcing.atmospheric sections.
        Falls back gracefully if PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for from_yaml(). Install with: pip install pyyaml")

        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Resolve _base inheritance
        if "_base" in data:
            base_name = data.pop("_base")
            base_path = yaml_path.parent / f"{base_name}.yaml"
            if base_path.exists():
                with open(base_path) as f:
                    base_data = yaml.safe_load(f)
                _deep_merge(base_data, data)
                data = base_data

        grid = data.get("grid", {})
        domain = grid.get("domain", {})
        model = data.get("model", {})
        run = model.get("run", model)
        forcing = data.get("forcing", {})
        atm = forcing.get("atmospheric", {})
        ocean = forcing.get("ocean", {})
        nudge = ocean.get("nudging", {}) if isinstance(ocean, dict) else {}
        river = forcing.get("river", {})
        tidal = forcing.get("tidal", {})
        runtime = model.get("runtime", {})

        # Infer met_num: 2 if secondary/HRRR is configured, else 1
        met_num = int(atm.get("met_num", 1))
        if met_num == 1 and (atm.get("secondary") or atm.get("forecast_source2")):
            met_num = 2

        # Vertical levels
        vert = model.get("vertical", {})
        n_levels = int(vert.get("nvrt", grid.get("n_levels", 51)))

        # NWS: check model.physics.nws
        physics = model.get("physics", {})
        nws = int(physics.get("nws", 2))

        # River config file
        river_files = river.get("files", {}) if isinstance(river, dict) else {}
        river_config_file = river_files.get("nwm_reach") or river_files.get("ctl_file")

        # Tidal template
        tidal_files = tidal.get("files", {}) if isinstance(tidal, dict) else {}
        bctides_template = tidal_files.get("harmonic_constants_ofs") or \
                          tidal_files.get("harmonic_constants_obc")

        # OBC SSH offset
        obc = ocean.get("obc", {}) if isinstance(ocean, dict) else {}
        obc_ssh_offset = float(obc.get("ssh_offset", 0.0))

        kwargs = dict(
            lon_min=domain.get("lon_min", -180.0),
            lon_max=domain.get("lon_max", 180.0),
            lat_min=domain.get("lat_min", -90.0),
            lat_max=domain.get("lat_max", 90.0),
            pdy=overrides.pop("pdy", ""),
            cyc=overrides.pop("cyc", 12),
            nowcast_hours=int(run.get("nowcast_hours", float(run.get("hindcast_days", 0.25)) * 24)),
            forecast_hours=int(run.get("forecast_hours", float(run.get("forecast_days", 5.0)) * 24)),
            igrd_met=int(grid.get("igrd_met", 0)),
            met_num=met_num,
            nws=nws,
            scale_hflux=float(atm.get("scale_hflux", 1.0)),
            n_levels=n_levels,
            nudging_enabled=nudge.get("enabled", False) if isinstance(nudge, dict) else False,
            nudging_timescale_seconds=float(
                nudge.get("timescale_seconds", nudge.get("timescale_days", 1.0) * 86400)
            ) if isinstance(nudge, dict) else 86400.0,
            obc_ssh_offset=obc_ssh_offset,
        )

        # Optional Path fields — only set if value is non-empty
        if river_config_file:
            kwargs["river_config_file"] = Path(river_config_file)
        if bctides_template:
            kwargs["bctides_template"] = Path(bctides_template)

        kwargs.update(overrides)
        return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override into base dict recursively (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
