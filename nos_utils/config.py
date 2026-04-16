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
    gfs_resolution: str = "0p50"  # SECOFS default; STOFS overrides to "0p25"
    scale_hflux: float = 1.0

    # Model grid (needed when igrd_met > 0)
    grid_file: Optional[Path] = None

    # Output mode
    nws: int = 2

    # Variable selection (empty = processor defaults)
    variables: List[str] = field(default_factory=list)

    # --- HRRR-specific domain (for STOFS, HRRR domain differs from GFS domain) ---
    hrrr_lon_min: Optional[float] = None  # None = use main domain
    hrrr_lon_max: Optional[float] = None
    hrrr_lat_min: Optional[float] = None
    hrrr_lat_max: Optional[float] = None

    # --- OBC (ocean boundary) settings ---
    # RTOFS ROI indices for 2D (ssh) extraction
    obc_roi_2d: Optional[dict] = None  # {x1, x2, y1, y2}
    # RTOFS ROI indices for 3D (T,S,U,V) extraction
    obc_roi_3d: Optional[dict] = None  # {x1, x2, y1, y2}
    # RTOFS ROI indices for nudging (slightly larger domain than OBC)
    nudge_roi_3d: Optional[dict] = None  # {x1, x2, y1, y2}
    # SSH offset applied to boundary elevation (meters)
    # Geoid-to-MSL datum offset (meters). OFS-specific — verify from Fortran source.
    # SECOFS: 1.25 (confirmed). STOFS-3D-ATL: 0.04. Other OFS: 0.0 until verified.
    obc_ssh_offset: float = 0.0
    # ADT satellite SSH blending (STOFS-3D-ATL uses CMEMS ADT to correct boundary SSH)
    adt_enabled: bool = False
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
    # Fortran nos_ofs_create_forcing_river uses 10.0 for msource.th temperature
    river_default_temp: float = 10.0
    river_default_salt: float = 0.0
    # NWM product type (STOFS uses medium_range_mem1, SECOFS uses analysis_assim)
    nwm_product: str = "analysis_assim"
    # Target and minimum NWM file counts for STOFS-style assembly
    nwm_n_list_target: int = 55
    nwm_n_list_min: int = 31

    # St. Lawrence River climatology (STOFS-3D-ATL only).
    # When True, the orchestrator runs StLawrenceProcessor which reads the
    # Environment Canada hydrometric CSV at $COMINlaw/<pdy>/can_streamgauge/
    # and produces flux.th / TEM_1.th. Falls back to previous-day CSV and
    # then to the previous cycle's archived .riv.obs.* files.
    st_lawrence_enabled: bool = False
    st_lawrence_csv_name: str = "02OA016_hydrometric.csv"

    # Dynamic SSH adjustment (STOFS-3D-ATL NOAA tide-gauge bias correction).
    # When True, the orchestrator runs DynamicAdjustProcessor after RTOFS
    # to subtract a per-cycle bias from elev2D.th.nc using 11 NOAA
    # reference stations. Needs the previous cycle's staout_1, param.nml,
    # and average_bias file in $COMOUT_PREV; degrades gracefully otherwise.
    dynamic_adjust_enabled: bool = False
    dynamic_adjust_window_days: int = 2

    # Minimum number of time records required in RTOFS-derived OBC files
    # (elev2D.th.nc, TEM_3D.th.nc, SAL_3D.th.nc, uv3D.th.nc). When any
    # file falls short, the orchestrator copies the previous cycle's
    # archived OBC from `$COMOUT_PREV/rerun/`. Set to 0 to disable the check.
    # Operational STOFS gates on N_dim_cr_max=21 — anything smaller triggers
    # the fallback (see stofs_3d_atl_create_obc_3d_th_non_adjust.sh:688).
    obc_min_timesteps: int = 21

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
        # grid_file is needed for igrd_met > 0 (met interpolation) or RTOFS OBC
        # Don't raise error here — it's resolved later by nco_bridge from FIXofs

    @property
    def domain(self):
        """Return domain bounds as tuple (lon_min, lon_max, lat_min, lat_max)."""
        return (self.lon_min, self.lon_max, self.lat_min, self.lat_max)

    @property
    def hrrr_domain(self):
        """HRRR domain bounds (falls back to main domain if not set)."""
        if self.hrrr_lon_min is not None:
            return (self.hrrr_lon_min, self.hrrr_lon_max,
                    self.hrrr_lat_min, self.hrrr_lat_max)
        return self.domain

    @classmethod
    def for_secofs(cls, pdy: str, cyc: int, **overrides) -> "ForcingConfig":
        """Factory with SECOFS defaults (SE Coastal Ocean Forecast System)."""
        defaults = dict(
            lon_min=-88.0, lon_max=-63.0,
            lat_min=17.0, lat_max=40.0,
            pdy=pdy, cyc=cyc,
            nowcast_hours=6, forecast_hours=48,
            met_num=2,
            obc_ssh_offset=1.25,  # Geoid-to-MSL datum offset for SECOFS
            nudging_enabled=True,  # COMF Fortran generates TEM_nu/SAL_nu
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
            obc_ssh_offset=1.25,
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
            gfs_resolution="0p25",
            met_num=2, n_levels=51,
            # HRRR domain (different from GFS/model domain)
            hrrr_lon_min=-98.5, hrrr_lon_max=-49.5,
            hrrr_lat_min=5.5, hrrr_lat_max=50.0,
            # OBC settings
            obc_roi_2d={"x1": 2805, "x2": 2923, "y1": 1598, "y2": 2325},
            obc_roi_3d={"x1": 482, "x2": 600, "y1": 94, "y2": 821},
            nudge_roi_3d={"x1": 422, "x2": 600, "y1": 94, "y2": 835},
            obc_ssh_offset=0.04,
            adt_enabled=True,
            # Nudging
            nudging_enabled=True,
            nudging_timescale_seconds=86400.0,
            # NWM river settings (medium_range_mem1, 121 target files)
            nwm_product="medium_range_mem1",
            nwm_n_list_target=121,
            nwm_n_list_min=97,
            # St. Lawrence climatology — always on for STOFS-3D-ATL.
            st_lawrence_enabled=True,
            # Dynamic SSH adjust — operational default for STOFS-3D-ATL.
            dynamic_adjust_enabled=True,
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
            gfs_resolution="0p25",
            met_num=2, nws=4, n_levels=51,
            hrrr_lon_min=-98.5, hrrr_lon_max=-49.5,
            hrrr_lat_min=5.5, hrrr_lat_max=50.0,
            obc_roi_2d={"x1": 2805, "x2": 2923, "y1": 1598, "y2": 2325},
            obc_roi_3d={"x1": 482, "x2": 600, "y1": 94, "y2": 821},
            nudge_roi_3d={"x1": 422, "x2": 600, "y1": 94, "y2": 835},
            obc_ssh_offset=0.04,
            adt_enabled=True,
            nudging_enabled=True,
            nudging_timescale_seconds=86400.0,
            nwm_product="medium_range_mem1",
            nwm_n_list_target=121,
            nwm_n_list_min=97,
            st_lawrence_enabled=True,
            dynamic_adjust_enabled=True,
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
        river_config_file = river_files.get("ctl_file") or river_files.get("nwm_reach")

        # Tidal template
        tidal_files = tidal.get("files", {}) if isinstance(tidal, dict) else {}
        bctides_template = tidal_files.get("harmonic_constants_ofs") or \
                          tidal_files.get("harmonic_constants_obc")

        # Grid file (hgrid.ll for boundary node extraction)
        grid_files = grid.get("files", {})
        grid_file = grid_files.get("horizontal_ll") or grid_files.get("horizontal")

        # OBC settings
        obc = ocean.get("obc", {}) if isinstance(ocean, dict) else {}
        obc_ssh_offset = float(obc.get("ssh_offset", 0.0))

        # ROI indices for RTOFS subsetting (STOFS-style index-based)
        roi_2ds = obc.get("roi_2ds", {})
        roi_3dz = obc.get("roi_3dz", {})
        nudge_roi = nudge.get("roi_3dz", {}) if isinstance(nudge, dict) else {}

        # HRRR blend domain (may differ from main domain)
        hrrr_blend = atm.get("hrrr_blend", {})

        # ADT satellite SSH blending
        adt = ocean.get("adt", {}) if isinstance(ocean, dict) else {}

        # NWM river product and target counts
        river_product = river.get("primary", "nwm") if isinstance(river, dict) else "nwm"
        nwm_product = "medium_range_mem1" if river_product == "nwm" and \
            river.get("version", "") == "v3.0" else "analysis_assim"

        # GFS resolution: "0.25", "0.50", or "sflux" (surface flux files)
        gfs_cfg = atm.get("gfs", {})
        gfs_resolution = gfs_cfg.get("resolution", "0.50")
        if gfs_resolution != "sflux":
            gfs_resolution = gfs_resolution.replace(".", "p")  # "0.25" -> "0p25"

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
            gfs_resolution=gfs_resolution,
            nws=nws,
            scale_hflux=float(atm.get("scale_hflux", 1.0)),
            n_levels=n_levels,
            nudging_enabled=nudge.get("enabled", False) if isinstance(nudge, dict) else False,
            nudging_timescale_seconds=float(
                nudge.get("timescale_seconds", nudge.get("timescale_days", 1.0) * 86400)
            ) if isinstance(nudge, dict) else 86400.0,
            obc_ssh_offset=obc_ssh_offset,
            adt_enabled=adt.get("enabled", False) if isinstance(adt, dict) else False,
            nwm_product=nwm_product,
        )

        # HRRR domain bounds (optional, falls back to main domain)
        if hrrr_blend and hrrr_blend.get("enabled", False):
            kwargs["hrrr_lon_min"] = float(hrrr_blend.get("lon_min", domain.get("lon_min", -180.0)))
            kwargs["hrrr_lon_max"] = float(hrrr_blend.get("lon_max", domain.get("lon_max", 180.0)))
            kwargs["hrrr_lat_min"] = float(hrrr_blend.get("lat_min", domain.get("lat_min", -90.0)))
            kwargs["hrrr_lat_max"] = float(hrrr_blend.get("lat_max", domain.get("lat_max", 90.0)))

        # OBC ROI indices (STOFS-style)
        if roi_2ds:
            kwargs["obc_roi_2d"] = {k: int(v) for k, v in roi_2ds.items()}
        if roi_3dz:
            kwargs["obc_roi_3d"] = {k: int(v) for k, v in roi_3dz.items()}
        if nudge_roi:
            kwargs["nudge_roi_3d"] = {k: int(v) for k, v in nudge_roi.items()}

        # NWM target counts
        if isinstance(river, dict):
            n_target = river.get("n_list_target")
            n_min = river.get("n_list_min")
            if n_target:
                kwargs["nwm_n_list_target"] = int(n_target)
            if n_min:
                kwargs["nwm_n_list_min"] = int(n_min)

        # St. Lawrence River climatology flag
        if isinstance(river, dict):
            stl = river.get("st_lawrence", {})
            if isinstance(stl, dict):
                kwargs["st_lawrence_enabled"] = bool(stl.get("enabled", False))
                csv_name = stl.get("csv_name")
                if csv_name:
                    kwargs["st_lawrence_csv_name"] = str(csv_name)

        # Dynamic SSH adjust: explicitly controlled by obc.obc_mode, or by
        # obc.dynamic_adjust.enabled if specified. For STOFS-3D-ATL YAMLs
        # that omit both (including `_base` inheritance), default to True
        # so from_yaml matches the operational semantics of the
        # for_stofs_3d_atl() factory. Non-STOFS OFSes leave it False.
        obc_mode = obc.get("obc_mode") if isinstance(obc, dict) else None
        dyn_cfg = obc.get("dynamic_adjust", {}) if isinstance(obc, dict) else {}
        if isinstance(dyn_cfg, dict) and "enabled" in dyn_cfg:
            kwargs["dynamic_adjust_enabled"] = bool(dyn_cfg["enabled"])
        elif obc_mode is not None:
            kwargs["dynamic_adjust_enabled"] = (str(obc_mode) == "dynamic_adjust")
        else:
            # Neither flag present: enable by default for STOFS-3D-ATL,
            # which is identified by having OBC ROI indices (the index-based
            # subsetting is STOFS-only).
            kwargs["dynamic_adjust_enabled"] = bool(roi_2ds or roi_3dz)

        # Optional Path fields — only set if value is non-empty
        # River config: try sources_json first (STOFS), then ctl_file (SECOFS)
        river_files = river.get("files", {}) if isinstance(river, dict) else {}
        river_config_file = river_files.get("sources_json") or \
                           river_files.get("ctl_file") or \
                           river_files.get("nwm_reach")
        if river_config_file:
            kwargs["river_config_file"] = Path(river_config_file)
        if bctides_template:
            kwargs["bctides_template"] = Path(bctides_template)
        if grid_file:
            kwargs["grid_file"] = Path(grid_file)

        kwargs.update(overrides)
        return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> None:
    """Merge override into base dict recursively (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
