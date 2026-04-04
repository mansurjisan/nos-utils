# SECOFS Preprocessing Pipeline Flowchart

## PBS Job Submission

```
qsub pbs/jnos_secofs_prep_00.pbs

Sets:
  PDY=20260403, cyc=00, OFS=secofs
  COMINgfs, COMINhrrr, COMINrtofs, COMINnwm
  FIXofs, DATA, COMOUT
  USE_PYTHON_PREP=YES
  FULL_PYTHON_PREP=YES
```

## JNOS_OFS_PREP Dispatch

```
USE_PYTHON_PREP=YES?
  YES → exnos_ofs_prep_python.sh
  NO  → exnos_ofs_prep.sh (legacy shell+Fortran, unchanged)
```

## exnos_ofs_prep_python.sh

### Step 0: Environment Setup

```
source nos_ofs_launch.sh secofs prep
  → Sets ~50 env vars: DBASE_MET_NOW, time_hotstart,
    GRIDFILE, OCEAN_MODEL, FIXofs, EXECnos, etc.

FULL_PYTHON_PREP=YES?
  YES → skip_legacy=False (Python does everything)
  NO  → skip_legacy=True  (Python met/param/tidal only)
```

### Step 1: nco_bridge.config_from_env()

```
NCO Environment Variables          ForcingConfig
─────────────────────────          ─────────────
PDY=20260403              ───►     pdy="20260403"
cyc=00                             cyc=0
OFS_CONFIG=secofs.yaml             lon_min=-88, lon_max=-63
FIXofs=/lfs/.../fix/secofs         lat_min=17, lat_max=40
COMINgfs=/lfs/.../gfs/v16.3       nowcast_hours=6
COMINhrrr=/lfs/.../hrrr/v4.1      forecast_hours=48
COMINrtofs=/lfs/.../rtofs/v2.5     met_num=2, nws=2
COMOUT=/lfs/.../secofs.20260403    grid_file=secofs.hgrid.ll
                                   bctides_template=secofs.bctides.in_template

Resolves relative paths against FIXofs:
  secofs.hgrid.ll         → /lfs/.../fix/secofs/secofs.hgrid.ll
  secofs.nwm.reach.dat    → /lfs/.../fix/secofs/secofs.nwm.reach.dat
  secofs.bctides.in_templ → /lfs/.../fix/secofs/secofs.bctides.in_template
```

---

## PrepOrchestrator.run(phase)

Runs twice: once for **nowcast**, once for **forecast**.

### Step 1: Hotstart

```
Search: $COMOUT parent (e.g., /lfs/.../com/nosofs/v3.7/)
  ├── secofs.20260401/
  ├── secofs.20260402/  ← has restart files
  └── secofs.20260403/

Parse cycle time from filename:
  secofs.t12z.20260402.rst.nowcast.nc → valid 2026-04-02 12:00

Select most recent file BEFORE current cycle (04-03 00z):
  → time_hotstart = 2026-04-02 12:00
  → ihot = 1 (hotstart found)

Link: $DATA/hotstart.nc → secofs.t12z.20260402.rst.nowcast.nc
```

### Step 2: GFS Atmospheric

```
Input: $COMINgfs/gfs.20260403/{00,06,12}/atmos/gfs.t*z.pgrb2.0p25.f*
Found: 147 files from multiple cycles

Dedup: removed 86 duplicate valid times (multi-cycle overlap)

Time window (phase-dependent):
  NOWCAST:  04-02 09:00 → 04-03 03:00 → kept 16/61
  FORECAST: 04-02 21:00 → 04-05 03:00 → kept 52/61

Extraction per file:
  wgrib2 -match ":UGRD:10 m above ground:"
         -small_grib -88:-63 17:40  (domain subset)
         -d 1 -no_header -bin      (binary dump, first record only)
  → numpy array (93 × 101) at 0.25°

Variables: uwind, vwind, prmsl, stmp, spfh, dlwrf, dswrf, prate

SfluxWriter → day-split NetCDF files:
  sflux_air_1.{1,2,3}.nc  (uwind, vwind, prmsl, stmp, spfh)
  sflux_rad_1.{1,2,3}.nc  (dlwrf, dswrf)
  sflux_prc_1.{1,2,3}.nc  (prate)
  sflux_inputs.txt
```

### Step 3: HRRR Secondary (optional, non-fatal)

```
Input: $COMINhrrr/hrrr.20260403/conus/hrrr.t*z.wrfsfcf*.grib2
Found: 54 files (hourly)
Time window: same as GFS per phase

Regrid: Lambert Conformal → regular lat/lon
  wgrib2 -new_grid_winds earth -new_grid latlon -88:834:0.03 17:768:0.03
  Uses spack wgrib2 v3.6.0 (has IPOLATES)
  All variables regridded together (U+V pair for wind rotation)

Extract from regridded file with skip_subset=True
  → 834 × 768 regular lat/lon grid at 0.03°

Output: sflux_air_2.{1,2,3}.nc (source_index=2 for SCHISM blending)
        sflux_rad_2.{1,2,3}.nc
        sflux_prc_2.{1,2,3}.nc

Note: MSLMA variable used instead of PRMSL (HRRR-specific)
```

### Step 4: NWM River

```
FULL_PYTHON_PREP=NO:
  → Skipped (legacy shell handles it)

FULL_PYTHON_PREP=YES:
  Read secofs.nwm.reach.dat → 2 NWM reaches (SECOFS has minimal rivers)
  Search $COMINnwm for channel_rt files
  Fallback: monthly climatology (April factor = 1.5×)

  Output:
    vsource.th     (55 timesteps × 2 rivers, flow in m³/s)
    msource.th     (temp=10°C, salt=0 PSU per river)
    source_sink.in (2 sources, 0 sinks)
```

### Step 5: RTOFS OBC

```
FULL_PYTHON_PREP=NO:
  → Skipped (legacy Fortran gen_3Dth_from_hycom handles it)

FULL_PYTHON_PREP=YES:
  Read secofs.hgrid.ll (321 MB, 1.68M nodes)
  Extract 1,494 open boundary nodes across 4 segments:
    Boundary 0: 1,199 nodes (main open ocean)
    Boundary 1:   281 nodes (deep boundary)
    Boundary 2:     8 nodes
    Boundary 3:     6 nodes

  RTOFS 2D (138 files, global 4500×3298):
    Build nearest-neighbor index (14.8M points → 1,494 nodes)
    For each file: interpolate SSH to boundary nodes
    Apply ssh_offset (+0.04m for STOFS, 0 for SECOFS)
    → elev2D.th.nc (138, 1494)

  RTOFS 3D (72 files, regional grids):
    Build separate interpolation indices per grid shape:
      US_east: (1710, 742) = 1.27M points
      Another: (1666, 1047) = 1.74M points
    For each file × each depth level (40 levels):
      Interpolate T/S/U/V to 1,494 boundary nodes
    → TEM_3D.th.nc (72, 1494, 40)
    → SAL_3D.th.nc (72, 1494, 40)
    → uv3D.th.nc   (72, 1494, 40)
```

### Step 6: Tidal

```
Read template: secofs.bctides.in_template (23,909 lines)

Phase-aware start time:
  NOWCAST:  04/02/2026 12:00:00 UTC (= time_hotstart)
  FORECAST: 04/03/2026 00:00:00 UTC (= cycle time)

Recompute nodal corrections for 8 constituents:
  M2:  f=0.966, u=355.3°    (principal lunar)
  S2:  f=1.000, u=0.0°      (principal solar, no nodal)
  N2:  f=0.966, u=233.2°
  K2:  f=1.293, u=28.0°     (largest variation)
  K1:  f=1.106, u=283.8°
  O1:  f=1.172, u=70.3°
  P1:  f=1.000, u=79.1°
  Q1:  f=1.172, u=308.1°

→ bctides.in (23,909 lines, 690 KB)
```

### Step 7: param.nml

```
Read template: secofs.param.nml (701 lines)

Substitute placeholders:

  NOWCAST:                       FORECAST:
  rnday     = 0.5000             rnday     = 2.0000
  start_year  = 2026             start_year  = 2026
  start_month = 04               start_month = 04
  start_day   = 02               start_day   = 03
  start_hour  = 12.0             start_hour  = 0.0
  ihot_value  = 1                ihot_value  = 2

  Nowcast: start from time_hotstart, ihot=1 (reset clock)
  Forecast: start from cycle time, ihot=2 (continue clock)

→ param.nml (36 KB)
```

### Step 8: Time Markers

```
→ time_hotstart.t00z    (2026040212)
→ time_nowcastend.t00z  (2026040300)
→ time_forecastend.t00z (2026040500)
→ base_date.t00z        (2026040212)
```

### Step 9: Archive to COMOUT

```
$DATA/sflux/sflux_*_1.*.nc  → secofs.t00z.20260403.met.{phase}.nc.tar
$DATA/sflux/sflux_*_2.*.nc  → secofs.t00z.20260403.met.{phase}.nc.2.tar
$DATA/elev2D + TEM/SAL_3D   → secofs.t00z.20260403.obc.{phase}.tar
$DATA/param.nml              → secofs.t00z.20260403.{phase}.in
$DATA/bctides.in             → secofs.t00z.20260403.bctides.in.{phase}
$DATA/source_sink.in         → secofs.source_sink.in
$DATA/vsource.th             → secofs.t00z.20260403.river.vsource.th
$DATA/msource.th             → secofs.t00z.20260403.river.msource.th
$DATA/sflux_inputs.txt       → sflux_inputs.txt
$DATA/time_*.t00z            → time_*.t00z
```

---

## Legacy Shell Phase (hybrid mode only)

When `FULL_PYTHON_PREP=NO`, after Python completes:

```
nos_ofs_create_forcing_river.sh
  → Fortran nos_ofs_create_forcing_river
  → Python schism_nwm_source_sink.py
  → nwm.source.sink.now.tar + fore.tar

nos_ofs_create_forcing_obc.sh
  → Fortran gen_3Dth_from_hycom
  → elev2D.th.nc (1801, 1488) at dt=120s
  → TEM_3D.th.nc (21, 1488, 63) at dt=3h
  → SAL_3D.th.nc (21, 1488, 63)
  → uv3D.th.nc   (21, 1488, 63, 2)
  → TEM_nu.nc    (21, 32613, 63) nudging
  → SAL_nu.nc    (21, 32613, 63) nudging
  → secofs.t00z.20260403.obc.tar (729 MB)
```

---

## Output Summary

| File | Size | Source |
|------|------|--------|
| met.nowcast.nc.tar (GFS) | ~19 MB | Python GFSProcessor |
| met.nowcast.nc.2.tar (HRRR) | ~1.1 GB | Python HRRRProcessor |
| met.forecast.nc.tar (GFS) | ~19 MB | Python GFSProcessor |
| met.forecast.nc.2.tar (HRRR) | ~1.1 GB | Python HRRRProcessor |
| obc.nowcast.tar | ~500 MB | Python RTOFSProcessor |
| obc.forecast.tar | ~500 MB | Python RTOFSProcessor |
| nowcast.in (param.nml) | 36 KB | Python ParamNmlProcessor |
| forecast.in (param.nml) | 36 KB | Python ParamNmlProcessor |
| bctides.in.nowcast | 690 KB | Python TidalProcessor |
| bctides.in.forecast | 690 KB | Python TidalProcessor |
| source_sink.in | 12 B | Python NWMProcessor |
| vsource.th | 1.4 KB | Python NWMProcessor |
| msource.th | 1.5 KB | Python NWMProcessor |
| sflux_inputs.txt | 275 B | Python SfluxWriter |
| time_hotstart.t00z | 11 B | Python Orchestrator |
| time_nowcastend.t00z | 11 B | Python Orchestrator |
| time_forecastend.t00z | 11 B | Python Orchestrator |
| base_date.t00z | 11 B | Python Orchestrator |
