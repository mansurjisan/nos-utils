# Python-Based Preprocessing for NOAA NOS-OFS
## Replacing Legacy Shell+Fortran with nos-utils

---

## Slide 1: The Problem

**Current NOS-OFS preprocessing (COMF/nosofs framework):**

- ~3,700 lines of shell scripts calling Fortran executables
- `exnos_ofs_prep.sh` → `nos_ofs_create_forcing_met.sh` (1,507 lines)
                       → `nos_ofs_create_forcing_obc.sh` (1,113 lines)
                       → `nos_ofs_create_forcing_river.sh` (299 lines)
                       → `nos_ofs_prep_schism_ctl.sh` (253 lines)

**Challenges:**
- Shell scripts are hard to unit test
- Fortran executables require full HPC compile chain (Intel, MPI, HDF5, NetCDF, BUFR, G2...)
- Adding a new OFS requires copying and modifying 4+ scripts
- No configuration inheritance — each OFS has its own `.ctl` control file
- Debugging requires reading 56K-line PBS error logs

---

## Slide 2: The Solution — nos-utils

**Standalone Python package** — no dependency on the existing nos_ofs codebase.

```
pip install nos-utils
```

```python
from nos_utils.config import ForcingConfig
from nos_utils.forcing import GFSProcessor

config = ForcingConfig.for_secofs(pdy="20260403", cyc=0)
gfs = GFSProcessor(config, "/data/gfs", "/data/sflux", phase="nowcast")
result = gfs.process()
# → sflux_air_1.1.nc, sflux_rad_1.1.nc, sflux_prc_1.1.nc
```

**Key design principles:**
- Each processor independently testable with mock data
- Configuration via simple Python dataclass — no shell env vars
- YAML config with inheritance (`_base: schism`)
- Same package works for SECOFS, STOFS-3D-ATL, CREOFS, any SCHISM OFS

---

## Slide 3: Architecture

```
nos-utils/
├── config.py                  ForcingConfig dataclass
├── nco_bridge.py              NCO env var → Python config bridge
├── orchestrator.py            Chains all processors in sequence
├── cli.py                     CLI: nos-utils prep --ofs secofs ...
│
├── forcing/
│   ├── gfs.py                 GFS 0.25° atmospheric (hourly)
│   ├── hrrr.py                HRRR 3km CONUS (LCC→latlon regrid)
│   ├── gefs.py                GEFS ensemble (RH→SPFH, APCP→PRATE)
│   ├── rtofs.py               RTOFS ocean boundary (→ boundary nodes)
│   ├── nwm.py                 NWM river (vsource/msource)
│   ├── tidal.py               Tidal (bctides.in + nodal corrections)
│   ├── param_nml.py           param.nml generation
│   ├── hotstart.py            Restart file discovery
│   ├── sflux_writer.py        SCHISM sflux NetCDF output
│   └── datm_writer.py         UFS-Coastal DATM output
│
└── io/
    ├── grib_extract.py        wgrib2 + cfgrib backends
    └── schism_grid.py         hgrid.gr3/ll parser (boundary nodes)
```

**20 modules | 124 unit tests | 7,500 lines of Python**

---

## Slide 4: Configuration — From .ctl to YAML

**Before (secofs.ctl — flat shell variables):**
```bash
MINLON=-88.0
MAXLON=-63.0
DBASE_MET_NOW=GFS
DBASE_MET_NOW2=HRRR
MET_NUM=2
DELT_MODEL=120.0
STEP_NU_VALUE=10800.0
```

**After (secofs.yaml — structured, inheritable):**
```yaml
_base: schism          # Inherits 100+ defaults

grid:
  domain: {lon_min: -88, lon_max: -63, lat_min: 17, lat_max: 40}
forcing:
  atmospheric: {primary: GFS, secondary: HRRR, met_num: 2}
  ocean:
    nudging: {enabled: true, timescale_seconds: 10800}
model:
  run: {nowcast_hours: 6, forecast_hours: 48}
```

**Adding a new OFS:**
```yaml
# new_ofs.yaml — 10 lines instead of 300
_base: schism
grid:
  domain: {lon_min: -95, lon_max: -80, lat_min: 25, lat_max: 32}
model:
  run: {nowcast_hours: 6, forecast_hours: 72}
```

---

## Slide 5: GRIB2 Extraction Pipeline

**Legacy:** Shell loops + Fortran executable per variable

```bash
$WGRIB2 $GRB2FILE -s | grep "UGRD:10 m" | $WGRIB2 -i ... -spread tmp.txt
# Repeat for 19 variables × 50+ files = 950 wgrib2 calls
$EXECnos/nos_ofs_create_forcing_met < Fortran_met.ctl
```

**Python:** Vectorized extraction + shared writer

```python
class GFSProcessor:
    GRIB2_VARIABLES = {
        "uwind": ("UGRD", "10 m above ground"),
        "prmsl": ("PRMSL", "mean sea level"),
        # ... 18 total
    }

    def process(self):
        files = self.find_input_files()        # Config-driven discovery
        data = self._extract_all(files)         # wgrib2 subprocess per var
        data = self._filter_to_time_window()    # Phase-aware filtering
        writer = SfluxWriter(output_dir)
        writer.write_all(data, times, lons, lats, base_date)
```

**SfluxWriter** — shared by GFS, HRRR, GEFS:
- Validates monotonic time axis (caught real bugs)
- `.{N}.nc` naming convention (not `.{NNNN}.nc`)
- Day-split files for SCHISM stack reading

---

## Slide 6: HRRR Lambert Conformal Regridding

**Problem:** HRRR native grid is Lambert Conformal (3km).
SCHISM's sflux reader can abort on non-convex LCC quadrilaterals.

**Solution:** Regrid to regular lat/lon before creating sflux:

```python
# wgrib2 with IPOLATES (spack-stack v3.6.0)
wgrib2 hrrr.grib2 \
  -match ":(UGRD|VGRD|TMP|SPFH|MSLMA|PRATE|DSWRF|DLWRF):" \
  -new_grid_winds earth \
  -new_grid latlon -88:834:0.03 17:768:0.03 \
  regridded.grb2
```

- U+V must be processed together for wind rotation (`-new_grid_winds earth`)
- Auto-detects spack wgrib2 (has IPOLATES) over system wgrib2 (doesn't)
- Falls back to scipy `griddata` interpolation if wgrib2 can't regrid

---

## Slide 7: Ocean Boundary Conditions — Pure Python

**Legacy:** Fortran `gen_3Dth_from_hycom` (compiled binary, ~2000 lines Fortran)

**Python replacement:**

```python
# 1. Parse SCHISM grid → extract boundary nodes
grid = SchismGrid.read("secofs.hgrid.ll")  # 1.68M nodes, 321 MB
bnd_lons, bnd_lats, bnd_depths, bnd_ids = grid.open_boundary_nodes()
# → 1,494 boundary nodes across 4 open boundary segments

# 2. Build nearest-neighbor index (cached per grid shape)
# Global 2D: (3298 × 4500) = 14.8M points
# Regional 3D: (1710 × 742) = 1.27M points

# 3. Interpolate RTOFS to boundary nodes
for rtofs_file in files:
    ssh_at_boundary = interpolate_2d(rtofs_lon, rtofs_lat, ssh, bnd_lons, bnd_lats)

# 4. Write SCHISM format
# elev2D.th.nc:  (138, 1494)         — SSH
# TEM_3D.th.nc:  (72, 1494, 40)      — Temperature
# SAL_3D.th.nc:  (72, 1494, 40)      — Salinity
# uv3D.th.nc:    (72, 1494, 40, 2)   — Velocity
```

Works with any SCHISM OFS — just change the grid file.

---

## Slide 8: Phase-Aware Processing

**Each processor knows which phase it's running:**

```
NOWCAST (6 hours):
  Start:     time_hotstart (from restart file)
  End:       cycle time (PDY + cyc)
  param.nml: rnday=0.5, ihot=1, start=hotstart_time
  bctides:   start=hotstart_time, nodal corrections for that time
  sflux:     GFS time window 09:00-03:00 → 16 hourly steps

FORECAST (48 hours):
  Start:     cycle time
  End:       cycle + 48h
  param.nml: rnday=2.0, ihot=2, start=cycle_time
  bctides:   start=cycle_time, different nodal corrections
  sflux:     GFS time window 21:00-03:00+2d → 52 hourly steps
```

Legacy shell computes these in separate code paths.
Python uses a single `phase` parameter throughout.

---

## Slide 9: Integration with NCO Workflow

**Zero changes to existing operational workflow required:**

```bash
# PBS script — one line activates Python prep
export USE_PYTHON_PREP=YES
${HOMEnos}/jobs/JNOS_OFS_PREP
```

**Hybrid mode** (default): Python for met/param/tidal, Fortran for OBC/river
**Full Python mode**: `FULL_PYTHON_PREP=YES` — Python handles everything

```
exnos_ofs_prep_python.sh:
  Step 0: source nos_ofs_launch.sh (sets NCO env vars)
  Step 1: Python nowcast  (met + param + tidal + OBC + river)
  Step 2: Python forecast (met + param + tidal + OBC + river)
  Step 3: Archive to COMOUT with NCO naming convention
```

**nco_bridge.py** translates NCO env vars → ForcingConfig:
```
PDY=20260403       →  config.pdy = "20260403"
COMINgfs=/lfs/...  →  paths["gfs"] = "/lfs/..."
FIXofs=/lfs/...    →  resolves grid_file, bctides_template
```

---

## Slide 10: Testing Strategy

**Unit tests (124 tests, no data needed):**
```bash
pytest -v  # Runs in 2.4 seconds
```
- Config validation, factory methods
- File discovery with mock directories
- sflux writer: dimension names, time monotonicity, naming convention
- GEFS: RH→SPFH conversion, APCP→PRATE conversion
- ESMF mesh: elementMask=1 (not 0)

**Integration tests (Docker container with real data):**
- GFS: 147 files → 9 sflux files, validated variable ranges
- HRRR: 54 files → regridded 834×768, 58% CONUS coverage
- SCHISM compatibility: bctides.in (23,909 lines), param.nml (701 lines), partition.prop (3.3M elements)

**WCOSS2 validation:**
- Side-by-side with Fortran prep output
- param.nml: rnday, ihot, start_time all match
- Time markers: byte-for-byte identical
- Station comparison plots: wind, pressure, temperature consistent

---

## Slide 11: Resource Comparison

| | Legacy (Shell+Fortran) | Python (nos-utils) |
|---|---|---|
| **CPUs** | 128 (MPI) | 1 |
| **Modules** | 20+ (Intel, MPI, HDF5, BUFR...) | 5 (Python, wgrib2, nco) |
| **Compile** | Required (Fortran) | None |
| **Lines of code** | ~3,700 shell + Fortran | ~7,500 Python |
| **Unit tests** | 0 | 124 |
| **Config format** | .ctl (flat) | YAML (inherited) |
| **Add new OFS** | Copy+modify 4 scripts | 10-line YAML file |
| **Wall time** | ~50 min | ~30 min |

---

## Slide 12: What's Next

- [ ] Run SCHISM with Python-generated forcing — validate water levels
- [ ] Multi-cycle consistency test (3 consecutive cycles)
- [ ] STOFS-3D-ATL integration (different domain, 7690 rivers)
- [ ] Nudging fields (TEM_nu.nc, SAL_nu.nc) in pure Python
- [ ] Post-processing module (station extraction, CO-OPS format)
- [ ] CI/CD pipeline (GitHub Actions for unit tests)
- [ ] Retire legacy shell scripts after 30-day parallel run

---

## Slide 13: Repository

**nos-utils:** https://github.com/mansurjisan/nos-utils
**nos-workflow:** https://github.com/mansurjisan/nos-workflow (branch: feature/python-prep)

```bash
# Install
pip install -e ".[full]"

# Run
nos-utils prep --ofs secofs --pdy 20260403 --cyc 0 \
  --gfs /data/gfs --hrrr /data/hrrr --fix /data/fix/secofs \
  --output /work/prep
```
