# Audit Issues — All Resolved (2026-04-04)

## MEDIUM Priority — Fixed

### 1. RTOFS 2D file dedup — FIXED
- Added `_parse_rtofs_hour()` and `_sort_and_dedup()` to RTOFSProcessor
- Files sorted by valid time (parsed from filename), nowcast preferred over forecast
- Phase-based time window filtering when phase is set

### 2. RTOFS 3D takes only first time step — FIXED
- Changed `data = data[0]` to iterate over all time steps in each file
- Each time step produces a separate boundary profile entry

### 3. RTOFS NaN leak when ALL boundary nodes are on land — FIXED
- Added `n_land == n_bnd` branch that fills with domain-mean ocean value
- Logs warning when this fallback is used

### 4. RTOFS/NWM don't receive phase parameter — FIXED
- Orchestrator now passes `phase` and `time_hotstart` to `_run_rtofs()` and `_run_nwm()`
- RTOFSProcessor: new `phase`, `time_hotstart` params + `_get_time_window()` method
- NWMProcessor: new `phase`, `time_hotstart` params stored for future use

### 5. Tidal V0 (equilibrium argument) not computed — FIXED
- Added astronomical argument V0 computation using Doodson numbers
- `compute_nodal_corrections()` now returns `{"f", "u", "v0"}` per constituent
- Template and python-native modes both write V0+u (mod 360) as equilibrium argument

### 6. param_nml start_hour formatted as "12.0" not "12" — FIXED
- Changed `f"{start_dt.hour:.1f}"` to `str(start_dt.hour)` in `_compute_substitutions()`
- SCHISM Fortran expects integer start_hour

### 7. Tidal processor not critical in orchestrator — FIXED
- Added "TIDAL" to `critical_sources` set (was only GFS and PARAM_NML)
- SCHISM crashes at runtime without bctides.in

## LOW Priority — Fixed

### 8. GFS _compute_base_date is phase-unaware — FIXED
- Both GFS and HRRR `_compute_base_date()` now use `time_hotstart` when available
- Falls back to `cycle - nowcast_hours` (phase-independent — sflux uses continuous time axis)

### 9. Data array length guards use silent truncation — FIXED
- Added `log.warning()` before every `if i < len(arrays)` guard in 6 locations:
  - `gfs.py`: `_filter_to_time_window`, dedup section
  - `hrrr.py`: sort section, `_filter_to_time_window`
  - `gefs.py`: sort section
  - `sflux_writer.py`: `write_all` day grouping
- Guards kept for IndexError safety, but mismatches are now visible

### 10. nco_bridge restart path assumes standard NCO layout — FIXED
- Added `RESTART_DIR` env var as explicit override (highest priority)
- Reordered priority: `RESTART_DIR` > `COMIN` > `COMOUT` parent (heuristic)
- Added `is_dir()` validation with warning when COMOUT parent doesn't exist
