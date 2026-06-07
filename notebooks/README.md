# CMIP7 cmor4 Example Notebooks

These notebooks demonstrate how to use `cmor4` to create CMIP7-style dataset files. They use the CMIP7 table checkout under `project_tables/cmip7-cmor-tables`.

The examples use table-backed metadata: variable attributes come from the
loaded CMIP7 variable table, coordinate attributes come from the coordinate
and grid tables, and hybrid-coordinate z-factor
attributes come from the formula terms table. The notebook dictionaries focus
on dataset and custom grid metadata. The examples use `project.variable(...)`,
`project.axis(...)`, `project.zfactor(...)`, and `project.dataset_info(...)`
to combine runtime values with table metadata. Fixed scalar coordinates such
as `height2m` are added during dataset creation when the selected variable
requires them.

## Setup

Create the notebook environment from the repository root:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .[notebook]
```

Start JupyterLab:

```bash
jupyter lab notebooks
```

The notebooks write generated NetCDF files under `notebooks/output/`, which is ignored by Git.

## Notebooks

1. [Basic Ocean Surface Temperature](01_basic_ocean_surface_temperature.ipynb): writes monthly sea surface temperature (`tos_tavg-u-hxy-sea`) on a latitude-longitude grid.
2. [Atmospheric Surface Air Temperature](02_atmos_surface_air_temperature.ipynb): writes 2 m air temperature (`tas_tavg-h2m-hxy-u`) and demonstrates the table-provided fixed-height coordinate.
3. [Atmospheric Temperature On Pressure Levels](03_pressure_level_air_temperature.ipynb): writes air temperature (`ta_tavg-p19-hxy-air`) on the CMIP7 19 pressure levels.
4. [Hybrid Sigma Humidity Tendency With Z-Factors](04_hybrid_sigma_humidity_tendency.ipynb): writes humidity tendency (`tnhusscpbl_tavg-al-hxy-u`) on a hybrid sigma pressure coordinate with cmor4 z-factor variables and surface pressure.
5. [Ocean Heat Transport By Basin](05_ocean_heat_transport_by_basin.ipynb): writes northward ocean heat transport (`htovgyre_tavg-u-hyb-sea`) using latitude and ocean basin coordinates.
6. [Land-Use Fraction](06_land_use_fraction.ipynb): writes land-use fraction (`fracLut_tpt-u-hxy-u`) using time-point, land-use, latitude, and longitude coordinates.
7. [Sea Ice Concentration On Projected Grids](07_non_lat_lon_sea_ice_concentration.ipynb): writes sea-ice area percentage (`siconc_tavg-u-hxy-u`) on projected Northern and Southern Hemisphere grids.
