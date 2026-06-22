"""CMOR4 reimplementations of the CMOR3 Python examples.

Each test writes a dataset with CMOR4 and verifies that the dimensions,
coordinate names, coordinate attributes, and variable attributes match those
produced by the corresponding CMOR3 example found at
https://github.com/PCMDI/cmor/tree/main/examples/python.

Only attributes that CMOR3 explicitly writes are asserted; CMOR4-specific
global attributes (e.g. ``host_collection``) are not checked.

All seven examples share the same dataset metadata unless noted:

    activity_id       : CMIP
    calendar          : 360_day
    experiment_id     : amip
    institution_id    : MOHC
    source_id         : DUMMY-MODEL
    grid_label        : g999
    nominal_resolution: 100 km
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
CMIP7_TABLE_ROOT = REPO_ROOT / "project_tables" / "cmip7-cmor-tables"
TABLES_DIR = CMIP7_TABLE_ROOT / "tables"
CV_PATH = CMIP7_TABLE_ROOT / "tables-cvs" / "cmor-cvs.json"


def _requires_tables(test):
    """Skip decorator applied when the CMIP7 submodule is not checked out."""
    import unittest

    if not TABLES_DIR.exists() or not CV_PATH.exists():
        return unittest.skip("CMIP7 tables submodule not initialised")(test)
    return test


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_DATASET = {
    "activity_id": "CMIP",
    "calendar": "360_day",
    "experiment_id": "amip",
    "forcing_index": "f1",
    "frequency": "mon",
    "grid_label": "g999",
    "host_collection": "CMIP7",
    "initialization_index": "i1",
    "institution_id": "MOHC",
    "license_id": "CC-BY-4.0",
    "nominal_resolution": "100 km",
    "physics_index": "p1",
    "realization_index": "r1",
    "region": "glb",
    "source_id": "DUMMY-MODEL",
}

# Coordinate values shared by examples 1–5 and 7
_LAT_VALS = np.array([10.0, 20.0, 30.0], dtype="d")
_LAT_BNDS = np.array([[5.0, 15.0], [15.0, 25.0], [25.0, 35.0]], dtype="d")
_LON_VALS = np.array([0.0, 90.0, 180.0, 270.0], dtype="d")
_LON_BNDS = np.array([[-45.0, 45.0], [45.0, 135.0], [135.0, 225.0], [225.0, 315.0]], dtype="d")
_TIME_VALS = np.array([15.5, 45.5], dtype="d")
_TIME_BNDS = np.array([[0.0, 31.0], [31.0, 60.0]], dtype="d")
_TIME_UNITS = "days since 1979-01-01"


def _project(*table_names: str):
    """Return a ProjectTables instance for the given variable table names."""
    import cmor4

    return cmor4.ProjectTables.from_directory(
        CMIP7_TABLE_ROOT,
        cv_file="tables-cvs/cmor-cvs.json",
        variable_tables=[f"tables/{n}" for n in table_names],
        coordinate_table="tables/CMIP7_coordinate.json",
        formula_table="tables/CMIP7_formula_terms.json",
        grid_table="tables/CMIP7_grids.json",
    )


def _make_dataset(project, base: dict, outpath: str):
    """Return a DatasetInfo with an outpath merged in."""
    return project.dataset_info({**base, "outpath": outpath})


def _time_axis(project):
    return project.axis(
        "time",
        values=_TIME_VALS,
        bounds=_TIME_BNDS,
        units=_TIME_UNITS,
    )


def _lat_axis(project):
    return project.axis(
        "latitude",
        values=_LAT_VALS,
        bounds=_LAT_BNDS,
    )


def _lon_axis(project):
    return project.axis(
        "longitude",
        values=_LON_VALS,
        bounds=_LON_BNDS,
    )


# ---------------------------------------------------------------------------
# Example 1 — usual 2-D field (tos, ocean)
# ---------------------------------------------------------------------------


@_requires_tables
class TestExample01UsualField(unittest.TestCase):
    """Example 1: monthly SST on a regular lat/lon grid (CMIP7_ocean)."""

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        project = _project("CMIP7_ocean.json")
        dataset = _make_dataset(project, _BASE_DATASET, self.tmp)
        variable = project.variable(
            "tos_tavg-u-hxy-sea", table_id="ocean", missing_value=np.float32(1.0e20)
        )
        axes = [_time_axis(project), _lat_axis(project), _lon_axis(project)]
        data = np.array(
            [
                254.0895, 258.4085, 1.0e20, 258.7101,
                258.6680, 258.2990, 1.0e20, 255.0432,
                253.7254, 251.2460, 1.0e20, 255.4808,
                254.0995, 258.5085, 1.0e20, 258.8101,
                258.8680, 258.4990, 1.0e20, 255.2432,
                254.0254, 251.5460, 1.0e20, 255.7808,
            ],
            dtype="f4",
        ).reshape(2, 3, 4)
        ds = cmor4.create_dataset(dataset, variable, axes, data)
        path = cmor4.write_netcdf(ds, dataset, variable)
        self.ds = xr.open_dataset(path, decode_times=False)

    def tearDown(self):
        self.ds.close()

    # --- dimensions ---

    def test_variable_dims(self):
        self.assertEqual(self.ds["tos"].dims, ("time", "lat", "lon"))

    def test_variable_shape(self):
        self.assertEqual(self.ds["tos"].shape, (2, 3, 4))

    # --- coordinate names ---

    def test_coords_present(self):
        for name in ("time", "lat", "lon"):
            self.assertIn(name, self.ds.coords)

    def test_bounds_present(self):
        for name in ("lat_bnds", "lon_bnds", "time_bnds"):
            self.assertIn(name, self.ds)

    # --- coordinate attributes ---

    def test_lat_attrs(self):
        a = self.ds["lat"].attrs
        self.assertEqual(a["standard_name"], "latitude")
        self.assertEqual(a["long_name"], "Latitude")
        self.assertEqual(a["units"], "degrees_north")
        self.assertEqual(a["axis"], "Y")
        self.assertEqual(a["bounds"], "lat_bnds")

    def test_lon_attrs(self):
        a = self.ds["lon"].attrs
        self.assertEqual(a["standard_name"], "longitude")
        self.assertEqual(a["long_name"], "Longitude")
        self.assertEqual(a["units"], "degrees_east")
        self.assertEqual(a["axis"], "X")
        self.assertEqual(a["bounds"], "lon_bnds")

    def test_time_attrs(self):
        a = self.ds["time"].attrs
        self.assertEqual(a["standard_name"], "time")
        self.assertEqual(a["long_name"], "Time Intervals")
        self.assertEqual(a["axis"], "T")
        self.assertEqual(a["bounds"], "time_bnds")

    # --- variable attributes ---

    def test_variable_standard_name(self):
        self.assertEqual(
            self.ds["tos"].attrs["standard_name"], "sea_surface_temperature"
        )

    def test_variable_long_name(self):
        self.assertEqual(self.ds["tos"].attrs["long_name"], "Sea Surface Temperature")

    def test_variable_units(self):
        self.assertEqual(self.ds["tos"].attrs["units"], "degC")

    def test_variable_cell_methods(self):
        self.assertEqual(
            self.ds["tos"].attrs["cell_methods"], "area: mean where sea time: mean"
        )

    # --- global attributes ---

    def test_global_variable_id(self):
        self.assertEqual(self.ds.attrs["variable_id"], "tos")

    def test_global_frequency(self):
        self.assertEqual(self.ds.attrs["frequency"], "mon")

    def test_global_realm(self):
        self.assertEqual(self.ds.attrs["realm"], "ocean")

    def test_global_mip_era(self):
        self.assertEqual(self.ds.attrs["mip_era"], "CMIP7")

    def test_global_experiment_id(self):
        self.assertEqual(self.ds.attrs["experiment_id"], "amip")

    def test_global_institution_id(self):
        self.assertEqual(self.ds.attrs["institution_id"], "MOHC")

    def test_global_source_id(self):
        self.assertEqual(self.ds.attrs["source_id"], "DUMMY-MODEL")

    def test_global_variant_label(self):
        self.assertEqual(self.ds.attrs["variant_label"], "r1i1p1f1")

    def test_global_title(self):
        self.assertIn("DUMMY-MODEL", self.ds.attrs.get("title", ""))


# ---------------------------------------------------------------------------
# Example 2 — pressure levels (ta, atmos)
# ---------------------------------------------------------------------------


@_requires_tables
class TestExample02PressureLevels(unittest.TestCase):
    """Example 2: air temperature on 19 pressure levels (CMIP7_atmos)."""

    _PLEV19 = np.array(
        [
            100000.0, 92500.0, 85000.0, 70000.0, 60000.0,
            50000.0, 40000.0, 30000.0, 25000.0, 20000.0,
            15000.0, 10000.0, 7000.0, 5000.0, 3000.0,
            2000.0, 1000.0, 500.0, 100.0,
        ],
        dtype="d",
    )

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        project = _project("CMIP7_atmos.json")
        dataset = _make_dataset(project, _BASE_DATASET, self.tmp)
        variable = project.variable(
            "ta_tavg-p19-hxy-air", table_id="atmos", missing_value=np.float32(1.0e20)
        )
        plev_axis = project.axis("plev19", values=self._PLEV19)
        axes = [_time_axis(project), plev_axis, _lat_axis(project), _lon_axis(project)]
        data = np.linspace(250.0, 275.0, 2 * 19 * 3 * 4, dtype="f4").reshape(2, 19, 3, 4)
        data[0, 0, 0, 0] = np.float32(1.0e20)
        ds = cmor4.create_dataset(dataset, variable, axes, data)
        path = cmor4.write_netcdf(ds, dataset, variable)
        self.ds = xr.open_dataset(path, decode_times=False)

    def tearDown(self):
        self.ds.close()

    def test_variable_dims(self):
        self.assertEqual(self.ds["ta"].dims, ("time", "plev", "lat", "lon"))

    def test_variable_shape(self):
        self.assertEqual(self.ds["ta"].shape, (2, 19, 3, 4))

    def test_plev_coord_present(self):
        self.assertIn("plev", self.ds.coords)

    def test_plev_attrs(self):
        a = self.ds["plev"].attrs
        self.assertEqual(a["standard_name"], "air_pressure")
        self.assertEqual(a["long_name"], "Pressure Levels (19)")
        self.assertEqual(a["units"], "Pa")
        self.assertEqual(a["axis"], "Z")
        self.assertEqual(a["positive"], "down")

    def test_no_plev_bounds(self):
        # plev19 must_have_bounds = no
        self.assertNotIn("bounds", self.ds["plev"].attrs)
        self.assertNotIn("plev_bnds", self.ds)

    def test_variable_standard_name(self):
        self.assertEqual(self.ds["ta"].attrs["standard_name"], "air_temperature")

    def test_variable_long_name(self):
        self.assertEqual(self.ds["ta"].attrs["long_name"], "Air Temperature")

    def test_variable_units(self):
        self.assertEqual(self.ds["ta"].attrs["units"], "K")

    def test_variable_cell_methods(self):
        self.assertEqual(
            self.ds["ta"].attrs["cell_methods"], "area: time: mean where air"
        )

    def test_global_variable_id(self):
        self.assertEqual(self.ds.attrs["variable_id"], "ta")

    def test_global_realm(self):
        self.assertEqual(self.ds.attrs["realm"], "atmos")

    def test_global_mip_era(self):
        self.assertEqual(self.ds.attrs["mip_era"], "CMIP7")

    def test_global_variant_label(self):
        self.assertEqual(self.ds.attrs["variant_label"], "r1i1p1f1")


# ---------------------------------------------------------------------------
# Example 3 — scalar dimension (tas, height2m)
# ---------------------------------------------------------------------------


@_requires_tables
class TestExample03ScalarDimension(unittest.TestCase):
    """Example 3: near-surface air temperature with 2 m height scalar coord."""

    _DATA = np.array(
        [
            254.0895, 258.4085, 250.5549, 258.7101,
            258.6680, 258.2990, 252.1237, 255.0432,
            253.7254, 251.2460, 254.3168, 255.4808,
            259.7908, 252.2754, 257.1892, 253.3132,
            253.8823, 253.4698, 253.5381, 254.9730,
            256.1002, 251.8168, 259.3698, 250.2994,
        ],
        dtype="f4",
    ).reshape(2, 3, 4)

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        base = {**_BASE_DATASET, "forcing_index": "f2", "realization_index": "r9"}
        project = _project("CMIP7_atmos.json")
        dataset = _make_dataset(project, base, self.tmp)
        variable = project.variable(
            "tas_tavg-h2m-hxy-u", table_id="atmos", missing_value=np.float32(1.0e20)
        )
        axes = [_time_axis(project), _lat_axis(project), _lon_axis(project)]
        ds = cmor4.create_dataset(dataset, variable, axes, self._DATA)
        path = cmor4.write_netcdf(ds, dataset, variable)
        self.ds = xr.open_dataset(path, decode_times=False)

    def tearDown(self):
        self.ds.close()

    def test_variable_dims(self):
        # height2m is scalar — not a dimension on the variable
        self.assertEqual(self.ds["tas"].dims, ("time", "lat", "lon"))

    def test_height_is_scalar_coord(self):
        self.assertIn("height", self.ds.coords)
        self.assertEqual(self.ds["height"].shape, ())

    def test_height_attrs(self):
        a = self.ds["height"].attrs
        self.assertEqual(a["standard_name"], "height")
        self.assertEqual(a["long_name"], "height")
        self.assertEqual(a["units"], "m")
        self.assertEqual(a["axis"], "Z")
        self.assertEqual(a["positive"], "up")

    def test_height_value(self):
        self.assertAlmostEqual(float(self.ds["height"].values), 2.0)

    def test_variable_standard_name(self):
        self.assertEqual(self.ds["tas"].attrs["standard_name"], "air_temperature")

    def test_variable_long_name(self):
        self.assertEqual(
            self.ds["tas"].attrs["long_name"], "Near-Surface Air Temperature"
        )

    def test_variable_units(self):
        self.assertEqual(self.ds["tas"].attrs["units"], "K")

    def test_variable_cell_methods(self):
        self.assertEqual(self.ds["tas"].attrs["cell_methods"], "area: time: mean")

    def test_global_variable_id(self):
        self.assertEqual(self.ds.attrs["variable_id"], "tas")

    def test_global_realm(self):
        self.assertEqual(self.ds.attrs["realm"], "atmos")

    def test_global_variant_label(self):
        self.assertEqual(self.ds.attrs["variant_label"], "r9i1p1f2")


# ---------------------------------------------------------------------------
# Example 4 — auxiliary coordinates / basin (htovgyre, ocean)
# ---------------------------------------------------------------------------


@_requires_tables
class TestExample04AuxiliaryCoordinates(unittest.TestCase):
    """Example 4: ocean heat transport with basin auxiliary coordinate."""

    _BASINS = np.array(
        ["atlantic_arctic_ocean", "indian_pacific_ocean", "global_ocean"],
        dtype="U21",
    )
    _DATA = np.array(
        [
            -80.0, -84.0, -88.0, -100.0, -104.0, -76.0,
            -120.0, -92.0, -96.0, -79.0, -83.0, -87.0,
            -99.0, -103.0, -75.0, -107.0, -111.0, -115.0,
        ],
        dtype="f4",
    ).reshape(2, 3, 3)

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        project = _project("CMIP7_ocean.json")
        dataset = _make_dataset(project, _BASE_DATASET, self.tmp)
        variable = project.variable(
            "htovgyre_tavg-u-hyb-sea",
            table_id="ocean",
            missing_value=np.float32(1.0e20),
        )
        basin_axis = project.axis("basin", values=self._BASINS)
        axes = [_time_axis(project), basin_axis, _lat_axis(project)]
        ds = cmor4.create_dataset(dataset, variable, axes, self._DATA)
        path = cmor4.write_netcdf(ds, dataset, variable)
        self.ds = xr.open_dataset(path, decode_times=False)

    def tearDown(self):
        self.ds.close()

    def test_variable_dims(self):
        self.assertEqual(self.ds["htovgyre"].dims, ("time", "basin", "lat"))

    def test_basin_coord_present(self):
        self.assertIn("basin", self.ds.coords)
        self.assertEqual(self.ds["basin"].dims, ("basin",))

    def test_basin_coord_attrs(self):
        a = self.ds["basin"].attrs
        self.assertEqual(a.get("standard_name"), "region")
        self.assertEqual(a.get("long_name"), "Ocean Basin")

    def test_basin_coord_values(self):
        vals = list(str(v) for v in self.ds["basin"].values)
        self.assertEqual(vals, list(self._BASINS))

    def test_variable_standard_name(self):
        self.assertEqual(
            self.ds["htovgyre"].attrs["standard_name"],
            "northward_ocean_heat_transport_due_to_gyre",
        )

    def test_variable_long_name(self):
        self.assertEqual(
            self.ds["htovgyre"].attrs["long_name"],
            "Northward Ocean Heat Transport Due to Gyre",
        )

    def test_variable_units(self):
        self.assertEqual(self.ds["htovgyre"].attrs["units"], "W")

    def test_variable_cell_methods(self):
        self.assertEqual(
            self.ds["htovgyre"].attrs["cell_methods"],
            "depth: longitude: sum where sea (along a zig-zag grid path spanning a basin)  time: mean",
        )

    def test_global_variable_id(self):
        self.assertEqual(self.ds.attrs["variable_id"], "htovgyre")

    def test_global_realm(self):
        self.assertEqual(self.ds.attrs["realm"], "ocean")

    def test_global_variant_label(self):
        self.assertEqual(self.ds.attrs["variant_label"], "r1i1p1f1")

    def test_no_time_dim_on_lat(self):
        self.assertIn("lat", self.ds.coords)
        self.assertIn("lat", self.ds["htovgyre"].dims)


# ---------------------------------------------------------------------------
# Example 5 — model levels / hybrid sigma (cl, atmos)
# ---------------------------------------------------------------------------


@_requires_tables
class TestExample05ModelLevels(unittest.TestCase):
    """Example 5: cloud fraction on hybrid-sigma pressure levels."""

    _LEV_VALS = np.array([0.92, 0.72, 0.50, 0.30, 0.10], dtype="d")
    _LEV_BNDS = np.array([[1.00, 0.83],[0.83, 0.61],[0.61, 0.40],[0.40, 0.20],[0.20, 0.00]], dtype="d")
    _A_VALS = np.array([0.12, 0.22, 0.30, 0.20, 0.10], dtype="d")
    _A_BNDS = np.array([[0.06, 0.18],[0.18, 0.26],[0.26, 0.25],[0.25, 0.15],[0.15, 0.00]], dtype="d")
    _B_VALS = np.array([0.80, 0.50, 0.20, 0.10, 0.00], dtype="d")
    _B_BNDS = np.array([[0.94, 0.65],[0.65, 0.35],[0.35, 0.15],[0.15, 0.05],[0.05, 0.00]], dtype="d")
    _PS = np.array(
        [
            97000.0, 97400.0, 97800.0, 98200.0, 98600.0, 99000.0,
            99400.0, 99800.0, 100200.0, 100600.0, 101000.0, 101400.0,
            97100.0, 97500.0, 97900.0, 98300.0, 98700.0, 99100.0,
            99500.0, 99900.0, 100300.0, 100700.0, 101100.0, 101500.0,
        ],
        dtype="f4",
    ).reshape(2, 3, 4)

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        project = _project("CMIP7_atmos.json")
        dataset = _make_dataset(project, _BASE_DATASET, self.tmp)
        variable = project.variable(
            "cl_tavg-al-hxy-u", table_id="atmos", missing_value=np.float32(1.0e20)
        )
        time_axis = _time_axis(project)
        lat_axis = _lat_axis(project)
        lon_axis = _lon_axis(project)
        lev_axis = project.axis(
            "standard_hybrid_sigma",
            values=self._LEV_VALS,
            bounds=self._LEV_BNDS,
        )
        a_zfactor = project.zfactor("a", values=self._A_VALS, bounds=self._A_BNDS)
        b_zfactor = project.zfactor("b", values=self._B_VALS, bounds=self._B_BNDS)
        p0_zfactor = project.zfactor("p0", values=100000.0)
        ps_zfactor = project.zfactor("ps", values=self._PS)
        data = np.array(
            [
                72.8, 73.2, 73.6, 74.0, 71.6, 72.0, 72.4, 72.4,
                70.4, 70.8, 70.8, 71.2, 67.6, 69.2, 69.6, 70.0,
                66.0, 66.4, 66.8, 67.2, 64.8, 65.2, 65.6, 66.0,
                63.6, 64.0, 64.4, 64.4, 60.8, 61.2, 62.8, 63.2,
                59.6, 59.6, 60.0, 60.4, 58.0, 58.4, 58.8, 59.2,
                56.8, 57.2, 57.6, 58.0, 54.0, 54.4, 54.8, 56.4,
                52.8, 53.2, 53.2, 53.6, 51.6, 51.6, 52.0, 52.4,
                50.0, 50.4, 50.8, 51.2, 72.9, 73.3, 73.7, 74.1,
                71.7, 72.1, 72.5, 72.5, 70.5, 70.9, 70.9, 71.3,
                67.7, 69.3, 69.7, 70.1, 66.1, 66.5, 66.9, 67.3,
                64.9, 65.3, 65.7, 66.1, 63.7, 64.1, 64.5, 64.5,
                60.9, 61.3, 62.9, 63.3, 59.7, 59.7, 60.1, 60.5,
                58.1, 58.5, 58.9, 59.3, 56.9, 57.3, 57.7, 58.1,
                54.1, 54.5, 54.9, 56.5, 52.9, 53.3, 53.3, 53.7,
                51.7, 51.7, 52.1, 52.5, 50.1, 50.5, 50.9, 51.3,
            ],
            dtype="f4",
        ).reshape(2, 5, 3, 4)
        ds = cmor4.create_dataset(
            dataset,
            variable,
            [time_axis, lev_axis, lat_axis, lon_axis],
            data,
            zfactors=[a_zfactor, b_zfactor, p0_zfactor, ps_zfactor],
        )
        path = cmor4.write_netcdf(ds, dataset, variable)
        self.ds = xr.open_dataset(path, decode_times=False)

    def tearDown(self):
        self.ds.close()

    def test_variable_dims(self):
        self.assertEqual(self.ds["cl"].dims, ("time", "lev", "lat", "lon"))

    def test_lev_attrs(self):
        a = self.ds["lev"].attrs
        self.assertEqual(
            a["standard_name"], "atmosphere_hybrid_sigma_pressure_coordinate"
        )
        self.assertEqual(a["long_name"], "hybrid sigma pressure coordinate")
        self.assertEqual(a["units"], "1")
        self.assertEqual(a["axis"], "Z")
        self.assertEqual(a["positive"], "down")
        self.assertEqual(a["formula"], "p = a*p0 + b*ps")
        self.assertIn("formula_terms", a)
        self.assertEqual(a["bounds"], "lev_bnds")

    def test_formula_terms_on_lev(self):
        ft = self.ds["lev"].attrs["formula_terms"]
        self.assertIn("p0", ft)
        self.assertIn("ps", ft)
        self.assertIn("a", ft)
        self.assertIn("b", ft)

    def test_zfactor_vars_present(self):
        for name in ("a", "b", "p0", "ps", "a_bnds", "b_bnds", "lev_bnds"):
            self.assertIn(name, self.ds)

    def test_a_attrs(self):
        self.assertEqual(
            self.ds["a"].attrs.get("long_name"),
            "vertical coordinate formula term: a",
        )

    def test_b_attrs(self):
        self.assertEqual(
            self.ds["b"].attrs.get("long_name"),
            "vertical coordinate formula term: b",
        )

    def test_p0_attrs(self):
        a = self.ds["p0"].attrs
        self.assertEqual(a["units"], "Pa")
        self.assertEqual(
            a["standard_name"],
            "reference_air_pressure_for_atmosphere_vertical_coordinate",
        )
        self.assertEqual(
            a["long_name"],
            "vertical coordinate formula term: reference pressure",
        )

    def test_ps_attrs(self):
        a = self.ds["ps"].attrs
        self.assertEqual(a["standard_name"], "air_pressure")
        self.assertEqual(a["long_name"], "Surface Air Pressure")
        self.assertEqual(a["units"], "Pa")

    def test_lev_bnds_formula_terms(self):
        a = self.ds["lev_bnds"].attrs
        self.assertIn("formula_terms", a)
        ft = a["formula_terms"]
        self.assertIn("a_bnds", ft)
        self.assertIn("b_bnds", ft)

    def test_variable_units(self):
        self.assertEqual(self.ds["cl"].attrs["units"], "%")

    def test_variable_cell_methods(self):
        self.assertEqual(self.ds["cl"].attrs["cell_methods"], "area: time: mean")

    def test_global_variable_id(self):
        self.assertEqual(self.ds.attrs["variable_id"], "cl")

    def test_global_realm(self):
        self.assertEqual(self.ds.attrs["realm"], "atmos")

    def test_global_variant_label(self):
        self.assertEqual(self.ds.attrs["variant_label"], "r1i1p1f1")


# ---------------------------------------------------------------------------
# Example 6 — complex curvilinear grid (hfls, atmos + grids)
# ---------------------------------------------------------------------------


@_requires_tables
class TestExample06ComplexGrid(unittest.TestCase):
    """Example 6: latent heat flux on a curvilinear grid with Lambert conformal
    conic mapping and lat/lon vertex arrays.

    The CMOR3 reference file for this example shows:
        latitude  : dims=(x, y), attrs=standard_name, long_name, units, bounds
        longitude : dims=(x, y), attrs=standard_name, long_name, units, bounds
        vertices_latitude  : dims=(x, y, vertices), attrs={units: degrees_north}
        vertices_longitude : dims=(x, y, vertices), attrs={units: degrees_east}
        lambert_conformal_conic: scalar, attrs=grid_mapping_name + params
        hfls: grid_mapping=lambert_conformal_conic, coordinates=latitude longitude
    """

    _Y_VALS = np.array([0.0, 10000.0, 20000.0], dtype="d")
    _X_VALS = np.array([0.0, 10000.0, 20000.0, 30000.0], dtype="d")

    _LAT = np.array(
        [[10.0, 8.0, 6.0, 4.0], [20.0, 18.0, 16.0, 14.0], [30.0, 28.0, 26.0, 24.0]],
        dtype="d",
    ).T  # (x=4, y=3) after CMOR3 axis reorder

    _LON = np.array(
        [
            [280.0, 290.0, 300.0, 310.0],
            [282.0, 292.0, 302.0, 312.0],
            [284.0, 294.0, 304.0, 314.0],
        ],
        dtype="d",
    ).T

    def _make_vertices(self, lat2d, lon2d):
        """Build (nj, ni, 4) vertex arrays matching the CMOR3 example."""
        nj, ni = 3, 4
        lat_v = np.empty((nj, ni, 4), dtype="d")
        lon_v = np.empty((nj, ni, 4), dtype="d")
        lat_src = lat2d.T  # back to (y=3, x=4)
        lon_src = lon2d.T
        for j in range(nj):
            for i in range(ni):
                lat_v[j, i] = [
                    lat_src[j, i] - 5.0,
                    lat_src[j, i] - 4.0,
                    lat_src[j, i] + 5.0,
                    lat_src[j, i] + 4.0,
                ]
                lon_v[j, i] = [
                    lon_src[j, i] - 5.0,
                    lon_src[j, i] + 5.0,
                    lon_src[j, i] + 5.0,
                    lon_src[j, i] - 5.0,
                ]
        return lat_v, lon_v

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        project = _project("CMIP7_atmos.json")
        dataset = _make_dataset(project, _BASE_DATASET, self.tmp)
        variable = project.variable(
            "hfls_tavg-u-hxy-u", table_id="atmos", missing_value=np.float32(1.0e20)
        )
        y_axis = project.axis(
            "y",
            values=self._Y_VALS,
            bounds=np.array([[-5000.0, 5000.0], [5000.0, 15000.0], [15000.0, 25000.0]], dtype="d"),
            units="m",
        )
        x_axis = project.axis(
            "x",
            values=self._X_VALS,
            bounds=np.array([[-5000.0, 5000.0], [5000.0, 15000.0], [15000.0, 25000.0], [25000.0, 35000.0]], dtype="d"),
            units="m",
        )
        lat_v, lon_v = self._make_vertices(self._LAT, self._LON)
        grid = project.grid(
            axes=[y_axis, x_axis],
            latitude=self._LAT.T,   # pass as (y, x) matching original example
            longitude=self._LON.T,
            latitude_vertices=lat_v,
            longitude_vertices=lon_v,
            mapping_name="lambert_conformal_conic",
            params={
                "standard_parallel1": [-20.0, ""],
                "longitude_of_central_meridian": [175.0, ""],
                "latitude_of_projection_origin": [13.0, ""],
                "false_easting": [8.0, ""],
                "false_northing": [0.0, ""],
                "standard_parallel2": [20.0, ""],
            },
        )
        time_axis = _time_axis(project)
        data = np.array(
            [
                80.0, 82.0, 84.0, 86.0, 88.0, 90.0, 92.0, 94.0,
                96.0, 98.0, 100.0, 102.0, 81.0, 83.0, 85.0, 87.0,
                89.0, 91.0, 93.0, 95.0, 97.0, 99.0, 101.0, 103.0,
            ],
            dtype="f4",
        ).reshape(2, 3, 4)
        ds = cmor4.create_dataset(dataset, variable, [time_axis], data, grid=grid)
        path = cmor4.write_netcdf(ds, dataset, variable)
        self.ds = xr.open_dataset(path, decode_times=False)

    def tearDown(self):
        self.ds.close()

    # --- dimensions and shape ---

    def test_variable_dims_include_x_y(self):
        dims = self.ds["hfls"].dims
        self.assertIn("x", dims)
        self.assertIn("y", dims)
        self.assertIn("time", dims)

    def test_grid_dim_sizes(self):
        self.assertEqual(self.ds.sizes["x"], 4)
        self.assertEqual(self.ds.sizes["y"], 3)
        self.assertEqual(self.ds.sizes["vertices"], 4)

    # --- x / y dimensional coordinates ---

    def test_x_coord_present(self):
        self.assertIn("x", self.ds.coords)

    def test_y_coord_present(self):
        self.assertIn("y", self.ds.coords)

    def test_x_attrs(self):
        a = self.ds["x"].attrs
        self.assertEqual(a["standard_name"], "projection_x_coordinate")
        self.assertEqual(a["long_name"], "x coordinate of projection")
        self.assertEqual(a["units"], "m")
        self.assertEqual(a["axis"], "X")

    def test_y_attrs(self):
        a = self.ds["y"].attrs
        self.assertEqual(a["standard_name"], "projection_y_coordinate")
        self.assertEqual(a["long_name"], "y coordinate of projection")
        self.assertEqual(a["units"], "m")
        self.assertEqual(a["axis"], "Y")

    def test_x_bounds_present(self):
        self.assertEqual(self.ds["x"].attrs.get("bounds"), "x_bnds")
        self.assertIn("x_bnds", self.ds)

    def test_y_bounds_present(self):
        self.assertEqual(self.ds["y"].attrs.get("bounds"), "y_bnds")
        self.assertIn("y_bnds", self.ds)

    # --- auxiliary latitude / longitude ---

    def test_latitude_coord_present(self):
        self.assertIn("latitude", self.ds.coords)

    def test_longitude_coord_present(self):
        self.assertIn("longitude", self.ds.coords)

    def test_latitude_dims(self):
        dims = set(self.ds["latitude"].dims)
        self.assertEqual(dims, {"x", "y"})

    def test_longitude_dims(self):
        dims = set(self.ds["longitude"].dims)
        self.assertEqual(dims, {"x", "y"})

    def test_latitude_attrs(self):
        a = self.ds["latitude"].attrs
        self.assertEqual(a["standard_name"], "latitude")
        self.assertEqual(a["long_name"], "latitude")
        self.assertEqual(a["units"], "degrees_north")
        self.assertEqual(a.get("bounds"), "vertices_latitude")

    def test_longitude_attrs(self):
        a = self.ds["longitude"].attrs
        self.assertEqual(a["standard_name"], "longitude")
        self.assertEqual(a["long_name"], "longitude")
        self.assertEqual(a["units"], "degrees_east")
        self.assertEqual(a.get("bounds"), "vertices_longitude")

    # --- vertex arrays ---

    def test_vertices_latitude_present(self):
        self.assertIn("vertices_latitude", self.ds)

    def test_vertices_longitude_present(self):
        self.assertIn("vertices_longitude", self.ds)

    def test_vertices_latitude_dims(self):
        dims = set(self.ds["vertices_latitude"].dims)
        self.assertEqual(dims, {"x", "y", "vertices"})

    def test_vertices_latitude_shape(self):
        s = self.ds["vertices_latitude"].shape
        self.assertEqual(sorted(s), [3, 4, 4])

    # --- grid mapping ---

    def test_grid_mapping_name_attr(self):
        gm_var = self.ds["hfls"].attrs.get("grid_mapping")
        self.assertIsNotNone(gm_var, "hfls missing grid_mapping attr")
        self.assertEqual(
            self.ds[gm_var].attrs.get("grid_mapping_name"),
            "lambert_conformal_conic",
        )

    def test_variable_references_grid_mapping(self):
        self.assertIn("grid_mapping", self.ds["hfls"].attrs)

    def test_lat_lon_in_dataset_coords(self):
        self.assertIn("latitude", self.ds.coords)
        self.assertIn("longitude", self.ds.coords)

    # --- variable attrs ---

    def test_variable_standard_name(self):
        self.assertEqual(
            self.ds["hfls"].attrs["standard_name"],
            "surface_upward_latent_heat_flux",
        )

    def test_variable_long_name(self):
        self.assertEqual(
            self.ds["hfls"].attrs["long_name"],
            "Surface Upward Latent Heat Flux",
        )

    def test_variable_units(self):
        self.assertEqual(self.ds["hfls"].attrs["units"], "W m-2")

    def test_variable_cell_methods(self):
        self.assertEqual(self.ds["hfls"].attrs["cell_methods"], "area: time: mean")

    def test_global_variable_id(self):
        self.assertEqual(self.ds.attrs["variable_id"], "hfls")

    def test_global_realm(self):
        self.assertEqual(self.ds.attrs["realm"], "atmos")

    def test_global_variant_label(self):
        self.assertEqual(self.ds.attrs["variant_label"], "r1i1p1f1")


# ---------------------------------------------------------------------------
# Example 7 — fixed (time-independent) field (rootd, land)
# ---------------------------------------------------------------------------


@_requires_tables
class TestExample07FixedField(unittest.TestCase):
    """Example 7: maximum root depth as a fixed (fx) land field."""

    _DATA = np.array(
        [[0.50, 0.45, 1.0e20, 0.55], [0.60, 0.60, 1.0e20, 0.55], [1.0e20, 0.45, 0.50, 0.50]],
        dtype="f4",
    )

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        base = {**_BASE_DATASET, "frequency": "fx"}
        project = _project("CMIP7_land.json")
        dataset = _make_dataset(project, base, self.tmp)
        variable = project.variable(
            "rootd_ti-u-hxy-lnd", table_id="land", missing_value=np.float32(1.0e20)
        )
        axes = [_lat_axis(project), _lon_axis(project)]
        ds = cmor4.create_dataset(dataset, variable, axes, self._DATA)
        path = cmor4.write_netcdf(ds, dataset, variable)
        self.ds = xr.open_dataset(path, decode_times=False)

    def tearDown(self):
        self.ds.close()

    def test_variable_dims(self):
        # Fixed field: no time dimension
        self.assertEqual(self.ds["rootd"].dims, ("lat", "lon"))

    def test_variable_shape(self):
        self.assertEqual(self.ds["rootd"].shape, (3, 4))

    def test_no_time_dim(self):
        self.assertNotIn("time", self.ds.dims)

    def test_variable_standard_name(self):
        self.assertEqual(self.ds["rootd"].attrs["standard_name"], "root_depth")

    def test_variable_long_name(self):
        self.assertEqual(self.ds["rootd"].attrs["long_name"], "Maximum Root Depth")

    def test_variable_units(self):
        self.assertEqual(self.ds["rootd"].attrs["units"], "m")

    def test_variable_cell_methods(self):
        self.assertEqual(
            self.ds["rootd"].attrs["cell_methods"], "area: mean where land"
        )

    def test_lat_attrs(self):
        a = self.ds["lat"].attrs
        self.assertEqual(a["standard_name"], "latitude")
        self.assertEqual(a["long_name"], "Latitude")
        self.assertEqual(a["units"], "degrees_north")
        self.assertEqual(a["axis"], "Y")
        self.assertEqual(a["bounds"], "lat_bnds")

    def test_lon_attrs(self):
        a = self.ds["lon"].attrs
        self.assertEqual(a["standard_name"], "longitude")
        self.assertEqual(a["long_name"], "Longitude")
        self.assertEqual(a["units"], "degrees_east")
        self.assertEqual(a["axis"], "X")
        self.assertEqual(a["bounds"], "lon_bnds")

    def test_global_variable_id(self):
        self.assertEqual(self.ds.attrs["variable_id"], "rootd")

    def test_global_frequency(self):
        self.assertEqual(self.ds.attrs["frequency"], "fx")

    def test_global_realm(self):
        self.assertEqual(self.ds.attrs["realm"], "land")

    def test_global_mip_era(self):
        self.assertEqual(self.ds.attrs["mip_era"], "CMIP7")

    def test_global_variant_label(self):
        self.assertEqual(self.ds.attrs["variant_label"], "r1i1p1f1")


if __name__ == "__main__":
    unittest.main()
