"""Tests for Grid.axes — the axis-based grid design.

Covers:
* Grid construction from Axis objects (axis-based path)
* Grid construction from dimension strings (name-based path, backward compat)
* isgridaxis flag set on axes stored inside Grid.axes
* Caller's original Axis instance is not mutated
* dimensions derived automatically from axes
* variable_dimensions() with axes, with strings, fallback
* Grid.axes are appended directly in create_dataset (no _grid_axes wrapper)
* add_grid_coords() writes lat/lon/vertices directly as dataset coords
* Grid.to_dataset_coords() produces correct coords/data_vars/aux names
* must_call_cmor_grid: isgridaxis axis in user axes without Grid → error
* ProjectTables.grid(axes=...) validates axes against grid table
* _validate_grid_dimensions() both paths
* Full round-trip: create_dataset with axis-based Grid
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from cmor4 import Axis, Grid, Variable
from cmor4.exceptions import TableValidationError
from cmor4.utils.construction import add_grid_coords

# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------


def _axis(name: str, n: int, **kw: Any) -> Axis:
    return Axis(name=name, values=np.arange(n, dtype="f4"), **kw)


def _latlon(nj: int, ni: int) -> tuple[np.ndarray, np.ndarray]:
    lons = np.linspace(-180, 180, ni)
    lats = np.linspace(-90, 90, nj)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lat2d, lon2d


# ---------------------------------------------------------------------------
# 1. Grid construction — axis-based path
# ---------------------------------------------------------------------------


class TestGridAxesConstruction(unittest.TestCase):
    def _make_axes(self) -> tuple[Axis, Axis]:
        return _axis("j_index", 4), _axis("i_index", 8)

    def test_axes_stored(self) -> None:
        j, i = self._make_axes()
        grid = Grid(axes=[j, i])
        self.assertEqual(len(grid.axes), 2)

    def test_dimensions_derived_from_axes(self) -> None:
        j = Axis(name="j_index", out_name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i_index", out_name="i", values=np.arange(8, dtype="f4"))
        grid = Grid(axes=[j, i])
        # dimensions should be derived from out_name
        self.assertEqual(grid.dimensions, ("j", "i"))

    def test_dimensions_derived_falls_back_to_name(self) -> None:
        j, i = self._make_axes()  # no out_name
        grid = Grid(axes=[j, i])
        self.assertEqual(grid.dimensions, ("j_index", "i_index"))

    def test_explicit_dimensions_not_overridden(self) -> None:
        j, i = self._make_axes()
        grid = Grid(axes=[j, i], dimensions=("rows", "cols"))
        self.assertEqual(grid.dimensions, ("rows", "cols"))

    def test_empty_axes_does_not_set_dimensions(self) -> None:
        grid = Grid(axes=[])
        self.assertIsNone(grid.dimensions)


# ---------------------------------------------------------------------------
# 2. isgridaxis flag
# ---------------------------------------------------------------------------


class TestIsgridaxis(unittest.TestCase):
    def test_isgridaxis_set_on_stored_axes(self) -> None:
        j, i = _axis("j", 4), _axis("i", 8)
        grid = Grid(axes=[j, i])
        for axis in grid.axes:
            self.assertTrue(axis.isgridaxis, f"{axis.name}.isgridaxis should be True")

    def test_caller_axis_not_mutated(self) -> None:
        """Original Axis instances must remain unchanged (immutable)."""
        j = _axis("j", 4)
        i = _axis("i", 8)
        self.assertFalse(j.isgridaxis)
        self.assertFalse(i.isgridaxis)
        Grid(axes=[j, i])
        # After Grid construction the caller's references must be unaffected.
        self.assertFalse(j.isgridaxis)
        self.assertFalse(i.isgridaxis)

    def test_already_flagged_axes_not_duplicated(self) -> None:
        flagged = Axis(name="x", values=np.arange(3, dtype="f4"), isgridaxis=True)
        grid = Grid(axes=[flagged])
        self.assertTrue(grid.axes[0].isgridaxis)
        # Should be the same object (no unnecessary copy)
        self.assertIs(grid.axes[0], flagged)

    def test_isgridaxis_default_false(self) -> None:
        axis = _axis("lat", 10)
        self.assertFalse(axis.isgridaxis)


# ---------------------------------------------------------------------------
# 3. variable_dimensions()
# ---------------------------------------------------------------------------


class TestVariableDimensions(unittest.TestCase):
    def _var(self, dims: tuple[str, ...]) -> Variable:
        return Variable(name="tas", units="K", dimensions=dims)

    def test_axes_path_prepends_time(self) -> None:
        j = Axis(name="j", out_name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", out_name="i", values=np.arange(8, dtype="f4"))
        grid = Grid(axes=[j, i])
        var = self._var(("time", "j", "i"))
        self.assertEqual(grid.variable_dimensions(var), ("time", "j", "i"))

    def test_axes_path_no_time_in_var(self) -> None:
        j = Axis(name="j", out_name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", out_name="i", values=np.arange(8, dtype="f4"))
        grid = Grid(axes=[j, i])
        var = self._var(("j", "i"))
        self.assertEqual(grid.variable_dimensions(var), ("j", "i"))

    def test_string_dimensions_path(self) -> None:
        grid = Grid(dimensions=("y", "x"))
        var = self._var(("time", "y", "x"))
        self.assertEqual(grid.variable_dimensions(var), ("time", "y", "x"))

    def test_fallback_to_variable_dimensions(self) -> None:
        grid = Grid()
        var = self._var(("time", "lat", "lon"))
        self.assertEqual(grid.variable_dimensions(var), ("time", "lat", "lon"))

    def test_no_dimensions_no_variable_dimensions(self) -> None:
        grid = Grid()
        var = Variable(name="tas", units="K")
        self.assertIsNone(grid.variable_dimensions(var))


# ---------------------------------------------------------------------------
# 4. add_grid_coords() in utils.construction
# ---------------------------------------------------------------------------
# Note: there is no _grid_axes() wrapper. create_dataset() directly appends
# grid.axes to the axis list (or nothing when grid is None). The round-trip
# tests in section 9 exercise that path end-to-end.


class TestAddGridCoords(unittest.TestCase):
    """add_grid_coords() writes lat/lon/vertices directly as dataset entries."""

    def _run(self, grid: Grid, spatial_dims: list[str]) -> tuple[dict, dict, list]:
        coords: dict = {}
        data_vars: dict = {}
        aux_names: list = []
        add_grid_coords(grid, tuple(spatial_dims), None, coords, data_vars, aux_names)
        return coords, data_vars, aux_names

    def test_lat_lon_written_as_coords(self) -> None:
        lat2d, lon2d = _latlon(4, 8)
        grid = Grid(dimensions=("j", "i"), latitude=lat2d, longitude=lon2d)
        coords, data_vars, aux_names = self._run(grid, ["j", "i"])
        self.assertIn("latitude", coords)
        self.assertIn("longitude", coords)
        self.assertEqual(coords["latitude"][0], ("j", "i"))
        self.assertEqual(coords["longitude"][0], ("j", "i"))

    def test_lat_lon_in_auxiliary_names(self) -> None:
        lat2d, lon2d = _latlon(4, 8)
        grid = Grid(dimensions=("j", "i"), latitude=lat2d, longitude=lon2d)
        _, _, aux_names = self._run(grid, ["j", "i"])
        self.assertIn("latitude", aux_names)
        self.assertIn("longitude", aux_names)

    def test_vertices_written_as_data_vars(self) -> None:
        nj, ni, nv = 4, 8, 4
        lat2d, lon2d = _latlon(nj, ni)
        blat = np.zeros((nj, ni, nv))
        blon = np.zeros((nj, ni, nv))
        grid = Grid(
            dimensions=("j", "i"),
            latitude=lat2d,
            longitude=lon2d,
            latitude_vertices=blat,
            longitude_vertices=blon,
        )
        coords, data_vars, _ = self._run(grid, ["j", "i"])
        self.assertIn("vertices_latitude", data_vars)
        self.assertIn("vertices_longitude", data_vars)
        self.assertEqual(data_vars["vertices_latitude"][0], ("j", "i", "vertices"))

    def test_bounds_attr_set_on_lat_lon(self) -> None:
        nj, ni, nv = 4, 8, 4
        lat2d, lon2d = _latlon(nj, ni)
        blat = np.zeros((nj, ni, nv))
        blon = np.zeros((nj, ni, nv))
        grid = Grid(
            dimensions=("j", "i"),
            latitude=lat2d,
            longitude=lon2d,
            latitude_vertices=blat,
            longitude_vertices=blon,
        )
        coords, _, _ = self._run(grid, ["j", "i"])
        self.assertEqual(coords["latitude"][2].get("bounds"), "vertices_latitude")
        self.assertEqual(coords["longitude"][2].get("bounds"), "vertices_longitude")

    def test_no_lat_lon_produces_no_coords(self) -> None:
        grid = Grid(dimensions=("j", "i"))
        coords, data_vars, aux_names = self._run(grid, ["j", "i"])
        self.assertEqual(coords, {})
        self.assertEqual(data_vars, {})
        self.assertEqual(aux_names, [])

    def test_spatial_dims_from_grid_dimensions(self) -> None:
        lat2d, lon2d = _latlon(3, 5)
        grid = Grid(dimensions=("row", "col"), latitude=lat2d, longitude=lon2d)
        coords, _, _ = self._run(grid, [])
        self.assertEqual(coords["latitude"][0], ("row", "col"))

    def test_spatial_dims_fallback_to_variable_dims(self) -> None:
        lat2d, lon2d = _latlon(3, 5)
        grid = Grid(latitude=lat2d, longitude=lon2d)
        coords: dict = {}
        data_vars: dict = {}
        aux_names: list = []
        add_grid_coords(grid, ("time", "ny", "nx"), None, coords, data_vars, aux_names)
        self.assertEqual(coords["latitude"][0], ("ny", "nx"))


class TestGridToDatasetCoords(unittest.TestCase):
    """Grid.to_dataset_coords() unit tests."""

    def test_lat_lon_only(self) -> None:
        lat2d, lon2d = _latlon(4, 8)
        grid = Grid(latitude=lat2d, longitude=lon2d)
        coords, data_vars, aux_names = grid.to_dataset_coords(["j", "i"])
        self.assertIn("latitude", coords)
        self.assertIn("longitude", coords)
        self.assertEqual(coords["latitude"][0], ("j", "i"))
        self.assertNotIn("vertices_latitude", data_vars)
        self.assertEqual(aux_names, ["latitude", "longitude"])

    def test_default_cf_attrs_no_table(self) -> None:
        lat2d, lon2d = _latlon(2, 3)
        grid = Grid(latitude=lat2d, longitude=lon2d)
        coords, _, _ = grid.to_dataset_coords(["j", "i"])
        lat_attrs = coords["latitude"][2]
        self.assertEqual(lat_attrs["standard_name"], "latitude")
        self.assertEqual(lat_attrs["units"], "degrees_north")
        lon_attrs = coords["longitude"][2]
        self.assertEqual(lon_attrs["standard_name"], "longitude")
        self.assertEqual(lon_attrs["units"], "degrees_east")

    def test_vertices_dims_include_vertices_dim(self) -> None:
        nj, ni, nv = 3, 5, 4
        lat2d, lon2d = _latlon(nj, ni)
        blat = np.zeros((nj, ni, nv))
        blon = np.zeros((nj, ni, nv))
        grid = Grid(
            latitude=lat2d,
            longitude=lon2d,
            latitude_vertices=blat,
            longitude_vertices=blon,
        )
        coords, data_vars, _ = grid.to_dataset_coords(["j", "i"])
        self.assertEqual(data_vars["vertices_latitude"][0], ("j", "i", "vertices"))

    def test_custom_vertices_dim_name(self) -> None:
        nj, ni, nv = 3, 5, 4
        lat2d, lon2d = _latlon(nj, ni)
        blat = np.zeros((nj, ni, nv))
        grid = Grid(
            latitude=lat2d, longitude=lon2d, latitude_vertices=blat, vertices_dim="nv"
        )
        _, data_vars, _ = grid.to_dataset_coords(["j", "i"])
        self.assertEqual(data_vars["vertices_latitude"][0], ("j", "i", "nv"))

    def test_no_lat_lon_returns_empty(self) -> None:
        grid = Grid()
        coords, data_vars, aux_names = grid.to_dataset_coords(["j", "i"])
        self.assertEqual(coords, {})
        self.assertEqual(data_vars, {})
        self.assertEqual(aux_names, [])


# ---------------------------------------------------------------------------
# 5. must_call_cmor_grid validation (isgridaxis without Grid)
# ---------------------------------------------------------------------------


class TestMustCallCmorGrid(unittest.TestCase):
    """Axes with isgridaxis=True in the user axes list without a Grid → error."""

    def _make_project(self) -> Any:
        """Build a minimal ProjectTables with a grids table containing i/j axes."""
        from cmor4 import ProjectTables

        cv = {
            "CV": {
                "activity_id": ["CMIP"],
                "experiment_id": {
                    "historical": {
                        "experiment_id": "historical",
                        "activity_id": ["CMIP"],
                        "parent_experiment_id": ["piControl"],
                        "parent_activity_id": ["CMIP"],
                    }
                },
                "institution_id": {"NCAR": "National Center for Atmospheric Research"},
                "source_id": {"CESM2": {"institution_id": ["NCAR"]}},
                "source_type": {"AOGCM": "coupled atmosphere-ocean GCM"},
                "required_global_attributes": [],
                "mip_era": "CMIP7",
            }
        }
        var_table = {
            "Header": {"table_id": "Omon"},
            "variable_entry": {
                "tos": {
                    "out_name": "tos",
                    "units": "K",
                    "dimensions": ["time", "j", "i"],
                    "frequency": "mon",
                    "realm": "ocean",
                }
            },
        }
        coord_table = {
            "axis_entry": {
                "time": {"axis": "T", "standard_name": "time", "out_name": "time"},
            }
        }
        grids_table = {
            "axis_entry": {
                "i_index": {"out_name": "i", "units": "1"},
                "j_index": {"out_name": "j", "units": "1"},
            },
            "variable_entry": {},
            "mapping_entry": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cv.json").write_text(json.dumps(cv))
            (root / "var.json").write_text(json.dumps(var_table))
            (root / "coord.json").write_text(json.dumps(coord_table))
            (root / "grids.json").write_text(json.dumps(grids_table))
            return ProjectTables(
                root / "cv.json",
                [root / "var.json"],
                coordinate_table=root / "coord.json",
                grid_table=root / "grids.json",
            )

    def test_isgridaxis_without_grid_raises(self) -> None:
        """Axis with isgridaxis=True in the user axes list, no grid → error."""
        # The isgridaxis guard is a fast-fail at the top of validate_components;
        # it fires before any variable-table lookup.
        # Build the simplest possible project (no variable table) so no other
        # validation noise interferes.

        project = self._make_project()
        time_axis = Axis(
            name="time",
            values=np.array([0.5]),
            units="days since 2000-01-01",
            axis="T",
        )
        # Axis flagged as a grid axis (as if it came out of Grid.axes)
        # but mistakenly placed in the user axes list.
        grid_axis = Axis(name="i", values=np.arange(8, dtype="f4"), isgridaxis=True)

        # We test the guard directly through the exceptions it raises.
        # Variable resolution isn't reached because the check is at the top.
        variable = Variable(name="tos", units="K", dimensions=("time", "i", "j"))

        with self.assertRaises(TableValidationError) as ctx:
            project.validate_dataset(None, variable, [time_axis, grid_axis], grid=None)
        msg = str(ctx.exception)
        self.assertIn("isgridaxis", msg)
        self.assertIn("grid=", msg)


# ---------------------------------------------------------------------------
# 6. ProjectTables.grid(axes=...)
# ---------------------------------------------------------------------------


class TestProjectTablesGridFactory(unittest.TestCase):
    def _make_project(self) -> Any:
        from cmor4 import ProjectTables

        cv = {"CV": {"required_global_attributes": [], "mip_era": "CMIP7"}}
        grids_table = {
            "axis_entry": {
                "i_index": {"out_name": "i", "units": "1"},
                "j_index": {"out_name": "j", "units": "1"},
            },
            "variable_entry": {},
            "mapping_entry": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cv.json").write_text(json.dumps(cv))
            (root / "var.json").write_text(
                json.dumps({"Header": {"table_id": "T"}, "variable_entry": {}})
            )
            (root / "grids.json").write_text(json.dumps(grids_table))
            return ProjectTables(
                root / "cv.json",
                [root / "var.json"],
                grid_table=root / "grids.json",
            )

    def test_grid_with_valid_axes(self) -> None:
        project = self._make_project()
        i = Axis(name="i_index", out_name="i", values=np.arange(8, dtype="f4"))
        j = Axis(name="j_index", out_name="j", values=np.arange(4, dtype="f4"))
        grid = project.grid(axes=[j, i])
        self.assertEqual(len(grid.axes), 2)
        self.assertTrue(all(a.isgridaxis for a in grid.axes))
        self.assertEqual(grid.dimensions, ("j", "i"))

    def test_grid_with_invalid_axis_raises(self) -> None:
        project = self._make_project()
        bad = Axis(name="lat", values=np.linspace(-90, 90, 10, dtype="f4"))
        with self.assertRaises(TableValidationError) as ctx:
            project.grid(axes=[bad])
        self.assertIn("lat", str(ctx.exception))

    def test_grid_without_axes_no_validation(self) -> None:
        project = self._make_project()
        grid = project.grid(dimensions=("y", "x"))
        self.assertEqual(grid.dimensions, ("y", "x"))
        self.assertEqual(grid.axes, [])

    def test_grid_axes_empty_table_skips_validation(self) -> None:
        """When the grids table has no axis entries, any axis is accepted."""
        from cmor4 import ProjectTables

        cv = {"CV": {"required_global_attributes": [], "mip_era": "CMIP7"}}
        grids_table: dict[str, dict[str, dict[str, str]]] = {
            "axis_entry": {},
            "variable_entry": {},
            "mapping_entry": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cv.json").write_text(json.dumps(cv))
            (root / "var.json").write_text(
                json.dumps({"Header": {"table_id": "T"}, "variable_entry": {}})
            )
            (root / "grids.json").write_text(json.dumps(grids_table))
            project = ProjectTables(
                root / "cv.json",
                [root / "var.json"],
                grid_table=root / "grids.json",
            )
        any_axis = Axis(name="x", values=np.arange(5, dtype="f4"))
        grid = project.grid(axes=[any_axis])  # should not raise
        self.assertEqual(len(grid.axes), 1)


# ---------------------------------------------------------------------------
# 7. _validate_grid_dimensions — both paths
# ---------------------------------------------------------------------------


class TestValidateGridDimensions(unittest.TestCase):
    def test_axis_path_passes_matching_shapes(self) -> None:
        from cmor4.tables import _validate_grid_dimensions

        j = Axis(name="j", out_name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", out_name="i", values=np.arange(8, dtype="f4"))
        lat2d, lon2d = _latlon(4, 8)
        grid = Grid(axes=[j, i], latitude=lat2d, longitude=lon2d)
        _validate_grid_dimensions(grid, [])  # should not raise

    def test_axis_path_raises_on_shape_mismatch(self) -> None:
        from cmor4.tables import _validate_grid_dimensions

        j = Axis(name="j", out_name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", out_name="i", values=np.arange(8, dtype="f4"))
        wrong_lat = np.zeros((5, 8))  # j should be 4
        grid = Grid(axes=[j, i], latitude=wrong_lat, longitude=np.zeros((5, 8)))
        with self.assertRaises(TableValidationError):
            _validate_grid_dimensions(grid, [])

    def test_name_path_passes_with_matching_axes(self) -> None:
        from cmor4.tables import _validate_grid_dimensions

        j = Axis(name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", values=np.arange(8, dtype="f4"))
        lat2d, lon2d = _latlon(4, 8)
        grid = Grid(dimensions=("j", "i"), latitude=lat2d, longitude=lon2d)
        _validate_grid_dimensions(grid, [j, i])  # should not raise

    def test_name_path_raises_on_missing_axis(self) -> None:
        from cmor4.tables import _validate_grid_dimensions

        i = Axis(name="i", values=np.arange(8, dtype="f4"))
        grid = Grid(dimensions=("j", "i"))
        with self.assertRaises(TableValidationError) as ctx:
            _validate_grid_dimensions(grid, [i])
        self.assertIn("j", str(ctx.exception))

    def test_name_path_raises_on_shape_mismatch(self) -> None:
        from cmor4.tables import _validate_grid_dimensions

        j = Axis(name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", values=np.arange(8, dtype="f4"))
        wrong_lat = np.zeros((3, 8))  # j should be 4
        grid = Grid(
            dimensions=("j", "i"), latitude=wrong_lat, longitude=np.zeros((3, 8))
        )
        with self.assertRaises(TableValidationError):
            _validate_grid_dimensions(grid, [j, i])


# ---------------------------------------------------------------------------
# 8. Backward compatibility — name-based Grid still works
# ---------------------------------------------------------------------------


class TestNameBasedGridBackcompat(unittest.TestCase):
    def test_string_dimensions_no_axes(self) -> None:
        grid = Grid(dimensions=("y", "x"))
        self.assertEqual(grid.dimensions, ("y", "x"))
        self.assertEqual(grid.axes, [])

    def test_variable_dimensions_string_path(self) -> None:
        grid = Grid(dimensions=("y", "x"))
        var = Variable(name="tas", units="K", dimensions=("time", "y", "x"))
        self.assertEqual(grid.variable_dimensions(var), ("time", "y", "x"))

    def test_isgridaxis_not_set_on_string_path(self) -> None:
        grid = Grid(dimensions=("y", "x"))
        # No axes to check — just confirm no error and axes is empty.
        self.assertEqual(grid.axes, [])


# ---------------------------------------------------------------------------
# 9. Round-trip: create_dataset with axis-based Grid
# ---------------------------------------------------------------------------


class TestCreateDatasetWithGridAxes(unittest.TestCase):
    """create_dataset should correctly write dimensional + auxiliary axes."""

    def _make_grid_and_axes(
        self, nj: int = 4, ni: int = 8
    ) -> tuple[Axis, Axis, Axis, Grid, np.ndarray, np.ndarray]:
        lat2d, lon2d = _latlon(nj, ni)
        time_axis = Axis(
            name="time",
            values=np.array([15.0]),
            units="days since 2000-01-01",
            axis="T",
        )
        j_axis = Axis(name="j", out_name="j", values=np.arange(nj, dtype="f4"))
        i_axis = Axis(name="i", out_name="i", values=np.arange(ni, dtype="f4"))
        grid = Grid(axes=[j_axis, i_axis], latitude=lat2d, longitude=lon2d)
        return time_axis, j_axis, i_axis, grid, lat2d, lon2d

    def test_curvilinear_grid_dataset(self) -> None:
        from cmor4.dataset import create_dataset
        from cmor4.datasetinfo import DatasetInfo

        nj, ni = 4, 8
        time_axis, j_axis, i_axis, grid, lat2d, lon2d = self._make_grid_and_axes(nj, ni)

        variable = Variable(
            name="tos",
            units="K",
            dimensions=("time", "j", "i"),
            frequency="mon",
        )
        data = np.random.rand(1, nj, ni).astype("f4") + 273.15
        dataset = DatasetInfo.from_prepared({"outpath": "/tmp", "frequency": "mon"})

        ds = create_dataset(dataset, variable, [time_axis], data, grid=grid)

        # Variable should have dims (time, j, i)
        self.assertIn("tos", ds.data_vars)
        self.assertEqual(tuple(ds["tos"].dims), ("time", "j", "i"))

        # Dimensional grid axes must appear as coordinates
        self.assertIn("j", ds.coords)
        self.assertIn("i", ds.coords)

        # Auxiliary lat/lon must appear as coordinates
        self.assertIn("latitude", ds.coords)
        self.assertIn("longitude", ds.coords)

        # coordinates attribute should reference lat and lon
        coord_attr = ds["tos"].attrs.get("coordinates", "")
        self.assertIn("latitude", coord_attr)
        self.assertIn("longitude", coord_attr)

    def test_curvilinear_grid_with_vertices(self) -> None:
        from cmor4.dataset import create_dataset
        from cmor4.datasetinfo import DatasetInfo

        nj, ni, nv = 3, 5, 4
        lat2d, lon2d = _latlon(nj, ni)
        blat = np.random.rand(nj, ni, nv).astype("f4")
        blon = np.random.rand(nj, ni, nv).astype("f4")

        j_axis = Axis(name="j", out_name="j", values=np.arange(nj, dtype="f4"))
        i_axis = Axis(name="i", out_name="i", values=np.arange(ni, dtype="f4"))
        time_axis = Axis(
            name="time",
            values=np.array([15.0]),
            units="days since 2000-01-01",
            axis="T",
        )
        grid = Grid(
            axes=[j_axis, i_axis],
            latitude=lat2d,
            longitude=lon2d,
            latitude_vertices=blat,
            longitude_vertices=blon,
        )
        variable = Variable(
            name="tos",
            units="K",
            dimensions=("time", "j", "i"),
            frequency="mon",
        )
        data = np.random.rand(1, nj, ni).astype("f4") + 273.15
        dataset = DatasetInfo.from_prepared({"outpath": "/tmp", "frequency": "mon"})

        ds = create_dataset(dataset, variable, [time_axis], data, grid=grid)

        # Bounds variables must be present
        self.assertIn("vertices_latitude", ds.data_vars)
        self.assertIn("vertices_longitude", ds.data_vars)
        # bounds attribute on lat coordinate
        self.assertEqual(ds["latitude"].attrs.get("bounds"), "vertices_latitude")

    def test_grid_axes_also_in_caller_axes_no_duplicates(self) -> None:
        """When grid axes are also passed in the caller's axes list they must
        not be processed twice — no duplicate coords or doubled coordinates
        attribute entries."""
        from cmor4.dataset import create_dataset
        from cmor4.datasetinfo import DatasetInfo

        nj, ni = 4, 8
        lat2d, lon2d = _latlon(nj, ni)
        time_axis = Axis(
            name="time",
            values=np.array([15.0]),
            units="days since 2000-01-01",
            axis="T",
        )
        j_axis = Axis(name="j", out_name="j", values=np.arange(nj, dtype="f4"))
        i_axis = Axis(name="i", out_name="i", values=np.arange(ni, dtype="f4"))
        grid = Grid(axes=[j_axis, i_axis], latitude=lat2d, longitude=lon2d)

        variable = Variable(
            name="tos",
            units="K",
            dimensions=("time", "j", "i"),
            frequency="mon",
        )
        data = np.random.rand(1, nj, ni).astype("f4") + 273.15
        dataset = DatasetInfo.from_prepared({"outpath": "/tmp", "frequency": "mon"})

        # Pass j and i explicitly in the caller's axes list as well as via grid
        ds = create_dataset(
            dataset, variable, [time_axis, j_axis, i_axis], data, grid=grid
        )

        self.assertEqual(tuple(ds["tos"].dims), ("time", "j", "i"))
        # Each coord appears exactly once
        self.assertIn("j", ds.coords)
        self.assertIn("i", ds.coords)
        # coordinates attribute lists lat and lon exactly once each
        coord_attr = ds["tos"].attrs.get("coordinates", "")
        self.assertEqual(coord_attr.split().count("latitude"), 1)
        self.assertEqual(coord_attr.split().count("longitude"), 1)


if __name__ == "__main__":
    unittest.main()
