"""Tests for Grid.axes — the axis-based grid design.

Covers:
* Grid construction from Axis objects (axis-based path)
* Grid construction from dimension strings (name-based path, backward compat)
* isgridaxis flag set on axes stored inside Grid.axes
* Caller's original Axis instance is not mutated
* dimensions derived automatically from axes
* variable_dimensions() with axes, with strings, fallback
* _grid_axes() in core includes dimensional axes before lat/lon auxiliaries
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
from cmor4.core import _grid_axes
from cmor4.exceptions import TableValidationError

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
# 4. _grid_axes() in core
# ---------------------------------------------------------------------------


class TestGridAxesInCore(unittest.TestCase):
    """_grid_axes() should return dimensional axes THEN auxiliary lat/lon."""

    def test_returns_dimensional_axes_first(self) -> None:
        j = Axis(name="j", out_name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", out_name="i", values=np.arange(8, dtype="f4"))
        lat2d, lon2d = _latlon(4, 8)
        grid = Grid(axes=[j, i], latitude=lat2d, longitude=lon2d)
        result = _grid_axes(grid, (), None)
        # First two should be the dimensional axes
        self.assertEqual(result[0].name, "j")
        self.assertEqual(result[1].name, "i")

    def test_returns_lat_lon_auxiliary_after_dimensional(self) -> None:
        j = Axis(name="j", out_name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", out_name="i", values=np.arange(8, dtype="f4"))
        lat2d, lon2d = _latlon(4, 8)
        grid = Grid(axes=[j, i], latitude=lat2d, longitude=lon2d)
        result = _grid_axes(grid, (), None)
        self.assertEqual(len(result), 4)  # j, i, lat, lon
        # lat and lon should be auxiliary
        for aux in result[2:]:
            self.assertTrue(bool(aux.auxiliary), f"{aux.name} should be auxiliary")

    def test_no_latlon_returns_only_dimensional(self) -> None:
        j = Axis(name="j", out_name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", out_name="i", values=np.arange(8, dtype="f4"))
        grid = Grid(axes=[j, i])
        result = _grid_axes(grid, (), None)
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0].auxiliary)
        self.assertFalse(result[1].auxiliary)

    def test_none_grid_returns_empty(self) -> None:
        self.assertEqual(_grid_axes(None, (), None), [])

    def test_no_axes_only_latlon(self) -> None:
        """Name-based grid: no axes, just lat/lon arrays."""
        lat2d, lon2d = _latlon(4, 8)
        grid = Grid(dimensions=("j", "i"), latitude=lat2d, longitude=lon2d)
        result = _grid_axes(grid, (), None)
        self.assertEqual(len(result), 2)
        self.assertTrue(bool(result[0].auxiliary))
        self.assertTrue(bool(result[1].auxiliary))

    def test_auxiliary_lat_uses_grid_dimensions(self) -> None:
        j = Axis(name="j", out_name="j", values=np.arange(4, dtype="f4"))
        i = Axis(name="i", out_name="i", values=np.arange(8, dtype="f4"))
        lat2d, lon2d = _latlon(4, 8)
        grid = Grid(axes=[j, i], latitude=lat2d, longitude=lon2d)
        result = _grid_axes(grid, (), None)
        lat_axis = next(a for a in result if a.name == "latitude")
        self.assertEqual(list(lat_axis.dimensions), ["j", "i"])


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
            project.validate_components(
                None, variable, [time_axis, grid_axis], grid=None
            )
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
        grids_table = {
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
        from cmor4.core import create_dataset
        from cmor4.dataset import DatasetInfo

        nj, ni = 4, 8
        time_axis, j_axis, i_axis, grid, lat2d, lon2d = self._make_grid_and_axes(nj, ni)

        variable = Variable(
            name="tos",
            units="K",
            dimensions=("time", "j", "i"),
            frequency="mon",
        )
        data = np.random.rand(1, nj, ni).astype("f4") + 273.15
        dataset = DatasetInfo.from_mapping({"outpath": "/tmp", "frequency": "mon"})

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
        from cmor4.core import create_dataset
        from cmor4.dataset import DatasetInfo

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
        dataset = DatasetInfo.from_mapping({"outpath": "/tmp", "frequency": "mon"})

        ds = create_dataset(dataset, variable, [time_axis], data, grid=grid)

        # Bounds variables must be present
        self.assertIn("vertices_latitude", ds.data_vars)
        self.assertIn("vertices_longitude", ds.data_vars)
        # bounds attribute on lat coordinate
        self.assertEqual(ds["latitude"].attrs.get("bounds"), "vertices_latitude")


if __name__ == "__main__":
    unittest.main()
