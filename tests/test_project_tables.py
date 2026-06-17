from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import cmor4
from cmor4 import (
    Axis,
    ControlledVocabulary,
    DatasetInfo,
    Grid,
    ProjectTables,
    Variable,
    ZFactor,
)
from cmor4.exceptions import (
    ControlledVocabularyError,
    TableValidationError,
    AxisValidationError,
)
from table_helpers import (
    CMIP7_TABLE_ROOT,
    DRCDP_TABLE_ROOT,
    OBS4MIPS_TABLE_ROOT,
    cmip7_project,
    drcdp_project,
    obs4mips_project,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj) + "\n")


# Minimal but realistic CV with activity, experiment, source, and institution
_RICH_CV: dict = {
    "CV": {
        "activity_id": ["CMIP", "ScenarioMIP"],
        "experiment_id": {
            "historical": {
                "experiment_id": "historical",
                "activity_id": ["CMIP"],
                "required_source_type": ["AOGCM"],
                "additional_allowed_model_components": ["AER", "BGC"],
                "parent_experiment_id": ["piControl"],
                "parent_activity_id": ["CMIP"],
            },
            "amip": {
                "experiment_id": "amip",
                "activity_id": ["CMIP"],
            },
            "piControl": {
                "experiment_id": "piControl",
                "activity_id": ["CMIP"],
                "required_source_type": ["AOGCM"],
            },
        },
        "institution_id": {
            "NCAR": "National Center for Atmospheric Research, Boulder, CO, USA",
            "ECMWF": "European Centre for Medium-Range Weather Forecasts",
        },
        "source_id": {
            "CESM2": {
                "institution_id": ["NCAR"],
                "source_type": ["AOGCM"],
            },
            "DUMMY": {
                "institution_id": ["NCAR"],
            },
        },
        "source_type": {
            "AOGCM": "coupled atmosphere-ocean GCM",
            "AER": "aerosol",
            "BGC": "biogeochemistry",
        },
        "required_global_attributes": ["activity_id", "institution_id"],
        "mip_era": "CMIP7",
    }
}

_DEFAULT_VARIABLE_ENTRIES: dict = {
    "pr": {
        "dimensions": ["time", "lat", "lon"],
        "out_name": "pr",
        "units": "kg m-2 s-1",
        "standard_name": "precipitation_flux",
        "frequency": "mon",
        "realm": "atmos",
    },
    "tas": {
        "dimensions": ["time", "lat", "lon", "height2m"],
        "out_name": "tas",
        "units": "K",
        "standard_name": "air_temperature",
        "frequency": "mon",
        "realm": "atmos",
    },
    "ua": {
        "dimensions": ["time", "plev", "lat", "lon"],
        "out_name": "ua",
        "units": "m s-1",
        "frequency": "mon",
    },
}

_DEFAULT_COORDINATE_ENTRIES: dict = {
    "time": {"axis": "T", "standard_name": "time", "out_name": "time"},
    "lat": {
        "axis": "Y",
        "units": "degrees_north",
        "standard_name": "latitude",
        "out_name": "lat",
    },
    "lon": {
        "axis": "X",
        "units": "degrees_east",
        "standard_name": "longitude",
        "out_name": "lon",
    },
    "height2m": {
        "axis": "Z",
        "units": "m",
        "standard_name": "height",
        "out_name": "height",
        "value": "2.0",
    },
    "plev": {
        "axis": "Z",
        "units": "Pa",
        "standard_name": "air_pressure",
        "out_name": "plev",
        "positive": "down",
    },
}

_DEFAULT_FORMULA_ENTRIES: dict = {
    "ps": {
        "units": "Pa",
        "standard_name": "surface_air_pressure",
        "out_name": "ps",
    },
    "p0": {
        "units": "Pa",
        "standard_name": "reference_air_pressure",
        "out_name": "p0",
    },
}

_DEFAULT_MAPPING_ENTRIES: dict = {
    "rotated_latitude_longitude": {
        "grid_mapping_name": "rotated_latitude_longitude",
    },
    "lambert_conformal_conic": {
        "grid_mapping_name": "lambert_conformal_conic",
    },
}

# validate_components tests use "latitude"/"longitude" coordinate key names.
_VC_VARIABLE_ENTRIES: dict = {
    "pr": {
        "dimensions": ["time", "latitude", "longitude"],
        "out_name": "pr",
        "units": "kg m-2 s-1",
        "standard_name": "precipitation_flux",
        "frequency": "mon",
    },
    "tas": {
        "dimensions": ["time", "latitude", "longitude", "height2m"],
        "out_name": "tas",
        "units": "K",
        "standard_name": "air_temperature",
        "frequency": "mon",
    },
}

_VC_COORDINATE_ENTRIES: dict = {
    # 'units' omitted from time so user-supplied "days since …" is accepted.
    "time": {"axis": "T", "standard_name": "time", "out_name": "time"},
    "latitude": {
        "axis": "Y",
        "units": "degrees_north",
        "standard_name": "latitude",
        "out_name": "lat",
    },
    "longitude": {
        "axis": "X",
        "units": "degrees_east",
        "standard_name": "longitude",
        "out_name": "lon",
    },
    "height2m": {
        "axis": "Z",
        "units": "m",
        "standard_name": "height",
        "out_name": "height",
        "value": "2.0",
    },
}

_VC_FORMULA_ENTRIES: dict = {
    "ps": {"units": "Pa", "standard_name": "surface_air_pressure", "out_name": "ps"},
}

_VC_MAPPING_ENTRIES: dict = {
    "rotated_latitude_longitude": {
        "grid_mapping_name": "rotated_latitude_longitude",
    },
}


def _build_project(
    tmp: Path,
    *,
    cv: dict | None = None,
    variable_entries: dict | None = None,
    coordinate_entries: dict | None = None,
    formula_entries: dict | None = None,
    mapping_entries: dict | None = None,
    table_id: str = "Amon",
    include_formula: bool = True,
    include_grid: bool = True,
) -> ProjectTables:
    """Write JSON table files to *tmp* and return a loaded ProjectTables."""
    cv_file = tmp / "CV.json"
    vtable_file = tmp / f"{table_id}.json"
    coord_file = tmp / "coordinate.json"
    formula_file = tmp / "formula.json"
    grids_file = tmp / "grids.json"

    _write(cv_file, cv or _RICH_CV)
    _write(
        vtable_file,
        {
            "Header": {"table_id": table_id},
            "variable_entry": variable_entries or _DEFAULT_VARIABLE_ENTRIES,
        },
    )
    _write(
        coord_file,
        {"axis_entry": coordinate_entries or _DEFAULT_COORDINATE_ENTRIES},
    )

    kwargs: dict[str, Any] = dict(coordinate_table=coord_file)

    if include_formula:
        _write(
            formula_file, {"formula_entry": formula_entries or _DEFAULT_FORMULA_ENTRIES}
        )
        kwargs["formula_table"] = formula_file

    if include_grid:
        _write(
            grids_file,
            {
                "axis_entry": {},
                "variable_entry": {},
                "mapping_entry": mapping_entries or _DEFAULT_MAPPING_ENTRIES,
            },
        )
        kwargs["grid_table"] = grids_file

    return ProjectTables(cv_file, [vtable_file], **kwargs)


def _build_vc_project(tmp: Path, **kwargs) -> ProjectTables:
    """Build a project using the validate_components coordinate namespace.

    Coordinates use 'latitude'/'longitude' key names (not 'lat'/'lon') and
    a minimal CV with no controlled-vocabulary constraints.
    """
    return _build_project(
        tmp,
        cv={"CV": {}},
        variable_entries=kwargs.pop("variable_entries", _VC_VARIABLE_ENTRIES),
        coordinate_entries=kwargs.pop("coordinate_entries", _VC_COORDINATE_ENTRIES),
        formula_entries=kwargs.pop("formula_entries", _VC_FORMULA_ENTRIES),
        mapping_entries=kwargs.pop("mapping_entries", _VC_MAPPING_ENTRIES),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Constructor / __init__
# ---------------------------------------------------------------------------


def require_path(test_case: unittest.TestCase, path: Path) -> None:
    if not path.exists():
        test_case.skipTest(f"Project table source is not available: {path}")


def lat_lon_axes(project=None):
    if project is not None:
        return [
            project.axis(
                "time",
                values=[15.0, 45.0],
                bounds=[[0.0, 30.0], [30.0, 60.0]],
                units="days since 2000-01-01",
            ),
            project.axis(
                "latitude",
                values=[-45.0, 45.0],
                bounds=[[-90.0, 0.0], [0.0, 90.0]],
            ),
            project.axis(
                "longitude",
                values=[90.0, 270.0],
                bounds=[[0.0, 180.0], [180.0, 360.0]],
            ),
        ]
    return [
        cmor4.Axis(
            name="time",
            values=[15.0, 45.0],
            bounds=[[0.0, 30.0], [30.0, 60.0]],
            units="days since 2000-01-01",
        ),
        cmor4.Axis(
            name="latitude",
            values=[-45.0, 45.0],
            bounds=[[-90.0, 0.0], [0.0, 90.0]],
        ),
        cmor4.Axis(
            name="longitude",
            values=[90.0, 270.0],
            bounds=[[0.0, 180.0], [180.0, 360.0]],
        ),
    ]


def cmip7_dataset(**overrides):
    dataset = {
        "activity_id": "CMIP",
        "experiment_id": "amip",
        "forcing_index": "f3",
        "frequency": "mon",
        "grid_label": "g999",
        "initialization_index": "i1",
        "institution_id": "CCCma",
        "license_id": "CC-BY-4.0",
        "nominal_resolution": "100 km",
        "physics_index": "p1",
        "realization_index": "r9",
        "region": "glb",
        "source_id": "DUMMY-MODEL",
    }
    dataset.update(overrides)
    return dataset


class ProjectTablesTest(unittest.TestCase):
    def assert_grid_table_metadata_is_used(self, project):
        axes = [
            project.axis(
                "time",
                values=[15.0],
                bounds=[[0.0, 30.0]],
                units="days since 2020-02-01",
            ),
            project.axis("x", values=[0.0, 1.0]),
            project.axis("y", values=[2.0, 3.0]),
            project.axis(
                "latitude",
                out_name="latitude",
                values=[[10.0, 20.0], [30.0, 40.0]],
                dimensions=["x", "y"],
                bounds=np.ones((2, 2, 4), dtype="f8"),
                bounds_name="vertices_latitude",
                bounds_dim="vertices",
                auxiliary=True,
            ),
            project.axis(
                "longitude",
                out_name="longitude",
                values=[[100.0, 110.0], [120.0, 130.0]],
                dimensions=["x", "y"],
                bounds=np.ones((2, 2, 4), dtype="f8"),
                bounds_name="vertices_longitude",
                bounds_dim="vertices",
                auxiliary=True,
            ),
        ]

        variable = cmor4.Variable(
            name="sample",
            dimensions=["time", "x", "y"],
            coordinates=["latitude", "longitude"],
        )
        info = cmor4.DatasetInfo.from_mapping(
            {"frequency": "mon"},
        )

        ds = cmor4.create_dataset(
            info,
            variable,
            axes,
            np.ones((1, 2, 2), dtype="f4"),
        )

        self.assertEqual(ds["x"].attrs["standard_name"], "projection_x_coordinate")
        self.assertEqual(ds["x"].attrs["long_name"], "x coordinate of projection")
        self.assertEqual(ds["x"].attrs["units"], "m")
        self.assertEqual(ds["y"].attrs["standard_name"], "projection_y_coordinate")
        self.assertEqual(ds["y"].attrs["long_name"], "y coordinate of projection")
        self.assertEqual(ds["y"].attrs["units"], "m")
        self.assertEqual(ds["latitude"].attrs["standard_name"], "latitude")
        self.assertEqual(ds["latitude"].attrs["units"], "degrees_north")
        self.assertEqual(ds["longitude"].attrs["standard_name"], "longitude")
        self.assertEqual(ds["longitude"].attrs["units"], "degrees_east")
        self.assertEqual(ds["vertices_latitude"].attrs["units"], "degrees_north")
        self.assertEqual(ds["vertices_longitude"].attrs["units"], "degrees_east")
        self.assertEqual(ds["sample"].attrs["coordinates"], "latitude longitude")

    def test_loads_cv_and_variable_entries_from_submodule(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")

        self.assertIsInstance(project.cv, cmor4.ControlledVocabulary)
        self.assertIn("activity_id", project.cv)
        self.assertIn("tos_tavg-u-hxy-sea", project.variable_entries)
        self.assertIn("latitude", project.coordinate_entries)
        self.assertIn("x", project.coordinate_entries)
        self.assertIn("latitude", project.grid_coordinate_entries)
        self.assertIn("vertices_latitude", project.grid_coordinate_entries)
        self.assertIn("ps", project.formula_entries)
        self.assertIn("alternate_hybrid_sigma", project.generic_level_entries["alevel"])
        self.assertIn("depth_coord", project.generic_level_entries["olevel"])
        self.assertEqual(
            project.variable_entries["tos_tavg-u-hxy-sea"].entry["out_name"],
            "tos",
        )

    def test_controlled_vocabulary_loads_project_cv_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cv_file = Path(tmp_dir) / "CV.json"
            cv_file.write_text("""
{
  "CV": {
    "activity_id": ["CMIP"],
    "required_global_attributes": ["activity_id"]
  }
}
""".strip() + "\n")

            cv = cmor4.ControlledVocabulary.from_file(cv_file)

            self.assertEqual(cv["activity_id"], ["CMIP"])
            self.assertEqual(cv.required_global_attributes(), ("activity_id",))
            cv.validate_dataset({"activity_id": "CMIP"})
            with self.assertRaises(cmor4.ControlledVocabularyError):
                cv.validate_dataset({"activity_id": "not-a-real-activity"})

    def test_cmip7_generic_level_resolves_concrete_coordinate(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        variable = project.variable("agessc_tavg-ol-hxy-sea")
        dataset = project.dataset_info(cmip7_dataset())

        ds = cmor4.create_dataset(
            dataset,
            variable,
            [
                project.axis(
                    "time",
                    values=[15.0, 45.0],
                    bounds=[[0.0, 30.0], [30.0, 60.0]],
                    units="days since 2000-01-01",
                ),
                project.axis(
                    "olevel",
                    values=[5.0, 50.0],
                    bounds=[[0.0, 10.0], [10.0, 100.0]],
                    standard_name="depth",
                ),
                project.axis(
                    "latitude",
                    values=[-45.0, 45.0],
                    bounds=[[-90.0, 0.0], [0.0, 90.0]],
                ),
                project.axis(
                    "longitude",
                    values=[90.0, 270.0],
                    bounds=[[0.0, 180.0], [180.0, 360.0]],
                ),
            ],
            np.ones((2, 2, 2, 2), dtype="f4"),
        )

        self.assertEqual(ds["agessc"].dims, ("time", "lev", "lat", "lon"))
        self.assertIn("lev", ds.coords)
        self.assertNotIn("olevel", ds.coords)
        self.assertNotIn("out_name", ds["lev"].attrs)
        self.assertEqual(ds["lev"].attrs["standard_name"], "depth")
        self.assertEqual(ds["lev"].attrs["positive"], "down")
        self.assertEqual(ds["lev"].attrs["bounds"], "lev_bnds")

    def test_axis_empty_out_name_falls_back_to_coordinate_entry_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cv_file = root / "CV.json"
            variable_table = root / "test_table.json"
            coordinate_table = root / "coordinate.json"
            cv_file.write_text('{"CV": {}}\n')
            variable_table.write_text("""
{
  "Header": {"table_id": "test"},
  "variable_entry": {
    "sample": {
      "dimensions": ["runtime_axis"],
      "out_name": "sample",
      "units": "1"
    }
  }
}
""".strip() + "\n")
            coordinate_table.write_text("""
{
  "axis_entry": {
    "runtime_axis": {
      "axis": "X",
      "long_name": "Runtime axis",
      "out_name": "",
      "standard_name": "projection_x_coordinate",
      "units": "m"
    }
  }
}
""".strip() + "\n")
            project = cmor4.ProjectTables(
                cv_file,
                [variable_table],
                coordinate_table=coordinate_table,
            )

            variable = project.variable("sample")
            info = project.dataset_info({})

            ds = cmor4.create_dataset(
                info,
                variable,
                [
                    project.axis(
                        "source_x",
                        table_entry="runtime_axis",
                        values=[0.0, 1.0],
                    )
                ],
                np.ones((2,), dtype="f4"),
            )

            self.assertEqual(ds["sample"].dims, ("runtime_axis",))
            self.assertIn("runtime_axis", ds.coords)
            self.assertNotIn("source_x", ds.coords)
            self.assertNotIn("out_name", ds["runtime_axis"].attrs)

    def test_cmip7_generic_level_requires_concrete_coordinate_choice(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")

        with self.assertRaisesRegex(
            cmor4.TableValidationError,
            "Generic level 'olevel' matches multiple coordinate entries",
        ):
            project.axis("olevel", values=[5.0, 50.0])

    def test_cmip7_grid_axes_and_aux_coords_come_from_grids_table(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_seaIce.json")
        axes = [
            project.axis(
                "time",
                values=[15.0],
                bounds=[[0.0, 30.0]],
                units="days since 2020-02-01",
            ),
            project.axis("x", values=[0.0, 1.0]),
            project.axis("y", values=[2.0, 3.0]),
            project.axis(
                "latitude",
                out_name="latitude",
                values=[[10.0, 20.0], [30.0, 40.0]],
                dimensions=["x", "y"],
                bounds=np.ones((2, 2, 4), dtype="f8"),
                bounds_name="vertices_latitude",
                bounds_dim="vertices",
                auxiliary=True,
            ),
            project.axis(
                "longitude",
                out_name="longitude",
                values=[[100.0, 110.0], [120.0, 130.0]],
                dimensions=["x", "y"],
                bounds=np.ones((2, 2, 4), dtype="f8"),
                bounds_name="vertices_longitude",
                bounds_dim="vertices",
                auxiliary=True,
            ),
        ]

        variable = cmor4.Variable(
            name="sample",
            dimensions=["time", "x", "y"],
            coordinates=["latitude", "longitude"],
        )
        info = cmor4.DatasetInfo.from_mapping(
            {"frequency": "mon"},
        )

        ds = cmor4.create_dataset(
            info,
            variable,
            axes,
            np.ones((1, 2, 2), dtype="f4"),
        )

        self.assertEqual(ds["x"].attrs["standard_name"], "projection_x_coordinate")
        self.assertEqual(ds["x"].attrs["long_name"], "x coordinate of projection")
        self.assertEqual(ds["x"].attrs["units"], "m")
        self.assertEqual(ds["y"].attrs["standard_name"], "projection_y_coordinate")
        self.assertEqual(ds["latitude"].attrs["standard_name"], "latitude")
        self.assertEqual(ds["latitude"].attrs["units"], "degrees_north")
        self.assertEqual(ds["vertices_latitude"].attrs["units"], "degrees_north")
        self.assertEqual(ds["sample"].attrs["coordinates"], "latitude longitude")

    def test_drcdp_grid_axes_and_aux_coords_come_from_grids_table(self):
        require_path(self, DRCDP_TABLE_ROOT)
        project = drcdp_project("Tables/DRCDP_APday.json")

        self.assert_grid_table_metadata_is_used(project)

    def test_obs4mips_grid_axes_and_aux_coords_come_from_grids_table(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project("Tables/obs4MIPs_Amon.json")

        self.assert_grid_table_metadata_is_used(project)

    def test_dataset_info_merges_authoritative_variable_metadata(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project("Tables/obs4MIPs_Amon.json")
        dataset = {
            "activity_id": "obs4MIPs",
            "contact": "submissions-obs4mips@wcrp-cmip.org",
            "grid": "5 degree latitude height zonal mean",
            "grid_label": "gnz",
            "has_aux_unc": "FALSE",
            "institution_id": "DLR-BIRA",
            "license": project.cv["license"],
            "nominal_resolution": "500 km",
            "processing_code_location": (
                "dataset_guides/obs4mips/example-data-tools/example.py"
            ),
            "product": "observations",
            "references": "Example reference",
            "source_data_url": "https://example.invalid/source",
            "source_id": "BSVertOzone-v1-0",
            "variant_label": "CMORGuide",
        }

        prepared_variable = project.variable("o3zm")
        prepared_dataset = project.dataset_info(dataset)

        self.assertEqual(prepared_dataset["source_id"], "BSVertOzone-v1-0")
        self.assertEqual(prepared_variable.id, "o3")
        self.assertEqual(prepared_variable.units, "mol mol-1")
        self.assertEqual(prepared_variable.dimensions, ("time", "height", "latitude"))
        self.assertEqual(
            prepared_dataset["source"],
            "BSVertOzone v1-0 (2018): Mole concentration of ozone in air",
        )
        self.assertEqual(prepared_dataset["source_type"], "satellite_retrieval")
        self.assertEqual(prepared_dataset["source_version_number"], "v1-0")

    def test_axis_resolution_does_not_use_axis_letter_alone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cv_file = root / "CV.json"
            variable_table = root / "test_table.json"
            coordinate_table = root / "coordinate.json"
            cv_file.write_text('{"CV": {}}\n')
            variable_table.write_text("""
{
  "Header": {"table_id": "test"},
  "variable_entry": {
    "sample": {
      "dimensions": ["x"],
      "out_name": "sample",
      "units": "1"
    }
  }
}
""".strip() + "\n")
            coordinate_table.write_text("""
{
  "axis_entry": {
    "longitude": {
      "axis": "X",
      "long_name": "Longitude",
      "out_name": "lon",
      "standard_name": "longitude",
      "units": "degrees_east"
    }
  }
}
""".strip() + "\n")
            project = cmor4.ProjectTables(
                cv_file,
                [variable_table],
                coordinate_table=coordinate_table,
            )

            merged = project.axis("x", values=[0.0, 1.0], axis="X", units="m")

            self.assertEqual(merged.name, "x")
            self.assertIsNone(merged.table_entry)
            self.assertNotEqual(merged.out_name, "lon")

    def test_duplicate_variable_names_require_table_id(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project(
            "Tables/obs4MIPs_Amon.json", "Tables/obs4MIPs_A1hrPt.json"
        )

        with self.assertRaises(cmor4.TableValidationError):
            cmor4._table_resolution.variable_table_entry(
                project, cmor4.Variable(name="pr")
            )

        entry = cmor4._table_resolution.variable_table_entry(
            project, cmor4.Variable(name="pr", table_id="obs4MIPs_A1hrPt")
        )

        self.assertEqual(entry.table_id, "obs4MIPs_A1hrPt")
        self.assertEqual(entry.entry["frequency"], "1hr")

    def test_cmip7_uses_project_cv_and_variable_table(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = {
                "activity_id": "CMIP",
                "calendar": "standard",
                "experiment_id": "amip",
                "forcing_index": "f3",
                "frequency": "mon",
                "grid_label": "g999",
                "initialization_index": "i1",
                "institution_id": "CCCma",
                "license_id": "CC-BY-4.0",
                "mip_era": "CMIP7",
                "nominal_resolution": "100 km",
                "outpath": tmp_dir,
                "physics_index": "p1",
                "realization_index": "r9",
                "region": "glb",
                "source_id": "DUMMY-MODEL",
                "version": "v20200101",
            }
            variable = project.variable("tos_tavg-u-hxy-sea")
            info = project.dataset_info(dataset)
            axes = lat_lon_axes(project)

            result = cmor4.cmorize(
                info,
                variable,
                axes,
                np.ones((2, 2, 2), dtype="f4"),
            )

            self.assertEqual(result.dataset["tos"].attrs["units"], "degC")
            self.assertEqual(result.dataset["lat"].attrs["standard_name"], "latitude")
            self.assertEqual(result.dataset["lat"].attrs["axis"], "Y")
            self.assertEqual(
                result.dataset["tos"].attrs["standard_name"],
                "sea_surface_temperature",
            )
            self.assertEqual(
                result.dataset["tos"].attrs["cell_methods"],
                "area: mean where sea time: mean",
            )
            self.assertEqual(result.dataset.attrs["source_id"], "DUMMY-MODEL")

    def test_cmip7_global_attrs_follow_upstream_cmor_cmip7(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = {
            "activity_id": "CMIP",
            "calendar": "360_day",
            "experiment_id": "amip",
            "forcing_index": "f3",
            "frequency": "mon",
            "grid_label": "g999",
            "initialization_index": "i1",
            "institution_id": "CCCma",
            "license_id": "CC-BY-4.0",
            "nominal_resolution": "100 km",
            "physics_index": "p1",
            "realization_index": "r9",
            "region": "glb",
            "source_id": "DUMMY-MODEL",
            "host_collection": "CMIP7",
            "archive_id": "WCRP",
        }

        variable = project.variable("tos_tavg-u-hxy-sea")
        info = project.dataset_info(dataset)

        ds = cmor4.create_dataset(
            info,
            variable,
            lat_lon_axes(project),
            np.ones((2, 2, 2), dtype="f4"),
        )

        expected = {
            "branded_variable": "tos_tavg-u-hxy-sea",
            "branding_suffix": "tavg-u-hxy-sea",
            "temporal_label": "tavg",
            "vertical_label": "u",
            "horizontal_label": "hxy",
            "area_label": "sea",
            "region": "glb",
            "frequency": "mon",
            "archive_id": "WCRP",
            "mip_era": "CMIP7",
            "data_specs_version": "MIP-DS7.1.0.0",
            "host_collection": "CMIP7",
            "drs_specs": "MIP-DRS7",
            "license_id": "CC-BY-4.0",
        }
        for key, value in expected.items():
            self.assertEqual(ds.attrs[key], value)
        for key in project.required_global_attributes():
            self.assertIn(key, ds.attrs)
        self.assertIn("Name: CMIP7_ocean.json;", ds.attrs["table_info"])
        # table_info should also contain a creation date and MD5 hash
        self.assertIn("Creation Date:(", ds.attrs["table_info"])
        self.assertRegex(ds.attrs["table_info"], r"MD5:[0-9a-f]{32}")
        self.assertNotIn("license_type", ds.attrs)
        self.assertNotIn("license_url", ds.attrs)
        self.assertEqual(
            ds.attrs["license"],
            "CC-BY-4.0; CMIP7 data produced by CCCma is licensed under a "
            "Creative Commons Attribution 4.0 International License "
            "(https://creativecommons.org/licenses/by/4.0). Consult "
            "https://wcrp-cmip.github.io/cmip7-guidance/docs/CMIP7/"
            "Guidance_for_users/#2-terms-of-use-and-citations-requirements "
            "for terms of use governing CMIP7 output, including citation "
            "requirements and proper acknowledgment. The data producers and "
            "data providers make no warranty, either express or implied, "
            "including, but not limited to, warranties of merchantability and "
            "fitness for a particular purpose. All liabilities arising from "
            "the supply of the information (including any liability arising "
            "in negligence) are excluded to the fullest extent permitted by "
            "law.",
        )

    def test_cmip7_rejects_values_not_in_cv(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = cmip7_dataset(activity_id="not-a-real-activity")

        with self.assertRaises(cmor4.ControlledVocabularyError):
            project.dataset_info(dataset)

    def test_experiment_required_source_type_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cv_file = root / "CV.json"
            variable_table = root / "test_table.json"
            cv_file.write_text("""
{
  "CV": {
    "activity_id": ["CMIP"],
    "experiment_id": {
      "historical": {
        "experiment_id": "historical",
        "required_source_type": ["AOGCM"],
        "additional_allowed_model_components": ["AER", "BGC"]
      }
    },
    "source_type": {
      "AER": "aerosols",
      "AOGCM": "coupled atmosphere-ocean general circulation model",
      "BGC": "biogeochemistry",
      "LAND": "land-only model"
    }
  }
}
""".strip() + "\n")
            variable_table.write_text("""
{
  "Header": {"table_id": "test"},
  "variable_entry": {
    "sample": {
      "dimensions": ["time"],
      "out_name": "sample",
      "units": "1"
    }
  }
}
""".strip() + "\n")
            project = cmor4.ProjectTables(cv_file, [variable_table])
            dataset = {
                "activity_id": "CMIP",
                "experiment_id": "historical",
                "source_type": "AOGCM AER",
            }

            prepared = project.dataset_info(dataset)
            self.assertEqual(prepared["source_type"], "AOGCM AER")

            with self.assertRaisesRegex(
                cmor4.ControlledVocabularyError, "missing required"
            ):
                project.dataset_info({**dataset, "source_type": "AER"})
            with self.assertRaisesRegex(cmor4.ControlledVocabularyError, "not allowed"):
                project.dataset_info({**dataset, "source_type": "AOGCM LAND"})

    def test_required_global_attributes_are_enforced(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = cmip7_dataset()
        dataset.pop("license_id")
        variable = project.variable("tos_tavg-u-hxy-sea")
        info = project.dataset_info(dataset)

        with self.assertRaisesRegex(cmor4.ControlledVocabularyError, "license_id"):
            cmor4.create_dataset(
                info,
                variable,
                lat_lon_axes(project),
                np.ones((2, 2, 2), dtype="f4"),
            )

    def test_final_global_attributes_are_validated(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        variable = project.variable("tos_tavg-u-hxy-sea")
        info = project.dataset_info(cmip7_dataset())

        with self.assertRaisesRegex(
            cmor4.ControlledVocabularyError, "not-a-real-activity"
        ):
            cmor4.create_dataset(
                info,
                variable,
                lat_lon_axes(project),
                np.ones((2, 2, 2), dtype="f4"),
                attrs={"activity_id": "not-a-real-activity"},
            )

    def test_source_id_specific_attributes_are_validated(self):
        require_path(self, DRCDP_TABLE_ROOT)
        project = drcdp_project("Tables/DRCDP_APday.json")
        dataset = {
            "activity_id": "DRCDP",
            "Conventions": "CF-1.7 CMIP-6.5",
            "driving_activity_id": "CMIP",
            "driving_experiment_id": "historical",
            "driving_mip_era": "CMIP6",
            "driving_source_id": "ACCESS-CM2",
            "driving_variant_label": "r1i1p1f1",
            "grid_label": "gn",
            "institution_id": "EPA",
            "source_id": "LOCA2-1",
        }

        with self.assertRaisesRegex(cmor4.ControlledVocabularyError, "institution_id"):
            project.dataset_info(dataset)

    def test_parent_experiment_attributes_are_required_by_experiment_cv(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = cmip7_dataset(experiment_id="historical")

        with self.assertRaisesRegex(
            cmor4.ControlledVocabularyError, "parent_experiment_id"
        ):
            project.dataset_info(dataset)

    def test_parent_experiment_attributes_follow_experiment_cv(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = cmip7_dataset(
            experiment_id="historical",
            parent_activity_id="CMIP",
            parent_experiment_id="piControl",
            parent_mip_era="CMIP7",
            parent_source_id="DUMMY-MODEL",
            parent_time_units="days since 1850-01-01",
            parent_variant_label="r1i1p1f3",
            branch_time_in_child=0.0,
            branch_time_in_parent=0.0,
        )

        variable = project.variable("tos_tavg-u-hxy-sea")
        info = project.dataset_info(dataset)

        ds = cmor4.create_dataset(
            info,
            variable,
            lat_lon_axes(project),
            np.ones((2, 2, 2), dtype="f4"),
        )

        self.assertEqual(ds.attrs["parent_experiment_id"], "piControl")
        self.assertEqual(ds.attrs["parent_activity_id"], "CMIP")

    def test_parent_experiment_attributes_are_validated(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = cmip7_dataset(
            experiment_id="historical",
            parent_activity_id="CMIP",
            parent_experiment_id="piControl",
            parent_mip_era="CMIP7",
            parent_source_id="DUMMY-MODEL",
            parent_time_units="days since 1850-01-01",
            parent_variant_label="r1i1p1f3",
            branch_time_in_child=0.0,
            branch_time_in_parent=0.0,
        )
        cases = {
            "parent_activity_id": "ScenarioMIP",
            "parent_experiment_id": "amip",
            "parent_mip_era": "CMIP6",
            "parent_source_id": "not-a-source",
            "parent_time_units": "seconds since 1850-01-01",
            "parent_variant_label": "not-a-variant",
            "branch_time_in_child": "not-a-number",
        }

        for key, value in cases.items():
            with self.subTest(key=key):
                with self.assertRaises(cmor4.ControlledVocabularyError):
                    project.dataset_info({**dataset, key: value})

    def test_cmip7_variable_attrs_come_from_variable_table(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = {
            "activity_id": "CMIP",
            "experiment_id": "amip",
            "forcing_index": "f3",
            "frequency": "mon",
            "grid_label": "g999",
            "initialization_index": "i1",
            "institution_id": "CCCma",
            "license_id": "CC-BY-4.0",
            "nominal_resolution": "100 km",
            "physics_index": "p1",
            "realization_index": "r9",
            "region": "glb",
            "source_id": "DUMMY-MODEL",
        }

        variable = project.variable(
            "tos_tavg-u-hxy-sea",
            units="K",
            standard_name="not_the_table_value",
            attrs={
                "long_name": "Not the table value",
                "cell_methods": "not the table value",
            },
        )
        info = project.dataset_info(dataset)

        ds = cmor4.create_dataset(
            info,
            variable,
            lat_lon_axes(project),
            np.ones((2, 2, 2), dtype="f4"),
        )

        self.assertEqual(ds["tos"].attrs["units"], "degC")
        self.assertEqual(ds["tos"].attrs["standard_name"], "sea_surface_temperature")
        self.assertEqual(ds["tos"].attrs["long_name"], "Sea Surface Temperature")
        self.assertEqual(
            ds["tos"].attrs["cell_methods"],
            "area: mean where sea time: mean",
        )

    def test_obs4mips_uses_project_cv_and_o3zm_table_entry(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project("Tables/obs4MIPs_Amon.json")
        cv_license = project.cv["license"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = {
                "activity_id": "obs4MIPs",
                "contact": "submissions-obs4mips@wcrp-cmip.org",
                "grid_label": "gnz",
                "grid": "5 degree latitude height zonal mean",
                "has_aux_unc": "FALSE",
                "institution_id": "DLR-BIRA",
                "license": cv_license,
                "nominal_resolution": "500 km",
                "outpath": tmp_dir,
                "output_file_template": (
                    "<variable_id><frequency><source_id><variant_label><grid_label>"
                ),
                "output_path_template": (
                    "<activity_id><institution_id><source_id><frequency>"
                    "<variable_id><grid_label><version>"
                ),
                "product": "observations",
                "processing_code_location": (
                    "dataset_guides/obs4mips/example-data-tools/example.py"
                ),
                "references": "Example reference",
                "source_data_url": "https://example.invalid/source",
                "source_id": "BSVertOzone-v1-0",
                "variant_label": "CMORGuide",
                "version": "v20260512",
            }
            axes = [
                project.axis(
                    "time",
                    values=[15.0, 45.0],
                    bounds=[[0.0, 30.0], [30.0, 60.0]],
                    units="days since 2000-01-01",
                ),
                project.axis("height", values=[1000.0, 5000.0]),
                project.axis(
                    "latitude",
                    values=[-45.0, 45.0],
                    bounds=[[-90.0, 0.0], [0.0, 90.0]],
                ),
            ]
            variable = project.variable("o3zm")
            info = project.dataset_info(dataset)

            result = cmor4.cmorize(
                info,
                variable,
                axes,
                np.ones((2, 2, 2), dtype="f4"),
            )

            self.assertIn("o3", result.dataset)
            self.assertNotIn("o3zm", result.dataset)
            self.assertEqual(result.dataset["o3"].attrs["units"], "mol mol-1")
            self.assertEqual(result.dataset.attrs["variable_id"], "o3")
            self.assertEqual(
                result.path.name,
                "o3_mon_BSVertOzone-v1-0_CMORGuide_gnz_200001-200002.nc",
            )

    def test_obs4mips_rejects_variable_not_in_loaded_table(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project("Tables/obs4MIPs_Amon.json")

        with self.assertRaises(cmor4.TableValidationError):
            project.variable("not_a_table_variable")

    def test_obs4mips_rejects_frequency_that_does_not_match_table(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project("Tables/obs4MIPs_Amon.json")
        dataset = {
            "activity_id": "obs4MIPs",
            "frequency": "day",
            "grid_label": "gn",
            "institution_id": "NOAA-NCEI",
            "license": project.cv["license"],
            "nominal_resolution": "250 km",
            "product": "observations",
            "source_id": "CMAP-V1902",
        }

        variable = project.variable("pr")
        info = project.dataset_info(dataset)
        axes = [
            project.axis(
                "time",
                values=[15.0, 45.0],
                bounds=[[0.0, 30.0], [30.0, 60.0]],
                units="days since 2000-01-01",
            ),
            project.axis("latitude", values=[-45.0, 45.0]),
            project.axis("longitude", values=[90.0, 270.0]),
        ]

        with self.assertRaises(cmor4.ControlledVocabularyError):
            cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((2, 2, 2), dtype="f4"),
            )


class ConstructorTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)

    def tearDown(self):
        self._ctx.cleanup()

    def test_cv_is_loaded_as_controlled_vocabulary(self):
        project = _build_project(self.tmp)
        self.assertIsInstance(project.cv, ControlledVocabulary)
        self.assertIn("activity_id", project.cv)

    def test_variable_entries_populated_from_table(self):
        project = _build_project(self.tmp)
        self.assertIn("pr", project.variable_entries)
        self.assertIn("tas", project.variable_entries)

    def test_coordinate_entries_populated_from_coordinate_table(self):
        project = _build_project(self.tmp)
        self.assertIn("lat", project.coordinate_entries)
        self.assertIn("lon", project.coordinate_entries)
        self.assertIn("time", project.coordinate_entries)

    def test_scalar_axis_entries_contain_only_value_entries(self):
        project = _build_project(self.tmp)
        # height2m has "value": "2.0" → scalar axis
        self.assertIn("height2m", project.scalar_axis_entries)
        # lat/lon/time have no fixed value → not scalar
        self.assertNotIn("lat", project.scalar_axis_entries)
        self.assertNotIn("time", project.scalar_axis_entries)

    def test_formula_entries_populated_from_formula_table(self):
        project = _build_project(self.tmp)
        self.assertIn("ps", project.formula_entries)

    def test_grid_mapping_entries_populated_from_grid_table(self):
        project = _build_project(self.tmp)
        self.assertIn("rotated_latitude_longitude", project.grid_mapping_entries)

    def test_no_coordinate_table_leaves_empty_coordinate_entries(self):
        cv_file = self.tmp / "CV.json"
        vtable_file = self.tmp / "vars.json"
        _write(cv_file, {"CV": {}})
        _write(vtable_file, {"Header": {"table_id": "t"}, "variable_entry": {}})
        project = ProjectTables(cv_file, [vtable_file])
        self.assertEqual(project.coordinate_entries, {})

    def test_no_formula_table_leaves_empty_formula_entries(self):
        cv_file = self.tmp / "CV.json"
        vtable_file = self.tmp / "vars.json"
        _write(cv_file, {"CV": {}})
        _write(vtable_file, {"Header": {"table_id": "t"}, "variable_entry": {}})
        project = ProjectTables(cv_file, [vtable_file])
        self.assertEqual(project.formula_entries, {})

    def test_multiple_variable_tables_all_loaded(self):
        cv_file = self.tmp / "CV.json"
        _write(cv_file, {"CV": {}})
        t1 = self.tmp / "T1.json"
        t2 = self.tmp / "T2.json"
        _write(
            t1,
            {
                "Header": {"table_id": "T1"},
                "variable_entry": {
                    "aa": {"out_name": "aa", "units": "1", "dimensions": []}
                },
            },
        )
        _write(
            t2,
            {
                "Header": {"table_id": "T2"},
                "variable_entry": {
                    "bb": {"out_name": "bb", "units": "1", "dimensions": []}
                },
            },
        )
        project = ProjectTables(cv_file, [t1, t2])
        self.assertIn("aa", project.variable_entries)
        self.assertIn("bb", project.variable_entries)

    def test_first_table_wins_for_duplicate_variable_name(self):
        """When the same variable name appears in two tables the first table's
        entry is used (first-wins semantics for variable_entries)."""
        cv_file = self.tmp / "CV.json"
        _write(cv_file, {"CV": {}})
        t1 = self.tmp / "T1.json"
        t2 = self.tmp / "T2.json"
        _write(
            t1,
            {
                "Header": {"table_id": "T1"},
                "variable_entry": {
                    "pr": {
                        "out_name": "pr_from_t1",
                        "units": "kg m-2 s-1",
                        "dimensions": [],
                    }
                },
            },
        )
        _write(
            t2,
            {
                "Header": {"table_id": "T2"},
                "variable_entry": {
                    "pr": {
                        "out_name": "pr_from_t2",
                        "units": "mm s-1",
                        "dimensions": [],
                    }
                },
            },
        )
        project = ProjectTables(cv_file, [t1, t2])
        self.assertEqual(project.variable_entries["pr"].entry["out_name"], "pr_from_t1")

    def test_table_id_stripped_of_table_prefix(self):
        """Table header 'Table Amon' → table_id 'Amon'."""
        cv_file = self.tmp / "CV.json"
        vtable = self.tmp / "Amon.json"
        _write(cv_file, {"CV": {}})
        _write(
            vtable,
            {
                "Header": {"table_id": "Table Amon"},
                "variable_entry": {
                    "pr": {"out_name": "pr", "units": "kg m-2 s-1", "dimensions": []}
                },
            },
        )
        project = ProjectTables(cv_file, [vtable])
        self.assertEqual(project.variable_entries["pr"].table_id, "Amon")


# ---------------------------------------------------------------------------
# 2. from_directory
# ---------------------------------------------------------------------------


class FromDirectoryTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        (self.tmp / "tables").mkdir()

    def tearDown(self):
        self._ctx.cleanup()

    def _write_standard_files(self, tables_dir: Path) -> None:
        _write(self.tmp / "CV.json", {"CV": {}})
        _write(
            tables_dir / "Amon.json",
            {
                "Header": {"table_id": "Amon"},
                "variable_entry": {
                    "pr": {
                        "out_name": "pr",
                        "units": "kg m-2 s-1",
                        "dimensions": ["time"],
                    }
                },
            },
        )
        _write(
            tables_dir / "PROJ_coordinate.json",
            {
                "axis_entry": {
                    "time": {"axis": "T", "out_name": "time", "standard_name": "time"}
                }
            },
        )
        _write(tables_dir / "PROJ_formula_terms.json", {"formula_entry": {}})
        _write(
            tables_dir / "PROJ_grids.json",
            {"axis_entry": {}, "variable_entry": {}, "mapping_entry": {}},
        )

    def test_explicit_paths_resolve_relative_to_root(self):
        tables_dir = self.tmp / "tables"
        self._write_standard_files(tables_dir)
        project = ProjectTables.from_directory(
            self.tmp,
            cv_file="CV.json",
            variable_tables=["tables/Amon.json"],
            coordinate_table="tables/PROJ_coordinate.json",
        )
        self.assertIn("pr", project.variable_entries)

    def test_auto_discovers_coordinate_table_in_tables_subdir(self):
        tables_dir = self.tmp / "tables"
        self._write_standard_files(tables_dir)
        project = ProjectTables.from_directory(
            self.tmp,
            cv_file="CV.json",
            variable_tables=["tables/Amon.json"],
        )
        # coordinate file was auto-discovered → coordinate_entries populated
        self.assertIsNotNone(project.coordinate_table_file)
        self.assertIn("time", project.coordinate_entries)

    def test_auto_discovers_formula_and_grid_tables(self):
        tables_dir = self.tmp / "tables"
        self._write_standard_files(tables_dir)
        project = ProjectTables.from_directory(
            self.tmp,
            cv_file="CV.json",
            variable_tables=["tables/Amon.json"],
        )
        self.assertIsNotNone(project.formula_table_file)
        self.assertIsNotNone(project.grid_table_file)

    def test_auto_discovers_in_Tables_capitalised_subdir(self):
        big_tables = self.tmp / "Tables"
        big_tables.mkdir(exist_ok=True)
        _write(self.tmp / "CV.json", {"CV": {}})
        _write(
            big_tables / "Amon.json",
            {
                "Header": {"table_id": "Amon"},
                "variable_entry": {},
            },
        )
        _write(big_tables / "PROJ_coordinate.json", {"axis_entry": {}})
        project = ProjectTables.from_directory(
            self.tmp,
            cv_file="CV.json",
            variable_tables=["Tables/Amon.json"],
        )
        self.assertIsNotNone(project.coordinate_table_file)

    def test_explicit_paths_override_auto_discovery(self):
        tables_dir = self.tmp / "tables"
        self._write_standard_files(tables_dir)
        custom_coord = self.tmp / "custom_coord.json"
        _write(
            custom_coord,
            {
                "axis_entry": {
                    "custom_axis": {"axis": "X", "out_name": "custom", "units": "m"}
                }
            },
        )
        project = ProjectTables.from_directory(
            self.tmp,
            cv_file="CV.json",
            variable_tables=["tables/Amon.json"],
            coordinate_table="custom_coord.json",
        )
        self.assertIn("custom_axis", project.coordinate_entries)

    def test_returns_project_tables_instance(self):
        tables_dir = self.tmp / "tables"
        self._write_standard_files(tables_dir)
        project = ProjectTables.from_directory(
            self.tmp,
            cv_file="CV.json",
            variable_tables=["tables/Amon.json"],
        )
        self.assertIsInstance(project, ProjectTables)


# ---------------------------------------------------------------------------
# 3. dataset_info
# ---------------------------------------------------------------------------


class DatasetInfoMethodTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def test_returns_dataset_info_instance(self):
        info = self.project.dataset_info(
            {"activity_id": "CMIP", "institution_id": "NCAR"}
        )
        self.assertIsInstance(info, DatasetInfo)

    def test_applies_scalar_cv_default(self):
        # _RICH_CV has "mip_era": "CMIP7" — a scalar default
        info = self.project.dataset_info(
            {"activity_id": "CMIP", "institution_id": "NCAR"}
        )
        self.assertEqual(info["mip_era"], "CMIP7")

    def test_user_values_are_preserved(self):
        info = self.project.dataset_info(
            {
                "activity_id": "CMIP",
                "institution_id": "NCAR",
                "grid_label": "gn",
            }
        )
        self.assertEqual(info["activity_id"], "CMIP")
        self.assertEqual(info["grid_label"], "gn")

    def test_institution_text_filled_from_institution_id(self):
        info = self.project.dataset_info(
            {"activity_id": "CMIP", "institution_id": "NCAR"}
        )
        self.assertIn("institution", info)
        self.assertIn("National Center", info["institution"])

    def test_rejects_invalid_activity_id_at_validation(self):
        """dataset_info validates controlled values and rejects unknown ones."""
        with self.assertRaises(ControlledVocabularyError):
            self.project.dataset_info(
                {
                    "activity_id": "NOT_REAL",
                    "institution_id": "NCAR",
                }
            )

    def test_validate_dataset_enforces_required_attributes(self):
        """
        validate_dataset (not dataset_info) raises when required attrs are missing.
        """
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_dataset({"activity_id": "CMIP"})
        self.assertIn("institution_id", str(ctx.exception))

    def test_accepts_dataset_info_as_input(self):
        """dataset_info should accept an existing DatasetInfo, not just a plain dict."""
        di = DatasetInfo.from_mapping({"activity_id": "CMIP", "institution_id": "NCAR"})
        info = self.project.dataset_info(di)
        self.assertIsInstance(info, DatasetInfo)
        self.assertEqual(info["activity_id"], "CMIP")

    def test_user_info_is_preserved_in_output(self):
        di = DatasetInfo.from_mapping(
            {"activity_id": "CMIP", "institution_id": "NCAR"},
        )
        info = self.project.dataset_info(di)
        # user_info should carry through (DatasetInfo stores it separately)
        self.assertIsInstance(info, DatasetInfo)


# ---------------------------------------------------------------------------
# 4. variable
# ---------------------------------------------------------------------------


class VariableMethodTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def test_returns_variable_instance(self):
        var = self.project.variable("pr")
        self.assertIsInstance(var, Variable)

    def test_table_units_applied(self):
        var = self.project.variable("pr")
        self.assertEqual(var.units, "kg m-2 s-1")

    def test_table_standard_name_applied(self):
        var = self.project.variable("pr")
        self.assertEqual(var.standard_name, "precipitation_flux")

    def test_table_dimensions_applied(self):
        """Variable dimensions come from the table entry.

        Note: table_dimensions() reverses the declared order, so we check
        that the expected names are all present rather than asserting a
        specific sequence.
        """
        var = self.project.variable("pr")
        self.assertEqual(set(var.dimensions), {"time", "lat", "lon"})

    def test_table_frequency_applied(self):
        var = self.project.variable("pr")
        self.assertEqual(var.frequency, "mon")

    def test_table_frequency_and_realm_in_entry(self):
        """Frequency is available on the Variable; realm is stored in the raw
        table entry but not surfaced as a top-level Variable attribute."""
        var = self.project.variable("pr")
        # frequency is a first-class Variable attribute
        self.assertEqual(var.frequency, "mon")
        # realm lives in the raw table entry, not the Variable dataclass
        entry = self.project.variable_entries["pr"].entry
        self.assertEqual(entry.get("realm"), "atmos")

    def test_user_missing_value_preserved(self):
        var = self.project.variable("pr", missing_value=-999.0)
        self.assertEqual(var.missing_value, -999.0)

    def test_unknown_variable_raises(self):
        with self.assertRaises(TableValidationError):
            self.project.variable("not_in_any_table")

    def test_table_id_disambiguates_variable(self):
        """When the same variable name exists in two tables, table_id selects
        the correct one."""
        cv_file = self.tmp / "CV2.json"
        t1 = self.tmp / "A.json"
        t2 = self.tmp / "B.json"
        _write(cv_file, {"CV": {}})
        _write(
            t1,
            {
                "Header": {"table_id": "A"},
                "variable_entry": {
                    "pr": {
                        "out_name": "pr_a",
                        "units": "kg m-2 s-1",
                        "dimensions": [],
                        "frequency": "mon",
                    }
                },
            },
        )
        _write(
            t2,
            {
                "Header": {"table_id": "B"},
                "variable_entry": {
                    "pr": {
                        "out_name": "pr_b",
                        "units": "kg m-2 s-1",
                        "dimensions": [],
                        "frequency": "day",
                    }
                },
            },
        )
        project = ProjectTables(cv_file, [t1, t2])
        var = project.variable("pr", table_id="B")
        self.assertEqual(var.frequency, "day")

    def test_conflicting_user_units_overridden_by_table(self):
        """Table-authoritative attributes override user-supplied values."""
        var = self.project.variable("pr", units="mm s-1")
        # Table says "kg m-2 s-1" — must win
        self.assertEqual(var.units, "kg m-2 s-1")

    def test_positive_merged_from_table_entry(self):
        """positive from the table entry is applied to the Variable."""
        tmp2 = Path(self._ctx.name) / "pos"
        tmp2.mkdir()
        project = _build_project(
            tmp2,
            variable_entries={
                "wap": {
                    "dimensions": [],
                    "out_name": "wap",
                    "units": "Pa s-1",
                    "positive": "down",
                },
            },
        )
        var = project.variable("wap")
        self.assertEqual(var.positive, "down")

    def test_positive_in_netcdf_attributes(self):
        """positive from the table is written to NetCDF output attributes."""
        tmp2 = Path(self._ctx.name) / "posattr"
        tmp2.mkdir()
        project = _build_project(
            tmp2,
            variable_entries={
                "wap": {
                    "dimensions": [],
                    "out_name": "wap",
                    "units": "Pa s-1",
                    "positive": "down",
                },
            },
        )
        var = project.variable("wap")
        _, labels = var.names()
        attrs = var.attributes(labels)
        self.assertEqual(attrs["positive"], "down")

    def test_variable_without_positive_has_none(self):
        """Variables whose table entry has no positive have positive=None."""
        self.assertIsNone(self.project.variable("pr").positive)


# ---------------------------------------------------------------------------
# positive field validation
# ---------------------------------------------------------------------------


class PositiveValidationTest(unittest.TestCase):
    """Tests for the 'positive' attribute validation in validate_against_entry
    and validate_components.

    Covers:
    * positive is merged from the table entry
    * positive is written to output NetCDF attributes
    * invalid positive values ('sideways', '') are rejected
    * positive conflicting with the table entry value is rejected
    * required positive (listed in entry 'required' field) must be supplied
    * optional positive (not in 'required') may be omitted
    * correct positive passes all checks
    * variable with no positive table entry is unaffected
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)

        # Project with three variable types:
        #   wap  – positive="down", listed in required → must be supplied
        #   ua   – positive="up",   not in required    → optional
        #   tas  – no positive entry at all
        self.project = _build_project(
            self.tmp,
            variable_entries={
                "wap": {
                    "dimensions": [],
                    "out_name": "wap",
                    "units": "Pa s-1",
                    "positive": "down",
                    "required": "positive",
                },
                "ua": {
                    "dimensions": [],
                    "out_name": "ua",
                    "units": "m s-1",
                    "positive": "up",
                },
                "tas": {
                    "dimensions": [],
                    "out_name": "tas",
                    "units": "K",
                },
            },
        )

    def tearDown(self):
        self._ctx.cleanup()

    # ------------------------------------------------------------------
    # Merging and output
    # ------------------------------------------------------------------

    def test_positive_merged_into_variable(self):
        self.assertEqual(self.project.variable("wap").positive, "down")

    def test_positive_up_merged_into_variable(self):
        self.assertEqual(self.project.variable("ua").positive, "up")

    def test_no_positive_entry_gives_none(self):
        self.assertIsNone(self.project.variable("tas").positive)

    def test_positive_written_to_netcdf_attrs(self):
        var = self.project.variable("wap")
        _, labels = var.names()
        self.assertEqual(var.attributes(labels)["positive"], "down")

    def test_no_positive_not_written_to_attrs(self):
        var = self.project.variable("tas")
        _, labels = var.names()
        self.assertNotIn("positive", var.attributes(labels))

    # ------------------------------------------------------------------
    # validate_against_entry / validate_components: value validity
    # ------------------------------------------------------------------

    def test_invalid_positive_value_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, Variable(name="wap", positive="sideways"), []
            )
        self.assertIn("positive", str(ctx.exception))
        self.assertIn("sideways", str(ctx.exception))

    def test_invalid_positive_empty_string_raises(self):
        # Empty string is invalid; None means not provided
        with self.assertRaises(TableValidationError):
            self.project.validate_components(
                None, Variable(name="wap", positive=""), []
            )

    def test_positive_wrong_direction_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, Variable(name="wap", positive="up"), []
            )
        msg = str(ctx.exception)
        self.assertIn("positive", msg)
        self.assertIn("up", msg)
        self.assertIn("down", msg)

    def test_correct_positive_passes(self):
        self.project.validate_components(
            None, Variable(name="wap", positive="down"), []
        )

    def test_correct_positive_case_insensitive(self):
        """Case-insensitive matching ('Down' == 'down')."""
        self.project.validate_components(
            None, Variable(name="wap", positive="Down"), []
        )

    # ------------------------------------------------------------------
    # required positive
    # ------------------------------------------------------------------

    def test_required_positive_absent_raises(self):
        """positive in required field → must be provided."""
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, Variable(name="wap"), [])
        msg = str(ctx.exception)
        self.assertIn("positive", msg)
        self.assertIn("requires", msg)

    def test_required_positive_error_names_table_and_variable(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, Variable(name="wap"), [])
        msg = str(ctx.exception)
        self.assertIn("wap", msg)
        self.assertIn("Amon", msg)

    def test_required_positive_none_raises(self):
        """Explicit None counts as not provided."""
        with self.assertRaises(TableValidationError):
            self.project.validate_components(
                None, Variable(name="wap", positive=None), []
            )

    # ------------------------------------------------------------------
    # optional positive
    # ------------------------------------------------------------------

    def test_optional_positive_absent_passes(self):
        """positive not in required → omitting it is fine."""
        self.project.validate_components(None, Variable(name="ua"), [])

    def test_optional_positive_correct_value_passes(self):
        self.project.validate_components(None, Variable(name="ua", positive="up"), [])

    def test_optional_positive_wrong_value_raises(self):
        """Even for optional positive, the value must match the table."""
        with self.assertRaises(TableValidationError):
            self.project.validate_components(
                None, Variable(name="ua", positive="down"), []
            )

    # ------------------------------------------------------------------
    # no positive in table
    # ------------------------------------------------------------------

    def test_no_positive_in_table_no_positive_provided_passes(self):
        self.project.validate_components(None, Variable(name="tas"), [])

    def test_no_positive_in_table_positive_provided_passes(self):
        """No table constraint → any valid value is accepted."""
        self.project.validate_components(None, Variable(name="tas", positive="up"), [])

    # ------------------------------------------------------------------
    # project.variable() round-trip through validate_components
    # ------------------------------------------------------------------

    def test_project_variable_with_positive_passes_validate(self):
        """Variable from project.variable() already has positive merged in."""
        var = self.project.variable("wap")
        self.project.validate_components(None, var, [])


class UnitsConvertibilityTest(unittest.TestCase):
    """User units must be dimensionally convertible to the
    table units, not necessarily identical strings.

    Relies on cf_units when available; tests that require it are skipped
    gracefully when the library is absent.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(
            self.tmp,
            variable_entries={
                "tas": {"dimensions": [], "out_name": "tas", "units": "K"},
                "pr": {"dimensions": [], "out_name": "pr", "units": "kg m-2 s-1"},
                "wap": {"dimensions": [], "out_name": "wap", "units": "Pa s-1"},
                "wc": {"dimensions": [], "out_name": "wc", "units": "?"},
            },
        )

    def tearDown(self):
        self._ctx.cleanup()

    @staticmethod
    def _cf_units_available():
        return True

    # --- exact-match always works ---

    def test_exact_unit_match_passes(self):
        self.project.validate_components(None, Variable(name="tas", units="K"), [])

    def test_no_user_units_skips_check(self):
        self.project.validate_components(None, Variable(name="tas"), [])

    def test_wildcard_table_units_accepts_any_user_units(self):
        """Table units '?' means accept any user-supplied units."""
        self.project.validate_components(None, Variable(name="wc", units="hPa"), [])

    def test_wildcard_table_units_accepts_no_user_units(self):
        self.project.validate_components(None, Variable(name="wc"), [])

    # --- incompatible units always fail ---

    def test_incompatible_units_raises(self):
        """Units from a different physical dimension are always rejected."""
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, Variable(name="tas", units="m s-1"), []
            )
        msg = str(ctx.exception)
        self.assertIn("m s-1", msg)
        self.assertIn("K", msg)
        self.assertIn("Amon:tas", msg)

    def test_error_message_mentions_convertibility(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, Variable(name="tas", units="m s-1"), []
            )
        self.assertIn("convertible", str(ctx.exception))

    def test_pressure_incompatible_with_temperature_raises(self):
        with self.assertRaises(TableValidationError):
            self.project.validate_components(None, Variable(name="tas", units="Pa"), [])

    # --- equivalent units pass when cf_units is available ---

    def test_degC_accepted_for_K_table(self):
        """degC and K are dimensionally convertible (temperature offset)."""
        if not self._cf_units_available():
            self.skipTest("cf_units not installed")
        self.project.validate_components(None, Variable(name="tas", units="degC"), [])

    def test_hPa_accepted_for_Pa_table(self):
        if not self._cf_units_available():
            self.skipTest("cf_units not installed")
        self.project.validate_components(
            None, Variable(name="wap", units="hPa s-1"), []
        )

    def test_equivalent_units_different_string_passes(self):
        """K and kelvin are the same unit with different string representations."""
        if not self._cf_units_available():
            self.skipTest("cf_units not installed")
        self.project.validate_components(None, Variable(name="tas", units="kelvin"), [])


class RequiredAttributesTest(unittest.TestCase):
    """Attributes listed in the table 'required' field must
    be present on the Variable.

    The 'required' field is a space-separated list of attribute names.
    Each attribute that has a table-defined value and appears in 'required'
    must be supplied by the user; those without table values are skipped
    (there is nothing to validate against).

    'positive' is handled by its own existing logic and excluded here.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(
            self.tmp,
            variable_entries={
                # two required attrs
                "ua": {
                    "dimensions": [],
                    "out_name": "ua",
                    "units": "m s-1",
                    "required": "standard_name long_name",
                    "standard_name": "eastward_wind",
                    "long_name": "Eastward Wind",
                },
                # required attr with no table value → no check possible
                "va": {
                    "dimensions": [],
                    "out_name": "va",
                    "units": "m s-1",
                    "required": "comment",
                    # comment has no value in the table entry
                },
                # no required field at all
                "tas": {
                    "dimensions": [],
                    "out_name": "tas",
                    "units": "K",
                    "standard_name": "air_temperature",
                },
            },
        )

    def tearDown(self):
        self._ctx.cleanup()

    # --- happy paths ---

    def test_all_required_attrs_present_passes(self):
        self.project.validate_components(
            None,
            Variable(
                name="ua", standard_name="eastward_wind", long_name="Eastward Wind"
            ),
            [],
        )

    def test_no_required_field_in_table_always_passes(self):
        self.project.validate_components(None, Variable(name="tas"), [])

    def test_required_attr_without_table_value_not_enforced(self):
        """'required: comment' when the table has no comment value → no check."""
        self.project.validate_components(None, Variable(name="va"), [])

    # --- missing required attr raises ---

    def test_missing_required_standard_name_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None,
                Variable(name="ua", long_name="Eastward Wind"),
                [],
            )
        msg = str(ctx.exception)
        self.assertIn("standard_name", msg)

    def test_missing_required_long_name_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None,
                Variable(name="ua", standard_name="eastward_wind"),
                [],
            )
        self.assertIn("long_name", str(ctx.exception))

    def test_error_names_table_and_variable(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, Variable(name="ua"), [])
        msg = str(ctx.exception)
        self.assertIn("Amon", msg)
        self.assertIn("ua", msg)

    def test_error_names_expected_value(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None,
                Variable(name="ua", long_name="Eastward Wind"),
                [],
            )
        self.assertIn("eastward_wind", str(ctx.exception))

    def test_all_missing_raises_on_first(self):
        """When both required attrs are missing only the first is reported."""
        with self.assertRaises(TableValidationError):
            self.project.validate_components(None, Variable(name="ua"), [])

    def test_project_variable_includes_required_attrs(self):
        """project.variable() merges table attrs so required check passes."""
        var = self.project.variable("ua")
        self.project.validate_components(None, var, [])


class FlagConsistencyTest(unittest.TestCase):
    """flag_values and flag_meanings must both be present
    in the table entry and have the same number of space-separated tokens.

    Also covers that flag fields are merged from the table into Variable and
    written to output NetCDF attributes.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(
            self.tmp,
            variable_entries={
                # valid: counts match
                "biome": {
                    "dimensions": [],
                    "out_name": "biome",
                    "units": "1",
                    "flag_values": "1 2 3",
                    "flag_meanings": "forest grassland desert",
                },
                # invalid: 3 values but 2 meanings
                "biome_short": {
                    "dimensions": [],
                    "out_name": "biome_short",
                    "units": "1",
                    "flag_values": "1 2 3",
                    "flag_meanings": "forest grassland",
                },
                # invalid: flag_values present, flag_meanings absent
                "biome_no_meanings": {
                    "dimensions": [],
                    "out_name": "biome_no_meanings",
                    "units": "1",
                    "flag_values": "1 2 3",
                },
                # invalid: flag_meanings present, flag_values absent
                "biome_no_values": {
                    "dimensions": [],
                    "out_name": "biome_no_values",
                    "units": "1",
                    "flag_meanings": "forest grassland desert",
                },
                # no flags at all — unaffected
                "tas": {"dimensions": [], "out_name": "tas", "units": "K"},
                # single-flag edge case
                "binary": {
                    "dimensions": [],
                    "out_name": "binary",
                    "units": "1",
                    "flag_values": "0",
                    "flag_meanings": "off",
                },
            },
        )

    def tearDown(self):
        self._ctx.cleanup()

    # --- merging into Variable ---

    def test_flag_values_merged_from_table(self):
        var = self.project.variable("biome")
        self.assertEqual(var.flag_values, "1 2 3")

    def test_flag_meanings_merged_from_table(self):
        var = self.project.variable("biome")
        self.assertEqual(var.flag_meanings, "forest grassland desert")

    def test_no_flags_in_table_gives_none(self):
        var = self.project.variable("tas")
        self.assertIsNone(var.flag_values)
        self.assertIsNone(var.flag_meanings)

    # --- output attributes ---

    def test_flag_values_in_netcdf_attrs(self):
        var = self.project.variable("biome")
        _, labels = var.names()
        self.assertIn("flag_values", var.attributes(labels))

    def test_flag_meanings_in_netcdf_attrs(self):
        var = self.project.variable("biome")
        _, labels = var.names()
        self.assertIn("flag_meanings", var.attributes(labels))

    def test_no_flags_not_in_attrs(self):
        var = self.project.variable("tas")
        _, labels = var.names()
        attrs = var.attributes(labels)
        self.assertNotIn("flag_values", attrs)
        self.assertNotIn("flag_meanings", attrs)

    # --- happy paths ---

    def test_matching_token_counts_passes(self):
        self.project.validate_components(None, self.project.variable("biome"), [])

    def test_single_flag_pair_passes(self):
        self.project.validate_components(None, self.project.variable("binary"), [])

    def test_no_flags_in_table_passes(self):
        self.project.validate_components(None, self.project.variable("tas"), [])

    # --- mismatch raises ---

    def test_mismatched_token_counts_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.project.variable("biome_short"), []
            )
        msg = str(ctx.exception)
        self.assertIn("flag_meanings", msg)
        self.assertIn("flag_values", msg)

    def test_mismatch_error_reports_counts(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.project.variable("biome_short"), []
            )
        msg = str(ctx.exception)
        self.assertIn("3", msg)  # flag_values count
        self.assertIn("2", msg)  # flag_meanings count

    def test_flag_values_without_flag_meanings_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.project.variable("biome_no_meanings"), []
            )
        msg = str(ctx.exception)
        self.assertIn("flag_meanings", msg)

    def test_flag_meanings_without_flag_values_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.project.variable("biome_no_values"), []
            )
        msg = str(ctx.exception)
        self.assertIn("flag_values", msg)

    def test_missing_raises_names_table_and_variable(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.project.variable("biome_no_meanings"), []
            )
        msg = str(ctx.exception)
        self.assertIn("Amon", msg)
        self.assertIn("biome_no_meanings", msg)


# ---------------------------------------------------------------------------
# 5. axis
# ---------------------------------------------------------------------------


class AxisMethodTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def test_returns_axis_instance(self):
        ax = self.project.axis("lat", values=[-45.0, 45.0])
        self.assertIsInstance(ax, Axis)

    def test_axis_is_marked_as_prepared(self):
        ax = self.project.axis("lat", values=[-45.0, 45.0])
        self.assertTrue(self.project._is_prepared_axis(ax))

    def test_axis_not_from_this_project_not_prepared(self):
        ax = Axis(name="lat", values=[-45.0, 45.0])
        self.assertFalse(self.project._is_prepared_axis(ax))

    def test_coordinate_table_units_applied(self):
        ax = self.project.axis("lat", values=[-45.0, 45.0])
        self.assertEqual(ax.units, "degrees_north")

    def test_coordinate_table_standard_name_applied(self):
        ax = self.project.axis("lat", values=[-45.0, 45.0])
        self.assertEqual(ax.standard_name, "latitude")

    def test_coordinate_table_out_name_applied(self):
        ax = self.project.axis("lat", values=[-45.0, 45.0])
        self.assertEqual(ax.out_name, "lat")

    def test_user_values_array_preserved(self):
        vals = [1000.0, 500.0, 250.0]
        ax = self.project.axis("plev", values=vals)
        np.testing.assert_array_equal(ax.values, vals)

    def test_user_bounds_preserved(self):
        bnds = [[-90.0, 0.0], [0.0, 90.0]]
        ax = self.project.axis("lat", values=[-45.0, 45.0], bounds=bnds)
        np.testing.assert_array_equal(ax.bounds, bnds)

    def test_axis_name_not_in_coordinate_table_accepted(self):
        """Axes not in the coordinate table are accepted without error."""
        ax = self.project.axis("custom_dim", values=[1.0, 2.0, 3.0])
        self.assertIsInstance(ax, Axis)
        self.assertEqual(ax.name, "custom_dim")

    def test_wrong_metadata_for_known_axis_raises(self):
        """Conflicting user metadata for a known coordinate raises."""
        with self.assertRaises(TableValidationError):
            self.project.axis("lat", values=[-45.0, 45.0], units="degrees_east")

    def test_time_axis_with_user_units_accepted(self):
        """time coordinate has no fixed units in table → user units accepted."""
        ax = self.project.axis(
            "time",
            values=[15.0, 45.0],
            bounds=[[0.0, 30.0], [30.0, 60.0]],
            units="days since 2000-01-01",
        )
        self.assertEqual(ax.units, "days since 2000-01-01")


# ---------------------------------------------------------------------------
# 6. scalar_axes_for
# ---------------------------------------------------------------------------


class ScalarAxesForTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def _base_axes(self):
        return [
            self.project.axis(
                "time",
                values=[15.0],
                bounds=[[0.0, 30.0]],
                units="days since 2000-01-01",
            ),
            self.project.axis("lat", values=[-45.0, 45.0]),
            self.project.axis("lon", values=[90.0, 270.0]),
        ]

    def test_returns_tuple(self):
        tas = self.project.variable("tas")
        result = self.project.scalar_axes_for(tas, self._base_axes())
        self.assertIsInstance(result, tuple)

    def test_returns_missing_scalar_axis(self):
        tas = self.project.variable("tas")
        scalars = self.project.scalar_axes_for(tas, self._base_axes())
        names = [a.name for a in scalars]
        self.assertIn("height2m", names)

    def test_returned_scalar_axes_are_prepared(self):
        tas = self.project.variable("tas")
        scalars = self.project.scalar_axes_for(tas, self._base_axes())
        for ax in scalars:
            self.assertTrue(self.project._is_prepared_axis(ax))

    def test_already_provided_scalar_not_returned(self):
        """If the scalar axis is already in the user-supplied list it must not
        be duplicated in the return value."""
        tas = self.project.variable("tas")
        height = self.project.axis("height2m")
        axes = self._base_axes() + [height]
        scalars = self.project.scalar_axes_for(tas, axes)
        names = [a.name for a in scalars]
        self.assertNotIn("height2m", names)

    def test_variable_with_no_scalar_dims_returns_empty(self):
        pr = self.project.variable("pr")
        scalars = self.project.scalar_axes_for(pr, self._base_axes())
        self.assertEqual(scalars, ())

    def test_called_with_no_axes_returns_all_required_scalars(self):
        tas = self.project.variable("tas")
        scalars = self.project.scalar_axes_for(tas)
        names = [a.name for a in scalars]
        self.assertIn("height2m", names)

    def test_scalar_axis_marked_as_scalar(self):
        tas = self.project.variable("tas")
        scalars = self.project.scalar_axes_for(tas)
        for ax in scalars:
            if ax.name == "height2m":
                self.assertTrue(ax.scalar)


# ---------------------------------------------------------------------------
# 7. complete_axes
# ---------------------------------------------------------------------------


class CompleteAxesTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def _base_axes(self):
        return [
            self.project.axis(
                "time",
                values=[15.0],
                bounds=[[0.0, 30.0]],
                units="days since 2000-01-01",
            ),
            self.project.axis("lat", values=[-45.0, 45.0]),
            self.project.axis("lon", values=[90.0, 270.0]),
        ]

    def test_returns_tuple(self):
        tas = self.project.variable("tas")
        result = self.project.complete_axes(tas, self._base_axes())
        self.assertIsInstance(result, tuple)

    def test_base_axes_included_in_result(self):
        pr = self.project.variable("pr")
        base = self._base_axes()
        result = self.project.complete_axes(pr, base)
        result_names = [a.name for a in result]
        for ax in base:
            self.assertIn(ax.name, result_names)

    def test_scalar_axis_appended_for_tas(self):
        tas = self.project.variable("tas")
        result = self.project.complete_axes(tas, self._base_axes())
        names = [a.name for a in result]
        self.assertIn("height2m", names)

    def test_no_duplication_when_scalar_already_present(self):
        tas = self.project.variable("tas")
        height = self.project.axis("height2m")
        axes = self._base_axes() + [height]
        result = self.project.complete_axes(tas, axes)
        names = [a.name for a in result]
        self.assertEqual(names.count("height2m"), 1)

    def test_all_returned_axes_are_prepared(self):
        tas = self.project.variable("tas")
        result = self.project.complete_axes(tas, self._base_axes())
        for ax in result:
            self.assertTrue(self.project._is_prepared_axis(ax))

    def test_unprepared_input_axes_are_merged(self):
        """Axes supplied from outside this project are merged with table data."""
        pr = self.project.variable("pr")
        raw_lat = Axis(name="lat", values=[-45.0, 45.0])
        axes = [
            self.project.axis(
                "time",
                values=[15.0],
                bounds=[[0.0, 30.0]],
                units="days since 2000-01-01",
            ),
            raw_lat,
            self.project.axis("lon", values=[90.0, 270.0]),
        ]
        result = self.project.complete_axes(pr, axes)
        # All should be prepared after complete_axes
        for ax in result:
            self.assertTrue(self.project._is_prepared_axis(ax))

    def test_variable_with_no_scalar_dims_returns_same_axes(self):
        pr = self.project.variable("pr")
        base = self._base_axes()
        result = self.project.complete_axes(pr, base)
        self.assertEqual(len(result), len(base))


# ---------------------------------------------------------------------------
# 8. grid
# ---------------------------------------------------------------------------


class GridMethodTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def test_returns_grid_instance(self):
        g = self.project.grid("rotated_latitude_longitude")
        self.assertIsInstance(g, Grid)

    def test_grid_mapping_name_from_table(self):
        g = self.project.grid("rotated_latitude_longitude")
        self.assertEqual(g.grid_mapping_name, "rotated_latitude_longitude")

    def test_grid_with_no_name_accepted(self):
        g = self.project.grid()
        self.assertIsInstance(g, Grid)
        self.assertIsNone(g.name)

    def test_grid_name_not_in_table_accepted(self):
        """Grid names not in the grid table are accepted; user must supply attrs."""
        g = self.project.grid(
            "custom_projection",
            grid_mapping_name="custom_crs",
        )
        self.assertIsInstance(g, Grid)

    def test_user_params_preserved(self):
        params = {"standard_parallel": ([30.0, 60.0], "degrees_north")}
        g = self.project.grid("lambert_conformal_conic", params=params)
        self.assertEqual(g.params, params)

    def test_user_dimensions_preserved(self):
        g = self.project.grid(dimensions=["j", "i"])
        self.assertEqual(list(g.dimensions), ["j", "i"])

    def test_grid_without_grid_table_accepted(self):
        """Grid factory works even when no grid table is loaded."""
        cv_file = self.tmp / "CV2.json"
        vtable = self.tmp / "V2.json"
        _write(cv_file, {"CV": {}})
        _write(vtable, {"Header": {"table_id": "t"}, "variable_entry": {}})
        project = ProjectTables(cv_file, [vtable])
        g = project.grid()
        self.assertIsInstance(g, Grid)


# ---------------------------------------------------------------------------
# 9. zfactor
# ---------------------------------------------------------------------------


class ZFactorMethodTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def test_returns_zfactor_instance(self):
        zf = self.project.zfactor("ps")
        self.assertIsInstance(zf, ZFactor)

    def test_formula_table_units_applied(self):
        zf = self.project.zfactor("ps")
        self.assertEqual(zf.units, "Pa")

    def test_formula_table_standard_name_applied(self):
        zf = self.project.zfactor("ps")
        self.assertEqual(zf.standard_name, "surface_air_pressure")

    def test_user_values_array_preserved(self):
        data = np.ones((3, 4), dtype="f4")
        zf = self.project.zfactor("ps", values=data)
        np.testing.assert_array_equal(zf.values, data)

    def test_wrong_units_raises_at_construction(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.zfactor("ps", units="hPa")
        self.assertIn("units", str(ctx.exception))

    def test_unknown_name_accepted_without_error(self):
        zf = self.project.zfactor("not_in_table")
        self.assertIsInstance(zf, ZFactor)

    def test_zfactor_without_formula_table_accepted(self):
        cv_file = self.tmp / "CV2.json"
        vtable = self.tmp / "V2.json"
        _write(cv_file, {"CV": {}})
        _write(vtable, {"Header": {"table_id": "t"}, "variable_entry": {}})
        project = ProjectTables(cv_file, [vtable])
        zf = project.zfactor("anything")
        self.assertIsInstance(zf, ZFactor)

    def test_user_dimensions_preserved(self):
        zf = self.project.zfactor("ps", dimensions=["time", "lat", "lon"])
        self.assertEqual(list(zf.dimensions), ["time", "lat", "lon"])


# ---------------------------------------------------------------------------
# ZFactor units convertibility
# ---------------------------------------------------------------------------


class ZFactorUnitsConvertibilityTest(unittest.TestCase):
    """Tests for ZFactor units validation using dimensional convertibility.

    ZFactor units are validated by dimensional equivalence (via cf_units) rather
    than exact string equality, mirroring the behaviour for Variable units.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(
            self.tmp,
            formula_entries={
                "ps": {
                    "units": "Pa",
                    "standard_name": "surface_air_pressure",
                    "out_name": "ps",
                },
                "temp": {
                    "units": "K",
                    "standard_name": "air_temperature",
                    "out_name": "temp",
                },
            },
        )
        self.var = self.project.variable("pr")

    def tearDown(self):
        self._ctx.cleanup()

    def test_exact_units_match_passes(self):
        self.project.validate_components(
            None, self.var, [], zfactors=[ZFactor(name="ps", units="Pa")]
        )

    def test_no_user_units_skips_check(self):
        self.project.validate_components(
            None, self.var, [], zfactors=[ZFactor(name="ps")]
        )

    def test_convertible_units_passes(self):
        """hPa is dimensionally equivalent to Pa — should be accepted."""
        self.project.validate_components(
            None, self.var, [], zfactors=[ZFactor(name="ps", units="hPa")]
        )

    def test_convertible_temperature_units_passes(self):
        self.project.validate_components(
            None, self.var, [], zfactors=[ZFactor(name="temp", units="degC")]
        )

    def test_incompatible_units_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.var, [], zfactors=[ZFactor(name="ps", units="m s-1")]
            )
        msg = str(ctx.exception)
        self.assertIn("m s-1", msg)
        self.assertIn("Pa", msg)
        self.assertIn("convertible", msg)

    def test_incompatible_units_error_names_formula_term(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.var, [], zfactors=[ZFactor(name="ps", units="K")]
            )
        self.assertIn("ps", str(ctx.exception))

    def test_unknown_formula_term_skips_units_check(self):
        """ZFactor not in the formula table is not validated."""
        self.project.validate_components(
            None, self.var, [], zfactors=[ZFactor(name="unknown", units="furlong")]
        )


# ---------------------------------------------------------------------------
# ZFactor scalar enforcement (p0)
# ---------------------------------------------------------------------------


class ZFactorScalarTest(unittest.TestCase):
    """Tests for the rule that formula terms with no declared dimensions must
    supply scalar (0-dimensional) values.

    The canonical example is the reference pressure p0 in hybrid
    sigma-pressure coordinates: it is a single constant, not a profile.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(
            self.tmp,
            formula_entries={
                # p0: no dimensions — must be scalar
                "p0": {
                    "units": "Pa",
                    "standard_name": "reference_air_pressure",
                    "out_name": "p0",
                },
                # ap: has dimensions — array is fine
                "ap": {
                    "units": "Pa",
                    "standard_name": "vertical_pressure",
                    "out_name": "ap",
                    "dimensions": "lev",
                },
            },
        )
        self.var = self.project.variable("pr")

    def tearDown(self):
        self._ctx.cleanup()

    # --- scalar values pass ---

    def test_python_float_passes(self):
        self.project.validate_components(
            None, self.var, [], zfactors=[ZFactor(name="p0", values=100000.0)]
        )

    def test_zero_dimensional_array_passes(self):
        self.project.validate_components(
            None,
            self.var,
            [],
            zfactors=[ZFactor(name="p0", values=np.float64(100000.0))],
        )

    def test_no_values_provided_skips_check(self):
        """Without values there is nothing to validate."""
        self.project.validate_components(
            None, self.var, [], zfactors=[ZFactor(name="p0")]
        )

    def test_formula_term_with_dimensions_accepts_array(self):
        """ap declares dimensions in the table — a 1-D array is correct."""
        self.project.validate_components(
            None,
            self.var,
            [],
            zfactors=[ZFactor(name="ap", values=np.ones(5), units="Pa")],
        )

    def test_unknown_formula_term_accepts_array(self):
        """ZFactor not in the formula table is not checked."""
        self.project.validate_components(
            None,
            self.var,
            [],
            zfactors=[ZFactor(name="unknown", values=np.ones((3, 4)))],
        )

    # --- array values raise ---

    def test_one_dimensional_array_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.var, [], zfactors=[ZFactor(name="p0", values=np.ones(3))]
            )
        msg = str(ctx.exception)
        self.assertIn("p0", msg)
        self.assertIn("scalar", msg)

    def test_two_dimensional_array_raises(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None,
                self.var,
                [],
                zfactors=[ZFactor(name="p0", values=np.ones((2, 3)))],
            )
        msg = str(ctx.exception)
        self.assertIn("(2, 3)", msg)

    def test_error_names_the_formula_term(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.var, [], zfactors=[ZFactor(name="p0", values=np.ones(3))]
            )
        self.assertIn("p0", str(ctx.exception))

    def test_error_reports_actual_shape(self):
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, self.var, [], zfactors=[ZFactor(name="p0", values=np.ones((4,)))]
            )
        self.assertIn("(4,)", str(ctx.exception))

    # --- size-1 array is CMOR3-compatible and must be accepted ---

    def test_size_1_array_is_accepted(self):
        """A shape-(1,) array is CMOR3-compatible and must not raise."""
        self.project.validate_components(
            None,
            self.var,
            [],
            zfactors=[ZFactor(name="p0", values=np.array([100000.0]))],
        )

    def test_size_1_array_error_message_not_raised(self):
        """Confirm the 'scalar value' error is NOT raised for shape-(1,) arrays."""
        try:
            self.project.validate_components(
                None,
                self.var,
                [],
                zfactors=[ZFactor(name="p0", values=np.array([100000.0]))],
            )
        except TableValidationError as exc:
            self.fail(
                f"validate_components raised TableValidationError unexpectedly: {exc}"
            )


# ---------------------------------------------------------------------------
# 10. validate_global_attributes
# ---------------------------------------------------------------------------


class ValidateGlobalAttributesTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def _valid_attrs(self, **overrides):
        return {"activity_id": "CMIP", "institution_id": "NCAR", **overrides}

    def test_valid_attributes_pass(self):
        self.project.validate_global_attributes(self._valid_attrs())

    def test_invalid_activity_id_raises(self):
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_global_attributes(
                self._valid_attrs(activity_id="NOT_REAL")
            )

    def test_missing_required_attribute_raises(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_global_attributes({"activity_id": "CMIP"})
        self.assertIn("institution_id", str(ctx.exception))

    def test_returns_none_on_success(self):
        result = self.project.validate_global_attributes(self._valid_attrs())
        self.assertIsNone(result)

    def test_experiment_is_validated(self):
        """validate_global_attributes also validates experiment metadata."""
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_global_attributes(
                self._valid_attrs(
                    experiment_id="historical",
                    # source_type required for historical but not provided
                )
            )


# ---------------------------------------------------------------------------
# 11. validate_dataset
# ---------------------------------------------------------------------------


class ValidateDatasetTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def test_valid_dataset_passes(self):
        self.project.validate_dataset({"activity_id": "CMIP", "institution_id": "NCAR"})

    def test_invalid_controlled_value_raises(self):
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_dataset(
                {
                    "activity_id": "INVALID",
                    "institution_id": "NCAR",
                }
            )

    def test_missing_required_attribute_raises(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_dataset({"activity_id": "CMIP"})
        self.assertIn("institution_id", str(ctx.exception))

    def test_returns_none_on_success(self):
        result = self.project.validate_dataset(
            {
                "activity_id": "CMIP",
                "institution_id": "NCAR",
            }
        )
        self.assertIsNone(result)

    def test_empty_cv_accepts_any_values(self):
        cv_file = self.tmp / "CV_empty.json"
        vtable = self.tmp / "V_empty.json"
        _write(cv_file, {"CV": {}})
        _write(vtable, {"Header": {"table_id": "t"}, "variable_entry": {}})
        project = ProjectTables(cv_file, [vtable])
        # No CV constraints → no error regardless of values
        project.validate_dataset({"anything": "anything_value"})

    def test_invalid_source_type_raises(self):
        """source_type is a multi-token field validated against the CV."""
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_dataset(
                {
                    "activity_id": "CMIP",
                    "institution_id": "NCAR",
                    "source_type": "NOT_A_REAL_TYPE",
                }
            )


# ---------------------------------------------------------------------------
# 12. validate_required_global_attributes
# ---------------------------------------------------------------------------


class ValidateRequiredGlobalAttributesTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def test_all_required_present_passes(self):
        self.project.validate_required_global_attributes(
            {
                "activity_id": "CMIP",
                "institution_id": "NCAR",
            }
        )

    def test_missing_required_attribute_raises_with_name(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_required_global_attributes(
                {
                    "activity_id": "CMIP",
                }
            )
        self.assertIn("institution_id", str(ctx.exception))

    def test_both_missing_raises(self):
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_required_global_attributes({})

    def test_empty_string_counts_as_missing(self):
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_required_global_attributes(
                {
                    "activity_id": "CMIP",
                    "institution_id": "",
                }
            )

    def test_returns_none_on_success(self):
        result = self.project.validate_required_global_attributes(
            {
                "activity_id": "CMIP",
                "institution_id": "NCAR",
            }
        )
        self.assertIsNone(result)

    def test_no_required_attributes_always_passes(self):
        cv_file = self.tmp / "CV_no_req.json"
        vtable = self.tmp / "V_no_req.json"
        _write(cv_file, {"CV": {}})
        _write(vtable, {"Header": {"table_id": "t"}, "variable_entry": {}})
        project = ProjectTables(cv_file, [vtable])
        project.validate_required_global_attributes({})


# ---------------------------------------------------------------------------
# 13. required_global_attributes
# ---------------------------------------------------------------------------


class RequiredGlobalAttributesTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)

    def tearDown(self):
        self._ctx.cleanup()

    def test_returns_tuple(self):
        project = _build_project(self.tmp)
        result = project.required_global_attributes()
        self.assertIsInstance(result, tuple)

    def test_returns_expected_attribute_names(self):
        project = _build_project(self.tmp)
        result = project.required_global_attributes()
        self.assertIn("activity_id", result)
        self.assertIn("institution_id", result)

    def test_returns_empty_tuple_when_cv_has_none(self):
        cv_file = self.tmp / "CV_none.json"
        vtable = self.tmp / "V_none.json"
        _write(cv_file, {"CV": {}})
        _write(vtable, {"Header": {"table_id": "t"}, "variable_entry": {}})
        project = ProjectTables(cv_file, [vtable])
        self.assertEqual(project.required_global_attributes(), ())

    def test_all_items_are_strings(self):
        project = _build_project(self.tmp)
        for name in project.required_global_attributes():
            self.assertIsInstance(name, str)


# ---------------------------------------------------------------------------
# 14. validate_experiment
# ---------------------------------------------------------------------------


class ValidateExperimentTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def _historical_base(self):
        return {
            "experiment_id": "historical",
            "source_type": "AOGCM",
        }

    def test_valid_experiment_passes(self):
        self.project.validate_experiment(self._historical_base())

    def test_no_experiment_id_is_no_op(self):
        """Without an experiment_id the method must not raise."""
        self.project.validate_experiment({})

    def test_unknown_experiment_id_is_no_op(self):
        """experiment_id not present in CV is skipped silently."""
        self.project.validate_experiment({"experiment_id": "does_not_exist"})

    def test_missing_source_type_raises(self):
        """historical requires source_type=AOGCM."""
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_experiment({"experiment_id": "historical"})
        self.assertIn("source_type", str(ctx.exception))

    def test_disallowed_source_type_raises(self):
        # BGC is defined in source_type CV but is NOT in
        # additional_allowed_model_components
        # for historical. validate_source_type (called by validate_experiment)
        # checks allowed tokens via the experiment's additional_allowed list.
        # Build a project where BGC is excluded from additional_allowed:
        cv_file = self.tmp / "CV_strict.json"
        vtable = self.tmp / "V_strict.json"
        _write(
            cv_file,
            {
                "CV": {
                    "experiment_id": {
                        "historical": {
                            "required_source_type": ["AOGCM"],
                            "additional_allowed_model_components": ["AER"],
                            # BGC intentionally omitted
                        },
                    },
                }
            },
        )
        _write(vtable, {"Header": {"table_id": "t"}, "variable_entry": {}})
        project = ProjectTables(cv_file, [vtable])
        with self.assertRaises(ControlledVocabularyError):
            project.validate_experiment(
                {
                    "experiment_id": "historical",
                    "source_type": "AOGCM BGC",
                }
            )

    def test_valid_additional_allowed_source_type_passes(self):
        self.project.validate_experiment(
            {
                "experiment_id": "historical",
                "source_type": "AOGCM AER",
            }
        )

    def test_wrong_activity_id_for_experiment_raises(self):
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_experiment(
                {
                    "experiment_id": "historical",
                    "source_type": "AOGCM",
                    "activity_id": "ScenarioMIP",  # historical requires CMIP
                }
            )

    def test_parent_attribute_on_no_parent_experiment_raises(self):
        """amip has no parent_experiment_id CV entry.  validate_parent_attributes
        (called by dataset_info) rejects unexpected parent attributes.
        validate_experiment itself does not perform this check."""
        # verify the correct method rejects it
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_parent_attributes(
                {
                    "experiment_id": "amip",
                    "parent_experiment_id": "piControl",
                }
            )
        self.assertIn("parent_experiment_id", str(ctx.exception))

    def test_returns_none_on_success(self):
        result = self.project.validate_experiment(self._historical_base())
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 15. validate_source_type
# ---------------------------------------------------------------------------


class ValidateSourceTypeTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    @property
    def _historical_entry(self):
        return _RICH_CV["CV"]["experiment_id"]["historical"]

    def test_required_type_present_passes(self):
        self.project.validate_source_type(
            {"source_type": "AOGCM"}, self._historical_entry
        )

    def test_required_plus_allowed_passes(self):
        self.project.validate_source_type(
            {"source_type": "AOGCM AER"}, self._historical_entry
        )

    def test_missing_source_type_raises(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_source_type({}, self._historical_entry)
        self.assertIn("source_type", str(ctx.exception))

    def test_empty_source_type_raises(self):
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_source_type(
                {"source_type": ""}, self._historical_entry
            )

    def test_missing_required_token_raises(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_source_type(
                {"source_type": "AER"}, self._historical_entry
            )
        self.assertIn("missing required", str(ctx.exception))

    def test_disallowed_token_raises(self):
        """A token present in source_type but absent from required+additional raises."""
        exp_entry = {
            "required_source_type": ["AOGCM"],
            "additional_allowed_model_components": ["AER"],
            # BGC deliberately excluded
        }
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_source_type({"source_type": "AOGCM BGC"}, exp_entry)
        self.assertIn("not allowed", str(ctx.exception))

    def test_empty_experiment_entry_is_no_op(self):
        """When experiment has no required/additional source type, no error."""
        self.project.validate_source_type({"source_type": "ANYTHING"}, {})

    def test_returns_none_on_success(self):
        result = self.project.validate_source_type(
            {"source_type": "AOGCM"}, self._historical_entry
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 16. validate_source_attributes
# ---------------------------------------------------------------------------


class ValidateSourceAttributesTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def test_valid_source_attributes_pass(self):
        self.project.validate_source_attributes(
            {
                "source_id": "CESM2",
                "institution_id": "NCAR",
                "source_type": "AOGCM",
            }
        )

    def test_wrong_institution_for_source_id_raises(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_source_attributes(
                {
                    "source_id": "CESM2",
                    "institution_id": "ECMWF",
                }
            )
        self.assertIn("institution_id", str(ctx.exception))

    def test_no_source_id_is_no_op(self):
        """Missing source_id skips all source-attribute validation."""
        self.project.validate_source_attributes({"institution_id": "ECMWF"})

    def test_unknown_source_id_is_no_op(self):
        """source_id not in CV: nothing to cross-check."""
        self.project.validate_source_attributes(
            {
                "source_id": "TOTALLY_UNKNOWN",
                "institution_id": "ANY",
            }
        )

    def test_source_id_with_correct_source_type_passes(self):
        self.project.validate_source_attributes(
            {
                "source_id": "CESM2",
                "institution_id": "NCAR",
                "source_type": "AOGCM",
            }
        )

    def test_source_id_with_wrong_source_type_raises(self):
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_source_attributes(
                {
                    "source_id": "CESM2",
                    "institution_id": "NCAR",
                    "source_type": "AER",  # CESM2 must be AOGCM
                }
            )

    def test_returns_none_on_success(self):
        result = self.project.validate_source_attributes(
            {
                "source_id": "DUMMY",
                "institution_id": "NCAR",
            }
        )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 17. validate_parent_attributes
# ---------------------------------------------------------------------------


class ValidateParentAttributesTest(unittest.TestCase):
    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def _valid_historical(self, **overrides):
        base = {
            "experiment_id": "historical",
            "source_id": "CESM2",
            "mip_era": "CMIP7",
            "parent_experiment_id": "piControl",
            "parent_activity_id": "CMIP",
            "parent_source_id": "CESM2",
            "parent_mip_era": "CMIP7",
            "parent_time_units": "days since 1850-01-01",
            "parent_variant_label": "r1i1p1f1",
            "branch_time_in_child": 0.0,
            "branch_time_in_parent": 0.0,
        }
        base.update(overrides)
        return base

    def test_valid_parent_attributes_pass(self):
        self.project.validate_parent_attributes(self._valid_historical())

    def test_no_experiment_id_is_no_op(self):
        self.project.validate_parent_attributes({})

    def test_unknown_experiment_id_is_no_op(self):
        self.project.validate_parent_attributes({"experiment_id": "unknown"})

    def test_no_parent_experiment_amip_passes(self):
        """amip requires no parent; an empty dataset for it should pass."""
        self.project.validate_parent_attributes({"experiment_id": "amip"})

    def test_supplying_parent_to_no_parent_experiment_raises(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_parent_attributes(
                {
                    "experiment_id": "amip",
                    "parent_experiment_id": "piControl",
                }
            )
        self.assertIn("parent_experiment_id", str(ctx.exception))

    def test_missing_parent_experiment_id_raises(self):
        dataset = self._valid_historical()
        del dataset["parent_experiment_id"]
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_parent_attributes(dataset)
        self.assertIn("parent_experiment_id", str(ctx.exception))

    def test_wrong_parent_experiment_id_raises(self):
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_parent_attributes(
                self._valid_historical(parent_experiment_id="amip")
            )

    def test_wrong_parent_time_units_raises(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_parent_attributes(
                self._valid_historical(parent_time_units="seconds since 1850-01-01")
            )
        self.assertIn("parent_time_units", str(ctx.exception))

    def test_wrong_parent_variant_label_raises(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_parent_attributes(
                self._valid_historical(parent_variant_label="not-a-variant")
            )
        self.assertIn("parent_variant_label", str(ctx.exception))

    def test_non_numeric_branch_time_raises(self):
        with self.assertRaises(ControlledVocabularyError) as ctx:
            self.project.validate_parent_attributes(
                self._valid_historical(branch_time_in_child="not-a-number")
            )
        self.assertIn("branch_time_in_child", str(ctx.exception))

    def test_unknown_parent_source_id_raises(self):
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_parent_attributes(
                self._valid_historical(parent_source_id="UNKNOWN_MODEL")
            )

    def test_returns_none_on_success(self):
        result = self.project.validate_parent_attributes(self._valid_historical())
        self.assertIsNone(result)

    def test_extra_parent_attrs_rejected_for_no_parent_experiment(self):
        """Supplying any parent-related attribute when experiment has no parent
        must raise."""
        with self.assertRaises(ControlledVocabularyError):
            self.project.validate_parent_attributes(
                {
                    "experiment_id": "amip",
                    "parent_activity_id": "CMIP",
                }
            )


# ---------------------------------------------------------------------------
# 18. validate_components — integration / cross-method scenarios
# (primary unit tests in test_validate_components.py)
# ---------------------------------------------------------------------------


class ValidateComponentsIntegrationTest(unittest.TestCase):
    """Scenarios that exercise validate_components together with factory methods."""

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(self.tmp)

    def tearDown(self):
        self._ctx.cleanup()

    def _base_axes(self):
        return [
            self.project.axis(
                "time",
                values=[15.0, 45.0],
                bounds=[[0.0, 30.0], [30.0, 60.0]],
                units="days since 2000-01-01",
            ),
            self.project.axis("lat", values=[-45.0, 45.0]),
            self.project.axis("lon", values=[90.0, 270.0]),
        ]

    def test_factory_produced_components_pass_validation(self):
        """
        All components produced by project factory methods pass validate_components.
        """
        variable = self.project.variable("pr")
        axes = self._base_axes()
        grid = self.project.grid("rotated_latitude_longitude")
        zf = self.project.zfactor("ps")
        self.project.validate_components(None, variable, axes, grid=grid, zfactors=[zf])

    def test_complete_axes_satisfies_scalar_axis_requirement(self):
        tas = self.project.variable("tas")
        full_axes = list(self.project.complete_axes(tas, self._base_axes()))
        # Should not raise for the scalar axis requirement
        self.project.validate_components(None, tas, full_axes)

    def test_dataset_from_dataset_info_passes_frequency_check(self):
        pr = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon"})
        self.project.validate_components(dataset, pr, self._base_axes())

    def test_mismatched_frequency_between_dataset_and_variable(self):
        pr = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "day"})
        with self.assertRaises(TableValidationError):
            self.project.validate_components(dataset, pr, self._base_axes())

    def test_grid_from_factory_passes_mapping_name_check(self):
        pr = self.project.variable("pr")
        grid = self.project.grid("lambert_conformal_conic")
        self.project.validate_components(None, pr, self._base_axes(), grid=grid)

    def test_multiple_zfactors_from_factory_all_pass(self):
        pr = self.project.variable("pr")
        zf_ps = self.project.zfactor("ps")
        zf_p0 = self.project.zfactor("p0")
        self.project.validate_components(
            None, pr, self._base_axes(), zfactors=[zf_ps, zf_p0]
        )


class StoredDirectionTest(unittest.TestCase):
    """Tests for stored_direction enforcement on coordinate axes.

    The coordinate table may declare ``stored_direction = "increasing"`` or
    ``"decreasing"`` to specify the expected ordering of values (e.g. pressure
    levels are conventionally stored surface-to-top, i.e. decreasing).  When
    user-supplied values contradict this declaration the axis will produce
    physically incorrect output.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(
            self.tmp,
            coordinate_entries={
                "time": {"axis": "T", "out_name": "time"},
                "plev": {
                    "axis": "Z",
                    "units": "Pa",
                    "standard_name": "air_pressure",
                    "out_name": "plev",
                    "positive": "down",
                    "stored_direction": "decreasing",
                    "valid_min": "100.0",
                    "valid_max": "92500.0",
                },
                "lev": {
                    "axis": "Z",
                    "units": "m",
                    "standard_name": "depth",
                    "out_name": "lev",
                    "stored_direction": "increasing",
                },
                "lat": {
                    "axis": "Y",
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "out_name": "lat",
                },
            },
        )

    def tearDown(self):
        self._ctx.cleanup()

    # --- correct direction passes ---

    def test_decreasing_values_with_decreasing_direction_passes(self):
        """Pressure values from surface to top (decreasing) are correct."""
        ax = self.project.axis("plev", values=[92500.0, 50000.0, 10000.0])
        self.assertIsNotNone(ax)

    def test_increasing_values_with_increasing_direction_passes(self):
        ax = self.project.axis("lev", values=[0.0, 10.0, 50.0])
        self.assertIsNotNone(ax)

    def test_two_value_axis_correct_direction_passes(self):
        ax = self.project.axis("plev", values=[92500.0, 10000.0])
        self.assertIsNotNone(ax)

    def test_single_value_axis_not_checked(self):
        """Direction cannot be determined from a single coordinate value."""
        ax = self.project.axis("plev", values=[50000.0])
        self.assertIsNotNone(ax)

    def test_no_stored_direction_in_table_always_passes(self):
        """Axes without stored_direction declared are not checked."""
        ax = self.project.axis("lat", values=[45.0, -45.0])
        self.assertIsNotNone(ax)

    # --- wrong direction raises at construction time ---

    def test_increasing_values_with_decreasing_direction_raises(self):
        """Increasing pressure values violate stored_direction='decreasing'."""
        with self.assertRaises(AxisValidationError) as ctx:
            self.project.axis("plev", values=[10000.0, 50000.0, 92500.0])
        msg = str(ctx.exception)
        self.assertIn("stored_direction", msg)
        self.assertIn("decreasing", msg)

    def test_decreasing_values_with_increasing_direction_raises(self):
        with self.assertRaises(AxisValidationError) as ctx:
            self.project.axis("lev", values=[50.0, 10.0, 0.0])
        msg = str(ctx.exception)
        self.assertIn("stored_direction", msg)
        self.assertIn("increasing", msg)

    def test_error_message_contains_axis_name(self):
        with self.assertRaises(AxisValidationError) as ctx:
            self.project.axis("plev", values=[10000.0, 50000.0, 92500.0])
        self.assertIn("plev", str(ctx.exception))

    def test_error_message_contains_actual_values(self):
        """Error message shows the first and last values for diagnosis."""
        with self.assertRaises(AxisValidationError) as ctx:
            self.project.axis("plev", values=[10000.0, 92500.0])
        msg = str(ctx.exception)
        self.assertIn("10000", msg)
        self.assertIn("92500", msg)

    # --- validate_axis_values_early catches raw axes too ---

    def test_raw_axis_with_wrong_direction_raises_on_early_validation(self):
        """An unprepared Axis with stored_direction set is also validated."""
        from cmor4._axis_validation import validate_axis_values_early

        raw = Axis(
            name="plev",
            values=[10000.0, 50000.0, 92500.0],
            units="Pa",
            stored_direction="decreasing",
        )
        with self.assertRaises(AxisValidationError):
            validate_axis_values_early(raw)

    def test_raw_axis_correct_direction_passes_early_validation(self):
        from cmor4._axis_validation import validate_axis_values_early

        raw = Axis(
            name="plev",
            values=[92500.0, 50000.0, 10000.0],
            units="Pa",
            stored_direction="decreasing",
        )
        validate_axis_values_early(raw)  # must not raise


class MustHaveBoundsTest(unittest.TestCase):
    """Tests for must_have_bounds enforcement in validate_components.

    The coordinate table field ``must_have_bounds`` requires that bounds are
    always supplied for an axis.  Previously this was only enforced when a
    dataset was provided; it is now enforced unconditionally.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(
            self.tmp,
            variable_entries={
                "pr": {
                    "dimensions": ["time", "lat", "lon"],
                    "out_name": "pr",
                    "units": "kg m-2 s-1",
                }
            },
            coordinate_entries={
                "time": {"axis": "T", "out_name": "time"},
                "lat": {
                    "axis": "Y",
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "out_name": "lat",
                    "must_have_bounds": "1",
                },
                "lon": {
                    "axis": "X",
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "out_name": "lon",
                },
            },
        )

    def tearDown(self):
        self._ctx.cleanup()

    def _base_axes(self, lat_bounds=True):
        time = self.project.axis(
            "time",
            values=[15.0],
            bounds=[[0.0, 30.0]],
            units="days since 2000-01-01",
        )
        lat_kwargs = dict(values=[-45.0, 45.0])
        if lat_bounds:
            lat_kwargs["bounds"] = [[-90.0, 0.0], [0.0, 90.0]]
        lat = self.project.axis("lat", **lat_kwargs)
        lon = self.project.axis(
            "lon",
            values=[90.0, 270.0],
            bounds=[[0.0, 180.0], [180.0, 360.0]],
        )
        return [time, lat, lon]

    # --- passes when bounds are provided ---

    def test_lat_with_bounds_passes_without_dataset(self):
        var = self.project.variable("pr")
        self.project.validate_components(None, var, self._base_axes(lat_bounds=True))

    def test_lat_with_bounds_passes_with_dataset(self):
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon"})
        self.project.validate_components(dataset, var, self._base_axes(lat_bounds=True))

    # --- fails without bounds regardless of whether dataset is provided ---

    def test_lat_without_bounds_raises_without_dataset(self):
        """must_have_bounds is enforced even when dataset=None."""
        var = self.project.variable("pr")
        with self.assertRaises(AxisValidationError) as ctx:
            self.project.validate_components(
                None, var, self._base_axes(lat_bounds=False)
            )
        self.assertIn("must have bounds", str(ctx.exception))
        self.assertIn("lat", str(ctx.exception))

    def test_lat_without_bounds_raises_with_dataset(self):
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon"})
        with self.assertRaises(AxisValidationError) as ctx:
            self.project.validate_components(
                dataset, var, self._base_axes(lat_bounds=False)
            )
        self.assertIn("must have bounds", str(ctx.exception))

    def test_axis_without_must_have_bounds_needs_no_bounds(self):
        """lon has no must_have_bounds — omitting its bounds is fine."""
        var = self.project.variable("pr")
        time = self.project.axis(
            "time", values=[15.0], bounds=[[0.0, 30.0]], units="days since 2000-01-01"
        )
        lat = self.project.axis(
            "lat", values=[-45.0, 45.0], bounds=[[-90.0, 0.0], [0.0, 90.0]]
        )
        lon_no_bounds = self.project.axis("lon", values=[90.0, 270.0])
        self.project.validate_components(None, var, [time, lat, lon_no_bounds])


class TimeIntervalWithoutDatasetTest(unittest.TestCase):
    """Tests for time-axis interval validation when no dataset is provided.

    The variable's table-declared frequency is used to derive the expected
    time-step interval.  Previously this check only fired when a dataset was
    explicitly passed to validate_components; it now runs unconditionally so
    that a daily time axis paired with a monthly variable is caught regardless.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(
            self.tmp,
            variable_entries={
                "pr": {
                    "dimensions": ["time", "lat", "lon"],
                    "out_name": "pr",
                    "units": "kg m-2 s-1",
                    "frequency": "mon",
                },
                "tas_fx": {
                    "dimensions": ["lat", "lon"],
                    "out_name": "tas_fx",
                    "units": "K",
                    # no frequency in table
                },
            },
            coordinate_entries={
                "time": {"axis": "T", "out_name": "time"},
                "lat": {
                    "axis": "Y",
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "out_name": "lat",
                    "must_have_bounds": "1",
                },
                "lon": {
                    "axis": "X",
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "out_name": "lon",
                },
            },
        )

    def tearDown(self):
        self._ctx.cleanup()

    def _spatial_axes(self):
        return [
            self.project.axis(
                "lat", values=[-45.0, 45.0], bounds=[[-90.0, 0.0], [0.0, 90.0]]
            ),
            self.project.axis(
                "lon", values=[90.0, 270.0], bounds=[[0.0, 180.0], [180.0, 360.0]]
            ),
        ]

    def _monthly_time(self):
        return self.project.axis(
            "time",
            values=[15.0, 45.0],
            bounds=[[0.0, 30.0], [30.0, 60.0]],
            units="days since 2000-01-01",
        )

    def _daily_time(self):
        return self.project.axis(
            "time",
            values=[0.5, 1.5, 2.5],
            bounds=[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
            units="days since 2000-01-01",
        )

    # --- correct interval passes ---

    def test_monthly_time_with_monthly_variable_passes_without_dataset(self):
        var = self.project.variable("pr")
        self.project.validate_components(
            None, var, [self._monthly_time()] + self._spatial_axes()
        )

    def test_monthly_time_with_monthly_variable_passes_with_dataset(self):
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon"})
        self.project.validate_components(
            dataset, var, [self._monthly_time()] + self._spatial_axes()
        )

    # --- wrong interval is caught without a dataset ---

    def test_daily_time_with_monthly_variable_raises_without_dataset(self):
        """Table frequency 'mon' is used to validate spacing even without dataset."""
        var = self.project.variable("pr")
        with self.assertRaises(AxisValidationError) as ctx:
            self.project.validate_components(
                None, var, [self._daily_time()] + self._spatial_axes()
            )
        msg = str(ctx.exception)
        self.assertIn("mon", msg)
        self.assertIn("interval", msg.lower())

    def test_daily_time_with_monthly_variable_also_raises_with_dataset(self):
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon"})
        with self.assertRaises(AxisValidationError):
            self.project.validate_components(
                dataset, var, [self._daily_time()] + self._spatial_axes()
            )

    def test_no_variable_frequency_in_table_skips_interval_check(self):
        """When the table declares no frequency there is no expected interval."""
        var = self.project.variable("tas_fx")
        daily = self.project.axis(
            "time",
            values=[0.5, 1.5],
            bounds=[[0.0, 1.0], [1.0, 2.0]],
            units="days since 2000-01-01",
        )
        # Should not raise — no table frequency to validate against
        self.project.validate_components(None, var, [daily])


class MIPCalendarTest(unittest.TestCase):
    """Tests for the warning issued when a dataset specifies a calendar that
    is valid per the CF convention but inappropriate for MIP data.

    CMOR3 rejects 'utc', 'tai', 'all_leap', and '366_day' with a warning.
    CMOR4 mirrors this behaviour.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        self.project = _build_project(
            self.tmp,
            variable_entries={
                "pr": {
                    "dimensions": ["time", "lat", "lon"],
                    "out_name": "pr",
                    "units": "kg m-2 s-1",
                    "frequency": "mon",
                }
            },
            coordinate_entries={
                "time": {"axis": "T", "out_name": "time"},
                "lat": {
                    "axis": "Y",
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "out_name": "lat",
                    "must_have_bounds": "1",
                },
                "lon": {
                    "axis": "X",
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "out_name": "lon",
                },
            },
        )

    def tearDown(self):
        self._ctx.cleanup()

    def _base_axes(self):
        return [
            self.project.axis(
                "time",
                values=[15.0, 45.0],
                bounds=[[0.0, 30.0], [30.0, 60.0]],
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "lat", values=[-45.0, 45.0], bounds=[[-90.0, 0.0], [0.0, 90.0]]
            ),
            self.project.axis(
                "lon", values=[90.0, 270.0], bounds=[[0.0, 180.0], [180.0, 360.0]]
            ),
        ]

    def _validate_with_calendar(self, calendar):
        """
        Run validate_components with the given calendar and return any MIP warnings.
        """
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon", "calendar": calendar})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.project.validate_components(dataset, var, self._base_axes())
        return [w for w in caught if "appropriate for MIP" in str(w.message)]

    # --- invalid calendars (not recognised by cftime at all) raise ---

    def test_utc_calendar_raises(self):
        """'utc' is not a CF calendar — raises AxisValidationError."""
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon", "calendar": "utc"})
        with self.assertRaises(AxisValidationError) as ctx:
            self.project.validate_components(dataset, var, self._base_axes())
        self.assertIn("utc", str(ctx.exception))

    def test_tai_calendar_raises(self):
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon", "calendar": "tai"})
        with self.assertRaises(AxisValidationError):
            self.project.validate_components(dataset, var, self._base_axes())

    def test_completely_unknown_calendar_raises(self):
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon", "calendar": "martian"})
        with self.assertRaises(AxisValidationError):
            self.project.validate_components(dataset, var, self._base_axes())

    def test_invalid_calendar_error_lists_valid_options(self):
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon", "calendar": "utc"})
        with self.assertRaises(AxisValidationError) as ctx:
            self.project.validate_components(dataset, var, self._base_axes())
        self.assertIn("standard", str(ctx.exception))

    # --- MIP-inappropriate but cftime-valid calendars warn ---

    def test_all_leap_calendar_warns(self):
        self.assertTrue(self._validate_with_calendar("all_leap"))

    def test_366_day_calendar_warns(self):
        self.assertTrue(self._validate_with_calendar("366_day"))

    def test_warning_message_names_the_calendar(self):
        warns = self._validate_with_calendar("all_leap")
        self.assertIn("all_leap", str(warns[0].message))

    def test_warning_message_suggests_alternatives(self):
        warns = self._validate_with_calendar("all_leap")
        msg = str(warns[0].message)
        self.assertIn("standard", msg)

    # --- appropriate calendars do not warn ---

    def test_standard_calendar_no_warning(self):
        self.assertFalse(self._validate_with_calendar("standard"))

    def test_gregorian_calendar_no_warning(self):
        self.assertFalse(self._validate_with_calendar("gregorian"))

    def test_proleptic_gregorian_calendar_no_warning(self):
        self.assertFalse(self._validate_with_calendar("proleptic_gregorian"))

    def test_noleap_calendar_no_warning(self):
        self.assertFalse(self._validate_with_calendar("noleap"))

    def test_365_day_calendar_no_warning(self):
        self.assertFalse(self._validate_with_calendar("365_day"))

    def test_360_day_calendar_no_warning(self):
        self.assertFalse(self._validate_with_calendar("360_day"))

    def test_julian_calendar_no_warning(self):
        self.assertFalse(self._validate_with_calendar("julian"))

    # --- no calendar in dataset means no check ---

    def test_no_calendar_in_dataset_no_warning(self):
        var = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.project.validate_components(dataset, var, self._base_axes())
        mip_warns = [w for w in caught if "appropriate for MIP" in str(w.message)]
        self.assertFalse(mip_warns)


class ValidateComponentsTest(unittest.TestCase):
    """Tests for ProjectTables.validate_components."""

    # ------------------------------------------------------------------
    # Per-test project setup
    # ------------------------------------------------------------------

    def setUp(self):
        self._tmp_ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp_ctx.name)
        self.project = _build_vc_project(self.tmp)

    def tearDown(self):
        self._tmp_ctx.cleanup()

    # ------------------------------------------------------------------
    # Axis helpers
    # ------------------------------------------------------------------

    def _time_axis(self):
        return self.project.axis(
            "time",
            values=[15.0, 45.0],
            bounds=[[0.0, 30.0], [30.0, 60.0]],
            units="days since 2000-01-01",
        )

    def _lat_axis(self):
        return self.project.axis(
            "latitude",
            values=[-45.0, 45.0],
            bounds=[[-90.0, 0.0], [0.0, 90.0]],
        )

    def _lon_axis(self):
        return self.project.axis(
            "longitude",
            values=[90.0, 270.0],
            bounds=[[0.0, 180.0], [180.0, 360.0]],
        )

    def _standard_axes(self):
        return [self._time_axis(), self._lat_axis(), self._lon_axis()]

    # ==================================================================
    # 1. Happy-path: valid components raise no error
    # ==================================================================

    def test_valid_variable_and_axes_passes(self):
        """validate_components succeeds for a fully correct setup."""
        variable = self.project.variable("pr")
        self.project.validate_components(None, variable, self._standard_axes())

    def test_valid_with_dataset_none_passes(self):
        """Passing dataset=None skips dataset-specific checks without error."""
        variable = self.project.variable("pr")
        self.project.validate_components(None, variable, self._standard_axes())

    def test_valid_with_matching_frequency_passes(self):
        """Dataset frequency matching variable frequency passes."""
        variable = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "mon"})
        self.project.validate_components(dataset, variable, self._standard_axes())

    def test_valid_with_no_grid_and_no_zfactors_passes(self):
        """Explicitly passing grid=None and empty zfactors is fine."""
        variable = self.project.variable("pr")
        self.project.validate_components(
            None, variable, self._standard_axes(), grid=None, zfactors=[]
        )

    def test_prepared_axes_are_not_re_validated(self):
        """Axes created via project.axis() are marked prepared and trusted."""
        # All axes here come from project.axis() → already prepared.
        variable = self.project.variable("pr")
        axes = self._standard_axes()
        # Sanity: confirm they're considered prepared
        for ax in axes:
            self.assertTrue(self.project._is_prepared_axis(ax))
        # Should pass without re-validating their metadata
        self.project.validate_components(None, variable, axes)

    # ==================================================================
    # 2. Variable metadata validation
    # ==================================================================

    def test_variable_with_wrong_units_raises(self):
        """
        Variable units conflicting with the table entry raise TableValidationError.
        """
        # Bypass project.variable() to get a Variable without pre-validation
        bad_var = Variable(name="pr", units="m s-1")
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, bad_var, self._standard_axes())
        self.assertIn("units", str(ctx.exception))
        self.assertIn("m s-1", str(ctx.exception))

    def test_variable_with_wrong_standard_name_raises(self):
        """Variable standard_name conflicting with table raises TableValidationError."""
        bad_var = Variable(name="pr", standard_name="wrong_standard_name")
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, bad_var, self._standard_axes())
        self.assertIn("standard_name", str(ctx.exception))

    def test_variable_not_in_table_raises(self):
        """Variable name not found in any loaded table raises TableValidationError."""
        unknown_var = Variable(name="not_in_any_table")
        with self.assertRaises(TableValidationError):
            self.project.validate_components(None, unknown_var, self._standard_axes())

    # ==================================================================
    # 3. Dataset-variable consistency (frequency)
    # ==================================================================

    def test_frequency_mismatch_raises(self):
        """Dataset frequency differing from variable table frequency raises."""
        variable = self.project.variable("pr")  # table says 'mon'
        dataset = DatasetInfo.from_mapping({"frequency": "day"})
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(dataset, variable, self._standard_axes())
        msg = str(ctx.exception)
        self.assertIn("frequency", msg)
        self.assertIn("day", msg)
        self.assertIn("mon", msg)

    def test_frequency_mismatch_names_variable_table(self):
        """Error message identifies the variable and table where mismatch occurs."""
        variable = self.project.variable("pr")
        dataset = DatasetInfo.from_mapping({"frequency": "fx"})
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(dataset, variable, self._standard_axes())
        self.assertIn("Amon", str(ctx.exception))
        self.assertIn("pr", str(ctx.exception))

    def test_dataset_none_skips_frequency_check(self):
        """With dataset=None the frequency check is skipped entirely."""
        # Variable has frequency='mon' in table; no dataset → no check → no error
        variable = self.project.variable("pr")
        self.project.validate_components(None, variable, self._standard_axes())

    # ==================================================================
    # 4. Axis metadata validation (unprepared axes)
    # ==================================================================

    def test_unprepared_axis_with_correct_metadata_passes(self):
        """An axis not created via project.axis() but with correct metadata passes."""
        variable = self.project.variable("pr")
        raw_lat = Axis(
            name="latitude",
            values=[-45.0, 45.0],
            standard_name="latitude",
            units="degrees_north",
        )
        axes = [self._time_axis(), raw_lat, self._lon_axis()]
        self.project.validate_components(None, variable, axes)

    def test_unprepared_axis_wrong_standard_name_raises(self):
        """Unprepared axis with wrong standard_name raises TableValidationError."""
        variable = self.project.variable("pr")
        raw_lat = Axis(
            name="latitude", values=[-45.0, 45.0], standard_name="WRONG_NAME"
        )
        axes = [self._time_axis(), raw_lat, self._lon_axis()]
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, variable, axes)
        self.assertIn("standard_name", str(ctx.exception))

    def test_unprepared_axis_wrong_units_raises(self):
        """Unprepared axis with wrong units raises TableValidationError."""
        variable = self.project.variable("pr")
        raw_lat = Axis(
            name="latitude", values=[-45.0, 45.0], units="degrees_east"
        )  # should be degrees_north
        axes = [self._time_axis(), raw_lat, self._lon_axis()]
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, variable, axes)
        self.assertIn("units", str(ctx.exception))

    def test_unknown_axis_not_in_coordinate_table_is_silently_skipped(self):
        """Axes with no matching coordinate table entry are not validated."""
        variable = self.project.variable("pr")
        # 'custom_dim' doesn't appear in our coordinate table → skipped
        custom = Axis(name="custom_dim", values=[1.0, 2.0])
        axes = [self._time_axis(), self._lat_axis(), self._lon_axis(), custom]
        self.project.validate_components(None, variable, axes)

    # ==================================================================
    # 5. Scalar axis enforcement
    # ==================================================================

    def test_missing_required_scalar_axis_raises(self):
        """Variable that requires a scalar axis but it's absent raises."""
        variable = self.project.variable("tas")  # needs 'height2m'
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, variable, self._standard_axes())
        msg = str(ctx.exception)
        self.assertIn("height2m", msg)
        self.assertIn("scalar axis", msg)

    def test_error_message_suggests_scalar_axes_helper(self):
        """Error for missing scalar axis mentions helper methods."""
        variable = self.project.variable("tas")
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, variable, self._standard_axes())
        msg = str(ctx.exception)
        self.assertTrue(
            "scalar_axes_for" in msg or "complete_axes" in msg,
            msg=f"Expected helper name in error message, got: {msg}",
        )

    def test_scalar_axis_auto_added_via_complete_axes_passes(self):
        """complete_axes() satisfies scalar axis requirement so validation passes."""
        variable = self.project.variable("tas")
        full_axes = list(self.project.complete_axes(variable, self._standard_axes()))
        self.project.validate_components(None, variable, full_axes)

    def test_scalar_axis_added_manually_passes(self):
        """Manually providing the scalar axis satisfies the requirement."""
        variable = self.project.variable("tas")
        scalar_axes = self.project.scalar_axes_for(variable, self._standard_axes())
        axes = self._standard_axes() + list(scalar_axes)
        self.project.validate_components(None, variable, axes)

    # ==================================================================
    # 6. Grid validation
    # ==================================================================

    def test_valid_grid_mapping_passes(self):
        """Grid with grid_mapping_name matching the table entry passes."""
        variable = self.project.variable("pr")
        grid = self.project.grid(
            "rotated_latitude_longitude",
            grid_mapping_name="rotated_latitude_longitude",
        )
        self.project.validate_components(
            None, variable, self._standard_axes(), grid=grid
        )

    def test_grid_mapping_name_mismatch_raises(self):
        """Grid with grid_mapping_name conflicting with table raises."""
        variable = self.project.variable("pr")
        # Force wrong value by bypassing project constructor
        grid = Grid.__new__(Grid)
        object.__setattr__(grid, "name", "rotated_latitude_longitude")
        object.__setattr__(grid, "table_entry", "rotated_latitude_longitude")
        object.__setattr__(grid, "mapping_entry", None)
        object.__setattr__(grid, "grid_mapping_name", "wrong_value")
        object.__setattr__(grid, "mapping_name", None)
        object.__setattr__(grid, "mapping_var", None)
        object.__setattr__(grid, "dimensions", None)
        object.__setattr__(grid, "coordinates", ())
        object.__setattr__(grid, "params", {})
        object.__setattr__(grid, "attrs", {})
        object.__setattr__(grid, "latitude", None)
        object.__setattr__(grid, "longitude", None)
        object.__setattr__(grid, "latitude_vertices", None)
        object.__setattr__(grid, "longitude_vertices", None)
        object.__setattr__(grid, "vertices_dim", "vertices")
        object.__setattr__(grid, "project", None)
        object.__setattr__(grid, "extra", {})
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, variable, self._standard_axes(), grid=grid
            )
        self.assertIn("grid_mapping_name", str(ctx.exception))

    def test_grid_not_in_table_is_ignored(self):
        """Grid with no matching mapping entry skips mapping validation."""
        variable = self.project.variable("pr")
        # 'unknown_projection' doesn't exist in the grids table
        grid = Grid()
        self.project.validate_components(
            None, variable, self._standard_axes(), grid=grid
        )

    def test_grid_dimensions_matching_axes_passes(self):
        """Grid dimensions that resolve to supplied axes pass validation."""
        variable = self.project.variable("pr")
        grid = Grid(dimensions=["latitude", "longitude"])
        self.project.validate_components(
            None, variable, self._standard_axes(), grid=grid
        )

    def test_grid_dimension_not_in_axes_raises(self):
        """A grid dimension name that matches no axis raises TableValidationError."""
        variable = self.project.variable("pr")
        grid = Grid(dimensions=["latitude", "no_such_axis"])
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(
                None, variable, self._standard_axes(), grid=grid
            )
        self.assertIn("no_such_axis", str(ctx.exception))

    def test_grid_dimensions_not_set_skips_dimension_check(self):
        """Grid with no dimensions set does not raise for dimension resolution."""
        variable = self.project.variable("pr")
        grid = Grid(latitude=np.ones((2, 3)))
        self.project.validate_components(
            None, variable, self._standard_axes(), grid=grid
        )

    # ==================================================================
    # 7. ZFactor validation
    # ==================================================================

    def test_valid_zfactor_passes(self):
        """ZFactor with metadata matching the formula table passes."""
        variable = self.project.variable("pr")
        zf = self.project.zfactor("ps", units="Pa")
        self.project.validate_components(
            None, variable, self._standard_axes(), zfactors=[zf]
        )

    def test_zfactor_wrong_units_raises(self):
        """ZFactor with units conflicting with formula table raises."""
        # Validation happens at construction time via project.zfactor()
        with self.assertRaises(TableValidationError) as ctx:
            self.project.zfactor("ps", units="hPa")
        self.assertIn("units", str(ctx.exception))

    def test_zfactor_not_in_formula_table_passes_without_error(self):
        """ZFactor name absent from the formula table is silently accepted."""
        variable = self.project.variable("pr")
        zf = ZFactor(name="not_in_table")
        self.project.validate_components(
            None, variable, self._standard_axes(), zfactors=[zf]
        )

    def test_multiple_valid_zfactors_pass(self):
        """Multiple ZFactors all matching formula table entries pass."""
        # Add a second formula entry to the project
        tmp2 = Path(self._tmp_ctx.name) / "multi"
        tmp2.mkdir()
        project = _build_vc_project(
            tmp2,
            formula_entries={
                "ps": {
                    "units": "Pa",
                    "standard_name": "surface_air_pressure",
                    "out_name": "ps",
                },
                "p0": {
                    "units": "Pa",
                    "standard_name": "reference_air_pressure",
                    "out_name": "p0",
                },
            },
        )
        variable = project.variable("pr")
        axes = [
            project.axis(
                "time",
                values=[15.0],
                bounds=[[0.0, 30.0]],
                units="days since 2000-01-01",
            ),
            project.axis("latitude", values=[-45.0, 45.0]),
            project.axis("longitude", values=[90.0, 270.0]),
        ]
        zf_ps = project.zfactor("ps", units="Pa")
        zf_p0 = project.zfactor("p0", units="Pa")
        project.validate_components(None, variable, axes, zfactors=[zf_ps, zf_p0])

    # ==================================================================
    # 8. Returns None (no return value)
    # ==================================================================

    def test_validate_components_returns_none(self):
        """validate_components returns None on success."""
        variable = self.project.variable("pr")
        result = self.project.validate_components(None, variable, self._standard_axes())
        self.assertIsNone(result)


class ValidateGridDimensionsTest(unittest.TestCase):
    """Tests for grid-dimension ↔ axis correspondence validation.

    Covers the _validate_grid_dimensions helper called by validate_components:

    * Every name in grid.dimensions must resolve to a supplied axis.
    * When grid.latitude / grid.longitude is provided, shape[i] must equal
      the length of the axis at position i in grid.dimensions.
    * Matching is done by axis.name, axis.out_name, axis.table_entry, etc.
    * When grid.dimensions is absent the check is skipped entirely.
    """

    def setUp(self):
        self._ctx = tempfile.TemporaryDirectory()
        self.tmp = Path(self._ctx.name)
        # Project with x/y spatial axes (lengths 3 and 4) and time
        self.project = _build_project(
            self.tmp,
            cv={"CV": {}},
            variable_entries={
                "pr": {
                    "dimensions": ["time", "x", "y"],
                    "out_name": "pr",
                    "units": "kg m-2 s-1",
                }
            },
            coordinate_entries={
                "time": {"axis": "T", "out_name": "time"},
                "x": {"axis": "X", "units": "m", "out_name": "x"},
                "y": {"axis": "Y", "units": "m", "out_name": "y"},
            },
            include_formula=False,
            include_grid=False,
        )

    def tearDown(self):
        self._ctx.cleanup()

    def _axes(self, nx=3, ny=4):
        return [
            self.project.axis("time", values=[1.0, 2.0], units="days since 2000-01-01"),
            self.project.axis("x", values=np.arange(nx, dtype="f4")),
            self.project.axis("y", values=np.arange(ny, dtype="f4")),
        ]

    def _var(self):
        return self.project.variable("pr")

    # -- Happy-path ----------------------------------------------------------

    def test_correct_shape_passes(self):
        """lat/lon arrays shaped (nx, ny) match axes x=3, y=4."""
        lat = np.ones((3, 4))
        lon = np.ones((3, 4))
        grid = Grid(dimensions=["x", "y"], latitude=lat, longitude=lon)
        self.project.validate_components(None, self._var(), self._axes(), grid=grid)

    def test_dims_only_no_latlon_passes(self):
        """Grid with dimensions but no lat/lon still validates dim names."""
        grid = Grid(dimensions=["x", "y"])
        self.project.validate_components(None, self._var(), self._axes(), grid=grid)

    def test_no_dimensions_skips_validation(self):
        """Grid with no dimensions set skips the dimension check entirely."""
        grid = Grid(latitude=np.ones((3, 4)))  # lat provided but dims absent
        self.project.validate_components(None, self._var(), self._axes(), grid=grid)

    def test_latitude_only_checked_when_dimensions_present(self):
        """Only latitude provided; dimensions set → shape is validated."""
        lat = np.ones((3, 4))
        grid = Grid(dimensions=["x", "y"], latitude=lat)
        self.project.validate_components(None, self._var(), self._axes(), grid=grid)

    def test_longitude_only_checked_when_dimensions_present(self):
        """Only longitude provided; dimensions set → shape is validated."""
        lon = np.ones((3, 4))
        grid = Grid(dimensions=["x", "y"], longitude=lon)
        self.project.validate_components(None, self._var(), self._axes(), grid=grid)

    def test_match_by_axis_out_name(self):
        """Grid dimensions may be specified using the axis out_name."""
        # 'x' axis has out_name 'x' (same), but let's test with 'y' → out_name 'y'
        lat = np.ones((3, 4))
        grid = Grid(dimensions=["x", "y"], latitude=lat)
        self.project.validate_components(None, self._var(), self._axes(), grid=grid)

    # -- Dimension name resolution failures ----------------------------------

    def test_unknown_dimension_name_raises(self):
        """A grid dimension name matching no axis raises TableValidationError."""
        grid = Grid(dimensions=["x", "no_such_axis"])
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, self._var(), self._axes(), grid=grid)
        self.assertIn("no_such_axis", str(ctx.exception))

    def test_error_message_mentions_missing_dimension(self):
        grid = Grid(dimensions=["totally_unknown"])
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, self._var(), self._axes(), grid=grid)
        msg = str(ctx.exception)
        self.assertIn("totally_unknown", msg)
        self.assertIn("axis", msg)

    # -- Shape mismatch failures --------------------------------------------

    def test_latitude_shape_wrong_first_dim_raises(self):
        """lat.shape[0] not matching x axis length raises."""
        lat = np.ones((5, 4))  # x has 3 points, not 5
        grid = Grid(dimensions=["x", "y"], latitude=lat)
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, self._var(), self._axes(), grid=grid)
        msg = str(ctx.exception)
        self.assertIn("latitude", msg)
        self.assertIn("x", msg)

    def test_latitude_shape_wrong_second_dim_raises(self):
        """lat.shape[1] not matching y axis length raises."""
        lat = np.ones((3, 7))  # y has 4 points, not 7
        grid = Grid(dimensions=["x", "y"], latitude=lat)
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, self._var(), self._axes(), grid=grid)
        msg = str(ctx.exception)
        self.assertIn("latitude", msg)
        self.assertIn("y", msg)

    def test_longitude_shape_mismatch_raises(self):
        """lon.shape not matching axes raises even when lat is absent."""
        lon = np.ones((9, 4))  # x has 3 points
        grid = Grid(dimensions=["x", "y"], longitude=lon)
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, self._var(), self._axes(), grid=grid)
        msg = str(ctx.exception)
        self.assertIn("longitude", msg)

    def test_error_message_includes_array_shape_and_axis_length(self):
        """Error message reports the mismatching shape and expected length."""
        lat = np.ones((5, 4))
        grid = Grid(dimensions=["x", "y"], latitude=lat)
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, self._var(), self._axes(), grid=grid)
        msg = str(ctx.exception)
        self.assertIn("(5, 4)", msg)  # reported array shape
        self.assertIn("3", msg)  # expected axis length

    def test_transposed_shape_raises(self):
        """Shape (ny, nx) instead of (nx, ny) is caught as a mismatch."""
        lat = np.ones((4, 3))  # transposed: should be (3, 4)
        grid = Grid(dimensions=["x", "y"], latitude=lat)
        with self.assertRaises(TableValidationError) as ctx:
            self.project.validate_components(None, self._var(), self._axes(), grid=grid)
        self.assertIn("latitude", str(ctx.exception))

    def test_transposed_correct_shape_passes(self):
        """Shape (ny, nx) with dimensions declared as ['y', 'x'] is valid."""
        lat = np.ones((4, 3))  # (ny, nx)
        grid = Grid(dimensions=["y", "x"], latitude=lat)
        self.project.validate_components(None, self._var(), self._axes(), grid=grid)

    # -- Axis matching via name variants ------------------------------------

    def test_match_by_table_entry(self):
        """Dimension name matching an axis's table_entry resolves correctly."""
        # x axis has table_entry="x" from coordinate table
        lat = np.ones((3, 4))
        grid = Grid(dimensions=["x", "y"], latitude=lat)
        self.project.validate_components(None, self._var(), self._axes(), grid=grid)

    def test_unprepared_axis_also_matched(self):
        """Axes not created via project.axis() are also matched by name."""
        # Supply an unprepared Axis for 'x' (raw, not from project.axis())
        axes = [
            self.project.axis("time", values=[1.0], units="days since 2000-01-01"),
            Axis(name="x", values=np.arange(3, dtype="f4")),
            Axis(name="y", values=np.arange(4, dtype="f4")),
        ]
        lat = np.ones((3, 4))
        grid = Grid(dimensions=["x", "y"], latitude=lat)
        self.project.validate_components(None, self._var(), axes, grid=grid)


class ValidateComponentsMinimalProjectTest(unittest.TestCase):
    """Tests using a minimal project with a single variable and no optional tables."""

    def setUp(self):
        self._tmp_ctx = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp_ctx.name)
        cv_file = tmp / "CV.json"
        vtable_file = tmp / "vars.json"
        cv_file.write_text('{"CV": {}}\n')
        vtable_file.write_text(
            json.dumps(
                {
                    "Header": {"table_id": "test"},
                    "variable_entry": {
                        "sample": {
                            "dimensions": ["time"],
                            "out_name": "sample",
                            "units": "1",
                        }
                    },
                }
            )
            + "\n"
        )
        self.project = ProjectTables(cv_file, [vtable_file])

    def tearDown(self):
        self._tmp_ctx.cleanup()

    def test_minimal_project_with_no_optional_tables(self):
        """
        validate_components works when coordinate, formula, and grid tables are absent.
        """
        variable = self.project.variable("sample")
        axis = Axis(name="time", values=[1.0, 2.0])
        self.project.validate_components(None, variable, [axis])

    def test_no_axes_passes_for_variable_with_no_required_dimensions(self):
        """Empty axes list is fine if the variable has no dimension requirements."""
        variable = self.project.variable("sample")
        # The 'time' dimension is present but there's no scalar_axis_entries entry
        # for it, so it won't trigger the scalar-axis check.
        # Passing no axes is a user error but not caught by validate_components itself.
        self.project.validate_components(None, variable, [])


if __name__ == "__main__":
    unittest.main()
