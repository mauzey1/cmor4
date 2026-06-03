from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import cmor4
from table_helpers import (
    CMIP7_TABLE_ROOT,
    DRCDP_TABLE_ROOT,
    OBS4MIPS_TABLE_ROOT,
    cmip7_project,
    drcdp_project,
    obs4mips_project,
)


def require_path(test_case: unittest.TestCase, path: Path) -> None:
    if not path.exists():
        test_case.skipTest(f"Project table source is not available: {path}")


def lat_lon_axes():
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
        axes = project.prepare_axes(
            [
                cmor4.Axis(
                    name="time",
                    values=[15.0],
                    units="days since 2020-02-01",
                ),
                cmor4.Axis(name="x", values=[0.0, 1.0]),
                cmor4.Axis(name="y", values=[2.0, 3.0]),
                cmor4.Axis(
                    name="latitude",
                    out_name="latitude",
                    values=[[10.0, 20.0], [30.0, 40.0]],
                    dimensions=["x", "y"],
                    bounds=np.ones((2, 2, 4), dtype="f8"),
                    bounds_name="vertices_latitude",
                    bounds_dim="vertices",
                    auxiliary=True,
                ),
                cmor4.Axis(
                    name="longitude",
                    out_name="longitude",
                    values=[[100.0, 110.0], [120.0, 130.0]],
                    dimensions=["x", "y"],
                    bounds=np.ones((2, 2, 4), dtype="f8"),
                    bounds_name="vertices_longitude",
                    bounds_dim="vertices",
                    auxiliary=True,
                ),
            ]
        )

        ds = cmor4.create_dataset(
            {"frequency": "mon"},
            cmor4.Variable(
                name="sample",
                dimensions=["time", "x", "y"],
                coordinates=["latitude", "longitude"],
            ),
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

        self.assertIn("activity_id", project.cv)
        self.assertIn("tos_tavg-u-hxy-sea", project.variable_entries)
        self.assertIn("latitude", project.coordinate_entries)
        self.assertIn("x", project.coordinate_entries)
        self.assertIn("latitude", project.grid_coordinate_entries)
        self.assertIn("vertices_latitude", project.grid_coordinate_entries)
        self.assertIn("ps", project.formula_entries)
        self.assertEqual(project.coordinate_aliases["latitude"], "lat")
        self.assertEqual(project.coordinate_aliases["height2m"], "height")
        self.assertEqual(
            project.variable_entries["tos_tavg-u-hxy-sea"].entry["out_name"],
            "tos",
        )

    def test_cmip7_grid_axes_and_aux_coords_come_from_grids_table(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_seaIce.json")
        axes = project.prepare_axes(
            [
                cmor4.Axis(
                    name="time",
                    values=[15.0],
                    units="days since 2020-02-01",
                ),
                cmor4.Axis(name="x", values=[0.0, 1.0]),
                cmor4.Axis(name="y", values=[2.0, 3.0]),
                cmor4.Axis(
                    name="latitude",
                    out_name="latitude",
                    values=[[10.0, 20.0], [30.0, 40.0]],
                    dimensions=["x", "y"],
                    bounds=np.ones((2, 2, 4), dtype="f8"),
                    bounds_name="vertices_latitude",
                    bounds_dim="vertices",
                    auxiliary=True,
                ),
                cmor4.Axis(
                    name="longitude",
                    out_name="longitude",
                    values=[[100.0, 110.0], [120.0, 130.0]],
                    dimensions=["x", "y"],
                    bounds=np.ones((2, 2, 4), dtype="f8"),
                    bounds_name="vertices_longitude",
                    bounds_dim="vertices",
                    auxiliary=True,
                ),
            ]
        )

        ds = cmor4.create_dataset(
            {"frequency": "mon"},
            cmor4.Variable(
                name="sample",
                dimensions=["time", "x", "y"],
                coordinates=["latitude", "longitude"],
            ),
            axes,
            np.ones((1, 2, 2), dtype="f4"),
        )

        self.assertEqual(ds["x"].attrs["standard_name"], "projection_x_coordinate")
        self.assertEqual(ds["x"].attrs["long_name"], "x coordinate of projection")
        self.assertEqual(ds["x"].attrs["units"], "m")
        self.assertEqual(ds["y"].attrs["standard_name"], "projection_y_coordinate")
        self.assertEqual(ds["latitude"].attrs["standard_name"], "latitude")
        self.assertEqual(ds["latitude"].attrs["units"], "degrees_north")
        self.assertEqual(
            ds["vertices_latitude"].attrs["units"], "degrees_north"
        )
        self.assertEqual(
            ds["sample"].attrs["coordinates"], "latitude longitude"
        )

    def test_drcdp_grid_axes_and_aux_coords_come_from_grids_table(self):
        require_path(self, DRCDP_TABLE_ROOT)
        project = drcdp_project("Tables/DRCDP_APday.json")

        self.assert_grid_table_metadata_is_used(project)

    def test_obs4mips_grid_axes_and_aux_coords_come_from_grids_table(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project("Tables/obs4MIPs_Amon.json")

        self.assert_grid_table_metadata_is_used(project)

    def test_prepare_inputs_merges_authoritative_variable_metadata(self):
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

        prepared_dataset, prepared_variable = project.prepare_inputs(
            dataset, cmor4.Variable(name="o3zm")
        )

        self.assertEqual(prepared_dataset["source_id"], "BSVertOzone-v1-0")
        self.assertEqual(prepared_variable.id, "o3")
        self.assertEqual(prepared_variable.units, "mol mol-1")
        self.assertEqual(
            prepared_variable.dimensions, ("time", "height", "latitude")
        )
        self.assertEqual(
            prepared_dataset["source"],
            "BSVertOzone v1-0 (2018): Mole concentration of ozone in air",
        )
        self.assertEqual(prepared_dataset["source_type"], "satellite_retrieval")
        self.assertEqual(prepared_dataset["source_version_number"], "v1-0")

    def test_variable_table_axis_entries_override_coordinate_table(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cv_file = root / "CV.json"
            variable_table = root / "test_table.json"
            coordinate_table = root / "coordinate.json"
            cv_file.write_text('{"CV": {}}\n')
            coordinate_table.write_text(
                """
{
  "axis_entry": {
    "sample_height": {
      "axis": "Z",
      "long_name": "Coordinate Table Height",
      "out_name": "height",
      "positive": "up",
      "standard_name": "height",
      "units": "m"
    },
    "time": {
      "axis": "T",
      "long_name": "time",
      "out_name": "time",
      "standard_name": "time",
      "units": "days since 2000-01-01"
    }
  }
}
""".strip()
                + "\n"
            )
            variable_table.write_text(
                """
{
  "Header": {"table_id": "test"},
  "axis_entry": {
    "sample_height": {
      "long_name": "Variable Table Height",
      "standard_name": "height_above_mean_sea_level"
    }
  },
  "variable_entry": {
    "sample": {
      "cell_measures": "area: areacella",
      "dimensions": ["sample_height", "time"],
      "long_name": "Sample",
      "out_name": "sample",
      "standard_name": "sample_standard_name",
      "units": "1"
    }
  }
}
""".strip()
                + "\n"
            )
            project = cmor4.ProjectTables(
                cv_file,
                [variable_table],
                coordinate_table=coordinate_table,
            )

            ds = cmor4.create_dataset(
                {},
                cmor4.Variable(name="sample"),
                [
                    cmor4.Axis(name="time", values=[15.0, 45.0]),
                    cmor4.Axis(
                        name="sample_height", values=[10.0, 20.0]
                    ),
                ],
                np.ones((2, 2), dtype="f4"),
                project=project,
            )

            self.assertEqual(
                ds["height"].attrs["long_name"], "Variable Table Height"
            )
            self.assertEqual(
                ds["height"].attrs["standard_name"],
                "height_above_mean_sea_level",
            )
            self.assertEqual(ds["height"].attrs["units"], "m")
            self.assertEqual(
                ds["sample"].attrs["cell_measures"], "area: areacella"
            )

    def test_axis_resolution_does_not_use_axis_letter_alone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cv_file = root / "CV.json"
            variable_table = root / "test_table.json"
            coordinate_table = root / "coordinate.json"
            cv_file.write_text('{"CV": {}}\n')
            variable_table.write_text(
                """
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
""".strip()
                + "\n"
            )
            coordinate_table.write_text(
                """
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
""".strip()
                + "\n"
            )
            project = cmor4.ProjectTables(
                cv_file,
                [variable_table],
                coordinate_table=coordinate_table,
            )

            merged = cmor4.Axis(
                name="x", values=[0.0, 1.0], axis="X", units="m"
            ).merge_table_entry(project)

            self.assertEqual(merged.name, "x")
            self.assertIsNone(merged.table_entry)
            self.assertNotEqual(merged.out_name, "lon")

    def test_duplicate_variable_names_require_table_id(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project(
            "Tables/obs4MIPs_Amon.json", "Tables/obs4MIPs_A1hrPt.json"
        )

        with self.assertRaises(cmor4.TableValidationError):
            cmor4.Variable(name="pr").resolve_table_entry(project)

        entry = cmor4.Variable(
            name="pr", table_id="obs4MIPs_A1hrPt"
        ).resolve_table_entry(project)

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
            variable = cmor4.Variable(name="tos_tavg-u-hxy-sea")

            result = cmor4.cmorize(
                dataset,
                variable,
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
            )

            self.assertEqual(result.dataset["tos"].attrs["units"], "degC")
            self.assertEqual(
                result.dataset["lat"].attrs["standard_name"], "latitude"
            )
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

        ds = cmor4.create_dataset(
            dataset,
            cmor4.Variable(name="tos_tavg-u-hxy-sea"),
            lat_lon_axes(),
            np.ones((2, 2, 2), dtype="f4"),
            project=project,
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

        with self.assertRaises(cmor4.TableValidationError):
            cmor4.create_dataset(
                dataset,
                cmor4.Variable(name="tos_tavg-u-hxy-sea"),
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
            )

    def test_experiment_required_source_type_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cv_file = root / "CV.json"
            variable_table = root / "test_table.json"
            cv_file.write_text(
                """
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
""".strip()
                + "\n"
            )
            variable_table.write_text(
                """
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
""".strip()
                + "\n"
            )
            project = cmor4.ProjectTables(cv_file, [variable_table])
            dataset = {
                "activity_id": "CMIP",
                "experiment_id": "historical",
                "source_type": "AOGCM AER",
            }

            prepared, _ = project.prepare_inputs(
                dataset, cmor4.Variable(name="sample")
            )
            self.assertEqual(prepared["source_type"], "AOGCM AER")

            with self.assertRaisesRegex(
                cmor4.TableValidationError, "missing required"
            ):
                project.prepare_inputs(
                    {**dataset, "source_type": "AER"},
                    cmor4.Variable(name="sample"),
                )
            with self.assertRaisesRegex(
                cmor4.TableValidationError, "not allowed"
            ):
                project.prepare_inputs(
                    {**dataset, "source_type": "AOGCM LAND"},
                    cmor4.Variable(name="sample"),
                )

    def test_required_global_attributes_are_enforced(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = cmip7_dataset()
        dataset.pop("license_id")

        with self.assertRaisesRegex(
            cmor4.TableValidationError, "license_id"
        ):
            cmor4.create_dataset(
                dataset,
                cmor4.Variable(name="tos_tavg-u-hxy-sea"),
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
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

        with self.assertRaisesRegex(
            cmor4.TableValidationError, "institution_id"
        ):
            cmor4.create_dataset(
                dataset,
                cmor4.Variable(name="tasmax"),
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
            )

    def test_parent_experiment_attributes_are_required_by_experiment_cv(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = cmip7_dataset(experiment_id="historical")

        with self.assertRaisesRegex(
            cmor4.TableValidationError, "parent_experiment_id"
        ):
            cmor4.create_dataset(
                dataset,
                cmor4.Variable(name="tos_tavg-u-hxy-sea"),
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
            )

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

        ds = cmor4.create_dataset(
            dataset,
            cmor4.Variable(name="tos_tavg-u-hxy-sea"),
            lat_lon_axes(),
            np.ones((2, 2, 2), dtype="f4"),
            project=project,
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
                with self.assertRaises(cmor4.TableValidationError):
                    cmor4.create_dataset(
                        {**dataset, key: value},
                        cmor4.Variable(name="tos_tavg-u-hxy-sea"),
                        lat_lon_axes(),
                        np.ones((2, 2, 2), dtype="f4"),
                        project=project,
                    )

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

        ds = cmor4.create_dataset(
            dataset,
            cmor4.Variable(
                name="tos_tavg-u-hxy-sea",
                units="K",
                standard_name="not_the_table_value",
                attrs={
                    "long_name": "Not the table value",
                    "cell_methods": "not the table value",
                },
            ),
            lat_lon_axes(),
            np.ones((2, 2, 2), dtype="f4"),
            project=project,
        )

        self.assertEqual(ds["tos"].attrs["units"], "degC")
        self.assertEqual(
            ds["tos"].attrs["standard_name"], "sea_surface_temperature"
        )
        self.assertEqual(
            ds["tos"].attrs["long_name"], "Sea Surface Temperature"
        )
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
                    "<variable_id><frequency><source_id><variant_label>"
                    "<grid_label>"
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
                lat_lon_axes()[0],
                cmor4.Axis(name="height", values=[1000.0, 5000.0]),
                lat_lon_axes()[1],
            ]

            result = cmor4.cmorize(
                dataset,
                cmor4.Variable(name="o3zm"),
                axes,
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
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
        dataset = {
            "activity_id": "obs4MIPs",
            "grid_label": "gn",
            "institution_id": "NOAA-NCEI",
            "license": project.cv["license"],
            "nominal_resolution": "250 km",
            "product": "observations",
            "source_id": "CMAP-V1902",
        }

        with self.assertRaises(cmor4.TableValidationError):
            cmor4.create_dataset(
                dataset,
                cmor4.Variable(name="not_a_table_variable"),
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
            )

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

        with self.assertRaises(cmor4.TableValidationError):
            cmor4.create_dataset(
                dataset,
                cmor4.Variable(name="pr"),
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
            )


if __name__ == "__main__":
    unittest.main()
