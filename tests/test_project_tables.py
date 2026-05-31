from __future__ import annotations

import tempfile
import unittest

import numpy as np

import cmor4
from table_helpers import (
    CMIP7_TABLE_ROOT,
    OBS4MIPS_TABLE_ROOT,
    cmip7_project,
    obs4mips_project,
)


def require_path(test_case: unittest.TestCase, path: Path) -> None:
    if not path.exists():
        test_case.skipTest(f"Project table source is not available: {path}")


def lat_lon_axes():
    return [
        {
            "name": "time",
            "values": [15.0, 45.0],
            "bounds": [[0.0, 30.0], [30.0, 60.0]],
            "units": "days since 2000-01-01",
            "standard_name": "time",
            "axis": "T",
        },
        {
            "name": "latitude",
            "values": [-45.0, 45.0],
            "bounds": [[-90.0, 0.0], [0.0, 90.0]],
            "units": "degrees_north",
        },
        {
            "name": "longitude",
            "values": [90.0, 270.0],
            "bounds": [[0.0, 180.0], [180.0, 360.0]],
            "units": "degrees_east",
        },
    ]


class ProjectTablesTest(unittest.TestCase):
    def test_loads_cv_and_variable_entries_from_submodule(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")

        self.assertIn("activity_id", project.cv)
        self.assertIn("tos_tavg-u-hxy-sea", project.variable_entries)
        self.assertIn("latitude", project.coordinate_entries)
        self.assertIn("ps", project.formula_entries)
        self.assertEqual(project.coordinate_aliases["latitude"], "lat")
        self.assertEqual(project.coordinate_aliases["height2m"], "height")
        self.assertEqual(
            project.variable_entries["tos_tavg-u-hxy-sea"].entry["out_name"],
            "tos",
        )

    def test_prepare_inputs_merges_authoritative_variable_metadata(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project("Tables/obs4MIPs_Amon.json")
        dataset = {
            "activity_id": "obs4MIPs",
            "grid_label": "gnz",
            "institution_id": "DLR-BIRA",
            "license": project.cv["license"],
            "nominal_resolution": "500 km",
            "product": "observations",
            "source_id": "BSVertOzone-v1-0",
        }

        prepared_dataset, prepared_variable = project.prepare_inputs(
            dataset, {"name": "o3zm"}
        )

        self.assertEqual(prepared_dataset["source_id"], "BSVertOzone-v1-0")
        self.assertEqual(prepared_variable["id"], "o3")
        self.assertEqual(prepared_variable["units"], "mol mol-1")
        self.assertEqual(
            prepared_variable["dimensions"], ("time", "height", "latitude")
        )

    def test_duplicate_variable_names_require_table_id(self):
        require_path(self, OBS4MIPS_TABLE_ROOT)
        project = obs4mips_project(
            "Tables/obs4MIPs_Amon.json", "Tables/obs4MIPs_A1hrPt.json"
        )

        with self.assertRaises(cmor4.TableValidationError):
            project.resolve_variable({"name": "pr"})

        entry = project.resolve_variable(
            {"name": "pr", "table_id": "obs4MIPs_A1hrPt"}
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
                "mip_era": "CMIP7",
                "nominal_resolution": "100 km",
                "outpath": tmp_dir,
                "physics_index": "p1",
                "realization_index": "r9",
                "region": "glb",
                "source_id": "DUMMY-MODEL",
                "version": "v20200101",
            }
            variable = {"name": "tos_tavg-u-hxy-sea"}

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

    def test_cmip7_rejects_values_not_in_cv(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = {
            "activity_id": "not-a-real-activity",
            "experiment_id": "amip",
            "forcing_index": "f3",
            "grid_label": "g999",
            "initialization_index": "i1",
            "institution_id": "CCCma",
            "nominal_resolution": "100 km",
            "physics_index": "p1",
            "realization_index": "r9",
            "region": "glb",
            "source_id": "DUMMY-MODEL",
        }

        with self.assertRaises(cmor4.TableValidationError):
            cmor4.create_dataset(
                dataset,
                {"name": "tos_tavg-u-hxy-sea"},
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
            )

    def test_cmip7_rejects_variable_metadata_not_in_table(self):
        require_path(self, CMIP7_TABLE_ROOT)
        project = cmip7_project("tables/CMIP7_ocean.json")
        dataset = {
            "activity_id": "CMIP",
            "experiment_id": "amip",
            "forcing_index": "f3",
            "grid_label": "g999",
            "initialization_index": "i1",
            "institution_id": "CCCma",
            "nominal_resolution": "100 km",
            "physics_index": "p1",
            "realization_index": "r9",
            "region": "glb",
            "source_id": "DUMMY-MODEL",
        }

        with self.assertRaises(cmor4.TableValidationError):
            cmor4.create_dataset(
                dataset,
                {"name": "tos_tavg-u-hxy-sea", "units": "K"},
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
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
                "source_id": "BSVertOzone-v1-0",
                "variant_label": "CMORGuide",
                "version": "v20260512",
            }
            axes = [
                lat_lon_axes()[0],
                {
                    "name": "height",
                    "values": [1000.0, 5000.0],
                    "units": "m",
                    "standard_name": "height",
                    "axis": "Z",
                    "positive": "up",
                },
                lat_lon_axes()[1],
            ]

            result = cmor4.cmorize(
                dataset,
                {"name": "o3zm"},
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
                {"name": "not_a_table_variable"},
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
                {"name": "pr"},
                lat_lon_axes(),
                np.ones((2, 2, 2), dtype="f4"),
                project=project,
            )


if __name__ == "__main__":
    unittest.main()
