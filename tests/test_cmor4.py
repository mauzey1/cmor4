from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

import cmor4
from table_helpers import cmip7_project


def dataset_info(tmp_path: Path):
    return {
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
        "outpath": str(tmp_path),
        "physics_index": "p1",
        "realization_index": "r9",
        "region": "glb",
        "source_id": "DUMMY-MODEL",
        "version": "v20200101",
    }


def horizontal_axes():
    return [
        {
            "name": "latitude",
            "values": [-45.0, 45.0],
            "bounds": [[-90.0, 0.0], [0.0, 90.0]],
            "units": "degrees_north",
            "standard_name": "latitude",
            "axis": "Y",
        },
        {
            "name": "longitude",
            "values": [90.0, 270.0],
            "bounds": [[0.0, 180.0], [180.0, 360.0]],
            "units": "degrees_east",
            "standard_name": "longitude",
            "axis": "X",
        },
    ]


def time_axis():
    return {
        "name": "time",
        "values": [15.0, 45.0],
        "bounds": [[0.0, 30.0], [30.0, 60.0]],
        "units": "days since 2000-01-01",
        "standard_name": "time",
        "axis": "T",
    }


class Cmor4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = cmip7_project()

    def test_writes_basic_ocean_surface_temperature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = dataset_info(Path(tmp_dir))
            variable = {
                "name": "tos_tavg-u-hxy-sea",
                "table_id": "ocean",
                "missing_value": np.float32(1.0e20),
            }
            axes = [time_axis(), *horizontal_axes()]
            data = np.arange(8, dtype="f4").reshape(2, 2, 2)

            result = cmor4.cmorize(
                info, variable, axes, data, project=self.project
            )

            self.assertEqual(
                result.path.name,
                "tos_tavg-u-hxy-sea_mon_glb_g999_DUMMY-MODEL_amip_"
                "r9i1p1f3_200001-200002.nc",
            )
            self.assertIn(
                "CMIP7/CMIP/CCCma/DUMMY-MODEL/amip/r9i1p1f3/glb/mon/"
                "tos/tavg-u-hxy-sea/g999/v20200101",
                str(result.path),
            )

            with xr.open_dataset(result.path, decode_times=False) as opened:
                ds = opened.load()

            self.assertEqual(ds["tos"].dims, ("time", "lat", "lon"))
            self.assertEqual(ds["tos"].attrs["units"], "degC")
            self.assertEqual(
                ds["tos"].attrs["branded_variable_name"], "tos_tavg-u-hxy-sea"
            )
            self.assertEqual(ds["lat"].attrs["bounds"], "lat_bnds")
            self.assertEqual(ds.attrs["variant_label"], "r9i1p1f3")

    def test_scalar_height_and_pressure_level_patterns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = dataset_info(Path(tmp_dir))
            axes = [
                time_axis(),
                {
                    "name": "height2m",
                    "values": [2.0],
                    "units": "m",
                    "standard_name": "height",
                    "positive": "up",
                    "scalar": True,
                },
                *horizontal_axes(),
            ]
            variable = {
                "name": "tas_tavg-h2m-hxy-u",
                "table_id": "atmos",
            }

            ds = cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((2, 2, 2), dtype="f4"),
                project=self.project,
            )

            self.assertEqual(ds["tas"].dims, ("time", "lat", "lon"))
            self.assertEqual(ds["height"].shape, ())
            self.assertEqual(ds["height"].attrs["units"], "m")
            self.assertEqual(ds["tas"].attrs["coordinates"], "height")

            plev_axes = [
                time_axis(),
                {
                    "name": "plev19",
                    "values": [100000.0, 50000.0],
                    "units": "Pa",
                    "positive": "down",
                },
                *horizontal_axes(),
            ]
            plev_variable = {
                "name": "ta_tavg-p19-hxy-air",
                "table_id": "atmos",
            }
            plev_ds = cmor4.create_dataset(
                info,
                plev_variable,
                plev_axes,
                np.ones((2, 2, 2, 2), dtype="f4"),
                project=self.project,
            )

            self.assertEqual(
                plev_ds["ta"].dims, ("time", "plev", "lat", "lon")
            )
            self.assertEqual(plev_ds["plev"].attrs["positive"], "down")

    def test_hybrid_sigma_zfactors_are_written(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = dataset_info(Path(tmp_dir))
            axes = [
                time_axis(),
                {
                    "name": "standard_hybrid_sigma",
                    "values": [0.1, 0.9],
                    "bounds": [[0.0, 0.5], [0.5, 1.0]],
                },
                *horizontal_axes(),
            ]
            variable = {
                "name": "tnhusscpbl_tavg-al-hxy-u",
                "table_id": "atmos",
            }
            zfactors = [
                {
                    "name": "a",
                    "values": [0.1, 0.9],
                    "bounds": [[0.0, 0.5], [0.5, 1.0]],
                },
                {
                    "name": "b",
                    "values": [0.9, 0.1],
                    "bounds": [[1.0, 0.5], [0.5, 0.0]],
                },
                {"name": "p0", "values": 100000.0},
                {
                    "name": "ps",
                    "values": np.ones((2, 2, 2), dtype="f4") * 99000.0,
                },
            ]

            ds = cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((2, 2, 2, 2), dtype="f4"),
                zfactors=zfactors,
                project=self.project,
            )

            self.assertEqual(
                ds["tnhusscpbl"].dims, ("time", "lev", "lat", "lon")
            )
            self.assertEqual(
                ds["lev"].attrs["formula_terms"], "p0: p0 a: a b: b ps: ps"
            )
            self.assertEqual(
                ds["lev"].attrs["formula"],
                "p = a*p0 + b*ps",
            )
            self.assertEqual(
                ds["a"].attrs["long_name"],
                "vertical coordinate formula term: a",
            )
            self.assertEqual(ds["p0"].attrs["units"], "Pa")
            self.assertEqual(ds["a_bnds"].dims, ("lev", "bnds"))
            self.assertEqual(ds["ps"].dims, ("time", "lat", "lon"))

    def test_land_use_fraction_time_point_axis(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = dataset_info(Path(tmp_dir))
            axes = [
                {
                    "name": "time1",
                    "values": [15.0, 45.0],
                    "units": "days since 2000-01-01",
                    "standard_name": "time",
                    "axis": "T",
                },
                {
                    "name": "landuse",
                    "values": [
                        "primary_and_secondary_land",
                        "pastures",
                        "crops",
                        "urban",
                    ],
                    "units": "1",
                },
                *horizontal_axes(),
            ]
            variable = {
                "name": "fracLut_tpt-u-hxy-u",
                "table_id": "land",
            }

            ds = cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((2, 4, 2, 2), dtype="f4"),
                project=self.project,
            )

            self.assertEqual(
                ds["fracLut"].dims, ("time", "landuse", "lat", "lon")
            )
            self.assertEqual(
                ds["landuse"].values.tolist(),
                ["primary_and_secondary_land", "pastures", "crops", "urban"],
            )
            self.assertEqual(ds["fracLut"].attrs["temporal_label"], "tpt")

    def test_basin_axis_and_projected_grid_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = dataset_info(Path(tmp_dir))
            basin_axes = [
                time_axis(),
                {
                    "name": "basin",
                    "values": [
                        "atlantic_arctic_ocean",
                        "indian_pacific_ocean",
                        "global_ocean",
                    ],
                    "auxiliary_name": "sector",
                    "auxiliary_attrs": {"long_name": "ocean basin"},
                },
                {
                    "name": "latitude",
                    "values": [-30.0, 30.0],
                    "bounds": [[-60.0, 0.0], [0.0, 60.0]],
                    "units": "degrees_north",
                },
            ]
            basin_variable = {
                "name": "htovgyre_tavg-u-hyb-sea",
                "table_id": "ocean",
            }

            basin_ds = cmor4.create_dataset(
                info,
                basin_variable,
                basin_axes,
                np.ones((2, 3, 2), dtype="f4"),
                project=self.project,
            )

            self.assertEqual(
                basin_ds["htovgyre"].dims, ("time", "basin", "lat")
            )
            self.assertEqual(
                basin_ds["htovgyre"].attrs["coordinates"], "sector"
            )
            self.assertEqual(
                basin_ds["sector"].values.tolist(),
                [
                    "atlantic_arctic_ocean",
                    "indian_pacific_ocean",
                    "global_ocean",
                ],
            )

            grid_axes = [
                time_axis(),
                *horizontal_axes(),
            ]
            grid_variable = {
                "name": "siconc_tavg-u-hxy-u",
                "table_id": "seaIce",
            }
            grid = {
                "mapping_name": "lambert_azimuthal_equal_area",
                "params": {
                    "latitude_of_projection_origin": [90.0, "degrees_north"]
                },
            }

            grid_ds = cmor4.create_dataset(
                info,
                grid_variable,
                grid_axes,
                np.ones((2, 2, 2), dtype="f4"),
                grid=grid,
                project=self.project,
            )

            self.assertEqual(grid_ds["siconc"].dims, ("time", "lat", "lon"))
            self.assertEqual(grid_ds["siconc"].attrs["grid_mapping"], "crs")
            self.assertEqual(
                grid_ds["crs"].attrs["grid_mapping_name"],
                "lambert_azimuthal_equal_area",
            )


if __name__ == "__main__":
    unittest.main()
