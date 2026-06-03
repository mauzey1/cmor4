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
        cmor4.Axis(
            name="latitude",
            values=[-45.0, 45.0],
            bounds=[[-90.0, 0.0], [0.0, 90.0]],
            units="degrees_north",
            standard_name="latitude",
            axis="Y",
        ),
        cmor4.Axis(
            name="longitude",
            values=[90.0, 270.0],
            bounds=[[0.0, 180.0], [180.0, 360.0]],
            units="degrees_east",
            standard_name="longitude",
            axis="X",
        ),
    ]


def time_axis():
    return cmor4.Axis(
        name="time",
        values=[15.0, 45.0],
        bounds=[[0.0, 30.0], [30.0, 60.0]],
        units="days since 2000-01-01",
        standard_name="time",
        axis="T",
    )


class Cmor4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = cmip7_project()

    def test_metadata_classes_own_table_preparation_behavior(self):
        variable = cmor4.Variable(
            name="sample",
            dimensions=["time"],
            attrs={"units": "1"},
            extra={"custom_key": "custom_value"},
        )
        axis = cmor4.Axis(name="time", values=np.arange(2))
        zfactor = cmor4.ZFactor(name="p0", values=100000.0)

        updated = variable.updated(dimensions=["time", "lat", "lon"])
        self.assertEqual(updated.name, "sample")
        self.assertEqual(updated.dimensions, ["time", "lat", "lon"])
        self.assertEqual(updated.extra["custom_key"], "custom_value")
        self.assertEqual(axis.name, "time")
        self.assertEqual(zfactor.values, 100000.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, prepared_variable = self.project.prepare_inputs(
                dataset_info(Path(tmp_dir)),
                cmor4.Variable(name="tos_tavg-u-hxy-sea", table_id="ocean"),
            )
        prepared_axis = axis.merge_table_entry(self.project)
        prepared_zfactor = zfactor.merge_table_entry(self.project)

        self.assertIsInstance(prepared_variable, cmor4.Variable)
        self.assertIsInstance(prepared_axis, cmor4.Axis)
        self.assertIsInstance(prepared_zfactor, cmor4.ZFactor)

    def test_writes_basic_ocean_surface_temperature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = dataset_info(Path(tmp_dir))
            variable = cmor4.Variable(
                name="tos_tavg-u-hxy-sea",
                table_id="ocean",
                missing_value=np.float32(1.0e20),
            )
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
                cmor4.Axis(
                    name="height2m",
                    values=[2.0],
                    units="m",
                    standard_name="height",
                    positive="up",
                    scalar=True,
                ),
                *horizontal_axes(),
            ]
            variable = cmor4.Variable(
                name="tas_tavg-h2m-hxy-u",
                table_id="atmos",
            )

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
                cmor4.Axis(
                    name="plev19",
                    values=[100000.0, 50000.0],
                    units="Pa",
                    positive="down",
                ),
                *horizontal_axes(),
            ]
            plev_variable = cmor4.Variable(
                name="ta_tavg-p19-hxy-air",
                table_id="atmos",
            )
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
                cmor4.Axis(
                    name="standard_hybrid_sigma",
                    values=[0.1, 0.9],
                    bounds=[[0.0, 0.5], [0.5, 1.0]],
                ),
                *horizontal_axes(),
            ]
            variable = cmor4.Variable(
                name="tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            zfactors = [
                cmor4.ZFactor(
                    name="a",
                    values=[0.1, 0.9],
                    bounds=[[0.0, 0.5], [0.5, 1.0]],
                ),
                cmor4.ZFactor(
                    name="b",
                    values=[0.9, 0.1],
                    bounds=[[1.0, 0.5], [0.5, 0.0]],
                ),
                cmor4.ZFactor(name="p0", values=100000.0),
                cmor4.ZFactor(
                    name="ps",
                    values=np.ones((2, 2, 2), dtype="f4") * 99000.0,
                ),
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
                cmor4.Axis(
                    name="time1",
                    values=[15.0, 45.0],
                    units="days since 2000-01-01",
                    standard_name="time",
                    axis="T",
                ),
                cmor4.Axis(
                    name="landuse",
                    values=[
                        "primary_and_secondary_land",
                        "pastures",
                        "crops",
                        "urban",
                    ],
                    units="1",
                ),
                *horizontal_axes(),
            ]
            variable = cmor4.Variable(
                name="fracLut_tpt-u-hxy-u",
                table_id="land",
            )

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
                cmor4.Axis(
                    name="basin",
                    values=[
                        "atlantic_arctic_ocean",
                        "indian_pacific_ocean",
                        "global_ocean",
                    ],
                    auxiliary_name="sector",
                    auxiliary_attrs={"long_name": "ocean basin"},
                ),
                cmor4.Axis(
                    name="latitude",
                    values=[-30.0, 30.0],
                    bounds=[[-60.0, 0.0], [0.0, 60.0]],
                    units="degrees_north",
                ),
            ]
            basin_variable = cmor4.Variable(
                name="htovgyre_tavg-u-hyb-sea",
                table_id="ocean",
            )

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
            grid_variable = cmor4.Variable(
                name="siconc_tavg-u-hxy-u",
                table_id="seaIce",
            )
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

    def test_filename_time_ranges_follow_cmor_frequency_formats(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_info = dataset_info(Path(tmp_dir))
            base_info.update(
                {
                    "output_file_template": (
                        "<variable_id>_<frequency>_<source_id>_"
                        "<variant_label>"
                    ),
                    "output_path_template": "<activity_id>",
                    "version": "v20200101",
                }
            )
            variable = cmor4.Variable(name="sample", dimensions=["time"])

            cases = [
                (
                    "yr",
                    [182.5, 547.5],
                    "days since 2000-01-01 00:00:00",
                    "sample_yr_DUMMY-MODEL_r9i1p1f3_2000-2001.nc",
                ),
                (
                    "day",
                    [0.9999999, 1.9999999],
                    "days since 1960-01-01 00:00:00",
                    "sample_day_DUMMY-MODEL_r9i1p1f3_19600102-19600103.nc",
                ),
                (
                    "1hr",
                    [12.6, 77.4],
                    "minutes since 2000-01-01 00:00:00",
                    "sample_1hr_DUMMY-MODEL_r9i1p1f3_"
                    "200001010013-200001010117.nc",
                ),
                (
                    "subhr",
                    [750.4, 2250.6],
                    "seconds since 2000-01-01 00:00:00",
                    "sample_subhr_DUMMY-MODEL_r9i1p1f3_"
                    "20000101001230-20000101003731.nc",
                ),
            ]
            for frequency, values, units, expected_name in cases:
                info = dict(base_info, frequency=frequency)
                axes = [
                    cmor4.Axis(
                        name="time",
                        values=values,
                        units=units,
                        standard_name="time",
                        axis="T",
                    )
                ]
                ds = cmor4.create_dataset(
                    info, variable, axes, np.ones(2, dtype="f4")
                )

                self.assertEqual(
                    cmor4.build_output_path(info, variable, ds).name,
                    expected_name,
                )

    def test_climatology_time_axis_uses_cmor_bounds_and_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = dataset_info(Path(tmp_dir))
            info.update(
                {
                    "calendar": "360_day",
                    "frequency": "mon",
                    "output_file_template": "<variable_id>_<frequency>",
                    "output_path_template": "<activity_id>",
                }
            )
            axes = [
                cmor4.Axis(
                    name="time2",
                    values=[15.0, 45.0],
                    bounds=[[0.0, 31.0], [31.0, 60.0]],
                    units="days since 2018",
                    standard_name="time",
                    axis="T",
                    climatology="yes",
                    out_name="time",
                )
            ]
            variable = cmor4.Variable(
                name="co2_tclm-u-hm-u", dimensions=["time2"]
            )

            ds = cmor4.create_dataset(
                info, variable, axes, np.ones(2, dtype="f4")
            )

            self.assertEqual(
                ds["time"].attrs["climatology"], "climatology_bnds"
            )
            self.assertNotIn("bounds", ds["time"].attrs)
            self.assertEqual(
                ds["climatology_bnds"].dims, ("time", "bnds")
            )
            self.assertEqual(
                cmor4.build_output_path(info, variable, ds).name,
                "co2_mon_201801-201802.nc",
            )


if __name__ == "__main__":
    unittest.main()
