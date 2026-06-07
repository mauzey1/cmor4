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


def horizontal_axes(project=None):
    if project is not None:
        return [
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


def time_axis(project=None):
    if project is not None:
        return project.axis(
            "time",
            values=[15.0, 45.0],
            bounds=[[0.0, 30.0], [30.0, 60.0]],
            units="days since 2000-01-01",
        )
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
        variable_id, labels = cmor4.Variable(
            name="tas_tavg-h2m-hxy-u"
        ).names()
        self.assertEqual(variable_id, "tas")
        self.assertEqual(labels["branded_name"], "tas_tavg-h2m-hxy-u")
        self.assertEqual(labels["vertical_label"], "h2m")

        with tempfile.TemporaryDirectory() as tmp_dir:
            prepared_info = self.project.dataset_info(
                dataset_info(Path(tmp_dir))
            )
            prepared_variable = self.project.variable(
                "tos_tavg-u-hxy-sea", table_id="ocean"
            )
        prepared_axis = axis.merge_table_entry(self.project)
        prepared_zfactor = zfactor.merge_table_entry(self.project)

        self.assertIsInstance(prepared_info, cmor4.DatasetInfo)
        self.assertIsInstance(prepared_variable, cmor4.Variable)
        self.assertIsInstance(prepared_axis, cmor4.Axis)
        self.assertIsInstance(prepared_zfactor, cmor4.ZFactor)

    def test_dataset_info_prepares_project_global_attributes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            variable = self.project.variable(
                "tos_tavg-u-hxy-sea", table_id="ocean"
            )
            info = self.project.dataset_info(dataset_info(Path(tmp_dir)))

        attrs = info.global_attributes(variable)

        self.assertEqual(info["source_id"], "DUMMY-MODEL")
        self.assertEqual(info.variant_label(), "r9i1p1f3")
        self.assertEqual(attrs["variable_id"], "tos")
        self.assertEqual(attrs["branded_variable"], "tos_tavg-u-hxy-sea")
        self.assertEqual(attrs["frequency"], "mon")
        self.assertIn("license", attrs)
        self.assertIn("tracking_id", attrs)

    def test_project_builds_table_backed_metadata_classes(self):
        variable = self.project.variable(
            "tos_tavg-u-hxy-sea",
            table_id="ocean",
        )
        axis = self.project.axis(
            "latitude",
            values=[-45.0, 45.0],
        )
        grid = self.project.grid("sample_user_mapping")
        zfactor = self.project.zfactor(
            "p0",
            values=100000.0,
        )

        self.assertEqual(variable.id, "tos")
        self.assertEqual(variable.units, "degC")
        self.assertEqual(
            variable.dimensions, ("time", "latitude", "longitude")
        )
        self.assertEqual(axis.out_name, "lat")
        self.assertEqual(axis.units, "degrees_north")
        self.assertEqual(axis.axis, "Y")
        self.assertEqual(grid.coordinates, ["rlon", "rlat"])
        self.assertIn("false_easting", grid.params)
        self.assertEqual(zfactor.units, "Pa")
        self.assertEqual(
            zfactor.standard_name,
            "reference_air_pressure_for_atmosphere_vertical_coordinate",
        )

    def test_writes_basic_ocean_surface_temperature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            variable = self.project.variable(
                "tos_tavg-u-hxy-sea",
                table_id="ocean",
                missing_value=np.float32(1.0e20),
            )
            info = self.project.dataset_info(dataset_info(Path(tmp_dir)))
            axes = [time_axis(self.project), *horizontal_axes(self.project)]
            data = np.arange(8, dtype="f4").reshape(2, 2, 2)

            result = cmor4.cmorize(info, variable, axes, data)

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

    def test_string_from_template_uses_global_attrs_and_special_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            variable = cmor4.Variable(name="sample", dimensions=["time"])
            info = cmor4.DatasetInfo.from_mapping(
                dataset_info(Path(tmp_dir)),
            )
            ds = cmor4.create_dataset(
                info, variable, [time_axis()], np.ones(2, dtype="f4")
            )

            self.assertEqual(
                cmor4.string_from_template(
                    "<variable_id>_<source_id>_<time_range>_<version>",
                    info,
                    variable,
                    ds=ds,
                ),
                "sample_DUMMY-MODEL_200001-200002_v20200101",
            )

            path_info = cmor4.DatasetInfo.from_mapping(
                {
                    **info.to_dict(),
                    "output_path_template": "<activity_id>",
                    "output_file_template": (
                        "<variable_id>_<frequency>_<time_range>"
                    ),
                },
            )
            self.assertEqual(
                cmor4.build_output_path(path_info, variable, ds=ds).name,
                "sample_mon_200001-200002.nc",
            )

    def test_scalar_height_and_pressure_level_patterns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_info = dataset_info(Path(tmp_dir))
            axes = [
                time_axis(self.project),
                self.project.axis(
                    "height2m",
                    values=[2.0],
                    scalar=True,
                ),
                *horizontal_axes(self.project),
            ]
            variable = self.project.variable(
                "tas_tavg-h2m-hxy-u", table_id="atmos"
            )
            info = self.project.dataset_info(base_info)

            ds = cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((2, 2, 2), dtype="f4"),
            )

            self.assertEqual(ds["tas"].dims, ("time", "lat", "lon"))
            self.assertEqual(ds["height"].shape, ())
            self.assertEqual(ds["height"].attrs["units"], "m")
            self.assertEqual(ds["tas"].attrs["coordinates"], "height")

            plev_axes = [
                time_axis(self.project),
                self.project.axis(
                    "plev19",
                    values=[100000.0, 50000.0],
                    units="Pa",
                    positive="down",
                ),
                *horizontal_axes(self.project),
            ]
            plev_variable = self.project.variable(
                "ta_tavg-p19-hxy-air",
                table_id="atmos",
            )
            plev_info = self.project.dataset_info(base_info)
            plev_ds = cmor4.create_dataset(
                plev_info,
                plev_variable,
                plev_axes,
                np.ones((2, 2, 2, 2), dtype="f4"),
            )

            self.assertEqual(
                plev_ds["ta"].dims, ("time", "plev", "lat", "lon")
            )
            self.assertEqual(plev_ds["plev"].attrs["positive"], "down")

    def test_hybrid_sigma_zfactors_are_written(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_info = dataset_info(Path(tmp_dir))
            axes = [
                time_axis(self.project),
                self.project.axis(
                    "standard_hybrid_sigma",
                    values=[0.1, 0.9],
                    bounds=[[0.0, 0.5], [0.5, 1.0]],
                ),
                *horizontal_axes(self.project),
            ]
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            info = self.project.dataset_info(base_info)
            zfactors = [
                self.project.zfactor(
                    "a",
                    values=[0.1, 0.9],
                    bounds=[[0.0, 0.5], [0.5, 1.0]],
                ),
                self.project.zfactor(
                    "b",
                    values=[0.9, 0.1],
                    bounds=[[1.0, 0.5], [0.5, 0.0]],
                ),
                self.project.zfactor("p0", values=100000.0),
                self.project.zfactor(
                    "ps",
                    values=np.ones((2, 2, 2), dtype="f4") * 99000.0,
                ),
            ]

            ds = cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((2, 2, 2, 2), dtype="f4"),
                zfactors=zfactors,
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
            base_info = dataset_info(Path(tmp_dir))
            axes = [
                self.project.axis(
                    "time1",
                    values=[15.0, 45.0],
                    units="days since 2000-01-01",
                ),
                self.project.axis(
                    "landuse",
                    values=[
                        "primary_and_secondary_land",
                        "pastures",
                        "crops",
                        "urban",
                    ],
                    units="1",
                ),
                *horizontal_axes(self.project),
            ]
            variable = self.project.variable(
                "fracLut_tpt-u-hxy-u",
                table_id="land",
            )
            info = self.project.dataset_info(base_info)

            ds = cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((2, 4, 2, 2), dtype="f4"),
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
            base_info = dataset_info(Path(tmp_dir))
            basin_axes = [
                time_axis(self.project),
                self.project.axis(
                    "basin",
                    values=[
                        "atlantic_arctic_ocean",
                        "indian_pacific_ocean",
                        "global_ocean",
                    ],
                    auxiliary_name="sector",
                    auxiliary_attrs={"long_name": "ocean basin"},
                ),
                self.project.axis(
                    "latitude",
                    values=[-30.0, 30.0],
                    bounds=[[-60.0, 0.0], [0.0, 60.0]],
                ),
            ]
            basin_variable = self.project.variable(
                "htovgyre_tavg-u-hyb-sea",
                table_id="ocean",
            )
            basin_info = self.project.dataset_info(base_info)

            basin_ds = cmor4.create_dataset(
                basin_info,
                basin_variable,
                basin_axes,
                np.ones((2, 3, 2), dtype="f4"),
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
                time_axis(self.project),
                *horizontal_axes(self.project),
            ]
            grid_variable = self.project.variable(
                "siconc_tavg-u-hxy-u",
                table_id="seaIce",
            )
            grid_info = self.project.dataset_info(base_info)
            grid = cmor4.Grid(
                mapping_name="lambert_azimuthal_equal_area",
                params={
                    "latitude_of_projection_origin": [90.0, "degrees_north"]
                },
            )

            grid_ds = cmor4.create_dataset(
                grid_info,
                grid_variable,
                grid_axes,
                np.ones((2, 2, 2), dtype="f4"),
                grid=grid,
            )

            self.assertEqual(grid_ds["siconc"].dims, ("time", "lat", "lon"))
            self.assertEqual(grid_ds["siconc"].attrs["grid_mapping"], "crs")
            self.assertEqual(
                grid_ds["crs"].attrs["grid_mapping_name"],
                "lambert_azimuthal_equal_area",
            )

    def test_grid_dimensions_override_table_variable_dimensions(self):
        axes = [
            cmor4.Axis(name="time", values=[15.0]),
            cmor4.Axis(name="x", values=[0.0, 1.0]),
            cmor4.Axis(name="y", values=[2.0, 3.0]),
            cmor4.Axis(
                name="latitude",
                values=[[10.0, 20.0], [30.0, 40.0]],
                dimensions=["x", "y"],
                auxiliary=True,
            ),
            cmor4.Axis(
                name="longitude",
                values=[[100.0, 110.0], [120.0, 130.0]],
                dimensions=["x", "y"],
                auxiliary=True,
            ),
        ]
        variable = cmor4.Variable(
            name="sample",
            dimensions=["time", "latitude", "longitude"],
        )
        info = cmor4.DatasetInfo.from_mapping(
            {"frequency": "mon"},
        )
        grid = cmor4.Grid(dimensions=["time", "x", "y"])

        ds = cmor4.create_dataset(
            info,
            variable,
            axes,
            np.ones((1, 2, 2), dtype="f4"),
            grid=grid,
        )

        self.assertEqual(ds["sample"].dims, ("time", "x", "y"))

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
                info = cmor4.DatasetInfo.from_mapping(
                    dict(base_info, frequency=frequency),
                )
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
                    cmor4.build_output_path(info, variable, ds=ds).name,
                    expected_name,
                )

    def test_climatology_time_axis_uses_cmor_bounds_and_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_info = dataset_info(Path(tmp_dir))
            raw_info.update(
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
            info = cmor4.DatasetInfo.from_mapping(raw_info)

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
                cmor4.build_output_path(info, variable, ds=ds).name,
                "co2_mon_201801-201802.nc",
            )


if __name__ == "__main__":
    unittest.main()
