from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import xarray as xr

import cmor4
import cmor4._axis_validation as axis_validation
import cmor4._time_utils as time_utils
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
        prepared_axis = self.project.axis("time", values=np.arange(2))
        prepared_zfactor = self.project.zfactor("p0", values=100000.0)

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
        self.assertNotIn("type", axis.extra)
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

    def test_axis_validation_matches_cmor_time_and_bounds_checks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            variable = self.project.variable(
                "tos_tavg-u-hxy-sea",
                table_id="ocean",
            )
            info = self.project.dataset_info(dataset_info(Path(tmp_dir)))
            axes = [
                self.project.axis(
                    "time",
                    values=[15.0, 45.0],
                    bounds=[[0.0, 31.0], [31.0, 60.0]],
                    units="days since 2000-01-01",
                ),
                *horizontal_axes(self.project),
            ]

            with self.assertWarnsRegex(RuntimeWarning, "bound midpoints"):
                ds = cmor4.create_dataset(
                    info,
                    variable,
                    axes,
                    np.ones((2, 2, 2), dtype="f4"),
                )

            np.testing.assert_allclose(ds["time"].values, [15.5, 45.5])

            bad_interval_axes = [
                self.project.axis(
                    "time",
                    values=[15.0, 45.0, 90.0],
                    bounds=[[0.0, 30.0], [30.0, 60.0], [60.0, 120.0]],
                    units="days since 2000-01-01",
                ),
                *horizontal_axes(self.project),
            ]
            with self.assertRaisesRegex(
                cmor4.AxisValidationError,
                "Time interval mismatch detected",
            ):
                cmor4.create_dataset(
                    info,
                    variable,
                    bad_interval_axes,
                    np.ones((3, 2, 2), dtype="f4"),
                )

    def test_axis_validation_rejects_bad_values_and_accepts_flat_bounds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            variable = self.project.variable(
                "tos_tavg-u-hxy-sea",
                table_id="ocean",
            )
            info = self.project.dataset_info(dataset_info(Path(tmp_dir)))

            flat_bound_axes = [
                time_axis(self.project),
                self.project.axis(
                    "latitude",
                    values=[-45.0, 45.0],
                    bounds=[-90.0, 0.0, 90.0],
                ),
                self.project.axis(
                    "longitude",
                    values=[90.0, 270.0],
                    bounds=[0.0, 180.0, 360.0],
                ),
            ]
            ds = cmor4.create_dataset(
                info,
                variable,
                flat_bound_axes,
                np.ones((2, 2, 2), dtype="f4"),
            )
            self.assertEqual(ds["lat_bnds"].shape, (2, 2))

            with self.assertRaisesRegex(
                cmor4.AxisValidationError,
                "valid_min",
            ):
                self.project.axis(
                    "latitude",
                    values=[-95.0, 45.0],
                    bounds=[[-100.0, 0.0], [0.0, 90.0]],
                )

            bad_lat_axes = [
                time_axis(self.project),
                cmor4.Axis(
                    name="latitude",
                    values=[-95.0, 45.0],
                    bounds=[[-100.0, 0.0], [0.0, 90.0]],
                    units="degrees_north",
                    standard_name="latitude",
                    axis="Y",
                    valid_min=-90.0,
                    valid_max=90.0,
                    out_name="lat",
                ),
                self.project.axis(
                    "longitude",
                    values=[90.0, 270.0],
                    bounds=[[0.0, 180.0], [180.0, 360.0]],
                ),
            ]
            with self.assertRaisesRegex(
                cmor4.AxisValidationError,
                "valid_min",
            ):
                cmor4.create_dataset(
                    info,
                    variable,
                    bad_lat_axes,
                    np.ones((2, 2, 2), dtype="f4"),
                )

    def test_time_interval_uses_cftime_with_numeric_fallback(self):
        with mock.patch.object(
            time_utils.cftime,
            "num2date",
            wraps=time_utils.cftime.num2date,
        ) as num2date:
            intervals = axis_validation._time_interval_days(
                np.asarray([0.0, 1.0, 2.0]),
                "months since 2001-01-01",
                "360_day",
            )

        self.assertTrue(num2date.called)
        np.testing.assert_allclose(intervals, [30.0, 30.0])

        with mock.patch.object(
            time_utils.cftime,
            "num2date",
            side_effect=ValueError("unsupported units"),
        ):
            fallback_intervals = axis_validation._time_interval_days(
                np.asarray([0.0, 1.0]),
                "years since 2001-01-01",
                "standard",
            )

        np.testing.assert_allclose(fallback_intervals, [365.0])

    def test_variable_value_validation_matches_cmor_nan_and_range_checks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = cmor4.DatasetInfo.from_mapping(dataset_info(Path(tmp_dir)))
            axis = cmor4.Axis(name="time", values=[0.0, 1.0, 2.0])
            variable = cmor4.Variable(
                name="sample",
                dimensions=["time"],
                table_id="Amon",
                valid_min=0.0,
                valid_max=10.0,
                missing_value=np.float32(1.0e20),
            )

            with self.assertRaisesRegex(
                cmor4.VariableValidationError,
                "1 values were NaNs",
            ):
                cmor4.create_dataset(
                    info,
                    variable,
                    [axis],
                    np.asarray([1.0, np.nan, 2.0], dtype="f4"),
                )

            with self.assertWarnsRegex(
                RuntimeWarning,
                "lower than minimum valid value",
            ):
                ds = cmor4.create_dataset(
                    info,
                    variable,
                    [axis],
                    np.asarray([1.0, -1.0, 1.0e20], dtype="f4"),
                )

            self.assertEqual(ds["sample"].shape, (3,))

    def test_variable_absolute_mean_validation_matches_cmor_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = cmor4.DatasetInfo.from_mapping(dataset_info(Path(tmp_dir)))
            axis = cmor4.Axis(name="time", values=[0.0, 1.0, 2.0])
            variable = cmor4.Variable(
                name="sample",
                dimensions=["time"],
                table_id="Amon",
                ok_min_mean_abs=10.0,
                ok_max_mean_abs=20.0,
            )

            with self.assertWarnsRegex(
                RuntimeWarning,
                "lower than minimum allowed",
            ):
                cmor4.create_dataset(
                    info,
                    variable,
                    [axis],
                    np.asarray([5.0, 5.0, 5.0], dtype="f4"),
                )

            with self.assertRaisesRegex(
                cmor4.VariableValidationError,
                "greater by more than an order of magnitude",
            ):
                cmor4.create_dataset(
                    info,
                    variable,
                    [axis],
                    np.asarray([250.0, 250.0, 250.0], dtype="f4"),
                )

    def test_zfactor_value_validation_uses_cmor_variable_checks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = cmor4.DatasetInfo.from_mapping(dataset_info(Path(tmp_dir)))
            axis = cmor4.Axis(name="time", values=[0.0, 1.0, 2.0])
            variable = cmor4.Variable(name="sample", dimensions=["time"])
            zfactor = cmor4.ZFactor(
                name="ps",
                values=[1.0, np.nan, 2.0],
                dimensions=["time"],
                valid_min=0.0,
                table_entry="formula_terms",
            )

            with self.assertRaisesRegex(
                cmor4.VariableValidationError,
                "ps.*1 values were NaNs",
            ):
                cmor4.create_dataset(
                    info,
                    variable,
                    [axis],
                    np.ones(3, dtype="f4"),
                    zfactors=[zfactor],
                )

            with self.assertWarnsRegex(
                RuntimeWarning,
                "ps.*lower than minimum valid value",
            ):
                cmor4.create_dataset(
                    info,
                    variable,
                    [axis],
                    np.ones(3, dtype="f4"),
                    zfactors=[zfactor.updated(values=[1.0, -1.0, 2.0])],
                )

    def test_grid_mapping_parameters_match_cmor_range_checks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = cmor4.DatasetInfo.from_mapping(dataset_info(Path(tmp_dir)))
            variable = cmor4.Variable(
                name="sample",
                dimensions=["time", "y", "x"],
            )
            axes = [
                cmor4.Axis(name="time", values=[0.0]),
                cmor4.Axis(name="y", values=[0.0]),
                cmor4.Axis(name="x", values=[0.0]),
            ]
            grid = cmor4.Grid(
                mapping_name="lambert_azimuthal_equal_area",
                params={
                    "latitude_of_projection_origin": [100.0, "degrees_north"],
                    "longitude_of_projection_origin": [200.0, "degrees_east"],
                    "scale_factor_at_projection_origin": -1.0,
                    "false_easting": [10.0, "m"],
                },
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ds = cmor4.create_dataset(
                    info,
                    variable,
                    axes,
                    np.ones((1, 1, 1), dtype="f4"),
                    grid=grid,
                )

            messages = [str(item.message) for item in caught]
            self.assertTrue(
                any("between -90 and 90" in message for message in messages)
            )
            self.assertTrue(
                any("between -180 and 180" in message for message in messages)
            )
            self.assertTrue(
                any("must be positive" in message for message in messages)
            )
            self.assertNotIn(
                "latitude_of_projection_origin", ds["crs"].attrs
            )
            self.assertNotIn(
                "longitude_of_projection_origin", ds["crs"].attrs
            )
            self.assertNotIn(
                "scale_factor_at_projection_origin", ds["crs"].attrs
            )
            self.assertEqual(ds["crs"].attrs["false_easting"], 10.0)

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
            scalar_variable = self.project.variable(
                "tas_tavg-h2m-hxy-u",
                table_id="atmos",
            )
            scalar_axes = self.project.scalar_axes_for(scalar_variable)
            self.assertEqual(scalar_variable.id, "tas")
            self.assertEqual([axis.name for axis in scalar_axes], ["height2m"])
            self.assertEqual(scalar_axes[0].values, [2.0])

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
                    values=[
                        100000.0,
                        92500.0,
                        85000.0,
                        70000.0,
                        60000.0,
                        50000.0,
                        40000.0,
                        30000.0,
                        25000.0,
                        20000.0,
                        15000.0,
                        10000.0,
                        7000.0,
                        5000.0,
                        3000.0,
                        2000.0,
                        1000.0,
                        500.0,
                        100.0,
                    ],
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
                np.ones((2, 19, 2, 2), dtype="f4"),
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
            self.assertIn("p0", ds.variables)
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
        grid = cmor4.Grid(dimensions=["x", "y"])

        ds = cmor4.create_dataset(
            info,
            variable,
            axes,
            np.ones((1, 2, 2), dtype="f4"),
            grid=grid,
        )

        self.assertEqual(ds["sample"].dims, ("time", "x", "y"))

    def test_grid_owns_latitude_longitude_coordinates_and_vertices(self):
        """Test that Grid can own lat/lon arrays and create auxiliary
        coordinates.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            latitude = np.array([[10.0, 20.0], [30.0, 40.0]], dtype="f8")
            longitude = np.array([[100.0, 110.0], [120.0, 130.0]], dtype="f8")
            latitude_vertices = np.array(
                [
                    [[9.0, 11.0, 11.0, 9.0], [19.0, 21.0, 21.0, 19.0]],
                    [[29.0, 31.0, 31.0, 29.0], [39.0, 41.0, 41.0, 39.0]],
                ],
                dtype="f8",
            )
            longitude_vertices = np.array(
                [
                    [[99.0, 101.0, 101.0, 99.0], [109.0, 111.0, 111.0, 109.0]],
                    [[119.0, 121.0, 121.0, 119.0], [129.0, 131.0, 131.0, 129.0]],
                ],
                dtype="f8",
            )

            base_info = dataset_info(Path(tmp_dir))
            variable = self.project.variable(
                "tos_tavg-u-hxy-sea",
                table_id="ocean",
                missing_value=np.float32(1.0e20),
            )
            info = self.project.dataset_info(base_info)
            axes = [
                self.project.axis(
                    "time",
                    values=[15.0],
                    bounds=[0.0, 31.0],
                    units="days since 2000-01-01",
                ),
                cmor4.Axis(name="x", values=[0.0, 1.0]),
                cmor4.Axis(name="y", values=[2.0, 3.0]),
            ]
            grid = self.project.grid(
                dimensions=["x", "y"],
                mapping_name="lambert_azimuthal_equal_area",
                params={
                    "latitude_of_projection_origin": [90.0, "degrees_north"],
                    "longitude_of_projection_origin": [0.0, "degrees_east"],
                },
                latitude=latitude,
                longitude=longitude,
                latitude_vertices=latitude_vertices,
                longitude_vertices=longitude_vertices,
            )

            ds = cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((1, 2, 2), dtype="f4"),
                grid=grid,
            )

            # Verify lat/lon auxiliary coordinates were created
            self.assertIn("latitude", ds.coords)
            self.assertIn("longitude", ds.coords)
            self.assertEqual(ds["latitude"].dims, ("x", "y"))
            self.assertEqual(ds["longitude"].dims, ("x", "y"))
            np.testing.assert_array_equal(ds["latitude"].values, latitude)
            np.testing.assert_array_equal(ds["longitude"].values, longitude)

            # Verify vertices were created
            self.assertIn("vertices_latitude", ds.data_vars)
            self.assertIn("vertices_longitude", ds.data_vars)
            self.assertEqual(
                ds["vertices_latitude"].dims, ("x", "y", "vertices")
            )
            self.assertEqual(
                ds["vertices_longitude"].dims, ("x", "y", "vertices")
            )
            np.testing.assert_array_equal(
                ds["vertices_latitude"].values, latitude_vertices
            )
            np.testing.assert_array_equal(
                ds["vertices_longitude"].values, longitude_vertices
            )

            # Verify grid mapping was created
            self.assertIn("crs", ds.data_vars)
            self.assertEqual(
                ds["crs"].attrs["grid_mapping_name"],
                "lambert_azimuthal_equal_area",
            )

            # Verify coordinates attribute includes lat/lon
            self.assertIn("latitude", ds["tos"].attrs["coordinates"])
            self.assertIn("longitude", ds["tos"].attrs["coordinates"])

            # Verify grid coordinate table attributes are applied
            self.assertEqual(ds["latitude"].attrs["units"], "degrees_north")
            self.assertEqual(
                ds["latitude"].attrs["standard_name"], "latitude"
            )
            self.assertEqual(ds["longitude"].attrs["units"], "degrees_east")
            self.assertEqual(
                ds["longitude"].attrs["standard_name"], "longitude"
            )

            # Verify 'axis' attribute NOT present (grid coords are auxiliary)
            self.assertNotIn("axis", ds["latitude"].attrs)
            self.assertNotIn("axis", ds["longitude"].attrs)

            # Verify vertices also have table attributes
            self.assertEqual(
                ds["vertices_latitude"].attrs["units"], "degrees_north"
            )
            self.assertEqual(
                ds["vertices_longitude"].attrs["units"], "degrees_east"
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
