from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

import cmor4
from table_helpers import cmip7_project


def dataset_info(tmp_path: Path) -> dict[str, str]:
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


def horizontal_axes(project: cmor4.ProjectTables) -> list[cmor4.Axis]:
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


def shifted_horizontal_axes(project: cmor4.ProjectTables) -> list[cmor4.Axis]:
    return [
        project.axis(
            "latitude",
            values=[-30.0, 60.0],
            bounds=[[-75.0, 15.0], [15.0, 90.0]],
        ),
        project.axis(
            "longitude",
            values=[90.0, 270.0],
            bounds=[[0.0, 180.0], [180.0, 360.0]],
        ),
    ]


def time_axis(
    project: cmor4.ProjectTables,
    values: list[float] | None = None,
    bounds: list[list[float]] | None = None,
) -> cmor4.Axis:
    kwargs = {"units": "days since 2000-01-01"}
    if values is not None:
        kwargs["values"] = values
    if bounds is not None:
        kwargs["bounds"] = bounds
    return project.axis("time", **kwargs)


def curvilinear_grid(
    project: cmor4.ProjectTables,
    *,
    mapping_latitude: float = 90.0,
) -> cmor4.Grid:
    y_axis = project.axis(
        "y",
        values=[0.0, 10000.0, 20000.0],
        bounds=[[-5000.0, 5000.0], [5000.0, 15000.0], [15000.0, 25000.0]],
        units="m",
    )
    x_axis = project.axis(
        "x",
        values=[0.0, 10000.0, 20000.0, 30000.0],
        bounds=[
            [-5000.0, 5000.0],
            [5000.0, 15000.0],
            [15000.0, 25000.0],
            [25000.0, 35000.0],
        ],
        units="m",
    )
    latitude = np.array(
        [[10.0, 12.0, 14.0, 16.0], [20.0, 22.0, 24.0, 26.0], [30.0, 32.0, 34.0, 36.0]],
        dtype="f8",
    )
    longitude = np.array(
        [
            [100.0, 110.0, 120.0, 130.0],
            [102.0, 112.0, 122.0, 132.0],
            [104.0, 114.0, 124.0, 134.0],
        ],
        dtype="f8",
    )
    latitude_vertices = np.stack(
        (latitude - 0.5, latitude - 0.25, latitude + 0.5, latitude + 0.25),
        axis=-1,
    )
    longitude_vertices = np.stack(
        (longitude - 0.5, longitude + 0.5, longitude + 0.5, longitude - 0.5),
        axis=-1,
    )
    return project.grid(
        axes=[y_axis, x_axis],
        mapping_name="lambert_azimuthal_equal_area",
        params={
            "latitude_of_projection_origin": [mapping_latitude, "degrees_north"],
            "longitude_of_projection_origin": [0.0, "degrees_east"],
        },
        latitude=latitude,
        longitude=longitude,
        latitude_vertices=latitude_vertices,
        longitude_vertices=longitude_vertices,
    )


def hybrid_axis(project: cmor4.ProjectTables) -> cmor4.Axis:
    return project.axis(
        "standard_hybrid_sigma",
        values=[0.9, 0.1],
        bounds=[[1.0, 0.5], [0.5, 0.0]],
    )


def hybrid_zfactors(
    project: cmor4.ProjectTables,
    ps_values: np.ndarray,
    *,
    a_values: list[float] | None = None,
) -> list[cmor4.ZFactor]:
    a_vals = a_values or [0.9, 0.1]
    return [
        project.zfactor(
            "a",
            values=a_vals,
            bounds=[[1.0, 0.5], [0.5, 0.0]],
        ),
        project.zfactor(
            "b",
            values=[0.9, 0.1],
            bounds=[[1.0, 0.5], [0.5, 0.0]],
        ),
        project.zfactor("p0", values=100000.0),
        project.zfactor("ps", values=ps_values),
    ]


def equivalent(ds: xr.Dataset) -> xr.Dataset:
    normalized = ds.load().copy(deep=True)
    normalized.attrs.pop("creation_date", None)
    normalized.attrs.pop("history", None)
    normalized.attrs.pop("tracking_id", None)
    for variable in normalized.variables.values():
        variable.attrs.pop("_FillValue", None)
    return normalized


class DatasetWriterAppendModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project = cmip7_project()

    def test_append_compatible_data_matches_cmorize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            output_path = tmp_path / "writer.nc"
            existing_data = np.arange(8, dtype="f4").reshape(2, 2, 2)
            append_data = np.arange(8, 12, dtype="f4").reshape(1, 2, 2)
            full_data = np.concatenate([existing_data, append_data], axis=0)

            full_time = time_axis(
                self.project,
                values=[15.0, 45.0, 75.0],
                bounds=[[0.0, 30.0], [30.0, 60.0], [60.0, 90.0]],
            )
            expected, _ = cmor4.cmorize(
                info,
                variable,
                [full_time, *horizontal_axes(self.project)],
                full_data,
                path=tmp_path / "expected.nc",
            )

            with cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *horizontal_axes(self.project)],
                path=output_path,
            ) as writer:
                writer.write(
                    existing_data,
                    time_values=[15.0, 45.0],
                    time_bounds=[[0.0, 30.0], [30.0, 60.0]],
                )

            with cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *horizontal_axes(self.project)],
                path=output_path,
                existing="append",
            ) as writer:
                writer.write(
                    append_data,
                    time_values=[75.0],
                    time_bounds=[[60.0, 90.0]],
                )
                actual, path = writer.close()

            self.assertEqual(path, output_path)
            self.assertEqual(len(actual["time"]), 3)
            np.testing.assert_array_equal(actual["time"].values, [15.0, 45.0, 75.0])
            np.testing.assert_array_equal(actual["tos"].values, full_data)
            xr.testing.assert_identical(equivalent(actual), equivalent(expected))

    def test_append_rejects_incompatible_non_time_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            output_path = tmp_path / "writer.nc"

            with cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *horizontal_axes(self.project)],
                path=output_path,
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

            writer = cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *shifted_horizontal_axes(self.project)],
                path=output_path,
                existing="append",
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
            )

            with self.assertRaisesRegex(ValueError, "lat.*values differ"):
                writer.close()

    def test_append_rejects_non_monotonic_time_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            output_path = tmp_path / "writer.nc"

            with cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *horizontal_axes(self.project)],
                path=output_path,
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[45.0],
                    time_bounds=[[30.0, 60.0]],
                )

            writer = cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *horizontal_axes(self.project)],
                path=output_path,
                existing="append",
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
            )

            with self.assertRaisesRegex(ValueError, "monotonic"):
                writer.close()

    def test_append_rejects_non_contiguous_time_bounds_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            output_path = tmp_path / "writer.nc"

            with cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *horizontal_axes(self.project)],
                path=output_path,
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

            writer = cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *horizontal_axes(self.project)],
                path=output_path,
                existing="append",
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[75.0],
                time_bounds=[[60.0, 90.0]],
            )

            with self.assertRaisesRegex(ValueError, "contiguous"):
                writer.close()

    def test_append_requires_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            writer = cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *horizontal_axes(self.project)],
                path=tmp_path / "missing.nc",
                existing="append",
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
            )

            with self.assertRaises(FileNotFoundError):
                writer.close()

    def test_append_with_2d_curvilinear_grid_mapping_matches_cmorize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("hfls_tavg-u-hxy-u", table_id="atmos")
            grid = curvilinear_grid(self.project)
            output_path = tmp_path / "writer-grid.nc"
            existing_data = np.arange(24, dtype="f4").reshape(2, 3, 4)
            append_data = np.arange(24, 36, dtype="f4").reshape(1, 3, 4)
            full_data = np.concatenate([existing_data, append_data], axis=0)
            full_time = time_axis(
                self.project,
                values=[15.0, 45.0, 75.0],
                bounds=[[0.0, 30.0], [30.0, 60.0], [60.0, 90.0]],
            )
            _, expected_path = cmor4.cmorize(
                info,
                variable,
                [full_time],
                full_data,
                grid=grid,
                path=tmp_path / "expected-grid.nc",
            )

            with cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project)],
                path=output_path,
                grid=grid,
            ) as writer:
                writer.write(
                    existing_data,
                    time_values=[15.0, 45.0],
                    time_bounds=[[0.0, 30.0], [30.0, 60.0]],
                )

            with cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project)],
                path=output_path,
                grid=grid,
                existing="append",
            ) as writer:
                writer.write(
                    append_data,
                    time_values=[75.0],
                    time_bounds=[[60.0, 90.0]],
                )
                actual, _ = writer.close()

            with xr.open_dataset(
                expected_path,
                decode_times=False,
                mask_and_scale=False,
            ) as expected_open:
                expected = expected_open.load()
            xr.testing.assert_identical(equivalent(actual), equivalent(expected))
            self.assertEqual(actual["hfls"].attrs["grid_mapping"], "crs")
            np.testing.assert_array_equal(
                actual["latitude"].values,
                expected["latitude"].values,
            )
            np.testing.assert_array_equal(
                actual["vertices_longitude"].values,
                expected["vertices_longitude"].values,
            )

    def test_append_rejects_incompatible_grid_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("hfls_tavg-u-hxy-u", table_id="atmos")
            output_path = tmp_path / "writer-grid.nc"

            with cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project)],
                path=output_path,
                grid=curvilinear_grid(self.project, mapping_latitude=90.0),
            ) as writer:
                writer.write(
                    np.ones((1, 3, 4), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

            writer = cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project)],
                path=output_path,
                grid=curvilinear_grid(self.project, mapping_latitude=80.0),
                existing="append",
            )
            writer.write(
                np.ones((1, 3, 4), dtype="f4"),
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
            )

            with self.assertRaisesRegex(
                ValueError,
                "crs.*latitude_of_projection_origin",
            ):
                writer.close()

    def test_append_with_zfactors_matches_cmorize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            output_path = tmp_path / "writer-zfactor.nc"
            axes = [
                time_axis(self.project),
                hybrid_axis(self.project),
                *horizontal_axes(self.project),
            ]
            existing_data = np.arange(8, dtype="f4").reshape(1, 2, 2, 2)
            append_data = np.arange(8, 16, dtype="f4").reshape(1, 2, 2, 2)
            full_data = np.concatenate([existing_data, append_data], axis=0)
            existing_ps = np.full((1, 2, 2), 99000.0, dtype="f4")
            append_ps = np.full((1, 2, 2), 99100.0, dtype="f4")
            full_ps = np.concatenate([existing_ps, append_ps], axis=0)
            full_time = time_axis(
                self.project,
                values=[15.0, 45.0],
                bounds=[[0.0, 30.0], [30.0, 60.0]],
            )
            _, expected_path = cmor4.cmorize(
                info,
                variable,
                [full_time, hybrid_axis(self.project), *horizontal_axes(self.project)],
                full_data,
                zfactors=hybrid_zfactors(self.project, full_ps),
                path=tmp_path / "expected-zfactor.nc",
            )

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                zfactors=hybrid_zfactors(self.project, existing_ps),
            ) as writer:
                writer.write(
                    existing_data,
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                zfactors=hybrid_zfactors(self.project, append_ps),
                existing="append",
            ) as writer:
                writer.write(
                    append_data,
                    time_values=[45.0],
                    time_bounds=[[30.0, 60.0]],
                )
                actual, _ = writer.close()

            with xr.open_dataset(
                expected_path,
                decode_times=False,
                mask_and_scale=False,
            ) as expected_open:
                expected = expected_open.load()
            xr.testing.assert_identical(equivalent(actual), equivalent(expected))
            np.testing.assert_array_equal(actual["ps"].values, full_ps)
            self.assertIn("formula_terms", actual["lev"].attrs)

    def test_append_rejects_incompatible_zfactor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            output_path = tmp_path / "writer-zfactor.nc"
            axes = [
                time_axis(self.project),
                hybrid_axis(self.project),
                *horizontal_axes(self.project),
            ]
            ps = np.full((1, 2, 2), 99000.0, dtype="f4")

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                zfactors=hybrid_zfactors(self.project, ps),
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

            writer = cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                zfactors=hybrid_zfactors(self.project, ps, a_values=[0.8, 0.2]),
                existing="append",
            )
            writer.write(
                np.ones((1, 2, 2, 2), dtype="f4"),
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
            )

            with self.assertRaisesRegex(ValueError, "variable 'a' values differ"):
                writer.close()

    def test_sequential_appends_accumulate_and_remain_appendable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]
            output_path = tmp_path / "writer.nc"
            chunks = [
                np.arange(8, dtype="f4").reshape(2, 2, 2),
                np.arange(8, 12, dtype="f4").reshape(1, 2, 2),
                np.arange(12, 20, dtype="f4").reshape(2, 2, 2),
            ]

            with cmor4.DatasetWriter(info, variable, axes, path=output_path) as writer:
                writer.write(
                    chunks[0],
                    time_values=[15.0, 45.0],
                    time_bounds=[[0.0, 30.0], [30.0, 60.0]],
                )

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                existing="append",
            ) as writer:
                writer.write(
                    chunks[1],
                    time_values=[75.0],
                    time_bounds=[[60.0, 90.0]],
                )
                writer.close()

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                existing="append",
            ) as writer:
                writer.write(
                    chunks[2],
                    time_values=[105.0, 135.0],
                    time_bounds=[[90.0, 120.0], [120.0, 150.0]],
                )
                actual, _ = writer.close()

            np.testing.assert_array_equal(
                actual["time"].values,
                [15.0, 45.0, 75.0, 105.0, 135.0],
            )
            np.testing.assert_array_equal(
                actual["tos"].values,
                np.concatenate(chunks, axis=0),
            )

    def test_append_preserves_history_and_refreshes_provenance_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]
            output_path = tmp_path / "writer.nc"
            old_history = "original history entry"
            old_creation_date = "1900-01-01T00:00:00Z"
            old_tracking_id = "old-tracking-id"

            with cmor4.DatasetWriter(info, variable, axes, path=output_path) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

            with xr.open_dataset(
                output_path,
                decode_times=False,
                mask_and_scale=False,
            ) as opened:
                rewritten = opened.load()
            rewritten.attrs.update(
                {
                    "history": old_history,
                    "creation_date": old_creation_date,
                    "tracking_id": old_tracking_id,
                }
            )
            rewritten.to_netcdf(output_path)
            rewritten.close()

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                existing="append",
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[45.0],
                    time_bounds=[[30.0, 60.0]],
                )
                actual, _ = writer.close()

            self.assertEqual(actual.attrs["history"], old_history)
            self.assertNotEqual(actual.attrs["creation_date"], old_creation_date)
            self.assertNotEqual(actual.attrs["tracking_id"], old_tracking_id)

    def test_append_reports_specific_attribute_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]
            output_path = tmp_path / "writer.nc"

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                attrs={"append_marker": "existing"},
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

            writer = cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                attrs={"append_marker": "new"},
                existing="append",
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
            )

            with self.assertRaisesRegex(
                ValueError,
                "global attribute 'append_marker' differs.*existing.*new",
            ):
                writer.close()

    def test_append_with_edge_format_time_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]
            output_path = tmp_path / "writer.nc"

            with cmor4.DatasetWriter(info, variable, axes, path=output_path) as writer:
                writer.write(
                    np.ones((2, 2, 2), dtype="f4"),
                    time_values=[15.0, 45.0],
                    time_bounds=[0.0, 30.0, 60.0],
                )

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=output_path,
                existing="append",
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[75.0],
                    time_bounds=[60.0, 90.0],
                )
                actual, _ = writer.close()

            np.testing.assert_array_equal(
                actual["time_bnds"].values,
                [[0.0, 30.0], [30.0, 60.0], [60.0, 90.0]],
            )

    def test_append_preserves_original_on_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            output_path = tmp_path / "writer.nc"

            with cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *horizontal_axes(self.project)],
                path=output_path,
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

            writer = cmor4.DatasetWriter(
                info,
                variable,
                [time_axis(self.project), *shifted_horizontal_axes(self.project)],
                path=output_path,
                existing="append",
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4") * 2.0,
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
            )
            with self.assertRaises(ValueError):
                writer.close()

            with xr.open_dataset(output_path, decode_times=False) as unchanged:
                self.assertEqual(len(unchanged["time"]), 1)
                np.testing.assert_array_equal(unchanged["time"].values, [15.0])


if __name__ == "__main__":
    unittest.main()
