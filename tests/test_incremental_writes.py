from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

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


def equivalent(ds: xr.Dataset) -> xr.Dataset:
    normalized = ds.load().copy(deep=True)
    normalized.attrs.pop("creation_date", None)
    normalized.attrs.pop("history", None)
    for variable in normalized.variables.values():
        variable.attrs.pop("_FillValue", None)
    return normalized


def assert_files_equivalent(
    test: unittest.TestCase,
    actual_path: Path,
    expected_path: Path,
) -> None:
    test.assertTrue(actual_path.exists())
    test.assertTrue(expected_path.exists())
    with xr.open_dataset(actual_path, decode_times=False) as actual_opened:
        actual = actual_opened.load()
    with xr.open_dataset(expected_path, decode_times=False) as expected_opened:
        expected = expected_opened.load()
    xr.testing.assert_identical(equivalent(actual), equivalent(expected))


def is_full_zarr_selection(selection: object, shape: tuple[int, ...]) -> bool:
    if selection == slice(None):
        return True
    if not isinstance(selection, tuple) or len(selection) != len(shape):
        return False
    for item, size in zip(selection, shape):
        if item == slice(None):
            continue
        if not isinstance(item, slice):
            return False
        start = 0 if item.start is None else item.start
        stop = size if item.stop is None else item.stop
        step = 1 if item.step is None else item.step
        if start != 0 or stop != size or step != 1:
            return False
    return True


class DatasetWriterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project = cmip7_project()

    def test_single_write_matches_cmorize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tos_tavg-u-hxy-sea",
                table_id="ocean",
                missing_value=np.float32(1.0e20),
            )
            full_time = time_axis(
                self.project,
                values=[15.0, 45.0],
                bounds=[[0.0, 30.0], [30.0, 60.0]],
            )
            axes = [full_time, *horizontal_axes(self.project)]
            writer_axes = [time_axis(self.project), *horizontal_axes(self.project)]
            data = np.arange(8, dtype="f4").reshape(2, 2, 2)

            expected, expected_path = cmor4.cmorize(
                info,
                variable,
                axes,
                data,
                path=tmp_path / "cmorize.nc",
            )
            writer = cmor4.DatasetWriter(
                info,
                variable,
                writer_axes,
                path=tmp_path / "writer.nc",
            )
            writer.write(
                data,
                time_values=[15.0, 45.0],
                time_bounds=[[0.0, 30.0], [30.0, 60.0]],
            )
            actual, path = writer.close()

            self.assertEqual(path, tmp_path / "writer.nc")
            xr.testing.assert_identical(equivalent(actual), equivalent(expected))
            assert_files_equivalent(self, path, expected_path)

    def test_multiple_writes_match_cmorize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            full_time = time_axis(
                self.project,
                values=[15.0, 45.0],
                bounds=[[0.0, 30.0], [30.0, 60.0]],
            )
            axes = [full_time, *horizontal_axes(self.project)]
            writer_axes = [time_axis(self.project), *horizontal_axes(self.project)]
            data = np.arange(8, dtype="f4").reshape(2, 2, 2)

            expected, expected_path = cmor4.cmorize(
                info,
                variable,
                axes,
                data,
                path=tmp_path / "cmorize.nc",
            )
            writer = cmor4.DatasetWriter(
                info,
                variable,
                writer_axes,
                path=tmp_path / "writer.nc",
            )
            writer.write(
                data[:1],
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
            )
            writer.write(
                data[1:],
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
            )

            actual, _ = writer.close()

            xr.testing.assert_identical(equivalent(actual), equivalent(expected))
            assert_files_equivalent(self, tmp_path / "writer.nc", expected_path)

    def test_write_rejects_bad_chunk_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]
            writer = cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=tmp_path / "writer.nc",
            )

            with self.assertRaisesRegex(ValueError, "shape"):
                writer.write(
                    np.ones((1, 2, 3), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

    def test_write_rejects_non_monotonic_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]
            writer = cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=tmp_path / "writer.nc",
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
            )

            with self.assertRaisesRegex(ValueError, "monotonic"):
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

    def test_close_does_not_read_full_zarr_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]
            writer = cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=tmp_path / "writer.nc",
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
            )

            zarr_array = writer._zarr_array
            assert zarr_array is not None
            zarr_type = type(zarr_array)
            original_getitem = zarr_type.__getitem__
            full_reads = []

            def guarded_getitem(array, selection):
                if array is zarr_array and is_full_zarr_selection(
                    selection,
                    tuple(zarr_array.shape),
                ):
                    full_reads.append(selection)
                    raise AssertionError("DatasetWriter read the full Zarr array")
                return original_getitem(array, selection)

            with mock.patch.object(zarr_type, "__getitem__", guarded_getitem):
                ds, path = writer.close()
                ds.close()

            self.assertTrue(path.exists())
            self.assertEqual(full_reads, [])
