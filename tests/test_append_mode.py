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


if __name__ == "__main__":
    unittest.main()
