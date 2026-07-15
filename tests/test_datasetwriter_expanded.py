"""Expanded test suite for DatasetWriter covering Phase 1 requirements.

This module provides comprehensive coverage of:
- Equivalence with cmorize() for various variable types
- Validation and error handling
- Encoding and chunking
- Edge cases and error recovery
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

import cmor4
from table_helpers import cmip7_project


def dataset_info(tmp_path: Path) -> dict[str, str]:
    """Standard CMIP7 dataset metadata for tests."""
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
    """Standard horizontal axes for tests."""
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
    **kwargs,
) -> cmor4.Axis:
    """Create a time axis with optional values and bounds."""
    axis_kwargs = {"units": "days since 2000-01-01"}
    if values is not None:
        axis_kwargs["values"] = values
    if bounds is not None:
        axis_kwargs["bounds"] = bounds
    axis_kwargs.update(kwargs)
    return project.axis("time", **axis_kwargs)


def pressure_axis(project: cmor4.ProjectTables) -> cmor4.Axis:
    """Create a standard pressure level axis."""
    return project.axis("plev19")


def equivalent(ds: xr.Dataset) -> xr.Dataset:
    """Normalize dataset for comparison by removing time-dependent attributes."""
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
    """Assert that two NetCDF files are equivalent (ignoring timestamps)."""
    test.assertTrue(actual_path.exists(), f"Actual file missing: {actual_path}")
    test.assertTrue(expected_path.exists(), f"Expected file missing: {expected_path}")
    with xr.open_dataset(actual_path, decode_times=False) as actual_opened:
        actual = actual_opened.load()
    with xr.open_dataset(expected_path, decode_times=False) as expected_opened:
        expected = expected_opened.load()
    xr.testing.assert_identical(equivalent(actual), equivalent(expected))


class DatasetWriterEquivalenceTests(unittest.TestCase):
    """Test that DatasetWriter produces output equivalent to cmorize()."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project = cmip7_project()

    def test_3d_variable_with_pressure_levels_matches_cmorize(self) -> None:
        """Test DatasetWriter with 3D atmospheric variable (ta)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("ta_tavg-p19-hxy-air", table_id="atmos")

            # Create 3D data: (time=2, plev=19, lat=2, lon=2)
            data = np.arange(152, dtype="f4").reshape(2, 19, 2, 2)

            # Full axes for cmorize
            full_axes = [
                time_axis(
                    self.project,
                    values=[15.0, 45.0],
                    bounds=[[0.0, 30.0], [30.0, 60.0]],
                ),
                pressure_axis(self.project),
                *horizontal_axes(self.project),
            ]

            # Writer axes (empty time)
            writer_axes = [
                time_axis(self.project),
                pressure_axis(self.project),
                *horizontal_axes(self.project),
            ]

            # Create expected output with cmorize
            expected, expected_path = cmor4.cmorize(
                info,
                variable,
                full_axes,
                data,
                path=tmp_path / "cmorize.nc",
            )

            # Create output with DatasetWriter
            with cmor4.DatasetWriter(
                info,
                variable,
                writer_axes,
                path=tmp_path / "writer.nc",
            ) as writer:
                writer.write(
                    data,
                    time_values=[15.0, 45.0],
                    time_bounds=[[0.0, 30.0], [30.0, 60.0]],
                )
                actual, actual_path = writer.close()

            # Compare
            xr.testing.assert_identical(equivalent(actual), equivalent(expected))
            assert_files_equivalent(self, actual_path, expected_path)

    def test_int16_dtype_preserved(self) -> None:
        """Test that int16 dtype is preserved through DatasetWriter."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

            # Create int16 data
            data = np.arange(8, dtype=np.int16).reshape(2, 2, 2)

            full_axes = [
                time_axis(
                    self.project,
                    values=[15.0, 45.0],
                    bounds=[[0.0, 30.0], [30.0, 60.0]],
                ),
                *horizontal_axes(self.project),
            ]
            writer_axes = [time_axis(self.project), *horizontal_axes(self.project)]

            expected, expected_path = cmor4.cmorize(
                info,
                variable,
                full_axes,
                data,
                path=tmp_path / "cmorize.nc",
            )

            with cmor4.DatasetWriter(
                info,
                variable,
                writer_axes,
                path=tmp_path / "writer.nc",
            ) as writer:
                writer.write(
                    data,
                    time_values=[15.0, 45.0],
                    time_bounds=[[0.0, 30.0], [30.0, 60.0]],
                )
                actual, actual_path = writer.close()

            # Check dtype
            self.assertEqual(actual["tos"].dtype, expected["tos"].dtype)
            xr.testing.assert_identical(equivalent(actual), equivalent(expected))

    def test_many_time_slices_match_cmorize(self) -> None:
        """Test DatasetWriter with many individual time slices."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

            # Create 12 monthly time slices
            n_times = 12
            time_values = [float(i * 30 + 15) for i in range(n_times)]
            time_bounds = [[float(i * 30), float((i + 1) * 30)] for i in range(n_times)]
            data = np.arange(n_times * 4, dtype="f4").reshape(n_times, 2, 2)

            full_axes = [
                time_axis(self.project, values=time_values, bounds=time_bounds),
                *horizontal_axes(self.project),
            ]
            writer_axes = [time_axis(self.project), *horizontal_axes(self.project)]

            expected, expected_path = cmor4.cmorize(
                info,
                variable,
                full_axes,
                data,
                path=tmp_path / "cmorize.nc",
            )

            # Write one time slice at a time
            with cmor4.DatasetWriter(
                info,
                variable,
                writer_axes,
                path=tmp_path / "writer.nc",
            ) as writer:
                for i in range(n_times):
                    writer.write(
                        data[i : i + 1],
                        time_values=[time_values[i]],
                        time_bounds=[time_bounds[i]],
                    )
                actual, actual_path = writer.close()

            xr.testing.assert_identical(equivalent(actual), equivalent(expected))
            assert_files_equivalent(self, actual_path, expected_path)

    def test_360day_calendar_matches_cmorize(self) -> None:
        """Test DatasetWriter with 360_day calendar."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info_dict = dataset_info(tmp_path)
            info_dict["calendar"] = "360_day"
            info = self.project.dataset_info(info_dict)
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

            data = np.arange(8, dtype="f4").reshape(2, 2, 2)

            full_axes = [
                time_axis(
                    self.project,
                    values=[15.0, 45.0],
                    bounds=[[0.0, 30.0], [30.0, 60.0]],
                    calendar="360_day",
                ),
                *horizontal_axes(self.project),
            ]
            writer_axes = [
                time_axis(self.project, calendar="360_day"),
                *horizontal_axes(self.project),
            ]

            expected, expected_path = cmor4.cmorize(
                info,
                variable,
                full_axes,
                data,
                path=tmp_path / "cmorize.nc",
            )

            with cmor4.DatasetWriter(
                info,
                variable,
                writer_axes,
                path=tmp_path / "writer.nc",
            ) as writer:
                writer.write(
                    data,
                    time_values=[15.0, 45.0],
                    time_bounds=[[0.0, 30.0], [30.0, 60.0]],
                )
                actual, actual_path = writer.close()

            xr.testing.assert_identical(equivalent(actual), equivalent(expected))


class DatasetWriterValidationTests(unittest.TestCase):
    """Test validation and error handling in DatasetWriter."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project = cmip7_project()

    def test_time_gaps_allowed_after_preserving_definition(self) -> None:
        """Test that preserve_definition permits gaps between output files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]

            writer = cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=tmp_path / "writer-1.nc",
            )
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
            )
            first, first_path = writer.close(preserve_definition=True)
            first.close()

            writer.path = tmp_path / "writer-2.nc"
            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[115.0],
                time_bounds=[[100.0, 130.0]],
            )
            second, second_path = writer.close()

            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())
            self.assertEqual(len(second["time"]), 1)
            np.testing.assert_array_equal(second["time"].values, [115.0])
            second.close()

    def test_time_bounds_must_be_contiguous_by_default(self) -> None:
        """Test that non-contiguous time bounds raise error by default."""
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

            # Gap in bounds should raise error
            with self.assertRaisesRegex(ValueError, "contiguous"):
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[115.0],
                    time_bounds=[[100.0, 130.0]],
                )

    def test_missing_time_values_without_initial_axis_raises_error(self) -> None:
        """Test that write() without time_values fails when axis is empty."""
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

            with self.assertRaisesRegex(ValueError, "time_values must be provided"):
                writer.write(np.ones((1, 2, 2), dtype="f4"))

    def test_time_bounds_length_mismatch_raises_error(self) -> None:
        """Test that time_bounds shape must match time_values."""
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

            with self.assertRaisesRegex(
                ValueError,
                "does not match time_values length",
            ):
                writer.write(
                    np.ones((2, 2, 2), dtype="f4"),
                    time_values=[15.0, 45.0],
                    time_bounds=[[0.0, 30.0]],  # Wrong: only 1 bound for 2 times
                )

    def test_inconsistent_time_bounds_across_writes_raises_error(self) -> None:
        """Test that time_bounds must be provided consistently."""
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

            with self.assertRaisesRegex(ValueError, "every write"):
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[45.0],
                    # Missing time_bounds
                )


class DatasetWriterErrorRecoveryTests(unittest.TestCase):
    """Test error recovery and edge cases in DatasetWriter."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project = cmip7_project()

    def test_close_without_writes_raises_error(self) -> None:
        """Test that close() fails if no data was written."""
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

            with self.assertRaisesRegex(ValueError, "before any data is written"):
                writer.close()

    def test_write_after_close_raises_error(self) -> None:
        """Test that write() after close() raises."""
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
            writer.close()

            with self.assertRaisesRegex(ValueError, "already closed"):
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[45.0],
                    time_bounds=[[30.0, 60.0]],
                )

    def test_staging_directory_cleaned_on_success(self) -> None:
        """Test that staging directory is removed after successful close()."""
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
            staging_root = writer.staging_root
            self.assertTrue(staging_root.exists())

            writer.write(
                np.ones((1, 2, 2), dtype="f4"),
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
            )
            writer.close()

            # Staging directory should be cleaned up
            self.assertFalse(staging_root.exists())

    def test_context_manager_closes_on_success(self) -> None:
        """Test that context manager calls close() automatically."""
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
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

            # File should exist
            self.assertTrue(output_path.exists())

            # Writer should be closed
            self.assertTrue(writer._closed)

    def test_context_manager_preserves_staging_on_error(self) -> None:
        """Test that staging directory is preserved when an exception occurs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]

            staging_root = None
            try:
                with cmor4.DatasetWriter(
                    info,
                    variable,
                    axes,
                    path=tmp_path / "writer.nc",
                ) as writer:
                    staging_root = writer.staging_root
                    writer.write(
                        np.ones((1, 2, 2), dtype="f4"),
                        time_values=[15.0],
                        time_bounds=[[0.0, 30.0]],
                    )
                    raise RuntimeError("Simulated error")
            except RuntimeError:
                pass

            # Staging directory should still exist for debugging
            self.assertTrue(staging_root.exists())


class DatasetWriterEdgeCaseTests(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project = cmip7_project()

    def test_time_values_as_list(self) -> None:
        """Test that time_values can be a Python list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=tmp_path / "writer.nc",
            ) as writer:
                # Pass time_values as list (not numpy array)
                writer.write(
                    np.ones((2, 2, 2), dtype="f4"),
                    time_values=[15.0, 45.0],  # List, not array
                    time_bounds=[[0.0, 30.0], [30.0, 60.0]],
                )
                ds, _ = writer.close()

            np.testing.assert_array_equal(ds["time"].values, [15.0, 45.0])

    def test_time_bounds_as_n_plus_one_edges(self) -> None:
        """Test N+1 edge format for time_bounds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=tmp_path / "writer.nc",
            ) as writer:
                # Pass time_bounds as N+1 edges (3 values for 2 times)
                writer.write(
                    np.ones((2, 2, 2), dtype="f4"),
                    time_values=[15.0, 45.0],
                    time_bounds=[0.0, 30.0, 60.0],  # N+1 format
                )
                ds, _ = writer.close()

            # Should be converted to Nx2 pairs
            expected_bounds = np.array([[0.0, 30.0], [30.0, 60.0]])
            np.testing.assert_array_equal(ds["time_bnds"].values, expected_bounds)

    def test_single_time_slice_works(self) -> None:
        """Test DatasetWriter with only one time slice."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

            axes = [time_axis(self.project), *horizontal_axes(self.project)]

            with cmor4.DatasetWriter(
                info,
                variable,
                axes,
                path=tmp_path / "writer.nc",
            ) as writer:
                writer.write(
                    np.ones((1, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )
                ds, path = writer.close()

            self.assertEqual(len(ds["time"]), 1)
            self.assertTrue(path.exists())

    def test_time_range_appears_in_output_filename(self) -> None:
        """Verify that output filename includes the time range from written data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]

            # Write January and February 2000 (days since 2000-01-01)
            # time_values=[15.0, 45.0] corresponds to Jan 16 and Feb 15, 2000
            with cmor4.DatasetWriter(info, variable, axes) as writer:
                writer.write(
                    np.ones((2, 2, 2), dtype="f4"),
                    time_values=[15.0, 45.0],
                    time_bounds=[[0.0, 30.0], [30.0, 60.0]],
                )
                ds, path = writer.close()

            # Check that the path exists
            self.assertTrue(path.exists(), f"Output file not found: {path}")

            # Filename should include time range for monthly data spanning Jan-Feb 2000
            filename = path.name

            # CMIP7 DRS typically formats monthly data as YYYYMM-YYYYMM
            # For Jan-Feb 2000, we expect 200001-200002 in the filename
            self.assertIn(
                "200001",
                filename,
                f"January 2000 (200001) not found in filename: {filename}",
            )
            self.assertIn(
                "200002",
                filename,
                f"February 2000 (200002) not found in filename: {filename}",
            )

            # Verify the dataset actually contains the expected time range
            self.assertEqual(len(ds["time"]), 2)
            ds.close()


if __name__ == "__main__":
    unittest.main()
