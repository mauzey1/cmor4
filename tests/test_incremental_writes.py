from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

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


def hybrid_axis(project: cmor4.ProjectTables) -> cmor4.Axis:
    return project.axis(
        "standard_hybrid_sigma",
        values=[0.9, 0.1],
        bounds=[[1.0, 0.5], [0.5, 0.0]],
    )


def hybrid_zfactors(
    project: cmor4.ProjectTables,
    ps_values: np.ndarray | None = None,
) -> list[cmor4.ZFactor]:
    return [
        project.zfactor(
            "a",
            values=[0.9, 0.1],
            bounds=[[1.0, 0.5], [0.5, 0.0]],
        ),
        project.zfactor(
            "b",
            values=[0.9, 0.1],
            bounds=[[1.0, 0.5], [0.5, 0.0]],
        ),
        project.zfactor("p0", values=100000.0),
        project.zfactor(
            "ps",
            **({} if ps_values is None else {"values": ps_values}),
        ),
    ]


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

    def test_chunked_zfactor_writes_match_cmorize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            full_time = time_axis(
                self.project,
                values=[15.0, 45.0],
                bounds=[[0.0, 30.0], [30.0, 60.0]],
            )
            axes = [
                full_time,
                hybrid_axis(self.project),
                *horizontal_axes(self.project),
            ]
            writer_axes = [
                time_axis(self.project),
                hybrid_axis(self.project),
                *horizontal_axes(self.project),
            ]
            data = np.arange(16, dtype="f4").reshape(2, 2, 2, 2)
            ps = np.stack(
                [
                    np.full((2, 2), 99000.0, dtype="f4"),
                    np.full((2, 2), 99100.0, dtype="f4"),
                ],
                axis=0,
            )

            _, expected_path = cmor4.cmorize(
                info,
                variable,
                axes,
                data,
                zfactors=hybrid_zfactors(self.project, ps),
                path=tmp_path / "cmorize-zfactor.nc",
            )
            writer = cmor4.DatasetWriter(
                info,
                variable,
                writer_axes,
                path=tmp_path / "writer-zfactor.nc",
                zfactors=hybrid_zfactors(self.project),
            )
            writer.write(
                data[:1],
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
                zfactors={"ps": ps[:1]},
            )
            writer.write(
                data[1:],
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
                zfactors={"ps": ps[1:]},
            )

            actual, path = writer.close()

            with xr.open_dataset(
                expected_path,
                decode_times=False,
                mask_and_scale=False,
            ) as expected_open:
                expected = expected_open.load()
            xr.testing.assert_identical(equivalent(actual), equivalent(expected))
            assert_files_equivalent(self, path, expected_path)
            self.assertEqual(
                actual["lev"].attrs["formula_terms"],
                "p0: p0 a: a b: b ps: ps",
            )
            np.testing.assert_array_equal(actual["ps"].values, ps)

    def test_write_requires_empty_time_varying_zfactor_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            writer = cmor4.DatasetWriter(
                info,
                variable,
                [
                    time_axis(self.project),
                    hybrid_axis(self.project),
                    *horizontal_axes(self.project),
                ],
                path=tmp_path / "writer-zfactor.nc",
                zfactors=hybrid_zfactors(self.project),
            )

            with self.assertRaisesRegex(ValueError, "zfactor 'ps'.*every write"):
                writer.write(
                    np.ones((1, 2, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                )

    def test_write_rejects_missing_chunked_zfactor_after_first_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            writer = cmor4.DatasetWriter(
                info,
                variable,
                [
                    time_axis(self.project),
                    hybrid_axis(self.project),
                    *horizontal_axes(self.project),
                ],
                path=tmp_path / "writer-zfactor.nc",
                zfactors=hybrid_zfactors(self.project),
            )
            writer.write(
                np.ones((1, 2, 2, 2), dtype="f4"),
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
                zfactors={"ps": np.ones((1, 2, 2), dtype="f4") * 99000.0},
            )

            with self.assertRaisesRegex(ValueError, "zfactor 'ps'.*every write"):
                writer.write(
                    np.ones((1, 2, 2, 2), dtype="f4"),
                    time_values=[45.0],
                    time_bounds=[[30.0, 60.0]],
                )

    def test_write_rejects_bad_zfactor_chunk_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            writer = cmor4.DatasetWriter(
                info,
                variable,
                [
                    time_axis(self.project),
                    hybrid_axis(self.project),
                    *horizontal_axes(self.project),
                ],
                path=tmp_path / "writer-zfactor.nc",
                zfactors=hybrid_zfactors(self.project),
            )

            with self.assertRaisesRegex(ValueError, "Zfactor 'ps' chunk shape"):
                writer.write(
                    np.ones((1, 2, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                    zfactors={"ps": np.ones((2, 2), dtype="f4")},
                )

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

    def test_close_does_not_warn_about_endian_mismatch(self) -> None:
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

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ds, _ = writer.close()
                ds.close()

            messages = [str(warning.message) for warning in caught]
            self.assertFalse(
                any("endian-ness of dtype and endian kwarg" in msg for msg in messages)
            )

    def test_write_rejects_unknown_zfactor_name(self) -> None:
        """Test that providing an unknown zfactor name raises an error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            writer = cmor4.DatasetWriter(
                info,
                variable,
                [
                    time_axis(self.project),
                    hybrid_axis(self.project),
                    *horizontal_axes(self.project),
                ],
                path=tmp_path / "writer-zfactor.nc",
                zfactors=hybrid_zfactors(self.project),
            )

            with self.assertRaisesRegex(ValueError, "Unknown zfactor.*'unknown_zf'"):
                writer.write(
                    np.ones((1, 2, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                    zfactors={
                        "ps": np.ones((1, 2, 2), dtype="f4") * 99000.0,
                        "unknown_zf": np.ones((1, 2, 2), dtype="f4"),
                    },
                )

    def test_write_rejects_non_time_varying_zfactor_per_chunk(self) -> None:
        """Test that providing a scalar zfactor per-chunk raises an error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            writer = cmor4.DatasetWriter(
                info,
                variable,
                [
                    time_axis(self.project),
                    hybrid_axis(self.project),
                    *horizontal_axes(self.project),
                ],
                path=tmp_path / "writer-zfactor.nc",
                zfactors=hybrid_zfactors(self.project),
            )

            # p0 is a scalar (no time dimension), should not be supplied per-chunk
            with self.assertRaisesRegex(
                ValueError, "Zfactor 'p0'.*does not include time"
            ):
                writer.write(
                    np.ones((1, 2, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                    zfactors={
                        "ps": np.ones((1, 2, 2), dtype="f4") * 99000.0,
                        "p0": np.array(100000.0),
                    },
                )

    def test_write_validates_zfactor_nan_values(self) -> None:
        """Test that NaN values in zfactor data are caught by validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            writer = cmor4.DatasetWriter(
                info,
                variable,
                [
                    time_axis(self.project),
                    hybrid_axis(self.project),
                    *horizontal_axes(self.project),
                ],
                path=tmp_path / "writer-zfactor.nc",
                zfactors=hybrid_zfactors(self.project),
            )

            # Create ps data with NaN values
            ps_with_nan = np.ones((1, 2, 2), dtype="f4") * 99000.0
            ps_with_nan[0, 0, 0] = np.nan

            with self.assertRaisesRegex(ValueError, "NaN"):
                writer.write(
                    np.ones((1, 2, 2, 2), dtype="f4"),
                    time_values=[15.0],
                    time_bounds=[[0.0, 30.0]],
                    zfactors={"ps": ps_with_nan},
                )

    def test_preserve_definition_with_zfactors(self) -> None:
        """Test preserve_definition=True with per-chunk zfactors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable(
                "tnhusscpbl_tavg-al-hxy-u",
                table_id="atmos",
            )
            axes = [
                time_axis(self.project),
                hybrid_axis(self.project),
                *horizontal_axes(self.project),
            ]

            writer = cmor4.DatasetWriter(
                info,
                variable,
                axes,
                zfactors=hybrid_zfactors(self.project),
            )

            # Write first file segment
            data1 = np.arange(8, dtype="f4").reshape(1, 2, 2, 2)
            ps1 = np.full((1, 2, 2), 99000.0, dtype="f4")
            writer.write(
                data1,
                time_values=[15.0],
                time_bounds=[[0.0, 30.0]],
                zfactors={"ps": ps1},
            )
            ds1, path1 = writer.close(preserve_definition=True)
            ds1.close()

            # Write second file segment with different data
            data2 = np.arange(8, 16, dtype="f4").reshape(1, 2, 2, 2)
            ps2 = np.full((1, 2, 2), 99500.0, dtype="f4")
            writer.write(
                data2,
                time_values=[45.0],
                time_bounds=[[30.0, 60.0]],
                zfactors={"ps": ps2},
            )
            ds2, path2 = writer.close()
            ds2.close()

            # Verify both files exist and have correct zfactor data
            with xr.open_dataset(path1, decode_times=False) as file1:
                np.testing.assert_array_equal(file1["ps"].values, ps1)
                self.assertEqual(file1["ps"].shape, (1, 2, 2))

            with xr.open_dataset(path2, decode_times=False) as file2:
                np.testing.assert_array_equal(file2["ps"].values, ps2)
                self.assertEqual(file2["ps"].shape, (1, 2, 2))

            # Files should be different
            self.assertNotEqual(path1, path2)


class TestDatasetWriterEncoding(unittest.TestCase):
    """Tests for NetCDF encoding with DatasetWriter."""

    def setUp(self):
        self.project = cmip7_project("tables/CMIP7_ocean.json")

    def test_compression(self):
        """Test that compression is applied correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]

            # Specify compression (DatasetWriter applies auto-chunking for CMIP7)
            encoding = {
                "tos": {
                    "zlib": True,
                    "complevel": 4,
                    "shuffle": True,
                }
            }

            with cmor4.DatasetWriter(info, variable, axes, encoding=encoding) as writer:
                data = np.arange(4, dtype="f4").reshape(1, 2, 2)
                writer.write(data, time_values=[15.0], time_bounds=[[0.0, 30.0]])

            # Verify compression was applied
            files = list(tmp_path.rglob("*.nc"))
            self.assertEqual(len(files), 1)

            with xr.open_dataset(files[0], decode_times=False) as ds:
                # Check compression encoding
                enc = ds["tos"].encoding
                self.assertTrue(enc.get("zlib", False))
                self.assertGreater(enc.get("complevel", 0), 0)
                self.assertTrue(enc.get("shuffle", False))

    def test_encoding_with_incremental_writes(self):
        """Test encoding is maintained across incremental writes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]

            encoding = {
                "tos": {
                    "zlib": True,
                    "complevel": 4,
                }
            }

            with cmor4.DatasetWriter(info, variable, axes, encoding=encoding) as writer:
                # First write
                data1 = np.arange(4, dtype="f4").reshape(1, 2, 2)
                writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])

                # Second write
                data2 = np.arange(4, 8, dtype="f4").reshape(1, 2, 2)
                writer.write(data2, time_values=[45.0], time_bounds=[[30.0, 60.0]])

            # Verify encoding was maintained
            files = list(tmp_path.rglob("*.nc"))
            self.assertEqual(len(files), 1)

            with xr.open_dataset(files[0], decode_times=False) as ds:
                enc = ds["tos"].encoding
                self.assertTrue(enc.get("zlib", False))
                self.assertEqual(enc.get("complevel"), 4)
                # Verify both time slices are present
                self.assertEqual(ds["tos"].shape, (2, 2, 2))

    def test_auto_chunking_applied(self):
        """Test that CMIP7 auto-chunking is applied when not explicitly specified."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            info = self.project.dataset_info(dataset_info(tmp_path))
            variable = self.project.variable("tos_tavg-u-hxy-sea", table_id="ocean")
            axes = [time_axis(self.project), *horizontal_axes(self.project)]

            # No encoding specified - should get auto-chunking
            with cmor4.DatasetWriter(info, variable, axes) as writer:
                data = np.arange(4, dtype="f4").reshape(1, 2, 2)
                writer.write(data, time_values=[15.0], time_bounds=[[0.0, 30.0]])

            # Verify chunking was applied (CMIP7 auto-chunking)
            files = list(tmp_path.rglob("*.nc"))
            self.assertEqual(len(files), 1)

            with xr.open_dataset(files[0], decode_times=False) as ds:
                # Check that variable has chunking
                self.assertIsNotNone(ds["tos"].encoding.get("chunksizes"))
                # Should have time-unlimited chunking
                chunks = ds["tos"].encoding["chunksizes"]
                self.assertEqual(chunks[0], 1)  # time dimension
