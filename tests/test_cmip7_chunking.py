"""Tests for CMIP7 chunking validation and auto-application."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyfive
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
CMIP7_TABLE_ROOT = REPO_ROOT / "project_tables" / "cmip7-cmor-tables"
TABLES_DIR = CMIP7_TABLE_ROOT / "tables"
CV_PATH = CMIP7_TABLE_ROOT / "tables-cvs" / "cmor-cvs.json"


def _requires_tables(test):
    """Skip decorator applied when the CMIP7 submodule is not checked out."""
    if not TABLES_DIR.exists() or not CV_PATH.exists():
        return unittest.skip("CMIP7 tables submodule not initialised")(test)
    return test


_BASE_CMIP7_DATASET = {
    "mip_era": "CMIP7",
    "activity_id": "CMIP",
    "calendar": "360_day",
    "experiment_id": "amip",
    "forcing_index": "f1",
    "frequency": "mon",
    "grid_label": "g999",
    "host_collection": "CMIP7",
    "initialization_index": "i1",
    "institution_id": "MOHC",
    "license_id": "CC-BY-4.0",
    "nominal_resolution": "100 km",
    "physics_index": "p1",
    "realization_index": "r1",
    "region": "glb",
    "source_id": "DUMMY-MODEL",
}

_BASE_CMIP6_DATASET = {
    "mip_era": "CMIP6",
    "activity_id": "CMIP",
    "calendar": "360_day",
    "experiment_id": "amip",
    "forcing_index": "f1",
    "frequency": "mon",
    "grid_label": "g999",
    "host_collection": "CMIP6",
    "initialization_index": "i1",
    "institution_id": "MOHC",
    "nominal_resolution": "100 km",
    "physics_index": "p1",
    "realization_index": "r1",
    "region": "glb",
    "source_id": "DUMMY-MODEL",
}


def _create_bounds(vals, coord_name=None):
    """Create simple bounds for coordinate values."""
    if len(vals) < 2:
        return None

    # For endpoints, use edge values instead of extrapolating
    bounds = np.zeros((len(vals), 2))
    for i in range(len(vals)):
        if i == 0:
            # First point: lower bound at first value, upper at midpoint to next
            bounds[i, 0] = vals[i]
            bounds[i, 1] = (vals[i] + vals[i + 1]) / 2
        elif i == len(vals) - 1:
            # Last point: lower bound at midpoint from previous, upper at last value
            bounds[i, 0] = (vals[i - 1] + vals[i]) / 2
            bounds[i, 1] = vals[i]
        else:
            # Middle points: centered bounds
            bounds[i, 0] = (vals[i - 1] + vals[i]) / 2
            bounds[i, 1] = (vals[i] + vals[i + 1]) / 2

    # Special handling for latitude to stay within [-90, 90]
    if coord_name == "latitude":
        bounds = np.clip(bounds, -90, 90)
    # Special handling for longitude to stay within [0, 360]
    elif coord_name == "longitude":
        bounds = np.clip(bounds, 0, 360)

    return bounds


class CMIP7ChunkingCheckAssertion:
    """Check if NetCDF files meet CMIP7 repack chunking requirements."""

    def assertCMIP7Chunking(self, path: Path, msg: str = None):
        BYTES_4MiB = 4 * 2**20

        # Chunking checks from cmip7_chunking.
        f = pyfive.File(path)
        for time_name in f:
            lower = str(time_name).lower()
            if lower != "time" and (
                not lower.startswith("time") or "bnds" in lower or "bounds" in lower
            ):
                continue

            # Check for the time coordinates variable having one chunk
            t = f[time_name]
            chunks = t.chunks
            if chunks is not None and t.id.get_num_chunks() > 1:
                # At least two chunks
                message = self._formatMessage(
                    msg,
                    f"FAIL: File {path.name!r} time coordinates variable "
                    f"{time_name!r} has {t.id.get_num_chunks()} chunks "
                    "(expected 1 chunk or contiguous)",
                )
                raise AssertionError(message)

            # Check for the time bounds variable having one chunk
            if "bounds" in t.attrs:
                bounds = str(np.array(t.attrs["bounds"]).astype("U"))
                if bounds in f:
                    b = f[bounds]
                    chunks = b.chunks
                    if chunks is not None and b.id.get_num_chunks() > 1:
                        # At least two chunks
                        message = self._formatMessage(
                            msg,
                            f"FAIL: File {path.name!r} time bounds variable "
                            f"{bounds!r} has {b.id.get_num_chunks()} chunks "
                            "(expected 1 chunk or contiguous)",
                        )
                        raise AssertionError(message)

        # Check for data variable chunks of at least ~4 MiB.
        if "variable_id" in f.attrs:
            variable_id = str(np.array(f.attrs["variable_id"]).astype("U"))
            if variable_id in f:
                d = f[variable_id]
                chunks = d.chunks
                if chunks is not None and d.id.get_num_chunks() > 1:
                    # At least two chunks
                    wordsize = d.dtype.itemsize
                    chunksize = np.prod(chunks) * wordsize

                    lee_way = 0
                    if len(chunks) > 1:
                        lee_way = np.prod(chunks[1:]) * wordsize

                    if chunksize + lee_way < BYTES_4MiB:
                        message = self._formatMessage(
                            msg,
                            f"FAIL: File {path.name!r} data variable "
                            f"{variable_id!r} has uncompressed chunk size "
                            f"{chunksize} B (expected at least "
                            f"{BYTES_4MiB - lee_way} B or 1 chunk "
                            "or contiguous)",
                        )
                        raise AssertionError(message)


@_requires_tables
class TestCMIP7AutoChunking(unittest.TestCase):
    """Test automatic CMIP7 chunking application."""

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        self.project = cmor4.ProjectTables.from_directory(
            CMIP7_TABLE_ROOT,
            cv_file="tables-cvs/cmor-cvs.json",
            variable_tables=["tables/CMIP7_atmos.json"],
            coordinate_table="tables/CMIP7_coordinate.json",
            formula_table="tables/CMIP7_formula_terms.json",
            grid_table="tables/CMIP7_grids.json",
        )

    def test_cmip7_auto_chunking_applied(self):
        """CMIP7 datasets get automatic chunking when not specified."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        # Create large enough data for meaningful chunks (monthly frequency)
        time_vals = np.array([
            15.0 + 30 * i for i in range(100)
        ])  # 100 monthly timesteps
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(100)])
        lat_vals = np.linspace(-90, 90, 90)
        lon_vals = np.linspace(0, 359, 180)

        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(100, 90, 180).astype(np.float32) * 30 + 270

        ds = cmor4.create_dataset(dataset, variable, axes, data)

        # Check that chunking was applied
        self.assertIn("chunksizes", ds["bldep"].encoding)
        chunksizes = ds["bldep"].encoding["chunksizes"]

        # Time dimension should be full length (single chunk)
        self.assertEqual(chunksizes[0], 100, "Time dimension should be unchunked")
        self.assertEqual(ds["time"].encoding["chunksizes"], (100,))
        self.assertEqual(ds["time_bnds"].encoding["chunksizes"], (100, 2))

        # Calculate chunk size
        itemsize = ds["bldep"].dtype.itemsize
        chunk_bytes = np.prod(chunksizes) * itemsize
        min_size = 4 * 1024 * 1024  # 4 MiB

        self.assertGreaterEqual(
            chunk_bytes,
            min_size,
            f"Chunk size {chunk_bytes} bytes should be ≥ 4 MiB ({min_size} bytes)",
        )

    def test_cmip7_auto_chunking_written_to_file(self):
        """CMIP7 auto-chunking persists when file is written."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(50)])  # 50 monthly timesteps
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(50)])
        lat_vals = np.linspace(-90, 90, 45)
        lon_vals = np.linspace(0, 359, 90)

        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(50, 45, 90).astype(np.float32) * 30 + 270

        ds, path = cmor4.cmorize(dataset, variable, axes, data)

        # Read back and check chunking
        ds_read = xr.open_dataset(path)
        self.assertTrue(
            ds_read["bldep"].chunks is not None
            or ds_read["bldep"].encoding.get("chunksizes") is not None
        )
        ds_read.close()

    def test_non_cmip7_no_auto_chunking(self):
        """Non-CMIP7 datasets don't get automatic chunking."""
        import cmor4

        # Use CMIP6 metadata
        dataset = cmor4.DatasetInfo({**_BASE_CMIP6_DATASET, "outpath": self.tmp})
        variable = cmor4.Variable(
            name="tas", units="K", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(10)])  # 10 monthly timesteps
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(10)])
        lat_vals = np.linspace(-90, 90, 10)
        lon_vals = np.linspace(0, 359, 20)

        axes = [
            cmor4.Axis(
                name="time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            cmor4.Axis(
                name="latitude",
                values=lat_vals,
                bounds=_create_bounds(lat_vals, "latitude"),
                units="degrees_north",
            ),
            cmor4.Axis(
                name="longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
                units="degrees_east",
            ),
        ]

        data = np.random.rand(10, 10, 20).astype(np.float32) * 30 + 270

        ds = cmor4.create_dataset(dataset, variable, axes, data)

        # Should not have automatic chunking
        self.assertNotIn("chunksizes", ds["tas"].encoding)


@_requires_tables
class TestCMIP7ChunkingValidation(unittest.TestCase):
    """Test validation of user-provided chunking for CMIP7."""

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        self.project = cmor4.ProjectTables.from_directory(
            CMIP7_TABLE_ROOT,
            cv_file="tables-cvs/cmor-cvs.json",
            variable_tables=["tables/CMIP7_atmos.json"],
            coordinate_table="tables/CMIP7_coordinate.json",
            formula_table="tables/CMIP7_formula_terms.json",
            grid_table="tables/CMIP7_grids.json",
        )

    def test_compliant_user_chunks_accepted(self):
        """Valid CMIP7-compliant user chunks are accepted."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(50)])  # 50 monthly timesteps
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(50)])
        lat_vals = np.linspace(-90, 90, 90)
        lon_vals = np.linspace(0, 359, 180)

        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(50, 90, 180).astype(np.float32) * 30 + 270

        # Provide compliant chunks: time=50 (full), lat=90, lon=180
        # Size: 50*90*180*4 = 3,240,000 bytes (3.09 MiB) - too small
        # Let's use full chunks: 50*90*180*4 = 3.09 MiB - still too small
        # Need at least 4 MiB, so 50*90*180*4 = 3,240,000 < 4,194,304
        # Use bigger chunks or validate will fail. Let me recalculate.
        # 4,194,304 / 4 = 1,048,576 elements needed
        # 50 * 90 * 180 = 810,000 < 1,048,576
        # So full array chunks won't meet 4MiB with this data

        # Let's create larger data
        time_vals = np.array([
            15.0 + 30 * i for i in range(100)
        ])  # 100 monthly timesteps
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(100)])
        lat_vals = np.linspace(-90, 90, 120)  # 120
        lon_vals = np.linspace(0, 359, 180)  # 180

        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(100, 120, 180).astype(np.float32) * 30 + 270

        # Compliant chunks: 100*120*180*4 = 8,640,000 bytes (8.24 MiB) > 4 MiB ✓
        encoding = {"bldep": {"chunksizes": (100, 120, 180)}}

        # Should not raise
        ds = cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)
        self.assertEqual(ds["bldep"].encoding["chunksizes"], (100, 120, 180))

    def test_data_variable_chunks_along_time_accepted(self):
        """Data chunks may split time when each chunk is large enough."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(200)])
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(200)])
        lat_vals = np.linspace(-90, 90, 64)
        lon_vals = np.linspace(0, 359, 384)

        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.zeros((200, 64, 384), dtype=np.float32)

        # 50*64*384*4 = 4,915,200 bytes, so the data chunk is CMIP7-sized
        # even though the data variable has multiple chunks along time.
        encoding = {"bldep": {"chunksizes": (50, 64, 384)}}

        ds = cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        self.assertEqual(ds["bldep"].encoding["chunksizes"], (50, 64, 384))
        self.assertEqual(ds["time"].encoding["chunksizes"], (200,))
        self.assertEqual(ds["time_bnds"].encoding["chunksizes"], (200, 2))

    def test_top_level_chunksizes_accepted(self):
        """Top-level chunksizes apply to data without splitting time bounds."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(200)])
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(200)])
        lat_vals = np.linspace(-90, 90, 64)
        lon_vals = np.linspace(0, 359, 384)
        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.zeros((200, 64, 384), dtype=np.float32)
        encoding = {"chunksizes": (50, 64, 384)}

        ds = cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        self.assertEqual(ds["bldep"].encoding["chunksizes"], (50, 64, 384))
        self.assertEqual(ds["time"].encoding["chunksizes"], (200,))
        self.assertEqual(ds["time_bnds"].encoding["chunksizes"], (200, 2))

    def test_small_chunks_rejected(self):
        """Chunks smaller than 4 MiB are rejected."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(10)])  # 10 monthly timesteps
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(10)])
        lat_vals = np.linspace(-90, 90, 10)
        lon_vals = np.linspace(0, 359, 20)

        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(10, 10, 20).astype(np.float32) * 30 + 270

        # Invalid: 10*10*20*4 = 8,000 bytes << 4 MiB
        encoding = {"bldep": {"chunksizes": (10, 10, 20)}}

        with self.assertRaises(ValueError) as cm:
            cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        self.assertIn("4 mib", str(cm.exception).lower())

    def test_top_level_small_chunks_rejected(self):
        """Top-level chunksizes are validated against CMIP7 chunk size rules."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(10)])
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(10)])
        lat_vals = np.linspace(-90, 90, 10)
        lon_vals = np.linspace(0, 359, 20)
        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(10, 10, 20).astype(np.float32) * 30 + 270
        encoding = {"chunksizes": (10, 10, 20)}

        with self.assertRaises(ValueError) as cm:
            cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        self.assertIn("4 mib", str(cm.exception).lower())

    def test_time_coordinate_chunksizes_rejected(self):
        """Variable-specific time chunks must keep time and bounds as one chunk."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(200)])
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(200)])
        lat_vals = np.linspace(-90, 90, 64)
        lon_vals = np.linspace(0, 359, 384)
        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]
        data = np.zeros((200, 64, 384), dtype=np.float32)

        cases = [
            {"time": {"chunksizes": (50,)}},
            {"time_bnds": {"chunksizes": (200, 1)}},
        ]
        for encoding in cases:
            with self.subTest(encoding=encoding):
                with self.assertRaises(ValueError) as cm:
                    cmor4.create_dataset(
                        dataset,
                        variable,
                        axes,
                        data,
                        encoding=encoding,
                    )

                self.assertIn("single chunk", str(cm.exception).lower())


@_requires_tables
class TestCMIP7EncodingParameters(unittest.TestCase):
    """Test that non-chunking encoding parameters work with CMIP7."""

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        self.project = cmor4.ProjectTables.from_directory(
            CMIP7_TABLE_ROOT,
            cv_file="tables-cvs/cmor-cvs.json",
            variable_tables=["tables/CMIP7_atmos.json"],
            coordinate_table="tables/CMIP7_coordinate.json",
            formula_table="tables/CMIP7_formula_terms.json",
            grid_table="tables/CMIP7_grids.json",
        )

    def test_compression_with_cmip7_chunks(self):
        """Compression encoding works alongside CMIP7 chunking."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([
            15.0 + 30 * i for i in range(100)
        ])  # 100 monthly timesteps
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(100)])
        lat_vals = np.linspace(-90, 90, 120)
        lon_vals = np.linspace(0, 359, 180)

        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(100, 120, 180).astype(np.float32) * 30 + 270

        # Provide compression alongside CMIP7 chunks
        encoding = {
            "bldep": {
                "chunksizes": (100, 120, 180),
                "zlib": True,
                "complevel": 4,
            }
        }

        ds = cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        # Check both chunking and compression are set
        self.assertEqual(ds["bldep"].encoding["chunksizes"], (100, 120, 180))
        self.assertEqual(ds["bldep"].encoding.get("zlib"), True)
        self.assertEqual(ds["bldep"].encoding.get("complevel"), 4)

    def test_compression_with_auto_chunks(self):
        """Compression works with auto-applied CMIP7 chunks."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([
            15.0 + 30 * i for i in range(100)
        ])  # 100 monthly timesteps
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(100)])
        lat_vals = np.linspace(-90, 90, 120)
        lon_vals = np.linspace(0, 359, 180)

        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(100, 120, 180).astype(np.float32) * 30 + 270

        # Only provide compression, let chunking be auto-applied
        encoding = {"bldep": {"zlib": True, "complevel": 4}}

        ds = cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        # Check chunking was auto-applied
        self.assertIn("chunksizes", ds["bldep"].encoding)
        # Check compression was applied
        self.assertEqual(ds["bldep"].encoding.get("zlib"), True)
        self.assertEqual(ds["bldep"].encoding.get("complevel"), 4)

    def test_top_level_compression_and_quantization(self):
        """Top-level encoding keys apply to the primary variable."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(100)])
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(100)])
        lat_vals = np.linspace(-90, 90, 120)
        lon_vals = np.linspace(0, 359, 180)
        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(100, 120, 180).astype(np.float32) * 30 + 270
        encoding = {
            "zlib": True,
            "complevel": 4,
            "least_significant_digit": 2,
        }

        ds = cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        self.assertIn("chunksizes", ds["bldep"].encoding)
        self.assertEqual(ds["bldep"].encoding.get("zlib"), True)
        self.assertEqual(ds["bldep"].encoding.get("complevel"), 4)
        self.assertEqual(ds["bldep"].encoding.get("least_significant_digit"), 2)

    def test_top_level_and_variable_specific_encoding_keys_pass_through(self):
        """CMOR4 only special-cases chunksizes and passes other encoding keys."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(100)])
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(100)])
        lat_vals = np.linspace(-90, 90, 120)
        lon_vals = np.linspace(0, 359, 180)
        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(100, 120, 180).astype(np.float32) * 30 + 270
        encoding = {
            "backend_specific_option": "global",
            "bldep": {
                "another_backend_option": 7,
            },
            "time": {
                "time_backend_option": "coordinate",
            },
        }

        ds = cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        self.assertEqual(ds["bldep"].encoding.get("backend_specific_option"), "global")
        self.assertEqual(ds["bldep"].encoding.get("another_backend_option"), 7)
        self.assertEqual(ds["time"].encoding.get("backend_specific_option"), "global")
        self.assertEqual(ds["time"].encoding.get("time_backend_option"), "coordinate")
        self.assertEqual(
            ds["time_bnds"].encoding.get("backend_specific_option"), "global"
        )
        self.assertNotIn("another_backend_option", ds["time"].encoding)

    def test_variable_encoding_overrides_top_level_encoding(self):
        """Variable-specific encoding has precedence over top-level defaults."""
        import cmor4

        dataset = self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )

        time_vals = np.array([15.0 + 30 * i for i in range(100)])
        time_bnds = np.array([[30 * i, 30 * (i + 1)] for i in range(100)])
        lat_vals = np.linspace(-90, 90, 120)
        lon_vals = np.linspace(0, 359, 180)
        axes = [
            self.project.axis(
                "time",
                values=time_vals,
                bounds=time_bnds,
                units="days since 2000-01-01",
            ),
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

        data = np.random.rand(100, 120, 180).astype(np.float32) * 30 + 270
        encoding = {
            "chunksizes": (100, 120, 180),
            "zlib": True,
            "complevel": 1,
            "bldep": {
                "chunksizes": (50, 120, 180),
                "zlib": False,
                "least_significant_digit": 3,
            },
        }

        ds = cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        self.assertEqual(ds["bldep"].encoding["chunksizes"], (50, 120, 180))
        self.assertEqual(ds["bldep"].encoding.get("zlib"), False)
        self.assertEqual(ds["bldep"].encoding.get("complevel"), 1)
        self.assertEqual(ds["bldep"].encoding.get("least_significant_digit"), 3)


@_requires_tables
class TestCheckCMIP7RepackChunking(unittest.TestCase, CMIP7ChunkingCheckAssertion):
    """Test if files generated have chunking that meets CMIP7 repack requirements."""

    def setUp(self):
        import cmor4

        self.tmp = tempfile.mkdtemp()
        self.project = cmor4.ProjectTables.from_directory(
            CMIP7_TABLE_ROOT,
            cv_file="tables-cvs/cmor-cvs.json",
            variable_tables=[
                "tables/CMIP7_atmos.json",
                "tables/CMIP7_land.json",
                "tables/CMIP7_ocean.json",
                "tables/CMIP7_seaIce.json",
            ],
            coordinate_table="tables/CMIP7_coordinate.json",
            formula_table="tables/CMIP7_formula_terms.json",
            grid_table="tables/CMIP7_grids.json",
        )

    def _dataset(self):
        return self.project.dataset_info({
            **_BASE_CMIP7_DATASET,
            "outpath": self.tmp,
        })

    def _time_axis(self, count=100, *, point=False):
        time_vals = np.array([15.0 + 30 * i for i in range(count)])
        name = "time1" if point else "time"
        kwargs = {
            "values": time_vals,
            "units": "days since 2000-01-01",
        }
        if not point:
            kwargs["bounds"] = np.array([[30 * i, 30 * (i + 1)] for i in range(count)])
        return self.project.axis(name, **kwargs)

    def _horizontal_axes(self, lat_count=120, lon_count=180):
        lat_vals = np.linspace(-90, 90, lat_count)
        lon_vals = np.linspace(0, 359, lon_count)
        return [
            self.project.axis(
                "latitude", values=lat_vals, bounds=_create_bounds(lat_vals, "latitude")
            ),
            self.project.axis(
                "longitude",
                values=lon_vals,
                bounds=_create_bounds(lon_vals, "longitude"),
            ),
        ]

    def _assert_chunking_compliant_file(self, variable, axes, data, *, encoding=None):
        import cmor4

        ds, path = cmor4.cmorize(
            self._dataset(),
            variable,
            axes,
            data,
            encoding=encoding,
        )

        self.assertCMIP7Chunking(path)
        return path

    def test_auto_chunked_file_passes_check_cmip7_chunking(self):
        """Auto-chunked time-mean latitude/longitude files pass the chunking checks."""
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )
        axes = [
            self._time_axis(),
            *self._horizontal_axes(),
        ]
        data = np.zeros((100, 120, 180), dtype=np.float32)

        self._assert_chunking_compliant_file(variable, axes, data)

    def test_multi_chunk_file_passes_check_cmip7_chunking(self):
        """Files with multiple compliant data chunks pass the chunking checks."""
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )
        axes = [
            self._time_axis(),
            *self._horizontal_axes(lat_count=64, lon_count=384),
        ]
        data = np.zeros((100, 64, 384), dtype=np.float32)
        encoding = {"bldep": {"chunksizes": (100, 32, 384)}}

        self._assert_chunking_compliant_file(
            variable,
            axes,
            data,
            encoding=encoding,
        )

    def test_multiple_chunks_along_time_passes_check_cmip7_chunking(self):
        """File with multiple compliant data chunks along time"""
        num_times = 1000
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )
        axes = [
            self._time_axis(count=num_times),
            *self._horizontal_axes(lat_count=64, lon_count=384),
        ]
        data = np.zeros((num_times, 64, 384), dtype=np.float32)
        encoding = {"bldep": {"chunksizes": (100, 64, 384)}}

        self._assert_chunking_compliant_file(
            variable,
            axes,
            data,
            encoding=encoding,
        )

    def test_top_level_chunksizes_passes_check_cmip7_chunking(self):
        """Top-level chunksizes can split data time but not time coordinates."""
        num_times = 1000
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )
        axes = [
            self._time_axis(count=num_times),
            *self._horizontal_axes(lat_count=64, lon_count=384),
        ]
        data = np.zeros((num_times, 64, 384), dtype=np.float32)
        encoding = {"chunksizes": (100, 64, 384)}

        self._assert_chunking_compliant_file(
            variable,
            axes,
            data,
            encoding=encoding,
        )

    def test_time_point_file_passes_check_cmip7_chunking(self):
        """Time-point files pass when the time coordinate is stored as one chunk."""
        variable = self.project.variable(
            "bldep_tpt-u-hxy-u", missing_value=np.float32(1.0e20)
        )
        axes = [
            self._time_axis(point=True),
            *self._horizontal_axes(),
        ]
        data = np.zeros((100, 120, 180), dtype=np.float32)

        self._assert_chunking_compliant_file(variable, axes, data)

    def test_landuse_time_point_file_passes_check_cmip7_chunking(self):
        """Four-dimensional files with a categorical axis pass the chunking checks."""
        variable = self.project.variable(
            "fracLut_tpt-u-hxy-u",
            table_id="land",
            missing_value=np.float32(1.0e20),
        )
        axes = [
            self._time_axis(point=True),
            self.project.axis(
                "landuse",
                values=[
                    "primary_and_secondary_land",
                    "pastures",
                    "crops",
                    "urban",
                ],
            ),
            *self._horizontal_axes(lat_count=64, lon_count=64),
        ]
        data = np.zeros((100, 4, 64, 64), dtype=np.float32)

        self._assert_chunking_compliant_file(variable, axes, data)

    def test_ocean_and_sea_ice_surface_files_pass_check_cmip7_chunking(self):
        """Non-atmospheric CMIP7 surface files pass the chunking checks."""
        cases = [
            ("tos_tavg-u-hxy-sea", "ocean", "tos", -1.8),
            ("siconc_tavg-u-hxy-u", "seaIce", "siconc", 0.0),
        ]

        for variable_name, table_id, out_name, value in cases:
            with self.subTest(variable=variable_name):
                variable = self.project.variable(
                    variable_name,
                    table_id=table_id,
                    missing_value=np.float32(1.0e20),
                )
                axes = [
                    self._time_axis(),
                    *self._horizontal_axes(),
                ]
                data = np.full((100, 120, 180), value, dtype=np.float32)

                path = self._assert_chunking_compliant_file(variable, axes, data)
                with xr.open_dataset(path) as ds:
                    self.assertEqual(ds.attrs["variable_id"], out_name)


if __name__ == "__main__":
    unittest.main()
