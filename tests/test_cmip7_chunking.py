"""Tests for CMIP7 chunking validation and auto-application."""

from __future__ import annotations

import sys
import sysconfig
import subprocess
import tempfile
import unittest
from pathlib import Path

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


def _has_check_cmip7_repack():
    """Check if check_cmip7_repack tool is available."""
    try:
        # Try command line tool first
        result = subprocess.run(
            ["check_cmip7_repack", "--help"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try Python module script
    try:
        import sys
        import sysconfig

        site_packages = sysconfig.get_path("purelib")
        check_script = Path(site_packages) / "cmip7_repack" / "check_cmip7_packing"
        if check_script.exists():
            result = subprocess.run(
                [sys.executable, str(check_script), "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
    except Exception:
        pass

    return False


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


class CMIP7RepackCheckAssertion:
    """Call cmip7-repack-check on NetCDF files to tests CMIP7 repack compliance."""

    def assertCMIP7Repack(self, path: Path, msg: str = None):
        # Run check_cmip7_repack on the output file
        # Try command line tool first, then Python script
        try:
            proc = subprocess.run(
                ["check_cmip7_repack", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except FileNotFoundError:
            # Use Python script directly
            site_packages = sysconfig.get_path("purelib")
            check_script = Path(site_packages) / "cmip7_repack" / "check_cmip7_packing"
            proc = subprocess.run(
                [sys.executable, str(check_script), str(path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

        if proc.returncode != 0:
            default_msg = (
                "check_cmip7_repack failed:\n"
                f"stdout: {proc.stdout}\n"
                f"stderr: {proc.stderr}"
            )
            full_msg = self._formatMessage(msg, default_msg)
            raise AssertionError(full_msg)


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

        result = cmor4.cmorize(dataset, variable, axes, data)

        # Read back and check chunking
        ds_read = xr.open_dataset(result.path)
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

    def test_chunked_time_rejected(self):
        """Time dimension with chunks < full length is rejected."""
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

        # Invalid: time is chunked (10 instead of 100)
        encoding = {"bldep": {"chunksizes": (10, 90, 180)}}

        with self.assertRaises(ValueError) as cm:
            cmor4.create_dataset(dataset, variable, axes, data, encoding=encoding)

        self.assertIn("time coordinate", str(cm.exception).lower())
        self.assertIn("single chunk", str(cm.exception).lower())

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


@unittest.skipUnless(_has_check_cmip7_repack(), "check_cmip7_repack not installed")
@_requires_tables
class TestCheckCMIP7RepackIntegration(unittest.TestCase, CMIP7RepackCheckAssertion):
    """Test integration with check_cmip7_repack tool."""

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

    def _assert_repack_compliant_file(self, variable, axes, data, *, encoding=None):
        import cmor4

        result = cmor4.cmorize(
            self._dataset(),
            variable,
            axes,
            data,
            encoding=encoding,
        )

        self.assertCMIP7Repack(result.path)
        return result.path

    def test_auto_chunked_file_passes_check_cmip7_repack(self):
        """Auto-chunked time-mean latitude/longitude files pass the repack check."""
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )
        axes = [
            self._time_axis(),
            *self._horizontal_axes(),
        ]
        data = np.zeros((100, 120, 180), dtype=np.float32)

        self._assert_repack_compliant_file(variable, axes, data)

    def test_multi_chunk_file_passes_check_cmip7_repack(self):
        """Files with multiple compliant data chunks pass the repack check."""
        variable = self.project.variable(
            "bldep_tavg-u-hxy-u", missing_value=np.float32(1.0e20)
        )
        axes = [
            self._time_axis(),
            *self._horizontal_axes(lat_count=64, lon_count=384),
        ]
        data = np.zeros((100, 64, 384), dtype=np.float32)
        encoding = {"bldep": {"chunksizes": (100, 32, 384)}}

        self._assert_repack_compliant_file(
            variable,
            axes,
            data,
            encoding=encoding,
        )

    def test_time_point_file_passes_check_cmip7_repack(self):
        """Time-point files pass when the time coordinate is stored as one chunk."""
        variable = self.project.variable(
            "bldep_tpt-u-hxy-u", missing_value=np.float32(1.0e20)
        )
        axes = [
            self._time_axis(point=True),
            *self._horizontal_axes(),
        ]
        data = np.zeros((100, 120, 180), dtype=np.float32)

        self._assert_repack_compliant_file(variable, axes, data)

    def test_landuse_time_point_file_passes_check_cmip7_repack(self):
        """Four-dimensional files with a categorical axis pass the repack check."""
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

        self._assert_repack_compliant_file(variable, axes, data)

    def test_ocean_and_sea_ice_surface_files_pass_check_cmip7_repack(self):
        """Non-atmospheric CMIP7 surface files pass the repack check."""
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

                path = self._assert_repack_compliant_file(variable, axes, data)
                with xr.open_dataset(path) as ds:
                    self.assertEqual(ds.attrs["variable_id"], out_name)


if __name__ == "__main__":
    unittest.main()
