from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

import cmor4
from table_helpers import cmip7_project, drcdp_project, obs4mips_project


def guide_time_axis(values, bounds, units, calendar="standard"):
    return {
        "name": "time",
        "values": values,
        "bounds": bounds,
        "units": units,
        "calendar": calendar,
    }


def guide_lat_axis(values=(-45.0, 45.0)):
    values = list(values)
    return {
        "name": "latitude",
        "values": values,
        "bounds": _regular_bounds(values),
        "units": "degrees_north",
    }


def guide_lon_axis(values=(90.0, 180.0, 270.0)):
    values = list(values)
    return {
        "name": "longitude",
        "values": values,
        "bounds": _regular_bounds(values),
        "units": "degrees_east",
    }


def _regular_bounds(values):
    if len(values) == 1:
        return [[values[0] - 0.5, values[0] + 0.5]]
    edges = [(left + right) / 2.0 for left, right in zip(values[:-1], values[1:])]
    first = values[0] - (edges[0] - values[0])
    last = values[-1] + (values[-1] - edges[-1])
    edges = [first, *edges, last]
    return [[edges[index], edges[index + 1]] for index in range(len(values))]


def drcdp_info(tmp_path: Path, source_id="EDDE2-0", institution_id="EPA"):
    return {
        "Conventions": "CF-1.7 CMIP-6.5",
        "activity_id": "DRCDP",
        "driving_activity_id": "CMIP",
        "driving_experiment_id": "historical",
        "driving_mip_era": "CMIP6",
        "driving_source_id": "ACCESS-CM2",
        "driving_variant_label": "r1i1p1f1",
        "grid_label": "gn",
        "institution_id": institution_id,
        "outpath": str(tmp_path),
        "output_file_template": (
            "<variable_id><region_id><institution_id><source_id>"
            "<driving_mip_era><driving_experiment_id>"
            "<driving_source_id><driving_variant_label><frequency>"
        ),
        "output_path_template": (
            "<activity_id><region_id><institution_id><source_id>"
            "<driving_mip_era><driving_activity_id>"
            "<driving_experiment_id><driving_source_id>"
            "<driving_variant_label><frequency><variable_id><version>"
        ),
        "region_id": "NAM",
        "source_id": source_id,
        "version": "v20260512",
    }


def obs4mips_info(
    tmp_path: Path,
    source_id: str,
    institution_id: str,
    license_text: str,
    grid_label="gn",
):
    return {
        "Conventions": "CF-1.11; ODS-2.5",
        "activity_id": "obs4MIPs",
        "calendar": "standard",
        "contact": "submissions-obs4mips@wcrp-cmip.org",
        "grid_label": grid_label,
        "has_aux_unc": "FALSE",
        "institution_id": institution_id,
        "license": license_text,
        "nominal_resolution": "250 km",
        "outpath": str(tmp_path),
        "output_file_template": (
            "<variable_id><frequency><source_id><variant_label>" "<grid_label>"
        ),
        "output_path_template": (
            "<activity_id><institution_id><source_id><frequency>"
            "<variable_id><grid_label><version>"
        ),
        "processing_code_location": (
            "dataset_guides/obs4mips/example-data-tools/example.py"
        ),
        "product": "observations",
        "references": "Example reference",
        "source_data_url": "https://example.invalid/source",
        "source_id": source_id,
        "variant_label": "CMORGuide",
        "version": "v20260512",
    }


class DatasetGuideProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmip7_atmos_project = cmip7_project("tables/CMIP7_atmos.json")
        cls.drcdp_ap1hr_project = drcdp_project("Tables/DRCDP_AP1hr.json")
        cls.drcdp_apday_project = drcdp_project("Tables/DRCDP_APday.json")
        cls.obs4mips_amon_project = obs4mips_project(
            "Tables/obs4MIPs_Amon.json"
        )
        cls.obs4mips_a1hrpt_project = obs4mips_project(
            "Tables/obs4MIPs_A1hrPt.json"
        )

    def test_drcdp_hourly_precipitation_uses_project_drs_template(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = drcdp_info(Path(tmp_dir))
            variable = {
                "name": "pr",
                "missing_value": np.float32(1.0e20),
            }
            axes = [
                guide_time_axis(
                    [23.5, 24.5],
                    [[23.0, 24.0], [24.0, 25.0]],
                    "hours since 2008-12-31",
                ),
                guide_lat_axis(),
                guide_lon_axis(),
            ]

            result = cmor4.cmorize(
                info,
                variable,
                axes,
                np.ones((2, 2, 3), dtype="f4"),
                project=self.drcdp_ap1hr_project,
            )

            self.assertEqual(
                result.path.name,
                "pr_NAM_EPA_EDDE2-0_CMIP6_historical_ACCESS-CM2_"
                "r1i1p1f1_1hr_200812312330-200901010030.nc",
            )
            self.assertIn(
                "DRCDP/NAM/EPA/EDDE2-0/CMIP6/CMIP/historical/"
                "ACCESS-CM2/r1i1p1f1/1hr/pr/v20260512",
                str(result.path),
            )
            with xr.open_dataset(result.path, decode_times=False) as opened:
                ds = opened.load()
            self.assertEqual(ds.attrs["activity_id"], "DRCDP")
            self.assertEqual(ds.attrs["frequency"], "1hr")
            self.assertEqual(ds["pr"].dims, ("time", "lat", "lon"))

    def test_drcdp_tasmax_auto_adds_table_height2m_scalar(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = drcdp_info(Path(tmp_dir), source_id="LOCA2-1")
            variable = {
                "name": "tasmax",
                "missing_value": np.float32(1.0e20),
            }
            axes = [
                guide_time_axis(
                    [39811.0, 39812.0],
                    [[39810.5, 39811.5], [39811.5, 39812.5]],
                    "days since 1900-01-01",
                ),
                guide_lat_axis((32.0, 33.0, 34.0)),
                guide_lon_axis((240.0, 241.0, 242.0, 243.0)),
            ]

            ds = cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((2, 3, 4), dtype="f4"),
                project=self.drcdp_apday_project,
            )

            self.assertEqual(ds["tasmax"].dims, ("time", "lat", "lon"))
            self.assertEqual(ds["height"].shape, ())
            self.assertEqual(float(ds["height"].values), 2.0)
            self.assertEqual(ds["height"].attrs["standard_name"], "height")
            self.assertEqual(ds["tasmax"].attrs["coordinates"], "height")

    def test_cmip7_tas_auto_adds_table_height2m_scalar(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = {
                "activity_id": "CMIP",
                "calendar": "360_day",
                "experiment_id": "amip",
                "forcing_index": "f3",
                "frequency": "mon",
                "grid_label": "g999",
                "initialization_index": "i1",
                "institution_id": "MOHC",
                "license_id": "CC-BY-4.0",
                "nominal_resolution": "100 km",
                "outpath": str(tmp_dir),
                "physics_index": "p1",
                "realization_index": "r9",
                "region": "glb",
                "source_id": "DUMMY-MODEL",
            }
            axes = [
                guide_time_axis(
                    [15.0, 45.0],
                    [[0.0, 30.0], [30.0, 60.0]],
                    "days since 1979-01-01",
                    calendar="360_day",
                ),
                guide_lat_axis((10.0, 20.0, 30.0)),
                guide_lon_axis((0.0, 90.0, 180.0, 270.0)),
            ]

            ds = cmor4.create_dataset(
                info,
                {"name": "tas_tavg-h2m-hxy-u"},
                axes,
                np.ones((2, 3, 4), dtype="f4"),
                project=self.cmip7_atmos_project,
            )

            self.assertEqual(ds["tas"].dims, ("time", "lat", "lon"))
            self.assertEqual(float(ds["height"].values), 2.0)
            self.assertEqual(ds["tas"].attrs["coordinates"], "height")

    def test_drcdp_tasmax_grid_crs_dataset_shape(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = drcdp_info(
                Path(tmp_dir), source_id="MACA3-0", institution_id="UCM-ACSL"
            )
            variable = {
                "name": "tasmax",
                "missing_value": np.float32(1.0e20),
            }
            axes = [
                guide_time_axis(
                    [39813.0, 39814.0],
                    [[39812.5, 39813.5], [39813.5, 39814.5]],
                    "days since 1900-01-01",
                ),
                guide_lat_axis((25.0, 26.0)),
                guide_lon_axis((240.0, 241.0, 242.0)),
                {
                    "name": "height2m",
                    "values": [2.0],
                    "units": "m",
                    "standard_name": "height",
                    "long_name": "height",
                    "axis": "Z",
                    "positive": "up",
                    "scalar": True,
                },
                {
                    "name": "latitude",
                    "out_name": "latitude",
                    "values": [[25.0, 25.1, 25.2], [26.0, 26.1, 26.2]],
                    "dimensions": ["latitude", "longitude"],
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "bounds": np.zeros((2, 3, 4)),
                    "bounds_name": "vertices_latitude",
                    "bounds_dim": "vertices",
                    "auxiliary": True,
                },
                {
                    "name": "longitude",
                    "out_name": "longitude",
                    "values": [[240.0, 241.0, 242.0], [240.0, 241.0, 242.0]],
                    "dimensions": ["latitude", "longitude"],
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "bounds": np.zeros((2, 3, 4)),
                    "bounds_name": "vertices_longitude",
                    "bounds_dim": "vertices",
                    "auxiliary": True,
                },
            ]
            grid = {
                "mapping_name": "latitude_longitude",
                "params": {
                    "longitude_of_prime_meridian": [0.0, "degrees_east"],
                    "semi_major_axis": [6378137.0, "m"],
                    "inverse_flattening": 298.257223563,
                },
            }

            ds = cmor4.create_dataset(
                info,
                variable,
                axes,
                np.ones((2, 2, 3), dtype="f4"),
                grid=grid,
                project=self.drcdp_apday_project,
            )

            self.assertEqual(ds["tasmax"].dims, ("time", "lat", "lon"))
            self.assertEqual(
                ds["tasmax"].attrs["coordinates"], "height latitude longitude"
            )
            self.assertEqual(ds["tasmax"].attrs["grid_mapping"], "crs")
            self.assertEqual(
                ds["crs"].attrs["grid_mapping_name"], "latitude_longitude"
            )

    def test_obs4mips_monthly_gridded_precipitation_template(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = obs4mips_info(
                Path(tmp_dir),
                "CMAP-V1902",
                "NOAA-NCEI",
                self.obs4mips_amon_project.cv["license"],
            )
            info["grid"] = "1x1 degree latitude x longitude"
            variable = {
                "name": "pr",
                "missing_value": np.float32(1.0e20),
            }
            axes = [
                guide_time_axis(
                    [15.0, 45.0],
                    [[0.0, 30.0], [30.0, 60.0]],
                    "days since 1979-01-01",
                ),
                guide_lat_axis((-45.0, 0.0)),
                guide_lon_axis((90.0, 180.0, 270.0)),
            ]

            result = cmor4.cmorize(
                info,
                variable,
                axes,
                np.ones((2, 2, 3), dtype="f4"),
                project=self.obs4mips_amon_project,
            )

            self.assertEqual(
                result.path.name,
                "pr_mon_CMAP-V1902_CMORGuide_gn_197901-197902.nc",
            )
            self.assertIn(
                "obs4MIPs/NOAA-NCEI/CMAP-V1902/mon/pr/gn/v20260512",
                str(result.path),
            )

    def test_obs4mips_point_site_precipitation_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = obs4mips_info(
                Path(tmp_dir),
                "ARMBE-atm-c1-1-8",
                "DOE-ARM",
                self.obs4mips_a1hrpt_project.cv["license"],
            )
            info.update(
                {
                    "grid": "site",
                    "nominal_resolution": "site",
                    "product": "site-observations",
                    "site_id": "AR-SLu",
                    "site_location": "San Luis",
                }
            )
            variable = {
                "name": "pr",
            }
            axes = [
                guide_time_axis(
                    [0.5, 1.5],
                    [[0.0, 1.0], [1.0, 2.0]],
                    "hours since 2018-01-01",
                ),
                {
                    "name": "latitude1",
                    "out_name": "lat",
                    "values": [36.605],
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "axis": "Y",
                },
                {
                    "name": "longitude1",
                    "out_name": "lon",
                    "values": [262.515],
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "axis": "X",
                },
            ]

            result = cmor4.cmorize(
                info,
                variable,
                axes,
                np.ones((2, 1, 1), dtype="f4"),
                project=self.obs4mips_a1hrpt_project,
            )

            self.assertEqual(
                result.path.name,
                "pr_1hr_ARMBE-atm-c1-1-8_CMORGuide_gn_"
                "201801010030-201801010130.nc",
            )
            self.assertEqual(result.dataset.attrs["site_id"], "AR-SLu")
            self.assertEqual(result.dataset["pr"].dims, ("time", "lat", "lon"))

    def test_obs4mips_zonal_mean_o3zm_writes_o3_variable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            info = obs4mips_info(
                Path(tmp_dir),
                "BSVertOzone-v1-0",
                "DLR-BIRA",
                self.obs4mips_amon_project.cv["license"],
                grid_label="gnz",
            )
            info.update(
                {
                    "grid": "5 degree latitude height zonal mean",
                    "nominal_resolution": "500 km",
                }
            )
            variable = {
                "name": "o3zm",
                "missing_value": np.float32(1.0e20),
            }
            axes = [
                guide_time_axis(
                    [15.0, 45.0],
                    [[0.0, 30.0], [30.0, 60.0]],
                    "days since 1979-01-01",
                ),
                {
                    "name": "height",
                    "values": [1000.0, 5000.0, 10000.0],
                    "units": "m",
                    "standard_name": "height",
                    "long_name": "height",
                    "axis": "Z",
                    "positive": "up",
                },
                guide_lat_axis((-60.0, -30.0)),
            ]

            result = cmor4.cmorize(
                info,
                variable,
                axes,
                np.ones((2, 3, 2), dtype="f4"),
                project=self.obs4mips_amon_project,
            )

            self.assertEqual(
                result.path.name,
                "o3_mon_BSVertOzone-v1-0_CMORGuide_gnz_197901-197902.nc",
            )
            self.assertIn(
                "obs4MIPs/DLR-BIRA/BSVertOzone-v1-0/mon/o3/gnz/v20260512",
                str(result.path),
            )
            self.assertIn("o3", result.dataset)
            self.assertNotIn("o3zm", result.dataset)


if __name__ == "__main__":
    unittest.main()
