"""Behavioral boundaries that must survive internal module reorganization."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import cmor
from cmor.dataset import create_dataset_from_validated_data
from cmor import _assembly, _netcdf
from cmor import axis as axis_module
from cmor._metadata import DatasetMetadata
from cmor._netcdf import prepare_append_encoding
from cmor._coordinate_tables import CoordinateTableEntry, GridTableDocument
from cmor._time import find_time_axis
from cmor._assembly import validate_metadata


@pytest.mark.parametrize(
    ("metadata", "validation", "forecast", "writer"),
    [
        ({"name": "time"}, False, True, True),
        ({"name": "time_counter"}, False, False, True),
        ({"name": "sample", "table_entry": "time"}, False, True, True),
        ({"name": "sample", "standard_name": "time"}, True, True, True),
        ({"name": "sample", "standard_name": "time_counter"}, False, False, True),
        ({"name": "sample", "axis": "T"}, True, True, True),
        ({"name": "sample"}, False, False, False),
    ],
)
def test_distinct_time_axis_recognition(metadata, validation, forecast, writer):
    axis = cmor.Axis(**metadata, values=[0.0])
    assert axis_module._is_time_axis(axis) is validation
    assert _assembly._is_time_axis(axis) is forecast
    if writer:
        assert find_time_axis([axis]) == (0, axis)
    else:
        with pytest.raises(ValueError, match="No time axis"):
            find_time_axis([axis])


def test_forecast_roles_preserve_first_match_and_exclude_reference_from_time():
    reference = cmor.Axis(
        name="reftime",
        standard_name="forecast_reference_time",
        axis="T",
        values=[1.0],
        scalar=True,
        units="days since 2000-01-01",
    )
    time = cmor.Axis(name="time", values=[2.0, 3.0], units=reference.units)
    axes = [
        reference,
        reference.updated(name="reftime_other", values=[100.0]),
        time,
        time.updated(name="later", axis="T", values=[200.0, 300.0]),
    ]
    coords = _assembly.build_axis_mappings(axes)[0]
    assert find_time_axis(axes) == (0, reference)
    assert _assembly.derive_forecast_coords(axes, coords) == "leadtime"
    np.testing.assert_array_equal(coords["leadtime"][1], [1.0, 2.0])


def test_grid_document_normalization_preserves_input_and_key_precedence():
    raw = {
        "axis_entry": {"x": {"name": "wrong_axis", "axis": "X"}},
        "variable_entry": {"latitude": {"name": "wrong_coord", "units": ""}},
        "mapping_entry": {"rotated": {"name": "wrong_mapping"}},
    }
    original = deepcopy(raw)
    document = GridTableDocument.model_validate(raw)
    assert raw == original
    assert document.axis_entry["x"].name == "x"
    assert document.variable_entry["latitude"].name == "latitude"
    assert document.mapping_entry["rotated"].name == "rotated"
    assert document.variable_entry["latitude"].is_grid_coord
    assert document.variable_entry["latitude"].units is None
    typed = CoordinateTableEntry(name="latitude")
    typed_document = GridTableDocument.model_validate({
        "variable_entry": {"latitude": typed}
    })
    assert not typed.is_grid_coord
    assert typed_document.variable_entry["latitude"] is not typed
    assert typed_document.variable_entry["latitude"].is_grid_coord


def test_project_reads_grid_file_once_and_orders_metadata_errors(tmp_path, monkeypatch):
    documents = {
        "cv": {"CV": {}},
        "variables": {"variable_entry": {"tas": {"units": "K", "dimensions": "x"}}},
        "coordinates": {"axis_entry": {"x": {"units": "m"}}},
        "grid": {"variable_entry": {"latitude": {"units": "degrees_north"}}},
    }
    for name, data in documents.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    reads = []
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    project = cmor.ProjectTables(
        tmp_path / "cv.json",
        [tmp_path / "variables.json"],
        coordinate_table=tmp_path / "coordinates.json",
        grid_table=tmp_path / "grid.json",
    )
    assert reads.count(tmp_path / "grid.json") == 1
    axis = cmor.Axis(name="x", values=[0.0], units="s")
    bad_variable = cmor.Variable(name="tas", units="s", dimensions=("x",))
    with pytest.raises(cmor.TableValidationError, match="grid axis"):
        project.validate_dataset(None, bad_variable, [axis.updated(isgridaxis=True)])
    with pytest.raises(cmor.TableValidationError, match="units") as error:
        project.validate_dataset(None, bad_variable, [axis])
    assert "tas" in str(error.value)
    with pytest.raises(cmor.TableValidationError, match="axis 'x'"):
        project.validate_dataset(None, project.variable("tas"), [axis])


def test_cv_serializes_once_and_preserves_required_error_order(monkeypatch):
    cv = cmor.ControlledVocabulary({
        "CV": {
            "required_global_attributes": ["institution"],
            "institution_id": {"test": "Expected institution"},
            "forcing": ["GHG"],
        }
    })
    metadata = DatasetMetadata(
        institution_id="test", institution="Expected institution", forcing="GHG"
    )
    serialize = DatasetMetadata.to_dict
    calls = []

    def to_dict(self):
        calls.append(self)
        return serialize(self)

    monkeypatch.setattr(DatasetMetadata, "to_dict", to_dict)
    cv.validate_dataset_info(metadata)
    assert calls == [metadata]
    invalid = DatasetMetadata(
        institution_id="test", institution="Wrong institution", forcing="UNKNOWN"
    )
    with pytest.raises(cmor.ControlledVocabularyError, match="institution="):
        cv.validate_dataset_info(invalid)
    with pytest.raises(cmor.ControlledVocabularyError, match="forcing term"):
        cv.validate_dataset_info(invalid, check_required=False)


def _simple_metadata():
    return (
        cmor.DatasetInfo(calendar="360_day", frequency="day"),
        cmor.Variable(name="tas", units="K", dimensions=("time",), chunksizes=(1,)),
        [
            cmor.Axis(
                name="time",
                axis="T",
                values=[0.5, 1.5, 2.5, 3.5],
                bounds=[0.0, 1.0, 2.0, 3.0, 4.0],
                units="days since 2000-01-01",
            )
        ],
    )


@pytest.mark.parametrize(
    "encoding, chunks",
    [
        ({}, (1,)),
        ({"chunksizes": (2,)}, (2,)),
        ({"chunksizes": (2,), "tas": {"chunksizes": (4,)}}, (4,)),
    ],
)
def test_public_entrypoints_and_encoding_precedence(tmp_path, encoding, chunks):
    dataset, variable, axes = _simple_metadata()
    encoding = {"zlib": True, "complevel": 1, **encoding}
    encoding.setdefault("tas", {})["complevel"] = 2
    encoding["time"] = {"_FillValue": -99.0}
    ctx = validate_metadata(dataset, variable, axes)
    ds = create_dataset_from_validated_data(ctx, np.arange(4.0), encoding=encoding)
    assert ds.tas.encoding["chunksizes"] == chunks
    assert ds.tas.encoding["complevel"] == 2
    assert ds.time.encoding["_FillValue"] is None
    assert ds.time_bnds.encoding["_FillValue"] is None
    paths = [
        entry(
            dataset,
            variable,
            axes,
            np.arange(4.0),
            path=tmp_path / f"{entry.__name__}.nc",
            encoding=encoding,
        )
        for entry in (cmor.cmorize, cmor.create_dataset)
    ]
    with (
        xr.open_dataset(paths[0], decode_times=False) as left,
        xr.open_dataset(paths[1], decode_times=False) as right,
    ):
        xr.testing.assert_identical(left, right)
        assert left.tas.encoding["chunksizes"] == chunks
        assert left.tas.encoding["complevel"] == 2


def test_append_encoding_keeps_its_distinct_bounds_rule():
    ds = xr.Dataset(
        coords={"latitude": ("latitude", [1.0, 2.0], {"bounds": "lat_bnds"})},
        data_vars={"lat_bnds": (("latitude", "bnds"), [[0.0, 1.5], [1.5, 3.0]])},
    )
    ds.lat_bnds.encoding.update(
        chunksizes=(1, 2), source="staging", original_shape=(2, 2)
    )
    merged = ds.copy(deep=True)
    prepare_append_encoding(merged, ds)
    assert merged.lat_bnds.encoding == {"chunksizes": (2, 2)}
    assert ds.lat_bnds.encoding["source"] == "staging"


@pytest.mark.parametrize("failed_write", [1, 2])
def test_append_write_failure_preserves_file_and_staging(
    tmp_path, monkeypatch, failed_write
):
    dataset, variable, axes = _simple_metadata()
    path = cmor.create_dataset(
        dataset, variable, axes, np.arange(4.0), path=tmp_path / "data.nc"
    )
    original = path.read_bytes()
    writer = cmor.DatasetWriter(
        dataset, variable, axes, path=path, existing="append", staging_dir=tmp_path
    )
    writer.write([4.0], time_values=[4.5], time_bounds=[[4.0, 5.0]])
    write = _netcdf.write_dataset
    calls = 0

    def interrupted(ds, dataset, output_path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_write:
            output_path.write_bytes(b"incomplete NetCDF")
            raise OSError("interrupted write")
        return write(ds, dataset, output_path, **kwargs)

    monkeypatch.setattr(_netcdf, "write_dataset", interrupted)
    with pytest.raises(OSError, match="interrupted write"):
        writer.close()
    assert path.read_bytes() == original
    assert writer._staging.path.exists()
    assert not writer._closed
    assert not list(tmp_path.glob(".data.nc.*.tmp"))
    monkeypatch.setattr(_netcdf, "write_dataset", write)
    assert writer.close() == path
    assert not writer._staging.path.exists()
    with xr.open_dataset(path, decode_times=False) as output:
        np.testing.assert_array_equal(output.tas, np.arange(5.0))
