from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cmor4 import Axis, DatasetInfo, Grid, Variable, ZFactor
from cmor4.utils.table_models import (
    CoordinateTableDocument,
    FormulaTableDocument,
    GridTableDocument,
    VariableTableDocument,
)


def test_coordinate_document_normalizes_runtime_values_and_dimensions() -> None:
    document = CoordinateTableDocument.model_validate({
        "axis_entry": {
            "levels": {
                "dimensions": "x y",
                "requested": ["1", "2.5"],
                "requested_bounds": "0 1.5 1.5 3",
            }
        }
    })

    entry = document.axis_entry["levels"]
    assert entry.name == "levels"
    assert entry.runtime_dimensions == ("y", "x")
    assert entry.runtime_values == [1, 2.5]
    assert entry.runtime_bounds == [[0, 1.5], [1.5, 3]]


def test_all_table_documents_reject_non_mapping_rows() -> None:
    documents_and_sections = (
        (CoordinateTableDocument, "axis_entry"),
        (FormulaTableDocument, "formula_entry"),
        (GridTableDocument, "mapping_entry"),
        (VariableTableDocument, "variable_entry"),
    )

    for document, section in documents_and_sections:
        with pytest.raises(ValidationError):
            document.model_validate({section: {"bad": ["not", "a", "row"]}})


def test_variable_document_attaches_source_provenance(tmp_path) -> None:
    path = tmp_path / "variables.json"
    path.write_text(json.dumps({
        "Header": {"table_id": "Table Amon"},
        "variable_entry": {
            "tas": {"out_name": "tas", "dimensions": "lon lat time", "units": "K"}
        },
    }))

    document = VariableTableDocument.model_validate_json(path.read_text())
    entry = document.resolved_entries(path)["tas"]

    assert entry.table_id == "Amon"
    assert entry.table_file == path
    assert entry.runtime_dimensions == ("time", "lat", "lon")


@pytest.mark.parametrize(
    ("model", "values"),
    (
        (Axis, {"name": "x"}),
        (Variable, {"name": "tas"}),
        (ZFactor, {"name": "p0"}),
        (Grid, {}),
    ),
)
def test_runtime_components_reject_unknown_constructor_fields(model, values) -> None:
    with pytest.raises(ValidationError, match="unexpected_field"):
        model.model_validate({**values, "unexpected_field": 1})


def test_dataset_info_has_an_explicit_serialization_boundary() -> None:
    info = DatasetInfo.from_prepared({
        "activity_id": "CMIP",
        "custom_attribute": "value",
    })

    assert info.activity_id == "CMIP"
    assert info.to_dict()["custom_attribute"] == "value"
    assert not hasattr(info, "get")
    with pytest.raises(TypeError):
        _ = info["activity_id"]  # type: ignore[index]
