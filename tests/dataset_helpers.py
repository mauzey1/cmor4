from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import xarray as xr

import cmor4


def load_dataset(path: str | Path) -> xr.Dataset:
    with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as opened:
        loaded = opened.load()
    for variable in loaded.variables.values():
        coordinates = variable.encoding.get("coordinates")
        if coordinates is not None and "coordinates" not in variable.attrs:
            variable.attrs["coordinates"] = coordinates
    return loaded


def open_created_dataset(*args: Any, **kwargs: Any) -> xr.Dataset:
    if "path" in kwargs and kwargs["path"] is not None:
        return load_dataset(cmor4.create_dataset(*args, **kwargs))
    with tempfile.TemporaryDirectory() as tmp_dir:
        kwargs["path"] = Path(tmp_dir) / "created.nc"
        return load_dataset(cmor4.create_dataset(*args, **kwargs))


def cmorize_and_open(*args: Any, **kwargs: Any) -> tuple[xr.Dataset, Path]:
    path = cmor4.cmorize(*args, **kwargs)
    return load_dataset(path), path
