from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


@dataclass
class ZarrStagingStore:
    """Manage temporary Zarr arrays used by DatasetWriter."""

    root: Path
    path: Path
    _group: Any
    _arrays: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, staging_dir: str | Path | None = None) -> ZarrStagingStore:
        root = Path(
            tempfile.mkdtemp(
                prefix="cmor4-datasetwriter-",
                dir=str(staging_dir) if staging_dir is not None else None,
            )
        )
        path = root / "staging.zarr"
        try:
            import zarr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "DatasetWriter requires the optional runtime dependency 'zarr'. "
                "Install cmor4 with current project dependencies before using it."
            ) from exc
        return cls(root=root, path=path, _group=zarr.open_group(str(path), mode="w"))

    def append(self, name: str, data: np.ndarray, time_dim_index: int) -> Any:
        array = self._arrays.get(name)
        if array is None:
            shape = list(data.shape)
            shape[time_dim_index] = 0
            chunks = tuple(max(1, int(size)) for size in data.shape)
            if hasattr(self._group, "create_array"):
                array = self._group.create_array(
                    name,
                    shape=tuple(shape),
                    chunks=chunks,
                    dtype=data.dtype,
                )
            else:
                array = self._group.create_dataset(
                    name,
                    shape=tuple(shape),
                    chunks=chunks,
                    dtype=data.dtype,
                )

        new_shape = list(array.shape)
        start = int(new_shape[time_dim_index])
        stop = start + int(data.shape[time_dim_index])
        new_shape[time_dim_index] = stop
        array.resize(tuple(new_shape))
        index = [slice(None)] * data.ndim
        index[time_dim_index] = slice(start, stop)
        array[tuple(index)] = data
        self._arrays[name] = array
        return array

    def array(self, name: str) -> Any | None:
        return self._arrays.get(name)

    def has_array(self, name: str) -> bool:
        return name in self._arrays

    def lazy_array(self, name: str) -> Any:
        array = self.array(name)
        if array is None:
            raise ValueError(f"No staged Zarr array exists for {name!r}.")
        try:
            import dask.array as da
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "DatasetWriter requires the runtime dependency 'dask[array]' "
                "to finalize staged Zarr data without loading it into memory."
            ) from exc
        data = da.from_zarr(array)
        byteorder = getattr(data.dtype, "byteorder", None)
        if byteorder not in (None, "=", "|"):
            dtype = data.dtype.newbyteorder("=")
            data = data.map_blocks(
                lambda block: block.astype(dtype, copy=False),
                meta=np.array([], dtype=dtype),
            )
        return data

    def reset(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        try:
            import zarr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "DatasetWriter requires the optional runtime dependency 'zarr'. "
                "Install cmor4 with current project dependencies before using it."
            ) from exc
        self._group = zarr.open_group(str(self.path), mode="w")
        self._arrays.clear()

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self._arrays.clear()
