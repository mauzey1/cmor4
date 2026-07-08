from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import xarray as xr

from .axis import Axis
from .dataset import (
    build_output_path,
    create_dataset_from_validated_data,
    write_netcdf,
)
from .datasetinfo import DatasetInfo
from .grid import Grid
from .utils.validation import (
    validate_data_chunk,
    validate_metadata,
)
from .utils.writer_helpers import find_time_axis
from .variable import Variable
from .zfactor import ZFactor


class DatasetWriter:
    """Incremental NetCDF writer for time-series datasets.

    Phase 1 supports creating a new output file from one or more writes along
    the time dimension. Data chunks are staged in a temporary Zarr store, then
    finalized through the same dataset construction and NetCDF writing path as
    :func:`cmor4.cmorize`.
    """

    def __init__(
        self,
        dataset: DatasetInfo,
        variable: Variable,
        axes: Sequence[Axis],
        path: str | Path | None = None,
        *,
        zfactors: Sequence[ZFactor] | None = None,
        grid: Grid | None = None,
        existing: Literal["error", "replace", "append"] = "error",
        encoding: Mapping[str, Any] | None = None,
        attrs: Mapping[str, Any] | None = None,
        staging_dir: str | Path | None = None,
        time_axis_name: str = "time",
        allow_time_gaps: bool = False,
        **to_netcdf_kwargs: Any,
    ) -> None:
        if existing not in {"error", "replace", "append"}:
            raise ValueError(
                "existing must be one of 'error', 'replace', or 'append'."
            )
        if existing == "append":
            raise NotImplementedError(
                "DatasetWriter append mode is planned for Phase 2."
            )

        self.path = Path(path) if path is not None else None
        self.existing = existing
        self.encoding = dict(encoding or {})
        self.attrs = dict(attrs or {})
        self.to_netcdf_kwargs = dict(to_netcdf_kwargs)
        self.allow_time_gaps = allow_time_gaps
        self._closed = False
        self._write_count = 0
        self._time_offset = 0
        self._time_values: list[np.ndarray] = []
        self._time_bounds: list[np.ndarray] = []
        self._saw_time_bounds = False

        input_axes = tuple(axes)
        input_time_index, input_time_axis = find_time_axis(input_axes, time_axis_name)
        metadata_axes = list(input_axes)
        metadata_axes[input_time_index] = _metadata_time_axis(input_time_axis)

        self._ctx = validate_metadata(
            dataset,
            variable,
            metadata_axes,
            zfactors,
            grid,
        )
        self._time_axis_index, self._time_axis = find_time_axis(
            self._ctx.axes,
            time_axis_name,
        )
        time_dims = self._ctx.axis_dims.get(self._time_axis.name, ())
        if len(time_dims) != 1 or time_dims[0] not in self._ctx.dims:
            raise ValueError(
                "DatasetWriter requires the time axis to map to exactly one "
                "data variable dimension."
            )
        self._time_dim = time_dims[0]
        self._time_dim_index = self._ctx.dims.index(self._time_dim)
        self._initial_time_values = input_time_axis.values_array()
        self._initial_time_bounds = (
            _bounds_as_pairs(
                input_time_axis.bounds_array(),
                len(self._initial_time_values),
            )
            if input_time_axis.bounds is not None
            else None
        )

        self.staging_root = Path(
            tempfile.mkdtemp(
                prefix="cmor4-datasetwriter-",
                dir=str(staging_dir) if staging_dir is not None else None,
            )
        )
        self.staging_path = self.staging_root / "staging.zarr"
        self._zarr_group = _open_zarr_group(self.staging_path)
        self._zarr_array: Any | None = None

    def write(
        self,
        data: Any,
        *,
        time_values: Any | None = None,
        time_bounds: Any | None = None,
        zfactors: Mapping[str, Any] | None = None,
    ) -> None:
        """Write one data chunk along the time dimension."""

        self._ensure_open()
        if zfactors:
            raise NotImplementedError(
                "Per-chunk zfactor writes are planned for DatasetWriter Phase 3."
            )

        data_array = np.asarray(data)
        chunk_time_values, chunk_time_bounds = self._time_chunk(
            data_array,
            time_values=time_values,
            time_bounds=time_bounds,
        )
        if self._write_count > 0 and self._saw_time_bounds != (
            chunk_time_bounds is not None
        ):
            raise ValueError(
                "time_bounds must be supplied for every write or omitted "
                "for every write."
            )
        chunk_axes = self._axes_with_time(chunk_time_values, chunk_time_bounds)
        chunk_ctx = validate_metadata(
            self._ctx.dataset,
            self._ctx.variable,
            chunk_axes,
            self._ctx.zfactors,
            self._ctx.grid,
            include_time_checks=not self.allow_time_gaps,
        )
        validated_data = validate_data_chunk(chunk_ctx, data_array)
        self._validate_chunk_shape(validated_data, len(chunk_time_values))
        self._validate_time_order(chunk_time_values, chunk_time_bounds)
        self._append_to_zarr(validated_data)

        self._time_values.append(chunk_time_values)
        if chunk_time_bounds is not None:
            self._saw_time_bounds = True
            self._time_bounds.append(chunk_time_bounds)
        self._time_offset += len(chunk_time_values)
        self._write_count += 1

    def close(
        self,
        *,
        preserve_definition: bool = False,
    ) -> tuple[xr.Dataset, Path]:
        """Finalize the staged data and write the NetCDF file."""

        self._ensure_open()
        if preserve_definition:
            raise NotImplementedError(
                "DatasetWriter preserve_definition is planned for Phase 3."
            )
        if self._zarr_array is None or self._write_count == 0:
            raise ValueError("Cannot close DatasetWriter before any data is written.")

        try:
            final_axes = self._final_axes()
            final_ctx = validate_metadata(
                self._ctx.dataset,
                self._ctx.variable,
                final_axes,
                self._ctx.zfactors,
                self._ctx.grid,
                include_time_checks=not self.allow_time_gaps,
            )
            data = _lazy_zarr_array(self._zarr_array)
            ds = create_dataset_from_validated_data(
                final_ctx,
                data,
                attrs=self.attrs,
                encoding=self.encoding,
                include_time_checks=not self.allow_time_gaps,
            )
            output_path = self.path or build_output_path(
                self._ctx.dataset,
                self._ctx.variable,
                ds,
            )
            if self.existing == "error" and output_path.exists():
                raise FileExistsError(
                    f"Output file {str(output_path)!r} already exists. "
                    "Use existing='replace' to overwrite it."
                )
            output_path = write_netcdf(
                ds,
                self._ctx.dataset,
                self._ctx.variable,
                path=output_path,
                **self.to_netcdf_kwargs,
            )
            result = xr.open_dataset(
                output_path,
                decode_times=False,
                mask_and_scale=False,
            )
        except Exception:
            raise
        else:
            self._closed = True
            shutil.rmtree(self.staging_root, ignore_errors=True)
            return result, output_path

    def __enter__(self) -> DatasetWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        if exc_type is None and not self._closed:
            self.close()
        return False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("DatasetWriter is already closed.")

    def _time_chunk(
        self,
        data_array: np.ndarray,
        *,
        time_values: Any | None,
        time_bounds: Any | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if data_array.ndim <= self._time_dim_index:
            raise ValueError(
                f"Data for {self._ctx.variable.names()[0]!r} does not include "
                f"the expected time dimension {self._time_dim!r}."
            )
        chunk_len = int(data_array.shape[self._time_dim_index])
        if time_values is None:
            if self._initial_time_values.size == 0:
                raise ValueError(
                    "time_values must be provided when the initial time axis "
                    "does not contain complete values."
                )
            end = self._time_offset + chunk_len
            if end > self._initial_time_values.shape[0]:
                raise ValueError(
                    "The number of written time records exceeds the initial "
                    "time axis length."
                )
            values = np.asarray(self._initial_time_values[self._time_offset : end])
            if time_bounds is None and self._initial_time_bounds is not None:
                bounds = np.asarray(self._initial_time_bounds[self._time_offset : end])
            else:
                bounds = _coerce_time_bounds(time_bounds, chunk_len)
        else:
            values = _coerce_time_values(time_values)
            if values.shape[0] != chunk_len:
                raise ValueError(
                    f"time_values length {values.shape[0]} does not match "
                    f"the data chunk time length {chunk_len}."
                )
            bounds = _coerce_time_bounds(time_bounds, chunk_len)
        return values, bounds

    def _axes_with_time(
        self,
        time_values: np.ndarray,
        time_bounds: np.ndarray | None,
    ) -> tuple[Axis, ...]:
        axes = list(self._ctx.axes)
        updates: dict[str, Any] = {
            "values": time_values.tolist(),
        }
        if time_bounds is not None:
            updates["bounds"] = time_bounds.tolist()
        elif self._time_axis.bounds is not None:
            updates["bounds"] = None
        axes[self._time_axis_index] = self._time_axis.updated(**updates)
        return tuple(axes)

    def _validate_chunk_shape(
        self,
        data: np.ndarray,
        chunk_time_len: int,
    ) -> None:
        if data.ndim != len(self._ctx.dims):
            expected = " x ".join(self._ctx.dims) if self._ctx.dims else "scalar"
            raise ValueError(
                f"Data for {self._ctx.variable.names()[0]!r} has {data.ndim} "
                f"dimensions, but variable dimensions resolve to {expected!r}."
            )
        expected_shape = []
        for index, dim in enumerate(self._ctx.dims):
            if index == self._time_dim_index:
                expected_shape.append(chunk_time_len)
                continue
            expected_shape.append(self._dimension_size(dim))
        actual = tuple(int(size) for size in data.shape)
        expected = tuple(expected_shape)
        if actual != expected:
            raise ValueError(
                f"Data chunk shape {actual!r} does not match expected shape "
                f"{expected!r} for dimensions {self._ctx.dims!r}."
            )

    def _dimension_size(self, dim: str) -> int:
        for axis in self._ctx.axes:
            dims = self._ctx.axis_dims.get(axis.name, ())
            if dim not in dims:
                continue
            values = axis.values_array()
            if values.ndim == 0:
                return 1
            if len(dims) == 1:
                return int(values.shape[0])
            dim_index = dims.index(dim)
            return int(values.shape[dim_index])
        raise ValueError(f"Cannot determine expected size for dimension {dim!r}.")

    def _validate_time_order(
        self,
        time_values: np.ndarray,
        time_bounds: np.ndarray | None,
    ) -> None:
        if not self._time_values:
            return
        previous = self._time_values[-1]
        if previous.size and time_values.size and time_values[0] <= previous[-1]:
            raise ValueError("Time values must be strictly monotonic across writes.")
        if self.allow_time_gaps or time_bounds is None or not self._time_bounds:
            return
        previous_bounds = self._time_bounds[-1]
        if previous_bounds.size == 0:
            return
        previous_end = previous_bounds[-1, 1]
        next_start = time_bounds[0, 0]
        if np.issubdtype(np.asarray(previous_end).dtype, np.number) and not np.isclose(
            previous_end,
            next_start,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Time bounds must be contiguous across writes unless "
                "allow_time_gaps=True."
            )

    def _append_to_zarr(self, data: np.ndarray) -> None:
        if self._zarr_array is None:
            shape = list(data.shape)
            shape[self._time_dim_index] = 0
            chunks = tuple(max(1, int(size)) for size in data.shape)
            self._zarr_array = _create_zarr_array(
                self._zarr_group,
                self._ctx.variable.names()[0],
                shape=tuple(shape),
                chunks=chunks,
                dtype=data.dtype,
            )
        new_shape = list(self._zarr_array.shape)
        start = int(new_shape[self._time_dim_index])
        stop = start + int(data.shape[self._time_dim_index])
        new_shape[self._time_dim_index] = stop
        self._zarr_array.resize(tuple(new_shape))
        index = [slice(None)] * data.ndim
        index[self._time_dim_index] = slice(start, stop)
        self._zarr_array[tuple(index)] = data

    def _final_axes(self) -> tuple[Axis, ...]:
        axes = list(self._ctx.axes)
        values = np.concatenate(self._time_values, axis=0)
        updates: dict[str, Any] = {"values": values.tolist()}
        if self._saw_time_bounds:
            if len(self._time_bounds) != len(self._time_values):
                raise ValueError(
                    "time_bounds must be supplied for every write or omitted "
                    "for every write."
                )
            updates["bounds"] = np.concatenate(self._time_bounds, axis=0).tolist()
        elif self._time_axis.bounds is not None:
            updates["bounds"] = None
        axes[self._time_axis_index] = self._time_axis.updated(**updates)
        return tuple(axes)


def _metadata_time_axis(axis: Axis) -> Axis:
    values = axis.values_array()
    value = values.reshape(-1)[0] if values.size else 0.0
    value_item = value.item() if hasattr(value, "item") else value
    updates: dict[str, Any] = {"values": [value_item]}
    if axis.bounds is not None:
        bounds = _bounds_as_pairs(
            axis.bounds_array(),
            len(values) if values.size else 1,
        )
        updates["bounds"] = bounds[:1].tolist()
    else:
        updates["bounds"] = [[float(value) - 0.5, float(value) + 0.5]]
    return axis.updated(**updates)


def _coerce_time_values(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim != 1:
        raise ValueError("time_values must be a one-dimensional array.")
    return array


def _coerce_time_bounds(bounds: Any | None, time_len: int) -> np.ndarray | None:
    if bounds is None:
        return None
    return _bounds_as_pairs(np.asarray(bounds), time_len)


def _bounds_as_pairs(bounds: np.ndarray, time_len: int) -> np.ndarray:
    if bounds.ndim == 1 and bounds.size == time_len + 1:
        return np.stack((bounds[:-1], bounds[1:]), axis=-1)
    if bounds.ndim == 1 and time_len == 1 and bounds.size == 2:
        return bounds.reshape(1, 2)
    if bounds.shape[:1] == (time_len,) and bounds.shape[-1] >= 2:
        return bounds
    raise ValueError(
        f"time_bounds shape {bounds.shape!r} does not match time_values length "
        f"{time_len}."
    )


def _open_zarr_group(path: Path) -> Any:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DatasetWriter requires the optional runtime dependency 'zarr'. "
            "Install cmor4 with current project dependencies before using it."
        ) from exc
    return zarr.open_group(str(path), mode="w")


def _lazy_zarr_array(array: Any) -> Any:
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


def _create_zarr_array(
    group: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype[Any],
) -> Any:
    if hasattr(group, "create_array"):
        return group.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
    return group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)
