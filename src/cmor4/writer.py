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

    ``DatasetWriter`` enables writing climate model output incrementally as data
    becomes available, rather than requiring the complete time series in memory.
    Data chunks are staged in a temporary Zarr store and streamed to NetCDF
    using dask lazy loading, keeping memory usage bounded regardless of total
    dataset size.

    **When to use DatasetWriter:**

    - Writing datasets one time slice at a time as a model runs
    - Processing datasets larger than available RAM
    - Splitting long time series across multiple output files
    - Reusing the same metadata definition for multiple files

    **When to use cmorize() instead:**

    - The complete dataset fits comfortably in memory
    - You already have all time slices in a single array
    - You're writing a single file with all data available

    **Key Features:**

    - **Memory-bounded operation**: Uses dask lazy loading to stream data
      chunk-by-chunk without loading the full array into memory
    - **Validation**: Applies the same CMOR table validation as
      :func:`~cmor4.cmorize`, ensuring output compliance
    - **Automatic cleanup**: Staging directory is automatically removed on
      success, preserved on error for debugging
    - **Flexible time specification**: Time values can be provided incrementally
      with each write, or pre-specified in the time axis

    **Usage Example:**

    .. code-block:: python

        import cmor4
        import numpy as np

        project = cmor4.ProjectTables.from_directory(...)
        dataset = project.dataset_info({...})
        variable = project.variable("tas")

        # Empty time axis - values provided with each write
        axes = [
            project.axis("time", units="days since 2000-01-01"),
            project.axis("latitude", values=lats),
            project.axis("longitude", values=lons),
        ]

        # Write data incrementally
        with cmor4.DatasetWriter(dataset, variable, axes) as writer:
            for year in range(1850, 2015):
                time_vals, data = load_year(year)  # Your function
                bounds = compute_bounds(time_vals)  # Your function
                writer.write(data, time_values=time_vals, time_bounds=bounds)

        # File is automatically written and closed

    **Memory Characteristics:**

    - **Per-write memory**: O(chunk size) - only the current chunk is in memory
    - **Finalization memory**: ~20-70 MB overhead for metadata and coordinates
    - **Disk staging**: Temporary Zarr store is ~2× final NetCDF size
    - **Total memory**: Independent of total dataset size

    **Phase 1 Limitations:**

    - ``append`` mode (extending existing files) is not yet implemented
    - Per-chunk zfactor writes are not yet implemented
    - Only time dimension supports incremental writes; spatial axes must have
      complete values at initialization

    Parameters
    ----------
    dataset : DatasetInfo
        Dataset-level metadata including institution, experiment, variant, etc.
        Created via :meth:`ProjectTables.dataset_info`.
    variable : Variable
        Variable metadata including dimensions, units, and CF attributes.
        Created via :meth:`ProjectTables.variable`.
    axes : Sequence[Axis]
        Coordinate axes. The time axis may have empty or incomplete values;
        other axes must have complete coordinate values at initialization.
        Created via :meth:`ProjectTables.axis`.
    path : str or Path, optional
        Output file path. If ``None``, path is generated from dataset and
        variable metadata according to project conventions (e.g., CMIP7 DRS).
    zfactors : Sequence[ZFactor], optional
        Formula term variables for dimensionless vertical coordinates.
        Per-chunk zfactor writes are not yet supported in Phase 1.
    grid : Grid, optional
        Grid mapping and spatial coordinate metadata for curvilinear grids.
    existing : {"error", "replace", "append"}, default "error"
        Behavior when output file already exists:

        - ``"error"``: Raise ``FileExistsError`` if file exists
        - ``"replace"``: Overwrite existing file
        - ``"append"``: Extend existing file with new time records (Phase 2)
    encoding : Mapping[str, Any], optional
        NetCDF encoding parameters (chunksizes, compression, fill values).
        If not specified, CMIP7 auto-chunking rules are applied.
    attrs : Mapping[str, Any], optional
        Additional global attributes to add to the dataset.
    staging_dir : str or Path, optional
        Directory for temporary Zarr store. If ``None``, uses system temp
        directory. Useful for directing staging to high-performance storage.
    time_axis_name : str, default "time"
        Name of the time axis. Used to identify which axis is the write
        dimension. Normally detected automatically from ``axis="T"`` or
        name starting with "time".
    **to_netcdf_kwargs
        Additional keyword arguments passed to :meth:`xarray.Dataset.to_netcdf`.

    Attributes
    ----------
    staging_root : Path
        Root directory of the temporary Zarr staging store.
    staging_path : Path
        Path to the Zarr store (``staging_root / "staging.zarr"``).

    Raises
    ------
    ValueError
        If ``existing`` is not one of "error", "replace", or "append".
    NotImplementedError
        If ``existing="append"`` (Phase 2 feature).
    AxisValidationError
        If axes fail validation (e.g., missing required bounds, invalid values).
    VariableValidationError
        If variable metadata fails validation.

    See Also
    --------
    cmorize : Single-pass dataset creation for data that fits in memory.
    create_dataset : Lower-level function for constructing xarray datasets.

    Notes
    -----
    ``DatasetWriter`` uses a Zarr store for staging and dask lazy arrays for
    finalization, enabling true memory-bounded operation. The implementation:

    1. **Initialization**: Validates metadata and creates an empty Zarr store
    2. **Write calls**: Append data chunks to the Zarr store on disk
    3. **Finalization**: Stream from Zarr to NetCDF using dask, never loading
       the full array into memory

    The staging directory is automatically cleaned up on successful close,
    but preserved on error to aid debugging. You can inspect the Zarr store at
    ``writer.staging_path`` if an error occurs.

    Examples
    --------
    **Basic incremental write:**

    >>> with cmor4.DatasetWriter(dataset, variable, axes) as writer:
    ...     for month in range(12):
    ...         data = load_month(month)  # Your function
    ...         time_val = [month * 30.0 + 15.0]
    ...         time_bnd = [[month * 30.0, (month + 1) * 30.0]]
    ...         writer.write(data, time_values=time_val, time_bounds=time_bnd)

    **Pre-specified time values:**

    >>> # Time axis with complete values
    >>> time = project.axis("time", values=time_vals, bounds=time_bnds)
    >>> axes = [time, lat, lon]
    >>>
    >>> with cmor4.DatasetWriter(dataset, variable, axes) as writer:
    ...     # Write uses pre-specified time values
    ...     writer.write(data[:6])    # First 6 time steps
    ...     writer.write(data[6:])    # Remaining time steps

    **Explicit path and encoding:**

    >>> encoding = {
    ...     "tas": {
    ...         "chunksizes": (1, 180, 360),
    ...         "zlib": True,
    ...         "complevel": 4,
    ...     }
    ... }
    >>> writer = cmor4.DatasetWriter(
    ...     dataset,
    ...     variable,
    ...     axes,
    ...     path="/path/to/output.nc",
    ...     encoding=encoding,
    ... )
    >>> writer.write(data, time_values=[15.0], time_bounds=[[0.0, 30.0]])
    >>> ds, path = writer.close()

    **Writing multiple file segments:**

    >>> writer = cmor4.DatasetWriter(
    ...     dataset, variable, axes, path="segment-1.nc"
    ... )
    >>> writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])
    >>> ds1, path1 = writer.close(preserve_definition=True)
    >>> writer.path = Path("segment-2.nc")
    >>> writer.write(data2, time_values=[115.0], time_bounds=[[100.0, 130.0]])
    >>> ds2, path2 = writer.close()
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
        **to_netcdf_kwargs: Any,
    ) -> None:
        if existing not in {"error", "replace", "append"}:
            raise ValueError("existing must be one of 'error', 'replace', or 'append'.")
        if existing == "append":
            raise NotImplementedError(
                "DatasetWriter append mode is planned for Phase 2."
            )

        self.path = Path(path) if path is not None else None
        self.existing = existing
        self.encoding = dict(encoding or {})
        self.attrs = dict(attrs or {})
        self.to_netcdf_kwargs = dict(to_netcdf_kwargs)
        self._closed = False
        self._write_count = 0
        self._time_offset = 0
        self._time_values: list[np.ndarray] = []
        self._time_bounds: list[np.ndarray] = []
        self._saw_time_bounds = False
        self._first_write_after_preserve = False
        self._last_closed_time_value: Any | None = None

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
        """Write one data chunk along the time dimension.

        Appends a chunk of data to the staging Zarr store. The chunk must match
        the expected non-time dimensions and contain valid data values. Time
        values must be strictly monotonically increasing across writes.

        Parameters
        ----------
        data : array-like
            Data values for this chunk. Shape must be consistent with variable
            dimensions, with the time dimension matching ``len(time_values)``.
            Accepts numpy arrays, lists, or other array-like objects.
        time_values : array-like, optional
            Time coordinate values for this chunk. If ``None``, uses values
            from the pre-initialized time axis (starting at the current write
            position). Must be strictly monotonically increasing relative to
            previous writes.
        time_bounds : array-like, optional
            Time bounds for this chunk. Can be provided as:

            - Shape ``(N, 2)``: Pairs of ``[lower, upper]`` bounds
            - Shape ``(N+1,)``: Edges (converted to pairs automatically)
            - Shape ``(N, >=2)``: Climatology or multi-bound format

            If ``None`` and bounds exist in the pre-initialized time axis, those
            bounds are used. If bounds are provided in any write for a given
            output file, they must be provided in every write for that file.
        zfactors : Mapping[str, array-like], optional
            Per-chunk zfactor data. Not yet implemented in Phase 1.

        Raises
        ------
        ValueError
            If the writer is already closed.
        ValueError
            If data chunk shape doesn't match expected dimensions.
        ValueError
            If time_values are not monotonically increasing.
        ValueError
            If time_bounds are not contiguous within one output file.
        ValueError
            If time_values length doesn't match data time dimension.
        ValueError
            If time_bounds shape doesn't match time_values length.
        ValueError
            If time_values are omitted but time axis has no pre-specified values.
        AxisValidationError
            If time values fail validation (e.g., units mismatch).
        NotImplementedError
            If zfactors are provided (Phase 3 feature).

        Examples
        --------
        **Write with explicit time values:**

        >>> writer.write(
        ...     data,
        ...     time_values=[15.0, 45.0],
        ...     time_bounds=[[0.0, 30.0], [30.0, 60.0]],
        ... )

        **Write using pre-specified time axis:**

        >>> # Time axis initialized with complete values
        >>> time = project.axis("time", values=[15.0, 45.0], ...)
        >>> writer = cmor4.DatasetWriter(dataset, variable, [time, ...])
        >>> writer.write(data)  # Uses pre-specified time values

        **Write single time slice:**

        >>> writer.write(
        ...     data[0:1],  # Shape (1, lat, lon)
        ...     time_values=[15.0],
        ...     time_bounds=[[0.0, 30.0]],
        ... )

        **Write with edge-format bounds:**

        >>> # Bounds as edges (N+1 values for N times)
        >>> writer.write(
        ...     data,
        ...     time_values=[15.0, 45.0],
        ...     time_bounds=[0.0, 30.0, 60.0],  # Converted to pairs
        ... )

        Notes
        -----
        Time values must be strictly monotonically increasing across all writes.
        For example, if the first write has ``time_values=[15.0]``, the second
        write must have ``time_values[0] > 15.0``.

        Time bounds must be contiguous (edge-to-edge) across writes within a
        single output file. The first write after
        ``close(preserve_definition=True)`` starts a new file segment, so it
        is not checked for contiguity against the previous file.

        The data chunk is validated for NaN values, valid_min/valid_max ranges,
        and other variable-specific constraints before being written to the
        Zarr store.
        """

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
        self._first_write_after_preserve = False

    def close(
        self,
        *,
        preserve_definition: bool = False,
    ) -> tuple[xr.Dataset, Path]:
        """Finalize the staged data and write the NetCDF file.

        Streams data from the Zarr staging store to the final NetCDF file using
        dask lazy loading. The full dataset is never loaded into memory;
        instead, data is read and written chunk-by-chunk.

        Parameters
        ----------
        preserve_definition : bool, default False
            If ``True``, keep metadata definition for writing another output
            file with the same variable/axis definitions. Staged data and
            per-file time state are cleared after the current file is written.

        Returns
        -------
        dataset : xarray.Dataset
            The written dataset, opened from the output file. This is a loaded
            dataset (not lazy) suitable for inspection or further processing.
        path : Path
            Absolute path to the written NetCDF file.

        Raises
        ------
        ValueError
            If the writer is already closed.
        ValueError
            If no data has been written (write() was never called).
        ValueError
            If time_bounds were provided inconsistently across writes.
        FileExistsError
            If output file exists and ``existing="error"``.
        AxisValidationError
            If final time axis fails validation.
        Notes
        -----
        The close operation:

        1. Validates the complete time axis with all accumulated time values
        2. Creates a lazy dask array from the Zarr store (no data loaded)
        3. Constructs an xarray Dataset with proper metadata and attributes
        4. Streams data chunk-by-chunk to the NetCDF file via xarray
        5. Cleans up the temporary Zarr staging directory
        6. Opens and loads the final NetCDF file for return

        Memory usage during close is bounded by chunk size plus ~20-70 MB
        for metadata and coordinate arrays.

        Examples
        --------
        **Basic usage:**

        >>> writer = cmor4.DatasetWriter(dataset, variable, axes)
        >>> writer.write(data1, time_values=[15.0], time_bounds=[[0.0, 30.0]])
        >>> writer.write(data2, time_values=[45.0], time_bounds=[[30.0, 60.0]])
        >>> ds, path = writer.close()
        >>> print(f"Wrote {path}")
        >>> print(f"Time range: {ds.time.values}")

        **With context manager (automatic close):**

        >>> with cmor4.DatasetWriter(dataset, variable, axes) as writer:
        ...     writer.write(data, time_values=times, time_bounds=bounds)
        ...     # close() called automatically on context exit

        **Inspect returned dataset:**

        >>> ds, path = writer.close()
        >>> print(ds.data_vars)  # Show variables
        >>> print(ds.attrs)      # Show global attributes
        >>> ds.close()           # Close when done inspecting

        See Also
        --------
        write : Write a data chunk to the staging store.
        __enter__ : Context manager entry (returns self).
        __exit__ : Context manager exit (calls close automatically).
        """

        self._ensure_open()
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
            )
            data = _lazy_zarr_array(self._zarr_array)
            ds = create_dataset_from_validated_data(
                final_ctx,
                data,
                attrs=self.attrs,
                encoding=self.encoding,
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
            if preserve_definition:
                self._prepare_next_segment()
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
            if self._write_count > 0:
                self.close()
            elif self._first_write_after_preserve:
                self._closed = True
                shutil.rmtree(self.staging_root, ignore_errors=True)
            else:
                self.close()
        return False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("DatasetWriter is already closed.")

    def _prepare_next_segment(self) -> None:
        if self._time_values and self._time_values[-1].size:
            value = self._time_values[-1][-1]
            self._last_closed_time_value = (
                value.item() if hasattr(value, "item") else value
            )
        else:
            self._last_closed_time_value = None
        self._time_values = []
        self._time_bounds = []
        self._saw_time_bounds = False
        self._time_offset = 0
        self._write_count = 0
        self._first_write_after_preserve = True
        shutil.rmtree(self.staging_path, ignore_errors=True)
        self._zarr_group = _open_zarr_group(self.staging_path)
        self._zarr_array = None

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
            if (
                self._first_write_after_preserve
                and self._last_closed_time_value is not None
                and time_values.size
                and time_values[0] <= self._last_closed_time_value
            ):
                raise ValueError("Time values must be strictly monotonic across files.")
            return
        previous = self._time_values[-1]
        if previous.size and time_values.size and time_values[0] <= previous[-1]:
            raise ValueError("Time values must be strictly monotonic across writes.")
        if time_bounds is None or not self._time_bounds:
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
                "Time bounds must be contiguous across writes within a single "
                "output file."
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
