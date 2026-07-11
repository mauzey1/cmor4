from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
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
    named_dimensions,
    validate_data_chunk,
    validate_metadata,
    validate_variable_values,
)
from .utils.writer_helpers import find_time_axis
from .utils.zarr_staging import ZarrStagingStore
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

    **Limitations:**

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
        Formula term variables for dimensionless vertical coordinates. Static
        zfactors can include complete values here. Time-varying zfactors may
        omit values and provide them per chunk via :meth:`write`.
    grid : Grid, optional
        Grid mapping and spatial coordinate metadata for curvilinear grids.
    existing : {"error", "replace", "append"}, default "error"
        Behavior when output file already exists:

        - ``"error"``: Raise ``FileExistsError`` if file exists
        - ``"replace"``: Overwrite existing file
        - ``"append"``: Extend existing file with new time records
    encoding : Mapping[str, Any], optional
        NetCDF encoding parameters (chunksizes, compression, fill values).
        If not specified, CMIP7 auto-chunking rules are applied.
    attrs : Mapping[str, Any], optional
        Additional global attributes to add to the dataset.
    staging_dir : str or Path, optional
        Directory for temporary Zarr store. If ``None``, uses system temp
        directory. Useful for directing staging to high-performance storage.
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
        **to_netcdf_kwargs: Any,
    ) -> None:
        if existing not in {"error", "replace", "append"}:
            raise ValueError("existing must be one of 'error', 'replace', or 'append'.")

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
        input_time_index, input_time_axis = find_time_axis(input_axes)
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

        self._staging = ZarrStagingStore.create(staging_dir)
        self.staging_root = self._staging.root
        self.staging_path = self._staging.path

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
            Per-chunk zfactor data. Keys may be zfactor names or output names;
            values must match the zfactor dimensions for this write chunk.

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
        ValueError
            If a per-chunk zfactor is unknown, missing for a chunk that requires
            it, or has a shape that does not match resolved zfactor dimensions.

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
        zfactor_chunks = self._validate_zfactor_chunks(
            zfactors,
            chunk_ctx,
            len(chunk_time_values),
        )
        self._validate_time_order(chunk_time_values, chunk_time_bounds)
        self._append_to_zarr(validated_data)
        self._append_zfactors_to_zarr(zfactor_chunks)

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
        var_name = self._ctx.variable.names()[0]
        if self._staging.array(var_name) is None or self._write_count == 0:
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
            final_ctx = replace(final_ctx, zfactors=self._final_zfactors(final_ctx))
            data = self._staging.lazy_array(var_name)
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
            if self.existing == "append":
                output_path = self._append_to_existing(output_path, ds)
            elif self.existing == "error" and output_path.exists():
                raise FileExistsError(
                    f"Output file {str(output_path)!r} already exists. "
                    "Use existing='replace' to overwrite it."
                )
            else:
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
                self._staging.cleanup()
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
                self._staging.cleanup()
            else:
                self.close()
        return False

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("DatasetWriter is already closed.")

    @property
    def _zarr_array(self) -> Any | None:
        return self._staging.array(self._ctx.variable.names()[0])

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
        self._staging.reset()

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
        self._staging.append(
            self._ctx.variable.names()[0],
            data,
            self._time_dim_index,
        )

    def _validate_zfactor_chunks(
        self,
        zfactors: Mapping[str, Any] | None,
        chunk_ctx: Any,
        chunk_time_len: int,
    ) -> dict[str, np.ndarray]:
        zfactor_values = {str(name): value for name, value in (zfactors or {}).items()}
        lookup = self._zfactor_lookup()
        unknown = sorted(
            str(name) for name in zfactor_values if str(name) not in lookup
        )
        if unknown:
            raise ValueError(f"Unknown zfactor chunk(s): {unknown!r}.")

        chunks: dict[str, np.ndarray] = {}
        for zfactor in self._ctx.zfactors:
            out_name = str(zfactor.out_name or zfactor.name)
            dims = self._zfactor_dims(zfactor)
            if self._time_dim not in dims:
                if zfactor.name in zfactor_values or out_name in zfactor_values:
                    raise ValueError(
                        f"Zfactor {out_name!r} does not include time and "
                        "cannot be supplied per write."
                    )
                continue

            value = _pop_zfactor_value(zfactor_values, zfactor)
            must_supply = self._staging.has_array(out_name) or _empty_zfactor(zfactor)
            if value is None:
                if must_supply:
                    raise ValueError(
                        f"Time-varying zfactor {out_name!r} must be supplied "
                        "for every write."
                    )
                continue
            if self._write_count > 0 and not self._staging.has_array(out_name):
                raise ValueError(
                    f"Time-varying zfactor {out_name!r} cannot start being "
                    "supplied after earlier writes omitted it."
                )

            data = np.asarray(value)
            self._validate_zfactor_chunk_shape(
                out_name,
                data,
                dims,
                chunk_time_len,
            )
            validate_variable_values(
                zfactor,
                chunk_ctx.axes,
                data,
                dims,
                chunk_ctx.axis_dims,
                name=out_name,
                table_id=str(zfactor.table_entry or "formula_terms"),
            )
            chunks[out_name] = data

        if zfactor_values:
            unknown = sorted(str(name) for name in zfactor_values)
            raise ValueError(f"Unknown zfactor chunk(s): {unknown!r}.")
        return chunks

    def _validate_zfactor_chunk_shape(
        self,
        name: str,
        data: np.ndarray,
        dims: tuple[str, ...],
        chunk_time_len: int,
    ) -> None:
        expected_shape = []
        for dim in dims:
            if dim == self._time_dim:
                expected_shape.append(chunk_time_len)
            else:
                expected_shape.append(self._dimension_size(dim))
        expected = tuple(expected_shape)
        actual = tuple(int(size) for size in data.shape)
        if actual != expected:
            raise ValueError(
                f"Zfactor {name!r} chunk shape {actual!r} does not match "
                f"expected shape {expected!r} for dimensions {dims!r}."
            )

    def _append_zfactors_to_zarr(self, zfactors: Mapping[str, np.ndarray]) -> None:
        for name, data in zfactors.items():
            zfactor = self._zfactor_by_output_name(name)
            dims = self._zfactor_dims(zfactor)
            time_dim_index = dims.index(self._time_dim)
            self._staging.append(
                name,
                data,
                time_dim_index,
            )

    def _final_zfactors(self, ctx: Any) -> tuple[ZFactor, ...]:
        if not any(
            self._staging.has_array(str(zfactor.out_name or zfactor.name))
            for zfactor in ctx.zfactors
        ):
            return ctx.zfactors

        final_zfactors: list[ZFactor] = []
        final_time_len = sum(int(values.shape[0]) for values in self._time_values)
        for zfactor in ctx.zfactors:
            out_name = str(zfactor.out_name or zfactor.name)
            array = self._staging.array(out_name)
            if array is None:
                final_zfactors.append(zfactor)
                continue
            dims = self._zfactor_dims(zfactor)
            if int(array.shape[dims.index(self._time_dim)]) != final_time_len:
                raise ValueError(
                    f"Time-varying zfactor {out_name!r} was not supplied "
                    "for every write."
                )
            final_zfactors.append(
                zfactor.updated(values=self._staging.lazy_array(out_name))
            )
        return tuple(final_zfactors)

    def _zfactor_lookup(self) -> dict[str, ZFactor]:
        lookup: dict[str, ZFactor] = {}
        for zfactor in self._ctx.zfactors:
            lookup[str(zfactor.name)] = zfactor
            lookup[str(zfactor.out_name or zfactor.name)] = zfactor
        return lookup

    def _zfactor_by_output_name(self, out_name: str) -> ZFactor:
        for zfactor in self._ctx.zfactors:
            if str(zfactor.out_name or zfactor.name) == out_name:
                return zfactor
        raise ValueError(f"Unknown zfactor {out_name!r}.")

    def _zfactor_dims(self, zfactor: ZFactor) -> tuple[str, ...]:
        return named_dimensions(zfactor.dimensions or (), self._ctx.axis_dims)

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

    def _append_to_existing(self, output_path: Path, new_ds: xr.Dataset) -> Path:
        if not output_path.exists():
            raise FileNotFoundError(
                f"Cannot append because output file {str(output_path)!r} "
                "does not exist."
            )

        temp_path: Path | None = None
        check_path: Path | None = None
        existing_ds = xr.open_dataset(
            output_path,
            decode_times=False,
            mask_and_scale=False,
        )
        existing_check_ds = xr.open_dataset(
            output_path,
            decode_times=False,
            mask_and_scale=False,
            decode_coords=False,
        )
        new_check_ds: xr.Dataset | None = None
        try:
            check_path = _temporary_netcdf_path(output_path)
            write_netcdf(
                new_ds,
                self._ctx.dataset,
                self._ctx.variable,
                path=check_path,
                **self.to_netcdf_kwargs,
            )
            _ensure_nonempty_file(check_path, "Append compatibility check")
            new_check_ds = xr.open_dataset(
                check_path,
                decode_times=False,
                mask_and_scale=False,
                decode_coords=False,
            )
            self._validate_append_compatible(existing_check_ds, new_check_ds)
            merged_ds = xr.concat(
                [existing_ds, new_ds],
                dim=self._time_dim,
                data_vars="minimal",
                coords="minimal",
                compat="override",
                combine_attrs="override",
            )
            self._validate_merged_time(merged_ds)
            _prepare_append_encoding(merged_ds, new_ds)
            _prepare_append_attrs(merged_ds, existing_ds, new_ds)
            temp_path = _temporary_netcdf_path(output_path)
            write_netcdf(
                merged_ds,
                self._ctx.dataset,
                self._ctx.variable,
                path=temp_path,
                **self.to_netcdf_kwargs,
            )
            _ensure_nonempty_file(temp_path, "Append write")
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
        finally:
            if new_check_ds is not None:
                new_check_ds.close()
            if check_path is not None:
                check_path.unlink(missing_ok=True)
            existing_check_ds.close()
            existing_ds.close()

        os.replace(temp_path, output_path)
        return output_path

    def _validate_append_compatible(
        self,
        existing_ds: xr.Dataset,
        new_ds: xr.Dataset,
    ) -> None:
        issues: list[str] = []
        var_name = self._ctx.variable.names()[0]

        if var_name not in existing_ds:
            issues.append(f"existing file is missing data variable {var_name!r}")
        if var_name not in new_ds:
            issues.append(f"new dataset is missing data variable {var_name!r}")

        existing_variables = {str(name) for name in existing_ds.variables}
        new_variables = {str(name) for name in new_ds.variables}
        if existing_variables != new_variables:
            missing = sorted(new_variables - existing_variables)
            extra = sorted(existing_variables - new_variables)
            if missing:
                issues.append(f"existing file is missing variables {missing!r}")
            if extra:
                issues.append(f"existing file has unexpected variables {extra!r}")

        for dim, size in new_ds.sizes.items():
            if dim == self._time_dim:
                continue
            existing_size = existing_ds.sizes.get(dim)
            if existing_size != size:
                issues.append(
                    f"dimension {dim!r} has size {existing_size!r} in the "
                    f"existing file and {size!r} in the new dataset"
                )

        existing_attrs = _normalize_attrs(
            existing_ds.attrs,
            ignored=_APPEND_IGNORED_GLOBAL_ATTRS,
        )
        new_attrs = _normalize_attrs(
            new_ds.attrs,
            ignored=_APPEND_IGNORED_GLOBAL_ATTRS,
        )
        if existing_attrs != new_attrs:
            issues.extend(
                _attribute_diff_messages(
                    "global attribute",
                    existing_attrs,
                    new_attrs,
                )
            )

        for name in sorted(existing_variables & new_variables):
            existing_var = existing_ds[name]
            new_var = new_ds[name]
            if existing_var.dims != new_var.dims:
                issues.append(
                    f"variable {name!r} has dimensions {existing_var.dims!r} "
                    f"in the existing file and {new_var.dims!r} in the new dataset"
                )
                continue
            if np.dtype(existing_var.dtype) != np.dtype(new_var.dtype):
                issues.append(
                    f"variable {name!r} has dtype {existing_var.dtype!r} in "
                    f"the existing file and {new_var.dtype!r} in the new dataset"
                )
            existing_var_attrs = _normalize_attrs(
                existing_var.attrs,
                ignored=_APPEND_IGNORED_VARIABLE_ATTRS,
            )
            new_var_attrs = _normalize_attrs(
                new_var.attrs,
                ignored=_APPEND_IGNORED_VARIABLE_ATTRS,
            )
            if existing_var_attrs != new_var_attrs:
                issues.extend(
                    _attribute_diff_messages(
                        f"variable {name!r} attribute",
                        existing_var_attrs,
                        new_var_attrs,
                    )
                )

            if self._time_dim in existing_var.dims:
                for dim, existing_size in existing_var.sizes.items():
                    if dim == self._time_dim:
                        continue
                    new_size = new_var.sizes[dim]
                    if existing_size != new_size:
                        issues.append(
                            f"variable {name!r} dimension {dim!r} has size "
                            f"{existing_size!r} in the existing file and "
                            f"{new_size!r} in the new dataset"
                        )
                continue

            if existing_var.shape != new_var.shape:
                issues.append(
                    f"variable {name!r} has shape {existing_var.shape!r} "
                    f"in the existing file and {new_var.shape!r} in the new dataset"
                )
                continue
            if not _array_values_equal(existing_var.values, new_var.values):
                issues.append(f"variable {name!r} values differ")

        if issues:
            details = "\n".join(f"- {issue}" for issue in issues)
            raise ValueError(
                "Existing file is not compatible with append mode:\n" + details
            )

    def _validate_merged_time(self, merged_ds: xr.Dataset) -> None:
        time_name = _time_coord_name(merged_ds, self._time_dim, self._time_axis)
        values = np.asarray(merged_ds[time_name].values)
        if values.ndim != 1:
            raise ValueError(
                f"Time coordinate {time_name!r} must be one-dimensional after append."
            )
        if values.size > 1 and np.any(values[1:] <= values[:-1]):
            raise ValueError(
                "Appended time values must be strictly monotonic with the "
                "existing file."
            )

        bounds_name = (
            merged_ds[time_name].attrs.get("bounds")
            or merged_ds[time_name].attrs.get("climatology")
        )
        if not bounds_name or str(bounds_name) not in merged_ds:
            return

        bounds = _bounds_as_pairs(
            np.asarray(merged_ds[str(bounds_name)].values),
            int(values.size),
        )
        if bounds.shape[0] <= 1:
            return
        previous_ends = bounds[:-1, 1]
        next_starts = bounds[1:, 0]
        numeric_bounds = np.issubdtype(np.asarray(previous_ends).dtype, np.number)
        if numeric_bounds and not np.allclose(
            previous_ends,
            next_starts,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "Appended time bounds must be contiguous with the existing file."
            )


_APPEND_IGNORED_GLOBAL_ATTRS = frozenset(
    {
        "creation_date",
        "history",
        "tracking_id",
    }
)
_APPEND_IGNORED_VARIABLE_ATTRS = frozenset({"_FillValue"})


def _temporary_netcdf_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        return Path(handle.name)


def _ensure_nonempty_file(path: Path, operation: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise OSError(
            f"{operation} did not create a valid temporary file {str(path)!r}."
        )


def _prepare_append_encoding(merged_ds: xr.Dataset, new_ds: xr.Dataset) -> None:
    for name in merged_ds.variables:
        name_str = str(name)
        encoding = (
            dict(new_ds[name_str].encoding) if name_str in new_ds.variables else {}
        )
        for transient_key in ("preferred_chunks", "source", "original_shape"):
            encoding.pop(transient_key, None)

        array = merged_ds[name_str]
        chunksizes = encoding.get("chunksizes")
        if chunksizes is not None:
            normalized_chunksizes = tuple(
                min(int(chunk), int(size))
                for chunk, size in zip(tuple(chunksizes), array.shape, strict=False)
            )
            if _is_time_metadata_variable(name_str, array, merged_ds):
                normalized_chunksizes = tuple(int(size) for size in array.shape)
            encoding["chunksizes"] = normalized_chunksizes

        array.encoding.clear()
        array.encoding.update(encoding)
        if "_FillValue" in array.attrs and "_FillValue" in array.encoding:
            array.attrs.pop("_FillValue", None)


def _prepare_append_attrs(
    merged_ds: xr.Dataset,
    existing_ds: xr.Dataset,
    new_ds: xr.Dataset,
) -> None:
    attrs = dict(merged_ds.attrs)
    if "history" in existing_ds.attrs:
        attrs["history"] = existing_ds.attrs["history"]
    for name in ("creation_date", "tracking_id"):
        if name in new_ds.attrs:
            attrs[name] = new_ds.attrs[name]
    merged_ds.attrs = attrs


def _is_time_metadata_variable(
    name: str,
    array: xr.DataArray,
    ds: xr.Dataset,
) -> bool:
    lower_name = name.lower()
    if lower_name == "time" or lower_name.startswith("time"):
        return True
    for coord_name in ds.coords:
        coord = ds[str(coord_name)]
        if name in {
            str(coord.attrs.get("bounds", "")),
            str(coord.attrs.get("climatology", "")),
        }:
            return True
    return array.ndim > 0 and all(
        str(dim).lower() in {"time", "bnds"} or str(dim).lower().startswith("time")
        for dim in array.dims
    )


def _normalize_attrs(
    attrs: Mapping[Any, Any],
    *,
    ignored: frozenset[str],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in attrs.items():
        key_str = str(key)
        if key_str in ignored:
            continue
        normalized[key_str] = _normalize_attr_value(value)
    return normalized


def _normalize_attr_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(_normalize_attr_value(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_attr_value(item) for item in value)
    return value


def _attribute_diff_messages(
    label: str,
    existing_attrs: Mapping[str, Any],
    new_attrs: Mapping[str, Any],
) -> list[str]:
    messages: list[str] = []
    keys = sorted(set(existing_attrs) | set(new_attrs))
    for key in keys:
        if key not in existing_attrs:
            messages.append(
                f"{label} {key!r} is missing from the existing file"
            )
        elif key not in new_attrs:
            messages.append(
                f"{label} {key!r} is missing from the new dataset"
            )
        elif existing_attrs[key] != new_attrs[key]:
            messages.append(
                f"{label} {key!r} differs: existing={existing_attrs[key]!r}, "
                f"new={new_attrs[key]!r}"
            )
    return messages


def _array_values_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    try:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left_array, right_array))


def _time_coord_name(ds: xr.Dataset, time_dim: str, axis: Axis) -> str:
    candidates = [
        time_dim,
        str(axis.out_name or ""),
        str(axis.name or ""),
    ]
    for candidate in candidates:
        if candidate and candidate in ds:
            return candidate
    for name in ds.coords:
        coord = ds[str(name)]
        if time_dim in coord.dims:
            return str(name)
    raise ValueError(
        f"No coordinate variable was found for time dimension {time_dim!r}."
    )


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


def _pop_zfactor_value(
    values: dict[str, Any],
    zfactor: ZFactor,
) -> Any | None:
    if zfactor.name in values:
        return values.pop(zfactor.name)
    out_name = str(zfactor.out_name or zfactor.name)
    if out_name in values:
        return values.pop(out_name)
    return None


def _empty_zfactor(zfactor: ZFactor) -> bool:
    if zfactor.values is None:
        return True
    return np.asarray(zfactor.values).size == 0
