Incremental Writing
===================

:class:`cmor.DatasetWriter` writes a variable in pieces, so you never
need the whole variable in memory. Memory use depends only on the size of
each piece you write.

How it works:

1. Each ``write()`` call checks one piece (a *chunk*) and saves it to a
   temporary Zarr store on disk.
2. ``close()`` copies the saved data into the final NetCDF file.

Chunks split the data along time only. Give the full spatial and vertical
axes when you create the writer. Pick a chunk size that fits easily in
memory.

Write data in chunks
--------------------

Create the time axis without values. Then give each ``write()`` call the
time values for its chunk and, if the variable needs them, the time
bounds:

.. code-block:: python

   import cmor
   import numpy as np

   project = cmor.ProjectTables.from_directory(
       "project_tables/cmip7-cmor-tables",
       cv_file="tables-cvs/cmor-cvs.json",
       variable_tables=["tables/CMIP7_ocean.json"],
   )
   dataset = project.dataset_info({
       "activity_id": "CMIP",
       "experiment_id": "amip",
       "forcing_index": "f1",
       "grid_label": "g101",
       "initialization_index": "i1",
       "institution_id": "CNRM-CERFACS",
       "license_id": "CC-BY-4.0",
       "nominal_resolution": "100 km",
       "physics_index": "p1",
       "realization_index": "r1",
       "region": "glb",
       "source_id": "CNRM-ESM2-1e",
       "outpath": "./output",
   })
   variable = project.variable("tos_tavg-u-hxy-sea", table_id="ocean")

   axes = [
       project.axis(
           "time", calendar="360_day", frequency="mon",
           units="days since 1979-01-01",
       ),
       project.axis(
           "latitude", values=[10.0, 20.0, 30.0],
           bounds=[[5.0, 15.0], [15.0, 25.0], [25.0, 35.0]],
       ),
       project.axis(
           "longitude", values=[0.0, 90.0, 180.0, 270.0],
           bounds=[[-45.0, 45.0], [45.0, 135.0],
                   [135.0, 225.0], [225.0, 315.0]],
       ),
   ]

   with cmor.DatasetWriter(
       dataset, variable, axes, path="./output/tos_incremental.nc",
   ) as writer:
       writer.write(
           np.full((1, 3, 4), 20.0, dtype="f4"),
           time_values=[15.5], time_bounds=[[0.0, 31.0]],
       )
       writer.write(
           np.full((1, 3, 4), 21.0, dtype="f4"),
           time_values=[45.5], time_bounds=[[31.0, 60.0]],
       )
   print(writer.path)

The ``with`` block calls ``close()`` for you when it ends without an
error. You can also call ``close()`` yourself; it returns the output path.

Rules for each write:

* time values must keep increasing from one write to the next;
* each chunk's shape must match the non-time axes;
* time bounds must be given in every write or in none, and must join up
  without gaps.

Pre-specified time values
-------------------------

If you know all the time values in advance, set them and their bounds on
the axis when you create it. Each ``write()`` call then uses the next time
values from the axis, so you pass only the data:

.. code-block:: python

   time = project.axis(
       "time", calendar="360_day", frequency="mon",
       values=[15.5, 45.5], bounds=[[0.0, 31.0], [31.0, 60.0]],
       units="days since 1979-01-01",
   )
   complete_axes = [time, *axes[1:]]
   with cmor.DatasetWriter(
       dataset, variable, complete_axes, path="./output/tos_prespecified.nc",
   ) as writer:
       writer.write(np.full((1, 3, 4), 20.0, dtype="f4"))
       writer.write(np.full((1, 3, 4), 21.0, dtype="f4"))

Append to an existing file
--------------------------

To add later times to a file that already exists, pass
``existing="append"``. The writer must use the same variable and non-time
axes as the existing file:

.. code-block:: python

   output = "./output/tos_series.nc"
   with cmor.DatasetWriter(dataset, variable, axes, path=output) as writer:
       writer.write(
           np.full((1, 3, 4), 20.0, dtype="f4"),
           time_values=[15.5], time_bounds=[[0.0, 31.0]],
       )

   with cmor.DatasetWriter(
       dataset, variable, axes, path=output, existing="append",
   ) as writer:
       writer.write(
           np.full((1, 3, 4), 21.0, dtype="f4"),
           time_values=[45.5], time_bounds=[[31.0, 60.0]],
       )

``existing`` also accepts ``"error"`` (the default, which fails if the
file exists) and ``"replace"``.

When appending, CMOR checks the new data against the existing file: its
axes, grid mapping, formula terms, metadata, and that the new times come
after the old ones. If the append fails, the original file is not
changed.

Write separate segments
-----------------------

To split a series into several files, call
``close(preserve_definition=True)``. This finishes the current file but
keeps the writer's variable and axes, so you can set a new ``path`` and keep
writing. Unlike chunks in one file, separate files may have a gap in
time:

.. code-block:: python

   from pathlib import Path

   writer = cmor.DatasetWriter(
       dataset, variable, axes, path="./output/segment_1.nc",
   )
   writer.write(
       np.full((1, 3, 4), 20.0, dtype="f4"),
       time_values=[15.5], time_bounds=[[0.0, 31.0]],
   )
   first_path = writer.close(preserve_definition=True)

   writer.path = Path("./output/segment_2.nc")
   writer.write(
       np.full((1, 3, 4), 21.0, dtype="f4"),
       time_values=[375.5], time_bounds=[[360.0, 391.0]],
   )
   second_path = writer.close()

Time-varying formula terms
--------------------------

For a hybrid vertical coordinate, give the constant formula terms their
values when you create the writer. Create a term that changes with time,
such as surface pressure ``ps``, without values. Then pass its values with
every chunk. This code writes hybrid-level cloud fraction one month at
a time, passing a new surface pressure array with each month:

.. code-block:: python

   zfactors = [
       project.zfactor("a", values=a, bounds=a_bounds),
       project.zfactor("b", values=b, bounds=b_bounds),
       project.zfactor("p0", values=100000.0),
       project.zfactor("ps"),
   ]
   cloud_chunk_1, cloud_chunk_2 = data[0:1], data[1:2]
   ps_chunk_1 = np.full((1, 3, 4), 100000.0, dtype="f4")
   ps_chunk_2 = np.full((1, 3, 4), 100100.0, dtype="f4")
   with cmor.DatasetWriter(
       dataset, cloud_variable, hybrid_axes, zfactors=zfactors,
       path="./output/cl_incremental.nc",
   ) as writer:
       writer.write(
           cloud_chunk_1, time_values=[15.5],
           time_bounds=[[0.0, 31.0]], zfactors={"ps": ps_chunk_1},
       )
       writer.write(
           cloud_chunk_2, time_values=[45.5],
           time_bounds=[[31.0, 60.0]], zfactors={"ps": ps_chunk_2},
       )

Each ``ps`` chunk must have the dimensions listed in the formula table
(here ``(time, lat, lon)``). Once you pass a time-varying term, you must
pass it in every later write.

Output and recovery options
---------------------------

``DatasetWriter`` accepts ``path``, ``attrs`` and ``encoding``, which
work as in :func:`cmor.cmorize`.

Use ``staging_dir`` to choose where the temporary Zarr data goes. After a
successful ``close()``, CMOR deletes it. After an error, CMOR keeps it so
you can inspect it; ``writer.staging_path`` gives its location. See
:class:`cmor.DatasetWriter` for the full API.
