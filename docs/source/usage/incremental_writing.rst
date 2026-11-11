Incremental Writing
===================

:class:`cmor.DatasetWriter` writes large variables while keeping data memory
use bounded by the size of each write. It validates and stages each chunk in
Zarr, then streams the staged data into a NetCDF file when ``close()`` runs.
Choose chunks that fit comfortably in memory. Only the time dimension can
grow: provide complete spatial and vertical axes when creating the writer.
Start with :doc:`basic_usage` for the project, dataset, and variable
definitions used below.

Write data in chunks
--------------------

The time axis can start without values. Give each ``write()`` call the time
values and, when the variable needs them, matching bounds. The following uses
``project``, ``dataset``, and ``variable`` from the Basic Usage example:

.. code-block:: python

   import cmor
   import numpy as np

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

The context manager calls ``close()`` after a successful block. You can call
``close()`` yourself to get the output path. The time values must increase,
chunk shapes must match the non-time axes, and bounds must be supplied
consistently across writes.

Pre-specified time values
-------------------------

When the complete time axis is known, set its values and bounds at creation.
``write()`` then takes successive slices from that axis:

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

Use ``existing="append"`` to add later time records to a file created with
the same variable, non-time axes, and compatible metadata:

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

The other ``existing`` choices are ``"error"`` (the default) and
``"replace"``. Append validates the new data against the existing file,
including its axes, grid mapping, formula terms, metadata, and time sequence.
A failed append leaves the original file intact.

Write separate segments
-----------------------

``close(preserve_definition=True)`` writes a file and keeps the variable and
axis definitions for another segment. Separate files may have a time gap:

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

For a hybrid vertical coordinate, define static formula terms when creating
the writer. Leave a time-varying term such as surface pressure without values,
then supply its values in every chunk. The following uses ``project``,
``dataset``, ``cloud_variable``, ``hybrid_axes``, the coefficient arrays, and
``data`` from :doc:`vertical_coordinates`:

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

Each ``ps`` chunk must have the dimensions required by its formula-term table
entry. Once a time-varying term is supplied, it must be supplied in every
subsequent write.

Output and recovery options
---------------------------

``DatasetWriter`` accepts ``path``, ``attrs``, and ``encoding`` as in
:func:`cmor.cmorize`. Use ``staging_dir`` to choose where temporary Zarr data
is stored. CMOR removes the staging directory after a successful close and
preserves it after an error; ``writer.staging_path`` identifies the staged
data for inspection. See :class:`cmor.DatasetWriter` for the full API.
