Time Coordinates
================

Set ``calendar`` and ``frequency`` on the time axis. From these, CMOR:

* checks that the calendar is valid and that the time values are spaced as
  the frequency requires;
* writes the calendar on the time coordinate;
* sets the dataset's ``frequency`` attribute.

If you set ``frequency``, it must match the variable table. If you leave it
out, CMOR uses the table's frequency. Fixed fields, such as land area, have
no time axis.

Calendars and bounds
--------------------

Set the calendar that your source data uses. Give time values and bounds
in the axis's ``units``. This code creates a monthly time axis on a
``noleap`` calendar:

.. code-block:: python

   time_axis = project.axis(
       "time", calendar="noleap", frequency="mon",
       values=[15.0, 45.0, 74.0],
       bounds=[[0.0, 30.0], [30.0, 60.0], [60.0, 90.0]],
       units="days since 1850-01-01",
   )

Pass ``calendar`` as an argument to the axis, as above. Putting it in the
axis's ``attrs`` or in the dataset metadata raises an error. A project's CV
can still require other calendar-related global attributes; see
:doc:`controlled_vocabularies`.

Forecast reference time
-----------------------

Forecast data has a *reference time*: the time the forecast started. When
you include a reference-time axis, CMOR adds a ``leadtime`` coordinate equal
to ``time - reference time``. This short example builds the axes and
variable directly, without project tables, to keep the focus on time:

.. code-block:: python

   import cmor
   import numpy as np

   time = cmor.Axis(
       name="time", standard_name="time", axis="T",
       values=[0.0, 6.0, 12.0], units="hours since 2020-01-01",
       calendar="standard", frequency="6hr",
   )
   reference_time = cmor.Axis(
       name="reftime1", out_name="reftime",
       standard_name="forecast_reference_time", scalar=True,
       values=[0.0], units="hours since 2020-01-01",
   )
   variable = cmor.Variable(
       name="tas", dimensions=("time",), units="K",
   )
   path = cmor.cmorize(
       cmor.DatasetInfo(), variable, [time, reference_time],
       np.full(3, 280.0), path="./output/forecast.nc",
   )
   print(path)

The output file has a ``leadtime`` coordinate with values ``[0, 6, 12]``
in hours. ``DatasetWriter`` also adds ``leadtime`` when you write in pieces;
see :doc:`incremental_writing`.
