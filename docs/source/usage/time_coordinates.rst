Time Coordinates
================

Put ``calendar`` and ``frequency`` on the sampling time axis. CMOR validates
the calendar and time spacing, writes the coordinate's calendar, and derives
the dataset frequency used for output metadata. An explicit frequency must
agree with the variable table; when omitted, CMOR uses the table frequency.
Fixed fields need no time axis.

Calendars and bounds
--------------------

The time axis in :doc:`basic_usage` uses a ``360_day`` calendar. Change the
calendar to match the source data and give time values and bounds in the
axis's stated units:

.. code-block:: python

   time_axis = project.axis(
       "time", calendar="noleap", frequency="mon",
       values=[15.0, 45.0, 74.0],
       bounds=[[0.0, 30.0], [30.0, 60.0], [60.0, 90.0]],
       units="days since 1850-01-01",
   )

``calendar`` is an Axis field; it cannot be supplied through ``attrs``.
Project CVs can separately require global calendar metadata. See
:doc:`controlled_vocabularies` for dataset metadata rules.

Forecast reference time
-----------------------

When a forecast reference-time coordinate is present, CMOR derives a
``forecast_period`` (lead time) coordinate from it and the sampling time.
This small example uses metadata objects directly so the coordinate behavior
is clear:

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
       np.full(3, 280.0), path="forecast.nc",
   )

The output includes ``leadtime`` with values ``[0, 6, 12]`` and units of
hours. ``DatasetWriter`` derives lead time across successive time chunks too;
see :doc:`incremental_writing` for the writing pattern.
