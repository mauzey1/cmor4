.. _controlled-vocabularies:

Controlled Vocabularies
=======================

A *controlled vocabulary* (CV) is a project-level JSON file. It defines
which global attributes a dataset must have, which values those attributes
may take, and which attributes CMOR can fill in for you. Each project
(CMIP7, obs4MIPs, DRCDP, …) ships its own CV alongside its MIP tables.

This page uses these terms:

Default
   A value CMOR fills in for an attribute you did not supply.
Constraint
   A rule that a value you supply must satisfy, such as "must be one of
   these keys".
Template
   A string containing ``<token>`` placeholders, such as
   ``"r<realization_index>i<...>"``.
Single-valued entry
   A CV entry with exactly one possible value: a scalar, a one-item list,
   or an object with one key.

Some behavior is surprising or applies only in narrow cases. Those
details are collected in :ref:`cv-caveats`.


How CMOR applies a CV
---------------------

CMOR works with a CV in two stages:

1. **Applying defaults**: attributes you did not supply are filled from the
   CV (for example, ``institution`` from ``institution_id``).
2. **Validation**: values with CV constraints, required attributes, and
   relationships between attributes are checked. A failed check raises
   :class:`~cmor.ControlledVocabularyError`. Attributes that the CV does not
   mention are generally allowed.

Defaults never replace a nonempty value you supplied. Most defaults fill
only attributes that are absent. The ``source_id``, ``institution`` and
``experiment_id`` defaults also treat ``None`` and ``""`` as missing and
replace them.

Defaults are applied in this order:

1. scalar CV values;
2. ``source_id`` record values;
3. ``institution`` from ``institution_id``;
4. ``experiment_id`` record values;
5. ``license`` rendered from its template;
6. runtime values (``Conventions``, ``creation_date``, ``variant_label``,
   ``tracking_id``) and single-valued required attributes;
7. attributes injected by lookup objects.

Validation happens at two points:

* **When dataset metadata is prepared** (``ProjectTables.dataset_info``),
  CMOR checks controlled values, RIPF indices, forcing, derived names, and
  source, experiment and parent consistency. It does *not* yet check that
  every required attribute is present.
* **When a variable is prepared for writing, and again before the final
  dataset is written**, CMOR checks the following in order and stops at
  the first error:

  1. supplied values that have a CV constraint;
  2. required attributes (see :ref:`cv-required-attributes`);
  3. ``source``, ``experiment`` and ``institution`` against their ``*_id``
     records;
  4. the RIPF indices and ``variant_label``;
  5. ``forcing`` tokens;
  6. source, experiment and parent consistency.


File structure
--------------

A CV file is a JSON object with a single top-level ``CV`` key:

.. code-block:: json

   {
     "CV": {
       "required_global_attributes": ["activity_id", "experiment_id", "..."],
       "activity_id": {"CMIP": "CMIP DECK ..."},
       "grid_label": ["gn", "gr", "gr1"],
       "mip_era": "CMIP7",
       "...": "..."
     }
   }

:class:`~cmor.ControlledVocabulary` accepts either this full document or
just the body of ``CV``.

Most keys inside ``CV`` are names of NetCDF global attributes. A few keys
control CMOR instead: ``DRS``, ``required_global_attributes`` and
``tracking_id_prefix``.


How CMOR reads an ordinary entry
--------------------------------

Most entries need no special handling. CMOR decides how to treat an entry
from the **JSON type of its value** alone. Projects can therefore add new
attributes without changes to CMOR.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - JSON type
     - Example
     - What CMOR does
   * - String, number or boolean
     - ``"mip_era": "CMIP7"``
     - The dataset value must equal it exactly. If you omit the attribute,
       CMOR uses this value as the default.
   * - String containing ``<tokens>``
     - ``"variant_label": "r<realization_index>i<...>"``
     - A template. The dataset value must equal either the rendered
       template or the unrendered template text. Templates are not used as
       defaults (with one exception; see :ref:`cv-caveats`).
   * - List
     - ``"grid_label": ["gn", "gr", "gr1"]``
     - The value must match one item. Each item is tried first as a literal
       string, then as a regular expression that must match the whole
       value. Some POSIX character classes are translated to Python syntax.
       Some one-item POSIX patterns are handled differently (see
       :ref:`cv-legacy-posix-regex`).
   * - Object
     - ``"activity_id": {"CMIP": "description"}``
     - The value must be one of the object's **keys**. The object's values
       are descriptions. For ``realm`` and ``source_type``, the dataset
       value is split on whitespace and every word must be a key.
   * - ``null`` or any other type
     - ``"further_info_url": null``
     - No constraint.

An attribute you leave out is only checked if another rule requires it.

Empty entries
~~~~~~~~~~~~~

An empty list or empty object gives no allowed values, so CMOR treats it as
an error in the CV:

* **Error**: an empty list or object as the value of a top-level attribute.
  CMOR raises ``ControlledVocabularyError`` when it loads the CV, even if
  the attribute is optional or not in the dataset. The error names the
  entry, and the source file name when available. If you do not want to
  constrain an attribute, leave its entry out.
* **Allowed**: an empty CV body, and empty structural sections
  (``required_global_attributes``, ``DRS`` and ``version_metadata``).
* **Allowed**: empty values nested inside an entry. For example,
  ``"grid_label": {"gn": {}}`` still allows the key ``gn``, and an
  experiment's ``"additional_allowed_model_components": []`` can mean no
  extra components are allowed.

To allow an empty string for an optional attribute, list it explicitly:
``[""]``.

Lookup objects that inject attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some objects map each key to a record whose values are all strings or
numbers. Selecting a key from such an object also **writes the record's
values as global attributes**. For example, an obs4MIPs CV might define:

.. code-block:: json

   "site_id": {
     "AR-SLu": {"latitude": "-33.47", "longitude": "-66.46", "location": "San Luis"}
   }

Setting ``site_id="AR-SLu"`` writes ``latitude``, ``longitude`` and
``location`` to the file.

This step runs last. It never overrides an attribute you supplied or one
filled by a :ref:`special section <cv-special-sections>`. It does not apply
to ``DRS``, ``experiment_id``, ``frequency``, ``institution_id``,
``license``, ``license_id``, ``source_id`` or ``source_type``.


.. _cv-required-attributes:

Required attributes
-------------------

``required_global_attributes`` lists the attributes every dataset must have
once defaults have been applied. CMOR can fill in some of them itself:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - How CMOR fills it when missing
   * - ``Conventions``
     - ``CF-1.12`` if the CV lists that value. Otherwise the CV's scalar
       value, or the last value in its list. With no usable CV value,
       ``CF-1.12``. See :ref:`cv-caveats` for a special case.
   * - ``creation_date``
     - The current time, formatted ``YYYY-MM-DDTHH:MM:SSZ`` (see
       :ref:`cv-caveats` about time zones).
   * - ``variant_label``
     - Built from ``realization_index``, ``initialization_index``,
       ``physics_index`` and ``forcing_index`` (see :ref:`cv-variant-label`).
   * - ``tracking_id``
     - A new UUID, with a prefix when one is available (see
       :ref:`cv-tracking-id`).
   * - Others
     - The CV value, if the entry is single-valued. One-item lists that
       look like regular expressions are not used.

If a required attribute is still missing at validation, validation fails.
(One exception for legacy POSIX patterns is described in
:ref:`cv-legacy-posix-regex`.)


.. _cv-special-sections:

Sections with special meaning
-----------------------------

The keys below get extra handling. Most of them also get the ordinary check
from the table above. For example, ``source_id`` must still be a key of its
catalog. For a few, the special check replaces or relaxes the ordinary one.

Each section below shows an excerpt from the CMIP7 CV
(``project_tables/cmip7-cmor-tables/tables-cvs/cmor-cvs.json``). Where
CMIP7 does not use a section, the excerpt comes from another CV. Long
strings are shortened with ``...``.

``source_id``
~~~~~~~~~~~~~

A catalog of models. Each key is a model identifier, and its record lists
attributes that model implies:

.. code-block:: json

   "source_id": {
     "CNRM-ESM2-1e": {
       "source": "CNRM-ESM2-1e: aerosol: tactic; atmosphere: arpege-climat-version-6-3; ...",
       "source_id": "CNRM-ESM2-1e"
     }
   }

When you select a ``source_id``:

* record values that are scalars or one-item lists become defaults;
* any record attribute already in your dataset must equal the record value
  (or be one of its values, if the record value is a list).

``source`` is checked only when it is a required attribute. The
``source_id`` catalog is also used to check ``parent_source_id``.

``experiment_id``
~~~~~~~~~~~~~~~~~

A catalog of experiments:

.. code-block:: json

   "experiment_id": {
     "historical": {
       "activity_id": ["CMIP"],
       "description": "Simulation of the climate of the recent past ...",
       "end_year": 2021,
       "experiment": "Simulation of the climate of the recent past ...",
       "experiment_id": "historical",
       "min_number_yrs_per_sim": 172,
       "parent_activity_id": ["CMIP"],
       "parent_experiment_id": ["piControl"],
       "start_year": 1850,
       "tier": 1
     }
   }

Selecting an experiment:

* fills defaults from the record's scalar and one-item-list values, such as
  ``experiment`` or ``activity_id``. These fields are not used as defaults:
  ``start_year``, ``end_year``, ``tier``, ``parent_activity_id``,
  ``parent_experiment_id``, ``required_source_type``, ``source_type`` and
  ``additional_allowed_model_components``;
* checks the other record attributes in your dataset against the record,
  except ``description`` and fields handled separately;
* restricts ``source_type`` (see :ref:`cv-source-type`);
* decides which parent attributes are allowed (see
  :ref:`cv-parent-attributes`).

``institution_id``
~~~~~~~~~~~~~~~~~~

Maps each institution identifier to the institution's full name:

.. code-block:: json

   "institution_id": {
     "CNRM-CERFACS": "CNRM (Centre National de Recherches Meteorologiques, Toulouse 31057, France), CERFACS (...)"
   }

CMOR fills ``institution`` from it. If ``institution`` is required and you
supplied a different value, validation fails.

``license``
~~~~~~~~~~~

Accepts three forms:

* **A string**: the exact license text. It is used as the default, and a
  supplied value must equal it.
* **A list**: patterns the license text must match.
* **An object** with ``license_id`` and ``license_template``. CMIP7 uses
  this form:

  .. code-block:: json

     "license": {
       "license_id": {
         "CC-BY-4.0": {
           "license_type": "Creative Commons Attribution 4.0 International",
           "license_url": "https://creativecommons.org/licenses/by/4.0"
         },
         "CC0-1.0": {"license_type": "...", "license_url": "https://creativecommons.org/publicdomain/zero/1.0"}
       },
       "license_template": "<license_id>; CMIP7 data produced by <institution_id> is licensed under a <license_type> License (<license_url>). ..."
     }

  A supplied ``license_id`` must be a key of the catalog. A supplied
  ``license_type`` or ``license_url`` must match the selected entry. If you
  do not supply ``license``, CMOR renders it from the template.

.. _cv-source-type:

``source_type``
~~~~~~~~~~~~~~~

The project's list of model component types (a list or an object). The
dataset value is a space-separated list, and every word must be in the
CV's list.

The CMIP7 CV does not define ``source_type``, and its experiments do not
list required components, so none of these checks apply to CMIP7. The
CMIP6 CV (``project_tables/cmip6-cmor-tables/Tables/CMIP6_CV.json``) shows
the form:

.. code-block:: json

   "source_type": {
     "AER": "aerosol treatment in an atmospheric model where concentrations are calculated ...",
     "AGCM": "atmospheric general circulation model run with prescribed ocean surface conditions ...",
     "AOGCM": "coupled atmosphere-ocean global climate model, ...",
     "...": "..."
   }

and, in each CMIP6 experiment record:

.. code-block:: json

   "historical": {
     "required_model_components": ["AOGCM"],
     "additional_allowed_model_components": ["AER", "CHEM", "BGC"],
     "...": "..."
   }

The selected experiment can restrict this further. CMOR uses the
experiment's ``required_source_type`` if it is nonempty, otherwise its
``required_model_components``. Each required pattern must be found in the
``source_type`` value. Each word in the value must then match a required
pattern or an entry in ``additional_allowed_model_components``. These are
regular-expression searches (see :ref:`cv-caveats`).

``frequency``
~~~~~~~~~~~~~

An object whose keys are the allowed frequency codes. An entry can include
``approx_interval``, ``approx_interval_warning`` and
``approx_interval_error``:

.. code-block:: json

   "frequency": {
     "1hr": {
       "approx_interval": 0.041666666666666664,
       "approx_interval_error": 0.2,
       "approx_interval_warning": 0.1,
       "description": "Hourly samples."
     },
     "mon": {
       "approx_interval": 30.0,
       "approx_interval_error": 0.2,
       "approx_interval_warning": 0.1,
       "description": "Monthly samples. ..."
     }
   }

CMOR uses these to check the spacing of time
coordinates. They are never written as global attributes.

``forcing``
~~~~~~~~~~~

Used by CMIP6-style projects. None of the bundled CVs defines it, but an
entry looks like this (taken from CMOR's test suite,
``tests/test_cmor3_parity.py``):

.. code-block:: json

   "forcing": ["GHG", "Oz", "SA", "Sl", "Vl", "BC", "OC", "Nat", "Ant"]

The dataset value is a list of abbreviations separated by spaces or
commas, optionally followed by a note in parentheses:

.. code-block:: text

   GHG Oz SA Sl Vl BC OC (GHG = CO2, N2O, CH4, …)

CMOR ignores everything from the first ``(`` onward. Every remaining
abbreviation must be in the CV's ``forcing`` list. If the CV has no
``forcing`` entry (as in CMIP7 and obs4MIPs), the attribute is not checked.

``DRS``
~~~~~~~

Holds ``directory_path_template`` and ``filename_template``, the templates
CMOR uses to build output directory paths and file names. CMOR ignores any
other keys in this section, such as CMIP7's ``*_example`` entries:

.. code-block:: json

   "DRS": {
     "directory_path_example": "MIP-DRS7/CMIP7/CMIP/CNRM-CERFACS/CNRM-ESM2-1e/1pctCO2/r1i1p1f1/glb/mon/rsus/tavg-h2m-hxy-u/g101/20251104",
     "directory_path_template": "<drs_specs><mip_era><activity_id><institution_id><source_id><experiment_id><variant_label><region><frequency><variable_id><branding_suffix><grid_label><version>",
     "filename_example": "rsus_tavg-h2m-hxy-u_mon_glb_g101_CNRM-ESM2-1e_1pctCO2_r1i1p1f1_185001-202112.nc",
     "filename_template": "<variable_id><branding_suffix><frequency><region><grid_label><source_id><experiment_id><variant_label>"
   }

.. _cv-tracking-id:

``tracking_id`` and ``tracking_id_prefix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the CV requires ``tracking_id`` and you do not supply one, CMOR
generates a UUID. With a prefix the ID has the form ``<prefix>/<uuid4>``.
Without one it is just the UUID. If ``tracking_id`` is not required and
you do not supply one, CMOR does not generate it.

CMOR keeps a ``tracking_id`` you supply. It checks the value only when the
CV defines a constraint for it.

The prefix for a generated ID comes from the first of these that is
available:

1. a ``tracking_prefix`` attribute you set on the dataset;
2. the CV's ``tracking_id_prefix`` (a string or one-item list);
3. the fixed leading part of the CV's ``tracking_id`` pattern. For example,
   ``"^hdl:21.14100/.*$"`` gives the prefix ``hdl:21.14100``.

The DRCDP CV (``project_tables/DRCDP/Tables/DRCDP_CV.json``) uses option 2
with a one-item list:

.. code-block:: json

   "tracking_id_prefix": ["hdl:21.14100"]

The CMIP7 CV has no ``tracking_id_prefix``. Instead it has a scalar
``tracking_prefix`` entry alongside the ``tracking_id`` pattern:

.. code-block:: json

   "tracking_id": [
     "^hdl:21\\.14107/[a-fA-F0-9]\\{8\\}-[a-fA-F0-9]\\{4\\}-[a-fA-F0-9]\\{4\\}-[a-fA-F0-9]\\{4\\}-[a-fA-F0-9]\\{12\\}$"
   ],
   "tracking_prefix": "hdl:21.14107"

The scalar ``tracking_prefix`` becomes a dataset default, so it is used as
the prefix in option 1: a generated ID looks like
``hdl:21.14107/1c4f33bb-12a4-46cd-b22f-3d78ada73c58``. ``tracking_prefix``
is never written as a global attribute.

.. _cv-variant-label:

``variant_label`` and the RIPF indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each of ``realization_index``, ``initialization_index``, ``physics_index``
and ``forcing_index`` must be a positive integer. You can give it alone
(``3``) or with its letter (``r3``). When all four are present, CMOR builds
``variant_label`` from them (``r3i1p1f1``).

The CMIP7 CV constrains the label and each index with legacy POSIX
patterns (see :ref:`cv-legacy-posix-regex`):

.. code-block:: json

   "realization_index": ["^r[[:digit:]]\\{1,\\}$"],
   "initialization_index": [
     "^i[[:digit:]]\\{1,\\}$",
     "^i[[:digit:]]\\{4\\}[[:digit:]]\\{2\\}[abcde]\\{0,\\}$"
   ],
   "physics_index": ["^p[[:digit:]]\\{1,\\}$"],
   "forcing_index": ["^f[[:digit:]]\\{1,\\}$"],
   "variant_label": [
     "^r[[:digit:]]\\{1,\\}i[[:digit:]]\\{1,\\}p[[:digit:]]\\{1,\\}f[[:digit:]]\\{1,\\}$",
     "^r[[:digit:]]\\{1,\\}i[[:digit:]]\\{4\\}[[:digit:]]\\{2\\}[abcde]\\{0,\\}p[[:digit:]]\\{1,\\}f[[:digit:]]\\{1,\\}$"
   ]

CMOR enforces the ``r<N>i<N>p<N>f<N>`` format only when the CV gives
``variant_label`` a list of allowed values or patterns. If the CV only
requires the attribute (as obs4MIPs does), any string is accepted.

``grid_label``
~~~~~~~~~~~~~~

CMIP7 defines ``grid_label`` as an object of grid codes and descriptions:

.. code-block:: json

   "grid_label": {
     "g010": "Horizontal grid cell with a regular latitude longitude grid type and 360 x 180 degree resolution.",
     "g100": "Horizontal grid cell with a regular latitude longitude grid type and 1.25 x 0.9 degree resolution.",
     "g101": "Horizontal grid cell with a linear spectral gaussian grid type and 1.4 x 1.4 degree resolution.",
     "...": "..."
   }

If the CV defines ``grid_label``, the ordinary rules apply. If it does not,
CMOR still requires a label that starts with ``g``, ``c`` or ``r``
followed by lowercase letters and digits (``gn``, ``gr1``, ``cn``).

.. _cv-parent-attributes:

Parent experiment attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The selected experiment's ``parent_experiment_id`` decides whether parent
attributes are allowed. The parent attributes are
``parent_experiment_id``, ``parent_activity_id``, ``parent_mip_era``,
``parent_source_id``, ``parent_time_units``, ``parent_variant_label``,
``branch_time_in_child`` and ``branch_time_in_parent``.

In CMIP7 these come from the experiment records. ``historical`` has a
parent and ``amip`` does not:

.. code-block:: json

   "historical": {
     "parent_activity_id": ["CMIP"],
     "parent_experiment_id": ["piControl"],
     "...": "..."
   },
   "amip": {
     "parent_activity_id": [],
     "parent_experiment_id": [],
     "...": "..."
   }

* If the experiment's ``parent_experiment_id`` is empty or
  ``"no parent"``, none of the parent attributes may be set.
* Otherwise all of them are required, and:

  * ``parent_experiment_id`` and ``parent_activity_id`` must match the
    experiment record;
  * ``parent_source_id`` must be in the ``source_id`` catalog;
  * ``parent_mip_era`` must equal ``mip_era``;
  * ``parent_time_units`` must look like ``days since YYYY-MM-DD``;
  * ``parent_variant_label`` must look like ``r1i1p1f1``;
  * both branch times must be numbers.


Common authoring mistakes
-------------------------

Double-nested entries
   Wrapping a value in an object keyed by its own name:

   .. code-block:: json

      "nominal_resolution": {"nominal_resolution": ["0.5 km", "1 km"]}

   instead of ``"nominal_resolution": ["0.5 km", "1 km"]``. When CMOR loads
   the CV, it emits a :class:`RuntimeWarning` that describes the malformed
   entry.

Legacy POSIX regular expressions
   CMOR checks these less strictly than other values (see
   :ref:`cv-legacy-posix-regex`). In new CVs, use explicit lists or Python
   regular expressions.

Templates in required attributes
   A required attribute defined by a template can end up with the
   unrendered template as its value (see :ref:`cv-caveats`). Supply the
   rendered value yourself.


.. _cv-caveats:

Caveats and edge cases
----------------------

These behaviors reflect the current implementation and may be surprising.

.. _cv-legacy-posix-regex:

Legacy POSIX regular expressions
   CMIP6 CVs write some constraints as one-item lists of POSIX patterns,
   such as ``"^[[:digit:]]\\{1,\\}$"``. For these entries CMOR:

   * skips the ordinary list check, and applies only the attribute's
     special checks, if it has any. For attributes without special checks,
     even an invalid value can pass;
   * does not use the pattern as a default;
   * allows the attribute to be missing even if it is required.

Templates copied as required defaults
   Templates (scalar values containing ``<`` or ``>``) are not used as
   ordinary defaults. But if the attribute is required, the
   single-valued-required-default step can still copy the unrendered
   template text as its value.

``Conventions`` fallback
   If the last value in the CV's ``Conventions`` list looks like a regular
   expression, CMOR uses ``CF-1.7 CMIP-6.2`` as the default.

``creation_date`` time zone
   The value ends in ``Z`` (meaning UTC), but CMOR uses local time and does
   not convert it to UTC.

``tracking_id_prefix`` written as an attribute
   If the CV's ``tracking_id_prefix`` is a string, CMOR also writes it as a
   global attribute. A one-item list is not written.

``required_model_components`` as an attribute
   A one-item ``required_model_components`` list in an experiment record
   is not excluded from experiment defaults, so it can become a global
   attribute.

``license`` text in object form
   When ``license`` is an object, CMOR accepts any ``license`` text you
   supply. It does not check it against the template.

Unanchored ``source_type`` patterns
   Experiment ``source_type`` requirements use regular-expression
   searches, so a pattern without ``^`` and ``$`` can match part of a
   word. For example, the required pattern ``A`` matches ``BAD``.

Supplied values are not always checked
   Supplying a value does not guarantee that CMOR checks it. The legacy
   POSIX and ``license`` cases above are examples.


See also
--------

* :class:`cmor.ControlledVocabulary`: loading and checking a CV from Python.
* :class:`cmor.ControlledVocabularyError`: the exception raised when a
  check fails.
