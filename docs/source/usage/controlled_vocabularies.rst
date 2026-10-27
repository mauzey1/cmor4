.. _controlled-vocabularies:

Controlled Vocabularies
=======================

A *controlled vocabulary* (CV) is the project-level JSON file that defines
which global attributes a dataset must carry, which values those attributes
may take, and which attributes CMOR can fill in for you. Every project
(CMIP7, obs4MIPs, DRCDP, …) ships its own CV alongside its MIP tables.

CMOR uses the CV in two passes when you create a dataset:

1. **Defaulting** — attributes you did not supply are filled from the CV
   (for example, ``institution`` from ``institution_id``).
2. **Validation** — every attribute is checked against the CV, and a
   :class:`~cmor.ControlledVocabularyError` is raised on the first problem.

Values you supply are never overwritten by defaults. If you set an attribute
yourself, CMOR validates it rather than replacing it.


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
just the body of ``CV``. Each key inside ``CV`` is normally the name of a
NetCDF global attribute. A few keys (``DRS``, ``required_global_attributes``,
``tracking_id_prefix``) describe CMOR behavior rather than an attribute.


How CMOR reads an ordinary entry
--------------------------------

Most entries need no special handling. CMOR decides how to treat them from
the **JSON shape** of the value alone, so projects can add new attributes
without changes to CMOR.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - JSON shape
     - Example
     - What CMOR does
   * - String, number or boolean
     - ``"mip_era": "CMIP7"``
     - The dataset value must equal it exactly. If you omit the attribute,
       CMOR writes this value as a default.
   * - String containing ``<tokens>``
     - ``"variant_label": "r<realization_index>i<...>"``
     - A template. The dataset value must equal the template rendered from
       the other attributes. Templates are never used as defaults.
   * - List
     - ``"grid_label": ["gn", "gr", "gr1"]``
     - The value must match one item, first as a literal string and then as
       a regular expression (full match). POSIX classes such as
       ``[[:digit:]]`` are translated to Python syntax.
   * - Object
     - ``"activity_id": {"CMIP": "description"}``
     - The value must be one of the object's **keys**; the values are
       descriptions. For ``realm`` and ``source_type`` the dataset value is
       split on whitespace and every token must be a key.
   * - Anything else (``null``, empty)
     - ``"further_info_url": null``
     - Not constrained.

Lookup objects that inject attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If an object's entries are themselves objects whose values are all scalars,
selecting an entry also **adds those scalars as global attributes**. For
example, an obs4MIPs CV might define:

.. code-block:: json

   "site_id": {
     "AR-SLu": {"latitude": "-33.47", "longitude": "-66.46", "location": "San Luis"}
   }

Setting ``site_id="AR-SLu"`` then writes ``latitude``, ``longitude`` and
``location`` to the file. This injection runs last, so it never overrides an
attribute you supplied or one filled by the special sections below. It is
disabled for ``DRS``, ``experiment_id``, ``frequency``, ``institution_id``,
``license``, ``license_id``, ``source_id`` and ``source_type``.


Required attributes
-------------------

``required_global_attributes`` lists the attributes every dataset must have
after defaulting. CMOR can produce some of them on its own:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - How CMOR fills it when missing
   * - ``Conventions``
     - ``CF-1.12`` if the CV lists it, otherwise the CV's value.
   * - ``creation_date``
     - The current time, formatted ``YYYY-MM-DDTHH:MM:SSZ``.
   * - ``variant_label``
     - Assembled from ``realization_index``, ``initialization_index``,
       ``physics_index`` and ``forcing_index`` (see below).
   * - ``tracking_id``
     - A new UUID with the project prefix (see below).
   * - Others
     - The CV value, when the entry has exactly one possible value: a
       scalar, a one-item list, or an object with one key.

If a required attribute is still missing after defaulting, validation fails.

.. TODO: creation_date uses datetime.now() (local time) with a "Z" suffix;
   confirm whether this should be UTC before documenting it as such.


Sections with special meaning
-----------------------------

The keys below get extra behavior on top of the generic rules. They still
get the ordinary membership check from the table above: for example,
``source_id`` must still be one of the keys of the ``source_id`` object.

``source_id``
~~~~~~~~~~~~~

An object mapping each model identifier to a record of attributes that
model implies. When you select a ``source_id``:

* any record value that is a scalar or a one-item list becomes a default;
* any record attribute that is also present in your dataset must match the
  record value (or be one of its values, if it is a list).

``source`` is only checked when it is a required attribute. The
``source_id`` catalog is also used to validate ``parent_source_id``.

``experiment_id``
~~~~~~~~~~~~~~~~~

An object mapping each experiment to its record. Selecting an experiment:

* fills scalar and one-item-list values (such as ``experiment`` or
  ``activity_id``) as defaults, except for structural fields like
  ``start_year``, ``end_year``, ``tier``, ``parent_*`` and source-type
  fields;
* requires any other record attribute present in the dataset to match;
* constrains ``source_type`` (see below);
* determines the parent attributes (see :ref:`cv-parent-attributes`).

``institution_id``
~~~~~~~~~~~~~~~~~~

An object mapping each identifier to the institution's full name. CMOR
fills ``institution`` from it, and if ``institution`` is required and you
supplied a different value, validation fails.

``license``
~~~~~~~~~~~

Three forms are accepted:

* **A string** — the literal license text, used as the default and
  required to match.
* **A list** — patterns the license text must match.
* **An object** with ``license_id`` and ``license_template``:

  .. code-block:: json

     "license": {
       "license_id": {
         "CC-BY-4.0": {"license_type": "...", "license_url": "https://..."}
       },
       "license_template": "<license_id>; ... see <license_url>"
     }

  ``license_id`` must be a key of the catalog, ``license_type`` and
  ``license_url`` must match the selected entry, and CMOR renders
  ``license`` from the template if you don't supply it.

``source_type``
~~~~~~~~~~~~~~~

The project-wide vocabulary of model component types (a list or an object).
The dataset value is a space-separated list, and every token must be in the
vocabulary.

The selected experiment can narrow this further. Every pattern in its
``required_source_type`` (or ``required_model_components``) must be present,
and every token must be either required or listed in
``additional_allowed_model_components``.

.. TODO: confirm the required/optional key names used by each project CV.

``frequency``
~~~~~~~~~~~~~

An object whose keys are the allowed frequency codes. Entries may carry
``approx_interval``, ``approx_interval_warning`` and
``approx_interval_error``, which CMOR uses to check the spacing of time
coordinates. These values are never written as global attributes.

``forcing``
~~~~~~~~~~~

Used by CMIP6-style projects. The dataset value is a space- or
comma-separated list of abbreviations, optionally followed by a
parenthetical note:

.. code-block:: text

   GHG Oz SA Sl Vl BC OC (GHG = CO2, N2O, CH4, …)

Everything from the first ``(`` onward is ignored, and every remaining token
must be in the CV's ``forcing`` list. If the CV has no ``forcing`` entry
(CMIP7, obs4MIPs), the attribute is not checked.

``DRS``
~~~~~~~

Holds ``directory_path_template`` and ``filename_template``, the
``<token>`` templates CMOR uses to build output paths and file names. Other
keys in this section are ignored.

``tracking_id`` and ``tracking_id_prefix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each file gets a ``tracking_id`` of the form ``<prefix>/<uuid4>``. The
prefix comes from, in order of preference:

1. a ``tracking_prefix`` attribute you set on the dataset;
2. the CV's ``tracking_id_prefix`` (a string or one-item list);
3. the literal leading part of the CV's ``tracking_id`` pattern, for
   example ``hdl:21.14100`` from ``"^hdl:21.14100/.*$"``.

``variant_label`` and the RIPF indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each of ``realization_index``, ``initialization_index``, ``physics_index``
and ``forcing_index`` must be a positive integer, given either bare (``3``)
or with its letter (``r3``). When all four are present, CMOR assembles
``variant_label`` (``r3i1p1f1``).

The ``r<N>i<N>p<N>f<N>`` format is only enforced when the CV gives
``variant_label`` a list of allowed values or patterns. If the CV just
requires the attribute (as obs4MIPs does), any string is accepted.

``grid_label``
~~~~~~~~~~~~~~

If the CV defines ``grid_label``, the generic rules apply. If it doesn't,
CMOR still requires a label that starts with ``g``, ``c`` or ``r`` followed
by lowercase letters and digits (``gn``, ``gr1``, ``cn``).

.. _cv-parent-attributes:

Parent experiment attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The selected experiment's ``parent_experiment_id`` decides whether parent
metadata is allowed:

* If it is empty or ``"no parent"``, none of ``parent_experiment_id``,
  ``parent_activity_id``, ``parent_mip_era``, ``parent_source_id``,
  ``parent_time_units``, ``parent_variant_label``, ``branch_time_in_child``
  or ``branch_time_in_parent`` may be set.
* Otherwise all of them are required, and:

  * ``parent_experiment_id`` and ``parent_activity_id`` must match the
    experiment record;
  * ``parent_source_id`` must be in the ``source_id`` catalog;
  * ``parent_mip_era`` must equal ``mip_era``;
  * ``parent_time_units`` must look like ``days since YYYY-MM-DD``;
  * ``parent_variant_label`` must look like ``r1i1p1f1``;
  * both branch times must be numeric.


The order CMOR applies a CV
---------------------------

Defaults are applied in this order. Each step only fills attributes that are
still missing, so earlier steps win over later ones:

1. scalar CV values;
2. ``source_id`` record values;
3. ``institution`` from ``institution_id``;
4. ``experiment_id`` record values;
5. ``license`` rendered from its template;
6. runtime values (``Conventions``, ``creation_date``, ``variant_label``,
   ``tracking_id``) and single-valued required attributes;
7. attributes injected by lookup objects.

Validation then checks, stopping at the first error:

1. every attribute value against its CV entry;
2. that all required attributes are present;
3. ``source``, ``experiment`` and ``institution`` against their ``*_id``
   records;
4. the RIPF indices and ``variant_label``;
5. ``forcing`` tokens.

Experiment, source and parent consistency checks run separately.

.. TODO: state where validate_experiment / validate_source_attributes /
   validate_parent_attributes are called (DatasetWriter? write time?).


Common authoring mistakes
-------------------------

Double-nested entries
   Wrapping a value in an object keyed by its own name:

   .. code-block:: json

      "nominal_resolution": {"nominal_resolution": ["0.5 km", "1 km"]}

   instead of ``"nominal_resolution": ["0.5 km", "1 km"]``. CMOR emits a
   :class:`RuntimeWarning` on load, and
   :meth:`~cmor.ControlledVocabulary.validate_structure` reports it.

Legacy POSIX regular expressions
   CMIP6 CVs express some constraints as one-item lists of POSIX patterns
   (``"^[[:digit:]]\\{1,\\}$"``). CMOR skips the generic check for these,
   relies on dedicated checks instead, and doesn't treat them as required
   when absent. Prefer explicit lists or Python regular expressions in new
   CVs.

Templates used as defaults
   A value containing ``<`` or ``>`` is treated as a template and is never
   copied into a file as a default.


See also
--------

* :class:`cmor.ControlledVocabulary` — loading and checking a CV from Python.
* :class:`cmor.ControlledVocabularyError` — the exception raised on failure.
