.. _controlled-vocabularies:

Controlled Vocabularies
=======================

A *controlled vocabulary* (CV) is the project-level JSON file that defines
which global attributes a dataset must carry, which values those attributes
may take, and which attributes CMOR can fill in for you. Every project
(CMIP7, obs4MIPs, DRCDP, …) ships its own CV alongside its MIP tables.

CMOR applies CV defaults when it prepares dataset metadata, then validates
controlled values. It checks required global attributes when it prepares a
variable for writing and again before writing the final dataset:

1. **Defaulting** — attributes you did not supply are filled from the CV
   (for example, ``institution`` from ``institution_id``).
2. **Validation** — attributes with CV constraints, required attributes and
   relationships between attributes are checked. A failed CV check raises
   :class:`~cmor.ControlledVocabularyError`. Attributes unknown to the CV
   are generally allowed.

Defaults preserve nonempty values you supply. Source, institution and
experiment defaults treat ``None`` and ``""`` as missing and replace them.
Supplying a value does not guarantee that it is constrained: the exceptions
in :ref:`cv-special-sections` and :ref:`cv-legacy-posix-regex` describe
checks CMOR skips or relaxes.


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
just the body of ``CV``. Most keys inside ``CV`` name NetCDF global
attributes. ``DRS`` and ``required_global_attributes`` control CMOR behavior.
``tracking_id_prefix`` also controls generated IDs; if its CV value is a
string, CMOR currently copies it into the global attributes as well.


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
     - A template. The dataset value may equal either the rendered template
       or the literal, unrendered template. An ordinary template is not
       defaulted, but one listed as required can be copied literally as a
       single-valued required default.
   * - List
     - ``"grid_label": ["gn", "gr", "gr1"]``
     - The value must match one item, first as a literal string and then as
       a regular expression (full match). Some POSIX classes are translated
       to Python syntax. The generic check is skipped for certain one-item
       legacy POSIX regex lists (see :ref:`cv-legacy-posix-regex`).
   * - Object
     - ``"activity_id": {"CMIP": "description"}``
     - The value must be one of the object's **keys**; the values are
       descriptions. For ``realm`` and ``source_type`` the dataset value is
       split on whitespace and every token must be a key.
   * - ``null`` or another unsupported shape
     - ``"further_info_url": null``
     - Not constrained.

Under the ordinary rules, an empty list or object rejects a nonempty
supplied value. Dedicated sections can behave differently. An absent
attribute is checked only if another rule requires it.

Lookup objects that inject attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If an object's entries are themselves objects whose values are all strings
or numbers, selecting an entry also **adds those scalars as global
attributes**. For example, an obs4MIPs CV might define:

.. code-block:: json

   "site_id": {
     "AR-SLu": {"latitude": "-33.47", "longitude": "-66.46", "location": "San Luis"}
   }

Setting ``site_id="AR-SLu"`` then writes ``latitude``, ``longitude`` and
``location`` to the file. This injection runs last, so it never overrides an
attribute you supplied or one filled by :ref:`cv-special-sections`. It is
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
     - ``CF-1.12`` if the CV lists it; otherwise a scalar CV value or the
       last value in a CV list. A regex-like final list value yields
       ``CF-1.7 CMIP-6.2`` instead. With no usable CV value, CMOR uses
       ``CF-1.12``.
   * - ``creation_date``
     - The current local time, formatted ``YYYY-MM-DDTHH:MM:SSZ``. The
       current implementation appends ``Z`` without converting to UTC.
   * - ``variant_label``
     - Assembled from ``realization_index``, ``initialization_index``,
       ``physics_index`` and ``forcing_index`` (see :ref:`cv-variant-label`).
   * - ``tracking_id``
     - A new UUID, with a prefix when one is available (see
       :ref:`cv-tracking-id`).
   * - Others
     - The CV value, when the entry has exactly one possible value: a
       scalar, a one-item list, or an object with one key. Regex-like
       one-item lists do not provide a default. A required scalar template
       can be copied literally.

If a required attribute is still missing when CMOR prepares a variable or
validates final global attributes, validation fails. One exception is a
required attribute defined by a one-item legacy POSIX regex list: CMOR
currently allows it to remain absent.


.. _cv-special-sections:

Sections with special meaning
-----------------------------

The following keys get extra behavior. Many also receive the ordinary check
from the table above: for example, ``source_id`` must be a key of its catalog.
Some dedicated checks replace or relax the ordinary check.

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
  ``activity_id``) as defaults. ``start_year``, ``end_year``, ``tier``,
  ``parent_activity_id``, ``parent_experiment_id``, ``required_source_type``,
  ``source_type`` and ``additional_allowed_model_components`` are excluded.
  A one-item ``required_model_components`` list is currently *not* excluded
  and may become a global attribute;
* checks other present record attributes against the selected experiment,
  except ``description`` and fields handled separately;
* constrains ``source_type`` (see :ref:`cv-source-type`);
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

  A supplied ``license_id`` must be a key of the catalog. Supplied
  ``license_type`` and ``license_url`` must match the selected entry. CMOR
  renders ``license`` from the template if you don't supply it, but
  currently accepts any explicitly supplied ``license`` text in this form.

.. _cv-source-type:

``source_type``
~~~~~~~~~~~~~~~

The project-wide vocabulary of model component types (a list or an object).
The dataset value is a space-separated list, and every token must be in the
vocabulary.

The selected experiment can narrow this further. CMOR uses a nonempty
``required_source_type`` if available, otherwise ``required_model_components``.
It searches the full ``source_type`` string for each required pattern, then
checks each token against the required patterns and
``additional_allowed_model_components``. These are regular-expression
searches, so an unanchored pattern can match a substring: required pattern
``A`` can match ``BAD``.

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

.. _cv-tracking-id:

``tracking_id`` and ``tracking_id_prefix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``tracking_id`` is required by the CV and you omit it, CMOR generates a
UUID. With a prefix, its form is ``<prefix>/<uuid4>``; without one, it is
just the UUID. An explicitly supplied ``tracking_id`` is preserved and is
checked only when the CV has an applicable value constraint. The prefix for
a generated ID comes from, in order of preference:

1. a ``tracking_prefix`` attribute you set on the dataset;
2. the CV's ``tracking_id_prefix`` (a string or one-item list);
3. the literal leading part of the CV's ``tracking_id`` pattern, for
   example ``hdl:21.14100`` from ``"^hdl:21.14100/.*$"``.

If ``tracking_id`` is not required and you do not supply one, CMOR does not
generate it. A string-valued CV ``tracking_id_prefix`` currently also
appears as a global attribute; a one-item list does not.

.. _cv-variant-label:

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

Defaults are applied in this order. Later steps generally fill absent keys;
source, institution and experiment defaults also replace ``None`` or ``""``:

1. scalar CV values;
2. ``source_id`` record values;
3. ``institution`` from ``institution_id``;
4. ``experiment_id`` record values;
5. ``license`` rendered from its template;
6. runtime values (``Conventions``, ``creation_date``, ``variant_label``,
   ``tracking_id``) and single-valued required attributes;
7. attributes injected by lookup objects.

``ProjectTables.dataset_info`` checks controlled values, RIPF indices,
forcing, derived names, and source, experiment and parent consistency. It
does not check that every required global attribute is present at this stage.
When CMOR prepares a variable for writing and validates final global
attributes, it checks the following in order, stopping at the first error:

1. supplied values with applicable CV constraints;
2. required attributes, except the legacy POSIX regex case above;
3. ``source``, ``experiment`` and ``institution`` against their ``*_id``
   records;
4. the RIPF indices and ``variant_label``;
5. ``forcing`` tokens;
6. source, experiment and parent consistency.


Common authoring mistakes
-------------------------

Double-nested entries
   Wrapping a value in an object keyed by its own name:

   .. code-block:: json

      "nominal_resolution": {"nominal_resolution": ["0.5 km", "1 km"]}

   instead of ``"nominal_resolution": ["0.5 km", "1 km"]``. CMOR emits a
   :class:`RuntimeWarning` on load describing the malformed entry.

.. _cv-legacy-posix-regex:

Legacy POSIX regular expressions
   CMIP6 CVs express some constraints as one-item lists of POSIX patterns
   (``"^[[:digit:]]\\{1,\\}$"``). CMOR skips the generic check for these,
   applies any separate checks available for the attribute, and doesn't
   treat them as required when absent. Some attributes have no separate
   check, so even an invalid supplied value may pass. Prefer explicit lists
   or Python regular expressions in new CVs.

Templates used as defaults
   A scalar CV value containing ``<`` or ``>`` is excluded from ordinary
   scalar defaults. If that attribute is required, the single-valued
   required-default step can still copy the template literally. Supply the
   intended rendered value explicitly or check the generated attribute.


See also
--------

* :class:`cmor.ControlledVocabulary` — loading and checking a CV from Python.
* :class:`cmor.ControlledVocabularyError` — the exception raised on failure.
