DatasetInfo
===============

``DatasetInfo`` is a typed Pydantic model rather than a mapping. Access declared
fields as attributes (for example, ``info.source_id``) and use
``info.to_dict()`` when a dictionary is required by an external API. Project-
specific global attributes are included in that serialized dictionary.

.. currentmodule:: cmor4

.. autoclass:: DatasetInfo
   :members:
   :undoc-members:
   :show-inheritance:
