# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "CMOR4"
copyright = "2026, Chris Mauzey"
author = "Chris Mauzey"
release = "4.0.0a1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",  # Adds links to source code
    "sphinx.ext.intersphinx",  # Links to other project docs
]

templates_path = ["_templates"]
exclude_patterns = []

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Type hint settings
autodoc_typehints = (
    "description"  # Show type hints only in parameter descriptions, not signatures
)
autodoc_typehints_description_target = "all"  # Add types to all params
autodoc_typehints_format = "short"  # Use short form (e.g., list instead of typing.List)
python_use_unqualified_type_names = (
    True  # Use short names (e.g., Dataset instead of xarray.Dataset)
)
autodoc_type_aliases = {
    "StrSeq": "list[str] | tuple[str, ...] | None",
    "StrTuple": "tuple[str, ...] | None",
    "IntTuple": "tuple[int, ...] | None",
    "StrOrTuple": "str | tuple[str, ...] | None",
    "CoercedF": "float | None",
    "AxisStr": "str | None",
    "BoolCoerced": "bool | None",
    "PositiveLiteral": "Literal['up', 'down'] | None",
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "StrSeq": "list[str] | tuple[str, ...] | None",
    "StrTuple": "tuple[str, ...] | None",
    "IntTuple": "tuple[int, ...] | None",
    "StrOrTuple": "str | tuple[str, ...] | None",
    "CoercedF": "float | None",
    "AxisStr": "str | None",
    "BoolCoerced": "bool | None",
    "PositiveLiteral": "Literal['up', 'down'] | None",
}
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
