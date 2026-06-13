# CMOR4 Documentation

This directory contains the Sphinx documentation for CMOR4.

## Prerequisites

Install the development dependencies including Sphinx, sphinx-autobuild, and the Furo theme:

```bash
pip install -e ".[dev]"
```

Or if you have an existing virtual environment:

```bash
source venv/bin/activate
pip install -e ".[dev]"
```

## Building the Documentation

### One-time Build

To build the HTML documentation:

```bash
cd docs
make html
```

The generated documentation will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser to view it.

### Clean Build

To remove all generated files and rebuild from scratch:

```bash
cd docs
make clean
make html
```

### Live Rebuild (Recommended for Development)

For automatic rebuilding while editing documentation:

```bash
cd docs
sphinx-autobuild source build/html
```

This will:
- Start a local web server (typically at http://127.0.0.1:8000)
- Watch for changes in the `source/` directory
- Automatically rebuild and refresh your browser when files change

Press `Ctrl+C` to stop the server.

## Documentation Structure

```
docs/
├── source/
│   ├── api/                    # API reference pages
│   │   ├── Axis.rst           # Individual class pages
│   │   ├── DatasetInfo.rst
│   │   ├── ProjectTables.rst
│   │   ├── Variable.rst
│   │   ├── ...
│   │   ├── functions.rst      # Public functions
│   │   ├── exceptions.rst     # Exception classes
│   │   └── classes.rst        # Classes index
│   ├── conf.py                # Sphinx configuration
│   ├── index.rst              # Documentation home page
│   └── _static/               # Static files (CSS, images)
└── build/                      # Generated documentation (gitignored)
    └── html/                   # HTML output
```

## Adding New Documentation

### Documenting a New Class

1. Create a new `.rst` file in `source/api/` (e.g., `NewClass.rst`):
   ```rst
   NewClass
   ========

   .. currentmodule:: cmor4

   .. autoclass:: NewClass
      :members:
      :undoc-members:
      :show-inheritance:
   ```

2. Add it to the table of contents in `source/api/classes.rst` and `source/index.rst`

### Documenting a New Function

Add the function to `source/api/functions.rst`:

```rst
.. autofunction:: new_function_name
```

## Tips

- **Type annotations**: All public functions and class methods should have type annotations. These are automatically extracted and displayed in the documentation.

- **Docstring format**: Use NumPy-style docstrings with sections like `Parameters`, `Returns`, `Raises`, and `Examples`.

- **Cross-references**: Link to other classes/functions using backticks with colons: `` :class:`DatasetInfo` `` or `` :func:`create_dataset` ``

- **Code blocks**: Use `::` followed by an indented block, or use `` ```python `` for syntax highlighting.

## Troubleshooting

### Module import errors

If Sphinx can't find the `cmor4` module, make sure:
1. The package is installed in editable mode: `pip install -e .`
2. Your virtual environment is activated
3. The `sys.path` modification in `conf.py` is correct

### Missing type annotations

If type hints aren't showing:
1. Verify functions have type annotations in the source code
2. Check `autodoc_typehints = 'both'` is set in `conf.py`
3. Try a clean rebuild: `make clean && make html`

### Theme not found

If you get a "theme 'furo' not found" error:
```bash
pip install furo
```
