# Contributing to CMOR

## Development setup

Install CMOR in editable mode with the development and test dependencies, and fetch all project table submodules (the tests use them):

```bash
git clone https://github.com/mauzey1/cmor4.git
cd cmor4
pip install -e ".[dev,test]"
git submodule update --init --recursive
```

For a Conda-based setup, run the following from the repository root instead of
the `pip install` command above:

```bash
conda env create -f environment.yaml
conda activate cmor-dev
```

This environment includes the native UDUNITS and NetCDF libraries through
Conda's `cf-units` and `netcdf4` packages, then installs CMOR and its development
and test extras in editable mode.

## Testing

To install only the test dependencies alongside CMOR, use `pip install -e ".[test]"`.
Run the test suite with coverage from the repository root:

```bash
python -m pytest --cov --cov-report=term-missing --cov-report=xml
```

Coverage is configured in `.coveragerc` to measure only the `cmor` package,
including branch coverage. CI runs this command on each supported Python/OS
combination and uploads `coverage.xml` as a workflow artifact. For a local HTML
report, add `--cov-report=html` and open `htmlcov/index.html`.

## Code style

Check code style (these checks run in CI):

```bash
ruff check cmor tests
ruff format cmor tests --check
```

Fix lint problems and format the code:

```bash
ruff check cmor tests --fix
ruff format cmor tests
```

## Documentation

See [docs/README.md](./docs/README.md) for how to build and edit the Sphinx documentation.

The `docs.yaml` workflow builds documentation independently of the `tests.yaml`
workflow. Pull requests build the documentation; pushes to `main` also deploy
it to GitHub Pages.

## Reporting issues

Report bugs and request features via [GitHub Issues](https://github.com/mauzey1/cmor4/issues).
