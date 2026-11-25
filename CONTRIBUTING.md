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
including branch coverage. The run fails if total branch coverage is below the
`fail_under` floor in `.coveragerc` (currently 90%). Raise the floor as coverage
improves; don't lower it to make a change pass. CI runs this command on each
supported Python/OS combination and uploads `coverage.xml` as a workflow
artifact. For a local HTML report, add `--cov-report=html` and open
`htmlcov/index.html`.

## CI checks

The `tests.yaml` workflow runs on pushes and pull requests to `main` and
`develop`. Run the same checks locally before opening a pull request:

```bash
ruff check cmor tests                      # lint, including complexity (C90)
ruff format cmor tests --check             # formatting
mypy --explicit-package-bases cmor         # type annotations
python -m pytest --cov --cov-report=term-missing   # tests and coverage floor
for f in examples/example_*.py; do                  # examples, warning-free
  python -W error::RuntimeWarning "$f" --output-dir /tmp/cmor-examples || break
done
```

The `lint` job runs the two ruff commands. Each job in the build matrix runs mypy
and the test suite with coverage. The job fails if coverage is below the floor.
The `examples` job runs every script in `examples/` with `RuntimeWarning` treated
as an error.

The test suite includes these checks:

- **Saved notebook outputs** (`tests/test_notebook_outputs.py`) must contain no
  errors, no stderr output and no warnings. The notebooks are not run in CI.
  After you edit a notebook, rebuild its outputs, for example
  `tools/rebuild_notebook_output.sh notebooks/07_non_lat_lon_sea_ice_concentration.ipynb`,
  and commit the result.
- **Golden snapshots** (`tests/test_golden_datasets.py`) compare complete
  `cmorize()` output with the files in `tests/golden/`. If your change is meant
  to alter output, regenerate them with
  `CMOR_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_datasets.py` and
  explain the diff in the pull request.
- **Known bugs** (`tests/test_known_issues.py`) are strict `xfail` tests. A
  fix makes its test pass, and strict mode then reports an XPASS failure.
  Remove the `xfail` marker in the same pull request.

## Code style

Fix lint problems and format the code:

```bash
ruff check cmor tests --fix
ruff format cmor tests
```

### Complexity

`ruff.toml` enables the mccabe check (`C90`) with a limit of 15. A small number
of existing functions are above the limit. Each one has a
`# noqa: C901 (see ruff.toml)` comment on its `def` line, and `ruff.toml` lists
them. Don't add new exemptions. When you refactor one of these functions below
the limit, remove its noqa comment and its line in `ruff.toml`.

To compare complexity before and after a change, run:

```bash
tools/complexity_report.sh            # the whole cmor package
tools/complexity_report.sh cmor/_assembly.py cmor/_cv_models.py
```

It prints radon's grade C-or-worse blocks and the maintainability index for each
module. `radon` is in the `dev` extra.

## Review checklist

For pull requests that touch `cmor/_assembly.py`, the `_cv` modules
(`cmor/_cv_models.py`, `cmor/cv.py`), `cmor/_netcdf.py`, `cmor/writer.py` or
`cmor/tables.py`, check that:

- no new function has a complexity above 10;
- no naming rule (for example, the `_bnds` suffix) or regex is duplicated where
  a shared helper already exists or should exist;
- each new branch has a test;
- a behavior change updates its tests and says so in the pull request
  description. Error and warning text counts as behavior.

## Documentation

See [docs/README.md](./docs/README.md) for how to build and edit the Sphinx documentation.

The `docs.yaml` workflow builds documentation independently of the `tests.yaml`
workflow. Pull requests and pushes to `main` and `develop` build the
documentation; only pushes to `main` deploy it to GitHub Pages.

## Reporting issues

Report bugs and request features via [GitHub Issues](https://github.com/mauzey1/cmor4/issues).
