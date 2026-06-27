"""Pytest configuration for CMOR4 tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--run-compliance",
        action="store_true",
        default=False,
        help="Run CMIP7 compliance checker tests (requires compliance-checker and cc-plugin-wcrp)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "compliance: mark test as a CMIP7 compliance check (use --run-compliance to enable)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip compliance tests unless --run-compliance is specified."""
    if config.getoption("--run-compliance"):
        # --run-compliance given in cli: do not skip compliance tests
        return

    skip_compliance = pytest.mark.skip(reason="need --run-compliance option to run")
    for item in items:
        if "compliance" in item.keywords:
            item.add_marker(skip_compliance)
