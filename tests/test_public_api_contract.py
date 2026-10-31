"""Freeze the public surface before reorganizing implementation modules."""

import importlib
import inspect
import json
from pathlib import Path

import cmor

MODULES = (
    "axis",
    "cv",
    "dataset",
    "datasetinfo",
    "exceptions",
    "grid",
    "tables",
    "variable",
    "writer",
    "zfactor",
)


def _signature(value):
    try:
        parameters = inspect.signature(value).parameters.values()
    except ValueError:
        return None
    return [[p.name, p.kind.name, repr(p.default)] for p in parameters]


def public_contract():
    result = {"exports": cmor.__all__, "modules": {}}
    for name in MODULES:
        module = importlib.import_module(f"cmor.{name}")
        members = {}
        for key, value in vars(module).items():
            if (
                key.startswith("_")
                or getattr(value, "__module__", None) != module.__name__
            ):
                continue
            if not (inspect.isclass(value) or inspect.isfunction(value)):
                continue
            entry = {"signature": _signature(value)}
            if inspect.isclass(value):
                methods = {}
                properties = []
                for base in reversed(value.__mro__):
                    if not base.__module__.startswith("cmor"):
                        continue
                    for attr, member in vars(base).items():
                        if attr.startswith("_"):
                            continue
                        if isinstance(member, property):
                            properties.append(attr)
                        elif callable(member) or isinstance(
                            member, (classmethod, staticmethod)
                        ):
                            methods[attr] = _signature(getattr(value, attr))
                entry.update(methods=methods, properties=sorted(set(properties)))
            members[key] = entry
        result["modules"][module.__name__] = members
    return result


def test_public_api_contract():
    expected = json.loads(
        Path(__file__).with_name("public_api_contract.json").read_text()
    )
    assert public_contract() == expected
    for name in cmor.__all__:
        assert getattr(cmor, name) is getattr(
            importlib.import_module(getattr(cmor, name).__module__), name
        )
