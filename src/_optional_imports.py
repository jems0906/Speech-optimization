from __future__ import annotations

from importlib import import_module
from typing import Any


def optional_import(module_name: str) -> Any | None:
    try:
        return import_module(module_name)
    except ImportError:
        return None


def optional_attr_import(module_name: str, attr_name: str) -> Any | None:
    module = optional_import(module_name)
    if module is None:
        return None
    return getattr(module, attr_name, None)
