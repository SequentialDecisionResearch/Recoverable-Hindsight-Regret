from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any

TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "yfinance",
    "pytest",
]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: package_version(name) for name in TRACKED_PACKAGES},
    }
