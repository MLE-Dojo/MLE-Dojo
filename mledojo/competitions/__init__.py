# mledojo/competitions/__init__.py

"""
Registry and accessors for MLE Dojo competitions.

Competitions are treated as data resources, not standard Python subpackages.
Use the functions provided here to list competitions and access their
metrics, preparation functions, and associated data files.
"""

from .registry import (
    get_metric,
    get_prepare
)

__all__ = [
    "get_metric",
    "get_prepare"
]

# Optional: Trigger initial discovery when the package is first imported.
# Generally lazy discovery (on first call) is fine.
# _discover_competitions()