"""Irrep-table adapters used by ValleyScope diagnostics."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "IrrepMatchResult",
    "OperationMappingReport",
    "StandardIrrep",
    "StandardIrrepTable",
    "StandardTableOperation",
    "decompose_characters_into_irreps",
    "load_standard_irrep_table",
    "match_table_operations",
]

_TABLE_EXPORTS = frozenset({
    "OperationMappingReport",
    "StandardIrrep",
    "StandardIrrepTable",
    "StandardTableOperation",
    "load_standard_irrep_table",
    "match_table_operations",
})
_MATCHING_EXPORTS = frozenset({
    "IrrepMatchResult",
    "decompose_characters_into_irreps",
})


def __getattr__(name: str):
    """Load optional table adapters only when their public names are used."""
    if name in _TABLE_EXPORTS:
        value = getattr(import_module("valleyscope.irreps.tables"), name)
    elif name in _MATCHING_EXPORTS:
        value = getattr(import_module("valleyscope.irreps.matching"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
