"""Irrep-table adapters used by ValleyScope diagnostics."""

from valleyscope.irreps.tables import (
    OperationMappingReport,
    StandardIrrep,
    StandardIrrepTable,
    StandardTableOperation,
    load_standard_irrep_table,
    match_table_operations,
)
from valleyscope.irreps.matching import IrrepMatchResult, decompose_characters_into_irreps

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
