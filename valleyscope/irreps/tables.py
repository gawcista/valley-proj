from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from irreptables import IrrepTable


@dataclass(frozen=True, slots=True)
class StandardTableOperation:
    table_index: int
    rotation_frac: np.ndarray
    translation_frac: np.ndarray
    spin_rotation: np.ndarray
    time_reversal: bool


@dataclass(frozen=True, slots=True)
class StandardIrrep:
    label: str
    kpoint_label: str
    k_frac: np.ndarray
    dimension: int
    characters: dict[int, complex]


@dataclass(frozen=True, slots=True)
class StandardIrrepTable:
    number: int
    name: str
    spinor: bool
    operations: tuple[StandardTableOperation, ...]
    irreps: tuple[StandardIrrep, ...]

    def operation_by_index(self, table_index: int) -> StandardTableOperation:
        for operation in self.operations:
            if operation.table_index == table_index:
                return operation
        raise KeyError(f"Unknown table operation index: {table_index}")

    def irreps_by_kpoint(self, kpoint_label: str) -> list[StandardIrrep]:
        return [irrep for irrep in self.irreps if irrep.kpoint_label == kpoint_label]

    def operation_indices_for_kpoint(self, kpoint_label: str) -> list[int]:
        indices: set[int] = set()
        for irrep in self.irreps_by_kpoint(kpoint_label):
            indices.update(irrep.characters)
        return sorted(indices)


@dataclass(frozen=True, slots=True)
class OperationMappingReport:
    status: str
    mapping_by_operation_id: dict[Any, int]
    unmatched_operation_ids: list[Any]
    unused_table_operation_indices: list[int]


def load_standard_irrep_table(spacegroup_number: int, *, spinor: bool) -> StandardIrrepTable:
    raw_table = IrrepTable(spacegroup_number, spinor=spinor)
    operations = tuple(
        StandardTableOperation(
            table_index=index,
            rotation_frac=np.asarray(symop.R, dtype=int),
            translation_frac=np.asarray(symop.t, dtype=float),
            spin_rotation=np.asarray(symop.S, dtype=complex),
            time_reversal=bool(symop.time_reversal),
        )
        for index, symop in enumerate(raw_table.symmetries, start=1)
    )
    irreps = tuple(
        StandardIrrep(
            label=str(irrep.name),
            kpoint_label=str(irrep.kpname),
            k_frac=np.asarray(irrep.k, dtype=float),
            dimension=int(irrep.dim),
            characters={
                int(index): complex(value)
                for index, value in irrep.characters.items()
            },
        )
        for irrep in raw_table.irreps
    )
    return StandardIrrepTable(
        number=int(raw_table.number),
        name=str(raw_table.name).strip(),
        spinor=bool(raw_table.spinor),
        operations=operations,
        irreps=irreps,
    )


def match_table_operations(
    detected_operations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    table: StandardIrrepTable,
    *,
    tolerance: float = 1e-8,
) -> OperationMappingReport:
    mapping: dict[Any, int] = {}
    unmatched: list[Any] = []
    used_table_indices: set[int] = set()

    for operation in detected_operations:
        operation_id = operation.get("operation_id")
        table_index = _match_one_operation(operation, table, tolerance=tolerance)
        if table_index is None:
            unmatched.append(operation_id)
            continue
        mapping[operation_id] = table_index
        used_table_indices.add(table_index)

    unused_table_indices = [
        operation.table_index
        for operation in table.operations
        if operation.table_index not in used_table_indices
    ]
    status = "complete" if not unmatched and not unused_table_indices else "incomplete"
    return OperationMappingReport(
        status=status,
        mapping_by_operation_id=mapping,
        unmatched_operation_ids=unmatched,
        unused_table_operation_indices=unused_table_indices,
    )


def _match_one_operation(
    operation: dict[str, Any],
    table: StandardIrrepTable,
    *,
    tolerance: float,
) -> int | None:
    rotation = np.rint(
        np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
    ).astype(int)
    translation = np.asarray(operation.get("translation_frac", np.zeros(3)), dtype=float)
    for table_operation in table.operations:
        if not np.array_equal(rotation, table_operation.rotation_frac):
            continue
        if _translation_matches(translation, table_operation.translation_frac, tolerance):
            return table_operation.table_index
    return None


def _translation_matches(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    delta_mod_lattice = delta - np.rint(delta)
    return bool(np.linalg.norm(delta_mod_lattice) <= tolerance)
