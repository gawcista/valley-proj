from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from valleyscope.irreps.tables import StandardIrrepTable


@dataclass(frozen=True, slots=True)
class IrrepMatchResult:
    status: str
    table_kpoint_label: str
    computed_characters: dict[int, complex]
    irrep_weights: dict[str, float]
    irrep_multiplicities: dict[str, int]
    missing_table_operation_indices: list[int]
    failure_reasons: list[str]


def decompose_characters_into_irreps(
    *,
    table: StandardIrrepTable,
    table_kpoint_label: str,
    computed_characters: dict[int, complex],
    tolerance: float = 1e-5,
) -> IrrepMatchResult:
    table_operation_indices = table.operation_indices_for_kpoint(table_kpoint_label)
    missing_indices = [
        table_index
        for table_index in table_operation_indices
        if table_index not in computed_characters
    ]
    if missing_indices:
        return IrrepMatchResult(
            status="missing_characters",
            table_kpoint_label=table_kpoint_label,
            computed_characters=dict(computed_characters),
            irrep_weights={},
            irrep_multiplicities={},
            missing_table_operation_indices=missing_indices,
            failure_reasons=[
                f"Missing computed characters for table operations: {missing_indices}"
            ],
        )

    irreps = table.irreps_by_kpoint(table_kpoint_label)
    group_order = len(table_operation_indices)
    weights: dict[str, float] = {}
    multiplicities: dict[str, int] = {}
    failure_reasons: list[str] = []

    for irrep in irreps:
        weight = sum(
            np.conj(irrep.characters[table_index]) * computed_characters[table_index]
            for table_index in table_operation_indices
        ) / group_order
        real_weight = float(np.real(weight))
        weights[irrep.label] = real_weight
        rounded = int(round(real_weight))
        if abs(weight.imag) > tolerance or abs(real_weight - rounded) > tolerance:
            failure_reasons.append(
                f"Irrep {irrep.label} has non-integer character weight {weight}"
            )
            continue
        if rounded > 0:
            multiplicities[irrep.label] = rounded

    status = "matched" if not failure_reasons else "non_integer_weights"
    return IrrepMatchResult(
        status=status,
        table_kpoint_label=table_kpoint_label,
        computed_characters=dict(computed_characters),
        irrep_weights=weights,
        irrep_multiplicities=multiplicities if status == "matched" else {},
        missing_table_operation_indices=[],
        failure_reasons=failure_reasons,
    )
