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


@dataclass(frozen=True, slots=True)
class SingleStateIrrepResult:
    state_index: int
    status: str
    table_kpoint_label: str
    computed_characters: dict[int, complex]
    irrep_weights: dict[str, float]
    irrep_multiplicities: dict[str, int]
    irrep_label: str | None
    missing_table_operation_indices: list[int]
    failure_reasons: list[str]


def match_single_state_irrep(
    *,
    table: StandardIrrepTable,
    table_kpoint_label: str,
    state_index: int,
    computed_characters: dict[int, complex],
    tolerance: float = 1e-5,
) -> SingleStateIrrepResult:
    """Match a single adapted state's characters to a one-dimensional irrep.

    Only emits a label when the characters uniquely select exactly one
    one-dimensional irrep with multiplicity 1.  All other cases return a
    conservative non-matched status.
    """
    decomposition = decompose_characters_into_irreps(
        table=table,
        table_kpoint_label=table_kpoint_label,
        computed_characters=computed_characters,
        tolerance=tolerance,
    )
    if decomposition.status != "matched":
        return SingleStateIrrepResult(
            state_index=state_index,
            status=decomposition.status,
            table_kpoint_label=table_kpoint_label,
            computed_characters=dict(computed_characters),
            irrep_weights=decomposition.irrep_weights,
            irrep_multiplicities={},
            irrep_label=None,
            missing_table_operation_indices=decomposition.missing_table_operation_indices,
            failure_reasons=decomposition.failure_reasons,
        )

    labels = [
        label
        for label, multiplicity in decomposition.irrep_multiplicities.items()
        if multiplicity > 0
    ]
    if len(labels) != 1 or decomposition.irrep_multiplicities.get(labels[0], 0) != 1:
        return SingleStateIrrepResult(
            state_index=state_index,
            status="ambiguous_irrep_label",
            table_kpoint_label=table_kpoint_label,
            computed_characters=dict(computed_characters),
            irrep_weights=decomposition.irrep_weights,
            irrep_multiplicities=decomposition.irrep_multiplicities,
            irrep_label=None,
            missing_table_operation_indices=[],
            failure_reasons=["Single-state characters do not select exactly one irrep with multiplicity 1"],
        )

    label = labels[0]
    irreps_by_label = {irrep.label: irrep for irrep in table.irreps_by_kpoint(table_kpoint_label)}
    matched_irrep = irreps_by_label.get(label)
    if matched_irrep is None or matched_irrep.dimension != 1:
        return SingleStateIrrepResult(
            state_index=state_index,
            status="not_one_dimensional",
            table_kpoint_label=table_kpoint_label,
            computed_characters=dict(computed_characters),
            irrep_weights=decomposition.irrep_weights,
            irrep_multiplicities=decomposition.irrep_multiplicities,
            irrep_label=None,
            missing_table_operation_indices=[],
            failure_reasons=["State-level irrep labels are only emitted for one-dimensional irreps"],
        )

    return SingleStateIrrepResult(
        state_index=state_index,
        status="matched",
        table_kpoint_label=table_kpoint_label,
        computed_characters=dict(computed_characters),
        irrep_weights=decomposition.irrep_weights,
        irrep_multiplicities=decomposition.irrep_multiplicities,
        irrep_label=label,
        missing_table_operation_indices=[],
        failure_reasons=[],
    )
