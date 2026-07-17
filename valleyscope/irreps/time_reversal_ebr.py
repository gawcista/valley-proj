"""Reviewed time-reversal closure of unitary EBR source data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import product

import numpy as np

from valleyscope.irreps.ebr_data_adapter import load_ebr_source_data
from valleyscope.irreps.magnetic_groups import derive_type_ii_bns_number
from valleyscope.irreps.time_reversal_geometry import (
    centered_k_equivalent,
    normalize_centering_vectors,
)
from valleyscope.irreps.tables import (
    ReviewedSourceIrrep,
    StandardIrrep,
    StandardIrrepTable,
    load_magnetic_irrep_table,
    match_table_operations,
)


_TOL = 5e-5


def validate_grey_group_time_reversal_source(
    *,
    unitary_table: StandardIrrepTable,
    reviewed_rows: Sequence[ReviewedSourceIrrep],
    unitary_source_data: Mapping[str, object],
    irrep_partner_by_label: Mapping[str, str],
    centering_vectors: Sequence[Sequence[float]],
    grey_source_loader: (
        Callable[[int | str, bool], Mapping[str, object]] | None
    ) = None,
) -> dict[str, object]:
    """Validate unitary TR pairs against an explicit type-II grey table.

    The grey-group rows and direct grey EBR columns are authoritative.
    """
    blockers: list[str] = []
    unit_basis = unitary_source_data.get("source_basis_labels", [])
    if (
        not isinstance(unit_basis, Sequence)
        or isinstance(unit_basis, (str, bytes))
    ):
        return _blocked("malformed_unitary_ebr_source_data")
    unit_basis = [str(label) for label in unit_basis]
    if not unit_basis or len(set(unit_basis)) != len(unit_basis) or any(
        not label for label in unit_basis
    ):
        blockers.append("invalid_or_duplicate_time_reversal_ebr_source_basis")
    if (
        set(irrep_partner_by_label) != set(unit_basis)
        or set(irrep_partner_by_label.values()) != set(unit_basis)
        or any(
            not isinstance(label, str)
            or not label
            or not isinstance(partner, str)
            or not partner
            for label, partner in irrep_partner_by_label.items()
        )
    ):
        blockers.append(
            "incomplete_or_nonbijective_time_reversal_irrep_row_mapping"
        )
    if any(
        irrep_partner_by_label.get(
            irrep_partner_by_label.get(label, ""), ""
        ) != label
        for label in unit_basis
    ):
        blockers.append("non_involutive_time_reversal_irrep_row_mapping")
    try:
        bns_number = derive_type_ii_bns_number(unitary_table.number)
        grey_table = load_magnetic_irrep_table(
            bns_number,
            unitary_spacegroup_number=unitary_table.number,
            spinor=unitary_table.spinor,
        )
        loader = grey_source_loader or load_ebr_source_data
        grey_source = loader(bns_number, unitary_table.spinor)
    except Exception as exc:
        return _blocked(
            "grey_group_source_unavailable:"
            f"{type(exc).__name__}:{exc}",
            blockers=blockers,
        )

    operation_map = _unitary_to_grey_operation_map(
        unitary_table, grey_table
    )
    if operation_map is None:
        blockers.append("grey_group_unitary_operation_inventory_mismatch")

    grey_basis_raw = grey_source.get("source_basis_labels", [])
    grey_ebrs_raw = grey_source.get("source_ebrs", [])
    if (
        not isinstance(grey_basis_raw, Sequence)
        or isinstance(grey_basis_raw, (str, bytes))
        or not isinstance(grey_ebrs_raw, Sequence)
        or isinstance(grey_ebrs_raw, (str, bytes))
    ):
        blockers.append("malformed_grey_group_ebr_source_data")
        grey_basis: list[str] = []
        grey_ebrs: list[Mapping[str, object]] = []
    else:
        grey_basis = [str(label) for label in grey_basis_raw]
        grey_ebrs = [row for row in grey_ebrs_raw if isinstance(row, Mapping)]
        if (
            not grey_basis
            or len(set(grey_basis)) != len(grey_basis)
            or any(
                not isinstance(label, str) or not label
                for label in grey_basis_raw
            )
        ):
            blockers.append("invalid_or_duplicate_grey_group_ebr_source_basis")
        if not grey_ebrs or len(grey_ebrs) != len(grey_ebrs_raw):
            blockers.append("grey_group_ebr_source_columns_missing_or_malformed")
        grey_ebr_labels: set[str] = set()
        for index, grey_ebr in enumerate(grey_ebrs):
            label = grey_ebr.get("ebr_label")
            vector = grey_ebr.get("vector")
            if (
                not isinstance(label, str)
                or not label
                or label in grey_ebr_labels
            ):
                blockers.append(
                    f"invalid_or_duplicate_grey_group_ebr_label:{index}"
                )
            else:
                grey_ebr_labels.add(label)
            if (
                not isinstance(vector, Sequence)
                or isinstance(vector, (str, bytes))
                or len(vector) != len(grey_basis)
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in vector
                )
            ):
                blockers.append(
                    f"malformed_grey_group_ebr_vector:{label or index}"
                )

    grey_by_label = {row.label: row for row in grey_table.irreps}
    missing_grey = [label for label in grey_basis if label not in grey_by_label]
    if missing_grey:
        blockers.append(f"grey_source_rows_missing_from_table:{missing_grey}")

    centering = normalize_centering_vectors(centering_vectors)
    if centering is None:
        blockers.append("grey_source_centering_vectors_missing_or_malformed")

    reviewed = list(reviewed_rows)
    reviewed_by_label = {row.label: row for row in reviewed}
    if set(reviewed_by_label) != set(unit_basis):
        blockers.append("reviewed_rows_do_not_cover_complete_unitary_ebr_basis")

    restrictions: dict[str, dict[str, int]] = {}
    restriction_cases: dict[str, str] = {}
    if not blockers and operation_map is not None and centering is not None:
        for grey_label in grey_basis:
            grey_row = grey_by_label[grey_label]
            decompositions = _unitary_restriction_decompositions(
                grey_row=grey_row,
                reviewed_rows=reviewed,
                operation_map=operation_map,
                centering_vectors=centering,
            )
            if len(decompositions) != 1:
                blockers.append(
                    "ambiguous_or_missing_grey_unitary_restriction:"
                    f"{grey_label}:{decompositions}"
                )
                continue
            restrictions[grey_label] = decompositions[0]
            restriction_case = _restriction_case(
                decompositions[0], irrep_partner_by_label
            )
            if restriction_case is None:
                blockers.append(
                    "unsupported_grey_unitary_restriction_case:"
                    f"{grey_label}:{decompositions[0]}"
                )
            else:
                restriction_cases[grey_label] = restriction_case

    status = "validated" if not blockers else "blocked"
    return {
        "status": status,
        "grey_bns_number": bns_number,
        "grey_source_basis_labels": grey_basis,
        "grey_source_hsp_by_irrep": {
            label: grey_by_label[label].kpoint_label
            for label in grey_basis if label in grey_by_label
        },
        "unitary_source_hsp_by_irrep": {
            row.label: row.kpoint_label for row in reviewed
        },
        "grey_source_ebrs": [dict(row) for row in grey_ebrs],
        "grey_unitary_restriction_by_irrep": restrictions,
        "grey_unitary_restriction_case_by_irrep": restriction_cases,
        "blockers": blockers,
    }


def _unitary_to_grey_operation_map(
    unitary: StandardIrrepTable,
    grey: StandardIrrepTable,
) -> dict[int, int] | None:
    grey_unitary = StandardIrrepTable(
        number=grey.number,
        name=grey.name,
        spinor=grey.spinor,
        operations=tuple(
            operation for operation in grey.operations
            if not operation.time_reversal
        ),
        irreps=grey.irreps,
    )
    detected = [{
        "operation_id": operation.table_index,
        "rotation_frac": operation.rotation_frac,
        "translation_frac": operation.translation_frac,
    } for operation in unitary.operations]
    report = match_table_operations(
        detected_operations=detected,
        table=grey_unitary,
        tolerance=_TOL,
    )
    if (
        report.unmatched_operation_ids
        or len(report.mapping_by_operation_id) != len(unitary.operations)
        or set(report.mapping_by_operation_id.values())
        != {operation.table_index for operation in grey_unitary.operations}
    ):
        return None
    return {
        int(unit_index): int(grey_index)
        for unit_index, grey_index in report.mapping_by_operation_id.items()
    }


def _unitary_restriction_decompositions(
    *,
    grey_row: StandardIrrep,
    reviewed_rows: Sequence[ReviewedSourceIrrep],
    operation_map: Mapping[int, int],
    centering_vectors: Sequence[np.ndarray],
) -> list[dict[str, int]]:
    candidates = [
        row for row in reviewed_rows
        if centered_k_equivalent(
            grey_row.k_frac,
            row.k_frac,
            centering_vectors,
            tolerance=_TOL,
        )
    ]
    if not candidates:
        return []
    grey_chars_by_unit = {
        unit_index: grey_row.characters[grey_index]
        for unit_index, grey_index in operation_map.items()
        if grey_index in grey_row.characters
    }
    candidates = [
        row for row in candidates
        if set(row.operation_indices) == set(grey_chars_by_unit)
    ]
    if not candidates:
        return []

    max_multiplicities = [
        grey_row.dimension // max(row.dimension, 1) for row in candidates
    ]
    decompositions: list[dict[str, int]] = []
    for multiplicities in product(*(
        range(limit + 1) for limit in max_multiplicities
    )):
        if sum(
            multiplicity * row.dimension
            for multiplicity, row in zip(multiplicities, candidates)
        ) != grey_row.dimension:
            continue
        if any(
            abs(
                sum(
                    multiplicity * row.characters[operation_index]
                    for multiplicity, row in zip(multiplicities, candidates)
                ) - grey_character
            ) > _TOL
            for operation_index, grey_character in grey_chars_by_unit.items()
        ):
            continue
        decompositions.append({
            row.label: int(multiplicity)
            for multiplicity, row in zip(multiplicities, candidates)
            if multiplicity
        })
    return decompositions


def _restriction_case(
    restriction: Mapping[str, int],
    irrep_partner_by_label: Mapping[str, str],
) -> str | None:
    """Classify a character-derived restriction without parsing labels."""
    labels = list(restriction)
    if len(labels) == 1:
        label = labels[0]
        multiplicity = restriction[label]
        partner = irrep_partner_by_label.get(label)
        if partner == label and multiplicity == 1:
            return "real"
        if partner == label and multiplicity == 2:
            return "quaternionic"
        if partner != label and partner is not None and multiplicity == 1:
            return "exchanged_hsp_arm"
        return None
    if (
        len(labels) == 2
        and all(restriction[label] == 1 for label in labels)
        and irrep_partner_by_label.get(labels[0]) == labels[1]
        and irrep_partner_by_label.get(labels[1]) == labels[0]
    ):
        return "complex_paired"
    return None


def _blocked(
    reason: str,
    *,
    blockers: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "status": "blocked",
        "grey_bns_number": None,
        "grey_source_basis_labels": [],
        "grey_source_hsp_by_irrep": {},
        "unitary_source_hsp_by_irrep": {},
        "grey_source_ebrs": [],
        "grey_unitary_restriction_by_irrep": {},
        "grey_unitary_restriction_case_by_irrep": {},
        "blockers": [*blockers, reason],
    }
