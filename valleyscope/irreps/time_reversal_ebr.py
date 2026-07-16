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


def derive_time_reversal_ebr_column_pairing(
    *,
    source_basis_labels: Sequence[str],
    source_ebrs: Sequence[Mapping[str, object]],
    irrep_partner_by_label: Mapping[str, str],
) -> dict[str, object]:
    """Apply the full row involution to every source EBR vector."""
    basis = [str(label) for label in source_basis_labels]
    blockers: list[str] = []
    if not basis or len(set(basis)) != len(basis):
        blockers.append("invalid_or_duplicate_time_reversal_ebr_source_basis")
    if (
        set(irrep_partner_by_label) != set(basis)
        or set(irrep_partner_by_label.values()) != set(basis)
    ):
        blockers.append("incomplete_time_reversal_irrep_row_mapping")
    if any(
        irrep_partner_by_label.get(
            irrep_partner_by_label.get(label, ""), ""
        ) != label
        for label in basis
    ):
        blockers.append("non_involutive_time_reversal_irrep_row_mapping")
    if not source_ebrs:
        blockers.append("time_reversal_ebr_source_columns_missing")

    labels: list[str] = []
    vectors: list[tuple[int, ...]] = []
    for index, raw in enumerate(source_ebrs):
        label = raw.get("ebr_label") if isinstance(raw, Mapping) else None
        vector = raw.get("vector") if isinstance(raw, Mapping) else None
        if not isinstance(label, str) or not label or label in labels:
            blockers.append(f"invalid_or_duplicate_source_ebr_label:{index}")
            continue
        if (
            not isinstance(vector, Sequence)
            or isinstance(vector, (str, bytes))
            or len(vector) != len(basis)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in vector
            )
        ):
            blockers.append(f"malformed_complete_source_ebr_vector:{label}")
            continue
        labels.append(label)
        vectors.append(tuple(int(value) for value in vector))

    basis_index = {label: index for index, label in enumerate(basis)}
    transformed: dict[str, list[int]] = {}
    candidates_by_label: dict[str, list[str]] = {}
    if not blockers:
        for label, vector in zip(labels, vectors):
            tr_vector = tuple(
                vector[basis_index[irrep_partner_by_label[row_label]]]
                for row_label in basis
            )
            transformed[label] = list(tr_vector)
            candidates = [
                candidate_label
                for candidate_label, candidate_vector in zip(labels, vectors)
                if candidate_vector == tr_vector
            ]
            candidates_by_label[label] = candidates
            if not candidates:
                blockers.append(f"missing_time_reversal_ebr_partner:{label}")

    ambiguous = [
        label for label, candidates in candidates_by_label.items()
        if len(candidates) > 1
    ]
    status = "blocked" if blockers else (
        "ambiguous" if ambiguous else "validated"
    )
    return {
        "status": status,
        "complete_source_basis_labels": basis,
        "partner_candidates_by_ebr_label": candidates_by_label,
        "time_reversed_vectors_by_ebr_label": transformed,
        "ambiguous_ebr_labels": ambiguous,
        "blockers": blockers,
    }


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

    The grey-group rows prove antiunitary corepresentation closure.  EBR
    columns are related only after comparison on the complete source basis.
    """
    blockers: list[str] = []
    unit_basis = unitary_source_data.get("source_basis_labels", [])
    unit_ebrs = unitary_source_data.get("source_ebrs", [])
    if (
        not isinstance(unit_basis, Sequence)
        or isinstance(unit_basis, (str, bytes))
        or not isinstance(unit_ebrs, Sequence)
        or isinstance(unit_ebrs, (str, bytes))
    ):
        return _blocked("malformed_unitary_ebr_source_data")
    unit_basis = [str(label) for label in unit_basis]
    pairing = derive_time_reversal_ebr_column_pairing(
        source_basis_labels=unit_basis,
        source_ebrs=[row for row in unit_ebrs if isinstance(row, Mapping)],
        irrep_partner_by_label=irrep_partner_by_label,
    )
    if pairing["status"] == "blocked":
        blockers.extend(str(item) for item in pairing["blockers"])

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

    pair_candidates_by_unit = pairing.get(
        "partner_candidates_by_ebr_label", {}
    )
    unit_labels = [
        str(row.get("ebr_label", "")) for row in unit_ebrs
        if isinstance(row, Mapping)
    ]
    unit_vectors = [
        tuple(int(value) for value in row.get("vector", []))
        for row in unit_ebrs if isinstance(row, Mapping)
    ]
    unit_index = {label: index for index, label in enumerate(unit_labels)}
    unitary_column_orbits: list[tuple[int, int]] = []
    if isinstance(pair_candidates_by_unit, Mapping):
        for left_label, raw_partners in pair_candidates_by_unit.items():
            if left_label not in unit_index or not isinstance(
                raw_partners, Sequence
            ):
                continue
            for right_label in raw_partners:
                if right_label not in unit_index:
                    continue
                pair = tuple(sorted((
                    unit_index[left_label], unit_index[right_label]
                )))
                if pair not in unitary_column_orbits:
                    unitary_column_orbits.append(pair)
    unitary_column_orbits.sort()
    grey_candidates: dict[str, list[tuple[int, int]]] = {}
    grey_unique: dict[str, tuple[int, int]] = {}
    if not blockers:
        for grey_ebr in grey_ebrs:
            label = str(grey_ebr.get("ebr_label", ""))
            vector = grey_ebr.get("vector", [])
            expanded = _expand_grey_vector(
                vector=vector,
                grey_basis=grey_basis,
                restrictions=restrictions,
                unit_basis=unit_basis,
            )
            candidates: list[tuple[int, int]] = []
            for left_label, raw_partners in (
                pair_candidates_by_unit.items()
                if isinstance(pair_candidates_by_unit, Mapping) else []
            ):
                if left_label not in unit_index or not isinstance(
                    raw_partners, Sequence
                ):
                    continue
                left_index = unit_index[left_label]
                for right_label in raw_partners:
                    if right_label not in unit_index:
                        continue
                    right_index = unit_index[right_label]
                    pair = tuple(sorted((left_index, right_index)))
                    if pair in candidates:
                        continue
                    summed = tuple(
                        unit_vectors[pair[0]][index]
                        + unit_vectors[pair[1]][index]
                        for index in range(len(unit_basis))
                    )
                    if summed == expanded:
                        candidates.append(pair)
            grey_candidates[label] = candidates
            if len(candidates) == 1:
                grey_unique[label] = candidates[0]
            else:
                blockers.append(
                    "ambiguous_or_missing_grey_ebr_unitary_columns:"
                    f"{label}:{candidates}"
                )

    expected_orbits = set(unitary_column_orbits)
    covered_orbits = set(grey_unique.values())
    if expected_orbits != covered_orbits:
        blockers.append(
            "incomplete_grey_ebr_unitary_orbit_coverage:"
            f"missing={sorted(expected_orbits - covered_orbits)}:"
            f"unexpected={sorted(covered_orbits - expected_orbits)}"
        )

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
        "unitary_ebr_partner_candidates": pairing.get(
            "partner_candidates_by_ebr_label", {}
        ),
        "unitary_ebr_column_orbits": unitary_column_orbits,
        "grey_ebr_unitary_column_candidate_sets": grey_candidates,
        "grey_ebr_unitary_column_candidates": grey_unique,
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


def _expand_grey_vector(
    *,
    vector: object,
    grey_basis: Sequence[str],
    restrictions: Mapping[str, Mapping[str, int]],
    unit_basis: Sequence[str],
) -> tuple[int, ...]:
    if (
        not isinstance(vector, Sequence)
        or isinstance(vector, (str, bytes))
        or len(vector) != len(grey_basis)
    ):
        return ()
    index = {label: position for position, label in enumerate(unit_basis)}
    out = [0] * len(unit_basis)
    for coefficient, grey_label in zip(vector, grey_basis):
        for unit_label, multiplicity in restrictions[grey_label].items():
            out[index[unit_label]] += int(coefficient) * int(multiplicity)
    return tuple(out)


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
        "unitary_ebr_partner_candidates": {},
        "unitary_ebr_column_orbits": [],
        "grey_ebr_unitary_column_candidate_sets": {},
        "grey_ebr_unitary_column_candidates": {},
        "blockers": [*blockers, reason],
    }
