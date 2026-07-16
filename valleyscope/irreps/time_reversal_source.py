"""Reviewed unitary source-irrep time-reversal orbits.

The implementation derives ``k -> -k`` and character conjugation from the
reviewed source rows.  Labels are opaque identifiers and are never parsed to
infer HSP or irrep partners.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from valleyscope.irreps.time_reversal_geometry import (
    centered_k_equivalent,
    normalize_centering_vectors,
)
from valleyscope.irreps.tables import ReviewedSourceIrrep


_TOL = 5e-5


def derive_time_reversal_source_irrep_orbits(
    *,
    reviewed_rows: Sequence[ReviewedSourceIrrep],
    centering_vectors: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Derive deterministic HSP and irrep involutions from reviewed rows."""
    rows = list(reviewed_rows)
    blockers: list[str] = []
    if not rows or not all(isinstance(row, ReviewedSourceIrrep) for row in rows):
        return _blocked("reviewed_source_irrep_rows_missing_or_malformed")
    labels = [row.label for row in rows]
    if len(set(labels)) != len(labels):
        return _blocked("duplicate_reviewed_source_irrep_label")
    inventory_ids = {row.operation_inventory_identity for row in rows}
    spin_conventions = {row.spin_convention for row in rows}
    if len(inventory_ids) != 1:
        blockers.append("reviewed_source_operation_inventory_mismatch")
    if len(spin_conventions) != 1:
        blockers.append("reviewed_source_spin_convention_mismatch")

    centering = normalize_centering_vectors(centering_vectors)
    if centering is None:
        blockers.append("time_reversal_centering_vectors_missing_or_malformed")
        return _blocked_many(blockers)

    hsp_rows: dict[str, list[ReviewedSourceIrrep]] = {}
    hsp_order: list[str] = []
    for row in rows:
        if row.kpoint_label not in hsp_rows:
            hsp_order.append(row.kpoint_label)
            hsp_rows[row.kpoint_label] = []
        hsp_rows[row.kpoint_label].append(row)

    hsp_partner: dict[str, str] = {}
    for hsp in hsp_order:
        coordinate = hsp_rows[hsp][0].k_frac
        if any(
            np.linalg.norm(row.k_frac - coordinate) > _TOL
            for row in hsp_rows[hsp]
        ):
            blockers.append(f"inconsistent_source_hsp_coordinate:{hsp}")
            continue
        candidates = [
            other for other in hsp_order
            if centered_k_equivalent(
                -coordinate,
                hsp_rows[other][0].k_frac,
                centering,
                tolerance=_TOL,
            )
        ]
        if len(candidates) != 1:
            blockers.append(
                f"ambiguous_or_missing_time_reversal_hsp_partner:{hsp}:"
                f"{candidates}"
            )
            continue
        hsp_partner[hsp] = candidates[0]

    hsp_validation = _validate_involution(hsp_partner, hsp_order, "hsp")
    blockers.extend(hsp_validation)

    irrep_candidates: dict[str, list[str]] = {}
    irrep_partner: dict[str, str] = {}
    for row in rows:
        partner_hsp = hsp_partner.get(row.kpoint_label)
        candidates = [] if partner_hsp is None else [
            candidate.label
            for candidate in hsp_rows[partner_hsp]
            if _characters_are_time_reversal_partners(row, candidate)
        ]
        irrep_candidates[row.label] = candidates
        if len(candidates) != 1:
            blockers.append(
                "ambiguous_or_missing_time_reversal_irrep_partner:"
                f"{row.label}:{candidates}"
            )
            continue
        irrep_partner[row.label] = candidates[0]
    blockers.extend(_validate_involution(irrep_partner, labels, "irrep"))

    hsp_orbits: list[dict[str, object]] = []
    seen: set[str] = set()
    for hsp in hsp_order:
        if hsp in seen or hsp not in hsp_partner:
            continue
        partner = hsp_partner[hsp]
        members = [hsp] if partner == hsp else [hsp, partner]
        seen.update(members)
        hsp_orbits.append({
            "representative": hsp,
            "members": members,
            "self_mapped": partner == hsp,
        })

    return {
        "status": "validated" if not blockers else "blocked",
        "time_reversal_hsp_mapping": hsp_partner,
        "time_reversal_hsp_orbits": hsp_orbits,
        "independent_hsp_labels": [
            str(orbit["representative"]) for orbit in hsp_orbits
        ],
        "irrep_partner_candidates_by_label": irrep_candidates,
        "irrep_partner_by_label": irrep_partner,
        "operation_inventory_identity": next(iter(inventory_ids), ""),
        "spin_convention": next(iter(spin_conventions), ""),
        "blockers": blockers,
    }


def _characters_are_time_reversal_partners(
    left: ReviewedSourceIrrep,
    right: ReviewedSourceIrrep,
) -> bool:
    if (
        left.dimension != right.dimension
        or left.operation_indices != right.operation_indices
        or left.operation_inventory_identity != right.operation_inventory_identity
        or left.spin_convention != right.spin_convention
    ):
        return False
    return all(
        abs(right.characters[index] - left.characters[index].conjugate())
        <= _TOL
        for index in left.operation_indices
    )


def _validate_involution(
    mapping: dict[str, str],
    expected: Sequence[str],
    kind: str,
) -> list[str]:
    blockers: list[str] = []
    if set(mapping) != set(expected) or set(mapping.values()) != set(expected):
        blockers.append(f"incomplete_or_nonbijective_time_reversal_{kind}_mapping")
    if any(mapping.get(mapping.get(item, "")) != item for item in expected):
        blockers.append(f"non_involutive_time_reversal_{kind}_mapping")
    return blockers


def _blocked(reason: str) -> dict[str, object]:
    return _blocked_many([reason])


def _blocked_many(blockers: list[str]) -> dict[str, object]:
    return {
        "status": "blocked",
        "time_reversal_hsp_mapping": {},
        "time_reversal_hsp_orbits": [],
        "independent_hsp_labels": [],
        "irrep_partner_candidates_by_label": {},
        "irrep_partner_by_label": {},
        "operation_inventory_identity": "",
        "spin_convention": "",
        "blockers": blockers,
    }
