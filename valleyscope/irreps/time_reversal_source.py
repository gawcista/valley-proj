"""Reviewed unitary source-irrep time-reversal orbits.

The implementation derives ``k -> -k`` and character conjugation from the
reviewed source rows.  Labels are opaque identifiers and are never parsed to
infer HSP or irrep partners.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Mapping

import numpy as np

from valleyscope.io.wavefunction_convention import (
    canonical_identity,
    valid_sha256_identity,
)
from valleyscope.irreps.time_reversal_geometry import (
    centered_k_equivalent,
    normalize_centering_vectors,
)
from valleyscope.irreps.tables import (
    ReviewedSourceIrrep,
    load_standard_irrep_table,
    resolve_ebr_source_irrep_label_evidence,
)


_TOL = 5e-5
TR_SOURCE_CONTEXT_SCHEMA_VERSION = "1.0.0"
_HSP_SPECIFIC_STANDARD_SETTING_FIELDS = frozenset({
    "parent_k_frac",
    "resolved_hsp_label",
})


def derive_time_reversal_source_irrep_orbits(
    *,
    reviewed_rows: Sequence[ReviewedSourceIrrep],
    centering_vectors: Sequence[Sequence[float]],
    source_table_identity: Mapping[str, object] | None = None,
    standard_setting_certificate: Mapping[str, object] | None = None,
    parent_affine_operations: Sequence[Mapping[str, object]] | None = None,
    parent_affine_lift_record: Mapping[str, object] | None = None,
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

    report = {
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
    if (
        source_table_identity is not None
        or standard_setting_certificate is not None
    ):
        context = build_reviewed_time_reversal_source_context(
            reviewed_rows=rows,
            centering_vectors=centering_vectors,
            source_table_identity=source_table_identity,
            standard_setting_certificate=standard_setting_certificate,
            parent_affine_operations=parent_affine_operations,
            parent_affine_lift_record=parent_affine_lift_record,
        )
        report["reviewed_source_context"] = context
        report["reviewed_source_context_identity"] = context.get(
            "context_identity", ""
        )
        if context.get("status") != "validated":
            report["status"] = "blocked"
            report["blockers"] = _deduplicate([
                *blockers,
                *[
                    str(blocker)
                    for blocker in context.get("blockers", [])
                    if isinstance(blocker, str)
                ],
            ])
    return report


def build_reviewed_time_reversal_source_context(
    *,
    reviewed_rows: Sequence[ReviewedSourceIrrep],
    centering_vectors: Sequence[Sequence[float]],
    source_table_identity: Mapping[str, object] | None,
    standard_setting_certificate: Mapping[str, object] | None,
    parent_affine_operations: Sequence[Mapping[str, object]] | None = None,
    parent_affine_lift_record: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialize the raw reviewed source needed to rederive TR pairings."""
    rows = list(reviewed_rows)
    centering = normalize_centering_vectors(centering_vectors)
    blockers: list[str] = []
    if not rows or not all(isinstance(row, ReviewedSourceIrrep) for row in rows):
        blockers.append("reviewed_source_irrep_rows_missing_or_malformed")
    if centering is None:
        blockers.append("time_reversal_centering_vectors_missing_or_malformed")
    if not isinstance(source_table_identity, Mapping) or not source_table_identity:
        blockers.append("time_reversal_source_table_identity_missing_or_malformed")
    if (
        not isinstance(standard_setting_certificate, Mapping)
        or not standard_setting_certificate
    ):
        blockers.append(
            "time_reversal_standard_setting_certificate_missing_or_malformed"
        )
    serialized_rows = [
        _serialize_reviewed_source_irrep(row)
        for row in rows
        if isinstance(row, ReviewedSourceIrrep)
    ]
    content: dict[str, object] = {
        "schema_version": TR_SOURCE_CONTEXT_SCHEMA_VERSION,
        "reviewed_rows": serialized_rows,
        "normalized_centering_vectors": (
            [list(vector) for vector in centering]
            if centering is not None else []
        ),
        "source_table_identity": (
            deepcopy(dict(source_table_identity))
            if isinstance(source_table_identity, Mapping) else {}
        ),
        "standard_setting_certificate": (
            normalize_time_reversal_standard_setting_context(
                standard_setting_certificate
            )
            if isinstance(standard_setting_certificate, Mapping) else {}
        ),
        "parent_affine_operations": _serialize_parent_affine_operations(
            parent_affine_operations
        ),
        "parent_affine_lift_record": (
            deepcopy(dict(parent_affine_lift_record))
            if isinstance(parent_affine_lift_record, Mapping) else {}
        ),
    }
    try:
        identity = canonical_identity(content)
    except (TypeError, ValueError):
        identity = ""
        blockers.append("time_reversal_source_context_not_canonicalizable")
    return {
        **content,
        "status": "validated" if not blockers else "blocked",
        "context_identity": identity,
        "blockers": _deduplicate(blockers),
    }


def validate_reviewed_time_reversal_source_context(
    context: object,
    *,
    require_reviewed_table: bool = True,
) -> dict[str, object]:
    """Rebuild reviewed rows and recompute the TR source model fail-closed."""
    if not isinstance(context, Mapping):
        return _blocked("time_reversal_reviewed_source_context_malformed")
    blockers: list[str] = []
    content = {
        key: deepcopy(value)
        for key, value in context.items()
        if key not in {"status", "context_identity", "blockers"}
    }
    try:
        identity = canonical_identity(content)
    except (TypeError, ValueError):
        identity = ""
    if (
        context.get("schema_version") != TR_SOURCE_CONTEXT_SCHEMA_VERSION
        or context.get("status") != "validated"
        or context.get("blockers") != []
        or context.get("context_identity") != identity
        or not valid_sha256_identity(identity)
    ):
        blockers.append("time_reversal_reviewed_source_context_identity_invalid")
    rows_raw = context.get("reviewed_rows")
    rows: list[ReviewedSourceIrrep] = []
    if not isinstance(rows_raw, list) or not rows_raw:
        blockers.append("reviewed_source_irrep_rows_missing_or_malformed")
    else:
        for raw in rows_raw:
            row = _deserialize_reviewed_source_irrep(raw)
            if row is None:
                blockers.append("reviewed_source_irrep_rows_missing_or_malformed")
                break
            rows.append(row)
    centering = context.get("normalized_centering_vectors")
    normalized = (
        normalize_centering_vectors(centering)
        if isinstance(centering, list) else None
    )
    if normalized is None:
        blockers.append("time_reversal_centering_vectors_missing_or_malformed")
    source_table_identity = context.get("source_table_identity")
    if (
        not isinstance(source_table_identity, Mapping)
        or not source_table_identity
        or not _source_table_context_matches_rows(source_table_identity, rows)
    ):
        blockers.append("time_reversal_source_table_identity_mismatch")
    elif (
        require_reviewed_table
        and not _rows_match_reviewed_source_table(
            source_table_identity, rows_raw
        )
    ):
        blockers.append("time_reversal_reviewed_rows_source_table_mismatch")
    setting = context.get("standard_setting_certificate")
    parent_affine_operations = context.get("parent_affine_operations")
    parent_affine_lift_record = context.get(
        "parent_affine_lift_record"
    )
    if not isinstance(setting, Mapping) or not setting:
        blockers.append(
            "time_reversal_standard_setting_certificate_missing_or_malformed"
        )
    elif normalized is not None:
        setting_centering = normalize_centering_vectors(
            setting.get(
                "normalized_centering_vectors",
                setting.get("centering_vectors"),
            )
        )
        if (
            setting_centering is None
            or len(setting_centering) != len(normalized)
            or any(
                not np.allclose(left, right, atol=_TOL, rtol=0.0)
                for left, right in zip(setting_centering, normalized)
            )
        ):
            blockers.append(
                "time_reversal_centering_standard_setting_mismatch"
            )
    if (
        require_reviewed_table
        and isinstance(source_table_identity, Mapping)
        and isinstance(setting, Mapping)
        and not _standard_setting_matches_reviewed_source_table(
            source_table_identity,
            setting,
        )
    ):
        blockers.append(
            "time_reversal_standard_setting_reviewed_table_mismatch"
        )
    if (
        require_reviewed_table
        and isinstance(source_table_identity, Mapping)
        and isinstance(setting, Mapping)
        and not _standard_setting_affine_context_valid(
            source_table_identity,
            setting,
            parent_affine_operations,
        )
    ):
        blockers.append(
            "time_reversal_standard_setting_affine_context_mismatch"
        )
    if (
        require_reviewed_table
        and isinstance(setting, Mapping)
        and not _parent_affine_context_matches_lift_record(
            setting=setting,
            parent_affine_operations=parent_affine_operations,
            lift_record=parent_affine_lift_record,
        )
    ):
        blockers.append(
            "time_reversal_parent_affine_producer_evidence_mismatch"
        )
    if blockers:
        return _blocked_many(_deduplicate(blockers))
    try:
        derived = derive_time_reversal_source_irrep_orbits(
            reviewed_rows=rows,
            centering_vectors=normalized,
        )
    except Exception:
        return _blocked(
            "time_reversal_reviewed_source_context_rederivation_failed"
        )
    derived["reviewed_source_context"] = deepcopy(dict(context))
    derived["reviewed_source_context_identity"] = identity
    return derived


def normalize_time_reversal_standard_setting_context(
    certificate: Mapping[str, object],
) -> dict[str, object]:
    """Keep the shared affine setting, excluding local HSP classification."""
    return {
        key: deepcopy(value)
        for key, value in certificate.items()
        if key not in _HSP_SPECIFIC_STANDARD_SETTING_FIELDS
    }


def _serialize_reviewed_source_irrep(
    row: ReviewedSourceIrrep,
) -> dict[str, object]:
    return {
        "label": row.label,
        "kpoint_label": row.kpoint_label,
        "k_frac": [float(value) for value in row.k_frac],
        "dimension": row.dimension,
        "characters": {
            str(index): [float(value.real), float(value.imag)]
            for index, value in sorted(row.characters.items())
        },
        "operation_indices": list(row.operation_indices),
        "operation_inventory_identity": row.operation_inventory_identity,
        "spinor": row.spinor,
        "spin_convention": row.spin_convention,
        "source_table": row.source_table,
        "source_table_status": row.source_table_status,
        "source_provenance": row.source_provenance,
    }


def _deserialize_reviewed_source_irrep(
    raw: object,
) -> ReviewedSourceIrrep | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        character_rows = raw["characters"]
        if not isinstance(character_rows, Mapping):
            return None
        characters = {
            int(index): complex(float(value[0]), float(value[1]))
            for index, value in character_rows.items()
            if isinstance(value, list) and len(value) == 2
        }
        operation_indices = tuple(int(value) for value in raw["operation_indices"])
        if set(characters) != set(operation_indices):
            return None
        return ReviewedSourceIrrep(
            label=str(raw["label"]),
            kpoint_label=str(raw["kpoint_label"]),
            k_frac=np.asarray(raw["k_frac"], dtype=float),
            dimension=int(raw["dimension"]),
            characters=characters,
            operation_indices=operation_indices,
            operation_inventory_identity=str(
                raw["operation_inventory_identity"]
            ),
            spinor=raw["spinor"],
            spin_convention=str(raw["spin_convention"]),
            source_table=str(raw["source_table"]),
            source_table_status=str(raw["source_table_status"]),
            source_provenance=str(raw["source_provenance"]),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _source_table_context_matches_rows(
    identity: Mapping[str, object],
    rows: Sequence[ReviewedSourceIrrep],
) -> bool:
    if not rows:
        return False
    return bool(
        isinstance(identity.get("space_group_number"), int)
        and not isinstance(identity.get("space_group_number"), bool)
        and identity.get("space_group_number", 0) > 0
        and isinstance(identity.get("space_group_symbol"), str)
        and identity.get("space_group_symbol")
        and identity.get("source_table_name")
        == identity.get("space_group_symbol")
        and identity.get("source_table_provenance")
        == rows[0].source_provenance
        and identity.get("spinor") is rows[0].spinor
        and len({row.source_provenance for row in rows}) == 1
        and len({row.operation_inventory_identity for row in rows}) == 1
        and len({row.spinor for row in rows}) == 1
    )


def _rows_match_reviewed_source_table(
    identity: Mapping[str, object],
    serialized_rows: object,
) -> bool:
    if not isinstance(serialized_rows, list):
        return False
    try:
        table = load_standard_irrep_table(
            int(identity["space_group_number"]),
            spinor=identity["spinor"],
        )
        labels = [
            str(row["label"])
            for row in serialized_rows
            if isinstance(row, Mapping)
        ]
        evidence = resolve_ebr_source_irrep_label_evidence(
            table=table,
            source_basis_labels=labels,
        )
        reviewed_rows = evidence.get("reviewed_rows", [])
        return bool(
            table.name == identity.get("space_group_symbol")
            and table.name == identity.get("source_table_name")
            and evidence.get("status") == "validated"
            and len(reviewed_rows) == len(serialized_rows)
            and [
                _serialize_reviewed_source_irrep(row)
                for row in reviewed_rows
            ] == serialized_rows
        )
    except Exception:
        return False


def _standard_setting_matches_reviewed_source_table(
    identity: Mapping[str, object],
    setting: Mapping[str, object],
) -> bool:
    """Rederive canonical setting identity from reviewed table operations."""
    try:
        table = load_standard_irrep_table(
            int(identity["space_group_number"]),
            spinor=identity["spinor"],
        )
        from valleyscope.analysis.standard_setting_kmap import (
            derive_irreptables_standard_setting_identity,
        )
        derived = derive_irreptables_standard_setting_identity(
            table,
            int(identity["space_group_number"]),
        )
    except Exception:
        return False
    field_pairs = {
        "subspace_sg_number": "space_group_number",
        "subspace_sg_symbol": "space_group_symbol",
        "hall_number": "hall_number",
        "hall_symbol": "hall_symbol",
        "centering_type": "centering_type",
        "primitive_conventional_index": "primitive_conventional_index",
        "standard_setting_operation_count": (
            "standard_setting_operation_count"
        ),
        "standard_operation_closure_validated": (
            "standard_operation_closure_validated"
        ),
        "centering_vectors": "centering_cosets",
        "canonical_setting_status": "status",
        "canonical_setting_source": "source",
        "canonical_hall_numbers": "affine_matching_hall_numbers",
        "canonical_candidate_hall_numbers": "candidate_hall_numbers",
    }
    return bool(
        derived.get("status") == "unique_match"
        and all(
            setting.get(setting_key) == derived.get(derived_key)
            for setting_key, derived_key in field_pairs.items()
        )
    )


def _standard_setting_affine_context_valid(
    identity: Mapping[str, object],
    setting: Mapping[str, object],
    parent_affine_operations: object,
) -> bool:
    """Recompute parent-to-standard affine equivalence from raw operations."""
    if not isinstance(parent_affine_operations, list) or not (
        parent_affine_operations
    ):
        return False
    required_ids = setting.get("parent_basis_operation_ids")
    transform = setting.get("parent_to_standard_direct_transform")
    origin = setting.get("origin_shift_fractional")
    if not isinstance(required_ids, list):
        return False
    standard_match = {
        "number": identity.get("space_group_number"),
        "international_short": identity.get("space_group_symbol"),
        "hall_number": setting.get("hall_number"),
        "hall_symbol": setting.get("hall_symbol"),
    }
    try:
        from valleyscope.analysis.standard_setting_kmap import (
            _validate_affine_operation_equivalence,
        )
        affine = _validate_affine_operation_equivalence(
            vp_operations=parent_affine_operations,
            vp_operation_ids=required_ids,
            standard_match=standard_match,
            parent_to_standard_direct_transform=np.asarray(
                transform, dtype=float
            ),
            origin_shift_fractional=(
                np.asarray(origin, dtype=float)
                if origin is not None else None
            ),
        )
    except Exception:
        return False
    field_pairs = {
        "translation_validation_status": "status",
        "total_parent_operations": "total_parent_operations",
        "matched_affine_operations": "matched_affine_operations",
        "standard_setting_operation_count": (
            "standard_setting_operation_count"
        ),
        "missing_affine_ingredients": "missing_ingredients",
        "mismatched_translation_count": "mismatched_translation_count",
        "operation_closure_validated": "operation_closure_validated",
        "affine_operation_map": "operation_map",
        "unmatched_parent_operations": "unmatched_parent_operations",
        "unused_standard_operation_indices": (
            "unused_standard_operation_indices"
        ),
        "parent_basis_operation_ids": "required_operation_ids",
        "required_operation_id_count": "required_operation_id_count",
        "centering_coset_count": "centering_cosets_count",
        "primitive_conventional_index": "primitive_conventional_index",
        "expanded_parent_operation_count": (
            "expanded_parent_operation_count"
        ),
        "matched_expanded_operations": "matched_expanded_operations",
        "standard_operation_closure_validated": (
            "standard_operation_closure_validated"
        ),
        "centered_affine_operation_map": "centered_operation_map",
        "unmatched_centered_operation_pairs": (
            "unmatched_centered_operation_pairs"
        ),
    }
    return bool(
        affine.get("status") == "passed"
        and all(
            setting.get(setting_key) == affine.get(affine_key)
            for setting_key, affine_key in field_pairs.items()
        )
    )


def _serialize_parent_affine_operations(
    operations: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, object]]:
    if operations is None:
        return []
    serialized: list[dict[str, object]] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            return []
        operation_id = operation.get("operation_id")
        if not isinstance(operation_id, int) or isinstance(operation_id, bool):
            return []
        try:
            rotation = np.asarray(
                operation.get("rotation_frac"), dtype=float
            )
            translation = np.asarray(
                operation.get("translation_frac"), dtype=float
            )
        except (TypeError, ValueError):
            return []
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(translation))
        ):
            return []
        serialized.append({
            "operation_id": operation_id,
            "rotation_frac": rotation.tolist(),
            "translation_frac": translation.tolist(),
        })
    return serialized


def _parent_affine_context_matches_lift_record(
    *,
    setting: Mapping[str, object],
    parent_affine_operations: object,
    lift_record: object,
) -> bool:
    """Bind raw parent operations to the producer-owned group-wide lift."""
    if not isinstance(lift_record, Mapping):
        return False
    content = {
        key: deepcopy(value)
        for key, value in lift_record.items()
        if key not in {"status", "reason_codes", "certificate_identity"}
    }
    operation_inventory = lift_record.get("operation_inventory")
    if not isinstance(operation_inventory, list):
        return False
    try:
        record_identity = canonical_identity(content)
        inventory_identity = canonical_identity({
            "operations": operation_inventory,
        })
    except (TypeError, ValueError):
        return False
    required_ids = setting.get("parent_basis_operation_ids")
    if not isinstance(required_ids, list):
        return False
    required_set = set(required_ids)
    raw_operations = _serialize_parent_affine_operations(
        parent_affine_operations
        if isinstance(parent_affine_operations, list) else None
    )
    lift_operations = _serialize_parent_affine_operations(
        operation_inventory
    )
    raw_required = [
        operation for operation in raw_operations
        if operation["operation_id"] in required_set
    ]
    lift_required = [
        operation for operation in lift_operations
        if operation["operation_id"] in required_set
    ]
    lift_setting = lift_record.get("standard_setting_identity")
    if not isinstance(lift_setting, Mapping):
        return False
    claimed_origin = setting.get(
        "origin_shift_fractional",
        [0.0, 0.0, 0.0],
    )
    return bool(
        lift_record.get("status") == "passed"
        and lift_record.get("reason_codes") == []
        and valid_sha256_identity(record_identity)
        and lift_record.get("certificate_identity") == record_identity
        and lift_record.get("operation_inventory_identity")
        == inventory_identity
        and len(raw_required) == len(required_set)
        and raw_required == lift_required
        and lift_setting.get("status") == "passed"
        and lift_setting.get("reason_codes") == []
        and lift_setting.get("parent_to_standard_direct_transform")
        == setting.get("parent_to_standard_direct_transform")
        and lift_setting.get("origin_shift_fractional") == claimed_origin
    )


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


def _deduplicate(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))
