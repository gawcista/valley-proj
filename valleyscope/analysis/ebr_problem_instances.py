"""EBR problem-instance collector from trusted EBR input candidates.

Groups trusted candidate irreps into per-valley/per-subspace-group EBR
problem instances with a certificate-aware physical identity key.

State model
-----------
1. ``sampled_basis`` — trusted irrep rows collected on a sampled HSP basis.
   ``ready_for_reduced_table_validation`` = true, ``ready_for_ebr_decomposition`` = false.
2. ``table_validated`` — HSP basis validated against a reviewed
   irreptables-derived reduced table (requires external table; not yet wired).
3. ``validated_basis`` — basis and certificate confirmed; instance is ready
   for exact reduced EBR decomposition.
4. Solve attempted/completed (downstream in ``reduced_ebr_mapping.py``).

Does NOT implement reduced EBR decomposition, EBR table matching,
compatibility relations, or new physics.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def build_ebr_problem_instances(
    *,
    ebr_input_candidates: dict[str, object] | None,
) -> dict[str, object]:
    """Build EBR problem instances from trusted input candidates.

    Returns a dict with instances grouped by certificate-aware physical
    identity (SG number, SG symbol, Hall number, certificate validation
    status, valley).
    """
    if ebr_input_candidates is None:
        return _empty_report("no EBR input candidates available")

    candidates: list[dict[str, object]] = []
    raw = ebr_input_candidates.get("candidates")
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, dict) and c.get("ready_for_ebr_input") is True:
                candidates.append(c)

    if not candidates:
        return _empty_report("no trusted EBR input candidates")

    # Group by certificate-aware physical identity.
    # Use the complete immutable _SettingIdentity as the key so that
    # any affine-evidence difference (transform, origin, centering,
    # provenance, operation-mapping, affine-validation status) produces
    # a distinct group.
    groups: dict[tuple[_SettingIdentity, str], list[dict[str, object]]] = {}
    for c in candidates:
        valley = str(c.get("valley", ""))
        fp = _certificate_fingerprint(c)
        groups.setdefault((fp, valley), []).append(c)

    instances: list[dict[str, object]] = []
    instance_counter = 0

    for (fp, valley), cands in groups.items():
        instance_counter += 1
        # Setting identity for the flat keys.
        sg = fp.sg_symbol or str(cands[0].get("subspace_group_candidate", ""))
        _sg_num = fp.sg_number
        instance_id = f"ebr_instance_{instance_counter:03d}"

        # --- Canonical subgroup identity ---
        first_candidate_ssg = _first_subspace_space_group(cands)
        subspace_space_group: dict[str, object] = (
            dict(first_candidate_ssg) if isinstance(first_candidate_ssg, dict) else {}
        )
        canonical_sg = (
            subspace_space_group.get("candidate_space_group_symbol")
            or sg
        )
        canonical_sg_number = (
            subspace_space_group.get("candidate_space_group_number")
            or _sg_num
        )

        # --- Certificate identity from merged candidates ---
        cert_identity = _certificate_identity(cands)

        # --- Aggregate provenance ---
        workflow_paths = sorted({
            str(c.get("workflow_path", ""))
            for c in cands if c.get("workflow_path")
        })
        readiness_levels = sorted({
            str(c.get("readiness_level", ""))
            for c in cands if c.get("readiness_level")
        })
        workflow_path = str(cands[0].get("workflow_path", ""))
        readiness_level = str(cands[0].get("readiness_level", ""))

        irreps_by_kpoint: dict[str, list[str]] = {}
        operations_by_kpoint: dict[str, list[object]] = {}
        irrep_records_by_kpoint: dict[str, list[dict[str, object]]] = {}
        for c in cands:
            kp = str(c.get("kpoint", ""))
            irrep = c.get("matched_irrep")
            op_id = c.get("operation_id")
            if irrep:
                multiplicity = _positive_multiplicity(c.get("irrep_multiplicity"))
                irreps_by_kpoint.setdefault(kp, []).extend(
                    [str(irrep)] * multiplicity
                )
            if op_id is not None:
                operations_by_kpoint.setdefault(kp, []).append(op_id)
            record: dict[str, object] = {
                "valley": valley,
                "operation_id": c.get("operation_id"),
                "operation_order": c.get("operation_order"),
                "matched_irrep": c.get("matched_irrep"),
                "irrep_multiplicity": _positive_multiplicity(
                    c.get("irrep_multiplicity")
                ),
                "character": c.get("character"),
                "eigenphases": c.get("eigenphases", []),
                "workflow_path": str(c.get("workflow_path", "")),
                "readiness_level": str(c.get("readiness_level", "")),
                "source": c.get("source", ""),
            }
            for key in (
                "matching_strategy",
                "subspace_space_group",
                "valley_preserving_operation_ids",
                "source_operation_map",
                "irrep_source_provenance",
            ):
                if key in c:
                    record[key] = c[key]
            irrep_records_by_kpoint.setdefault(kp, []).append(record)

        # --- HSP basis ---
        actual_hsps = sorted(irreps_by_kpoint.keys())
        expected_hsps = list(actual_hsps)
        optional_hsps: list[str] = []

        has_hsps = bool(actual_hsps)
        # State 1 (sampled_basis): trusted irrep rows collected, not yet
        # validated against a reviewed reduced table.
        hsp_basis_status = "sampled_basis" if has_hsps else "no_data"
        status = "sampled_basis" if has_hsps else "no_data"

        instances.append({
            "instance_id": instance_id,
            "valley": valley,
            "subspace_group_candidate": canonical_sg,
            "subspace_sg_number": canonical_sg_number,
            "subspace_space_group": subspace_space_group,
            "certificate_identity": cert_identity,
            "workflow_path": workflow_path,
            "workflow_paths": workflow_paths,
            "readiness_level": readiness_level,
            "readiness_evidence": readiness_levels,
            "irreps_by_kpoint": {k: v for k, v in sorted(irreps_by_kpoint.items())},
            "operations_by_kpoint": {
                k: sorted(v, key=_sort_key)
                for k, v in sorted(operations_by_kpoint.items())
            },
            "irrep_records_by_kpoint": {
                k: sorted(v, key=lambda r: (_sort_key(r.get("operation_id"))))
                for k, v in sorted(irrep_records_by_kpoint.items())
            },
            "candidate_count": len(cands),
            "status": status,
            "ready_for_reduced_table_validation": has_hsps,
            "ready_for_ebr_decomposition": False,
            "blocked_by": [],
            "expected_hsps": expected_hsps,
            "expected_hsp_policy_source": "sampled_irrep_basis",
            "hsp_basis_status": hsp_basis_status,
            "optional_hsps": optional_hsps,
            "actual_hsps": actual_hsps,
            "missing_optional_hsps": [],
        })

    overall_status = "has_instances" if instances else "no_instances"
    return {
        "status": overall_status,
        "instance_count": len(instances),
        "reduced_ebr_decomposition_status": "not_implemented",
        "interpretation": (
            "Per-valley/per-subspace-group EBR problem instances grouped from "
            "trusted input candidates by certificate-aware (SG number, SG symbol, "
            "Hall number, certificate validation status, valley) identity.  "
            "State 1 (sampled_basis): irrep rows collected on a sampled HSP basis, "
            "ready_for_reduced_table_validation=true, "
            "ready_for_ebr_decomposition=false.  Promotion to state 2 "
            "(table_validated) and state 3 (validated_basis) requires a reviewed "
            "irreptables-derived reduced table."
        ),
        "instances": instances,
    }


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _first_subspace_space_group(
    cands: list[dict[str, object]],
) -> dict[str, object]:
    """Return the canonical subspace_space_group from the first candidate."""
    for c in cands:
        ssg = c.get("subspace_space_group")
        if isinstance(ssg, dict) and ssg:
            return ssg
    return {}


def _sort_key(op_id: object) -> tuple[int, object]:
    try:
        return (0, int(str(op_id)))
    except (TypeError, ValueError):
        return (1, str(op_id))


def _positive_multiplicity(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 1


def _int_or_zero(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _empty_report(reason: str) -> dict[str, object]:
    return {
        "status": "no_instances",
        "instance_count": 0,
        "reduced_ebr_decomposition_status": "not_implemented",
        "interpretation": reason,
        "instances": [],
    }


# ---------------------------------------------------------------------------
# Certificate-aware identity
# ---------------------------------------------------------------------------

_CERT_TOL = 1e-9
_SENTINEL_MISSING = object()  # distinguishes absent from empty in fingerprints


class _SettingIdentity:
    """Hashable normalized standard-setting certificate identity.

    Captures the setting-level affine evidence needed to distinguish
    physically inequivalent conventions for the same space group.
    HSP-specific fields (parent_k_frac, resolved_hsp_label) are excluded
    because they vary per k-point and belong in per-row provenance.
    """

    __slots__ = (
        "_hash",
        "sg_number", "sg_symbol",
        "hall_number", "hall_symbol",
        "transform_key", "origin_shift_key",
        "centering_type", "centering_vectors_key",
        "primitive_conventional_relation",
        "transform_provenance",
        "validation_status",
        "operation_mapping_status",
        "affine_validation_status",
        "affine_matched_operations",
        "affine_total_operations",
        "affine_mismatch_count",
        "affine_missing_ingredients",
        "affine_standard_setting_op_count",
        "affine_operation_map",
        "affine_unmatched_parents",
        "affine_unused_std",
        "affine_required_operation_ids",
        "affine_required_op_count",
        "operation_closure_validated",
        "canonical_setting_status",
        "canonical_setting_source",
        "canonical_hall_numbers",
        "canonical_candidate_hall_numbers",
        "centering_coset_count",
        "primitive_conventional_index",
        "expanded_parent_operation_count",
        "matched_expanded_operations",
        "centered_affine_operation_map",
        "affine_unmatched_centered_pairs",
        "standard_operation_closure_validated",
    )

    def __init__(
        self,
        sg_number: int,
        sg_symbol: str,
        hall_number: int,
        hall_symbol: str,
        transform_key: tuple[tuple[float, float, float],
                            tuple[float, float, float],
                            tuple[float, float, float]] | None,
        origin_shift_key: tuple[float, float, float] | None,
        centering_type: str,
        centering_vectors_key: tuple[tuple[float, float, float], ...] | None,
        primitive_conventional_relation: str,
        transform_provenance: str,
        validation_status: str,
        operation_mapping_status: str,
        affine_validation_status: str,
        affine_matched_operations: int | None = None,
        affine_total_operations: int | None = None,
        affine_mismatch_count: int | None = None,
        affine_missing_ingredients: tuple[str, ...] | None = None,
        affine_standard_setting_op_count: int | None = None,
        affine_operation_map: tuple[tuple[int, int], ...] | None = None,
        affine_unmatched_parents: tuple[int, ...] | None = None,
        affine_unused_std: tuple[int, ...] | None = None,
        affine_required_operation_ids: tuple[int, ...] | None = None,
        affine_required_op_count: int | None = None,
        operation_closure_validated: bool | None = None,
        canonical_setting_status: str = "not_evaluated",
        canonical_setting_source: str = "",
        canonical_hall_numbers: tuple[int, ...] | None = None,
        canonical_candidate_hall_numbers: tuple[int, ...] | None = None,
        centering_coset_count: int | None = None,
        primitive_conventional_index: int | None = None,
        expanded_parent_operation_count: int | None = None,
        matched_expanded_operations: int | None = None,
        centered_affine_operation_map: tuple[tuple[int, int, int], ...] | None = None,
        affine_unmatched_centered_pairs: tuple[tuple[int, int], ...] | None = None,
        standard_operation_closure_validated: bool | None = None,
    ):
        self.sg_number = sg_number
        self.sg_symbol = sg_symbol
        self.hall_number = hall_number
        self.hall_symbol = hall_symbol
        self.transform_key = transform_key
        self.origin_shift_key = origin_shift_key
        self.centering_type = centering_type
        self.centering_vectors_key = centering_vectors_key
        self.primitive_conventional_relation = primitive_conventional_relation
        self.transform_provenance = transform_provenance
        self.validation_status = validation_status
        self.operation_mapping_status = operation_mapping_status
        self.affine_validation_status = affine_validation_status
        self.affine_matched_operations = affine_matched_operations
        self.affine_total_operations = affine_total_operations
        self.affine_mismatch_count = affine_mismatch_count
        self.affine_missing_ingredients = affine_missing_ingredients
        self.affine_standard_setting_op_count = affine_standard_setting_op_count
        self.affine_operation_map = affine_operation_map
        self.affine_unmatched_parents = affine_unmatched_parents
        self.affine_unused_std = affine_unused_std
        self.affine_required_operation_ids = affine_required_operation_ids
        self.affine_required_op_count = affine_required_op_count
        self.operation_closure_validated = operation_closure_validated
        self.canonical_setting_status = canonical_setting_status
        self.canonical_setting_source = canonical_setting_source
        self.canonical_hall_numbers = canonical_hall_numbers
        self.canonical_candidate_hall_numbers = canonical_candidate_hall_numbers
        self.centering_coset_count = centering_coset_count
        self.primitive_conventional_index = primitive_conventional_index
        self.expanded_parent_operation_count = expanded_parent_operation_count
        self.matched_expanded_operations = matched_expanded_operations
        self.centered_affine_operation_map = centered_affine_operation_map
        self.affine_unmatched_centered_pairs = affine_unmatched_centered_pairs
        self.standard_operation_closure_validated = \
            standard_operation_closure_validated
        self._hash = hash((
            sg_number, sg_symbol,
            hall_number, hall_symbol,
            transform_key, origin_shift_key,
            centering_type, centering_vectors_key,
            primitive_conventional_relation,
            transform_provenance,
            validation_status,
            operation_mapping_status,
            affine_validation_status,
            affine_matched_operations,
            affine_total_operations,
            affine_mismatch_count,
            affine_missing_ingredients,
            affine_standard_setting_op_count,
            affine_operation_map,
            affine_unmatched_parents,
            affine_unused_std,
            affine_required_operation_ids,
            affine_required_op_count,
            operation_closure_validated,
            canonical_setting_status,
            canonical_setting_source,
            canonical_hall_numbers,
            canonical_candidate_hall_numbers,
            centering_coset_count,
            primitive_conventional_index,
            expanded_parent_operation_count,
            matched_expanded_operations,
            centered_affine_operation_map,
            affine_unmatched_centered_pairs,
            standard_operation_closure_validated,
        ))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SettingIdentity):
            return NotImplemented
        return self._hash == other._hash and (
            self.sg_number == other.sg_number
            and self.sg_symbol == other.sg_symbol
            and self.hall_number == other.hall_number
            and self.hall_symbol == other.hall_symbol
            and self.transform_key == other.transform_key
            and self.origin_shift_key == other.origin_shift_key
            and self.centering_type == other.centering_type
            and self.centering_vectors_key == other.centering_vectors_key
            and self.primitive_conventional_relation == other.primitive_conventional_relation
            and self.transform_provenance == other.transform_provenance
            and self.validation_status == other.validation_status
            and self.operation_mapping_status == other.operation_mapping_status
            and self.affine_validation_status == other.affine_validation_status
            and self.affine_matched_operations == other.affine_matched_operations
            and self.affine_total_operations == other.affine_total_operations
            and self.affine_mismatch_count == other.affine_mismatch_count
            and self.affine_missing_ingredients == other.affine_missing_ingredients
            and self.affine_standard_setting_op_count == other.affine_standard_setting_op_count
            and self.affine_operation_map == other.affine_operation_map
            and self.affine_unmatched_parents == other.affine_unmatched_parents
            and self.affine_unused_std == other.affine_unused_std
            and self.affine_required_operation_ids == other.affine_required_operation_ids
            and self.affine_required_op_count == other.affine_required_op_count
            and self.operation_closure_validated == other.operation_closure_validated
            and self.canonical_setting_status == other.canonical_setting_status
            and self.canonical_setting_source == other.canonical_setting_source
            and self.canonical_hall_numbers == other.canonical_hall_numbers
            and self.canonical_candidate_hall_numbers == other.canonical_candidate_hall_numbers
            and self.centering_coset_count == other.centering_coset_count
            and self.primitive_conventional_index == other.primitive_conventional_index
            and self.expanded_parent_operation_count == other.expanded_parent_operation_count
            and self.matched_expanded_operations == other.matched_expanded_operations
            and self.centered_affine_operation_map == other.centered_affine_operation_map
            and self.affine_unmatched_centered_pairs == other.affine_unmatched_centered_pairs
            and self.standard_operation_closure_validated == other.standard_operation_closure_validated
        )


def _normalize_transform(
    matrix: object,
) -> tuple[tuple[float, float, float],
           tuple[float, float, float],
           tuple[float, float, float]] | None:
    """Normalize a 3×3 transform matrix to a hashable tuple.

    Rounds to tolerance, replaces -0.0 with 0.0.  Returns None for
    non-finite or non-3×3 input.
    """
    if not isinstance(matrix, (list, tuple)):
        return None
    if len(matrix) != 3:
        return None
    rows: list[tuple[float, float, float]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        r: list[float] = []
        for v in row:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(f):
                return None
            f = round(f / _CERT_TOL) * _CERT_TOL
            if f == 0.0:
                f = 0.0  # normalize -0.0
            r.append(f)
        rows.append((r[0], r[1], r[2]))
    return (rows[0], rows[1], rows[2])


def _normalize_origin_shift(
    vector: object,
) -> tuple[float, float, float] | None:
    """Normalize a 3-vector origin shift to a hashable tuple."""
    if not isinstance(vector, (list, tuple)) or len(vector) != 3:
        return None
    comps: list[float] = []
    for v in vector:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(f):
            return None
        # Modulo lattice: shift into [0, 1).
        f = f - np.floor(f)
        f = round(f / _CERT_TOL) * _CERT_TOL
        if f == 0.0:
            f = 0.0
        # Remap 1.0 (from rounding) back to 0.0.
        if f >= 1.0 - _CERT_TOL:
            f = 0.0
        comps.append(f)
    return (comps[0], comps[1], comps[2])


def _normalize_centering_vectors(
    vectors: object,
) -> tuple[tuple[float, float, float], ...] | None:
    """Normalize centering vectors to a sorted hashable tuple."""
    if not isinstance(vectors, list):
        return None
    normed: list[tuple[float, float, float]] = []
    for v in vectors:
        normalized = _normalize_origin_shift(v)
        if normalized is None:
            return None
        normed.append(normalized)
    return tuple(sorted(normed))


def _normalize_strict_int_list(
    value: object,
    *,
    unique: bool = False,
) -> tuple[int, ...] | None:
    """Normalize an exact runtime ``list[int]`` without coercion."""
    if not isinstance(value, list):
        return None
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        return None
    if unique and len(value) != len(set(value)):
        return None
    return tuple(sorted(value))


def _normalize_strict_string_list(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    if any(not isinstance(item, str) for item in value):
        return None
    return tuple(sorted(value))


def _normalize_operation_id_key(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = int(value)
    except ValueError:
        return None
    if str(normalized) != value:
        return None
    return normalized


def _normalize_affine_operation_map(
    value: object,
) -> tuple[tuple[int, int], ...] | None:
    """Normalize an operation map and reject key aliases after coercion."""
    if not isinstance(value, dict):
        return None
    normalized: dict[int, int] = {}
    for raw_key, raw_value in value.items():
        key = _normalize_operation_id_key(raw_key)
        if key is None or key in normalized:
            return None
        if not isinstance(raw_value, int) or isinstance(raw_value, bool):
            return None
        normalized[key] = raw_value
    return tuple(sorted(normalized.items()))


def _normalize_unmatched_parent_operations(
    value: object,
) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    operation_ids: list[int] = []
    for row in value:
        if not isinstance(row, dict):
            return None
        operation_id = row.get("operation_id")
        if not isinstance(operation_id, int) or isinstance(operation_id, bool):
            return None
        operation_ids.append(operation_id)
    if len(operation_ids) != len(set(operation_ids)):
        return None
    return tuple(sorted(operation_ids))


def _normalize_centered_affine_operation_map(
    value: object,
) -> tuple[tuple[int, int, int], ...] | None:
    if not isinstance(value, list):
        return None
    normalized: list[tuple[int, int, int]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "parent_operation_id",
            "centering_coset_index",
            "standard_operation_index",
        }:
            return None
        fields = (
            row["parent_operation_id"],
            row["centering_coset_index"],
            row["standard_operation_index"],
        )
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in fields
        ):
            return None
        normalized.append(fields)
    if len(normalized) != len(set(normalized)):
        return None
    return tuple(sorted(normalized))


def _normalize_unmatched_centered_pairs(
    value: object,
) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(value, list):
        return None
    normalized: list[tuple[int, int]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "parent_operation_id", "centering_coset_index",
        }:
            return None
        fields = (
            row["parent_operation_id"], row["centering_coset_index"],
        )
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in fields
        ):
            return None
        normalized.append(fields)
    if len(normalized) != len(set(normalized)):
        return None
    return tuple(sorted(normalized))


def _certificate_fingerprint(candidate: dict[str, object]) -> _SettingIdentity:
    """Extract a normalized setting identity from one candidate."""
    prov = candidate.get("irrep_source_provenance")
    cert: dict[str, object] = {}
    if isinstance(prov, dict):
        kmap = prov.get("standard_setting_hsp_mapping")
        if isinstance(kmap, dict):
            c = kmap.get("standard_setting_certificate")
            if isinstance(c, dict):
                cert = c

    sg_number = 0
    sg_symbol = ""
    hall_number = 0
    hall_symbol = ""
    centering_type = ""
    primitive_conventional_relation = ""
    transform_provenance = ""
    validation_status = "not_evaluated"
    operation_mapping_status = "not_attempted"
    affine_validation_status = "not_attempted"

    # Also check subspace_space_group for SG identity.
    ssg = candidate.get("subspace_space_group")
    if isinstance(ssg, dict):
        sn = ssg.get("candidate_space_group_number")
        if isinstance(sn, int) and not isinstance(sn, bool):
            sg_number = int(sn)
        sy = ssg.get("candidate_space_group_symbol")
        if isinstance(sy, str) and sy:
            sg_symbol = str(sy)

    hn = cert.get("hall_number")
    if isinstance(hn, int) and not isinstance(hn, bool):
        hall_number = int(hn)
    hs = cert.get("hall_symbol")
    if isinstance(hs, str) and hs:
        hall_symbol = str(hs)
    vs = cert.get("validation_status")
    if isinstance(vs, str) and vs:
        validation_status = str(vs)
    ct = cert.get("centering_type")
    if isinstance(ct, str) and ct:
        centering_type = str(ct)
    pcr = cert.get("primitive_conventional_relation")
    if isinstance(pcr, str) and pcr:
        primitive_conventional_relation = str(pcr)
    tp = cert.get("transform_provenance")
    if isinstance(tp, str) and tp:
        transform_provenance = str(tp)
    oms = cert.get("operation_mapping_status")
    if isinstance(oms, str) and oms:
        operation_mapping_status = str(oms)
    avs = cert.get("translation_validation_status")
    if isinstance(avs, str) and avs:
        affine_validation_status = str(avs)

    def _opt_int(value: object) -> int | None:
        return int(value) if isinstance(value, int) \
            and not isinstance(value, bool) else None

    affine_matched_operations = _opt_int(cert.get("matched_affine_operations"))
    affine_total_operations = _opt_int(cert.get("total_parent_operations"))
    affine_mismatch_count = _opt_int(cert.get("mismatched_translation_count"))
    # Distinguish missing/absent from empty: None = evidence absent.
    # The list may still be empty — that is explicit evidence (no ingredients
    # missing).  A missing key / None is unknown.
    missing = cert.get("missing_affine_ingredients", _SENTINEL_MISSING)
    affine_missing_ingredients = (
        None
        if missing is _SENTINEL_MISSING
        else _normalize_strict_string_list(missing)
    )

    closure = cert.get("operation_closure_validated")
    operation_closure_validated = closure if isinstance(closure, bool) else None
    standard_closure = cert.get("standard_operation_closure_validated")
    standard_operation_closure_validated = (
        standard_closure if isinstance(standard_closure, bool) else None
    )

    std_op_count = _opt_int(cert.get("standard_setting_operation_count"))
    req_op_count = _opt_int(cert.get("required_operation_id_count"))
    affine_operation_map = _normalize_affine_operation_map(
        cert.get("affine_operation_map", _SENTINEL_MISSING)
    )
    affine_unmatched_parents = _normalize_unmatched_parent_operations(
        cert.get("unmatched_parent_operations", _SENTINEL_MISSING)
    )
    affine_unused_std = _normalize_strict_int_list(
        cert.get("unused_standard_operation_indices", _SENTINEL_MISSING),
        unique=True,
    )
    affine_required_operation_ids = _normalize_strict_int_list(
        cert.get("parent_basis_operation_ids", _SENTINEL_MISSING),
        unique=True,
    )
    canonical_setting_status = (
        str(cert.get("canonical_setting_status"))
        if isinstance(cert.get("canonical_setting_status"), str)
        else "not_evaluated"
    )
    canonical_setting_source = (
        str(cert.get("canonical_setting_source"))
        if isinstance(cert.get("canonical_setting_source"), str)
        else ""
    )
    canonical_hall_numbers = _normalize_strict_int_list(
        cert.get("canonical_hall_numbers", _SENTINEL_MISSING), unique=True,
    )
    canonical_candidate_hall_numbers = _normalize_strict_int_list(
        cert.get("canonical_candidate_hall_numbers", _SENTINEL_MISSING),
        unique=True,
    )
    centering_coset_count = _opt_int(cert.get("centering_coset_count"))
    primitive_conventional_index = _opt_int(
        cert.get("primitive_conventional_index")
    )
    expanded_parent_operation_count = _opt_int(
        cert.get("expanded_parent_operation_count")
    )
    matched_expanded_operations = _opt_int(
        cert.get("matched_expanded_operations")
    )
    centered_affine_operation_map = _normalize_centered_affine_operation_map(
        cert.get("centered_affine_operation_map", _SENTINEL_MISSING)
    )
    affine_unmatched_centered_pairs = _normalize_unmatched_centered_pairs(
        cert.get("unmatched_centered_operation_pairs", _SENTINEL_MISSING)
    )

    transform_key = _normalize_transform(
        cert.get("parent_to_standard_direct_transform")
    )
    origin_shift_key = _normalize_origin_shift(
        cert.get("origin_shift_fractional")
    )
    centering_vectors_key = _normalize_centering_vectors(
        cert.get("centering_vectors")
    )

    return _SettingIdentity(
        sg_number=sg_number,
        sg_symbol=sg_symbol,
        hall_number=hall_number,
        hall_symbol=hall_symbol,
        transform_key=transform_key,
        origin_shift_key=origin_shift_key,
        centering_type=centering_type,
        centering_vectors_key=centering_vectors_key,
        primitive_conventional_relation=primitive_conventional_relation,
        transform_provenance=transform_provenance,
        validation_status=validation_status,
        operation_mapping_status=operation_mapping_status,
        affine_validation_status=affine_validation_status,
        affine_matched_operations=affine_matched_operations,
        affine_total_operations=affine_total_operations,
        affine_mismatch_count=affine_mismatch_count,
        affine_missing_ingredients=affine_missing_ingredients,
        affine_standard_setting_op_count=std_op_count,
        affine_operation_map=affine_operation_map,
        affine_unmatched_parents=affine_unmatched_parents,
        affine_unused_std=affine_unused_std,
        affine_required_operation_ids=affine_required_operation_ids,
        affine_required_op_count=req_op_count,
        operation_closure_validated=operation_closure_validated,
        canonical_setting_status=canonical_setting_status,
        canonical_setting_source=canonical_setting_source,
        canonical_hall_numbers=canonical_hall_numbers,
        canonical_candidate_hall_numbers=canonical_candidate_hall_numbers,
        centering_coset_count=centering_coset_count,
        primitive_conventional_index=primitive_conventional_index,
        expanded_parent_operation_count=expanded_parent_operation_count,
        matched_expanded_operations=matched_expanded_operations,
        centered_affine_operation_map=centered_affine_operation_map,
        affine_unmatched_centered_pairs=affine_unmatched_centered_pairs,
        standard_operation_closure_validated=(
            standard_operation_closure_validated
        ),
    )


def _certificate_identity(
    cands: list[dict[str, object]],
) -> dict[str, object]:
    """Build certificate-identity dict from merged candidates.

    All candidates in one group share the same ``_SettingIdentity``, so the
    canonical identity is taken from the first candidate.  The complete
    setting-level fields are serialized so that the promotion function can
    validate affine evidence.
    """
    fps = [_certificate_fingerprint(c) for c in cands]
    fp0 = fps[0] if fps else None
    hall_numbers = sorted({fp.hall_number for fp in fps if fp.hall_number})
    hall_symbols = sorted({fp.hall_symbol for fp in fps if fp.hall_symbol})
    validation_statuses = sorted({fp.validation_status for fp in fps})
    centering_types = sorted({fp.centering_type for fp in fps if fp.centering_type})
    distinct = len({fp._hash for fp in fps})

    result: dict[str, object] = {
        "hall_numbers": hall_numbers,
        "hall_symbols": hall_symbols,
        "centering_types": centering_types,
        "certificate_validation_statuses": validation_statuses,
        "any_unresolved": (
            "unresolved" in validation_statuses
            or "not_evaluated" in validation_statuses
            or "rejected" in validation_statuses
        ),
        "distinct_setting_identities": distinct,
    }

    # Serialize the canonical _SettingIdentity fields.
    if fp0 is not None:
        result["sg_number"] = fp0.sg_number
        result["sg_symbol"] = fp0.sg_symbol
        result["hall_number"] = fp0.hall_number
        result["hall_symbol"] = fp0.hall_symbol
        result["centering_type"] = fp0.centering_type
        result["primitive_conventional_relation"] = (
            fp0.primitive_conventional_relation
        )
        result["transform_provenance"] = fp0.transform_provenance
        result["validation_status"] = fp0.validation_status
        result["operation_mapping_status"] = fp0.operation_mapping_status
        result["affine_validation_status"] = fp0.affine_validation_status
        result["affine_matched_operations"] = fp0.affine_matched_operations
        result["affine_total_operations"] = fp0.affine_total_operations
        result["affine_mismatch_count"] = fp0.affine_mismatch_count
        result["affine_missing_ingredients"] = (
            list(fp0.affine_missing_ingredients)
            if fp0.affine_missing_ingredients is not None else None
        )
        result["affine_standard_setting_op_count"] = \
            fp0.affine_standard_setting_op_count
        result["affine_operation_map"] = (
            {str(key): value for key, value in fp0.affine_operation_map}
            if fp0.affine_operation_map is not None else None
        )
        result["affine_required_operation_ids"] = (
            list(fp0.affine_required_operation_ids)
            if fp0.affine_required_operation_ids is not None else None
        )
        result["affine_required_op_count"] = fp0.affine_required_op_count
        result["affine_unmatched_parent_operations"] = (
            list(fp0.affine_unmatched_parents)
            if fp0.affine_unmatched_parents is not None else None
        )
        result["affine_unused_standard_operation_indices"] = (
            list(fp0.affine_unused_std)
            if fp0.affine_unused_std is not None else None
        )
        result["operation_closure_validated"] = fp0.operation_closure_validated
        result["canonical_setting_status"] = fp0.canonical_setting_status
        result["canonical_setting_source"] = fp0.canonical_setting_source
        result["canonical_hall_numbers"] = (
            list(fp0.canonical_hall_numbers)
            if fp0.canonical_hall_numbers is not None else None
        )
        result["canonical_candidate_hall_numbers"] = (
            list(fp0.canonical_candidate_hall_numbers)
            if fp0.canonical_candidate_hall_numbers is not None else None
        )
        result["centering_coset_count"] = fp0.centering_coset_count
        result["primitive_conventional_index"] = (
            fp0.primitive_conventional_index
        )
        result["expanded_parent_operation_count"] = (
            fp0.expanded_parent_operation_count
        )
        result["matched_expanded_operations"] = fp0.matched_expanded_operations
        result["centered_affine_operation_map"] = (
            [
                {
                    "parent_operation_id": parent_id,
                    "centering_coset_index": coset_index,
                    "standard_operation_index": standard_index,
                }
                for parent_id, coset_index, standard_index
                in fp0.centered_affine_operation_map
            ]
            if fp0.centered_affine_operation_map is not None else None
        )
        result["affine_unmatched_centered_operation_pairs"] = (
            [
                {
                    "parent_operation_id": parent_id,
                    "centering_coset_index": coset_index,
                }
                for parent_id, coset_index
                in fp0.affine_unmatched_centered_pairs
            ]
            if fp0.affine_unmatched_centered_pairs is not None else None
        )
        result["standard_operation_closure_validated"] = (
            fp0.standard_operation_closure_validated
        )
        if fp0.transform_key is not None:
            result["normalized_direct_transform"] = list(
                list(row) for row in fp0.transform_key
            )
        if fp0.origin_shift_key is not None:
            result["normalized_origin_shift"] = list(fp0.origin_shift_key)
        if fp0.centering_vectors_key is not None:
            result["normalized_centering_vectors"] = [
                list(v) for v in fp0.centering_vectors_key
            ]

    return result
