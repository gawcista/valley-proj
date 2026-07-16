"""Standard-setting HSP k-coordinate mapping for valley-projected subspace SG.

When a valley-projected subspace space group uses a conventional or
centered setting that differs from the parent moire primitive reciprocal
basis, the parent-setting k_frac may not directly match any
irreptables standard-setting HSP coordinate.

This module builds a ``standard_setting_certificate`` that records the
crystallographic equivalence between the parent moire basis and the
Bilbao/irreptables standard setting.  HSP labels are trusted only when
the certificate validates the coordinate convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Standard-setting certificate data model
# ---------------------------------------------------------------------------

@dataclass
class StandardSettingCertificate:
    """Affine space-group equivalence certificate for conventional setting.

    Records the evidence that links the parent moire reciprocal basis
    to the irreptables standard-setting reciprocal basis.  HSP labels
    may be trusted only when ``validation_status == "validated"``.
    """

    # --- Source identity ---
    subspace_sg_number: int | None = None
    subspace_sg_symbol: str | None = None
    hall_number: int | None = None
    hall_symbol: str | None = None
    standard_setting_source: str = "spglib.get_spacegroup_type_from_symmetry"
    canonical_setting_status: str = "not_evaluated"
    canonical_setting_source: str | None = None
    canonical_hall_numbers: list[int] | None = None
    canonical_candidate_hall_numbers: list[int] | None = None

    # --- Validation ---
    validation_status: str = "not_evaluated"
    """One of ``"validated"``, ``"unresolved"``, ``"rejected"``."""

    # --- Operation mapping ---
    parent_basis_operation_ids: list[int] | None = None
    standard_setting_operation_count: int | None = None
    operation_mapping_status: str = "not_attempted"

    # --- Basis transform ---
    parent_to_standard_direct_transform: list[list[float]] | None = None
    """3×3 matrix T: x_std = T · x_parent (direct space)."""
    reciprocal_k_transform_rule: str = (
        "k_std = T^(-T) · k_parent"
    )
    transform_provenance: str = "not_provided"

    # --- Origin shift ---
    origin_shift_fractional: list[float] | None = None
    """Fractional origin shift when available; None if unresolved."""
    origin_shift_status: str = "unavailable"

    # --- Primitive-conventional relation ---
    primitive_conventional_relation: str | None = None
    """Describes how the parent primitive cell relates to the standard
    conventional cell: ``"direct_coordinate_match"`` when parent k_frac
    directly matches irreptables HSP coordinates, ``"explicit_transform"``
    when an explicit T matrix is used, ``"operation_basis_reconstruction"``
    when a transform is derived from operation content,
    ``"centered_unresolved"`` when centering vectors are unavailable,
    ``"not_applicable"``."""

    # --- Centering ---
    centering_type: str | None = None
    """P, C, I, F, R from Hall symbol first character."""
    centering_status: str = "not_evaluated"
    """One of ``"primitive_direct_match"``, ``"centered_unresolved"``,
    ``"not_evaluated"``."""
    centering_vectors: list[list[float]] | None = None
    """Conventional centering cosets when known; None if unavailable."""
    centering_coset_count: int | None = None
    primitive_conventional_index: int | None = None

    # --- Affine translation validation ---
    translation_validation_status: str = "not_attempted"
    """One of ``"passed"``, ``"failed"``, ``"unresolved"``,
    ``"not_attempted"``."""
    total_parent_operations: int | None = None
    """Number of parent VP operations with rotation+translation data."""
    matched_affine_operations: int | None = None
    """Number whose transformed affine content matches standard setting."""
    mismatched_translation_count: int | None = None
    """Number of parent VP operations with inconsistent affine translation."""
    mismatched_translations: list[dict[str, object]] = field(default_factory=list)
    """Short diagnostic sample for failed affine translation matches."""

    # --- Affine missing ingredients ---
    missing_affine_ingredients: list[str] | None = None
    """List of missing affine ingredients when unresolved, e.g.
    ``"conventional_centering_vectors"``, ``"origin_shift_fractional"``,
    ``"standard_setting_translations"``, ``"direct_lattice_transform"``."""

    # --- Operation closure ---
    operation_closure_validated: bool | None = None
    """Whether the VP operations form a closed set under multiplication
    (validated via spglib standard match or explicit check)."""

    # --- Operation bijection ---
    affine_operation_map: dict[str, int] | None = None
    """Parent operation ID -> standard operation index (bijection)."""
    unmatched_parent_operations: list[dict[str, object]] | None = None
    """Parent operations that could not be matched to any standard op."""
    unused_standard_operation_indices: list[int] | None = None
    """Standard operation indices not matched by any parent op."""
    required_operation_id_count: int | None = None
    """Number of distinct required VP operation IDs."""
    centered_affine_operation_map: list[dict[str, int]] | None = None
    """Expanded centered operation map keyed by parent ID and coset index."""
    unmatched_centered_operation_pairs: list[dict[str, int]] | None = None
    expanded_parent_operation_count: int | None = None
    matched_expanded_operations: int | None = None
    standard_operation_closure_validated: bool | None = None

    # --- Blockers ---
    unresolved_reason: str | None = None
    """Human-readable explanation when validation_status != "validated"."""

    # --- k-point ---
    parent_k_frac: list[float] | None = None
    resolved_hsp_label: str | None = None
    """Source HSP label, only when validated."""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict, omitting None defaults.

        Successful empty affine audit collections are explicit evidence and
        are preserved.  ``None`` remains absent/unknown.
        """
        d: dict[str, object] = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            # Preserve empty lists/dicts that are explicit evidence.
            if isinstance(value, list) and not value:
                if key in (
                    "missing_affine_ingredients",
                    "unmatched_parent_operations",
                    "unmatched_centered_operation_pairs",
                    "unused_standard_operation_indices",
                ):
                    d[key] = []
                    continue
                continue
            if isinstance(value, dict) and not value:
                d[key] = value
                continue
            d[key] = value
        return d


@dataclass
class StandardSettingTransformCandidate:
    """Standard-setting transform candidate for validation.

    Records a candidate parent-to-standard direct-lattice transform
    plus the centering/affine evidence needed to validate it.  The
    transform is accepted only when all field-level gates pass.
    """

    # --- Transform ---
    direct_transform: np.ndarray | None = None
    """3×3 matrix T: x_std = T · x_parent (direct space)."""
    reciprocal_k_rule: str = "k_std = T^(-T) · k_parent"
    transform_provenance: str = "not_provided"

    # --- Centering ---
    centering_type: str | None = None
    """P, C, I, F, R from Hall symbol."""
    centering_vectors: list[list[float]] | None = None
    """Centering vectors/cosets when known."""
    centering_status: str = "not_evaluated"

    # --- Origin shift ---
    origin_shift_fractional: list[float] | None = None

    # --- Validation ---
    validation_status: str = "not_evaluated"
    """One of ``"validated"``, ``"unresolved"``, ``"rejected"``."""

    # --- Operation mapping ---
    operation_mapping_status: str = "not_attempted"
    affine_validation_status: str = "not_attempted"
    matched_affine_operations: int | None = None
    total_affine_operations: int | None = None

    # --- Blockers ---
    unresolved_reason: str | None = None
    missing_ingredients: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {}
        for key, value in asdict(self).items():
            if key == "direct_transform":
                if value is not None:
                    d[key] = np.asarray(value, dtype=float).tolist()
            elif value is not None and value != []:
                d[key] = value
        return d


def _build_transform_candidate(
    *,
    standard_match: dict[str, object] | None,
    parent_to_standard_direct_transform: np.ndarray | None = None,
    origin_shift_fractional: np.ndarray | None = None,
    transform_provenance: str = "not_provided",
    operation_mapping_status: str = "not_attempted",
    affine_result: dict[str, object] | None = None,
    unresolved_reason: str | None = None,
) -> StandardSettingTransformCandidate:
    """Build a standard-setting transform candidate from available evidence.

    The candidate records the transform matrix, centering data, origin
    shift, and validation provenance.  It is accepted only when all
    validation gates (operation mapping, affine translation) pass.
    """
    c = StandardSettingTransformCandidate()

    if isinstance(standard_match, dict):
        v = standard_match.get("hall_symbol")
        if isinstance(v, str) and v:
            c.centering_type = _hall_centering_symbol(str(v)) or None
            hall_number = standard_match.get("hall_number")
            coset_evidence = _centering_cosets_from_hall_database(hall_number)
            if coset_evidence.get("status") == "passed":
                cv = coset_evidence.get("centering_cosets")
                if isinstance(cv, list) and len(cv) > 1:
                    c.centering_vectors = [list(vec) for vec in cv]

    if parent_to_standard_direct_transform is not None:
        T = np.asarray(parent_to_standard_direct_transform, dtype=float)
        if T.shape == (3, 3) and np.all(np.isfinite(T)):
            c.direct_transform = T
            c.transform_provenance = transform_provenance

    if origin_shift_fractional is not None:
        o = np.asarray(origin_shift_fractional, dtype=float)
        if o.shape == (3,):
            c.origin_shift_fractional = o.tolist()

    c.operation_mapping_status = operation_mapping_status

    if affine_result is not None:
        c.affine_validation_status = str(affine_result.get("status", ""))
        v = affine_result.get("matched_affine_operations")
        if isinstance(v, int):
            c.matched_affine_operations = int(v)
        v = affine_result.get("total_parent_operations")
        if isinstance(v, int):
            c.total_affine_operations = int(v)
        missing = affine_result.get("missing_ingredients")
        if isinstance(missing, list):
            c.missing_ingredients = list(missing)

    if unresolved_reason and c.affine_validation_status == "failed":
        c.validation_status = "rejected"
        c.unresolved_reason = str(unresolved_reason)
    elif unresolved_reason:
        c.validation_status = "unresolved"
        c.unresolved_reason = str(unresolved_reason)
    elif (
        c.direct_transform is not None
        and c.operation_mapping_status == "operation_basis_verification_passed"
        and c.affine_validation_status in ("passed",)
    ):
        c.validation_status = "validated"
    elif c.direct_transform is not None:
        c.validation_status = "rejected"
        c.unresolved_reason = (
            "transform candidate rejected: operation mapping or affine "
            "validation did not pass"
        )
    else:
        c.validation_status = "unresolved"

    if c.centering_type in ("A", "B", "C", "I", "F", "R"):
        c.centering_status = "centered_unresolved"
        if c.centering_vectors is None:
            if "conventional_centering_vectors" not in c.missing_ingredients:
                c.missing_ingredients.append("conventional_centering_vectors")

    return c


def _attach_transform_candidate(
    *,
    provenance: dict[str, object],
    certificate: StandardSettingCertificate,
    candidate: StandardSettingTransformCandidate,
) -> None:
    """Attach transform-candidate provenance to resolver output."""
    if candidate.centering_vectors is not None:
        certificate.centering_vectors = candidate.centering_vectors
    provenance["transform_candidate"] = candidate.to_dict()


def _validate_affine_operation_equivalence(
    *,
    vp_operations: list[dict[str, object]] | None,
    vp_operation_ids: object,
    standard_match: dict[str, object],
    parent_to_standard_direct_transform: np.ndarray | None = None,
    origin_shift_fractional: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Validate affine operation-group bijection.

    Builds a one-to-one parent-to-standard operation map under the basis
    transform T for the complete valley-preserving operation set that defines
    the valley-projected subspace space group.  Every required operation must
    map to a distinct standard operation and every primitive standard
    operation must be used exactly once.  Closure is checked only on that
    complete selected set, never on the full parent moire operation set or an
    HSP-local ``G_k^(a)`` representation subset.

    Returns a status dict with ``status`` (``"passed"``, ``"failed"``, or
    ``"unresolved"``), bijection evidence (operation map, unmatched/unused
    indices), mismatch diagnostics, missing ingredients, and closure status.
    """
    result: dict[str, object] = {
        "status": "not_attempted",
        "missing_ingredients": [],
        "required_operation_ids": None,
        "operation_map": None,
        "centered_operation_map": None,
        "unmatched_parent_operations": None,
        "unmatched_centered_operation_pairs": None,
        "unused_standard_operation_indices": None,
    }

    # ---- A. Operation-ID coverage ----
    if not isinstance(vp_operation_ids, list) or any(
        not isinstance(i, int) or isinstance(i, bool)
        for i in vp_operation_ids
    ):
        result["status"] = "failed"
        result["missing_ingredients"].append(
            "malformed_required_operation_ids"
        )
        return result
    required_ids = list(vp_operation_ids)
    required_set = set(required_ids)
    if len(required_ids) != len(required_set):
        result["status"] = "failed"
        result["missing_ingredients"].append("duplicate_required_operation_id")
        return result
    result["required_operation_ids"] = required_ids
    result["required_operation_id_count"] = len(required_ids)
    if not required_set:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("vp_operation_data")
        return result
    if not vp_operations:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("vp_operation_data")
        return result

    # Index by ID, checking for duplicates and malformed entries.
    # Operations outside the required valley-preserving subset are silently
    # excluded (valley-changing parent operations are expected and must not
    # block the selected subgroup validation).
    by_id: dict[int, dict[str, object]] = {}
    duplicate_ids: list[int] = []
    missing_required: list[int] = []
    malformed: list[dict[str, object]] = []
    excluded_extra: list[int] = []
    seen_required_ids: set[int] = set()
    for op in vp_operations:
        if not isinstance(op, dict):
            continue
        oid = op.get("operation_id")
        if not _is_exact_operation_id(oid):
            continue
        if oid not in required_set:
            excluded_extra.append(oid)
            continue
        if oid in seen_required_ids:
            duplicate_ids.append(oid)
            continue
        seen_required_ids.add(oid)
        rot = op.get("rotation_frac")
        trans = op.get("translation_frac")
        if rot is None or trans is None:
            malformed.append({"operation_id": oid,
                              "reason": "missing_rotation_or_translation"})
            continue
        by_id[oid] = op

    for rid in required_ids:
        if rid not in by_id:
            missing_required.append(rid)

    if excluded_extra:
        result["excluded_parent_operation_ids"] = excluded_extra
    if duplicate_ids:
        result["status"] = "failed"
        result["missing_ingredients"].append("duplicate_detected_operation_id")
        result["duplicate_detected_operation_ids"] = duplicate_ids
    if malformed:
        result["missing_ingredients"].append("malformed_detected_operations")
        result["malformed_detected_operations"] = malformed
    if missing_required:
        result["status"] = "failed"
        result["missing_ingredients"].append("missing_required_operation_ids")
        result["missing_required_operation_ids"] = missing_required

    if result["status"] == "failed":
        result["mismatched_translation_count"] = 0
        return result

    # Derive the ordered list that bijection analysis consumes.
    ops_with_translations: list[dict[str, object]] = [
        by_id[rid] for rid in required_ids if rid in by_id
    ]
    if not ops_with_translations and not missing_required:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("parent_translation_frac")
        return result

    # ---- B. Hall / spglib standard operations ----
    hall_number = standard_match.get("hall_number")
    if not isinstance(hall_number, int) or isinstance(hall_number, bool):
        if result["status"] not in ("failed",):
            result["status"] = "unresolved"
        result["missing_ingredients"].append("standard_setting_translations")
        return result
    hall_number = int(hall_number)

    sg_number = standard_match.get("number")
    _hall_ok, _hall_blocker = _validate_hall_sg_consistency(
        hall_number=hall_number,
        sg_number=int(sg_number)
        if isinstance(sg_number, int) and not isinstance(sg_number, bool)
        else None,
    )
    if not _hall_ok:
        if result["status"] not in ("failed",):
            result["status"] = "failed"
        result["missing_ingredients"].append("hall_number")
        result["hall_sg_consistency"] = _hall_blocker
        return result

    try:
        import spglib
        std_sym = spglib.get_symmetry_from_database(hall_number)
    except Exception:
        std_sym = None
    if std_sym is None:
        if result["status"] not in ("failed",):
            result["status"] = "unresolved"
        result["missing_ingredients"].append("spglib_database_access")
        return result

    std_rotations = [np.asarray(r, dtype=float) for r in std_sym["rotations"]]
    std_translations = [np.asarray(t, dtype=float) for t in std_sym["translations"]]
    std_op_count = len(std_translations)
    result["standard_setting_operation_count"] = std_op_count

    # Centering evidence is derived from the Hall operation set itself.
    hall_symbol = str(standard_match.get("hall_symbol", "") or "")
    centering_evidence = _centering_cosets_from_hall_database(
        hall_number, tolerance=tolerance,
    )
    if centering_evidence.get("status") != "passed":
        result["status"] = (
            "failed" if centering_evidence.get("status") == "failed"
            else "unresolved"
        )
        result["missing_ingredients"].append("conventional_centering_vectors")
        result["centering_evidence_reason"] = centering_evidence.get("reason")
        return result
    centering_vectors = [
        np.asarray(vector, dtype=float)
        for vector in centering_evidence["centering_cosets"]
    ]
    centering_index = int(
        centering_evidence["primitive_conventional_index"]
    )
    result["centering_cosets_count"] = len(centering_vectors)
    result["primitive_conventional_index"] = centering_index
    result["standard_operation_closure_validated"] = centering_evidence.get(
        "standard_operation_closure_validated"
    )

    # A parent primitive operation expands over every conventional centering
    # coset.  The expanded order must equal the full Hall operation count.
    parent_op_count = len(ops_with_translations)
    expanded_parent_count = parent_op_count * centering_index
    result["expanded_parent_operation_count"] = expanded_parent_count
    if expanded_parent_count != std_op_count:
        if result["status"] not in ("failed",):
            result["status"] = "failed"
        result["missing_ingredients"].append(
            f"parent_standard_group_order_mismatch: parent has "
            f"{parent_op_count} primitive operations, centering index is "
            f"{centering_index}, expanded parent has {expanded_parent_count}, "
            f"standard has {std_op_count}"
        )
    result["total_parent_operations"] = parent_op_count

    # If already failed on coverage issues, stop.
    if result["status"] in ("failed",) and result["missing_ingredients"]:
        result["mismatched_translation_count"] = 0
        return result

    # ---- C. Transform validation ----
    if parent_to_standard_direct_transform is None:
        if result["status"] not in ("failed",):
            result["status"] = "unresolved"
        result["missing_ingredients"].append("direct_lattice_transform")
        return result
    T = np.asarray(parent_to_standard_direct_transform, dtype=float)
    if T.shape != (3, 3):
        if result["status"] not in ("failed",):
            result["status"] = "unresolved"
        result["missing_ingredients"].append("direct_lattice_transform")
        return result
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        if result["status"] not in ("failed",):
            result["status"] = "failed"
        result["missing_ingredients"].append("singular_transform")
        return result
    determinant = float(np.linalg.det(T))
    expected_abs_determinant = 1.0 / float(centering_index)
    result["transform_determinant"] = determinant
    result["expected_transform_abs_determinant"] = expected_abs_determinant
    if not np.isclose(
        abs(determinant), expected_abs_determinant, atol=tolerance, rtol=0.0,
    ):
        result["status"] = "failed"
        result["missing_ingredients"].append(
            "primitive_conventional_transform_index"
        )
        return result
    if centering_index > 1 and origin_shift_fractional is None:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("origin_shift_fractional")
        return result
    if origin_shift_fractional is not None:
        origin = np.asarray(origin_shift_fractional, dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            result["status"] = "failed"
            result["missing_ingredients"].append("origin_shift_fractional")
            return result
    else:
        origin = np.zeros(3, dtype=float)

    # ---- D. Build expanded parent/coset -> standard-operation bijection ----
    Pair = tuple[int, int]
    candidate_graph: dict[Pair, list[int]] = {}
    for op in ops_with_translations:
        r_p = np.asarray(op["rotation_frac"], dtype=float)
        t_p = np.asarray(op["translation_frac"], dtype=float)
        r_transformed = T @ r_p @ T_inv
        t_base = T @ t_p + origin - r_transformed @ origin
        oid = op.get("operation_id")
        if not _is_exact_operation_id(oid):
            continue
        for coset_index, centering_vector in enumerate(centering_vectors):
            expanded_translation = t_base + centering_vector
            matched_indices = [
                standard_index
                for standard_index, (standard_rotation, standard_translation)
                in enumerate(zip(std_rotations, std_translations))
                if np.allclose(r_transformed, standard_rotation, atol=tolerance)
                and _mod1_norm(
                    expanded_translation - standard_translation
                ) <= tolerance
            ]
            candidate_graph[(oid, coset_index)] = matched_indices

    # Deterministic augmenting-path bipartite matching (Hopcroft-Karp style,
    # single-path DFS for simplicity).  Required IDs are processed in sorted
    # order for deterministic output.
    pair_match: dict[Pair, int] = {}
    std_match: dict[int, Pair] = {}
    parent_candidates = sorted(by_id)
    pair_candidates = [
        (parent_id, coset_index)
        for parent_id in parent_candidates
        for coset_index in range(centering_index)
    ]

    # Build adjacency for augmenting-path.
    adj: dict[Pair, list[int]] = {
        pair: list(candidate_graph.get(pair, [])) for pair in pair_candidates
    }

    def _dfs_augment(u: Pair, visited: set[int]) -> bool:
        for v in adj.get(u, []):
            if v in visited:
                continue
            visited.add(v)
            if v not in std_match or _dfs_augment(std_match[v], visited):
                pair_match[u] = v
                std_match[v] = u
                return True
        return False

    # Match in sorted parent order for deterministic result.
    for pair in pair_candidates:
        _dfs_augment(pair, set())

    unmatched_pairs = [
        {
            "parent_operation_id": parent_id,
            "centering_coset_index": coset_index,
        }
        for parent_id, coset_index in pair_candidates
        if (parent_id, coset_index) not in pair_match
    ]
    unmatched_parents: list[dict[str, object]] = []
    for pid in parent_candidates:
        if any(
            (pid, coset_index) not in pair_match
            for coset_index in range(centering_index)
        ):
            op_data = by_id.get(pid, {})
            unmatched_parents.append({
                "operation_id": pid,
                "parent_translation_frac":
                    op_data.get("translation_frac"),
            })

    unused_std = sorted(
        set(range(std_op_count)) - set(std_match.keys())
    )
    if centering_index == 1:
        result["operation_map"] = {
            str(parent_id): int(pair_match[(parent_id, 0)])
            for parent_id in parent_candidates
            if (parent_id, 0) in pair_match
        }
        result["centered_operation_map"] = None
    else:
        result["operation_map"] = None
        result["centered_operation_map"] = [
            {
                "parent_operation_id": parent_id,
                "centering_coset_index": coset_index,
                "standard_operation_index": int(standard_index),
            }
            for (parent_id, coset_index), standard_index
            in sorted(pair_match.items())
        ]
    result["unmatched_parent_operations"] = unmatched_parents
    result["unmatched_centered_operation_pairs"] = unmatched_pairs
    result["unused_standard_operation_indices"] = unused_std
    result["matched_expanded_operations"] = len(pair_match)
    result["matched_affine_operations"] = sum(
        all(
            (parent_id, coset_index) in pair_match
            for coset_index in range(centering_index)
        )
        for parent_id in parent_candidates
    )
    result["total_parent_operations"] = parent_op_count

    # ---- E. Determine status ----
    if unmatched_pairs or unused_std:
        result["mismatched_translation_count"] = max(
            len(unmatched_pairs), len(unused_std),
        )
        if unmatched_parents:
            result["mismatched_translations"] = unmatched_parents[:3]
        if result["status"] not in ("failed",):
            result["status"] = "failed"
    else:
        result["mismatched_translation_count"] = 0
        if result["status"] not in ("failed",):
            result["status"] = "passed"

    # ---- F. Affine group closure of the selected subgroup ----
    # Closure failure must override a prior passed status so a trusted HSP
    # label is never emitted on a non-closed set.
    _validate_affine_group_closure(by_id, result)
    if result.get("operation_closure_validated") is not True:
        if result["status"] == "passed":
            result["status"] = "failed"
            result["missing_ingredients"].append("affine_group_not_closed")

    if result["status"] == "passed" and not (
        result.get("missing_ingredients") == []
        and result.get("unmatched_parent_operations") == []
        and result.get("unmatched_centered_operation_pairs") == []
        and result.get("unused_standard_operation_indices") == []
        and result.get("operation_closure_validated") is True
        and result.get("standard_operation_closure_validated") is True
        and result.get("matched_expanded_operations") == std_op_count
        and (
            centering_index > 1
            and isinstance(result.get("centered_operation_map"), list)
            and len(result["centered_operation_map"]) == std_op_count
            or centering_index == 1
            and isinstance(result.get("operation_map"), dict)
            and len(result["operation_map"]) == len(required_ids)
        )
    ):
        result["status"] = "failed"
        if "incomplete_affine_success_evidence" not in result["missing_ingredients"]:
            result["missing_ingredients"].append(
                "incomplete_affine_success_evidence"
            )

    return result


def _validate_affine_group_closure(
    by_id: dict[int, dict[str, object]], result: dict[str, object],
) -> None:
    """Validate the detected parent affine operation set is closed under
    composition modulo lattice translations.

    Checks: a unique identity exists, every operation has an inverse in the
    set, and pairwise products stay in the set (mod lattice).  Sets
    ``operation_closure_validated`` to True or False in *result*.
    """
    if not by_id:
        result["operation_closure_validated"] = False
        return
    ops = list(by_id.values())
    try:
        rotations = [np.asarray(op["rotation_frac"], dtype=float) for op in ops]
        translations = [np.asarray(op["translation_frac"], dtype=float) for op in ops]
    except (KeyError, ValueError):
        result["operation_closure_validated"] = False
        return

    n = len(ops)
    # Identity: ⟨identity rotation, zero translation-mod-1⟩ in the set
    def _is_identity(idx: int) -> bool:
        r = rotations[idx]
        return np.allclose(r, np.eye(3, dtype=float), atol=1e-6) \
            and _mod1_norm(translations[idx]) < 1e-6

    identity_indices = [i for i in range(n) if _is_identity(i)]
    if len(identity_indices) != 1:
        result["operation_closure_validated"] = False
        result["affine_closure_diagnostic"] = (
            f"expected 1 identity; found {len(identity_indices)}"
        )
        return
    e_idx = identity_indices[0]

    # Inverse: for every op g, find h such that g*h = identity (mod lattice).
    for i in range(n):
        r_i = rotations[i].astype(float)
        t_i = translations[i].astype(float)
        found_inv = False
        for j in range(n):
            r_j = rotations[j].astype(float)
            t_j = translations[j].astype(float)
            # g · h = (R_i · R_j,  t_i + R_i · t_j) mod 1
            r_prod = r_i @ r_j
            t_prod = t_i + r_i @ t_j
            if np.allclose(r_prod, np.eye(3), atol=1e-6) \
                    and _mod1_norm(t_prod) < 1e-6:
                found_inv = True
                break
        if not found_inv:
            result["operation_closure_validated"] = False
            result["affine_closure_diagnostic"] = (
                f"operation {i} has no inverse in the detected set"
            )
            return

    # Closure under multiplication: R_i * R_j, T_i + R_i * T_j (mod 1)
    for i in range(n):
        for j in range(n):
            r_ij = rotations[i] @ rotations[j]
            t_ij = translations[i] + rotations[i] @ translations[j]
            # t_ij modulo 1
            t_ij_mod = t_ij - np.rint(t_ij)
            # Heuristic: only check that each product has SOME element with
            # matching rotation and a translation within tolerance.
            found_prod = False
            for k in range(n):
                if np.allclose(rotations[k], r_ij, atol=1e-6) \
                        and _mod1_norm(translations[k] - t_ij_mod) < 1e-6:
                    found_prod = True
                    break
            if not found_prod:
                result["operation_closure_validated"] = False
                result["affine_closure_diagnostic"] = (
                    f"product of operations {i} and {j} is not in the set"
                )
                return

    result["operation_closure_validated"] = True


def _mod1_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec - np.rint(vec)))


def _apply_affine_validation_to_certificate(
    cert: StandardSettingCertificate,
    affine_result: dict[str, object],
) -> None:
    """Copy affine validation diagnostics including bijection evidence into
    the certificate."""
    cert.translation_validation_status = str(
        affine_result.get("status", "unresolved")
    )
    cert.total_parent_operations = affine_result.get("total_parent_operations")
    cert.matched_affine_operations = affine_result.get(
        "matched_affine_operations"
    )
    cert.standard_setting_operation_count = affine_result.get(
        "standard_setting_operation_count"
    )
    missing = affine_result.get("missing_ingredients")
    cert.missing_affine_ingredients = (
        list(missing)
        if isinstance(missing, list)
        and all(isinstance(item, str) for item in missing)
        else None
    )
    cert.mismatched_translation_count = affine_result.get(
        "mismatched_translation_count"
    )
    mismatched = affine_result.get("mismatched_translations")
    cert.mismatched_translations = (
        list(mismatched) if isinstance(mismatched, list) else []
    )
    cert.operation_closure_validated = affine_result.get(
        "operation_closure_validated"
    )
    # Bijection evidence.
    op_map = affine_result.get("operation_map")
    if isinstance(op_map, dict) and all(
        isinstance(k, str)
        and isinstance(v, int)
        and not isinstance(v, bool)
        for k, v in op_map.items()
    ):
        cert.affine_operation_map = {str(k): int(v) for k, v in op_map.items()}
    unmatched = affine_result.get("unmatched_parent_operations")
    cert.unmatched_parent_operations = (
        [dict(item) for item in unmatched]
        if isinstance(unmatched, list)
        and all(isinstance(item, dict) for item in unmatched)
        else None
    )
    unused = affine_result.get("unused_standard_operation_indices")
    cert.unused_standard_operation_indices = (
        list(unused)
        if isinstance(unused, list)
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in unused
        )
        else None
    )
    required_ids = affine_result.get("required_operation_ids")
    cert.parent_basis_operation_ids = (
        list(required_ids)
        if isinstance(required_ids, list)
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in required_ids
        )
        and len(required_ids) == len(set(required_ids))
        else None
    )
    cert.required_operation_id_count = affine_result.get("required_operation_id_count")
    cert.centering_coset_count = affine_result.get("centering_cosets_count")
    cert.primitive_conventional_index = affine_result.get(
        "primitive_conventional_index"
    )
    cert.expanded_parent_operation_count = affine_result.get(
        "expanded_parent_operation_count"
    )
    cert.matched_expanded_operations = affine_result.get(
        "matched_expanded_operations"
    )
    cert.standard_operation_closure_validated = affine_result.get(
        "standard_operation_closure_validated"
    )
    centered_map = affine_result.get("centered_operation_map")
    cert.centered_affine_operation_map = (
        [dict(row) for row in centered_map]
        if isinstance(centered_map, list)
        and all(
            isinstance(row, dict)
            and set(row) == {
                "parent_operation_id",
                "centering_coset_index",
                "standard_operation_index",
            }
            and all(
                isinstance(row[key], int) and not isinstance(row[key], bool)
                for key in row
            )
            for row in centered_map
        )
        else None
    )
    unmatched_pairs = affine_result.get("unmatched_centered_operation_pairs")
    cert.unmatched_centered_operation_pairs = (
        [dict(row) for row in unmatched_pairs]
        if isinstance(unmatched_pairs, list)
        and all(isinstance(row, dict) for row in unmatched_pairs)
        else None
    )


def _affine_failure_blocker(
    affine_result: dict[str, object],
    *,
    source: str,
) -> str:
    """Build a blocker for a failed affine operation equivalence check."""
    hall_blocker = affine_result.get("hall_sg_consistency")
    if hall_blocker:
        return (
            "standard_setting_hsp_mapping_unresolved: "
            f"{source} rejected because {hall_blocker}."
        )
    count = affine_result.get("mismatched_translation_count", "?")
    return (
        "standard_setting_hsp_mapping_unresolved: "
        f"{source} rejected because affine operation validation failed: "
        f"{count} parent valley-preserving translation(s) did not match "
        "standard-setting translations after basis transformation."
    )


def build_standard_setting_certificate(
    *,
    standard_match: dict[str, object] | None = None,
    validation_status: str = "not_evaluated",
    unresolved_reason: str | None = None,
    parent_basis_operation_ids: list[int] | None = None,
    parent_to_standard_direct_transform: np.ndarray | None = None,
    origin_shift_fractional: np.ndarray | None = None,
    transform_provenance: str = "not_provided",
    parent_k_frac: np.ndarray | None = None,
    resolved_hsp_label: str | None = None,
    transform_candidate: StandardSettingTransformCandidate | None = None,
) -> StandardSettingCertificate:
    """Build a standard-setting certificate from available evidence."""
    cert = StandardSettingCertificate()

    if isinstance(standard_match, dict):
        v = standard_match.get("number")
        if isinstance(v, int) and not isinstance(v, bool):
            cert.subspace_sg_number = int(v)
        v = standard_match.get("international_short")
        if isinstance(v, str) and v:
            cert.subspace_sg_symbol = str(v)
        v = standard_match.get("hall_number")
        if isinstance(v, int) and not isinstance(v, bool):
            cert.hall_number = int(v)
        v = standard_match.get("hall_symbol")
        if isinstance(v, str) and v:
            cert.hall_symbol = str(v)
            cert.centering_type = _hall_centering_symbol(str(v)) or None
        canonical_status = standard_match.get("canonical_setting_status")
        if isinstance(canonical_status, str) and canonical_status:
            cert.canonical_setting_status = canonical_status
        canonical_source = standard_match.get("canonical_setting_source")
        if isinstance(canonical_source, str) and canonical_source:
            cert.canonical_setting_source = canonical_source
        canonical_halls = standard_match.get("canonical_hall_numbers")
        if isinstance(canonical_halls, list) and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in canonical_halls
        ):
            cert.canonical_hall_numbers = list(canonical_halls)
        candidate_halls = standard_match.get("canonical_candidate_hall_numbers")
        if isinstance(candidate_halls, list) and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in candidate_halls
        ):
            cert.canonical_candidate_hall_numbers = list(candidate_halls)
        hall_number = standard_match.get("hall_number")
        coset_evidence = _centering_cosets_from_hall_database(hall_number)
        if coset_evidence.get("status") == "passed":
            cert.centering_vectors = [
                list(vector) for vector in coset_evidence["centering_cosets"]
            ]
            cert.centering_coset_count = int(
                coset_evidence["primitive_conventional_index"]
            )
            cert.primitive_conventional_index = int(
                coset_evidence["primitive_conventional_index"]
            )

    if (
        isinstance(parent_basis_operation_ids, list)
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in parent_basis_operation_ids
        )
        and len(parent_basis_operation_ids) == len(set(parent_basis_operation_ids))
    ):
        cert.parent_basis_operation_ids = list(parent_basis_operation_ids)

    if parent_to_standard_direct_transform is not None:
        T = np.asarray(parent_to_standard_direct_transform, dtype=float)
        if T.shape == (3, 3) and np.all(np.isfinite(T)):
            cert.parent_to_standard_direct_transform = T.tolist()
            cert.transform_provenance = transform_provenance

    cert.validation_status = validation_status
    if unresolved_reason:
        cert.unresolved_reason = str(unresolved_reason)

    if parent_k_frac is not None:
        cert.parent_k_frac = np.asarray(parent_k_frac, dtype=float).tolist()
    if resolved_hsp_label is not None:
        cert.resolved_hsp_label = str(resolved_hsp_label)

    if transform_candidate is not None:
        cert_data = cert.to_dict()
        cand_data = transform_candidate.to_dict()
        # Merge non-empty candidate fields into certificate.
        for key in (
            "centering_vectors", "centering_status",
        ):
            if cand_data.get(key):
                cert_data[key] = cand_data[key]
        # Rebuild cert from merged data by setting fields.
        if "centering_vectors" in cand_data:
            cert.centering_vectors = cand_data["centering_vectors"]

    if origin_shift_fractional is not None:
        o = np.asarray(origin_shift_fractional, dtype=float)
        if o.shape == (3,) and np.all(np.isfinite(o)):
            cert.origin_shift_fractional = o.tolist()
            cert.origin_shift_status = "explicit"

    return cert


def _match_table_kpoint_label(
    table: object,
    k_frac: np.ndarray,
    *,
    tolerance: float,
    standard_match: dict[str, object] | None,
) -> str | None:
    """Match an HSP using Hall-derived centering evidence when available."""
    try:
        label = table.match_kpoint_label(k_frac, tolerance=tolerance)
    except TypeError:
        label = table.match_kpoint_label(k_frac)
    if label is not None:
        return label
    if not isinstance(standard_match, dict):
        return None
    evidence = _centering_cosets_from_hall_database(
        standard_match.get("hall_number"), tolerance=tolerance,
    )
    cosets = evidence.get("centering_cosets")
    irreps = getattr(table, "irreps", None)
    if evidence.get("status") != "passed" or not isinstance(cosets, list) \
            or not isinstance(irreps, (list, tuple)):
        return None
    labels_by_coordinate: dict[str, np.ndarray] = {}
    for irrep in irreps:
        irrep_label = getattr(irrep, "kpoint_label", None)
        irrep_k = getattr(irrep, "k_frac", None)
        if isinstance(irrep_label, str) and irrep_k is not None:
            labels_by_coordinate.setdefault(
                irrep_label, np.asarray(irrep_k, dtype=float),
            )
    for irrep_label, table_k in labels_by_coordinate.items():
        delta = np.asarray(k_frac, dtype=float) - table_k
        integer_delta = np.rint(delta)
        if np.linalg.norm(delta - integer_delta) > tolerance:
            continue
        if all(
            abs(float(np.dot(integer_delta, np.asarray(coset, dtype=float)))
                - np.rint(np.dot(integer_delta, np.asarray(coset, dtype=float))))
            <= tolerance
            for coset in cosets
        ):
            return irrep_label
    return None


def resolve_standard_setting_hsp_label(
    *,
    k_frac: np.ndarray,
    table,
    standard_match: dict[str, object] | None,
    tolerance: float = 1e-6,
    lattice_direct_cart: np.ndarray | None = None,
    detected_operations: list[dict[str, object]] | None = None,
    parent_to_standard_direct_transform: np.ndarray | None = None,
    origin_shift_fractional: np.ndarray | None = None,
    transform_provenance: str | None = None,
) -> tuple[str | None, str | None, dict[str, object]]:
    """Resolve a standard-setting Bilbao HSP label for a sampled k-point.

    Parameters
    ----------
    k_frac : np.ndarray
        Sampled k-point fractional coordinates in the parent moire
        reciprocal basis.
    table : StandardIrrepTable
        Loaded irreptables irrep table for the subspace SG.
    standard_match : dict or None
        spglib per-valley standard group match with fields:
        ``number``, ``international_short``, ``hall_number``,
        ``hall_symbol``, ``operation_ids``.
    tolerance : float
        Coordinate comparison tolerance.
    lattice_direct_cart : np.ndarray or None
        Parent moire direct lattice in Cartesian coordinates (3×3).
    detected_operations : list[dict] or None
        Detected symmetry operations, each with ``operation_id``,
        ``rotation_frac``, ``order`` fields.
    parent_to_standard_direct_transform : np.ndarray or None
        Optional explicit parent-to-standard direct-lattice
        transformation matrix (3×3).  Maps direct-space fractional
        coordinates from the parent basis to the standard setting:
        x_std = T * x_parent.  For k-points:
        k_std = T^(-T) * k_parent.
    origin_shift_fractional : np.ndarray or None
        Optional origin shift in standard fractional coordinates.
    transform_provenance : str or None
        Short provenance string for an explicit transform.

    Returns
    -------
    hsp_label : str or None
        Resolved HSP label, or None if unresolved.
    blocker : str or None
        Blocker reason if unresolved, None if resolved.
    provenance : dict
        Diagnostic provenance for the resolution attempt.
    """
    k_frac = np.asarray(k_frac, dtype=float)
    prov: dict[str, object] = {
        "attempted_direct_match": True,
    }

    canonical_blocker: str | None = None
    if isinstance(standard_match, dict):
        standard_match = dict(standard_match)
        declared_sg_number = standard_match.get("number")
        declared_hall_number = standard_match.get("hall_number")
        declared_hall_ok, declared_hall_blocker = _validate_hall_sg_consistency(
            hall_number=(
                declared_hall_number
                if isinstance(declared_hall_number, int)
                and not isinstance(declared_hall_number, bool)
                else None
            ),
            sg_number=(
                declared_sg_number
                if isinstance(declared_sg_number, int)
                and not isinstance(declared_sg_number, bool)
                else None
            ),
        )
        canonical_identity = derive_irreptables_standard_setting_identity(
            table, declared_sg_number,
        )
        prov["canonical_setting_evidence"] = dict(canonical_identity)
        if not declared_hall_ok:
            canonical_blocker = (
                "standard_setting_hsp_mapping_unresolved: declared Hall/SG "
                f"identity is inconsistent: {declared_hall_blocker}."
            )
            standard_match["canonical_setting_status"] = "input_conflict"
        elif canonical_identity.get("status") != "unique_match":
            canonical_blocker = (
                "standard_setting_hsp_mapping_unresolved: irreptables "
                "canonical Hall setting is not uniquely established "
                f"(status={canonical_identity.get('status')!r}, matching_halls="
                f"{canonical_identity.get('affine_matching_hall_numbers')!r})."
            )
            standard_match["canonical_setting_status"] = str(
                canonical_identity.get("status", "unresolved")
            )
            standard_match["canonical_setting_source"] = str(
                canonical_identity.get("source", "")
            )
            standard_match["canonical_hall_numbers"] = list(
                canonical_identity.get("affine_matching_hall_numbers", [])
            )
            standard_match["canonical_candidate_hall_numbers"] = list(
                canonical_identity.get("candidate_hall_numbers", [])
            )
        else:
            standard_match["detected_setting_hall_number"] = declared_hall_number
            standard_match["detected_setting_hall_symbol"] = standard_match.get(
                "hall_symbol"
            )
            standard_match["hall_number"] = canonical_identity["hall_number"]
            standard_match["hall_symbol"] = canonical_identity["hall_symbol"]
            standard_match["international_short"] = canonical_identity[
                "space_group_symbol"
            ]
            standard_match["canonical_setting_status"] = "unique_match"
            standard_match["canonical_setting_source"] = canonical_identity["source"]
            standard_match["canonical_hall_numbers"] = [
                canonical_identity["hall_number"]
            ]
            standard_match["canonical_candidate_hall_numbers"] = list(
                canonical_identity["candidate_hall_numbers"]
            )

    # 1. Direct coordinate match (existing behavior).
    direct_label = _match_table_kpoint_label(
        table, k_frac, tolerance=tolerance, standard_match=standard_match,
    )

    if canonical_blocker is not None:
        prov["canonical_setting_blocker"] = canonical_blocker
    direct_hall_symbol = (
        str(standard_match.get("hall_symbol", "") or "")
        if isinstance(standard_match, dict)
        else ""
    )
    direct_sg_symbol = (
        str(standard_match.get("international_short", "") or "")
        if isinstance(standard_match, dict)
        else ""
    )
    direct_match_is_primitive = (
        _hall_centering_symbol(direct_hall_symbol) == "P"
        or (not direct_hall_symbol and direct_sg_symbol.startswith("P"))
    )
    direct_match_trusted = (
        not isinstance(standard_match, dict)
        or direct_match_is_primitive
    )
    if (
        direct_label is not None
        and parent_to_standard_direct_transform is None
        and direct_match_trusted
    ):
        prov["direct_match_succeeded"] = True
        is_primitive_group = (
            isinstance(standard_match, dict) and direct_match_is_primitive
        )
        if not is_primitive_group:
            # Coordinate-only match with no primitive standard-group claim:
            # record the label as diagnostic provenance with a weak certificate
            # that cannot enable trusted irrep/EBR promotion.
            cert = build_standard_setting_certificate(
                standard_match=standard_match,
                validation_status="validated",
                parent_basis_operation_ids=(
                    _operation_ids_list(standard_match)
                    if isinstance(standard_match, dict) else None
                ),
                parent_k_frac=k_frac,
                resolved_hsp_label=direct_label,
                origin_shift_fractional=origin_shift_fractional,
            )
            cert.operation_mapping_status = "not_attempted"
            cert.standard_setting_source = "coordinate_match_only"
            prov["standard_setting_certificate"] = cert.to_dict()
            return direct_label, None, prov

        # Primitive standard group: a bare Gamma/coordinate match is true in
        # every reciprocal basis and is NOT sufficient.  Trust the primitive
        # direct-coordinate setting only when generic affine {R|tau} operation
        # equivalence passes under the identity direct transform.
        vp_ids = _operation_ids_list(standard_match)
        aff = _validate_affine_operation_equivalence(
            vp_operations=(
                list(detected_operations) if detected_operations else None
            ),
            vp_operation_ids=vp_ids,
            standard_match=standard_match,
            parent_to_standard_direct_transform=np.eye(3),
            origin_shift_fractional=origin_shift_fractional,
        )
        prov["affine_validation"] = aff
        if aff.get("status") == "passed" and canonical_blocker is None:
            cert = build_standard_setting_certificate(
                standard_match=standard_match,
                validation_status="validated",
                parent_basis_operation_ids=vp_ids,
                parent_to_standard_direct_transform=np.eye(3),
                origin_shift_fractional=origin_shift_fractional,
                transform_provenance="primitive_direct_identity",
                parent_k_frac=k_frac,
                resolved_hsp_label=direct_label,
            )
            cert.centering_status = "primitive_direct_match"
            cert.primitive_conventional_relation = "direct_coordinate_match"
            cert.operation_mapping_status = "operation_basis_verification_passed"
            cert.standard_setting_source = "spglib.per_valley_standard_matches"
            _apply_affine_validation_to_certificate(cert, aff)
            prov["standard_setting_certificate"] = cert.to_dict()
            return direct_label, None, prov

        # Affine equivalence did not pass: coordinate match is diagnostic only.
        if aff.get("status") == "failed":
            blocker = _affine_failure_blocker(
                aff, source="primitive direct coordinate match")
            validation_status = "rejected"
        elif canonical_blocker is not None:
            blocker = canonical_blocker
            validation_status = "unresolved"
        else:
            blocker = (
                "standard_setting_hsp_mapping_unresolved: primitive direct "
                "coordinate match requires passed affine operation equivalence "
                f"(affine status={aff.get('status')!r}, missing="
                f"{aff.get('missing_ingredients')})"
            )
            validation_status = "unresolved"
        cert = build_standard_setting_certificate(
            standard_match=standard_match,
            validation_status=validation_status,
            unresolved_reason=blocker,
            parent_basis_operation_ids=vp_ids,
            parent_to_standard_direct_transform=np.eye(3),
            origin_shift_fractional=origin_shift_fractional,
            transform_provenance="primitive_direct_identity",
            parent_k_frac=k_frac,
        )
        cert.centering_status = "primitive_direct_match"
        cert.primitive_conventional_relation = "direct_coordinate_match"
        cert.operation_mapping_status = "not_attempted"
        cert.standard_setting_source = "spglib.per_valley_standard_matches"
        _apply_affine_validation_to_certificate(cert, aff)
        prov["standard_setting_certificate"] = cert.to_dict()
        return None, blocker, prov

    prov["direct_match_succeeded"] = False
    if direct_label is not None:
        if parent_to_standard_direct_transform is not None:
            prov["direct_match_reason"] = (
                "direct parent-coordinate match was skipped because an explicit "
                "parent-to-standard transform was supplied"
            )
        else:
            prov["direct_match_reason"] = (
                "direct parent-coordinate match was not trusted because the "
                "standard setting is not primitive; a validated "
                "parent-to-standard transform is required"
            )
    else:
        prov["direct_match_reason"] = (
            "parent-setting k_frac does not match any standard-setting "
            "Bilbao HSP coordinate in the irreptables table"
        )

    # 2. Explicit parent-to-standard direct-lattice transform.
    #    When an explicit, validated transform matrix is provided, apply
    #    the reciprocal rule k_std = T^(-T) k_parent and check against
    #    the irreptables table.
    if parent_to_standard_direct_transform is not None:
        T = np.asarray(parent_to_standard_direct_transform, dtype=float)
        tf_result = _validate_explicit_transform(
            transform_matrix=T,
            standard_match=standard_match,
            vp_operations=(
                list(detected_operations) if detected_operations else None
            ),
        )
        prov["explicit_transform"] = tf_result
        if tf_result.get("status") == "valid":
            try:
                explicit_provenance = (
                    str(transform_provenance)
                    if transform_provenance
                    else "explicit_user_input"
                )
                T_inv = np.linalg.inv(T)
                transformed_k = k_frac @ T_inv.T
                prov["transformed_k_frac"] = transformed_k.tolist()
                label = _match_table_kpoint_label(
                    table, transformed_k, tolerance=tolerance,
                    standard_match=standard_match,
                )
                if label is not None:
                    # Affine gate: explicit transform must not bypass
                    # translation validation when affine data is available.
                    aff = None
                    if isinstance(standard_match, dict) and detected_operations:
                        vp_ids = _operation_ids_list(standard_match)
                        aff = _validate_affine_operation_equivalence(
                            vp_operations=list(detected_operations),
                            vp_operation_ids=vp_ids,
                            standard_match=standard_match,
                            parent_to_standard_direct_transform=T,
                            origin_shift_fractional=origin_shift_fractional,
                        )
                    if aff is not None and aff.get("status") != "passed":
                        prov["affine_validation"] = aff
                        affine_failed = aff.get("status") == "failed"
                        tf_result["status"] = (
                            "rejected" if affine_failed else "unresolved"
                        )
                        tf_result["rejection_reason"] = (
                            _affine_failure_blocker(
                                aff, source="explicit transform"
                            )
                            if affine_failed
                            else (
                                "standard_setting_hsp_mapping_unresolved: "
                                "explicit transform requires complete affine "
                                "operation evidence; "
                                f"status={aff.get('status')!r}, missing="
                                f"{aff.get('missing_ingredients')!r}."
                            )
                        )
                        prov["explicit_transform"] = tf_result
                        cert = build_standard_setting_certificate(
                            standard_match=standard_match,
                            validation_status=(
                                "rejected" if affine_failed else "unresolved"
                            ),
                            unresolved_reason=tf_result["rejection_reason"],
                            parent_basis_operation_ids=(
                                _operation_ids_list(standard_match)
                                if isinstance(standard_match, dict) else None
                            ),
                            parent_to_standard_direct_transform=T,
                            origin_shift_fractional=origin_shift_fractional,
                            transform_provenance=explicit_provenance,
                            parent_k_frac=k_frac,
                        )
                        cert.standard_setting_source = "explicit_transform"
                        cert.primitive_conventional_relation = "explicit_transform"
                        cert.operation_mapping_status = (
                            "operation_basis_verification_passed"
                            if isinstance(
                                tf_result.get("operation_basis_verification"),
                                dict,
                            )
                            else "not_attempted"
                        )
                        _apply_affine_validation_to_certificate(cert, aff)
                        candidate = _build_transform_candidate(
                            standard_match=standard_match,
                            parent_to_standard_direct_transform=T,
                            origin_shift_fractional=origin_shift_fractional,
                            transform_provenance=explicit_provenance,
                            operation_mapping_status=cert.operation_mapping_status,
                            affine_result=aff,
                            unresolved_reason=tf_result["rejection_reason"],
                        )
                        _attach_transform_candidate(
                            provenance=prov,
                            certificate=cert,
                            candidate=candidate,
                        )
                        prov["standard_setting_certificate"] = cert.to_dict()
                        return None, tf_result["rejection_reason"], prov
                    else:
                        if canonical_blocker is not None:
                            cert = build_standard_setting_certificate(
                                standard_match=standard_match,
                                validation_status="unresolved",
                                unresolved_reason=canonical_blocker,
                                parent_basis_operation_ids=(
                                    _operation_ids_list(standard_match)
                                    if isinstance(standard_match, dict) else None
                                ),
                                parent_to_standard_direct_transform=T,
                                origin_shift_fractional=origin_shift_fractional,
                                transform_provenance=explicit_provenance,
                                parent_k_frac=k_frac,
                            )
                            cert.standard_setting_source = "explicit_transform"
                            cert.primitive_conventional_relation = "explicit_transform"
                            cert.operation_mapping_status = (
                                "operation_basis_verification_passed"
                            )
                            if aff is not None:
                                _apply_affine_validation_to_certificate(cert, aff)
                            prov["standard_setting_certificate"] = cert.to_dict()
                            return None, canonical_blocker, prov
                        prov["explicit_transformed_match_succeeded"] = True
                        prov["explicit_transformed_hsp_label"] = label
                        cert = build_standard_setting_certificate(
                            standard_match=standard_match,
                            validation_status="validated",
                            parent_basis_operation_ids=(
                                _operation_ids_list(standard_match)
                                if isinstance(standard_match, dict) else None
                            ),
                            parent_to_standard_direct_transform=T,
                            origin_shift_fractional=origin_shift_fractional,
                            transform_provenance=explicit_provenance,
                            parent_k_frac=k_frac,
                            resolved_hsp_label=label,
                        )
                        cert.standard_setting_source = "explicit_transform"
                        cert.primitive_conventional_relation = "explicit_transform"
                        cert.operation_mapping_status = (
                            "operation_basis_verification_passed"
                            if isinstance(
                                tf_result.get("operation_basis_verification"),
                                dict,
                            )
                            else "not_attempted"
                        )
                        if aff is not None:
                            _apply_affine_validation_to_certificate(cert, aff)
                            if aff.get("primitive_conventional_index", 1) > 1:
                                cert.centering_status = "centered_affine_validated"
                            candidate = _build_transform_candidate(
                                standard_match=standard_match,
                                parent_to_standard_direct_transform=T,
                                origin_shift_fractional=origin_shift_fractional,
                                transform_provenance=explicit_provenance,
                                operation_mapping_status=cert.operation_mapping_status,
                                affine_result=aff,
                            )
                            _attach_transform_candidate(
                                provenance=prov,
                                certificate=cert,
                                candidate=candidate,
                            )
                        prov["standard_setting_certificate"] = cert.to_dict()
                        return label, None, prov
            except np.linalg.LinAlgError:
                tf_result["status"] = "rejected"
                tf_result["rejection_reason"] = "singular transformation matrix"
                prov["explicit_transform"] = tf_result
        # Explicit transform rejected — fall through to provenance-only paths.
        # The rejection reason is recorded in prov["explicit_transform"].

    if standard_match is None or not isinstance(standard_match, dict):
        cert = build_standard_setting_certificate(
            standard_match=None,
            validation_status="unresolved",
            unresolved_reason=(
                "standard_setting_hsp_mapping_unresolved: "
                "no per-valley standard group match available "
                "for setting-aware k-coordinate transformation"
            ),
            parent_k_frac=k_frac,
        )
        prov["standard_setting_certificate"] = cert.to_dict()
        return None, cert.unresolved_reason, prov

    sg_number = standard_match.get("number")
    hall_number = standard_match.get("hall_number")
    hall_symbol = standard_match.get("hall_symbol", "")
    sg_symbol = standard_match.get("international_short", "")

    prov["subspace_sg_number"] = sg_number
    prov["subspace_sg_symbol"] = sg_symbol
    prov["hall_number"] = hall_number
    prov["hall_symbol"] = hall_symbol

    # 3. Attempt standard-setting basis transform using crystallographic
    #    data (lattice, VP operation matrices, Hall symbol).
    basis_result = _compute_standard_setting_basis_transform(
        lattice_direct_cart=lattice_direct_cart,
        vp_operations=list(detected_operations) if detected_operations else None,
        standard_match=dict(standard_match),
    )
    prov["basis_transform"] = basis_result

    basis_verification = basis_result.get("operation_basis_verification")
    basis_verification_passed = (
        isinstance(basis_verification, dict)
        and basis_verification.get("status") == "passed"
    )
    if (
        basis_result.get("status") == "accepted"
        and basis_verification_passed
        and basis_result.get("transform_matrix") is not None
    ):
        T = np.asarray(basis_result["transform_matrix"], dtype=float)
        derived_origin = basis_result.get("origin_shift_fractional")
        basis_origin_shift = (
            np.asarray(origin_shift_fractional, dtype=float)
            if origin_shift_fractional is not None
            else (
                np.asarray(derived_origin, dtype=float)
                if derived_origin is not None
                else None
            )
        )
        basis_provenance = str(
            basis_result.get(
                "transform_provenance", "operation_basis_reconstruction",
            )
        )
        # Transform k_frac from parent reciprocal basis to standard setting.
        # The reciprocal transform is the transpose of the inverse of the
        # direct-lattice transform: k_std = T^(-T) * k_parent.
        try:
            T_inv = np.linalg.inv(T)
            transformed_k = k_frac @ T_inv.T
            prov["transformed_k_frac"] = transformed_k.tolist()
            transformed_label = _match_table_kpoint_label(
                table, transformed_k, tolerance=tolerance,
                standard_match=standard_match,
            )
            aff = None
            if isinstance(standard_match, dict) and detected_operations:
                vp_ids = _operation_ids_list(standard_match)
                aff = _validate_affine_operation_equivalence(
                    vp_operations=list(detected_operations),
                    vp_operation_ids=vp_ids,
                    standard_match=standard_match,
                    parent_to_standard_direct_transform=T,
                    origin_shift_fractional=basis_origin_shift,
                )
                if aff.get("status") == "failed":
                    prov["affine_validation"] = aff
                    blocker = _affine_failure_blocker(
                        aff, source="operation-basis reconstruction"
                    )
                    cert = build_standard_setting_certificate(
                        standard_match=standard_match,
                        validation_status="rejected",
                        unresolved_reason=blocker,
                        parent_basis_operation_ids=(
                            _operation_ids_list(standard_match)
                            if isinstance(standard_match, dict) else None
                        ),
                        parent_to_standard_direct_transform=T,
                        origin_shift_fractional=basis_origin_shift,
                        transform_provenance=basis_provenance,
                        parent_k_frac=k_frac,
                    )
                    cert.standard_setting_source = basis_provenance
                    cert.primitive_conventional_relation = basis_provenance
                    cert.operation_mapping_status = (
                        "operation_basis_verification_passed"
                    )
                    _apply_affine_validation_to_certificate(cert, aff)
                    candidate = _build_transform_candidate(
                        standard_match=standard_match,
                        parent_to_standard_direct_transform=T,
                        origin_shift_fractional=basis_origin_shift,
                        transform_provenance=basis_provenance,
                        operation_mapping_status=cert.operation_mapping_status,
                        affine_result=aff,
                        unresolved_reason=blocker,
                    )
                    _attach_transform_candidate(
                        provenance=prov,
                        certificate=cert,
                        candidate=candidate,
                    )
                    prov["standard_setting_certificate"] = cert.to_dict()
                    return None, blocker, prov
            if canonical_blocker is not None:
                cert = build_standard_setting_certificate(
                    standard_match=standard_match,
                    validation_status="unresolved",
                    unresolved_reason=canonical_blocker,
                    parent_basis_operation_ids=_operation_ids_list(standard_match),
                    parent_to_standard_direct_transform=T,
                    origin_shift_fractional=basis_origin_shift,
                    transform_provenance=basis_provenance,
                    parent_k_frac=k_frac,
                )
                cert.primitive_conventional_relation = basis_provenance
                if aff is not None:
                    _apply_affine_validation_to_certificate(cert, aff)
                prov["standard_setting_certificate"] = cert.to_dict()
                return None, canonical_blocker, prov

            cert = build_standard_setting_certificate(
                standard_match=standard_match,
                validation_status="validated",
                parent_basis_operation_ids=(
                    _operation_ids_list(standard_match)
                    if isinstance(standard_match, dict) else None
                ),
                parent_to_standard_direct_transform=T,
                origin_shift_fractional=basis_origin_shift,
                transform_provenance=basis_provenance,
                parent_k_frac=k_frac,
                resolved_hsp_label=transformed_label,
            )
            cert.standard_setting_source = basis_provenance
            cert.primitive_conventional_relation = basis_provenance
            cert.operation_mapping_status = (
                "operation_basis_verification_passed"
            )
            if aff is not None:
                _apply_affine_validation_to_certificate(cert, aff)
                if aff.get("primitive_conventional_index", 1) > 1:
                    cert.centering_status = "centered_affine_validated"
                candidate = _build_transform_candidate(
                    standard_match=standard_match,
                    parent_to_standard_direct_transform=T,
                    origin_shift_fractional=basis_origin_shift,
                    transform_provenance=basis_provenance,
                    operation_mapping_status=cert.operation_mapping_status,
                    affine_result=aff,
                )
                _attach_transform_candidate(
                    provenance=prov,
                    certificate=cert,
                    candidate=candidate,
                )

            if transformed_label is not None:
                prov["basis_transformed_match_succeeded"] = True
                prov["transformed_hsp_label"] = transformed_label
                prov["standard_setting_certificate"] = cert.to_dict()
                return transformed_label, None, prov

            prov["basis_transformed_match_succeeded"] = False
            prov["standard_setting_certificate"] = cert.to_dict()
            blocker = (
                "standard_setting_hsp_label_unavailable: validated "
                "standard-setting transformation maps parent k-point "
                f"{k_frac.tolist()} to standard k-point "
                f"{transformed_k.tolist()}, but no source HSP label matches "
                f"for valley-projected subspace SG {sg_symbol} "
                f"(No. {sg_number})"
            )
            return None, blocker, prov
        except np.linalg.LinAlgError:
            prov["basis_transform_error"] = "singular transformation matrix"

    # 3. Attempt setting-aware transformation using Hall-symbol centering.
    #    For centered settings (C, I, F, R), the conventional reciprocal
    #    cell differs from the primitive cell used in the parent DFT.
    transform_result = _attempt_setting_transform(
        k_frac=k_frac,
        hall_number=int(hall_number) if isinstance(hall_number, int) and not isinstance(hall_number, bool) else None,
        hall_symbol=str(hall_symbol) if hall_symbol else "",
        sg_number=int(sg_number) if isinstance(sg_number, int) and not isinstance(sg_number, bool) else None,
    )
    prov["setting_transform"] = transform_result

    if transform_result.get("transformed_k_frac") is not None:
        transformed_k = np.asarray(transform_result["transformed_k_frac"], dtype=float)
        transformed_label = _match_table_kpoint_label(
            table, transformed_k, tolerance=tolerance,
            standard_match=standard_match,
        )
        if transformed_label is not None and canonical_blocker is None:
            prov["transformed_match_succeeded"] = True
            prov["transformed_k_frac"] = transformed_k.tolist()
            prov["transformed_hsp_label"] = transformed_label
            return transformed_label, None, prov

    # 3. Setting transform not available or not unique.
    reason_parts = [
        "standard_setting_hsp_mapping_unresolved: "
        f"valley-projected subspace SG {sg_symbol} (No. {sg_number})",
    ]
    if hall_symbol:
        hall_str = f"Hall {hall_number} ({hall_symbol})" if hall_number else hall_symbol
        reason_parts.append(f"standard setting {hall_str}")
    reason_parts.append(
        "is resolved, but no unique standard-setting k-coordinate "
        "transform is established from the parent moire reciprocal "
        "basis to the irreptables standard reciprocal basis"
    )
    if transform_result.get("reason"):
        reason_parts.append(f"- {transform_result['reason']}")

    blocker = " ".join(reason_parts) + "."
    cert = build_standard_setting_certificate(
        standard_match=standard_match,
        validation_status="unresolved",
        unresolved_reason=blocker,
        parent_basis_operation_ids=(
            _operation_ids_list(standard_match)
            if isinstance(standard_match, dict) else None
        ),
        parent_k_frac=k_frac,
    )
    cert.standard_setting_source = (
        "spglib.get_spacegroup_type_from_symmetry"
    )
    cert.centering_status = "centered_unresolved" if (
        isinstance(standard_match, dict)
        and _hall_centering_symbol(str(standard_match.get("hall_symbol", "") or ""))
        in ("A", "B", "C", "I", "F", "R")
    ) else "not_evaluated"
    cert.primitive_conventional_relation = (
        "centered_unresolved"
        if cert.centering_status == "centered_unresolved"
        else "not_applicable"
    )
    # Affine validation for unresolved path.
    if isinstance(standard_match, dict) and detected_operations:
        vp_ids = _operation_ids_list(standard_match)
        aff = _validate_affine_operation_equivalence(
            vp_operations=list(detected_operations),
            vp_operation_ids=vp_ids,
            standard_match=standard_match,
        )
        _apply_affine_validation_to_certificate(cert, aff)
    # Build transform candidate with centering/affine evidence.
    candidate = _build_transform_candidate(
        standard_match=standard_match,
        parent_to_standard_direct_transform=parent_to_standard_direct_transform,
        origin_shift_fractional=origin_shift_fractional,
        transform_provenance=(
            "explicit_config"
            if parent_to_standard_direct_transform is not None
            else "not_provided"
        ),
        affine_result=aff if isinstance(standard_match, dict) and detected_operations else None,
        unresolved_reason=blocker if cert.validation_status != "validated" else None,
    )
    cert.centering_vectors = candidate.centering_vectors
    prov["transform_candidate"] = candidate.to_dict()
    prov["standard_setting_certificate"] = cert.to_dict()
    return None, blocker, prov


def _is_exact_operation_id(value: object) -> bool:
    """Return whether value is a trusted opaque Python integer ID."""
    return isinstance(value, int) and not isinstance(value, bool)


def _operation_ids_list(sm: dict[str, object]) -> list[int] | None:
    """Return exact operation IDs, preserving malformed input as unknown."""
    value = sm.get("operation_ids")
    if not isinstance(value, list):
        return None
    if any(not _is_exact_operation_id(item) for item in value):
        return None
    return list(value)


def _hall_centering_symbol(hall_symbol: str) -> str:
    """Return the lattice centering symbol from a Hall symbol.

    Hall symbols may begin with ``-`` for centrosymmetric groups, e.g.
    ``-P 1`` or ``-R 3``.  The lattice centering symbol follows that
    sign and must be parsed before primitive/centered decisions.
    """
    token = str(hall_symbol or "").strip()
    if token.startswith("-"):
        token = token[1:].lstrip()
    return token[:1].upper()


_CENTERING_INDICES = {
    "P": 1,
    "A": 2,
    "B": 2,
    "C": 2,
    "I": 2,
    "F": 4,
    "R": 3,
}


def _translation_key(vector: object, tolerance: float = 1e-8) -> tuple[int, int, int]:
    values = np.asarray(vector, dtype=float)
    reduced = values - np.floor(values + tolerance)
    reduced[np.abs(reduced - 1.0) <= tolerance] = 0.0
    return tuple(int(np.rint(value / tolerance)) for value in reduced)


def _canonical_centering_cosets(
    vectors: list[np.ndarray],
    tolerance: float = 1e-8,
) -> list[np.ndarray]:
    """Return normalized cosets in the one order used by map indices."""
    normalized: list[np.ndarray] = []
    for vector in vectors:
        reduced = np.asarray(vector, dtype=float)
        reduced = reduced - np.floor(reduced + tolerance)
        reduced[np.abs(reduced - 1.0) <= tolerance] = 0.0
        normalized.append(reduced)
    return sorted(
        normalized,
        key=lambda vector: _translation_key(vector, tolerance),
    )


def _affine_operation_key(
    rotation: object,
    translation: object,
    tolerance: float = 1e-8,
) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    rotation_array = np.asarray(rotation, dtype=float)
    return (
        tuple(int(value) for value in np.rint(rotation_array).reshape(-1)),
        _translation_key(translation, tolerance),
    )


def _affine_operation_set_is_closed(
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    tolerance: float = 1e-8,
) -> bool:
    if not rotations or len(rotations) != len(translations):
        return False
    keys = {
        _affine_operation_key(rotation, translation, tolerance)
        for rotation, translation in zip(rotations, translations)
    }
    if len(keys) != len(rotations):
        return False
    for left_rotation, left_translation in zip(rotations, translations):
        for right_rotation, right_translation in zip(rotations, translations):
            product_rotation = left_rotation @ right_rotation
            product_translation = (
                left_translation + left_rotation @ right_translation
            )
            if _affine_operation_key(
                product_rotation, product_translation, tolerance,
            ) not in keys:
                return False
    return True


def _centering_cosets_from_hall_database(
    hall_number: object,
    *,
    tolerance: float = 1e-8,
) -> dict[str, object]:
    """Derive conventional centering cosets from one Hall operation set.

    The cosets are the distinct translations attached to the identity
    rotation in spglib's Hall database.  The Hall lattice symbol supplies the
    crystallographic primitive/conventional index, but never the coordinates
    of the cosets themselves.
    """
    result: dict[str, object] = {"status": "unresolved"}
    if not isinstance(hall_number, int) or isinstance(hall_number, bool):
        result["reason"] = "hall_number_missing_or_malformed"
        return result
    try:
        import spglib
        sg_type = spglib.get_spacegroup_type(hall_number)
        symmetry = spglib.get_symmetry_from_database(hall_number)
    except Exception:
        sg_type = None
        symmetry = None
    if sg_type is None or symmetry is None:
        result["reason"] = "spglib_hall_database_unavailable"
        return result

    centering_type = _hall_centering_symbol(str(sg_type.hall_symbol))
    expected_index = _CENTERING_INDICES.get(centering_type)
    if expected_index is None:
        result["reason"] = "unrecognized_hall_centering"
        return result

    rotations = [np.asarray(value, dtype=float) for value in symmetry["rotations"]]
    translations = [
        np.asarray(value, dtype=float) for value in symmetry["translations"]
    ]
    identity = np.eye(3, dtype=float)
    raw_cosets = [
        translation for rotation, translation in zip(rotations, translations)
        if np.allclose(rotation, identity, atol=tolerance)
    ]
    raw_keys = [_translation_key(vector, tolerance) for vector in raw_cosets]
    if len(raw_keys) != len(set(raw_keys)):
        result["status"] = "failed"
        result["reason"] = "duplicate_centering_cosets"
        return result
    if _translation_key(np.zeros(3), tolerance) not in raw_keys:
        result["status"] = "failed"
        result["reason"] = "identity_centering_coset_missing"
        return result
    if len(raw_cosets) != expected_index:
        result["status"] = "failed"
        result["reason"] = (
            "centering_coset_count_mismatch: "
            f"Hall centering {centering_type} requires {expected_index}, "
            f"database operations provide {len(raw_cosets)}"
        )
        return result

    centering_cosets = _canonical_centering_cosets(raw_cosets, tolerance)
    coset_keys = {
        _translation_key(vector, tolerance) for vector in centering_cosets
    }
    for left in centering_cosets:
        for right in centering_cosets:
            if _translation_key(left + right, tolerance) not in coset_keys:
                result["status"] = "failed"
                result["reason"] = "centering_cosets_not_closed"
                return result

    result.update({
        "status": "passed",
        "centering_type": centering_type,
        "primitive_conventional_index": expected_index,
        "centering_cosets": [
            vector.tolist() for vector in centering_cosets
        ],
        "standard_setting_operation_count": len(rotations),
        "standard_operation_closure_validated": (
            _affine_operation_set_is_closed(rotations, translations, tolerance)
        ),
        "source": "spglib.get_symmetry_from_database",
    })
    if result["standard_operation_closure_validated"] is not True:
        result["status"] = "failed"
        result["reason"] = "standard_hall_operation_set_not_closed"
    return result


def _table_operations_match_hall(
    table: object,
    hall_number: int,
    *,
    tolerance: float = 1e-8,
) -> bool:
    operations = getattr(table, "operations", None)
    if not isinstance(operations, (list, tuple)) or not operations:
        return False
    evidence = _centering_cosets_from_hall_database(
        hall_number, tolerance=tolerance,
    )
    if evidence.get("status") != "passed":
        return False
    try:
        import spglib
        symmetry = spglib.get_symmetry_from_database(hall_number)
    except Exception:
        symmetry = None
    if symmetry is None:
        return False
    standard_keys = [
        _affine_operation_key(rotation, translation, tolerance)
        for rotation, translation in zip(
            symmetry["rotations"], symmetry["translations"],
        )
    ]
    cosets = evidence.get("centering_cosets")
    if not isinstance(cosets, list):
        return False
    expanded_keys = []
    for operation in operations:
        rotation = getattr(operation, "rotation_frac", None)
        translation = getattr(operation, "translation_frac", None)
        if rotation is None or translation is None:
            return False
        for coset in cosets:
            expanded_keys.append(_affine_operation_key(
                rotation,
                np.asarray(translation, dtype=float) + np.asarray(coset, dtype=float),
                tolerance,
            ))
    return (
        len(expanded_keys) == len(standard_keys)
        and len(set(expanded_keys)) == len(expanded_keys)
        and set(expanded_keys) == set(standard_keys)
    )


def derive_irreptables_standard_setting_identity(
    table: object,
    sg_number: object,
) -> dict[str, object]:
    """Identify the irreptables Hall setting from independent source data.

    Every Hall setting for the same international SG number is enumerated.
    Candidates are retained only when the irreptables source centering and
    source affine operation set agree.  No candidate ordering is used as a
    fallback.
    """
    result: dict[str, object] = {
        "status": "unresolved",
        "source": "irreptables.StandardIrrepTable+spglib.HallDatabase",
        "candidate_hall_numbers": [],
        "affine_matching_hall_numbers": [],
    }
    if not isinstance(sg_number, int) or isinstance(sg_number, bool) \
            or sg_number <= 0:
        result["reason"] = "space_group_number_missing_or_malformed"
        return result
    operations = getattr(table, "operations", None)
    has_source_operations = (
        isinstance(operations, (list, tuple)) and bool(operations)
    )
    table_number = getattr(table, "number", None)
    if has_source_operations \
            and isinstance(table_number, int) and not isinstance(table_number, bool) \
            and table_number != sg_number:
        result["status"] = "no_match"
        result["reason"] = "irreptables_space_group_number_mismatch"
        return result
    try:
        import spglib
    except Exception:
        result["reason"] = "spglib_unavailable"
        return result

    candidates: list[tuple[int, object]] = []
    for hall_number in range(1, 531):
        sg_type = spglib.get_spacegroup_type(hall_number)
        if sg_type is not None and int(sg_type.number) == sg_number:
            candidates.append((hall_number, sg_type))
    result["candidate_hall_numbers"] = [number for number, _ in candidates]
    if not candidates:
        result["status"] = "no_match"
        result["reason"] = "no_spglib_hall_setting_for_space_group"
        return result

    if not has_source_operations:
        if len(candidates) == 1:
            matching = candidates
        else:
            result["status"] = "ambiguous"
            result["affine_matching_hall_numbers"] = [
                number for number, _ in candidates
            ]
            result["reason"] = "irreptables_affine_operations_unavailable"
            return result
    else:
        source_centering = _hall_centering_symbol(
            str(getattr(table, "name", "") or "")
        )
        matching = []
        for hall_number, sg_type in candidates:
            hall_centering = _hall_centering_symbol(str(sg_type.hall_symbol))
            if source_centering in _CENTERING_INDICES \
                    and hall_centering != source_centering:
                continue
            if _table_operations_match_hall(table, hall_number):
                matching.append((hall_number, sg_type))

    result["affine_matching_hall_numbers"] = [
        number for number, _ in matching
    ]
    if len(matching) == 0:
        result["status"] = "no_match"
        result["reason"] = "irreptables_affine_operations_match_no_hall_setting"
        return result
    if len(matching) > 1:
        result["status"] = "ambiguous"
        result["reason"] = "irreptables_affine_operations_match_multiple_hall_settings"
        return result

    hall_number, sg_type = matching[0]
    coset_evidence = _centering_cosets_from_hall_database(hall_number)
    if coset_evidence.get("status") != "passed":
        result["status"] = "no_match"
        result["reason"] = str(coset_evidence.get("reason", "centering_unresolved"))
        return result
    result.update({
        "status": "unique_match",
        "hall_number": hall_number,
        "hall_symbol": str(sg_type.hall_symbol),
        "centering_type": str(coset_evidence["centering_type"]),
        "primitive_conventional_index": int(
            coset_evidence["primitive_conventional_index"]
        ),
        "space_group_number": int(sg_type.number),
        "space_group_symbol": str(getattr(table, "name", "") or sg_type.international_short),
        "centering_cosets": [
            list(vector) for vector in coset_evidence["centering_cosets"]
        ],
        "standard_setting_operation_count": int(
            coset_evidence["standard_setting_operation_count"]
        ),
        "standard_operation_closure_validated": bool(
            coset_evidence["standard_operation_closure_validated"]
        ),
    })
    return result


def _centering_cosets(hall_symbol: str) -> list[np.ndarray] | None:
    """Return legacy diagnostic centering cosets from a Hall symbol.

    This helper is used only by the small-integer transform-candidate search,
    which performs set-membership diagnostics and never assigns serialized
    coset indices.  Trusted affine expansion and certificate evidence use
    ``_centering_cosets_from_hall_database`` and its canonical ordering.

    Returns fractional translation vectors that define the centering.
    For primitive settings (P), returns [ (0,0,0) ] only.
    For A/B/C/I/F centered settings, returns the standard coset
    representatives from ITA Tables 2.1.1.1.  Rhombohedral settings
    stay unresolved here because obverse/reverse convention must be
    explicit.

    Returns None when the centering convention is unavailable,
    ambiguous, or unrecognised.
    """
    identity = np.array([0.0, 0.0, 0.0])
    if not hall_symbol:
        return [identity]
    centering = _hall_centering_symbol(hall_symbol)
    if centering == "P":
        return [identity]
    if centering == "A":
        return [identity, np.array([0.0, 0.5, 0.5])]
    if centering == "B":
        return [identity, np.array([0.5, 0.0, 0.5])]
    if centering == "C":
        return [identity, np.array([0.5, 0.5, 0.0])]
    if centering == "I":
        return [identity, np.array([0.5, 0.5, 0.5])]
    if centering == "F":
        return [
            identity,
            np.array([0.0, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.5, 0.0]),
        ]
    if centering == "R":
        # Rhombohedral settings require an explicit obverse/reverse
        # convention.  Do not guess from the Hall symbol prefix alone.
        return None
    return None


def _validate_hall_sg_consistency(
    *,
    hall_number: int | None,
    sg_number: int | None,
) -> tuple[bool, str | None]:
    """Check that Hall number and SG number are internally consistent.

    spglib maps each Hall number to a unique space-group number.
    The standard_match from spglib should carry both consistently.
    A mismatch indicates an input convention error (e.g. passing the
    SG number as the Hall number).

    Returns (ok, blocker_reason).
    """
    if hall_number is None:
        return False, "standard_setting_hall_number_missing: Hall number is None"
    if sg_number is None:
        # Cannot validate; do not block — Hall alone is sufficient
        # for standard-setting operation lookup.
        return True, None
    try:
        import spglib
        sg_type = spglib.get_spacegroup_type(hall_number)
    except Exception:
        return False, (
            f"standard_setting_hall_number_invalid: spglib could not "
            f"resolve Hall number {hall_number}"
        )
    if sg_type is None:
        return False, (
            f"standard_setting_hall_number_invalid: Hall number "
            f"{hall_number} returned None from spglib"
        )
    if int(sg_type.number) != int(sg_number):
        return False, (
            f"standard_setting_hall_number_mismatch: Hall number "
            f"{hall_number} corresponds to SG {sg_type.number} "
            f"({sg_type.international_short}), but standard_match "
            f"declares SG {sg_number}"
        )
    return True, None


def _validate_explicit_transform(
    *,
    transform_matrix: np.ndarray,
    standard_match: dict[str, object] | None,
    vp_operations: list[dict[str, object]] | None,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Validate an explicit parent-to-standard direct-lattice transform.

    Checks: finite 3×3 real matrix, nonsingular, provenance present,
    operation-basis verification passes (when VP ops available).

    Returns a dict with ``status`` (``"valid"`` or ``"rejected"``)
    and a ``rejection_reason`` when rejected.
    """
    T = np.asarray(transform_matrix, dtype=float)
    result: dict[str, object] = {"status": "valid"}

    # Shape check.
    if T.shape != (3, 3):
        result["status"] = "rejected"
        result["rejection_reason"] = (
            f"transform matrix has shape {T.shape}; expected (3, 3)"
        )
        return result

    # Finite check.
    if not np.all(np.isfinite(T)):
        result["status"] = "rejected"
        result["rejection_reason"] = (
            "transform matrix is not finite (contains NaN or inf)"
        )
        return result

    # Nonsingular check.
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        result["status"] = "rejected"
        result["rejection_reason"] = "transform matrix is singular"
        return result

    det = np.linalg.det(T)
    if abs(det) < 1e-10:
        result["status"] = "rejected"
        result["rejection_reason"] = (
            f"transform matrix has near-zero determinant ({det:.2e})"
        )
        return result

    result["determinant"] = float(det)

    # Provenance check: standard_match must be present.
    if not isinstance(standard_match, dict):
        result["status"] = "rejected"
        result["rejection_reason"] = (
            "explicit transform provided without standard group "
            "match provenance"
        )
        return result

    # Operation-basis verification (when VP ops available).
    if vp_operations is not None:
        vp_id_list = _operation_ids_list(standard_match)
        if vp_id_list is None:
            result["status"] = "rejected"
            result["rejection_reason"] = (
                "standard group match operation_ids are malformed"
            )
            return result
        vp_ids = set(vp_id_list)
        parent_rotations: list[np.ndarray] = []
        for op in vp_operations:
            if not isinstance(op, dict):
                continue
            oid = op.get("operation_id")
            if not _is_exact_operation_id(oid):
                continue
            if oid not in vp_ids:
                continue
            rot = op.get("rotation_frac")
            if rot is None:
                continue
            parent_rotations.append(np.asarray(rot, dtype=float))

        if parent_rotations:
            hall_number = standard_match.get("hall_number")
            if isinstance(hall_number, int) and not isinstance(hall_number, bool):
                try:
                    import spglib
                    std_sym = spglib.get_symmetry_from_database(int(hall_number))
                except Exception:
                    std_sym = None
                if std_sym is not None:
                    std_rotations = [
                        np.asarray(r, dtype=float)
                        for r in std_sym["rotations"]
                    ]
                    vf = _verify_operation_basis(
                        parent_rotations=parent_rotations,
                        std_rotations=std_rotations,
                        transform_matrix=T,
                        tolerance=tolerance,
                    )
                    result["operation_basis_verification"] = vf
                    if vf.get("status") != "passed":
                        result["status"] = "rejected"
                        result["rejection_reason"] = (
                            "operation-basis verification failed: "
                            f"{vf.get('reason', 'unknown')}"
                        )
                        return result

    return result


def _reconstruct_subgroup_standard_cell(
    *,
    lattice_direct_cart: np.ndarray,
    vp_operations: list[dict[str, object]],
    standard_match: dict[str, object],
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Reconstruct a subgroup standard-setting direct cell from VP operations.

    Uses spglib's standard-setting symmetry database and the detected
    VP rotation matrices to determine the basis transformation from
    the parent fractional basis to the standard-setting fractional
    basis.  Only accepts the result when the transformed VP operation
    matrices match the standard-setting operation content.

    Returns a dict with ``status``, optional ``transform_matrix``,
    ``operation_basis_verification``, and/or ``reason``.
    """
    result: dict[str, object] = {"status": "not_attempted"}

    hall_number = standard_match.get("hall_number")
    if not isinstance(hall_number, int) or isinstance(hall_number, bool):
        result["status"] = "unavailable"
        result["reason"] = (
            "Hall number not available from spglib standard match; "
            "cannot retrieve standard-setting symmetry operations"
        )
        return result
    hall_number = int(hall_number)
    hall_symbol = str(standard_match.get("hall_symbol") or "").strip()
    centering = _hall_centering_symbol(hall_symbol)
    if centering in {"A", "B", "C", "F", "I", "R"}:
        return _standardize_affine_subgroup_cell(
            lattice_direct_cart=lattice_direct_cart,
            vp_operations=vp_operations,
            standard_match=standard_match,
            tolerance=tolerance,
        )

    # 1. Load standard-setting operations from spglib database.
    try:
        import spglib
        std_sym = spglib.get_symmetry_from_database(hall_number)
    except Exception:
        result["status"] = "unavailable"
        result["reason"] = (
            f"spglib.get_symmetry_from_database(Hall {hall_number}) "
            f"failed; cannot retrieve standard-setting operations"
        )
        return result

    if std_sym is None:
        result["status"] = "unavailable"
        result["reason"] = (
            f"spglib returned None for Hall {hall_number}; "
            f"standard-setting operations not available"
        )
        return result

    std_rotations = [np.asarray(r, dtype=float) for r in std_sym["rotations"]]

    # 2. Collect VP rotation matrices and match by order/type.
    vp_id_list = _operation_ids_list(standard_match)
    if vp_id_list is None:
        result["status"] = "unavailable"
        result["reason"] = "standard group match operation_ids are malformed"
        return result
    vp_ids = set(vp_id_list)
    parent_by_id: dict[int, dict[str, object]] = {}
    for op in vp_operations:
        if not isinstance(op, dict):
            continue
        oid = op.get("operation_id")
        if not _is_exact_operation_id(oid):
            continue
        if oid in vp_ids:
            parent_by_id[oid] = op

    parent_rotations: list[np.ndarray] = []
    for oid in sorted(vp_ids):
        op = parent_by_id.get(oid)
        if op is None:
            continue
        rot = op.get("rotation_frac")
        if rot is None:
            continue
        parent_rotations.append(np.asarray(rot, dtype=float))

    nontrivial_parent = [r for r in parent_rotations
                         if not np.allclose(r, np.eye(3), atol=tolerance)]
    nontrivial_std = [r for r in std_rotations
                      if not np.allclose(r, np.eye(3), atol=tolerance)]

    # Need at least one nontrivial VP rotation to align axes.
    if not nontrivial_parent or not nontrivial_std:
        result["status"] = "unavailable"
        result["reason"] = (
            "no nontrivial rotation matrices available in either "
            "parent or standard setting; cannot determine basis "
            "orientation from operation content"
        )
        return result

    # 3. Align the parent and standard rotation axes.
    #    Each rotation matrix defines an axis (rotation eigenvector).
    #    We find the basis transformation T such that
    #    T · R_parent · T⁻¹ = R_standard for matched rotation pairs.
    #    For a single nontrivial rotation (e.g., C2), the axis alignment
    #    determines T up to a scale and an in-plane rotation.
    parent_rot = nontrivial_parent[0]
    std_rot = nontrivial_std[0]

    # Find the rotation axis of the parent rotation (eigenvector with
    # eigenvalue +1 for proper rotations).
    # For a 2-fold rotation: eigenvalues are [1, -1, -1]; the axis is
    # the eigenvector with eigenvalue 1.
    parent_eigvals, parent_eigvecs = np.linalg.eig(parent_rot)
    rotation_axis_idx = np.argmin(np.abs(parent_eigvals - 1.0))
    parent_axis = parent_eigvecs[:, rotation_axis_idx].real

    # The standard rotation axis in the standard setting is also the
    # eigenvector with eigenvalue 1.
    std_eigvals, std_eigvecs = np.linalg.eig(std_rot)
    std_axis_idx = np.argmin(np.abs(std_eigvals - 1.0))
    std_axis = std_eigvecs[:, std_axis_idx].real

    # The transformation T maps the parent axis to the standard axis.
    # For C2: parent_axis in hexagonal basis → std_axis in standard basis.
    # We construct T by aligning the two coordinate frames.

    # 4. Build a candidate basis transformation T (direct space).
    #    T transforms coordinates FROM parent basis TO standard basis:
    #    x_std = T · x_parent
    #    For k-points (reciprocal space): k_std = T⁻ᵀ · k_parent
    try:
        T = _align_bases_from_rotation_axes(
            parent_axis=parent_axis,
            std_axis=std_axis,
            parent_rot=parent_rot,
            std_rot=std_rot,
            lattice=lattice_direct_cart,
        )
    except Exception as exc:
        result["status"] = "unavailable"
        result["reason"] = (
            f"basis alignment from rotation axes failed: {exc}"
        )
        return result

    if T is None:
        result["status"] = "unavailable"
        result["reason"] = (
            "could not uniquely determine basis transformation from "
            "rotation axes; the parent and standard settings may have "
            "incompatible Bravais lattice types"
        )
        return result

    # 5. Verify: transform each parent rotation into the standard basis
    #    and check against the standard rotation set.
    vf = _verify_operation_basis(
        parent_rotations=parent_rotations,
        std_rotations=std_rotations,
        transform_matrix=T,
        tolerance=tolerance,
    )
    result["operation_basis_verification"] = vf

    if vf.get("status") != "passed":
        result["status"] = "rejected"
        result["reason"] = (
            "operation-basis verification failed: "
            f"{vf.get('reason', 'transformed parent rotations do not '
                       'match standard-setting rotations')}"
        )
        return result

    result["status"] = "accepted"
    result["transform_matrix"] = T.tolist()
    return result


def _standardize_affine_subgroup_cell(
    *,
    lattice_direct_cart: np.ndarray,
    vp_operations: list[dict[str, object]],
    standard_match: dict[str, object],
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Derive a centered standard cell from complete affine subgroup data."""
    result: dict[str, object] = {
        "status": "unavailable",
        "transform_provenance": "spglib_affine_subgroup_standardization",
        "operation_basis_verification": {
            "status": "not_attempted",
            "reason": "complete affine subgroup evidence is not yet validated",
        },
    }
    required_ids = _operation_ids_list(standard_match)
    if required_ids is None or not required_ids:
        result["reason"] = "standard group match operation_ids are malformed"
        return result

    operations_by_id: dict[int, dict[str, object]] = {}
    for operation in vp_operations:
        if not isinstance(operation, dict):
            continue
        operation_id = operation.get("operation_id")
        if not _is_exact_operation_id(operation_id) or operation_id not in required_ids:
            continue
        rotation = operation.get("rotation_frac")
        translation = operation.get("translation_frac")
        if rotation is None or translation is None:
            continue
        operations_by_id[operation_id] = operation
    if set(operations_by_id) != set(required_ids):
        result["reason"] = (
            "complete valley-preserving affine operation data are required "
            "for subgroup standardization"
        )
        return result
    subgroup_operations = [operations_by_id[operation_id] for operation_id in required_ids]

    seeds = (
        np.array([0.137, 0.271, 0.389]),
        np.array([0.219, 0.413, 0.173]),
        np.array([0.347, 0.119, 0.283]),
    )
    positions: list[np.ndarray] = []
    atom_types: list[int] = []
    for atom_type, seed in enumerate(seeds, start=1):
        orbit: list[np.ndarray] = []
        for operation in subgroup_operations:
            rotation = np.asarray(operation["rotation_frac"], dtype=float)
            translation = np.asarray(operation["translation_frac"], dtype=float)
            if rotation.shape != (3, 3) or translation.shape != (3,):
                result["reason"] = "malformed valley-preserving affine operation"
                return result
            position = np.mod(rotation @ seed + translation, 1.0)
            if not any(
                _mod1_norm(position - existing) <= tolerance
                for existing in orbit
            ):
                orbit.append(position)
        if len(orbit) != len(required_ids):
            result["reason"] = (
                "deterministic generic subgroup orbit has a nontrivial site "
                "stabilizer; standard-setting transform remains unresolved"
            )
            return result
        positions.extend(orbit)
        atom_types.extend([atom_type] * len(orbit))

    hall_number = standard_match.get("hall_number")
    sg_number = standard_match.get("number")
    if not _is_exact_operation_id(hall_number) or not _is_exact_operation_id(sg_number):
        result["reason"] = "Hall and space-group numbers are required"
        return result
    symprec = standard_match.get("symprec", tolerance)
    try:
        symprec_value = float(symprec)
    except (TypeError, ValueError):
        symprec_value = tolerance
    if not np.isfinite(symprec_value) or symprec_value <= 0.0:
        symprec_value = tolerance

    try:
        import spglib

        dataset = spglib.get_symmetry_dataset(
            (
                np.asarray(lattice_direct_cart, dtype=float),
                np.asarray(positions, dtype=float),
                np.asarray(atom_types, dtype=int),
            ),
            symprec=symprec_value,
            angle_tolerance=-1.0,
            hall_number=int(hall_number),
        )
    except Exception as exc:
        result["reason"] = f"spglib affine subgroup standardization failed: {exc}"
        return result
    if dataset is None:
        result["reason"] = "spglib affine subgroup standardization returned no dataset"
        return result
    if int(dataset.number) != int(sg_number) or int(dataset.hall_number) != int(hall_number):
        result["reason"] = (
            "spglib affine subgroup standardization disagrees with the "
            "canonical space-group or Hall identity"
        )
        return result
    if len(dataset.rotations) != len(required_ids):
        result["reason"] = (
            "constructed generic orbit has unexpected additional or missing "
            "input-basis symmetry operations"
        )
        return result

    transform = np.asarray(dataset.transformation_matrix, dtype=float)
    origin_shift = np.asarray(dataset.origin_shift, dtype=float)
    affine = _validate_affine_operation_equivalence(
        vp_operations=vp_operations,
        vp_operation_ids=required_ids,
        standard_match=standard_match,
        parent_to_standard_direct_transform=transform,
        origin_shift_fractional=origin_shift,
        tolerance=tolerance,
    )
    result["affine_validation"] = affine
    result["spglib_dataset"] = {
        "space_group_number": int(dataset.number),
        "hall_number": int(dataset.hall_number),
        "input_basis_operation_count": len(dataset.rotations),
        "constructed_orbit_count": len(seeds),
        "constructed_position_count": len(positions),
    }
    if affine.get("status") != "passed":
        result["status"] = (
            "rejected" if affine.get("status") == "failed" else "unavailable"
        )
        result["reason"] = (
            "spglib-derived subgroup standardization did not pass complete "
            "affine operation equivalence"
        )
        result["operation_basis_verification"] = {
            "status": "failed" if affine.get("status") == "failed" else "not_attempted",
            "reason": result["reason"],
        }
        return result

    result.update({
        "status": "accepted",
        "transform_matrix": transform.tolist(),
        "origin_shift_fractional": origin_shift.tolist(),
        "operation_basis_verification": {
            "status": "passed",
            "source": "complete_affine_operation_bijection",
        },
    })
    return result


def _align_bases_from_rotation_axes(
    *,
    parent_axis: np.ndarray,
    std_axis: np.ndarray,
    parent_rot: np.ndarray,
    std_rot: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray | None:
    """Compute the basis transform T mapping parent axis to standard axis.

    For a C2 rotation, the rotation axis is unique.  T is constructed
    by rotating the parent axis to align with the standard axis, then
    completing the basis with two orthogonal directions.

    Returns T (3×3) or None if the axes are degenerate.
    """
    # Check that axes are well-defined (nonzero norm)
    if np.linalg.norm(parent_axis) < 1e-10 or np.linalg.norm(std_axis) < 1e-10:
        return None

    # Normalize axes
    p_axis = parent_axis / np.linalg.norm(parent_axis)
    s_axis = std_axis / np.linalg.norm(std_axis)

    # For a proper rotation, the axis is preserved.  T maps parent_axis → std_axis.
    # We use Rodrigues' rotation formula to find the rotation that maps one to the
    # other, then express it in the parent fractional basis.

    cross = np.cross(p_axis, s_axis)
    dot = np.dot(p_axis, s_axis)

    if np.linalg.norm(cross) < 1e-10:
        # Axes are parallel — T is identity (up to lattice scaling)
        return np.eye(3)

    # Rotation matrix in Cartesian space that maps p_axis to s_axis
    cross_norm = np.linalg.norm(cross)
    k = cross / cross_norm
    # Rodrigues formula: R = I + sin(θ)·[k]× + (1-cos(θ))·[k]×²
    cos_theta = np.clip(dot, -1.0, 1.0)
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    R_cart = np.eye(3) + sin_theta * K + (1.0 - cos_theta) * (K @ K)

    # Express R_cart in fractional coordinates.
    # From fractional to Cartesian: x_cart = lattice^T · x_frac
    # So from Cartesian to fractional: x_frac = lattice^(-T) · x_cart
    lat_T = lattice.T
    lat_inv_T = np.linalg.inv(lat_T)
    T = lat_inv_T @ R_cart @ lat_T

    # T should be close to integer
    T_rounded = np.rint(T).astype(int)
    if not np.allclose(T, T_rounded, atol=1e-3):
        return T  # Non-integer transform — still usable but flagged

    return T_rounded.astype(float)


def _verify_operation_basis(
    *,
    parent_rotations: list[np.ndarray],
    std_rotations: list[np.ndarray],
    transform_matrix: np.ndarray,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Verify that transformed parent rotations match standard rotations.

    For each parent rotation R_p, compute R_s' = T · R_p · T⁻¹ and
    check whether it matches any standard rotation R_s within tolerance.
    """
    T = transform_matrix
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        return {"status": "failed", "reason": "singular transformation matrix"}

    matches: list[dict[str, object]] = []
    unmatched_parent: list[int] = []
    for i, r_p in enumerate(parent_rotations):
        r_transformed = T @ r_p @ T_inv
        found = False
        for j, r_s in enumerate(std_rotations):
            if np.allclose(r_transformed, r_s, atol=tolerance):
                matches.append({"parent_index": i, "std_index": j})
                found = True
                break
        if not found:
            unmatched_parent.append(i)

    if unmatched_parent:
        return {
            "status": "failed",
            "reason": (
                f"{len(unmatched_parent)} parent VP rotation(s) could "
                f"not be matched to standard-setting rotations "
                f"(indices: {unmatched_parent})"
            ),
            "matched_count": len(matches),
            "unmatched_count": len(unmatched_parent),
        }

    return {
        "status": "passed",
        "matched_count": len(matches),
        "unmatched_count": 0,
    }


def _compute_standard_setting_basis_transform(
    *,
    lattice_direct_cart: np.ndarray | None,
    vp_operations: list[dict[str, object]] | None,
    standard_match: dict[str, object],
) -> dict[str, object]:
    """Attempt to compute a standard-setting reciprocal-basis transform.

    Uses the parent direct lattice and valley-preserving operations to
    determine the basis-transformation matrix from the parent reciprocal
    basis to the standard-setting reciprocal basis.  Accepts the
    transformation only when operation matrices transformed into the
    candidate basis match the standard-setting operation set.

    Returns a dict with ``status`` and optional ``transform_matrix``,
    ``transformed_k_frac``, and/or ``reason``.
    """
    result: dict[str, object] = {"status": "not_attempted"}

    if lattice_direct_cart is None:
        result["status"] = "unavailable"
        result["reason"] = (
            "parent direct lattice (lattice_direct_cart) is not available; "
            "cannot compute standard-setting basis transform"
        )
        return result

    lattice = np.asarray(lattice_direct_cart, dtype=float)
    if lattice.shape != (3, 3):
        result["status"] = "unavailable"
        result["reason"] = (
            f"parent direct lattice has unexpected shape {lattice.shape}; "
            f"expected (3, 3)"
        )
        return result

    vp_op_ids = _operation_ids_list(standard_match)
    if vp_op_ids is None:
        result["status"] = "unavailable"
        result["reason"] = "standard group match operation_ids are malformed"
        return result
    if len(vp_op_ids) < 2:
        result["status"] = "unavailable"
        result["reason"] = (
            "fewer than two valley-preserving operations "
            "(identity + at least one nontrivial operation) "
            "in the standard match; cannot verify the basis "
            "orientation against operation content"
        )
        return result

    if vp_operations is None:
        result["status"] = "unavailable"
        result["reason"] = (
            "detected operation matrices (rotation_frac) are not "
            "available; cannot verify basis orientation"
        )
        return result

    # Extract non-identity VP rotation matrices to verify orientation.
    vp_set = set(vp_op_ids)
    vp_rotations: list[tuple[int, np.ndarray]] = []
    for op in vp_operations:
        if not isinstance(op, dict):
            continue
        op_id = op.get("operation_id")
        if not _is_exact_operation_id(op_id):
            continue
        if op_id not in vp_set:
            continue
        rot = op.get("rotation_frac")
        if rot is None:
            continue
        rotation = np.asarray(rot, dtype=float)
        if rotation.shape != (3, 3) or np.allclose(
            rotation, np.eye(3), atol=1e-8,
        ):
            continue
        vp_rotations.append((op_id, rotation))

    # 0. Diagnostic affine transform derivation from parent VP operations.
    #    Searches small-integer trial matrices T with det=±1 that map
    #    parent {R_p, τ_p} to standard {R_s, τ_s}.  This is a provenance/
    #    diagnostic step, not an accepted auto-derivation path —
    #    the search space is limited and ambiguous results are common
    #    for groups with few operations (e.g. C2).
    derivation = _derive_transform_candidate(
        vp_operations=list(vp_operations),
        vp_operation_ids=list(vp_op_ids),
        standard_match=dict(standard_match),
    )
    # Always record derivation attempt for provenance.
    result["derivation_attempt"] = {
        "status": derivation.get("status", "not_attempted"),
        "candidate_count": derivation.get("candidate_count"),
        "missing_ingredients": derivation.get("missing_ingredients", []),
        "derivation_provenance": derivation.get("derivation_provenance", ""),
    }

    if not vp_rotations:
        recon = {
            "status": "unavailable",
            "reason": (
                "no non-identity valley-preserving operation with a "
                "fractional rotation matrix found; cannot verify "
                "basis orientation against operation content"
            ),
            "operation_basis_verification": {
                "status": "not_attempted",
                "reason": "no nontrivial VP rotation available",
            },
        }
        recon["derivation_attempt"] = result.get("derivation_attempt")
        return recon

    # Attempt subgroup standard-cell reconstruction using VP operation
    # matrices and spglib's standard-setting symmetry database.
    recon = _reconstruct_subgroup_standard_cell(
        lattice_direct_cart=lattice,
        vp_operations=list(vp_operations),
        standard_match=standard_match,
    )
    recon["derivation_attempt"] = result.get("derivation_attempt")
    return recon


def _attempt_setting_transform(
    *,
    k_frac: np.ndarray,
    hall_number: int | None,
    hall_symbol: str,
    sg_number: int | None,
) -> dict[str, object]:
    """Attempt to transform k_frac using setting provenance.

    Currently implements a provenance-documenting structure.  Future
    work: use spglib basis-transformation matrices or Hall-symbol-
    directed centering transforms to map the parent-setting k_frac
    into the standard-setting reciprocal basis.

    The Hall symbol classifies the standard setting by centering
    type.  Primitive settings (P) share the same Bravais lattice
    class as the parent cell; centered settings (C, I, F, R) and
    rhombohedral settings (R) use a larger conventional cell where
    the reciprocal lattice differs from the primitive reciprocal
    lattice.
    """
    result: dict[str, object] = {}

    if hall_number is None or not hall_symbol:
        result["reason"] = (
            "Hall number/symbol not available from spglib match"
        )
        return result

    centering = _hall_centering_symbol(hall_symbol)
    if centering == "P":
        result["reason"] = (
            f"primitive setting (Hall {hall_symbol}); "
            "parent-to-standard lattice vectors both primitive; "
            "the difference is in Bravais type/orientation, "
            "not in centering.  A Bravais-type change requires "
            "the full direct-lattice transformation, which is not "
            "available for subgroup-only standard matches."
        )
    elif centering in ("C", "I", "F", "R"):
        result["reason"] = (
            f"centered setting (Hall {hall_symbol}); "
            "the conventional reciprocal lattice for a "
            f"{centering}-centered cell differs from the parent "
            "primitive reciprocal lattice.  The transformation "
            "requires the full direct-to-conventional "
            "reciprocal-basis mapping, which is not yet "
            "implemented."
        )
    else:
        result["reason"] = (
            f"unrecognised Hall symbol centering '{centering}' "
            f"(Hall {hall_symbol})"
        )

    if sg_number is not None:
        result["sg_number"] = int(sg_number)

    return result


def _derive_transform_candidate(
    *,
    vp_operations: list[dict[str, object]],
    vp_operation_ids: object,
    standard_match: dict[str, object],
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Derive a standard-setting transform candidate from affine operation data.

    Searches small-integer trial transform matrices T (entries in {-1,0,1},
    determinant ±1) and checks whether each T maps parent VP rotations to
    standard-setting rotations and parent translations to standard-setting
    translations modulo lattice + centering cosets.

    Returns a dict with ``status`` (``"unique_found"``, ``"ambiguous"``,
    or ``"unresolved"``) and optional ``transform_matrix`` and
    ``affine_validation`` provenance.
    """
    result: dict[str, object] = {
        "status": "not_attempted",
        "missing_ingredients": [],
    }

    if not isinstance(vp_operation_ids, list) or any(
        not _is_exact_operation_id(item) for item in vp_operation_ids
    ):
        result["status"] = "unresolved"
        result["missing_ingredients"].append("malformed_required_operation_ids")
        return result
    vp_id_set = set(vp_operation_ids)
    parent_ops: list[dict[str, object]] = []
    for op in vp_operations:
        if not isinstance(op, dict):
            continue
        oid = op.get("operation_id")
        if not _is_exact_operation_id(oid):
            continue
        if oid not in vp_id_set:
            continue
        rot = op.get("rotation_frac")
        if rot is None:
            continue
        parent_ops.append(op)

    if len(parent_ops) < 1:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("vp_operation_rotation_data")
        return result

    hall_number = standard_match.get("hall_number")
    if not isinstance(hall_number, int) or isinstance(hall_number, bool):
        result["status"] = "unresolved"
        result["missing_ingredients"].append("hall_number")
        return result

    try:
        import spglib
        std_sym = spglib.get_symmetry_from_database(int(hall_number))
    except Exception:
        std_sym = None

    if std_sym is None:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("spglib_database_access")
        return result

    std_rotations = [np.asarray(r, dtype=float) for r in std_sym["rotations"]]
    std_translations = [np.asarray(t, dtype=float) for t in std_sym["translations"]]
    hall_symbol = str(standard_match.get("hall_symbol", "") or "")
    centering_vectors = _centering_cosets(hall_symbol)
    if centering_vectors is None:
        if _hall_centering_symbol(hall_symbol) in ("A", "B", "C", "I", "F", "R"):
            result["status"] = "unresolved"
            result["missing_ingredients"].append("conventional_centering_vectors")
            return result
        centering_vectors = [np.array([0.0, 0.0, 0.0])]
    elif not centering_vectors:
        centering_vectors = [np.array([0.0, 0.0, 0.0])]

    # Build trial transform matrices: entries in {-1, 0, 1}, det = ±1
    trial_matrices: list[np.ndarray] = []
    entries = [-1, 0, 1]
    for a00 in entries:
        for a01 in entries:
            for a02 in entries:
                for a10 in entries:
                    for a11 in entries:
                        for a12 in entries:
                            for a20 in entries:
                                for a21 in entries:
                                    for a22 in entries:
                                        T = np.array(
                                            [[a00, a01, a02],
                                             [a10, a11, a12],
                                             [a20, a21, a22]],
                                            dtype=float,
                                        )
                                        det = np.linalg.det(T)
                                        if abs(abs(det) - 1.0) <= 1e-10:
                                            trial_matrices.append(T)

    valid_transforms: list[dict[str, object]] = []
    for T in trial_matrices:
        try:
            T_inv = np.linalg.inv(T)
        except np.linalg.LinAlgError:
            continue

        # Check rotation mapping: each parent rotation must map to a standard one
        all_rot_match = True
        for op in parent_ops:
            r_p = np.asarray(op["rotation_frac"], dtype=float)
            r_trans = np.rint(T @ r_p @ T_inv).astype(int)
            if not any(np.allclose(r_trans, r_s, atol=tolerance) for r_s in std_rotations):
                all_rot_match = False
                break

        if not all_rot_match:
            continue

        # Check translation mapping (if parent ops carry translations)
        all_trans_match = True
        has_translations = False
        for op in parent_ops:
            t_p = op.get("translation_frac")
            if t_p is None:
                continue
            has_translations = True
            t_p_arr = np.asarray(t_p, dtype=float)
            t_trans = T @ t_p_arr
            r_p = np.asarray(op["rotation_frac"], dtype=float)
            r_trans = T @ r_p @ T_inv  # float version, not rounded

            # Find standard op with matching rotation, compare translation
            found_trans = False
            for r_s, t_s in zip(std_rotations, std_translations):
                if not np.allclose(np.rint(r_trans).astype(int), r_s, atol=tolerance):
                    continue
                t_diff = t_trans - t_s
                for cv in centering_vectors:
                    t_diff_cv = t_diff - cv
                    t_diff_mod = t_diff_cv - np.rint(t_diff_cv)
                    if np.linalg.norm(t_diff_mod) <= tolerance:
                        found_trans = True
                        break
                if found_trans:
                    break
            if not found_trans:
                all_trans_match = False
                break

        if not all_trans_match:
            continue

        valid_transforms.append({
            "transform_matrix": T,
            "has_translation_validation": has_translations,
        })

    if len(valid_transforms) == 1:
        v = valid_transforms[0]
        result["status"] = "unique_found"
        result["transform_matrix"] = v["transform_matrix"].tolist()
        result["affine_validation"] = {
            "status": "passed" if v["has_translation_validation"] else "rotation_only",
            "derived": True,
        }
        result["derivation_provenance"] = (
            "small_integer_search over parent->standard rotation mapping; "
            f"{len(trial_matrices)} trial matrices, 1 valid"
        )
    elif len(valid_transforms) > 1:
        result["status"] = "ambiguous"
        result["candidate_count"] = len(valid_transforms)
        result["missing_ingredients"].append(
            "ambiguous_transform: multiple parent->standard transforms found"
        )
    else:
        result["status"] = "unresolved"
        result["missing_ingredients"].append(
            "no_valid_transform: no small-integer T maps parent rotations "
            "to standard rotations"
        )

    return result
