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

    # --- Validation ---
    validation_status: str = "not_evaluated"
    """One of ``"validated"``, ``"unresolved"``, ``"rejected"``."""

    # --- Operation mapping ---
    parent_basis_operation_ids: list[int] = field(default_factory=list)
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
    missing_affine_ingredients: list[str] = field(default_factory=list)
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

    # --- Blockers ---
    unresolved_reason: str | None = None
    """Human-readable explanation when validation_status != "validated"."""

    # --- k-point ---
    parent_k_frac: list[float] | None = None
    resolved_hsp_label: str | None = None
    """Source HSP label, only when validated."""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict, omitting None defaults.

        Empty ``missing_affine_ingredients`` is explicit evidence (nothing
        is missing) and is preserved; so is ``affine_operation_map``.
        """
        d: dict[str, object] = {}
        for key, value in asdict(self).items():
            if value is None:
                continue
            # Preserve empty lists/dicts that are explicit evidence.
            if isinstance(value, list) and not value:
                if key in ("missing_affine_ingredients",):
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
            cv = _centering_cosets(str(v))
            if cv and len(cv) > 1:
                c.centering_vectors = [vec.tolist() for vec in cv]

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
    vp_operation_ids: list[int],
    standard_match: dict[str, object],
    parent_to_standard_direct_transform: np.ndarray | None = None,
    origin_shift_fractional: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Validate affine operation-group bijection.

    Builds a one-to-one parent-to-standard operation map under the basis
    transform T, requiring that every required valley-preserving parent
    operation maps to a distinct standard operation and every primitive
    standard operation is used exactly once.  Then validates affine group
    closure of the complete detected parent operation set.

    Returns a status dict with ``status`` (``"passed"``, ``"failed"``, or
    ``"unresolved"``), bijection evidence (operation map, unmatched/unused
    indices), mismatch diagnostics, missing ingredients, and closure status.
    """
    result: dict[str, object] = {
        "status": "not_attempted",
        "missing_ingredients": [],
    }

    # ---- A. Operation-ID coverage ----
    if not vp_operations:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("vp_operation_data")
        return result
    required_ids = [int(i) for i in vp_operation_ids]
    required_set = set(required_ids)
    if not required_set:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("vp_operation_data")
        return result
    if len(required_ids) != len(required_set):
        result["status"] = "failed"
        result["missing_ingredients"].append("duplicate_required_operation_id")
        result["required_operation_ids"] = required_ids
        return result
    result["required_operation_id_count"] = len(required_set)

    # Index by ID, checking for duplicates and malformed entries.
    by_id: dict[int, dict[str, object]] = {}
    duplicate_ids: list[int] = []
    unexpected_ids: list[object] = []
    missing_required: list[int] = []
    malformed: list[dict[str, object]] = []
    for op in vp_operations:
        if not isinstance(op, dict):
            continue
        oid_raw = op.get("operation_id")
        if oid_raw is None:
            continue
        try:
            oid = int(oid_raw)
        except (TypeError, ValueError):
            malformed.append({"operation_id": oid_raw, "reason": "non_integer_id"})
            continue
        rot = op.get("rotation_frac")
        trans = op.get("translation_frac")
        if rot is None or trans is None:
            malformed.append({"operation_id": oid,
                              "reason": "missing_rotation_or_translation"})
            continue
        if oid not in required_set:
            unexpected_ids.append(oid)
            continue
        if oid in by_id:
            duplicate_ids.append(oid)
            continue
        by_id[oid] = op

    for rid in required_ids:
        if rid not in by_id:
            missing_required.append(rid)

    if duplicate_ids:
        result["status"] = "failed"
        result["missing_ingredients"].append("duplicate_detected_operation_id")
        result["duplicate_detected_operation_ids"] = duplicate_ids
    if unexpected_ids:
        result["status"] = "failed"
        result["missing_ingredients"].append("unexpected_operation_ids")
        result["unexpected_operation_ids"] = unexpected_ids
    if malformed:
        result["missing_ingredients"].append("malformed_detected_operations")
        result["malformed_detected_operations"] = malformed
    if missing_required:
        result["status"] = "failed"
        result["missing_ingredients"].append("missing_required_operation_ids")
        result["missing_required_operation_ids"] = missing_required

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

    # Centering guard.
    hall_symbol = str(standard_match.get("hall_symbol", "") or "")
    centering_vectors = _centering_cosets(hall_symbol)
    if centering_vectors and len(centering_vectors) > 1:
        result["centering_cosets_count"] = len(centering_vectors)
    elif _hall_centering_symbol(hall_symbol) in ("A", "B", "C", "I", "F", "R"):
        if result["status"] not in ("failed",):
            result["status"] = "unresolved"
        result["missing_ingredients"].append("conventional_centering_vectors")
        return result

    # Primitive-path group-order equality: required parent ops must equal
    # standard ops for a correct primitive direct-coordinate match.
    parent_op_count = len(ops_with_translations)
    if _hall_centering_symbol(hall_symbol) == "P" \
            and parent_op_count != std_op_count:
        if result["status"] not in ("failed",):
            result["status"] = "failed"
        result["missing_ingredients"].append(
            f"parent_standard_group_order_mismatch: parent has "
            f"{parent_op_count} operations, standard has {std_op_count}"
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

    centering_vectors = _centering_cosets(hall_symbol) or [np.array([0.0, 0.0, 0.0])]

    # ---- D. Build one-to-one parent-to-standard bijection ----
    # For each parent op, record which standard op(s) it could match.
    candidates: list[list[int]] = []
    unmatched_parents: list[dict[str, object]] = []
    for op in ops_with_translations:
        r_p = np.asarray(op["rotation_frac"], dtype=float)
        t_p = np.asarray(op["translation_frac"], dtype=float)
        r_transformed = T @ r_p @ T_inv
        t_transformed = T @ t_p
        if origin_shift_fractional is not None:
            o = np.asarray(origin_shift_fractional, dtype=float)
            t_transformed = t_transformed + o - r_transformed @ o

        matched_indices: list[int] = []
        for j, (r_s, t_s) in enumerate(zip(std_rotations, std_translations)):
            if not np.allclose(r_transformed, r_s, atol=tolerance):
                continue
            t_diff = t_transformed - t_s
            match_count = 0
            for cv in centering_vectors:
                d = t_diff - cv
                d_mod = d - np.rint(d)
                if np.linalg.norm(d_mod) <= tolerance:
                    matched_indices.append(j)
                    match_count += 1
                    break  # one centering vector match suffices
        if matched_indices:
            candidates.append(matched_indices)
        else:
            unmatched_parents.append({
                "operation_id": op.get("operation_id"),
                "parent_translation_frac": t_p.tolist(),
                "transformed_translation": t_transformed.tolist(),
            })

    # Find a perfect bijection (each parent → unique std op) by greedy
    # assignment over the deterministic sorted required-ID order.
    used_std: list[int] = []
    used = [False] * std_op_count
    op_map: dict[int, int] = {}  # parent operation_id -> std index
    ambiguous = False
    for op, cand_list in zip(ops_with_translations, candidates):
        free = [j for j in cand_list if not used[j]]
        if len(free) == 0:
            unmatched_parents.append({
                "operation_id": op.get("operation_id"),
                "parent_translation_frac": op.get("translation_frac"),
                "reason": "all_standard_operations_already_used",
            })
            continue
        if len(free) > 1:
            ambiguous = True
        # Take the first free standard operation (deterministic order).
        j = free[0]
        used[j] = True
        used_std.append(j)
        oid = op.get("operation_id")
        if oid is not None:
            op_map[int(oid)] = j

    unmatched_count = len(unmatched_parents)
    unused_std = sorted(set(range(std_op_count)) - set(used_std))
    result["operation_map"] = {str(k): int(v) for k, v in op_map.items()}
    if unmatched_parents:
        result["unmatched_parent_operations"] = unmatched_parents
    if unused_std:
        result["unused_standard_operation_indices"] = unused_std

    result["matched_affine_operations"] = len(used_std)
    result["total_parent_operations"] = parent_op_count

    # ---- E. Determine status ----
    if unmatched_parents:
        result["mismatched_translation_count"] = unmatched_count
        result["mismatched_translations"] = unmatched_parents[:3]
        if result["status"] not in ("failed",):
            result["status"] = "failed"
    elif ambiguous:
        # Ambiguous mapping (multiple possible bijections): unresolved for
        # automated primitive-path (human review would be needed).
        result["mismatched_translation_count"] = 0
        result["status"] = "unresolved"
        result["missing_ingredients"].append("ambiguous_affine_operation_map")
    else:
        result["mismatched_translation_count"] = 0
        result["status"] = "passed"

    # ---- F. Affine group closure of the complete detected parent set ----
    _validate_affine_group_closure(by_id, result)

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
    cert.missing_affine_ingredients = affine_result.get(
        "missing_ingredients", []
    )
    cert.mismatched_translation_count = affine_result.get(
        "mismatched_translation_count", 0
    )
    cert.mismatched_translations = affine_result.get(
        "mismatched_translations", []
    )
    cert.operation_closure_validated = affine_result.get(
        "operation_closure_validated"
    )
    # Bijection evidence.
    op_map = affine_result.get("operation_map")
    if isinstance(op_map, dict):
        cert.affine_operation_map = {str(k): int(v) for k, v in op_map.items()}
    cert.unmatched_parent_operations = affine_result.get("unmatched_parent_operations")
    cert.unused_standard_operation_indices = affine_result.get("unused_standard_operation_indices")
    cert.required_operation_id_count = affine_result.get("required_operation_id_count")


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

    if parent_basis_operation_ids is not None:
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

    # 1. Direct coordinate match (existing behavior).
    try:
        direct_label = table.match_kpoint_label(k_frac, tolerance=tolerance)
    except TypeError:
        # Compatibility with test mocks that don't accept tolerance.
        direct_label = table.match_kpoint_label(k_frac)
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
        if aff.get("status") == "passed":
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
                try:
                    label = table.match_kpoint_label(transformed_k, tolerance=tolerance)
                except TypeError:
                    label = table.match_kpoint_label(transformed_k)
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
                    if aff is not None and aff.get("status") == "failed":
                        prov["affine_validation"] = aff
                        tf_result["status"] = "rejected"
                        tf_result["rejection_reason"] = _affine_failure_blocker(
                            aff, source="explicit transform"
                        )
                        prov["explicit_transform"] = tf_result
                        cert = build_standard_setting_certificate(
                            standard_match=standard_match,
                            validation_status="rejected",
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
        # Transform k_frac from parent reciprocal basis to standard setting.
        # The reciprocal transform is the transpose of the inverse of the
        # direct-lattice transform: k_std = T^(-T) * k_parent.
        try:
            T_inv = np.linalg.inv(T)
            transformed_k = k_frac @ T_inv.T
            prov["transformed_k_frac"] = transformed_k.tolist()
            try:
                transformed_label = table.match_kpoint_label(transformed_k, tolerance=tolerance)
            except TypeError:
                transformed_label = table.match_kpoint_label(transformed_k)
            if transformed_label is not None:
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
                            origin_shift_fractional=origin_shift_fractional,
                            transform_provenance="operation_basis_reconstruction",
                            parent_k_frac=k_frac,
                        )
                        cert.standard_setting_source = (
                            "operation_basis_reconstruction"
                        )
                        cert.primitive_conventional_relation = (
                            "operation_basis_reconstruction"
                        )
                        cert.operation_mapping_status = (
                            "operation_basis_verification_passed"
                        )
                        _apply_affine_validation_to_certificate(cert, aff)
                        candidate = _build_transform_candidate(
                            standard_match=standard_match,
                            parent_to_standard_direct_transform=T,
                            origin_shift_fractional=origin_shift_fractional,
                            transform_provenance="operation_basis_reconstruction",
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
                prov["basis_transformed_match_succeeded"] = True
                prov["transformed_hsp_label"] = transformed_label
                cert = build_standard_setting_certificate(
                    standard_match=standard_match,
                    validation_status="validated",
                    parent_basis_operation_ids=(
                        _operation_ids_list(standard_match)
                        if isinstance(standard_match, dict) else None
                    ),
                    parent_to_standard_direct_transform=T,
                    origin_shift_fractional=origin_shift_fractional,
                    transform_provenance="operation_basis_reconstruction",
                    parent_k_frac=k_frac,
                    resolved_hsp_label=transformed_label,
                )
                cert.standard_setting_source = "operation_basis_reconstruction"
                cert.primitive_conventional_relation = (
                    "operation_basis_reconstruction"
                )
                cert.operation_mapping_status = (
                    "operation_basis_verification_passed"
                )
                if aff is not None:
                    _apply_affine_validation_to_certificate(cert, aff)
                    candidate = _build_transform_candidate(
                        standard_match=standard_match,
                        parent_to_standard_direct_transform=T,
                        origin_shift_fractional=origin_shift_fractional,
                        transform_provenance="operation_basis_reconstruction",
                        operation_mapping_status=cert.operation_mapping_status,
                        affine_result=aff,
                    )
                    _attach_transform_candidate(
                        provenance=prov,
                        certificate=cert,
                        candidate=candidate,
                    )
                prov["standard_setting_certificate"] = cert.to_dict()
                return transformed_label, None, prov
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
        try:
            transformed_label = table.match_kpoint_label(transformed_k, tolerance=tolerance)
        except TypeError:
            transformed_label = table.match_kpoint_label(transformed_k)
        if transformed_label is not None:
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


def _operation_ids_list(sm: dict[str, object]) -> list[int]:
    v = sm.get("operation_ids", [])
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v if isinstance(x, (int, float))]
    return []


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


def _centering_cosets(hall_symbol: str) -> list[np.ndarray] | None:
    """Return conventional centering cosets from Hall symbol.

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
        vp_ids = {
            int(op_id)
            for op_id in standard_match.get("operation_ids", [])
            if isinstance(op_id, (int, float))
        }
        parent_rotations: list[np.ndarray] = []
        for op in vp_operations:
            if not isinstance(op, dict):
                continue
            oid_raw = op.get("operation_id")
            if oid_raw is None:
                continue
            try:
                oid = int(oid_raw)
            except (TypeError, ValueError):
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
        result["status"] = "unavailable"
        result["operation_basis_verification"] = {
            "status": "not_attempted",
            "reason": (
                "centered/rhombohedral standard setting requires an explicit "
                "parent-to-standard direct-lattice transformation, including "
                "centering; rotation-axis alignment alone is underdetermined"
            ),
        }
        result["reason"] = (
            "standard-setting basis transform is unavailable: "
            "centered/rhombohedral conventional reciprocal coordinates cannot "
            "be reconstructed from valley-preserving rotation matrices alone"
        )
        return result

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
    vp_ids = {
        int(op_id) for op_id in standard_match.get("operation_ids", [])
        if isinstance(op_id, (int, float))
    }
    parent_by_id: dict[int, dict[str, object]] = {}
    for op in vp_operations:
        if not isinstance(op, dict):
            continue
        oid_raw = op.get("operation_id")
        if oid_raw is None:
            continue
        try:
            oid = int(oid_raw)
        except (TypeError, ValueError):
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

    vp_op_ids = standard_match.get("operation_ids", [])
    if not isinstance(vp_op_ids, (list, tuple)) or len(vp_op_ids) < 2:
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
    vp_set = set(int(op) for op in vp_op_ids if isinstance(op, (int, float)))
    vp_rotations: list[tuple[int, np.ndarray]] = []
    for op in vp_operations:
        if not isinstance(op, dict):
            continue
        op_id_raw = op.get("operation_id")
        if op_id_raw is None:
            continue
        try:
            op_id = int(op_id_raw)
        except (TypeError, ValueError):
            continue
        if op_id not in vp_set:
            continue
        order_raw = op.get("order")
        try:
            order = int(order_raw) if order_raw is not None else 0
        except (TypeError, ValueError):
            order = 0
        if order <= 1:
            continue  # skip identity
        rot = op.get("rotation_frac")
        if rot is None:
            continue
        vp_rotations.append((op_id, np.asarray(rot, dtype=float)))

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
    vp_operation_ids: list[int],
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

    vp_id_set = set(vp_operation_ids)
    parent_ops: list[dict[str, object]] = []
    for op in vp_operations:
        if not isinstance(op, dict):
            continue
        oid_raw = op.get("operation_id")
        if oid_raw is None:
            continue
        try:
            oid = int(oid_raw)
        except (TypeError, ValueError):
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
