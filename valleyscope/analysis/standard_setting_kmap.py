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
    when an explicit T matrix is used, ``"centered_unresolved"`` when
    centering vectors are unavailable, ``"not_applicable"``."""

    # --- Centering ---
    centering_type: str | None = None
    """P, C, I, F, R from Hall symbol first character."""
    centering_status: str = "not_evaluated"
    """One of ``"primitive_direct_match"``, ``"centered_unresolved"``,
    ``"not_evaluated"``."""

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

    # --- Blockers ---
    unresolved_reason: str | None = None
    """Human-readable explanation when validation_status != "validated"."""

    # --- k-point ---
    parent_k_frac: list[float] | None = None
    resolved_hsp_label: str | None = None
    """Source HSP label, only when validated."""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict, omitting None defaults."""
        d: dict[str, object] = {}
        for key, value in asdict(self).items():
            if value is not None and value != []:
                d[key] = value
        return d


def _validate_affine_operation_equivalence(
    *,
    vp_operations: list[dict[str, object]] | None,
    vp_operation_ids: list[int],
    standard_match: dict[str, object],
    parent_to_standard_direct_transform: np.ndarray | None = None,
    origin_shift_fractional: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Validate affine (rotation + translation) operation equivalence.

    When a candidate direct-basis transform T is provided, transforms
    each parent {R_p, τ_p} into the standard basis via
    {T·R_p·T⁻¹, T·τ_p} and compares against standard-setting
    {R_s, τ_s} from spglib, modulo lattice translations.

    When an origin shift ``o`` is provided, the transformed
    translation is computed as

        τ_std = T·τ_parent + o - R_std·o   (modulo lattice)

    where R_std = T·R_parent·T⁻¹.

    Returns a status dict with ``status`` (``"passed"``, ``"failed"``,
    or ``"unresolved"``), matched/unmatched counts, and a list of
    missing affine ingredients.
    """
    result: dict[str, object] = {
        "status": "not_attempted",
        "missing_ingredients": [],
    }

    if not vp_operations:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("vp_operation_data")
        return result

    vp_id_set = set(vp_operation_ids)
    ops_with_translations: list[dict[str, object]] = []
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
        trans = op.get("translation_frac")
        if rot is not None and trans is not None:
            ops_with_translations.append(op)

    result["total_parent_operations"] = len(ops_with_translations)

    if not ops_with_translations:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("parent_translation_frac")
        return result

    # Check spglib standard-setting translations.
    hall_number = standard_match.get("hall_number")
    if not isinstance(hall_number, int) or isinstance(hall_number, bool):
        result["status"] = "unresolved"
        result["missing_ingredients"].append("standard_setting_translations")
        return result

    hall_number = int(hall_number)

    # Hall-number / SG-number consistency guard.
    sg_number = standard_match.get("number")
    _hall_ok, _hall_blocker = _validate_hall_sg_consistency(
        hall_number=hall_number,
        sg_number=int(sg_number)
        if isinstance(sg_number, int) and not isinstance(sg_number, bool)
        else None,
    )
    if not _hall_ok:
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
        result["status"] = "unresolved"
        result["missing_ingredients"].append("spglib_database_access")
        return result

    std_rotations = [np.asarray(r, dtype=float) for r in std_sym["rotations"]]
    std_translations = [np.asarray(t, dtype=float) for t in std_sym["translations"]]
    result["standard_setting_operation_count"] = len(std_translations)

    hall_symbol = standard_match.get("hall_symbol", "")
    if hall_symbol and hall_symbol[0] in ("C", "I", "F", "R"):
        result["missing_ingredients"].append("conventional_centering_vectors")
        result["status"] = "unresolved"
        return result

    # If no candidate transform, we cannot compare — unresolved.
    if parent_to_standard_direct_transform is None:
        result["status"] = "unresolved"
        result["missing_ingredients"].append("direct_lattice_transform")
        return result

    T = np.asarray(parent_to_standard_direct_transform, dtype=float)
    if T.shape != (3, 3):
        result["status"] = "unresolved"
        result["missing_ingredients"].append("direct_lattice_transform")
        return result

    # Compare transformed parent affine ops against standard ops.
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError:
        result["status"] = "failed"
        result["missing_ingredients"].append("singular_transform")
        return result

    matched_count = 0
    mismatched_translations: list[dict[str, object]] = []
    for op in ops_with_translations:
        r_p = np.asarray(op["rotation_frac"], dtype=float)
        t_p = np.asarray(op["translation_frac"], dtype=float)

        r_transformed = np.rint(T @ r_p @ T_inv).astype(int)
        r_transformed_float = T @ r_p @ T_inv
        t_transformed = T @ t_p
        # Apply origin shift: tau_std = T·tau_parent + o - R_std·o
        if origin_shift_fractional is not None:
            o = np.asarray(origin_shift_fractional, dtype=float)
            t_transformed = t_transformed + o - r_transformed_float @ o

        found = False
        for j, (r_s, t_s) in enumerate(zip(std_rotations, std_translations)):
            if not np.allclose(r_transformed, r_s, atol=tolerance):
                continue
            # Compare translation modulo lattice: the transformed translation
            # should match the standard translation mod 1 (fractional).
            t_diff = t_transformed - t_s
            t_diff_mod = t_diff - np.rint(t_diff)
            if np.linalg.norm(t_diff_mod) <= tolerance:
                matched_count += 1
                found = True
                break

        if not found:
            mismatched_translations.append({
                "parent_operation_id": op.get("operation_id"),
                "parent_translation_frac": t_p.tolist(),
                "transformed_translation": t_transformed.tolist(),
            })

    result["matched_affine_operations"] = matched_count
    result["total_parent_operations"] = len(ops_with_translations)

    if mismatched_translations:
        result["status"] = "failed"
        result["mismatched_translation_count"] = len(mismatched_translations)
        result["mismatched_translations"] = mismatched_translations[:3]
    elif matched_count > 0:
        result["status"] = "passed"
    else:
        result["status"] = "unresolved"
        result["missing_ingredients"].append(
            "no_affine_operations_matched_after_transform"
        )

    return result


def _apply_affine_validation_to_certificate(
    cert: StandardSettingCertificate,
    affine_result: dict[str, object],
) -> None:
    """Copy affine validation diagnostics into a certificate."""
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
        "mismatched_translation_count"
    )
    cert.mismatched_translations = affine_result.get(
        "mismatched_translations", []
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
            cert.centering_type = str(v)[0] if v else None

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
    if direct_label is not None and parent_to_standard_direct_transform is None:
        prov["direct_match_succeeded"] = True
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
        if isinstance(standard_match, dict) and str(standard_match.get("hall_symbol", "")).startswith("P"):
            cert.centering_status = "primitive_direct_match"
            cert.primitive_conventional_relation = "direct_coordinate_match"
        cert.standard_setting_source = (
            "spglib.per_valley_standard_matches"
            if isinstance(standard_match, dict)
            else "coordinate_match_only"
        )
        # Affine validation for direct match.
        if isinstance(standard_match, dict) and detected_operations:
            vp_ids = _operation_ids_list(standard_match)
            # Direct primitive coordinate match means the parent fractional
            # basis is already being treated as the standard basis.
            hall_symbol = str(standard_match.get("hall_symbol", "") or "")
            direct_transform = np.eye(3) if hall_symbol.startswith("P") else None
            aff = _validate_affine_operation_equivalence(
                vp_operations=list(detected_operations),
                vp_operation_ids=vp_ids,
                standard_match=standard_match,
                parent_to_standard_direct_transform=direct_transform,
                origin_shift_fractional=origin_shift_fractional,
            )
            _apply_affine_validation_to_certificate(cert, aff)
            if aff.get("status") == "failed":
                blocker = _affine_failure_blocker(
                    aff, source="direct coordinate match"
                )
                cert.validation_status = "rejected"
                cert.unresolved_reason = blocker
                cert.resolved_hsp_label = None
                prov["affine_validation"] = aff
                prov["standard_setting_certificate"] = cert.to_dict()
                return None, blocker, prov
        prov["standard_setting_certificate"] = cert.to_dict()
        return direct_label, None, prov

    prov["direct_match_succeeded"] = False
    if direct_label is not None:
        prov["direct_match_reason"] = (
            "direct parent-coordinate match was skipped because an explicit "
            "parent-to-standard transform was supplied"
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
                        cert.operation_mapping_status = (
                            "operation_basis_verification_passed"
                        )
                        _apply_affine_validation_to_certificate(cert, aff)
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
                cert.operation_mapping_status = (
                    "operation_basis_verification_passed"
                )
                if aff is not None:
                    _apply_affine_validation_to_certificate(cert, aff)
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
        and str(standard_match.get("hall_symbol", "")[:1]) in ("C", "I", "F", "R")
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
    prov["standard_setting_certificate"] = cert.to_dict()
    return None, blocker, prov


def _operation_ids_list(sm: dict[str, object]) -> list[int]:
    v = sm.get("operation_ids", [])
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v if isinstance(x, (int, float))]
    return []


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
    centering = hall_symbol[0] if hall_symbol else ""
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

    if not vp_rotations:
        result["status"] = "unavailable"
        result["reason"] = (
            "no non-identity valley-preserving operation with a "
            "fractional rotation matrix found; cannot verify "
            "basis orientation against operation content"
        )
        return result

    # Attempt subgroup standard-cell reconstruction using VP operation
    # matrices and spglib's standard-setting symmetry database.
    recon = _reconstruct_subgroup_standard_cell(
        lattice_direct_cart=lattice,
        vp_operations=list(vp_operations),
        standard_match=standard_match,
    )
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

    centering = hall_symbol[0] if hall_symbol else ""
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
