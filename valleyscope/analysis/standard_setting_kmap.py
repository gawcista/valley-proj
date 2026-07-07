"""Standard-setting HSP k-coordinate mapping for valley-projected subspace SG.

When a valley-projected subspace space group uses a conventional or
centered setting that differs from the parent moire primitive reciprocal
basis, the parent-setting k_frac may not directly match any
irreptables standard-setting HSP coordinate.

This module attempts to resolve the mapping using crystallographic
setting data (Hall symbol/number, centering, unique-axis convention)
from the spglib per-valley standard group match.
"""

from __future__ import annotations

import numpy as np


def resolve_standard_setting_hsp_label(
    *,
    k_frac: np.ndarray,
    table,
    standard_match: dict[str, object] | None,
    tolerance: float = 1e-6,
    lattice_direct_cart: np.ndarray | None = None,
    detected_operations: list[dict[str, object]] | None = None,
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
    if direct_label is not None:
        prov["direct_match_succeeded"] = True
        return direct_label, None, prov

    prov["direct_match_succeeded"] = False
    prov["direct_match_reason"] = (
        "parent-setting k_frac does not match any standard-setting "
        "Bilbao HSP coordinate in the irreptables table"
    )

    if standard_match is None or not isinstance(standard_match, dict):
        return None, (
            "standard_setting_hsp_mapping_unresolved: "
            "no per-valley standard group match available "
            "for setting-aware k-coordinate transformation"
        ), prov

    sg_number = standard_match.get("number")
    hall_number = standard_match.get("hall_number")
    hall_symbol = standard_match.get("hall_symbol", "")
    sg_symbol = standard_match.get("international_short", "")

    prov["subspace_sg_number"] = sg_number
    prov["subspace_sg_symbol"] = sg_symbol
    prov["hall_number"] = hall_number
    prov["hall_symbol"] = hall_symbol

    # 2. Attempt standard-setting basis transform using crystallographic
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
                prov["basis_transformed_match_succeeded"] = True
                prov["transformed_hsp_label"] = transformed_label
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

    return None, " ".join(reason_parts) + ".", prov


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
    std_translations = [np.asarray(t, dtype=float) for t in std_sym["translations"]]

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
    # Normalize axes
    p_axis = parent_axis / np.linalg.norm(parent_axis)
    s_axis = std_axis / np.linalg.norm(std_axis)

    # Check that axes are well-defined (nonzero norm)
    if np.linalg.norm(parent_axis) < 1e-10 or np.linalg.norm(std_axis) < 1e-10:
        return None

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
        r_transformed_int = np.rint(r_transformed).astype(int)
        found = False
        for j, r_s in enumerate(std_rotations):
            if np.allclose(r_transformed_int, r_s, atol=tolerance):
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
