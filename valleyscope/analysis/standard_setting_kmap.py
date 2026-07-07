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

    # For the standard setting, we need the Hall symbol to determine
    # the conventional cell centering and unique-axis convention.
    hall_number = standard_match.get("hall_number")
    hall_symbol = standard_match.get("hall_symbol", "")
    centering = hall_symbol[0] if hall_symbol else ""

    if centering in ("C", "I", "F", "R"):
        result["status"] = "unavailable"
        result["reason"] = (
            f"centered conventional setting (Hall {hall_symbol}); "
            f"the {centering}-centered conventional reciprocal lattice "
            "differs from the parent primitive reciprocal lattice. "
            "The reciprocal-basis transformation from the parent "
            "primitive reciprocal basis to the conventional centered "
            "reciprocal basis requires the corresponding standard-setting "
            "direct cell (lattice parameters + Bravais type + centering "
            "vectors), which spglib does not provide for subgroup-only "
            "standard matches because the parent cell's full symmetry "
            "differs from the valley-preserving subgroup."
        )
        return result

    if centering == "P":
        # For primitive settings, the Bravais lattice class is the
        # same as the parent cell, but the conventional cell
        # orientation may differ.  The operation rotation matrices
        # in the parent fractional basis encode the orientation.
        # A full solution requires comparing the parent-setting
        # rotation to the irreptables standard-setting rotation for
        # each VP operation and solving for the basis change.
        result["status"] = "unavailable"
        result["reason"] = (
            f"primitive setting (Hall {hall_symbol}); the parent "
            "and standard reciprocal bases share the same Bravais "
            "lattice class, but the conventional-cell orientation "
            "may differ.  Orientation verification requires mapping "
            "each VP rotation matrix from the parent fractional "
            "basis to the standard-setting fractional basis, which "
            "has not been implemented for the general case."
        )
        return result

    result["status"] = "unavailable"
    result["reason"] = (
        f"unrecognised Hall symbol centering '{centering}' "
        f"(Hall {hall_symbol}); cannot determine standard-setting "
        "reciprocal basis"
    )
    return result


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
