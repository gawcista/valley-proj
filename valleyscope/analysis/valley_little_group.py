from __future__ import annotations

from typing import Any
import numpy as np
import spglib

from valleyscope.symmetry.little_group import (
    DEFAULT_HSP_LITTLE_GROUP_K_RESIDUAL_TOLERANCE,
    hsp_little_group_evidence,
    is_little_group_operation,
)


def update_valley_preserving_operation_inventory(
    *,
    symmetry_payload: dict[str, Any],
    kpoint_name: str,
    k_frac: np.ndarray,
    valley_names: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Classify detected operations per (kpoint, valley) pair."""
    if valley_names is None:
        valley_names = _infer_valley_names(symmetry_payload)

    tolerance = float(
        symmetry_payload.get(
            "hsp_little_group_k_residual_tolerance",
            DEFAULT_HSP_LITTLE_GROUP_K_RESIDUAL_TOLERANCE,
        )
    )

    per_valley: dict[str, list[dict[str, Any]]] = {}
    flat_inventory: list[dict[str, Any]] = []

    for operation in symmetry_payload.get("detected_operations", []):
        rotation = np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
        evidence = hsp_little_group_evidence(rotation, k_frac, tolerance=tolerance)
        little_group_passed = bool(evidence["passed"])
        sector_mapping = operation.get("sector_mapping", {})

        # Per-valley classification within the HSP little group.
        for valley_name in valley_names:
            mapped_valley = sector_mapping.get(valley_name)
            valley_preserving = bool(
                mapped_valley is not None and str(mapped_valley) == str(valley_name)
            )
            allowed = bool(little_group_passed and valley_preserving)
            reason = _per_valley_rejection_reason(
                little_group_passed=little_group_passed,
                valley_preserving=valley_preserving,
                mapped_valley=mapped_valley,
            )

            row = {
                "operation_id": operation.get("operation_id"),
                "kind": operation.get("kind", ""),
                "order": operation.get("order"),
                "little_group_passed": little_group_passed,
                "target_valley": valley_name,
                "mapped_valley": mapped_valley,
                "valley_preserving": valley_preserving,
                "allowed_for_valley_preserving_representation": allowed,
                "reason": reason,
                "hsp_little_group_k_residual_max_abs": evidence["residual_max_abs"],
                "hsp_little_group_k_residual_tolerance": evidence["tolerance"],
            }
            per_valley.setdefault(valley_name, []).append(row)

        # Flat all-valley-intersection row for diagnostics only.
        old_preserves_all = _preserves_all_valley_subspaces(operation)
        old_exchanging = _is_valley_exchanging(operation)
        old_allowed = bool(little_group_passed and old_preserves_all)
        old_reason = _rejection_reason(
            little_group_passed=little_group_passed,
            valley_preserving=old_preserves_all,
            valley_exchanging=old_exchanging,
        )
        flat_row = {
            "operation_id": operation.get("operation_id"),
            "kind": operation.get("kind", ""),
            "order": operation.get("order"),
            "little_group_passed": little_group_passed,
            "valley_preserving": old_preserves_all,
            "valley_exchanging": old_exchanging,
            "allowed_for_valley_preserving_representation": old_allowed,
            "reason": old_reason,
            "hsp_little_group_k_residual_max_abs": evidence["residual_max_abs"],
            "hsp_little_group_k_residual_tolerance": evidence["tolerance"],
        }
        flat_inventory.append(flat_row)

        operation.setdefault("little_group_by_kpoint", {})[kpoint_name] = little_group_passed
        operation.setdefault("allowed_for_valley_preserving_representation_by_kpoint", {})[kpoint_name] = old_allowed
        operation["allowed_for_valley_preserving_representation"] = old_allowed
        operation.setdefault("rejection_reason_by_kpoint", {})[kpoint_name] = old_reason

    symmetry_payload.setdefault("hsp_little_group_inventory", {})[kpoint_name] = flat_inventory
    symmetry_payload.setdefault("per_valley_preserving_operation_inventory", {}).setdefault(kpoint_name, {}).update(per_valley)
    symmetry_payload.setdefault("kpoint_frac_by_name", {})[kpoint_name] = np.asarray(
        k_frac, dtype=float,
    ).tolist()
    symmetry_payload.setdefault("valley_names", list(valley_names))
    return per_valley


def build_valley_preserving_subgroup_report(
    *,
    symmetry_payload: dict[str, Any],
    target_kpoints: list[str] | tuple[str, ...] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Build the valley orbit and valley-preserving subgroup report."""
    per_valley_inventories = symmetry_payload.get("per_valley_preserving_operation_inventory", {})
    valley_names = symmetry_payload.get("valley_names", [])
    inventories = symmetry_payload.get("hsp_little_group_inventory", {})

    if not isinstance(inventories, dict) or not inventories:
        return {
            "status": "not_available",
            "standard_group_match": None,
            "standard_group_match_status": "not_attempted",
            "by_kpoint": {},
        }

    operation_lookup = {
        operation.get("operation_id"): operation
        for operation in symmetry_payload.get("detected_operations", [])
    }

    # All-valley intersection is debug-only and not used for valley-preserving irreps.
    all_valley_intersection = _global_valley_preserving_operation_set_report(
        operation_lookup=operation_lookup,
        tolerance=tolerance,
    )
    all_valley_intersection["interpretation"] = (
        "intersection of operations preserving ALL selected valleys; "
        "this is NOT a per-valley preserving subgroup and is provided for debugging only"
    )

    valley_preserving_subgroups: dict[str, Any] = {}
    for valley_name in valley_names:
        preserving_ids = [
            op_id for op_id, op in operation_lookup.items()
            if _operation_preserves_valley(op, valley_name)
        ]
        closure = _operation_set_closure_report(
            allowed_ids=preserving_ids,
            operation_lookup=operation_lookup,
            tolerance=tolerance,
        )
        valley_preserving_subgroups[valley_name] = {
            "operation_ids": preserving_ids,
            "closure_status": closure["closure_status"],
            "identity_operation_id": closure["identity_operation_id"],
            "missing_products": closure["missing_products"],
        }

    # Valley orbits
    valley_orbits = _compute_valley_orbits(operation_lookup, valley_names)

    if target_kpoints is None:
        ordered_kpoints = list(inventories)
    else:
        ordered_kpoints = [kpoint for kpoint in target_kpoints if kpoint in inventories]
        ordered_kpoints.extend(kpoint for kpoint in inventories if kpoint not in ordered_kpoints)

    by_kpoint: dict[str, Any] = {}
    for kpoint in ordered_kpoints:
        kp_entry: dict[str, Any] = {}
        for valley_name in valley_names:
            # Get per-valley inventory for this (kpoint, valley)
            pv_inv = (
                per_valley_inventories.get(kpoint, {}).get(valley_name, [])
                if per_valley_inventories
                else []
            )
            if not pv_inv:
                # Fallback: build from flat inventory
                flat_inv = inventories.get(kpoint, [])
                if isinstance(flat_inv, list):
                    pv_inv = _build_per_valley_from_flat(flat_inv, valley_name, operation_lookup)

            allowed_ids = [
                row.get("operation_id")
                for row in pv_inv
                if bool(row.get("allowed_for_valley_preserving_representation", False))
            ]
            little_ids = [
                row.get("operation_id")
                for row in pv_inv
                if bool(row.get("little_group_passed", False))
            ]
            valley_changing_ids = [
                row.get("operation_id")
                for row in pv_inv
                if bool(row.get("little_group_passed", False))
                and not bool(row.get("valley_preserving", False))
                and row.get("mapped_valley") is not None
                and str(row.get("mapped_valley", "")) != str(valley_name)
            ]
            outside_little_ids = [
                row.get("operation_id")
                for row in pv_inv
                if not bool(row.get("little_group_passed", False))
            ]
            closure = _operation_set_closure_report(
                allowed_ids=allowed_ids,
                operation_lookup=operation_lookup,
                tolerance=tolerance,
            )
            kp_entry[valley_name] = {
                "little_group_operation_ids": little_ids,
                "allowed_operation_ids": allowed_ids,
                "valley_changing_operation_ids": valley_changing_ids,
                "outside_little_group_operation_ids": outside_little_ids,
                "operation_count": len(allowed_ids),
                "closure_status": closure["closure_status"],
                "identity_operation_id": closure["identity_operation_id"],
                "missing_products": closure["missing_products"],
            }

        # Legacy backward-compat fields at kpoint level
        flat_inv = inventories.get(kpoint, [])
        if isinstance(flat_inv, list):
            lg_ids = [
                row.get("operation_id")
                for row in flat_inv
                if bool(row.get("little_group_passed", False))
            ]
            old_allowed = [
                row.get("operation_id")
                for row in flat_inv
                if bool(row.get("allowed_for_valley_preserving_representation", False))
            ]
            kp_entry["hsp_little_group_operation_ids"] = lg_ids
            kp_entry["all_valley_intersection_operation_ids"] = old_allowed

        by_kpoint[kpoint] = kp_entry

    # Standard group match from per-valley preserving subgroups.
    per_valley_standard_matches: dict[str, Any] = {}
    for valley_name in valley_names:
        preserving_ids = valley_preserving_subgroups.get(valley_name, {}).get("operation_ids", [])
        match, match_status = _match_standard_group_from_operations(
            allowed_ids=preserving_ids,
            operation_lookup=operation_lookup,
            lattice_direct_cart=symmetry_payload.get("lattice_direct_cart"),
            symprec=float(symmetry_payload.get("symprec", tolerance)),
        )
        per_valley_standard_matches[valley_name] = {
            "standard_group_match": match,
            "standard_group_match_status": match_status,
        }

    status = "per_valley_preserving_subgroups_computed"
    report = {
        "status": status,
        "interpretation": (
            "For each valley a, G_k^(a) = {g in G_k | pi_g(a) = a}. "
            "Valley-preserving irreps use these subgroups, NOT the all-valley intersection."
        ),
        "valley_orbits": valley_orbits,
        "valley_preserving_subgroups": valley_preserving_subgroups,
        "per_valley_standard_matches": per_valley_standard_matches,
        "all_valley_intersection": all_valley_intersection,
        "by_kpoint": by_kpoint,
    }
    symmetry_payload["valley_preserving_subgroup_report"] = report
    return report

def _operation_preserves_valley(operation: dict[str, Any], valley_name: str) -> bool:
    mapping = operation.get("sector_mapping", {})
    target = mapping.get(valley_name)
    return target is not None and str(target) == str(valley_name)


def _preserves_all_valley_subspaces(operation: dict[str, Any]) -> bool:
    preserved = operation.get("preserved", {})
    if not isinstance(preserved, dict) or not preserved:
        return False
    return all(bool(value) for value in preserved.values())


def _per_valley_rejection_reason(
    *,
    little_group_passed: bool,
    valley_preserving: bool,
    mapped_valley: Any,
) -> str:
    if not little_group_passed:
        return "not in little group"
    if valley_preserving:
        return ""
    if mapped_valley is not None:
        return f"valley-changing (maps to {mapped_valley})"
    return "not valley preserving"


def _infer_valley_names(symmetry_payload: dict[str, Any]) -> list[str]:
    for op in symmetry_payload.get("detected_operations", []):
        preserved = op.get("preserved", {})
        if isinstance(preserved, dict) and preserved:
            return list(preserved.keys())
    mapping = op.get("sector_mapping", {}) if symmetry_payload.get("detected_operations") else {}
    if isinstance(mapping, dict) and mapping:
        return list(mapping.keys())
    return []


def _build_per_valley_from_flat(
    flat_inventory: list[dict[str, Any]],
    valley_name: str,
    operation_lookup: dict[Any, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fallback: build per-valley rows from flat inventory and operation data."""
    rows: list[dict[str, Any]] = []
    for flat_row in flat_inventory:
        op_id = flat_row.get("operation_id")
        op = operation_lookup.get(op_id, {})
        sector_mapping = op.get("sector_mapping", {})
        mapped_valley = sector_mapping.get(valley_name)
        valley_preserving = bool(
            mapped_valley is not None and str(mapped_valley) == str(valley_name)
        )
        little = bool(flat_row.get("little_group_passed", False))
        allowed = bool(little and valley_preserving)
        reason = _per_valley_rejection_reason(
            little_group_passed=little,
            valley_preserving=valley_preserving,
            mapped_valley=mapped_valley,
        )
        rows.append({
            "operation_id": op_id,
            "kind": flat_row.get("kind", ""),
            "order": flat_row.get("order"),
            "little_group_passed": little,
            "target_valley": valley_name,
            "mapped_valley": mapped_valley,
            "valley_preserving": valley_preserving,
            "allowed_for_valley_preserving_representation": allowed,
            "reason": reason,
        })
    return rows


def _compute_valley_orbits(
    operation_lookup: dict[Any, dict[str, Any]],
    valley_names: list[str],
) -> list[dict[str, Any]]:
    """Compute valley orbits under the detected operations."""
    if not valley_names:
        return []

    # Build adjacency: which valleys map to which
    adjacency: dict[str, set[str]] = {v: {v} for v in valley_names}
    for op in operation_lookup.values():
        mapping = op.get("sector_mapping", {})
        for src, tgt in mapping.items():
            if tgt is not None and src in adjacency:
                adjacency[src].add(str(tgt))

    # Find connected components
    visited: set[str] = set()
    orbits: list[dict[str, Any]] = []
    valley_order = {name: index for index, name in enumerate(valley_names)}

    for valley_name in valley_names:
        if valley_name in visited:
            continue
        # BFS
        component: list[str] = []
        queue = [valley_name]
        while queue:
            v = queue.pop(0)
            if v in visited:
                continue
            visited.add(v)
            component.append(v)
            for neighbor in sorted(
                adjacency.get(v, set()),
                key=lambda name: (valley_order.get(name, len(valley_order)), name),
            ):
                if neighbor not in visited:
                    queue.append(neighbor)

        # Operation mappings for this orbit
        op_mappings: list[dict[str, Any]] = []
        coset_reps: list[Any] = []
        for op_id, op in operation_lookup.items():
            mapping = op.get("sector_mapping", {})
            relevant = False
            for v in component:
                tgt = mapping.get(v)
                if tgt is not None and str(tgt) != str(v):
                    relevant = True
                    break
            if relevant:
                op_mappings.append({
                    "operation_id": op_id,
                    "kind": op.get("kind", ""),
                    "order": op.get("order"),
                    "mapping": {k: v for k, v in mapping.items() if k in component},
                })
            # Check if this operation maps between different valleys in the orbit
            mapped_set = set()
            for v in component:
                tgt = mapping.get(v)
                if tgt is not None:
                    mapped_set.add(str(tgt))
            if len(mapped_set) > 1:
                coset_reps.append(op_id)

        orbits.append({
            "valleys": component,
            "operation_mappings": op_mappings,
            "valley_permuting_operation_ids": coset_reps,
            "coset_representative_operation_ids": coset_reps,
        })

    return orbits


def _global_valley_preserving_operation_set_report(
    *,
    operation_lookup: dict[Any, dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    allowed_ids = [
        operation_id
        for operation_id, operation in operation_lookup.items()
        if _preserves_all_valley_subspaces(operation)
    ]
    valley_exchanging_ids = [
        operation_id
        for operation_id, operation in operation_lookup.items()
        if _is_valley_exchanging(operation)
    ]
    not_valley_preserving_ids = [
        operation_id
        for operation_id, operation in operation_lookup.items()
        if not _preserves_all_valley_subspaces(operation)
    ]
    closure = _operation_set_closure_report(
        allowed_ids=allowed_ids,
        operation_lookup=operation_lookup,
        tolerance=tolerance,
    )
    return {
        "operation_set_label": "all_valley_intersection",
        "interpretation": "intersection of operations preserving ALL selected valleys (debug only)",
        "allowed_operation_ids": allowed_ids,
        "valley_exchanging_operation_ids": valley_exchanging_ids,
        "not_valley_preserving_operation_ids": not_valley_preserving_ids,
        "operation_count": len(allowed_ids),
        "closure_status": closure["closure_status"],
        "identity_operation_id": closure["identity_operation_id"],
        "missing_products": closure["missing_products"],
    }


def _match_standard_group_from_operations(
    *,
    allowed_ids: list[Any],
    operation_lookup: dict[Any, dict[str, Any]],
    lattice_direct_cart: Any,
    symprec: float,
) -> tuple[dict[str, Any] | None, str]:
    if lattice_direct_cart is None:
        return None, "not_attempted"
    if not allowed_ids:
        return None, "not_matched"
    match_from_symmetry = getattr(spglib, "get_spacegroup_type_from_symmetry", None)
    if match_from_symmetry is None:
        return None, "not_matched"

    rotations = []
    translations = []
    for operation_id in allowed_ids:
        operation = operation_lookup.get(operation_id)
        if operation is None:
            return None, "not_matched"
        rotation = np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
        translation = np.asarray(operation.get("translation_frac", np.zeros(3)), dtype=float)
        rotations.append(np.rint(rotation).astype(int))
        translations.append(translation)

    try:
        spacegroup_type = match_from_symmetry(
            np.asarray(rotations, dtype=int),
            np.asarray(translations, dtype=float),
            lattice=np.asarray(lattice_direct_cart, dtype=float),
            symprec=symprec,
        )
    except Exception:
        return None, "not_matched"
    if spacegroup_type is None:
        return None, "not_matched"

    return {
        "number": int(spacegroup_type.number),
        "international_short": str(spacegroup_type.international_short),
        "international": str(spacegroup_type.international),
        "hall_number": int(spacegroup_type.hall_number),
        "hall_symbol": str(spacegroup_type.hall_symbol),
        "pointgroup_international": str(spacegroup_type.pointgroup_international),
        "source": "spglib.get_spacegroup_type_from_symmetry",
        "symprec": float(symprec),
        "operation_ids": list(allowed_ids),
    }, "matched"


def _is_valley_exchanging(operation: dict[str, Any]) -> bool:
    mapping = operation.get("sector_mapping", {})
    if not isinstance(mapping, dict):
        return False
    for source, target in mapping.items():
        if target is not None and str(target) != str(source):
            return True
    return False


def _rejection_reason(
    *,
    little_group_passed: bool,
    valley_preserving: bool,
    valley_exchanging: bool,
) -> str:
    if not little_group_passed:
        return "not in little group"
    if valley_preserving:
        return ""
    if valley_exchanging:
        return "valley-exchanging"
    return "not valley preserving"


def _operation_set_closure_report(
    *,
    allowed_ids: list[Any],
    operation_lookup: dict[Any, dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    missing_products: list[dict[str, Any]] = []
    identity_operation_id = _find_identity_operation_id(
        allowed_ids=allowed_ids,
        operation_lookup=operation_lookup,
        tolerance=tolerance,
    )
    if not allowed_ids:
        return {
            "closure_status": "empty",
            "identity_operation_id": identity_operation_id,
            "missing_products": missing_products,
        }

    for left_id in allowed_ids:
        for right_id in allowed_ids:
            left = operation_lookup.get(left_id)
            right = operation_lookup.get(right_id)
            if left is None or right is None:
                missing_products.append(
                    {"left_operation_id": left_id, "right_operation_id": right_id}
                )
                continue
            product_rotation, product_translation = _compose_fractional_operations(left, right)
            if _find_matching_operation_id(
                allowed_ids=allowed_ids,
                operation_lookup=operation_lookup,
                rotation_frac=product_rotation,
                translation_frac=product_translation,
                tolerance=tolerance,
            ) is None:
                missing_products.append(
                    {"left_operation_id": left_id, "right_operation_id": right_id}
                )

    return {
        "closure_status": "closed" if not missing_products else "not_closed",
        "identity_operation_id": identity_operation_id,
        "missing_products": missing_products,
    }


def _compose_fractional_operations(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    left_rotation = np.asarray(left.get("rotation_frac", np.eye(3)), dtype=float)
    right_rotation = np.asarray(right.get("rotation_frac", np.eye(3)), dtype=float)
    left_translation = np.asarray(left.get("translation_frac", np.zeros(3)), dtype=float)
    right_translation = np.asarray(right.get("translation_frac", np.zeros(3)), dtype=float)
    rotation = left_rotation @ right_rotation
    translation = left_rotation @ right_translation + left_translation
    return rotation, translation


def _find_identity_operation_id(
    *,
    allowed_ids: list[Any],
    operation_lookup: dict[Any, dict[str, Any]],
    tolerance: float,
) -> Any:
    for operation_id in allowed_ids:
        operation = operation_lookup.get(operation_id)
        if operation is None:
            continue
        if _rotation_matches(np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float), np.eye(3), tolerance):
            translation = np.asarray(operation.get("translation_frac", np.zeros(3)), dtype=float)
            if _translation_matches(translation, np.zeros(3), tolerance):
                return operation_id
    return None


def _find_matching_operation_id(
    *,
    allowed_ids: list[Any],
    operation_lookup: dict[Any, dict[str, Any]],
    rotation_frac: np.ndarray,
    translation_frac: np.ndarray,
    tolerance: float,
) -> Any:
    for operation_id in allowed_ids:
        operation = operation_lookup.get(operation_id)
        if operation is None:
            continue
        candidate_rotation = np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
        candidate_translation = np.asarray(operation.get("translation_frac", np.zeros(3)), dtype=float)
        if _rotation_matches(rotation_frac, candidate_rotation, tolerance) and _translation_matches(
            translation_frac, candidate_translation, tolerance,
        ):
            return operation_id
    return None


def _rotation_matches(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    return bool(np.allclose(left, right, atol=tolerance, rtol=0.0))


def _translation_matches(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    delta_mod_lattice = delta - np.rint(delta)
    return bool(np.linalg.norm(delta_mod_lattice) <= tolerance)
