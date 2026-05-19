from __future__ import annotations

from typing import Any

import numpy as np

from valleyscope.symmetry.little_group import is_little_group_operation


def update_valley_little_group_inventory(
    *,
    symmetry_payload: dict[str, Any],
    kpoint_name: str,
    k_frac: np.ndarray,
) -> list[dict[str, Any]]:
    """Classify detected operations for valley-preserving little-group diagnostics."""
    inventory: list[dict[str, Any]] = []
    for operation in symmetry_payload.get("detected_operations", []):
        rotation = np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
        little_group_passed = bool(is_little_group_operation(rotation, k_frac))
        valley_preserving = _preserves_all_valley_subspaces(operation)
        valley_exchanging = _is_valley_exchanging(operation)
        allowed = bool(little_group_passed and valley_preserving)
        reason = _rejection_reason(
            little_group_passed=little_group_passed,
            valley_preserving=valley_preserving,
            valley_exchanging=valley_exchanging,
        )

        row = {
            "operation_id": operation.get("operation_id"),
            "kind": operation.get("kind", ""),
            "order": operation.get("order"),
            "little_group_passed": little_group_passed,
            "valley_preserving": valley_preserving,
            "valley_exchanging": valley_exchanging,
            "allowed_for_single_valley_representation": allowed,
            "reason": reason,
        }
        inventory.append(row)

        operation.setdefault("little_group_by_kpoint", {})[kpoint_name] = little_group_passed
        operation.setdefault("allowed_for_single_valley_representation_by_kpoint", {})[kpoint_name] = allowed
        operation["allowed_for_single_valley_representation"] = allowed
        operation.setdefault("rejection_reason_by_kpoint", {})[kpoint_name] = reason

    symmetry_payload.setdefault("valley_little_group_inventory", {})[kpoint_name] = inventory
    return inventory


def build_valley_preserving_subgroup_report(
    *,
    symmetry_payload: dict[str, Any],
    target_kpoints: list[str] | tuple[str, ...] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    inventories = symmetry_payload.get("valley_little_group_inventory", {})
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
    if target_kpoints is None:
        ordered_kpoints = list(inventories)
    else:
        ordered_kpoints = [kpoint for kpoint in target_kpoints if kpoint in inventories]
        ordered_kpoints.extend(kpoint for kpoint in inventories if kpoint not in ordered_kpoints)

    by_kpoint: dict[str, Any] = {}
    for kpoint in ordered_kpoints:
        inventory = inventories.get(kpoint, [])
        if not isinstance(inventory, list):
            continue
        allowed_ids = [
            row.get("operation_id")
            for row in inventory
            if bool(row.get("allowed_for_single_valley_representation", False))
        ]
        little_ids = [
            row.get("operation_id")
            for row in inventory
            if bool(row.get("little_group_passed", False))
        ]
        valley_exchanging_ids = [
            row.get("operation_id")
            for row in inventory
            if bool(row.get("valley_exchanging", False))
        ]
        not_valley_preserving_ids = [
            row.get("operation_id")
            for row in inventory
            if bool(row.get("little_group_passed", False))
            and not bool(row.get("valley_preserving", False))
        ]
        outside_little_group_ids = [
            row.get("operation_id")
            for row in inventory
            if not bool(row.get("little_group_passed", False))
        ]
        closure = _operation_set_closure_report(
            allowed_ids=allowed_ids,
            operation_lookup=operation_lookup,
            tolerance=tolerance,
        )
        by_kpoint[kpoint] = {
            "operation_set_label": f"G_tau({kpoint})",
            "interpretation": "valley-preserving little-group operation set",
            "standard_group_match": None,
            "standard_group_match_status": "not_attempted",
            "little_group_operation_ids": little_ids,
            "allowed_operation_ids": allowed_ids,
            "valley_exchanging_operation_ids": valley_exchanging_ids,
            "not_valley_preserving_operation_ids": not_valley_preserving_ids,
            "outside_little_group_operation_ids": outside_little_group_ids,
            "operation_count": len(allowed_ids),
            "closure_status": closure["closure_status"],
            "identity_operation_id": closure["identity_operation_id"],
            "missing_products": closure["missing_products"],
        }

    report = {
        "status": "operation_set_only",
        "interpretation": (
            "G_tau is reported as a detected valley-preserving operation set; "
            "standard space-group matching and irrep-table matching are deferred"
        ),
        "standard_group_match": None,
        "standard_group_match_status": "not_attempted",
        "by_kpoint": by_kpoint,
    }
    symmetry_payload["valley_preserving_subgroup_report"] = report
    return report


def _preserves_all_valley_subspaces(operation: dict[str, Any]) -> bool:
    preserved = operation.get("preserved", {})
    if not isinstance(preserved, dict) or not preserved:
        return False
    return all(bool(value) for value in preserved.values())


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
