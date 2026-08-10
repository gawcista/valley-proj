from __future__ import annotations

from urllib.parse import quote

import numpy as np

from valleyscope.symmetry.little_group import (
    DEFAULT_HSP_LITTLE_GROUP_K_RESIDUAL_TOLERANCE,
    hsp_little_group_evidence,
    is_little_group_operation,
)
from valleyscope.symmetry.double_space_group_lift import (
    spin_lift_from_orthogonal,
)
from valleyscope.symmetry.plane_wave_action import (
    DEFAULT_RECIPROCAL_GRID_MAPPING_TOLERANCE,
    RECIPROCAL_GRID_ACTION_CONVENTION,
    build_plane_wave_representation,
    reciprocal_grid_identity,
    unitarity_deviation,
)


def symmetry_eigenvalue_diagnostics_for_kpoint(
    *,
    kpoint_name: str,
    k_frac: np.ndarray,
    q_cart: np.ndarray,
    coefficients: np.ndarray,
    symmetry_payload: dict[str, object],
    basis_payload: dict[str, np.ndarray] | None,
    representation_payload: dict[str, object],
    valley_names: list[str] | None = None,
) -> list[dict[str, object]]:
    """Compute diagnostic eigenvalues for valley-preserving little-group operations.

    Readiness is not decided here.  The generic raw representation producer and
    ``scoped_representation_evidence`` own the C-prime trust boundary.

    Each row is keyed by (kpoint, operation, state_index, target_valley).
    """
    rows: list[dict[str, object]] = []

    if valley_names is None:
        valley_names = _infer_valley_names(symmetry_payload)

    lg_tol = float(symmetry_payload.get(
        "hsp_little_group_k_residual_tolerance",
        DEFAULT_HSP_LITTLE_GROUP_K_RESIDUAL_TOLERANCE,
    ))
    for operation in symmetry_payload["detected_operations"]:
        little = is_little_group_operation(
            np.asarray(operation["rotation_frac"]), k_frac, tolerance=lg_tol,
        )
        sector_mapping = operation.get("sector_mapping", {})
        preserved = operation.get("preserved", {})

        operation.setdefault("little_group_by_kpoint", {})[kpoint_name] = little

        # Per-valley classification
        for target_valley in valley_names:
            # Prefer sector_mapping, fall back to preserved dict
            mapped_valley = sector_mapping.get(target_valley)
            if mapped_valley is None and target_valley in preserved:
                mapped_valley = target_valley if preserved[target_valley] else None
            valley_preserves = bool(
                mapped_valley is not None and str(mapped_valley) == str(target_valley)
            )
            rejection_reason = _valley_rejection_reason_v11(
                little_group_passed=little,
                valley_preserving=valley_preserves,
                mapped_valley=mapped_valley,
            )

            operation.setdefault("rejection_reason_by_kpoint", {})[kpoint_name] = (
                _legacy_rejection_reason(little, preserved, operation.get("sector_mapping", {}))
            )

            if rejection_reason:
                continue

            # Build representation and eigenvalue rows for this (operation, target_valley)
            _append_operation_rows(
                rows=rows,
                operation=operation,
                kpoint_name=kpoint_name,
                k_frac=k_frac,
                q_cart=q_cart,
                coefficients=coefficients,
                basis_payload=basis_payload,
                representation_payload=representation_payload,
                little_group_passed=little,
                target_valley=target_valley,
                valley_preserving=valley_preserves,
            )

    return rows


def _append_operation_rows(
    *,
    rows: list[dict[str, object]],
    operation: dict[str, object],
    kpoint_name: str,
    k_frac: np.ndarray,
    q_cart: np.ndarray,
    coefficients: np.ndarray,
    basis_payload: dict[str, np.ndarray] | None,
    representation_payload: dict[str, object],
    little_group_passed: bool,
    target_valley: str,
    valley_preserving: bool,
) -> None:
    """Append eigenvalue rows for a single (operation, target_valley) pair."""

    spin_rotation = None
    spinor_rotation_applied = coefficients.shape[1] == 2
    if coefficients.shape[1] == 2:
        try:
            spin_rotation = spin_lift_from_orthogonal(
                np.asarray(operation["rotation_cart"])
            )
        except ValueError as exc:
            operation.setdefault("representation_quality", {})[kpoint_name] = {
                "skipped_reason": f"spinor rotation skipped: {exc}",
                "spinor_rotation_applied": False,
                "diagnostic_only": True,
            }
            return
    elif coefficients.shape[1] != 1:
        operation.setdefault("representation_quality", {})[kpoint_name] = {
            "skipped_reason": f"unsupported nspinor={coefficients.shape[1]}",
            "spinor_rotation_applied": False,
            "diagnostic_only": True,
        }
        return

    representation = build_plane_wave_representation(
        coefficients,
        q_cart,
        np.asarray(operation["rotation_cart"]),
        np.asarray(operation["translation_cart"]),
        spin_rotation=spin_rotation,
    )
    basis = "raw_diagnostic"
    matrix_for_eigen = representation.matrix
    d_valley = None
    valley_eta = None
    reason = "not valley-adapted"
    valid_valley_subspace = False
    d_block_leakage_norm = None

    if basis_payload is not None and "transform" in basis_payload:
        transform = np.asarray(basis_payload["transform"], dtype=np.complex128)
        d_valley = transform.conj().T @ representation.matrix @ transform
        valid_valley_subspace = bool(np.asarray(basis_payload.get("valid_valley_subspace", False)))
        assigned_valleys = _decode_assigned_valleys(basis_payload.get("assigned_valleys"))
        if assigned_valleys and target_valley:
            d_valley, d_block_leakage_norm = _select_valley_block(
                d_valley, assigned_valleys, target_valley
            )
        matrix_for_eigen = d_valley
        basis = "valley_adapted"
        valley_eta = np.asarray(basis_payload.get("eta", []), dtype=float)
        reason = "" if valid_valley_subspace else "valley subspace not clean"

    eigenvalues = np.linalg.eigvals(matrix_for_eigen)
    phases = np.angle(eigenvalues) / (2.0 * np.pi)
    modulus_deviation = np.abs(np.abs(eigenvalues) - 1.0)
    matrix_unitarity = unitarity_deviation(matrix_for_eigen)
    order = int(operation.get("order") or 1)
    plane_wave_mapping_complete = representation.mapping_miss_count == 0

    operation.setdefault("representation_quality", {})[kpoint_name] = {
        "mapping_miss_count": representation.mapping_miss_count,
        "norm_preservation_residual": representation.norm_preservation_residual,
        "unitarity_deviation": matrix_unitarity,
        "max_modulus_deviation": (
            float(np.max(modulus_deviation)) if len(modulus_deviation) else 0.0
        ),
        "plane_wave_mapping_complete": plane_wave_mapping_complete,
        "spinor_rotation_applied": spinor_rotation_applied,
        "diagnostic_only": True,
        "D_block_leakage_norm": np.nan if d_block_leakage_norm is None else d_block_leakage_norm,
        "basis": basis,
    }

    op_key = _representation_payload_key(operation["operation_id"], target_valley)
    op_payload: dict[str, object] = {
        "target_valley": target_valley,
        "source_operation_key": f"operation_{operation['operation_id']}",
        "D_raw": representation.matrix,
        "eigenvalues": eigenvalues,
        "plane_wave_mapping": representation.mapping,
        "plane_wave_action_convention": RECIPROCAL_GRID_ACTION_CONVENTION,
        "reciprocal_grid_identity": reciprocal_grid_identity(q_cart),
        "reciprocal_grid_dimension": int(len(q_cart)),
        "plane_wave_mapping_tolerance": (
            DEFAULT_RECIPROCAL_GRID_MAPPING_TOLERANCE
        ),
        "q_cart": np.asarray(q_cart, dtype=float),
        "mapping_miss_count": representation.mapping_miss_count,
        "norm_preservation_residual": representation.norm_preservation_residual,
        "relative_norm_preservation_residual": (
            representation.relative_norm_preservation_residual
        ),
        "unitarity_deviation": matrix_unitarity,
        "operation_order": int(operation["order"]),
        "rotation_frac": np.asarray(operation.get("rotation_frac", np.eye(3))),
        "translation_frac": np.asarray(operation.get("translation_frac", np.zeros(3))),
        "rotation_cart": np.asarray(operation["rotation_cart"]),
        "translation_cart": np.asarray(operation["translation_cart"]),
        "basis": basis,
        "plane_wave_mapping_complete": plane_wave_mapping_complete,
        "spinor_rotation_applied": spinor_rotation_applied,
        "local_irrep_ready": np.zeros(len(eigenvalues), dtype=bool),
        "diagnostic_only": np.ones(len(eigenvalues), dtype=bool),
        "D_block_leakage_norm": np.nan if d_block_leakage_norm is None else d_block_leakage_norm,
    }
    if d_valley is not None:
        op_payload["D_valley"] = d_valley
    representation_payload.setdefault(kpoint_name, {})[op_key] = op_payload

    char_raw = complex(np.trace(representation.matrix))
    char_valley = complex(np.trace(d_valley)) if d_valley is not None else None

    for state_index, (value, phase, modulus_error) in enumerate(
        zip(eigenvalues, phases, modulus_deviation)
    ):
        row_reason = reason or "awaiting scoped_representation_evidence"
        rows.append(
            {
                "kpoint": kpoint_name,
                "target_valley": target_valley,
                "operation_id": operation["operation_id"],
                "kind": operation.get("kind", ""),
                "order": order,
                "rotation_frac": str(np.asarray(operation.get("rotation_frac", [])).tolist()),
                "translation_frac": str(np.asarray(operation.get("translation_frac", [])).tolist()),
                "basis": basis,
                "state_index": state_index,
                "eigenvalue_real": float(value.real),
                "eigenvalue_imag": float(value.imag),
                "phase_2pi": float(phase),
                "modulus_deviation": float(modulus_error),
                "unitarity_deviation": float(matrix_unitarity),
                "character_raw": f"{char_raw.real:.6f}{char_raw.imag:+.6f}j",
                "character_valley": "" if char_valley is None else f"{char_valley.real:.6f}{char_valley.imag:+.6f}j",
                "little_group_passed": bool(little_group_passed),
                "valley_preserving": bool(valley_preserving),
                "plane_wave_mapping_complete": plane_wave_mapping_complete,
                "spinor_rotation_applied": spinor_rotation_applied,
                "local_irrep_ready": False,
                "diagnostic_only": True,
                "D_block_leakage_norm": "" if d_block_leakage_norm is None else d_block_leakage_norm,
                "reason": row_reason,
                "valley_eta": "" if valley_eta is None or state_index >= len(valley_eta) else float(valley_eta[state_index]),
            }
        )


def _select_valley_block(
    d_valley: np.ndarray,
    assigned_valleys: list[str],
    target_valley: str,
) -> tuple[np.ndarray, float | None]:
    """Extract the diagonal block for target_valley and compute leakage norm.

    Returns (block_matrix, leakage_norm) where leakage_norm is the Frobenius
    norm of the off-diagonal elements connecting target_valley states to other
    valley states.
    """
    n = d_valley.shape[0]
    target_indices = [i for i, v in enumerate(assigned_valleys) if v == target_valley]
    other_indices = [i for i in range(n) if i not in target_indices]

    if not target_indices:
        return d_valley, None

    block = d_valley[np.ix_(target_indices, target_indices)]

    if not other_indices:
        return block, 0.0

    # Leakage: elements connecting target to other valleys
    leakage_rows = d_valley[np.ix_(target_indices, other_indices)]
    leakage_cols = d_valley[np.ix_(other_indices, target_indices)]
    leakage_norm = float(np.sqrt(np.linalg.norm(leakage_rows) ** 2 + np.linalg.norm(leakage_cols) ** 2))

    return block, leakage_norm


def _representation_payload_key(operation_id: object, target_valley: str) -> str:
    return f"operation_{operation_id}__valley_{quote(str(target_valley), safe='')}"


def _decode_assigned_valleys(data: Any) -> list[str]:
    if data is None:
        return []
    arr = np.asarray(data)
    if arr.dtype.kind == "S":
        return [v.decode("utf-8") for v in arr]
    return [str(v) for v in arr]


def _infer_valley_names(symmetry_payload: dict[str, object]) -> list[str]:
    for op in symmetry_payload.get("detected_operations", []):
        preserved = op.get("preserved", {})
        if isinstance(preserved, dict) and preserved:
            return list(preserved.keys())
    return []


def _valley_rejection_reason_v11(
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


def _legacy_rejection_reason(
    little_group_passed: bool,
    preserved: dict[str, bool],
    sector_mapping: dict[str, Any],
) -> str:
    """Reproduce legacy rejection reason for backward compat."""
    if not little_group_passed:
        return "not in little group"
    if preserved and all(bool(v) for v in preserved.values()):
        return ""
    if sector_mapping:
        for source, target in sector_mapping.items():
            if target is not None and str(target) != str(source):
                return "valley-exchanging"
    return "not valley preserving"


def build_raw_representations_for_kpoint(
    *,
    kpoint_name: str,
    k_frac: np.ndarray,
    q_cart: np.ndarray,
    coefficients: np.ndarray,
    symmetry_payload: dict[str, object],
) -> dict[object, dict[str, object]]:
    """Build or explain D_raw once per (kpoint, operation_id).

    This is independent of the per-valley preservation gate and covers
    valley-permuting operations. Every little-group affine operation
    that cannot provide D_raw keep an explicit skipped_reason for audit output.

    Returns
    -------
    dict[operation_id, {"D_raw": ndarray, "kind": str, "order": int,
                         "sector_mapping": dict, "little_group_passed": bool}]
    """
    result: dict[object, dict[str, object]] = {}
    lg_tol = float(symmetry_payload.get(
        "hsp_little_group_k_residual_tolerance",
        DEFAULT_HSP_LITTLE_GROUP_K_RESIDUAL_TOLERANCE,
    ))

    for operation in symmetry_payload.get("detected_operations", []):
        if not isinstance(operation, dict):
            continue
        order = int(operation.get("order") or 1)
        operation_id = operation.get("operation_id")
        sector_mapping = operation.get("sector_mapping", {})
        if not isinstance(sector_mapping, dict):
            sector_mapping = {}

        little = is_little_group_operation(
            np.asarray(operation["rotation_frac"]), k_frac, tolerance=lg_tol,
        )
        if not little:
            result[operation_id] = _raw_representation_skip_payload(
                operation=operation,
                sector_mapping=sector_mapping,
                little_group_passed=False,
                reason="not in little group",
            )
            continue
        if not sector_mapping:
            result[operation_id] = _raw_representation_skip_payload(
                operation=operation,
                sector_mapping=sector_mapping,
                little_group_passed=True,
                reason="missing sector_mapping",
            )
            continue

        spin_rotation = None
        n_spinor = coefficients.shape[1]
        if n_spinor == 2:
            try:
                spin_rotation = spin_lift_from_orthogonal(
                    np.asarray(operation["rotation_cart"])
                )
            except ValueError as exc:
                result[operation_id] = _raw_representation_skip_payload(
                    operation=operation,
                    sector_mapping=sector_mapping,
                    little_group_passed=True,
                    reason=f"spinor rotation skipped: {exc}",
                )
                continue
        elif n_spinor != 1:
            result[operation_id] = _raw_representation_skip_payload(
                operation=operation,
                sector_mapping=sector_mapping,
                little_group_passed=True,
                reason=f"unsupported nspinor={n_spinor}",
            )
            continue

        representation = build_plane_wave_representation(
            coefficients,
            q_cart,
            np.asarray(operation["rotation_cart"]),
            np.asarray(operation["translation_cart"]),
            spin_rotation=spin_rotation,
        )
        if representation.mapping_miss_count > 0:
            result[operation_id] = _raw_representation_skip_payload(
                operation=operation,
                sector_mapping=sector_mapping,
                little_group_passed=True,
                reason=f"plane-wave mapping_miss_count={representation.mapping_miss_count}",
                mapping_miss_count=representation.mapping_miss_count,
            )
            continue

        result[operation_id] = {
            "D_raw": representation.matrix,
            "kind": operation.get("kind", ""),
            "order": order,
            "sector_mapping": dict(sector_mapping),
            "little_group_passed": True,
            "plane_wave_mapping": representation.mapping,
            "plane_wave_action_convention": (
                RECIPROCAL_GRID_ACTION_CONVENTION
            ),
            "reciprocal_grid_identity": reciprocal_grid_identity(q_cart),
            "reciprocal_grid_dimension": int(len(q_cart)),
            "plane_wave_mapping_tolerance": (
                DEFAULT_RECIPROCAL_GRID_MAPPING_TOLERANCE
            ),
            "q_cart": np.asarray(q_cart, dtype=float),
            "rotation_cart": np.asarray(
                operation["rotation_cart"], dtype=float
            ),
            "mapping_miss_count": representation.mapping_miss_count,
            "norm_preservation_residual": (
                representation.norm_preservation_residual
            ),
            "relative_norm_preservation_residual": (
                representation.relative_norm_preservation_residual
            ),
        }

    return result


def _raw_representation_skip_payload(
    *,
    operation: dict[str, object],
    sector_mapping: dict[str, object],
    little_group_passed: bool,
    reason: str,
    mapping_miss_count: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "D_raw": None,
        "kind": operation.get("kind", ""),
        "order": int(operation.get("order", 0)),
        "sector_mapping": dict(sector_mapping),
        "little_group_passed": bool(little_group_passed),
        "skipped_reason": reason,
    }
    if mapping_miss_count is not None:
        payload["mapping_miss_count"] = int(mapping_miss_count)
    return payload
