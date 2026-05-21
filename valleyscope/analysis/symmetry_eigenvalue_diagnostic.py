from __future__ import annotations

from urllib.parse import quote

import numpy as np

from valleyscope.symmetry.little_group import is_little_group_operation
from valleyscope.symmetry.operation_classifier import rotation_axis_angle
from valleyscope.symmetry.plane_wave_action import build_plane_wave_representation, spin_rotation_matrix
from valleyscope.symmetry.rotation_eigenvalues import extract_rotation_eigenvalues, nearest_root_of_unity


unitarity_tol = 1e-4
ROOT_DEVIATION_TOL = 1e-6
D_VALLEY_OFFDIAG_TOL = 1e-6


def symmetry_eigenvalue_diagnostics_for_kpoint(
    *,
    kpoint_name: str,
    k_frac: np.ndarray,
    q_cart: np.ndarray,
    coefficients: np.ndarray,
    symmetry_payload: dict[str, object],
    basis_payload: dict[str, np.ndarray] | None,
    representation_payload: dict[str, object],
    spinor_convention_verified: bool = False,
    spinor_convention: str = "vasp_up_down_saxis_z",
    spinor_benchmark: str | None = None,
    unitarity_tol: float = unitarity_tol,
    root_deviation_tol: float = ROOT_DEVIATION_TOL,
    d_valley_offdiag_tol: float = D_VALLEY_OFFDIAG_TOL,
    valley_names: list[str] | None = None,
    generators_only: bool = False,
) -> list[dict[str, object]]:
    """Compute symmetry eigenvalue diagnostics for all proper little-group operations.

    V1.1: enumerates ALL detected proper little-group operations (order 2,3,4,6)
    and checks per-valley preservation.  ``rotation_order`` does NOT control
    which operations enter the analysis.

    Each row is keyed by (kpoint, operation, state_index, target_valley).
    """
    rows: list[dict[str, object]] = []

    if valley_names is None:
        valley_names = _infer_valley_names(symmetry_payload)

    for operation in symmetry_payload["detected_operations"]:
        order = operation.get("order")
        if operation.get("det", 1) != 1 or order not in (2, 3, 4, 6):
            continue
        if generators_only and not operation.get("candidate_rotation", False):
            continue

        little = is_little_group_operation(np.asarray(operation["rotation_frac"]), k_frac)
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
            allowed = bool(little and valley_preserves)
            rejection_reason = _valley_rejection_reason_v11(
                little_group_passed=little,
                valley_preserving=valley_preserves,
                mapped_valley=mapped_valley,
            )

            # Record rejection reason for backward compat
            operation.setdefault("rejection_reason_by_kpoint", {})[kpoint_name] = (
                _legacy_rejection_reason(little, preserved, operation.get("sector_mapping", {}))
            )
            # Legacy all-valley compat
            old_preserves_all = all(bool(v) for v in preserved.values()) if preserved else False
            operation["allowed_for_single_valley_representation"] = bool(little and old_preserves_all)
            operation.setdefault("allowed_for_single_valley_representation_by_kpoint", {})[kpoint_name] = bool(little and old_preserves_all)

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
                spinor_convention_verified=spinor_convention_verified,
                spinor_convention=spinor_convention,
                spinor_benchmark=spinor_benchmark,
                unitarity_tol=unitarity_tol,
                root_deviation_tol=root_deviation_tol,
                d_valley_offdiag_tol=d_valley_offdiag_tol,
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
    spinor_convention_verified: bool,
    spinor_convention: str,
    spinor_benchmark: str | None,
    unitarity_tol: float,
    root_deviation_tol: float,
    d_valley_offdiag_tol: float,
    little_group_passed: bool,
    target_valley: str,
    valley_preserving: bool,
) -> None:
    """Append eigenvalue rows for a single (operation, target_valley) pair."""

    spin_rotation = None
    spinor_rotation_applied = False
    spinor_verified = coefficients.shape[1] == 1
    if coefficients.shape[1] == 2:
        try:
            axis, angle = rotation_axis_angle(np.asarray(operation["rotation_cart"]))
            spin_rotation = spin_rotation_matrix(axis, angle)
            spinor_rotation_applied = True
            spinor_verified = bool(spinor_convention_verified)
        except ValueError as exc:
            operation.setdefault("representation_quality", {})[kpoint_name] = {
                "skipped_reason": f"spinor rotation skipped: {exc}",
                "spinor_rotation_applied": False,
                "spinor_convention_verified": False,
                "diagnostic_only": True,
            }
            return
    elif coefficients.shape[1] != 1:
        operation.setdefault("representation_quality", {})[kpoint_name] = {
            "skipped_reason": f"unsupported nspinor={coefficients.shape[1]}",
            "spinor_rotation_applied": False,
            "spinor_convention_verified": False,
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

    eigen = extract_rotation_eigenvalues(matrix_for_eigen, spinor_convention_verified=spinor_verified)
    rotation_ready = bool(
        representation.mapping_miss_count == 0 and eigen.unitarity_deviation <= unitarity_tol
    )
    d_valley_offdiag_norm = _two_sector_offdiag_norm(d_valley)
    order = int(operation["order"])
    root_order = 2 * order if spinor_rotation_applied else order
    root_info = [nearest_root_of_unity(value, order=root_order) for value in eigen.eigenvalues]
    root_deviations = np.asarray([item[2] for item in root_info], dtype=float)
    topology_input_ready_by_state = np.asarray(
        [
            _topology_input_ready(
                rotation_ready=rotation_ready,
                basis=basis,
                valid_valley_subspace=valid_valley_subspace,
                spinor_convention_verified=spinor_verified,
                root_deviation=root_deviation,
                d_valley_offdiag_norm=d_valley_offdiag_norm,
                d_block_leakage_norm=d_block_leakage_norm,
                root_deviation_tol=root_deviation_tol,
                d_valley_offdiag_tol=d_valley_offdiag_tol,
            )
            for root_deviation in root_deviations
        ],
        dtype=bool,
    )
    diagnostic_any = bool(np.any(~topology_input_ready_by_state))

    operation.setdefault("representation_quality", {})[kpoint_name] = {
        "mapping_miss_count": representation.mapping_miss_count,
        "unitarity_deviation": eigen.unitarity_deviation,
        "max_modulus_deviation": float(np.max(eigen.modulus_deviation)) if len(eigen.modulus_deviation) else 0.0,
        "max_root_deviation": float(np.max(root_deviations)) if len(root_deviations) else 0.0,
        "rotation_ready": rotation_ready,
        "spinor_rotation_applied": spinor_rotation_applied,
        "spinor_convention_verified": spinor_verified,
        "diagnostic_only": diagnostic_any,
        "D_valley_offdiag_norm": np.nan if d_valley_offdiag_norm is None else d_valley_offdiag_norm,
        "D_block_leakage_norm": np.nan if d_block_leakage_norm is None else d_block_leakage_norm,
        "basis": basis,
    }

    op_key = _representation_payload_key(operation["operation_id"], target_valley)
    op_payload: dict[str, object] = {
        "target_valley": target_valley,
        "source_operation_key": f"operation_{operation['operation_id']}",
        "D_raw": representation.matrix,
        "eigenvalues": eigen.eigenvalues,
        "root_deviation": root_deviations,
        "mapping_miss_count": representation.mapping_miss_count,
        "unitarity_deviation": eigen.unitarity_deviation,
        "operation_order": int(operation["order"]),
        "root_order": root_order,
        "rotation_frac": np.asarray(operation.get("rotation_frac", np.eye(3))),
        "translation_frac": np.asarray(operation.get("translation_frac", np.zeros(3))),
        "rotation_cart": np.asarray(operation["rotation_cart"]),
        "translation_cart": np.asarray(operation["translation_cart"]),
        "basis": basis,
        "rotation_ready": rotation_ready,
        "spinor_rotation_applied": spinor_rotation_applied,
        "spinor_convention_verified": spinor_verified,
        "spinor_convention": spinor_convention,
        "spinor_benchmark": "" if spinor_benchmark is None else spinor_benchmark,
        "topology_input_ready": topology_input_ready_by_state,
        "diagnostic_only": ~topology_input_ready_by_state,
        "D_valley_offdiag_norm": np.nan if d_valley_offdiag_norm is None else d_valley_offdiag_norm,
        "D_block_leakage_norm": np.nan if d_block_leakage_norm is None else d_block_leakage_norm,
    }
    if d_valley is not None:
        op_payload["D_valley"] = d_valley
    representation_payload.setdefault(kpoint_name, {})[op_key] = op_payload

    char_raw = complex(np.trace(representation.matrix))
    char_valley = complex(np.trace(d_valley)) if d_valley is not None else None

    for state_index, (value, phase, modulus_deviation, root, topology_input_ready) in enumerate(
        zip(
            eigen.eigenvalues,
            eigen.phases_2pi,
            eigen.modulus_deviation,
            root_info,
            topology_input_ready_by_state,
        )
    ):
        root_index, _root, root_deviation = root
        row_reason = _readiness_reason(
            base_reason=reason,
            rotation_ready=rotation_ready,
            spinor_rotation_applied=spinor_rotation_applied,
            spinor_convention_verified=spinor_verified,
            root_deviation=root_deviation,
            d_valley_offdiag_norm=d_valley_offdiag_norm,
            d_block_leakage_norm=d_block_leakage_norm,
            topology_input_ready=bool(topology_input_ready),
            root_deviation_tol=root_deviation_tol,
            d_valley_offdiag_tol=d_valley_offdiag_tol,
        )
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
                "modulus_deviation": float(modulus_deviation),
                "unitarity_deviation": float(eigen.unitarity_deviation),
                "character_raw": f"{char_raw.real:.6f}{char_raw.imag:+.6f}j",
                "character_valley": "" if char_valley is None else f"{char_valley.real:.6f}{char_valley.imag:+.6f}j",
                "little_group_passed": bool(little_group_passed),
                "valley_preserving": bool(valley_preserving),
                "rotation_ready": rotation_ready,
                "spinor_rotation_applied": spinor_rotation_applied,
                "spinor_convention_verified": spinor_verified,
                "spinor_convention": spinor_convention,
                "spinor_benchmark": "" if spinor_benchmark is None else spinor_benchmark,
                "nearest_root_of_unity": f"exp(2pii*{root_index}/{root_order})",
                "root_deviation": root_deviation,
                "topology_input_ready": bool(topology_input_ready),
                "topology_ready": bool(topology_input_ready),
                "diagnostic_only": bool(not topology_input_ready),
                "D_valley_offdiag_norm": "" if d_valley_offdiag_norm is None else d_valley_offdiag_norm,
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
    valley states.  This is the general multi-valley block-closure diagnostic,
    replacing the two-valley-only D_valley_offdiag_norm.
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


def _two_sector_offdiag_norm(matrix: np.ndarray | None) -> float | None:
    if matrix is None or matrix.shape != (2, 2):
        return None
    return float(np.linalg.norm(np.array([matrix[0, 1], matrix[1, 0]], dtype=np.complex128)))


def _topology_input_ready(
    *,
    rotation_ready: bool,
    basis: str,
    valid_valley_subspace: bool,
    spinor_convention_verified: bool,
    root_deviation: float,
    d_valley_offdiag_norm: float | None,
    d_block_leakage_norm: float | None = None,
    root_deviation_tol: float = ROOT_DEVIATION_TOL,
    d_valley_offdiag_tol: float = D_VALLEY_OFFDIAG_TOL,
) -> bool:
    ready = bool(
        rotation_ready
        and basis == "valley_adapted"
        and valid_valley_subspace
        and spinor_convention_verified
        and root_deviation <= root_deviation_tol
    )
    if not ready:
        return False
    # Use block-leakage for multi-valley, two-valley offdiag for legacy
    if d_block_leakage_norm is not None:
        return d_block_leakage_norm <= d_valley_offdiag_tol
    if d_valley_offdiag_norm is not None:
        return d_valley_offdiag_norm <= d_valley_offdiag_tol
    return False


def _readiness_reason(
    *,
    base_reason: str,
    rotation_ready: bool,
    spinor_rotation_applied: bool,
    spinor_convention_verified: bool,
    root_deviation: float,
    d_valley_offdiag_norm: float | None,
    d_block_leakage_norm: float | None = None,
    topology_input_ready: bool = False,
    root_deviation_tol: float = ROOT_DEVIATION_TOL,
    d_valley_offdiag_tol: float = D_VALLEY_OFFDIAG_TOL,
) -> str:
    if topology_input_ready:
        return ""
    if base_reason:
        return base_reason
    if not rotation_ready:
        return "rotation representation not ready"
    if spinor_rotation_applied and not spinor_convention_verified:
        return "spinor convention unverified"
    if root_deviation > root_deviation_tol:
        return "root deviation too large"
    if d_block_leakage_norm is not None:
        if d_block_leakage_norm > d_valley_offdiag_tol:
            return "valley block leakage too large"
        return ""
    if d_valley_offdiag_norm is None:
        return "two-valley D_valley offdiag diagnostic unavailable"
    if d_valley_offdiag_norm > d_valley_offdiag_tol:
        return "two-valley D_valley offdiag diagnostic too large"
    return "diagnostic-only"
