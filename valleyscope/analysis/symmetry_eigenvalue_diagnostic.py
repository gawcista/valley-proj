from __future__ import annotations

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
    generators_only: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for operation in symmetry_payload["detected_operations"]:
        if generators_only and not operation["candidate_rotation"]:
            continue
        if not generators_only:
            order = operation.get("order")
            if order not in (2, 3, 4, 6):
                continue
        little = is_little_group_operation(np.asarray(operation["rotation_frac"]), k_frac)
        preserves_all = all(bool(value) for value in operation["preserved"].values())
        operation.setdefault("little_group_by_kpoint", {})[kpoint_name] = little
        operation["allowed_for_single_valley_representation"] = bool(little and preserves_all)
        operation.setdefault("allowed_for_single_valley_representation_by_kpoint", {})[kpoint_name] = bool(little and preserves_all)
        rejection_reason = ""
        if not little:
            rejection_reason = "not in little group"
        elif not preserves_all:
            rejection_reason = "not valley preserving"
        operation.setdefault("rejection_reason_by_kpoint", {})[kpoint_name] = rejection_reason
        if rejection_reason:
            continue
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
                continue
        elif coefficients.shape[1] != 1:
            operation.setdefault("representation_quality", {})[kpoint_name] = {
                "skipped_reason": f"unsupported nspinor={coefficients.shape[1]}",
                "spinor_rotation_applied": False,
                "spinor_convention_verified": False,
                "diagnostic_only": True,
            }
            continue
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
        if basis_payload is not None and bool(np.asarray(basis_payload.get("valid_valley_subspace", False))):
            transform = np.asarray(basis_payload["transform"], dtype=np.complex128)
            d_valley = transform.conj().T @ representation.matrix @ transform
            matrix_for_eigen = d_valley
            basis = "valley_adapted"
            valley_eta = np.asarray(basis_payload.get("eta", []), dtype=float)
            reason = ""
            valid_valley_subspace = True
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
            "basis": basis,
        }
        op_key = f"operation_{operation['operation_id']}"
        op_payload: dict[str, object] = {
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
        }
        if d_valley is not None:
            op_payload["D_valley"] = d_valley
        representation_payload.setdefault(kpoint_name, {})[op_key] = op_payload
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
                topology_input_ready=bool(topology_input_ready),
                root_deviation_tol=root_deviation_tol,
                d_valley_offdiag_tol=d_valley_offdiag_tol,
            )
            char_raw = complex(np.trace(representation.matrix)) if state_index == 0 else None
            char_valley = complex(np.trace(d_valley)) if d_valley is not None and state_index == 0 else None
            rows.append(
                {
                    "kpoint": kpoint_name,
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
                    "character_raw": "" if char_raw is None else f"{char_raw.real:.6f}{char_raw.imag:+.6f}j",
                    "character_valley": "" if char_valley is None else f"{char_valley.real:.6f}{char_valley.imag:+.6f}j",
                    "little_group_passed": bool(little),
                    "valley_preserving": bool(preserves_all),
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
                    "reason": row_reason,
                    "valley_eta": "" if valley_eta is None or state_index >= len(valley_eta) else float(valley_eta[state_index]),
                }
            )
    return rows


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
    root_deviation_tol: float = ROOT_DEVIATION_TOL,
    d_valley_offdiag_tol: float = D_VALLEY_OFFDIAG_TOL,
) -> bool:
    return bool(
        rotation_ready
        and basis == "valley_adapted"
        and valid_valley_subspace
        and spinor_convention_verified
        and root_deviation <= root_deviation_tol
        and d_valley_offdiag_norm is not None
        and d_valley_offdiag_norm <= d_valley_offdiag_tol
    )


def _readiness_reason(
    *,
    base_reason: str,
    rotation_ready: bool,
    spinor_rotation_applied: bool,
    spinor_convention_verified: bool,
    root_deviation: float,
    d_valley_offdiag_norm: float | None,
    topology_input_ready: bool,
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
    if d_valley_offdiag_norm is None:
        return "two-sector D_valley offdiag diagnostic unavailable"
    if d_valley_offdiag_norm > d_valley_offdiag_tol:
        return "two-sector D_valley offdiag diagnostic too large"
    return "diagnostic-only"
