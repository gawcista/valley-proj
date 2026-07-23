"""Target-subspace symmetry-closure diagnostic with provenance classification.

Checks whether the raw subspace D_raw is closed under each symmetry operation
and classifies the root cause of any failure: target-subspace leakage,
plane-wave mapping loss, non-orthonormal input wavefunctions, or
insufficient provenance data.

This is independent of projector symmetry-consistency; a non-unitary
or non-closed D_raw means the target subspace is not a faithful
representation of the symmetry group, and EBR mapping must be blocked.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CLOSURE_UNITARITY_TOL = 1e-3
DEFAULT_CLOSURE_GROUP_RELATION_TOL = 1e-2
DEFAULT_GRAM_ORTHONORMALITY_TOL = 1e-6
DEFAULT_USABLE_WITH_CAUTION_UNITARITY_TOL = 3e-2
DEFAULT_USABLE_WITH_CAUTION_RESIDUAL_TOL = 3e-2
DEFAULT_USABLE_WITH_CAUTION_SV_DEVIATION_TOL = 3e-2
SMALL_NUMBER = 1e-14


def build_target_subspace_closure_report(
    *,
    raw_representations_by_kpoint: dict[str, dict[object, dict[str, object]]],
    operation_orders: dict[object, int] | None = None,
    spinor_wavefunction: bool = False,
    unitarity_tol: float = DEFAULT_CLOSURE_UNITARITY_TOL,
    group_relation_tol: float = DEFAULT_CLOSURE_GROUP_RELATION_TOL,
    coefficients_by_kpoint: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Build per-(kpoint, operation) closure diagnostic with provenance.

    Returns a report keyed by kpoint, each containing a list of per-operation
    diagnostics with provenance classification.
    """
    operation_orders = operation_orders or {}
    coeffs_by_kp = dict(coefficients_by_kpoint or {})
    by_kpoint: dict[str, object] = {}
    overall_status = "ok"

    for kpoint_name, op_payloads in raw_representations_by_kpoint.items():
        rows: list[dict[str, object]] = []

        # Per-kpoint: target wavefunction Gram matrix diagnostic
        coeffs = coeffs_by_kp.get(kpoint_name)
        gram_err: float | None = None
        gram_diag: str = "not_available"
        if coeffs is not None:
            c_arr = np.asarray(coeffs, dtype=np.complex128)
            if c_arr.ndim == 3:
                n_bands = c_arr.shape[0]
                c_flat = c_arr.reshape(n_bands, -1)
                gram = c_flat @ c_flat.conj().T
                gram_err = float(
                    np.linalg.norm(gram - np.eye(n_bands, dtype=np.complex128), ord="fro")
                )
                gram_diag = "available"

        for op_id, op_data in op_payloads.items():
            if not isinstance(op_data, dict):
                continue
            d_raw = op_data.get("D_raw")
            little_group_passed = bool(op_data.get("little_group_passed", True))
            mapping_miss_count = int(op_data.get("mapping_miss_count", 0))
            order = int(operation_orders.get(op_id, op_data.get("order", 0)))

            if d_raw is None:
                rows.append(_closure_row(
                    op_id=op_id, kpoint=kpoint_name,
                    little_group_passed=little_group_passed,
                    mapping_miss_count=mapping_miss_count,
                    classification="insufficient_provenance",
                    closure_quality="blocked",
                    status="not_evaluated",
                    reason="D_raw not available",
                    provenance_notes="D_raw was not built (skipped by upstream diagnostic)",
                ))
                continue

            d_raw = np.asarray(d_raw, dtype=np.complex128)
            n = d_raw.shape[0]

            # --- Singular values of D_raw ---
            sv = np.linalg.svd(d_raw, compute_uv=False)
            sv_list = [float(v) for v in sv.tolist()]

            # --- Per-source-state projected norm and closure residual ---
            d_dag_d = d_raw.conj().T @ d_raw
            diag_norms = np.real(np.diag(d_dag_d))
            projected_norms = [float(v) for v in diag_norms.tolist()]
            residuals = [float(max(0.0, 1.0 - v)) for v in diag_norms]
            max_residual = float(np.max(residuals)) if residuals else 0.0
            worst_idx = int(np.argmax(residuals)) if residuals else -1

            raw_unitarity = float(
                np.linalg.norm(d_dag_d - np.eye(n, dtype=np.complex128), ord="fro")
            )

            group_rel_err, group_rel_label = _check_group_relation(
                d_raw=d_raw,
                order=order,
                spinor_wavefunction=spinor_wavefunction,
            )

            # --- Provenance classification ---
            classification, class_reason = _classify_provenance(
                little_group_passed=little_group_passed,
                mapping_miss_count=mapping_miss_count,
                gram_err=gram_err,
                raw_unitarity=raw_unitarity,
                singular_values=sv_list,
                unitarity_tol=unitarity_tol,
                gram_tol=DEFAULT_GRAM_ORTHONORMALITY_TOL,
            )
            closure_quality = _closure_quality(
                classification=classification,
                raw_unitarity=raw_unitarity,
                singular_values=sv_list,
                max_residual=max_residual,
                unitarity_tol=unitarity_tol,
            )

            status = "ok"
            status_reason = ""
            if classification == "raw_representation_ok":
                if group_rel_err is not None and group_rel_err > group_relation_tol:
                    status = "warn"
                    if group_rel_err > 10.0 * group_relation_tol:
                        status = "failed"
                    status_reason = (
                        f"group_relation_error ({group_rel_label})"
                        f"={group_rel_err:.2e} > tol={group_relation_tol:.2e}"
                    )
            elif classification == "insufficient_provenance":
                status = "not_evaluated"
                status_reason = class_reason
            elif closure_quality == "usable_with_caution":
                status = "warn"
                status_reason = class_reason
            else:
                status = "failed"
                status_reason = class_reason

            row = _closure_row(
                op_id=op_id, kpoint=kpoint_name,
                little_group_passed=little_group_passed,
                mapping_miss_count=mapping_miss_count,
                raw_unitarity_error=raw_unitarity,
                D_raw_singular_values=sv_list,
                group_relation_error=group_rel_err,
                group_relation_label=group_rel_label,
                projected_norm_by_source_state=projected_norms,
                closure_residual_by_source_state=residuals,
                max_closure_residual=max_residual,
                worst_source_state=worst_idx if worst_idx >= 0 else None,
                target_wavefunction_gram_error=gram_err,
                target_wavefunction_gram_status=gram_diag,
                classification=classification,
                closure_quality=closure_quality,
                provenance_notes=class_reason,
                status=status,
                reason=status_reason if status_reason else (
                    "target subspace closed under this operation"
                ),
            )
            rows.append(row)
            if status == "failed":
                overall_status = "closure_failures_detected"
            elif status == "warn" and overall_status == "ok":
                overall_status = "closure_warnings_detected"

        if rows:
            by_kpoint[kpoint_name] = rows

    if not by_kpoint:
        overall_status = "no_data"

    return {
        "status": overall_status,
        "unitarity_tol": unitarity_tol,
        "group_relation_tol": group_relation_tol,
        "interpretation": (
            "Checks whether D_raw for each (kpoint, operation) is a closed "
            "representation in the target subspace. "
            "raw_unitarity_error = ||D^dag D - I||_F. "
            "projected_norm_by_source_state = diag(D^dag D) measures per-state "
            "fidelity under the operation. "
            "closure_residual_by_source_state = max(0, 1 - projected_norm) "
            "quantifies target-subspace leakage. "
            "D_raw_singular_values reports the SVD spectrum. "
            "target_wavefunction_gram_error = ||C^dag C - I||_F checks input "
            "wavefunction orthonormality. "
            "plane_wave_norm_preservation: not_available from current "
            "build_plane_wave_representation (no pre-projection norm tracked). "
            "expanded_band_sensitivity: not_available unless HDF5 contains "
            "bands outside the target window. "
            "classification distinguishes root cause: "
            "raw_representation_ok, target_subspace_not_closed, "
            "plane_wave_mapping_loss, input_wavefunctions_nonorthonormal, "
            "insufficient_provenance. "
            "closure_quality is the user-facing three-level quality: clean, "
            "usable_with_caution, blocked."
        ),
        "by_kpoint": by_kpoint,
    }


def check_target_subspace_closure_blocked(
    report: dict[str, object],
) -> list[str]:
    """Return list of blocker keys from the closure report (global summary)."""
    blockers: list[str] = []
    for rows in report.get("by_kpoint", {}).values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("status") == "failed":
                if "target_subspace_closure_failed" not in blockers:
                    blockers.append("target_subspace_closure_failed")
            if row.get("status") == "not_evaluated":
                key = "target_subspace_closure_not_evaluated"
                if key not in blockers:
                    blockers.append(key)
    return blockers


def check_target_subspace_closure_blocked_for_operation(
    report: dict[str, object],
    kpoint: str,
    operation_id: object,
) -> bool:
    """Check whether a specific (kpoint, operation) has a closure failure."""
    rows = report.get("by_kpoint", {}).get(kpoint, [])
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("operation_id") == operation_id and row.get("status") == "failed":
            return True
    return False


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _closure_row(
    *,
    op_id: object,
    kpoint: str,
    little_group_passed: bool,
    mapping_miss_count: int,
    status: str,
    reason: str,
    classification: str,
    closure_quality: str,
    provenance_notes: str = "",
    raw_unitarity_error: float | None = None,
    D_raw_singular_values: list[float] | None = None,
    group_relation_error: float | None = None,
    group_relation_label: str = "",
    projected_norm_by_source_state: list[float] | None = None,
    closure_residual_by_source_state: list[float] | None = None,
    max_closure_residual: float | None = None,
    worst_source_state: int | None = None,
    target_wavefunction_gram_error: float | None = None,
    target_wavefunction_gram_status: str = "not_available",
) -> dict[str, object]:
    row: dict[str, object] = {
        "operation_id": op_id,
        "kpoint": kpoint,
        "little_group_passed": little_group_passed,
        "mapping_miss_count": mapping_miss_count,
        "classification": classification,
        "closure_quality": closure_quality,
        "status": status,
        "reason": reason,
    }
    if provenance_notes:
        row["provenance_notes"] = provenance_notes
    if raw_unitarity_error is not None:
        row["raw_unitarity_error"] = raw_unitarity_error
    if D_raw_singular_values is not None:
        row["D_raw_singular_values"] = D_raw_singular_values
    if group_relation_error is not None:
        row["group_relation_error"] = group_relation_error
        row["group_relation_label"] = group_relation_label
    if projected_norm_by_source_state is not None:
        row["projected_norm_by_source_state"] = projected_norm_by_source_state
    if closure_residual_by_source_state is not None:
        row["closure_residual_by_source_state"] = closure_residual_by_source_state
    if max_closure_residual is not None:
        row["max_closure_residual"] = max_closure_residual
    if worst_source_state is not None and worst_source_state >= 0:
        row["worst_source_state"] = worst_source_state
    if target_wavefunction_gram_error is not None:
        row["target_wavefunction_gram_error"] = target_wavefunction_gram_error
    row["target_wavefunction_gram_status"] = target_wavefunction_gram_status
    # Provenance fields that are not available from current implementation
    row["plane_wave_norm_preservation"] = (
        "not_available: build_plane_wave_representation does not track "
        "pre-projection norm"
    )
    row["expanded_band_sensitivity"] = (
        "not_available: HDF5 contains only target window bands; "
        "expanded-band analysis requires bands outside iband"
    )
    return row


def _check_group_relation(
    *,
    d_raw: np.ndarray,
    order: int,
    spinor_wavefunction: bool,
) -> tuple[float | None, str]:
    if order < 2:
        return None, ""
    d_pow = np.linalg.matrix_power(d_raw, order)
    n = d_raw.shape[0]
    if spinor_wavefunction:
        expected = -np.eye(n, dtype=np.complex128)
        label = f"D^{order} + I (spinful)"
    else:
        expected = np.eye(n, dtype=np.complex128)
        label = f"D^{order} - I (spinless)"
    err = float(np.linalg.norm(d_pow - expected, ord="fro"))
    return err, label


def _classify_provenance(
    *,
    little_group_passed: bool,
    mapping_miss_count: int,
    gram_err: float | None,
    raw_unitarity: float,
    singular_values: list[float],
    unitarity_tol: float,
    gram_tol: float,
) -> tuple[str, str]:
    """Classify the D_raw non-unitarity into a root-cause category.

    Priority order:
    1. insufficient_provenance (not in little group)
    2. plane_wave_mapping_loss
    3. input_wavefunctions_nonorthonormal (gram_err > 1e-3, physical)
    4. target_subspace_not_closed (dominant: D_raw non-unitary)

    Mild gram errors (< 1e-3) are noted but do not override the
    target-subspace closure classification.
    """
    if not little_group_passed:
        return "insufficient_provenance", "not in little group"
    if mapping_miss_count > 0:
        return (
            "plane_wave_mapping_loss",
            f"plane-wave mapping_miss_count={mapping_miss_count}",
        )
    # Only classify as input non-orthonormal if physically significant
    gram_fail_tol = max(gram_tol, 1e-3)
    if gram_err is not None and gram_err > gram_fail_tol:
        return (
            "input_wavefunctions_nonorthonormal",
            f"target wavefunction Gram error={gram_err:.2e} > tol={gram_fail_tol:.2e}",
        )
    if raw_unitarity <= unitarity_tol:
        msg = "D_raw is unitary in target subspace"
        if gram_err is not None and gram_err > gram_tol:
            msg += f" (mild gram_error={gram_err:.2e} noted)"
        return "raw_representation_ok", msg
    # Subspace not closed
    min_sv = float(np.min(singular_values)) if singular_values else 0.0
    max_sv = float(np.max(singular_values)) if singular_values else 0.0
    msg_parts = [
        f"D_raw unitarity error={raw_unitarity:.2e} > tol={unitarity_tol:.2e}"
    ]
    if gram_err is not None and gram_err > gram_tol:
        msg_parts.append(
            f"mild gram_error={gram_err:.2e} noted but not primary cause"
        )
    if min_sv < 0.9 or max_sv > 1.1:
        msg_parts.append(
            f"singular values [{min_sv:.4f}, {max_sv:.4f}] "
            f"indicate significant subspace leakage"
        )
    else:
        msg_parts.append(
            f"singular values within [0.9, 1.1] suggest mild non-closure "
            f"(possible expanded-band sensitivity)"
        )
    return "target_subspace_not_closed", "; ".join(msg_parts)


def _closure_quality(
    *,
    classification: str,
    raw_unitarity: float,
    singular_values: list[float],
    max_residual: float,
    unitarity_tol: float,
) -> str:
    """Map detailed provenance to a three-level user-facing quality label."""
    if classification == "raw_representation_ok":
        return "clean"
    if classification != "target_subspace_not_closed":
        return "blocked"
    min_sv = float(np.min(singular_values)) if singular_values else 0.0
    max_sv = float(np.max(singular_values)) if singular_values else 0.0
    usable_unitarity_tol = max(
        3.0 * unitarity_tol,
        DEFAULT_USABLE_WITH_CAUTION_UNITARITY_TOL,
    )
    if (
        raw_unitarity <= usable_unitarity_tol
        and max_residual <= DEFAULT_USABLE_WITH_CAUTION_RESIDUAL_TOL
        and abs(min_sv - 1.0) <= DEFAULT_USABLE_WITH_CAUTION_SV_DEVIATION_TOL
        and abs(max_sv - 1.0) <= DEFAULT_USABLE_WITH_CAUTION_SV_DEVIATION_TOL
    ):
        return "usable_with_caution"
    return "blocked"
