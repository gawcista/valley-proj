"""Target-subspace symmetry-closure diagnostic.

Checks whether the raw six-band subspace D_raw is closed under each symmetry
operation.  This is independent of projector symmetry-consistency; a non-unitary
or non-closed D_raw here means the target subspace itself is not a faithful
representation of the symmetry group, and EBR mapping must be blocked.
"""

from __future__ import annotations

import numpy as np

DEFAULT_UNITARITY_TOL = 1e-3
DEFAULT_GROUP_RELATION_TOL = 1e-2


def build_target_subspace_closure_report(
    *,
    raw_representations_by_kpoint: dict[str, dict[object, dict[str, object]]],
    operation_orders: dict[object, int] | None = None,
    spinor_wavefunction: bool = False,
    unitarity_tol: float = DEFAULT_UNITARITY_TOL,
    group_relation_tol: float = DEFAULT_GROUP_RELATION_TOL,
) -> dict[str, object]:
    """Build per-(kpoint, operation) closure diagnostic for D_raw.

    Returns a report keyed by kpoint, each containing a list of per-operation
    diagnostics.
    """
    operation_orders = operation_orders or {}
    by_kpoint: dict[str, object] = {}
    overall_status = "ok"

    for kpoint_name, op_payloads in raw_representations_by_kpoint.items():
        rows: list[dict[str, object]] = []
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
                    status="not_evaluated",
                    reason="D_raw not available",
                ))
                continue

            d_raw = np.asarray(d_raw, dtype=np.complex128)
            n = d_raw.shape[0]
            raw_unitarity = float(
                np.linalg.norm(d_raw.conj().T @ d_raw - np.eye(n, dtype=np.complex128), ord="fro")
            )

            group_rel_err, group_rel_label = _check_group_relation(
                d_raw=d_raw,
                order=order,
                spinor_wavefunction=spinor_wavefunction,
            )

            # Determine status
            status, reason = _closure_status(
                little_group_passed=little_group_passed,
                mapping_miss_count=mapping_miss_count,
                raw_unitarity=raw_unitarity,
                group_rel_err=group_rel_err,
                group_rel_label=group_rel_label,
                unitarity_tol=unitarity_tol,
                group_relation_tol=group_relation_tol,
            )

            row = _closure_row(
                op_id=op_id, kpoint=kpoint_name,
                little_group_passed=little_group_passed,
                mapping_miss_count=mapping_miss_count,
                raw_unitarity_error=raw_unitarity,
                group_relation_error=group_rel_err,
                group_relation_label=group_rel_label,
                status=status,
                reason=reason,
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
            "Checks whether D_raw for each (kpoint, operation) is consistent "
            "with a unitary representation closed under the target subspace. "
            "raw_unitarity_error = ||D^dag D - I||_F. "
            "group_relation_error reports ||D^order - (+/-)I||_F for spinless "
            "or spinful convention as appropriate. "
            "This diagnostic is independent of projector symmetry-consistency."
        ),
        "by_kpoint": by_kpoint,
    }


def check_target_subspace_closure_blocked(
    report: dict[str, object],
) -> list[str]:
    """Return list of blocker keys from the closure report (global summary).

    Example: ["target_subspace_closure_failed"] if any operation has status=failed.
    """
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
    raw_unitarity_error: float | None = None,
    group_relation_error: float | None = None,
    group_relation_label: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "operation_id": op_id,
        "kpoint": kpoint,
        "little_group_passed": little_group_passed,
        "mapping_miss_count": mapping_miss_count,
        "status": status,
        "reason": reason,
    }
    if raw_unitarity_error is not None:
        row["raw_unitarity_error"] = raw_unitarity_error
    if group_relation_error is not None:
        row["group_relation_error"] = group_relation_error
        row["group_relation_label"] = group_relation_label
    return row


def _check_group_relation(
    *,
    d_raw: np.ndarray,
    order: int,
    spinor_wavefunction: bool,
) -> tuple[float | None, str]:
    """Check D^order relation.  For spinless: D^order ≈ I.  For spinful: D^order ≈ -I.

    Returns (error, label).  error is None if order < 2 (identity, not evaluated).
    """
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


def _closure_status(
    *,
    little_group_passed: bool,
    mapping_miss_count: int,
    raw_unitarity: float,
    group_rel_err: float | None,
    group_rel_label: str,
    unitarity_tol: float,
    group_relation_tol: float,
) -> tuple[str, str]:
    if not little_group_passed:
        return "not_evaluated", "not in little group"
    if mapping_miss_count > 0:
        return "failed", f"plane-wave mapping_miss_count={mapping_miss_count}"
    reasons: list[str] = []
    failed = False
    warned = False
    if raw_unitarity > unitarity_tol:
        failed = True
        reasons.append(f"raw_unitarity_error={raw_unitarity:.2e} > tol={unitarity_tol:.2e}")
    if group_rel_err is not None and group_rel_err > group_relation_tol:
        reasons.append(
            f"group_relation_error ({group_rel_label})={group_rel_err:.2e} > tol={group_relation_tol:.2e}"
        )
        if group_rel_err > 10.0 * group_relation_tol:
            failed = True
        else:
            warned = True
    if failed:
        return "failed", "; ".join(reasons)
    if warned:
        return "warn", "; ".join(reasons)
    return "ok", "target subspace closed under this operation"
