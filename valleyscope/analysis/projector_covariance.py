from __future__ import annotations

import numpy as np

SEED_COVARIANCE_WARN_TOL = 1.0e-2
SEED_COVARIANCE_FAIL_TOL = 1.0e-1
SMALL_NUMBER = 1e-14


def compute_projector_covariance(
    *,
    valley_matrices_by_kpoint: dict[str, dict[str, np.ndarray]],
    raw_representations_by_kpoint: dict[str, dict[object, dict[str, object]]],
    valley_names: list[str],
    warn_tol: float = SEED_COVARIANCE_WARN_TOL,
    fail_tol: float = SEED_COVARIANCE_FAIL_TOL,
    small_number: float = SMALL_NUMBER,
) -> dict[str, object]:
    """Compute seed projector covariance under little-group representations.

    For each (kpoint, operation_id, source_valley) with available D_raw and
    projected valley matrix P_a^0:

        epsilon_seed(g,a) = || D_g P_a^0 D_g^† - P_{pi_g(a)}^0 ||_F
                            / max(||P_a^0||_F, small_number)

    Uses D_raw matrices indexed by (kpoint, operation_id), not by
    target-valley payload keys.  This deduplicates rows and naturally
    covers valley-permuting operations such as C3 cycling M1/M2/M3.

    Parameters
    ----------
    valley_matrices_by_kpoint : {kpoint: {valley_name: P_a^0 matrix}}
        Seed projected valley matrices in the raw DFT subspace.
    raw_representations_by_kpoint : {kpoint: {operation_id: payload}}
        Per-operation payloads from ``build_raw_representations_for_kpoint``.
        Each payload contains ``D_raw``, ``sector_mapping``, etc.
    valley_names : list[str]
        Ordered list of valley names.
    warn_tol : float
        Threshold above which covariance is flagged as ``warn``.
    fail_tol : float
        Threshold above which covariance is flagged as ``failed``.

    Returns
    -------
    report : dict
        ``projector_covariance_report`` structure.
    """
    by_kpoint: dict[str, object] = {}
    overall_status = "ok"
    any_missing_mapping = False
    any_failed = False
    any_warn = False

    for kpoint_name, valley_matrices in valley_matrices_by_kpoint.items():
        op_payloads = raw_representations_by_kpoint.get(kpoint_name, {})
        if not isinstance(op_payloads, dict) or not op_payloads:
            continue

        kp_rows: list[dict[str, object]] = []

        for operation_id, op_data in op_payloads.items():
            if not isinstance(op_data, dict):
                continue
            d_raw = op_data.get("D_raw")
            sector_mapping = op_data.get("sector_mapping", {})
            little_group_passed = bool(op_data.get("little_group_passed", True))
            if d_raw is None:
                reason = str(op_data.get("skipped_reason", "D_raw not available"))
                for source_valley in valley_matrices:
                    mapped_valley = sector_mapping.get(source_valley)
                    kp_rows.append({
                        "operation_id": operation_id,
                        "source_valley": source_valley,
                        "mapped_valley": None if mapped_valley is None else str(mapped_valley),
                        "epsilon_seed": None,
                        "little_group_passed": little_group_passed,
                        "status": "not_evaluated",
                        "reason": reason,
                    })
                    any_missing_mapping = True
                continue
            d_raw = np.asarray(d_raw, dtype=np.complex128)

            for source_valley in valley_matrices:
                p_a = np.asarray(valley_matrices[source_valley], dtype=np.complex128)
                mapped_valley = sector_mapping.get(source_valley)
                if mapped_valley is None:
                    kp_rows.append({
                        "operation_id": operation_id,
                        "source_valley": source_valley,
                        "mapped_valley": None,
                        "epsilon_seed": None,
                        "little_group_passed": little_group_passed,
                        "status": "not_evaluated",
                        "reason": f"pi_g({source_valley}) not in sector_mapping",
                    })
                    any_missing_mapping = True
                    continue
                mapped_valley = str(mapped_valley)

                p_mapped = valley_matrices.get(mapped_valley)
                if p_mapped is None:
                    kp_rows.append({
                        "operation_id": operation_id,
                        "source_valley": source_valley,
                        "mapped_valley": mapped_valley,
                        "epsilon_seed": None,
                        "little_group_passed": little_group_passed,
                        "status": "not_evaluated",
                        "reason": f"P_{mapped_valley}^0 not found in seed matrices",
                    })
                    any_missing_mapping = True
                    continue
                p_mapped = np.asarray(p_mapped, dtype=np.complex128)

                transformed = d_raw @ p_a @ d_raw.conj().T
                epsilon = float(
                    np.linalg.norm(transformed - p_mapped, ord="fro")
                    / max(np.linalg.norm(p_a, ord="fro"), small_number)
                )

                if epsilon <= warn_tol:
                    status = "passed"
                elif epsilon <= fail_tol:
                    status = "warn"
                    any_warn = True
                else:
                    status = "failed"
                    any_failed = True

                kp_rows.append({
                    "operation_id": operation_id,
                    "source_valley": source_valley,
                    "mapped_valley": mapped_valley,
                    "epsilon_seed": epsilon,
                    "little_group_passed": little_group_passed,
                    "status": status,
                    "reason": "",
                })

        if kp_rows:
            by_kpoint[kpoint_name] = {"seed_projector_covariance": kp_rows}

    if any_failed:
        overall_status = "covariance_failures_detected"
    elif any_warn:
        overall_status = "covariance_warnings_detected"
    elif any_missing_mapping:
        overall_status = "partial"
    elif not by_kpoint:
        overall_status = "no_data"

    return {
        "status": overall_status,
        "normalization": "frobenius_source_projector",
        "small_number": small_number,
        "warn_tol": warn_tol,
        "fail_tol": fail_tol,
        "interpretation": (
            "epsilon_seed(g,a) = ||D_g P_a^0 D_g^dag - P_{pi_g(a)}^0||_F "
            "/ max(||P_a^0||_F, small_number). "
            "epsilon <= warn_tol: passed. "
            "warn_tol < epsilon <= fail_tol: warn / diagnostic caution. "
            "epsilon > fail_tol: failed_covariance, local valley irrep "
            "interpretation is diagnostic_only."
        ),
        "by_kpoint": by_kpoint,
    }


def apply_projector_covariance_gate(
    *,
    symmetry_rows: list[dict[str, object]],
    covariance_report: dict[str, object] | None,
) -> None:
    """Attach seed-covariance diagnostics to symmetry rows and demote failures."""
    if not covariance_report:
        return

    covariance_by_row: dict[tuple[str, str, str], dict[str, object]] = {}
    by_kpoint = covariance_report.get("by_kpoint", {})
    if not isinstance(by_kpoint, dict):
        return
    for kpoint, kp_data in by_kpoint.items():
        if not isinstance(kp_data, dict):
            continue
        rows = kp_data.get("seed_projector_covariance", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            operation_id = row.get("operation_id")
            source_valley = row.get("source_valley")
            if operation_id is None or source_valley is None:
                continue
            covariance_by_row[(str(kpoint), str(operation_id), str(source_valley))] = row

    for row in symmetry_rows:
        kpoint = str(row.get("kpoint", ""))
        operation_id = row.get("operation_id")
        target_valley = row.get("target_valley")
        if operation_id is None or target_valley is None:
            continue
        covariance = covariance_by_row.get((kpoint, str(operation_id), str(target_valley)))
        if covariance is None:
            continue

        status = str(covariance.get("status", ""))
        row["projector_covariance_status"] = status
        row["projector_covariance_mapped_valley"] = covariance.get("mapped_valley")
        row["epsilon_seed"] = covariance.get("epsilon_seed")
        row["projector_covariance_reason"] = covariance.get("reason", "")

        if not bool(row.get("valley_preserving", False)):
            continue
        if status != "failed":
            continue

        row["topology_input_ready"] = False
        row["topology_ready"] = False
        row["diagnostic_only"] = True
        row["reason"] = _append_reason(
            row.get("reason", ""),
            "failed seed projector covariance",
        )


def _append_reason(existing: object, new_reason: str) -> str:
    existing_text = str(existing) if existing not in (None, "") else ""
    if not existing_text:
        return new_reason
    if new_reason in existing_text:
        return existing_text
    return f"{existing_text}; {new_reason}"
