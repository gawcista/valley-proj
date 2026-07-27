from __future__ import annotations

import numpy as np

SEED_PROJECTOR_SYMMETRY_WARN_TOL = 1.0e-2
SEED_PROJECTOR_SYMMETRY_FAIL_TOL = 1.0e-1
SMALL_NUMBER = 1e-14


def build_projector_symmetry_report(
    *,
    valley_matrices_by_kpoint: dict[str, dict[str, np.ndarray]],
    raw_representations_by_kpoint: dict[str, dict[object, dict[str, object]]],
    valley_names: list[str],
    warn_tol: float = SEED_PROJECTOR_SYMMETRY_WARN_TOL,
    fail_tol: float = SEED_PROJECTOR_SYMMETRY_FAIL_TOL,
    small_number: float = SMALL_NUMBER,
) -> dict[str, object]:
    """Build the q-cut seed projector symmetry-consistency report.

    For each (kpoint, operation_id, source_valley) with available D_raw and
    projected q-cut valley seed matrix P_a^0:

        epsilon_seed(g,a) =
            || D_g P_a^0 D_g^dag - P_{pi_g(a)}^0 ||_F
            / max(||P_a^0||_F, small_number)

    epsilon_seed is the seed projector symmetry error.  The check follows the
    valley mapping pi_g(a); it is not an invariance check for valley-changing
    operations.
    """
    by_kpoint: dict[str, object] = {}
    overall_status = "ok"
    any_not_evaluated = False
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
            valley_mapping = op_data.get("sector_mapping", {})
            little_group_passed = bool(op_data.get("little_group_passed", True))
            if d_raw is None:
                reason = str(op_data.get("skipped_reason", "D_raw not available"))
                for source_valley in valley_matrices:
                    mapped_valley = valley_mapping.get(source_valley)
                    kp_rows.append({
                        "operation_id": operation_id,
                        "source_valley": source_valley,
                        "mapped_valley": None if mapped_valley is None else str(mapped_valley),
                        "epsilon_seed": None,
                        "seed_projector_symmetry_error": None,
                        "little_group_passed": little_group_passed,
                        "seed_projector_symmetry_status": "not_evaluated",
                        "status": "not_evaluated",
                        "reason": reason,
                    })
                    any_not_evaluated = True
                continue
            d_raw = np.asarray(d_raw, dtype=np.complex128)

            for source_valley in valley_matrices:
                p_a = np.asarray(valley_matrices[source_valley], dtype=np.complex128)
                mapped_valley = valley_mapping.get(source_valley)
                if mapped_valley is None:
                    kp_rows.append({
                        "operation_id": operation_id,
                        "source_valley": source_valley,
                        "mapped_valley": None,
                        "epsilon_seed": None,
                        "seed_projector_symmetry_error": None,
                        "little_group_passed": little_group_passed,
                        "seed_projector_symmetry_status": "not_evaluated",
                        "status": "not_evaluated",
                        "reason": f"pi_g({source_valley}) not in valley_mapping",
                    })
                    any_not_evaluated = True
                    continue
                mapped_valley = str(mapped_valley)

                p_mapped = valley_matrices.get(mapped_valley)
                if p_mapped is None:
                    kp_rows.append({
                        "operation_id": operation_id,
                        "source_valley": source_valley,
                        "mapped_valley": mapped_valley,
                        "epsilon_seed": None,
                        "seed_projector_symmetry_error": None,
                        "little_group_passed": little_group_passed,
                        "seed_projector_symmetry_status": "not_evaluated",
                        "status": "not_evaluated",
                        "reason": f"P_{mapped_valley}^0 not found in seed matrices",
                    })
                    any_not_evaluated = True
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
                    "seed_projector_symmetry_error": epsilon,
                    "little_group_passed": little_group_passed,
                    "seed_projector_symmetry_status": status,
                    "status": status,
                    "reason": "",
                })

        if kp_rows:
            by_kpoint[kpoint_name] = {"seed_projector_symmetry": kp_rows}

    if any_failed:
        overall_status = "symmetry_consistency_failures_detected"
    elif any_warn:
        overall_status = "symmetry_consistency_warnings_detected"
    elif any_not_evaluated:
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
            "/ max(||P_a^0||_F, small_number). epsilon_seed is the seed "
            "projector symmetry error. The condition follows valley_mapping "
            "pi_g(a), so valley-changing operations are not tested as "
            "invariance conditions."
        ),
        "by_kpoint": by_kpoint,
    }


def apply_projector_symmetry_gate(
    *,
    symmetry_rows: list[dict[str, object]],
    projector_symmetry_report: dict[str, object] | None,
) -> None:
    """Attach seed projector symmetry status and demote failed ready rows."""
    if not projector_symmetry_report:
        return

    symmetry_by_row: dict[tuple[str, str, str], dict[str, object]] = {}
    by_kpoint = projector_symmetry_report.get("by_kpoint", {})
    if not isinstance(by_kpoint, dict):
        return
    for kpoint, kp_data in by_kpoint.items():
        if not isinstance(kp_data, dict):
            continue
        rows = kp_data.get("seed_projector_symmetry", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            operation_id = row.get("operation_id")
            source_valley = row.get("source_valley")
            if operation_id is None or source_valley is None:
                continue
            symmetry_by_row[(str(kpoint), str(operation_id), str(source_valley))] = row

    for row in symmetry_rows:
        kpoint = str(row.get("kpoint", ""))
        operation_id = row.get("operation_id")
        target_valley = row.get("target_valley")
        if operation_id is None or target_valley is None:
            continue
        symmetry = symmetry_by_row.get((kpoint, str(operation_id), str(target_valley)))
        if symmetry is None:
            continue

        status = str(symmetry.get("seed_projector_symmetry_status", symmetry.get("status", "")))
        row["projector_symmetry_status"] = status
        row["seed_projector_symmetry_status"] = status
        row["projector_symmetry_mapped_valley"] = symmetry.get("mapped_valley")
        row["epsilon_seed"] = symmetry.get("epsilon_seed")
        row["seed_projector_symmetry_error"] = symmetry.get("seed_projector_symmetry_error")
        row["projector_symmetry_reason"] = symmetry.get("reason", "")

        if not bool(row.get("valley_preserving", False)):
            continue
        if status != "failed":
            continue

        row["local_irrep_ready"] = False
        row["diagnostic_only"] = True
        row["reason"] = _append_reason(
            row.get("reason", ""),
            "seed projector symmetry-consistency failed",
        )


def _append_reason(existing: object, new_reason: str) -> str:
    existing_text = str(existing) if existing not in (None, "") else ""
    if not existing_text:
        return new_reason
    if new_reason in existing_text:
        return existing_text
    return f"{existing_text}; {new_reason}"
