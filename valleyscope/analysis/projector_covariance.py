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
            if d_raw is None:
                continue
            d_raw = np.asarray(d_raw, dtype=np.complex128)
            sector_mapping = op_data.get("sector_mapping", {})
            little_group_passed = bool(op_data.get("little_group_passed", True))

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
