"""Subspace representation quality diagnostics.

Decomposes the unitarity error of valley-preserving local representations
D_a(g) = U_a^dag D_g U_a into contributions from:

  - basis orthonormality (U_a)
  - raw representation unitarity (D_g)
  - projector invariance (P_a^sym)
  - local representation unitarity
  - group relation consistency

This is a diagnostic-only layer.  It does not modify readiness, ebr_mapping,
or character outputs.
"""

from __future__ import annotations

import numpy as np

SMALL_NUMBER = 1e-14


def build_subspace_representation_quality_report(
    *,
    valley_bases: dict[str, np.ndarray],
    projectors: dict[str, np.ndarray],
    representations: dict[object, np.ndarray],
    valley_mappings: dict[object, dict[str, str]],
    operation_orders: dict[object, int] | None = None,
    spinor_wavefunction: bool = False,
    target_valleys: list[str] | None = None,
) -> dict[str, object]:
    """Build per-(valley, operation) subspace representation quality diagnostics.

    Only valley-preserving operations are analysed in depth.
    Valley-changing operations get a ``not_valley_preserving`` status.
    """
    operation_orders = operation_orders or {}
    valleys = list(target_valleys) if target_valleys else sorted(valley_bases.keys())
    rows: list[dict[str, object]] = []

    for valley in valleys:
        u_a = _validated_basis(valley_bases, valley)
        p_a = _validated_square_matrix(projectors.get(valley))
        for op_id, d_g_raw in representations.items():
            mapping = valley_mappings.get(op_id, {})
            mapped = mapping.get(valley)
            is_vp = mapped is not None and str(mapped) == str(valley)
            order = int(operation_orders.get(op_id, 0))

            if not is_vp:
                rows.append(_quality_row(
                    valley=valley, operation_id=op_id,
                    operation_order=order,
                    is_valley_preserving=False,
                    diagnosis="not_valley_preserving",
                ))
                continue

            d_g = _validated_square_matrix(d_g_raw)
            if d_g is None or u_a is None or p_a is None:
                rows.append(_quality_row(
                    valley=valley, operation_id=op_id,
                    operation_order=order,
                    is_valley_preserving=True,
                    diagnosis="missing_inputs",
                ))
                continue

            n = d_g.shape[0]
            basis_shape = list(u_a.shape)

            # 1. Basis orthonormality
            basis_ortho_err = float(
                np.linalg.norm(
                    u_a.conj().T @ u_a - np.eye(u_a.shape[1], dtype=np.complex128),
                    ord="fro",
                )
            )

            # 2. Raw representation unitarity
            d_raw_unitarity = float(
                np.linalg.norm(
                    d_g.conj().T @ d_g - np.eye(n, dtype=np.complex128),
                    ord="fro",
                )
            )

            # 3. Projector invariance: ||D P_a D^dag - P_a||_F / max(||P_a||_F, small)
            transformed_p = d_g @ p_a @ d_g.conj().T
            p_norm = max(float(np.linalg.norm(p_a, ord="fro")), SMALL_NUMBER)
            proj_inv_err = float(
                np.linalg.norm(transformed_p - p_a, ord="fro") / p_norm
            )

            # 4. Local representation D_a = U^dag D U
            d_a = u_a.conj().T @ d_g @ u_a
            local_unitarity = float(
                np.linalg.norm(
                    d_a.conj().T @ d_a - np.eye(d_a.shape[0], dtype=np.complex128),
                    ord="fro",
                )
            )

            # 5. Singular values of U^dag D U
            sv = np.linalg.svd(d_a, compute_uv=False)
            sv_list = [float(v) for v in sv.tolist()]

            # 6. Eigenvalue modulus deviation
            eigvals = np.linalg.eigvals(d_a)
            modulus_dev = float(max(abs(abs(lam) - 1.0) for lam in eigvals))

            # 7. Group relation
            group_rel_err, group_rel_label = _check_group_relation(
                matrix=d_a, order=order, spinor=spinor_wavefunction,
            )

            # 8. Closest unitary eigenphases (diagnostic-only)
            closest_phases = _closest_unitary_eigenphases(d_a)

            # 9. Polar unitarity distance (diagnostic-only)
            polar_dist = _polar_unitarity_distance(d_a)

            # Diagnosis
            diagnosis = _classify_diagnosis(
                basis_ortho_err=basis_ortho_err,
                d_raw_unitarity=d_raw_unitarity,
                proj_inv_err=proj_inv_err,
                local_unitarity=local_unitarity,
                group_rel_err=group_rel_err,
            )

            rows.append(_quality_row(
                valley=valley,
                operation_id=op_id,
                operation_order=order,
                is_valley_preserving=True,
                basis_shape=basis_shape,
                basis_orthonormality_error=basis_ortho_err,
                D_raw_unitarity_error=d_raw_unitarity,
                projector_invariance_error=proj_inv_err,
                local_representation_unitarity_error=local_unitarity,
                local_group_relation_error=group_rel_err,
                local_group_relation_label=group_rel_label,
                singular_values_of_UdagDU=sv_list,
                eigenvalue_modulus_deviation=modulus_dev,
                closest_unitary_eigenphases_diagnostic_only=closest_phases,
                polar_unitarity_distance_diagnostic_only=polar_dist,
                diagnosis=diagnosis,
            ))

    status = "ok"
    for row in rows:
        if row.get("diagnosis") not in ("ok", "not_valley_preserving", "missing_inputs"):
            status = "quality_issues_detected"
            break

    return {
        "status": status,
        "interpretation": (
            "Per-(valley, operation) subspace representation quality. "
            "Decomposes local representation unitarity error into basis "
            "orthonormality, D_raw unitarity, projector invariance, and "
            "group relation contributions.  This is diagnostic-only and "
            "does not modify readiness or ebr_mapping decisions."
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validated_basis(
    bases: dict[str, np.ndarray],
    valley: str,
) -> np.ndarray | None:
    u = bases.get(valley)
    if u is None:
        return None
    u = np.asarray(u, dtype=np.complex128)
    if u.ndim != 2 or u.shape[0] < 1 or u.shape[1] < 1:
        return None
    return u


def _validated_square_matrix(mat: object) -> np.ndarray | None:
    if mat is None:
        return None
    arr = np.asarray(mat, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    return arr


def _check_group_relation(
    *,
    matrix: np.ndarray,
    order: int,
    spinor: bool,
) -> tuple[float | None, str]:
    if order < 2:
        return None, ""
    d_pow = np.linalg.matrix_power(matrix, order)
    n = matrix.shape[0]
    if spinor:
        expected = -np.eye(n, dtype=np.complex128)
        label = f"D_a^{order} + I (spinful)"
    else:
        expected = np.eye(n, dtype=np.complex128)
        label = f"D_a^{order} - I (spinless)"
    err = float(np.linalg.norm(d_pow - expected, ord="fro"))
    return err, label


def _closest_unitary_eigenphases(
    d_a: np.ndarray,
) -> list[float]:
    """Extract eigenphases from the closest unitary to D_a via eigendecomposition.

    Diagnostic-only: the actual D_a eigenvalues are used for character
    computation; this is only an auxiliary diagnostic.
    """
    try:
        eigvals = np.linalg.eigvals(d_a)
    except np.linalg.LinAlgError:
        return []
    phases: list[float] = []
    for lam in eigvals:
        norm = abs(lam)
        if norm < SMALL_NUMBER:
            phases.append(0.0)
        else:
            phase = float(np.angle(lam / norm) / (2.0 * np.pi))
            if phase <= -0.5:
                phase += 1.0
            phases.append(phase)
    phases.sort()
    return phases


def _polar_unitarity_distance(d_a: np.ndarray) -> float:
    """Frobenius distance between D_a and its closest unitary via polar decomposition.

    Diagnostic-only.
    """
    try:
        u_polar, s, vh = np.linalg.svd(d_a)
        closest_unitary = u_polar @ vh
        return float(np.linalg.norm(d_a - closest_unitary, ord="fro"))
    except np.linalg.LinAlgError:
        return -1.0


def _classify_diagnosis(
    *,
    basis_ortho_err: float,
    d_raw_unitarity: float,
    proj_inv_err: float,
    local_unitarity: float,
    group_rel_err: float | None,
    ortho_tol: float = 1e-6,
    unitarity_tol: float = 1e-3,
    proj_inv_tol: float = 1e-1,
    group_rel_tol: float = 1e-2,
) -> str:
    if basis_ortho_err > ortho_tol:
        return "non_orthonormal_basis"
    if d_raw_unitarity > unitarity_tol:
        return "raw_representation_nonunitary"
    if proj_inv_err > proj_inv_tol:
        return "projector_not_invariant_under_valley_preserving_operation"
    if local_unitarity > unitarity_tol:
        return "local_representation_nonunitary"
    if group_rel_err is not None and group_rel_err > group_rel_tol:
        return "local_group_relation_failed"
    return "ok"


def _quality_row(
    *,
    valley: str,
    operation_id: object,
    operation_order: int,
    is_valley_preserving: bool,
    basis_shape: list[int] | None = None,
    basis_orthonormality_error: float | None = None,
    D_raw_unitarity_error: float | None = None,
    projector_invariance_error: float | None = None,
    local_representation_unitarity_error: float | None = None,
    local_group_relation_error: float | None = None,
    local_group_relation_label: str = "",
    singular_values_of_UdagDU: list[float] | None = None,
    eigenvalue_modulus_deviation: float | None = None,
    closest_unitary_eigenphases_diagnostic_only: list[float] | None = None,
    polar_unitarity_distance_diagnostic_only: float | None = None,
    diagnosis: str = "not_evaluated",
) -> dict[str, object]:
    row: dict[str, object] = {
        "valley": valley,
        "operation_id": operation_id,
        "operation_order": operation_order,
        "is_valley_preserving": is_valley_preserving,
        "diagnosis": diagnosis,
    }
    if basis_shape is not None:
        row["basis_shape"] = basis_shape
    if basis_orthonormality_error is not None:
        row["basis_orthonormality_error"] = basis_orthonormality_error
    if D_raw_unitarity_error is not None:
        row["D_raw_unitarity_error"] = D_raw_unitarity_error
    if projector_invariance_error is not None:
        row["projector_invariance_error"] = projector_invariance_error
    if local_representation_unitarity_error is not None:
        row["local_representation_unitarity_error"] = local_representation_unitarity_error
    if local_group_relation_error is not None:
        row["local_group_relation_error"] = local_group_relation_error
        row["local_group_relation_label"] = local_group_relation_label
    if singular_values_of_UdagDU is not None:
        row["singular_values_of_UdagDU"] = singular_values_of_UdagDU
    if eigenvalue_modulus_deviation is not None:
        row["eigenvalue_modulus_deviation"] = eigenvalue_modulus_deviation
    if closest_unitary_eigenphases_diagnostic_only is not None:
        row["closest_unitary_eigenphases_diagnostic_only"] = \
            closest_unitary_eigenphases_diagnostic_only
    if polar_unitarity_distance_diagnostic_only is not None:
        row["polar_unitarity_distance_diagnostic_only"] = \
            polar_unitarity_distance_diagnostic_only
    return row
