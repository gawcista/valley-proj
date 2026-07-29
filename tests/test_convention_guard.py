"""Guard: verify projector matrix convention and transformation.

_projector_matrix stores P[i,j] = <psi_i|P|psi_j>.
Correct symmetry action: D @ P @ D.conj().T = D P D^dag.
D.T @ P @ D.conj() is incorrect for non-symmetric D.
"""

import numpy as np

from valleyscope.subspace.valley_basis import _projector_matrix
from valleyscope.symmetry.plane_wave_action import build_plane_wave_representation


def test_stored_projector_is_standard_convention():
    """P[i,j] = <psi_i|P|psi_j> = sum_G psi_i(G)* psi_j(G) in mask."""
    coeffs = np.zeros((2, 1, 4), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 2] = 1.0 + 1.0j
    mask = np.array([True, False, True, False])
    P = _projector_matrix(coeffs, mask)

    # Manual: <psi_i|P|psi_j>
    for i in range(2):
        for j in range(2):
            expected = sum(
                coeffs[i, 0, g].conj() * coeffs[j, 0, g]
                for g in range(4) if mask[g]
            )
            assert abs(P[i, j] - expected) < 1e-14, (
                f"P[{i},{j}]={P[i,j]:.6f} != expected={expected:.6f}"
            )


def test_stored_projector_uses_bra_conjugation_for_complex_overlap():
    """Shared complex support fixes the off-diagonal bra/ket convention."""
    coeffs = np.array(
        [
            [[1.0 + 1.0j, 0.0]],
            [[2.0 - 1.0j, 0.0]],
        ],
        dtype=np.complex128,
    )

    projector = _projector_matrix(
        coeffs,
        np.array([True, False]),
    )
    expected = coeffs[:, 0, :1].conj() @ coeffs[:, 0, :1].T

    assert np.allclose(projector, expected)


def test_correct_transformation_is_d_p_ddag():
    """D @ P_M1 @ Ddag = P_M2 under C3 cyclic mapping."""
    angle = 2 * np.pi / 3
    c3 = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    q_cart = np.array([
        [1., 0., 0.],
        [-0.5, np.sqrt(3) / 2, 0.],
        [-0.5, -np.sqrt(3) / 2, 0.],
    ])
    coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0
    coeffs[2, 0, 2] = 1.0

    mask_m1 = np.array([True, False, False])
    mask_m2 = np.array([False, True, False])
    P_m1 = _projector_matrix(coeffs, mask_m1)
    P_m2 = _projector_matrix(coeffs, mask_m2)

    result = build_plane_wave_representation(coeffs, q_cart, c3, np.zeros(3))
    D = result.matrix

    # Correct: D @ P @ Ddag
    transformed = D @ P_m1 @ D.conj().T
    err_correct = float(np.linalg.norm(transformed - P_m2))
    assert err_correct < 1e-12, f"D@P@Ddag should map M1→M2, err={err_correct:.2e}"

    # Bonus: idempotency preserved
    assert float(np.linalg.norm(transformed @ transformed - transformed)) < 1e-12


def test_wrong_transformation_dt_p_dconj_fails():
    """D.T @ P @ D.conj() gives O(1) error for C3 cyclic mapping."""
    angle = 2 * np.pi / 3
    c3 = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    q_cart = np.array([
        [1., 0., 0.],
        [-0.5, np.sqrt(3) / 2, 0.],
        [-0.5, -np.sqrt(3) / 2, 0.],
    ])
    coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 1] = 1.0
    coeffs[2, 0, 2] = 1.0

    mask_m1 = np.array([True, False, False])
    mask_m2 = np.array([False, True, False])
    P_m1 = _projector_matrix(coeffs, mask_m1)
    P_m2 = _projector_matrix(coeffs, mask_m2)

    result = build_plane_wave_representation(coeffs, q_cart, c3, np.zeros(3))
    D = result.matrix

    wrong = D.T @ P_m1 @ D.conj()
    err_wrong = float(np.linalg.norm(wrong - P_m2))
    assert err_wrong > 0.1, (
        f"D.T@P@D.conj() should NOT map M1→M2, but err={err_wrong:.2e} (O(1) expected)"
    )
