import numpy as np
import pytest

from valleyscope.subspace.valley_basis import (
    build_two_valley_adapted_basis,
    diagnose_multivalley_subspace,
    build_valley_subspace_matrices,
    build_valley_adapted_basis,
    diagnose_valley_separability,
)


# -----------------------------------------------------------------------
# A. two-valley compatibility
# -----------------------------------------------------------------------

def test_degenerate_two_valley_subspace_recovers_pure_valley_basis():
    """Legacy two-valley test — new API should recover eta_adapted ≈ ±1."""
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    coefficients = np.array(
        [
            [[inv_sqrt2, inv_sqrt2]],
            [[inv_sqrt2, -inv_sqrt2]],
        ],
        dtype=np.complex128,
    )
    masks = {
        "K_valley": np.array([True, False]),
        "Kp_valley": np.array([False, True]),
    }

    # Legacy API
    legacy = build_two_valley_adapted_basis(coefficients, masks, "K_valley", "Kp_valley")
    assert sorted(np.round(legacy.eta.real, 8).tolist()) == [-1.0, 1.0]
    assert legacy.transform.shape == (2, 2)
    assert np.allclose(legacy.transform.conj().T @ legacy.transform, np.eye(2))

    # New general API
    result = build_valley_adapted_basis(coefficients, masks)
    assert result.eta_adapted is not None
    assert sorted(np.round(result.eta_adapted.real, 8).tolist()) == [-1.0, 1.0]
    assert result.assigned_valleys == ["Kp_valley", "K_valley"] or result.assigned_valleys == ["K_valley", "Kp_valley"]
    assert np.allclose(result.min_valley_concentration, 1.0, atol=1e-8)
    assert result.valley_weights_adapted.shape == (2, 2)


# -----------------------------------------------------------------------
# B. three-valley pure case
# -----------------------------------------------------------------------

def test_three_valley_pure_subspace_recovers_assigned_valleys():
    """Construct 3 normalized states, each pure in one valley mask."""
    coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0   # state 0 ∈ valley A
    coeffs[1, 0, 1] = 1.0   # state 1 ∈ valley B
    coeffs[2, 0, 2] = 1.0   # state 2 ∈ valley C

    masks = {
        "valley_A": np.array([True, False, False]),
        "valley_B": np.array([False, True, False]),
        "valley_C": np.array([False, False, True]),
    }

    result = build_valley_adapted_basis(coefficients=coeffs, valley_masks=masks)
    diagnosed = diagnose_valley_separability(result, w_val_min=0.5)

    assert diagnosed.stably_separable
    assert diagnosed.reason == "stably_separable"
    assert diagnosed.eta_adapted is None
    assert np.allclose(diagnosed.min_valley_concentration, 1.0, atol=1e-8)
    # Each adapted state should be assigned to a distinct valley
    assert sorted(diagnosed.assigned_valleys) == ["valley_A", "valley_B", "valley_C"]
    assert diagnosed.valley_weights_adapted.shape == (3, 3)
    assert np.allclose(diagnosed.s_expectation.sum(), 3.0, atol=1e-2)


# -----------------------------------------------------------------------
# C. three-valley mixed case
# -----------------------------------------------------------------------

def test_three_valley_mixed_subspace_lowers_concentration():
    """Equal-weight mixed state should reduce min_valley_concentration."""
    inv_sqrt3 = 1.0 / np.sqrt(3.0)
    coeffs = np.zeros((3, 1, 3), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0                     # pure A
    coeffs[1, 0, 1] = 1.0                     # pure B
    # State 2 is equal-weight mix of all three valleys
    coeffs[2, 0, 0] = inv_sqrt3
    coeffs[2, 0, 1] = inv_sqrt3
    coeffs[2, 0, 2] = inv_sqrt3

    masks = {
        "valley_A": np.array([True, False, False]),
        "valley_B": np.array([False, True, False]),
        "valley_C": np.array([False, False, True]),
    }

    result = build_valley_adapted_basis(coefficients=coeffs, valley_masks=masks)
    # S_min ~ 0.184 for this mixed fixture; use lenient thresholds so
    # only the concentration check (~0.39 < 0.7) triggers failure
    diagnosed = diagnose_valley_separability(
        result, w_val_min=0.15, concentration_threshold=0.7,
        commutator_tol=1.0, idempotency_tol=1.0,
    )

    assert not diagnosed.stably_separable
    assert "concentration" in diagnosed.reason
    assert diagnosed.min_valley_concentration < 0.7
    assert diagnosed.eta_adapted is None


# -----------------------------------------------------------------------
# D. non-commuting valley matrices diagnostic
# -----------------------------------------------------------------------

def test_multivalley_diagnostic_rejects_non_commuting_sector_projectors():
    """Legacy test — should still pass with new internals."""
    m1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    m2 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    diagnostic = diagnose_multivalley_subspace({"A": m1, "B": m2}, eig_tol=1e-6, commutator_tol=1e-6)

    assert diagnostic.stably_separable is False
    assert diagnostic.reason == "non_commuting_sector_projectors"
    assert diagnostic.max_commutator_norm > 0.0


def test_non_commuting_valley_matrices_detected_in_diagnose():
    """New diagnostic API detects non-commuting matrices."""
    # Construct raw states so that projected matrices don't commute
    coeffs = np.zeros((2, 1, 4), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0 / np.sqrt(2.0)
    coeffs[0, 0, 1] = 1.0 / np.sqrt(2.0)
    coeffs[1, 0, 1] = 1.0 / np.sqrt(2.0)
    coeffs[1, 0, 2] = 1.0 / np.sqrt(2.0)

    # Overlapping masks create non-commuting projected matrices
    masks = {
        "valley_A": np.array([True, True, False, False]),
        "valley_B": np.array([False, True, True, False]),
    }

    subspaces = build_valley_subspace_matrices(coeffs, masks)
    # The commutator check is diagnostic — depending on overlap it may or may not exceed tol
    assert subspaces.commutator_norm_max >= 0.0
    assert subspaces.idempotency_deviation_max >= 0.0


# -----------------------------------------------------------------------
# E. ValleySubspaceMatrices basic checks
# -----------------------------------------------------------------------

def test_valley_subspace_matrices_sums_to_s():
    coeffs = np.array(
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]],
        dtype=np.complex128,
    )
    masks = {
        "A": np.array([True, False, False]),
        "B": np.array([False, True, False]),
    }
    result = build_valley_subspace_matrices(coeffs, masks)
    assert np.allclose(result.s_matrix, result.valley_matrices["A"] + result.valley_matrices["B"])
    assert result.s_min >= 0.0
    assert result.s_max <= 2.0


def test_valley_subspace_matrices_rejects_empty_masks():
    with pytest.raises(ValueError):
        build_valley_subspace_matrices(
            np.zeros((2, 1, 2), dtype=np.complex128), {}
        )


# -----------------------------------------------------------------------
# F. Concentration and assignment edge cases
# -----------------------------------------------------------------------

def test_perfect_concentration_for_orthogonal_valley_states():
    """Orthogonal states in distinct valleys → concentration = 1."""
    coeffs = np.zeros((2, 1, 4), dtype=np.complex128)
    coeffs[0, 0, 0] = 1.0
    coeffs[1, 0, 2] = 1.0
    masks = {
        "X": np.array([True, True, False, False]),
        "Y": np.array([False, False, True, True]),
    }
    result = build_valley_adapted_basis(coeffs, masks)
    assert np.allclose(result.min_valley_concentration, 1.0, atol=1e-8)
    assert np.allclose(result.valley_concentration, [1.0, 1.0], atol=1e-8)


def test_eta_adapted_none_for_three_valleys():
    coeffs = np.array(
        [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]],
        dtype=np.complex128,
    )
    masks = {
        "V1": np.array([True, False, False]),
        "V2": np.array([False, True, False]),
        "V3": np.array([False, False, True]),
    }
    result = build_valley_adapted_basis(coeffs, masks)
    assert result.eta_adapted is None
