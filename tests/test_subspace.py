import numpy as np
import pytest

from valley_proj.subspace.valley_basis import (
    build_two_valley_adapted_basis,
    diagnose_multivalley_subspace,
)


def test_degenerate_two_valley_subspace_recovers_pure_valley_basis():
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    coefficients = np.array(
        [
            [[inv_sqrt2, inv_sqrt2]],
            [[inv_sqrt2, -inv_sqrt2]],
        ],
        dtype=np.complex128,
    )
    masks = {
        "K_sector": np.array([True, False]),
        "Kp_sector": np.array([False, True]),
    }

    result = build_two_valley_adapted_basis(coefficients, masks, "K_sector", "Kp_sector")

    assert sorted(np.round(result.eta.real, 8).tolist()) == [-1.0, 1.0]
    assert result.transform.shape == (2, 2)
    assert np.allclose(result.transform.conj().T @ result.transform, np.eye(2))


def test_multivalley_diagnostic_rejects_non_commuting_sector_projectors():
    m1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    m2 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    diagnostic = diagnose_multivalley_subspace({"A": m1, "B": m2}, eig_tol=1e-6, commutator_tol=1e-6)

    assert diagnostic.stably_separable is False
    assert diagnostic.reason == "non_commuting_sector_projectors"
    assert diagnostic.max_commutator_norm > 0.0
