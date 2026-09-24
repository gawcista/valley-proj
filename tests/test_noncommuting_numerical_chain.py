"""Numerical acceptance for noncommuting spinful two-dimensional irreps."""

import h5py
import numpy as np
import pytest


@pytest.mark.parametrize("broken_coefficients", [False, True])
def test_noncommuting_fixture_is_orthonormal_and_time_reversal_closed(
    tmp_path, broken_coefficients,
):
    """Even the closure-negative input stays in the nonmagnetic SOC domain."""
    from tests.noncommuting_numerical_chain import write_noncommuting_fixture

    write_noncommuting_fixture(tmp_path, broken_coefficients=broken_coefficients)
    with h5py.File(tmp_path / "wavefunctions.h5") as h5:
        assert len(h5["kpoints"]) == 3
        for group in h5["kpoints"].values():
            grid = group["g_vectors_frac"][()]
            frac = group["frac"][()]
            lookup = {tuple(g): index for index, g in enumerate(grid)}
            # All three sampled points are TRIM; no production mapping helper.
            np.testing.assert_allclose(2 * frac, np.rint(2 * frac), atol=1e-12)
            permutation = [lookup[tuple(-g - np.rint(2 * frac).astype(int))] for g in grid]
            assert sorted(permutation) == list(range(4))
            coefficients = group["coefficients"][()]
            assert coefficients.shape == (2, 2, 4)
            frame = coefficients.reshape(2, -1)
            np.testing.assert_allclose(frame @ frame.conj().T, np.eye(2), atol=1e-12)
            transformed = np.zeros_like(coefficients)
            transformed[:, 0, permutation] = coefficients[:, 1, :].conj()
            transformed[:, 1, permutation] = -coefficients[:, 0, :].conj()
            transformed = transformed.reshape(2, -1)
            sewing = transformed @ frame.conj().T
            np.testing.assert_allclose(sewing @ frame, transformed, atol=1e-12)
            np.testing.assert_allclose(sewing @ sewing.conj(), -np.eye(2), atol=1e-12)


@pytest.mark.parametrize("profile", ["debug", "standard"])
def test_noncommuting_two_dimensional_irreps_reach_exact_reduced_ebr(tmp_path, profile):
    """Losing mirror mixing, full scope or reviewed spin labels must fail."""
    from tests.noncommuting_numerical_chain import (
        assert_noncommuting_positive,
        run_noncommuting_workflow,
    )

    assert_noncommuting_positive(run_noncommuting_workflow(tmp_path, profile=profile))


@pytest.mark.parametrize("profile", ["debug", "standard"])
def test_noncommuting_high_purity_without_closure_is_blocked(tmp_path, profile):
    """Purity/Gram/TR checks cannot replace actual target-space closure."""
    from tests.noncommuting_numerical_chain import (
        assert_noncommuting_negative,
        run_noncommuting_workflow,
    )

    assert_noncommuting_negative(
        run_noncommuting_workflow(tmp_path, profile=profile, broken_coefficients=True)
    )


@pytest.mark.parametrize("broken_coefficients", [False, True])
def test_debug_acceptance_rejects_missing_numerical_diagnostics(tmp_path, broken_coefficients):
    """The acceptance itself must fail closed if debug evidence disappears."""
    from tests.noncommuting_numerical_chain import (
        assert_noncommuting_negative,
        assert_noncommuting_positive,
        run_noncommuting_workflow,
    )

    result = run_noncommuting_workflow(tmp_path, broken_coefficients=broken_coefficients)
    if broken_coefficients:
        del result["reports"]["target_subspace_closure_json"]
        assertion = assert_noncommuting_negative
    else:
        del result["outputs"]["diagnostics_h5"]
        assertion = assert_noncommuting_positive
    with pytest.raises(AssertionError):
        assertion(result)
