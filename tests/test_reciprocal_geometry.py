"""Nearest-image reciprocal geometry: independent oracles and invariants.

The expected values in this file come from brute-force integer enumeration
and from exact lattice rebasing, never from the production search itself.
"""

from itertools import product

import numpy as np
import pytest

from valleyscope.geometry.reciprocal import (
    RunLocalPeriodicDistanceCache,
    equivalent_mod_reciprocal,
    minimum_periodic_distance,
)
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.projection.folded_center import build_folded_center_report
from valleyscope.projection.sector_projectors import build_sector_projectors
from valleyscope.projection.weights import compute_valley_weights
from valleyscope.subspace.valley_basis import build_valley_subspace_matrices


HEX_BASIS = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]])
HEX_Q = np.array([0.51, -0.25, 0.0]) @ HEX_BASIS
C3_CART = np.array(
    [
        [-0.5, -np.sqrt(3) / 2, 0.0],
        [np.sqrt(3) / 2, -0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def brute_force_minimum(deltas: np.ndarray, lattice: np.ndarray, shell: int = 6) -> np.ndarray:
    """Independent oracle: exhaustive enumeration over a generous integer box."""
    dim = lattice.shape[0]
    coeffs = np.array(list(product(range(-shell, shell + 1), repeat=dim)), dtype=float)
    shifts = coeffs @ lattice
    return np.array(
        [float(np.linalg.norm(delta - shifts, axis=1).min()) for delta in deltas]
    )


def embed_2d(matrix_2d: np.ndarray) -> np.ndarray:
    out = np.eye(3)
    out[:2, :2] = matrix_2d
    return out


def coeffs_for(amplitudes) -> np.ndarray:
    arr = np.asarray(amplitudes, dtype=np.complex128).reshape(1, 1, -1)
    norm = np.linalg.norm(arr)
    return arr / norm if norm else arr


# --- D1: hexagonal counterexample through distance, mask, weight and seed ---

def test_hexagonal_minimum_distance():
    basis = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]])
    q = (np.array([0.51, -0.25, 0.0]) @ basis)[None, :]
    actual = minimum_periodic_distance(q, np.zeros(3), basis)
    np.testing.assert_allclose(actual, [np.sqrt(0.1951)], rtol=0.0, atol=1e-12)


def test_hexagonal_counterexample_reaches_mask_weight_and_seed():
    q = HEX_Q[None, :]
    oracle = brute_force_minimum(HEX_Q[None, :2], HEX_BASIS[:2, :2])
    assert oracle[0] == pytest.approx(np.sqrt(0.1951), abs=1e-15)

    projectors = build_sector_projectors(
        q,
        [ValleyCenter("Q", np.zeros(3))],
        [ValleySector("a", ["Q"])],
        HEX_BASIS,
        0.5,
    )
    weights = compute_valley_weights(coeffs_for([1.0]), projectors)[0]
    seeds = build_valley_subspace_matrices(coeffs_for([1.0]), projectors.sector_masks)

    assert projectors.center_masks["Q"].tolist() == [True]
    assert projectors.sector_masks["a"].tolist() == [True]
    assert weights.w_val == pytest.approx(1.0)
    assert weights.center_weights["Q"] == pytest.approx(1.0)
    np.testing.assert_allclose(seeds.valley_matrices["a"], [[1.0 + 0.0j]], atol=1e-12)


def test_hexagonal_counterexample_is_not_the_component_rounded_answer():
    """The component-wise rounded image is a worse candidate and must not win."""
    fractional = HEX_Q[:2] @ np.linalg.inv(HEX_BASIS[:2, :2])
    rounded = fractional - np.rint(fractional)
    rounded_distance = float(np.linalg.norm(rounded @ HEX_BASIS[:2, :2]))
    assert rounded_distance == pytest.approx(0.6519969325081215, abs=1e-12)
    actual = float(minimum_periodic_distance(HEX_Q[None, :], np.zeros(3), HEX_BASIS)[0])
    assert actual < rounded_distance


# --- D2: C3-related momenta carry equal distances and equal masks ---

def test_c3_related_momenta_have_equal_minimum_distance_and_mask():
    momenta = np.array([HEX_Q @ np.linalg.matrix_power(C3_CART, j).T for j in range(3)])
    distances = minimum_periodic_distance(momenta, np.zeros(3), HEX_BASIS)
    np.testing.assert_allclose(distances, [np.sqrt(0.1951)] * 3, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        distances, brute_force_minimum(momenta[:, :2], HEX_BASIS[:2, :2]), rtol=0.0, atol=1e-12
    )

    masks = []
    for momentum in momenta:
        projectors = build_sector_projectors(
            momentum[None, :],
            [ValleyCenter("Q", np.zeros(3))],
            [ValleySector("a", ["Q"])],
            HEX_BASIS,
            0.5,
        )
        masks.append(projectors.sector_masks["a"][0])
    assert masks == [True, True, True]


def test_rotated_center_covariance_matches_rotated_momentum():
    """The mask is covariant: rotating every q and the center together is a no-op."""
    center_cart = np.array([0.2, 0.05, 0.0])
    offsets = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.75, 0.0]])
    qs = center_cart + offsets
    plain = build_sector_projectors(
        qs,
        [ValleyCenter("Q", center_cart)],
        [ValleySector("a", ["Q"])],
        HEX_BASIS,
        0.5,
    )
    rotated_center = C3_CART @ center_cart
    rotated_qs = np.array([C3_CART @ q for q in qs])
    rotated = build_sector_projectors(
        rotated_qs,
        [ValleyCenter("Q", rotated_center)],
        [ValleySector("a", ["Q"])],
        HEX_BASIS,
        0.5,
    )
    assert plain.sector_masks["a"].tolist() == rotated.sector_masks["a"].tolist()
    assert plain.sector_masks["a"].tolist() == [True, True, False]


def test_equivalent_mod_reciprocal_uses_true_periodic_equivalence():
    assert equivalent_mod_reciprocal(HEX_Q, np.zeros(3), HEX_BASIS, tolerance=0.4418) is True
    assert equivalent_mod_reciprocal(HEX_Q, np.zeros(3), HEX_BASIS, tolerance=0.4416) is False


# --- D3: integer rebase invariance in two dimensions ---

def test_unimodular_rebase_preserves_minimum_distance_2d():
    rebase = embed_2d(np.array([[1.0, 37.0], [0.0, 1.0]]))
    rebased_basis = rebase @ HEX_BASIS
    deltas = np.array([HEX_Q, HEX_Q + np.array([1.0, 0.0, 0.0])])
    actual = minimum_periodic_distance(deltas, np.zeros(3), rebased_basis)
    np.testing.assert_allclose(
        actual, brute_force_minimum(deltas[:, :2], rebased_basis[:2, :2], shell=60), rtol=0.0, atol=1e-12
    )
    reference = minimum_periodic_distance(deltas, np.zeros(3), HEX_BASIS)
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-12)


def test_unimodular_rebase_preserves_mask_2d():
    rebase = embed_2d(np.array([[1.0, 37.0], [0.0, 1.0]]))
    rebased_basis = rebase @ HEX_BASIS
    args = (
        HEX_Q[None, :],
        [ValleyCenter("Q", np.zeros(3))],
        [ValleySector("a", ["Q"])],
    )
    plain = build_sector_projectors(*args, HEX_BASIS, 0.5)
    rebased = build_sector_projectors(*args, rebased_basis, 0.5)
    assert plain.sector_masks["a"].tolist() == rebased.sector_masks["a"].tolist() == [True]


# --- D4: three-dimensional skew cell and GL(3,Z) rebase ---

SKEW_BASIS_3D = np.array(
    [
        [0.10, 0.00, 0.00],
        [0.03, 0.09, 0.00],
        [0.01, 0.02, 0.40],
    ]
)


def test_three_dimensional_skew_minimum_distance():
    rng = np.random.default_rng(20260915)
    deltas = rng.normal(size=(8, 3)) * 0.1
    actual = minimum_periodic_distance(deltas, np.zeros(3), SKEW_BASIS_3D, use_2d=False)
    np.testing.assert_allclose(
        actual, brute_force_minimum(deltas, SKEW_BASIS_3D, shell=5), rtol=0.0, atol=1e-12
    )


def test_gl3z_rebase_preserves_minimum_distance():
    rebase = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    rebased_basis = rebase @ SKEW_BASIS_3D
    rng = np.random.default_rng(7)
    deltas = rng.normal(size=(6, 3)) * 0.1
    actual = minimum_periodic_distance(deltas, np.zeros(3), rebased_basis, use_2d=False)
    np.testing.assert_allclose(
        actual,
        brute_force_minimum(deltas, rebased_basis, shell=40),
        rtol=0.0,
        atol=1e-12,
    )
    reference = minimum_periodic_distance(deltas, np.zeros(3), SKEW_BASIS_3D, use_2d=False)
    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=1e-12)


# --- D5: large integer reciprocal shifts ---

def test_large_integer_reciprocal_shift_keeps_distance():
    shift = np.array([12000.0, -9000.0, 0.0])
    shifted = (HEX_Q + shift @ HEX_BASIS)[None, :]
    reference = float(minimum_periodic_distance(HEX_Q[None, :], np.zeros(3), HEX_BASIS)[0])
    actual = float(minimum_periodic_distance(shifted, np.zeros(3), HEX_BASIS)[0])
    # Finite-precision-aware margin: the shift is applied in floating point.
    scale = float(np.linalg.norm(shift @ HEX_BASIS))
    assert abs(actual - reference) <= 1e-12 * max(1.0, scale)


def test_large_center_shift_keeps_distance():
    shift = np.array([5000.0, 5000.0, 0.0]) @ HEX_BASIS
    reference = float(minimum_periodic_distance(HEX_Q[None, :], np.zeros(3), HEX_BASIS)[0])
    actual = float(minimum_periodic_distance(HEX_Q[None, :], shift, HEX_BASIS)[0])
    scale = float(np.linalg.norm(shift))
    assert abs(actual - reference) <= 1e-12 * max(1.0, scale)


# --- D6: semantics, edge cases and rejected input ---

def test_use_2d_ignores_z_components():
    basis = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]])
    plain = float(minimum_periodic_distance(HEX_Q[None, :], np.zeros(3), basis)[0])
    lifted = float(
        minimum_periodic_distance(
            (HEX_Q + np.array([0.0, 0.0, 3.5]))[None, :],
            np.array([0.0, 0.0, -2.0]),
            basis,
        )[0]
    )
    assert plain == pytest.approx(lifted, abs=1e-15)
    three_dimensional = float(
        minimum_periodic_distance(
            (HEX_Q + np.array([0.0, 0.0, 1.5]))[None, :],
            np.zeros(3),
            basis,
            use_2d=False,
        )[0]
    )
    assert three_dimensional == pytest.approx(np.sqrt(0.1951 + 0.25), abs=1e-12)


def test_orthogonal_basis_matches_component_answer():
    basis = np.diag([2.0, 0.5, 1.25])
    deltas = np.array([[0.7, -1.3, 0.2], [3.1, 0.9, -0.6]])
    actual = minimum_periodic_distance(deltas, np.zeros(3), basis)
    np.testing.assert_allclose(
        actual, brute_force_minimum(deltas[:, :2], basis[:2, :2]), rtol=0.0, atol=1e-14
    )
    # Diagonal metric: the component-rounded image is already the minimum.
    # wrapped xy are (0.7, 0.2) and (-0.9, -0.1).
    np.testing.assert_allclose(
        actual, [np.sqrt(0.53), np.sqrt(0.82)], rtol=0.0, atol=1e-14
    )


def test_tie_between_two_images_is_exact_half_separation():
    basis = np.diag([1.0, 1.0, 1.0])
    delta = np.array([[0.5, 0.0, 0.0]])
    actual = minimum_periodic_distance(delta, np.zeros(3), basis)
    np.testing.assert_allclose(actual, [0.5], rtol=0.0, atol=1e-15)


def test_empty_batch_returns_empty_result():
    basis = np.diag([1.0, 1.0, 1.0])
    actual = minimum_periodic_distance(np.zeros((0, 3)), np.zeros(3), basis)
    assert actual.shape == (0,)


def test_uniform_scaling_scales_distance():
    basis = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.0]])
    reference = float(minimum_periodic_distance(HEX_Q[None, :], np.zeros(3), basis)[0])
    scaled_basis = 3.5 * basis
    actual = float(
        minimum_periodic_distance((3.5 * HEX_Q)[None, :], np.zeros(3), scaled_basis)[0]
    )
    assert actual == pytest.approx(3.5 * reference, rel=1e-14)


def test_shell_argument_does_not_change_the_result():
    values = [
        float(minimum_periodic_distance(HEX_Q[None, :], np.zeros(3), HEX_BASIS, shell=shell)[0])
        for shell in (0, 1, 2, 5)
    ]
    assert len(set(values)) == 1


def test_skew_but_valid_cell_is_accepted():
    basis = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    delta = np.array([[0.31, 0.22, 0.0]])
    actual = minimum_periodic_distance(delta, np.zeros(3), basis)
    np.testing.assert_allclose(
        actual, brute_force_minimum(delta[:, :2], basis[:2, :2]), rtol=0.0, atol=1e-13
    )


@pytest.mark.parametrize(
    "q,center,basis",
    [
        (np.zeros((2, 2)), np.zeros(3), np.eye(3)),
        (np.zeros((2, 3)), np.zeros(2), np.eye(3)),
        (np.zeros((2, 3)), np.zeros(3), np.eye(2)),
        (np.array([[np.nan, 0.0, 0.0]]), np.zeros(3), np.eye(3)),
        (np.zeros((1, 3)), np.array([np.inf, 0.0, 0.0]), np.eye(3)),
        (np.zeros((1, 3)), np.zeros(3), np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
        (np.zeros((1, 3)), np.zeros(3), np.zeros((3, 3))),
    ],
)
def test_malformed_or_degenerate_input_is_rejected(q, center, basis):
    with pytest.raises(ValueError):
        minimum_periodic_distance(q, center, basis)


def test_exhausted_search_raises_instead_of_returning_partial_minimum(monkeypatch):
    import valleyscope.geometry.reciprocal as reciprocal_module

    monkeypatch.setattr(reciprocal_module, "_MAX_SEARCH_NODES", 1)
    with pytest.raises(RuntimeError):
        minimum_periodic_distance(HEX_Q[None, :], np.zeros(3), HEX_BASIS)


# --- folded-center diagnostic consumes the same geometry owner ---

def test_folded_center_distances_use_periodic_minimum_on_non_orthogonal_cell():
    centers = [ValleyCenter("Q", np.zeros(3), reciprocal_cart=HEX_BASIS)]
    report = build_folded_center_report(centers, HEX_BASIS, {"K1": np.array([0.51, -0.25, 0.0])})
    distance = report.kpoint_distances["Q"][0]
    component_wrapped = float(
        np.linalg.norm(
            (np.array([0.51, -0.25]) - np.rint(np.array([0.51, -0.25]))) @ HEX_BASIS[:2, :2]
        )
    )
    assert component_wrapped == pytest.approx(0.6519969325081215, abs=1e-12)
    assert distance == pytest.approx(np.sqrt(0.1951), abs=1e-14)
    assert distance < component_wrapped


# --- Exact coordinate compression and run-local distance reuse ---


class _RowCountingSearch:
    """Count rows and batches reaching the complete nearest-image search."""

    def __init__(self, monkeypatch):
        import valleyscope.geometry.reciprocal as reciprocal_module

        self.rows = 0
        self.batches = 0
        self._original = reciprocal_module._nearest_lattice_distances
        monkeypatch.setattr(
            reciprocal_module, "_nearest_lattice_distances", self
        )

    def __call__(self, deltas, lattice):
        self.rows += int(deltas.shape[0])
        self.batches += 1
        return self._original(deltas, lattice)


def _random_cart_rows(count: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((count, 3))
    out[:, :2] = rng.normal(size=(count, 2)) @ HEX_BASIS[:2, :2]
    return out


def test_exact_duplicate_rows_share_one_search_and_match_oracle(monkeypatch):
    counter = _RowCountingSearch(monkeypatch)
    base = _random_cart_rows(37)
    q = np.vstack([base, base[::-1]])
    q[:, 2] = 0.0

    distances = minimum_periodic_distance(
        q, np.zeros(3), HEX_BASIS, use_2d=True
    )

    assert counter.batches == 1
    assert counter.rows == 37
    assert np.array_equal(distances[:37], distances[37:][::-1])
    oracle = brute_force_minimum(q[:, :2], HEX_BASIS[:2, :2])
    assert np.array_equal(distances, oracle)


def test_lattice_shifted_rows_keep_equal_distances_without_merge(monkeypatch):
    # Rows separated by an exact lattice translate are physically equivalent.
    # Exact-bit deduplication groups only bit-equal coordinates, so such rows
    # may or may not share one search node -- but their distances are equal.
    counter = _RowCountingSearch(monkeypatch)
    base = _random_cart_rows(11, seed=13)
    shifted = base.copy()
    shifted[:, :2] = base[:, :2] + np.array([3.0, -2.0]) @ HEX_BASIS[:2, :2]
    q = np.vstack([base, shifted])
    q[:, 2] = 0.0

    distances = minimum_periodic_distance(
        q, np.zeros(3), HEX_BASIS, use_2d=True
    )

    assert 11 <= counter.rows <= 22
    assert np.allclose(distances[:11], distances[11:], rtol=0.0, atol=1e-12)


def test_use_2d_false_keeps_distinct_z_rows(monkeypatch):
    counter = _RowCountingSearch(monkeypatch)
    rows = _random_cart_rows(9, seed=21)
    q = np.vstack([rows, rows.copy()])
    q[:9, 2] = 0.0
    q[9:, 2] = 0.7

    distances = minimum_periodic_distance(
        q, np.zeros(3), HEX_BASIS, use_2d=False
    )

    assert counter.rows == 18
    assert not np.array_equal(distances[:9], distances[9:])


def test_dedup_preserves_empty_and_invalid_input_behavior():
    with pytest.raises(ValueError):
        minimum_periodic_distance(
            np.array([[0.0, np.nan, 0.0]]), np.zeros(3), HEX_BASIS
        )
    empty = minimum_periodic_distance(
        np.zeros((0, 3)), np.zeros(3), HEX_BASIS
    )
    assert empty.shape == (0,)


def test_cache_hits_only_on_identical_numeric_geometry(monkeypatch):
    counter = _RowCountingSearch(monkeypatch)
    cache = RunLocalPeriodicDistanceCache()
    q = _random_cart_rows(15)

    first = cache.distances(q, np.zeros(3), HEX_BASIS, use_2d=True)
    second = cache.distances(
        q.copy(), np.zeros(3), HEX_BASIS.copy(), use_2d=True
    )
    assert counter.batches == 1
    assert np.array_equal(first, second)

    misses = [
        (q + np.array([1e-12, 0.0, 0.0]), np.zeros(3), HEX_BASIS, True),
        (q, np.array([1e-12, 0.0, 0.0]), HEX_BASIS, True),
        (q, np.zeros(3), HEX_BASIS * (1.0 + 1e-12), True),
        (q, np.zeros(3), HEX_BASIS, False),
    ]
    for miss_q, center, basis, use_2d in misses:
        cache.distances(miss_q, center, basis, use_2d=use_2d)
    assert counter.batches == 1 + len(misses)


def test_cache_is_bounded_and_evicts_oldest_entries():
    cache = RunLocalPeriodicDistanceCache(max_entries=2)
    for seed in (1, 2, 3):
        cache.distances(
            _random_cart_rows(3, seed=seed), np.zeros(3), HEX_BASIS,
            use_2d=True,
        )
    assert len(cache) == 2


def test_cached_distances_are_read_only():
    cache = RunLocalPeriodicDistanceCache()
    distances = cache.distances(
        _random_cart_rows(4), np.zeros(3), HEX_BASIS, use_2d=True
    )
    with pytest.raises(ValueError):
        distances[0] = 0.0
