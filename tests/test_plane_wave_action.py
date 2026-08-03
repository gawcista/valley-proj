"""Focused regressions for the vectorized reciprocal-grid plane-wave action.

The profile-driven contraction replaced the per-row reciprocal-grid map
builder and the per-band action loops with vectorized equivalents.  These
tests pin the optimized map and source-ordered accumulation bit-for-bit to
the reference loop, prove the permutation validator fails closed on
non-bijective (many-to-one) maps that satisfy ``mapping_miss_count == 0``,
and bound the vectorized spin coefficients to the reference loop at
rounding scale.
"""

import numpy as np
import pytest

from valleyscope.symmetry.plane_wave_action import (
    apply_plane_wave_action,
    build_reciprocal_grid_map,
    validate_reciprocal_grid_permutation,
)


def _reference_map(q, rotation, tolerance, target_q=None):
    """Pre-vectorization per-row nearest-neighbour map semantics."""
    target = q if target_q is None else target_q
    mapping = np.full(len(q), -1, dtype=int)
    for source_idx, q_source in enumerate(q):
        q_rotated = rotation @ q_source
        best_idx, best_distance_sq = -1, float("inf")
        for target_idx, q_target in enumerate(target):
            distance_sq = float(np.sum((q_rotated - q_target) ** 2))
            if distance_sq <= tolerance * tolerance and distance_sq < best_distance_sq:
                best_idx, best_distance_sq = target_idx, distance_sq
        mapping[source_idx] = best_idx
    return mapping


def _reference_action(coeffs, q, rotation, translation, spin=None, target_q=None):
    """Pre-vectorization per-band action loop semantics."""
    n_bands, n_spinor, n_g = coeffs.shape
    target = q if target_q is None else target_q
    spin_matrix = np.eye(n_spinor, dtype=np.complex128) if spin is None else spin
    mapping = _reference_map(q, rotation, 1e-6, target)
    transformed = np.zeros(
        (n_bands, n_spinor, len(target)), dtype=np.complex128
    )
    for source_idx in range(n_g):
        target_idx = int(mapping[source_idx])
        if target_idx < 0:
            continue
        phase = np.exp(-1.0j * ((rotation @ q[source_idx]) @ translation))
        for band in range(n_bands):
            transformed[band, :, target_idx] += (
                phase * (spin_matrix @ coeffs[band, :, source_idx])
            )
    return transformed


def _cubic_grid(radius):
    """Integer cubic grid of cartesian q vectors in [-radius, radius]**3."""
    axes = np.arange(-radius, radius + 1, dtype=float)
    return np.array(
        [[x, y, z] for x in axes for y in axes for z in axes], dtype=float
    )


_ROT_Z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_ROT_X_180 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def _perturbed_rotation(angle_degrees):
    """90-degree z-rotation perturbed slightly, keeping q off exact grid."""
    angle = np.radians(90.0 + angle_degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


@pytest.mark.parametrize(
    "rotation",
    [
        np.eye(3),
        _ROT_Z_90,
        _ROT_X_180,
        _perturbed_rotation(0.01),
    ],
)
def test_vectorized_map_matches_reference_loop_exactly(rotation):
    q = _cubic_grid(2)
    mapping = build_reciprocal_grid_map(q, rotation).mapping
    assert mapping.tolist() == _reference_map(q, rotation, 1e-6).tolist()


def test_vectorized_map_matches_reference_for_distinct_target_grid():
    q = _cubic_grid(2)
    target = _cubic_grid(1)
    mapping = build_reciprocal_grid_map(
        q, _ROT_Z_90, target_q_cart=target
    ).mapping
    assert mapping.tolist() == _reference_map(
        q, _ROT_Z_90, 1e-6, target_q=target
    ).tolist()


def test_map_misses_off_axis_sources_but_keeps_rotation_axis():
    # A 135-degree z-rotation maps every off-axis lattice point off the
    # grid (miss), while the z-axis sources are fixed points and must map
    # to themselves at distance zero.
    q = _cubic_grid(2)
    rotation = _perturbed_rotation(45.0)
    result = build_reciprocal_grid_map(q, rotation)
    mapping = result.mapping.tolist()
    axis_indices = [
        i for i, vector in enumerate(q) if vector[0] == 0.0 and vector[1] == 0.0
    ]
    for i, vector in enumerate(q):
        if i in axis_indices:
            assert mapping[i] == i, f"axis source {vector} must self-map"
        else:
            assert mapping[i] == -1, f"off-axis source {vector} must miss"
    assert result.mapping_miss_count == len(q) - len(axis_indices)


def test_identity_mapping_on_shared_grid_is_exact_permutation():
    q = _cubic_grid(2)
    result = build_reciprocal_grid_map(q, np.eye(3))
    assert result.mapping.tolist() == list(range(len(q)))
    assert result.mapping_miss_count == 0
    assert validate_reciprocal_grid_permutation(
        result.mapping, dimension=len(q)
    ).status == "passed"


def test_vectorized_action_matches_reference_loop_arithmetic():
    # The vectorized action performs the same products in the same
    # summation order as the per-band reference loop; vectorized complex
    # kernels may fuse multiply-adds, so coefficients must agree to
    # rounding scale (observed max ~2e-15, bound asserted at 1e-13).
    # The reciprocal-grid map and the source-ordered accumulation are
    # pinned bit-for-bit by the other tests in this module.
    q = _cubic_grid(2)
    rng = np.random.default_rng(7)
    coeffs = rng.normal(size=(2, 2, len(q))) + 1.0j * rng.normal(
        size=(2, 2, len(q))
    )
    spin = rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2))
    translation = np.array([0.25, -0.5, 1.0 / 3.0])
    action = apply_plane_wave_action(
        coeffs,
        q,
        _ROT_Z_90,
        translation,
        spin_rotation=spin,
    )
    expected = _reference_action(
        coeffs, q, _ROT_Z_90, translation, spin=spin
    )
    difference = np.abs(action.transformed_coefficients - expected)
    scale = max(1.0, float(np.max(np.abs(expected))))
    assert float(np.max(difference)) <= 1e-13 * scale
    assert action.mapping.tolist() == _reference_map(
        q, _ROT_Z_90, 1e-6
    ).tolist()
    assert action.mapping_miss_count == 0


def test_action_accumulates_multiple_sources_into_one_target():
    # Three distinct source vectors all land within tolerance of the same
    # target; the vectorized np.add.at accumulation must sum all three
    # contributions in source order, matching the reference loop.
    q = np.array(
        [[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [0.0, 1e-9, 0.0]]
    )
    target_q = np.array([[0.0, 0.0, 0.0]])
    coeffs = np.zeros((1, 1, 3), dtype=np.complex128)
    coeffs[0, 0, :] = [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]
    action = apply_plane_wave_action(
        coeffs, q, np.eye(3), np.zeros(3), target_q_cart=target_q
    )
    assert action.mapping.tolist() == [0, 0, 0]
    expected = _reference_action(
        coeffs, q, np.eye(3), np.zeros(3), target_q=target_q
    )
    assert np.array_equal(action.transformed_coefficients, expected)
    assert action.transformed_coefficients[0, 0, 0] == 3.0 + 0.0j


@pytest.mark.parametrize(
    ("mapping", "dimension", "expected"),
    [
        # Many-to-one maps pass mapping_miss_count == 0 and norm residuals;
        # the validator must fail them closed on the collision check.
        (np.array([1, 1, 2, 3]), 4, ["target_index_collision", "target_coverage_incomplete"]),
        (np.array([0, 0, 2, 3]), 4, ["target_index_collision", "target_coverage_incomplete"]),
        (np.array([0, 1, 1]), 3, ["target_index_collision", "target_coverage_incomplete"]),
        # Length mismatch and negative entries drop sources, so the
        # independent coverage check accumulates too.
        (np.array([0, 1]), 3, ["source_coverage_incomplete", "target_coverage_incomplete"]),
        (np.array([0, -1, 2]), 3, ["source_coverage_incomplete", "target_coverage_incomplete"]),
        # Out-of-range target index drops a target from coverage.
        (np.array([0, 1, 3]), 3, ["target_index_out_of_range", "target_coverage_incomplete"]),
        # Valid permutations.
        (np.array([2, 0, 1]), 3, []),
        (np.array([0, 1, 2, 3], dtype=np.uint8), 4, []),
        # Unsigned dtype still reports collisions.
        (np.array([0, 0, 1], dtype=np.uint8), 3, ["target_index_collision", "target_coverage_incomplete"]),
    ],
)
def test_permutation_validation_fails_closed_on_ndarray_maps(
    mapping, dimension, expected,
):
    result = validate_reciprocal_grid_permutation(mapping, dimension=dimension)
    assert result.status == ("passed" if not expected else "blocked")
    assert list(result.reason_codes) == expected


@pytest.mark.parametrize(
    ("mapping", "dimension", "expected"),
    [
        ([0, 1, 1], 3, ["target_index_collision", "target_coverage_incomplete"]),
        # Negative and malformed entries are dropped from the candidate set,
        # so the independent coverage check accumulates as well.
        ([0, -1, 2], 3, ["source_coverage_incomplete", "target_coverage_incomplete"]),
        ([0, 1, 3], 3, ["target_index_out_of_range", "target_coverage_incomplete"]),
        ([0, 1.5, 2], 3, ["mapping_index_malformed", "target_coverage_incomplete"]),
        ([0, True, 2], 3, ["mapping_index_malformed", "target_coverage_incomplete"]),
        ([2, 0, 1], 3, []),
    ],
)
def test_permutation_validation_list_path_reports_identical_failures(
    mapping, dimension, expected,
):
    result = validate_reciprocal_grid_permutation(mapping, dimension=dimension)
    assert result.status == ("passed" if not expected else "blocked")
    assert list(result.reason_codes) == expected


def test_permutation_validation_rejects_malformed_maps():
    assert validate_reciprocal_grid_permutation(
        "0,1,2", dimension=3
    ).reason_codes == ("mapping_collection_malformed",)
    assert validate_reciprocal_grid_permutation(
        np.zeros((3, 3), dtype=int), dimension=3
    ).reason_codes == ("mapping_collection_malformed",)
    assert validate_reciprocal_grid_permutation(
        np.array([0.0, 1.0, 2.0]), dimension=3
    ).reason_codes == ("mapping_collection_malformed",)
    assert validate_reciprocal_grid_permutation(
        [0, 1], dimension=True
    ).reason_codes == ("reciprocal_grid_dimension_malformed",)
    assert validate_reciprocal_grid_permutation(
        [0, 1], dimension=-1
    ).reason_codes == ("reciprocal_grid_dimension_malformed",)


def test_map_builder_rejects_malformed_inputs():
    with pytest.raises(ValueError, match="q_cart"):
        build_reciprocal_grid_map(np.zeros((2, 2)), np.eye(3))
    with pytest.raises(ValueError, match="finite"):
        build_reciprocal_grid_map(
            np.array([[0.0, 0.0, np.nan]]), np.eye(3)
        )
    with pytest.raises(ValueError, match="rotation_cart"):
        build_reciprocal_grid_map(np.zeros((1, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="tolerance"):
        build_reciprocal_grid_map(
            np.zeros((1, 3)), np.eye(3), tolerance=0.0
        )
