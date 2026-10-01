"""Unit tests for rotation geometry and residue arithmetic, not physics trust."""

import cmath

import numpy as np
import pytest

from valleyscope.analysis.rotation_chern_math import (
    rotation_chern_residue,
    rotation_fixed_point_orbits,
)


_ROTATIONS = {
    2: np.array([[-1, 0], [0, -1]], dtype=int),
    3: np.array([[0, -1], [1, -1]], dtype=int),
    4: np.array([[0, -1], [1, 0]], dtype=int),
    6: np.array([[0, -1], [1, 1]], dtype=int),
}


@pytest.mark.parametrize("order,expected_sizes", [
    (2, [(1, 1)] * 4),
    (3, [(1, 1)] * 3),
    (4, [(1, 1), (1, 1), (2, 2)]),
    (6, [(1, 1), (2, 2), (3, 3)]),
])
def test_fixed_point_orbits_have_exact_power_and_full_orbit(order, expected_sizes):
    rows = rotation_fixed_point_orbits(_ROTATIONS[order], order)
    assert [(row["power"], len(row["points"])) for row in rows] == expected_sizes
    assert rows == rotation_fixed_point_orbits(_ROTATIONS[order], order)
    for row in rows:
        assert row["points"] == sorted(row["points"])
        assert all(0 <= value < 1 for point in row["points"] for value in point)
        point_set = {tuple(point) for point in row["points"]}
        for point in row["points"]:
            moved = _ROTATIONS[order] @ np.asarray(point)
            assert tuple(np.mod(moved, 1).round(12)) in {
                tuple(np.asarray(p).round(12)) for p in point_set
            }
            fixed = np.linalg.matrix_power(_ROTATIONS[order], row["power"]) @ point
            residual = fixed - point
            assert np.allclose(residual, np.rint(residual), atol=1e-12)


@pytest.mark.parametrize("order", [2, 3, 4, 6])
def test_unimodular_rebasis_transports_exact_fixed_point_geometry(order):
    change = np.array([[1, 1], [0, 1]], dtype=int)
    inverse = np.array([[1, -1], [0, 1]], dtype=int)
    rebased = change @ _ROTATIONS[order] @ inverse
    original = rotation_fixed_point_orbits(_ROTATIONS[order], order)
    transformed = rotation_fixed_point_orbits(rebased, order)
    for power in {row["power"] for row in original}:
        expected = {
            tuple(np.mod(change @ point, 1).round(12))
            for row in original if row["power"] == power
            for point in np.asarray(row["points"])
        }
        actual = {
            tuple(np.asarray(point).round(12))
            for row in transformed if row["power"] == power
            for point in row["points"]
        }
        assert actual == expected


@pytest.mark.parametrize("matrix,order", [
    (np.eye(2, dtype=int), 2),
    (_ROTATIONS[3], 4),
    (np.array([[1, 1], [0, 1]], dtype=int), 2),
    (np.array([[0, 1], [1, 0]], dtype=int), 2),
    (np.array([[1.0, 0.0], [0.0, -1.0]]), 2),
    (np.array([[True, False], [False, True]]), 2),
    (np.array([[np.nan, 0], [0, 1]]), 2),
])
def test_fixed_point_orbits_reject_nonrotation_or_noninteger_matrix(matrix, order):
    with pytest.raises(ValueError):
        rotation_fixed_point_orbits(matrix, order)


@pytest.mark.parametrize("order", [True, 0, 5, 2.0])
def test_fixed_point_orbits_reject_invalid_order(order):
    with pytest.raises(ValueError):
        rotation_fixed_point_orbits(_ROTATIONS[2], order)


@pytest.mark.parametrize("order,phase", [
    (2, -1),
    (3, cmath.exp(2j * cmath.pi / 3)),
    (4, 1j),
    (6, cmath.exp(1j * cmath.pi / 3)),
])
def test_nonzero_residue_for_each_supported_rotation(order, phase):
    length = 4 if order == 2 else 3
    determinants = [phase] + [1] * (length - 1)
    assert rotation_chern_residue(order, determinants, 2, False) == 1


@pytest.mark.parametrize("order", [2, 3, 4, 6])
def test_spinful_multiband_parity_factor_is_order_specific(order):
    length = 4 if order == 2 else 3
    determinants = [-1] + [1] * (length - 1)
    if order == 2:
        assert rotation_chern_residue(order, determinants, 1, True) == 1
    elif order == 3:
        with pytest.raises(ValueError):
            rotation_chern_residue(order, determinants, 2, True)
        assert rotation_chern_residue(order, determinants, 1, True) == 0
    else:
        assert rotation_chern_residue(order, determinants, 1, True) == 0
        assert rotation_chern_residue(order, determinants, 2, True) == order // 2


@pytest.mark.parametrize("order,determinants,rank,spinful,tolerance", [
    (2, [1, 1, 1], 1, False, 1e-6),
    (3, [1, 1, 1, 1], 1, False, 1e-6),
    (4, [1, 1, 1.1], 1, False, 1e-6),
    (6, [1, 1, cmath.exp(0.2j)], 1, False, 1e-6),
    (2, [1, 1, 1, complex(float("nan"), 0)], 1, False, 1e-6),
    (2, [1, 1, 1, 1], True, False, 1e-6),
    (2, [1, 1, 1, 1], 1, False, float("inf")),
    (2, [1, 1, 1, 1], 1, False, 0),
    (2, [1, 1, 1, 1], 1, False, 2),
])
def test_residue_rejects_malformed_or_ambiguous_evidence(
    order, determinants, rank, spinful, tolerance,
):
    with pytest.raises(ValueError):
        rotation_chern_residue(
            order, determinants, rank, spinful, tolerance=tolerance,
        )


@pytest.mark.parametrize("order", [True, 0, 5, 2.0])
def test_residue_rejects_invalid_order(order):
    with pytest.raises(ValueError):
        rotation_chern_residue(order, [1, 1, 1, 1], 1, False)
