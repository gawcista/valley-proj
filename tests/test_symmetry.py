import numpy as np
import pytest

from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.symmetry.little_group import is_little_group_operation
from valleyscope.symmetry.operation_classifier import classify_operation, operation_order
from valleyscope.symmetry.plane_wave_action import build_plane_wave_representation, spin_rotation_matrix
from valleyscope.symmetry.spglib_finder import find_symmetry_operations
from valleyscope.symmetry.valley_preservation import map_valley_sectors


RECIP = np.array(
    [
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def test_operation_order_and_classification_for_c3z():
    w = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
    assert operation_order(w) == 3

    info = classify_operation(w, np.zeros(3), allowed_orders=[2, 3, 4, 6])
    assert info.det == 1
    assert info.order == 3
    assert info.allowed_for_rotation_workflow is True


def test_little_group_uses_inverse_transpose_on_reciprocal_fractional_k():
    c2z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert is_little_group_operation(c2z, np.array([0.0, 0.0, 0.0]))
    assert is_little_group_operation(c2z, np.array([0.5, 0.0, 0.0]))
    assert not is_little_group_operation(c2z, np.array([1.0 / 3.0, 0.0, 0.0]))


def test_valley_preservation_allows_c3_and_rejects_c2_mapping_between_sectors():
    centers = [
        ValleyCenter("K", np.array([1.0, 0.0, 0.0])),
        ValleyCenter("Kp", np.array([-1.0, 0.0, 0.0])),
    ]
    sectors = [ValleySector("K_sector", ["K"]), ValleySector("Kp_sector", ["Kp"])]
    identity = np.eye(3, dtype=int)
    c2z_cart = np.diag([-1.0, -1.0, 1.0])

    id_mapping = map_valley_sectors(identity, np.eye(3), centers, sectors, RECIP, tolerance=1e-8)
    assert id_mapping.sector_mapping["K_sector"] == "K_sector"
    assert id_mapping.preserved["K_sector"] is True

    c2_mapping = map_valley_sectors(identity, c2z_cart, centers, sectors, RECIP, tolerance=1e-8)
    assert c2_mapping.sector_mapping["K_sector"] == "Kp_sector"
    assert c2_mapping.preserved["K_sector"] is False


def test_spinful_two_pi_rotation_returns_minus_identity():
    rot = spin_rotation_matrix(axis=np.array([0.0, 0.0, 1.0]), angle=2.0 * np.pi)
    assert np.allclose(rot, -np.eye(2), atol=1e-12)


def test_spinful_c3_rotation_has_unitary_matrix():
    rot = spin_rotation_matrix(axis=np.array([0.0, 0.0, 1.0]), angle=2.0 * np.pi / 3.0)
    assert np.allclose(rot.conj().T @ rot, np.eye(2), atol=1e-12)


def test_plane_wave_action_recovers_known_c2_eigenvalues():
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    coefficients = np.array(
        [
            [[inv_sqrt2, inv_sqrt2]],
            [[inv_sqrt2, -inv_sqrt2]],
        ],
        dtype=np.complex128,
    )
    q_cart = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    c2z_cart = np.diag([-1.0, -1.0, 1.0])

    result = build_plane_wave_representation(coefficients, q_cart, c2z_cart, np.zeros(3))

    assert result.mapping_miss_count == 0
    assert np.allclose(result.matrix, np.diag([1.0, -1.0]), atol=1e-12)


def test_spglib_finds_c3_candidate_for_simple_hexagonal_cell():
    lattice = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 8.0],
        ]
    )
    positions = np.array([[0.0, 0.0, 0.0]])
    numbers = np.array([1])

    dataset = find_symmetry_operations((lattice, positions, numbers), symprec=1e-5)
    infos = [classify_operation(rot, trans) for rot, trans in zip(dataset.rotations, dataset.translations)]

    assert any(info.order == 3 and info.det == 1 for info in infos)
