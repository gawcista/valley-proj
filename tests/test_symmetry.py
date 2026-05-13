import numpy as np
import pytest

from valleyscope.geometry.lattice import cart_rotation_from_fractional, cart_translation_from_fractional
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.symmetry.little_group import is_little_group_operation
from valleyscope.symmetry.operation_classifier import classify_operation, operation_order, rotation_axis_angle
from valleyscope.symmetry.plane_wave_action import build_plane_wave_representation, spin_rotation_matrix
from valleyscope.symmetry.rotation_eigenvalues import nearest_root_of_unity
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


def test_fractional_operation_matches_cartesian_column_convention_for_nonorthogonal_lattice():
    lattice = np.array(
        [
            [2.0, 0.2, 0.0],
            [0.7, 1.8, 0.0],
            [0.0, 0.1, 5.0],
        ]
    )
    rotation_frac = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation_frac = np.array([0.25, 0.5, 0.125])
    x_frac = np.array([0.2, 0.3, 0.4])

    rotation_cart = cart_rotation_from_fractional(rotation_frac, lattice)
    translation_cart = cart_translation_from_fractional(translation_frac, lattice)

    r_cart = lattice.T @ x_frac
    x_rot = rotation_frac @ x_frac
    x_rot_translated = rotation_frac @ x_frac + translation_frac

    np.testing.assert_allclose(lattice.T @ x_rot, rotation_cart @ r_cart, atol=1e-12)
    np.testing.assert_allclose(
        lattice.T @ x_rot_translated,
        rotation_cart @ r_cart + translation_cart,
        atol=1e-12,
    )


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


def test_valley_preservation_requires_every_center_to_map():
    centers = [
        ValleyCenter("top_K", np.array([1.0, 0.0, 0.0])),
        ValleyCenter("bottom_K", np.array([0.0, 1.0, 0.0])),
        ValleyCenter("top_Kp", np.array([-1.0, 0.0, 0.0])),
    ]
    sectors = [
        ValleySector("K_sector", ["top_K", "bottom_K"]),
        ValleySector("Kp_sector", ["top_Kp"]),
    ]
    c2z_cart = np.diag([-1.0, -1.0, 1.0])

    mapping = map_valley_sectors(np.eye(3), c2z_cart, centers, sectors, RECIP, tolerance=1e-8)

    assert mapping.center_mapping["top_K"] == "top_Kp"
    assert mapping.center_mapping["bottom_K"] is None
    assert mapping.sector_mapping["K_sector"] is None
    assert mapping.preserved["K_sector"] is False


def test_valley_preservation_uses_center_specific_reciprocal_lattice():
    top_recip = np.diag([20.0, 20.0, 1.0])
    bottom_recip = np.diag([10.0, 10.0, 1.0])
    centers = [
        ValleyCenter("top_K", np.array([2.0, 0.0, 0.0]), reciprocal_cart=top_recip),
        ValleyCenter("bottom_K", np.array([9.0, 0.0, 0.0]), reciprocal_cart=bottom_recip),
        ValleyCenter("top_Kp", np.array([-2.0, 0.0, 0.0]), reciprocal_cart=top_recip),
        ValleyCenter("bottom_Kp", np.array([1.0, 0.0, 0.0]), reciprocal_cart=bottom_recip),
    ]
    sectors = [
        ValleySector("K_sector", ["top_K", "bottom_K"]),
        ValleySector("Kp_sector", ["top_Kp", "bottom_Kp"]),
    ]
    fallback_recip = np.diag([7.0, 7.0, 1.0])
    c2z_cart = np.diag([-1.0, -1.0, 1.0])

    mapping = map_valley_sectors(np.eye(3), c2z_cart, centers, sectors, fallback_recip, tolerance=1e-8)

    assert mapping.center_mapping["bottom_K"] == "bottom_Kp"
    assert mapping.sector_mapping["K_sector"] == "Kp_sector"
    assert mapping.preserved["K_sector"] is False


def test_c3_preserves_multicenter_k_valley_sector():
    angle = 2.0 * np.pi / 3.0
    c3z_cart = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    k1 = np.array([1.0, 0.0, 0.0])
    k2 = c3z_cart @ k1
    k3 = c3z_cart @ k2
    centers = [
        ValleyCenter("K1", k1),
        ValleyCenter("K2", k2),
        ValleyCenter("K3", k3),
    ]
    sectors = [ValleySector("K_sector", ["K1", "K2", "K3"])]

    mapping = map_valley_sectors(np.eye(3), c3z_cart, centers, sectors, RECIP, tolerance=1e-8)

    assert mapping.sector_mapping["K_sector"] == "K_sector"
    assert mapping.preserved["K_sector"] is True


def test_rotation_axis_angle_recovers_c2_and_c3_about_z():
    axis, angle = rotation_axis_angle(np.diag([-1.0, -1.0, 1.0]))
    np.testing.assert_allclose(np.abs(axis), [0.0, 0.0, 1.0], atol=1e-12)
    assert angle == pytest.approx(np.pi)

    c3_angle = 2.0 * np.pi / 3.0
    c3z_cart = np.array(
        [
            [np.cos(c3_angle), -np.sin(c3_angle), 0.0],
            [np.sin(c3_angle), np.cos(c3_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    axis, angle = rotation_axis_angle(c3z_cart)
    np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-12)
    assert angle == pytest.approx(c3_angle)


def test_spinful_c2_rotation_matches_su2_generator():
    rot = spin_rotation_matrix(axis=np.array([0.0, 0.0, 1.0]), angle=np.pi)
    np.testing.assert_allclose(rot, np.diag([-1.0j, 1.0j]), atol=1e-12)


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


def test_nearest_root_of_unity_diagnostic():
    index, root, deviation = nearest_root_of_unity(1.001 * np.exp(2.0j * np.pi / 3.0), order=3)

    assert index == 1
    assert root == pytest.approx(np.exp(2.0j * np.pi / 3.0))
    assert deviation == pytest.approx(0.001)


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
