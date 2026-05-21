import numpy as np
import pytest

from valleyscope.geometry.lattice import cart_rotation_from_fractional, cart_translation_from_fractional
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.symmetry.little_group import is_little_group_operation
from valleyscope.symmetry.operation_classifier import classify_operation, operation_order, rotation_axis_angle
from valleyscope.symmetry.plane_wave_action import build_plane_wave_representation, spin_rotation_matrix
from valleyscope.symmetry.rotation_eigenvalues import nearest_root_of_unity
from valleyscope.symmetry.rotation_selection import mark_rotation_generators, resolve_rotation_order
from valleyscope.symmetry.spglib_finder import find_symmetry_operations
from valleyscope.symmetry.valley_preservation import map_valley_sectors
from valleyscope.analysis.symmetry_eigenvalue_diagnostic import symmetry_eigenvalue_diagnostics_for_kpoint
from valleyscope.analysis.valley_little_group import (
    add_valley_irrep_results,
    build_valley_preserving_subgroup_report,
    update_valley_little_group_inventory,
)


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


def test_plane_wave_action_pure_translation_phase():
    coefficients = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
    q_cart = np.array([[2.0, 0.0, 0.0]])
    translation = np.array([0.3, 0.0, 0.0])

    result = build_plane_wave_representation(coefficients, q_cart, np.eye(3), translation)

    assert result.mapping_miss_count == 0
    np.testing.assert_allclose(result.matrix, [[np.exp(-0.6j)]], atol=1e-12)


def test_plane_wave_action_non_origin_c2_rotation_phase():
    coefficients = np.array(
        [
            [[1.0 + 0.0j, 0.0 + 0.0j]],
            [[0.0 + 0.0j, 1.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    q_cart = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    rotation = np.diag([-1.0, -1.0, 1.0])
    center = np.array([0.25, 0.0, 0.0])
    translation = center - rotation @ center

    result = build_plane_wave_representation(coefficients, q_cart, rotation, translation)

    expected = np.array(
        [
            [0.0, np.exp(-0.5j)],
            [np.exp(0.5j), 0.0],
        ],
        dtype=np.complex128,
    )
    assert result.mapping_miss_count == 0
    np.testing.assert_allclose(result.matrix, expected, atol=1e-12)


def test_plane_wave_action_recovers_c3_angular_momentum_eigenvalue():
    angle = 2.0 * np.pi / 3.0
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    q_cart = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.cos(angle), np.sin(angle), 0.0],
            [np.cos(2.0 * angle), np.sin(2.0 * angle), 0.0],
        ]
    )
    phases = np.exp(1.0j * np.array([0.0, angle, 2.0 * angle]))
    coefficients = phases.reshape(1, 1, 3) / np.sqrt(3.0)

    result = build_plane_wave_representation(coefficients, q_cart, rotation, np.zeros(3), tolerance=1e-7)

    assert result.mapping_miss_count == 0
    np.testing.assert_allclose(result.matrix, [[np.exp(-1.0j * angle)]], atol=1e-12)


def test_spinor_symmetry_rows_are_diagnostic_only_without_convention_benchmark():
    coefficients = np.array(
        [
            [
                [1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j],
            ]
        ],
        dtype=np.complex128,
    )
    q_cart = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    symmetry_payload = {
        "detected_operations": [
            {
                "operation_id": 0,
                "candidate_rotation": True,
                "rotation_frac": np.diag([-1, -1, 1]),
                "rotation_cart": np.diag([-1.0, -1.0, 1.0]),
                "translation_cart": np.zeros(3),
                "preserved": {"K_sector": True},
                "order": 2,
                "kind": "C2",
            }
        ]
    }
    representation_payload: dict[str, object] = {}

    rows = symmetry_eigenvalue_diagnostics_for_kpoint(
        kpoint_name="GammaM",
        k_frac=np.zeros(3),
        q_cart=q_cart,
        coefficients=coefficients,
        symmetry_payload=symmetry_payload,
        basis_payload=None,
        representation_payload=representation_payload,
    )

    assert rows
    assert rows[0]["spinor_rotation_applied"] is True
    assert rows[0]["spinor_convention_verified"] is False
    assert rows[0]["diagnostic_only"] is True
    assert rows[0]["topology_input_ready"] is False
    assert rows[0]["topology_ready"] is False


def test_spinful_c3_root_diagnostic_uses_double_group_order():
    angle = 2.0 * np.pi / 3.0
    rotation_cart = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    coefficients = np.array([[[1.0 + 0.0j], [0.0 + 0.0j]]], dtype=np.complex128)
    symmetry_payload = {
        "detected_operations": [
            {
                "operation_id": 0,
                "candidate_rotation": True,
                "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
                "rotation_cart": rotation_cart,
                "translation_cart": np.zeros(3),
                "preserved": {"K_sector": True},
                "order": 3,
                "kind": "C3",
            }
        ]
    }
    representation_payload: dict[str, object] = {}

    rows = symmetry_eigenvalue_diagnostics_for_kpoint(
        kpoint_name="GammaM",
        k_frac=np.zeros(3),
        q_cart=np.zeros((1, 3)),
        coefficients=coefficients,
        symmetry_payload=symmetry_payload,
        basis_payload=None,
        representation_payload=representation_payload,
    )

    assert rows[0]["nearest_root_of_unity"] == "exp(2pii*5/6)"
    assert rows[0]["root_deviation"] == pytest.approx(0.0, abs=1e-12)


def test_spinor_convention_benchmark_marks_spinor_rows_verified():
    angle = 2.0 * np.pi / 3.0
    rotation_cart = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    coefficients = np.array([[[1.0 + 0.0j], [0.0 + 0.0j]]], dtype=np.complex128)
    symmetry_payload = {
        "detected_operations": [
            {
                "operation_id": 0,
                "candidate_rotation": True,
                "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
                "rotation_cart": rotation_cart,
                "translation_cart": np.zeros(3),
                "preserved": {"K_sector": True},
                "order": 3,
                "kind": "C3",
            }
        ]
    }

    rows = symmetry_eigenvalue_diagnostics_for_kpoint(
        kpoint_name="GammaM",
        k_frac=np.zeros(3),
        q_cart=np.zeros((1, 3)),
        coefficients=coefficients,
        symmetry_payload=symmetry_payload,
        basis_payload=None,
        representation_payload={},
        spinor_convention_verified=True,
    )

    assert rows[0]["spinor_rotation_applied"] is True
    assert rows[0]["spinor_convention_verified"] is True
    assert rows[0]["reason"] == "not valley-adapted"


def test_rotation_ready_tolerates_small_truncation_unitarity_error():
    coefficients = np.array([[[np.sqrt(0.99999) + 0.0j]]], dtype=np.complex128)
    symmetry_payload = {
        "detected_operations": [
            {
                "operation_id": 0,
                "candidate_rotation": True,
                "rotation_frac": np.diag([-1, -1, 1]),
                "rotation_cart": np.diag([-1.0, -1.0, 1.0]),
                "translation_cart": np.zeros(3),
                "preserved": {"K_sector": True},
                "order": 2,
                "kind": "C2",
            }
        ]
    }
    rows = symmetry_eigenvalue_diagnostics_for_kpoint(
        kpoint_name="GammaM",
        k_frac=np.zeros(3),
        q_cart=np.zeros((1, 3)),
        coefficients=coefficients,
        symmetry_payload=symmetry_payload,
        basis_payload=None,
        representation_payload={},
    )

    assert rows[0]["unitarity_deviation"] < 1.0e-4
    assert rows[0]["rotation_ready"] is True


def test_plane_wave_mapping_uses_local_lookup_not_dense_all_pairs(monkeypatch):
    original_norm = np.linalg.norm

    def guarded_norm(value, *args, **kwargs):
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[0] > 64:
            raise AssertionError("dense all-pairs q-vector matching is too expensive")
        return original_norm(value, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", guarded_norm)
    n_pairs = 128
    q_positive = np.column_stack(
        [
            np.arange(1, n_pairs + 1, dtype=float),
            np.zeros(n_pairs),
            np.zeros(n_pairs),
        ]
    )
    q_cart = np.vstack([q_positive, -q_positive])
    coefficients = np.zeros((1, 1, 2 * n_pairs), dtype=np.complex128)
    coefficients[0, 0, 0] = 1.0

    result = build_plane_wave_representation(
        coefficients,
        q_cart,
        np.diag([-1.0, -1.0, 1.0]),
        np.zeros(3),
    )

    assert result.mapping_miss_count == 0
    assert result.mapping[0] == n_pairs


def test_plane_wave_mapping_default_tolerance_handles_real_wavecar_roundoff():
    q_cart = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0 + 2.0e-7, 0.0, 0.0],
        ]
    )
    coefficients = np.array([[[1.0 + 0.0j, 0.0 + 0.0j]]], dtype=np.complex128)

    result = build_plane_wave_representation(
        coefficients,
        q_cart,
        np.diag([-1.0, -1.0, 1.0]),
        np.zeros(3),
    )

    assert result.mapping_miss_count == 0
    assert result.mapping[0] == 1


def test_rotation_order_selection_from_user_value_and_spacegroup():
    assert resolve_rotation_order(3, international="P422", candidate_orders=[2, 3, 4]) == 3
    assert resolve_rotation_order("none", international="P321", candidate_orders=[3]) is None
    assert resolve_rotation_order("None", international="P321", candidate_orders=[3]) is None
    assert resolve_rotation_order("auto", international="P321", candidate_orders=[2, 3]) == 3
    assert resolve_rotation_order("AUTO", international="P312", candidate_orders=[2, 3]) == 3
    assert resolve_rotation_order("auto", international="P422", candidate_orders=[2, 4]) == 4


def test_rotation_generator_filter_keeps_one_cyclic_generator():
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])
    c3_squared = c3 @ c3
    operations = [
        {"operation_id": 1, "rotation_frac": c3, "order": 3, "candidate_rotation": True},
        {"operation_id": 2, "rotation_frac": c3_squared, "order": 3, "candidate_rotation": True},
        {"operation_id": 3, "rotation_frac": np.diag([-1, -1, 1]), "order": 2, "candidate_rotation": False},
    ]

    mark_rotation_generators(operations)

    assert [op["operation_id"] for op in operations if op["candidate_rotation"]] == [1]
    assert operations[0]["rotation_generator_operation_id"] == 1
    assert operations[0]["rotation_power_of_generator"] == 1
    assert operations[1]["rotation_generator_operation_id"] == 1
    assert operations[1]["rotation_power_of_generator"] == 2
    assert operations[1]["candidate_rotation"] is False
    assert operations[1]["candidate_rejection_reason"] == "power_of_rotation_generator"


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


class TestLittleGroupExtendedDiagnostics:
    """V1.1: extended diagnostics for ALL valley-preserving little-group operations."""

    def _toy_symmetry_payload(self, operations):
        return {"status": "ok", "detected_operations": operations, "little_group_check": {}, "valley_preservation_check": {}}

    def test_generators_only_default_behavior_unchanged(self):
        """Default generators_only=False enumerates all proper rotations. generators_only=True matches old behavior."""
        from valleyscope.analysis.symmetry_eigenvalue_diagnostic import symmetry_eigenvalue_diagnostics_for_kpoint
        coeffs = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
        ops = [
            {
                "operation_id": 0, "candidate_rotation": True, "order": 2, "kind": "C2",
                "rotation_frac": np.diag([-1, -1, 1]), "rotation_cart": np.diag([-1.0, -1.0, 1.0]),
                "translation_cart": np.zeros(3), "preserved": {"K": True},
            },
            {
                "operation_id": 1, "candidate_rotation": False, "order": 3, "kind": "C3",
                "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
                "rotation_cart": np.eye(3), "translation_cart": np.zeros(3),
                "preserved": {"K": True},
            },
        ]
        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3), q_cart=np.zeros((1, 3)),
            coefficients=coeffs, symmetry_payload=self._toy_symmetry_payload(ops),
            basis_payload=None, representation_payload={}, generators_only=True,
        )
        assert len(rows) > 0
        assert all(row["operation_id"] == 0 for row in rows)

    def test_generators_only_false_includes_all_proper_rotations(self):
        """With generators_only=False, C3 (order=3, non-candidate) should be included."""
        from valleyscope.analysis.symmetry_eigenvalue_diagnostic import symmetry_eigenvalue_diagnostics_for_kpoint
        coeffs = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
        ops = [
            {
                "operation_id": 0, "candidate_rotation": True, "order": 2, "kind": "C2",
                "rotation_frac": np.diag([-1, -1, 1]), "rotation_cart": np.diag([-1.0, -1.0, 1.0]),
                "translation_cart": np.zeros(3), "preserved": {"K": True},
            },
            {
                "operation_id": 1, "candidate_rotation": False, "order": 3, "kind": "C3",
                "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
                "rotation_cart": np.eye(3), "translation_cart": np.zeros(3),
                "preserved": {"K": True},
            },
        ]
        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3), q_cart=np.zeros((1, 3)),
            coefficients=coeffs, symmetry_payload=self._toy_symmetry_payload(ops),
            basis_payload=None, representation_payload={}, generators_only=False,
        )
        op_ids = {row["operation_id"] for row in rows}
        assert 1 in op_ids, "C3 should be included with generators_only=False"

    def test_non_little_group_skipped_with_reason(self):
        """Operations not in little group should be skipped with clear reason."""
        from valleyscope.analysis.symmetry_eigenvalue_diagnostic import symmetry_eigenvalue_diagnostics_for_kpoint
        coeffs = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
        c2z = np.diag([-1, -1, 1])
        ops = [
            {
                "operation_id": 0, "candidate_rotation": True, "order": 2, "kind": "C2",
                "rotation_frac": c2z, "rotation_cart": np.diag([-1.0, -1.0, 1.0]),
                "translation_cart": np.zeros(3), "preserved": {"K": True},
            },
        ]
        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="K", k_frac=np.array([1.0/3.0, 0.0, 0.0]),
            q_cart=np.zeros((1, 3)), coefficients=coeffs,
            symmetry_payload=self._toy_symmetry_payload(ops),
            basis_payload=None, representation_payload={}, generators_only=False,
        )
        assert rows == []

    def test_valley_exchanging_operation_gets_specific_rejection_reason(self):
        """A little-group operation that maps K to Kp is valley-exchanging."""
        from valleyscope.analysis.symmetry_eigenvalue_diagnostic import symmetry_eigenvalue_diagnostics_for_kpoint
        coeffs = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
        ops = [
            {
                "operation_id": 2,
                "candidate_rotation": True,
                "order": 2,
                "kind": "C2",
                "rotation_frac": np.eye(3, dtype=int),
                "rotation_cart": np.eye(3),
                "translation_cart": np.zeros(3),
                "preserved": {"K_valley": False, "Kp_valley": False},
                "sector_mapping": {"K_valley": "Kp_valley", "Kp_valley": "K_valley"},
            },
        ]

        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3), q_cart=np.zeros((1, 3)),
            coefficients=coeffs, symmetry_payload=self._toy_symmetry_payload(ops),
            basis_payload=None, representation_payload={},
        )

        assert rows == []
        assert ops[0]["rejection_reason_by_kpoint"]["GM"] == "valley-exchanging"

    def test_non_generator_operation_still_gets_little_group_and_valley_checks(self):
        """Non-generator rotations are not diagonalized by default but are still classified."""
        from valleyscope.analysis.symmetry_eigenvalue_diagnostic import symmetry_eigenvalue_diagnostics_for_kpoint
        coeffs = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
        ops = [
            {
                "operation_id": 2,
                "candidate_rotation": False,
                "order": 2,
                "kind": "C2",
                "rotation_frac": np.eye(3, dtype=int),
                "rotation_cart": np.eye(3),
                "translation_cart": np.zeros(3),
                "preserved": {"K_valley": False, "Kp_valley": False},
                "sector_mapping": {"K_valley": "Kp_valley", "Kp_valley": "K_valley"},
            },
        ]

        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3), q_cart=np.zeros((1, 3)),
            coefficients=coeffs, symmetry_payload=self._toy_symmetry_payload(ops),
            basis_payload=None, representation_payload={},
        )

        assert rows == []
        assert ops[0]["little_group_by_kpoint"]["GM"] is True
        assert ops[0]["rejection_reason_by_kpoint"]["GM"] == "valley-exchanging"

    def test_character_fields_present_in_output(self):
        """Output rows should contain character_raw and character_valley."""
        from valleyscope.analysis.symmetry_eigenvalue_diagnostic import symmetry_eigenvalue_diagnostics_for_kpoint
        coeffs = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
        ops = [
            {
                "operation_id": 0, "candidate_rotation": True, "order": 2, "kind": "C2",
                "rotation_frac": np.diag([-1, -1, 1]), "rotation_cart": np.diag([-1.0, -1.0, 1.0]),
                "translation_cart": np.zeros(3), "preserved": {"K": True},
            },
        ]
        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3), q_cart=np.zeros((1, 3)),
            coefficients=coeffs, symmetry_payload=self._toy_symmetry_payload(ops),
            basis_payload=None, representation_payload={},
        )
        assert "character_raw" in rows[0]
        assert "character_valley" in rows[0]
        assert "little_group_passed" in rows[0]
        assert "valley_preserving" in rows[0]

    def test_spinful_c3_double_valued_character_in_valley_adapted_basis(self):
        coeffs = np.zeros((2, 2, 1), dtype=np.complex128)
        coeffs[0, 0, 0] = 1.0
        coeffs[1, 1, 0] = 1.0
        angle = 2.0 * np.pi / 3.0
        c3_cart = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        operations = [
            {
                "operation_id": 3,
                "candidate_rotation": True,
                "order": 3,
                "kind": "C3",
                "rotation_frac": np.eye(3, dtype=int),
                "rotation_cart": c3_cart,
                "translation_cart": np.zeros(3),
                "preserved": {"K_valley": True},
                "sector_mapping": {"K_valley": "K_valley"},
            }
        ]
        representation_payload: dict[str, object] = {}

        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM",
            k_frac=np.zeros(3),
            q_cart=np.zeros((1, 3)),
            coefficients=coeffs,
            symmetry_payload=self._toy_symmetry_payload(operations),
            basis_payload={
                "valid_valley_subspace": True,
                "transform": np.eye(2, dtype=np.complex128),
                "eta": np.array([1.0, -1.0]),
            },
            representation_payload=representation_payload,
            spinor_convention_verified=True,
        )

        assert {row["nearest_root_of_unity"] for row in rows} == {
            "exp(2pii*1/6)",
            "exp(2pii*5/6)",
        }
        assert all(row["topology_input_ready"] for row in rows)
        assert rows[0]["character_valley"] == "1.000000+0.000000j"


def test_valley_little_group_inventory_marks_allowed_and_valley_exchanging_operations():
    operations = [
        {
            "operation_id": 0,
            "kind": "identity",
            "order": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
        {
            "operation_id": 1,
            "kind": "C2",
            "order": 2,
            "rotation_frac": np.eye(3, dtype=int),
            "preserved": {"K_valley": False, "Kp_valley": False},
            "sector_mapping": {"K_valley": "Kp_valley", "Kp_valley": "K_valley"},
        },
        {
            "operation_id": 2,
            "kind": "C2",
            "order": 2,
            "rotation_frac": np.diag([-1, -1, 1]),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
    ]
    symmetry_payload = {"detected_operations": operations}

    per_valley = update_valley_little_group_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="KM",
        k_frac=np.array([1.0 / 3.0, 0.0, 0.0]),
        valley_names=["K_valley", "Kp_valley"],
    )

    # Per-valley: K_valley inventory
    kv_rows = per_valley["K_valley"]
    assert [row["operation_id"] for row in kv_rows] == [0, 1, 2]
    assert kv_rows[0]["allowed_for_single_valley_representation"] is True
    assert kv_rows[1]["little_group_passed"] is True
    assert kv_rows[1]["valley_preserving"] is False
    assert "valley-changing" in kv_rows[1]["reason"]
    assert kv_rows[2]["little_group_passed"] is False
    assert kv_rows[2]["reason"] == "not in little group"
    assert operations[1]["rejection_reason_by_kpoint"]["KM"] == "valley-exchanging"

    # Flat inventory still stored for backward compat
    flat = symmetry_payload["valley_little_group_inventory"]["KM"]
    assert [row["operation_id"] for row in flat] == [0, 1, 2]
    assert flat[0]["allowed_for_single_valley_representation"] is True
    assert flat[1]["valley_exchanging"] is True
    assert flat[1]["reason"] == "valley-exchanging"


def test_valley_preserving_subgroup_report_checks_operation_set_closure():
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    c3_square = c3 @ c3
    operations = [
        {
            "operation_id": 0,
            "kind": "identity",
            "order": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True},
            "sector_mapping": {"K_valley": "K_valley"},
        },
        {
            "operation_id": 1,
            "kind": "C3",
            "order": 3,
            "rotation_frac": c3,
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True},
            "sector_mapping": {"K_valley": "K_valley"},
        },
        {
            "operation_id": 2,
            "kind": "C3^2",
            "order": 3,
            "rotation_frac": c3_square,
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True},
            "sector_mapping": {"K_valley": "K_valley"},
        },
        {
            "operation_id": 3,
            "kind": "C2",
            "order": 2,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": False},
            "sector_mapping": {"K_valley": "Kp_valley"},
        },
    ]
    symmetry_payload = {"detected_operations": operations}
    update_valley_little_group_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="GM",
        k_frac=np.zeros(3),
        valley_names=["K_valley"],
    )

    report = build_valley_preserving_subgroup_report(
        symmetry_payload=symmetry_payload,
        target_kpoints=["GM"],
    )

    assert report["status"] == "per_valley_stabilizers_computed"
    # Per-valley stabilizer for K_valley
    assert report["valley_stabilizers"]["K_valley"]["operation_ids"] == [0, 1, 2]
    # All-valley intersection (debug)
    assert report["all_valley_intersection"]["allowed_operation_ids"] == [0, 1, 2]
    # Per-valley by_kpoint
    gm_k = report["by_kpoint"]["GM"]["K_valley"]
    assert gm_k["allowed_operation_ids"] == [0, 1, 2]
    assert gm_k["valley_changing_operation_ids"] == [3]
    assert gm_k["closure_status"] == "closed"
    assert gm_k["missing_products"] == []
    assert report["irrep_matching"]["status"] == "table_mapping_incomplete"


def test_valley_preserving_subgroup_report_identifies_standard_global_subgroup():
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    c3_square = c3 @ c3
    lattice = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    operations = [
        {
            "operation_id": 0,
            "kind": "identity",
            "order": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
        {
            "operation_id": 1,
            "kind": "C3",
            "order": 3,
            "rotation_frac": c3,
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
        {
            "operation_id": 2,
            "kind": "C3^2",
            "order": 3,
            "rotation_frac": c3_square,
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
        {
            "operation_id": 3,
            "kind": "C2",
            "order": 2,
            "rotation_frac": np.diag([-1, -1, 1]),
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": False, "Kp_valley": False},
            "sector_mapping": {"K_valley": "Kp_valley", "Kp_valley": "K_valley"},
        },
    ]
    symmetry_payload = {
        "detected_operations": operations,
        "lattice_direct_cart": lattice,
    }
    update_valley_little_group_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="GM",
        k_frac=np.zeros(3),
        valley_names=["K_valley", "Kp_valley"],
    )

    report = build_valley_preserving_subgroup_report(
        symmetry_payload=symmetry_payload,
        target_kpoints=["GM"],
    )

    assert report["status"] == "per_valley_stabilizers_computed"
    # All-valley intersection (debug)
    assert report["all_valley_intersection"]["operation_set_label"] == "all_valley_intersection"
    assert report["all_valley_intersection"]["allowed_operation_ids"] == [0, 1, 2]
    assert report["all_valley_intersection"]["valley_exchanging_operation_ids"] == [3]
    assert report["all_valley_intersection"]["closure_status"] == "closed"
    # Per-valley stabilizers both match P3 since both valleys are preserved by C3
    k_match = report["per_valley_standard_matches"]["K_valley"]
    assert k_match["standard_group_match_status"] == "matched"
    assert k_match["standard_group_match"]["number"] == 143
    assert k_match["standard_group_match"]["international_short"] == "P3"
    # Per-valley by_kpoint
    gm_k = report["by_kpoint"]["GM"]["K_valley"]
    assert gm_k["closure_status"] == "closed"


def test_valley_preserving_subgroup_report_maps_operations_to_irreptables():
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    c3_square = c3 @ c3
    lattice = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    operations = [
        {
            "operation_id": 0,
            "kind": "identity",
            "order": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
        {
            "operation_id": 1,
            "kind": "C3",
            "order": 3,
            "rotation_frac": c3,
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
        {
            "operation_id": 2,
            "kind": "C3^2",
            "order": 3,
            "rotation_frac": c3_square,
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
    ]
    symmetry_payload = {
        "detected_operations": operations,
        "lattice_direct_cart": lattice,
        "spinor_wavefunction": True,
    }
    update_valley_little_group_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="KM",
        k_frac=np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
        valley_names=["K_valley", "Kp_valley"],
    )

    report = build_valley_preserving_subgroup_report(
        symmetry_payload=symmetry_payload,
        target_kpoints=["KM"],
    )

    matching = report["irrep_matching"]
    assert matching["status"] == "table_mapping_complete"
    assert matching["table_source"] == "irreptables"
    # Per-valley matching
    kv_match = matching["per_valley"]["K_valley"]
    assert kv_match["spacegroup_number"] == 143
    assert kv_match["spinor"] is True
    assert kv_match["operation_to_table_mapping_status"] == "complete"
    assert kv_match["operation_to_table_mapping"] == {0: 1, 1: 2, 2: 3}
    assert kv_match["unmatched_operation_ids"] == []
    assert kv_match["unused_table_operation_indices"] == []
    km_matching = kv_match["by_kpoint"]["KM"]
    assert km_matching["status"] == "table_kpoint_matched"
    assert km_matching["table_kpoint_label"] == "K"
    assert km_matching["table_operation_indices"] == [1, 2, 3]
    assert km_matching["mapped_allowed_table_operation_indices"] == [1, 2, 3]
    assert km_matching["missing_table_operation_indices"] == []
    assert km_matching["extra_mapped_table_operation_indices"] == []
    assert {"-K4", "-K5", "-K6"} <= set(km_matching["available_irrep_labels"])


def test_valley_irrep_results_match_characters_to_irrep_multiplicities():
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    c3_square = c3 @ c3
    lattice = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    operations = [
        {
            "operation_id": 0,
            "kind": "identity",
            "order": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
        {
            "operation_id": 1,
            "kind": "C3",
            "order": 3,
            "rotation_frac": c3,
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
        {
            "operation_id": 2,
            "kind": "C3^2",
            "order": 3,
            "rotation_frac": c3_square,
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True, "Kp_valley": True},
            "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
        },
    ]
    symmetry_payload = {
        "detected_operations": operations,
        "lattice_direct_cart": lattice,
        "spinor_wavefunction": True,
    }
    update_valley_little_group_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="KM",
        k_frac=np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
        valley_names=["K_valley", "Kp_valley"],
    )
    build_valley_preserving_subgroup_report(
        symmetry_payload=symmetry_payload,
        target_kpoints=["KM"],
    )
    symmetry_rows = [
        {
            "kpoint": "KM",
            "target_valley": "K_valley",
            "operation_id": 1,
            "state_index": 0,
            "character_valley": "1.000000+0.000000j",
            "little_group_passed": True,
            "valley_preserving": True,
            "topology_input_ready": True,
        },
        {
            "kpoint": "KM",
            "target_valley": "K_valley",
            "operation_id": 1,
            "state_index": 1,
            "character_valley": "",
            "little_group_passed": True,
            "valley_preserving": True,
            "topology_input_ready": True,
        },
        {
            "kpoint": "KM",
            "target_valley": "K_valley",
            "operation_id": 2,
            "state_index": 0,
            "character_valley": "1.000000+0.000000j",
            "little_group_passed": True,
            "valley_preserving": True,
            "topology_input_ready": True,
        },
        {
            "kpoint": "KM",
            "target_valley": "K_valley",
            "operation_id": 2,
            "state_index": 1,
            "character_valley": "",
            "little_group_passed": True,
            "valley_preserving": True,
            "topology_input_ready": True,
        },
    ]

    add_valley_irrep_results(
        symmetry_payload=symmetry_payload,
        symmetry_rows=symmetry_rows,
    )

    matching = symmetry_payload["valley_preserving_subgroup_report"]["irrep_matching"]
    km_result = matching["irrep_results_by_kpoint"]["KM"]["K_valley"]
    assert km_result["status"] == "matched"
    assert km_result["table_kpoint_label"] == "K"
    assert km_result["identity_character_source"] == "inferred_from_ready_rows"
    assert km_result["computed_characters"] == {
        "1": "2.000000+0.000000j",
        "2": "1.000000+0.000000j",
        "3": "1.000000+0.000000j",
    }
    assert km_result["irrep_multiplicities"] == {"-K5": 1, "-K6": 1}
    assert km_result["failure_reasons"] == []
    # Kp_valley has table mapping but no character rows -> missing_characters
    kp_result = matching["irrep_results_by_kpoint"]["KM"]["Kp_valley"]
    assert kp_result["status"] == "missing_characters"


def test_valley_preserving_subgroup_report_records_missing_products():
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    operations = [
        {
            "operation_id": 0,
            "kind": "identity",
            "order": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True},
            "sector_mapping": {"K_valley": "K_valley"},
        },
        {
            "operation_id": 1,
            "kind": "C3",
            "order": 3,
            "rotation_frac": c3,
            "translation_frac": np.zeros(3),
            "preserved": {"K_valley": True},
            "sector_mapping": {"K_valley": "K_valley"},
        },
    ]
    symmetry_payload = {"detected_operations": operations}
    update_valley_little_group_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="GM",
        k_frac=np.zeros(3),
        valley_names=["K_valley"],
    )

    report = build_valley_preserving_subgroup_report(
        symmetry_payload=symmetry_payload,
        target_kpoints=["GM"],
    )

    gm_k = report["by_kpoint"]["GM"]["K_valley"]
    assert gm_k["closure_status"] == "not_closed"
    assert gm_k["missing_products"] == [
        {"left_operation_id": 1, "right_operation_id": 1}
    ]


class TestV11PerValleyStabilizer:
    """V1.1 per-valley stabilizer tests: three-valley orbit, per-valley gates,
    block-leakage diagnostic, and rotation_order independence."""

    @staticmethod
    def _c3_cart():
        angle = 2.0 * np.pi / 3.0
        return np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def _c2_m1_cart():
        """C2 that preserves M1 valley (at [1,0,0]) and exchanges M2/M3."""
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def _c2_m2_cart():
        """C2 that preserves M2 valley (at [-1/2, sqrt(3)/2, 0]) and exchanges M1/M3."""
        angle = 2.0 * np.pi / 3.0
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        c3 = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
        c2_m1 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        return c3 @ c2_m1 @ c3.T

    @staticmethod
    def _c2_m3_cart():
        """C2 that preserves M3 valley (at [-1/2, -sqrt(3)/2, 0]) and exchanges M1/M2."""
        angle = 2.0 * np.pi / 3.0
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        c3_inv = np.array([[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
        c2_m1 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        return c3_inv @ c2_m1 @ c3_inv.T

    def _three_valley_operations(self):
        """Build operations for 3-valley M1/M2/M3 orbit under C3.

        M1 at [1, 0, 0], M2 at [-1/2, sqrt(3)/2, 0], M3 at [-1/2, -sqrt(3)/2, 0].
        Operations:
          E  (id=0): identity
          C3 (id=1): cycles M1->M2->M3->M1
          C3^2 (id=2): cycles M1->M3->M2->M1
          C2_M1 (id=3): fixes M1, exchanges M2<->M3
          C2_M2 (id=4): fixes M2, exchanges M1<->M3
          C2_M3 (id=5): fixes M3, exchanges M1<->M2
        """
        c3 = self._c3_cart()
        c3_sq = c3 @ c3
        c2_m1 = self._c2_m1_cart()
        c2_m2 = self._c2_m2_cart()
        c2_m3 = self._c2_m3_cart()

        def sector_map(m1_tgt, m2_tgt, m3_tgt):
            return {"M1_valley": m1_tgt, "M2_valley": m2_tgt, "M3_valley": m3_tgt}

        def preserved(m1, m2, m3):
            return {"M1_valley": m1, "M2_valley": m2, "M3_valley": m3}

        return [
            {
                "operation_id": 0, "kind": "identity", "order": 1,
                "rotation_frac": np.eye(3, dtype=int),
                "translation_frac": np.zeros(3),
                "rotation_cart": np.eye(3),
                "translation_cart": np.zeros(3),
                "det": 1,
                "sector_mapping": sector_map("M1_valley", "M2_valley", "M3_valley"),
                "preserved": preserved(True, True, True),
            },
            {
                "operation_id": 1, "kind": "C3", "order": 3,
                "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
                "translation_frac": np.zeros(3),
                "rotation_cart": c3,
                "translation_cart": np.zeros(3),
                "det": 1,
                "sector_mapping": sector_map("M2_valley", "M3_valley", "M1_valley"),
                "preserved": preserved(False, False, False),
            },
            {
                "operation_id": 2, "kind": "C3^2", "order": 3,
                "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]) @ np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
                "translation_frac": np.zeros(3),
                "rotation_cart": c3_sq,
                "translation_cart": np.zeros(3),
                "det": 1,
                "sector_mapping": sector_map("M3_valley", "M1_valley", "M2_valley"),
                "preserved": preserved(False, False, False),
            },
            {
                "operation_id": 3, "kind": "C2", "order": 2,
                "rotation_frac": np.diag([1, -1, 1]),
                "translation_frac": np.zeros(3),
                "rotation_cart": c2_m1,
                "translation_cart": np.zeros(3),
                "det": 1,
                "sector_mapping": sector_map("M1_valley", "M3_valley", "M2_valley"),
                "preserved": preserved(True, False, False),
            },
            {
                "operation_id": 4, "kind": "C2", "order": 2,
                "rotation_frac": np.eye(3, dtype=int),  # simplified
                "translation_frac": np.zeros(3),
                "rotation_cart": c2_m2,
                "translation_cart": np.zeros(3),
                "det": 1,
                "sector_mapping": sector_map("M3_valley", "M2_valley", "M1_valley"),
                "preserved": preserved(False, True, False),
            },
            {
                "operation_id": 5, "kind": "C2", "order": 2,
                "rotation_frac": np.eye(3, dtype=int),  # simplified
                "translation_frac": np.zeros(3),
                "rotation_cart": c2_m3,
                "translation_cart": np.zeros(3),
                "det": 1,
                "sector_mapping": sector_map("M2_valley", "M1_valley", "M3_valley"),
                "preserved": preserved(False, False, True),
            },
        ]

    def test_three_valley_orbit_and_per_valley_stabilizers(self):
        """C3 cycles M1/M2/M3; each C2 fixes one valley and exchanges the other two.
        Each stabilizer is E + corresponding C2.  All-valley intersection is just E."""
        operations = self._three_valley_operations()
        symmetry_payload = {"detected_operations": operations}
        update_valley_little_group_inventory(
            symmetry_payload=symmetry_payload,
            kpoint_name="GM",
            k_frac=np.zeros(3),
            valley_names=["M1_valley", "M2_valley", "M3_valley"],
        )

        report = build_valley_preserving_subgroup_report(
            symmetry_payload=symmetry_payload,
            target_kpoints=["GM"],
        )

        # Valley orbits: one orbit with all three M valleys
        orbits = report["valley_orbits"]
        assert len(orbits) == 1
        assert set(orbits[0]["valleys"]) == {"M1_valley", "M2_valley", "M3_valley"}
        # C3 should appear as a coset/valley-orbit operation
        assert 1 in orbits[0]["coset_representative_operation_ids"]

        # Per-valley stabilizers
        stabilizers = report["valley_stabilizers"]
        # M1 stabilizer: E(0) + C2_M1(3)
        assert stabilizers["M1_valley"]["operation_ids"] == [0, 3]
        assert stabilizers["M1_valley"]["closure_status"] == "closed"
        # M2 stabilizer: E(0) + C2_M2(4)
        assert stabilizers["M2_valley"]["operation_ids"] == [0, 4]
        # M3 stabilizer: E(0) + C2_M3(5)
        assert stabilizers["M3_valley"]["operation_ids"] == [0, 5]

        # All-valley intersection (debug only): just identity
        intersection = report["all_valley_intersection"]
        assert intersection["allowed_operation_ids"] == [0]
        assert "debug" in intersection.get("interpretation", "").lower()

        # Per-valley by_kpoint
        gm = report["by_kpoint"]["GM"]
        assert gm["M1_valley"]["allowed_operation_ids"] == [0, 3]
        assert 1 in gm["M1_valley"]["valley_changing_operation_ids"]  # C3
        assert gm["M2_valley"]["allowed_operation_ids"] == [0, 4]
        assert gm["M3_valley"]["allowed_operation_ids"] == [0, 5]

    def test_operation_preserving_m1_allowed_for_m1_not_for_m2_m3(self):
        """C2_M1 preserves M1 and exchanges M2/M3.
        It is allowed for M1 single-valley representation but NOT for M2 or M3."""
        operations = self._three_valley_operations()
        symmetry_payload = {"detected_operations": operations}
        per_valley = update_valley_little_group_inventory(
            symmetry_payload=symmetry_payload,
            kpoint_name="GM",
            k_frac=np.zeros(3),
            valley_names=["M1_valley", "M2_valley", "M3_valley"],
        )

        # C2_M1 (id=3) in M1_valley inventory
        m1_rows = {row["operation_id"]: row for row in per_valley["M1_valley"]}
        assert m1_rows[3]["valley_preserving"] is True
        assert m1_rows[3]["allowed_for_single_valley_representation"] is True
        assert m1_rows[3]["mapped_valley"] == "M1_valley"

        # C2_M1 in M2_valley inventory
        m2_rows = {row["operation_id"]: row for row in per_valley["M2_valley"]}
        assert m2_rows[3]["valley_preserving"] is False
        assert m2_rows[3]["allowed_for_single_valley_representation"] is False
        assert "valley-changing" in m2_rows[3]["reason"]

        # C2_M1 in M3_valley inventory
        m3_rows = {row["operation_id"]: row for row in per_valley["M3_valley"]}
        assert m3_rows[3]["valley_preserving"] is False
        assert m3_rows[3]["allowed_for_single_valley_representation"] is False

    def test_symmetry_rows_contain_target_valley(self):
        """Symmetry eigenvalue rows must include target_valley field."""
        c2z = np.diag([-1.0, -1.0, 1.0])
        operations = [
            {
                "operation_id": 0, "candidate_rotation": True,
                "order": 2, "kind": "C2",
                "rotation_frac": np.diag([-1, -1, 1]),
                "rotation_cart": c2z,
                "translation_cart": np.zeros(3),
                "preserved": {"K_valley": True, "Kp_valley": True},
                "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
            },
        ]
        coefficients = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
        symmetry_payload = {"detected_operations": operations}

        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3), q_cart=np.zeros((1, 3)),
            coefficients=coefficients, symmetry_payload=symmetry_payload,
            basis_payload=None, representation_payload={},
            valley_names=["K_valley", "Kp_valley"],
        )

        # C2z preserves both valleys at GM → 1 row per valley
        assert len(rows) == 2
        target_valleys = {row["target_valley"] for row in rows}
        assert target_valleys == {"K_valley", "Kp_valley"}
        for row in rows:
            assert row["valley_preserving"] is True
            assert row["target_valley"] in ("K_valley", "Kp_valley")

    def test_representation_payload_is_keyed_per_target_valley(self):
        """A single operation can have different valley blocks for different target valleys.

        The representation payload must therefore be keyed by operation and target valley,
        otherwise the second valley overwrites the first.
        """
        operations = [
            {
                "operation_id": 0, "candidate_rotation": True,
                "order": 2, "kind": "C2",
                "rotation_frac": np.eye(3, dtype=int),
                "rotation_cart": np.eye(3),
                "translation_cart": np.zeros(3),
                "preserved": {"K_valley": True, "Kp_valley": True},
                "sector_mapping": {"K_valley": "K_valley", "Kp_valley": "Kp_valley"},
            },
        ]
        coefficients = np.array(
            [
                [[1.0 + 0.0j, 0.0 + 0.0j]],
                [[0.0 + 0.0j, 1.0 + 0.0j]],
            ],
            dtype=np.complex128,
        )
        basis_payload = {
            "valid_valley_subspace": True,
            "transform": np.eye(2, dtype=np.complex128),
            "assigned_valleys": np.array([b"K_valley", b"Kp_valley"]),
        }
        representation_payload: dict[str, object] = {}

        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3),
            q_cart=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            coefficients=coefficients,
            symmetry_payload={"detected_operations": operations},
            basis_payload=basis_payload,
            representation_payload=representation_payload,
            valley_names=["K_valley", "Kp_valley"],
        )

        assert {row["target_valley"] for row in rows} == {"K_valley", "Kp_valley"}
        kp_payload = representation_payload["GM"]
        assert set(kp_payload) == {
            "operation_0__valley_K_valley",
            "operation_0__valley_Kp_valley",
        }
        assert kp_payload["operation_0__valley_K_valley"]["target_valley"] == "K_valley"
        assert kp_payload["operation_0__valley_Kp_valley"]["target_valley"] == "Kp_valley"
        assert kp_payload["operation_0__valley_K_valley"]["D_valley"].shape == (1, 1)
        assert kp_payload["operation_0__valley_Kp_valley"]["D_valley"].shape == (1, 1)

    def test_non_clean_valley_basis_is_diagnostic_only(self):
        """Approximate valley bases can be used for diagnostics without becoming ready."""
        operations = [
            {
                "operation_id": 0, "candidate_rotation": True,
                "order": 2, "kind": "C2",
                "rotation_frac": np.eye(3, dtype=int),
                "rotation_cart": np.eye(3),
                "translation_cart": np.zeros(3),
                "preserved": {"K_valley": True},
                "sector_mapping": {"K_valley": "K_valley"},
            },
        ]
        coefficients = np.array(
            [
                [[1.0 + 0.0j, 0.0 + 0.0j]],
                [[0.0 + 0.0j, 1.0 + 0.0j]],
            ],
            dtype=np.complex128,
        )
        basis_payload = {
            "valid_valley_subspace": False,
            "transform": np.eye(2, dtype=np.complex128),
            "assigned_valleys": np.array([b"K_valley", b"K_valley"]),
        }
        representation_payload: dict[str, object] = {}

        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3),
            q_cart=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            coefficients=coefficients,
            symmetry_payload={"detected_operations": operations},
            basis_payload=basis_payload,
            representation_payload=representation_payload,
            valley_names=["K_valley"],
        )

        assert rows
        assert all(row["basis"] == "valley_adapted" for row in rows)
        assert all(row["topology_input_ready"] is False for row in rows)
        assert all(row["reason"] == "valley subspace not clean" for row in rows)
        assert representation_payload["GM"]["operation_0__valley_K_valley"]["D_valley"].shape == (2, 2)

    def test_valley_changing_operation_no_single_valley_eigenvalue(self):
        """C2x exchanges K and Kp. It must NOT produce eigenvalue rows for either valley."""
        c2x = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        operations = [
            {
                "operation_id": 0, "candidate_rotation": True,
                "order": 2, "kind": "C2",
                "rotation_frac": np.diag([1, -1, 1]),
                "rotation_cart": c2x,
                "translation_cart": np.zeros(3),
                "preserved": {"K_valley": False, "Kp_valley": False},
                "sector_mapping": {"K_valley": "Kp_valley", "Kp_valley": "K_valley"},
            },
        ]
        coefficients = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
        symmetry_payload = {"detected_operations": operations}

        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3), q_cart=np.zeros((1, 3)),
            coefficients=coefficients, symmetry_payload=symmetry_payload,
            basis_payload=None, representation_payload={},
            valley_names=["K_valley", "Kp_valley"],
        )

        # Operation is valley-changing for both valleys → no eigenvalue rows
        assert rows == []

    def test_block_leakage_reported_as_diagnostic_only(self):
        """A representation matrix with leakage out of the target valley block
        should be diagnostic-only."""
        from valleyscope.analysis.symmetry_eigenvalue_diagnostic import _select_valley_block

        n = 4
        # Two K_valley states (indices 0,1) and two Kp_valley states (indices 2,3)
        assigned = ["K_valley", "K_valley", "Kp_valley", "Kp_valley"]
        # D_valley with significant coupling between K_valley and Kp_valley blocks
        d_valley = np.eye(n, dtype=np.complex128)
        d_valley[0, 2] = 0.3  # leakage from K to Kp
        d_valley[2, 0] = 0.3  # leakage from Kp to K
        d_valley[1, 3] = 0.2
        d_valley[3, 1] = 0.2

        block, leakage = _select_valley_block(d_valley, assigned, "K_valley")

        assert block.shape == (2, 2)
        assert leakage > 0.3  # significant leakage
        # Block should still be extractable but leakage is non-zero
        np.testing.assert_allclose(np.diag(block), [1.0, 1.0])

        # Test with no leakage
        d_clean = np.eye(n, dtype=np.complex128)
        block_clean, leakage_clean = _select_valley_block(d_clean, assigned, "K_valley")
        assert block_clean.shape == (2, 2)
        assert leakage_clean == 0.0

    def test_block_leakage_makes_topology_input_not_ready(self):
        """When D_block_leakage_norm exceeds D_valley_offdiag_tol,
        topology_input_ready should be False."""
        from valleyscope.analysis.symmetry_eigenvalue_diagnostic import _topology_input_ready

        # All other gates pass
        ready = _topology_input_ready(
            rotation_ready=True,
            basis="valley_adapted",
            valid_valley_subspace=True,
            spinor_convention_verified=True,
            root_deviation=1e-10,
            d_valley_offdiag_norm=None,
            d_block_leakage_norm=0.5,
            root_deviation_tol=1e-6,
            d_valley_offdiag_tol=1e-4,
        )
        assert ready is False

        # With small leakage it should pass
        ready_small = _topology_input_ready(
            rotation_ready=True,
            basis="valley_adapted",
            valid_valley_subspace=True,
            spinor_convention_verified=True,
            root_deviation=1e-10,
            d_valley_offdiag_norm=None,
            d_block_leakage_norm=1e-8,
            root_deviation_tol=1e-6,
            d_valley_offdiag_tol=1e-4,
        )
        assert ready_small is True

    def test_operation_inclusion_independent_of_rotation_order(self):
        """V1.1: all proper little-group operations (order 2,3,4,6) enter
        the analysis regardless of rotation_order.  Non-candidate operations
        are not excluded."""
        c3_angle = 2.0 * np.pi / 3.0
        c3_cart = np.array([
            [np.cos(c3_angle), -np.sin(c3_angle), 0.0],
            [np.sin(c3_angle), np.cos(c3_angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        c2z = np.diag([-1.0, -1.0, 1.0])
        operations = [
            {
                "operation_id": 0, "candidate_rotation": True, "order": 3, "kind": "C3",
                "rotation_frac": np.eye(3, dtype=int),
                "rotation_cart": c3_cart, "translation_cart": np.zeros(3),
                "preserved": {"K": True},
                "sector_mapping": {"K": "K"},
            },
            {
                "operation_id": 1, "candidate_rotation": False, "order": 2, "kind": "C2",
                "rotation_frac": np.eye(3, dtype=int),
                "rotation_cart": c2z, "translation_cart": np.zeros(3),
                "preserved": {"K": True},
                "sector_mapping": {"K": "K"},
            },
        ]
        coefficients = np.array([[[1.0 + 0.0j]]], dtype=np.complex128)
        symmetry_payload = {"detected_operations": operations}

        rows = symmetry_eigenvalue_diagnostics_for_kpoint(
            kpoint_name="GM", k_frac=np.zeros(3), q_cart=np.zeros((1, 3)),
            coefficients=coefficients, symmetry_payload=symmetry_payload,
            basis_payload=None, representation_payload={},
            valley_names=["K"], generators_only=False,
        )

        # Both C3 (candidate) and C2 (non-candidate) are included
        op_ids = {row["operation_id"] for row in rows}
        assert op_ids == {0, 1}, f"Expected both C3 and C2, got {op_ids}"

    def test_not_degenerate_shown_in_valley_subspace_analysis(self):
        """not_degenerate status should not be squashed to n/a in summary."""
        from valleyscope.reports.summary_report import _short_valley_status, _subspace_basis_label

        assert _short_valley_status("not_degenerate") == "not_degenerate"
        assert _short_valley_status("single_band") == "n/a"

        label = _subspace_basis_label({
            "basis_status": "not_degenerate",
            "energy_span_meV": 5.2,
            "subspace_energy_tol_meV": 1.0,
        })
        assert "not_degenerate" in label
        assert "5.2" in label
        assert "1" in label
