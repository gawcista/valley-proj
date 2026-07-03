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
    build_valley_preserving_subgroup_report,
    update_valley_preserving_operation_inventory,
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


@pytest.mark.parametrize("angle, expected, check_unitary", [
    pytest.param(np.pi, np.diag([-1.0j, 1.0j]), False, id="c2_matches_su2"),
    pytest.param(2.0 * np.pi, -np.eye(2), False, id="two_pi_minus_identity"),
    pytest.param(2.0 * np.pi / 3.0, None, True, id="c3_unitary"),
])
def test_spin_rotation_matrix(angle, expected, check_unitary):
    rot = spin_rotation_matrix(axis=np.array([0.0, 0.0, 1.0]), angle=angle)
    if check_unitary:
        assert np.allclose(rot.conj().T @ rot, np.eye(2), atol=1e-12)
    else:
        np.testing.assert_allclose(rot, expected, atol=1e-12)


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


def test_valley_preserving_operation_inventory_marks_allowed_and_valley_exchanging_operations():
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

    per_valley = update_valley_preserving_operation_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="KM",
        k_frac=np.array([1.0 / 3.0, 0.0, 0.0]),
        valley_names=["K_valley", "Kp_valley"],
    )

    # Per-valley: K_valley inventory
    kv_rows = per_valley["K_valley"]
    assert [row["operation_id"] for row in kv_rows] == [0, 1, 2]
    assert kv_rows[0]["allowed_for_valley_preserving_representation"] is True
    assert kv_rows[1]["little_group_passed"] is True
    assert kv_rows[1]["valley_preserving"] is False
    assert "valley-changing" in kv_rows[1]["reason"]
    assert kv_rows[2]["little_group_passed"] is False
    assert kv_rows[2]["reason"] == "not in little group"
    assert operations[1]["rejection_reason_by_kpoint"]["KM"] == "valley-exchanging"

    # Flat inventory still stored for backward compat
    flat = symmetry_payload["hsp_little_group_inventory"]["KM"]
    assert [row["operation_id"] for row in flat] == [0, 1, 2]
    assert flat[0]["allowed_for_valley_preserving_representation"] is True
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
    update_valley_preserving_operation_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="GM",
        k_frac=np.zeros(3),
        valley_names=["K_valley"],
    )

    report = build_valley_preserving_subgroup_report(
        symmetry_payload=symmetry_payload,
        target_kpoints=["GM"],
    )

    assert report["status"] == "per_valley_preserving_subgroups_computed"
    # Per-valley subgroup for K_valley
    assert report["valley_preserving_subgroups"]["K_valley"]["operation_ids"] == [0, 1, 2]
    # All-valley intersection (debug)
    assert report["all_valley_intersection"]["allowed_operation_ids"] == [0, 1, 2]
    # Per-valley by_kpoint
    gm_k = report["by_kpoint"]["GM"]["K_valley"]
    assert gm_k["allowed_operation_ids"] == [0, 1, 2]
    assert gm_k["valley_changing_operation_ids"] == [3]
    assert gm_k["closure_status"] == "closed"
    assert gm_k["missing_products"] == []
    assert "K_valley" in report["per_valley_standard_matches"]


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
    update_valley_preserving_operation_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="GM",
        k_frac=np.zeros(3),
        valley_names=["K_valley", "Kp_valley"],
    )

    report = build_valley_preserving_subgroup_report(
        symmetry_payload=symmetry_payload,
        target_kpoints=["GM"],
    )

    assert report["status"] == "per_valley_preserving_subgroups_computed"
    # All-valley intersection (debug)
    assert report["all_valley_intersection"]["operation_set_label"] == "all_valley_intersection"
    assert report["all_valley_intersection"]["allowed_operation_ids"] == [0, 1, 2]
    assert report["all_valley_intersection"]["valley_exchanging_operation_ids"] == [3]
    assert report["all_valley_intersection"]["closure_status"] == "closed"
    # Per-valley subgroups both match P3 since both valleys are preserved by C3
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
    update_valley_preserving_operation_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="KM",
        k_frac=np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
        valley_names=["K_valley", "Kp_valley"],
    )

    report = build_valley_preserving_subgroup_report(
        symmetry_payload=symmetry_payload,
        target_kpoints=["KM"],
    )

    # Standard group match still works via per_valley_standard_matches.
    kv_match = report["per_valley_standard_matches"]["K_valley"]
    assert kv_match["standard_group_match_status"] == "matched"
    assert kv_match["standard_group_match"]["number"] == 143
    assert kv_match["standard_group_match"]["international_short"] == "P3"


def test_three_m_valley_c2_subgroups_map_operations_but_report_kpoint_ambiguity():
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    c3_square = c3 @ c3
    operations = [
        {
            "operation_id": 0,
            "kind": "identity",
            "order": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "preserved": {"M1_valley": True, "M2_valley": True, "M3_valley": True},
            "sector_mapping": {
                "M1_valley": "M1_valley",
                "M2_valley": "M2_valley",
                "M3_valley": "M3_valley",
            },
        },
        {
            "operation_id": 1,
            "kind": "C3",
            "order": 3,
            "rotation_frac": c3,
            "translation_frac": np.zeros(3),
            "preserved": {"M1_valley": False, "M2_valley": False, "M3_valley": False},
            "sector_mapping": {
                "M1_valley": "M3_valley",
                "M2_valley": "M1_valley",
                "M3_valley": "M2_valley",
            },
        },
        {
            "operation_id": 2,
            "kind": "C3^2",
            "order": 3,
            "rotation_frac": c3_square,
            "translation_frac": np.zeros(3),
            "preserved": {"M1_valley": False, "M2_valley": False, "M3_valley": False},
            "sector_mapping": {
                "M1_valley": "M2_valley",
                "M2_valley": "M3_valley",
                "M3_valley": "M1_valley",
            },
        },
        {
            "operation_id": 4,
            "kind": "C2",
            "order": 2,
            "rotation_frac": np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]], dtype=int),
            "translation_frac": np.zeros(3),
            "preserved": {"M1_valley": True, "M2_valley": False, "M3_valley": False},
            "sector_mapping": {
                "M1_valley": "M1_valley",
                "M2_valley": "M3_valley",
                "M3_valley": "M2_valley",
            },
        },
    ]
    symmetry_payload = {
        "detected_operations": operations,
        "lattice_direct_cart": np.array(
            [
                [1.0, 0.0, 0.0],
                [-0.5, np.sqrt(3.0) / 2.0, 0.0],
                [0.0, 0.0, 20.0],
            ]
        ),
        "spinor_wavefunction": True,
    }
    update_valley_preserving_operation_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="GammaM",
        k_frac=np.array([0.0, 0.0, 0.0]),
        valley_names=["M1_valley", "M2_valley", "M3_valley"],
    )
    update_valley_preserving_operation_inventory(
        symmetry_payload=symmetry_payload,
        kpoint_name="MM",
        k_frac=np.array([0.0, 0.5, 0.0]),
        valley_names=["M1_valley", "M2_valley", "M3_valley"],
    )

    report = build_valley_preserving_subgroup_report(
        symmetry_payload=symmetry_payload,
        target_kpoints=["GammaM", "MM"],
    )

    # Each M valley still resolves its own C2 subgroup with distinct
    # per-valley standard group matches.
    m1_sm = report["per_valley_standard_matches"]["M1_valley"]
    assert m1_sm["standard_group_match_status"] == "matched"
    m2_sm = report["per_valley_standard_matches"]["M2_valley"]
    assert m2_sm["standard_group_match_status"] == "matched"
