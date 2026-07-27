import numpy as np
import pytest

from valleyscope.geometry.lattice import cart_rotation_from_fractional, cart_translation_from_fractional
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.symmetry.little_group import is_little_group_operation
from valleyscope.symmetry.operation_classifier import classify_operation, operation_order, rotation_axis_angle
from valleyscope.symmetry.plane_wave_action import (
    build_plane_wave_representation,
    build_reciprocal_grid_map,
    reciprocal_grid_identity,
    spin_rotation_matrix,
    validate_reciprocal_grid_permutation,
)
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


def test_spinor_symmetry_rows_require_scoped_evidence_for_readiness():
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
    assert "spinor_convention_verified" not in rows[0]
    assert rows[0]["diagnostic_only"] is True
    assert rows[0]["local_irrep_ready"] is False


def test_plane_wave_mapping_complete_tolerates_small_truncation_unitarity_error():
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
    assert rows[0]["plane_wave_mapping_complete"] is True


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


def test_reciprocal_grid_map_rejects_duplicate_grid_collision():
    q_cart = np.zeros((2, 3), dtype=float)

    result = build_reciprocal_grid_map(
        q_cart,
        np.eye(3),
    )
    validation = validate_reciprocal_grid_permutation(
        result.mapping,
        dimension=2,
    )

    assert result.mapping.tolist() == [0, 0]
    assert validation.status == "blocked"
    assert "target_index_collision" in validation.reason_codes
    assert "target_coverage_incomplete" in validation.reason_codes


@pytest.mark.parametrize(
    ("mapping", "reason"),
    [
        ([0, 0], "target_index_collision"),
        ([0, 2], "target_index_out_of_range"),
        ([0, -1], "source_coverage_incomplete"),
        ([0, 1.0], "mapping_index_malformed"),
        ([0, True], "mapping_index_malformed"),
    ],
)
def test_reciprocal_grid_permutation_validation_fails_closed(mapping, reason):
    validation = validate_reciprocal_grid_permutation(
        mapping,
        dimension=2,
    )

    assert validation.status == "blocked"
    assert reason in validation.reason_codes


def test_reciprocal_grid_identity_binds_order_and_values():
    q_cart = np.array(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]],
        dtype=float,
    )

    identity = reciprocal_grid_identity(q_cart)

    assert identity.startswith("sha256:")
    assert identity != reciprocal_grid_identity(q_cart[::-1])


def test_plane_wave_norm_residual_is_relative_finite_and_nonnegative():
    coefficients = np.array([[[1.0, 1.0j]]], dtype=np.complex128)
    q_cart = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )

    result = build_plane_wave_representation(
        coefficients,
        q_cart,
        np.zeros((3, 3)),
        np.zeros(3),
    )

    assert np.isfinite(result.relative_norm_preservation_residual)
    assert result.relative_norm_preservation_residual >= 0.0


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

    def test_all_little_group_operations_are_evaluated(self):
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
            basis_payload=None, representation_payload={},
        )
        assert {row["operation_id"] for row in rows} == {0, 1}

    def test_operation_inclusion_does_not_use_candidate_flags(self):
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
            basis_payload=None, representation_payload={},
        )
        op_ids = {row["operation_id"] for row in rows}
        assert 1 in op_ids

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
            basis_payload=None, representation_payload={},
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
        )

        assert sorted(row["phase_2pi"] for row in rows) == pytest.approx(
            [-1.0 / 6.0, 1.0 / 6.0]
        )
        assert all(not row["local_irrep_ready"] for row in rows)
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


def test_valley_preserving_subgroup_report_resolves_standard_subgroup_match():
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

    # Standard group match via per_valley_standard_matches.
    kv_match = report["per_valley_standard_matches"]["K_valley"]
    assert kv_match["standard_group_match_status"] == "matched"
    assert kv_match["standard_group_match"]["number"] == 143
    assert kv_match["standard_group_match"]["international_short"] == "P3"


def test_three_m_valley_c2_subgroups_report_per_valley_standard_matches():
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

    # Per-valley standard group matches for each M valley.
    m1_sm = report["per_valley_standard_matches"]["M1_valley"]
    assert m1_sm["standard_group_match_status"] == "matched"
    m2_sm = report["per_valley_standard_matches"]["M2_valley"]
    assert m2_sm["standard_group_match_status"] == "matched"


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

    gm_k = report["by_kpoint"]["GM"]["K_valley"]
    assert gm_k["closure_status"] == "not_closed"
    assert gm_k["missing_products"] == [
        {"left_operation_id": 1, "right_operation_id": 1}
    ]


class TestV11PerValleySubgroup:
    """V1.1 per-valley subgroup tests: three-valley orbit, per-valley gates,
    block-leakage diagnostic, and operation-inventory independence."""

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

    def test_three_valley_orbit_and_per_valley_preserving_subgroups(self):
        """C3 cycles M1/M2/M3; each C2 fixes one valley and exchanges the other two.
        Each subgroup is E + corresponding C2.  All-valley intersection is just E."""
        operations = self._three_valley_operations()
        symmetry_payload = {"detected_operations": operations}
        update_valley_preserving_operation_inventory(
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
        assert orbits[0]["valleys"] == [
            "M1_valley",
            "M2_valley",
            "M3_valley",
        ]
        # C3 should appear as a coset/valley-orbit operation
        assert 1 in orbits[0]["valley_permuting_operation_ids"]
        assert 1 in orbits[0]["coset_representative_operation_ids"]  # legacy alias

        # Per-valley subgroups
        subgroups = report["valley_preserving_subgroups"]
        # M1 subgroup: E(0) + C2_M1(3)
        assert subgroups["M1_valley"]["operation_ids"] == [0, 3]
        assert subgroups["M1_valley"]["closure_status"] == "closed"
        # M2 subgroup: E(0) + C2_M2(4)
        assert subgroups["M2_valley"]["operation_ids"] == [0, 4]
        # M3 subgroup: E(0) + C2_M3(5)
        assert subgroups["M3_valley"]["operation_ids"] == [0, 5]

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
        It is allowed for M1 valley-preserving representation but NOT for M2 or M3."""
        operations = self._three_valley_operations()
        symmetry_payload = {"detected_operations": operations}
        per_valley = update_valley_preserving_operation_inventory(
            symmetry_payload=symmetry_payload,
            kpoint_name="GM",
            k_frac=np.zeros(3),
            valley_names=["M1_valley", "M2_valley", "M3_valley"],
        )

        # C2_M1 (id=3) in M1_valley inventory
        m1_rows = {row["operation_id"]: row for row in per_valley["M1_valley"]}
        assert m1_rows[3]["valley_preserving"] is True
        assert m1_rows[3]["allowed_for_valley_preserving_representation"] is True
        assert m1_rows[3]["mapped_valley"] == "M1_valley"

        # C2_M1 in M2_valley inventory
        m2_rows = {row["operation_id"]: row for row in per_valley["M2_valley"]}
        assert m2_rows[3]["valley_preserving"] is False
        assert m2_rows[3]["allowed_for_valley_preserving_representation"] is False
        assert "valley-changing" in m2_rows[3]["reason"]

        # C2_M1 in M3_valley inventory
        m3_rows = {row["operation_id"]: row for row in per_valley["M3_valley"]}
        assert m3_rows[3]["valley_preserving"] is False
        assert m3_rows[3]["allowed_for_valley_preserving_representation"] is False

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
        assert all(row["local_irrep_ready"] is False for row in rows)
        assert all(row["reason"] == "valley subspace not clean" for row in rows)
        assert representation_payload["GM"]["operation_0__valley_K_valley"]["D_valley"].shape == (2, 2)

    def test_valley_changing_operation_no_valley_preserving_eigenvalue(self):
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

    def test_operation_inclusion_uses_exact_little_group_inventory(self):
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
            valley_names=["K"],
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
