from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from valleyscope.io.spinor_source_basis import SpinorSourceBasisCertificate
from valleyscope.io.wavefunction_convention import canonical_identity
from valleyscope.geometry.lattice import (
    cart_rotation_from_fractional,
    cart_translation_from_fractional,
)
from valleyscope.symmetry.double_space_group_lift import (
    axial_spin_rotation,
    build_double_space_group_lift_certificate,
    seitz_product,
    spin_lift_from_orthogonal,
    validate_double_space_group_lift_record,
)
from valleyscope.symmetry.plane_wave_action import apply_plane_wave_action


def _source_record() -> dict[str, object]:
    return SpinorSourceBasisCertificate(
        extracted_wavefunction_payload_identity="sha256:" + "1" * 64,
        nspinor=2,
        parser_identity="valleyscope_h5_reader_v1",
        hdf5_layout_identity="valleyscope_wavefunction_h5_layout_v1",
        extractor_provenance=None,
    ).to_record()


def _op(
    operation_id,
    rotation,
    translation=(0.0, 0.0, 0.0),
    *,
    direct_lattice=None,
):
    rotation = np.asarray(rotation, dtype=int)
    translation = np.asarray(translation, dtype=float)
    direct = (
        np.eye(3)
        if direct_lattice is None
        else np.asarray(direct_lattice, dtype=float)
    )
    return {
        "operation_id": operation_id,
        "rotation_frac": rotation,
        "translation_frac": translation,
        "rotation_cart": cart_rotation_from_fractional(rotation, direct),
        "translation_cart": cart_translation_from_fractional(
            translation, direct
        ),
    }


def _complex_matrix_record(matrix):
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in np.asarray(matrix, dtype=np.complex128)
    ]


def _identities(
    operations,
    *,
    direct_lattice=None,
) -> tuple[dict[str, object], dict[str, object]]:
    direct = (
        np.eye(3)
        if direct_lattice is None
        else np.asarray(direct_lattice, dtype=float)
    )
    ordered = sorted(
        (
            operation
            for operation in operations
            if isinstance(operation["operation_id"], int)
            and not isinstance(operation["operation_id"], bool)
        ),
        key=lambda operation: operation["operation_id"],
    )
    source_table = {
        "schema_version": "1.0.0",
        "provider": "irreptables",
        "data_source": "irreptables.StandardIrrepTable",
        "space_group_number": 1,
        "spinor": True,
        "operations": [
            {
                "table_index": table_index,
                "rotation_frac": np.asarray(
                    operation["rotation_frac"], dtype=int
                ).tolist(),
                "translation_frac": np.mod(
                    np.asarray(operation["translation_frac"], dtype=float),
                    1.0,
                ).tolist(),
                "spin_rotation": _complex_matrix_record(
                    spin_lift_from_orthogonal(
                        cart_rotation_from_fractional(
                            np.asarray(operation["rotation_frac"], dtype=int),
                            direct,
                        )
                    )
                ),
            }
            for table_index, operation in enumerate(ordered)
        ],
    }
    standard_setting = {
        "schema_version": "1.0.0",
        "parent_to_standard_direct_transform": np.eye(3).tolist(),
        "origin_shift_fractional": [0.0, 0.0, 0.0],
        "parent_to_standard_operation_map": {
            str(operation["operation_id"]): table_index
            for table_index, operation in enumerate(ordered)
        },
    }
    return source_table, standard_setting


def _build(operations, *, direct_lattice=None):
    direct = (
        np.eye(3)
        if direct_lattice is None
        else np.asarray(direct_lattice, dtype=float)
    )
    source_table, setting = _identities(
        operations,
        direct_lattice=direct,
    )
    return build_double_space_group_lift_certificate(
        _source_record(),
        operations,
        source_table_identity=source_table,
        standard_setting_identity=setting,
        direct_lattice_cart=(
            direct
        ),
    )


def test_noncommuting_generators_have_group_consistent_cocycle_and_central_minus_e():
    eye = np.eye(3, dtype=int)
    c2x = np.diag([1, -1, -1])
    c2y = np.diag([-1, 1, -1])
    c2z = np.diag([-1, -1, 1])
    certificate = _build(
        [
            _op(10, eye),
            _op(20, c2x),
            _op(30, c2y),
            _op(40, c2z),
        ]
    )
    record = certificate.to_record()

    assert record["status"] == "passed"
    assert record["operation_ids"] == [10, 20, 30, 40]
    assert record["central_element"]["label"] == "-E"
    assert record["central_element"]["matrix"] == [
        [[-1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [-1.0, 0.0]],
    ]
    pair_xy = record["pairwise_products"]["20,30"]
    pair_yx = record["pairwise_products"]["30,20"]
    assert pair_xy["product_operation_id"] == 40
    assert pair_yx["product_operation_id"] == 40
    assert pair_xy["cocycle_sign"] == -pair_yx["cocycle_sign"]
    assert pair_xy["spin_residual"] < 1e-12
    assert pair_yx["spin_residual"] < 1e-12
    assert validate_double_space_group_lift_record(
        record,
        source_basis_record=_source_record(),
        source_table_identity=_identities(
            [
                _op(10, eye),
                _op(20, c2x),
                _op(30, c2y),
                _op(40, c2z),
            ]
        )[0],
        standard_setting_identity=_identities(
            [
                _op(10, eye),
                _op(20, c2x),
                _op(30, c2y),
                _op(40, c2z),
            ]
        )[1],
        direct_lattice_cart=np.eye(3),
        expected_operations=[
            _op(10, eye),
            _op(20, c2x),
            _op(30, c2y),
            _op(40, c2z),
        ],
    ).status == "passed"


@pytest.mark.parametrize(
    ("rotation", "expected_axial"),
    [
        (-np.eye(3, dtype=int), np.eye(3)),
        (np.diag([1, 1, -1]), np.diag([-1, -1, 1])),
        (
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]], dtype=int),
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float),
        ),
    ],
)
def test_improper_operations_use_axial_o3_action(rotation, expected_axial):
    axial = axial_spin_rotation(rotation)
    lift = spin_lift_from_orthogonal(rotation)

    assert np.allclose(axial, expected_axial, atol=1e-12)
    assert np.allclose(lift.conj().T @ lift, np.eye(2), atol=1e-12)
    assert np.isclose(np.linalg.det(lift), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    ("generator_rotation", "generator_translation", "expected_lattice_translation"),
    [
        (
            np.diag([1, 1, -1]),
            (0.5, 0.0, 0.0),
            [1, 0, 0],
        ),
        (
            np.diag([-1, -1, 1]),
            (0.0, 0.0, 0.5),
            [0, 0, 1],
        ),
    ],
)
def test_glide_and_screw_square_record_lattice_translation_and_phase_composition(
    generator_rotation,
    generator_translation,
    expected_lattice_translation,
):
    certificate = _build(
        [
            _op(3, np.eye(3, dtype=int)),
            _op(11, generator_rotation, generator_translation),
        ]
    )
    pair = certificate.to_record()["pairwise_products"]["11,11"]

    assert pair["product_operation_id"] == 3
    assert pair["lattice_translation_frac"] == expected_lattice_translation
    assert pair["reciprocal_permutation_composition_passed"] is True
    assert pair["bloch_phase_composition_residual"] < 1e-12
    if expected_lattice_translation == [0, 0, 1]:
        sample = next(
            item
            for item in pair["factor_system_phase_samples"]
            if item["k_frac"] == [0.0, 0.0, 0.25]
        )
        assert np.allclose(sample["phase"], [0.0, -1.0], atol=1e-12)


def test_seitz_product_uses_active_left_after_right_order():
    c2z = np.diag([-1, -1, 1])
    mirror_x = np.diag([-1, 1, 1])
    rotation, translation = seitz_product(
        c2z,
        np.array([0.25, 0.0, 0.0]),
        mirror_x,
        np.array([0.0, 0.5, 0.0]),
    )

    assert np.array_equal(rotation, c2z @ mirror_x)
    assert np.allclose(
        translation,
        np.array([0.25, 0.0, 0.0]) + c2z @ np.array([0.0, 0.5, 0.0]),
    )


def test_reciprocal_grid_permutation_and_phase_compose_before_projection():
    q_cart = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    coefficients = np.array(
        [
            [
                [1.0, 2.0j, -0.5, 0.25j],
                [0.5j, -1.0, 0.75j, 2.0],
            ]
        ],
        dtype=np.complex128,
    )
    left_rotation = np.diag([1.0, -1.0, -1.0])
    right_rotation = np.diag([-1.0, 1.0, -1.0])
    left_translation = np.array([0.25, 0.0, 0.0])
    right_translation = np.array([0.0, 0.5, 0.0])
    left_spin = spin_lift_from_orthogonal(left_rotation)
    right_spin = spin_lift_from_orthogonal(right_rotation)

    right_action = apply_plane_wave_action(
        coefficients,
        q_cart,
        right_rotation,
        right_translation,
        spin_rotation=right_spin,
    )
    composed_action = apply_plane_wave_action(
        right_action.transformed_coefficients,
        q_cart,
        left_rotation,
        left_translation,
        spin_rotation=left_spin,
    )
    product_rotation, product_translation = seitz_product(
        left_rotation,
        left_translation,
        right_rotation,
        right_translation,
    )
    product_action = apply_plane_wave_action(
        coefficients,
        q_cart,
        product_rotation,
        product_translation,
        spin_rotation=left_spin @ right_spin,
    )

    assert right_action.mapping_miss_count == 0
    assert composed_action.mapping_miss_count == 0
    assert product_action.mapping_miss_count == 0
    assert np.allclose(
        composed_action.transformed_coefficients,
        product_action.transformed_coefficients,
        atol=1e-12,
    )


def test_noncontiguous_operation_ids_are_opaque_and_must_be_exact_integers():
    inversion = -np.eye(3, dtype=int)
    record = _build(
        [_op(0, np.eye(3, dtype=int)), _op(4, inversion)]
    ).to_record()
    assert record["status"] == "passed"
    assert record["operation_ids"] == [0, 4]

    for malformed in (True, 4.0, "4"):
        malformed_record = _build(
            [_op(0, np.eye(3, dtype=int)), _op(malformed, inversion)]
        ).to_record()
        assert malformed_record["status"] == "blocked"
        assert "operation_id_malformed" in malformed_record["reason_codes"]


def test_fractional_and_cartesian_operations_are_bound_by_direct_lattice():
    direct = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    c3_squared = c3 @ c3
    operations = [
        _op(1, np.eye(3, dtype=int), direct_lattice=direct),
        _op(7, c3, direct_lattice=direct),
        _op(12, c3_squared, direct_lattice=direct),
    ]
    assert _build(operations, direct_lattice=direct).to_record()["status"] == "passed"

    inconsistent = deepcopy(operations)
    inconsistent[1]["rotation_cart"] = np.eye(3)
    record = _build(inconsistent, direct_lattice=direct).to_record()
    assert record["status"] == "blocked"
    assert "cartesian_affine_operation_mismatch" in record["reason_codes"]


def test_one_common_spin_basis_transform_is_derived_for_all_operations():
    direct = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    c3 = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=int)
    operations = [
        _op(1, np.eye(3, dtype=int), direct_lattice=direct),
        _op(7, c3, direct_lattice=direct),
        _op(12, c3 @ c3, direct_lattice=direct),
    ]
    source_table, setting = _identities(
        operations, direct_lattice=direct
    )
    common_basis = 1.0j * np.array(
        [[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128
    )
    for source_operation in source_table["operations"]:
        spin = np.asarray(
            [
                [complex(*value) for value in row]
                for row in source_operation["spin_rotation"]
            ],
            dtype=np.complex128,
        )
        source_operation["spin_rotation"] = _complex_matrix_record(
            common_basis @ spin @ common_basis.conj().T
        )

    record = build_double_space_group_lift_certificate(
        _source_record(),
        operations,
        source_table_identity=source_table,
        standard_setting_identity=setting,
        direct_lattice_cart=direct,
    ).to_record()

    assert record["status"] == "passed"
    assert record["source_table_identity"]["status"] == "passed"
    assert max(
        row["residual"]
        for row in record["source_table_identity"]["spin_mapping_rows"]
    ) < 1e-12
    assert record["source_table_identity"][
        "common_spin_basis_transform"
    ] != _complex_matrix_record(np.eye(2))


def test_failed_source_table_or_standard_setting_evidence_blocks_lift():
    operations = [_op(0, np.eye(3, dtype=int))]
    source_table, setting = _identities(operations)
    source_table["operations"][0]["spin_rotation"] = []
    setting["parent_to_standard_operation_map"] = {}

    record = build_double_space_group_lift_certificate(
        _source_record(),
        operations,
        source_table_identity=source_table,
        standard_setting_identity=setting,
        direct_lattice_cart=np.eye(3),
    ).to_record()

    assert record["status"] == "blocked"
    assert "source_table_convention_not_validated" in record["reason_codes"]
    assert "standard_setting_not_validated" in record["reason_codes"]


def test_positive_validation_statuses_are_not_accepted_as_input_evidence():
    operations = [_op(0, np.eye(3, dtype=int))]
    source_table, setting = _identities(operations)
    source_table["common_spin_basis_status"] = "passed"
    setting.update(
        {
            "validation_status": "validated",
            "operation_mapping_status": "operation_basis_verification_passed",
            "affine_validation_status": "passed",
        }
    )

    record = build_double_space_group_lift_certificate(
        _source_record(),
        operations,
        source_table_identity=source_table,
        standard_setting_identity=setting,
        direct_lattice_cart=np.eye(3),
    ).to_record()

    assert record["status"] == "blocked"
    assert record["source_table_identity"]["status"] == "blocked"
    assert record["standard_setting_identity"]["status"] == "blocked"
    assert "source_table_convention_not_validated" in record["reason_codes"]
    assert "standard_setting_not_validated" in record["reason_codes"]


def test_incomplete_operation_inventory_fails_closed():
    c4z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    record = _build(
        [_op(0, np.eye(3, dtype=int)), _op(4, c4z)]
    ).to_record()

    assert record["status"] == "blocked"
    assert "operation_inventory_not_closed" in record["reason_codes"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda record: record["pairwise_products"]["20,30"].__setitem__(
                "cocycle_sign", 0
            ),
            "certificate_identity_mismatch",
        ),
        (
            lambda record: record.__setitem__("status", "blocked"),
            "derived_status_mismatch",
        ),
        (
            lambda record: record.__setitem__(
                "source_basis_certificate_identity", "sha256:" + "0" * 64
            ),
            "source_basis_identity_mismatch",
        ),
        (
            lambda record: record.__setitem__("operation_ids", [0, 1, 2, 3]),
            "operation_inventory_identity_mismatch",
        ),
    ],
)
def test_lift_certificate_rejects_tampered_records(mutate, reason):
    operations = [
        _op(10, np.eye(3, dtype=int)),
        _op(20, np.diag([1, -1, -1])),
        _op(30, np.diag([-1, 1, -1])),
        _op(40, np.diag([-1, -1, 1])),
    ]
    record = _build(operations).to_record()
    tampered = deepcopy(record)
    mutate(tampered)

    validation = validate_double_space_group_lift_record(
        tampered,
        source_basis_record=_source_record(),
        source_table_identity=_identities(operations)[0],
        standard_setting_identity=_identities(operations)[1],
        direct_lattice_cart=np.eye(3),
        expected_operations=operations,
    )

    assert validation.status == "blocked"
    assert reason in validation.reason_codes


def test_validator_recomputes_physical_evidence_and_rejects_self_signed_forgery():
    operations = [
        _op(10, np.eye(3, dtype=int)),
        _op(20, np.diag([1, -1, -1])),
        _op(30, np.diag([-1, 1, -1])),
        _op(40, np.diag([-1, -1, 1])),
    ]
    record = _build(operations).to_record()
    forged = deepcopy(record)
    forged["pairwise_products"] = {}
    forged["central_element"]["matrix"] = [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0]],
    ]
    forged["reciprocal_action_convention"] = "forged"
    forged["max_spin_residual"] = 999.0
    content = {
        key: value
        for key, value in forged.items()
        if key not in {"status", "reason_codes", "certificate_identity"}
    }
    forged["certificate_identity"] = canonical_identity(content)
    source_table, setting = _identities(operations)

    validation = validate_double_space_group_lift_record(
        forged,
        source_basis_record=_source_record(),
        source_table_identity=source_table,
        standard_setting_identity=setting,
        direct_lattice_cart=np.eye(3),
        expected_operations=operations,
        required_operation_ids=[10, 20],
    )

    assert validation.status == "blocked"
    assert "recomputed_certificate_mismatch" in validation.reason_codes


def test_validator_binds_opaque_ids_to_current_affine_operation_inventory():
    current_operations = [
        _op(10, np.eye(3, dtype=int)),
        _op(20, np.diag([-1, -1, 1])),
    ]
    substituted_operations = [
        _op(10, np.eye(3, dtype=int)),
        _op(20, np.diag([1, -1, -1])),
    ]
    source_table, setting = _identities(current_operations)
    substituted_source_table, substituted_setting = _identities(
        substituted_operations
    )
    substituted_record = build_double_space_group_lift_certificate(
        _source_record(),
        substituted_operations,
        source_table_identity=substituted_source_table,
        standard_setting_identity=substituted_setting,
        direct_lattice_cart=np.eye(3),
    ).to_record()
    assert substituted_record["status"] == "passed"

    validation = validate_double_space_group_lift_record(
        substituted_record,
        source_basis_record=_source_record(),
        source_table_identity=source_table,
        standard_setting_identity=setting,
        direct_lattice_cart=np.eye(3),
        expected_operations=current_operations,
    )

    assert validation.status == "blocked"
    assert "recomputed_certificate_mismatch" in validation.reason_codes
