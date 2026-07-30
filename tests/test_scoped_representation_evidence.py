from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from valleyscope.analysis.representation_readiness import (
    compose_representation_readiness,
)
from valleyscope.analysis.scoped_representation_evidence import (
    build_directed_valley_sewing_evidence,
    build_scoped_representation_evidence,
    validate_directed_valley_sewing_evidence_record,
    validate_scoped_representation_evidence_record,
)
from valleyscope.analysis.target_frame import build_target_frame
from valleyscope.geometry.lattice import (
    cart_rotation_from_fractional,
    cart_translation_from_fractional,
)
from valleyscope.io.spinor_source_basis import SpinorSourceBasisCertificate
from valleyscope.symmetry.double_space_group_lift import (
    build_double_space_group_lift_certificate,
    spin_lift_from_orthogonal,
)
from valleyscope.symmetry.plane_wave_action import (
    RECIPROCAL_GRID_ACTION_CONVENTION,
    reciprocal_grid_identity,
)


def _source_record() -> dict[str, object]:
    return SpinorSourceBasisCertificate(
        extracted_wavefunction_payload_identity="sha256:" + "1" * 64,
        nspinor=2,
        parser_identity="valleyscope_h5_reader_v1",
        hdf5_layout_identity="valleyscope_wavefunction_h5_layout_v1",
        extractor_provenance=None,
    ).to_record()


def _operation(operation_id: int, rotation: np.ndarray) -> dict[str, object]:
    rotation = np.asarray(rotation, dtype=int)
    return {
        "operation_id": operation_id,
        "rotation_frac": rotation,
        "translation_frac": np.zeros(3),
        "rotation_cart": cart_rotation_from_fractional(rotation, np.eye(3)),
        "translation_cart": cart_translation_from_fractional(
            np.zeros(3), np.eye(3)
        ),
    }


def _complex_record(matrix: np.ndarray) -> list[list[list[float]]]:
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in np.asarray(matrix, dtype=np.complex128)
    ]


def _lift_inputs() -> dict[str, object]:
    operations = [
        _operation(2, np.eye(3, dtype=int)),
        _operation(5, np.diag([1, -1, -1])),
    ]
    source_table = {
        "schema_version": "1.0.0",
        "provider": "irreptables",
        "data_source": "irreptables.StandardIrrepTable",
        "space_group_number": 1,
        "spinor": True,
        "operations": [
            {
                "table_index": index,
                "rotation_frac": operation["rotation_frac"].tolist(),
                "translation_frac": [0.0, 0.0, 0.0],
                "spin_rotation": _complex_record(
                    spin_lift_from_orthogonal(operation["rotation_cart"])
                ),
            }
            for index, operation in enumerate(operations)
        ],
    }
    setting = {
        "schema_version": "1.0.0",
        "parent_to_standard_direct_transform": np.eye(3).tolist(),
        "origin_shift_fractional": [0.0, 0.0, 0.0],
        "parent_to_standard_operation_map": {"2": 0, "5": 1},
    }
    return {
        "expected_operations": operations,
        "source_table_identity": source_table,
        "standard_setting_identity": setting,
        "direct_lattice_cart": np.eye(3),
    }


def _lift_record() -> dict[str, object]:
    source = _source_record()
    inputs = _lift_inputs()
    return build_double_space_group_lift_certificate(
        source,
        inputs["expected_operations"],
        source_table_identity=inputs["source_table_identity"],
        standard_setting_identity=inputs["standard_setting_identity"],
        direct_lattice_cart=inputs["direct_lattice_cart"],
    ).to_record()


def _raw_inputs(
    *,
    scope_kind: str,
    required_operation_ids: tuple[int, ...],
    bad_valley_changing_operation: bool = False,
) -> dict[str, object]:
    q_cart = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=float,
    )
    grid_identity = reciprocal_grid_identity(q_cart)
    rotations = {
        2: np.eye(3, dtype=float),
        5: np.diag([1.0, -1.0, -1.0]),
    }

    def plane_wave_row(operation_id: int) -> dict[str, object]:
        return {
            "action_convention": RECIPROCAL_GRID_ACTION_CONVENTION,
            "reciprocal_grid_identity": grid_identity,
            "reciprocal_grid_dimension": 2,
            "q_cart": q_cart,
            "rotation_cart": rotations[operation_id],
            "mapping_tolerance": 1.0e-6,
            "source_to_target_map": [0, 1],
            "mapping_miss_count": 0,
            "relative_norm_residual": 0.0,
            "norm_preservation_residual": 0.0,
        }

    d_changing = np.array(
        [[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128
    )
    if bad_valley_changing_operation:
        d_changing[0, 1] = 0.0
    return {
        "source_basis_record": _source_record(),
        "lift_record": _lift_record(),
        "lift_validation_inputs": _lift_inputs(),
        "extracted_wavefunction_payload_identity": "sha256:" + "1" * 64,
        "kpoint_label": "K0",
        "kpoint_frac": np.zeros(3),
        "scope_kind": scope_kind,
        "source_valleys": ("v0",),
        "valley_orbit": ("v0", "v1"),
        "required_operation_ids": required_operation_ids,
        "representations": {
            2: np.eye(2, dtype=np.complex128),
            5: d_changing,
        },
        "plane_wave_evidence": {
            2: plane_wave_row(2),
            5: plane_wave_row(5),
        },
        "target_coefficients": np.eye(
            2, dtype=np.complex128
        ).reshape(2, 1, 2),
        "projectors": {
            "v0": np.diag([1.0, 0.0]),
            "v1": np.diag([0.0, 1.0]),
        },
        "valley_bases": {
            "v0": np.array([[1.0], [0.0]], dtype=np.complex128),
            "v1": np.array([[0.0], [1.0]], dtype=np.complex128),
        },
        "valley_mappings": {
            2: {"v0": "v0", "v1": "v1"},
            5: {"v0": "v1", "v1": "v0"},
        },
    }


def _build(**kwargs) -> tuple[dict[str, object], dict[str, object]]:
    inputs = _raw_inputs(**kwargs)
    return build_scoped_representation_evidence(**inputs).to_record(), inputs


def _directed_sewing_inputs() -> dict[str, object]:
    source = _source_record()
    operations = [
        {
            **_operation(11, np.eye(3, dtype=int)),
            "sector_mapping": {"v0": "v0", "v1": "v1"},
        },
        {
            **_operation(47, -np.eye(3, dtype=int)),
            "sector_mapping": {"v0": "v1", "v1": "v0"},
        },
    ]
    source_table = {
        "schema_version": "1.0.0",
        "provider": "irreptables",
        "data_source": "irreptables.StandardIrrepTable",
        "space_group_number": 2,
        "spinor": True,
        "operations": [
            {
                "table_index": index,
                "rotation_frac": operation["rotation_frac"].tolist(),
                "translation_frac": [0.0, 0.0, 0.0],
                "spin_rotation": _complex_record(
                    spin_lift_from_orthogonal(operation["rotation_cart"])
                ),
            }
            for index, operation in enumerate(operations)
        ],
    }
    setting = {
        "schema_version": "1.0.0",
        "parent_to_standard_direct_transform": np.eye(3).tolist(),
        "origin_shift_fractional": [0.0, 0.0, 0.0],
        "parent_to_standard_operation_map": {"11": 0, "47": 1},
    }
    lift_inputs = {
        "expected_operations": operations,
        "source_table_identity": source_table,
        "standard_setting_identity": setting,
        "direct_lattice_cart": np.eye(3),
    }
    lift = build_double_space_group_lift_certificate(
        source,
        operations,
        source_table_identity=source_table,
        standard_setting_identity=setting,
        direct_lattice_cart=np.eye(3),
    ).to_record()
    source_coefficients = np.zeros((2, 2, 2), dtype=np.complex128)
    source_coefficients[0, 0, 0] = 1.0
    source_coefficients[1, 1, 1] = 1.0
    target_coefficients = np.zeros_like(source_coefficients)
    target_coefficients[0, 0, 1] = 1.0
    target_coefficients[1, 1, 0] = 1.0
    q_cart = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=float,
    )
    return {
        "source_basis_record": source,
        "lift_record": lift,
        "lift_validation_inputs": lift_inputs,
        "extracted_wavefunction_payload_identity": "sha256:" + "1" * 64,
        "source_kpoint_label": "source_arm",
        "target_kpoint_label": "target_arm",
        "source_k_frac": np.array([1.0 / 3.0, 0.0, 0.0]),
        "target_k_frac": np.array([-1.0 / 3.0, 0.0, 0.0]),
        "source_valley": "v0",
        "target_valley": "v1",
        "valley_orbit": ("v0", "v1"),
        "sewing_operation_id": 47,
        "source_little_group_operation_ids": (11,),
        "target_little_group_operation_ids": (11,),
        "detected_operations": operations,
        "source_q_cart": q_cart,
        "target_q_cart": q_cart,
        "source_coefficients": source_coefficients,
        "target_coefficients": target_coefficients,
        "source_projector": np.diag([1.0, 0.0]),
        "target_projector": np.diag([1.0, 0.0]),
        "source_valley_basis": np.array([[1.0], [0.0]]),
        "target_valley_basis": np.array([[1.0], [0.0]]),
        "wavecar_rtag": None,
    }


def test_directed_valley_sewing_scope_recomputes_cross_arm_groupoid_evidence():
    """Catches replacing cross-arm scoped evidence with a self-signed B matrix."""
    inputs = _directed_sewing_inputs()

    record = build_directed_valley_sewing_evidence(**inputs).to_record()

    assert record["status"] == "passed"
    assert record["scope"]["scope_kind"] == "valley_sewing"
    assert record["scope"]["source_kpoint_label"] == "source_arm"
    assert record["scope"]["target_kpoint_label"] == "target_arm"
    assert record["reciprocal_grid"]["source_to_target_map"] == [1, 0]
    assert record["reciprocal_grid"]["permutation_status"] == "passed"
    assert record["directed_group_law"]["rows"][0]["passed"] is True
    assert record["projector_covariance"]["passed"] is True
    assert record["valley_block_quality"]["passed"] is True
    assert record["intertwining"]["rows"][0][
        "rank_normalized_residual"
    ] == pytest.approx(0.0)
    assert record["intertwining"]["rows"][0]["target_character"] == [
        1.0,
        0.0,
    ]
    assert validate_directed_valley_sewing_evidence_record(
        record, **inputs
    ).status == "passed"


@pytest.mark.parametrize(
    ("tamper", "reason"),
    [
        ("target_k", "directed_hsp_mapping_failed"),
        ("valley_mapping", "directed_valley_mapping_failed"),
        ("target_grid", "source_coverage_incomplete"),
        ("target_projector", "projector_covariance_failed"),
        ("source_coefficients", "valley_sewing_unitarity_failed"),
        ("lift_inputs", "double_space_group_lift_not_passed"),
    ],
)
def test_directed_valley_sewing_scope_fails_closed_on_raw_tamper(
    tamper,
    reason,
):
    """Catches accepting one altered producer boundary as trusted sewing."""
    inputs = _directed_sewing_inputs()
    if tamper == "target_k":
        inputs["target_k_frac"][0] = 1.0 / 3.0
    elif tamper == "valley_mapping":
        inputs["detected_operations"][1]["sector_mapping"]["v0"] = "v0"
    elif tamper == "target_grid":
        inputs["target_q_cart"][1] = inputs["target_q_cart"][0]
    elif tamper == "target_projector":
        inputs["target_projector"] = np.diag([0.0, 1.0])
    elif tamper == "source_coefficients":
        inputs["source_coefficients"][0, 0, 0] = 0.5
    else:
        inputs["lift_validation_inputs"]["expected_operations"][0][
            "translation_frac"
        ][0] = 0.25

    record = build_directed_valley_sewing_evidence(**inputs).to_record()

    assert record["status"] == "blocked"
    assert reason in record["reason_codes"]


def test_directed_valley_sewing_validator_rejects_serialized_tamper():
    """Catches trusting a rewritten target character plus a stale identity."""
    inputs = _directed_sewing_inputs()
    record = build_directed_valley_sewing_evidence(**inputs).to_record()
    forged = deepcopy(record)
    forged["intertwining"]["rows"][0]["target_character"] = [0.0, 0.0]

    validation = validate_directed_valley_sewing_evidence_record(
        forged, **inputs
    )

    assert validation.status == "blocked"
    assert validation.reason_codes == ("recomputed_evidence_mismatch",)


def test_directed_valley_sewing_accepts_generic_three_valley_permutation():
    inputs = _directed_sewing_inputs()
    inputs["detected_operations"][0]["sector_mapping"]["v2"] = "v2"
    inputs["detected_operations"][1]["sector_mapping"]["v2"] = "v2"

    record = build_directed_valley_sewing_evidence(**inputs).to_record()

    assert record["status"] == "passed"
    assert len(inputs["detected_operations"][1]["sector_mapping"]) == 3


def test_directed_valley_block_uses_range_not_seed_eigenvalue():
    inputs = _directed_sewing_inputs()
    inputs["source_projector"] = np.diag([0.9, 0.0])
    inputs["target_projector"] = np.diag([0.9, 0.0])

    record = build_directed_valley_sewing_evidence(**inputs).to_record()

    assert record["status"] == "passed"
    assert record["valley_block_quality"]["relative_residual"] == (
        pytest.approx(0.0)
    )


def test_local_irrep_consumes_only_exact_valley_preserving_scope():
    record, inputs = _build(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
        bad_valley_changing_operation=True,
    )

    assert record["status"] == "passed"
    assert record["scope"]["scope_kind"] == "local_irrep"
    assert record["scope"]["required_operation_ids"] == [2]
    assert record["scope"]["valley_preserving_operation_ids"] == [2]
    assert record["scope"]["valley_changing_operation_ids"] == []
    assert "antiunitary_evidence" not in record
    assert record["grey_group_matching_allowed"] is False
    assert record["tolerances"] == {
        "group_law": 1.0e-2,
        "plane_wave_norm": 1.0e-8,
        "target_subspace": 1.0e-2,
        "projector_covariance": 1.0e-2,
        "valley_block": 1.0e-2,
    }
    assert validate_scoped_representation_evidence_record(
        record, **inputs
    ).status == "passed"


def test_scoped_evidence_binds_recomputed_canonical_target_frame():
    inputs = _raw_inputs(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
    )
    overlap = 2.0e-5
    source_coefficients = np.array(
        [
            [[1.0, 0.0]],
            [[overlap, np.sqrt(1.0 - overlap**2)]],
        ],
        dtype=np.complex128,
    )
    frame = build_target_frame(
        source_coefficients,
        wavecar_rtag=45200,
    )
    inputs["source_target_coefficients"] = source_coefficients
    inputs["target_coefficients"] = frame.coefficients
    inputs["wavecar_rtag"] = 45200
    inputs["target_frame_record"] = frame.record

    record = build_scoped_representation_evidence(**inputs).to_record()

    assert record["status"] == "passed"
    assert record["target_frame"]["status"] == "passed"
    assert record["target_frame"]["contract_identity"] == (
        frame.record["contract_identity"]
    )


def test_scoped_evidence_rejects_tampered_target_frame_record():
    inputs = _raw_inputs(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
    )
    frame = build_target_frame(
        inputs["target_coefficients"],
        wavecar_rtag=None,
    )
    tampered = deepcopy(frame.record)
    tampered["canonicalization"]["post_gram_error"] = 1.0
    inputs["target_frame_record"] = tampered

    record = build_scoped_representation_evidence(**inputs).to_record()

    assert record["status"] == "blocked"
    assert "target_frame_recomputation_mismatch" in record["reason_codes"]


def test_valley_changing_failure_blocks_sewing_but_not_local_irrep():
    local, _ = _build(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
        bad_valley_changing_operation=True,
    )
    sewing, _ = _build(
        scope_kind="valley_sewing",
        required_operation_ids=(2, 5),
        bad_valley_changing_operation=True,
    )

    assert local["status"] == "passed"
    assert sewing["status"] == "blocked"
    assert sewing["scope"]["valley_changing_operation_ids"] == [5]
    assert any(
        reason in sewing["reason_codes"]
        for reason in (
            "target_subspace_closure_failed",
            "projected_representation_group_law_failed",
            "valley_block_quality_failed",
        )
    )


def test_clean_valley_sewing_scope_passes_without_antiunitary_evidence():
    record, _ = _build(
        scope_kind="valley_sewing",
        required_operation_ids=(2, 5),
    )

    assert record["status"] == "passed"
    assert record["scope"]["valley_changing_operation_ids"] == [5]
    assert "antiunitary_evidence" not in record


@pytest.mark.parametrize("rank", [2, 4])
def test_matrix_residuals_are_normalized_by_subspace_rank(rank: int):
    inputs = _raw_inputs(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
    )
    epsilon = 1.0e-4
    scale = np.sqrt(1.0 + epsilon)
    inputs["representations"] = {
        2: scale * np.eye(rank, dtype=np.complex128),
    }
    inputs["target_coefficients"] = (
        scale * np.eye(rank, dtype=np.complex128)
    ).reshape(rank, 1, rank)
    inputs["projectors"] = {
        "v0": np.eye(rank, dtype=np.complex128),
        "v1": np.eye(rank, dtype=np.complex128),
    }
    inputs["valley_bases"] = {
        "v0": np.eye(rank, dtype=np.complex128),
        "v1": np.eye(rank, dtype=np.complex128),
    }
    record = build_scoped_representation_evidence(**inputs).to_record()
    target = record["target_subspace"]
    operation = target["operation_rows"][0]
    group_law = record["projected_representation_group_law"][
        "pair_rows"
    ][0]
    block = record["valley_block_quality"]["operation_rows"][0]

    original_gram = record["target_frame"]["original_gram"]
    assert original_gram["normalization"] == pytest.approx(np.sqrt(rank))
    assert original_gram["normalized_error"] == pytest.approx(epsilon)
    assert operation["unitarity_residual_normalization"] == pytest.approx(
        np.sqrt(rank)
    )
    assert operation["unitarity_residual"] == pytest.approx(epsilon)
    assert group_law["residual_normalization"] == pytest.approx(
        np.sqrt(rank)
    )
    assert group_law["residual"] == pytest.approx(
        scale * abs(scale - 1.0)
    )
    assert block[
        "source_basis_orthonormality_normalization"
    ] == pytest.approx(np.sqrt(rank))
    assert block[
        "target_basis_orthonormality_normalization"
    ] == pytest.approx(np.sqrt(rank))
    assert block["block_unitarity_normalization"] == pytest.approx(
        np.sqrt(rank)
    )
    assert block["block_unitarity_residual"] == pytest.approx(epsilon)


def test_tr_completed_scope_is_not_a_supported_representation_scope():
    record, _ = _build(
        scope_kind="tr_completed",
        required_operation_ids=(2,),
    )

    assert record["status"] == "blocked"
    assert "scope_kind_unsupported" in record["reason_codes"]
    assert "antiunitary_evidence" not in record


def test_scoped_evidence_binds_payload_certificates_kpoint_valley_and_opaque_ids():
    record, _ = _build(
        scope_kind="valley_sewing",
        required_operation_ids=(2, 5),
    )

    assert record["source_basis_certificate_identity"] == _source_record()[
        "certificate_identity"
    ]
    assert record["double_space_group_lift_certificate_identity"] == (
        _lift_record()["certificate_identity"]
    )
    assert record["extracted_wavefunction_payload_identity"] == (
        "sha256:" + "1" * 64
    )
    assert record["scope"]["kpoint_label"] == "K0"
    assert record["scope"]["kpoint_frac"] == [0.0, 0.0, 0.0]
    assert record["scope"]["source_valleys"] == ["v0"]
    assert record["scope"]["valley_orbit"] == ["v0", "v1"]
    assert record["scope"]["required_operation_ids"] == [2, 5]
    assert record["evidence_identity"].startswith("sha256:")


def test_scoped_evidence_recomputes_exact_reciprocal_grid_permutation():
    record, _ = _build(
        scope_kind="valley_sewing",
        required_operation_ids=(2, 5),
    )

    rows = record["plane_wave_mapping"]["operation_rows"]
    assert [row["source_to_target_map"] for row in rows] == [[0, 1], [0, 1]]
    assert all(row["reciprocal_grid_permutation_passed"] for row in rows)
    assert record["plane_wave_mapping"]["composition_rows"]
    assert all(
        row["passed"]
        for row in record["plane_wave_mapping"]["composition_rows"]
    )


@pytest.mark.parametrize(
    ("mapping", "reason"),
    [
        ([0, 0], "plane_wave_mapping_not_bijective"),
        ([0, 1.0], "plane_wave_mapping_evidence_malformed"),
        ([0, True], "plane_wave_mapping_evidence_malformed"),
        ([0, 2], "plane_wave_mapping_not_bijective"),
    ],
)
def test_scoped_evidence_rejects_malformed_or_nonbijective_exact_map(
    mapping,
    reason,
):
    inputs = _raw_inputs(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
    )
    inputs["plane_wave_evidence"][2]["source_to_target_map"] = mapping

    record = build_scoped_representation_evidence(**inputs).to_record()

    assert record["status"] == "blocked"
    assert reason in record["reason_codes"]


def test_scoped_evidence_rejects_tampered_but_bijective_map():
    inputs = _raw_inputs(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
    )
    inputs["plane_wave_evidence"][2]["source_to_target_map"] = [1, 0]

    record = build_scoped_representation_evidence(**inputs).to_record()

    assert record["status"] == "blocked"
    assert "plane_wave_mapping_recomputation_mismatch" in record["reason_codes"]


@pytest.mark.parametrize("residual", [-1.0, float("nan"), float("inf")])
def test_scoped_evidence_rejects_negative_or_nonfinite_norm_residual(
    residual,
):
    inputs = _raw_inputs(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
    )
    inputs["plane_wave_evidence"][2]["relative_norm_residual"] = residual

    record = build_scoped_representation_evidence(**inputs).to_record()

    assert record["status"] == "blocked"
    assert "plane_wave_mapping_evidence_malformed" in record["reason_codes"]


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (
            ("extracted_wavefunction_payload_identity",),
            "sha256:" + "9" * 64,
            "recomputed_evidence_mismatch",
        ),
        (
            ("source_basis_certificate_identity",),
            "sha256:" + "8" * 64,
            "recomputed_evidence_mismatch",
        ),
        (
            ("scope", "required_operation_ids"),
            [2],
            "recomputed_evidence_mismatch",
        ),
        (
            ("status",),
            "passed",
            "derived_status_mismatch",
        ),
        (
            ("evidence_identity",),
            "sha256:" + "7" * 64,
            "evidence_identity_mismatch",
        ),
    ],
)
def test_scoped_evidence_rejects_serialized_tampering(path, value, reason):
    record, inputs = _build(
        scope_kind="valley_sewing",
        required_operation_ids=(2, 5),
        bad_valley_changing_operation=True,
    )
    tampered = deepcopy(record)
    target = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    validation = validate_scoped_representation_evidence_record(
        tampered, **inputs
    )

    assert validation.status == "blocked"
    assert reason in validation.reason_codes


def test_readiness_composer_revalidates_and_fails_closed_on_identity_tampering():
    record, inputs = _build(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
    )
    ready = compose_representation_readiness(record, **inputs)
    tampered = deepcopy(record)
    tampered["evidence_identity"] = "sha256:" + "0" * 64
    blocked = compose_representation_readiness(tampered, **inputs)

    assert ready["local_irrep_ready"] is True
    assert ready["valley_sewing_ready"] is False
    assert "tr_completed_ready" not in ready
    assert blocked["local_irrep_ready"] is False
    assert "scoped_representation_evidence_invalid" in blocked["blockers"]
