from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from valleyscope.analysis.representation_readiness import (
    compose_representation_readiness,
)
from valleyscope.analysis.scoped_representation_evidence import (
    build_scoped_representation_evidence,
    validate_scoped_representation_evidence_record,
)
from valleyscope.geometry.lattice import (
    cart_rotation_from_fractional,
    cart_translation_from_fractional,
)
from valleyscope.io.spinor_source_basis import SpinorSourceBasisCertificate
from valleyscope.symmetry.double_space_group_lift import (
    build_double_space_group_lift_certificate,
    spin_lift_from_orthogonal,
)


def _source_record() -> dict[str, object]:
    return SpinorSourceBasisCertificate(
        extracted_wavefunction_payload_identity="sha256:" + "1" * 64,
        nspinor=2,
        parser_identity="valleyscope_h5_reader_v1",
        extractor_identity="valleyscope_wavecar_h5_layout_v1",
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
    antiunitary_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
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
            2: {
                "mapping_miss_count": 0,
                "norm_preservation_residual": 0.0,
            },
            5: {
                "mapping_miss_count": 0,
                "norm_preservation_residual": 0.0,
            },
        },
        "target_coefficients": np.eye(2, dtype=np.complex128),
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
        "antiunitary_evidence": antiunitary_evidence,
    }


def _build(**kwargs) -> tuple[dict[str, object], dict[str, object]]:
    inputs = _raw_inputs(**kwargs)
    return build_scoped_representation_evidence(**inputs).to_record(), inputs


def _clean_tr_evidence() -> dict[str, object]:
    return {
        "source_valley": "v0",
        "target_valley": "v1",
        "forward_sewing_matrix": np.array([[1.0]]),
        "reverse_sewing_matrix": np.array([[-1.0]]),
        "expected_square_sign": -1,
    }


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
    assert record["antiunitary_evidence"]["required"] is False
    assert record["antiunitary_evidence"]["evaluated"] is False
    assert record["grey_group_matching_allowed"] is False
    assert validate_scoped_representation_evidence_record(
        record, **inputs
    ).status == "passed"


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
    assert record["antiunitary_evidence"]["required"] is False


def test_tr_completed_scope_requires_antiunitary_evidence_and_does_not_grey_match_one_valley():
    missing, _ = _build(
        scope_kind="tr_completed",
        required_operation_ids=(2, 5),
    )
    present, _ = _build(
        scope_kind="tr_completed",
        required_operation_ids=(2, 5),
        antiunitary_evidence=_clean_tr_evidence(),
    )

    assert missing["status"] == "blocked"
    assert "antiunitary_evidence_missing" in missing["reason_codes"]
    assert present["status"] == "passed"
    assert present["antiunitary_evidence"]["square_residual"] < 1e-12
    assert present["grey_group_matching_allowed"] is False


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
    assert ready["tr_completed_ready"] is False
    assert blocked["local_irrep_ready"] is False
    assert "scoped_representation_evidence_invalid" in blocked["blockers"]
