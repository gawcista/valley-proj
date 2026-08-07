from copy import deepcopy

import numpy as np

from tests.reduced_ebr_promo_helpers import (
    _cprime_fixture_scope,
    real_primitive_certificate_dict,
)
from tests.test_reduced_ebr_promotion import (
    _real_centered_certificate_dict,
)
from tests.test_scoped_representation_evidence import (
    _directed_sewing_inputs,
)
from valleyscope.analysis.database_ingestion_record import (
    _extract_irrep_records,
    build_database_ingestion_record,
)
from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
from valleyscope.analysis.reduced_ebr_mapping import build_auto_reduced_ebr_mapping
from valleyscope.analysis.projected_hsp_coverage import (
    classify_projected_subspace_kpoint,
    derive_projected_subspace_source_hsp_basis,
)
from valleyscope.analysis.scoped_representation_evidence import (
    build_directed_valley_sewing_evidence,
)
from valleyscope.analysis.unitary_provenance import (
    validate_unitary_bundle_provenance,
)
from valleyscope.analysis.unitary_valley_sewing_completion import (
    build_unitary_valley_sewing_certificate,
    build_unitary_valley_sewing_completion_report,
    validate_unitary_valley_sewing_certificate,
)
from valleyscope.geometry.lattice import (
    cart_rotation_from_fractional,
    cart_translation_from_fractional,
)
from valleyscope.irreps.tables import (
    build_spinful_source_table_evidence,
    load_standard_irrep_table,
)
from valleyscope.irreps.ebr_data_adapter import load_ebr_source_data
from valleyscope.irreps.source_payload import (
    build_source_payload_for_projected_hsp_matching,
)
from valleyscope.io.wavefunction_convention import (
    canonical_identity,
    valid_sha256_identity,
)
from valleyscope.symmetry.double_space_group_lift import (
    build_double_space_group_lift_certificate,
    spin_lift_from_orthogonal,
)
from valleyscope.workflows.analyze_hsp import (
    _build_parent_double_group_lift_context,
    _build_unitary_valley_sewing_attempts,
)


def _completion_inputs():
    directed_raw = _directed_sewing_inputs()
    cprime_record, cprime_raw = _cprime_fixture_scope(
        bundle={"valley": "v0", "valley_orbit": ["v0"]},
        kpoint="source_arm",
        source_table_sg_number=1,
    )
    source_basis = cprime_raw["source_basis_record"]
    directed_raw["source_basis_record"] = source_basis
    directed_raw["extracted_wavefunction_payload_identity"] = source_basis[
        "extracted_wavefunction_payload_identity"
    ]
    lift_inputs = directed_raw["lift_validation_inputs"]
    directed_raw["lift_record"] = build_double_space_group_lift_certificate(
        source_basis,
        lift_inputs["expected_operations"],
        source_table_identity=lift_inputs["source_table_identity"],
        standard_setting_identity=lift_inputs["standard_setting_identity"],
        direct_lattice_cart=lift_inputs["direct_lattice_cart"],
    ).to_record()
    directed_record = build_directed_valley_sewing_evidence(
        **directed_raw
    ).to_record()
    assert directed_record["status"] == "passed", directed_record[
        "reason_codes"
    ]

    certificate = real_primitive_certificate_dict(1, "P1", spinor=True)
    assert certificate is not None
    certificate["parent_basis_operation_ids"] = [11]
    certificate["affine_operation_map"] = {"11": 0}
    table = load_standard_irrep_table(1, spinor=True)
    source_basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=load_ebr_source_data(1, True)[
            "source_basis_labels"
        ],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )
    classification = classify_projected_subspace_kpoint(
        parent_k_frac=np.zeros(3),
        table=table,
        source_hsp_basis=source_basis,
        standard_setting_certificate=certificate,
        mapped_standard_little_group_operation_ids=[1],
        override_source_hsp_label="GM",
        valley="v1",
    )
    source_payload_inputs = {
        "table": table,
        "projected_hsp_classification": classification,
        "detected_operations": directed_raw["detected_operations"],
        "valley_preserving_operation_ids": [11],
        "source_hsp_basis": source_basis,
        "standard_setting_certificate": certificate,
    }
    target_payload = build_source_payload_for_projected_hsp_matching(
        **source_payload_inputs
    )
    assert target_payload["status"] == "ok"
    cprime_links = {
        "spinor_source_basis_certificate_identity": cprime_record[
            "source_basis_certificate_identity"
        ],
        "double_space_group_lift_certificate_identity": cprime_record[
            "double_space_group_lift_certificate_identity"
        ],
        "scoped_representation_evidence_identity": cprime_record[
            "evidence_identity"
        ],
    }
    return {
        "source_candidate": {
            "kpoint": "source_arm",
            "valley": "v0",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "matched_irrep": "source_irrep",
            "irrep_multiplicity": 1,
            "source": "valley_irrep_matching/generic/source_arm/v0",
            "valley_preserving_operation_ids": [11],
            "irrep_source_provenance": {
                "source_hsp_label": "GM",
                "source_table_sg_number": 1,
                "source_table_spinor": True,
                "cprime": cprime_links,
            },
        },
        "target_context": {
            "valley": "v1",
            "source_hsp_label": "GM",
            "valley_preserving_operation_ids": [11],
            "source_operation_map": target_payload["source_operation_map"],
            "source_irrep_characters": target_payload[
                "source_irrep_characters"
            ],
            "source_payload_provenance": target_payload["provenance"],
            "source_payload_context": {
                "record": target_payload,
                "raw_inputs": source_payload_inputs,
            },
            "subspace_space_group": {
                "status": "resolved",
                "candidate_space_group_number": 1,
                "candidate_space_group_symbol": "P1",
                "valley_preserving_operation_ids": [11],
            },
        },
        "directed_scoped_evidence_context": {
            "record": directed_record,
            "raw_inputs": directed_raw,
        },
        "source_standard_setting_certificate": certificate,
        "target_standard_setting_certificate": deepcopy(certificate),
        "source_table": table,
        "target_table": table,
        "detected_operations": directed_raw["detected_operations"],
        "source_cprime_context": {
            "record": cprime_record,
            "raw_inputs": cprime_raw,
            "standard_setting_certificate": deepcopy(certificate),
        },
    }


def _coverage():
    return {
        "by_valley": {
            "v1": {
                "required_source_hsp_labels": ["GM"],
                "covered_source_hsp_labels": [],
                "missing_source_hsp_labels": ["GM"],
                "trusted_matched_source_hsp_labels": [],
                "trusted_missing_source_hsp_labels": ["GM"],
                "source_hsp_to_sampled_kpoint": {},
                "complete": False,
                "ready_for_ebr_promotion": False,
            }
        }
    }


def test_certificate_consumes_directed_scope_and_matches_target_not_source_label():
    raw = _completion_inputs()

    certificate = build_unitary_valley_sewing_certificate(**raw)

    assert certificate["status"] == "passed"
    assert certificate["source"]["irrep"] == "source_irrep"
    assert certificate["target"]["irrep_multiplicities"] == {"-GM2": 1}
    assert certificate["source"]["irrep"] not in certificate["target"][
        "irrep_multiplicities"
    ]
    assert certificate["directed_scoped_evidence_identity"] == raw[
        "directed_scoped_evidence_context"
    ]["record"]["evidence_identity"]
    assert certificate["target_character_vector"] == {"11": [1.0, 0.0]}
    assert "time_reversal" not in certificate
    assert certificate["grey_group_matching_allowed"] is False
    assert validate_unitary_valley_sewing_certificate(
        certificate, **raw
    ).status == "passed"


def test_certificate_recomputation_rejects_raw_and_serialized_tamper():
    raw = _completion_inputs()
    certificate = build_unitary_valley_sewing_certificate(**raw)
    raw_tamper = deepcopy(raw)
    raw_tamper["directed_scoped_evidence_context"]["raw_inputs"][
        "source_projector"
    ] = np.diag([0.0, 1.0])

    raw_validation = validate_unitary_valley_sewing_certificate(
        certificate, **raw_tamper
    )
    forged = deepcopy(certificate)
    forged["target"]["irrep_multiplicities"] = {"forged": 1}
    serialized_validation = validate_unitary_valley_sewing_certificate(
        forged, **raw
    )

    assert raw_validation.status == "blocked"
    assert "recomputed_certificate_mismatch" in raw_validation.reason_codes
    assert serialized_validation.status == "blocked"
    assert "certificate_identity_mismatch" in serialized_validation.reason_codes


def test_certificate_fails_closed_on_reviewed_table_and_setting_tamper():
    for mutate in (
        lambda raw: raw["target_context"]["source_payload_provenance"].update(
            table_sg_number=2
        ),
        lambda raw: raw["source_candidate"]["irrep_source_provenance"][
            "cprime"
        ].update(scoped_representation_evidence_identity="sha256:" + "0" * 64),
        lambda raw: raw["target_standard_setting_certificate"].update(
            validation_status="blocked"
        ),
    ):
        raw = _completion_inputs()
        mutate(raw)
        certificate = build_unitary_valley_sewing_certificate(**raw)
        assert certificate["status"] == "blocked_unknown"


def test_optional_grey_blocker_does_not_block_unitary_completion():
    raw = _completion_inputs()
    raw["target_context"]["joint_grey_source_status"] = "blocked"

    certificate = build_unitary_valley_sewing_certificate(**raw)

    assert certificate["status"] == "passed"
    assert "time_reversal" not in certificate
    assert certificate["grey_group_matching_allowed"] is False


def test_multiple_unitary_paths_must_agree():
    first = _completion_inputs()
    agreeing = deepcopy(first)

    report = build_unitary_valley_sewing_completion_report(
        attempts=[first, agreeing]
    )

    assert report["status"] == "has_inferred_rows"
    assert report["inferred_candidate_count"] == 1
    assert len(report["inferred_candidates"][0][
        "unitary_valley_sewing_certificates"
    ]) == 2

    disagreeing = deepcopy(first)
    disagreeing["source_candidate"]["irrep_multiplicity"] = 2
    directed = disagreeing[
        "directed_scoped_evidence_context"
    ]["raw_inputs"]
    directed["source_projector"] = np.eye(2)
    directed["source_valley_basis"] = np.eye(2)
    disagreeing["directed_scoped_evidence_context"]["record"] = (
        build_directed_valley_sewing_evidence(**directed).to_record()
    )
    blocked = build_unitary_valley_sewing_completion_report(
        attempts=[first, disagreeing]
    )

    assert blocked["status"] == "blocked_unknown"
    assert blocked["inferred_candidate_count"] == 0
    assert blocked["blocked_targets"][0]["reason"] == (
        "multiple_unitary_sewing_paths_disagree"
    )


def test_distinct_source_paths_are_grouped_by_the_same_target():
    first = _completion_inputs()
    second = deepcopy(first)
    cprime_record, cprime_raw = _cprime_fixture_scope(
        bundle={"valley": "v0", "valley_orbit": ["v0"]},
        kpoint="source_arm_2",
        source_table_sg_number=1,
    )
    second["source_candidate"]["kpoint"] = "source_arm_2"
    links = second["source_candidate"]["irrep_source_provenance"]["cprime"]
    links.update({
        "spinor_source_basis_certificate_identity": cprime_record[
            "source_basis_certificate_identity"
        ],
        "double_space_group_lift_certificate_identity": cprime_record[
            "double_space_group_lift_certificate_identity"
        ],
        "scoped_representation_evidence_identity": cprime_record[
            "evidence_identity"
        ],
    })
    second["source_cprime_context"] = {
        "record": cprime_record,
        "raw_inputs": cprime_raw,
        "standard_setting_certificate": deepcopy(
            second["source_standard_setting_certificate"]
        ),
    }
    directed = second["directed_scoped_evidence_context"]["raw_inputs"]
    directed["source_kpoint_label"] = "source_arm_2"
    second["directed_scoped_evidence_context"]["record"] = (
        build_directed_valley_sewing_evidence(**directed).to_record()
    )

    report = build_unitary_valley_sewing_completion_report(
        attempts=[first, second]
    )
    problems = build_ebr_problem_instances(
        ebr_input_candidates={"candidates": report["inferred_candidates"]},
        projected_hsp_coverage=_coverage(),
        unitary_valley_sewing_validation_contexts=report[
            "_validation_contexts"
        ],
    )

    assert report["status"] == "has_inferred_rows"
    assert len(report["inferred_candidates"][0][
        "unitary_valley_sewing_certificates"
    ]) == 2
    assert problems["ready_instance_count"] == 1
    assert set(problems["instances"][0]["cprime_identity_by_kpoint"]) == {
        "scope_001", "scope_002",
    }
    export = build_ebr_export_bundle(ebr_problem_instances=problems)
    cprime_contexts = {
        first["source_cprime_context"]["record"]["evidence_identity"]: first[
            "source_cprime_context"
        ],
        cprime_record["evidence_identity"]: second[
            "source_cprime_context"
        ],
        **report["_validation_contexts"],
    }
    solved = build_auto_reduced_ebr_mapping(
        ebr_export_bundle=export,
        spinor=True,
        cprime_validation_context=cprime_contexts,
    )

    assert solved["status"] == "solved_exact", (
        solved.get("excluded_bundles"),
        solved.get("auto_canonical_bundles"),
    )
    ingestion = build_database_ingestion_record(
        valley_summary={
            "target_kpoints": [],
            "iband": [1, 2],
            "cprime": {
                "spinor_source_basis": {
                    "status": "passed",
                    "identity": cprime_record[
                        "source_basis_certificate_identity"
                    ],
                    "blockers": [],
                },
                "acceptance_matrix": [{
                    "kpoint": kpoint,
                    "valley": "v0",
                    "double_space_group_lift_status": "passed",
                    "double_space_group_lift_identity": record[
                        "double_space_group_lift_certificate_identity"
                    ],
                    "scoped_representation_status": "passed",
                    "scoped_representation_evidence_identity": record[
                        "evidence_identity"
                    ],
                } for kpoint, record in (
                    (
                        "source_arm",
                        first["source_cprime_context"]["record"],
                    ),
                    ("source_arm_2", cprime_record),
                )],
            },
        },
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=solved,
    )

    assert ingestion["record_status"] == "has_final_reduced_ebr_results"


def test_observed_and_inferred_same_kpoint_keep_distinct_valley_scopes():
    raw = _completion_inputs()
    completion = build_unitary_valley_sewing_completion_report(
        attempts=[raw]
    )
    inferred = completion["inferred_candidates"][0]
    cprime_record, _ = _cprime_fixture_scope(
        bundle={"valley": "v1", "valley_orbit": ["v1"]},
        kpoint="source_arm",
        source_table_sg_number=1,
    )
    observed = deepcopy(inferred)
    for key in (
        "completion_kind",
        "evidence_sampled_kpoint",
        "evidence_valley",
        "evidence_source_hsp_label",
        "evidence_irrep_vector",
        "unitary_valley_sewing_certificates",
    ):
        observed.pop(key, None)
    observed.update({
        "kpoint": "source_arm",
        "valley": "v1",
        "workflow_path": "direct_qcut",
        "matched_irrep": "-X2",
        "irrep_multiplicity": 1,
        "source": "valley_irrep_matching/generic/source_arm/v1",
    })
    provenance = observed["irrep_source_provenance"]
    provenance["source_hsp_label"] = "X"
    provenance["cprime"] = {
        "spinor_source_basis_certificate_identity": cprime_record[
            "source_basis_certificate_identity"
        ],
        "double_space_group_lift_certificate_identity": cprime_record[
            "double_space_group_lift_certificate_identity"
        ],
        "scoped_representation_evidence_identity": cprime_record[
            "evidence_identity"
        ],
    }
    coverage = {"by_valley": {"v1": {
        "required_source_hsp_labels": ["GM", "X"],
        "covered_source_hsp_labels": ["X"],
        "missing_source_hsp_labels": ["GM"],
        "trusted_matched_source_hsp_labels": ["X"],
        "trusted_missing_source_hsp_labels": ["GM"],
        "source_hsp_to_sampled_kpoint": {"X": "source_arm"},
        "complete": False,
        "ready_for_ebr_promotion": False,
    }}}

    problems = build_ebr_problem_instances(
        ebr_input_candidates={"candidates": [observed, inferred]},
        projected_hsp_coverage=coverage,
        unitary_valley_sewing_validation_contexts=completion[
            "_validation_contexts"
        ],
    )
    bundle = build_ebr_export_bundle(
        ebr_problem_instances=problems
    )["bundles"][0]

    assert set(bundle["cprime_identity_by_kpoint"]) == {
        "scope_001", "scope_002",
    }
    assert validate_unitary_bundle_provenance(
        bundle,
        unitary_valley_sewing_validation_contexts=completion[
            "_validation_contexts"
        ],
    )
    records = []
    assert _extract_irrep_records(bundle, records) is None
    assert {record["completion_kind"] for record in records} == {
        "observed_at_sampled_kpoint",
        "inferred_by_unitary_valley_sewing",
    }


def test_inferred_row_reaches_export_exact_mapping_and_ingestion():
    raw = _completion_inputs()
    completion = build_unitary_valley_sewing_completion_report(attempts=[raw])
    problems = build_ebr_problem_instances(
        ebr_input_candidates={
            "candidates": completion["inferred_candidates"]
        },
        projected_hsp_coverage=_coverage(),
        unitary_valley_sewing_validation_contexts=completion[
            "_validation_contexts"
        ],
    )
    export = build_ebr_export_bundle(ebr_problem_instances=problems)

    assert problems["ready_instance_count"] == 1
    bundle = export["bundles"][0]
    assert bundle["unitary_vector_construction"]["kind"] == (
        "unitary_valley_sewing_completed_unitary_rows"
    )
    inferred = bundle["unitary_irrep_completion_records_by_hsp"]["GM"][0]
    assert "sampled_kpoint" not in inferred
    assert inferred["evidence_sampled_kpoint"] == "source_arm"
    assert validate_unitary_bundle_provenance(
        bundle,
        unitary_valley_sewing_validation_contexts=completion[
            "_validation_contexts"
        ],
    )
    forged = deepcopy(bundle)
    forged_certificate = forged[
        "unitary_irrep_completion_records_by_hsp"
    ]["GM"][0]["unitary_valley_sewing_certificates"][0]
    forged_certificate["target_character_vector"]["11"] = [0.0, 0.0]
    assert not validate_unitary_bundle_provenance(
        forged,
        unitary_valley_sewing_validation_contexts=completion[
            "_validation_contexts"
        ],
    )

    forged_rehashed = deepcopy(bundle)
    forged_certificate = forged_rehashed[
        "unitary_irrep_completion_records_by_hsp"
    ]["GM"][0]["unitary_valley_sewing_certificates"][0]
    forged_certificate["target_character_vector"]["11"] = [0.0, 0.0]
    forged_content = {
        key: value for key, value in forged_certificate.items()
        if key not in {"status", "reason_codes", "certificate_identity"}
    }
    forged_certificate["certificate_identity"] = canonical_identity(
        forged_content
    )
    assert valid_sha256_identity(forged_certificate["certificate_identity"])
    assert not validate_unitary_bundle_provenance(forged_rehashed)
    assert not validate_unitary_bundle_provenance(
        forged_rehashed,
        unitary_valley_sewing_validation_contexts=completion[
            "_validation_contexts"
        ],
    )

    forged_candidate = deepcopy(completion["inferred_candidates"][0])
    forged_candidate["unitary_valley_sewing_certificates"] = [
        deepcopy(forged_certificate)
    ]
    forged_problems = build_ebr_problem_instances(
        ebr_input_candidates={"candidates": [forged_candidate]},
        projected_hsp_coverage=_coverage(),
        unitary_valley_sewing_validation_contexts=completion[
            "_validation_contexts"
        ],
    )
    assert forged_problems["ready_instance_count"] == 0

    cprime_record = raw["source_cprime_context"]["record"]
    validation_context = {
        cprime_record["evidence_identity"]: raw["source_cprime_context"],
        **completion["_validation_contexts"],
    }
    solved = build_auto_reduced_ebr_mapping(
        ebr_export_bundle=export,
        spinor=True,
        cprime_validation_context=validation_context,
    )
    assert solved["status"] == "solved_exact"
    forged_export = deepcopy(export)
    forged_export["bundles"][0] = forged_rehashed
    forged_mapping = build_auto_reduced_ebr_mapping(
        ebr_export_bundle=forged_export,
        spinor=True,
        cprime_validation_context=validation_context,
    )
    assert forged_mapping["status"] == "blocked"
    summary = {
        "schema_version": "2.0.0",
        "target_kpoints": [],
        "iband": [1, 2],
        "symmetry_analysis": {},
        "cprime": {
            "spinor_source_basis": {
                "status": "passed",
                "identity": cprime_record[
                    "source_basis_certificate_identity"
                ],
                "blockers": [],
            },
            "acceptance_matrix": [{
                "kpoint": "source_arm",
                "valley": "v0",
                "double_space_group_lift_status": "passed",
                "double_space_group_lift_identity": cprime_record[
                    "double_space_group_lift_certificate_identity"
                ],
                "scoped_representation_status": "passed",
                "scoped_representation_evidence_identity": cprime_record[
                    "evidence_identity"
                ],
            }],
        },
    }
    ingestion = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=solved,
    )

    assert ingestion["record_status"] == "has_final_reduced_ebr_results"
    assert ingestion["valley_irrep_records"][0]["completion_kind"] == (
        "inferred_by_unitary_valley_sewing"
    )
    structural_only = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=export,
    )
    assert structural_only["valley_irrep_records"] == []


def test_analyze_hsp_producer_builds_missing_unitary_valley_row():
    raw = _completion_inputs()
    directed = raw["directed_scoped_evidence_context"]["raw_inputs"]
    directed.update({
        "source_k_frac": np.zeros(3),
        "target_k_frac": np.zeros(3),
    })
    operations = directed["detected_operations"]
    certificate = raw["target_standard_setting_certificate"]
    ebr = load_ebr_source_data(1, True)
    table = raw["target_table"]
    basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=ebr["source_basis_labels"],
        standard_setting_certificate=certificate,
        use_2d_momentum_only=True,
    )
    gm = next(
        row for row in basis["source_hsps"]
        if row["source_hsp_label"] == "GM"
    )
    candidate = raw["source_candidate"]
    candidate["ready_for_ebr_input"] = True
    cprime_identity = candidate["irrep_source_provenance"]["cprime"][
        "scoped_representation_evidence_identity"
    ]

    attempts = _build_unitary_valley_sewing_attempts(
        ebr_input_candidates={"candidates": [candidate]},
        projected_hsp_coverage={"by_valley": {"v1": {
            "missing_source_hsp_representatives": [gm],
        }}},
        source_hsp_basis_by_valley={"v1": basis},
        symmetry_payload={
            "detected_operations": operations,
            "spacegroup_number": 2,
            "lattice_direct_cart": np.eye(3).tolist(),
        },
        kpoint_frac_by_name={"source_arm": np.zeros(3)},
        q_cart_by_kpoint={"source_arm": directed["source_q_cart"]},
        coefficients_by_kpoint={
            "source_arm": directed["source_coefficients"]
        },
        seed_projectors_by_kpoint={
            "source_arm": {"v0": directed["source_projector"]}
        },
        symmetry_adapted_projectors_by_kpoint={},
        workflow_decisions={"by_kpoint": {"source_arm": {"v0": {
            "workflow_path": "direct_qcut",
        }}}},
        source_tables={"v0": table, "v1": table},
        source_certificates={
            "v0": certificate,
            "v1": certificate,
        },
        cprime_validation_context={
            cprime_identity: raw["source_cprime_context"]
        },
        source_basis_record=directed["source_basis_record"],
        wavecar_rtag=directed["wavecar_rtag"],
    )
    report = build_unitary_valley_sewing_completion_report(
        attempts=attempts
    )

    assert len(attempts) == 1
    evidence = attempts[0]["directed_scoped_evidence_context"]["record"]
    assert evidence["status"] == "passed"
    assert evidence["scope"]["sewing_operation_id"] == 47
    assert evidence["source_target_frames"]["target_basis_kind"] == (
        "canonical_unitary_transport"
    )
    assert evidence["source_target_frames"][
        "independent_target_numerical_evidence"
    ] is False
    parent_context = _build_parent_double_group_lift_context(
        symmetry_payload={
            "detected_operations": operations,
            "spacegroup_number": 2,
            "lattice_direct_cart": np.eye(3).tolist(),
        },
        source_cprime_context=raw["source_cprime_context"],
    )
    assert parent_context is not None
    parent_certificate = parent_context["standard_setting_certificate"]
    assert parent_certificate["validation_status"] == "validated"
    assert parent_certificate["transform_provenance"] == (
        "spglib_affine_subgroup_standardization"
    )
    assert np.allclose(
        parent_certificate["parent_to_standard_direct_transform"], np.eye(3)
    )
    assert np.allclose(parent_certificate["origin_shift_fractional"], 0.0)
    assert parent_certificate["standard_setting_source"] == (
        "complete_parent_affine_operation_derivation"
    )
    assert report["status"] == "has_inferred_rows"
    assert report["inferred_candidates"][0]["valley"] == "v1"


def test_parent_affine_setting_derives_nontrivial_basis_and_origin():
    table = load_standard_irrep_table(75, spinor=True)
    expected_transform = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    expected_origin = np.array([0.125, 0.0, 0.0])
    direct_lattice = expected_transform.T @ np.diag([1.0, 1.0, 2.0])
    operations = []
    for index, standard in enumerate(table.operations):
        rotation = (
            np.linalg.inv(expected_transform)
            @ standard.rotation_frac
            @ expected_transform
        )
        translation = np.linalg.inv(expected_transform) @ (
            standard.translation_frac
            - expected_origin
            + standard.rotation_frac @ expected_origin
        )
        operations.append({
            "operation_id": 10 + 3 * index,
            "rotation_frac": np.rint(rotation).astype(int),
            "translation_frac": translation,
            "rotation_cart": cart_rotation_from_fractional(
                rotation, direct_lattice
            ),
            "translation_cart": cart_translation_from_fractional(
                translation, direct_lattice
            ),
        })

    raw = _completion_inputs()
    context = _build_parent_double_group_lift_context(
        symmetry_payload={
            "detected_operations": operations,
            "spacegroup_number": 75,
            "lattice_direct_cart": direct_lattice,
        },
        source_cprime_context=raw["source_cprime_context"],
    )
    assert context is not None
    certificate = context["standard_setting_certificate"]
    assert certificate["validation_status"] == "validated"
    assert certificate["transform_provenance"] == (
        "spglib_affine_subgroup_standardization"
    )
    assert not np.allclose(
        certificate["parent_to_standard_direct_transform"], np.eye(3)
    )
    assert not np.allclose(certificate["origin_shift_fractional"], 0.0)
    assert certificate["unmatched_parent_operations"] == []
    assert certificate["unused_standard_operation_indices"] == []

    tampered = deepcopy(operations)
    tampered[1]["translation_frac"][0] += 0.2
    rejected = _build_parent_double_group_lift_context(
        symmetry_payload={
            "detected_operations": tampered,
            "spacegroup_number": 75,
            "lattice_direct_cart": direct_lattice,
        },
        source_cprime_context=raw["source_cprime_context"],
    )
    assert rejected is None


def test_centered_setting_keeps_all_coset_transport_rows_and_bloch_phases():
    centered = _real_centered_certificate_dict()
    table = load_standard_irrep_table(79, spinor=True)
    transform = np.asarray(
        centered["parent_to_standard_direct_transform"], dtype=float
    )
    direct_lattice = transform.T @ np.diag([1.0, 1.0, 2.0])
    operation_ids = centered["parent_basis_operation_ids"]
    operations = []
    for operation_id, standard in zip(operation_ids, table.operations):
        rotation = np.linalg.inv(transform) @ standard.rotation_frac @ transform
        translation = np.linalg.inv(transform) @ standard.translation_frac
        operations.append({
            "operation_id": operation_id,
            "rotation_frac": np.rint(rotation).astype(int),
            "translation_frac": translation,
            "rotation_cart": cart_rotation_from_fractional(
                rotation, direct_lattice
            ),
            "translation_cart": cart_translation_from_fractional(
                translation, direct_lattice
            ),
        })
    identity_id, sewing_id = operation_ids[0], operation_ids[2]
    cycle = {"v0": "v1", "v1": "v2", "v2": "v3", "v3": "v0"}
    for operation in operations:
        operation["sector_mapping"] = (
            cycle if operation["operation_id"] == sewing_id
            else {valley: valley for valley in cycle}
        )
    parent_context = _build_parent_double_group_lift_context(
        symmetry_payload={
            "detected_operations": operations,
            "spacegroup_number": 79,
            "lattice_direct_cart": direct_lattice,
        },
        source_cprime_context=_completion_inputs()["source_cprime_context"],
    )
    assert parent_context is not None
    derived_parent = parent_context["standard_setting_certificate"]
    assert derived_parent["validation_status"] == "validated"
    assert len(derived_parent["centered_affine_operation_map"]) == 8
    cprime_record, cprime_raw = _cprime_fixture_scope(
        bundle={"valley": "v0", "valley_orbit": list(cycle)},
        kpoint="centered_source",
        source_table_sg_number=79,
    )
    source_basis = cprime_raw["source_basis_record"]
    source_table = build_spinful_source_table_evidence(
        table,
        required_operation_indices=[
            operation.table_index for operation in table.operations
        ],
    )
    lift_inputs = {
        "source_table_identity": source_table,
        "standard_setting_identity": {
            "schema_version": "1.0.0",
            "parent_to_standard_direct_transform": transform.tolist(),
            "origin_shift_fractional": [0.0, 0.0, 0.0],
            "parent_to_standard_operation_map": {
                str(operation_id): standard.table_index
                for operation_id, standard in zip(
                    operation_ids, table.operations
                )
            },
        },
        "direct_lattice_cart": direct_lattice,
        "expected_operations": operations,
    }
    lift = build_double_space_group_lift_certificate(
        source_basis,
        operations,
        source_table_identity=source_table,
        standard_setting_identity=lift_inputs[
            "standard_setting_identity"
        ],
        direct_lattice_cart=direct_lattice,
    ).to_record()
    source_coefficients = np.eye(
        2, dtype=np.complex128
    ).reshape(2, 2, 1)
    directed_raw = {
        "source_basis_record": source_basis,
        "lift_record": lift,
        "lift_validation_inputs": lift_inputs,
        "extracted_wavefunction_payload_identity": source_basis[
            "extracted_wavefunction_payload_identity"
        ],
        "source_kpoint_label": "centered_source",
        "target_kpoint_label": "centered_target",
        "source_k_frac": np.zeros(3),
        "target_k_frac": np.zeros(3),
        "source_valley": "v0",
        "target_valley": "v1",
        "sewing_operation_id": sewing_id,
        "source_little_group_operation_ids": [identity_id],
        "target_little_group_operation_ids": [identity_id],
        "detected_operations": operations,
        "source_q_cart": np.zeros((1, 3)),
        "source_coefficients": source_coefficients,
        "source_projector": np.eye(2),
        "source_valley_basis": np.eye(2),
        "wavecar_rtag": None,
    }
    directed_record = build_directed_valley_sewing_evidence(
        **directed_raw
    ).to_record()
    links = {
        "spinor_source_basis_certificate_identity": cprime_record[
            "source_basis_certificate_identity"
        ],
        "double_space_group_lift_certificate_identity": cprime_record[
            "double_space_group_lift_certificate_identity"
        ],
        "scoped_representation_evidence_identity": cprime_record[
            "evidence_identity"
        ],
    }
    source_basis = derive_projected_subspace_source_hsp_basis(
        table=table,
        ebr_source_basis_labels=load_ebr_source_data(79, True)[
            "source_basis_labels"
        ],
        standard_setting_certificate=centered,
        use_2d_momentum_only=True,
    )
    classification = classify_projected_subspace_kpoint(
        parent_k_frac=np.zeros(3),
        table=table,
        source_hsp_basis=source_basis,
        standard_setting_certificate=centered,
        mapped_standard_little_group_operation_ids=[
            table.operations[0].table_index
        ],
        override_source_hsp_label="GM",
        valley="v1",
    )
    payload_inputs = {
        "table": table,
        "projected_hsp_classification": classification,
        "detected_operations": operations,
        "valley_preserving_operation_ids": [identity_id],
        "source_hsp_basis": source_basis,
        "standard_setting_certificate": centered,
    }
    target_payload = build_source_payload_for_projected_hsp_matching(
        **payload_inputs
    )
    raw = {
        "source_candidate": {
            "kpoint": "centered_source",
            "valley": "v0",
            "readiness_level": "trusted",
            "matched_irrep": "source_irrep",
            "irrep_multiplicity": 1,
            "valley_preserving_operation_ids": [identity_id],
            "irrep_source_provenance": {
                "source_hsp_label": "GM",
                "source_table_sg_number": 79,
                "source_table_spinor": True,
                "cprime": links,
            },
        },
        "target_context": {
            "valley": "v1",
            "source_hsp_label": "GM",
            "valley_preserving_operation_ids": [identity_id],
            "source_operation_map": target_payload.get(
                "source_operation_map", {}
            ),
            "source_irrep_characters": target_payload.get(
                "source_irrep_characters", {}
            ),
            "source_payload_provenance": target_payload.get(
                "provenance", {}
            ),
            "source_payload_context": {
                "record": target_payload,
                "raw_inputs": payload_inputs,
            },
            "subspace_space_group": {
                "candidate_space_group_number": 79,
                "candidate_space_group_symbol": "I4",
            },
        },
        "directed_scoped_evidence_context": {
            "record": directed_record,
            "raw_inputs": directed_raw,
        },
        "source_standard_setting_certificate": centered,
        "target_standard_setting_certificate": deepcopy(centered),
        "source_table": table,
        "target_table": table,
        "detected_operations": operations,
        "source_cprime_context": {
            "record": cprime_record,
            "raw_inputs": cprime_raw,
            "standard_setting_certificate": centered,
        },
    }

    certificate = build_unitary_valley_sewing_certificate(**raw)

    assert directed_record["status"] == "passed", directed_record[
        "reason_codes"
    ]
    assert certificate["status"] == "blocked_unknown"
    assert "target_irrep_matching_not_unique" in certificate["reason_codes"]
    transports = certificate["source_target_standard_setting_transport"]
    for side in ("source", "target"):
        rows = transports[side]["operation_rows"]
        assert len(rows) == 8
        assert {row["centering_coset_index"] for row in rows} == {0, 1}
        assert all("bloch_phase" in row for row in rows)
