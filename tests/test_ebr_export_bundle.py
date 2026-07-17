import json
from pathlib import Path

from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle


_SAMPLED_BASIS_INSTANCE = {
    "instances": [
        {
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {
                "candidate_space_group_symbol": "P3",
                "valley_preserving_operation_ids": [0, 1],
            },
            "certificate_identity": {
                "hall_numbers": [],
                "certificate_validation_statuses": ["not_evaluated"],
                "any_unresolved": True,
            },
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "status": "canonical_hsp_vector_ready",
            "canonical_hsp_vector_complete": True,
            "canonical_hsp_vector_ready": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"], "KM": ["C3_spinor_phase_+1/6"]},
            "operations_by_kpoint": {"GammaM": [1], "KM": [1]},
            "irrep_records_by_kpoint": {},
            "expected_hsps": ["GammaM", "KM"],
            "optional_hsps": ["MM"],
            "missing_optional_hsps": ["MM"],
        },
    ],
}

_VALIDATED_BASIS_INSTANCE = {
    "instances": [
        {
            "instance_id": "ebr_instance_002",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {
                "candidate_space_group_symbol": "P3",
            },
            "certificate_identity": {
                "hall_numbers": [430],
                "certificate_validation_statuses": ["validated"],
                "any_unresolved": False,
            },
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "status": "canonical_hsp_vector_ready",
            "canonical_hsp_vector_complete": True,
            "canonical_hsp_vector_ready": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
            "operations_by_kpoint": {"GammaM": [1]},
            "irrep_records_by_kpoint": {},
            "expected_hsps": ["GammaM"],
            "optional_hsps": [],
            "missing_optional_hsps": [],
        },
    ],
}


# -----------------------------------------------------------------------
# 1. Complete canonical vector is exported for table validation
# -----------------------------------------------------------------------

def test_canonical_vector_exported_for_table_validation():
    r = build_ebr_export_bundle(ebr_problem_instances=_SAMPLED_BASIS_INSTANCE)
    assert r["bundle_count"] == 1
    assert r["excluded_count"] == 0
    b = r["bundles"][0]
    assert b["ready_for_reduced_table_validation"] is True
    assert "ready_for_external_solver" not in b
    assert b["subspace_group_candidate"] == "P3"
    assert b["missing_optional_hsps"] == ["MM"]


# -----------------------------------------------------------------------
# 2. Incomplete vectors are excluded
# -----------------------------------------------------------------------

def test_partial_instance_excluded():
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "certificate_identity": {},
            "status": "incomplete_canonical_hsp_vector",
            "canonical_hsp_vector_complete": False,
            "blocked_by": ["source_hsp_coverage_incomplete"],
        }],
    })
    assert r["bundle_count"] == 0
    assert r["excluded_count"] == 1
    e = r["excluded_instances"][0]
    assert e["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
    assert e["exclusion_reasons"] == ["source_hsp_coverage_incomplete"]


def test_incomplete_canonical_vector_is_excluded():
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "M1_valley",
            "subspace_group_candidate": "P2",
            "subspace_space_group": {"candidate_space_group_symbol": "P2"},
            "certificate_identity": {},
            "status": "incomplete_canonical_hsp_vector",
            "canonical_hsp_vector_complete": False,
            "blocked_by": ["projected_hsp_coverage_missing"],
        }],
    })
    assert r["bundle_count"] == 0
    assert r["excluded_count"] == 1


def test_complete_but_untrusted_vector_is_excluded_with_both_states():
    instance = dict(_SAMPLED_BASIS_INSTANCE["instances"][0])
    instance.update({
        "status": "canonical_hsp_vector_complete_but_untrusted",
        "canonical_hsp_vector_complete": True,
        "canonical_hsp_vector_ready": False,
        "blocked_by": [
            "source_hsp_coverage_not_ready_for_ebr_promotion",
        ],
    })

    report = build_ebr_export_bundle(
        ebr_problem_instances={"instances": [instance]},
    )

    assert report["bundle_count"] == 0
    assert report["excluded_count"] == 1
    excluded = report["excluded_instances"][0]
    assert excluded["canonical_hsp_vector_complete"] is True
    assert excluded["canonical_hsp_vector_ready"] is False
    assert excluded["exclusion_reasons"] == [
        "source_hsp_coverage_not_ready_for_ebr_promotion",
    ]


# -----------------------------------------------------------------------
# 4. No instances → no_bundles
# -----------------------------------------------------------------------

def test_null_input():
    r = build_ebr_export_bundle(ebr_problem_instances=None)
    assert r["status"] == "no_bundles"
    assert r["bundle_count"] == 0


def test_empty_instances():
    r = build_ebr_export_bundle(
        ebr_problem_instances={"status": "no_instances", "instances": []},
    )
    assert r["status"] == "no_bundles"


# -----------------------------------------------------------------------
# 4. Multiple complete canonical vectors
# -----------------------------------------------------------------------

def test_multiple_canonical_vectors_export_once_each():
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [
            _SAMPLED_BASIS_INSTANCE["instances"][0],
            _VALIDATED_BASIS_INSTANCE["instances"][0],
        ],
    })
    assert r["bundle_count"] == 2
    assert r["excluded_count"] == 0
    assert all(
        bundle["ready_for_reduced_table_validation"] is True
        for bundle in r["bundles"]
    )


# -----------------------------------------------------------------------
# 6. JSON serializable
# -----------------------------------------------------------------------

def test_json_serializable():
    r = build_ebr_export_bundle(ebr_problem_instances=_SAMPLED_BASIS_INSTANCE)
    encoded = json.dumps(r)
    assert len(encoded) > 0
    assert "dtype" not in encoded


# -----------------------------------------------------------------------
# 7. Schema
# -----------------------------------------------------------------------

def test_schema_fields():
    r = build_ebr_export_bundle(ebr_problem_instances=_SAMPLED_BASIS_INSTANCE)
    keys = {"status", "bundle_count", "excluded_count", "schema_version",
            "bundles",
            "excluded_instances", "interpretation"}
    assert keys <= set(r)
    assert "reduced_ebr_decomposition_status" not in r
    assert r["schema_version"] == "1.6.0"
    b = r["bundles"][0]
    for k in ["bundle_id", "source_instance_id", "valley",
              "irreps_by_kpoint", "operations_by_kpoint",
              "ready_for_reduced_table_validation",
              "canonical_hsp_vector_complete",
              "canonical_hsp_vector_ready"]:
        assert k in b, f"missing: {k}"
    assert "ready_for_external_solver" not in b
    assert "hsp_basis_status" not in b
    assert "certificate_identity" in b


def test_source_hsp_coverage_fields_are_exported_without_loss():
    instance = dict(_SAMPLED_BASIS_INSTANCE["instances"][0])
    instance.update({
        "required_source_hsp_labels": ["GM", "K"],
        "covered_source_hsp_labels": ["GM", "K"],
        "missing_source_hsp_labels": [],
        "trusted_matched_source_hsp_labels": ["GM", "K"],
        "source_hsp_to_sampled_kpoint": {"GM": "GammaM", "K": "KM"},
        "source_hsp_coverage_complete": True,
        "source_hsp_coverage_provenance": {"source": "irreptables"},
    })

    bundle = build_ebr_export_bundle(
        ebr_problem_instances={"instances": [instance]},
    )["bundles"][0]

    assert bundle["required_source_hsp_labels"] == ["GM", "K"]
    assert bundle["covered_source_hsp_labels"] == ["GM", "K"]
    assert bundle["missing_source_hsp_labels"] == []
    assert bundle["source_hsp_to_sampled_kpoint"] == {
        "GM": "GammaM", "K": "KM",
    }
    assert bundle["source_hsp_coverage_complete"] is True
    assert bundle["source_hsp_coverage_provenance"] == {
        "source": "irreptables",
    }


def test_schema_doc_covers_centered_export_and_ingestion_versions():
    schema = Path("docs/schema.md").read_text(encoding="utf-8")
    normalized_schema = " ".join(schema.split())
    assert 'Schema version `"1.6.0"`' in schema
    assert 'Current ingestion-record schema version: `"1.6.0"`' in schema
    assert "centered_affine_operation_map" in schema
    assert "centering_coset_index" in schema
    assert (
        "no downstream grouping or serialization step may reorder"
        in normalized_schema
    )
    assert "fail closed at promotion" in schema


def test_schema_1_2_preserves_required_operation_ids_in_certificate_identity():
    instance = dict(_VALIDATED_BASIS_INSTANCE["instances"][0])
    certificate_identity = dict(instance["certificate_identity"])
    certificate_identity["affine_required_operation_ids"] = [-3, 4]
    instance["certificate_identity"] = certificate_identity

    report = build_ebr_export_bundle(
        ebr_problem_instances={"instances": [instance]},
    )

    assert report["schema_version"] == "1.6.0"
    exported_identity = report["bundles"][0]["certificate_identity"]
    assert exported_identity["affine_required_operation_ids"] == [-3, 4]


def test_schema_1_2_preserves_centered_expansion_identity_without_loss():
    instance = dict(_VALIDATED_BASIS_INSTANCE["instances"][0])
    certificate_identity = dict(instance["certificate_identity"])
    centered_map = [
        {
            "parent_operation_id": -3,
            "centering_coset_index": 0,
            "standard_operation_index": 0,
        },
        {
            "parent_operation_id": -3,
            "centering_coset_index": 1,
            "standard_operation_index": 1,
        },
        {
            "parent_operation_id": 4,
            "centering_coset_index": 0,
            "standard_operation_index": 2,
        },
        {
            "parent_operation_id": 4,
            "centering_coset_index": 1,
            "standard_operation_index": 3,
        },
    ]
    certificate_identity.update({
        "centering_coset_count": 2,
        "primitive_conventional_index": 2,
        "expanded_parent_operation_count": 4,
        "matched_expanded_operations": 4,
        "centered_affine_operation_map": centered_map,
        "affine_unmatched_centered_operation_pairs": [],
    })
    instance["certificate_identity"] = certificate_identity

    report = build_ebr_export_bundle(
        ebr_problem_instances={"instances": [instance]},
    )

    assert report["schema_version"] == "1.6.0"
    assert report["bundles"][0]["certificate_identity"] == certificate_identity


def test_no_forbidden_terms():
    r = build_ebr_export_bundle(ebr_problem_instances=_SAMPLED_BASIS_INSTANCE)
    encoded = json.dumps(r)
    for forbidden in ["covariance", "equivariance", "stabilizer",
                      "valley_little_group"]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"
