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
            "status": "sampled_basis",
            "hsp_basis_status": "sampled_basis",
            "ready_for_reduced_table_validation": True,
            "ready_for_ebr_decomposition": False,
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
            "status": "validated_basis",
            "hsp_basis_status": "validated_basis",
            "ready_for_reduced_table_validation": True,
            "ready_for_ebr_decomposition": True,
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
# 1. Sampled-basis: exported but NOT solver-ready
# -----------------------------------------------------------------------

def test_sampled_basis_exported_without_solver_readiness():
    """State 1: sampled_basis instance is exported for validation,
    ready_for_external_solver=false."""
    r = build_ebr_export_bundle(ebr_problem_instances=_SAMPLED_BASIS_INSTANCE)
    assert r["bundle_count"] == 1
    assert r["excluded_count"] == 0
    b = r["bundles"][0]
    assert b["ready_for_external_solver"] is False
    assert b["ready_for_reduced_table_validation"] is True
    assert b["hsp_basis_status"] == "sampled_basis"
    assert b["subspace_group_candidate"] == "P3"
    assert b["missing_optional_hsps"] == ["MM"]


# -----------------------------------------------------------------------
# 2. Validated-basis: exported WITH solver readiness
# -----------------------------------------------------------------------

def test_validated_basis_exported_with_solver_readiness():
    """State 3+: validated_basis instance is exported with ready_for_external_solver=true."""
    r = build_ebr_export_bundle(ebr_problem_instances=_VALIDATED_BASIS_INSTANCE)
    assert r["bundle_count"] == 1
    b = r["bundles"][0]
    assert b["ready_for_external_solver"] is True
    assert b["ready_for_reduced_table_validation"] is True
    assert b["hsp_basis_status"] == "validated_basis"


# -----------------------------------------------------------------------
# 3. Partial / no_data excluded
# -----------------------------------------------------------------------

def test_partial_instance_excluded():
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "certificate_identity": {},
            "status": "no_data",
            "hsp_basis_status": "no_data",
            "ready_for_reduced_table_validation": False,
            "ready_for_ebr_decomposition": False,
        }],
    })
    assert r["bundle_count"] == 0
    assert r["excluded_count"] == 1
    e = r["excluded_instances"][0]
    assert e["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
    assert any(
        "ready_for_reduced_table_validation=false" in reason
        for reason in e["exclusion_reasons"]
    )


def test_sampled_basis_without_validation_gate_excluded():
    """Instance without ready_for_reduced_table_validation is excluded."""
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "M1_valley",
            "subspace_group_candidate": "P2",
            "subspace_space_group": {"candidate_space_group_symbol": "P2"},
            "certificate_identity": {},
            "status": "sampled_basis",
            "hsp_basis_status": "sampled_basis",
            "ready_for_reduced_table_validation": False,
            "ready_for_ebr_decomposition": False,
        }],
    })
    assert r["bundle_count"] == 0
    assert r["excluded_count"] == 1


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
# 5. Mixed: sampled + validated
# -----------------------------------------------------------------------

def test_mixed_sampled_and_validated():
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [
            _SAMPLED_BASIS_INSTANCE["instances"][0],
            _VALIDATED_BASIS_INSTANCE["instances"][0],
        ],
    })
    assert r["bundle_count"] == 2
    assert r["excluded_count"] == 0
    solver_ready = [b["ready_for_external_solver"] for b in r["bundles"]]
    assert solver_ready == [False, True]


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
            "reduced_ebr_decomposition_status", "bundles",
            "excluded_instances", "interpretation"}
    assert keys <= set(r)
    assert r["reduced_ebr_decomposition_status"] == "not_implemented"
    assert r["schema_version"] == "1.2.0"
    b = r["bundles"][0]
    for k in ["bundle_id", "source_instance_id", "valley",
              "irreps_by_kpoint", "operations_by_kpoint",
              "ready_for_external_solver",
              "ready_for_reduced_table_validation",
              "hsp_basis_status"]:
        assert k in b, f"missing: {k}"
    assert "certificate_identity" in b


def test_schema_doc_covers_centered_export_and_ingestion_versions():
    schema = Path("docs/schema.md").read_text(encoding="utf-8")
    normalized_schema = " ".join(schema.split())
    assert 'Schema version `"1.2.0"`' in schema
    assert 'Current ingestion-record schema version: `"1.4.0"`' in schema
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

    assert report["schema_version"] == "1.2.0"
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

    assert report["schema_version"] == "1.2.0"
    assert report["bundles"][0]["certificate_identity"] == certificate_identity


def test_no_forbidden_terms():
    r = build_ebr_export_bundle(ebr_problem_instances=_SAMPLED_BASIS_INSTANCE)
    encoded = json.dumps(r)
    for forbidden in ["covariance", "equivariance", "stabilizer",
                      "valley_little_group"]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"
