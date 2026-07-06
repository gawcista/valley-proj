import json

from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle


_COMPLETE_INSTANCES = {
    "instances": [
        {
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {
                "candidate_space_group_symbol": "P3",
                "valley_preserving_operation_ids": [0, 1],
            },
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "status": "complete",
            "ready_for_ebr_decomposition": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"], "KM": ["C3_spinor_phase_+1/6"]},
            "operations_by_kpoint": {"GammaM": [1], "KM": [1]},
            "expected_hsps": ["GammaM", "KM"],
            "optional_hsps": ["MM"],
            "missing_optional_hsps": ["MM"],
        },
    ],
}


# -----------------------------------------------------------------------
# 1. Complete instances become bundles
# -----------------------------------------------------------------------

def test_complete_instances_exported():
    r = build_ebr_export_bundle(ebr_problem_instances=_COMPLETE_INSTANCES)
    assert r["status"] == "ready_for_external_solver"
    assert r["bundle_count"] == 1
    assert r["excluded_count"] == 0
    b = r["bundles"][0]
    assert b["ready_for_external_solver"] is True
    assert b["subspace_group_candidate"] == "P3"
    assert b["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
    assert b["missing_optional_hsps"] == ["MM"]


# -----------------------------------------------------------------------
# 2. Partial / no_policy excluded
# -----------------------------------------------------------------------

def test_partial_instance_excluded():
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "status": "partial",
            "ready_for_ebr_decomposition": False,
            "blocked_by": ["missing required HSPs: ['KM']"],
        }],
    })
    assert r["bundle_count"] == 0
    assert r["excluded_count"] == 1
    e = r["excluded_instances"][0]
    assert e["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
    assert any("status=partial" in r for r in e["exclusion_reasons"])


def test_no_policy_excluded():
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "M1_valley",
            "subspace_group_candidate": "P2",
            "subspace_space_group": {"candidate_space_group_symbol": "P2"},
            "status": "no_policy",
            "ready_for_ebr_decomposition": False,
            "blocked_by": ["no expected-HSP policy for P2"],
        }],
    })
    assert r["bundle_count"] == 0
    assert r["excluded_count"] == 1
    assert (
        r["excluded_instances"][0]["subspace_space_group"][
            "candidate_space_group_symbol"
        ]
        == "P2"
    )


# -----------------------------------------------------------------------
# 3. No instances → no_bundles
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
# 4. Mixed complete + excluded
# -----------------------------------------------------------------------

def test_mixed_partial_export():
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [
            {
                "instance_id": "ebr_instance_001",
                "valley": "K_valley",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                "workflow_path": "direct_qcut",
                "readiness_level": "trusted",
                "status": "complete",
                "ready_for_ebr_decomposition": True,
                "irreps_by_kpoint": {"GammaM": []},
                "operations_by_kpoint": {},
                "expected_hsps": ["GammaM", "KM"],
                "optional_hsps": [],
                "missing_optional_hsps": [],
            },
            {
                "instance_id": "ebr_instance_002",
                "valley": "M1_valley",
                "subspace_group_candidate": "P2",
                "subspace_space_group": {"candidate_space_group_symbol": "P2"},
                "status": "no_policy",
                "ready_for_ebr_decomposition": False,
                "blocked_by": ["no expected-HSP policy for P2"],
            },
        ],
    })
    assert r["status"] == "partial_export"
    assert r["bundle_count"] == 1
    assert r["excluded_count"] == 1


def test_complete_but_not_trusted_is_excluded():
    r = build_ebr_export_bundle(ebr_problem_instances={
        "instances": [{
            "instance_id": "ebr_instance_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "workflow_path": "symmetry_adapted",
            "readiness_level": "usable_with_caution",
            "status": "complete",
            "ready_for_ebr_decomposition": True,
            "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"]},
            "operations_by_kpoint": {"GammaM": [1]},
            "expected_hsps": ["GammaM", "KM"],
            "optional_hsps": ["MM"],
            "missing_optional_hsps": ["MM"],
        }],
    })
    assert r["bundle_count"] == 0
    assert r["excluded_count"] == 1
    assert any(
        "readiness_level=usable_with_caution" in reason
        for reason in r["excluded_instances"][0]["exclusion_reasons"]
    )


# -----------------------------------------------------------------------
# 5. JSON serializable
# -----------------------------------------------------------------------

def test_json_serializable():
    r = build_ebr_export_bundle(ebr_problem_instances=_COMPLETE_INSTANCES)
    encoded = json.dumps(r)
    assert len(encoded) > 0
    assert "dtype" not in encoded


# -----------------------------------------------------------------------
# 6. Schema
# -----------------------------------------------------------------------

def test_schema_fields():
    r = build_ebr_export_bundle(ebr_problem_instances=_COMPLETE_INSTANCES)
    keys = {"status", "bundle_count", "excluded_count", "schema_version",
            "reduced_ebr_decomposition_status", "bundles",
            "excluded_instances", "interpretation"}
    assert keys <= set(r)
    assert r["reduced_ebr_decomposition_status"] == "not_implemented"
    assert r["schema_version"] == "1.0.0"
    b = r["bundles"][0]
    for k in ["bundle_id", "source_instance_id", "valley",
              "irreps_by_kpoint", "operations_by_kpoint",
              "ready_for_external_solver"]:
        assert k in b, f"missing: {k}"
    assert "subspace_space_group" in b


def test_no_forbidden_terms():
    r = build_ebr_export_bundle(ebr_problem_instances=_COMPLETE_INSTANCES)
    encoded = json.dumps(r)
    for forbidden in ["covariance", "equivariance", "stabilizer",
                      "valley_little_group"]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"
