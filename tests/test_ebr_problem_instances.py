import json

from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances


def _make_tmo_te2_candidates():
    return {
        "status": "has_candidates",
        "candidates": [
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C3_like",
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C3_like",
                "operation_id": 2, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "KM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C3_like",
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/6",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "KM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C3_like",
                "operation_id": 2, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_-1/6",
                "ready_for_ebr_input": True,
            },
        ],
    }


# -----------------------------------------------------------------------
# 1. tMoTe2-like candidates group into instances
# -----------------------------------------------------------------------

def test_candidates_group_into_instances():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_tmo_te2_candidates(),
    )
    assert r["status"] == "has_instances"
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["subspace_group_candidate"] == "C3_like"
    assert inst["valley"] == "K_valley"
    assert inst["status"] == "complete"
    assert inst["ready_for_ebr_decomposition"] is True
    assert "GammaM" in inst["irreps_by_kpoint"]
    assert "KM" in inst["irreps_by_kpoint"]
    assert inst["optional_hsps"] == ["MM"]
    assert inst["missing_optional_hsps"] == ["MM"]


# -----------------------------------------------------------------------
# 2. Partial HSP data
# -----------------------------------------------------------------------

def test_partial_hsp_not_marked_ready():
    """Only GammaM data, missing KM → partial."""
    cands = {
        "candidates": [
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C3_like",
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["status"] == "partial"
    assert inst["ready_for_ebr_decomposition"] is False
    assert "missing required HSPs" in inst["blocked_by"][0]


# -----------------------------------------------------------------------
# 3. Blocked/diagnostic rows do not enter instances
# -----------------------------------------------------------------------

def test_blocked_rows_excluded():
    cands = {
        "candidates": [
            {
                "kpoint": "MM", "valley": "M3_valley",
                "workflow_path": "symmetry_adapted",
                "readiness_level": "usable_with_caution",
                "subspace_group_candidate": "C2_like",
                "operation_id": 5, "operation_order": 2,
                "matched_irrep": None,
                "ready_for_ebr_input": False,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 0
    assert r["status"] == "no_instances"


# -----------------------------------------------------------------------
# 4. P2/C2_like has no policy → no_instances
# -----------------------------------------------------------------------

def test_c2_like_no_policy():
    cands = {
        "candidates": [
            {
                "kpoint": "GammaM", "valley": "M1_valley",
                "workflow_path": "symmetry_adapted",
                "readiness_level": "trusted",
                "subspace_group_candidate": "C2_like",
                "operation_id": 4, "operation_order": 2,
                "matched_irrep": "C2_spinor_phase_+1/4",
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["status"] == "no_policy"
    assert inst["ready_for_ebr_decomposition"] is False


# -----------------------------------------------------------------------
# 5. Empty / null inputs
# -----------------------------------------------------------------------

def test_null_input():
    r = build_ebr_problem_instances(ebr_input_candidates=None)
    assert r["status"] == "no_instances"
    assert r["instance_count"] == 0


def test_empty_candidates():
    r = build_ebr_problem_instances(
        ebr_input_candidates={"status": "no_candidates", "candidates": []},
    )
    assert r["status"] == "no_instances"


# -----------------------------------------------------------------------
# 6. JSON serializable
# -----------------------------------------------------------------------

def test_json_serializable():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_tmo_te2_candidates(),
    )
    encoded = json.dumps(r)
    assert len(encoded) > 0
    assert "dtype" not in encoded


# -----------------------------------------------------------------------
# 7. Schema fields
# -----------------------------------------------------------------------

def test_schema_fields():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_tmo_te2_candidates(),
    )
    inst = r["instances"][0]
    for key in ["instance_id", "valley", "subspace_group_candidate",
                "workflow_path", "readiness_level", "irreps_by_kpoint",
                "operations_by_kpoint", "candidate_count", "status",
                "ready_for_ebr_decomposition", "blocked_by",
                "expected_hsps", "optional_hsps", "actual_hsps",
                "missing_optional_hsps"]:
        assert key in inst, f"missing key: {key}"
    assert r["reduced_ebr_decomposition_status"] == "not_implemented"


# -----------------------------------------------------------------------
# 8. Forbidden terms
# -----------------------------------------------------------------------

def test_no_forbidden_terms():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_tmo_te2_candidates(),
    )
    encoded = json.dumps(r)
    for forbidden in ["covariance", "equivariance", "stabilizer",
                      "valley_little_group"]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"


# -----------------------------------------------------------------------
# 9. Multiple valleys
# -----------------------------------------------------------------------

def test_multiple_valleys():
    cands = {
        "candidates": [
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C3_like",
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "GammaM", "valley": "Kp_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C3_like",
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 2
    valleys = {inst["valley"] for inst in r["instances"]}
    assert valleys == {"K_valley", "Kp_valley"}


def test_same_valley_different_provenance_not_merged():
    cands = {
        "candidates": [
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C3_like",
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "symmetry_adapted", "readiness_level": "trusted",
                "subspace_group_candidate": "C3_like",
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 2
    assert {inst["workflow_path"] for inst in r["instances"]} == {
        "direct_qcut",
        "symmetry_adapted",
    }


def test_generic_multiplicity_records_expand_irrep_counts():
    """Generic character multiplicities expand only at problem-instance level."""
    cands = {
        "candidates": [
            {
                "kpoint": "GammaM",
                "valley": "K_valley",
                "workflow_path": "direct_qcut",
                "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "valley_preserving_operation_ids": [0, 4],
                },
                "legacy_subspace_group_candidate": "C3_like",
                "matched_irrep": "-GM6_a",
                "irrep_multiplicity": 2,
                "matching_strategy": "bilbao_restricted_character",
                "valley_preserving_operation_ids": [0, 4],
                "source_operation_map": {0: 1, 4: 2},
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "KM",
                "valley": "K_valley",
                "workflow_path": "direct_qcut",
                "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "valley_preserving_operation_ids": [0, 4],
                },
                "legacy_subspace_group_candidate": "C3_like",
                "matched_irrep": "-K5",
                "irrep_multiplicity": 1,
                "matching_strategy": "bilbao_restricted_character",
                "valley_preserving_operation_ids": [0, 4],
                "source_operation_map": {0: 1, 4: 2},
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    inst = r["instances"][0]
    assert inst["status"] == "complete"
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["irreps_by_kpoint"]["GammaM"] == ["-GM6_a", "-GM6_a"]
    assert inst["irreps_by_kpoint"]["KM"] == ["-K5"]
    rec = inst["irrep_records_by_kpoint"]["GammaM"][0]
    assert rec["irrep_multiplicity"] == 2
    assert rec["matching_strategy"] == "bilbao_restricted_character"
    assert rec["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
