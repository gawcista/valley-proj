import json

from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances


def _make_c3_preserving_candidates():
    """Generic C3 valley-preserving candidate fixture with subspace_space_group."""
    return {
        "status": "has_candidates",
        "candidates": [
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "valley_preserving_operation_ids": [0, 1, 2],
                },
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "valley_preserving_operation_ids": [0, 1, 2],
                },
                "operation_id": 2, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "KM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "valley_preserving_operation_ids": [0, 1, 2],
                },
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/6",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "KM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "valley_preserving_operation_ids": [0, 1, 2],
                },
                "operation_id": 2, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_-1/6",
                "ready_for_ebr_input": True,
            },
        ],
    }


# -----------------------------------------------------------------------
# 1. C3 valley-preserving candidates group into instances
# -----------------------------------------------------------------------

def test_candidates_group_into_instances():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )
    assert r["status"] == "has_instances"
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["subspace_group_candidate"] == "P3"
    assert inst["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
    assert inst["valley"] == "K_valley"
    assert inst["status"] == "complete"
    assert inst["ready_for_ebr_decomposition"] is True
    assert "GammaM" in inst["irreps_by_kpoint"]
    assert "KM" in inst["irreps_by_kpoint"]
    # Table-authoritative: expected_hsps from sampled irrep basis.
    assert inst["expected_hsps"] == ["GammaM", "KM"]
    assert inst["expected_hsp_policy_source"] == "sampled_irrep_basis"
    assert inst["optional_hsps"] == []
    assert inst["missing_optional_hsps"] == []


# -----------------------------------------------------------------------
# 2. Partial HSP data
# -----------------------------------------------------------------------

def test_partial_hsp_still_complete_table_authoritative():
    """Table-authoritative: expected HSPs = actual HSPs; no hard-coded policy blocks."""
    cands = {
        "candidates": [
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "candidate_space_group_number": 143,
                    "valley_preserving_operation_ids": [0, 1],
                },
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["status"] == "complete"
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["expected_hsps"] == ["GammaM"]
    assert inst["expected_hsp_policy_source"] == "sampled_irrep_basis"
    assert inst["hsp_basis_status"] == "sampled_basis"
    assert inst["subspace_sg_number"] == 143


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
                "subspace_group_candidate": "P2",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P2",
                    "valley_preserving_operation_ids": [0, 5],
                },
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
# 4. P2 valley-preserving: table-authoritative, no legacy policy needed
# -----------------------------------------------------------------------

def test_p2_valley_preserving_table_authoritative():
    """Table-authoritative: P2 valley-preserving data is not blocked by legacy policy."""
    cands = {
        "candidates": [
            {
                "kpoint": "GammaM", "valley": "M1_valley",
                "workflow_path": "symmetry_adapted",
                "readiness_level": "trusted",
                "subspace_group_candidate": "P2",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P2",
                    "candidate_space_group_number": 3,
                    "valley_preserving_operation_ids": [0, 4],
                },
                "operation_id": 4, "operation_order": 2,
                "matched_irrep": "C2_spinor_phase_+1/4",
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["status"] == "complete"
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["expected_hsps"] == ["GammaM"]
    assert inst["hsp_basis_status"] == "sampled_basis"


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
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )
    encoded = json.dumps(r)
    assert len(encoded) > 0
    assert "dtype" not in encoded


# -----------------------------------------------------------------------
# 7. Schema fields
# -----------------------------------------------------------------------

def test_schema_fields():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )
    inst = r["instances"][0]
    for key in ["instance_id", "valley", "subspace_group_candidate",
                "subspace_sg_number", "subspace_space_group",
                "workflow_path", "workflow_paths",
                "readiness_level", "readiness_evidence",
                "irreps_by_kpoint", "operations_by_kpoint",
                "candidate_count", "status",
                "ready_for_ebr_decomposition", "blocked_by",
                "expected_hsps", "optional_hsps", "actual_hsps",
                "missing_optional_hsps", "hsp_basis_status"]:
        assert key in inst, f"missing key: {key}"
    assert r["reduced_ebr_decomposition_status"] == "not_implemented"


# -----------------------------------------------------------------------
# 8. Forbidden terms
# -----------------------------------------------------------------------

def test_no_forbidden_terms():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
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
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "candidate_space_group_number": 143,
                    "valley_preserving_operation_ids": [0, 1],
                },
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "GammaM", "valley": "Kp_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "candidate_space_group_number": 143,
                    "valley_preserving_operation_ids": [0, 1],
                },
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


def test_same_valley_different_provenance_merged_into_one_instance():
    """Different workflow paths for same physical identity merge into one instance."""
    cands = {
        "candidates": [
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "candidate_space_group_number": 143,
                    "valley_preserving_operation_ids": [0, 1],
                },
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "GammaM", "valley": "K_valley",
                "workflow_path": "symmetry_adapted", "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "candidate_space_group_number": 143,
                    "valley_preserving_operation_ids": [0, 1],
                },
                "operation_id": 1, "operation_order": 3,
                "matched_irrep": "C3_spinor_phase_+1/2",
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    # Aggregate provenance: both workflow paths recorded.
    assert set(inst["workflow_paths"]) == {"direct_qcut", "symmetry_adapted"}
    assert inst["workflow_path"] in ("direct_qcut", "symmetry_adapted")
    # Per-row provenance: each record carries its own workflow_path.
    gamma_records = inst["irrep_records_by_kpoint"]["GammaM"]
    row_paths = {rec["workflow_path"] for rec in gamma_records}
    assert row_paths == {"direct_qcut", "symmetry_adapted"}


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
                    "candidate_space_group_number": 143,
                    "valley_preserving_operation_ids": [0, 4],
                },
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
                    "candidate_space_group_number": 143,
                    "valley_preserving_operation_ids": [0, 4],
                },
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
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["status"] == "complete"
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["irreps_by_kpoint"]["GammaM"] == ["-GM6_a", "-GM6_a"]
    assert inst["irreps_by_kpoint"]["KM"] == ["-K5"]
    rec = inst["irrep_records_by_kpoint"]["GammaM"][0]
    assert rec["irrep_multiplicity"] == 2
    assert rec["matching_strategy"] == "bilbao_restricted_character"
    assert rec["subspace_space_group"]["candidate_space_group_symbol"] == "P3"


# -----------------------------------------------------------------------
# Commit 2 new tests: identity-only HSP, sampled-basis, provenance
# -----------------------------------------------------------------------

def test_identity_only_hsp_preserved_not_dropped():
    """Identity-only HSP little group carries trivial irrep — must not be dropped."""
    cands = {
        "candidates": [
            {
                "kpoint": "MM", "valley": "K_valley",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P3",
                    "candidate_space_group_number": 143,
                    "valley_preserving_operation_ids": [0],
                },
                "operation_id": 0, "operation_order": 1,
                "matched_irrep": "GM1",
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["status"] == "complete"
    # Identity-only HSP is valid — must appear in HSP lists.
    assert "MM" in inst["actual_hsps"]
    assert "MM" in inst["expected_hsps"]
    assert inst["hsp_basis_status"] == "sampled_basis"


def test_hsp_basis_status_is_sampled_basis():
    """All instances built from sampled candidates have hsp_basis_status=sampled_basis."""
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )
    inst = r["instances"][0]
    assert inst["hsp_basis_status"] == "sampled_basis"
    assert inst["expected_hsp_policy_source"] == "sampled_irrep_basis"


def test_aggregate_workflow_provenance():
    """Instance-level workflow_paths records all unique workflow paths."""
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )
    inst = r["instances"][0]
    assert isinstance(inst["workflow_paths"], list)
    assert len(inst["workflow_paths"]) >= 1
    assert isinstance(inst["readiness_evidence"], list)
    # Backward compat: single fields still present.
    assert isinstance(inst["workflow_path"], str)
    assert isinstance(inst["readiness_level"], str)


def test_inequivalent_settings_not_merged():
    """Same symbol but different SG numbers → separate instances."""
    cands = {
        "candidates": [
            {
                "kpoint": "GM", "valley": "V1",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P2",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P2",
                    "candidate_space_group_number": 3,
                    "valley_preserving_operation_ids": [0],
                },
                "operation_id": 0, "operation_order": 1,
                "matched_irrep": "GM1",
                "ready_for_ebr_input": True,
            },
            {
                "kpoint": "GM", "valley": "V1",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "P2",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P2",
                    "candidate_space_group_number": 5,  # different SG number
                    "valley_preserving_operation_ids": [0],
                },
                "operation_id": 0, "operation_order": 1,
                "matched_irrep": "GM1",
                "ready_for_ebr_input": True,
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    # Different SG numbers → two separate instances.
    assert r["instance_count"] == 2


def test_per_row_provenance_preserved():
    """Each irrep record carries its own workflow_path and readiness_level."""
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )
    inst = r["instances"][0]
    for kp, records in inst["irrep_records_by_kpoint"].items():
        for rec in records:
            assert "workflow_path" in rec
            assert "readiness_level" in rec
            assert isinstance(rec["workflow_path"], str)
            assert isinstance(rec["readiness_level"], str)


def test_candidates_group_into_instances_exact_count():
    """C3-preserving candidates: exactly 1 instance, 2 HSPs."""
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["candidate_count"] == 4  # 4 specific candidate rows
    assert inst["status"] == "complete"
    assert inst["ready_for_ebr_decomposition"] is True
    assert "GammaM" in inst["irreps_by_kpoint"]
    assert "KM" in inst["irreps_by_kpoint"]
    assert inst["expected_hsps"] == ["GammaM", "KM"]
    assert inst["actual_hsps"] == ["GammaM", "KM"]
    assert inst["hsp_basis_status"] == "sampled_basis"
