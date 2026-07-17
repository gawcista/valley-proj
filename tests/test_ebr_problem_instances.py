import json

from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
from tests.reduced_ebr_promo_helpers import real_primitive_certificate_dict


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


def _complete_coverage(*, valley="K_valley", mapping=None):
    mapping = mapping or {"GM": "GammaM", "K": "KM"}
    required = list(mapping)
    return {
        "by_valley": {
            valley: {
                "required_source_hsp_labels": required,
                "covered_source_hsp_labels": required,
                "missing_source_hsp_labels": [],
                "trusted_matched_source_hsp_labels": required,
                "trusted_missing_source_hsp_labels": [],
                "source_hsp_to_sampled_kpoint": dict(mapping),
                "complete": True,
                "ready_for_ebr_promotion": True,
                "source_basis_provenance": {"data_source": "irreptables"},
            }
        }
    }


# -----------------------------------------------------------------------
# 1. Complete canonical HSP vector
# -----------------------------------------------------------------------

def test_candidates_group_into_canonical_hsp_vector():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
        projected_hsp_coverage=_complete_coverage(),
    )
    assert r["status"] == "canonical_hsp_vectors_ready"
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["subspace_group_candidate"] == "P3"
    assert inst["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
    assert inst["valley"] == "K_valley"
    # State 1: sampled basis, not ready for decomposition.
    assert inst["status"] == "canonical_hsp_vector_ready"
    assert inst["canonical_hsp_vector_complete"] is True
    assert "hsp_basis_status" not in inst
    assert "ready_for_reduced_table_validation" not in inst
    assert "ready_for_ebr_decomposition" not in inst
    assert "GammaM" in inst["irreps_by_kpoint"]
    assert "KM" in inst["irreps_by_kpoint"]
    assert inst["expected_hsps"] == ["GammaM", "KM"]
    assert inst["expected_hsp_policy_source"] == "certified_source_hsp_basis"
    assert inst["optional_hsps"] == []
    assert inst["missing_optional_hsps"] == []
    assert "certificate_identity" in inst


# -----------------------------------------------------------------------
# 2. Partial HSP data
# -----------------------------------------------------------------------

def test_single_hsp_without_certified_coverage_is_incomplete():
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
    assert inst["status"] == "incomplete_canonical_hsp_vector"
    assert inst["canonical_hsp_vector_complete"] is False
    assert inst["expected_hsps"] == ["GammaM"]
    assert inst["expected_hsp_policy_source"] == "certified_source_hsp_basis"
    assert "projected_hsp_coverage_missing" in inst["blocked_by"]
    assert inst["subspace_sg_number"] == 143


def test_missing_coverage_fails_closed_for_canonical_vector():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )

    inst = r["instances"][0]
    assert inst["status"] == "incomplete_canonical_hsp_vector"
    assert inst["canonical_hsp_vector_complete"] is False
    assert "projected_hsp_coverage_missing" in inst["blocked_by"]


def test_complete_but_untrusted_coverage_has_explicit_blocker():
    coverage = _complete_coverage()
    coverage["by_valley"]["K_valley"]["ready_for_ebr_promotion"] = False
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
        projected_hsp_coverage=coverage,
    )

    inst = r["instances"][0]
    assert inst["canonical_hsp_vector_complete"] is False
    assert "source_hsp_coverage_not_ready_for_ebr_promotion" in (
        inst["blocked_by"]
    )


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
    assert r["status"] == "no_canonical_hsp_vectors"


# -----------------------------------------------------------------------
# 4. P2 sampled-basis
# -----------------------------------------------------------------------

def test_p2_valley_preserving_canonical_vector():
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
    r = build_ebr_problem_instances(
        ebr_input_candidates=cands,
        projected_hsp_coverage=_complete_coverage(
            valley="M1_valley", mapping={"GM": "GammaM"}
        ),
    )
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["status"] == "canonical_hsp_vector_ready"
    assert inst["canonical_hsp_vector_complete"] is True
    assert inst["expected_hsps"] == ["GammaM"]


# -----------------------------------------------------------------------
# 5. Empty / null inputs
# -----------------------------------------------------------------------

def test_null_input():
    r = build_ebr_problem_instances(ebr_input_candidates=None)
    assert r["status"] == "no_canonical_hsp_vectors"
    assert r["instance_count"] == 0


def test_empty_candidates():
    r = build_ebr_problem_instances(
        ebr_input_candidates={"status": "no_candidates", "candidates": []},
    )
    assert r["status"] == "no_canonical_hsp_vectors"


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
        projected_hsp_coverage=_complete_coverage(),
    )
    inst = r["instances"][0]
    for key in ["instance_id", "valley", "subspace_group_candidate",
                "subspace_sg_number", "subspace_space_group",
                "workflow_path", "workflow_paths",
                "readiness_level", "readiness_evidence",
                "irreps_by_kpoint", "operations_by_kpoint",
                "candidate_count", "status",
                "canonical_hsp_vector_complete",
                "certificate_identity",
                "blocked_by",
                "expected_hsps", "optional_hsps", "actual_hsps",
                "missing_optional_hsps"]:
        assert key in inst, f"missing key: {key}"
    assert "reduced_ebr_decomposition_status" not in r


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
    r = build_ebr_problem_instances(
        ebr_input_candidates=cands,
        projected_hsp_coverage=_complete_coverage(),
    )
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
    assert set(inst["workflow_paths"]) == {"direct_qcut", "symmetry_adapted"}
    assert inst["workflow_path"] == "direct_qcut"
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
    r = build_ebr_problem_instances(
        ebr_input_candidates=cands,
        projected_hsp_coverage=_complete_coverage(),
    )
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["status"] == "canonical_hsp_vector_ready"
    assert inst["canonical_hsp_vector_complete"] is True
    assert inst["irreps_by_kpoint"]["GammaM"] == ["-GM6_a", "-GM6_a"]
    assert inst["irreps_by_kpoint"]["KM"] == ["-K5"]
    rec = inst["irrep_records_by_kpoint"]["GammaM"][0]
    assert rec["irrep_multiplicity"] == 2
    assert rec["matching_strategy"] == "bilbao_restricted_character"
    assert rec["subspace_space_group"]["candidate_space_group_symbol"] == "P3"


# -----------------------------------------------------------------------
# Identity-only HSP, sampled-basis, provenance
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
    r = build_ebr_problem_instances(
        ebr_input_candidates=cands,
        projected_hsp_coverage=_complete_coverage(
            mapping={"M": "MM"}
        ),
    )
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    assert inst["status"] == "canonical_hsp_vector_ready"
    assert "MM" in inst["actual_hsps"]
    assert "MM" in inst["expected_hsps"]
    assert inst["canonical_hsp_vector_complete"] is True


def test_obsolete_hsp_basis_status_is_absent():
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )
    inst = r["instances"][0]
    assert "hsp_basis_status" not in inst
    assert inst["expected_hsp_policy_source"] == "certified_source_hsp_basis"


def test_aggregate_workflow_provenance():
    """Instance-level workflow_paths records all unique workflow paths."""
    r = build_ebr_problem_instances(
        ebr_input_candidates=_make_c3_preserving_candidates(),
    )
    inst = r["instances"][0]
    assert isinstance(inst["workflow_paths"], list)
    assert inst["workflow_paths"] == ["direct_qcut"]
    assert isinstance(inst["readiness_evidence"], list)
    assert isinstance(inst["workflow_path"], str)
    assert isinstance(inst["readiness_level"], str)


# -----------------------------------------------------------------------
# Certificate-aware identity key tests
# -----------------------------------------------------------------------

def test_same_sg_different_certificate_not_merged():
    """Same SG number, same symbol, same valley, but different Hall numbers
    (from different certificates) → separate instances."""
    cands = {
        "candidates": [
            {
                "kpoint": "GM", "valley": "V1",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C2",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "C2",
                    "candidate_space_group_number": 5,
                    "valley_preserving_operation_ids": [0],
                },
                "operation_id": 0, "operation_order": 1,
                "matched_irrep": "GM1",
                "ready_for_ebr_input": True,
                "irrep_source_provenance": {
                    "standard_setting_hsp_mapping": {
                        "standard_setting_certificate": {
                            "hall_number": 9,
                            "validation_status": "validated",
                        },
                    },
                },
            },
            {
                "kpoint": "GM", "valley": "V1",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C2",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "C2",
                    "candidate_space_group_number": 5,
                    "valley_preserving_operation_ids": [0],
                },
                "operation_id": 0, "operation_order": 1,
                "matched_irrep": "GM1",
                "ready_for_ebr_input": True,
                "irrep_source_provenance": {
                    "standard_setting_hsp_mapping": {
                        "standard_setting_certificate": {
                            "hall_number": 12,  # different Hall setting
                            "validation_status": "validated",
                        },
                    },
                },
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    # Different Hall numbers → two separate instances.
    assert r["instance_count"] == 2


def test_same_sg_same_certificate_merged():
    """Same SG number, symbol, valley, Hall number, certificate status → merged."""
    cands = {
        "candidates": [
            {
                "kpoint": "GM", "valley": "V1",
                "workflow_path": "direct_qcut", "readiness_level": "trusted",
                "subspace_group_candidate": "C2",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "C2",
                    "candidate_space_group_number": 5,
                    "valley_preserving_operation_ids": [0],
                },
                "operation_id": 0, "operation_order": 1,
                "matched_irrep": "GM1",
                "ready_for_ebr_input": True,
                "irrep_source_provenance": {
                    "standard_setting_hsp_mapping": {
                        "standard_setting_certificate": {
                            "hall_number": 9,
                            "validation_status": "validated",
                        },
                    },
                },
            },
            {
                "kpoint": "GM", "valley": "V1",
                "workflow_path": "symmetry_adapted", "readiness_level": "trusted",
                "subspace_group_candidate": "C2",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "C2",
                    "candidate_space_group_number": 5,
                    "valley_preserving_operation_ids": [0],
                },
                "operation_id": 0, "operation_order": 1,
                "matched_irrep": "GM1",
                "ready_for_ebr_input": True,
                "irrep_source_provenance": {
                    "standard_setting_hsp_mapping": {
                        "standard_setting_certificate": {
                            "hall_number": 9,
                            "validation_status": "validated",
                        },
                    },
                },
            },
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 1
    inst = r["instances"][0]
    ci = inst["certificate_identity"]
    assert ci["hall_numbers"] == [9]
    assert "validated" in ci["certificate_validation_statuses"]


def test_no_certificate_defaults_to_zero_hall():
    """Candidates without certificate data get hall_number=0, status=not_evaluated."""
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
        ],
    }
    r = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert r["instance_count"] == 1
    ci = r["instances"][0]["certificate_identity"]
    assert ci["hall_numbers"] == []
    assert "not_evaluated" in ci["certificate_validation_statuses"]
    assert ci["any_unresolved"] is True


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


def test_malformed_raw_operation_map_does_not_crash_instance_construction():
    cert = real_primitive_certificate_dict(143, "P3")
    assert cert is not None
    cert["affine_operation_map"] = {"0": "not-an-integer"}
    cands = {
        "candidates": [{
            "kpoint": "GM",
            "valley": "K_valley",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "subspace_group_candidate": "P3",
            "subspace_space_group": {
                "candidate_space_group_symbol": "P3",
                "candidate_space_group_number": 143,
                "valley_preserving_operation_ids": [0, 1, 2],
            },
            "operation_id": 0,
            "operation_order": 1,
            "matched_irrep": "GM1",
            "ready_for_ebr_input": True,
            "irrep_source_provenance": {
                "standard_setting_hsp_mapping": {
                    "standard_setting_certificate": cert,
                },
            },
        }],
    }
    report = build_ebr_problem_instances(ebr_input_candidates=cands)
    assert report["instance_count"] == 1
    identity = report["instances"][0]["certificate_identity"]
    assert identity["affine_operation_map"] is None
