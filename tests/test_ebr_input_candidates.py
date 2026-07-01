import json

from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_generic_inputs():
    """Shared generic-matching inputs: trusted K_valley @ GammaM."""
    decisions = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                },
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "matching_status": "matched",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {"C3_spinor_phase_+1/2": 1},
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P3",
                        "valley_preserving_operation_ids": [0, 1],
                        "status": "candidate",
                    },
                    "valley_preserving_operation_ids": [0, 1],
                    "hsp_little_group_operation_ids": [0, 1],
                    "source_operation_map": {0: 1, 1: 2},
                },
            },
        },
    }
    return decisions, matching


# -----------------------------------------------------------------------
# 1. Trusted matched rows become candidates
# -----------------------------------------------------------------------

def test_trusted_matched_becomes_candidate():
    decisions, matching = _make_generic_inputs()
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    assert r["status"] == "has_candidates"
    assert r["candidate_count"] == 1
    assert r["blocked_count"] == 0
    c = r["candidates"][0]
    assert c["ready_for_ebr_input"] is True
    assert c["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert c["matching_strategy"] == "bilbao_restricted_character"


# -----------------------------------------------------------------------
# 2. diagnostic / usable_with_caution does not become candidate
# -----------------------------------------------------------------------

def test_usable_with_caution_does_not_become_candidate():
    decisions = {
        "by_kpoint": {
            "MM": {
                "M3_valley": {
                    "workflow_path": "symmetry_adapted",
                    "readiness_level": "usable_with_caution",
                },
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "MM": {
                "M3_valley": {
                    "matching_status": "diagnostic",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {},
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P2",
                        "valley_preserving_operation_ids": [0, 5],
                    },
                    "valley_preserving_operation_ids": [0, 5],
                    "hsp_little_group_operation_ids": [0, 5],
                    "diagnostic_only": False,
                },
            },
        },
    }
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    assert r["candidate_count"] == 0
    assert r["blocked_count"] == 1
    b = r["blocked"][0]
    assert "readiness=usable_with_caution" in b["reason"]


# -----------------------------------------------------------------------
# 3. Ambiguous / missing matched_irrep does not become candidate
# -----------------------------------------------------------------------

def test_failed_ambiguous_not_candidate():
    decisions = {
        "by_kpoint": {
            "MM": {
                "M3_valley": {
                    "workflow_path": "symmetry_adapted",
                    "readiness_level": "trusted",
                },
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "MM": {
                "M3_valley": {
                    "matching_status": "failed_ambiguous",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {},
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P2",
                        "valley_preserving_operation_ids": [0, 5],
                    },
                    "valley_preserving_operation_ids": [0, 5],
                    "hsp_little_group_operation_ids": [0, 5],
                },
            },
        },
    }
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    assert r["candidate_count"] == 0
    assert r["blocked_count"] == 1
    assert "matching_status=failed_ambiguous" in r["blocked"][0]["reason"]


def test_diagnostic_only_flag_blocks_even_if_status_is_matched():
    decisions = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                },
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "matching_status": "matched",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {"C3_spinor_phase_+1/2": 1},
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P3",
                        "valley_preserving_operation_ids": [0, 1],
                    },
                    "valley_preserving_operation_ids": [0, 1],
                    "hsp_little_group_operation_ids": [0, 1],
                    "diagnostic_only": True,
                },
            },
        },
    }
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    assert r["candidate_count"] == 0
    assert r["blocked_count"] == 1
    assert "diagnostic_only=true" in r["blocked"][0]["reason"]


# -----------------------------------------------------------------------
# 4. Blocked readiness
# -----------------------------------------------------------------------

def test_blocked_readiness_not_candidate():
    decisions = {
        "by_kpoint": {
            "MM": {
                "K_valley": {
                    "workflow_path": "blocked",
                    "readiness_level": "blocked",
                },
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "MM": {
                "K_valley": {
                    "matching_status": "blocked",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {},
                    "subspace_space_group": {},
                    "valley_preserving_operation_ids": [],
                    "hsp_little_group_operation_ids": [],
                },
            },
        },
    }
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    assert r["candidate_count"] == 0
    assert "readiness=blocked" in r["blocked"][0]["reason"]


# -----------------------------------------------------------------------
# 5. No matching results
# -----------------------------------------------------------------------

def test_no_matching_results_blocked():
    """Empty generic_matches_by_kpoint produces no candidates and no blocked rows."""
    decisions = {
        "by_kpoint": {
            "KM": {
                "M1_valley": {
                    "workflow_path": "blocked",
                    "readiness_level": "blocked",
                },
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {},
    }
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    assert r["candidate_count"] == 0
    assert r["blocked_count"] == 0
    assert r["status"] == "no_candidates"


# -----------------------------------------------------------------------
# 6. JSON serializability
# -----------------------------------------------------------------------

def test_json_serializable():
    decisions, matching = _make_generic_inputs()
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    encoded = json.dumps(r)
    assert len(encoded) > 0
    assert "dtype" not in encoded


# -----------------------------------------------------------------------
# 7. Schema
# -----------------------------------------------------------------------

def test_schema_fields_present():
    decisions, matching = _make_generic_inputs()
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    keys = {"status", "candidate_count", "blocked_count",
            "reduced_ebr_decomposition_status", "by_kpoint",
            "candidates", "blocked", "interpretation"}
    assert keys <= set(r)
    assert r["reduced_ebr_decomposition_status"] == "not_implemented"


def test_no_forbidden_terms():
    decisions, matching = _make_generic_inputs()
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    encoded = json.dumps(r)
    for forbidden in ["covariance", "equivariance", "stabilizer",
                      "valley_little_group"]:
        assert forbidden not in encoded.lower(), f"forbidden: {forbidden}"


# -----------------------------------------------------------------------
# 8. Multiple valleys, mixed readiness
# -----------------------------------------------------------------------

def test_mixed_readiness():
    decisions = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                },
                "Kp_valley": {
                    "workflow_path": "symmetry_adapted",
                    "readiness_level": "usable_with_caution",
                },
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "matching_status": "matched",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {"C3_spinor_phase_+1/2": 1},
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P3",
                        "valley_preserving_operation_ids": [0, 1],
                    },
                    "valley_preserving_operation_ids": [0, 1],
                    "hsp_little_group_operation_ids": [0, 1],
                },
                "Kp_valley": {
                    "matching_status": "diagnostic",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {},
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P3",
                        "valley_preserving_operation_ids": [0, 1],
                    },
                    "valley_preserving_operation_ids": [0, 1],
                    "hsp_little_group_operation_ids": [0, 1],
                },
            },
        },
    }
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    assert r["candidate_count"] == 1
    assert r["blocked_count"] == 1
    assert r["candidates"][0]["valley"] == "K_valley"
    assert "readiness=usable_with_caution" in r["blocked"][0]["reason"]


# -----------------------------------------------------------------------
# 9. Empty inputs
# -----------------------------------------------------------------------

def test_null_inputs_gives_empty():
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=None,
        valley_irrep_matching=None,
    )
    assert r["status"] == "no_candidates"
    assert r["candidate_count"] == 0


def test_generic_matches_produce_ebr_candidates():
    """Generic matched multiplicities become EBR input candidates."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    decisions = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
        },
    }
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "matching_status": "matched",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {"-GM5": 1, "-GM6_a": 2},
                    "subspace_group_candidate": "P3",
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P3",
                        "valley_preserving_operation_ids": [0, 4],
                        "status": "candidate",
                    },
                    "source_operation_map": {0: 1, 4: 2},
                    "valley_preserving_operation_ids": [0, 4],
                    "diagnostic_only": False,
                    "reason": "",
                },
            },
        },
    }
    result = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    assert result["status"] == "has_candidates"
    assert result["candidate_count"] == 2
    cands = result["candidates"]
    by_label = {c["matched_irrep"]: c for c in cands}
    assert by_label["-GM5"]["irrep_multiplicity"] == 1
    assert by_label["-GM6_a"]["irrep_multiplicity"] == 2
    for c in cands:
        assert c["matching_strategy"] == "bilbao_restricted_character"
        assert c["ready_for_ebr_input"] is True
        assert c["subspace_group_candidate"] == "P3"
        assert c["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
