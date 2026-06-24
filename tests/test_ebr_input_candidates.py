import json

from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_tmo_te2_like_inputs():
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
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "1": {
                        "matching_status": "matched",
                        "matched_irrep": "C3_spinor_phase_+1/2",
                        "operation_order": 3,
                        "subspace_group_candidate": "P3",
                        "eigenphases": [0.5],
                        "readiness_level": "trusted",
                        "workflow_path": "direct_qcut",
                    },
                },
            },
        },
    }
    sa = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [{
                                "operation_id": 1,
                                "character": {"real": -1.0, "imag": 0.0},
                            }],
                        },
                    },
                }],
            },
        },
    }
    return decisions, matching, sa


# -----------------------------------------------------------------------
# 1. Trusted matched rows become candidates
# -----------------------------------------------------------------------

def test_trusted_matched_becomes_candidate():
    decisions, matching, sa = _make_tmo_te2_like_inputs()
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
        symmetry_adapted_valley_report=sa,
    )
    assert r["status"] == "has_candidates"
    assert r["candidate_count"] == 1
    assert r["blocked_count"] == 0
    c = r["candidates"][0]
    assert c["ready_for_ebr_input"] is True
    assert c["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert c["character"] == {"real": -1.0, "imag": 0.0}


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
        "by_kpoint": {
            "MM": {
                "M3_valley": {
                    "5": {
                        "matching_status": "diagnostic_only",
                        "matched_irrep": None,
                        "operation_order": 2,
                        "subspace_group_candidate": "P2",
                        "eigenphases": [-0.25, 0.25],
                    },
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
    assert "matching_status=diagnostic_only" in b["reason"]


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
        "by_kpoint": {
            "MM": {
                "M3_valley": {
                    "5": {
                        "matching_status": "failed_ambiguous",
                        "matched_irrep": None,
                        "operation_order": 2,
                        "subspace_group_candidate": "P2",
                        "eigenphases": [-0.25, 0.25],
                    },
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
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "1": {
                        "matching_status": "matched",
                        "diagnostic_only": True,
                        "matched_irrep": "C3_spinor_phase_+1/2",
                        "operation_order": 3,
                    },
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
        "by_kpoint": {
            "MM": {
                "K_valley": {
                    "1": {
                        "matching_status": "blocked",
                        "matched_irrep": None,
                    },
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
    matching = {"by_kpoint": {}}
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
    )
    assert r["candidate_count"] == 0
    assert "no irrep matching results" in r["blocked"][0]["reason"]


# -----------------------------------------------------------------------
# 6. JSON serializability
# -----------------------------------------------------------------------

def test_json_serializable():
    decisions, matching, sa = _make_tmo_te2_like_inputs()
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
        symmetry_adapted_valley_report=sa,
    )
    encoded = json.dumps(r)
    assert len(encoded) > 0
    assert "dtype" not in encoded


# -----------------------------------------------------------------------
# 7. Schema
# -----------------------------------------------------------------------

def test_schema_fields_present():
    decisions, matching, sa = _make_tmo_te2_like_inputs()
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
        symmetry_adapted_valley_report=sa,
    )
    keys = {"status", "candidate_count", "blocked_count",
            "reduced_ebr_decomposition_status", "by_kpoint",
            "candidates", "blocked", "interpretation"}
    assert keys <= set(r)
    assert r["reduced_ebr_decomposition_status"] == "not_implemented"


def test_no_forbidden_terms():
    decisions, matching, sa = _make_tmo_te2_like_inputs()
    r = build_ebr_input_candidates(
        irrep_workflow_decisions=decisions,
        valley_irrep_matching=matching,
        symmetry_adapted_valley_report=sa,
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
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "1": {
                        "matching_status": "matched",
                        "matched_irrep": "C3_spinor_phase_+1/2",
                        "operation_order": 3,
                    },
                },
                "Kp_valley": {
                    "1": {
                        "matching_status": "diagnostic_only",
                        "matched_irrep": None,
                        "operation_order": 3,
                    },
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
        "by_kpoint": {},
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
