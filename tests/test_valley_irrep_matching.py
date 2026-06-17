import json
import pytest

from valleyscope.analysis.valley_irrep_matching import (
    match_valley_irrep,
    build_valley_irrep_matching_report,
)


# -----------------------------------------------------------------------
# C3 phase matching
# -----------------------------------------------------------------------

def test_c3_phase_plus_one_sixth():
    r = match_valley_irrep(
        eigenphases=[1.0 / 6.0],
        operation_order=3,
        subspace_group_candidate="C3_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "matched"
    assert "1/6" in r["matched_irrep"]


def test_c3_phase_plus_one_half():
    r = match_valley_irrep(
        eigenphases=[0.5],
        operation_order=3,
        subspace_group_candidate="C3_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "matched"
    assert "1/2" in r["matched_irrep"]


def test_c3_phase_minus_one_sixth():
    r = match_valley_irrep(
        eigenphases=[-1.0 / 6.0],
        operation_order=3,
        subspace_group_candidate="C3_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "matched"
    assert "-1/6" in r["matched_irrep"]


def test_c3_modulo_wrapping_near_boundary():
    # Phase = +5/6 ~ -1/6
    r = match_valley_irrep(
        eigenphases=[5.0 / 6.0],
        operation_order=3,
        subspace_group_candidate="C3_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "matched"
    assert "-1/6" in r["matched_irrep"]


def test_c3_phase_wrap_near_minus_half():
    # Phase near +1/2 modulo
    r = match_valley_irrep(
        eigenphases=[-0.5],
        operation_order=3,
        subspace_group_candidate="C3_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "matched"
    assert "1/2" in r["matched_irrep"]


# -----------------------------------------------------------------------
# C2 phase matching
# -----------------------------------------------------------------------

def test_c2_phase_plus_one_quarter():
    r = match_valley_irrep(
        eigenphases=[0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "matched"
    assert "1/4" in r["matched_irrep"]


def test_c2_phase_minus_one_quarter():
    r = match_valley_irrep(
        eigenphases=[-0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "matched"
    assert "-1/4" in r["matched_irrep"]


def test_rank2_plus_minus_is_ambiguous_not_single_irrep():
    r = match_valley_irrep(
        eigenphases=[-0.25, 0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "failed_ambiguous"
    assert r["matched_irrep"] is None


# -----------------------------------------------------------------------
# Ambiguous / missing cases
# -----------------------------------------------------------------------

def test_unknown_phase_no_match():
    r = match_valley_irrep(
        eigenphases=[0.123],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "failed_no_table"
    assert r["matched_irrep"] is None


def test_empty_phases_not_applicable():
    r = match_valley_irrep(
        eigenphases=[],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "not_applicable"


def test_unsupported_order_not_applicable():
    r = match_valley_irrep(
        eigenphases=[0.1],
        operation_order=4,
        subspace_group_candidate="C4_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "failed_no_table"


def test_unsupported_rank_failed_ambiguous():
    r = match_valley_irrep(
        eigenphases=[0.1, 0.2, 0.3],
        operation_order=3,
        subspace_group_candidate="C3_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "failed_ambiguous"


def test_candidate_mismatch_does_not_match_by_order_only():
    r = match_valley_irrep(
        eigenphases=[0.25],
        operation_order=2,
        subspace_group_candidate="C3_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "failed_no_table"
    assert r["matched_irrep"] is None


def test_missing_candidate_does_not_match_by_order_only():
    r = match_valley_irrep(
        eigenphases=[0.25],
        operation_order=2,
        subspace_group_candidate=None,
        readiness_level="trusted",
    )
    assert r["matching_status"] == "failed_no_table"
    assert r["matched_irrep"] is None


# -----------------------------------------------------------------------
# Readiness gates
# -----------------------------------------------------------------------

def test_trusted_can_match():
    r = match_valley_irrep(
        eigenphases=[0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "matched"


def test_usable_with_caution_is_diagnostic_only():
    r = match_valley_irrep(
        eigenphases=[0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="usable_with_caution",
    )
    assert r["matching_status"] == "diagnostic_only"


def test_usable_with_caution_allow_caution_matches():
    r = match_valley_irrep(
        eigenphases=[0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="usable_with_caution",
        allow_caution=True,
    )
    assert r["matching_status"] == "matched"


def test_blocked_is_blocked():
    r = match_valley_irrep(
        eigenphases=[0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="blocked",
    )
    assert r["matching_status"] == "blocked"


def test_unknown_readiness_is_blocked():
    r = match_valley_irrep(
        eigenphases=[0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="unknown",
    )
    assert r["matching_status"] == "blocked"


# -----------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------

def test_build_matching_report_empty():
    r = build_valley_irrep_matching_report(
        irrep_workflow_decisions=None,
        symmetry_adapted_valley_report=None,
    )
    assert r["status"] == "not_evaluated"


def test_build_matching_report_with_data():
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
    sa = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "subspace_group": {"subspace_group_candidate": "C3_like", "operation_orders": {"1": 3}},
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [{
                                "operation_id": 1,
                                
                                "eigenphases": [0.5],
                            }],
                        },
                    },
                }],
            },
        },
    }
    r = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa,
    )
    assert "GammaM" in r["by_kpoint"]
    assert "K_valley" in r["by_kpoint"]["GammaM"]
    match = r["by_kpoint"]["GammaM"]["K_valley"]["1"]
    assert match["matching_status"] == "matched"
    assert "1/2" in match["matched_irrep"]


def test_build_matching_report_missing_operation_order_does_not_default_to_c2():
    decisions = {
        "by_kpoint": {
            "MM": {
                "M_valley": {
                    "workflow_path": "symmetry_adapted",
                    "readiness_level": "trusted",
                },
            },
        },
    }
    sa = {
        "by_kpoint": {
            "MM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["M_valley"],
                    "subspace_group": {"subspace_group_candidate": "C2_like"},
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "M_valley": [{
                                "operation_id": 5,
                                "eigenphases": [0.25],
                            }],
                        },
                    },
                }],
            },
        },
    }
    r = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa,
    )
    match = r["by_kpoint"]["MM"]["M_valley"]["5"]
    assert match["matching_status"] == "failed_no_table"
    assert "missing operation order" in match["reason"]


# -----------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------

def test_schema_json_serializable():
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
    sa = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "subspace_group": {"subspace_group_candidate": "C3_like", "operation_orders": {"1": 3}},
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [{
                                "operation_id": 1,
                                
                                "eigenphases": [1.0/6.0],
                            }],
                        },
                    },
                }],
            },
        },
    }
    r = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa,
    )
    encoded = json.dumps(r)
    assert len(encoded) > 0
    assert "dtype" not in encoded


def test_schema_no_forbidden_terms():
    r = match_valley_irrep(
        eigenphases=[0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="trusted",
    )
    encoded = json.dumps(r)
    for forbidden in [
        "covariance", "equivariance", "stabilizer", "valley_little_group",
    ]:
        assert forbidden not in encoded.lower()


def test_flattened_per_row_path_matches_with_identity():
    """Flattened path with identity [0, 4] returns multiplicities {A:1, B:1}."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    decisions = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    sa = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "reference_valley": "K_valley",
                    "orbit": ["K_valley"],
                    "subspace_group": {
                        "subspace_group_candidate": "C3_like",
                    },
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P3",
                        "valley_preserving_operation_ids": [0, 4],
                    },
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                {"operation_id": 4, "eigenphases": [0.0, 0.5]},
                            ],
                        },
                    },
                }],
            },
        },
    }
    src_chars = {"A": {1: 1.0 + 0j, 2: 1.0 + 0j}, "B": {1: 1.0 + 0j, 2: -1.0 + 0j}}
    op_maps = {"GammaM": {"K_valley": {0: 1, 4: 2}}}
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa,
        source_irrep_characters_flattened={
            "GammaM": {"K_valley": src_chars},
        },
        source_operation_maps=op_maps,
    )
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "matched"
    assert gm["irrep_multiplicities"] == {"A": 1, "B": 1}
    assert gm["diagnostic_only"] is False
    assert gm["subspace_group_candidate"] == "C3_like"
    assert gm["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
    assert gm["workflow_path"] == "direct_qcut"
    assert gm["readiness_level"] == "trusted"


def test_source_payload_blocked_row_surfaces_in_generic_matches():
    """Blocked source payload rows must remain visible to downstream review."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    decisions = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report={"by_kpoint": {}},
        source_payload_blocked_rows=[{
            "kpoint": "GammaM",
            "valley": "K_valley",
            "source_hsp_label": "GM",
            "table_sg_number": 150,
            "table_spinor": True,
            "valley_preserving_operation_ids": [0, 4],
            "blocker_reasons": [
                "table_operation_matching_failed: no mapped VP op",
            ],
        }],
    )
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "blocked"
    assert gm["matching_strategy"] == "bilbao_restricted_character"
    assert gm["diagnostic_only"] is True
    assert gm["irrep_multiplicities"] == {}
    assert gm["valley_preserving_operation_ids"] == [0, 4]
    assert "table_operation_matching_failed" in gm["reason"]
    assert gm["source_payload_provenance"]["source_hsp_label"] == "GM"
