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


def test_c2_rank2_plus_minus():
    r = match_valley_irrep(
        eigenphases=[-0.25, 0.25],
        operation_order=2,
        subspace_group_candidate="C2_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "matched"
    assert "+1/4&-1/4" in r["matched_irrep"]


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
    assert r["matching_status"] == "not_applicable"


def test_unsupported_rank_not_applicable():
    r = match_valley_irrep(
        eigenphases=[0.1, 0.2, 0.3],
        operation_order=3,
        subspace_group_candidate="C3_like",
        readiness_level="trusted",
    )
    assert r["matching_status"] == "not_applicable"


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
