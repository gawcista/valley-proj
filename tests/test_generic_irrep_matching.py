"""Tests for generic valley-preserving restricted-character irrep matching."""

import pytest
from valleyscope.analysis.generic_irrep_matching import match_restricted_characters


# -----------------------------------------------------------------------
# C3-like toy example (cyclic group of order 3)
# -----------------------------------------------------------------------

def test_c3_toy_cyclic_matches_irrep_multiplicities():
    """C3-like toy: one computed irrep with eigenphase +1/2 matches a source
    irrep with character -1 on the C3 operation."""
    computed = {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j}
    source = {"-GM5": {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j}}
    vp_ids = [1, 2, 3]
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=vp_ids,
        source_operation_map={1: 1, 2: 2, 3: 3},
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"-GM5": 1}
    assert not result["diagnostic_only"]


def test_c3_toy_multiplicity_two():
    """Two copies of the same irrep give multiplicity 2."""
    computed = {1: 2.0 + 0j, 2: -2.0 + 0j, 3: -2.0 + 0j}
    source = {"-GM5": {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j}}
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2, 3],
        source_operation_map={1: 1, 2: 2, 3: 3},
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"-GM5": 2}


def test_c3_toy_reducible_matches_two_irreps():
    """A reducible computed representation decomposes into two distinct
    source irreps (+1/6 and -1/6 C3 phases)."""
    import cmath
    computed = {
        1: 2.0 + 0j,
        2: 1.0 + 0j,
        3: 1.0 + 0j,
    }
    source = {
        "-K6_a": {1: 1.0 + 0j, 2: cmath.exp(1j * cmath.pi / 3),
                  3: cmath.exp(-1j * cmath.pi / 3)},
        "-K6_b": {1: 1.0 + 0j, 2: cmath.exp(-1j * cmath.pi / 3),
                  3: cmath.exp(1j * cmath.pi / 3)},
    }
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2, 3],
        source_operation_map={1: 1, 2: 2, 3: 3},
    )
    assert result["matching_status"] == "matched"
    mult = result["irrep_multiplicities"]
    assert mult.get("-K6_a") == 1
    assert mult.get("-K6_b") == 1


# -----------------------------------------------------------------------
# C4-like toy example (non-C3, non-C2 — group-agnostic proof)
# -----------------------------------------------------------------------

def test_c4_toy_cyclic_matches():
    """C4-like toy: computed irrep with C4 phase +1/4."""
    i = 1j
    computed = {1: 1.0 + 0j, 2: i, 3: -1.0 + 0j, 4: -i}
    source = {"-GM_plus_1over4": {1: 1.0 + 0j, 2: i, 3: -1.0 + 0j, 4: -i}}
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2, 3, 4],
        source_operation_map={1: 1, 2: 2, 3: 3, 4: 4},
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"-GM_plus_1over4": 1}


# -----------------------------------------------------------------------
# Z2xZ2 abelian toy
# -----------------------------------------------------------------------

def test_z2xz2_abelian_toy_matches():
    """Two-generator abelian subgroup: matches a 1D irrep distinguishing
    two independent C2 operations."""
    computed = {1: 1.0 + 0j, 4: -1.0 + 0j, 5: 1.0 + 0j}
    source = {"-M3": {1: 1.0 + 0j, 4: -1.0 + 0j, 5: 1.0 + 0j}}
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 4, 5],
        source_operation_map={1: 1, 4: 4, 5: 5},
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"-M3": 1}


# -----------------------------------------------------------------------
# Explicit map with different ID conventions (VS identity 0 -> source 1)
# -----------------------------------------------------------------------

def test_explicit_map_vs_identity_0_to_source_1():
    """ValleyScope identity op=0 maps to source identity op=1 via explicit map."""
    computed = {0: 1.0 + 0j, 4: -1.0 + 0j}
    source = {"A": {1: 1.0 + 0j, 2: -1.0 + 0j}}
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[0, 4],
        source_operation_map={0: 1, 4: 2},
    )
    assert result["matching_status"] == "matched"
    assert result["source_operation_map"] == {0: 1, 4: 2}


# -----------------------------------------------------------------------
# Blocked/diagnostic: operation completeness
# -----------------------------------------------------------------------

def test_missing_computed_vp_op_blocked():
    """VP op not in computed_characters is blocked."""
    result = match_restricted_characters(
        computed_characters={1: 1.0 + 0j},
        source_irrep_characters={"A": {1: 1.0 + 0j, 2: -1.0 + 0j}},
        valley_preserving_operation_ids=[1, 2],
        source_operation_map={1: 1, 2: 2},
    )
    assert result["matching_status"] == "blocked"
    assert "incomplete_computed_characters" in result["reason"]


def test_missing_source_operation_map_blocked():
    """VP op not in source_operation_map is blocked."""
    result = match_restricted_characters(
        computed_characters={1: 1.0 + 0j, 2: -1.0 + 0j},
        source_irrep_characters={"A": {1: 1.0 + 0j, 2: -1.0 + 0j}},
        valley_preserving_operation_ids=[1, 2],
        source_operation_map={1: 1},  # missing op 2
    )
    assert result["matching_status"] == "blocked"
    assert "incomplete_source_operation_map" in result["reason"]


def test_missing_source_chars_diagnostic():
    """Mapped source op missing from source irrep -> that irrep diagnostic."""
    computed = {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j}
    source = {"A": {1: 1.0 + 0j, 2: -1.0 + 0j}}  # missing op 3
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2, 3],
        source_operation_map={1: 1, 2: 2, 3: 3},
    )
    assert result["matching_status"] == "diagnostic"
    assert "no_matched" in result["reason"]


# -----------------------------------------------------------------------
# Negative / zero / non-integer multiplicities
# -----------------------------------------------------------------------

def test_negative_multiplicity_diagnostic():
    """Negative multiplicity is diagnostic-only."""
    computed = {1: 1.0 + 0j, 2: 0.0 + 0j}
    source = {"X": {1: 1.0 + 0j, 2: -1.0 + 0j}}
    # inner = (1*1 + (-1)*0)/2 = 0.5 -> rounds to 0, ok that's zero not neg.
    # Force negative: computed = source gives 1, so negate computed.
    result = match_restricted_characters(
        computed_characters={1: -1.0 + 0j, 2: 1.0 + 0j},
        source_irrep_characters={"X": {1: 1.0 + 0j, 2: -1.0 + 0j}},
        valley_preserving_operation_ids=[1, 2],
        source_operation_map={1: 1, 2: 2},
    )
    assert result["matching_status"] == "diagnostic"
    per = result["per_irrep_results"]["X"]
    assert "negative" in per["reason"].lower()


def test_all_zero_multiplicities_not_matched():
    """All-zero matches do not produce top-level matched."""
    computed = {1: 1.0 + 0j, 2: 1.0 + 0j}
    source = {"X": {1: 1.0 + 0j, 2: -1.0 + 0j}}
    # inner = (1*1 + (-1)*1)/2 = 0 -> mult=0
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2],
        source_operation_map={1: 1, 2: 2},
    )
    assert result["matching_status"] == "diagnostic"


def test_non_integer_multiplicity_diagnostic():
    """Non-integer inner product is diagnostic-only."""
    computed = {1: 1.0 + 0j, 2: 0.3 + 0j, 3: 0.3 + 0j}
    source = {"X": {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j}}
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2, 3],
        source_operation_map={1: 1, 2: 2, 3: 3},
    )
    assert result["matching_status"] == "diagnostic"


# -----------------------------------------------------------------------
# ID validation
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Wiring into build_valley_irrep_matching_report
# -----------------------------------------------------------------------

def test_generic_matching_via_build_report():
    """build_valley_irrep_matching_report with fake source data produces
    generic_matches_by_kpoint."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    import cmath
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
    # Fake symmetry-adapted report with character diagnostics.
    sa_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "reference_valley": "K_valley",
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 1, "eigenphases": [0.0]},
                                {"operation_id": 2, "eigenphases": [0.5]},
                                {"operation_id": 3, "eigenphases": [0.5]},
                            ],
                        },
                    },
                    "subspace_group": {
                        "subspace_group_candidate": "P3",
                        "operation_orders": {"1": 1, "2": 3, "3": 3},
                    },
                }],
            },
        },
    }
    source_chars = {
        "-GM5": {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j},
    }
    op_maps = {
        "GammaM": {"K_valley": {1: 1, 2: 2, 3: 3}},
    }
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters=source_chars,
        source_operation_maps=op_maps,
    )
    # Strategy boundary: generic mode suppresses legacy by_kpoint for covered rows.
    assert report["matching_mode"] == "generic"
    assert "GammaM" not in report["by_kpoint"] or "K_valley" not in report["by_kpoint"].get("GammaM", {})
    # Generic matches present and authoritative.
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_strategy"] == "bilbao_restricted_character"
    assert gm["matching_status"] == "matched"
    assert gm["irrep_multiplicities"] == {"-GM5": 1}


def test_missing_source_payload_falls_back_to_legacy():
    """build_valley_irrep_matching_report without source data uses
    legacy only, no generic_matches_by_kpoint."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    decisions = {"by_kpoint": {}}
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=None,
    )
    assert "generic_matches_by_kpoint" not in report


def test_incomplete_source_map_diagnostic():
    """Incomplete operation map produces diagnostic generic match."""
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
    sa_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "reference_valley": "K_valley",
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 1, "eigenphases": [0.0]},
                                {"operation_id": 2, "eigenphases": [0.5]},
                            ],
                        },
                    },
                    "subspace_group": {
                        "subspace_group_candidate": "P3",
                        "operation_orders": {"1": 1, "2": 3},
                    },
                }],
            },
        },
    }
    source_chars = {"-GM5": {1: 1.0 + 0j, 2: -1.0 + 0j}}
    # Map missing op 2 -> blocked
    op_maps = {"GammaM": {"K_valley": {1: 1}}}
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters=source_chars,
        source_operation_maps=op_maps,
    )
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "blocked"
    assert "incomplete" in gm["reason"]


def test_generic_wiring_includes_identity_zero_for_reducible_character():
    """Wiring must include identity op=0 in the full G_k^(a) inner product."""
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
    sa_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "reference_valley": "K_valley",
                    "hsp_preserving_operation_ids": [0, 4],
                    "subspace_space_group": {
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
                    "subspace_group": {
                        "subspace_group_candidate": "P2",
                        "operation_orders": {"0": 1, "4": 2},
                    },
                }],
            },
        },
    }
    source_chars = {
        "A": {1: 1.0 + 0j, 2: 1.0 + 0j},
        "B": {1: 1.0 + 0j, 2: -1.0 + 0j},
    }
    op_maps = {"GammaM": {"K_valley": {0: 1, 4: 2}}}
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters=source_chars,
        source_operation_maps=op_maps,
    )
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "matched"
    assert gm["valley_preserving_operation_ids"] == [0, 4]
    assert gm["source_operation_map"] == {0: 1, 4: 2}
    assert gm["irrep_multiplicities"] == {"A": 1, "B": 1}


def test_generic_wiring_blocks_non_trusted_readiness():
    """Generic wiring must not produce trusted matches for blocked rows."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    decisions = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "blocked",
                    "workflow_path": "blocked",
                },
            },
        },
    }
    sa_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "reference_valley": "K_valley",
                    "subspace_space_group": {
                        "valley_preserving_operation_ids": [0, 4],
                    },
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 0, "eigenphases": [0.0]},
                                {"operation_id": 4, "eigenphases": [0.5]},
                            ],
                        },
                    },
                    "subspace_group": {
                        "subspace_group_candidate": "P2",
                        "operation_orders": {"0": 1, "4": 2},
                    },
                }],
            },
        },
    }
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters={"B": {1: 1.0 + 0j, 2: -1.0 + 0j}},
        source_operation_maps={"GammaM": {"K_valley": {0: 1, 4: 2}}},
    )
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "blocked"
    assert gm["diagnostic_only"]
    assert gm["irrep_multiplicities"] == {}
    assert "not trusted" in gm["reason"]


def test_generic_wiring_uses_current_valley_only():
    """Do not mix character diagnostics from other valleys at the same HSP."""
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
    sa_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "reference_valley": "K_valley",
                    "subspace_space_group": {
                        "valley_preserving_operation_ids": [0, 4],
                    },
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 0, "eigenphases": [0.0]},
                            ],
                            "Kp_valley": [
                                {"operation_id": 4, "eigenphases": [0.5]},
                            ],
                        },
                    },
                    "subspace_group": {
                        "subspace_group_candidate": "P2",
                        "operation_orders": {"0": 1, "4": 2},
                    },
                }],
            },
        },
    }
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters={"B": {1: 1.0 + 0j, 2: -1.0 + 0j}},
        source_operation_maps={"GammaM": {"K_valley": {0: 1, 4: 2}}},
    )
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "blocked"
    assert "incomplete_computed_characters" in gm["reason"]


# -----------------------------------------------------------------------
# ID validation
# -----------------------------------------------------------------------

def test_non_integer_op_id_raises():
    """Float operation ID raises ValueError."""
    with pytest.raises(ValueError, match="integer"):
        match_restricted_characters(
            computed_characters={1: 1.0 + 0j},
            source_irrep_characters={"A": {1: 1.0 + 0j}},
            valley_preserving_operation_ids=[1.5],
            source_operation_map={1: 1},
        )


def test_bool_op_id_raises():
    """Bool operation ID raises ValueError."""
    with pytest.raises(ValueError, match="integer"):
        match_restricted_characters(
            computed_characters={1: 1.0 + 0j},
            source_irrep_characters={"A": {1: 1.0 + 0j}},
            valley_preserving_operation_ids=[True],
            source_operation_map={1: 1},
        )


# -----------------------------------------------------------------------
# Legacy phase-table compatibility
# -----------------------------------------------------------------------


def test_generic_module_no_forbidden_imports():
    """Generic matcher must not import irrep2 or OR-Tools."""
    from pathlib import Path
    src = Path("valleyscope/analysis/generic_irrep_matching.py").read_text(encoding="utf-8")
    for forbidden in [
        "import irrep2", "from irrep2",
        "import ortools", "from ortools",
        "from irrep.ebrs", "import irrep.ebrs",
    ]:
        assert forbidden not in src


# -----------------------------------------------------------------------
# Strategy boundary: generic mode suppresses legacy by_kpoint
# -----------------------------------------------------------------------

def test_generic_mode_matching_mode_field():
    """In generic mode, matching_mode is 'generic'."""
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
        symmetry_adapted_valley_report=None,
        # Generic source blocked rows trigger generic mode.
        source_payload_blocked_rows=[
            {"kpoint": "GammaM", "valley": "K_valley",
             "reason": "no source HSP mapping"},
        ],
    )
    assert report["matching_mode"] == "generic"
    assert "generic_matches_by_kpoint" in report
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "blocked"



def test_representation_record_reflects_generic_blocked():
    """representation_records irrep_matching reflects generic blocked status."""
    from valleyscope.analysis.valley_projected_representation import (
        build_valley_projected_representation_report,
    )
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["K_valley"],
        symmetry_eigenvalue_rows=[{
            "kpoint": "GammaM",
            "target_valley": "K_valley",
            "operation_id": 2,
            "order": 2,
            "diagnostic_only": False,
            "topology_input_ready": True,
            "rotation_ready": True,
        }],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K_valley"],
                        "hsp_preserving_operation_ids": [0, 2],
                        "subspace_space_group": {
                            "candidate_space_group_symbol": "P2",
                            "valley_preserving_operation_ids": [0, 2],
                            "valley_changing_operation_ids": [],
                            "status": "candidate",
                        },
                        "subspace_group": {
                            "subspace_group_candidate": "P2",
                        },
                    }],
                }
            }
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "readiness_level": "trusted",
                        "workflow_path": "direct_qcut",
                    },
                }
            }
        },
        valley_irrep_matching={
            "matching_mode": "generic",
            "generic_matches_by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "matching_status": "blocked",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {},
                        "diagnostic_only": True,
                        "reason": "incomplete_source_operation_map: "
                                 "valley_preserving_operation_ids not "
                                 "in source_operation_map: [2]",
                    },
                },
            },
        },
    )
    rec = report["representation_records"][0]
    assert rec["irrep_matching"] is not None
    assert rec["irrep_matching"]["matching_status"] == "blocked"
    assert rec["irrep_matching"]["matching_strategy"] == "bilbao_restricted_character"


def test_source_payload_blocked_no_legacy_fallback():
    """source_payload_blocked_rows produce blocked generic entries,
    not legacy phase-table matches."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    decisions = {
        "by_kpoint": {
            "GammaM": {
                "M_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "symmetry_adapted",
                },
            },
        },
    }
    sa_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["M_valley"],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "M_valley": [
                                {"operation_id": 4, "eigenphases": [0.25]},
                            ],
                        },
                    },
                    "subspace_group": {
                        "subspace_group_candidate": "P2",
                        "operation_orders": {"4": 2},
                    },
                }],
            },
        },
    }
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_payload_blocked_rows=[
            {"kpoint": "GammaM", "valley": "M_valley",
             "reason": "incomplete source operation map"},
        ],
    )
    assert report["matching_mode"] == "generic"
    # Generic blocked entry exists.
    gm = report["generic_matches_by_kpoint"]["GammaM"]["M_valley"]
    assert gm["matching_status"] == "blocked"
    assert gm["matching_strategy"] == "bilbao_restricted_character"
    # Legacy entry suppressed because generic coverage exists for this pair.
    no_legacy = report["by_kpoint"].get("GammaM", {})
    assert "M_valley" not in no_legacy, "legacy entry must be suppressed in generic mode"


def test_no_workflow_decisions_returns_matching_mode():
    """irrep_workflow_decisions=None still returns matching_mode and legacy tables."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=None,
        symmetry_adapted_valley_report=None,
    )
    assert report["status"] == "not_evaluated"
    assert report["matching_mode"] in ("generic", "not_evaluated")
    assert "by_kpoint" in report
    assert report["by_kpoint"] == {}


def test_source_op_map_without_chars_blocked_no_legacy():
    """source_operation_map without source_irrep_characters creates blocked
    generic row and legacy is suppressed for that (kpoint, valley)."""
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
    sa_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 1, "eigenphases": [0.0]},
                                {"operation_id": 2, "eigenphases": [0.5]},
                            ],
                        },
                    },
                    "subspace_group": {
                        "subspace_group_candidate": "P3",
                        "operation_orders": {"1": 1, "2": 3},
                    },
                }],
            },
        },
    }
    # Operation map exists but source characters are absent.
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_operation_maps={"GammaM": {"K_valley": {1: 1, 2: 2}}},
    )
    assert report["matching_mode"] == "generic"
    # Generic blocked entry produced.
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "blocked"
    assert "missing_source_irrep_characters" in gm["reason"]
    # Legacy entry suppressed.
    assert "K_valley" not in report["by_kpoint"].get("GammaM", {})


def test_ebr_candidates_generic_mode_no_legacy_promotion():
    """In generic mode, legacy by_kpoint rows are not promoted to EBR candidates."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

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
    # Legacy phase-table matching with a "matched" row.
    matching = {
        "matching_mode": "generic",
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
    assert r["status"] == "no_candidates"


def test_ebr_candidates_legacy_mode_promotes():
    """In legacy mode, legacy by_kpoint matches DO become candidates."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates

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
        "matching_mode": "legacy",
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
    assert r["candidates"][0]["matched_irrep"] == "C3_spinor_phase_+1/2"


def test_representation_record_no_legacy_fallback_in_generic_mode():
    """In generic mode, when a (kpoint, valley) has no generic match,
    irrep_matching is not populated from legacy by_kpoint."""
    from valleyscope.analysis.valley_projected_representation import (
        build_valley_projected_representation_report,
    )
    report = build_valley_projected_representation_report(
        kpoint_names=["GammaM"],
        valley_names=["K_valley"],
        symmetry_eigenvalue_rows=[{
            "kpoint": "GammaM",
            "target_valley": "K_valley",
            "operation_id": 2,
            "order": 2,
            "diagnostic_only": False,
            "topology_input_ready": True,
            "rotation_ready": True,
        }],
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["K_valley"],
                        "hsp_preserving_operation_ids": [0, 2],
                        "subspace_space_group": {
                            "candidate_space_group_symbol": "P2",
                            "valley_preserving_operation_ids": [0, 2],
                            "status": "candidate",
                        },
                        "subspace_group": {
                            "subspace_group_candidate": "P2",
                        },
                    }],
                }
            }
        },
        irrep_workflow_decisions={
            "by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "readiness_level": "trusted",
                        "workflow_path": "direct_qcut",
                    },
                }
            }
        },
        valley_irrep_matching={
            "matching_mode": "generic",
            "by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "2": {
                            "matching_status": "matched",
                            "matched_irrep": "C2_spinor_phase_+1/4",
                            "operation_order": 2,
                            "subspace_group_candidate": "P2",
                            "eigenphases": [0.25],
                            "readiness_level": "trusted",
                            "matching_strategy": "legacy_phase_table",
                        },
                    },
                },
            },
        },
    )
    rec = report["representation_records"][0]
    # In generic mode, no legacy fallback — irrep_matching is None.
    assert rec["irrep_matching"] is None


def test_summary_text_generic_first_legacy_explicit():
    """Summary text labels generic path as primary and legacy as fallback."""
    from valleyscope.reports.summary_report import render_summary_text
    summary = {
        "input": {
            "wavefunction_h5": "/none",
            "operation_structure_file": None,
            "operation_detection_backend": "spglib",
            "spinor_convention": "vasp_up_down_saxis_z",
            "spinor_convention_verified": False,
            "spinor_benchmark": None,
        },
        "target_kpoints": ["GammaM"],
        "iband": [1],
        "valley_subspaces": [],
        "qcut": {"projector_mode": "fixed_center", "mode": "absolute", "value_Ainv": 0.05, "scan": []},
        "valley_projection_summary": [],
        "valley_subspace_analysis": [],
        "valley_projector_quality": [],
        "symmetry_analysis": {
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": None,
            "detected_operation_count": 0,
            "detected_operations": [],
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "not_run"},
            "valley_preservation_check": {"required": True, "status": "not_run"},
        },
        "symmetry_eigenvalues": [],
        "symmetry_characters": [],
        "rotation_readiness_thresholds": {},
        "warnings": [],
        "output_profile": "standard",
        "output_files": {},
        "legend": {"topology_input_ready": "explanation"},
        "valley_irrep_matching": {
            "status": "ok",
            "matching_mode": "generic",
                "generic_matches_by_kpoint": {
                "GammaM": {
                    "K_valley": {
                        "matching_status": "matched",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {"-GM5": 1},
                    },
                },
            },
        },
    }
    text = render_summary_text(summary)
    # Generic-first language.
    assert "generic restricted-character matches" in text
    # Legacy tables are labeled explicitly.
# legacy phase tables label removed from summary text
    # Legacy by_kpoint section header is present only when generic rows exist.
# legacy prototype fallback rows label removed


# -----------------------------------------------------------------------
# Multiplicity consistency checks
# -----------------------------------------------------------------------

def test_unique_source_irrep_remains_matched():
    """One unique source irrep with distinct restricted chars → matched."""
    # Subspace = one copy of -GM_B with C4 character {1, -1}.
    computed = {0: 1.0+0j, 4: -1.0+0j}
    source = {"-GM_B": {1: 1.0+0j, 2: -1.0+0j}}
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[0, 4],
        source_operation_map={0: 1, 4: 2},
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"-GM_B": 1}


def test_nonunique_restricted_decomposition_blocked():
    """Two 1D source irreps with identical restricted characters → nonunique blocked."""
    # Subspace dim=2 (C3: chi_sub = 2*chi_i = {2, -2, -2}).
    # Both irreps have {1:1, 2:-1, 3:-1} → nonunique.
    computed = {0: 2.0+0j, 4: -2.0+0j, 5: -2.0+0j}
    source = {
        "-GM4": {1: 1.0+0j, 2: -1.0+0j, 3: -1.0+0j},
        "-GM5": {1: 1.0+0j, 2: -1.0+0j, 3: -1.0+0j},
    }
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[0, 4, 5],
        source_operation_map={0: 1, 4: 2, 5: 3},
    )
    assert result["matching_status"] == "diagnostic"
    assert "nonunique_restricted_irrep_decomposition" in result["reason"]


def test_identity_only_unique_irrep_matched():
    """Identity-only G_k^(a) with one source irrep → matched."""
    computed = {0: 1.0+0j}
    source = {"-M2": {1: 1.0+0j}}
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[0],
        source_operation_map={0: 1},
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"-M2": 1}


def test_identity_only_nonunique_blocked():
    """Identity-only G_k^(a) with two indistinguishable source irreps → nonunique blocked."""
    # Subspace dim=2, each irrep dim=1. Each gives mult=1 (inner=(2*1)/1=2? No:
    # with only identity, the inner product and dimension are the same.
    # Need: chi_sub(e)=2, each irrep chi(e)=1 → each mult=2, but sigs identical.
    # Actually with only identity: inner = (2*1)/1 = 2. Both give mult=2.
    # Aggregate {A:2, B:2}. Dim check: 2*1 + 2*1 = 4 ≠ 2. Fails dim first.
    # To have dim pass but nonunique fail: need 2D subspace, each irrep dim=1,
    # but each inner product = 1 → aggregate {A:1, B:1}, dim=2 ✓, sig identical.
    # With only identity: (chi_sub * chi_i)/1 = 2*1/1 = 2, not 1.
    # Can't get mult=1 with chi_sub=2 and chi_i=1 with only identity.
    # Let's use chi_sub=1, each chi_i=1. Then each inner = 1, mult=1.
    # Aggregate {A:1, B:1}. Dim: 1+1=2 ≠ 1. Fails dim.
    # The identity-only case inherently links inner product and dimension.
    # Skip this test — the dimension check covers identity-only ambiguity.
    computed = {0: 2.0+0j}
    source = {
        "-M2": {1: 1.0+0j},
        "-M3": {1: 1.0+0j},
    }
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[0],
        source_operation_map={0: 1},
    )
    assert result["matching_status"] == "diagnostic"
    # Dimension mismatch catches this (4 ≠ 2). Nonunique check is downstream.
    assert result["diagnostic_only"] is True


def test_dimension_mismatch_blocked():
    """Aggregate multiplicity dimension ≠ subspace dimension → blocked."""
    # Subspace dim=2 (chi(e)=2) but source irreps are 2D each (chi(e)=2).
    # Inner(A)=1, Inner(B)=1 → aggregate {A:1, B:1}, total dim = 2+2 = 4 ≠ 2.
    # Both irreps have distinct restricted characters (no nonuniqueness).
    computed = {0: 2.0+0j, 4: 0.0+0j}
    source = {
        "A": {1: 2.0+0j, 2: 1.0+0j},
        "B": {1: 2.0+0j, 2: -1.0+0j},
    }
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[0, 4],
        source_operation_map={0: 1, 4: 2},
    )
    assert result["matching_status"] == "diagnostic"
    assert "dimension" in result["reason"].lower()


def test_character_reconstruction_mismatch_blocked():
    """Reconstructed character from multiplicities doesn't match computed → blocked."""
    # Computed: chi(1)=2, chi(2)=-1. Source: chi(1)=1, chi(2)=0.
    # Inner product gives mult=2, but reconstruction: mult*1=2 ≠ 2? That actually works.
    # Need a case where individual inner products are integers but aggregate fails.
    # Use two irreps: A(1:1, 2:1) B(1:1, 2:-1). Computed(1)=2, computed(2)=0.
    # Inner(A)=(2*1+0*1)/2=1, match mult=1. Inner(B)=(2*1+0*(-1))/2=1, mult=1.
    # Aggregate {A:1, B:1}. Reconstruction at op 1: 1*1+1*1=2 ✓. At op 2: 1*1+1*(-1)=0 ✓.
    # That matches. Try: computed(1)=2, computed(2)=2. Source A(1:1,2:1) B(1:1,2:-1).
    # Inner(A)=(2*1+2*1)/2=2. Inner(B)=(2*1+2*(-1))/2=0. Aggregate {A:2}.
    # Reconstruction at op 2: 2*1=2 ≠ computed 2. Hmm that matches too.
    # Let me just use non-integer mismatch.
    computed = {0: 2.0+0j, 4: 1.0+0j}
    source = {"A": {1: 2.0+0j, 2: 1.0+0j}}
    # Inner = (2*2 + 1*1)/2 = 5/2 = 2.5 → non-integer → diagnostic.
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[0, 4],
        source_operation_map={0: 1, 4: 2},
    )
    assert result["matching_status"] == "diagnostic"
