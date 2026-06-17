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

def test_legacy_phase_table_still_works():
    """Existing C3 phase-table matching still produces matched_irrep."""
    from valleyscope.analysis.valley_irrep_matching import match_valley_irrep
    result = match_valley_irrep(
        eigenphases=[0.5],
        operation_order=3,
        subspace_group_candidate="C3_like",
        readiness_level="trusted",
    )
    assert result["matching_status"] == "matched"
    assert result["matched_irrep"] == "C3_spinor_phase_+1/2"


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
