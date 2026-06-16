"""Tests for generic valley-preserving restricted-character irrep matching."""

import pytest
from valleyscope.analysis.generic_irrep_matching import match_restricted_characters


# -----------------------------------------------------------------------
# C3-like toy example (cyclic group of order 3)
# -----------------------------------------------------------------------

def test_c3_toy_cyclic_matches_irrep_multiplicities():
    """C3-like toy: one computed irrep with eigenphase +1/2 matches a source
    irrep with character -1 on the C3 operation."""
    # ValleyScope computed char: identity=1, C3=-1 (phase +1/2), C3^2=-1
    computed = {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j}
    # Source irrep table: one 1D irrep with C3 char = -1
    source = {
        "-GM5": {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j},
    }
    vp_ids = [1, 2, 3]
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=vp_ids,
    )
    assert result["matching_status"] == "matched"
    assert result["matching_strategy"] == "bilbao_restricted_character"
    assert not result["diagnostic_only"]
    assert result["irrep_multiplicities"] == {"-GM5": 1}


def test_c3_toy_multiplicity_two():
    """Two copies of the same irrep give multiplicity 2."""
    computed = {1: 2.0 + 0j, 2: -2.0 + 0j, 3: -2.0 + 0j}
    source = {
        "-GM5": {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j},
    }
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2, 3],
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"-GM5": 2}


def test_c3_toy_reducible_matches_two_irreps():
    """A reducible computed representation decomposes into two distinct
    source irreps (+1/6 and -1/6 C3 phases)."""
    import cmath
    computed = {
        1: 2.0 + 0j,
        2: 1.0 + 0j,   # exp(+i*pi/3) + exp(-i*pi/3) = 1
        3: 1.0 + 0j,
    }
    source = {
        "-K6_a": {1: 1.0 + 0j, 2: cmath.exp(1j * cmath.pi / 3), 3: cmath.exp(-1j * cmath.pi / 3)},
        "-K6_b": {1: 1.0 + 0j, 2: cmath.exp(-1j * cmath.pi / 3), 3: cmath.exp(1j * cmath.pi / 3)},
    }
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2, 3],
    )
    assert result["matching_status"] == "matched"
    mult = result["irrep_multiplicities"]
    assert mult.get("-K6_a") == 1
    assert mult.get("-K6_b") == 1


# -----------------------------------------------------------------------
# C4-like toy example (cyclic group of order 4, non-C3 non-C2)
# -----------------------------------------------------------------------

def test_c4_toy_cyclic_matches():
    """C4-like toy: computed irrep with C4 phase +1/4 matches source."""
    import cmath
    i = 1j
    computed = {
        1: 1.0 + 0j,
        2: i,
        3: -1.0 + 0j,
        4: -i,
    }
    source = {
        "-GM_plus_1over4": {1: 1.0 + 0j, 2: i, 3: -1.0 + 0j, 4: -i},
    }
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2, 3, 4],
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"-GM_plus_1over4": 1}


# -----------------------------------------------------------------------
# Two-generator abelian toy (Z2 x Z2)
# -----------------------------------------------------------------------

def test_z2xz2_abelian_toy_matches():
    """Two-generator abelian subgroup: matches a 1D irrep distinguishing
    two independent C2 operations."""
    computed = {1: 1.0 + 0j, 4: -1.0 + 0j, 5: 1.0 + 0j}
    source = {
        "-M3": {1: 1.0 + 0j, 4: -1.0 + 0j, 5: 1.0 + 0j},
    }
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 4, 5],
    )
    assert result["matching_status"] == "matched"
    assert result["irrep_multiplicities"] == {"-M3": 1}


# -----------------------------------------------------------------------
# Incomplete / blocked cases
# -----------------------------------------------------------------------

def test_empty_vp_ids_blocked():
    """Empty valley-preserving operation list blocks matching."""
    result = match_restricted_characters(
        computed_characters={1: 1.0 + 0j},
        source_irrep_characters={"A": {1: 1.0 + 0j}},
        valley_preserving_operation_ids=[],
    )
    assert result["matching_status"] == "blocked"


def test_no_computed_characters_diagnostic():
    """Missing computed characters is diagnostic-only."""
    result = match_restricted_characters(
        computed_characters={},
        source_irrep_characters={"A": {1: 1.0 + 0j}},
        valley_preserving_operation_ids=[1],
    )
    assert result["matching_status"] == "diagnostic"


def test_no_common_operations_diagnostic():
    """Non-overlapping operation sets produce diagnostic on all irreps."""
    computed = {1: 1.0 + 0j, 2: -1.0 + 0j, 3: 1.0 + 0j}
    source = {"X": {7: 1.0 + 0j, 8: -1.0 + 0j}}
    result = match_restricted_characters(
        computed_characters=computed,
        source_irrep_characters=source,
        valley_preserving_operation_ids=[1, 2, 3],
    )
    assert result["matching_status"] == "diagnostic"
    assert "no_matched" in result.get("reason", "")


def test_non_integer_multiplicity_diagnostic():
    """Non-integer inner product is diagnostic-only."""
    computed = {1: 1.0 + 0j, 2: 0.5 + 0j}
    source = {"X": {1: 1.0 + 0j, 2: -1.0 + 0j}}
    # With only 2 ops, inner product = (1*1 + (-1)*0.5)/2 = 0.25 -> rounds to 0
    # -> passes as matched with multiplicity 0 -> no_matched_irreps
    # Use 3 ops for a clear non-integer case.
    computed2 = {1: 1.0 + 0j, 2: 0.3 + 0j, 3: 0.3 + 0j}
    source2 = {"X": {1: 1.0 + 0j, 2: -1.0 + 0j, 3: -1.0 + 0j}}
    result = match_restricted_characters(
        computed_characters=computed2,
        source_irrep_characters=source2,
        valley_preserving_operation_ids=[1, 2, 3],
    )
    assert result["matching_status"] == "diagnostic"


# -----------------------------------------------------------------------
# Legacy phase-table compatibility
# -----------------------------------------------------------------------

def test_legacy_phase_table_still_works():
    """Existing C3 phase-table matching still produces matched_irrep when
    called without generic source data (legacy fallback)."""
    from valleyscope.analysis.valley_irrep_matching import match_valley_irrep

    # C3_spinor_phase_+1/2 has eigenphase [0.5] * 2pi
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
