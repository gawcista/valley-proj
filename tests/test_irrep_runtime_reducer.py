"""Tests for irrep runtime source -> ValleyScope reduced external table reducer."""

import json
import pytest
from pathlib import Path

from valleyscope.analysis.irrep_runtime_reducer import (
    build_reduced_table_from_runtime_source,
)
from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table

_SAMPLE_BASIS = [
    {"source_label": "K5@Gamma", "hsp": "GammaM",
     "valleyscope_irrep_key": "GammaM:C3_spinor_phase_+1/2"},
    {"source_label": "K6@K", "hsp": "KM",
     "valleyscope_irrep_key": "KM:C3_spinor_phase_+1/6"},
    {"source_label": "K6@K'", "hsp": "KM",
     "valleyscope_irrep_key": "KM:C3_spinor_phase_-1/6"},
    # Extra 3D HSP rows not in sampled HSPs — should be filtered out.
    {"source_label": "A@H", "hsp": "A",
     "valleyscope_irrep_key": "A:C1_spinor"},
]

_SAMPLE_EBRS = [
    {"label": "EBR_A", "vector": [1, 0, 1, 1]},
    {"label": "EBR_B", "vector": [1, 1, 0, 0]},
]

_HSP = ["GammaM", "KM"]
_KEYS = ["GammaM:C3_spinor_phase_+1/2", "KM:C3_spinor_phase_+1/6", "KM:C3_spinor_phase_-1/6"]


def _source_payload(basis=_SAMPLE_BASIS, ebrs=_SAMPLE_EBRS, source=None):
    return {
        "basis": basis,
        "ebrs": ebrs,
        "source": source or {"package": "irrep", "version": "fake-test"},
    }


# -----------------------------------------------------------------------
# 1. Happy path
# -----------------------------------------------------------------------

def test_happy_path_reduces_3d_to_sampled_hsps():
    """Extra 3D HSP rows are filtered; sampled HSP rows are preserved."""
    result = build_reduced_table_from_runtime_source(
        source_payload=_source_payload(),
        expected_hsps=_HSP,
        allowed_irrep_keys=_KEYS,
        subspace_group_candidate="C3_like",
    )
    assert result["subspace_group_candidate"] == "C3_like"
    assert result["expected_hsps"] == ["GammaM", "KM"]
    assert result["irreps"] == _KEYS
    # Vectors reduced from 4 -> 3 (A@H filtered out).
    for ebr in result["ebrs"]:
        assert len(ebr["vector"]) == 3


def test_happy_path_output_passes_load_reduced_ebr_table(tmp_path):
    """Reducer output is accepted by the existing table validator."""
    result = build_reduced_table_from_runtime_source(
        source_payload=_source_payload(),
        expected_hsps=_HSP,
        allowed_irrep_keys=_KEYS,
        subspace_group_candidate="C3_like",
    )
    path = tmp_path / "table.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    loaded = load_reduced_ebr_table(path)
    assert loaded["subspace_group_candidate"] == "C3_like"
    assert loaded["irreps"] == _KEYS


# -----------------------------------------------------------------------
# 2. Missing/empty basis
# -----------------------------------------------------------------------

def test_empty_basis_raises():
    with pytest.raises(ValueError, match="basis"):
        build_reduced_table_from_runtime_source(
            source_payload=_source_payload(basis=[], ebrs=[]),
            expected_hsps=_HSP, allowed_irrep_keys=_KEYS,
            subspace_group_candidate="C3_like",
        )


def test_no_basis_matches_hsp_and_keys_raises():
    with pytest.raises(ValueError, match="no source basis entries match"):
        build_reduced_table_from_runtime_source(
            source_payload=_source_payload(),
            expected_hsps=["ZZZ"],
            allowed_irrep_keys=["XXX:irrep"],
            subspace_group_candidate="C1",
        )


def test_partial_missing_allowed_irrep_key_mapping_raises():
    """Every trusted valley-preserving irrep key must map to a source basis row."""
    with pytest.raises(ValueError, match="missing source basis mapping"):
        build_reduced_table_from_runtime_source(
            source_payload=_source_payload(basis=_SAMPLE_BASIS[:-2]),
            expected_hsps=_HSP,
            allowed_irrep_keys=_KEYS,
            subspace_group_candidate="C3_like",
        )


def test_expected_hsp_order_is_preserved_in_output():
    """HSP order is an explicit reduced-basis contract, not an alphabetical side effect."""
    result = build_reduced_table_from_runtime_source(
        source_payload=_source_payload(),
        expected_hsps=["KM", "GammaM"],
        allowed_irrep_keys=_KEYS,
        subspace_group_candidate="C3_like",
    )
    assert result["expected_hsps"] == ["KM", "GammaM"]
    assert result["provenance"]["expected_hsps"] == ["KM", "GammaM"]


def test_allowed_irrep_key_order_controls_reduced_basis_order():
    """External source ordering must not define ValleyScope's reduced basis order."""
    shuffled_basis = [_SAMPLE_BASIS[2], _SAMPLE_BASIS[0], _SAMPLE_BASIS[3], _SAMPLE_BASIS[1]]
    shuffled_ebrs = [
        {"label": "EBR_A", "vector": [1, 1, 1, 0]},
        {"label": "EBR_B", "vector": [0, 1, 0, 1]},
    ]
    result = build_reduced_table_from_runtime_source(
        source_payload=_source_payload(basis=shuffled_basis, ebrs=shuffled_ebrs),
        expected_hsps=_HSP,
        allowed_irrep_keys=_KEYS,
        subspace_group_candidate="C3_like",
    )
    assert result["irreps"] == _KEYS
    assert result["ebrs"][0]["vector"] == [1, 0, 1]
    assert result["ebrs"][1]["vector"] == [1, 1, 0]


# -----------------------------------------------------------------------
# 3. Duplicate basis keys rejected
# -----------------------------------------------------------------------

def test_duplicate_basis_key_rejected():
    dup_basis = [
        {"source_label": "l1", "hsp": "GammaM",
         "valleyscope_irrep_key": "GammaM:C3_spinor_phase_+1/2"},
        {"source_label": "l2", "hsp": "GammaM",
         "valleyscope_irrep_key": "GammaM:C3_spinor_phase_+1/2"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        build_reduced_table_from_runtime_source(
            source_payload=_source_payload(basis=dup_basis, ebrs=[{"label": "X", "vector": [1, 0]}]),
            expected_hsps=["GammaM"],
            allowed_irrep_keys=["GammaM:C3_spinor_phase_+1/2"],
            subspace_group_candidate="C3_like",
        )


# -----------------------------------------------------------------------
# 4. EBR vector length mismatch
# -----------------------------------------------------------------------

def test_ebr_vector_length_mismatch_raises():
    bad_ebrs = [{"label": "X", "vector": [1]}]
    with pytest.raises(ValueError, match="vector length"):
        build_reduced_table_from_runtime_source(
            source_payload=_source_payload(ebrs=bad_ebrs),
            expected_hsps=_HSP, allowed_irrep_keys=_KEYS,
            subspace_group_candidate="C3_like",
        )


# -----------------------------------------------------------------------
# 5. Reduced zero-vector EBR rejected
# -----------------------------------------------------------------------

def test_reduced_zero_vector_skipped_with_provenance():
    """EBR with all weight in filtered-out HSPs is skipped and recorded."""
    basis = [
        {"source_label": "only_in_sampled", "hsp": "GammaM",
         "valleyscope_irrep_key": "GammaM:C3_spinor_phase_+1/2"},
        {"source_label": "only_in_extra", "hsp": "A",
         "valleyscope_irrep_key": "A:C1_spinor"},
    ]
    # Ghost has weight only at A (non-sampled) -> reduced [0] — skipped.
    # Real has weight at GammaM (sampled) -> kept.
    ebrs = [
        {"label": "Ghost", "vector": [0, 1]},
        {"label": "Real", "vector": [1, 0]},
    ]
    result = build_reduced_table_from_runtime_source(
        source_payload=_source_payload(basis=basis, ebrs=ebrs),
        expected_hsps=["GammaM"],
        allowed_irrep_keys=["GammaM:C3_spinor_phase_+1/2"],
        subspace_group_candidate="C3_like",
    )
    labels = {e["label"] for e in result["ebrs"]}
    assert labels == {"Real"}
    assert result["provenance"]["filtered_zero_vector_ebr_count"] == 1
    assert result["provenance"]["filtered_zero_vector_ebrs"] == ["Ghost"]


# -----------------------------------------------------------------------
# 6. Provenance
# -----------------------------------------------------------------------

def test_provenance_included_in_output():
    provenance = {"package": "irrep", "version": "2.6.3",
                  "detected_space_group": "P321", "space_group_number": 150}
    result = build_reduced_table_from_runtime_source(
        source_payload=_source_payload(),
        expected_hsps=_HSP, allowed_irrep_keys=_KEYS,
        subspace_group_candidate="C3_like",
        provenance=provenance,
    )
    assert result["provenance"]["package"] == "irrep"
    assert result["provenance"]["version"] == "2.6.3"
    assert result["provenance"]["detected_space_group"] == "P321"
    assert result["provenance"]["space_group_number"] == 150
    assert result["provenance"]["expected_hsps"] == _HSP
    assert result["provenance"]["source_basis_count"] == 4
    assert result["provenance"]["reduction_basis_count"] == 3


# -----------------------------------------------------------------------
# 7. No irrep2 / irrep imports in reducer
# -----------------------------------------------------------------------

def test_reducer_no_irrep2_import():
    src = Path("valleyscope/analysis/irrep_runtime_reducer.py").read_text(encoding="utf-8")
    # Check for actual import statements, not docstring references.
    for forbidden in ["import irrep2", "from irrep2"]:
        assert forbidden not in src, f"reducer must not import irrep2"
    assert "import irrep " not in src and "from irrep " not in src, (
        "reducer should not import irrep package yet"
    )


# -----------------------------------------------------------------------
# 8. No raw 3D decomposition fields
# -----------------------------------------------------------------------

def test_output_has_no_raw_3d_decomposition_fields():
    result = build_reduced_table_from_runtime_source(
        source_payload=_source_payload(),
        expected_hsps=_HSP, allowed_irrep_keys=_KEYS,
        subspace_group_candidate="C3_like",
    )
    encoded = json.dumps(result)
    for forbidden in ["decomposition", "3d_ebr", "raw_ebr", "irrep_decomposition"]:
        assert forbidden not in encoded.lower(), f"output must not contain '{forbidden}'"


# -----------------------------------------------------------------------
# 9. Nonnegative integer vectors preserved
# -----------------------------------------------------------------------

def test_ebr_vectors_preserve_exact_nonnegative_integers():
    result = build_reduced_table_from_runtime_source(
        source_payload=_source_payload(),
        expected_hsps=_HSP, allowed_irrep_keys=_KEYS,
        subspace_group_candidate="C3_like",
    )
    ebr_a = result["ebrs"][0]
    assert ebr_a["label"] == "EBR_A"
    assert ebr_a["vector"] == [1, 0, 1]  # [1,0,1,1] -> reduced [1,0,1]
    ebr_b = result["ebrs"][1]
    assert ebr_b["vector"] == [1, 1, 0]  # [1,1,0,0] -> reduced [1,1,0]


# -----------------------------------------------------------------------
# 10. No material names
# -----------------------------------------------------------------------

def test_reducer_no_material_names():
    src = Path("valleyscope/analysis/irrep_runtime_reducer.py").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src, f"reducer must not contain {name!r}"
