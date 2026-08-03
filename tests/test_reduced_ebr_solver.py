"""Tests for ValleyScope's reduced-dimensional EBR solver API."""

from pathlib import Path

import pytest

from valleyscope.analysis.reduced_ebr_solver import (
    check_integer_span,
    classify_bundle,
    derive_coefficient_bounds,
    search_nonnegative_witnesses,
)


def test_classifier_returns_exact_nonnegative_decomposition():
    result = classify_bundle(
        target=[2, 1, 1],
        ebr_vectors=[[1, 0, 1], [1, 1, 0]],
        ebr_labels=["EBR_A", "EBR_B"],
        max_coefficient=6,
    )
    assert result["status"] == "solved_exact"
    assert result["classification"] == "atomic-compatible-candidate"
    assert result["ebr_decomposition"] == [
        {"label": "EBR_A", "coefficient": 1},
        {"label": "EBR_B", "coefficient": 1},
    ]


def test_solver_rejects_misaligned_vector_lengths():
    with pytest.raises(ValueError, match="vector length"):
        classify_bundle(
            target=[1, 0],
            ebr_vectors=[[1]],
            ebr_labels=["EBR_A"],
            max_coefficient=6,
        )


def test_solver_rejects_misaligned_ebr_labels():
    with pytest.raises(ValueError, match="ebr_labels length"):
        classify_bundle(
            target=[1, 0],
            ebr_vectors=[[1, 0], [0, 1]],
            ebr_labels=["EBR_A"],
            max_coefficient=6,
        )


def test_solver_rejects_negative_target_or_ebr_entries():
    with pytest.raises(ValueError, match="target"):
        classify_bundle([-1, 0], [[1, 0]], ["EBR_A"], 6)
    with pytest.raises(ValueError, match="EBR vector"):
        classify_bundle([1, 0], [[1, -1]], ["EBR_A"], 6)


def test_integer_span_bounds_and_witness_search():
    in_span, signed = check_integer_span([0, 1], [[1, 0], [1, 1]])
    assert in_span is True
    assert signed == [-1, 1]
    assert derive_coefficient_bounds([2, 3], [[1, 0], [0, 1]]) == [2, 3]
    assert search_nonnegative_witnesses(
        [2, 3],
        [[1, 0], [0, 1]],
        [2, 3],
        max_witnesses=1,
    ) == [[2, 3]]


def test_reduced_ebr_solver_has_no_forbidden_dependencies_or_material_names():
    src = Path("valleyscope/analysis/reduced_ebr_solver.py").read_text(
        encoding="utf-8"
    )
    for forbidden in [
        "from irrep.ebrs",
        "import irrep.ebrs",
        "import ortools",
        "from ortools",
        "import irrep2",
        "from irrep2",
        "tMoTe2",
        "tZrSe2",
        "MoTe2",
        "ZrSe2",
    ]:
        assert forbidden not in src
