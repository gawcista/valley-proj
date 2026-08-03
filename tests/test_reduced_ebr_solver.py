"""Tests for ValleyScope's reduced-dimensional EBR solver API."""

from pathlib import Path

import pytest

from valleyscope.analysis.reduced_ebr_solver import (
    check_integer_span,
    classify_bundle,
    derive_coefficient_bounds,
    search_nonnegative_witnesses,
    _validate_nonnegative_integer_vector,
)


# Test-side compact wrappers over the production solver API.  Production
# never calls them, so they live in the test package instead of the solver
# module; they exercise the public classification/witness code paths.

def _validate_irrep_basis(irrep_basis):
    if not isinstance(irrep_basis, (list, tuple)):
        raise ValueError("irrep_basis must be a sequence")
    basis: list[str] = []
    seen: set[str] = set()
    for i, label in enumerate(irrep_basis):
        if not isinstance(label, str) or not label:
            raise ValueError(f"irrep_basis[{i}] must be a non-empty string")
        if label in seen:
            raise ValueError(f"duplicate reduced irrep basis label {label!r}")
        basis.append(label)
        seen.add(label)
    if not basis:
        raise ValueError("irrep_basis must be non-empty")
    return basis


def get_reduced_ebr_matrix(table):
    """Return reduced EBR matrix columns from a validated table-like mapping."""
    ebrs = table.get("ebrs", [])
    vectors: list[list[int]] = []
    expected_length: int | None = None
    for i, ebr in enumerate(ebrs):
        vector = list(ebr.get("vector", []))
        _validate_nonnegative_integer_vector(vector, field=f"EBR vector {i}")
        if expected_length is None:
            expected_length = len(vector)
        elif len(vector) != expected_length:
            raise ValueError("all EBR vector lengths must match")
        vectors.append(vector)
    if not vectors:
        raise ValueError("table['ebrs'] must be non-empty")
    return vectors


def create_reduced_symmetry_vector(
    irrep_counts, irrep_basis, *, strict: bool = True,
):
    """Create an ordered reduced irrep vector from multiplicity counts."""
    basis = _validate_irrep_basis(irrep_basis)
    index = {label: i for i, label in enumerate(basis)}
    vector = [0 for _ in basis]
    if not isinstance(irrep_counts, dict):
        raise ValueError("irrep_counts must be a mapping")
    for label, count in irrep_counts.items():
        if not isinstance(label, str) or not label:
            raise ValueError("irrep_counts keys must be non-empty strings")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"irrep_counts[{label!r}] must be a nonnegative integer")
        if label not in index:
            if strict:
                raise ValueError(f"irrep count {label!r} is not in reduced irrep basis")
            continue
        vector[index[label]] = count
    return vector


def compute_reduced_ebr_decomposition(
    *, target_vector, ebr_matrix, ebr_labels, max_coefficient,
):
    """Compute exact reduced EBR decomposition/classification."""
    return classify_bundle(
        list(target_vector),
        [list(vector) for vector in ebr_matrix],
        list(ebr_labels),
        max_coefficient,
    )


def search_nonnegative_bounded(target, ebr_vectors, bounds):
    """Bounded search returning the first nonnegative solution (or None)."""
    witnesses = search_nonnegative_witnesses(
        target, ebr_vectors, bounds, max_witnesses=1,
    )
    return witnesses[0] if witnesses else None


_TABLE = {
    "schema_version": "1.0.0",
    "subspace_group_candidate": "P3",
    "expected_hsps": ["GammaM", "KM"],
    "irreps": [
        "GammaM:C3_spinor_phase_+1/2",
        "KM:C3_spinor_phase_+1/6",
        "KM:C3_spinor_phase_-1/6",
    ],
    "ebrs": [
        {"label": "EBR_A", "vector": [1, 0, 1]},
        {"label": "EBR_B", "vector": [1, 1, 0]},
    ],
}


def test_reduced_ebr_matrix_uses_table_ebr_columns():
    assert get_reduced_ebr_matrix(_TABLE) == [[1, 0, 1], [1, 1, 0]]


def test_create_reduced_symmetry_vector_orders_counts_by_irrep_basis():
    counts = {
        "GammaM:C3_spinor_phase_+1/2": 2,
        "KM:C3_spinor_phase_-1/6": 1,
    }
    assert create_reduced_symmetry_vector(counts, _TABLE["irreps"]) == [2, 0, 1]


def test_create_reduced_symmetry_vector_rejects_unknown_irrep_by_default():
    with pytest.raises(ValueError, match="not in reduced irrep basis"):
        create_reduced_symmetry_vector({"GammaM:unknown": 1}, _TABLE["irreps"])


def test_compute_reduced_ebr_decomposition_wraps_three_way_classifier():
    result = compute_reduced_ebr_decomposition(
        target_vector=[2, 1, 1],
        ebr_matrix=get_reduced_ebr_matrix(_TABLE),
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


def test_integer_span_and_bounded_search_helpers_remain_available():
    in_span, signed = check_integer_span([0, 1], [[1, 0], [1, 1]])
    assert in_span is True
    assert signed == [-1, 1]
    assert derive_coefficient_bounds([2, 3], [[1, 0], [0, 1]]) == [2, 3]
    assert search_nonnegative_bounded(
        [2, 3],
        [[1, 0], [0, 1]],
        [2, 3],
    ) == [2, 3]


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
