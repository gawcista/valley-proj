"""Reduced-dimensional EBR solver: exact integer algebra for ValleyScope.

Provides Smith-normal-form integer-span testing, bounded nonnegative
integer search, and three-way classification for valley-preserving
irrep vectors against reduced EBR tables.

This is ValleyScope's own solver API.  It works on ValleyScope reduced
EBR tables and valley-preserving irrep vectors.  It does not import
``irrep.ebrs``, OR-Tools, or the private ``irrep2`` repository.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def get_reduced_ebr_matrix(table: Mapping[str, Any]) -> list[list[int]]:
    """Return reduced EBR matrix columns from a validated table-like mapping."""
    ebrs = table.get("ebrs", [])
    if not isinstance(ebrs, Sequence) or isinstance(ebrs, (str, bytes)):
        raise ValueError("table['ebrs'] must be a sequence")
    vectors: list[list[int]] = []
    expected_length: int | None = None
    for i, ebr in enumerate(ebrs):
        if not isinstance(ebr, Mapping):
            raise ValueError(f"table['ebrs'][{i}] must be a mapping")
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
    irrep_counts: Mapping[str, int],
    irrep_basis: Sequence[str],
    *,
    strict: bool = True,
) -> list[int]:
    """Create an ordered reduced irrep vector from multiplicity counts."""
    basis = _validate_irrep_basis(irrep_basis)
    index = {label: i for i, label in enumerate(basis)}
    vector = [0 for _ in basis]
    if not isinstance(irrep_counts, Mapping):
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
    *,
    target_vector: Sequence[int],
    ebr_matrix: Sequence[Sequence[int]],
    ebr_labels: Sequence[str],
    max_coefficient: int,
) -> dict:
    """Compute exact reduced EBR decomposition/classification."""
    return classify_bundle(
        list(target_vector),
        [list(vector) for vector in ebr_matrix],
        list(ebr_labels),
        max_coefficient,
    )


def check_integer_span(
    target: list[int],
    ebr_vectors: list[list[int]],
) -> tuple[bool, list[int] | None]:
    """Test whether *target* lies in the integer span of EBR columns.

    Uses Smith normal form over ZZ via sympy.

    Parameters
    ----------
    target : list[int]
        The valley-preserving irrep count vector.
    ebr_vectors : list[list[int]]
        Reduced EBR matrix columns (each inner list is one EBR column).

    Returns
    -------
    (in_span, solution) : tuple[bool, list[int] | None]
        Whether the target is in the integer span, and if so, one integer
        coefficient solution (may contain negative values).
    """
    _validate_solver_inputs(target, ebr_vectors, ebr_labels=None)
    n_rows = len(target)
    n_cols = len(ebr_vectors)

    from sympy import ZZ, Matrix
    from sympy.matrices.normalforms import smith_normal_decomp
    matrix = Matrix([[ebr_vectors[j][i] for j in range(n_cols)] for i in range(n_rows)])
    diagonal, left, right = smith_normal_decomp(matrix, domain=ZZ)

    y_prime = list(left * Matrix(target))

    rank = sum(1 for i in range(min(diagonal.rows, diagonal.cols)) if int(diagonal[i, i]) != 0)

    for i in range(rank):
        if int(y_prime[i]) % int(diagonal[i, i]) != 0:
            return False, None

    for i in range(rank, len(y_prime)):
        if int(y_prime[i]) != 0:
            return False, None

    z = [0] * n_cols
    for i in range(rank):
        z[i] = int(y_prime[i]) // int(diagonal[i, i])
    solution = list(right * Matrix(z))
    return True, [int(v) for v in solution]


def derive_coefficient_bounds(
    target: list[int],
    ebr_vectors: list[list[int]],
) -> list[int | None]:
    """Derive physical upper bounds per EBR column from positive entries.

    For each EBR column j, the coefficient c_j cannot exceed
    min_i (target[i] / EBR[i][j]) over positive EBR entries.
    """
    _validate_solver_inputs(target, ebr_vectors, ebr_labels=None)
    n_ebrs = len(ebr_vectors)
    n_rows = len(target)
    bounds: list[int | None] = []
    for j in range(n_ebrs):
        bound = None
        for i in range(n_rows):
            v = ebr_vectors[j][i]
            if v > 0:
                cap = target[i] // v
                if bound is None or cap < bound:
                    bound = cap
        bounds.append(bound)
    return bounds


def search_nonnegative_bounded(
    target: list[int],
    ebr_vectors: list[list[int]],
    bounds: list[int],
) -> list[int] | None:
    """Bounded search returning the first nonnegative solution (or None)."""
    witnesses = _search_nonnegative_witnesses(
        target, ebr_vectors, bounds, max_witnesses=1,
    )
    return witnesses[0] if witnesses else None


def _search_nonnegative_witnesses(
    target: list[int],
    ebr_vectors: list[list[int]],
    bounds: list[int],
    *,
    max_witnesses: int = 2,
) -> list[list[int]]:
    """Bounded search returning at most ``max_witnesses`` nonnegative solutions.

    Uses pruning: aborts a branch when any component of the accumulated
    vector exceeds the target.
    """
    _validate_solver_inputs(target, ebr_vectors, ebr_labels=None)
    n_ebrs = len(ebr_vectors)
    n_rows = len(target)
    if len(bounds) != n_ebrs:
        raise ValueError(
            f"bounds length {len(bounds)} != EBR vector count {n_ebrs}"
        )
    for i, bound in enumerate(bounds):
        if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
            raise ValueError(f"bounds[{i}] must be a nonnegative integer")

    solutions: list[list[int]] = []

    def _search(idx: int, accum: list[int], coeffs: list[int]) -> None:
        if len(solutions) >= max_witnesses:
            return
        if idx == n_ebrs:
            if accum == target:
                solutions.append(list(coeffs))
            return
        vec = ebr_vectors[idx]
        max_c = bounds[idx]
        for c in range(max_c + 1):
            new_accum = [accum[i] + c * vec[i] for i in range(n_rows)]
            if any(new_accum[i] > target[i] for i in range(n_rows)):
                continue
            _search(idx + 1, new_accum, coeffs + [c])
            if len(solutions) >= max_witnesses:
                return

    _search(0, [0] * n_rows, [])
    return solutions


def classify_bundle(
    target: list[int],
    ebr_vectors: list[list[int]],
    ebr_labels: list[str],
    max_coefficient: int,
) -> dict:
    """Classify a valley-preserving irrep vector against reduced EBR vectors.

    Classification and uniqueness contract:

    - Outside integer span → ``stable-topology-candidate``, no uniqueness.
    - Inside span, no nonnegative solution after complete search →
      ``fragile-topology-candidate``, no uniqueness.
    - Inside span, search truncated, no witness →
      ``indeterminate_truncated``, no uniqueness.
    - Inside span, one nonnegative witness exists →
      ``atomic-compatible-candidate``.  Uniqueness:
      * ``unique`` — complete search found exactly one.
      * ``non_unique`` — at least two witnesses found.
      * ``unknown_truncated`` — one witness but search incomplete.
    """
    max_coefficient = int(max_coefficient)
    if max_coefficient < 0:
        raise ValueError("max_coefficient must be nonnegative")
    _validate_solver_inputs(target, ebr_vectors, ebr_labels=ebr_labels)

    in_span, integer_solution = check_integer_span(target, ebr_vectors)

    if not in_span:
        return {
            "status": "no_exact_solution",
            "classification": "stable-topology-candidate",
            "integer_span_status": "outside_integer_span",
            "nonnegative_solution_status": "no_nonnegative_solution",
        }

    bounds = derive_coefficient_bounds(target, ebr_vectors)
    truncated = any(
        bounds[i] is not None and bounds[i] > max_coefficient
        for i in range(len(ebr_vectors))
    )
    effective_bounds = [
        min(b, max_coefficient) if b is not None else max_coefficient
        for b in bounds
    ]

    witnesses = _search_nonnegative_witnesses(
        target, ebr_vectors, effective_bounds, max_witnesses=2,
    )

    if not witnesses:
        # No nonnegative witness found.
        if truncated:
            result: dict = {
                "status": "indeterminate_truncated",
                "classification": "indeterminate_truncated",
                "integer_span_status": "in_integer_span",
                "nonnegative_solution_status": "no_nonnegative_solution_truncated",
            }
        else:
            result = {
                "status": "no_exact_solution",
                "classification": "fragile-topology-candidate",
                "integer_span_status": "in_integer_span",
                "nonnegative_solution_status": "no_nonnegative_solution",
            }
        if integer_solution is not None:
            result["integer_solution"] = [
                {"label": ebr_labels[i], "coefficient": int(c)}
                for i, c in enumerate(integer_solution) if c != 0
            ]
        if truncated:
            result["search_status"] = "truncated_by_max_coefficient"
        return result

    # At least one nonnegative witness exists.
    primary = [
        {"label": ebr_labels[i], "coefficient": int(c)}
        for i, c in enumerate(witnesses[0]) if c > 0
    ]

    if len(witnesses) >= 2:
        uniqueness = "non_unique"
    elif truncated:
        uniqueness = "unknown_truncated"
    else:
        uniqueness = "unique"

    result = {
        "status": "solved_exact",
        "classification": "atomic-compatible-candidate",
        "integer_span_status": "in_integer_span",
        "nonnegative_solution_status": "solved_exact",
        "decomposition_uniqueness": uniqueness,
        "ebr_decomposition": primary,
    }

    if len(witnesses) >= 2:
        result["decomposition_witnesses"] = [
            [
                {"label": ebr_labels[i], "coefficient": int(c)}
                for i, c in enumerate(w) if c > 0
            ]
            for w in witnesses[:2]
        ]

    if truncated:
        result["search_status"] = "truncated_by_max_coefficient"

    return result


def _validate_solver_inputs(
    target: Sequence[int],
    ebr_vectors: Sequence[Sequence[int]],
    *,
    ebr_labels: Sequence[str] | None,
) -> None:
    _validate_nonnegative_integer_vector(target, field="target")
    if not ebr_vectors:
        raise ValueError("ebr_vectors must be non-empty")
    target_len = len(target)
    for i, vector in enumerate(ebr_vectors):
        values = list(vector)
        if len(values) != target_len:
            raise ValueError(
                f"EBR vector length {len(values)} != target length {target_len}"
            )
        _validate_nonnegative_integer_vector(values, field=f"EBR vector {i}")
    if ebr_labels is not None:
        if len(ebr_labels) != len(ebr_vectors):
            raise ValueError(
                f"ebr_labels length {len(ebr_labels)} != "
                f"EBR vector count {len(ebr_vectors)}"
            )
        seen: set[str] = set()
        for i, label in enumerate(ebr_labels):
            if not isinstance(label, str) or not label:
                raise ValueError(f"ebr_labels[{i}] must be a non-empty string")
            if label in seen:
                raise ValueError(f"duplicate EBR label {label!r}")
            seen.add(label)


def _validate_nonnegative_integer_vector(
    vector: Sequence[int],
    *,
    field: str,
) -> None:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    if not vector:
        raise ValueError(f"{field} must be non-empty")
    for i, value in enumerate(vector):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field}[{i}] must be a nonnegative integer")


def _validate_irrep_basis(irrep_basis: Sequence[str]) -> list[str]:
    if not isinstance(irrep_basis, Sequence) or isinstance(irrep_basis, (str, bytes)):
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
