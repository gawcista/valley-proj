"""Reduced-dimensional EBR solver: exact integer algebra for ValleyScope.

Provides Smith-normal-form integer-span testing, bounded nonnegative
integer search, and three-way classification for valley-preserving
irrep vectors against reduced EBR tables.

This is ValleyScope's own solver API.  It works on ValleyScope reduced
EBR tables and valley-preserving irrep vectors.  It does not import
``irrep.ebrs``, OR-Tools, or the private ``irrep2`` repository.
"""

from __future__ import annotations


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
    if not ebr_vectors:
        return False, None
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
    """Exhaustive bounded search for nonnegative integer solution.

    Uses pruning: aborts a branch when any component of the accumulated
    vector exceeds the target.
    """
    n_ebrs = len(ebr_vectors)
    n_rows = len(target)

    def _search(idx: int, accum: list[int], coeffs: list[int]) -> list[int] | None:
        if idx == n_ebrs:
            return coeffs if accum == target else None
        vec = ebr_vectors[idx]
        max_c = bounds[idx]
        for c in range(max_c + 1):
            new_accum = [accum[i] + c * vec[i] for i in range(n_rows)]
            if any(new_accum[i] > target[i] for i in range(n_rows)):
                continue
            result = _search(idx + 1, new_accum, coeffs + [c])
            if result is not None:
                return result
        return None

    return _search(0, [0] * n_rows, [])


def classify_bundle(
    target: list[int],
    ebr_vectors: list[list[int]],
    ebr_labels: list[str],
    max_coefficient: int,
) -> dict:
    """Classify a valley-preserving irrep vector against reduced EBR vectors.

    Three-way classification:
    - ``atomic-compatible-candidate``: nonnegative exact solution exists.
    - ``fragile-topology-candidate``: integer solution exists but needs
      negative coefficients; signed witness provided.
    - ``stable-topology-candidate``: target is outside integer EBR span.

    Parameters
    ----------
    target : list[int]
        Valley-preserving irrep count vector.
    ebr_vectors : list[list[int]]
        Reduced EBR matrix columns.
    ebr_labels : list[str]
        Labels for each EBR column.
    max_coefficient : int
        Safety cap; if a derived physical bound exceeds this, the search
        is truncated and ``search_status`` is set.

    Returns
    -------
    dict
        Per-solution dict with ``status``, ``classification``,
        ``integer_span_status``, ``nonnegative_solution_status``, and
        optionally ``ebr_decomposition``, ``integer_solution``, or
        ``search_status``.
    """
    max_coefficient = int(max_coefficient)
    if max_coefficient < 0:
        raise ValueError("max_coefficient must be nonnegative")

    # Integer-span test.
    in_span, integer_solution = check_integer_span(target, ebr_vectors)

    if not in_span:
        return {
            "status": "no_exact_solution",
            "classification": "stable-topology-candidate",
            "integer_span_status": "outside_integer_span",
            "nonnegative_solution_status": "no_nonnegative_solution",
        }

    # Nonnegative search with physical bounds, capped by max_coefficient.
    bounds = derive_coefficient_bounds(target, ebr_vectors)
    truncated = any(
        bounds[i] is not None and bounds[i] > max_coefficient
        for i in range(len(ebr_vectors))
    )
    effective_bounds = [
        min(b, max_coefficient) if b is not None else max_coefficient
        for b in bounds
    ]

    nonneg_solution = search_nonnegative_bounded(
        target, ebr_vectors, effective_bounds,
    )

    if nonneg_solution is not None:
        result = {
            "status": "solved_exact",
            "classification": "atomic-compatible-candidate",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "solved_exact",
            "ebr_decomposition": [
                {"label": ebr_labels[i], "coefficient": int(c)}
                for i, c in enumerate(nonneg_solution) if c > 0
            ],
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
