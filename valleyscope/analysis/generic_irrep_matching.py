"""Generic valley-preserving HSP little-subgroup restricted-character irrep matching.

Compares ValleyScope computed characters ``chi_a(g)`` for ``g in G_k^(a)``
against Bilbao/irreptable source irreps restricted to the same valley-preserving
operation set.  Uses the character inner product to compute integer
multiplicities when a complete operation set and matching character data are
available.

This is an internal matching strategy.  It does not infer HSP names, operation
IDs, valley labels, or spinor conventions from source labels.  It does not
import real ``irreptables`` — source payloads are supplied explicitly.
"""

from __future__ import annotations

from typing import Any


def match_restricted_characters(
    *,
    computed_characters: dict[int, complex],
    source_irrep_characters: dict[str, dict[int, complex]],
    valley_preserving_operation_ids: list[int],
    hsp_little_group_operation_ids: list[int] | None = None,
    tol: float = 5e-5,
) -> dict[str, Any]:
    """Match ValleyScope computed characters against restricted source irreps.

    Parameters
    ----------
    computed_characters : dict[int, complex]
        ValleyScope computed characters ``chi(g)`` keyed by ValleyScope
        operation ID.  Only operations in ``valley_preserving_operation_ids``
        are used for matching.
    source_irrep_characters : dict[str, dict[int, complex]]
        Source irrep character data keyed by source irrep label.  Each value
        is a dict mapping source operation ID to complex character.
        The keys of the inner dict define the available operation set for
        that source irrep.
    valley_preserving_operation_ids : list[int]
        ValleyScope operation IDs of the valley-preserving subgroup operations.
    hsp_little_group_operation_ids : list[int] or None
        ValleyScope operation IDs of the full HSP little group (valley-preserving
        and valley-changing).  For provenance only — does not affect matching.
    tol : float
        Tolerance for integer multiplicity checks.

    Returns
    -------
    dict with ``matching_status``, ``irrep_multiplicities``,
    ``source_operation_map``, and diagnostics.
    """
    vp_ids = _valid_ids(valley_preserving_operation_ids)
    hsp_ids = _valid_ids(hsp_little_group_operation_ids or vp_ids)

    # Build the operation set for matching.
    vp_set = set(vp_ids)

    if not vp_set:
        return _blocked("empty_valley_preserving_operation_set",
                        "valley_preserving_operation_ids is empty")

    if not computed_characters:
        return _diagnostic("missing_computed_characters",
                           "no computed characters provided")

    # For each source irrep, compute multiplicities via inner product.
    results: dict[str, dict[str, Any]] = {}
    for label, source_chars in sorted(source_irrep_characters.items()):
        if not isinstance(source_chars, dict) or not source_chars:
            continue
        source_op_ids = _valid_ids(list(source_chars.keys()))

        # Build operation map: ValleyScope op ID -> source op ID, if available.
        op_map = _build_operation_map(vp_ids, source_op_ids)

        # Restrict both computed and source characters to the common op set.
        common_ops = [op_id for op_id in vp_ids
                      if op_id in op_map and op_id in computed_characters]
        if not common_ops:
            results[label] = _diagnostic(
                "no_common_operations",
                f"no common operations between computed ({vp_ids}) "
                f"and source ({source_op_ids}) for {label}",
            )
            continue

        chi_computed = [computed_characters[op] for op in common_ops]
        chi_source = [source_chars[op_map[op]] for op in common_ops]

        # Inner product: (1/N) sum conj(chi_lambda(g)) chi_a(g)
        n = len(common_ops)
        inner = sum(
            _conj(source_chars[op_map[op]]) * computed_characters[op]
            for op in common_ops
        ) / n

        # Check for integer result.
        real_part = inner.real
        imag_part = inner.imag
        if abs(imag_part) > tol:
            results[label] = _diagnostic(
                "non_real_multiplicity",
                f"inner product for {label} is {inner}, "
                f"imaginary part {imag_part} exceeds tol {tol}",
            )
            continue

        mult = round(real_part)
        if abs(real_part - mult) > tol:
            results[label] = _diagnostic(
                "non_integer_multiplicity",
                f"inner product for {label} is {real_part}, "
                f"not integer within tol {tol}",
            )
            continue

        results[label] = {
            "matching_status": "matched",
            "irrep_multiplicities": {label: mult} if mult > 0 else {},
            "source_operation_map": op_map,
            "common_operation_ids": common_ops,
            "diagnostic_only": False,
        }

    # Aggregate multiplicities.
    aggregate: dict[str, int] = {}
    matched_count = 0
    diagnostic_count = 0
    for label, result in results.items():
        if result.get("matching_status") == "matched":
            matched_count += 1
            for irr_label, m in result.get("irrep_multiplicities", {}).items():
                if m > 0:
                    aggregate[irr_label] = aggregate.get(irr_label, 0) + m
        else:
            diagnostic_count += 1

    if matched_count == 0:
        return _diagnostic(
            "no_matched_irreps",
            f"all {len(results)} source irreps are diagnostic or blocked",
            multiplicities={},
        )

    return {
        "matching_status": "matched",
        "matching_strategy": "bilbao_restricted_character",
        "irrep_multiplicities": aggregate,
        "matched_irrep_count": matched_count,
        "diagnostic_irrep_count": diagnostic_count,
        "source_operation_map": _build_operation_map(
            vp_ids,
            list({op for r in results.values()
                  for op in r.get("common_operation_ids", [])}),
        ),
        "valley_preserving_operation_ids": vp_ids,
        "hsp_little_group_operation_ids": hsp_ids,
        "per_irrep_results": results,
        "diagnostic_only": False,
        "reason": "",
    }


def _build_operation_map(
    valleyscope_ids: list[int],
    source_ids: list[int],
) -> dict[int, int]:
    """Map ValleyScope op IDs to source op IDs.

    When both sides have the same cardinality and ordering, use direct
    1:1 correspondence.  Otherwise, identity is matched explicitly and
    remaining ops are paired by order when possible.
    """
    if len(valleyscope_ids) == len(source_ids):
        return dict(zip(valleyscope_ids, source_ids))
    op_map: dict[int, int] = {}
    # Match identity first.
    if 1 in valleyscope_ids and 1 in source_ids:
        op_map[1] = 1
    remaining_vs = [op for op in valleyscope_ids if op not in op_map]
    remaining_src = [op for op in source_ids if op not in op_map.values()]
    if len(remaining_vs) == len(remaining_src):
        for vs, src in zip(remaining_vs, remaining_src):
            op_map[vs] = src
    return op_map


def _valid_ids(ids: list[Any]) -> list[int]:
    return [int(i) for i in ids if isinstance(i, (int, float))
            and not isinstance(i, bool)]


def _conj(c: complex) -> complex:
    return c.real - 1j * c.imag


def _blocked(reason_key: str, reason: str) -> dict[str, Any]:
    return {
        "matching_status": "blocked",
        "matching_strategy": "bilbao_restricted_character",
        "irrep_multiplicities": {},
        "source_operation_map": {},
        "diagnostic_only": True,
        "reason": f"{reason_key}: {reason}",
    }


def _diagnostic(
    reason_key: str, reason: str, multiplicities: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "matching_status": "diagnostic",
        "matching_strategy": "bilbao_restricted_character",
        "irrep_multiplicities": multiplicities or {},
        "source_operation_map": {},
        "diagnostic_only": True,
        "reason": f"{reason_key}: {reason}",
    }
