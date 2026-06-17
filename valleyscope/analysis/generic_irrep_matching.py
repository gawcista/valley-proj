"""Generic valley-preserving HSP little-subgroup restricted-character irrep matching.

Compares ValleyScope computed characters ``chi_a(g)`` for ``g in G_k^(a)``
against Bilbao/irreptable source irreps restricted to the same valley-preserving
operation set via an explicit ``source_operation_map``.  Uses the character
inner product over the full ``G_k^(a)`` operation set to compute integer
multiplicities when all required data is present.

This is an internal matching strategy.  It does not infer HSP names, operation
IDs, valley labels, or spinor conventions from source labels.  It does not
import real ``irreptables`` — source payloads are supplied explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def match_restricted_characters(
    *,
    computed_characters: Mapping[int, complex],
    source_irrep_characters: Mapping[str, Mapping[int, complex]],
    valley_preserving_operation_ids: list[int],
    source_operation_map: Mapping[int, int],
    hsp_little_group_operation_ids: list[int] | None = None,
    tol: float = 5e-5,
) -> dict[str, Any]:
    """Match ValleyScope computed characters against restricted source irreps.

    Parameters
    ----------
    computed_characters : dict[int, complex]
        ValleyScope computed characters ``chi(g)`` keyed by ValleyScope
        operation ID.  Every ``valley_preserving_operation_ids`` entry must
        be present.
    source_irrep_characters : dict[str, dict[int, complex]]
        Source irrep character data keyed by source irrep label.  Each value
        is a dict mapping source operation ID to complex character.
    valley_preserving_operation_ids : list[int]
        ValleyScope operation IDs of the valley-preserving subgroup operations.
        Must be non-empty.
    source_operation_map : dict[int, int]
        Explicit mapping from ValleyScope operation ID to source/Bilbao
        operation ID.  Every entry in ``valley_preserving_operation_ids``
        must have a mapping.  Do NOT infer this map — the caller must
        supply it.
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
    vp_ids = _validate_op_ids(
        valley_preserving_operation_ids, "valley_preserving_operation_ids"
    )
    hsp_ids = _validate_op_ids(
        hsp_little_group_operation_ids if hsp_little_group_operation_ids is not None
        else vp_ids, "hsp_little_group_operation_ids",
    )
    omap = _validate_operation_map(source_operation_map, "source_operation_map")
    n_g = len(vp_ids)

    # --- Pre-match completeness checks ---
    if not vp_ids:
        return _blocked("empty_valley_preserving_operation_set",
                        "valley_preserving_operation_ids is empty")

    # Every VP op must be in computed_characters.
    missing_comp = [op for op in vp_ids if op not in computed_characters]
    if missing_comp:
        return _blocked(
            "incomplete_computed_characters",
            f"valley_preserving_operation_ids not in computed_characters: "
            f"{missing_comp}",
        )

    # Every VP op must be in the explicit source_operation_map.
    missing_map = [op for op in vp_ids if op not in omap]
    if missing_map:
        return _blocked(
            "incomplete_source_operation_map",
            f"valley_preserving_operation_ids not in source_operation_map: "
            f"{missing_map}",
        )

    # For each source irrep, every mapped source op must be present.
    results: dict[str, dict[str, Any]] = {}
    for label, source_chars in sorted(source_irrep_characters.items()):
        if not isinstance(source_chars, Mapping) or not source_chars:
            results[label] = _diagnostic(
                "missing_source_characters",
                f"source irrep {label!r} has no character data",
            )
            continue
        mapped_source_ids = [omap[op] for op in vp_ids]
        missing_src = [sid for sid in mapped_source_ids
                       if sid not in source_chars]
        if missing_src:
            results[label] = _diagnostic(
                "incomplete_source_characters",
                f"source irrep {label!r} missing characters for "
                f"source operation IDs: {missing_src}",
            )
            continue

        # --- Compute inner product over FULL G_k^(a) set ---
        chi_computed = [computed_characters[op] for op in vp_ids]
        chi_source = [source_chars[omap[op]] for op in vp_ids]

        inner = sum(
            _conj(chi_source[i]) * chi_computed[i]
            for i in range(n_g)
        ) / n_g

        # Check for integer result.
        real_part = inner.real
        imag_part = inner.imag
        if abs(imag_part) > tol:
            results[label] = _diagnostic(
                "non_real_multiplicity",
                f"inner product for {label!r} is {inner}, "
                f"imaginary part {imag_part} exceeds tol {tol}",
            )
            continue

        mult = round(real_part)
        if abs(real_part - mult) > tol:
            results[label] = _diagnostic(
                "non_integer_multiplicity",
                f"inner product for {label!r} is {real_part}, "
                f"not integer within tol {tol}",
            )
            continue

        if mult < 0:
            results[label] = _diagnostic(
                "negative_multiplicity",
                f"inner product for {label!r} is {mult}, "
                f"negative multiplicities are not physically valid",
            )
            continue

        results[label] = {
            "matching_status": "matched",
            "irrep_multiplicities": {label: mult} if mult > 0 else {},
            "diagnostic_only": False,
            "reason": "",
        }

    # --- Aggregate multiplicities (only positive counts) ---
    aggregate: dict[str, int] = {}
    matched_count = 0
    diagnostic_count = 0
    for label, result in results.items():
        if result.get("matching_status") == "matched":
            for irr_label, m in result.get("irrep_multiplicities", {}).items():
                if m > 0:
                    aggregate[irr_label] = aggregate.get(irr_label, 0) + m
                    matched_count += 1
        else:
            diagnostic_count += 1

    if matched_count == 0:
        return _diagnostic(
            "no_matched_irreps",
            f"all {len(results)} source irreps are diagnostic or blocked",
            per_irrep_results=results if results else None,
        )

    return {
        "matching_status": "matched",
        "matching_strategy": "bilbao_restricted_character",
        "irrep_multiplicities": aggregate,
        "matched_irrep_count": matched_count,
        "diagnostic_irrep_count": diagnostic_count,
        "source_operation_map": dict(omap),
        "valley_preserving_operation_ids": vp_ids,
        "hsp_little_group_operation_ids": hsp_ids,
        "per_irrep_results": results,
        "diagnostic_only": False,
        "reason": "",
    }


def _validate_op_ids(
    ids: list[Any], field: str,
) -> list[int]:
    """Validate operation IDs: ints only, no floats, no bools, no coercing."""
    result: list[int] = []
    for i, op_id in enumerate(ids):
        if not isinstance(op_id, int) or isinstance(op_id, bool):
            raise ValueError(
                f"{field}[{i}] must be an integer, got "
                f"{type(op_id).__name__} {op_id!r}"
            )
        result.append(op_id)
    return result


def _validate_operation_map(
    op_map: Mapping[Any, Any], field: str,
) -> dict[int, int]:
    """Validate an operation map: both keys and values must be ints."""
    if not isinstance(op_map, Mapping):
        raise ValueError(
            f"{field} must be a mapping, got {type(op_map).__name__}"
        )
    result: dict[int, int] = {}
    for key, val in op_map.items():
        if not isinstance(key, int) or isinstance(key, bool):
            raise ValueError(
                f"{field} keys must be integers, got "
                f"{type(key).__name__} {key!r}"
            )
        if not isinstance(val, int) or isinstance(val, bool):
            raise ValueError(
                f"{field}[{key!r}] must be an integer, got "
                f"{type(val).__name__} {val!r}"
            )
        result[key] = val
    return result


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
    reason_key: str, reason: str,
    multiplicities: dict[str, int] | None = None,
    per_irrep_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "matching_status": "diagnostic",
        "matching_strategy": "bilbao_restricted_character",
        "irrep_multiplicities": multiplicities or {},
        "source_operation_map": {},
        "diagnostic_only": True,
        "reason": f"{reason_key}: {reason}",
    }
    if per_irrep_results is not None:
        result["per_irrep_results"] = per_irrep_results
    return result
