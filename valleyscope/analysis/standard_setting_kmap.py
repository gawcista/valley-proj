"""Standard-setting HSP k-coordinate mapping for valley-projected subspace SG.

When a valley-projected subspace space group uses a conventional or
centered setting that differs from the parent moire primitive reciprocal
basis, the parent-setting k_frac may not directly match any
irreptables standard-setting HSP coordinate.

This module attempts to resolve the mapping using crystallographic
setting data (Hall symbol/number, centering, unique-axis convention)
from the spglib per-valley standard group match.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def resolve_standard_setting_hsp_label(
    *,
    k_frac: np.ndarray,
    table,
    standard_match: dict[str, object] | None,
    tolerance: float = 1e-6,
) -> tuple[str | None, str | None, dict[str, object]]:
    """Resolve a standard-setting Bilbao HSP label for a sampled k-point.

    Parameters
    ----------
    k_frac : np.ndarray
        Sampled k-point fractional coordinates in the parent moire
        reciprocal basis.
    table : StandardIrrepTable
        Loaded irreptables irrep table for the subspace SG.
    standard_match : dict or None
        spglib per-valley standard group match with fields:
        ``number``, ``international_short``, ``hall_number``,
        ``hall_symbol``, ``operation_ids``.
    tolerance : float
        Coordinate comparison tolerance.

    Returns
    -------
    hsp_label : str or None
        Resolved HSP label, or None if unresolved.
    blocker : str or None
        Blocker reason if unresolved, None if resolved.
    provenance : dict
        Diagnostic provenance for the resolution attempt.
    """
    k_frac = np.asarray(k_frac, dtype=float)
    prov: dict[str, object] = {
        "attempted_direct_match": True,
    }

    # 1. Direct coordinate match (existing behavior).
    try:
        direct_label = table.match_kpoint_label(k_frac, tolerance=tolerance)
    except TypeError:
        # Compatibility with test mocks that don't accept tolerance.
        direct_label = table.match_kpoint_label(k_frac)
    if direct_label is not None:
        prov["direct_match_succeeded"] = True
        return direct_label, None, prov

    prov["direct_match_succeeded"] = False
    prov["direct_match_reason"] = (
        "parent-setting k_frac does not match any standard-setting "
        "Bilbao HSP coordinate in the irreptables table"
    )

    if standard_match is None or not isinstance(standard_match, dict):
        return None, (
            "standard_setting_hsp_mapping_unresolved: "
            "no per-valley standard group match available "
            "for setting-aware k-coordinate transformation"
        ), prov

    sg_number = standard_match.get("number")
    hall_number = standard_match.get("hall_number")
    hall_symbol = standard_match.get("hall_symbol", "")
    sg_symbol = standard_match.get("international_short", "")

    prov["subspace_sg_number"] = sg_number
    prov["subspace_sg_symbol"] = sg_symbol
    prov["hall_number"] = hall_number
    prov["hall_symbol"] = hall_symbol

    # 2. Attempt setting-aware transformation.
    #    The Hall symbol encodes the conventional cell centering and
    #    unique-axis convention.  For centered settings (C, I, F, R),
    #    the conventional reciprocal cell differs from the primitive
    #    cell used in the parent DFT calculation.
    transform_result = _attempt_setting_transform(
        k_frac=k_frac,
        hall_number=int(hall_number) if isinstance(hall_number, int) and not isinstance(hall_number, bool) else None,
        hall_symbol=str(hall_symbol) if hall_symbol else "",
        sg_number=int(sg_number) if isinstance(sg_number, int) and not isinstance(sg_number, bool) else None,
    )
    prov["setting_transform"] = transform_result

    if transform_result.get("transformed_k_frac") is not None:
        transformed_k = np.asarray(transform_result["transformed_k_frac"], dtype=float)
        try:
            transformed_label = table.match_kpoint_label(transformed_k, tolerance=tolerance)
        except TypeError:
            transformed_label = table.match_kpoint_label(transformed_k)
        if transformed_label is not None:
            prov["transformed_match_succeeded"] = True
            prov["transformed_k_frac"] = transformed_k.tolist()
            prov["transformed_hsp_label"] = transformed_label
            return transformed_label, None, prov

    # 3. Setting transform not available or not unique.
    reason_parts = [
        "standard_setting_hsp_mapping_unresolved: "
        f"valley-projected subspace SG {sg_symbol} (No. {sg_number})",
    ]
    if hall_symbol:
        hall_str = f"Hall {hall_number} ({hall_symbol})" if hall_number else hall_symbol
        reason_parts.append(f"standard setting {hall_str}")
    reason_parts.append(
        "is resolved, but no unique standard-setting k-coordinate "
        "transform is established from the parent moire reciprocal "
        "basis to the irreptables standard reciprocal basis"
    )
    if transform_result.get("reason"):
        reason_parts.append(f" — {transform_result['reason']}")

    return None, " ".join(reason_parts) + ".", prov


def _attempt_setting_transform(
    *,
    k_frac: np.ndarray,
    hall_number: int | None,
    hall_symbol: str,
    sg_number: int | None,
) -> dict[str, object]:
    """Attempt to transform k_frac using setting provenance.

    Currently implements a provenance-documenting structure.  Future
    work: use spglib basis-transformation matrices or Hall-symbol-
    directed centering transforms to map the parent-setting k_frac
    into the standard-setting reciprocal basis.

    The Hall symbol classifies the standard setting by centering
    type.  Primitive settings (P) share the same Bravais lattice
    class as the parent cell; centered settings (C, I, F, R) and
    rhombohedral settings (R) use a larger conventional cell where
    the reciprocal lattice differs from the primitive reciprocal
    lattice.
    """
    result: dict[str, object] = {}

    if hall_number is None or not hall_symbol:
        result["reason"] = (
            "Hall number/symbol not available from spglib match"
        )
        return result

    centering = hall_symbol[0] if hall_symbol else ""
    if centering == "P":
        result["reason"] = (
            f"primitive setting (Hall {hall_symbol}); "
            "parent-to-standard lattice vectors both primitive; "
            "the difference is in Bravais type/orientation, "
            "not in centering.  A Bravais-type change requires "
            "the full direct-lattice transformation, which is not "
            "available for subgroup-only standard matches."
        )
    elif centering in ("C", "I", "F", "R"):
        result["reason"] = (
            f"centered setting (Hall {hall_symbol}); "
            "the conventional reciprocal lattice for a "
            f"{centering}-centered cell differs from the parent "
            "primitive reciprocal lattice.  The transformation "
            "requires the full direct-to-conventional "
            "reciprocal-basis mapping, which is not yet "
            "implemented."
        )
    else:
        result["reason"] = (
            f"unrecognised Hall symbol centering '{centering}' "
            f"(Hall {hall_symbol})"
        )

    if sg_number is not None:
        result["sg_number"] = int(sg_number)

    return result
