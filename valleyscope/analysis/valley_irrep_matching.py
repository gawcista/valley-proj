"""Minimal valley-preserving character-table matching for C3 and C2 spinful irreps.

Matches already-computed valley-preserving eigenphases to named irrep labels
using tiny internal tables.  Only ``trusted`` readiness enables matching;
``usable_with_caution`` produces ``diagnostic_only`` output.

Non-goals: no reduced EBR, no compatibility relations, no full induced rep logic.
"""

from __future__ import annotations

from typing import Any

READINESS_TRUSTED = "trusted"
READINESS_USABLE_WITH_CAUTION = "usable_with_caution"
READINESS_BLOCKED = "blocked"

# ---------------------------------------------------------------------------
# Internal irrep tables — spinful C3 and C2 only
# ---------------------------------------------------------------------------

# Spinful C3 irreps (order 3, double group).  Each entry is a label and
# the expected phases (sorted, each in (-0.5, 0.5]).
_C3_SPINFUL_IRREPS: list[dict[str, object]] = [
    {"label": "C3_spinor_phase_+1/6",  "phases": [1.0 / 6.0]},
    {"label": "C3_spinor_phase_+1/2",  "phases": [0.5]},
    {"label": "C3_spinor_phase_-1/6",  "phases": [-1.0 / 6.0]},
]

# Spinful C2 irreps (order 2, double group).
_C2_SPINFUL_IRREPS: list[dict[str, object]] = [
    {"label": "C2_spinor_phase_+1/4",  "phases": [0.25]},
    {"label": "C2_spinor_phase_-1/4",  "phases": [-0.25]},
]

def _canonical_phase(phase: float) -> float:
    """Wrap phase to (-0.5, 0.5]."""
    p = phase % 1.0
    if p > 0.5:
        p -= 1.0
    if p <= -0.5:
        p += 1.0
    if abs(p) < 1e-12:
        p = 0.0
    return p


def _sort_phases(phases: list[float]) -> list[float]:
    return sorted(_canonical_phase(p) for p in phases)


def _phase_periodic_delta(a: float, b: float) -> float:
    """Minimum distance between two phases accounting for mod-1 periodicity."""
    d = abs(a - b)
    return min(d, abs(d - 1.0))


def _match_phases_to_table(
    phases: list[float],
    table: list[dict[str, object]],
    tolerance: float = 1e-5,
) -> str | None:
    """Match a sorted list of phases to a table entry, using mod-1 periodicity."""
    canonical = _sort_phases(phases)
    for entry in table:
        expected = sorted(float(p) for p in entry["phases"])
        if len(expected) != len(canonical):
            continue
        if all(
            _phase_periodic_delta(c, e) <= tolerance
            for c, e in zip(canonical, expected)
        ):
            return str(entry["label"])
    return None


def _candidate_matches_order(candidate: str | None, operation_order: int) -> bool:
    if candidate in ("C3_like", "P3"):
        return operation_order == 3
    if candidate in ("C2_like", "P2"):
        return operation_order == 2
    return False


# ---------------------------------------------------------------------------
# Public matching interface
# ---------------------------------------------------------------------------

def match_valley_irrep(
    *,
    eigenphases: list[float],
    operation_order: int,
    subspace_group_candidate: str | None,
    readiness_level: str,
    allow_caution: bool = False,
    tolerance: float = 1e-5,
) -> dict[str, object]:
    """Match valley-preserving eigenphases to a named irrep.

    Parameters
    ----------
    eigenphases : list[float]
        Phases in (-0.5, 0.5] convention, one per state in the valley subspace.
    operation_order : int
        Order of the operation (2 or 3).
    subspace_group_candidate : str or None
        e.g. ``C3_like``, ``C2_like``.  Used to select the irrep table.
    readiness_level : str
        Must be ``trusted``, ``usable_with_caution``, or ``blocked``.
    allow_caution : bool
        If True, ``usable_with_caution`` readiness allows matching.
        Default False: only ``trusted`` enables matching;
        ``usable_with_caution`` gives ``diagnostic_only``.

    Returns
    -------
    dict with: matched_irrep, matching_status, reason, eigenphases
    """
    canonical = _sort_phases(eigenphases)
    result_base: dict[str, object] = {
        "eigenphases": canonical,
        "operation_order": operation_order,
    }

    if readiness_level not in {
        READINESS_TRUSTED,
        READINESS_USABLE_WITH_CAUTION,
        READINESS_BLOCKED,
    }:
        return {
            **result_base,
            "matched_irrep": None,
            "matching_status": "blocked",
            "reason": f"unknown readiness_level={readiness_level}",
        }

    if readiness_level == READINESS_BLOCKED:
        return {
            **result_base,
            "matched_irrep": None,
            "matching_status": "blocked",
            "reason": f"readiness_level={readiness_level}",
        }

    if readiness_level == READINESS_USABLE_WITH_CAUTION and not allow_caution:
        return {
            **result_base,
            "matched_irrep": None,
            "matching_status": "diagnostic_only",
            "reason": (
                "readiness is usable_with_caution; set allow_caution=True "
                "to enable matching"
            ),
        }

    if not canonical:
        return {
            **result_base,
            "matched_irrep": None,
            "matching_status": "not_applicable",
            "reason": "no eigenphases available",
        }

    rank = len(canonical)
    if rank != 1:
        return {
            **result_base,
            "matched_irrep": None,
            "matching_status": "failed_ambiguous",
            "reason": (
                f"rank={rank} contains multiple eigenphases; minimal matcher "
                "does not decompose direct sums into one-dimensional irreps"
            ),
        }

    if not _candidate_matches_order(subspace_group_candidate, operation_order):
        return {
            **result_base,
            "matched_irrep": None,
            "matching_status": "failed_no_table",
            "reason": (
                f"no irrep table for subspace_group_candidate="
                f"{subspace_group_candidate!r} and operation_order={operation_order}"
            ),
        }

    # Select table
    if operation_order == 3:
        table = _C3_SPINFUL_IRREPS
    elif operation_order == 2:
        table = _C2_SPINFUL_IRREPS
    else:
        return {
            **result_base,
            "matched_irrep": None,
            "matching_status": "not_applicable",
            "reason": f"no irrep table for order={operation_order}",
        }

    label = _match_phases_to_table(canonical, table, tolerance=tolerance)
    if label is not None:
        return {
            **result_base,
            "matched_irrep": label,
            "matching_status": "matched",
            "reason": f"phases {canonical} match {label}",
        }
    return {
        **result_base,
        "matched_irrep": None,
        "matching_status": "failed_no_table",
        "reason": f"phases {canonical} did not match any known irrep for order={operation_order} rank={rank}",
    }


# ---------------------------------------------------------------------------
# Batch matching across the workflow decision report
# ---------------------------------------------------------------------------

def build_valley_irrep_matching_report(
    *,
    irrep_workflow_decisions: dict[str, object] | None,
    symmetry_adapted_valley_report: dict[str, object] | None,
    allow_caution: bool = False,
) -> dict[str, object]:
    """Build per-(kpoint, valley) irrep matching results.

    Cross-references the workflow decision report with character diagnostics
    from the symmetry-adapted valley analysis.
    """
    if irrep_workflow_decisions is None:
        return {"status": "not_evaluated", "by_kpoint": {}}

    by_kpoint: dict[str, object] = {}
    decisions_by_kp = irrep_workflow_decisions.get("by_kpoint", {})

    # Collect subspace diagnostics for character data
    sa_by_kp: dict[str, dict[str, dict[str, Any]]] = {}
    if isinstance(symmetry_adapted_valley_report, dict):
        for kp_name, kp_data in symmetry_adapted_valley_report.get("by_kpoint", {}).items():
            if not isinstance(kp_data, dict):
                continue
            for subspace in kp_data.get("valley_preserving_subspaces", []):
                if not isinstance(subspace, dict):
                    continue
                orbit = subspace.get("orbit", [])
                if not orbit:
                    continue
                v = str(orbit[0])
                char_diag = subspace.get("valley_preserving_character_diagnostics", {})
                sg = subspace.get("subspace_group", {})
                sa_by_kp.setdefault(kp_name, {})[v] = {
                    "char_diag": char_diag,
                    "subspace_group": sg,
                }

    for kp_name, valley_decisions in decisions_by_kp.items():
        if not isinstance(valley_decisions, dict):
            continue
        kp_matches: dict[str, dict[str, object]] = {}
        for v_name, decision in valley_decisions.items():
            if not isinstance(decision, dict):
                continue
            readiness = str(decision.get("readiness_level", ""))
            path = str(decision.get("workflow_path", ""))

            sa = sa_by_kp.get(kp_name, {}).get(v_name, {})
            char_diag = sa.get("char_diag", {})
            sg = sa.get("subspace_group", {})
            sg_candidate = sg.get("subspace_group_candidate")
            op_orders = sg.get("operation_orders", {})
            if not isinstance(op_orders, dict):
                op_orders = {}

            # Collect all non-identity character entries
            op_matches: dict[str, dict[str, object]] = {}
            per_valley = char_diag.get("per_valley", {})
            if isinstance(per_valley, dict):
                for _, items in per_valley.items():
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        op_id = item.get("operation_id", 0)
                        if op_id == 0:
                            continue
                        phases = item.get("eigenphases")
                        if not phases:
                            continue
                        order_raw = op_orders.get(str(op_id), op_orders.get(op_id))
                        if order_raw is None:
                            result = {
                                "matched_irrep": None,
                                "matching_status": "failed_no_table",
                                "reason": f"missing operation order for operation_id={op_id}",
                                "eigenphases": list(phases),
                                "operation_order": None,
                            }
                        else:
                            order = int(order_raw)
                            result = match_valley_irrep(
                                eigenphases=list(phases),
                                operation_order=order,
                                subspace_group_candidate=sg_candidate,
                                readiness_level=readiness,
                                allow_caution=allow_caution,
                            )
                        result["workflow_path"] = path
                        result["readiness_level"] = readiness
                        result["subspace_group_candidate"] = sg_candidate
                        result["operation_id"] = op_id
                        op_matches[str(op_id)] = result
            if op_matches:
                kp_matches[v_name] = op_matches
        if kp_matches:
            by_kpoint[kp_name] = kp_matches

    return {
        "status": "ok" if by_kpoint else "not_evaluated",
        "matching_statuses": ["matched", "diagnostic_only", "not_applicable",
                              "failed_no_table", "failed_ambiguous", "blocked"],
        "tables_implemented": ["spinful_C3", "spinful_C2"],
        "by_kpoint": by_kpoint,
    }
