"""[PROTOTYPE_LEGACY] Minimal valley-preserving character-table matching for C3 and C2 spinful irreps.

Matches already-computed valley-preserving eigenphases to named irrep labels
using validated package-data phase tables.  Only ``trusted`` readiness enables
matching; ``usable_with_caution`` produces ``diagnostic_only`` output.

Non-goals: no reduced EBR, no compatibility relations, no full induced rep logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from cmath import exp
import math
from typing import Any

from valleyscope.data.valley_irreps.catalog import get_irrep_phase_list

READINESS_TRUSTED = "trusted"
READINESS_USABLE_WITH_CAUTION = "usable_with_caution"
READINESS_BLOCKED = "blocked"
MATCHING_STATUSES = [
    "matched", "diagnostic", "diagnostic_only", "not_applicable",
    "failed_no_table", "failed_ambiguous", "blocked",
]


def _irrep_table_for_order(order: int) -> list[dict[str, object]]:
    """Return the validated irrep phase table for a given operation order."""
    if order == 3:
        return get_irrep_phase_list("spinful_C3_phase_v1")
    if order == 2:
        return get_irrep_phase_list("spinful_C2_phase_v1")
    return []

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

    # Select table from validated package data.
    table = _irrep_table_for_order(operation_order)
    if not table:
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
    source_irrep_characters: Mapping[str, Mapping[int, complex]] | None = None,
    source_operation_maps: (
        Mapping[str, Mapping[str, Mapping[int, int]]] | None
    ) = None,
    source_irrep_characters_flattened: (
        Mapping[str, Mapping[str, Mapping[str, Mapping[int, complex]]]] | None
    ) = None,
    source_payload_blocked_rows: list[Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    """Build per-(kpoint, valley) irrep matching results.

    Cross-references the workflow decision report with character diagnostics
    from the symmetry-adapted valley analysis.

    Strategy boundary:

    - **Generic mode** (``source_irrep_characters``,
      ``source_irrep_characters_flattened``, ``source_operation_maps``, or
      ``source_payload_blocked_rows`` supplied): uses the generic
      ``match_restricted_characters`` strategy.
      Legacy ``by_kpoint`` phase-table entries are suppressed for any
      ``(kpoint, valley)`` that has generic coverage (including blocked rows),
      because the generic source is the authoritative matching path.

    - **Legacy mode** (no generic source attempted): uses the legacy
      C2/C3 phase-table fallback exclusively.  ``generic_matches_by_kpoint``
      is absent.
    """
    _generic_mode = (
        source_irrep_characters is not None
        or source_irrep_characters_flattened is not None
        or source_operation_maps is not None
        or (source_payload_blocked_rows is not None and len(source_payload_blocked_rows) > 0)
    )

    if irrep_workflow_decisions is None:
        return {
            "status": "not_evaluated",
            "matching_mode": "generic" if _generic_mode else "legacy",
            "matching_statuses": list(MATCHING_STATUSES),
            "tables_implemented": ["spinful_C3", "spinful_C2"],
            "legacy_tables_implemented": ["spinful_C3", "spinful_C2"],
            "by_kpoint": {},
        }

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
                v_raw = subspace.get("reference_valley")
                if not v_raw:
                    orbit = subspace.get("orbit", [])
                    if not orbit:
                        continue
                    v_raw = orbit[0]
                v = str(v_raw)
                char_diag = subspace.get("valley_preserving_character_diagnostics", {})
                sg = subspace.get("subspace_group", {})
                ssg = subspace.get("subspace_space_group", {})
                sa_by_kp.setdefault(kp_name, {})[v] = {
                    "char_diag": char_diag,
                    "subspace_group": sg,
                    "subspace_space_group": ssg,
                    "hsp_preserving_operation_ids": subspace.get(
                        "hsp_preserving_operation_ids", []
                    ),
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
                            result["matching_strategy"] = "legacy_phase_table"
                        result["workflow_path"] = path
                        result["readiness_level"] = readiness
                        result["subspace_group_candidate"] = sg_candidate
                        result["operation_id"] = op_id
                        op_matches[str(op_id)] = result
            if op_matches:
                kp_matches[v_name] = op_matches
        if kp_matches:
            by_kpoint[kp_name] = kp_matches

    # --- Generic restricted-character matching (optional) ---
    generic_matches: dict[str, dict[str, dict[str, object]]] = {}
    if source_irrep_characters is not None and source_operation_maps is not None:
        from valleyscope.analysis.generic_irrep_matching import (
            match_restricted_characters,
        )
        for kp_name, v_maps in source_operation_maps.items():
            if not isinstance(v_maps, Mapping):
                continue
            for v_name, op_map in v_maps.items():
                if not isinstance(op_map, Mapping):
                    continue
                # Collect ValleyScope computed characters from character
                # diagnostics for this (kpoint, valley).
                sa = sa_by_kp.get(kp_name, {}).get(v_name, {})
                sg = sa.get("subspace_group", {})
                ssg = sa.get("subspace_space_group", {})
                decision = (
                    decisions_by_kp.get(kp_name, {}).get(v_name, {})
                    if isinstance(decisions_by_kp.get(kp_name, {}), dict)
                    else {}
                )
                readiness = (
                    str(decision.get("readiness_level", ""))
                    if isinstance(decision, dict)
                    else ""
                )
                path = (
                    str(decision.get("workflow_path", ""))
                    if isinstance(decision, dict)
                    else ""
                )
                char_diag_g = sa.get("char_diag", {})
                per_valley_g = char_diag_g.get("per_valley", {})
                items = (
                    per_valley_g.get(v_name, [])
                    if isinstance(per_valley_g, dict)
                    else []
                )
                computed, item_op_ids = _computed_characters_from_items(items)
                vp_ids = (
                    _operation_ids_from_value(
                        ssg.get("valley_preserving_operation_ids")
                        if isinstance(ssg, Mapping)
                        else None
                    )
                    or item_op_ids
                )
                hsp_ids = (
                    _operation_ids_from_value(
                        sa.get("hsp_preserving_operation_ids")
                    )
                    or vp_ids
                )
                if not vp_ids or not computed:
                    continue

                if readiness != READINESS_TRUSTED:
                    status = (
                        "blocked"
                        if readiness == READINESS_BLOCKED
                        else "diagnostic_only"
                    )
                    generic_matches.setdefault(kp_name, {})[v_name] = {
                        "matching_status": status,
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {},
                        "source_operation_map": dict(op_map),
                        "valley_preserving_operation_ids": vp_ids,
                        "hsp_little_group_operation_ids": hsp_ids,
                        "diagnostic_only": True,
                        "reason": f"readiness_level={readiness} is not trusted",
                        "workflow_path": path,
                        "readiness_level": readiness,
                        "subspace_group_candidate": _generic_group_identity(
                            sg=sg, ssg=ssg,
                        ),
                        "subspace_space_group": dict(ssg)
                        if isinstance(ssg, Mapping) else {},
                    }
                    continue

                # Run generic matcher.
                g_result = match_restricted_characters(
                    computed_characters=computed,
                    source_irrep_characters=source_irrep_characters,
                    valley_preserving_operation_ids=vp_ids,
                    source_operation_map=dict(op_map),
                    hsp_little_group_operation_ids=hsp_ids,
                )
                generic_matches.setdefault(kp_name, {})[v_name] = {
                    "matching_status": g_result["matching_status"],
                    "matching_strategy": g_result["matching_strategy"],
                    "irrep_multiplicities": g_result.get("irrep_multiplicities", {}),
                    "source_operation_map": g_result.get("source_operation_map", {}),
                    "valley_preserving_operation_ids": vp_ids,
                    "hsp_little_group_operation_ids": hsp_ids,
                    "diagnostic_only": g_result.get("diagnostic_only", False),
                    "reason": g_result.get("reason", ""),
                    "workflow_path": path,
                    "readiness_level": readiness,
                    "subspace_group_candidate": _generic_group_identity(
                        sg=sg, ssg=ssg,
                    ),
                    "subspace_space_group": dict(ssg)
                    if isinstance(ssg, Mapping) else {},
                }

    # --- Generic matching from flattened per-row source payloads ---
    if source_irrep_characters_flattened is not None:
        from valleyscope.analysis.generic_irrep_matching import (
            match_restricted_characters,
        )
        for kp_name, valleys in source_irrep_characters_flattened.items():
            if not isinstance(valleys, Mapping):
                continue
            for v_name, per_row_chars in valleys.items():
                if not isinstance(per_row_chars, Mapping):
                    continue
                sa = sa_by_kp.get(kp_name, {}).get(v_name, {})
                sg_fl = sa.get("subspace_group", {})
                ssg_fl = sa.get("subspace_space_group", {})
                decision = (
                    decisions_by_kp.get(kp_name, {}).get(v_name, {})
                    if isinstance(decisions_by_kp.get(kp_name, {}), dict)
                    else {}
                )
                readiness = str(decision.get("readiness_level", ""))
                path_wf = str(decision.get("workflow_path", ""))
                op_map = (
                    source_operation_maps.get(kp_name, {}).get(v_name, {})
                    if source_operation_maps and isinstance(source_operation_maps, Mapping)
                    else {}
                )
                # Extract character items list for this valley, matching existing path.
                char_diag_fl = sa.get("char_diag", {})
                per_valley_fl = char_diag_fl.get("per_valley", {}) if isinstance(char_diag_fl, dict) else {}
                items_fl = (
                    per_valley_fl.get(v_name, [])
                    if isinstance(per_valley_fl, dict)
                    else []
                )
                computed_fl, item_op_ids = _computed_characters_from_items(items_fl)
                # Use full G_k^(a) from subspace_space_group, including identity.
                vp_ids_fl = (
                    _operation_ids_from_value(
                        ssg_fl.get("valley_preserving_operation_ids", [])
                    ) if isinstance(ssg_fl, dict) and _operation_ids_from_value(
                        ssg_fl.get("valley_preserving_operation_ids", [])
                    ) else item_op_ids
                )
                if not vp_ids_fl or not computed_fl or not isinstance(op_map, Mapping) or not op_map:
                    generic_matches.setdefault(kp_name, {})[v_name] = {
                        "matching_status": "blocked",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {},
                        "source_operation_map": dict(op_map) if isinstance(op_map, Mapping) else {},
                        "valley_preserving_operation_ids": vp_ids_fl,
                        "diagnostic_only": True,
                        "reason": "no_computed_character_data_or_operation_map",
                        "workflow_path": path_wf,
                        "readiness_level": readiness,
                        "subspace_group_candidate": (
                            _generic_group_identity(sg=sg_fl, ssg=ssg_fl)
                        ),
                        "subspace_space_group": dict(ssg_fl)
                        if isinstance(ssg_fl, Mapping) else {},
                    }
                    continue
                if readiness not in ("trusted",):
                    generic_matches.setdefault(kp_name, {})[v_name] = {
                        "matching_status": "blocked",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {},
                        "source_operation_map": dict(op_map),
                        "valley_preserving_operation_ids": vp_ids_fl,
                        "diagnostic_only": True,
                        "reason": f"readiness_level={readiness} is not trusted",
                        "workflow_path": path_wf,
                        "readiness_level": readiness,
                        "subspace_group_candidate": (
                            _generic_group_identity(sg=sg_fl, ssg=ssg_fl)
                        ),
                        "subspace_space_group": dict(ssg_fl)
                        if isinstance(ssg_fl, Mapping) else {},
                    }
                    continue
                g_result = match_restricted_characters(
                    computed_characters=computed_fl,
                    source_irrep_characters=per_row_chars,
                    valley_preserving_operation_ids=vp_ids_fl,
                    source_operation_map=dict(op_map),
                )
                generic_matches.setdefault(kp_name, {})[v_name] = {
                    "matching_status": g_result["matching_status"],
                    "matching_strategy": g_result["matching_strategy"],
                    "irrep_multiplicities": g_result.get("irrep_multiplicities", {}),
                    "source_operation_map": g_result.get("source_operation_map", {}),
                    "valley_preserving_operation_ids": vp_ids_fl,
                    "diagnostic_only": g_result.get("diagnostic_only", False),
                    "reason": g_result.get("reason", ""),
                    "workflow_path": path_wf,
                    "readiness_level": readiness,
                    "subspace_group_candidate": (
                        _generic_group_identity(sg=sg_fl, ssg=ssg_fl)
                    ),
                    "subspace_space_group": dict(ssg_fl)
                    if isinstance(ssg_fl, Mapping) else {},
                }

    # --- Blocked source-payload rows (adapter/preflight diagnostics) ---
    if source_payload_blocked_rows:
        for row in source_payload_blocked_rows:
            if not isinstance(row, Mapping):
                continue
            kp_name = row.get("kpoint")
            v_name = row.get("valley")
            if not isinstance(kp_name, str) or not isinstance(v_name, str):
                continue
            existing = generic_matches.get(kp_name, {}).get(v_name)
            if existing is not None and existing.get("matching_status") == "matched":
                continue
            blocker_reasons = row.get("blocker_reasons", row.get("reason", ""))
            if isinstance(blocker_reasons, list):
                reason = "; ".join(str(reason) for reason in blocker_reasons)
            else:
                reason = str(blocker_reasons)
            provenance = {
                key: value
                for key, value in row.items()
                if key not in {"kpoint", "valley", "blocker_reasons", "reason"}
            }
            generic_matches.setdefault(kp_name, {})[v_name] = {
                "matching_status": "blocked",
                "matching_strategy": "bilbao_restricted_character",
                "irrep_multiplicities": {},
                "source_operation_map": dict(row.get("source_operation_map", {}))
                if isinstance(row.get("source_operation_map", {}), Mapping)
                else {},
                "valley_preserving_operation_ids": _operation_ids_from_value(
                    row.get("valley_preserving_operation_ids", [])
                ),
                "diagnostic_only": True,
                "reason": reason,
                "source_payload_status": "blocked",
                "source_payload_provenance": provenance,
            }

    # --- Rows with operation maps but no source characters ---
    # A source_operation_map is already a row-level generic matching attempt.
    # If no source characters produced a generic row, expose the row as
    # blocked instead of letting legacy phase-table output become authoritative.
    if source_operation_maps is not None and isinstance(source_operation_maps, Mapping):
        for kp_name, v_maps in source_operation_maps.items():
            if not isinstance(kp_name, str) or not isinstance(v_maps, Mapping):
                continue
            for v_name, op_map in v_maps.items():
                if not isinstance(v_name, str) or not isinstance(op_map, Mapping):
                    continue
                existing = generic_matches.get(kp_name, {}).get(v_name)
                if existing is not None:
                    continue
                sa = sa_by_kp.get(kp_name, {}).get(v_name, {})
                ssg = sa.get("subspace_space_group", {})
                decision = (
                    decisions_by_kp.get(kp_name, {}).get(v_name, {})
                    if isinstance(decisions_by_kp.get(kp_name, {}), dict)
                    else {}
                )
                readiness = (
                    str(decision.get("readiness_level", ""))
                    if isinstance(decision, dict)
                    else ""
                )
                path = (
                    str(decision.get("workflow_path", ""))
                    if isinstance(decision, dict)
                    else ""
                )
                char_diag = sa.get("char_diag", {})
                per_valley = char_diag.get("per_valley", {}) if isinstance(char_diag, dict) else {}
                items = per_valley.get(v_name, []) if isinstance(per_valley, dict) else []
                _, item_op_ids = _computed_characters_from_items(items)
                vp_ids = (
                    _operation_ids_from_value(
                        ssg.get("valley_preserving_operation_ids")
                        if isinstance(ssg, Mapping)
                        else None
                    )
                    or item_op_ids
                    or _operation_ids_from_value(op_map)
                )
                hsp_ids = (
                    _operation_ids_from_value(sa.get("hsp_preserving_operation_ids"))
                    or vp_ids
                )
                generic_matches.setdefault(kp_name, {})[v_name] = {
                    "matching_status": "blocked",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {},
                    "source_operation_map": dict(op_map),
                    "valley_preserving_operation_ids": vp_ids,
                    "hsp_little_group_operation_ids": hsp_ids,
                    "diagnostic_only": True,
                    "reason": (
                        "missing_source_irrep_characters: source_operation_maps "
                        "was supplied without source irrep characters for this row"
                    ),
                    "workflow_path": path,
                    "readiness_level": readiness,
                    "subspace_group_candidate": (
                        sa.get("subspace_group", {}).get("subspace_group_candidate")
                        if isinstance(sa.get("subspace_group", {}), Mapping)
                        else None
                    ),
                    "subspace_space_group": dict(ssg)
                    if isinstance(ssg, Mapping) else {},
                }

    # --- Strategy boundary: in generic mode, suppress legacy by_kpoint entries ---
    # for (kpoint, valley) pairs that have generic coverage.  The generic
    # source (including blocked rows) is the authoritative matching path.
    if _generic_mode and generic_matches:
        _filter_legacy_rows_in_generic_mode(
            by_kpoint=by_kpoint,
            generic_matches=generic_matches,
        )

    return {
        "status": "ok" if (by_kpoint or generic_matches) else "not_evaluated",
        "matching_mode": "generic" if _generic_mode else "legacy",
        "matching_statuses": list(MATCHING_STATUSES),
        "tables_implemented": ["spinful_C3", "spinful_C2"],
        "legacy_tables_implemented": ["spinful_C3", "spinful_C2"],
        "by_kpoint": by_kpoint,
        **({"generic_matches_by_kpoint": generic_matches}
           if generic_matches else {}),
    }


def _operation_ids_from_value(value: object) -> list[int]:
    if isinstance(value, Mapping):
        iterable = value.keys()
    elif isinstance(value, (list, tuple, set)):
        iterable = value
    else:
        return []
    out: list[int] = []
    for op_id in iterable:
        if isinstance(op_id, int) and not isinstance(op_id, bool) and op_id not in out:
            out.append(op_id)
    return out


def _computed_characters_from_items(
    items: object,
) -> tuple[dict[int, complex], list[int]]:
    computed: dict[int, complex] = {}
    op_ids: list[int] = []
    if not isinstance(items, list):
        return computed, op_ids
    for item in items:
        if not isinstance(item, dict):
            continue
        op_id = item.get("operation_id")
        if not isinstance(op_id, int) or isinstance(op_id, bool):
            continue
        eigenphases = item.get("eigenphases")
        if not isinstance(eigenphases, list) or not eigenphases:
            continue
        phases = [
            float(phase)
            for phase in eigenphases
            if isinstance(phase, (int, float)) and not isinstance(phase, bool)
        ]
        if not phases:
            continue
        computed[op_id] = sum(exp(2j * math.pi * phase) for phase in phases)
        if op_id not in op_ids:
            op_ids.append(op_id)
    return computed, op_ids


def _generic_group_identity(
    *,
    sg: object,
    ssg: object,
) -> str | None:
    """Return the physical subspace-space-group symbol for generic matches.

    In generic mode, ``subspace_group_candidate`` should use the physical
    ``candidate_space_group_symbol`` from ``subspace_space_group`` when
    available, falling back to the legacy ``subspace_group_candidate``
    only when no physical symbol exists.
    """
    if isinstance(ssg, Mapping):
        physical = ssg.get("candidate_space_group_symbol")
        if physical and isinstance(physical, str):
            return str(physical)
    if isinstance(sg, Mapping):
        legacy = sg.get("subspace_group_candidate")
        if legacy and isinstance(legacy, str):
            return str(legacy)
    return None


def _filter_legacy_rows_in_generic_mode(
    *,
    by_kpoint: dict[str, object],
    generic_matches: dict[str, dict[str, dict[str, object]]],
) -> None:
    """Remove legacy by_kpoint entries for (kpoint, valley) pairs that have
    generic coverage in generic_matches (success, diagnostic, or blocked).

    Generic source is the authoritative matching path.  Legacy phase-table
    entries are kept only for (kpoint, valley) pairs with no generic data.
    """
    # Collect (kpoint, valley) pairs with generic coverage.
    covered: set[tuple[str, str]] = set()
    for kp_name, valleys in generic_matches.items():
        if not isinstance(valleys, dict):
            continue
        for v_name in valleys:
            covered.add((str(kp_name), str(v_name)))

    if not covered:
        return

    for kp_name in list(by_kpoint):
        v_dict = by_kpoint.get(kp_name)
        if not isinstance(v_dict, dict):
            continue
        for v_name in list(v_dict):
            if (str(kp_name), str(v_name)) in covered:
                del v_dict[str(v_name)]
        if not v_dict:
            del by_kpoint[str(kp_name)]
