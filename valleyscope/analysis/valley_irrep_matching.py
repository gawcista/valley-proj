"""Valley-preserving irrep matching report builder.

Builds the irrep matching report from workflow decisions, symmetry-adapted
valley analysis, and optional generic Bilbao/irreptables restricted-character
source data.  The generic restricted-character path is the production matching
strategy.  No legacy C2/C3 phase-table matching is available.
"""

from __future__ import annotations

from collections.abc import Mapping
from cmath import exp
import math
from typing import Any

READINESS_TRUSTED = "trusted"
READINESS_USABLE_WITH_CAUTION = "usable_with_caution"
READINESS_BLOCKED = "blocked"
MATCHING_STATUSES = [
    "matched", "diagnostic", "diagnostic_only", "not_applicable",
    "failed_no_table", "failed_ambiguous", "blocked",
    "identity_only_not_irrep_distinguishing",
]


# ---------------------------------------------------------------------------
# Batch matching across the workflow decision report
# ---------------------------------------------------------------------------

def build_valley_irrep_matching_report(
    *,
    irrep_workflow_decisions: dict[str, object] | None,
    symmetry_adapted_valley_report: dict[str, object] | None,
    allow_caution: bool = False,
    source_operation_maps: (
        Mapping[str, Mapping[str, Mapping[int, int]]] | None
    ) = None,
    source_irrep_characters_flattened: (
        Mapping[str, Mapping[str, Mapping[str, Mapping[int, complex]]]] | None
    ) = None,
    source_payload_provenance: (
        Mapping[str, Mapping[str, Mapping[str, Any]]] | None
    ) = None,
    source_payload_blocked_rows: list[Mapping[str, Any]] | None = None,
    source_payload_classification_rows: list[Mapping[str, Any]] | None = None,
    resolved_subspace_groups: (
        Mapping[str, Mapping[str, Mapping[str, object]]] | None
    ) = None,
) -> dict[str, object]:
    """Build per-(kpoint, valley) irrep matching results.

    Cross-references the workflow decision report with character diagnostics
    from the symmetry-adapted valley analysis using the generic
    ``match_restricted_characters`` strategy over ``G_k^(a)``.

    ``resolved_subspace_groups`` carries the canonical per-valley
    subspace-space-group identity resolved from
    ``per_valley_standard_matches``.  When provided, it replaces the
    SA-report ``subspace_space_group`` (which may be unresolved).
    """
    _generic_mode = (
        source_irrep_characters_flattened is not None
        or source_operation_maps is not None
        or source_payload_provenance is not None
        or (source_payload_blocked_rows is not None and len(source_payload_blocked_rows) > 0)
        or (
            source_payload_classification_rows is not None
            and len(source_payload_classification_rows) > 0
        )
    )

    if irrep_workflow_decisions is None:
        return {
            "status": "not_evaluated",
            "matching_mode": "generic" if _generic_mode else "not_evaluated",
            "matching_statuses": list(MATCHING_STATUSES),
        }

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

    # --- Resolve canonical subgroup identity ---
    def _canonical_ssg(
        kp_name: str, v_name: str, sa_ssg: object,
    ) -> dict[str, object]:
        """Return the canonical subspace_space_group for a (kpoint, valley).

        Merge the resolved identity from per_valley_standard_matches over the
        SA-report operation inventory so valley-changing provenance is kept.
        """
        merged = dict(sa_ssg) if isinstance(sa_ssg, Mapping) else {}
        if isinstance(resolved_subspace_groups, Mapping):
            r = resolved_subspace_groups.get(kp_name, {}).get(v_name)
            if isinstance(r, Mapping) and r:
                merged.update(r)
        return merged

    generic_matches: dict[str, dict[str, dict[str, object]]] = {}

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
                ssg_raw = sa.get("subspace_space_group", {})
                ssg_fl = _canonical_ssg(kp_name, v_name, ssg_raw)
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
                source_provenance = _source_payload_provenance_fields(
                    source_payload_provenance,
                    kp_name,
                    v_name,
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
                # G_k^(a) = HSP little-group ops ∩ valley-preserving ops.
                vp_ids_fl = (
                    _operation_ids_from_value(
                        ssg_fl.get("valley_preserving_operation_ids", [])
                    ) if isinstance(ssg_fl, dict) and _operation_ids_from_value(
                        ssg_fl.get("valley_preserving_operation_ids", [])
                    ) else item_op_ids
                )
                hsp_lg_fl = (
                    _operation_ids_from_value(
                        sa.get("hsp_preserving_operation_ids")
                    )
                )
                hsp_ids_fl = hsp_lg_fl or vp_ids_fl
                if hsp_lg_fl:
                    vp_ids_fl = [op for op in vp_ids_fl if op in hsp_lg_fl]
                if not vp_ids_fl or not computed_fl or not isinstance(op_map, Mapping) or not op_map:
                    generic_matches.setdefault(kp_name, {})[v_name] = {
                        "matching_status": "blocked",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {},
                        "source_operation_map": dict(op_map) if isinstance(op_map, Mapping) else {},
                        "valley_preserving_operation_ids": vp_ids_fl,
                        "hsp_little_group_operation_ids": hsp_ids_fl,
                        "diagnostic_only": True,
                        "reason": "no_computed_character_data_or_operation_map",
                        "workflow_path": path_wf,
                        "readiness_level": readiness,
                        "subspace_group_candidate": (
                            _generic_group_identity(sg=sg_fl, ssg=ssg_fl)
                        ),
                        "subspace_space_group": dict(ssg_fl)
                        if isinstance(ssg_fl, Mapping) else {},
                        **source_provenance,
                    }
                    continue
                if readiness not in ("trusted",):
                    generic_matches.setdefault(kp_name, {})[v_name] = {
                        "matching_status": "blocked",
                        "matching_strategy": "bilbao_restricted_character",
                        "irrep_multiplicities": {},
                        "source_operation_map": dict(op_map),
                        "valley_preserving_operation_ids": vp_ids_fl,
                        "hsp_little_group_operation_ids": hsp_ids_fl,
                        "diagnostic_only": True,
                        "reason": f"readiness_level={readiness} is not trusted",
                        "workflow_path": path_wf,
                        "readiness_level": readiness,
                        "subspace_group_candidate": (
                            _generic_group_identity(sg=sg_fl, ssg=ssg_fl)
                        ),
                        "subspace_space_group": dict(ssg_fl)
                        if isinstance(ssg_fl, Mapping) else {},
                        **source_provenance,
                    }
                    continue
                g_result = match_restricted_characters(
                    computed_characters=computed_fl,
                    source_irrep_characters=per_row_chars,
                    valley_preserving_operation_ids=vp_ids_fl,
                    source_operation_map=dict(op_map),
                    hsp_little_group_operation_ids=hsp_ids_fl,
                )
                generic_matches.setdefault(kp_name, {})[v_name] = {
                    "matching_status": g_result["matching_status"],
                    "matching_strategy": g_result["matching_strategy"],
                    "irrep_multiplicities": g_result.get("irrep_multiplicities", {}),
                    "source_operation_map": g_result.get("source_operation_map", {}),
                    "valley_preserving_operation_ids": vp_ids_fl,
                    "hsp_little_group_operation_ids": hsp_ids_fl,
                    "diagnostic_only": g_result.get("diagnostic_only", False),
                    "reason": g_result.get("reason", ""),
                    "workflow_path": path_wf,
                    "readiness_level": readiness,
                    "subspace_group_candidate": (
                        _generic_group_identity(sg=sg_fl, ssg=ssg_fl)
                    ),
                    "subspace_space_group": dict(ssg_fl)
                    if isinstance(ssg_fl, Mapping) else {},
                    **source_provenance,
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
            # Carry resolved subspace_space_group to top level so downstream
            # layers (compact rows, EBR candidates) see the SG identity even
            # when the source HSP label cannot be assigned.
            ssg = row.get("subspace_space_group")
            if not isinstance(ssg, Mapping):
                ssg = provenance.get("subspace_space_group")
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
                "hsp_little_group_operation_ids": _operation_ids_from_value(
                    row.get("hsp_little_group_operation_ids", [])
                ),
                "diagnostic_only": True,
                "reason": reason,
                "source_payload_status": "blocked",
                "source_payload_provenance": provenance,
                **({"subspace_space_group": dict(ssg)}
                   if isinstance(ssg, Mapping) else {}),
                **({
                    "projected_hsp_classification": dict(
                        row["projected_hsp_classification"]
                    ),
                    "source_hsp_membership": bool(
                        row["projected_hsp_classification"].get(
                            "source_hsp_membership", False
                        )
                    ),
                } if isinstance(
                    row.get("projected_hsp_classification"), Mapping
                ) else {}),
            }

    # --- Valid local rows outside the reviewed source-HSP basis ---
    if source_payload_classification_rows:
        for row in source_payload_classification_rows:
            if not isinstance(row, Mapping):
                continue
            kp_name = row.get("kpoint")
            v_name = row.get("valley")
            classification = row.get("projected_hsp_classification")
            if (
                not isinstance(kp_name, str)
                or not isinstance(v_name, str)
                or not isinstance(classification, Mapping)
                or classification.get("classification") != "generic"
                or classification.get("validation_status") != "validated"
            ):
                continue
            decision = (
                decisions_by_kp.get(kp_name, {}).get(v_name, {})
                if isinstance(decisions_by_kp.get(kp_name, {}), dict)
                else {}
            )
            ssg = row.get("subspace_space_group", {})
            generic_matches.setdefault(kp_name, {})[v_name] = {
                "matching_status": "not_applicable",
                "matching_strategy": "bilbao_restricted_character",
                "irrep_multiplicities": {},
                "source_operation_map": {},
                "valley_preserving_operation_ids": _operation_ids_from_value(
                    row.get("valley_preserving_operation_ids", [])
                ),
                "hsp_little_group_operation_ids": _operation_ids_from_value(
                    row.get("hsp_little_group_operation_ids", [])
                ),
                "diagnostic_only": False,
                "reason": "generic_projected_subspace_k",
                "workflow_path": str(decision.get("workflow_path", "")),
                "readiness_level": str(decision.get("readiness_level", "")),
                "subspace_space_group": dict(ssg)
                if isinstance(ssg, Mapping) else {},
                "projected_hsp_classification": dict(classification),
                "source_hsp_membership": False,
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
                source_provenance = _source_payload_provenance_fields(
                    source_payload_provenance,
                    kp_name,
                    v_name,
                )
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
                if hsp_ids:
                    vp_ids = [op for op in vp_ids if op in hsp_ids]
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
                        _generic_group_identity(sg=sg, ssg=ssg)
                    ),
                    "subspace_space_group": dict(ssg)
                    if isinstance(ssg, Mapping) else {},
                    **source_provenance,
                }

    # --- Strategy boundary: in generic mode, suppress legacy by_kpoint entries ---
    # for (kpoint, valley) pairs that have generic coverage.  The generic
    # source (including blocked rows) is the authoritative matching path.

    return {
        "status": "ok" if generic_matches else "not_evaluated",
        "matching_mode": "generic" if _generic_mode else "not_evaluated",
        "matching_statuses": list(MATCHING_STATUSES),
        **({"generic_matches_by_kpoint": generic_matches}
           if generic_matches else {}),
    }


def _source_payload_provenance_fields(
    source_payload_provenance: (
        Mapping[str, Mapping[str, Mapping[str, Any]]] | None
    ),
    kpoint: str,
    valley: str,
) -> dict[str, Any]:
    if not isinstance(source_payload_provenance, Mapping):
        return {}
    by_valley = source_payload_provenance.get(kpoint)
    if not isinstance(by_valley, Mapping):
        return {}
    provenance = by_valley.get(valley)
    if not isinstance(provenance, Mapping):
        return {}
    out: dict[str, Any] = {"source_payload_provenance": dict(provenance)}
    op_mapping = provenance.get("operation_mapping_provenance")
    if isinstance(op_mapping, str) and op_mapping:
        out["operation_mapping_provenance"] = op_mapping
    classification = provenance.get("projected_hsp_classification")
    if isinstance(classification, Mapping):
        out["projected_hsp_classification"] = dict(classification)
        out["source_hsp_membership"] = bool(
            classification.get("source_hsp_membership", False)
        )
    return out


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
    available, falling back to ``subspace_group_candidate`` only when no
    physical symbol exists.
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
