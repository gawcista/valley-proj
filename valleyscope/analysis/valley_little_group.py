from __future__ import annotations

from typing import Any
from urllib.parse import quote

import numpy as np
import spglib

from valleyscope.irreps.matching import (
    decompose_characters_into_irreps,
    match_single_state_irrep,
)
from valleyscope.irreps.tables import load_standard_irrep_table, match_table_operations
from valleyscope.symmetry.little_group import is_little_group_operation


def update_valley_preserving_operation_inventory(
    *,
    symmetry_payload: dict[str, Any],
    kpoint_name: str,
    k_frac: np.ndarray,
    valley_names: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Classify detected operations per (kpoint, valley) pair."""
    if valley_names is None:
        valley_names = _infer_valley_names(symmetry_payload)

    per_valley: dict[str, list[dict[str, Any]]] = {}
    flat_inventory: list[dict[str, Any]] = []

    for operation in symmetry_payload.get("detected_operations", []):
        rotation = np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
        little_group_passed = bool(is_little_group_operation(rotation, k_frac))
        sector_mapping = operation.get("sector_mapping", {})

        # Per-valley classification within the HSP little group.
        for valley_name in valley_names:
            mapped_valley = sector_mapping.get(valley_name)
            valley_preserving = bool(
                mapped_valley is not None and str(mapped_valley) == str(valley_name)
            )
            allowed = bool(little_group_passed and valley_preserving)
            reason = _per_valley_rejection_reason(
                little_group_passed=little_group_passed,
                valley_preserving=valley_preserving,
                mapped_valley=mapped_valley,
            )

            row = {
                "operation_id": operation.get("operation_id"),
                "kind": operation.get("kind", ""),
                "order": operation.get("order"),
                "little_group_passed": little_group_passed,
                "target_valley": valley_name,
                "mapped_valley": mapped_valley,
                "valley_preserving": valley_preserving,
                "allowed_for_valley_preserving_representation": allowed,
                "reason": reason,
            }
            per_valley.setdefault(valley_name, []).append(row)

        # Flat all-valley-intersection row for diagnostics only.
        old_preserves_all = _preserves_all_valley_subspaces(operation)
        old_exchanging = _is_valley_exchanging(operation)
        old_allowed = bool(little_group_passed and old_preserves_all)
        old_reason = _rejection_reason(
            little_group_passed=little_group_passed,
            valley_preserving=old_preserves_all,
            valley_exchanging=old_exchanging,
        )
        flat_row = {
            "operation_id": operation.get("operation_id"),
            "kind": operation.get("kind", ""),
            "order": operation.get("order"),
            "little_group_passed": little_group_passed,
            "valley_preserving": old_preserves_all,
            "valley_exchanging": old_exchanging,
            "allowed_for_valley_preserving_representation": old_allowed,
            "reason": old_reason,
        }
        flat_inventory.append(flat_row)

        operation.setdefault("little_group_by_kpoint", {})[kpoint_name] = little_group_passed
        operation.setdefault("allowed_for_valley_preserving_representation_by_kpoint", {})[kpoint_name] = old_allowed
        operation["allowed_for_valley_preserving_representation"] = old_allowed
        operation.setdefault("rejection_reason_by_kpoint", {})[kpoint_name] = old_reason

    symmetry_payload.setdefault("hsp_little_group_inventory", {})[kpoint_name] = flat_inventory
    symmetry_payload.setdefault("per_valley_preserving_operation_inventory", {}).setdefault(kpoint_name, {}).update(per_valley)
    symmetry_payload.setdefault("kpoint_frac_by_name", {})[kpoint_name] = np.asarray(
        k_frac, dtype=float,
    ).tolist()
    symmetry_payload.setdefault("valley_names", list(valley_names))
    return per_valley


def build_valley_preserving_subgroup_report(
    *,
    symmetry_payload: dict[str, Any],
    target_kpoints: list[str] | tuple[str, ...] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Build the valley orbit and valley-preserving subgroup report."""
    per_valley_inventories = symmetry_payload.get("per_valley_preserving_operation_inventory", {})
    valley_names = symmetry_payload.get("valley_names", [])
    inventories = symmetry_payload.get("hsp_little_group_inventory", {})

    if not isinstance(inventories, dict) or not inventories:
        return {
            "status": "not_available",
            "standard_group_match": None,
            "standard_group_match_status": "not_attempted",
            "by_kpoint": {},
        }

    operation_lookup = {
        operation.get("operation_id"): operation
        for operation in symmetry_payload.get("detected_operations", [])
    }

    # All-valley intersection is debug-only and not used for valley-preserving irreps.
    all_valley_intersection = _global_valley_preserving_operation_set_report(
        operation_lookup=operation_lookup,
        tolerance=tolerance,
    )
    all_valley_intersection["interpretation"] = (
        "intersection of operations preserving ALL selected valleys; "
        "this is NOT a per-valley preserving subgroup and is provided for debugging only"
    )

    valley_preserving_subgroups: dict[str, Any] = {}
    for valley_name in valley_names:
        preserving_ids = [
            op_id for op_id, op in operation_lookup.items()
            if _operation_preserves_valley(op, valley_name)
        ]
        closure = _operation_set_closure_report(
            allowed_ids=preserving_ids,
            operation_lookup=operation_lookup,
            tolerance=tolerance,
        )
        valley_preserving_subgroups[valley_name] = {
            "operation_ids": preserving_ids,
            "closure_status": closure["closure_status"],
            "identity_operation_id": closure["identity_operation_id"],
            "missing_products": closure["missing_products"],
        }

    # Valley orbits
    valley_orbits = _compute_valley_orbits(operation_lookup, valley_names)

    if target_kpoints is None:
        ordered_kpoints = list(inventories)
    else:
        ordered_kpoints = [kpoint for kpoint in target_kpoints if kpoint in inventories]
        ordered_kpoints.extend(kpoint for kpoint in inventories if kpoint not in ordered_kpoints)

    by_kpoint: dict[str, Any] = {}
    for kpoint in ordered_kpoints:
        kp_entry: dict[str, Any] = {}
        for valley_name in valley_names:
            # Get per-valley inventory for this (kpoint, valley)
            pv_inv = (
                per_valley_inventories.get(kpoint, {}).get(valley_name, [])
                if per_valley_inventories
                else []
            )
            if not pv_inv:
                # Fallback: build from flat inventory
                flat_inv = inventories.get(kpoint, [])
                if isinstance(flat_inv, list):
                    pv_inv = _build_per_valley_from_flat(flat_inv, valley_name, operation_lookup)

            allowed_ids = [
                row.get("operation_id")
                for row in pv_inv
                if bool(row.get("allowed_for_valley_preserving_representation", False))
            ]
            little_ids = [
                row.get("operation_id")
                for row in pv_inv
                if bool(row.get("little_group_passed", False))
            ]
            valley_changing_ids = [
                row.get("operation_id")
                for row in pv_inv
                if bool(row.get("little_group_passed", False))
                and not bool(row.get("valley_preserving", False))
                and row.get("mapped_valley") is not None
                and str(row.get("mapped_valley", "")) != str(valley_name)
            ]
            outside_little_ids = [
                row.get("operation_id")
                for row in pv_inv
                if not bool(row.get("little_group_passed", False))
            ]
            closure = _operation_set_closure_report(
                allowed_ids=allowed_ids,
                operation_lookup=operation_lookup,
                tolerance=tolerance,
            )
            kp_entry[valley_name] = {
                "little_group_operation_ids": little_ids,
                "allowed_operation_ids": allowed_ids,
                "valley_changing_operation_ids": valley_changing_ids,
                "outside_little_group_operation_ids": outside_little_ids,
                "operation_count": len(allowed_ids),
                "closure_status": closure["closure_status"],
                "identity_operation_id": closure["identity_operation_id"],
                "missing_products": closure["missing_products"],
            }

        # Legacy backward-compat fields at kpoint level
        flat_inv = inventories.get(kpoint, [])
        if isinstance(flat_inv, list):
            lg_ids = [
                row.get("operation_id")
                for row in flat_inv
                if bool(row.get("little_group_passed", False))
            ]
            old_allowed = [
                row.get("operation_id")
                for row in flat_inv
                if bool(row.get("allowed_for_valley_preserving_representation", False))
            ]
            kp_entry["hsp_little_group_operation_ids"] = lg_ids
            kp_entry["all_valley_intersection_operation_ids"] = old_allowed

        by_kpoint[kpoint] = kp_entry

    # Standard group match from per-valley preserving subgroups.
    per_valley_standard_matches: dict[str, Any] = {}
    for valley_name in valley_names:
        preserving_ids = valley_preserving_subgroups.get(valley_name, {}).get("operation_ids", [])
        match, match_status = _match_standard_group_from_operations(
            allowed_ids=preserving_ids,
            operation_lookup=operation_lookup,
            lattice_direct_cart=symmetry_payload.get("lattice_direct_cart"),
            symprec=float(symmetry_payload.get("symprec", tolerance)),
        )
        per_valley_standard_matches[valley_name] = {
            "standard_group_match": match,
            "standard_group_match_status": match_status,
        }

    status = "per_valley_preserving_subgroups_computed"
    report = {
        "status": status,
        "interpretation": (
            "For each valley a, G_k^(a) = {g in G_k | pi_g(a) = a}. "
            "Valley-preserving irreps use these subgroups, NOT the all-valley intersection."
        ),
        "valley_orbits": valley_orbits,
        "valley_preserving_subgroups": valley_preserving_subgroups,
        "per_valley_standard_matches": per_valley_standard_matches,
        "all_valley_intersection": all_valley_intersection,
        "irrep_matching": _build_per_valley_irrep_table_matching(
            symmetry_payload=symmetry_payload,
            operation_lookup=operation_lookup,
            valley_names=valley_names,
            valley_preserving_subgroups=valley_preserving_subgroups,
            per_valley_standard_matches=per_valley_standard_matches,
            by_kpoint=by_kpoint,
            tolerance=tolerance,
        ),
        "by_kpoint": by_kpoint,
    }
    symmetry_payload["valley_preserving_subgroup_report"] = report
    return report


def add_valley_irrep_results(
    *,
    symmetry_payload: dict[str, Any],
    symmetry_rows: list[dict[str, Any]],
    representation_payload: dict[str, Any] | None = None,
    state_diagonal_tol: float = 1e-3,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Match characters to valley-preserving irrep tables.

    Keys irrep matching by (kpoint, valley), not only by kpoint.
    """
    report = symmetry_payload.get("valley_preserving_subgroup_report", {})
    if not isinstance(report, dict):
        return {}
    matching = report.get("irrep_matching", {})
    if not isinstance(matching, dict):
        return {}
    if matching.get("status") != "table_mapping_complete" and not matching.get("per_valley"):
        matching["character_matching_status"] = "not_attempted"
        return matching

    valley_names = symmetry_payload.get("valley_names", [])
    per_valley_table_info = matching.get("per_valley", {})
    results_by_kpoint: dict[str, Any] = {}
    all_matched = True

    for kpoint, kpoint_info in matching.get("by_kpoint", {}).items():
        if not isinstance(kpoint_info, dict):
            all_matched = False
            continue
        kp_results: dict[str, Any] = {}

        for valley_name in valley_names:
            valley_table_info = per_valley_table_info.get(valley_name, {})
            if valley_table_info.get("operation_to_table_mapping_status") != "complete":
                kp_results[valley_name] = {
                    "status": "table_mapping_incomplete",
                    "failure_reasons": ["Per-valley table mapping is not complete"],
                    "state_irrep_assignment_status": "not_attempted",
                    "state_irrep_results": [],
                }
                all_matched = False
                continue

            table = _cached_load_irrep_table(
                int(valley_table_info["spacegroup_number"]),
                spinor=bool(valley_table_info.get("spinor", False)),
            )
            if table is None:
                kp_results[valley_name] = {
                    "status": "table_load_failed",
                    "failure_reasons": ["Failed to load irrep table"],
                    "state_irrep_assignment_status": "not_attempted",
                    "state_irrep_results": [],
                }
                all_matched = False
                continue

            operation_to_table = dict(valley_table_info.get("operation_to_table_mapping", {}))
            table_to_operation = {v: k for k, v in operation_to_table.items()}

            v_kp_info = valley_table_info.get("by_kpoint", {}).get(kpoint, {})
            if not isinstance(v_kp_info, dict) or v_kp_info.get("status") != "table_kpoint_matched":
                kp_results[valley_name] = {
                    "status": "table_kpoint_not_ready",
                    "failure_reasons": ["Table k-point mapping is not complete"],
                    "state_irrep_assignment_status": "not_attempted",
                    "state_irrep_results": [],
                }
                all_matched = False
                continue

            character_data = _collect_valley_characters(
                symmetry_rows=symmetry_rows,
                kpoint=kpoint,
                target_valley=valley_name,
                operation_to_table=operation_to_table,
            )
            computed_characters = dict(character_data["computed_characters"])
            identity_source = _fill_identity_character_if_needed(
                computed_characters=computed_characters,
                table_operation_indices=v_kp_info.get("table_operation_indices", []),
                ready_row_counts=character_data["ready_row_counts"],
            )
            match_result = decompose_characters_into_irreps(
                table=table,
                table_kpoint_label=str(v_kp_info["table_kpoint_label"]),
                computed_characters=computed_characters,
                tolerance=tolerance,
            )

            state_diag = _collect_state_diagonal_characters(
                representation_payload=representation_payload,
                kpoint=kpoint,
                table_to_operation=table_to_operation,
                table_operation_indices=v_kp_info.get("table_operation_indices", []),
                symmetry_rows=symmetry_rows,
                operation_to_table=operation_to_table,
                target_valley=valley_name,
                state_diagonal_tol=state_diagonal_tol,
            )
            state_irrep_result = _match_single_state_irreps(
                table=table,
                table_kpoint_label=str(v_kp_info["table_kpoint_label"]),
                table_operation_indices=v_kp_info.get("table_operation_indices", []),
                state_characters=state_diag["state_characters"],
                tolerance=tolerance,
            )
            kp_results[valley_name] = {
                "status": match_result.status,
                "table_kpoint_label": match_result.table_kpoint_label,
                "computed_characters": _format_complex_character_dict(match_result.computed_characters),
                "identity_character_source": identity_source,
                "irrep_weights": match_result.irrep_weights,
                "irrep_multiplicities": match_result.irrep_multiplicities,
                "missing_table_operation_indices": match_result.missing_table_operation_indices,
                "failure_reasons": match_result.failure_reasons,
                "state_irrep_assignment_status": state_irrep_result["status"],
                "state_irrep_results": state_irrep_result["results"],
            }
            if match_result.status != "matched":
                all_matched = False

        results_by_kpoint[kpoint] = kp_results

    all_matched = bool(results_by_kpoint) and all_matched
    matching["character_matching_status"] = "matched" if all_matched else "incomplete"
    matching["label_matching"] = "matched" if all_matched else "deferred"
    matching["irrep_results_by_kpoint"] = results_by_kpoint
    return matching


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IRREP_TABLE_CACHE: dict[tuple[int, bool], Any] = {}


def _cached_load_irrep_table(number: int, spinor: bool = False):
    key = (number, spinor)
    if key not in _IRREP_TABLE_CACHE:
        try:
            _IRREP_TABLE_CACHE[key] = load_standard_irrep_table(number, spinor=spinor)
        except Exception:
            _IRREP_TABLE_CACHE[key] = None
    return _IRREP_TABLE_CACHE[key]


def _operation_preserves_valley(operation: dict[str, Any], valley_name: str) -> bool:
    mapping = operation.get("sector_mapping", {})
    target = mapping.get(valley_name)
    return target is not None and str(target) == str(valley_name)


def _preserves_all_valley_subspaces(operation: dict[str, Any]) -> bool:
    preserved = operation.get("preserved", {})
    if not isinstance(preserved, dict) or not preserved:
        return False
    return all(bool(value) for value in preserved.values())


def _per_valley_rejection_reason(
    *,
    little_group_passed: bool,
    valley_preserving: bool,
    mapped_valley: Any,
) -> str:
    if not little_group_passed:
        return "not in little group"
    if valley_preserving:
        return ""
    if mapped_valley is not None:
        return f"valley-changing (maps to {mapped_valley})"
    return "not valley preserving"


def _infer_valley_names(symmetry_payload: dict[str, Any]) -> list[str]:
    for op in symmetry_payload.get("detected_operations", []):
        preserved = op.get("preserved", {})
        if isinstance(preserved, dict) and preserved:
            return list(preserved.keys())
    mapping = op.get("sector_mapping", {}) if symmetry_payload.get("detected_operations") else {}
    if isinstance(mapping, dict) and mapping:
        return list(mapping.keys())
    return []


def _build_per_valley_from_flat(
    flat_inventory: list[dict[str, Any]],
    valley_name: str,
    operation_lookup: dict[Any, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fallback: build per-valley rows from flat inventory and operation data."""
    rows: list[dict[str, Any]] = []
    for flat_row in flat_inventory:
        op_id = flat_row.get("operation_id")
        op = operation_lookup.get(op_id, {})
        sector_mapping = op.get("sector_mapping", {})
        mapped_valley = sector_mapping.get(valley_name)
        valley_preserving = bool(
            mapped_valley is not None and str(mapped_valley) == str(valley_name)
        )
        little = bool(flat_row.get("little_group_passed", False))
        allowed = bool(little and valley_preserving)
        reason = _per_valley_rejection_reason(
            little_group_passed=little,
            valley_preserving=valley_preserving,
            mapped_valley=mapped_valley,
        )
        rows.append({
            "operation_id": op_id,
            "kind": flat_row.get("kind", ""),
            "order": flat_row.get("order"),
            "little_group_passed": little,
            "target_valley": valley_name,
            "mapped_valley": mapped_valley,
            "valley_preserving": valley_preserving,
            "allowed_for_valley_preserving_representation": allowed,
            "reason": reason,
        })
    return rows


def _compute_valley_orbits(
    operation_lookup: dict[Any, dict[str, Any]],
    valley_names: list[str],
) -> list[dict[str, Any]]:
    """Compute valley orbits under the detected operations."""
    if not valley_names:
        return []

    # Build adjacency: which valleys map to which
    adjacency: dict[str, set[str]] = {v: {v} for v in valley_names}
    for op in operation_lookup.values():
        mapping = op.get("sector_mapping", {})
        for src, tgt in mapping.items():
            if tgt is not None and src in adjacency:
                adjacency[src].add(str(tgt))

    # Find connected components
    visited: set[str] = set()
    orbits: list[dict[str, Any]] = []

    for valley_name in valley_names:
        if valley_name in visited:
            continue
        # BFS
        component: list[str] = []
        queue = [valley_name]
        while queue:
            v = queue.pop(0)
            if v in visited:
                continue
            visited.add(v)
            component.append(v)
            for neighbor in adjacency.get(v, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        # Operation mappings for this orbit
        op_mappings: list[dict[str, Any]] = []
        coset_reps: list[Any] = []
        for op_id, op in operation_lookup.items():
            mapping = op.get("sector_mapping", {})
            relevant = False
            for v in component:
                tgt = mapping.get(v)
                if tgt is not None and str(tgt) != str(v):
                    relevant = True
                    break
            if relevant:
                op_mappings.append({
                    "operation_id": op_id,
                    "kind": op.get("kind", ""),
                    "order": op.get("order"),
                    "mapping": {k: v for k, v in mapping.items() if k in component},
                })
            # Check if this operation maps between different valleys in the orbit
            mapped_set = set()
            for v in component:
                tgt = mapping.get(v)
                if tgt is not None:
                    mapped_set.add(str(tgt))
            if len(mapped_set) > 1:
                coset_reps.append(op_id)

        orbits.append({
            "valleys": component,
            "operation_mappings": op_mappings,
            "valley_permuting_operation_ids": coset_reps,
            "coset_representative_operation_ids": coset_reps,
        })

    return orbits


def _build_per_valley_irrep_table_matching(
    *,
    symmetry_payload: dict[str, Any],
    operation_lookup: dict[Any, dict[str, Any]],
    valley_names: list[str],
    valley_preserving_subgroups: dict[str, Any],
    per_valley_standard_matches: dict[str, Any],
    by_kpoint: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    base = {
        "table_source": "irreptables",
        "label_matching": "deferred",
        "reason": (
            "automatic irrep labels use valley-preserving subgroups; "
            "not emitted until character-to-table matching succeeds"
        ),
    }

    spinor = bool(symmetry_payload.get("spinor_wavefunction", False))
    per_valley: dict[str, Any] = {}
    all_complete = True

    for valley_name in valley_names:
        match_info = per_valley_standard_matches.get(valley_name, {})
        standard_match = match_info.get("standard_group_match")
        match_status = match_info.get("standard_group_match_status", "not_attempted")

        if match_status != "matched" or standard_match is None:
            per_valley[valley_name] = {
                "status": "table_mapping_deferred",
                "reason": f"No standard group match for {valley_name} valley-preserving subgroup",
                "by_kpoint": {},
            }
            all_complete = False
            continue

        try:
            table = load_standard_irrep_table(int(standard_match["number"]), spinor=spinor)
        except Exception as exc:
            per_valley[valley_name] = {
                "status": "table_load_failed",
                "spacegroup_number": int(standard_match["number"]),
                "spinor": spinor,
                "reason": str(exc),
                "by_kpoint": {},
            }
            all_complete = False
            continue

        preserving_ids = valley_preserving_subgroups.get(valley_name, {}).get("operation_ids", [])
        preserving_operations = [
            operation_lookup[op_id]
            for op_id in preserving_ids
            if op_id in operation_lookup
        ]
        mapping_report = match_table_operations(
            preserving_operations, table, tolerance=tolerance,
        )

        kpoint_matching = _build_table_kpoint_matching(
            by_kpoint=by_kpoint,
            kpoint_frac_by_name=symmetry_payload.get("kpoint_frac_by_name", {}),
            table=table,
            operation_mapping=mapping_report.mapping_by_operation_id,
            tolerance=max(tolerance, 1e-6),
            valley_name=valley_name,
        )
        complete = mapping_report.status == "complete" and all(
            row.get("status") == "table_kpoint_matched"
            for row in kpoint_matching.values()
        )
        if not complete:
            all_complete = False

        per_valley[valley_name] = {
            "status": "table_mapping_complete" if complete else "table_mapping_incomplete",
            "spacegroup_number": table.number,
            "table_name": table.name,
            "spinor": table.spinor,
            "operation_to_table_mapping_status": mapping_report.status,
            "operation_to_table_mapping": mapping_report.mapping_by_operation_id,
            "unmatched_operation_ids": mapping_report.unmatched_operation_ids,
            "unused_table_operation_indices": mapping_report.unused_table_operation_indices,
            "by_kpoint": kpoint_matching,
        }

    return {
        **base,
        "status": "table_mapping_complete" if all_complete else "table_mapping_incomplete",
        "per_valley": per_valley,
        "by_kpoint": {
            kpoint: {
                valley_name: per_valley.get(valley_name, {}).get("by_kpoint", {}).get(kpoint, {})
                for valley_name in valley_names
            }
            for kpoint in by_kpoint
        },
    }


def _build_table_kpoint_matching(
    *,
    by_kpoint: dict[str, Any],
    kpoint_frac_by_name: Any,
    table,
    operation_mapping: dict[Any, int],
    tolerance: float,
    valley_name: str | None = None,
) -> dict[str, Any]:
    if not isinstance(kpoint_frac_by_name, dict):
        kpoint_frac_by_name = {}
    matching_by_kpoint: dict[str, Any] = {}
    for kpoint, payload in by_kpoint.items():
        k_frac = kpoint_frac_by_name.get(kpoint)
        # Get per-valley allowed ids
        if valley_name is not None and isinstance(payload, dict):
            v_payload = payload.get(valley_name, {})
            allowed_ids = list(v_payload.get("allowed_operation_ids", [])) if isinstance(v_payload, dict) else []
        else:
            allowed_ids = list(payload.get("allowed_operation_ids", [])) if isinstance(payload, dict) else []
        mapped_indices = sorted(
            operation_mapping[operation_id]
            for operation_id in allowed_ids
            if operation_id in operation_mapping
        )

        if k_frac is None:
            matching_by_kpoint[kpoint] = {
                "status": "missing_kpoint_coordinate",
                "table_kpoint_label": None,
                "mapped_allowed_table_operation_indices": mapped_indices,
            }
            continue
        table_label = table.match_kpoint_label(np.asarray(k_frac, dtype=float), tolerance=tolerance)
        match_source = "coordinate"
        if table_label is None:
            candidates = _table_labels_with_operation_indices(table, mapped_indices)
            if len(candidates) == 1:
                table_label = candidates[0]
                match_source = "operation_set_fallback"
            else:
                matching_by_kpoint[kpoint] = {
                    "status": "table_kpoint_ambiguous" if candidates else "table_kpoint_not_matched",
                    "input_k_frac": list(np.asarray(k_frac, dtype=float)),
                    "table_kpoint_label": None,
                    "mapped_allowed_table_operation_indices": mapped_indices,
                    "candidate_table_kpoint_labels": candidates,
                }
                continue
        table_indices = table.operation_indices_for_kpoint(table_label)
        missing_indices = [
            table_index
            for table_index in table_indices
            if table_index not in mapped_indices
        ]
        extra_indices = [
            table_index
            for table_index in mapped_indices
            if table_index not in table_indices
        ]
        status = (
            "table_kpoint_matched"
            if not missing_indices and not extra_indices
            else "table_operation_set_mismatch"
        )
        matching_by_kpoint[kpoint] = {
            "status": status,
            "input_k_frac": list(np.asarray(k_frac, dtype=float)),
            "table_kpoint_label": table_label,
            "table_kpoint_match_source": match_source,
            "table_k_frac": list(table.irreps_by_kpoint(table_label)[0].k_frac),
            "table_operation_indices": table_indices,
            "mapped_allowed_table_operation_indices": mapped_indices,
            "missing_table_operation_indices": missing_indices,
            "extra_mapped_table_operation_indices": extra_indices,
            "available_irrep_labels": [
                irrep.label for irrep in table.irreps_by_kpoint(table_label)
            ],
        }
    return matching_by_kpoint


def _table_labels_with_operation_indices(table, operation_indices: list[int]) -> list[str]:
    target = sorted(operation_indices)
    labels = sorted({irrep.kpoint_label for irrep in table.irreps})
    return [
        label for label in labels
        if table.operation_indices_for_kpoint(label) == target
    ]


def _collect_valley_characters(
    *,
    symmetry_rows: list[dict[str, Any]],
    kpoint: str,
    target_valley: str,
    operation_to_table: dict[Any, int],
) -> dict[str, Any]:
    """Collect trace-level characters for a specific (kpoint, valley) pair."""
    computed_characters: dict[int, complex] = {}
    ready_row_counts: dict[int, int] = {}
    rows_by_table_operation: dict[int, list[dict[str, Any]]] = {}

    for row in symmetry_rows:
        if str(row.get("kpoint", "")) != kpoint:
            continue
        row_valley = str(row.get("target_valley", ""))
        if row_valley and row_valley != target_valley:
            continue
        if not bool(row.get("little_group_passed", False)):
            continue
        if not bool(row.get("valley_preserving", False)):
            continue
        operation_id = row.get("operation_id")
        if operation_id not in operation_to_table:
            continue
        table_index = operation_to_table[operation_id]
        rows_by_table_operation.setdefault(table_index, []).append(row)

    for table_index, rows in rows_by_table_operation.items():
        if not rows or not all(bool(row.get("topology_input_ready", False)) for row in rows):
            continue
        ready_row_counts[table_index] = len(rows)
        character = next(
            (row.get("character_valley", "") for row in rows if row.get("character_valley", "")),
            "",
        )
        if character:
            computed_characters[table_index] = _parse_complex_character(character)
    return {
        "computed_characters": computed_characters,
        "ready_row_counts": ready_row_counts,
    }


def _collect_computed_characters(
    *,
    symmetry_rows: list[dict[str, Any]],
    kpoint: str,
    operation_to_table: dict[Any, int],
) -> dict[str, Any]:
    """Legacy: collect characters without valley filtering (backward compat)."""
    computed_characters: dict[int, complex] = {}
    ready_row_counts: dict[int, int] = {}
    rows_by_table_operation: dict[int, list[dict[str, Any]]] = {}

    for row in symmetry_rows:
        if str(row.get("kpoint", "")) != kpoint:
            continue
        if not bool(row.get("little_group_passed", False)):
            continue
        if not bool(row.get("valley_preserving", False)):
            continue
        operation_id = row.get("operation_id")
        if operation_id not in operation_to_table:
            continue
        table_index = operation_to_table[operation_id]
        rows_by_table_operation.setdefault(table_index, []).append(row)

    for table_index, rows in rows_by_table_operation.items():
        if not rows or not all(bool(row.get("topology_input_ready", False)) for row in rows):
            continue
        ready_row_counts[table_index] = len(rows)
        character = next(
            (row.get("character_valley", "") for row in rows if row.get("character_valley", "")),
            "",
        )
        if character:
            computed_characters[table_index] = _parse_complex_character(character)
    return {
        "computed_characters": computed_characters,
        "ready_row_counts": ready_row_counts,
    }


def _collect_state_diagonal_characters(
    *,
    representation_payload: dict[str, Any] | None,
    kpoint: str,
    table_to_operation: dict[int, Any],
    table_operation_indices: Any,
    symmetry_rows: list[dict[str, Any]],
    operation_to_table: dict[Any, int],
    target_valley: str | None = None,
    state_diagonal_tol: float = 1e-3,
) -> dict[str, Any]:
    """Collect per-state characters from D_valley diagonal, filtered by target_valley."""
    state_characters: dict[int, dict[int, complex]] = {}
    offdiag_warnings: list[str] = []

    if representation_payload is None:
        return {"state_characters": state_characters, "offdiag_warnings": ["no_representation_payload"]}

    kp_representations = representation_payload.get(kpoint, {})
    if not isinstance(kp_representations, dict):
        return {"state_characters": state_characters, "offdiag_warnings": ["no_kpoint_representations"]}

    rows_by_table: dict[int, list[dict[str, Any]]] = {}
    for row in symmetry_rows:
        if str(row.get("kpoint", "")) != kpoint:
            continue
        row_valley = str(row.get("target_valley", ""))
        if target_valley is not None and row_valley and row_valley != target_valley:
            continue
        if not bool(row.get("little_group_passed", False)):
            continue
        if not bool(row.get("valley_preserving", False)):
            continue
        operation_id = row.get("operation_id")
        if operation_id not in operation_to_table:
            continue
        table_index = operation_to_table[operation_id]
        rows_by_table.setdefault(table_index, []).append(row)

    ready_table_indices: set[int] = set()
    for table_index, rows in rows_by_table.items():
        if rows and all(bool(r.get("topology_input_ready", False)) for r in rows):
            ready_table_indices.add(table_index)

    for table_index in table_operation_indices:
        if table_index not in ready_table_indices:
            continue
        operation_id = table_to_operation.get(table_index)
        if operation_id is None:
            continue
        op_payload = kp_representations.get(_representation_payload_key(operation_id, target_valley or ""))
        if op_payload is None:
            op_payload = kp_representations.get(f"operation_{operation_id}")
        if op_payload is None and target_valley is not None:
            prefix = f"operation_{operation_id}__valley_"
            op_payload = next(
                (
                    payload
                    for key, payload in kp_representations.items()
                    if str(key).startswith(prefix)
                    and isinstance(payload, dict)
                    and str(payload.get("target_valley", "")) == str(target_valley)
                ),
                None,
            )
        if op_payload is None:
            op_payload = kp_representations.get(str(operation_id), {})
        if not isinstance(op_payload, dict):
            continue
        d_valley = op_payload.get("D_valley")
        if d_valley is None:
            continue
        d_valley = np.asarray(d_valley, dtype=np.complex128)
        if d_valley.ndim != 2 or d_valley.shape[0] != d_valley.shape[1]:
            continue
        n = d_valley.shape[0]
        if n < 1:
            continue

        off_diag = d_valley.copy()
        np.fill_diagonal(off_diag, 0.0)
        off_norm = float(np.linalg.norm(off_diag))
        if off_norm > state_diagonal_tol:
            offdiag_warnings.append(
                f"D_valley off-diagonal norm {off_norm:.2e} exceeds "
                f"state_diagonal_tol={state_diagonal_tol:.2e} for operation {operation_id}"
            )
            continue

        for i in range(n):
            state_characters.setdefault(i, {})[table_index] = d_valley[i, i]

    return {
        "state_characters": state_characters,
        "offdiag_warnings": offdiag_warnings,
    }


def _match_single_state_irreps(
    *,
    table,
    table_kpoint_label: str,
    table_operation_indices: Any,
    state_characters: dict[int, dict[int, complex]],
    tolerance: float,
) -> dict[str, Any]:
    table_indices = list(table_operation_indices)
    results: list[dict[str, Any]] = []
    for state_index in sorted(state_characters):
        characters = dict(state_characters[state_index])
        if 1 in table_indices and 1 not in characters:
            characters[1] = 1.0 + 0.0j
        result = match_single_state_irrep(
            table=table,
            table_kpoint_label=table_kpoint_label,
            state_index=state_index,
            computed_characters=characters,
            tolerance=tolerance,
        )
        results.append(
            {
                "state_index": result.state_index,
                "status": result.status,
                "irrep_label": result.irrep_label,
                "computed_characters": _format_complex_character_dict(result.computed_characters),
                "irrep_multiplicities": result.irrep_multiplicities,
                "missing_table_operation_indices": result.missing_table_operation_indices,
                "failure_reasons": result.failure_reasons,
            }
        )

    if not results:
        status = "not_attempted"
    elif all(result["status"] == "matched" for result in results):
        status = "matched"
    else:
        status = "incomplete"
    return {"status": status, "results": results}


def _fill_identity_character_if_needed(
    *,
    computed_characters: dict[int, complex],
    table_operation_indices: Any,
    ready_row_counts: dict[int, int],
) -> str:
    if 1 not in table_operation_indices:
        return "not_required"
    if 1 in computed_characters:
        return "computed"
    if not ready_row_counts:
        return "missing"
    inferred_dimension = max(ready_row_counts.values())
    computed_characters[1] = complex(float(inferred_dimension), 0.0)
    return "inferred_from_ready_rows"


def _parse_complex_character(value: Any) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float, np.number)):
        return complex(float(value), 0.0)
    return complex(str(value).replace(" ", ""))


def _format_complex_character_dict(values: dict[int, complex]) -> dict[str, str]:
    return {
        str(table_index): f"{value.real:.6f}{value.imag:+.6f}j"
        for table_index, value in sorted(values.items())
    }


def _representation_payload_key(operation_id: Any, target_valley: str) -> str:
    return f"operation_{operation_id}__valley_{quote(str(target_valley), safe='')}"


def _global_valley_preserving_operation_set_report(
    *,
    operation_lookup: dict[Any, dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    allowed_ids = [
        operation_id
        for operation_id, operation in operation_lookup.items()
        if _preserves_all_valley_subspaces(operation)
    ]
    valley_exchanging_ids = [
        operation_id
        for operation_id, operation in operation_lookup.items()
        if _is_valley_exchanging(operation)
    ]
    not_valley_preserving_ids = [
        operation_id
        for operation_id, operation in operation_lookup.items()
        if not _preserves_all_valley_subspaces(operation)
    ]
    closure = _operation_set_closure_report(
        allowed_ids=allowed_ids,
        operation_lookup=operation_lookup,
        tolerance=tolerance,
    )
    return {
        "operation_set_label": "all_valley_intersection",
        "interpretation": "intersection of operations preserving ALL selected valleys (debug only)",
        "allowed_operation_ids": allowed_ids,
        "valley_exchanging_operation_ids": valley_exchanging_ids,
        "not_valley_preserving_operation_ids": not_valley_preserving_ids,
        "operation_count": len(allowed_ids),
        "closure_status": closure["closure_status"],
        "identity_operation_id": closure["identity_operation_id"],
        "missing_products": closure["missing_products"],
    }


def _match_standard_group_from_operations(
    *,
    allowed_ids: list[Any],
    operation_lookup: dict[Any, dict[str, Any]],
    lattice_direct_cart: Any,
    symprec: float,
) -> tuple[dict[str, Any] | None, str]:
    if lattice_direct_cart is None:
        return None, "not_attempted"
    if not allowed_ids:
        return None, "not_matched"
    match_from_symmetry = getattr(spglib, "get_spacegroup_type_from_symmetry", None)
    if match_from_symmetry is None:
        return None, "not_matched"

    rotations = []
    translations = []
    for operation_id in allowed_ids:
        operation = operation_lookup.get(operation_id)
        if operation is None:
            return None, "not_matched"
        rotation = np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
        translation = np.asarray(operation.get("translation_frac", np.zeros(3)), dtype=float)
        rotations.append(np.rint(rotation).astype(int))
        translations.append(translation)

    try:
        spacegroup_type = match_from_symmetry(
            np.asarray(rotations, dtype=int),
            np.asarray(translations, dtype=float),
            lattice=np.asarray(lattice_direct_cart, dtype=float),
            symprec=symprec,
        )
    except Exception:
        return None, "not_matched"
    if spacegroup_type is None:
        return None, "not_matched"

    return {
        "number": int(spacegroup_type.number),
        "international_short": str(spacegroup_type.international_short),
        "international": str(spacegroup_type.international),
        "hall_number": int(spacegroup_type.hall_number),
        "hall_symbol": str(spacegroup_type.hall_symbol),
        "pointgroup_international": str(spacegroup_type.pointgroup_international),
        "source": "spglib.get_spacegroup_type_from_symmetry",
        "symprec": float(symprec),
        "operation_ids": list(allowed_ids),
    }, "matched"


def _is_valley_exchanging(operation: dict[str, Any]) -> bool:
    mapping = operation.get("sector_mapping", {})
    if not isinstance(mapping, dict):
        return False
    for source, target in mapping.items():
        if target is not None and str(target) != str(source):
            return True
    return False


def _rejection_reason(
    *,
    little_group_passed: bool,
    valley_preserving: bool,
    valley_exchanging: bool,
) -> str:
    if not little_group_passed:
        return "not in little group"
    if valley_preserving:
        return ""
    if valley_exchanging:
        return "valley-exchanging"
    return "not valley preserving"


def _operation_set_closure_report(
    *,
    allowed_ids: list[Any],
    operation_lookup: dict[Any, dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    missing_products: list[dict[str, Any]] = []
    identity_operation_id = _find_identity_operation_id(
        allowed_ids=allowed_ids,
        operation_lookup=operation_lookup,
        tolerance=tolerance,
    )
    if not allowed_ids:
        return {
            "closure_status": "empty",
            "identity_operation_id": identity_operation_id,
            "missing_products": missing_products,
        }

    for left_id in allowed_ids:
        for right_id in allowed_ids:
            left = operation_lookup.get(left_id)
            right = operation_lookup.get(right_id)
            if left is None or right is None:
                missing_products.append(
                    {"left_operation_id": left_id, "right_operation_id": right_id}
                )
                continue
            product_rotation, product_translation = _compose_fractional_operations(left, right)
            if _find_matching_operation_id(
                allowed_ids=allowed_ids,
                operation_lookup=operation_lookup,
                rotation_frac=product_rotation,
                translation_frac=product_translation,
                tolerance=tolerance,
            ) is None:
                missing_products.append(
                    {"left_operation_id": left_id, "right_operation_id": right_id}
                )

    return {
        "closure_status": "closed" if not missing_products else "not_closed",
        "identity_operation_id": identity_operation_id,
        "missing_products": missing_products,
    }


def _compose_fractional_operations(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    left_rotation = np.asarray(left.get("rotation_frac", np.eye(3)), dtype=float)
    right_rotation = np.asarray(right.get("rotation_frac", np.eye(3)), dtype=float)
    left_translation = np.asarray(left.get("translation_frac", np.zeros(3)), dtype=float)
    right_translation = np.asarray(right.get("translation_frac", np.zeros(3)), dtype=float)
    rotation = left_rotation @ right_rotation
    translation = left_rotation @ right_translation + left_translation
    return rotation, translation


def _find_identity_operation_id(
    *,
    allowed_ids: list[Any],
    operation_lookup: dict[Any, dict[str, Any]],
    tolerance: float,
) -> Any:
    for operation_id in allowed_ids:
        operation = operation_lookup.get(operation_id)
        if operation is None:
            continue
        if _rotation_matches(np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float), np.eye(3), tolerance):
            translation = np.asarray(operation.get("translation_frac", np.zeros(3)), dtype=float)
            if _translation_matches(translation, np.zeros(3), tolerance):
                return operation_id
    return None


def _find_matching_operation_id(
    *,
    allowed_ids: list[Any],
    operation_lookup: dict[Any, dict[str, Any]],
    rotation_frac: np.ndarray,
    translation_frac: np.ndarray,
    tolerance: float,
) -> Any:
    for operation_id in allowed_ids:
        operation = operation_lookup.get(operation_id)
        if operation is None:
            continue
        candidate_rotation = np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
        candidate_translation = np.asarray(operation.get("translation_frac", np.zeros(3)), dtype=float)
        if _rotation_matches(rotation_frac, candidate_rotation, tolerance) and _translation_matches(
            translation_frac, candidate_translation, tolerance,
        ):
            return operation_id
    return None


def _rotation_matches(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    return bool(np.allclose(left, right, atol=tolerance, rtol=0.0))


def _translation_matches(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    delta_mod_lattice = delta - np.rint(delta)
    return bool(np.linalg.norm(delta_mod_lattice) <= tolerance)
