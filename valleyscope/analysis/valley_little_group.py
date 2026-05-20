from __future__ import annotations

from typing import Any

import numpy as np
import spglib

from valleyscope.irreps.matching import (
    decompose_characters_into_irreps,
    match_single_state_irrep,
)
from valleyscope.irreps.tables import load_standard_irrep_table, match_table_operations
from valleyscope.symmetry.little_group import is_little_group_operation


def update_valley_little_group_inventory(
    *,
    symmetry_payload: dict[str, Any],
    kpoint_name: str,
    k_frac: np.ndarray,
) -> list[dict[str, Any]]:
    """Classify detected operations for valley-preserving little-group analysis."""
    inventory: list[dict[str, Any]] = []
    for operation in symmetry_payload.get("detected_operations", []):
        rotation = np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
        little_group_passed = bool(is_little_group_operation(rotation, k_frac))
        valley_preserving = _preserves_all_valley_subspaces(operation)
        valley_exchanging = _is_valley_exchanging(operation)
        allowed = bool(little_group_passed and valley_preserving)
        reason = _rejection_reason(
            little_group_passed=little_group_passed,
            valley_preserving=valley_preserving,
            valley_exchanging=valley_exchanging,
        )

        row = {
            "operation_id": operation.get("operation_id"),
            "kind": operation.get("kind", ""),
            "order": operation.get("order"),
            "little_group_passed": little_group_passed,
            "valley_preserving": valley_preserving,
            "valley_exchanging": valley_exchanging,
            "allowed_for_single_valley_representation": allowed,
            "reason": reason,
        }
        inventory.append(row)

        operation.setdefault("little_group_by_kpoint", {})[kpoint_name] = little_group_passed
        operation.setdefault("allowed_for_single_valley_representation_by_kpoint", {})[kpoint_name] = allowed
        operation["allowed_for_single_valley_representation"] = allowed
        operation.setdefault("rejection_reason_by_kpoint", {})[kpoint_name] = reason

    symmetry_payload.setdefault("valley_little_group_inventory", {})[kpoint_name] = inventory
    symmetry_payload.setdefault("kpoint_frac_by_name", {})[kpoint_name] = np.asarray(
        k_frac,
        dtype=float,
    ).tolist()
    return inventory


def build_valley_preserving_subgroup_report(
    *,
    symmetry_payload: dict[str, Any],
    target_kpoints: list[str] | tuple[str, ...] | None = None,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    inventories = symmetry_payload.get("valley_little_group_inventory", {})
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
    global_operation_set = _global_valley_preserving_operation_set_report(
        operation_lookup=operation_lookup,
        tolerance=tolerance,
    )
    standard_match, standard_match_status = _match_standard_group_from_operations(
        allowed_ids=global_operation_set["allowed_operation_ids"],
        operation_lookup=operation_lookup,
        lattice_direct_cart=symmetry_payload.get("lattice_direct_cart"),
        symprec=float(symmetry_payload.get("symprec", tolerance)),
    )
    if target_kpoints is None:
        ordered_kpoints = list(inventories)
    else:
        ordered_kpoints = [kpoint for kpoint in target_kpoints if kpoint in inventories]
        ordered_kpoints.extend(kpoint for kpoint in inventories if kpoint not in ordered_kpoints)

    by_kpoint: dict[str, Any] = {}
    for kpoint in ordered_kpoints:
        inventory = inventories.get(kpoint, [])
        if not isinstance(inventory, list):
            continue
        allowed_ids = [
            row.get("operation_id")
            for row in inventory
            if bool(row.get("allowed_for_single_valley_representation", False))
        ]
        little_ids = [
            row.get("operation_id")
            for row in inventory
            if bool(row.get("little_group_passed", False))
        ]
        valley_exchanging_ids = [
            row.get("operation_id")
            for row in inventory
            if bool(row.get("valley_exchanging", False))
        ]
        not_valley_preserving_ids = [
            row.get("operation_id")
            for row in inventory
            if bool(row.get("little_group_passed", False))
            and not bool(row.get("valley_preserving", False))
        ]
        outside_little_group_ids = [
            row.get("operation_id")
            for row in inventory
            if not bool(row.get("little_group_passed", False))
        ]
        closure = _operation_set_closure_report(
            allowed_ids=allowed_ids,
            operation_lookup=operation_lookup,
            tolerance=tolerance,
        )
        by_kpoint[kpoint] = {
            "operation_set_label": f"G_tau,k({kpoint})",
            "interpretation": "valley-preserving little-group operation set",
            "standard_group_match": None,
            "standard_group_match_status": "not_attempted",
            "little_group_operation_ids": little_ids,
            "allowed_operation_ids": allowed_ids,
            "valley_exchanging_operation_ids": valley_exchanging_ids,
            "not_valley_preserving_operation_ids": not_valley_preserving_ids,
            "outside_little_group_operation_ids": outside_little_group_ids,
            "operation_count": len(allowed_ids),
            "closure_status": closure["closure_status"],
            "identity_operation_id": closure["identity_operation_id"],
            "missing_products": closure["missing_products"],
        }

    status = (
        "standard_group_matched"
        if standard_match_status == "matched"
        else "operation_set_only"
    )
    report = {
        "status": status,
        "interpretation": (
            "G_tau is determined from detected valley-preserving operations; "
            "G_tau,k entries are valley-preserving little-group operation sets"
        ),
        "global_operation_set": global_operation_set,
        "standard_group_match": standard_match,
        "standard_group_match_status": standard_match_status,
        "irrep_matching": _build_irrep_table_matching(
            symmetry_payload=symmetry_payload,
            operation_lookup=operation_lookup,
            global_operation_set=global_operation_set,
            standard_match=standard_match,
            standard_match_status=standard_match_status,
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
    report = symmetry_payload.get("valley_preserving_subgroup_report", {})
    if not isinstance(report, dict):
        return {}
    matching = report.get("irrep_matching", {})
    if not isinstance(matching, dict):
        return {}
    if matching.get("status") != "table_mapping_complete":
        matching["character_matching_status"] = "not_attempted"
        return matching

    table = load_standard_irrep_table(
        int(matching["spacegroup_number"]),
        spinor=bool(matching.get("spinor", False)),
    )
    operation_to_table = dict(matching.get("operation_to_table_mapping", {}))
    table_to_operation = {v: k for k, v in operation_to_table.items()}
    results_by_kpoint: dict[str, Any] = {}
    for kpoint, kpoint_info in matching.get("by_kpoint", {}).items():
        if not isinstance(kpoint_info, dict) or kpoint_info.get("status") != "table_kpoint_matched":
            results_by_kpoint[kpoint] = {
                "status": "table_kpoint_not_ready",
                "failure_reasons": ["Table k-point mapping is not complete"],
                "state_irrep_assignment_status": "not_attempted",
                "state_irrep_results": [],
            }
            continue
        character_data = _collect_computed_characters(
            symmetry_rows=symmetry_rows,
            kpoint=kpoint,
            operation_to_table=operation_to_table,
        )
        computed_characters = dict(character_data["computed_characters"])
        identity_source = _fill_identity_character_if_needed(
            computed_characters=computed_characters,
            table_operation_indices=kpoint_info.get("table_operation_indices", []),
            ready_row_counts=character_data["ready_row_counts"],
        )
        match_result = decompose_characters_into_irreps(
            table=table,
            table_kpoint_label=str(kpoint_info["table_kpoint_label"]),
            computed_characters=computed_characters,
            tolerance=tolerance,
        )

        # State-level: collect from D_valley diagonal, not eigenvalue ordering
        state_diag = _collect_state_diagonal_characters(
            representation_payload=representation_payload,
            kpoint=kpoint,
            table_to_operation=table_to_operation,
            table_operation_indices=kpoint_info.get("table_operation_indices", []),
            symmetry_rows=symmetry_rows,
            operation_to_table=operation_to_table,
            state_diagonal_tol=state_diagonal_tol,
        )
        state_irrep_result = _match_single_state_irreps(
            table=table,
            table_kpoint_label=str(kpoint_info["table_kpoint_label"]),
            table_operation_indices=kpoint_info.get("table_operation_indices", []),
            state_characters=state_diag["state_characters"],
            tolerance=tolerance,
        )
        results_by_kpoint[kpoint] = {
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

    all_matched = bool(results_by_kpoint) and all(
        result.get("status") == "matched"
        for result in results_by_kpoint.values()
    )
    matching["character_matching_status"] = "matched" if all_matched else "incomplete"
    matching["label_matching"] = "matched" if all_matched else "deferred"
    matching["irrep_results_by_kpoint"] = results_by_kpoint
    return matching


def _preserves_all_valley_subspaces(operation: dict[str, Any]) -> bool:
    preserved = operation.get("preserved", {})
    if not isinstance(preserved, dict) or not preserved:
        return False
    return all(bool(value) for value in preserved.values())


def _build_irrep_table_matching(
    *,
    symmetry_payload: dict[str, Any],
    operation_lookup: dict[Any, dict[str, Any]],
    global_operation_set: dict[str, Any],
    standard_match: dict[str, Any] | None,
    standard_match_status: str,
    by_kpoint: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    base = {
        "table_source": "irreptables",
        "label_matching": "deferred",
        "reason": (
            "automatic irrep labels are not emitted until character-to-table "
            "matching is explicitly enabled"
        ),
    }
    if standard_match_status != "matched" or standard_match is None:
        return {
            **base,
            "status": "table_mapping_deferred",
            "operation_to_table_mapping_status": "not_attempted",
            "reason": "standard valley-preserving subgroup is not matched",
            "by_kpoint": {},
        }

    spinor = bool(symmetry_payload.get("spinor_wavefunction", False))
    try:
        table = load_standard_irrep_table(int(standard_match["number"]), spinor=spinor)
    except Exception as exc:
        return {
            **base,
            "status": "table_load_failed",
            "spacegroup_number": int(standard_match["number"]),
            "spinor": spinor,
            "operation_to_table_mapping_status": "not_attempted",
            "reason": str(exc),
            "by_kpoint": {},
        }

    allowed_operations = [
        operation_lookup[operation_id]
        for operation_id in global_operation_set.get("allowed_operation_ids", [])
        if operation_id in operation_lookup
    ]
    mapping_report = match_table_operations(
        allowed_operations,
        table,
        tolerance=tolerance,
    )
    kpoint_matching = _build_table_kpoint_matching(
        by_kpoint=by_kpoint,
        kpoint_frac_by_name=symmetry_payload.get("kpoint_frac_by_name", {}),
        table=table,
        operation_mapping=mapping_report.mapping_by_operation_id,
        tolerance=max(tolerance, 1e-6),
    )
    by_kpoint_complete = all(
        row.get("status") == "table_kpoint_matched"
        for row in kpoint_matching.values()
    )
    mapping_complete = mapping_report.status == "complete" and by_kpoint_complete
    return {
        **base,
        "status": "table_mapping_complete" if mapping_complete else "table_mapping_incomplete",
        "spacegroup_number": table.number,
        "table_name": table.name,
        "spinor": table.spinor,
        "operation_to_table_mapping_status": mapping_report.status,
        "operation_to_table_mapping": mapping_report.mapping_by_operation_id,
        "unmatched_operation_ids": mapping_report.unmatched_operation_ids,
        "unused_table_operation_indices": mapping_report.unused_table_operation_indices,
        "by_kpoint": kpoint_matching,
    }


def _build_table_kpoint_matching(
    *,
    by_kpoint: dict[str, Any],
    kpoint_frac_by_name: Any,
    table,
    operation_mapping: dict[Any, int],
    tolerance: float,
) -> dict[str, Any]:
    if not isinstance(kpoint_frac_by_name, dict):
        kpoint_frac_by_name = {}
    matching_by_kpoint: dict[str, Any] = {}
    for kpoint, payload in by_kpoint.items():
        k_frac = kpoint_frac_by_name.get(kpoint)
        if k_frac is None:
            matching_by_kpoint[kpoint] = {
                "status": "missing_kpoint_coordinate",
                "table_kpoint_label": None,
            }
            continue
        table_label = table.match_kpoint_label(np.asarray(k_frac, dtype=float), tolerance=tolerance)
        if table_label is None:
            matching_by_kpoint[kpoint] = {
                "status": "table_kpoint_not_matched",
                "input_k_frac": list(np.asarray(k_frac, dtype=float)),
                "table_kpoint_label": None,
            }
            continue
        table_indices = table.operation_indices_for_kpoint(table_label)
        allowed_ids = list(payload.get("allowed_operation_ids", []))
        mapped_indices = sorted(
            operation_mapping[operation_id]
            for operation_id in allowed_ids
            if operation_id in operation_mapping
        )
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


def _collect_computed_characters(
    *,
    symmetry_rows: list[dict[str, Any]],
    kpoint: str,
    operation_to_table: dict[Any, int],
) -> dict[str, Any]:
    """Collect aggregate (trace-level) characters from valley-adapted character rows.

    State-level characters are collected separately via
    ``_collect_state_diagonal_characters`` from the fixed valley-adapted
    D_valley matrix.
    """
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
    state_diagonal_tol: float,
) -> dict[str, Any]:
    """Collect per-state characters from D_valley diagonal entries.

    Uses the fixed valley-adapted basis D_valley(g)[i,i] rather than
    eigenvalue ordering, which is not stable across different operations.
    Only includes entries where ALL rows for a given operation pass
    readiness gates and the off-diagonal norm is below tolerance.
    """
    state_characters: dict[int, dict[int, complex]] = {}
    offdiag_warnings: list[str] = []
    if representation_payload is None:
        return {"state_characters": state_characters, "offdiag_warnings": ["no_representation_payload"]}

    kp_representations = representation_payload.get(kpoint, {})
    if not isinstance(kp_representations, dict):
        return {"state_characters": state_characters, "offdiag_warnings": ["no_kpoint_representations"]}

    # Collect rows by table index and check full readiness (mirrors aggregate gate)
    rows_by_table: dict[int, list[dict[str, Any]]] = {}
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
        rows_by_table.setdefault(table_index, []).append(row)

    # Only use operations where ALL rows pass topology_input_ready
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
        op_payload = kp_representations.get(f"operation_{operation_id}")
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

        # Check off-diagonal norm
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
        "operation_set_label": "G_tau",
        "interpretation": "global valley-preserving operation set",
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
