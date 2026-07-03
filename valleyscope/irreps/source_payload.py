"""Adapter that builds generic-irrep-matcher source payloads from a
``StandardIrrepTable`` and explicit ValleyScope operation metadata.

This is an offline API-only module.  It does not change ``analyze_hsp``
default behavior and does not add new output files.
"""

from __future__ import annotations

from typing import Any

from valleyscope.irreps.tables import (
    StandardIrrepTable,
    match_table_operations,
)


def build_source_payload_for_generic_matching(
    *,
    table: StandardIrrepTable,
    source_hsp_label: str,
    detected_operations: list[dict[str, Any]],
    valley_preserving_operation_ids: list[int],
    tol: float = 5e-5,
) -> dict[str, Any]:
    """Build explicit generic-matcher payloads from a standard irrep table.

    Parameters
    ----------
    table : StandardIrrepTable
        A reviewed Bilbao/irreptable irrep table loaded through
        ``load_standard_irrep_table(...)`` or an equivalent table object.
    source_hsp_label : str
        The Bilbao source HSP label to use (e.g. ``"K"``, ``"GM"``).
        Not inferred from ValleyScope labels.
    detected_operations : list[dict[str, object]]
        Detected ValleyScope operations.  Each entry must have
        ``operation_id`` (int), ``rotation_frac`` (3x3 array), and
        ``translation_frac`` (3-vector).
    valley_preserving_operation_ids : list[int]
        ValleyScope operation IDs for ``G_k^(a)``.  Every ID must be found
        in ``detected_operations``.
    tol : float
        Tolerance for spatial rotation/translation matching.

    Returns
    -------
    dict with ``source_irrep_characters``, ``source_operation_map``,
    ``provenance``, ``status``, and if blocked, ``blocker_reasons``.
    """
    # --- Validate VP operation IDs ---
    vp_ids = _validate_operation_ids(valley_preserving_operation_ids)
    if vp_ids is None:
        return _blocked(
            "invalid_valley_preserving_operation_ids",
            "valley-preserving operation IDs must be distinct integers",
        )
    if not vp_ids:
        return _blocked("empty_valley_preserving_operation_ids",
                        "no valley-preserving operation IDs provided")

    if not isinstance(source_hsp_label, str) or not source_hsp_label:
        return _blocked(
            "missing_source_hsp_label",
            "source_hsp_label must be an explicit non-empty string",
        )

    # --- Check source HSP has irreps before operation matching ---
    irreps = table.irreps_by_kpoint(source_hsp_label)
    if not irreps:
        return _blocked(
            "no_source_irreps_for_hsp",
            f"source HSP {source_hsp_label!r} has no irreps in the table",
        )

    # Build lookup from ValleyScope op ID to detected operation metadata.
    vs_op_lookup: dict[int, dict[str, Any]] = {}
    for det in detected_operations:
        if not isinstance(det, dict):
            continue
        op_id = det.get("operation_id")
        if not isinstance(op_id, int) or isinstance(op_id, bool):
            continue
        vs_op_lookup[op_id] = det

    missing_vs = [op for op in vp_ids if op not in vs_op_lookup]
    if missing_vs:
        return _blocked(
            "missing_detected_operations",
            f"valley-preserving operation IDs not in detected_operations: "
            f"{missing_vs}",
        )

    # --- Match ValleyScope operations to table operations ---
    vp_detected_operations = [vs_op_lookup[op_id] for op_id in vp_ids]
    op_match = match_table_operations(
        table=table,
        detected_operations=vp_detected_operations,
        tolerance=tol,
        source_hsp_label=source_hsp_label,
    )
    if op_match.unmatched_operation_ids:
        return _blocked(
            "table_operation_matching_failed",
            "valley-preserving operations could not be mapped to table "
            f"operation indices: {op_match.unmatched_operation_ids}",
        )

    # Build source_operation_map: VS op ID -> table operation index.
    source_operation_map: dict[int, int] = {}
    for op_id in vp_ids:
        table_index = op_match.mapping_by_operation_id.get(op_id)
        if table_index is None:
            return _blocked(
                "unmapped_valley_preserving_operation",
                f"VS operation {op_id} could not be mapped to a table "
                f"operation index",
            )
        source_operation_map[op_id] = table_index

    mapped_table_indices = list(source_operation_map.values())
    missing_chars = {
        irrep.label: [
            table_index for table_index in mapped_table_indices
            if table_index not in irrep.characters
        ]
        for irrep in irreps
    }
    missing_chars = {
        label: missing for label, missing in missing_chars.items() if missing
    }
    if missing_chars:
        return _blocked(
            "missing_source_irrep_characters",
            f"source irreps are missing mapped operation characters: "
            f"{missing_chars}",
        )

    ambiguous = _ambiguous_restricted_irreps(
        irreps=irreps,
        table_indices=mapped_table_indices,
    )
    if ambiguous:
        return _blocked(
            "ambiguous_restricted_source_irreps",
            "source irreps are not distinguishable on the restricted "
            f"operation set: {ambiguous}",
        )

    # --- Build source_irrep_characters ---
    source_irrep_characters: dict[str, dict[int, complex]] = {}
    for irrep in irreps:
        source_irrep_characters[irrep.label] = dict(irrep.characters)

    return {
        "status": "ok",
        "source_irrep_characters": source_irrep_characters,
        "source_operation_map": source_operation_map,
        "provenance": {
            "table_sg_number": table.number,
            "table_name": table.name,
            "table_spinor": table.spinor,
            "source_hsp_label": source_hsp_label,
            "valley_preserving_operation_ids": vp_ids,
            "source_table_operation_indices": mapped_table_indices,
            "unused_table_operation_indices": op_match.unused_table_operation_indices,
            "table_operations_mapped": len(source_operation_map),
            "operation_mapping_provenance": (
                op_match.provenance if hasattr(op_match, "provenance")
                else "exact_spatial"
            ),
        },
        "blocker_reasons": [],
    }


def _validate_operation_ids(values: list[int]) -> list[int] | None:
    out: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        if value in out:
            return None
        out.append(value)
    return out


def _ambiguous_restricted_irreps(
    *,
    irreps: object,
    table_indices: list[int],
) -> dict[tuple[tuple[float, float], ...], list[str]]:
    groups: dict[tuple[tuple[float, float], ...], list[str]] = {}
    for irrep in irreps:
        key = tuple(
            (
                round(float(irrep.characters[idx].real), 12),
                round(float(irrep.characters[idx].imag), 12),
            )
            for idx in table_indices
        )
        groups.setdefault(key, []).append(irrep.label)
    return {key: labels for key, labels in groups.items() if len(labels) > 1}


def _blocked(reason_key: str, reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "source_irrep_characters": {},
        "source_operation_map": {},
        "provenance": {},
        "blocker_reasons": [f"{reason_key}: {reason}"],
    }
