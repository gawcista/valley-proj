"""Build generic-irrep-matcher source payloads from reviewed tables.

The projected-subspace adapter is used by ``analyze_hsp`` after certified
k-point classification.  It does not add a standalone output file.
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


def build_source_payload_for_projected_hsp_matching(
    *,
    table: StandardIrrepTable,
    projected_hsp_classification: dict[str, Any],
    detected_operations: list[dict[str, Any]],
    valley_preserving_operation_ids: list[int],
    tol: float = 5e-5,
) -> dict[str, Any]:
    """Build a source payload for a representative or validated star arm.

    Star-arm characters are transported only when the classification carries
    a complete affine conjugation map with zero residual lattice translation.
    A nontrivial Bloch lattice phase therefore fails closed instead of reusing
    representative characters by assumption.
    """
    classification = projected_hsp_classification.get("classification")
    source_hsp = projected_hsp_classification.get("source_hsp_label")
    transport_status = projected_hsp_classification.get(
        "representation_transport_status"
    )
    if classification not in ("representative", "star_equivalent"):
        return _blocked(
            "not_source_hsp_member",
            f"classification={classification!r} is not a source-HSP member",
        )
    if not isinstance(source_hsp, str) or not source_hsp:
        return _blocked("missing_source_hsp_label", "classification has no source HSP")
    if transport_status != "validated":
        return _blocked(
            "star_character_transport_unresolved",
            str(projected_hsp_classification.get("matching_blocker", ""))
            or f"representation_transport_status={transport_status}",
        )

    vp_ids = _validate_operation_ids(valley_preserving_operation_ids)
    if vp_ids is None or not vp_ids:
        return _blocked(
            "invalid_valley_preserving_operation_ids",
            "valley-preserving operation IDs must be non-empty distinct integers",
        )
    detected_by_id = {
        row.get("operation_id"): row
        for row in detected_operations
        if isinstance(row, dict)
        and isinstance(row.get("operation_id"), int)
        and not isinstance(row.get("operation_id"), bool)
    }
    missing_detected = [op_id for op_id in vp_ids if op_id not in detected_by_id]
    if missing_detected:
        return _blocked(
            "missing_detected_operations",
            f"operation IDs absent from detected operations: {missing_detected}",
        )

    expected_table_ids = projected_hsp_classification.get(
        "standard_little_group_operation_ids", []
    )
    if (
        not isinstance(expected_table_ids, list)
        or any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in expected_table_ids
        )
        or not expected_table_ids
    ):
        return _blocked(
            "missing_standard_little_group",
            "classification has no validated standard little-group operation IDs",
        )
    expected_set = set(expected_table_ids)
    restricted_table = StandardIrrepTable(
        number=table.number,
        name=table.name,
        spinor=table.spinor,
        operations=tuple(
            operation for operation in table.operations
            if operation.table_index in expected_set
        ),
        irreps=table.irreps,
    )
    operation_match = match_table_operations(
        detected_operations=[detected_by_id[op_id] for op_id in vp_ids],
        table=restricted_table,
        tolerance=tol,
        source_hsp_label=None,
    )
    source_operation_map = {
        op_id: operation_match.mapping_by_operation_id[op_id]
        for op_id in vp_ids
        if op_id in operation_match.mapping_by_operation_id
    }
    if operation_match.unmatched_operation_ids:
        return _blocked_after_operation_mapping(
            "table_operation_matching_failed",
            "valley-preserving operations could not be mapped to the "
            f"classified standard little group: "
            f"{operation_match.unmatched_operation_ids}",
            source_operation_map,
        )
    if set(source_operation_map.values()) != expected_set:
        return _blocked_after_operation_mapping(
            "little_group_operation_mismatch",
            f"expected standard {sorted(expected_set)}, mapped "
            f"{sorted(source_operation_map.values())}",
            source_operation_map,
        )

    source_irrep_labels = projected_hsp_classification.get(
        "source_irrep_labels", []
    )
    if (
        not isinstance(source_irrep_labels, list)
        or not source_irrep_labels
        or any(
            not isinstance(label, str) or not label
            for label in source_irrep_labels
        )
    ):
        return _blocked_after_operation_mapping(
            "missing_ebr_source_irrep_labels",
            "classification has no reviewed EBR source irrep labels",
            source_operation_map,
        )
    source_label_set = set(source_irrep_labels)
    irreps = tuple(
        irrep for irrep in table.irreps_by_kpoint(source_hsp)
        if irrep.label in source_label_set
    )
    if not irreps:
        return _blocked_after_operation_mapping(
            "no_source_irreps_for_hsp",
            f"source HSP {source_hsp!r} has no irreps in the table",
            source_operation_map,
        )
    transport_by_target: dict[int, tuple[int, int]] = {}
    if classification == "representative":
        transport_by_target = {
            index: (index, 1) for index in expected_table_ids
        }
    else:
        raw_transport = projected_hsp_classification.get(
            "character_transport_map", []
        )
        if not isinstance(raw_transport, list):
            return _blocked(
                "star_character_transport_unresolved",
                "character_transport_map is not a list",
            )
        for row in raw_transport:
            if not isinstance(row, dict):
                continue
            lattice_translation = row.get("affine_lattice_translation", [])
            if not isinstance(lattice_translation, list) or any(
                int(value) != 0 for value in lattice_translation
            ):
                return _blocked_after_operation_mapping(
                    "star_character_transport_requires_bloch_phase",
                    "affine conjugation differs by a nonzero lattice translation",
                    source_operation_map,
                )
            target = row.get("star_arm_operation_id")
            source = row.get("source_representative_operation_id")
            spin_lift_factor = row.get("spin_lift_factor")
            if (
                isinstance(target, int)
                and isinstance(source, int)
                and spin_lift_factor in (-1, 1)
            ):
                transport_by_target[target] = (
                    source, int(spin_lift_factor)
                )
        if set(transport_by_target) != expected_set:
            return _blocked_after_operation_mapping(
                "incomplete_star_character_transport",
                f"expected targets {sorted(expected_set)}, transported "
                f"{sorted(transport_by_target)}",
                source_operation_map,
            )

    source_irrep_characters: dict[str, dict[int, complex]] = {}
    for irrep in irreps:
        transported: dict[int, complex] = {}
        for target_index, (
            representative_index,
            spin_lift_factor,
        ) in transport_by_target.items():
            if representative_index not in irrep.characters:
                return _blocked_after_operation_mapping(
                    "missing_source_irrep_characters",
                    f"source irrep {irrep.label!r} lacks representative "
                    f"operation {representative_index}",
                    source_operation_map,
                )
            transported[target_index] = (
                spin_lift_factor * irrep.characters[representative_index]
            )
        source_irrep_characters[irrep.label] = transported

    ambiguous = _ambiguous_restricted_irreps(
        irreps=[
            _CharacterView(label, source_irrep_characters[label])
            for label in source_irrep_characters
        ],
        table_indices=expected_table_ids,
    )
    if ambiguous:
        return _blocked_after_operation_mapping(
            "ambiguous_restricted_source_irreps",
            "source irreps are not distinguishable on the classified "
            f"operation set: {ambiguous}",
            source_operation_map,
        )

    return {
        "status": "ok",
        "operation_mapping_evaluated": True,
        "source_irrep_characters": source_irrep_characters,
        "source_operation_map": source_operation_map,
        "provenance": {
            "table_sg_number": table.number,
            "table_name": table.name,
            "table_spinor": table.spinor,
            "source_hsp_label": source_hsp,
            "source_irrep_labels": list(source_irrep_labels),
            "source_hsp_classification": classification,
            "valley_preserving_operation_ids": vp_ids,
            "source_table_operation_indices": expected_table_ids,
            "unused_table_operation_indices": (
                operation_match.unused_table_operation_indices
            ),
            "table_operations_mapped": len(source_operation_map),
            "operation_mapping_provenance": operation_match.provenance,
            "character_transport_status": transport_status,
            "character_transport_map": projected_hsp_classification.get(
                "character_transport_map", []
            ),
        },
        "blocker_reasons": [],
    }


class _CharacterView:
    """Minimal irrep view for the restricted-ambiguity helper."""

    def __init__(self, label: str, characters: dict[int, complex]) -> None:
        self.label = label
        self.characters = characters


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


def _blocked_after_operation_mapping(
    reason_key: str,
    reason: str,
    source_operation_map: dict[int, int],
) -> dict[str, Any]:
    result = _blocked(reason_key, reason)
    result["operation_mapping_evaluated"] = True
    result["source_operation_map"] = dict(source_operation_map)
    return result
