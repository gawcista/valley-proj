"""Build reviewed-table source payloads for projected-HSP irrep matching.

The projected-subspace adapter is used by ``analyze_hsp`` after certified
k-point classification.  It does not add a standalone output file.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from valleyscope.analysis.standard_setting_kmap import (
    build_standard_setting_transport_view,
)
from valleyscope.irreps.tables import (
    ReviewedSourceIrrep,
    StandardIrrepTable,
    match_table_operations,
)


def build_source_payload_for_projected_hsp_matching(
    *,
    table: StandardIrrepTable,
    projected_hsp_classification: dict[str, Any],
    detected_operations: list[dict[str, Any]],
    valley_preserving_operation_ids: list[int],
    source_hsp_basis: Mapping[str, object] | None = None,
    standard_setting_certificate: Mapping[str, object] | None = None,
    tol: float = 5e-5,
) -> dict[str, Any]:
    """Build a source payload for a representative or validated star arm.

    Star-arm characters require a complete affine conjugation map.  Reciprocal
    lattice translations are retained through their Bloch phase.
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
    parent_phase_by_table: dict[int, complex] = {}
    certified_transport: dict[str, object] = {}
    if standard_setting_certificate is not None:
        mapped = build_certified_standard_operation_transport(
            table=table,
            certificate=standard_setting_certificate,
            detected_operations=detected_operations,
            operation_ids=vp_ids,
            standard_k_frac=projected_hsp_classification.get("standard_k_frac"),
            tol=tol,
        )
        source_operation_map = dict(mapped.get("source_operation_map", {}))
        if mapped.get("status") != "validated":
            return _blocked_after_operation_mapping(
                "standard_setting_transport_failed",
                str(mapped.get("blocker", "unresolved")),
                source_operation_map,
            )
        parent_phase_by_table = dict(mapped["parent_phase_by_table"])
        certified_transport = dict(mapped["provenance"])
        unused_table_indices = sorted(
            {operation.table_index for operation in table.operations}
            - set(source_operation_map.values())
        )
        mapping_provenance = "revalidated_affine_centering_coset_bijection"
    else:
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
        unused_table_indices = operation_match.unused_table_operation_indices
        mapping_provenance = operation_match.provenance
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
    reviewed_by_label = (
        source_hsp_basis.get("_reviewed_source_irreps_by_label", {})
        if isinstance(source_hsp_basis, Mapping) else {}
    )
    if not isinstance(reviewed_by_label, Mapping) or not reviewed_by_label:
        return _blocked_after_operation_mapping(
            "reviewed_source_irrep_model_missing",
            "projected-HSP matching requires the reviewed EBR source-row "
            "model",
            source_operation_map,
        )
    if len(set(source_irrep_labels)) != len(source_irrep_labels):
        return _blocked_after_operation_mapping(
            "duplicate_reviewed_source_irrep_labels",
            f"source HSP {source_hsp!r} contains duplicate irrep labels",
            source_operation_map,
        )
    missing_or_malformed = [
        label for label in source_irrep_labels
        if not isinstance(reviewed_by_label.get(label), ReviewedSourceIrrep)
        or reviewed_by_label[label].kpoint_label != source_hsp
    ]
    if missing_or_malformed:
        return _blocked_after_operation_mapping(
            "incomplete_reviewed_source_irrep_model",
            "missing reviewed source irreps "
            f"{missing_or_malformed} for HSP {source_hsp!r}",
            source_operation_map,
        )
    irreps = tuple(reviewed_by_label[label] for label in source_irrep_labels)
    if (
        len({row.operation_inventory_identity for row in irreps}) != 1
        or len({row.spin_convention for row in irreps}) != 1
    ):
        return _blocked_after_operation_mapping(
            "inconsistent_reviewed_source_irrep_model",
            f"source HSP {source_hsp!r} rows disagree on operation or spin "
            "inventory",
            source_operation_map,
        )
    transport_by_target: dict[int, tuple[int, int, complex]] = {}
    transport_evidence: list[dict[str, object]] = []
    if classification == "representative":
        transport_by_target = {
            index: (index, 1, 1.0 + 0.0j)
            for index in expected_table_ids
        }
    else:
        try:
            target_standard_k = np.asarray(
                projected_hsp_classification.get("standard_k_frac"),
                dtype=float,
            )
        except (TypeError, ValueError):
            target_standard_k = np.asarray([], dtype=float)
        if (
            target_standard_k.shape != (3,)
            or not np.all(np.isfinite(target_standard_k))
        ):
            return _blocked_after_operation_mapping(
                "star_character_transport_target_k_missing",
                "classification has no finite target standard k coordinate",
                source_operation_map,
            )
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
            if (
                not isinstance(lattice_translation, list)
                or len(lattice_translation) != 3
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in lattice_translation
                )
            ):
                return _blocked_after_operation_mapping(
                    "star_character_transport_lattice_translation_malformed",
                    "affine conjugation lattice translation must be list[int,3]",
                    source_operation_map,
                )
            target = row.get("star_arm_operation_id")
            source = row.get("source_representative_operation_id")
            spin_lift_factor = row.get("spin_lift_factor")
            if (
                isinstance(target, int)
                and not isinstance(target, bool)
                and isinstance(source, int)
                and not isinstance(source, bool)
                and spin_lift_factor in (-1, 1)
                and not isinstance(spin_lift_factor, bool)
            ):
                try:
                    target_operation = table.operation_by_index(target)
                    transformed_k = (
                        np.linalg.inv(target_operation.rotation_frac).T
                        @ target_standard_k
                    )
                except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                    return _blocked_after_operation_mapping(
                        "star_character_transport_target_operation_invalid",
                        f"target source-table operation {target!r} is invalid",
                        source_operation_map,
                    )
                bloch_phase = np.exp(
                    -2.0j
                    * np.pi
                    * float(
                        transformed_k
                        @ np.asarray(lattice_translation, dtype=float)
                    )
                )
                transport_by_target[target] = (
                    source, int(spin_lift_factor), complex(bloch_phase)
                )
                transport_evidence.append({
                    **row,
                    "bloch_phase": [
                        float(np.real(bloch_phase)),
                        float(np.imag(bloch_phase)),
                    ],
                    "bloch_phase_convention": (
                        "exp(-2pii*(R_target^-T k_target)_dot_L)"
                    ),
                })
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
            bloch_phase,
        ) in transport_by_target.items():
            if representative_index not in irrep.characters:
                return _blocked_after_operation_mapping(
                    "missing_source_irrep_characters",
                    f"source irrep {irrep.label!r} lacks representative "
                    f"operation {representative_index}",
                    source_operation_map,
                )
            transported[target_index] = (
                spin_lift_factor
                * irrep.characters[representative_index]
                / bloch_phase
                * parent_phase_by_table.get(target_index, 1.0 + 0.0j)
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
        "_group_source_operation_map": mapped.get(
            "group_source_operation_map", {}
        ) if standard_setting_certificate is not None else {},
        "provenance": {
            "table_sg_number": table.number,
            "table_name": table.name,
            "table_spinor": table.spinor,
            "source_hsp_label": source_hsp,
            "source_irrep_labels": list(source_irrep_labels),
            "source_irrep_model": "reviewed_source_rows",
            "source_hsp_classification": classification,
            "valley_preserving_operation_ids": vp_ids,
            "source_table_operation_indices": expected_table_ids,
            "unused_table_operation_indices": unused_table_indices,
            "table_operations_mapped": len(source_operation_map),
            "operation_mapping_provenance": mapping_provenance,
            "character_transport_status": transport_status,
            "character_transport_map": transport_evidence,
            "standard_setting_transport": certified_transport,
        },
        "blocker_reasons": [],
    }


def build_certified_standard_operation_transport(
    *,
    table: StandardIrrepTable,
    certificate: Mapping[str, object],
    detected_operations: list[dict[str, Any]],
    operation_ids: list[int],
    standard_k_frac: object,
    tol: float,
) -> dict[str, object]:
    view = build_standard_setting_transport_view(
        table=table,
        standard_setting_certificate=certificate,
        detected_operations=detected_operations,
        tolerance=min(tol, 1e-6),
    )
    if view.get("status") != "validated":
        return {"status": "blocked", "blocker": view.get("blocker", "")}
    try:
        kpoint = np.asarray(standard_k_frac, dtype=float)
    except (TypeError, ValueError):
        kpoint = np.asarray([])
    if kpoint.shape != (3,) or not np.all(np.isfinite(kpoint)):
        return {"status": "blocked", "blocker": "standard_k_frac_missing"}
    centering = view["centering_vectors"]
    rows = view["operation_rows"]
    if (
        not isinstance(centering, list)
        or not centering
        or not np.allclose(centering[0], np.zeros(3), atol=tol)
        or not isinstance(rows, list)
    ):
        return {"status": "blocked", "blocker": "centering_rows_malformed"}

    grouped = {
        parent_id: sorted(
            (
                row for row in rows
                if row.get("parent_operation_id") == parent_id
            ),
            key=lambda row: row.get("centering_coset_index", -1),
        )
        for parent_id in view["parent_operation_ids"]
    }
    group_map: dict[int, int] = {}
    for parent_id, selected in grouped.items():
        indices = {
            row.get("source_table_operation_index") for row in selected
        }
        if (
            len(selected) != len(centering)
            or [row.get("centering_coset_index") for row in selected]
            != list(range(len(centering)))
            or len(indices) != 1
        ):
            return {"status": "blocked", "blocker": "incomplete_group_cosets"}
        group_map[int(parent_id)] = int(indices.pop())

    source_map: dict[int, int] = {}
    phases: dict[int, complex] = {}
    phase_rows: list[dict[str, object]] = []
    max_residual = 0.0
    for operation_id in operation_ids:
        selected = grouped.get(operation_id, [])
        if not selected:
            return {
                "status": "blocked",
                "blocker": f"incomplete_centering_cosets_for_operation_{operation_id}",
                "source_operation_map": source_map,
            }
        table_index = group_map[operation_id]
        try:
            operation = table.operation_by_index(table_index)
            transformed_k = np.linalg.inv(operation.rotation_frac).T @ kpoint
            row_phases = [
                np.exp(
                    -2.0j * np.pi * float(
                        transformed_k
                        @ np.asarray(
                            row["parent_to_source_translation_frac"],
                            dtype=float,
                        )
                    )
                )
                for row in selected
            ]
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
            return {"status": "blocked", "blocker": "bloch_phase_malformed"}
        base_phase = complex(row_phases[0])
        for row_phase, vector in zip(row_phases, centering):
            expected = base_phase * np.exp(
                -2.0j * np.pi
                * float(transformed_k @ np.asarray(vector, dtype=float))
            )
            max_residual = max(max_residual, float(abs(row_phase - expected)))
        phase_rows.extend({
            **row,
            "bloch_phase": [
                float(row_phase.real), float(row_phase.imag),
            ],
            "bloch_phase_convention": (
                "exp(-2pii*(R^-T k)_dot_delta_t)"
            ),
        } for row, row_phase in zip(selected, row_phases))
        source_map[operation_id] = int(table_index)
        phases[int(table_index)] = base_phase
    if max_residual > tol:
        return {"status": "blocked", "blocker": "centering_bloch_phase_failed"}
    return {
        "status": "validated",
        "source_operation_map": source_map,
        "group_source_operation_map": group_map,
        "parent_phase_by_table": phases,
        "operation_rows": phase_rows,
        "provenance": {
            "centering_coset_count": len(centering),
            "operation_pair_count": len(rows),
            "max_bloch_phase_relation_residual": max_residual,
            "bloch_phase_convention": "exp(-2pii*(R^-T k)_dot_delta_t)",
        },
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
