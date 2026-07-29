"""Projected-subspace source-HSP basis and sampled-k coverage.

The module keeps three statements separate:

* a sampled row can carry a valid local ``G_k^(a)`` representation;
* its standard-setting k point can belong to a reviewed source-HSP star;
* a complete trusted source-HSP basis can be promoted to reduced EBR input.

No material, point-group-order, or HSP-label special cases live here.  Source
HSP order comes from the reviewed irreptables EBR basis, while coordinates and
little groups come from ``StandardIrrepTable`` and a validated affine setting
certificate.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Mapping, Sequence

import numpy as np

from valleyscope.irreps.tables import (
    ReviewedSourceIrrep,
    StandardIrrepTable,
    StandardTableOperation,
    resolve_ebr_source_irrep_label_evidence,
)


_TOL = 5e-6
_MAX_RATIONAL_DENOMINATOR = 96
_MAX_RESIDUE_CANDIDATES = 1_000_000


def derive_projected_subspace_source_hsp_basis(
    *,
    table: StandardIrrepTable,
    ebr_source_basis_labels: Sequence[str],
    standard_setting_certificate: Mapping[str, object],
    use_2d_momentum_only: bool,
) -> dict[str, object]:
    """Derive the reviewed source-HSP basis lying in the parent 2D plane."""
    certificate_blocker = _certificate_blocker(standard_setting_certificate)
    space_group_identity, identity_blocker = _projected_space_group_identity(
        table, standard_setting_certificate
    )
    if identity_blocker:
        return _blocked_basis(identity_blocker)
    transform = _certificate_transform(standard_setting_certificate)
    if certificate_blocker or transform is None:
        return _blocked_basis(certificate_blocker or "missing reciprocal transform")
    if not use_2d_momentum_only:
        return _blocked_basis(
            "projected_subspace_plane_undefined: "
            "projection.use_2d_momentum_only must define the parent 2D plane"
        )

    reciprocal_transform = np.linalg.inv(transform).T
    standard_plane_basis = reciprocal_transform[:, :2]
    centering_vectors = _certificate_centering_vectors(
        standard_setting_certificate
    )
    if centering_vectors is None:
        return _blocked_basis(
            "projected_subspace_plane_undefined: normalized centering vectors "
            "are missing from the standard-setting certificate"
        )

    source_label_evidence = resolve_ebr_source_irrep_label_evidence(
        table=table,
        source_basis_labels=[str(x) for x in ebr_source_basis_labels],
    )
    if source_label_evidence.get("status") != "validated":
        return _blocked_basis(str(source_label_evidence.get("blocker", "")))
    reviewed_rows = source_label_evidence.get("reviewed_rows", [])
    if (
        not isinstance(reviewed_rows, list)
        or len(reviewed_rows) != len(ebr_source_basis_labels)
        or not all(isinstance(row, ReviewedSourceIrrep) for row in reviewed_rows)
    ):
        return _blocked_basis(
            "reviewed_source_irrep_model_incomplete_or_malformed"
        )
    records_by_hsp: dict[str, dict[str, object]] = {}
    ordered_hsps: list[str] = []
    out_of_plane_source_labels: list[str] = []
    reviewed_source_rows_provenance: list[dict[str, object]] = []
    plane_result_by_k: dict[tuple[float, float, float], object] = {}

    for irrep in reviewed_rows:
        label = irrep.label
        k_key = tuple(round(float(value), 12) for value in irrep.k_frac)
        if k_key not in plane_result_by_k:
            try:
                plane_result_by_k[k_key] = _inverse_parent_representative(
                    standard_k=np.asarray(irrep.k_frac, dtype=float),
                    transform=transform,
                    centering_vectors=centering_vectors,
                )
            except _PlaneMembershipUnresolved as exc:
                return _blocked_basis(
                    "source_hsp_plane_membership_unresolved: "
                    f"irrep={label}: {exc}"
                )
        plane_result = plane_result_by_k[k_key]
        reviewed_source_rows_provenance.append({
            "label": label,
            "source_hsp_label": irrep.kpoint_label,
            "standard_k_frac": _vector(irrep.k_frac),
            "dimension": irrep.dimension,
            "operation_indices": list(irrep.operation_indices),
            "operation_inventory_identity": (
                irrep.operation_inventory_identity
            ),
            "spin_convention": irrep.spin_convention,
            "source_table": irrep.source_table,
            "source_table_status": irrep.source_table_status,
            "source_provenance": irrep.source_provenance,
            "in_parent_plane": plane_result is not None,
        })
        if plane_result is None:
            out_of_plane_source_labels.append(label)
            continue
        assert isinstance(plane_result, dict)

        hsp = irrep.kpoint_label
        if hsp not in records_by_hsp:
            ordered_hsps.append(hsp)
            records_by_hsp[hsp] = {
                "source_hsp_label": hsp,
                "source_basis_order": len(ordered_hsps) - 1,
                "standard_representative_k_frac": _vector(irrep.k_frac),
                "standard_reciprocal_shift": plane_result[
                    "standard_reciprocal_shift"
                ],
                "inverse_parent_k_frac": plane_result[
                    "inverse_parent_k_frac"
                ],
                "inverse_reciprocal_shift": plane_result[
                    "inverse_reciprocal_shift"
                ],
                "standard_little_group_operation_ids": (
                    list(irrep.operation_indices)
                ),
                "source_irrep_labels": [],
            }
        operation_ids = records_by_hsp[hsp][
            "standard_little_group_operation_ids"
        ]
        if isinstance(operation_ids, list):
            records_by_hsp[hsp]["standard_little_group_operation_ids"] = (
                sorted(set(operation_ids) | set(irrep.operation_indices))
            )
        labels = records_by_hsp[hsp]["source_irrep_labels"]
        if isinstance(labels, list) and label not in labels:
            labels.append(label)

    source_hsps = [records_by_hsp[hsp] for hsp in ordered_hsps]
    if not source_hsps:
        return _blocked_basis(
            "no reviewed irreptables EBR source HSP lies in the certified "
            "parent 2D reciprocal plane"
        )

    return {
        "status": "validated",
        "validation_status": "validated",
        "required_source_hsp_labels": ordered_hsps,
        "source_hsps": source_hsps,
        "standard_plane_basis": [
            _vector(standard_plane_basis[:, 0]),
            _vector(standard_plane_basis[:, 1]),
        ],
        "parent_plane_definition": "parent reciprocal fractional k3 = 0",
        "projected_subspace_space_group": space_group_identity,
        "_reviewed_source_irreps_by_label": {
            row.label: row for row in reviewed_rows
        },
        "provenance": {
            "source": (
                "irreptables StandardIrrepTable + irreptables EBR source basis"
            ),
            "basis_order_source": "irreptables EBR source basis order",
            "source_table_sg_number": table.number,
            "source_table_name": table.name,
            "source_table_spinor": table.spinor,
            "source_ebr_basis_labels": [str(x) for x in ebr_source_basis_labels],
            "reviewed_source_rows": reviewed_source_rows_provenance,
            "auxiliary_source_table": source_label_evidence.get(
                "auxiliary_source_table"
            ),
            "out_of_plane_source_labels": out_of_plane_source_labels,
            "hall_number": standard_setting_certificate.get("hall_number"),
            "hall_symbol": standard_setting_certificate.get("hall_symbol"),
            "reciprocal_k_transform_rule": "k_std = T^(-T) @ k_parent",
        },
        "blocker": "",
    }


def classify_projected_subspace_kpoint(
    *,
    parent_k_frac: Sequence[float] | None,
    table: StandardIrrepTable,
    source_hsp_basis: Mapping[str, object],
    standard_setting_certificate: Mapping[str, object],
    mapped_standard_little_group_operation_ids: Sequence[int] | None = None,
    override_source_hsp_label: str | None = None,
    kpoint: str | None = None,
    valley: str | None = None,
) -> dict[str, object]:
    """Classify one sampled projected-subspace k point against source HSPs."""
    base: dict[str, object] = {}
    if kpoint is not None:
        base["kpoint"] = kpoint
    if valley is not None:
        base["valley"] = valley

    space_group_identity, identity_blocker = _projected_space_group_identity(
        table, standard_setting_certificate
    )
    base["projected_subspace_space_group"] = space_group_identity

    try:
        parent_k = np.asarray(parent_k_frac, dtype=float)
    except (TypeError, ValueError):
        parent_k = np.asarray([], dtype=float)
    if parent_k.shape != (3,) or not np.all(np.isfinite(parent_k)):
        return {
            **base,
            "parent_k_frac": None,
            "standard_k_frac": None,
            "classification": "unresolved",
            "geometric_classification": "unresolved",
            "source_hsp_label": None,
            "source_hsp_membership": False,
            "validation_status": "blocked",
            "local_representation_status": "unresolved",
            "representation_transport_status": "not_evaluated",
            "blocker": "missing_parent_k_coordinate",
        }

    if source_hsp_basis.get("status") != "validated":
        return {
            **base,
            "parent_k_frac": _vector(parent_k),
            "standard_k_frac": None,
            "classification": "unresolved",
            "geometric_classification": "unresolved",
            "source_hsp_label": None,
            "source_hsp_membership": False,
            "validation_status": "blocked",
            "local_representation_status": "unresolved",
            "representation_transport_status": "not_evaluated",
            "blocker": str(
                source_hsp_basis.get("blocker", "source basis blocked")
            ),
        }

    transform = _certificate_transform(standard_setting_certificate)
    certificate_blocker = _certificate_blocker(standard_setting_certificate)
    basis_identity = source_hsp_basis.get(
        "projected_subspace_space_group"
    )
    if basis_identity != space_group_identity:
        identity_blocker = (
            "source_basis_certificate_identity_mismatch"
        )
    if transform is None or certificate_blocker or identity_blocker:
        return {
            **base,
            "parent_k_frac": _vector(parent_k_frac),
            "standard_k_frac": None,
            "classification": "unresolved",
            "geometric_classification": "unresolved",
            "source_hsp_label": None,
            "source_hsp_membership": False,
            "validation_status": "blocked",
            "local_representation_status": "unresolved",
            "representation_transport_status": "not_evaluated",
            "blocker": (
                identity_blocker
                or certificate_blocker
                or "missing reciprocal transform"
            ),
        }
    standard_k = np.linalg.inv(transform).T @ parent_k
    centering_vectors = _certificate_centering_vectors(
        standard_setting_certificate
    )
    if centering_vectors is None:
        return {
            **base,
            "parent_k_frac": _vector(parent_k),
            "standard_k_frac": _vector(standard_k),
            "classification": "unresolved",
            "geometric_classification": "unresolved",
            "source_hsp_label": None,
            "source_hsp_membership": False,
            "validation_status": "blocked",
            "local_representation_status": "unresolved",
            "representation_transport_status": "not_evaluated",
            "blocker": "missing normalized centering vectors",
        }

    source_rows = source_hsp_basis.get("source_hsps", [])
    matches: list[dict[str, object]] = []
    if isinstance(source_rows, list):
        for source_row in source_rows:
            if not isinstance(source_row, Mapping):
                continue
            representative = np.asarray(
                source_row.get("standard_representative_k_frac"), dtype=float
            )
            direct_shift = _equivalent_reciprocal_shift(
                standard_k, representative, centering_vectors
            )
            if direct_shift is not None:
                matches.append({
                    "classification": "representative",
                    "source_row": source_row,
                    "reciprocal_shift": direct_shift,
                    "witness": None,
                })
                continue
            for operation in sorted(table.operations, key=lambda op: op.table_index):
                arm = np.linalg.inv(operation.rotation_frac).T @ representative
                shift = _equivalent_reciprocal_shift(
                    standard_k, arm, centering_vectors
                )
                if shift is None:
                    continue
                matches.append({
                    "classification": "star_equivalent",
                    "source_row": source_row,
                    "reciprocal_shift": shift,
                    "witness": operation,
                })

    matches = _deduplicate_geometric_matches(matches)
    distinct_hsps = {
        str(match["source_row"].get("source_hsp_label"))
        for match in matches
        if isinstance(match.get("source_row"), Mapping)
    }
    if len(distinct_hsps) > 1:
        return {
            **base,
            "parent_k_frac": _vector(parent_k),
            "standard_k_frac": _vector(standard_k),
            "classification": "unresolved",
            "geometric_classification": "ambiguous",
            "source_hsp_label": None,
            "source_hsp_membership": False,
            "validation_status": "blocked",
            "local_representation_status": "valid",
            "representation_transport_status": "not_evaluated",
            "blocker": (
                "ambiguous_source_hsp_star_membership: "
                + ", ".join(sorted(distinct_hsps))
            ),
        }

    if not matches:
        if override_source_hsp_label is not None:
            return {
                **base,
                "parent_k_frac": _vector(parent_k),
                "standard_k_frac": _vector(standard_k),
                "classification": "unresolved",
                "geometric_classification": "generic",
                "source_hsp_label": None,
                "source_hsp_membership": False,
                "validation_status": "blocked",
                "local_representation_status": "valid",
                "representation_transport_status": "not_applicable",
                "blocker": (
                    "source_hsp_override_cannot_promote_generic_k: "
                    f"{override_source_hsp_label!r}"
                ),
            }
        return {
            **base,
            "parent_k_frac": _vector(parent_k),
            "standard_k_frac": _vector(standard_k),
            "classification": "generic",
            "geometric_classification": "generic",
            "source_hsp_label": None,
            "source_hsp_membership": False,
            "reciprocal_shift": None,
            "standard_operation_witness": None,
            "standard_little_group_operation_ids": [],
            "mapped_parent_little_group_operation_ids": list(
                mapped_standard_little_group_operation_ids or []
            ),
            "validation_status": "validated",
            "local_representation_status": "valid",
            "representation_transport_status": "not_applicable",
            "blocker": "",
        }

    chosen = matches[0]
    source_row = chosen["source_row"]
    assert isinstance(source_row, Mapping)
    source_hsp = str(source_row.get("source_hsp_label"))
    geometric_classification = str(chosen["classification"])
    if (
        override_source_hsp_label is not None
        and override_source_hsp_label != source_hsp
    ):
        return {
            **base,
            "parent_k_frac": _vector(parent_k),
            "standard_k_frac": _vector(standard_k),
            "classification": "unresolved",
            "geometric_classification": geometric_classification,
            "source_hsp_label": source_hsp,
            "source_hsp_membership": True,
            "validation_status": "blocked",
            "local_representation_status": "valid",
            "representation_transport_status": "not_evaluated",
            "blocker": (
                f"source_hsp_override_mismatch: override "
                f"{override_source_hsp_label!r}, derived {source_hsp!r}"
            ),
        }

    expected_little_group = list(
        source_row.get("standard_little_group_operation_ids", [])
    )
    transport_status = "validated"
    transport_map: list[dict[str, object]] = []
    transport_blocker = ""
    witness = chosen.get("witness")
    if isinstance(witness, StandardTableOperation):
        conjugation = _conjugated_little_group(
            table=table,
            source_operation_ids=expected_little_group,
            witness=witness,
            target_standard_k=standard_k,
        )
        if conjugation["status"] != "validated":
            transport_status = "blocked"
            transport_blocker = str(conjugation.get("blocker", ""))
            expected_little_group = list(
                conjugation.get("conjugated_operation_ids", [])
            )
        else:
            expected_little_group = list(
                conjugation["conjugated_operation_ids"]
            )
            transport_map = list(conjugation["character_transport_map"])

    mapped_little_group = sorted(
        int(value) for value in (mapped_standard_little_group_operation_ids or [])
    )
    if mapped_standard_little_group_operation_ids is not None:
        if sorted(expected_little_group) != mapped_little_group:
            return {
                **base,
                "parent_k_frac": _vector(parent_k),
                "standard_k_frac": _vector(standard_k),
                "classification": "unresolved",
                "geometric_classification": geometric_classification,
                "source_hsp_label": source_hsp,
                "source_hsp_membership": True,
                "reciprocal_shift": chosen["reciprocal_shift"],
                "standard_operation_witness": _serialize_witness(witness),
                "standard_little_group_operation_ids": expected_little_group,
                "mapped_parent_little_group_operation_ids": mapped_little_group,
                "validation_status": "blocked",
                "local_representation_status": "valid",
                "representation_transport_status": transport_status,
                "blocker": (
                    "little_group_operation_mismatch: expected standard "
                    f"{sorted(expected_little_group)}, mapped local "
                    f"{mapped_little_group}"
                ),
            }

    return {
        **base,
        "parent_k_frac": _vector(parent_k),
        "standard_k_frac": _vector(standard_k),
        "classification": geometric_classification,
        "geometric_classification": geometric_classification,
        "source_hsp_label": source_hsp,
        "source_hsp_representative_k_frac": source_row.get(
            "standard_representative_k_frac"
        ),
        "source_irrep_labels": list(
            source_row.get("source_irrep_labels", [])
        ),
        "source_hsp_membership": True,
        "reciprocal_shift": chosen["reciprocal_shift"],
        "standard_operation_witness": _serialize_witness(witness),
        "standard_little_group_operation_ids": expected_little_group,
        "mapped_parent_little_group_operation_ids": mapped_little_group,
        "validation_status": "validated",
        "local_representation_status": "valid",
        "representation_transport_status": transport_status,
        "character_transport_map": transport_map,
        "matching_blocker": transport_blocker,
        "blocker": "",
    }


def build_projected_hsp_coverage_report(
    *,
    source_hsp_basis_by_valley: Mapping[str, Mapping[str, object]],
    classifications: Sequence[Mapping[str, object]],
    matching_by_kpoint: Mapping[str, Mapping[str, Mapping[str, object]]],
    workflow_decisions_by_kpoint: (
        Mapping[str, Mapping[str, Mapping[str, object]]]
    ),
) -> dict[str, object]:
    """Aggregate source-HSP coverage independently for every valley."""
    by_valley: dict[str, dict[str, object]] = {}
    rows_by_valley: dict[str, list[Mapping[str, object]]] = {}
    for row in classifications:
        valley = row.get("valley")
        if isinstance(valley, str) and valley:
            rows_by_valley.setdefault(valley, []).append(row)

    for valley, basis in source_hsp_basis_by_valley.items():
        required = list(basis.get("required_source_hsp_labels", []))
        source_rows = basis.get("source_hsps", [])
        source_by_label = {
            str(row.get("source_hsp_label")): row
            for row in source_rows
            if isinstance(row, Mapping)
        } if isinstance(source_rows, list) else {}
        valley_rows = rows_by_valley.get(valley, [])

        covered_rows_by_label: dict[str, list[Mapping[str, object]]] = {}
        generic_rows: list[dict[str, object]] = []
        unresolved_rows: list[dict[str, object]] = []
        for row in valley_rows:
            classification = str(row.get("classification", "unresolved"))
            if classification in ("representative", "star_equivalent") and (
                row.get("validation_status") == "validated"
            ):
                label = row.get("source_hsp_label")
                if isinstance(label, str) and label in required:
                    covered_rows_by_label.setdefault(label, []).append(row)
            elif classification == "generic":
                generic_rows.append(_compact_classification(row))
            else:
                unresolved_rows.append(_compact_classification(row))

        covered = [label for label in required if label in covered_rows_by_label]
        missing = [label for label in required if label not in covered_rows_by_label]
        duplicates = [
            {
                "source_hsp_label": label,
                "sampled_rows": [
                    _sampled_row_identity(row)
                    for row in covered_rows_by_label[label]
                ],
            }
            for label in required
            if len(covered_rows_by_label.get(label, [])) > 1
        ]

        trusted_covered: list[str] = []
        coverage_rows: list[dict[str, object]] = []
        source_hsp_to_sampled_kpoint: dict[str, str] = {}
        for label in covered:
            row_summaries: list[dict[str, object]] = []
            label_trusted = False
            for row in covered_rows_by_label[label]:
                kpoint = str(row.get("kpoint", ""))
                match = matching_by_kpoint.get(kpoint, {}).get(valley, {})
                decision = workflow_decisions_by_kpoint.get(kpoint, {}).get(
                    valley, {}
                )
                trusted = (
                    decision.get("readiness_level") == "trusted"
                    and match.get("matching_status") == "matched"
                    and not bool(match.get("diagnostic_only", False))
                    and row.get("representation_transport_status") == "validated"
                )
                label_trusted = label_trusted or trusted
                row_summaries.append({
                    **_sampled_row_identity(row),
                    "trusted_source_match": trusted,
                })
            if len(row_summaries) == 1:
                source_hsp_to_sampled_kpoint[label] = str(
                    row_summaries[0]["kpoint"]
                )
            if label_trusted:
                trusted_covered.append(label)
            coverage_rows.append({
                "source_hsp_label": label,
                "sampled_rows": row_summaries,
            })

        trusted_missing = [
            label for label in required if label not in trusted_covered
        ]
        missing_representatives = []
        for label in missing:
            source_row = source_by_label.get(label, {})
            missing_representatives.append({
                "source_hsp_label": label,
                "standard_representative_k_frac": source_row.get(
                    "standard_representative_k_frac"
                ),
                "inverse_parent_k_frac": source_row.get(
                    "inverse_parent_k_frac"
                ),
                "inverse_reciprocal_shift": source_row.get(
                    "inverse_reciprocal_shift"
                ),
                "provenance": basis.get("provenance", {}),
            })

        complete = (
            basis.get("status") == "validated"
            and not missing
            and not duplicates
            and not unresolved_rows
        )
        trusted_complete = complete and not trusted_missing
        by_valley[valley] = {
            "source_hsp_basis_status": basis.get("status", "blocked"),
            "required_source_hsp_labels": required,
            "covered_source_hsp_labels": covered,
            "missing_source_hsp_labels": missing,
            "trusted_matched_source_hsp_labels": trusted_covered,
            "trusted_missing_source_hsp_labels": trusted_missing,
            "source_hsp_to_sampled_kpoint": source_hsp_to_sampled_kpoint,
            "coverage_rows": coverage_rows,
            "generic_sampled_rows": generic_rows,
            "unresolved_sampled_rows": unresolved_rows,
            "duplicate_source_hsp_rows": duplicates,
            "missing_source_hsp_representatives": missing_representatives,
            "complete": complete,
            "trusted_matching_complete": trusted_complete,
            "ready_for_ebr_promotion": trusted_complete,
            "source_basis_provenance": basis.get("provenance", {}),
        }

    return {
        "status": "ok" if by_valley else "not_evaluated",
        "classification_model": (
            "representative | star_equivalent | generic | unresolved"
        ),
        "by_valley": by_valley,
        "all_valleys_complete": bool(by_valley) and all(
            bool(row.get("complete")) for row in by_valley.values()
        ),
        "all_valleys_ready_for_ebr_promotion": bool(by_valley) and all(
            bool(row.get("ready_for_ebr_promotion"))
            for row in by_valley.values()
        ),
    }


def _certificate_blocker(certificate: Mapping[str, object]) -> str:
    if certificate.get("validation_status") != "validated":
        return (
            "standard_setting_certificate_not_validated: "
            f"status={certificate.get('validation_status', 'missing')}"
        )
    if certificate.get("standard_operation_closure_validated") is not True:
        return "standard_setting_operation_closure_not_validated"
    return ""


def _certificate_transform(
    certificate: Mapping[str, object],
) -> np.ndarray | None:
    raw = certificate.get("parent_to_standard_direct_transform")
    try:
        transform = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        return None
    if abs(float(np.linalg.det(transform))) <= 1e-12:
        return None
    return transform


def _projected_space_group_identity(
    table: StandardIrrepTable,
    certificate: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    certificate_number = certificate.get("subspace_sg_number")
    hall_number = certificate.get("hall_number")
    hall_symbol = certificate.get("hall_symbol")
    symbol = certificate.get("subspace_sg_symbol")
    identity = {
        "number": table.number,
        "symbol": str(symbol) if isinstance(symbol, str) and symbol else table.name,
        "hall_number": hall_number,
        "hall_symbol": hall_symbol,
    }
    if certificate_number != table.number:
        return identity, (
            "source_table_certificate_space_group_mismatch: "
            f"table={table.number}, certificate={certificate_number}"
        )
    if (
        not isinstance(hall_number, int)
        or isinstance(hall_number, bool)
        or hall_number <= 0
        or not isinstance(hall_symbol, str)
        or not hall_symbol
    ):
        return identity, "standard_setting_hall_identity_missing"
    return identity, ""


def _certificate_centering_vectors(
    certificate: Mapping[str, object],
) -> list[np.ndarray] | None:
    raw = certificate.get("normalized_centering_vectors")
    if raw is None:
        raw = certificate.get("centering_vectors")
    if not isinstance(raw, list) or not raw:
        return None
    out: list[np.ndarray] = []
    for vector in raw:
        try:
            array = np.asarray(vector, dtype=float)
        except (TypeError, ValueError):
            return None
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            return None
        out.append(array)
    return out


def _inverse_parent_representative(
    *,
    standard_k: np.ndarray,
    transform: np.ndarray,
    centering_vectors: list[np.ndarray],
) -> dict[str, list[float] | list[int]] | None:
    # k_parent = T^T k_standard.  A source representative belongs to the
    # parent 2D plane iff an allowed standard reciprocal shift makes the
    # parent third coordinate integral.  Congruence residues are enumerated
    # exactly after rationalization, rather than assuming standard kz=0.
    row = transform.T[2, :]
    fractions = [_fraction(value) for value in row]
    k_fractions = [_fraction(value) for value in standard_k]
    centering_fractions = [
        [_fraction(value) for value in vector]
        for vector in centering_vectors
    ]
    modulus = 1
    for value in fractions + k_fractions:
        modulus = _lcm(modulus, value.denominator)
    for vector in centering_fractions:
        for value in vector:
            modulus = _lcm(modulus, value.denominator)
    if modulus ** 3 > _MAX_RESIDUE_CANDIDATES:
        raise _PlaneMembershipUnresolved(
            f"reciprocal residue search size {modulus ** 3} exceeds "
            f"the validated limit {_MAX_RESIDUE_CANDIDATES}"
        )

    candidates: list[tuple[tuple[int, tuple[int, int, int]], np.ndarray]] = []
    for i in range(modulus):
        for j in range(modulus):
            for k in range(modulus):
                raw_residue = (i, j, k)
                residue = np.asarray([
                    _symmetric_residue(value, modulus)
                    for value in raw_residue
                ], dtype=int)
                if not _reciprocal_shift_allowed(residue, centering_vectors):
                    continue
                parent_z = sum(
                    fractions[index] * (
                        k_fractions[index] + int(residue[index])
                    )
                    for index in range(3)
                )
                if parent_z.denominator != 1:
                    continue
                score = (
                    int(np.sum(np.abs(residue))),
                    tuple(int(x) for x in residue.tolist()),
                )
                candidates.append((score, residue))
    if not candidates:
        return None
    shift = min(candidates, key=lambda item: item[0])[1]
    parent_raw = transform.T @ (standard_k + shift)
    parent_z_integer = int(round(float(parent_raw[2])))
    inverse_shift = np.asarray([0, 0, -parent_z_integer], dtype=int)
    parent_plane = parent_raw + inverse_shift
    if abs(float(parent_plane[2])) > _TOL:
        return None
    return {
        "standard_reciprocal_shift": [int(x) for x in shift.tolist()],
        "inverse_parent_k_frac": _vector(parent_plane),
        "inverse_reciprocal_shift": [int(x) for x in inverse_shift.tolist()],
    }


def _equivalent_reciprocal_shift(
    target: np.ndarray,
    source: np.ndarray,
    centering_vectors: list[np.ndarray],
) -> list[int] | None:
    delta = np.asarray(target, dtype=float) - np.asarray(source, dtype=float)
    rounded = np.rint(delta).astype(int)
    if np.linalg.norm(delta - rounded) > _TOL:
        return None
    if not _reciprocal_shift_allowed(rounded, centering_vectors):
        return None
    return [int(value) for value in rounded.tolist()]


def _reciprocal_shift_allowed(
    shift: np.ndarray,
    centering_vectors: Sequence[np.ndarray],
) -> bool:
    return all(
        abs(float(np.dot(shift, vector)) - round(float(np.dot(shift, vector))))
        <= _TOL
        for vector in centering_vectors
    )


def _deduplicate_geometric_matches(
    matches: list[dict[str, object]],
) -> list[dict[str, object]]:
    # Representative evidence always wins.  For a star, the lowest source
    # operation index is the deterministic witness for the same HSP arm.
    matches.sort(key=lambda match: (
        0 if match["classification"] == "representative" else 1,
        str(match["source_row"].get("source_hsp_label")),
        (
            match["witness"].table_index
            if isinstance(match.get("witness"), StandardTableOperation)
            else 0
        ),
    ))
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for match in matches:
        key = (
            str(match["source_row"].get("source_hsp_label")),
            str(match["classification"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(match)
    return out


def _conjugated_little_group(
    *,
    table: StandardIrrepTable,
    source_operation_ids: Sequence[int],
    witness: StandardTableOperation,
    target_standard_k: np.ndarray,
) -> dict[str, object]:
    witness_inverse = _inverse_affine(witness)
    conjugated_ids: list[int] = []
    transport: list[dict[str, object]] = []
    nonzero_lattice_translation = False
    for source_id in source_operation_ids:
        source_operation = table.operation_by_index(int(source_id))
        rotation, translation = _compose_affine(
            witness.rotation_frac,
            witness.translation_frac,
            *_compose_affine(
                source_operation.rotation_frac,
                source_operation.translation_frac,
                *witness_inverse,
            ),
        )
        matched = _match_affine_operation(table, rotation, translation)
        if matched is None:
            return {
                "status": "blocked",
                "conjugated_operation_ids": conjugated_ids,
                "character_transport_map": transport,
                "blocker": (
                    "star_affine_conjugation_unmatched: source operation "
                    f"{source_id}, witness {witness.table_index}"
                ),
            }
        target_id, lattice_translation = matched
        target_operation = table.operation_by_index(target_id)
        if (
            witness.time_reversal
            or source_operation.time_reversal
            or target_operation.time_reversal
        ):
            return {
                "status": "blocked",
                "conjugated_operation_ids": conjugated_ids,
                "character_transport_map": transport,
                "blocker": (
                    "antiunitary_star_character_transport_not_implemented"
                ),
            }
        spin_lift_factor = _spin_lift_factor(
            table=table,
            witness=witness,
            source_operation=source_operation,
            target_operation=target_operation,
        )
        if spin_lift_factor is None:
            return {
                "status": "blocked",
                "conjugated_operation_ids": conjugated_ids,
                "character_transport_map": transport,
                "blocker": (
                    "spin_lift_conjugation_unresolved: source operation "
                    f"{source_id}, target operation {target_id}, witness "
                    f"{witness.table_index}"
                ),
            }
        conjugated_ids.append(target_id)
        lattice_shift = [int(x) for x in lattice_translation.tolist()]
        nonzero_lattice_translation = (
            nonzero_lattice_translation or any(lattice_shift)
        )
        transformed_k = (
            np.linalg.inv(target_operation.rotation_frac).T
            @ np.asarray(target_standard_k, dtype=float)
        )
        bloch_phase = np.exp(
            -2.0j
            * np.pi
            * float(transformed_k @ lattice_translation)
        )
        transport.append({
            "source_representative_operation_id": int(source_id),
            "star_arm_operation_id": target_id,
            "affine_lattice_translation": lattice_shift,
            "spin_lift_factor": spin_lift_factor,
            "bloch_phase": [
                float(np.real(bloch_phase)),
                float(np.imag(bloch_phase)),
            ],
            "bloch_phase_convention": (
                "exp(-2pii*(R_target^-T k_target)_dot_L)"
            ),
            "character_transport": (
                "unitary_conjugation"
                if not any(lattice_shift) and spin_lift_factor == 1
                else "unitary_conjugation_with_spin_lift"
                if not any(lattice_shift)
                else "unitary_conjugation_with_bloch_phase"
            ),
        })
    return {
        "status": "validated",
        "conjugated_operation_ids": sorted(conjugated_ids),
        "character_transport_map": transport,
        "nonzero_lattice_translation": nonzero_lattice_translation,
        "blocker": "",
    }


def _inverse_affine(
    operation: StandardTableOperation,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_inverse = np.linalg.inv(operation.rotation_frac).astype(int)
    return (
        rotation_inverse,
        -rotation_inverse @ operation.translation_frac,
    )


def _spin_lift_factor(
    *,
    table: StandardIrrepTable,
    witness: StandardTableOperation,
    source_operation: StandardTableOperation,
    target_operation: StandardTableOperation,
) -> int | None:
    if not table.spinor:
        return 1
    try:
        witness_spin = np.asarray(witness.spin_rotation, dtype=complex)
        source_spin = np.asarray(source_operation.spin_rotation, dtype=complex)
        target_spin = np.asarray(target_operation.spin_rotation, dtype=complex)
        conjugated = (
            witness_spin
            @ source_spin
            @ np.linalg.inv(witness_spin)
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if (
        conjugated.shape != (2, 2)
        or target_spin.shape != (2, 2)
        or not np.all(np.isfinite(conjugated))
        or not np.all(np.isfinite(target_spin))
    ):
        return None
    if np.linalg.norm(conjugated - target_spin) <= _TOL:
        return 1
    if np.linalg.norm(conjugated + target_spin) <= _TOL:
        return -1
    return None


def _compose_affine(
    rotation_left: np.ndarray,
    translation_left: np.ndarray,
    rotation_right: np.ndarray,
    translation_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.rint(rotation_left @ rotation_right).astype(int),
        rotation_left @ translation_right + translation_left,
    )


def _match_affine_operation(
    table: StandardIrrepTable,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[int, np.ndarray] | None:
    matches: list[tuple[int, np.ndarray]] = []
    for operation in table.operations:
        if not np.array_equal(rotation, operation.rotation_frac):
            continue
        delta = translation - operation.translation_frac
        rounded = np.rint(delta).astype(int)
        if np.linalg.norm(delta - rounded) <= _TOL:
            matches.append((operation.table_index, rounded))
    if len(matches) != 1:
        return None
    return matches[0]


def _serialize_witness(operation: object) -> dict[str, object] | None:
    if not isinstance(operation, StandardTableOperation):
        return None
    return {
        "table_index": operation.table_index,
        "rotation_frac": operation.rotation_frac.astype(int).tolist(),
        "translation_frac": _vector(operation.translation_frac),
    }


def _sampled_row_identity(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "kpoint": str(row.get("kpoint", "")),
        "classification": row.get("classification"),
        "standard_k_frac": row.get("standard_k_frac"),
        "reciprocal_shift": row.get("reciprocal_shift"),
        "standard_operation_witness": row.get("standard_operation_witness"),
        "validation_status": row.get("validation_status"),
        "representation_transport_status": row.get(
            "representation_transport_status"
        ),
    }


def _compact_classification(row: Mapping[str, object]) -> dict[str, object]:
    return {
        **_sampled_row_identity(row),
        "source_hsp_label": row.get("source_hsp_label"),
        "blocker": row.get("blocker", ""),
    }


def _blocked_basis(reason: str) -> dict[str, object]:
    return {
        "status": "blocked",
        "validation_status": "blocked",
        "required_source_hsp_labels": [],
        "source_hsps": [],
        "standard_plane_basis": [],
        "provenance": {},
        "blocker": reason,
    }


def _fraction(value: float) -> Fraction:
    raw = float(value)
    rational = Fraction(raw).limit_denominator(_MAX_RATIONAL_DENOMINATOR)
    if abs(float(rational) - raw) > _TOL:
        raise _PlaneMembershipUnresolved(
            f"value {raw:.12g} has no certified rational representation "
            f"with denominator <= {_MAX_RATIONAL_DENOMINATOR}"
        )
    return rational


class _PlaneMembershipUnresolved(ValueError):
    """Raised when exact-enough reciprocal-plane membership is unavailable."""


def _symmetric_residue(value: int, modulus: int) -> int:
    if value > modulus // 2:
        return value - modulus
    return value


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // gcd(left, right)


def _vector(value: object) -> list[float]:
    array = np.asarray(value, dtype=float).reshape(3)
    out: list[float] = []
    for component in array:
        normalized = 0.0 if abs(float(component)) <= 1e-12 else float(component)
        rounded = round(normalized, 12)
        out.append(0.0 if rounded == -0.0 else rounded)
    return out
