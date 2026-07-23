"""Center-derived time-reversal mapping of valley-projected subspaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from valleyscope.analysis.reduced_ebr_solver import (
    derive_coefficient_bounds,
    search_nonnegative_witnesses,
)
from valleyscope.analysis.time_reversal_sewing import (
    validate_time_reversal_sewing_report,
)
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector


_TOL = 5e-6


def derive_time_reversal_valley_mapping(
    *,
    enabled: bool,
    centers: Sequence[ValleyCenter],
    valley_subspaces: Sequence[ValleySector],
    spinor: bool,
) -> dict[str, object]:
    """Derive ``a -> a_bar`` from ``Q -> -Q`` in each layer lattice."""
    if not enabled:
        return {
            "status": "disabled",
            "enabled": False,
            "theta_square": None,
            "time_reversal_valley_mapping": {},
            "center_mapping": {},
            "valley_orbits": [],
            "blockers": [],
        }

    blockers: list[str] = []
    center_rows = list(centers)
    center_by_name = {center.name: center for center in center_rows}
    if len(center_by_name) != len(center_rows):
        blockers.append("duplicate_valley_center_name")

    center_mapping: dict[str, str] = {}
    for center in center_rows:
        candidates = [
            candidate.name for candidate in center_rows
            if _centers_are_time_reversal_partners(center, candidate)
        ]
        if len(candidates) != 1:
            blockers.append(
                "ambiguous_time_reversal_center_partner:"
                f"{center.name}:{candidates}"
            )
            continue
        center_mapping[center.name] = candidates[0]

    center_validation = validate_time_reversal_valley_mapping(
        mapping=center_mapping,
        valley_names=list(center_by_name),
        label="center",
    )
    blockers.extend(center_validation["blockers"])

    subspaces = list(valley_subspaces)
    subspace_by_name = {subspace.name: subspace for subspace in subspaces}
    if len(subspace_by_name) != len(subspaces):
        blockers.append("duplicate_valley_subspace_name")
    center_sets = {
        subspace.name: set(subspace.centers) for subspace in subspaces
    }
    valley_mapping: dict[str, str] = {}
    for subspace in subspaces:
        if any(name not in center_by_name for name in subspace.centers):
            blockers.append(
                f"unknown_center_in_valley_subspace:{subspace.name}"
            )
            continue
        mapped_centers = {
            center_mapping[name]
            for name in subspace.centers
            if name in center_mapping
        }
        candidates = [
            name for name, target_centers in center_sets.items()
            if target_centers == mapped_centers
        ]
        if len(mapped_centers) != len(subspace.centers) or len(candidates) != 1:
            blockers.append(
                "ambiguous_or_incomplete_time_reversal_valley_partner:"
                f"{subspace.name}:{candidates}"
            )
            continue
        valley_mapping[subspace.name] = candidates[0]

    valley_names = [subspace.name for subspace in subspaces]
    valley_validation = validate_time_reversal_valley_mapping(
        mapping=valley_mapping,
        valley_names=valley_names,
    )
    blockers.extend(valley_validation["blockers"])

    orbits: list[dict[str, object]] = []
    seen: set[str] = set()
    for valley in valley_names:
        if valley in seen or valley not in valley_mapping:
            continue
        partner = valley_mapping[valley]
        members = [valley] if partner == valley else [valley, partner]
        seen.update(members)
        row: dict[str, object] = {
            "representative": valley,
            "members": members,
            "mapping_type": (
                "self_mapped" if partner == valley else "exchanged"
            ),
        }
        if partner == valley:
            row["antiunitary_corepresentation_status"] = (
                "required_not_proven"
            )
        orbits.append(row)

    return {
        "status": "validated" if not blockers else "blocked",
        "enabled": True,
        "evidence": "configured_nonmagnetic_parent",
        "theta_square": -1 if spinor else 1,
        "spin_convention": (
            "spinful_theta_square_minus_one"
            if spinor else "scalar_theta_square_plus_one"
        ),
        "time_reversal_valley_mapping": valley_mapping,
        "center_mapping": center_mapping,
        "valley_orbits": orbits,
        "blockers": _deduplicate(blockers),
    }


def validate_time_reversal_valley_mapping(
    *,
    mapping: Mapping[str, str],
    valley_names: Sequence[str],
    label: str = "valley",
) -> dict[str, object]:
    """Validate exact coverage, bijectivity, and involution."""
    names = list(valley_names)
    blockers: list[str] = []
    if len(set(names)) != len(names):
        blockers.append(f"duplicate_time_reversal_{label}_name")
    if set(mapping) != set(names) or set(mapping.values()) != set(names):
        blockers.append(
            f"incomplete_or_nonbijective_time_reversal_{label}_mapping"
        )
    if any(mapping.get(mapping.get(name, "")) != name for name in names):
        blockers.append(f"non_involutive_time_reversal_{label}_mapping")
    return {
        "status": "validated" if not blockers else "blocked",
        "blockers": blockers,
    }


def build_time_reversal_valley_orbit_report(
    *,
    valley_mapping_report: Mapping[str, object],
    source_irrep_orbits_by_valley: Mapping[str, Mapping[str, object]],
    grey_source_by_valley: Mapping[str, Mapping[str, object]],
    ebr_input_candidates: Mapping[str, object] | None,
    antiunitary_sewing_report: Mapping[str, object] | None = None,
    trusted_projector_provenance_by_kpoint: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ] | None = None,
) -> dict[str, object]:
    """Build structural TR corep candidates and gate their physical trust."""
    if valley_mapping_report.get("status") != "validated":
        return _blocked_orbit_report(
            "time_reversal_valley_mapping_not_validated"
        )
    raw_orbits = valley_mapping_report.get("valley_orbits", [])
    if not isinstance(raw_orbits, list):
        return _blocked_orbit_report("time_reversal_valley_orbits_malformed")
    if not raw_orbits:
        return _blocked_orbit_report("time_reversal_valley_orbits_missing")
    candidates = (
        ebr_input_candidates.get("candidates", [])
        if isinstance(ebr_input_candidates, Mapping) else []
    )
    if not isinstance(candidates, list):
        candidates = []

    orbit_rows: list[dict[str, object]] = []
    for orbit_index, raw_orbit in enumerate(raw_orbits, start=1):
        if not isinstance(raw_orbit, Mapping):
            return _blocked_orbit_report(
                "time_reversal_valley_orbit_row_malformed"
            )
        members = raw_orbit.get("members", [])
        if not isinstance(members, list) or not all(
            isinstance(member, str) and member for member in members
        ):
            return _blocked_orbit_report(
                "time_reversal_valley_orbit_members_malformed"
            )
        structural_blockers: list[str] = []
        readiness_blockers: list[str] = []
        mapping_type = str(raw_orbit.get("mapping_type", ""))
        if mapping_type == "exchanged":
            if len(members) != 2:
                structural_blockers.append(
                    "malformed_exchanged_time_reversal_valley_orbit"
                )
        elif mapping_type == "self_mapped":
            if len(members) != 1:
                structural_blockers.append(
                    "malformed_self_mapped_time_reversal_valley_orbit"
                )
        else:
            structural_blockers.append(
                "unknown_time_reversal_valley_mapping_type"
            )
        representative = raw_orbit.get("representative")
        if representative not in members:
            structural_blockers.append(
                "time_reversal_valley_orbit_representative_malformed"
            )
            representative = members[0] if members else ""

        source_reports = [
            source_irrep_orbits_by_valley.get(member, {})
            for member in members
        ]
        grey_reports = [
            grey_source_by_valley.get(member, {}) for member in members
        ]
        if any(report.get("status") != "validated" for report in source_reports):
            structural_blockers.append(
                "time_reversal_source_irrep_orbits_not_validated"
            )
        if any(report.get("status") != "validated" for report in grey_reports):
            structural_blockers.append(
                "grey_group_time_reversal_source_not_validated"
            )
        if source_reports and any(
            report.get("time_reversal_hsp_mapping")
            != source_reports[0].get("time_reversal_hsp_mapping")
            or report.get("irrep_partner_by_label")
            != source_reports[0].get("irrep_partner_by_label")
            for report in source_reports[1:]
        ):
            structural_blockers.append(
                "valley_source_time_reversal_models_disagree"
            )
        if grey_reports and any(
            report.get("grey_bns_number")
            != grey_reports[0].get("grey_bns_number")
            or report.get("grey_unitary_restriction_by_irrep")
            != grey_reports[0].get("grey_unitary_restriction_by_irrep")
            for report in grey_reports[1:]
        ):
            structural_blockers.append(
                "valley_grey_source_models_disagree"
            )

        source_report = source_reports[0] if source_reports else {}
        grey_report = grey_reports[0] if grey_reports else {}
        hsp_mapping = source_report.get("time_reversal_hsp_mapping", {})
        irrep_mapping = source_report.get("irrep_partner_by_label", {})
        independent_hsps = source_report.get("independent_hsp_labels", [])
        if (
            not _is_nonempty_involutive_string_mapping(hsp_mapping)
            or not _is_nonempty_involutive_string_mapping(irrep_mapping)
            or not isinstance(independent_hsps, list)
            or not independent_hsps
            or any(
                not isinstance(hsp, str) or not hsp
                for hsp in independent_hsps
            )
            or len(set(independent_hsps)) != len(independent_hsps)
            or not set(independent_hsps).issubset(hsp_mapping)
        ):
            structural_blockers.append(
                "time_reversal_source_mapping_malformed"
            )
            hsp_mapping = {}
            irrep_mapping = {}
            independent_hsps = []

        grey_restrictions = grey_report.get(
            "grey_unitary_restriction_by_irrep", {}
        )
        grey_hsp_by_irrep = grey_report.get(
            "grey_source_hsp_by_irrep", {}
        )
        unitary_hsp_by_irrep = grey_report.get(
            "unitary_source_hsp_by_irrep", {}
        )
        if any(
            not isinstance(value, Mapping) or not value
            for value in (
                grey_restrictions,
                grey_hsp_by_irrep,
                unitary_hsp_by_irrep,
            )
        ):
            structural_blockers.append(
                "grey_group_time_reversal_source_mapping_malformed"
            )

        observed_completion_records = _candidate_unitary_completion_records(
            candidates, members
        )
        actual = _counts_from_completion_records(observed_completion_records)
        candidate_provenance_blockers = [
            str(blocker)
            for by_hsp in observed_completion_records.values()
            for records in by_hsp.values()
            for record in records
            for blocker in record.get("blockers", [])
            if isinstance(blocker, str)
            and blocker.startswith("source_candidate_provenance_incomplete:")
        ]
        structural_blockers.extend(candidate_provenance_blockers)
        (
            source_to_sampled,
            source_to_sampled_by_valley,
            source_mapping_blockers,
        ) = (
            _candidate_source_hsp_to_sampled_kpoint(
                candidates,
                members,
                independent_hsps=independent_hsps,
                representative=str(representative),
            )
        )
        structural_blockers.extend(source_mapping_blockers)
        (
            observed_source_to_sampled_by_valley,
            observed_source_mapping_blockers,
        ) = _observed_source_hsp_to_sampled_kpoint(
            observed_completion_records,
            members,
        )
        structural_blockers.extend(observed_source_mapping_blockers)
        _apply_source_mapping_readiness(
            records_by_valley=observed_completion_records,
            source_to_sampled_by_valley=(
                observed_source_to_sampled_by_valley
            ),
            blockers=observed_source_mapping_blockers,
        )
        projector_workflows: dict[str, dict[str, str]] = {}
        projector_provenance: dict[
            str, dict[str, dict[str, object]]
        ] = {}
        source_hsp_bindings: dict[
            str, dict[str, dict[str, object]]
        ] = {}
        if mapping_type == "self_mapped":
            (
                projector_workflows,
                projector_workflow_blockers,
            ) = _candidate_projector_workflows(candidates, members)
            readiness_blockers.extend(projector_workflow_blockers)
            (
                projector_provenance,
                projector_provenance_blockers,
            ) = _candidate_projector_provenance(
                projector_workflows,
                trusted_projector_provenance_by_kpoint,
            )
            readiness_blockers.extend(projector_provenance_blockers)
            (
                source_hsp_bindings,
                source_hsp_binding_blockers,
            ) = _candidate_source_hsp_bindings(candidates, members)
            structural_blockers.extend(source_hsp_binding_blockers)
            if not validate_time_reversal_sewing_report(
                antiunitary_sewing_report,
                valley_members=members,
                theta_square=valley_mapping_report.get("theta_square"),
                required_kpoints=list(source_to_sampled.values()),
                required_projector_workflows=projector_workflows,
                required_projector_provenance=projector_provenance,
            ):
                readiness_blockers.append(
                    "antiunitary_corepresentation_sewing_not_validated"
                )
        for member in members:
            for hsp in independent_hsps:
                if not actual.get(member, {}).get(str(hsp)):
                    structural_blockers.append(
                        f"missing_trusted_independent_hsp:{member}:{hsp}"
                    )

        inferred_records: dict[
            str, dict[str, list[dict[str, object]]]
        ] = {member: {} for member in members}
        if len(members) in (1, 2):
            raw_valley_mapping = valley_mapping_report.get(
                "time_reversal_valley_mapping", {}
            )
            partner_valley = {
                member: raw_valley_mapping.get(member) for member in members
            } if isinstance(raw_valley_mapping, Mapping) else {}
            for valley, by_hsp in observed_completion_records.items():
                for hsp, records in by_hsp.items():
                    partner_hsp = hsp_mapping.get(hsp)
                    if not isinstance(partner_hsp, str):
                        structural_blockers.append(
                            f"missing_time_reversal_hsp_partner:{hsp}"
                        )
                        continue
                    target_valley = partner_valley.get(valley)
                    if target_valley not in inferred_records:
                        structural_blockers.append(
                            f"missing_time_reversal_valley_partner:{valley}"
                        )
                        continue
                    target = inferred_records[target_valley].setdefault(
                        partner_hsp, []
                    )
                    for record in records:
                        irrep = record.get("irrep")
                        multiplicity = record.get("multiplicity")
                        partner_irrep = irrep_mapping.get(irrep)
                        if not isinstance(partner_irrep, str):
                            structural_blockers.append(
                                f"missing_time_reversal_irrep_partner:{irrep}"
                            )
                            continue
                        if (
                            not isinstance(multiplicity, int)
                            or isinstance(multiplicity, bool)
                            or multiplicity <= 0
                        ):
                            structural_blockers.append(
                                "time_reversal_evidence_multiplicity_invalid:"
                                f"{valley}:{hsp}:{irrep}"
                            )
                            continue
                        target.append(_inferred_completion_record(
                            observed=record,
                            target_valley=str(target_valley),
                            target_source_hsp=str(partner_hsp),
                            target_irrep=partner_irrep,
                        ))

        inferred = _counts_from_completion_records(inferred_records)

        completed: dict[str, dict[str, dict[str, int]]] = {
            member: {
                hsp: dict(counts) for hsp, counts in actual.get(member, {}).items()
            }
            for member in members
        }
        completion_records: dict[
            str, dict[str, list[dict[str, object]]]
        ] = {
            member: {
                hsp: [dict(record) for record in records]
                for hsp, records in observed_completion_records.get(
                    member, {}
                ).items()
            }
            for member in members
        }
        for member in members:
            for hsp, inferred_counts in inferred.get(member, {}).items():
                actual_counts = actual.get(member, {}).get(hsp)
                if actual_counts is not None and actual_counts != inferred_counts:
                    blocker = (
                        "time_reversal_multiplicity_or_irrep_mismatch:"
                        f"{member}:{hsp}:actual={actual_counts}:"
                        f"inferred={inferred_counts}"
                    )
                    structural_blockers.append(blocker)
                    _block_completion_records(
                        completion_records.get(member, {}).get(hsp, []),
                        blocker,
                    )
                    continue
                if actual_counts is not None:
                    _attach_time_reversal_consistency(
                        observed_records=(
                            completion_records.get(member, {}).get(hsp, [])
                        ),
                        inferred_records=inferred_records.get(
                            member, {}
                        ).get(hsp, []),
                    )
                if actual_counts is None:
                    completed[member][hsp] = dict(inferred_counts)
                    completion_records[member][hsp] = [
                        dict(record) for record in inferred_records.get(
                            member, {}
                        ).get(hsp, [])
                    ]

        if mapping_type == "self_mapped" and readiness_blockers:
            for by_hsp in completion_records.values():
                for records in by_hsp.values():
                    for record in records:
                        if record.get("completion_kind") != (
                            "inferred_by_time_reversal"
                        ):
                            continue
                        for blocker in readiness_blockers:
                            _block_completion_record_readiness(record, blocker)

        full_hsps = list(hsp_mapping)
        for member in members:
            for hsp in full_hsps:
                if not completed.get(member, {}).get(hsp):
                    structural_blockers.append(
                        f"incomplete_time_reversal_component:{member}:{hsp}"
                    )

        combined: dict[str, dict[str, int]] = {}
        for member in members:
            for hsp, counts in completed.get(member, {}).items():
                target = combined.setdefault(hsp, {})
                for irrep, multiplicity in counts.items():
                    target[irrep] = target.get(irrep, 0) + multiplicity
        if not combined:
            structural_blockers.append(
                "time_reversal_unitary_irrep_decomposition_empty"
            )

        grey_counts: dict[str, dict[str, int]] = {}
        if not structural_blockers:
            grey_counts, decomposition_blockers = _decompose_grey_counts(
                unitary_counts_by_hsp=combined,
                grey_restrictions=grey_restrictions,
                grey_hsp_by_irrep=grey_hsp_by_irrep,
                unitary_hsp_by_irrep=unitary_hsp_by_irrep,
            )
            structural_blockers.extend(decomposition_blockers)
            if not set(independent_hsps).issubset(grey_counts):
                structural_blockers.append(
                    "time_reversal_grey_decomposition_basis_incomplete"
                )

        all_blockers = _deduplicate([
            *structural_blockers,
            *readiness_blockers,
        ])

        irreps_by_kpoint = {
            hsp: [
                irrep
                for irrep, multiplicity in grey_counts.get(hsp, {}).items()
                for _ in range(multiplicity)
            ]
            for hsp in independent_hsps
            if grey_counts.get(hsp)
        }
        orbit_rows.append({
            "orbit_id": f"time_reversal_valley_orbit_{orbit_index:03d}",
            "representative": representative,
            "members": members,
            "mapping_type": mapping_type,
            "status": "validated" if not all_blockers else "blocked",
            "antiunitary_corepresentation_status": (
                "validated"
                if mapping_type == "self_mapped" and not all_blockers
                else "not_required_for_exchanged_orbit"
                if mapping_type == "exchanged"
                else "blocked"
            ),
            "unitary_valley_irreps": actual,
            "time_reversal_completed_unitary_valley_irreps": completed,
            "unitary_valley_irrep_completion_records": completion_records,
            "time_reversal_hsp_orbits": source_report.get(
                "time_reversal_hsp_orbits", []
            ),
            "time_reversal_irrep_pairing": dict(irrep_mapping),
            "full_unitary_source_hsp_labels": full_hsps,
            "independent_time_reversal_hsp_labels": independent_hsps,
            "source_hsp_to_sampled_kpoint": source_to_sampled,
            "source_hsp_to_sampled_kpoint_by_valley": (
                source_to_sampled_by_valley
            ),
            "independent_source_hsp_to_sampled_kpoint": source_to_sampled,
            "independent_source_hsp_to_sampled_kpoint_by_valley": (
                source_to_sampled_by_valley
            ),
            "observed_source_hsp_to_sampled_kpoint_by_valley": (
                observed_source_to_sampled_by_valley
            ),
            "projector_workflow_by_sampled_kpoint": projector_workflows,
            "projector_provenance_by_sampled_kpoint": projector_provenance,
            "source_hsp_binding_by_sampled_kpoint": source_hsp_bindings,
            "grey_bns_number": grey_report.get("grey_bns_number"),
            "grey_irrep_multiplicities_by_hsp": grey_counts,
            "irreps_by_kpoint": irreps_by_kpoint,
            "expected_hsps": list(independent_hsps),
            "structural_blockers": _deduplicate(structural_blockers),
            "readiness_blockers": _deduplicate(readiness_blockers),
            "blockers": all_blockers,
        })

    return {
        "status": (
            "validated" if orbit_rows and all(
                row["status"] == "validated" for row in orbit_rows
            ) else "blocked"
        ),
        "enabled": True,
        "theta_square": valley_mapping_report.get("theta_square"),
        "time_reversal_valley_mapping": valley_mapping_report.get(
            "time_reversal_valley_mapping", {}
        ),
        "antiunitary_sewing_evidence": (
            dict(antiunitary_sewing_report)
            if isinstance(antiunitary_sewing_report, Mapping) else {}
        ),
        "valley_orbits": orbit_rows,
        "blockers": _deduplicate([
            blocker for row in orbit_rows for blocker in row["blockers"]
        ]),
    }


def _candidate_unitary_completion_records(
    candidates: Sequence[object],
    members: Sequence[str],
) -> dict[str, dict[str, list[dict[str, object]]]]:
    """Build row-level observed provenance for trusted unitary candidates."""
    out: dict[str, dict[str, list[dict[str, object]]]] = {
        member: {} for member in members
    }
    for raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        valley = raw.get("valley")
        irrep = raw.get("matched_irrep")
        multiplicity = raw.get("irrep_multiplicity", 1)
        source_hsp = _candidate_source_hsp(raw)
        if (
            valley not in out
            or raw.get("ready_for_ebr_input") is not True
            or not isinstance(irrep, str)
            or not irrep
            or not isinstance(source_hsp, str)
            or not source_hsp
            or not isinstance(multiplicity, int)
            or isinstance(multiplicity, bool)
            or multiplicity <= 0
        ):
            continue
        sampled = raw.get("kpoint")
        record_blockers: list[str] = []
        if not isinstance(sampled, str) or not sampled:
            record_blockers.append(
                f"source_hsp_sampled_kpoint_missing:{valley}:{source_hsp}"
            )
            sampled = None
        record_blockers.extend(_candidate_provenance_blockers(
            raw,
            valley=str(valley),
            source_hsp=source_hsp,
            sampled=sampled,
            irrep=irrep,
            multiplicity=multiplicity,
        ))
        record: dict[str, object] = {
            "completion_kind": "observed_at_sampled_kpoint",
            "target_valley": str(valley),
            "target_source_hsp_label": source_hsp,
            "irrep": irrep,
            "multiplicity": multiplicity,
            "evidence_valley": str(valley),
            "evidence_source_hsp_label": source_hsp,
            "evidence_sampled_kpoint": sampled,
            "source_candidate_identity": _source_candidate_identity(
                raw, source_hsp=source_hsp
            ),
            "source_candidate_provenance": (
                _source_candidate_provenance(raw)
            ),
            "structural_status": (
                "validated" if not record_blockers else "blocked"
            ),
            "readiness_status": (
                "trusted" if not record_blockers else "blocked"
            ),
            "blockers": record_blockers,
        }
        if sampled is not None:
            record["sampled_kpoint"] = sampled
        out[str(valley)].setdefault(source_hsp, []).append(record)
    return out


def _candidate_source_hsp(raw: Mapping[str, object]) -> str | None:
    provenance = raw.get("irrep_source_provenance", {})
    classification = raw.get("projected_hsp_classification", {})
    source_hsp = (
        classification.get("source_hsp_label")
        if isinstance(classification, Mapping) else None
    ) or (
        provenance.get("source_hsp_label")
        if isinstance(provenance, Mapping) else None
    )
    return source_hsp if isinstance(source_hsp, str) and source_hsp else None


def _source_candidate_identity(
    raw: Mapping[str, object],
    *,
    source_hsp: str,
) -> dict[str, object]:
    return {
        "source": str(raw.get("source", "")),
        "workflow_path": str(raw.get("workflow_path", "")),
        "valley": str(raw.get("valley", "")),
        "source_hsp_label": source_hsp,
        "sampled_kpoint": raw.get("kpoint"),
        "irrep": raw.get("matched_irrep"),
        "multiplicity": raw.get("irrep_multiplicity", 1),
    }


def _source_candidate_provenance(
    raw: Mapping[str, object],
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "source": str(raw.get("source", "")),
        "workflow_path": str(raw.get("workflow_path", "")),
    }
    for key in (
        "irrep_source_provenance",
        "projected_hsp_classification",
        "subspace_space_group",
    ):
        value = raw.get(key)
        if isinstance(value, Mapping):
            provenance[key] = dict(value)
    return provenance


def _candidate_provenance_blockers(
    raw: Mapping[str, object],
    *,
    valley: str,
    source_hsp: str,
    sampled: object,
    irrep: str,
    multiplicity: int,
) -> list[str]:
    """Require the identity and table evidence used by TR completion."""
    missing: list[str] = []
    source = raw.get("source")
    workflow_path = raw.get("workflow_path")
    source_provenance = raw.get("irrep_source_provenance")
    if not isinstance(source, str) or not source:
        missing.append("source")
    if workflow_path not in ("direct_qcut", "symmetry_adapted"):
        missing.append("workflow_path")
    if raw.get("valley") != valley:
        missing.append("valley")
    if not isinstance(sampled, str) or not sampled:
        missing.append("sampled_kpoint")
    if raw.get("matched_irrep") != irrep:
        missing.append("irrep")
    if raw.get("irrep_multiplicity", 1) != multiplicity:
        missing.append("multiplicity")
    if not isinstance(source_provenance, Mapping):
        missing.append("irrep_source_provenance")
    else:
        if source_provenance.get("source_hsp_label") != source_hsp:
            missing.append("source_hsp_label")
        if not isinstance(source_provenance.get("source_table_spinor"), bool):
            missing.append("source_table_spinor")
    return [
        "source_candidate_provenance_incomplete:"
        f"{valley}:{source_hsp}:{field}"
        for field in missing
    ]


def _apply_source_mapping_readiness(
    *,
    records_by_valley: dict[
        str, dict[str, list[dict[str, object]]]
    ],
    source_to_sampled_by_valley: Mapping[str, Mapping[str, str]],
    blockers: Sequence[str],
) -> None:
    """Mark observed records whose sampled binding is missing or ambiguous."""
    for valley, by_hsp in records_by_valley.items():
        mapping = source_to_sampled_by_valley.get(valley, {})
        for source_hsp, records in by_hsp.items():
            if not records:
                continue
            relevant = [
                blocker for blocker in blockers
                if f":{valley}:{source_hsp}" in blocker
            ]
            expected = mapping.get(source_hsp)
            if not isinstance(expected, str) or not expected:
                relevant.append(
                    "source_hsp_sampled_kpoint_mapping_incomplete:"
                    f"{valley}:{source_hsp}"
                )
            for record in records:
                sampled = record.get("sampled_kpoint")
                if expected is not None and sampled != expected:
                    relevant.append(
                        "source_hsp_sampled_kpoint_binding_conflict:"
                        f"{valley}:{source_hsp}:{sampled}:{expected}"
                    )
            for blocker in _deduplicate(relevant):
                _block_completion_records(records, blocker)


def _observed_source_hsp_to_sampled_kpoint(
    records_by_valley: Mapping[
        str, Mapping[str, Sequence[Mapping[str, object]]]
    ],
    members: Sequence[str],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Resolve every directly observed source-HSP/sample binding."""
    mappings: dict[str, dict[str, str]] = {
        str(member): {} for member in members
    }
    blockers: list[str] = []
    for valley, by_hsp in records_by_valley.items():
        for source_hsp, records in by_hsp.items():
            sampled_values = {
                str(record.get("sampled_kpoint"))
                for record in records
                if isinstance(record.get("sampled_kpoint"), str)
                and record.get("sampled_kpoint")
            }
            if len(sampled_values) == 1:
                mappings[str(valley)][str(source_hsp)] = next(
                    iter(sampled_values)
                )
            elif not sampled_values:
                blockers.append(
                    "observed_source_hsp_sampled_kpoint_mapping_incomplete:"
                    f"{valley}:{source_hsp}"
                )
            else:
                blockers.append(
                    "observed_source_hsp_sampled_kpoint_mapping_ambiguous:"
                    f"{valley}:{source_hsp}:"
                    + ":".join(sorted(sampled_values))
                )
    for valley, by_source_hsp in mappings.items():
        by_sample: dict[str, list[str]] = {}
        for source_hsp, sampled in by_source_hsp.items():
            by_sample.setdefault(sampled, []).append(source_hsp)
        for sampled, source_hsps in by_sample.items():
            if len(source_hsps) > 1:
                blockers.append(
                    "observed_source_hsp_sampled_kpoint_mapping_noninjective:"
                    f"{valley}:{sampled}:"
                    + ":".join(sorted(source_hsps))
                )
    return mappings, _deduplicate(blockers)


def _attach_time_reversal_consistency(
    *,
    observed_records: Sequence[dict[str, object]],
    inferred_records: Sequence[dict[str, object]],
) -> None:
    """Preserve the reviewed inference that a redundant observation matched."""
    remaining = list(inferred_records)
    for observed in observed_records:
        match_index = next((
            index for index, inferred in enumerate(remaining)
            if inferred.get("irrep") == observed.get("irrep")
            and inferred.get("multiplicity") == observed.get("multiplicity")
        ), None)
        if match_index is None:
            continue
        inferred = remaining.pop(match_index)
        observed["time_reversal_consistency"] = {
            "evidence_valley": inferred.get("evidence_valley"),
            "evidence_source_hsp_label": inferred.get(
                "evidence_source_hsp_label"
            ),
            "evidence_sampled_kpoint": inferred.get(
                "evidence_sampled_kpoint"
            ),
            "reviewed_time_reversal_relation": dict(
                inferred.get("reviewed_time_reversal_relation", {})
            ),
            "source_candidate_identity": dict(
                inferred.get("source_candidate_identity", {})
            ),
            "source_candidate_provenance": dict(
                inferred.get("source_candidate_provenance", {})
            ),
            "status": "validated",
        }


def _inferred_completion_record(
    *,
    observed: Mapping[str, object],
    target_valley: str,
    target_source_hsp: str,
    target_irrep: str,
) -> dict[str, object]:
    evidence_valley = str(observed.get("target_valley", ""))
    evidence_source_hsp = str(
        observed.get("target_source_hsp_label", "")
    )
    evidence_irrep = str(observed.get("irrep", ""))
    blockers = [
        str(value) for value in observed.get("blockers", [])
        if isinstance(value, str)
    ] if isinstance(observed.get("blockers"), list) else []
    return {
        "completion_kind": "inferred_by_time_reversal",
        "target_valley": target_valley,
        "target_source_hsp_label": target_source_hsp,
        "irrep": target_irrep,
        "multiplicity": observed.get("multiplicity"),
        "evidence_valley": evidence_valley,
        "evidence_source_hsp_label": evidence_source_hsp,
        "evidence_sampled_kpoint": observed.get("sampled_kpoint"),
        "reviewed_time_reversal_relation": {
            "evidence_valley": evidence_valley,
            "target_valley": target_valley,
            "evidence_source_hsp_label": evidence_source_hsp,
            "target_source_hsp_label": target_source_hsp,
            "evidence_irrep": evidence_irrep,
            "target_irrep": target_irrep,
        },
        "source_candidate_identity": dict(
            observed.get("source_candidate_identity", {})
        ) if isinstance(
            observed.get("source_candidate_identity"), Mapping
        ) else {},
        "source_candidate_provenance": dict(
            observed.get("source_candidate_provenance", {})
        ) if isinstance(
            observed.get("source_candidate_provenance"), Mapping
        ) else {},
        "structural_status": (
            "validated" if not blockers else "blocked"
        ),
        "readiness_status": (
            "trusted"
            if observed.get("readiness_status") == "trusted" and not blockers
            else "blocked"
        ),
        "blockers": blockers,
    }


def _counts_from_completion_records(
    records_by_valley: Mapping[
        str, Mapping[str, Sequence[Mapping[str, object]]]
    ],
) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {
        str(valley): {} for valley in records_by_valley
    }
    for valley, by_hsp in records_by_valley.items():
        for source_hsp, records in by_hsp.items():
            target = counts[str(valley)].setdefault(str(source_hsp), {})
            for record in records:
                irrep = record.get("irrep")
                multiplicity = record.get("multiplicity")
                if (
                    not isinstance(irrep, str)
                    or not irrep
                    or not isinstance(multiplicity, int)
                    or isinstance(multiplicity, bool)
                    or multiplicity <= 0
                ):
                    continue
                target[irrep] = target.get(irrep, 0) + multiplicity
    return counts


def _block_completion_records(
    records: Sequence[dict[str, object]],
    blocker: str,
) -> None:
    for record in records:
        _block_completion_record(record, blocker)


def _block_completion_record(
    record: dict[str, object],
    blocker: str,
) -> None:
    blockers = record.get("blockers")
    if not isinstance(blockers, list):
        blockers = []
        record["blockers"] = blockers
    if blocker not in blockers:
        blockers.append(blocker)
    record["structural_status"] = "blocked"
    record["readiness_status"] = "blocked"


def _block_completion_record_readiness(
    record: dict[str, object],
    blocker: str,
) -> None:
    """Block promotion without erasing structurally complete TR evidence."""
    blockers = record.get("blockers")
    if not isinstance(blockers, list):
        blockers = []
        record["blockers"] = blockers
    if blocker not in blockers:
        blockers.append(blocker)
    record["readiness_status"] = "blocked"


def _candidate_source_hsp_to_sampled_kpoint(
    candidates: Sequence[object],
    members: Sequence[str],
    *,
    independent_hsps: Sequence[object],
    representative: str,
) -> tuple[dict[str, str], dict[str, dict[str, str]], list[str]]:
    """Resolve representative and lossless per-valley sampled-HSP maps."""
    independent = _deduplicate([
        str(value) for value in independent_hsps
        if isinstance(value, str) and value
    ])
    independent_set = set(independent)
    sampled_by_valley: dict[str, dict[str, str]] = {
        member: {} for member in members
    }
    blockers: list[str] = []
    for raw in candidates:
        if (
            not isinstance(raw, Mapping)
            or raw.get("valley") not in members
            or raw.get("ready_for_ebr_input") is not True
        ):
            continue
        provenance = raw.get("irrep_source_provenance", {})
        classification = raw.get("projected_hsp_classification", {})
        source_hsp = (
            classification.get("source_hsp_label")
            if isinstance(classification, Mapping) else None
        ) or (
            provenance.get("source_hsp_label")
            if isinstance(provenance, Mapping) else None
        )
        valley = str(raw.get("valley"))
        sampled = raw.get("kpoint")
        if not isinstance(source_hsp, str) or not source_hsp:
            continue
        if source_hsp not in independent_set:
            continue
        if not isinstance(sampled, str) or not sampled:
            blockers.append(
                f"source_hsp_sampled_kpoint_missing:{valley}:{source_hsp}"
            )
            continue
        by_source = sampled_by_valley[valley]
        previous = by_source.setdefault(source_hsp, sampled)
        if previous != sampled:
            blockers.append(
                "source_hsp_sampled_kpoint_mapping_ambiguous:"
                f"{valley}:{source_hsp}:{previous}:{sampled}"
            )
    for member, by_source in sampled_by_valley.items():
        for source_hsp in independent:
            if source_hsp not in by_source:
                blockers.append(
                    "source_hsp_sampled_kpoint_mapping_incomplete:"
                    f"{member}:{source_hsp}"
                )
        if len(set(by_source.values())) != len(by_source):
            blockers.append(
                f"source_hsp_sampled_kpoint_mapping_noninjective:{member}"
            )
    representative_mapping = sampled_by_valley.get(representative, {})
    return (
        dict(representative_mapping),
        sampled_by_valley,
        _deduplicate(blockers),
    )


def _is_nonempty_involutive_string_mapping(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    if any(
        not isinstance(label, str)
        or not label
        or not isinstance(partner, str)
        or not partner
        for label, partner in value.items()
    ):
        return False
    return all(value.get(value.get(label)) == label for label in value)


def _candidate_projector_workflows(
    candidates: Sequence[object],
    members: Sequence[str],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    workflows: dict[str, dict[str, str]] = {}
    blockers: list[str] = []
    for raw in candidates:
        if (
            not isinstance(raw, Mapping)
            or raw.get("valley") not in members
            or raw.get("ready_for_ebr_input") is not True
        ):
            continue
        valley = raw.get("valley")
        sampled = raw.get("kpoint")
        workflow_path = raw.get("workflow_path")
        if not isinstance(sampled, str) or not sampled:
            blockers.append(
                f"antiunitary_candidate_sampled_kpoint_missing:{valley}"
            )
            continue
        if workflow_path not in ("direct_qcut", "symmetry_adapted"):
            blockers.append(
                "antiunitary_candidate_projector_workflow_invalid:"
                f"{sampled}:{valley}:{workflow_path}"
            )
            continue
        by_valley = workflows.setdefault(sampled, {})
        previous = by_valley.setdefault(str(valley), str(workflow_path))
        if previous != workflow_path:
            blockers.append(
                "antiunitary_candidate_projector_workflow_ambiguous:"
                f"{sampled}:{valley}:{previous}:{workflow_path}"
            )
    return workflows, _deduplicate(blockers)


def _candidate_projector_provenance(
    workflows: Mapping[str, Mapping[str, str]],
    trusted: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ] | None,
) -> tuple[dict[str, dict[str, dict[str, object]]], list[str]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    blockers: list[str] = []
    if not isinstance(trusted, Mapping):
        return {}, ["antiunitary_trusted_projector_provenance_missing"]
    for sampled, by_valley in workflows.items():
        trusted_by_valley = trusted.get(sampled)
        if not isinstance(trusted_by_valley, Mapping):
            blockers.append(
                f"antiunitary_trusted_projector_provenance_missing:{sampled}"
            )
            continue
        for valley, workflow_path in by_valley.items():
            raw = trusted_by_valley.get(valley)
            if (
                not isinstance(raw, Mapping)
                or raw.get("workflow_path") != workflow_path
            ):
                blockers.append(
                    "antiunitary_trusted_projector_provenance_mismatch:"
                    f"{sampled}:{valley}:{workflow_path}"
                )
                continue
            result.setdefault(sampled, {})[valley] = dict(raw)
    return result, _deduplicate(blockers)


def _candidate_source_hsp_bindings(
    candidates: Sequence[object],
    members: Sequence[str],
) -> tuple[dict[str, dict[str, dict[str, object]]], list[str]]:
    result: dict[str, dict[str, dict[str, object]]] = {}
    blockers: list[str] = []
    for raw in candidates:
        if (
            not isinstance(raw, Mapping)
            or raw.get("valley") not in members
            or raw.get("ready_for_ebr_input") is not True
        ):
            continue
        sampled = raw.get("kpoint")
        valley = raw.get("valley")
        classification = raw.get("projected_hsp_classification")
        if (
            not isinstance(sampled, str)
            or not sampled
            or not isinstance(valley, str)
            or not isinstance(classification, Mapping)
        ):
            blockers.append(
                f"antiunitary_source_hsp_binding_missing:{sampled}:{valley}"
            )
            continue
        source_hsp = classification.get("source_hsp_label")
        parent_k = _finite_vector3(classification.get("parent_k_frac"))
        standard_k = _finite_vector3(classification.get("standard_k_frac"))
        representative_k = _finite_vector3(
            classification.get("source_hsp_representative_k_frac")
        )
        witness = classification.get("standard_operation_witness")
        standard_operation_index: int | None = None
        if classification.get("classification") == "star_equivalent":
            if not isinstance(witness, Mapping):
                blockers.append(
                    "antiunitary_source_hsp_binding_not_validated:"
                    f"{sampled}:{valley}:{source_hsp}"
                )
                continue
            raw_operation_index = witness.get("table_index")
            if (
                not isinstance(raw_operation_index, int)
                or isinstance(raw_operation_index, bool)
                or raw_operation_index <= 0
            ):
                blockers.append(
                    "antiunitary_source_hsp_binding_not_validated:"
                    f"{sampled}:{valley}:{source_hsp}"
                )
                continue
            standard_operation_index = raw_operation_index
        elif witness is not None:
            blockers.append(
                "antiunitary_source_hsp_binding_not_validated:"
                f"{sampled}:{valley}:{source_hsp}"
            )
            continue
        if (
            not isinstance(source_hsp, str)
            or not source_hsp
            or classification.get("classification")
            not in ("representative", "star_equivalent")
            or classification.get("source_hsp_membership") is not True
            or classification.get("validation_status") != "validated"
            or parent_k is None
            or standard_k is None
            or representative_k is None
        ):
            blockers.append(
                "antiunitary_source_hsp_binding_not_validated:"
                f"{sampled}:{valley}:{source_hsp}"
            )
            continue
        binding = {
            "source_hsp_label": source_hsp,
            "classification": str(classification.get("classification")),
            "validation_status": "validated",
            "parent_k_frac": parent_k,
            "standard_k_frac": standard_k,
            "source_hsp_representative_k_frac": representative_k,
            "standard_operation_index": standard_operation_index,
        }
        by_valley = result.setdefault(sampled, {})
        previous = by_valley.setdefault(valley, binding)
        if previous != binding:
            blockers.append(
                "antiunitary_source_hsp_binding_ambiguous:"
                f"{sampled}:{valley}"
            )
    return result, _deduplicate(blockers)


def _finite_vector3(value: object) -> list[float] | None:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    return [float(item) for item in vector]


def _decompose_grey_counts(
    *,
    unitary_counts_by_hsp: Mapping[str, Mapping[str, int]],
    grey_restrictions: object,
    grey_hsp_by_irrep: object,
    unitary_hsp_by_irrep: object,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    if not all(isinstance(value, Mapping) for value in (
        grey_restrictions, grey_hsp_by_irrep, unitary_hsp_by_irrep
    )):
        return {}, ["grey_group_restriction_model_malformed"]
    result: dict[str, dict[str, int]] = {}
    blockers: list[str] = []
    for hsp, target_raw in unitary_counts_by_hsp.items():
        if not isinstance(target_raw, Mapping) or not target_raw:
            blockers.append(f"missing_unitary_irrep_target_for_hsp:{hsp}")
            continue
        target = dict(target_raw)
        invalid_multiplicities = [
            str(label) for label, multiplicity in target.items()
            if not isinstance(multiplicity, int)
            or isinstance(multiplicity, bool)
            or multiplicity <= 0
        ]
        if invalid_multiplicities:
            blockers.append(
                f"invalid_unitary_irrep_multiplicity:{hsp}:"
                f"{invalid_multiplicities}"
            )
            continue
        unknown = [
            str(label) for label in target
            if label not in unitary_hsp_by_irrep
        ]
        if unknown:
            blockers.extend(
                f"unknown_unitary_irrep_in_grey_target:{hsp}:{label}"
                for label in unknown
            )
            continue
        wrong_hsp = [
            (str(label), str(unitary_hsp_by_irrep[label]))
            for label in target
            if unitary_hsp_by_irrep[label] != hsp
        ]
        if wrong_hsp:
            blockers.extend(
                f"wrong_hsp_unitary_irrep_in_grey_target:{hsp}:"
                f"{label}:{source_hsp}"
                for label, source_hsp in wrong_hsp
            )
            continue
        labels = [
            str(label) for label, source_hsp in grey_hsp_by_irrep.items()
            if source_hsp == hsp and label in grey_restrictions
        ]
        if not labels:
            blockers.append(f"missing_grey_irrep_basis_for_hsp:{hsp}")
            continue
        unit_labels = sorted({
            str(label) for label, source_hsp in unitary_hsp_by_irrep.items()
            if source_hsp == hsp
        })
        if not unit_labels:
            blockers.append(f"missing_unitary_irrep_basis_for_hsp:{hsp}")
            continue
        malformed_restrictions = [
            label for label in labels
            if not isinstance(grey_restrictions[label], Mapping)
            or not grey_restrictions[label]
            or any(
                unit_label not in unit_labels
                or not isinstance(multiplicity, int)
                or isinstance(multiplicity, bool)
                or multiplicity <= 0
                for unit_label, multiplicity
                in grey_restrictions[label].items()
            )
        ]
        if malformed_restrictions:
            blockers.append(
                f"malformed_grey_irrep_restriction:{hsp}:"
                f"{malformed_restrictions}"
            )
            continue
        target_vector = tuple(int(target.get(label, 0)) for label in unit_labels)
        columns = [
            tuple(
                int(grey_restrictions[label].get(unit_label, 0))
                for unit_label in unit_labels
            )
            for label in labels
        ]
        target_list = list(target_vector)
        column_lists = [list(column) for column in columns]
        raw_bounds = derive_coefficient_bounds(target_list, column_lists)
        if any(bound is None for bound in raw_bounds):
            blockers.append(f"zero_grey_irrep_restriction:{hsp}")
            continue
        solutions = search_nonnegative_witnesses(
            target_list,
            column_lists,
            [int(bound) for bound in raw_bounds],
            max_witnesses=2,
        )
        if len(solutions) != 1:
            blockers.append(
                f"ambiguous_or_missing_grey_irrep_decomposition:{hsp}:"
                f"solution_count={len(solutions)}"
            )
            continue
        result[hsp] = {
            label: coefficient
            for label, coefficient in zip(labels, solutions[0])
            if coefficient
        }
        if not result[hsp]:
            blockers.append(f"empty_grey_irrep_decomposition:{hsp}")
            result.pop(hsp, None)
    return result, blockers


def _blocked_orbit_report(reason: str) -> dict[str, object]:
    return {
        "status": "blocked",
        "enabled": True,
        "theta_square": None,
        "time_reversal_valley_mapping": {},
        "valley_orbits": [],
        "blockers": [reason],
    }


def _centers_are_time_reversal_partners(
    source: ValleyCenter,
    candidate: ValleyCenter,
) -> bool:
    if source.layer != candidate.layer or source.reciprocal_cart is None:
        return False
    if candidate.reciprocal_cart is None or not np.allclose(
        source.reciprocal_cart, candidate.reciprocal_cart, atol=_TOL, rtol=0.0
    ):
        return False
    try:
        delta_frac = (
            (-source.cart - candidate.cart)
            @ np.linalg.inv(source.reciprocal_cart)
        )
    except np.linalg.LinAlgError:
        return False
    return np.linalg.norm(delta_frac - np.rint(delta_frac)) <= _TOL


def _deduplicate(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
