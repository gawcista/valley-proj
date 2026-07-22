"""Fail-closed provenance validation for serialized unitary EBR vectors."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

from valleyscope.analysis.time_reversal_sewing import (
    validate_time_reversal_sewing_report,
)


_PROJECTOR_WORKFLOWS = frozenset({"direct_qcut", "symmetry_adapted"})


def unitary_bundle_claims_time_reversal_completion(
    bundle: Mapping[str, object],
) -> bool:
    """Return whether a unitary bundle claims or implies TR completion."""
    if bundle.get("problem_kind", "unitary_valley_reduced_ebr") != (
        "unitary_valley_reduced_ebr"
    ):
        return False
    construction = bundle.get("unitary_vector_construction")
    construction_kind = (
        construction.get("kind")
        if isinstance(construction, Mapping) else None
    )
    return any((
        construction_kind == "time_reversal_completed_unitary_rows",
        bundle.get("workflow_path")
        == "time_reversal_completed_unitary_valley",
        bool(bundle.get("valley_orbit")),
        bool(bundle.get("unitary_irrep_completion_records_by_hsp")),
        bool(bundle.get("time_reversal")),
    ))


def validate_unitary_bundle_provenance(
    bundle: Mapping[str, object],
) -> bool:
    """Validate either disjoint unitary-vector construction contract."""
    if unitary_bundle_claims_time_reversal_completion(bundle):
        return validate_tr_completed_unitary_bundle(bundle)
    return validate_direct_unitary_bundle(bundle)


def validate_direct_unitary_bundle(
    bundle: Mapping[str, object],
) -> bool:
    """Reconstruct a direct unitary vector from producer-owned records."""
    construction = bundle.get("unitary_vector_construction")
    records_by_sample = bundle.get("irrep_records_by_kpoint")
    irreps_by_sample = bundle.get("irreps_by_kpoint")
    source_to_sample = bundle.get("source_hsp_to_sampled_kpoint")
    required_source_hsps = bundle.get("required_source_hsp_labels")
    expected_samples = bundle.get("expected_hsps")
    valley = bundle.get("valley")
    spinor = bundle.get("spinor")
    certificate_identity = bundle.get("certificate_identity")
    if (
        bundle.get("problem_kind") != "unitary_valley_reduced_ebr"
        or bundle.get("physical_object_kind")
        != "unitary_valley_projected_subspace"
        or bundle.get("workflow_path") not in _PROJECTOR_WORKFLOWS
        or not isinstance(valley, str)
        or not valley
        or not isinstance(spinor, bool)
        or bundle.get("valley_orbit") not in ([], None)
        or bundle.get("time_reversal") not in ({}, None)
        or bundle.get("unitary_irrep_completion_records_by_hsp")
        not in ({}, None)
        or not isinstance(construction, Mapping)
        or construction.get("kind") != "direct_observed_unitary_rows"
        or construction.get("source") != "trusted_ebr_input_candidates"
        or not isinstance(records_by_sample, Mapping)
        or not records_by_sample
        or not isinstance(irreps_by_sample, Mapping)
        or not isinstance(source_to_sample, Mapping)
        or not isinstance(required_source_hsps, list)
        or not required_source_hsps
        or not isinstance(expected_samples, list)
        or not expected_samples
        or not isinstance(certificate_identity, Mapping)
        or not certificate_identity
    ):
        return False
    source_hsps = set(required_source_hsps)
    samples = set(expected_samples)
    if (
        len(source_hsps) != len(required_source_hsps)
        or len(samples) != len(expected_samples)
        or set(source_to_sample) != source_hsps
        or set(source_to_sample.values()) != samples
        or len(set(source_to_sample.values())) != len(source_to_sample)
        or set(records_by_sample) != samples
        or set(irreps_by_sample) != samples
    ):
        return False
    source_by_sample = {
        sampled: source_hsp
        for source_hsp, sampled in source_to_sample.items()
    }
    rebuilt: dict[str, Counter[str]] = {}
    for sampled, records in records_by_sample.items():
        expected_source_hsp = source_by_sample.get(sampled)
        if (
            not isinstance(sampled, str)
            or not sampled
            or not isinstance(expected_source_hsp, str)
            or not isinstance(records, list)
            or not records
        ):
            return False
        counts: Counter[str] = Counter()
        for record in records:
            if not isinstance(record, Mapping):
                return False
            irrep = record.get("matched_irrep")
            multiplicity = record.get("irrep_multiplicity")
            workflow = record.get("workflow_path")
            source_provenance = record.get("irrep_source_provenance")
            identity = record.get("source_candidate_identity")
            provenance = record.get("source_candidate_provenance")
            if (
                not isinstance(irrep, str)
                or not irrep
                or not _positive_int(multiplicity)
                or record.get("valley") != valley
                or record.get("sampled_kpoint") != sampled
                or record.get("source_hsp_label") != expected_source_hsp
                or workflow not in _PROJECTOR_WORKFLOWS
                or record.get("readiness_level") != "trusted"
                or not isinstance(record.get("source"), str)
                or not record.get("source")
                or not isinstance(source_provenance, Mapping)
                or source_provenance.get("source_hsp_label")
                != expected_source_hsp
                or source_provenance.get("source_table_spinor") is not spinor
                or record.get("certificate_identity")
                != certificate_identity
                or not isinstance(identity, Mapping)
                or identity.get("source") != record.get("source")
                or identity.get("workflow_path") != workflow
                or not _source_candidate_provenance_valid(
                    identity=identity,
                    provenance=provenance,
                    expected_valley=valley,
                    expected_hsp=expected_source_hsp,
                    expected_sample=sampled,
                    expected_irrep=irrep,
                    expected_multiplicity=multiplicity,
                    expected_spinor=spinor,
                )
            ):
                return False
            counts[irrep] += multiplicity
        rebuilt[sampled] = counts
    return all(
        isinstance(labels, list)
        and all(isinstance(label, str) and label for label in labels)
        and Counter(labels) == rebuilt[sampled]
        for sampled, labels in irreps_by_sample.items()
    )


def validate_tr_completed_unitary_bundle(
    bundle: Mapping[str, object],
) -> bool:
    """Validate serialized row-level TR completion and sewing dependencies."""
    valley = bundle.get("valley")
    valley_orbit = bundle.get("valley_orbit")
    time_reversal = bundle.get("time_reversal")
    irreps_by_hsp = bundle.get("irreps_by_kpoint")
    records_by_hsp = bundle.get("unitary_irrep_completion_records_by_hsp")
    independent_source_to_sample = bundle.get(
        "independent_source_hsp_to_sampled_kpoint"
    )
    observed_source_to_sample = bundle.get(
        "observed_source_hsp_to_sampled_kpoint"
    )
    expected_hsps = bundle.get("expected_hsps")
    construction = bundle.get("unitary_vector_construction")
    spinor = bundle.get("spinor")
    if (
        bundle.get("problem_kind") != "unitary_valley_reduced_ebr"
        or bundle.get("physical_object_kind")
        != "unitary_valley_projected_subspace"
        or bundle.get("workflow_path")
        != "time_reversal_completed_unitary_valley"
        or not isinstance(construction, Mapping)
        or construction.get("kind")
        != "time_reversal_completed_unitary_rows"
        or construction.get("source")
        != "validated_time_reversal_valley_orbit"
        or not isinstance(construction.get("orbit_id"), str)
        or not construction.get("orbit_id")
        or not isinstance(valley, str)
        or not valley
        or not isinstance(spinor, bool)
        or not isinstance(valley_orbit, list)
        or not valley_orbit
        or len(set(valley_orbit)) != len(valley_orbit)
        or not isinstance(time_reversal, Mapping)
        or time_reversal.get("mapping_type") not in {
            "exchanged", "self_mapped",
        }
        or time_reversal.get("valley_orbit") != valley_orbit
        or valley not in valley_orbit
        or time_reversal.get("theta_square") != (-1 if spinor else 1)
        or not isinstance(irreps_by_hsp, Mapping)
        or not isinstance(records_by_hsp, Mapping)
        or not isinstance(independent_source_to_sample, Mapping)
        or bundle.get("source_hsp_to_sampled_kpoint")
        != independent_source_to_sample
        or not _injective_string_mapping(observed_source_to_sample)
        or not isinstance(expected_hsps, list)
        or not expected_hsps
        or len(set(expected_hsps)) != len(expected_hsps)
        or set(expected_hsps) != set(irreps_by_hsp)
        or set(expected_hsps) != set(records_by_hsp)
        or bundle.get("irrep_records_by_kpoint") not in ({}, None)
    ):
        return False

    valley_mapping = time_reversal.get("time_reversal_valley_mapping")
    if (
        not _nonempty_involutive_string_mapping(valley_mapping)
        or set(valley_mapping) != set(valley_orbit)
        or set(valley_mapping.values()) != set(valley_orbit)
        or (
            time_reversal.get("mapping_type") == "exchanged"
            and (
                len(valley_orbit) != 2
                or any(valley_mapping[item] == item for item in valley_orbit)
            )
        )
        or (
            time_reversal.get("mapping_type") == "self_mapped"
            and (
                len(valley_orbit) != 1
                or valley_mapping.get(valley_orbit[0]) != valley_orbit[0]
            )
        )
    ):
        return False

    hsp_mapping, representative_hsps = _reviewed_hsp_involution(
        time_reversal.get("time_reversal_hsp_orbits")
    )
    full_hsps = time_reversal.get("full_unitary_source_hsp_labels")
    independent_hsps = time_reversal.get(
        "independent_time_reversal_hsp_labels"
    )
    if (
        hsp_mapping is None
        or not isinstance(full_hsps, list)
        or set(full_hsps) != set(expected_hsps)
        or set(hsp_mapping) != set(expected_hsps)
        or not isinstance(independent_hsps, list)
        or set(independent_hsps) != representative_hsps
        or set(independent_source_to_sample) != set(independent_hsps)
        or not _injective_string_mapping(independent_source_to_sample)
        or any(
            observed_source_to_sample.get(source_hsp) != sampled
            for source_hsp, sampled in independent_source_to_sample.items()
        )
    ):
        return False

    irrep_mapping = time_reversal.get("time_reversal_irrep_pairing")
    if not _nonempty_involutive_string_mapping(irrep_mapping):
        return False
    counts_by_hsp: dict[str, dict[str, int]] = {}
    vector_irreps: set[str] = set()
    for hsp, labels in irreps_by_hsp.items():
        if not isinstance(hsp, str) or not isinstance(labels, list):
            return False
        counts: dict[str, int] = {}
        for label in labels:
            if not isinstance(label, str) or not label:
                return False
            counts[label] = counts.get(label, 0) + 1
            vector_irreps.add(label)
        counts_by_hsp[hsp] = counts
    if not vector_irreps.issubset(irrep_mapping):
        return False
    records_valid, sewing_kpoints = _completion_records_valid(
        valley=valley,
        counts_by_hsp=counts_by_hsp,
        records_by_hsp=records_by_hsp,
        observed_source_to_sample=observed_source_to_sample,
        valley_mapping=valley_mapping,
        hsp_mapping=hsp_mapping,
        irrep_mapping=irrep_mapping,
        independent_hsps=set(independent_hsps),
        expected_spinor=spinor,
    )
    if not records_valid:
        return False
    if time_reversal.get("mapping_type") != "self_mapped" or not sewing_kpoints:
        return True
    sewing_evidence = time_reversal.get("antiunitary_sewing_evidence")
    if (
        not isinstance(sewing_evidence, Mapping)
        or sewing_evidence.get("status") != "validated"
        or sewing_evidence.get("blockers") != []
    ):
        return False
    return validate_time_reversal_sewing_report(
        sewing_evidence,
        valley_members=valley_orbit,
        theta_square=(-1 if spinor else 1),
        required_kpoints=sorted(sewing_kpoints),
        required_projector_workflows=time_reversal.get(
            "projector_workflow_by_sampled_kpoint"
        ),
        required_projector_provenance=time_reversal.get(
            "projector_provenance_by_sampled_kpoint"
        ),
    )


def _completion_records_valid(
    *,
    valley: str,
    counts_by_hsp: Mapping[str, Mapping[str, int]],
    records_by_hsp: Mapping[str, object],
    observed_source_to_sample: Mapping[str, str],
    valley_mapping: Mapping[str, str],
    hsp_mapping: Mapping[str, str],
    irrep_mapping: Mapping[str, str],
    independent_hsps: set[str],
    expected_spinor: bool,
) -> tuple[bool, set[str]]:
    if set(counts_by_hsp) != set(records_by_hsp):
        return False, set()
    direct_observation_complete = (
        set(observed_source_to_sample) == set(counts_by_hsp)
    )
    observed_hsps: set[str] = set()
    rebuilt: dict[str, dict[str, int]] = {}
    sewing_kpoints: set[str] = set()
    for hsp, records in records_by_hsp.items():
        if not isinstance(hsp, str) or not hsp or not isinstance(records, list):
            return False, set()
        target: dict[str, int] = {}
        for record in records:
            if not isinstance(record, Mapping):
                return False, set()
            irrep = record.get("irrep")
            multiplicity = record.get("multiplicity")
            kind = record.get("completion_kind")
            identity = record.get("source_candidate_identity")
            provenance = record.get("source_candidate_provenance")
            evidence_valley = (
                valley if kind == "observed_at_sampled_kpoint"
                else record.get("evidence_valley")
            )
            evidence_hsp = (
                hsp if kind == "observed_at_sampled_kpoint"
                else record.get("evidence_source_hsp_label")
            )
            evidence_sample = (
                record.get("sampled_kpoint")
                if kind == "observed_at_sampled_kpoint"
                else record.get("evidence_sampled_kpoint")
            )
            relation = record.get("reviewed_time_reversal_relation")
            evidence_irrep = (
                irrep if kind == "observed_at_sampled_kpoint"
                else relation.get("evidence_irrep")
                if isinstance(relation, Mapping) else None
            )
            if (
                record.get("target_valley") != valley
                or record.get("target_source_hsp_label") != hsp
                or not isinstance(irrep, str)
                or not irrep
                or not _positive_int(multiplicity)
                or record.get("structural_status") != "validated"
                or record.get("readiness_status") != "trusted"
                or record.get("blockers") not in ([], None)
                or not _source_candidate_provenance_valid(
                    identity=identity,
                    provenance=provenance,
                    expected_valley=evidence_valley,
                    expected_hsp=evidence_hsp,
                    expected_sample=evidence_sample,
                    expected_irrep=evidence_irrep,
                    expected_multiplicity=multiplicity,
                    expected_spinor=expected_spinor,
                )
            ):
                return False, set()
            if kind == "observed_at_sampled_kpoint":
                sampled = record.get("sampled_kpoint")
                if observed_source_to_sample.get(hsp) != sampled:
                    return False, set()
                observed_hsps.add(hsp)
                consistency = record.get("time_reversal_consistency")
                if (
                    not direct_observation_complete
                    and hsp not in independent_hsps
                ):
                    if not _observed_consistency_valid(
                        consistency=consistency,
                        valley=valley,
                        hsp=hsp,
                        irrep=irrep,
                        multiplicity=multiplicity,
                        valley_mapping=valley_mapping,
                        hsp_mapping=hsp_mapping,
                        irrep_mapping=irrep_mapping,
                        expected_spinor=expected_spinor,
                    ) or (
                        consistency.get("evidence_valley") == valley
                        and observed_source_to_sample.get(
                            consistency.get("evidence_source_hsp_label")
                        ) != consistency.get("evidence_sampled_kpoint")
                    ):
                        return False, set()
                    sewing_kpoints.add(
                        consistency["evidence_sampled_kpoint"]
                    )
            elif kind == "inferred_by_time_reversal":
                if (
                    "sampled_kpoint" in record
                    or (
                        evidence_valley == valley
                        and observed_source_to_sample.get(evidence_hsp)
                        != evidence_sample
                    )
                    or not _reviewed_relation_valid(
                        relation=relation,
                        evidence_valley=evidence_valley,
                        target_valley=valley,
                        evidence_hsp=evidence_hsp,
                        target_hsp=hsp,
                        evidence_irrep=evidence_irrep,
                        target_irrep=irrep,
                        valley_mapping=valley_mapping,
                        hsp_mapping=hsp_mapping,
                        irrep_mapping=irrep_mapping,
                    )
                ):
                    return False, set()
                sewing_kpoints.add(evidence_sample)
            else:
                return False, set()
            target[irrep] = target.get(irrep, 0) + multiplicity
        rebuilt[hsp] = target
    return (
        rebuilt == counts_by_hsp
        and set(observed_source_to_sample) == observed_hsps,
        sewing_kpoints,
    )


def _source_candidate_provenance_valid(
    *,
    identity: object,
    provenance: object,
    expected_valley: object,
    expected_hsp: object,
    expected_sample: object,
    expected_irrep: object,
    expected_multiplicity: object,
    expected_spinor: bool,
) -> bool:
    if not isinstance(identity, Mapping) or not isinstance(provenance, Mapping):
        return False
    workflow = provenance.get("workflow_path")
    source_irrep = provenance.get("irrep_source_provenance")
    return (
        isinstance(identity.get("source"), str)
        and bool(identity.get("source"))
        and provenance.get("source") == identity.get("source")
        and workflow in _PROJECTOR_WORKFLOWS
        and identity.get("workflow_path") == workflow
        and identity.get("valley") == expected_valley
        and identity.get("source_hsp_label") == expected_hsp
        and identity.get("sampled_kpoint") == expected_sample
        and identity.get("irrep") == expected_irrep
        and identity.get("multiplicity") == expected_multiplicity
        and isinstance(source_irrep, Mapping)
        and source_irrep.get("source_hsp_label") == expected_hsp
        and source_irrep.get("source_table_spinor") is expected_spinor
    )


def _observed_consistency_valid(
    *,
    consistency: object,
    valley: str,
    hsp: str,
    irrep: str,
    multiplicity: int,
    valley_mapping: Mapping[str, str],
    hsp_mapping: Mapping[str, str],
    irrep_mapping: Mapping[str, str],
    expected_spinor: bool,
) -> bool:
    if not isinstance(consistency, Mapping) or consistency.get("status") != (
        "validated"
    ):
        return False
    relation = consistency.get("reviewed_time_reversal_relation")
    evidence_irrep = (
        relation.get("evidence_irrep")
        if isinstance(relation, Mapping) else None
    )
    return (
        isinstance(consistency.get("evidence_sampled_kpoint"), str)
        and bool(consistency.get("evidence_sampled_kpoint"))
        and _reviewed_relation_valid(
            relation=relation,
            evidence_valley=consistency.get("evidence_valley"),
            target_valley=valley,
            evidence_hsp=consistency.get("evidence_source_hsp_label"),
            target_hsp=hsp,
            evidence_irrep=evidence_irrep,
            target_irrep=irrep,
            valley_mapping=valley_mapping,
            hsp_mapping=hsp_mapping,
            irrep_mapping=irrep_mapping,
        )
        and _source_candidate_provenance_valid(
            identity=consistency.get("source_candidate_identity"),
            provenance=consistency.get("source_candidate_provenance"),
            expected_valley=consistency.get("evidence_valley"),
            expected_hsp=consistency.get("evidence_source_hsp_label"),
            expected_sample=consistency.get("evidence_sampled_kpoint"),
            expected_irrep=evidence_irrep,
            expected_multiplicity=multiplicity,
            expected_spinor=expected_spinor,
        )
    )


def _reviewed_relation_valid(
    *,
    relation: object,
    evidence_valley: object,
    target_valley: object,
    evidence_hsp: object,
    target_hsp: object,
    evidence_irrep: object,
    target_irrep: object,
    valley_mapping: Mapping[str, str],
    hsp_mapping: Mapping[str, str],
    irrep_mapping: Mapping[str, str],
) -> bool:
    return (
        isinstance(relation, Mapping)
        and relation.get("evidence_valley") == evidence_valley
        and relation.get("target_valley") == target_valley
        and relation.get("evidence_source_hsp_label") == evidence_hsp
        and relation.get("target_source_hsp_label") == target_hsp
        and relation.get("evidence_irrep") == evidence_irrep
        and relation.get("target_irrep") == target_irrep
        and valley_mapping.get(evidence_valley) == target_valley
        and hsp_mapping.get(evidence_hsp) == target_hsp
        and irrep_mapping.get(evidence_irrep) == target_irrep
    )


def _nonempty_involutive_string_mapping(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(key, str)
            and bool(key)
            and isinstance(partner, str)
            and bool(partner)
            and value.get(partner) == key
            for key, partner in value.items()
        )
        and set(value) == set(value.values())
    )


def _injective_string_mapping(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and all(
            isinstance(key, str)
            and bool(key)
            and isinstance(sampled, str)
            and bool(sampled)
            for key, sampled in value.items()
        )
        and len(set(value.values())) == len(value)
    )


def _reviewed_hsp_involution(
    rows: object,
) -> tuple[dict[str, str] | None, set[str]]:
    if not isinstance(rows, list) or not rows:
        return None, set()
    mapping: dict[str, str] = {}
    representatives: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return None, set()
        members = row.get("members")
        representative = row.get("representative")
        self_mapped = row.get("self_mapped")
        if (
            not isinstance(members, list)
            or len(members) not in (1, 2)
            or any(not isinstance(item, str) or not item for item in members)
            or len(set(members)) != len(members)
            or representative not in members
            or self_mapped != (len(members) == 1)
            or set(members).intersection(mapping)
        ):
            return None, set()
        representatives.add(str(representative))
        if len(members) == 1:
            mapping[members[0]] = members[0]
        else:
            mapping[members[0]] = members[1]
            mapping[members[1]] = members[0]
    return mapping, representatives


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
