"""Fail-closed provenance validation for serialized unitary EBR vectors."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

from valleyscope.analysis.time_reversal_sewing import (
    validate_time_reversal_sewing_report,
)
from valleyscope.analysis.tr_irrep_completion import (
    _positive_int,
    validate_tr_irrep_completion_certificate,
)
from valleyscope.analysis.unitary_valley_sewing_completion import (
    validate_unitary_valley_sewing_certificate_context,
)
from valleyscope.io.wavefunction_convention import (
    canonical_identity,
    valid_sha256_identity,
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
        bool(bundle.get("time_reversal")),
    ))


def unitary_bundle_claims_valley_sewing_completion(
    bundle: Mapping[str, object],
) -> bool:
    construction = bundle.get("unitary_vector_construction")
    return bool(
        isinstance(construction, Mapping)
        and construction.get("kind")
        == "unitary_valley_sewing_completed_unitary_rows"
    )


def validate_unitary_bundle_provenance(
    bundle: Mapping[str, object],
    *,
    unitary_valley_sewing_validation_contexts: (
        Mapping[str, object] | None
    ) = None,
) -> bool:
    """Validate one explicitly dispatched unitary-vector construction."""
    if unitary_bundle_claims_time_reversal_completion(bundle):
        return validate_tr_completed_unitary_bundle(bundle)
    if unitary_bundle_claims_valley_sewing_completion(bundle):
        return validate_valley_sewing_completed_unitary_bundle(
            bundle,
            validation_contexts=(
                unitary_valley_sewing_validation_contexts
            ),
        )
    return validate_direct_unitary_bundle(bundle)


def validate_valley_sewing_completed_unitary_bundle(
    bundle: Mapping[str, object],
    *,
    validation_contexts: Mapping[str, object] | None,
) -> bool:
    """Rebuild all directed sewing certificates and the canonical vector."""
    if not isinstance(validation_contexts, Mapping):
        return False

    def certificate_valid(certificate):
        context = validation_contexts.get(certificate.get("certificate_identity"))
        return isinstance(context, Mapping) and (
            validate_unitary_valley_sewing_certificate_context(
                certificate, context
            )
        )

    return _validate_valley_sewing_completed_unitary_bundle(
        bundle, certificate_validator=certificate_valid
    )


def validate_serialized_valley_sewing_completed_unitary_bundle(
    bundle: Mapping[str, object],
) -> bool:
    """Validate serialized structure only; this is not raw-evidence trust."""
    return _validate_valley_sewing_completed_unitary_bundle(
        bundle,
        certificate_validator=_serialized_sewing_certificate_valid,
    )


def _validate_valley_sewing_completed_unitary_bundle(
    bundle: Mapping[str, object], *, certificate_validator
) -> bool:
    valley, expected = bundle.get("valley"), bundle.get("required_source_hsp_labels")
    completed = bundle.get("unitary_irrep_completion_records_by_hsp")
    direct = bundle.get("irrep_records_by_kpoint")
    irreps, source_to_sample = (
        bundle.get("irreps_by_kpoint"),
        bundle.get("source_hsp_to_sampled_kpoint"),
    )
    if (
        bundle.get("problem_kind") != "unitary_valley_reduced_ebr"
        or bundle.get("physical_object_kind") != "unitary_valley_projected_subspace"
        or bundle.get("workflow_path") != "unitary_valley_sewing_completion"
        or bundle.get("valley_orbit") not in ([], None)
        or bundle.get("time_reversal") not in ({}, None)
        or not isinstance(valley, str) or not valley
        or not isinstance(completed, Mapping) or not completed
        or not isinstance(irreps, Mapping)
        or not isinstance(direct, Mapping)
        or not isinstance(source_to_sample, Mapping)
        or not isinstance(expected, list)
        or set(expected) != set(irreps)
        or set(direct) != set(expected) - set(completed)
        or set(source_to_sample) != set(direct)
    ):
        return False
    declared_scopes = bundle.get("cprime_scope_metadata")
    if not isinstance(declared_scopes, Mapping):
        return False
    rebuilt, cprime_inventory, used_scopes = {}, {}, {}
    for hsp in expected:
        records, inferred = (
            (completed.get(hsp), True) if hsp in completed
            else (direct.get(hsp), False)
        )
        if not isinstance(records, list) or not records:
            return False
        counts = Counter()
        for record in records:
            if not isinstance(record, Mapping):
                return False
            if inferred:
                valid, irrep, multiplicity, cprime_rows = _validate_sewing_record(
                    record, hsp, valley, certificate_validator
                )
            else:
                irrep, multiplicity = (
                    record.get("matched_irrep"),
                    record.get("irrep_multiplicity"),
                )
                provenance = record.get("irrep_source_provenance")
                cprime = provenance.get("cprime") if isinstance(provenance, Mapping) else None
                valid = bool(
                    record.get("valley") == valley
                    and record.get("source_hsp_label") == hsp
                    and record.get("sampled_kpoint") == source_to_sample.get(hsp)
                    and record.get("workflow_path") in _PROJECTOR_WORKFLOWS
                    and record.get("readiness_level") == "trusted"
                    and isinstance(cprime, Mapping)
                )
                cprime_rows = [(
                    record.get("sampled_kpoint"), valley, cprime
                )]
            if not valid or not isinstance(irrep, str) or not _positive_int(multiplicity):
                return False
            counts[irrep] += multiplicity
            for sampled, evidence_valley, cprime in cprime_rows:
                key = _scope_key(
                    declared_scopes, sampled, evidence_valley
                )
                if key is None:
                    return False
                cprime_inventory[key] = _cprime_links(cprime)
                used_scopes[key] = declared_scopes[key]
        rebuilt[hsp] = counts
    return (
        dict(bundle.get("cprime_identity_by_kpoint", {})) == cprime_inventory
        and dict(declared_scopes) == used_scopes
        and all(
            isinstance(labels, list) and Counter(labels) == rebuilt.get(hsp)
            for hsp, labels in irreps.items()
        )
    )


def _validate_sewing_record(record, hsp, valley, certificate_validator):
    certificates = record.get("unitary_valley_sewing_certificates")
    if (
        record.get("completion_kind") != "inferred_by_unitary_valley_sewing"
        or "sampled_kpoint" in record
        or record.get("target_valley") != valley
        or record.get("target_source_hsp_label") != hsp
        or record.get("structural_status") != "validated"
        or record.get("readiness_status") != "trusted"
        or record.get("blockers") not in ([], None)
        or not isinstance(certificates, list) or not certificates
    ):
        return False, None, None, []
    first_source = certificates[0].get("source")
    if (
        not isinstance(first_source, Mapping)
        or record.get("evidence_sampled_kpoint")
        != first_source.get("sampled_kpoint")
        or record.get("evidence_valley") != first_source.get("valley")
        or record.get("evidence_source_hsp_label")
        != first_source.get("source_hsp_label")
    ):
        return False, None, None, []
    sources, cprime_rows = set(), []
    for certificate in certificates:
        source, target = (
            certificate.get("source"), certificate.get("target")
        ) if isinstance(certificate, Mapping) else (None, None)
        if (
            not isinstance(source, Mapping) or not isinstance(target, Mapping)
            or not certificate_validator(certificate)
            or target.get("valley") != valley
            or target.get("source_hsp_label") != hsp
            or target.get("irrep_multiplicities", {}).get(record.get("irrep"))
            != record.get("multiplicity")
            or not isinstance(source.get("cprime"), Mapping)
        ):
            return False, None, None, []
        sources.add((source.get("irrep"), source.get("multiplicity")))
        cprime_rows.append((
            source.get("sampled_kpoint"),
            source.get("valley"),
            source["cprime"],
        ))
    declared = record.get("evidence_irrep_vector")
    declared_set = {
        (item.get("irrep"), item.get("multiplicity"))
        for item in declared if isinstance(item, Mapping)
    } if isinstance(declared, list) else set()
    return (
        sources == declared_set,
        record.get("irrep"),
        record.get("multiplicity"),
        cprime_rows,
    )


def _serialized_sewing_certificate_valid(certificate):
    if not isinstance(certificate, Mapping):
        return False
    content = {
        key: value for key, value in certificate.items()
        if key not in {"status", "reason_codes", "certificate_identity"}
    }
    try:
        return bool(
            certificate.get("status") == "passed"
            and certificate.get("reason_codes") == []
            and canonical_identity(content) == certificate.get(
                "certificate_identity"
            )
        )
    except (TypeError, ValueError):
        return False


def _cprime_links(cprime):
    return {
        key: cprime.get(key)
        for key in (
            "spinor_source_basis_certificate_identity",
            "double_space_group_lift_certificate_identity",
            "scoped_representation_evidence_identity",
        )
    }


def _scope_key(scopes, sampled, valley):
    matches = [
        key for key, value in scopes.items()
        if isinstance(value, Mapping)
        and value.get("sampled_kpoint") == sampled
        and value.get("evidence_valley") == valley
    ]
    return matches[0] if len(matches) == 1 else None


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
    *,
    require_reviewed_table: bool = True,
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

    source_context = time_reversal.get(
        "reviewed_time_reversal_source_context"
    )
    from valleyscope.irreps.time_reversal_source import (
        validate_reviewed_time_reversal_source_context,
    )
    rederived_source = validate_reviewed_time_reversal_source_context(
        source_context,
        require_reviewed_table=require_reviewed_table,
    )
    if rederived_source.get("status") != "validated":
        return False
    expected_source_content = {
        "operation_inventory_identity": rederived_source.get(
            "operation_inventory_identity"
        ),
        "spin_convention": rederived_source.get("spin_convention"),
        "hsp_involution": rederived_source.get(
            "time_reversal_hsp_mapping"
        ),
        "irrep_pairing": rederived_source.get("irrep_partner_by_label"),
    }
    expected_source_identity = {
        **expected_source_content,
        "identity": canonical_identity(expected_source_content),
    }
    if (
        time_reversal.get("time_reversal_hsp_orbits")
        != rederived_source.get("time_reversal_hsp_orbits")
        or time_reversal.get("time_reversal_irrep_pairing")
        != rederived_source.get("irrep_partner_by_label")
        or time_reversal.get("reviewed_time_reversal_source_identity")
        != expected_source_identity
        or (
            require_reviewed_table
            and not _source_context_matches_bundle_certificate(
                source_context=source_context,
                certificate_identity=bundle.get("certificate_identity"),
            )
        )
        or (
            time_reversal.get("mapping_type") == "exchanged"
            and require_reviewed_table
            and not _completion_context_matches_reviewed_source(
                records_by_hsp=records_by_hsp,
                source_context=source_context,
            )
        )
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
        reviewed_source_identity=time_reversal.get(
            "reviewed_time_reversal_source_identity", {}
        ),
        reviewed_source_context=source_context,
        independent_hsps=set(independent_hsps),
        expected_spinor=spinor,
    )
    if not records_valid:
        return False
    if not _tr_completed_cprime_identity_valid(
        records_by_hsp=records_by_hsp,
        expected_hsps=expected_hsps,
        inventory=bundle.get("cprime_identity_by_kpoint"),
    ) or not _tr_completion_scope_metadata_valid(
        valley=valley,
        records_by_hsp=records_by_hsp,
        expected_hsps=expected_hsps,
        declared=bundle.get("cprime_scope_metadata"),
    ):
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


def _completion_context_matches_reviewed_source(
    *,
    records_by_hsp: Mapping[str, object],
    source_context: object,
) -> bool:
    if not isinstance(source_context, Mapping):
        return False
    source_table = source_context.get("source_table_identity")
    setting = source_context.get("standard_setting_certificate")
    lift_record = source_context.get("parent_affine_lift_record")
    if (
        not isinstance(source_table, Mapping)
        or not isinstance(setting, Mapping)
        or not isinstance(lift_record, Mapping)
    ):
        return False
    from valleyscope.irreps.time_reversal_source import (
        normalize_time_reversal_standard_setting_context,
    )
    expected_table = {
        "matching_strategy": "bilbao_restricted_character",
        "subspace_space_group_number": source_table.get(
            "space_group_number"
        ),
        "subspace_space_group_symbol": source_table.get(
            "space_group_symbol"
        ),
        "source_table_sg_number": source_table.get("space_group_number"),
        "source_table_name": source_table.get("source_table_name"),
        "source_table_spinor": source_table.get("spinor"),
    }
    inferred_count = 0
    for records in records_by_hsp.values():
        if not isinstance(records, list):
            return False
        for record in records:
            if not isinstance(record, Mapping):
                return False
            if record.get("completion_kind") != "inferred_by_time_reversal":
                continue
            inferred_count += 1
            certificate = record.get("tr_irrep_completion_certificate")
            reviewed = (
                certificate.get("reviewed_time_reversal")
                if isinstance(certificate, Mapping) else None
            )
            provenance = record.get("source_candidate_provenance")
            source_irrep = (
                provenance.get("irrep_source_provenance")
                if isinstance(provenance, Mapping) else None
            )
            cprime = (
                source_irrep.get("cprime")
                if isinstance(source_irrep, Mapping) else None
            )
            setting_mapping = (
                source_irrep.get("standard_setting_hsp_mapping")
                if isinstance(source_irrep, Mapping) else None
            )
            source_setting = (
                setting_mapping.get("standard_setting_certificate")
                if isinstance(setting_mapping, Mapping) else None
            )
            try:
                source_setting_identity = (
                    canonical_identity(source_setting)
                    if isinstance(source_setting, Mapping) else None
                )
            except (TypeError, ValueError):
                return False
            if (
                not isinstance(reviewed, Mapping)
                or not isinstance(source_setting, Mapping)
                or not isinstance(cprime, Mapping)
                or reviewed.get("source_table_identity") != expected_table
                or lift_record.get("certificate_identity")
                != cprime.get(
                    "double_space_group_lift_certificate_identity"
                )
                or certificate.get(
                    "standard_setting_certificate_identity"
                ) != source_setting_identity
                or normalize_time_reversal_standard_setting_context(
                    source_setting
                ) != setting
            ):
                return False
    return inferred_count > 0


def _source_context_matches_bundle_certificate(
    *,
    source_context: object,
    certificate_identity: object,
) -> bool:
    """Bind serialized TR source setting to the independent bundle setting."""
    if not isinstance(source_context, Mapping) or not isinstance(
        certificate_identity, Mapping
    ):
        return False
    setting = source_context.get("standard_setting_certificate")
    if not isinstance(setting, Mapping):
        return False
    field_map = {
        "subspace_sg_number": "sg_number",
        "subspace_sg_symbol": "sg_symbol",
        "hall_number": "hall_number",
        "hall_symbol": "hall_symbol",
        "centering_type": "centering_type",
        "primitive_conventional_relation": (
            "primitive_conventional_relation"
        ),
        "transform_provenance": "transform_provenance",
        "validation_status": "validation_status",
        "operation_mapping_status": "operation_mapping_status",
        "translation_validation_status": "affine_validation_status",
        "matched_affine_operations": "affine_matched_operations",
        "total_parent_operations": "affine_total_operations",
        "mismatched_translation_count": "affine_mismatch_count",
        "missing_affine_ingredients": "affine_missing_ingredients",
        "standard_setting_operation_count": (
            "affine_standard_setting_op_count"
        ),
        "affine_operation_map": "affine_operation_map",
        "parent_basis_operation_ids": "affine_required_operation_ids",
        "required_operation_id_count": "affine_required_op_count",
        "unmatched_parent_operations": (
            "affine_unmatched_parent_operations"
        ),
        "unused_standard_operation_indices": (
            "affine_unused_standard_operation_indices"
        ),
        "operation_closure_validated": "operation_closure_validated",
        "canonical_setting_status": "canonical_setting_status",
        "canonical_setting_source": "canonical_setting_source",
        "canonical_hall_numbers": "canonical_hall_numbers",
        "canonical_candidate_hall_numbers": (
            "canonical_candidate_hall_numbers"
        ),
        "centering_coset_count": "centering_coset_count",
        "primitive_conventional_index": "primitive_conventional_index",
        "expanded_parent_operation_count": (
            "expanded_parent_operation_count"
        ),
        "matched_expanded_operations": "matched_expanded_operations",
        "centered_affine_operation_map": "centered_affine_operation_map",
        "unmatched_centered_operation_pairs": (
            "affine_unmatched_centered_operation_pairs"
        ),
        "standard_operation_closure_validated": (
            "standard_operation_closure_validated"
        ),
    }
    if any(
        setting.get(source_key) != certificate_identity.get(bundle_key)
        for source_key, bundle_key in field_map.items()
    ):
        return False
    if (
        setting.get("parent_to_standard_direct_transform")
        != certificate_identity.get("normalized_direct_transform")
        or setting.get("origin_shift_fractional")
        != certificate_identity.get("normalized_origin_shift")
        or setting.get("centering_vectors")
        != certificate_identity.get("normalized_centering_vectors")
    ):
        return False
    hall_number = setting.get("hall_number")
    hall_symbol = setting.get("hall_symbol")
    centering_type = setting.get("centering_type")
    validation_status = setting.get("validation_status")
    return bool(
        certificate_identity.get("distinct_setting_identities") == 1
        and certificate_identity.get("any_unresolved") is False
        and certificate_identity.get("hall_numbers") == [hall_number]
        and certificate_identity.get("hall_symbols") == [hall_symbol]
        and certificate_identity.get("centering_types") == [centering_type]
        and certificate_identity.get(
            "certificate_validation_statuses"
        ) == [validation_status]
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
    reviewed_source_identity: Mapping[str, object],
    reviewed_source_context: Mapping[str, object] | None,
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
                    or (
                        valley_mapping.get(valley) != valley
                        and not validate_tr_irrep_completion_certificate(
                            record.get(
                                "tr_irrep_completion_certificate"
                            ),
                            completion_record=record,
                            valley_mapping=valley_mapping,
                            hsp_mapping=hsp_mapping,
                            irrep_pairing=irrep_mapping,
                            reviewed_source_identity=(
                                reviewed_source_identity
                            ),
                            reviewed_source_context=(
                                reviewed_source_context
                            ),
                        )
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


def _tr_completed_cprime_identity_valid(
    *,
    records_by_hsp: Mapping[str, object],
    expected_hsps: list[str],
    inventory: object,
) -> bool:
    """Require a complete C-prime inventory bound to every source HSP.

    Every record under a source HSP must carry the exact inventory identity
    for that HSP: observed records and self-mapped inferred records carry it
    in the candidate provenance, exact inferred records in the validated
    completion certificate.
    """
    if (
        not isinstance(inventory, Mapping)
        or set(inventory) != set(expected_hsps)
    ):
        return False
    for hsp in expected_hsps:
        identity = inventory.get(hsp)
        if not _valid_cprime_identity(identity):
            return False
        records = records_by_hsp.get(hsp)
        if not isinstance(records, list) or not records:
            return False
        for record in records:
            if not isinstance(record, Mapping):
                return False
            if _tr_record_cprime(record) != identity:
                return False
    return True


def _tr_record_cprime(record: Mapping[str, object]) -> object:
    """The C-prime identity bound by one completion record."""
    kind = record.get("completion_kind")
    if kind == "inferred_by_time_reversal":
        certificate = record.get("tr_irrep_completion_certificate")
        observed = (
            certificate.get("observed_source")
            if isinstance(certificate, Mapping)
            else None
        )
        if isinstance(observed, Mapping):
            return observed.get("local_cprime_identity")
    provenance = record.get("source_candidate_provenance")
    source_irrep = (
        provenance.get("irrep_source_provenance")
        if isinstance(provenance, Mapping)
        else None
    )
    return (
        source_irrep.get("cprime")
        if isinstance(source_irrep, Mapping)
        else None
    )


def _valid_cprime_identity(value: object) -> bool:
    required_keys = {
        "spinor_source_basis_certificate_identity",
        "double_space_group_lift_certificate_identity",
        "scoped_representation_evidence_identity",
    }
    return (
        isinstance(value, Mapping)
        and set(value) == required_keys
        and all(
            valid_sha256_identity(value.get(key))
            for key in required_keys
        )
    )


def _tr_completion_scope_metadata_valid(
    *,
    valley: str,
    records_by_hsp: Mapping[str, object],
    expected_hsps: list[str],
    declared: object,
) -> bool:
    """Recompute the expected C-prime scope metadata from the validated
    records and require exact equality with the serialized metadata."""
    if not isinstance(declared, Mapping):
        return False
    rebuilt: dict[str, dict[str, str]] = {}
    for hsp in expected_hsps:
        records = records_by_hsp.get(hsp)
        if not isinstance(records, list) or not records:
            return False
        scopes: list[tuple[str, str]] = []
        for record in records:
            if not isinstance(record, Mapping):
                return False
            scope = _tr_completion_record_scope(record, valley, hsp)
            if scope is None:
                return False
            scopes.append(scope)
        if any(scope != scopes[0] for scope in scopes[1:]):
            return False
        sampled, evidence_valley = scopes[0]
        rebuilt[hsp] = {
            "sampled_kpoint": sampled,
            "evidence_valley": evidence_valley,
        }
    return rebuilt == dict(declared)


def _tr_completion_record_scope(
    record: Mapping[str, object],
    valley: str,
    hsp: str,
) -> tuple[str, str] | None:
    """Exact evidence scope of one TR completion record, or None."""
    kind = record.get("completion_kind")
    if kind == "observed_at_sampled_kpoint":
        sampled = record.get("sampled_kpoint")
        if (
            record.get("target_source_hsp_label") == hsp
            and record.get("target_valley") == valley
            and record.get("evidence_sampled_kpoint") == sampled
            and record.get("evidence_valley") == valley
            and isinstance(sampled, str)
            and sampled
        ):
            return sampled, valley
        return None
    if kind == "inferred_by_time_reversal":
        certificate = record.get("tr_irrep_completion_certificate")
        observed = (
            certificate.get("observed_source")
            if isinstance(certificate, Mapping)
            else None
        )
        if isinstance(observed, Mapping):
            sampled = observed.get("sampled_kpoint")
            evidence_valley = observed.get("valley")
            if (
                record.get("evidence_sampled_kpoint") != sampled
                or record.get("evidence_valley") != evidence_valley
            ):
                return None
        else:
            # Self-mapped orbits attach no certificate; the record-level
            # evidence fields are the trusted evidence scope.
            sampled = record.get("evidence_sampled_kpoint")
            evidence_valley = record.get("evidence_valley")
        if (
            record.get("target_source_hsp_label") == hsp
            and isinstance(sampled, str)
            and sampled
            and isinstance(evidence_valley, str)
            and evidence_valley
        ):
            return sampled, evidence_valley
        return None
    return None


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
