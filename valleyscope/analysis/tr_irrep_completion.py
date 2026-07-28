"""Exact algebraic TR irrep completion for unobserved HSP rows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from valleyscope.analysis.scoped_representation_evidence import (
    validate_scoped_representation_evidence_record,
)
from valleyscope.io.wavefunction_convention import (
    V1_PROFILE_ASSUMPTIONS,
    V1_PROFILE_IDENTITY,
    canonical_identity,
    valid_sha256_identity,
)


TR_IRREP_COMPLETION_CERTIFICATE_SCHEMA_VERSION = "1.0.0"

_CPRIME_IDENTITY_KEYS = frozenset({
    "spinor_source_basis_certificate_identity",
    "double_space_group_lift_certificate_identity",
    "scoped_representation_evidence_identity",
})


def attach_tr_irrep_completion_certificates(
    *,
    time_reversal_orbit_report: dict[str, object],
    cprime_validation_context: dict[str, object],
) -> dict[str, object]:
    """Attach exact certificates only to algebraically inferred rows.

    Observed rows keep their local C-prime identities.  The completion
    certificate binds the trusted observed row to reviewed valley, HSP, and
    irreptables irrep involutions.  It does not construct or claim numerical
    antiunitary sewing evidence.
    """
    report = deepcopy(time_reversal_orbit_report)
    raw_orbits = report.get("valley_orbits")
    valley_mapping = report.get("time_reversal_valley_mapping")
    if not isinstance(raw_orbits, list) or not _involution(valley_mapping):
        return report

    for orbit in raw_orbits:
        if not isinstance(orbit, dict):
            continue
        if orbit.get("mapping_type") != "exchanged":
            orbit["tr_irrep_completion_status"] = "not_applicable"
            continue
        blockers = [
            value
            for value in orbit.get("blockers", [])
            if isinstance(value, str) and value
        ]
        records_by_valley = orbit.get(
            "unitary_valley_irrep_completion_records"
        )
        hsp_mapping = _hsp_mapping(
            orbit.get("time_reversal_hsp_orbits")
        )
        irrep_pairing = orbit.get("time_reversal_irrep_pairing")
        reviewed_source_identity = orbit.get(
            "reviewed_time_reversal_source_identity"
        )
        if (
            not isinstance(records_by_valley, Mapping)
            or not _involution(hsp_mapping)
            or not _involution(irrep_pairing)
            or not _valid_reviewed_source_identity(
                reviewed_source_identity
            )
        ):
            blockers.append("tr_irrep_completion_inputs_malformed")
            _block_orbit(orbit, blockers)
            continue

        inferred_count = 0
        inferred_row_count = 0
        for target_valley, raw_by_hsp in records_by_valley.items():
            if not isinstance(target_valley, str) or not isinstance(
                raw_by_hsp, Mapping
            ):
                blockers.append("tr_irrep_completion_records_malformed")
                continue
            for target_hsp, raw_records in raw_by_hsp.items():
                if not isinstance(target_hsp, str) or not isinstance(
                    raw_records, list
                ):
                    blockers.append(
                        "tr_irrep_completion_records_malformed:"
                        f"{target_valley}:{target_hsp}"
                    )
                    continue
                for record in raw_records:
                    if not isinstance(record, dict):
                        blockers.append(
                            "tr_irrep_completion_record_malformed:"
                            f"{target_valley}:{target_hsp}"
                        )
                        continue
                    kind = record.get("completion_kind")
                    if kind == "observed_at_sampled_kpoint":
                        record.pop(
                            "tr_irrep_completion_certificate", None
                        )
                        continue
                    if kind != "inferred_by_time_reversal":
                        blockers.append(
                            "tr_irrep_completion_record_kind_invalid:"
                            f"{target_valley}:{target_hsp}"
                        )
                        _block_record(
                            record,
                            "tr_irrep_completion_record_kind_invalid",
                        )
                        continue
                    inferred_row_count += 1
                    certificate, certificate_blockers = (
                        _build_completion_certificate(
                            record=record,
                            target_valley=target_valley,
                            target_hsp=target_hsp,
                            valley_mapping=valley_mapping,
                            hsp_mapping=hsp_mapping,
                            irrep_pairing=irrep_pairing,
                            reviewed_source_identity=(
                                reviewed_source_identity
                            ),
                            cprime_validation_context=(
                                cprime_validation_context
                            ),
                        )
                    )
                    if certificate is None:
                        for blocker in certificate_blockers:
                            _block_record(record, blocker)
                            blockers.append(
                                f"{blocker}:{target_valley}:{target_hsp}"
                            )
                        continue
                    record["tr_irrep_completion_certificate"] = certificate
                    inferred_count += 1

        if inferred_row_count and inferred_count == 0:
            blockers.append("tr_irrep_completion_certificate_missing")
        if blockers:
            _block_orbit(orbit, blockers)
            continue
        orbit["tr_irrep_completion_status"] = (
            "passed" if inferred_row_count else "not_applicable"
        )
        orbit["tr_irrep_completion_certificate_count"] = inferred_count

    report["blockers"] = _deduplicate(
        blocker
        for orbit in raw_orbits
        if isinstance(orbit, Mapping)
        for blocker in orbit.get("blockers", [])
        if isinstance(blocker, str)
    )
    report["status"] = (
        "validated"
        if raw_orbits
        and all(
            isinstance(orbit, Mapping)
            and orbit.get("status") == "validated"
            for orbit in raw_orbits
        )
        else "blocked"
    )
    return report


def validate_tr_irrep_completion_certificate(
    certificate: object,
    *,
    completion_record: Mapping[str, object],
    valley_mapping: Mapping[str, str],
    hsp_mapping: Mapping[str, str],
    irrep_pairing: Mapping[str, str],
    reviewed_source_identity: Mapping[str, object],
    cprime_validation_context: Mapping[str, object] | None = None,
) -> bool:
    """Recompute a serialized exact completion certificate fail-closed."""
    if (
        not isinstance(certificate, Mapping)
        or not _valid_reviewed_source_identity(reviewed_source_identity)
    ):
        return False
    content = {
        key: deepcopy(value)
        for key, value in certificate.items()
        if key != "certificate_identity"
    }
    try:
        expected_identity = canonical_identity(content)
    except (TypeError, ValueError):
        return False
    if (
        certificate.get("schema_version")
        != TR_IRREP_COMPLETION_CERTIFICATE_SCHEMA_VERSION
        or certificate.get("certificate_kind")
        != "exact_tr_irrep_completion"
        or certificate.get("status") != "passed"
        or certificate.get("certificate_identity") != expected_identity
        or not valid_sha256_identity(expected_identity)
    ):
        return False

    observed = certificate.get("observed_source")
    inferred = certificate.get("inferred_target")
    reviewed = certificate.get("reviewed_time_reversal")
    profile = certificate.get("supported_parent_profile")
    provenance = completion_record.get("source_candidate_provenance")
    source_irrep = (
        provenance.get("irrep_source_provenance")
        if isinstance(provenance, Mapping)
        else None
    )
    cprime = (
        source_irrep.get("cprime")
        if isinstance(source_irrep, Mapping)
        else None
    )
    setting_mapping = (
        source_irrep.get("standard_setting_hsp_mapping")
        if isinstance(source_irrep, Mapping)
        else None
    )
    setting_certificate = (
        setting_mapping.get("standard_setting_certificate")
        if isinstance(setting_mapping, Mapping)
        else None
    )
    if not all(
        isinstance(value, Mapping)
        for value in (
            observed,
            inferred,
            reviewed,
            profile,
            source_irrep,
            cprime,
            setting_certificate,
        )
    ):
        return False
    try:
        setting_identity = canonical_identity(setting_certificate)
    except (TypeError, ValueError):
        return False
    relation = completion_record.get("reviewed_time_reversal_relation")
    multiplicity = completion_record.get("multiplicity")
    source_identity = completion_record.get("source_candidate_identity")
    return bool(
        certificate.get("observed_source") == {
            "candidate_identity": source_identity,
            "local_cprime_identity": cprime,
            "valley": completion_record.get("evidence_valley"),
            "source_hsp_label": completion_record.get(
                "evidence_source_hsp_label"
            ),
            "sampled_kpoint": completion_record.get(
                "evidence_sampled_kpoint"
            ),
            "irrep": (
                relation.get("evidence_irrep")
                if isinstance(relation, Mapping)
                else None
            ),
            "multiplicity": multiplicity,
        }
        and certificate.get("inferred_target") == {
            "valley": completion_record.get("target_valley"),
            "source_hsp_label": completion_record.get(
                "target_source_hsp_label"
            ),
            "irrep": completion_record.get("irrep"),
            "multiplicity": multiplicity,
        }
        and reviewed.get("valley_involution") == dict(valley_mapping)
        and reviewed.get("hsp_involution") == dict(hsp_mapping)
        and reviewed.get("irrep_pairing") == dict(irrep_pairing)
        and reviewed.get("source_model_identity")
        == dict(reviewed_source_identity)
        and _mapping_is_reviewed_subset(
            hsp_mapping,
            reviewed_source_identity.get("hsp_involution"),
        )
        and _mapping_is_reviewed_subset(
            irrep_pairing,
            reviewed_source_identity.get("irrep_pairing"),
        )
        and reviewed.get("source_table_identity")
        == _source_table_identity(source_irrep)
        and certificate.get("standard_setting_certificate_identity")
        == setting_identity
        and profile.get("profile_identity") == V1_PROFILE_IDENTITY
        and profile.get("profile_assumptions") == V1_PROFILE_ASSUMPTIONS
        and profile.get("spinor_source_basis_certificate_identity")
        == cprime.get("spinor_source_basis_certificate_identity")
        and relation == {
            "evidence_valley": observed.get("valley"),
            "target_valley": inferred.get("valley"),
            "evidence_source_hsp_label": observed.get(
                "source_hsp_label"
            ),
            "target_source_hsp_label": inferred.get(
                "source_hsp_label"
            ),
            "evidence_irrep": observed.get("irrep"),
            "target_irrep": inferred.get("irrep"),
        }
        and valley_mapping.get(observed.get("valley"))
        == inferred.get("valley")
        and hsp_mapping.get(observed.get("source_hsp_label"))
        == inferred.get("source_hsp_label")
        and irrep_pairing.get(observed.get("irrep"))
        == inferred.get("irrep")
        and (
            cprime_validation_context is None
            or _producer_context_matches_certificate(
                certificate=certificate,
                completion_record=completion_record,
                cprime_validation_context=cprime_validation_context,
            )
        )
    )


def _build_completion_certificate(
    *,
    record: Mapping[str, object],
    target_valley: str,
    target_hsp: str,
    valley_mapping: Mapping[str, str],
    hsp_mapping: Mapping[str, str],
    irrep_pairing: Mapping[str, str],
    reviewed_source_identity: Mapping[str, object],
    cprime_validation_context: Mapping[str, object],
) -> tuple[dict[str, object] | None, list[str]]:
    blockers: list[str] = []
    relation = record.get("reviewed_time_reversal_relation")
    provenance = record.get("source_candidate_provenance")
    source_irrep = (
        provenance.get("irrep_source_provenance")
        if isinstance(provenance, Mapping)
        else None
    )
    cprime = (
        source_irrep.get("cprime")
        if isinstance(source_irrep, Mapping)
        else None
    )
    setting_mapping = (
        source_irrep.get("standard_setting_hsp_mapping")
        if isinstance(source_irrep, Mapping)
        else None
    )
    setting_certificate = (
        setting_mapping.get("standard_setting_certificate")
        if isinstance(setting_mapping, Mapping)
        else None
    )
    evidence_valley = record.get("evidence_valley")
    evidence_hsp = record.get("evidence_source_hsp_label")
    evidence_sample = record.get("evidence_sampled_kpoint")
    evidence_irrep = (
        relation.get("evidence_irrep")
        if isinstance(relation, Mapping)
        else None
    )
    target_irrep = record.get("irrep")
    multiplicity = record.get("multiplicity")
    if (
        not isinstance(relation, Mapping)
        or not isinstance(source_irrep, Mapping)
        or not _valid_cprime_identity(cprime)
        or not isinstance(setting_certificate, Mapping)
        or not setting_certificate
        or not isinstance(evidence_valley, str)
        or not evidence_valley
        or not isinstance(evidence_hsp, str)
        or not evidence_hsp
        or not isinstance(evidence_sample, str)
        or not evidence_sample
        or not isinstance(evidence_irrep, str)
        or not evidence_irrep
        or not isinstance(target_irrep, str)
        or not target_irrep
        or not _positive_int(multiplicity)
    ):
        return None, ["tr_irrep_completion_source_evidence_malformed"]
    if (
        record.get("target_valley") != target_valley
        or record.get("target_source_hsp_label") != target_hsp
        or valley_mapping.get(evidence_valley) != target_valley
        or hsp_mapping.get(evidence_hsp) != target_hsp
        or irrep_pairing.get(evidence_irrep) != target_irrep
        or relation != {
            "evidence_valley": evidence_valley,
            "target_valley": target_valley,
            "evidence_source_hsp_label": evidence_hsp,
            "target_source_hsp_label": target_hsp,
            "evidence_irrep": evidence_irrep,
            "target_irrep": target_irrep,
        }
    ):
        blockers.append("tr_irrep_completion_involution_mismatch")

    (
        source_basis,
        producer_context_identity,
        context_blockers,
    ) = _validated_producer_context(
        completion_record=record,
        cprime=cprime,
        standard_setting_certificate=setting_certificate,
        cprime_validation_context=cprime_validation_context,
    )
    blockers.extend(context_blockers)
    if (
        not isinstance(source_basis, Mapping)
        or source_basis.get("status") != "passed"
        or source_basis.get("profile_identity") != V1_PROFILE_IDENTITY
        or source_basis.get("certificate_identity")
        != cprime["spinor_source_basis_certificate_identity"]
        or not isinstance(source_basis.get("profile_assumptions"), Mapping)
        or source_basis["profile_assumptions"].get("time_reversal") is not True
    ):
        blockers.append("tr_irrep_completion_supported_profile_invalid")

    source_table_identity = _source_table_identity(source_irrep)
    if source_table_identity is None:
        blockers.append("tr_irrep_completion_source_table_identity_missing")
    try:
        setting_identity = canonical_identity(setting_certificate)
    except (TypeError, ValueError):
        setting_identity = None
        blockers.append(
            "tr_irrep_completion_standard_setting_identity_invalid"
        )
    if blockers:
        return None, _deduplicate(blockers)

    content: dict[str, object] = {
        "schema_version": (
            TR_IRREP_COMPLETION_CERTIFICATE_SCHEMA_VERSION
        ),
        "certificate_kind": "exact_tr_irrep_completion",
        "status": "passed",
        "observed_source": {
            "candidate_identity": deepcopy(
                record.get("source_candidate_identity")
            ),
            "local_cprime_identity": deepcopy(cprime),
            "valley": evidence_valley,
            "source_hsp_label": evidence_hsp,
            "sampled_kpoint": evidence_sample,
            "irrep": evidence_irrep,
            "multiplicity": multiplicity,
        },
        "inferred_target": {
            "valley": target_valley,
            "source_hsp_label": target_hsp,
            "irrep": target_irrep,
            "multiplicity": multiplicity,
        },
        "reviewed_time_reversal": {
            "valley_involution": dict(valley_mapping),
            "hsp_involution": dict(hsp_mapping),
            "irrep_pairing": dict(irrep_pairing),
            "source_model_identity": dict(reviewed_source_identity),
            "source_table_identity": source_table_identity,
        },
        "standard_setting_certificate_identity": setting_identity,
        "producer_context_identity": producer_context_identity,
        "supported_parent_profile": {
            "profile_identity": source_basis["profile_identity"],
            "profile_assumptions": deepcopy(
                source_basis["profile_assumptions"]
            ),
            "spinor_source_basis_certificate_identity": source_basis[
                "certificate_identity"
            ],
        },
    }
    certificate = dict(content)
    certificate["certificate_identity"] = canonical_identity(content)
    return certificate, []


def _source_table_identity(
    source_irrep: Mapping[str, object],
) -> dict[str, object] | None:
    keys = (
        "matching_strategy",
        "subspace_space_group_number",
        "subspace_space_group_symbol",
        "source_table_sg_number",
        "source_table_name",
        "source_table_spinor",
    )
    identity = {key: source_irrep.get(key) for key in keys}
    if (
        identity["matching_strategy"] != "bilbao_restricted_character"
        or not _positive_int(identity["subspace_space_group_number"])
        or not isinstance(identity["subspace_space_group_symbol"], str)
        or not identity["subspace_space_group_symbol"]
        or not _positive_int(identity["source_table_sg_number"])
        or not isinstance(identity["source_table_name"], str)
        or not identity["source_table_name"]
        or identity["source_table_spinor"] is not True
    ):
        return None
    return identity


def _validated_producer_context(
    *,
    completion_record: Mapping[str, object],
    cprime: Mapping[str, object],
    standard_setting_certificate: Mapping[str, object],
    cprime_validation_context: Mapping[str, object],
) -> tuple[
    Mapping[str, object] | None,
    str | None,
    list[str],
]:
    blockers: list[str] = []
    context = cprime_validation_context.get(
        str(cprime["scoped_representation_evidence_identity"])
    )
    if not isinstance(context, Mapping):
        return None, None, [
            "tr_irrep_completion_local_cprime_context_missing"
        ]
    scoped_record = context.get("record")
    raw_inputs = context.get("raw_inputs")
    context_setting_certificate = context.get(
        "standard_setting_certificate"
    )
    if (
        not isinstance(scoped_record, Mapping)
        or not isinstance(raw_inputs, Mapping)
        or not isinstance(context_setting_certificate, Mapping)
    ):
        return None, None, [
            "tr_irrep_completion_local_cprime_context_malformed"
        ]

    validation = validate_scoped_representation_evidence_record(
        scoped_record,
        **raw_inputs,
    )
    scope = scoped_record.get("scope")
    source_basis = raw_inputs.get("source_basis_record")
    evidence_valley = completion_record.get("evidence_valley")
    evidence_sample = completion_record.get("evidence_sampled_kpoint")
    if (
        validation.status != "passed"
        or scoped_record.get("status") != "passed"
        or scoped_record.get("evidence_identity")
        != cprime["scoped_representation_evidence_identity"]
        or scoped_record.get("source_basis_certificate_identity")
        != cprime["spinor_source_basis_certificate_identity"]
        or scoped_record.get(
            "double_space_group_lift_certificate_identity"
        )
        != cprime["double_space_group_lift_certificate_identity"]
        or not isinstance(scope, Mapping)
        or scope.get("scope_kind") != "local_irrep"
        or scope.get("source_valleys") != [evidence_valley]
        or scope.get("kpoint_label") != evidence_sample
    ):
        blockers.append("tr_irrep_completion_local_cprime_invalid")
    if dict(context_setting_certificate) != dict(
        standard_setting_certificate
    ):
        blockers.append(
            "tr_irrep_completion_standard_setting_context_mismatch"
        )

    lift_inputs = raw_inputs.get("lift_validation_inputs")
    source_table_evidence = (
        lift_inputs.get("source_table_identity")
        if isinstance(lift_inputs, Mapping)
        else None
    )
    standard_setting_evidence = (
        lift_inputs.get("standard_setting_identity")
        if isinstance(lift_inputs, Mapping)
        else None
    )
    provenance = completion_record.get("source_candidate_provenance")
    source_irrep = (
        provenance.get("irrep_source_provenance")
        if isinstance(provenance, Mapping)
        else None
    )
    if (
        not isinstance(source_irrep, Mapping)
        or not isinstance(source_table_evidence, Mapping)
        or not isinstance(standard_setting_evidence, Mapping)
        or source_irrep.get("source_table_sg_number")
        != source_table_evidence.get("space_group_number")
        or source_irrep.get("source_table_spinor")
        is not source_table_evidence.get("spinor")
    ):
        blockers.append(
            "tr_irrep_completion_source_table_context_mismatch"
        )
    if blockers:
        return (
            source_basis if isinstance(source_basis, Mapping) else None,
            None,
            _deduplicate(blockers),
        )
    try:
        producer_context_identity = canonical_identity({
            "scoped_representation_evidence_identity": scoped_record[
                "evidence_identity"
            ],
            "source_table_evidence_identity": canonical_identity(
                source_table_evidence
            ),
            "standard_setting_evidence_identity": canonical_identity(
                standard_setting_evidence
            ),
            "standard_setting_certificate_identity": canonical_identity(
                context_setting_certificate
            ),
        })
    except (KeyError, TypeError, ValueError):
        return None, None, [
            "tr_irrep_completion_producer_context_identity_invalid"
        ]
    return (
        source_basis if isinstance(source_basis, Mapping) else None,
        producer_context_identity,
        [],
    )


def _producer_context_matches_certificate(
    *,
    certificate: Mapping[str, object],
    completion_record: Mapping[str, object],
    cprime_validation_context: Mapping[str, object],
) -> bool:
    observed = certificate.get("observed_source")
    local_cprime = (
        observed.get("local_cprime_identity")
        if isinstance(observed, Mapping)
        else None
    )
    provenance = completion_record.get("source_candidate_provenance")
    source_irrep = (
        provenance.get("irrep_source_provenance")
        if isinstance(provenance, Mapping)
        else None
    )
    setting_mapping = (
        source_irrep.get("standard_setting_hsp_mapping")
        if isinstance(source_irrep, Mapping)
        else None
    )
    setting_certificate = (
        setting_mapping.get("standard_setting_certificate")
        if isinstance(setting_mapping, Mapping)
        else None
    )
    if (
        not isinstance(local_cprime, Mapping)
        or not isinstance(setting_certificate, Mapping)
    ):
        return False
    _, producer_identity, blockers = _validated_producer_context(
        completion_record=completion_record,
        cprime=local_cprime,
        standard_setting_certificate=setting_certificate,
        cprime_validation_context=cprime_validation_context,
    )
    return (
        not blockers
        and certificate.get("producer_context_identity")
        == producer_identity
    )


def _valid_reviewed_source_identity(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "operation_inventory_identity",
        "spin_convention",
        "hsp_involution",
        "irrep_pairing",
        "identity",
    }:
        return False
    content = {
        "operation_inventory_identity": value.get(
            "operation_inventory_identity"
        ),
        "spin_convention": value.get("spin_convention"),
        "hsp_involution": value.get("hsp_involution"),
        "irrep_pairing": value.get("irrep_pairing"),
    }
    return bool(
        _valid_operation_inventory_identity(
            content["operation_inventory_identity"]
        )
        and isinstance(content["spin_convention"], str)
        and content["spin_convention"]
        and _involution(content["hsp_involution"])
        and _involution(content["irrep_pairing"])
        and value.get("identity") == canonical_identity(content)
    )


def _valid_operation_inventory_identity(value: object) -> bool:
    return bool(
        valid_sha256_identity(value)
        or (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )
    )


def _mapping_is_reviewed_subset(
    scoped: Mapping[str, str],
    reviewed: object,
) -> bool:
    return bool(
        isinstance(reviewed, Mapping)
        and scoped
        and all(reviewed.get(key) == value for key, value in scoped.items())
    )


def _valid_cprime_identity(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == _CPRIME_IDENTITY_KEYS
        and all(valid_sha256_identity(value.get(key)) for key in value)
    )


def _hsp_mapping(raw_orbits: object) -> dict[str, str]:
    if not isinstance(raw_orbits, list) or not raw_orbits:
        return {}
    mapping: dict[str, str] = {}
    for raw in raw_orbits:
        if not isinstance(raw, Mapping):
            return {}
        members = raw.get("members")
        if (
            not isinstance(members, list)
            or len(members) not in (1, 2)
            or not all(isinstance(value, str) and value for value in members)
            or any(value in mapping for value in members)
        ):
            return {}
        if len(members) == 1:
            mapping[members[0]] = members[0]
        else:
            mapping[members[0]] = members[1]
            mapping[members[1]] = members[0]
    return mapping


def _involution(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and set(value) == set(value.values())
        and all(
            isinstance(key, str)
            and key
            and isinstance(partner, str)
            and partner
            and value.get(partner) == key
            for key, partner in value.items()
        )
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _block_record(record: dict[str, object], blocker: str) -> None:
    record["structural_status"] = "blocked"
    record["readiness_status"] = "blocked"
    record["blockers"] = _deduplicate([
        *(
            value
            for value in record.get("blockers", [])
            if isinstance(value, str)
        ),
        blocker,
    ])
    record.pop("tr_irrep_completion_certificate", None)


def _block_orbit(orbit: dict[str, object], blockers: list[str]) -> None:
    combined = _deduplicate(blockers)
    orbit["tr_irrep_completion_status"] = "blocked"
    orbit["readiness_blockers"] = _deduplicate([
        *(
            value
            for value in orbit.get("readiness_blockers", [])
            if isinstance(value, str)
        ),
        *combined,
    ])
    orbit["blockers"] = combined
    orbit["status"] = "blocked"


def _deduplicate(values) -> list[str]:
    return list(dict.fromkeys(values))
