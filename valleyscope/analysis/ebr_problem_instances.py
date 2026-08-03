"""EBR problem-instance collector from trusted EBR input candidates.

Groups trusted candidate irreps into per-valley/per-subspace-group EBR
problem instances with a certificate-aware physical identity key.

This stage owns only construction of a complete canonical source-HSP vector.
Reviewed-table validation and exact reduced EBR outcomes are downstream.

Does NOT implement reduced EBR decomposition, EBR table matching,
compatibility relations, or new physics.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from valleyscope.analysis.tr_irrep_completion import (
    validate_tr_irrep_completion_certificate,
)
from valleyscope.analysis.unitary_valley_sewing_completion import (
    validate_unitary_valley_sewing_certificate_context,
)
from valleyscope.io.wavefunction_convention import valid_sha256_identity


def build_ebr_problem_instances(
    *,
    ebr_input_candidates: dict[str, object] | None,
    projected_hsp_coverage: dict[str, object] | None = None,
    time_reversal_orbit_report: dict[str, object] | None = None,
    unitary_valley_sewing_validation_contexts: (
        dict[str, object] | None
    ) = None,
) -> dict[str, object]:
    """Build EBR problem instances from trusted input candidates.

    Returns a dict with instances grouped by certificate-aware physical
    identity (SG number, SG symbol, Hall number, certificate validation
    status, valley).
    """
    if ebr_input_candidates is None:
        return _empty_report("no EBR input candidates available")

    candidates: list[dict[str, object]] = []
    raw = ebr_input_candidates.get("candidates")
    if isinstance(raw, list):
        for c in raw:
            if isinstance(c, dict) and c.get("ready_for_ebr_input") is True:
                candidates.append(c)

    if not candidates:
        return _empty_report("no trusted EBR input candidates")

    if isinstance(time_reversal_orbit_report, dict) and (
        time_reversal_orbit_report.get("enabled") is True
    ):
        return _build_time_reversal_problem_instances(
            candidates=candidates,
            time_reversal_orbit_report=time_reversal_orbit_report,
        )

    # Group by certificate-aware physical identity.
    # Use the complete immutable _SettingIdentity as the key so that
    # any affine-evidence difference (transform, origin, centering,
    # provenance, operation-mapping, affine-validation status) produces
    # a distinct group.
    groups: dict[tuple[_SettingIdentity, str], list[dict[str, object]]] = {}
    for c in candidates:
        valley = str(c.get("valley", ""))
        fp = _certificate_fingerprint(c)
        groups.setdefault((fp, valley), []).append(c)

    instances: list[dict[str, object]] = []
    instance_counter = 0

    for (fp, valley), cands in groups.items():
        instance_counter += 1
        # Setting identity for the flat keys.
        sg = fp.sg_symbol or str(cands[0].get("subspace_group_candidate", ""))
        _sg_num = fp.sg_number
        instance_id = f"ebr_instance_{instance_counter:03d}"

        # --- Canonical subgroup identity ---
        first_candidate_ssg = _first_subspace_space_group(cands)
        subspace_space_group: dict[str, object] = (
            dict(first_candidate_ssg) if isinstance(first_candidate_ssg, dict) else {}
        )
        canonical_sg = (
            subspace_space_group.get("candidate_space_group_symbol")
            or sg
        )
        canonical_sg_number = (
            subspace_space_group.get("candidate_space_group_number")
            or _sg_num
        )

        # --- Certificate identity from merged candidates ---
        cert_identity = _certificate_identity(cands)

        direct_provenance_blockers: list[str] = []
        spin_values = set()
        cprime_identities_by_scope: dict[
            tuple[str, str], set[tuple[str, str, str]]
        ] = {}
        for candidate in cands:
            candidate_source = candidate.get("source")
            candidate_workflow = candidate.get("workflow_path")
            inferred_by_sewing = candidate.get("completion_kind") == (
                "inferred_by_unitary_valley_sewing"
            )
            source_provenance = candidate.get("irrep_source_provenance")
            if not isinstance(candidate_source, str) or not candidate_source:
                direct_provenance_blockers.append(
                    "direct_candidate_source_provenance_missing"
                )
            if candidate_workflow not in (
                "direct_qcut",
                "symmetry_adapted",
                "unitary_valley_sewing_completion",
            ):
                direct_provenance_blockers.append(
                    "direct_candidate_projector_workflow_invalid"
                )
            if inferred_by_sewing and not _unitary_sewing_candidate_valid(
                candidate,
                unitary_valley_sewing_validation_contexts,
            ):
                direct_provenance_blockers.append(
                    "unitary_valley_sewing_certificate_revalidation_failed"
                )
            if not isinstance(source_provenance, dict):
                direct_provenance_blockers.append(
                    "direct_candidate_irrep_source_provenance_missing"
                )
                continue
            source_hsp = source_provenance.get("source_hsp_label")
            source_spinor = source_provenance.get("source_table_spinor")
            if not isinstance(source_hsp, str) or not source_hsp:
                direct_provenance_blockers.append(
                    "direct_candidate_source_hsp_provenance_missing"
                )
            if not isinstance(source_spinor, bool):
                direct_provenance_blockers.append(
                    "direct_candidate_spin_provenance_missing"
                )
            else:
                spin_values.add(source_spinor)
            cprime_rows = _candidate_cprime_rows(candidate)
            if not cprime_rows:
                direct_provenance_blockers.append(
                    "direct_candidate_cprime_provenance_missing"
                )
                continue
            for evidence_kpoint, evidence_valley, cprime in cprime_rows:
                identity_values = tuple(
                    cprime.get(key) for key in (
                        "spinor_source_basis_certificate_identity",
                        "double_space_group_lift_certificate_identity",
                        "scoped_representation_evidence_identity",
                    )
                )
                if not all(
                    valid_sha256_identity(value) for value in identity_values
                ):
                    direct_provenance_blockers.append(
                        "direct_candidate_cprime_identity_malformed"
                    )
                else:
                    cprime_identities_by_scope.setdefault(
                        (
                            str(evidence_kpoint or ""),
                            str(evidence_valley or ""),
                        ),
                        set(),
                    ).add(tuple(str(value) for value in identity_values))
        if len(spin_values) != 1:
            direct_provenance_blockers.append(
                "direct_candidate_spin_provenance_mismatch"
            )
        if any(
            len(values) != 1
            for values in cprime_identities_by_scope.values()
        ):
            direct_provenance_blockers.append(
                "direct_candidate_cprime_identity_mismatch"
            )
        sewing_candidates = any(
            candidate.get("completion_kind")
            == "inferred_by_unitary_valley_sewing"
            for candidate in cands
        )
        cprime_identity_by_kpoint: dict[str, dict[str, object]] = {}
        cprime_scope_metadata: dict[str, dict[str, str]] = {}
        for (kpoint, evidence_valley), identities in sorted(
            cprime_identities_by_scope.items()
        ):
            if len(identities) != 1:
                continue
            key = (
                f"scope_{len(cprime_scope_metadata) + 1:03d}"
                if sewing_candidates else kpoint
            )
            values = next(iter(identities))
            cprime_identity_by_kpoint[key] = {
                "spinor_source_basis_certificate_identity": (
                    values[0]
                ),
                "double_space_group_lift_certificate_identity": (
                    values[1]
                ),
                "scoped_representation_evidence_identity": values[2],
            }
            if sewing_candidates:
                cprime_scope_metadata[key] = {
                    "sampled_kpoint": kpoint,
                    "evidence_valley": evidence_valley,
                }

        # --- Aggregate provenance ---
        workflow_paths = sorted({
            str(c.get("workflow_path", ""))
            for c in cands if c.get("workflow_path")
        })
        readiness_levels = sorted({
            str(c.get("readiness_level", ""))
            for c in cands if c.get("readiness_level")
        })
        workflow_path = str(cands[0].get("workflow_path", ""))
        readiness_level = str(cands[0].get("readiness_level", ""))

        irreps_by_kpoint: dict[str, list[str]] = {}
        operations_by_kpoint: dict[str, list[object]] = {}
        irrep_records_by_kpoint: dict[str, list[dict[str, object]]] = {}
        completion_records_by_hsp: dict[
            str, list[dict[str, object]]
        ] = {}
        for c in cands:
            inferred_by_sewing = c.get("completion_kind") == (
                "inferred_by_unitary_valley_sewing"
            )
            kp = "" if inferred_by_sewing else str(c.get("kpoint", ""))
            irrep = c.get("matched_irrep")
            op_id = c.get("operation_id")
            if irrep and not inferred_by_sewing:
                multiplicity = _positive_multiplicity(c.get("irrep_multiplicity"))
                irreps_by_kpoint.setdefault(kp, []).extend(
                    [str(irrep)] * multiplicity
                )
            if op_id is not None:
                operations_by_kpoint.setdefault(kp, []).append(op_id)
            record: dict[str, object] = {
                "valley": valley,
                "sampled_kpoint": kp,
                "source_hsp_label": (
                    c.get("irrep_source_provenance", {}).get(
                        "source_hsp_label", ""
                    )
                    if isinstance(
                        c.get("irrep_source_provenance"), dict
                    ) else ""
                ),
                "operation_id": c.get("operation_id"),
                "operation_order": c.get("operation_order"),
                "matched_irrep": c.get("matched_irrep"),
                "irrep_multiplicity": _positive_multiplicity(
                    c.get("irrep_multiplicity")
                ),
                "character": c.get("character"),
                "eigenphases": c.get("eigenphases", []),
                "workflow_path": str(c.get("workflow_path", "")),
                "readiness_level": str(c.get("readiness_level", "")),
                "source": c.get("source", ""),
                "certificate_identity": dict(cert_identity),
                "source_candidate_identity": {
                    "source": c.get("source", ""),
                    "workflow_path": str(c.get("workflow_path", "")),
                    "valley": valley,
                    "source_hsp_label": (
                        c.get("irrep_source_provenance", {}).get(
                            "source_hsp_label", ""
                        )
                        if isinstance(
                            c.get("irrep_source_provenance"), dict
                        ) else ""
                    ),
                    "sampled_kpoint": kp,
                    "irrep": c.get("matched_irrep"),
                    "multiplicity": _positive_multiplicity(
                        c.get("irrep_multiplicity")
                    ),
                },
                "source_candidate_provenance": {
                    "source": c.get("source", ""),
                    "workflow_path": str(c.get("workflow_path", "")),
                    "irrep_source_provenance": dict(
                        c.get("irrep_source_provenance", {})
                    ) if isinstance(
                        c.get("irrep_source_provenance"), dict
                    ) else {},
                },
            }
            for key in (
                "matching_strategy",
                "subspace_space_group",
                "valley_preserving_operation_ids",
                "source_operation_map",
                "irrep_source_provenance",
                "projected_hsp_classification",
            ):
                if key in c:
                    record[key] = c[key]
            if inferred_by_sewing:
                source_hsp = record["source_hsp_label"]
                completion_records_by_hsp.setdefault(
                    str(source_hsp), []
                ).append({
                    "completion_kind": (
                        "inferred_by_unitary_valley_sewing"
                    ),
                    "target_valley": valley,
                    "target_source_hsp_label": source_hsp,
                    "irrep": c.get("matched_irrep"),
                    "multiplicity": _positive_multiplicity(
                        c.get("irrep_multiplicity")
                    ),
                    "evidence_sampled_kpoint": c.get(
                        "evidence_sampled_kpoint"
                    ),
                    "evidence_valley": c.get("evidence_valley"),
                    "evidence_source_hsp_label": c.get(
                        "evidence_source_hsp_label"
                    ),
                    "evidence_irrep_vector": deepcopy(
                        c.get("evidence_irrep_vector", [])
                    ),
                    "structural_status": "validated",
                    "readiness_status": "trusted",
                    "blockers": [],
                    "unitary_valley_sewing_certificates": deepcopy(
                        c.get("unitary_valley_sewing_certificates", [])
                    ),
                    "source_candidate_identity": record[
                        "source_candidate_identity"
                    ],
                    "source_candidate_provenance": record[
                        "source_candidate_provenance"
                    ],
                })
            else:
                irrep_records_by_kpoint.setdefault(kp, []).append(record)

        # --- HSP basis ---
        actual_hsps = sorted(irreps_by_kpoint.keys())
        expected_hsps = list(actual_hsps)
        optional_hsps: list[str] = []

        coverage = _coverage_for_valley(projected_hsp_coverage, valley)
        required_source_hsps = _string_list(
            coverage.get("required_source_hsp_labels", [])
        )
        covered_source_hsps = _string_list(
            coverage.get("covered_source_hsp_labels", [])
        )
        missing_source_hsps = _string_list(
            coverage.get("missing_source_hsp_labels", [])
        )
        trusted_source_hsps = _string_list(
            coverage.get("trusted_matched_source_hsp_labels", [])
        )
        trusted_missing_source_hsps = _string_list(
            coverage.get("trusted_missing_source_hsp_labels", [])
        )
        source_to_sampled = (
            dict(coverage.get("source_hsp_to_sampled_kpoint", {}))
            if isinstance(
                coverage.get("source_hsp_to_sampled_kpoint", {}), dict
            )
            else {}
        )
        coverage_present = bool(coverage)
        coverage_complete = bool(
            coverage.get("complete", False)
        ) if coverage_present else True
        coverage_ready = bool(
            coverage.get("ready_for_ebr_promotion", False)
        ) if coverage_present else True
        mapped_expected = [
            source_to_sampled.get(label)
            for label in required_source_hsps
        ]
        source_mapping_complete = (
            set(source_to_sampled) == set(required_source_hsps)
            and bool(required_source_hsps)
            and all(
                isinstance(label, str) and label
                for label in mapped_expected
            )
            and len(set(mapped_expected)) == len(mapped_expected)
        )
        if completion_records_by_hsp:
            source_by_sample = {
                sampled: source_hsp
                for source_hsp, sampled in source_to_sampled.items()
            }
            canonical_irreps: dict[str, list[str]] = {}
            canonical_records: dict[str, list[dict[str, object]]] = {}
            for sampled, labels in irreps_by_kpoint.items():
                source_hsp = source_by_sample.get(sampled)
                if isinstance(source_hsp, str):
                    canonical_irreps.setdefault(source_hsp, []).extend(labels)
                    canonical_records.setdefault(source_hsp, []).extend(
                        irrep_records_by_kpoint.get(sampled, [])
                    )
            for source_hsp, records in completion_records_by_hsp.items():
                for record in records:
                    canonical_irreps.setdefault(source_hsp, []).extend(
                        [str(record["irrep"])] * int(record["multiplicity"])
                    )
            irreps_by_kpoint = canonical_irreps
            irrep_records_by_kpoint = canonical_records
            actual_hsps = sorted(canonical_irreps)
            expected_hsps = list(required_source_hsps)
            completed_hsps = set(completion_records_by_hsp)
            canonical_hsp_vector_complete = (
                bool(required_source_hsps)
                and set(actual_hsps) == set(required_source_hsps)
                and set(missing_source_hsps) == completed_hsps
            )
            canonical_hsp_vector_ready = (
                canonical_hsp_vector_complete
                and not direct_provenance_blockers
            )
            coverage_complete = canonical_hsp_vector_complete
            coverage_ready = canonical_hsp_vector_ready
            source_mapping_complete = True
            covered_source_hsps = list(required_source_hsps)
            trusted_source_hsps = list(required_source_hsps)
            missing_source_hsps = []
            trusted_missing_source_hsps = []
        if source_mapping_complete and not completion_records_by_hsp:
            source_by_sample = {
                sampled: source_hsp
                for source_hsp, sampled in source_to_sampled.items()
            }
            if any(
                record.get("source_hsp_label")
                != source_by_sample.get(sampled)
                for sampled, records in irrep_records_by_kpoint.items()
                for record in records
            ):
                direct_provenance_blockers.append(
                    "direct_candidate_source_hsp_binding_mismatch"
                )
        if (
            coverage_present
            and coverage_complete
            and source_mapping_complete
            and not completion_records_by_hsp
        ):
            expected_hsps = [str(label) for label in mapped_expected]

        has_hsps = bool(actual_hsps)
        if not completion_records_by_hsp:
            canonical_hsp_vector_complete = (
                has_hsps
                and coverage_present
                and coverage_complete
                and source_mapping_complete
                and actual_hsps == sorted(expected_hsps)
            )
            canonical_hsp_vector_ready = (
                canonical_hsp_vector_complete and coverage_ready
                and not trusted_missing_source_hsps
                and not direct_provenance_blockers
            )
        status = _canonical_vector_status(
            complete=canonical_hsp_vector_complete,
            ready=canonical_hsp_vector_ready,
        )
        blocked_by: list[str] = list(dict.fromkeys(
            direct_provenance_blockers
        ))
        if not coverage_present:
            blocked_by.append("projected_hsp_coverage_missing")
        elif not coverage_complete:
            blocked_by.append("source_hsp_coverage_incomplete")
        elif not source_mapping_complete:
            blocked_by.append(
                "source_hsp_mapping_incomplete_or_ambiguous"
            )
        elif not coverage_ready:
            blocked_by.append(
                "source_hsp_coverage_not_ready_for_ebr_promotion"
            )
        if coverage_present and trusted_missing_source_hsps:
            blocked_by.append(
                f"missing trusted source HSPs: {trusted_missing_source_hsps}"
            )
        if has_hsps and actual_hsps != sorted(expected_hsps):
            blocked_by.append(
                "canonical sampled HSP basis does not match certified "
                f"source-HSP mapping: actual={actual_hsps}, "
                f"expected={sorted(expected_hsps)}"
            )

        instances.append({
            "instance_id": instance_id,
            "problem_kind": "unitary_valley_reduced_ebr",
            "physical_object_kind": "unitary_valley_projected_subspace",
            "valley": valley,
            "valley_orbit": [],
            "subspace_group_candidate": canonical_sg,
            "subspace_sg_number": canonical_sg_number,
            "subspace_space_group": subspace_space_group,
            "spinor": next(iter(spin_values), None),
            "certificate_identity": cert_identity,
            "cprime_identity_by_kpoint": cprime_identity_by_kpoint,
            "cprime_scope_metadata": cprime_scope_metadata,
            "workflow_path": (
                "unitary_valley_sewing_completion"
                if completion_records_by_hsp else workflow_path
            ),
            "workflow_paths": workflow_paths,
            "unitary_vector_construction": (
                {
                    "kind": (
                        "unitary_valley_sewing_completed_unitary_rows"
                    ),
                    "source": (
                        "validated_unitary_valley_sewing_certificates"
                    ),
                }
                if completion_records_by_hsp
                else {
                    "kind": "direct_observed_unitary_rows",
                    "source": "trusted_ebr_input_candidates",
                }
            ),
            "readiness_level": readiness_level,
            "readiness_evidence": readiness_levels,
            "irreps_by_kpoint": {k: v for k, v in sorted(irreps_by_kpoint.items())},
            "operations_by_kpoint": {
                k: sorted(v, key=_sort_key)
                for k, v in sorted(operations_by_kpoint.items())
            },
            "irrep_records_by_kpoint": {
                k: sorted(v, key=lambda r: (_sort_key(r.get("operation_id"))))
                for k, v in sorted(irrep_records_by_kpoint.items())
            },
            "unitary_irrep_completion_records_by_hsp": {
                key: value
                for key, value in sorted(
                    completion_records_by_hsp.items()
                )
            },
            "candidate_count": len(cands),
            "status": status,
            "canonical_hsp_vector_complete": canonical_hsp_vector_complete,
            "canonical_hsp_vector_ready": canonical_hsp_vector_ready,
            "blocked_by": blocked_by,
            "expected_hsps": expected_hsps,
            "expected_hsp_policy_source": "certified_source_hsp_basis",
            "optional_hsps": optional_hsps,
            "actual_hsps": actual_hsps,
            "missing_optional_hsps": [],
            "required_source_hsp_labels": required_source_hsps,
            "covered_source_hsp_labels": covered_source_hsps,
            "missing_source_hsp_labels": missing_source_hsps,
            "trusted_matched_source_hsp_labels": trusted_source_hsps,
            "trusted_missing_source_hsp_labels": trusted_missing_source_hsps,
            "source_hsp_to_sampled_kpoint": source_to_sampled,
            "source_hsp_coverage_complete": coverage_complete,
            "source_hsp_coverage_provenance": coverage.get(
                "source_basis_provenance", {}
            ),
        })

    counts = _problem_report_counts(instances)
    return {
        "status": _problem_report_status(instances),
        "instance_count": len(instances),
        **counts,
        "interpretation": (
            "Per-valley/per-subspace-group canonical source-HSP vectors "
            "grouped from trusted input candidates by certificate-aware "
            "physical identity. Reviewed-table validation is downstream."
        ),
        "instances": instances,
    }


def _coverage_for_valley(
    report: dict[str, object] | None,
    valley: str,
) -> dict[str, object]:
    if not isinstance(report, dict):
        return {}
    by_valley = report.get("by_valley", {})
    if not isinstance(by_valley, dict):
        return {}
    row = by_valley.get(valley, {})
    return dict(row) if isinstance(row, dict) else {}


def _unitary_sewing_candidate_valid(
    candidate: dict[str, object],
    contexts: dict[str, object] | None,
) -> bool:
    certificates = candidate.get("unitary_valley_sewing_certificates")
    if not isinstance(certificates, list) or not certificates:
        return False
    for certificate in certificates:
        if not isinstance(certificate, dict):
            return False
        identity = certificate.get("certificate_identity")
        context = contexts.get(identity) if isinstance(contexts, dict) else None
        if (
            not isinstance(context, dict)
            or context.get("certificate") != certificate
            or not validate_unitary_valley_sewing_certificate_context(
                certificate, context
            )
        ):
            return False
    return True


def _candidate_cprime_rows(candidate):
    if candidate.get("completion_kind") != "inferred_by_unitary_valley_sewing":
        provenance = candidate.get("irrep_source_provenance")
        cprime = provenance.get("cprime") if isinstance(provenance, dict) else None
        return [(
            candidate.get("kpoint"), candidate.get("valley"), cprime
        )] if isinstance(cprime, dict) else []
    rows = []
    for certificate in candidate.get("unitary_valley_sewing_certificates", []):
        source = certificate.get("source") if isinstance(certificate, dict) else None
        cprime = source.get("cprime") if isinstance(source, dict) else None
        if isinstance(cprime, dict):
            rows.append((
                source.get("sampled_kpoint"), source.get("valley"), cprime
            ))
    return rows


def _build_time_reversal_problem_instances(
    *,
    candidates: list[dict[str, object]],
    time_reversal_orbit_report: dict[str, object],
) -> dict[str, object]:
    """Build per-valley unitary problems and the joint grey-orbit problem."""
    raw_orbits = time_reversal_orbit_report.get("valley_orbits", [])
    if not isinstance(raw_orbits, list):
        raw_orbits = []
    instances: list[dict[str, object]] = []
    for index, raw_orbit in enumerate(raw_orbits, start=1):
        if not isinstance(raw_orbit, dict):
            continue
        members = raw_orbit.get("members", [])
        if not isinstance(members, list):
            members = []
        component_candidates = [
            candidate for candidate in candidates
            if candidate.get("valley") in members
        ]
        fingerprints = {
            _certificate_fingerprint(candidate)
            for candidate in component_candidates
        }
        blockers = [
            str(value) for value in raw_orbit.get("blockers", [])
            if isinstance(value, str)
        ]
        if not component_candidates:
            blockers.append("no_trusted_unitary_valley_irrep_components")
        if len(fingerprints) != 1:
            blockers.append(
                "time_reversal_component_standard_setting_mismatch"
            )
        spin_values = {
            provenance.get("source_table_spinor")
            for candidate in component_candidates
            if isinstance(
                provenance := candidate.get("irrep_source_provenance"), dict
            )
            and isinstance(provenance.get("source_table_spinor"), bool)
        }
        if len(spin_values) != 1:
            blockers.append("time_reversal_component_spin_evidence_mismatch")
        ssg = _first_subspace_space_group(component_candidates)
        sg_number = ssg.get("candidate_space_group_number")
        sg_symbol = ssg.get("candidate_space_group_symbol")
        irreps_by_kpoint = raw_orbit.get("irreps_by_kpoint", {})
        expected_hsps = raw_orbit.get("expected_hsps", [])
        if not isinstance(irreps_by_kpoint, dict) or not irreps_by_kpoint:
            blockers.append("time_reversal_grey_irrep_target_missing")
            irreps_by_kpoint = {}
        if not isinstance(expected_hsps, list) or set(irreps_by_kpoint) != set(
            expected_hsps
        ):
            blockers.append("time_reversal_independent_hsp_basis_mismatch")
            expected_hsps = []
        # Exact algebraic completion certifies unitary irrep rows only.  It is
        # not a numerical antiunitary sewing/corepresentation certificate for
        # the joint time-reversal orbit.
        blockers.append(
            "joint_time_reversal_corepresentation_not_certified"
        )
        tr_cprime_identity_by_hsp: dict[str, dict[str, object]] = {}
        canonical_hsp_vector_complete = (
            bool(irreps_by_kpoint)
            and bool(expected_hsps)
            and set(irreps_by_kpoint) == set(expected_hsps)
        )
        canonical_hsp_vector_ready = (
            canonical_hsp_vector_complete
            and raw_orbit.get("status") == "validated"
            and not blockers
        )
        instance_id = f"ebr_instance_{index:03d}"
        instances.append({
            "instance_id": instance_id,
            "problem_kind": "valley_orbit_reduced_ebr",
            "physical_object_kind": "joint_time_reversal_valley_orbit",
            "valley": "",
            "valley_orbit": list(members),
            "subspace_group_candidate": sg_symbol or "",
            "subspace_sg_number": sg_number,
            "subspace_space_group": dict(ssg),
            "spinor": next(iter(spin_values), None),
            "certificate_identity": _certificate_identity(
                component_candidates
            ),
            "cprime_identity_by_kpoint": dict(
                tr_cprime_identity_by_hsp
            ),
            "workflow_path": "time_reversal_valley_orbit",
            "workflow_paths": sorted({
                str(candidate.get("workflow_path", ""))
                for candidate in component_candidates
                if candidate.get("workflow_path")
            }),
            "readiness_level": (
                "trusted" if canonical_hsp_vector_ready else "blocked"
            ),
            "readiness_evidence": [
                "trusted_unitary_valley_irreps",
                "validated_time_reversal_valley_orbit",
                "reviewed_grey_group_source",
            ] if canonical_hsp_vector_ready else [],
            "irreps_by_kpoint": dict(irreps_by_kpoint),
            "operations_by_kpoint": {},
            "irrep_records_by_kpoint": {},
            "candidate_count": len(component_candidates),
            "status": _canonical_vector_status(
                complete=canonical_hsp_vector_complete,
                ready=canonical_hsp_vector_ready,
            ),
            "canonical_hsp_vector_complete": canonical_hsp_vector_complete,
            "canonical_hsp_vector_ready": canonical_hsp_vector_ready,
            "blocked_by": _deduplicate_strings(blockers),
            "expected_hsps": list(expected_hsps),
            "expected_hsp_policy_source": (
                "validated_time_reversal_hsp_orbits"
            ),
            "optional_hsps": [],
            "actual_hsps": list(irreps_by_kpoint),
            "missing_optional_hsps": [],
            "required_source_hsp_labels": raw_orbit.get(
                "full_unitary_source_hsp_labels", []
            ),
            "covered_source_hsp_labels": raw_orbit.get(
                "full_unitary_source_hsp_labels", []
            ) if canonical_hsp_vector_complete else [],
            "missing_source_hsp_labels": [],
            "trusted_matched_source_hsp_labels": raw_orbit.get(
                "full_unitary_source_hsp_labels", []
            ) if canonical_hsp_vector_ready else [],
            "trusted_missing_source_hsp_labels": [],
            "source_hsp_to_sampled_kpoint": raw_orbit.get(
                "source_hsp_to_sampled_kpoint", {}
            ),
            "source_hsp_coverage_complete": canonical_hsp_vector_complete,
            "source_hsp_coverage_provenance": {
                "source": "time_reversal_valley_orbit_completion",
            },
            "unitary_valley_irreps": raw_orbit.get(
                "time_reversal_completed_unitary_valley_irreps", {}
            ),
            "time_reversal": {
                "theta_square": time_reversal_orbit_report.get(
                    "theta_square"
                ),
                "representative_valley": raw_orbit.get("representative"),
                "time_reversal_valley_mapping": (
                    time_reversal_orbit_report.get(
                        "time_reversal_valley_mapping", {}
                    )
                ),
                "time_reversal_hsp_orbits": raw_orbit.get(
                    "time_reversal_hsp_orbits", []
                ),
                "full_unitary_source_hsp_labels": raw_orbit.get(
                    "full_unitary_source_hsp_labels", []
                ),
                "time_reversal_irrep_pairing": raw_orbit.get(
                    "time_reversal_irrep_pairing", {}
                ),
                "reviewed_time_reversal_source_identity": raw_orbit.get(
                    "reviewed_time_reversal_source_identity", {}
                ),
                "reviewed_time_reversal_source_context": raw_orbit.get(
                    "reviewed_time_reversal_source_context", {}
                ),
                "projector_workflow_by_sampled_kpoint": raw_orbit.get(
                    "projector_workflow_by_sampled_kpoint", {}
                ),
                "projector_provenance_by_sampled_kpoint": raw_orbit.get(
                    "projector_provenance_by_sampled_kpoint", {}
                ),
                "source_hsp_binding_by_sampled_kpoint": raw_orbit.get(
                    "source_hsp_binding_by_sampled_kpoint", {}
                ),
                "source_hsp_to_sampled_kpoint_by_valley": raw_orbit.get(
                    "source_hsp_to_sampled_kpoint_by_valley", {}
                ),
                "independent_source_hsp_to_sampled_kpoint_by_valley": (
                    raw_orbit.get(
                        "independent_source_hsp_to_sampled_kpoint_by_valley",
                        {},
                    )
                ),
                "observed_source_hsp_to_sampled_kpoint_by_valley": (
                    raw_orbit.get(
                        "observed_source_hsp_to_sampled_kpoint_by_valley",
                        {},
                    )
                ),
                "unitary_valley_irrep_completion_records": raw_orbit.get(
                    "unitary_valley_irrep_completion_records", {}
                ),
                "antiunitary_sewing_evidence": (
                    time_reversal_orbit_report.get(
                        "antiunitary_sewing_evidence", {}
                    )
                    if raw_orbit.get("mapping_type") == "self_mapped"
                    else {}
                ),
                "grey_bns_number": raw_orbit.get("grey_bns_number"),
            },
        })
        instances.extend(_build_tr_unitary_component_instances(
            orbit_index=index,
            raw_orbit=raw_orbit,
            component_candidates=component_candidates,
            time_reversal_orbit_report=time_reversal_orbit_report,
        ))
    counts = _problem_report_counts(instances)
    return {
        "status": _problem_report_status(instances),
        "instance_count": len(instances),
        **counts,
        "interpretation": (
            "Separate time-reversal-completed unitary valley problems and "
            "joint grey-orbit EBR problems. Inferred unitary rows retain "
            "their opposite-valley sampled evidence and are not reported as "
            "independently sampled rows."
        ),
        "instances": instances,
    }


def _build_tr_unitary_component_instances(
    *,
    orbit_index: int,
    raw_orbit: dict[str, object],
    component_candidates: list[dict[str, object]],
    time_reversal_orbit_report: dict[str, object],
) -> list[dict[str, object]]:
    members = raw_orbit.get("members", [])
    if not isinstance(members, list):
        return []
    full_hsps = _string_list(
        raw_orbit.get("full_unitary_source_hsp_labels", [])
    )
    completed = raw_orbit.get(
        "time_reversal_completed_unitary_valley_irreps", {}
    )
    all_records = raw_orbit.get(
        "unitary_valley_irrep_completion_records", {}
    )
    sampled_by_valley = raw_orbit.get(
        "independent_source_hsp_to_sampled_kpoint_by_valley",
        raw_orbit.get("source_hsp_to_sampled_kpoint_by_valley", {}),
    )
    observed_sampled_by_valley = raw_orbit.get(
        "observed_source_hsp_to_sampled_kpoint_by_valley", {}
    )
    if not isinstance(completed, dict):
        completed = {}
    if not isinstance(all_records, dict):
        all_records = {}
    if not isinstance(sampled_by_valley, dict):
        sampled_by_valley = {}
    if not isinstance(observed_sampled_by_valley, dict):
        observed_sampled_by_valley = {}

    instances: list[dict[str, object]] = []
    for component_index, valley in enumerate(members, start=1):
        if not isinstance(valley, str) or not valley:
            continue
        candidates = [
            candidate for candidate in component_candidates
            if candidate.get("valley") == valley
        ]
        counts_by_hsp = completed.get(valley, {})
        records_by_hsp = all_records.get(valley, {})
        if not isinstance(counts_by_hsp, dict):
            counts_by_hsp = {}
        if not isinstance(records_by_hsp, dict):
            records_by_hsp = {}

        blockers: list[str] = []
        orbit_unitary_blockers = [
            str(blocker)
            for blocker in raw_orbit.get(
                "unitary_completion_blockers", []
            )
            if isinstance(blocker, str) and blocker
        ]
        if (
            raw_orbit.get("unitary_completion_status") != "validated"
            or orbit_unitary_blockers
        ):
            blockers.append("time_reversal_unitary_source_not_validated")
            blockers.extend(orbit_unitary_blockers)
        fingerprints = {
            _certificate_fingerprint(candidate) for candidate in candidates
        }
        if not candidates:
            blockers.append(
                f"no_trusted_unitary_valley_irrep_component:{valley}"
            )
        if len(fingerprints) != 1:
            blockers.append(
                f"unitary_component_standard_setting_mismatch:{valley}"
            )
        spin_values = {
            provenance.get("source_table_spinor")
            for candidate in candidates
            if isinstance(
                provenance := candidate.get("irrep_source_provenance"), dict
            )
            and isinstance(provenance.get("source_table_spinor"), bool)
        }
        if len(spin_values) != 1:
            blockers.append(
                f"unitary_component_spin_evidence_mismatch:{valley}"
            )
        (
            tr_cprime_identity_by_hsp,
            cprime_blockers,
        ) = _completion_cprime_identity_inventory(
            valley=valley,
            records_by_hsp=records_by_hsp,
            required_hsps=full_hsps,
            valley_mapping=time_reversal_orbit_report.get(
                "time_reversal_valley_mapping", {}
            ),
            hsp_mapping=_time_reversal_hsp_mapping(
                raw_orbit.get("time_reversal_hsp_orbits", [])
            ),
            irrep_pairing=raw_orbit.get(
                "time_reversal_irrep_pairing", {}
            ),
            reviewed_source_identity=raw_orbit.get(
                "reviewed_time_reversal_source_identity", {}
            ),
            reviewed_source_context=raw_orbit.get(
                "reviewed_time_reversal_source_context", {}
            ),
            require_exact_completion=(
                raw_orbit.get("mapping_type") == "exchanged"
            ),
        )
        blockers.extend(cprime_blockers)

        complete_counts = (
            bool(full_hsps)
            and set(counts_by_hsp) == set(full_hsps)
            and all(
                isinstance(counts_by_hsp.get(hsp), dict)
                and bool(counts_by_hsp.get(hsp))
                for hsp in full_hsps
            )
        )
        complete_records = (
            bool(full_hsps)
            and set(records_by_hsp) == set(full_hsps)
            and _completion_records_match_counts(
                counts_by_hsp=counts_by_hsp,
                records_by_hsp=records_by_hsp,
            )
        )
        if not complete_counts:
            blockers.append(
                f"unitary_component_source_hsp_vector_incomplete:{valley}"
            )
        if not complete_records:
            blockers.append(
                f"unitary_component_completion_provenance_incomplete:{valley}"
            )
        for hsp, records in records_by_hsp.items():
            if not isinstance(records, list):
                blockers.append(
                    "unitary_component_completion_records_malformed:"
                    f"{valley}:{hsp}"
                )
                continue
            for record in records:
                if not isinstance(record, dict):
                    blockers.append(
                        "unitary_component_completion_record_malformed:"
                        f"{valley}:{hsp}"
                    )
                    continue
                if (
                    record.get("structural_status") != "validated"
                    or record.get("readiness_status") != "trusted"
                ):
                    record_blockers = record.get("blockers", [])
                    if isinstance(record_blockers, list) and record_blockers:
                        blockers.extend(
                            str(value) for value in record_blockers
                            if isinstance(value, str)
                        )
                    else:
                        blockers.append(
                            "unitary_component_completion_record_untrusted:"
                            f"{valley}:{hsp}"
                        )

        canonical_complete = complete_counts and complete_records
        canonical_ready = canonical_complete and not blockers
        ssg = _first_subspace_space_group(candidates)
        sampled_mapping = sampled_by_valley.get(valley, {})
        if not isinstance(sampled_mapping, dict):
            sampled_mapping = {}
        observed_sampled_mapping = observed_sampled_by_valley.get(
            valley, {}
        )
        if not isinstance(observed_sampled_mapping, dict):
            observed_sampled_mapping = {}
        irreps_by_kpoint = {
            hsp: [
                irrep
                for irrep, multiplicity in sorted(
                    counts_by_hsp.get(hsp, {}).items()
                )
                for _ in range(multiplicity)
            ]
            for hsp in full_hsps
            if isinstance(counts_by_hsp.get(hsp), dict)
            and counts_by_hsp.get(hsp)
        }
        workflow_paths = sorted({
            str(candidate.get("workflow_path", ""))
            for candidate in candidates if candidate.get("workflow_path")
        })
        instances.append({
            "instance_id": (
                f"ebr_instance_{orbit_index:03d}_unitary_"
                f"{component_index:03d}"
            ),
            "problem_kind": "unitary_valley_reduced_ebr",
            "physical_object_kind": "unitary_valley_projected_subspace",
            "valley": valley,
            "valley_orbit": list(members),
            "subspace_group_candidate": ssg.get(
                "candidate_space_group_symbol", ""
            ),
            "subspace_sg_number": ssg.get(
                "candidate_space_group_number"
            ),
            "subspace_space_group": dict(ssg),
            "spinor": next(iter(spin_values), None),
            "certificate_identity": _certificate_identity(candidates),
            "cprime_identity_by_kpoint": dict(
                tr_cprime_identity_by_hsp
            ),
            "workflow_path": "time_reversal_completed_unitary_valley",
            "workflow_paths": workflow_paths,
            "unitary_vector_construction": {
                "kind": "time_reversal_completed_unitary_rows",
                "source": "validated_time_reversal_valley_orbit",
                "orbit_id": raw_orbit.get("orbit_id", ""),
            },
            "readiness_level": (
                "trusted" if canonical_ready else "blocked"
            ),
            "readiness_evidence": [
                "trusted_unitary_valley_irreps",
                "validated_row_level_time_reversal_completion",
            ] if canonical_ready else [],
            "irreps_by_kpoint": irreps_by_kpoint,
            "operations_by_kpoint": {},
            "irrep_records_by_kpoint": {},
            "unitary_irrep_completion_records_by_hsp": {
                hsp: [dict(record) for record in records]
                for hsp, records in records_by_hsp.items()
                if isinstance(records, list)
            },
            "candidate_count": len(candidates),
            "status": _canonical_vector_status(
                complete=canonical_complete,
                ready=canonical_ready,
            ),
            "canonical_hsp_vector_complete": canonical_complete,
            "canonical_hsp_vector_ready": canonical_ready,
            "blocked_by": _deduplicate_strings(blockers),
            "expected_hsps": list(full_hsps),
            "expected_hsp_policy_source": (
                "time_reversal_completed_unitary_source_hsp_basis"
            ),
            "optional_hsps": [],
            "actual_hsps": list(irreps_by_kpoint),
            "missing_optional_hsps": [],
            "required_source_hsp_labels": list(full_hsps),
            "covered_source_hsp_labels": (
                list(full_hsps) if canonical_complete else []
            ),
            "missing_source_hsp_labels": [
                hsp for hsp in full_hsps if hsp not in irreps_by_kpoint
            ],
            "trusted_matched_source_hsp_labels": (
                list(full_hsps) if canonical_ready else []
            ),
            "trusted_missing_source_hsp_labels": (
                [] if canonical_ready else list(full_hsps)
            ),
            "source_hsp_to_sampled_kpoint": dict(sampled_mapping),
            "independent_source_hsp_to_sampled_kpoint": dict(
                sampled_mapping
            ),
            "observed_source_hsp_to_sampled_kpoint": dict(
                observed_sampled_mapping
            ),
            "source_hsp_coverage_complete": canonical_complete,
            "source_hsp_coverage_provenance": {
                "source": "time_reversal_completed_unitary_valley",
                "observed_rows_are_sampled": True,
                "inferred_rows_are_sampled": False,
            },
            "time_reversal": {
                "theta_square": time_reversal_orbit_report.get(
                    "theta_square"
                ),
                "mapping_type": raw_orbit.get("mapping_type"),
                "valley_orbit": list(members),
                "time_reversal_valley_mapping": (
                    time_reversal_orbit_report.get(
                        "time_reversal_valley_mapping", {}
                    )
                ),
                "time_reversal_hsp_orbits": raw_orbit.get(
                    "time_reversal_hsp_orbits", []
                ),
                "full_unitary_source_hsp_labels": list(full_hsps),
                "independent_time_reversal_hsp_labels": raw_orbit.get(
                    "independent_time_reversal_hsp_labels", []
                ),
                "time_reversal_irrep_pairing": raw_orbit.get(
                    "time_reversal_irrep_pairing", {}
                ),
                "reviewed_time_reversal_source_identity": raw_orbit.get(
                    "reviewed_time_reversal_source_identity", {}
                ),
                "reviewed_time_reversal_source_context": raw_orbit.get(
                    "reviewed_time_reversal_source_context", {}
                ),
                "projector_workflow_by_sampled_kpoint": raw_orbit.get(
                    "projector_workflow_by_sampled_kpoint", {}
                ),
                "projector_provenance_by_sampled_kpoint": raw_orbit.get(
                    "projector_provenance_by_sampled_kpoint", {}
                ),
                "source_hsp_binding_by_sampled_kpoint": raw_orbit.get(
                    "source_hsp_binding_by_sampled_kpoint", {}
                ),
                "antiunitary_sewing_evidence": (
                    time_reversal_orbit_report.get(
                        "antiunitary_sewing_evidence", {}
                    )
                    if raw_orbit.get("mapping_type") == "self_mapped"
                    else {}
                ),
            },
        })
    return instances


def _completion_records_match_counts(
    *,
    counts_by_hsp: dict[str, object],
    records_by_hsp: dict[str, object],
) -> bool:
    for hsp, raw_counts in counts_by_hsp.items():
        records = records_by_hsp.get(hsp)
        if not isinstance(raw_counts, dict) or not isinstance(records, list):
            return False
        derived: dict[str, int] = {}
        for record in records:
            if not isinstance(record, dict):
                return False
            irrep = record.get("irrep")
            multiplicity = record.get("multiplicity")
            if (
                not isinstance(irrep, str)
                or not irrep
                or not isinstance(multiplicity, int)
                or isinstance(multiplicity, bool)
                or multiplicity <= 0
            ):
                return False
            derived[irrep] = derived.get(irrep, 0) + multiplicity
        if derived != raw_counts:
            return False
    return set(counts_by_hsp) == set(records_by_hsp)


def _completion_cprime_identity_inventory(
    *,
    valley: str,
    records_by_hsp: dict[str, object],
    required_hsps: Sequence[str],
    valley_mapping: object,
    hsp_mapping: object,
    irrep_pairing: object,
    reviewed_source_identity: object,
    reviewed_source_context: object,
    require_exact_completion: bool,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    inventory: dict[str, dict[str, object]] = {}
    blockers: list[str] = []
    if not all(
        isinstance(value, dict)
        for value in (valley_mapping, hsp_mapping, irrep_pairing)
    ):
        return {}, [
            f"tr_irrep_completion_mapping_invalid:{valley}"
        ]
    for hsp in required_hsps:
        records = records_by_hsp.get(hsp)
        identities: list[dict[str, object]] = []
        if not isinstance(records, list) or not records:
            blockers.append(
                f"tr_irrep_completion_record_missing:{valley}:{hsp}"
            )
            continue
        for record in records:
            identity = _completion_record_cprime_identity(
                record=record,
                valley_mapping=valley_mapping,
                hsp_mapping=hsp_mapping,
                irrep_pairing=irrep_pairing,
                reviewed_source_identity=reviewed_source_identity,
                reviewed_source_context=reviewed_source_context,
                require_exact_completion=require_exact_completion,
            )
            if identity is None:
                blockers.append(
                    f"tr_irrep_completion_cprime_invalid:{valley}:{hsp}"
                )
                continue
            identities.append(identity)
        if not identities:
            continue
        if any(identity != identities[0] for identity in identities[1:]):
            blockers.append(
                f"tr_irrep_completion_cprime_mismatch:{valley}:{hsp}"
            )
            continue
        inventory[hsp] = identities[0]
    if not _valid_cprime_identity_inventory(
        inventory,
        required_hsps=required_hsps,
    ):
        blockers.append(
            f"tr_irrep_completion_cprime_inventory_incomplete:{valley}"
        )
    return inventory, _deduplicate_strings(blockers)


def _completion_record_cprime_identity(
    *,
    record: object,
    valley_mapping: dict[str, object],
    hsp_mapping: dict[str, object],
    irrep_pairing: dict[str, object],
    reviewed_source_identity: object,
    reviewed_source_context: object,
    require_exact_completion: bool,
) -> dict[str, object] | None:
    if not isinstance(record, dict):
        return None
    kind = record.get("completion_kind")
    if kind == "observed_at_sampled_kpoint":
        provenance = record.get("source_candidate_provenance")
        source_irrep = (
            provenance.get("irrep_source_provenance")
            if isinstance(provenance, dict)
            else None
        )
        identity = (
            source_irrep.get("cprime")
            if isinstance(source_irrep, dict)
            else None
        )
    elif kind == "inferred_by_time_reversal" and require_exact_completion:
        certificate = record.get("tr_irrep_completion_certificate")
        if not validate_tr_irrep_completion_certificate(
            certificate,
            completion_record=record,
            valley_mapping=valley_mapping,
            hsp_mapping=hsp_mapping,
            irrep_pairing=irrep_pairing,
            reviewed_source_identity=(
                reviewed_source_identity
                if isinstance(reviewed_source_identity, dict)
                else {}
            ),
            reviewed_source_context=(
                reviewed_source_context
                if isinstance(reviewed_source_context, dict)
                else {}
            ),
        ):
            return None
        observed = certificate.get("observed_source")
        identity = (
            observed.get("local_cprime_identity")
            if isinstance(observed, dict)
            else None
        )
    elif kind == "inferred_by_time_reversal":
        provenance = record.get("source_candidate_provenance")
        source_irrep = (
            provenance.get("irrep_source_provenance")
            if isinstance(provenance, dict)
            else None
        )
        identity = (
            source_irrep.get("cprime")
            if isinstance(source_irrep, dict)
            else None
        )
    else:
        return None
    if not _valid_cprime_identity(identity):
        return None
    return dict(identity)


def _valid_cprime_identity(value: object) -> bool:
    required_keys = {
        "spinor_source_basis_certificate_identity",
        "double_space_group_lift_certificate_identity",
        "scoped_representation_evidence_identity",
    }
    return (
        isinstance(value, dict)
        and set(value) == required_keys
        and all(
            valid_sha256_identity(value.get(key))
            for key in required_keys
        )
    )


def _time_reversal_hsp_mapping(value: object) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not isinstance(value, list):
        return mapping
    for orbit in value:
        if not isinstance(orbit, dict):
            return {}
        members = orbit.get("members")
        if (
            not isinstance(members, list)
            or len(members) not in (1, 2)
            or not all(
                isinstance(member, str) and member for member in members
            )
        ):
            return {}
        if len(members) == 1:
            mapping[members[0]] = members[0]
        else:
            mapping[members[0]] = members[1]
            mapping[members[1]] = members[0]
    return mapping


def _deduplicate_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _valid_cprime_identity_inventory(
    value: object,
    *,
    required_hsps: Sequence[str],
) -> bool:
    if (
        not required_hsps
        or not isinstance(value, dict)
        or set(value) != set(required_hsps)
    ):
        return False
    return all(
        _valid_cprime_identity(identity)
        for identity in value.values()
    )


def _problem_report_status(instances: list[dict[str, object]]) -> str:
    ready = sum(
        instance.get("canonical_hsp_vector_ready") is True
        for instance in instances
    )
    complete = sum(
        instance.get("canonical_hsp_vector_complete") is True
        for instance in instances
    )
    if not instances:
        return "no_canonical_hsp_vectors"
    if ready == len(instances):
        return "canonical_hsp_vectors_ready"
    if ready:
        return "partial_canonical_hsp_vectors_ready"
    if complete == len(instances):
        return "canonical_hsp_vectors_complete_but_untrusted"
    if complete:
        return "canonical_hsp_vectors_blocked"
    return "incomplete_canonical_hsp_vectors"


def _problem_report_counts(
    instances: list[dict[str, object]],
) -> dict[str, int]:
    ready = sum(
        instance.get("canonical_hsp_vector_ready") is True
        for instance in instances
    )
    complete = sum(
        instance.get("canonical_hsp_vector_complete") is True
        for instance in instances
    )
    return {
        "ready_instance_count": ready,
        "structurally_complete_instance_count": complete,
        "structurally_complete_blocked_count": complete - ready,
        "incomplete_instance_count": len(instances) - complete,
    }


def _canonical_vector_status(*, complete: bool, ready: bool) -> str:
    if ready:
        return "canonical_hsp_vector_ready"
    if complete:
        return "canonical_hsp_vector_complete_but_untrusted"
    return "incomplete_canonical_hsp_vector"


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _first_subspace_space_group(
    cands: list[dict[str, object]],
) -> dict[str, object]:
    """Return the canonical subspace_space_group from the first candidate."""
    for c in cands:
        ssg = c.get("subspace_space_group")
        if isinstance(ssg, dict) and ssg:
            return ssg
    return {}


def _sort_key(op_id: object) -> tuple[int, object]:
    try:
        return (0, int(str(op_id)))
    except (TypeError, ValueError):
        return (1, str(op_id))


def _positive_multiplicity(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 1


def _empty_report(reason: str) -> dict[str, object]:
    return {
        "status": "no_canonical_hsp_vectors",
        "instance_count": 0,
        "ready_instance_count": 0,
        "structurally_complete_instance_count": 0,
        "structurally_complete_blocked_count": 0,
        "incomplete_instance_count": 0,
        "interpretation": reason,
        "instances": [],
    }


# ---------------------------------------------------------------------------
# Certificate-aware identity
# ---------------------------------------------------------------------------

_CERT_TOL = 1e-9
_SENTINEL_MISSING = object()  # distinguishes absent from empty in fingerprints


class _SettingIdentity:
    """Hashable normalized standard-setting certificate identity.

    Captures the setting-level affine evidence needed to distinguish
    physically inequivalent conventions for the same space group.
    HSP-specific fields (parent_k_frac, resolved_hsp_label) are excluded
    because they vary per k-point and belong in per-row provenance.
    """

    __slots__ = (
        "_hash",
        "sg_number", "sg_symbol",
        "hall_number", "hall_symbol",
        "transform_key", "origin_shift_key",
        "centering_type", "centering_vectors_key",
        "primitive_conventional_relation",
        "transform_provenance",
        "validation_status",
        "operation_mapping_status",
        "affine_validation_status",
        "affine_matched_operations",
        "affine_total_operations",
        "affine_mismatch_count",
        "affine_missing_ingredients",
        "affine_standard_setting_op_count",
        "affine_operation_map",
        "affine_unmatched_parents",
        "affine_unused_std",
        "affine_required_operation_ids",
        "affine_required_op_count",
        "operation_closure_validated",
        "canonical_setting_status",
        "canonical_setting_source",
        "canonical_hall_numbers",
        "canonical_candidate_hall_numbers",
        "centering_coset_count",
        "primitive_conventional_index",
        "expanded_parent_operation_count",
        "matched_expanded_operations",
        "centered_affine_operation_map",
        "affine_unmatched_centered_pairs",
        "standard_operation_closure_validated",
    )

    def __init__(
        self,
        sg_number: int,
        sg_symbol: str,
        hall_number: int,
        hall_symbol: str,
        transform_key: tuple[tuple[float, float, float],
                            tuple[float, float, float],
                            tuple[float, float, float]] | None,
        origin_shift_key: tuple[float, float, float] | None,
        centering_type: str,
        centering_vectors_key: tuple[tuple[float, float, float], ...] | None,
        primitive_conventional_relation: str,
        transform_provenance: str,
        validation_status: str,
        operation_mapping_status: str,
        affine_validation_status: str,
        affine_matched_operations: int | None = None,
        affine_total_operations: int | None = None,
        affine_mismatch_count: int | None = None,
        affine_missing_ingredients: tuple[str, ...] | None = None,
        affine_standard_setting_op_count: int | None = None,
        affine_operation_map: tuple[tuple[int, int], ...] | None = None,
        affine_unmatched_parents: tuple[int, ...] | None = None,
        affine_unused_std: tuple[int, ...] | None = None,
        affine_required_operation_ids: tuple[int, ...] | None = None,
        affine_required_op_count: int | None = None,
        operation_closure_validated: bool | None = None,
        canonical_setting_status: str = "not_evaluated",
        canonical_setting_source: str = "",
        canonical_hall_numbers: tuple[int, ...] | None = None,
        canonical_candidate_hall_numbers: tuple[int, ...] | None = None,
        centering_coset_count: int | None = None,
        primitive_conventional_index: int | None = None,
        expanded_parent_operation_count: int | None = None,
        matched_expanded_operations: int | None = None,
        centered_affine_operation_map: tuple[tuple[int, int, int], ...] | None = None,
        affine_unmatched_centered_pairs: tuple[tuple[int, int], ...] | None = None,
        standard_operation_closure_validated: bool | None = None,
    ):
        self.sg_number = sg_number
        self.sg_symbol = sg_symbol
        self.hall_number = hall_number
        self.hall_symbol = hall_symbol
        self.transform_key = transform_key
        self.origin_shift_key = origin_shift_key
        self.centering_type = centering_type
        self.centering_vectors_key = centering_vectors_key
        self.primitive_conventional_relation = primitive_conventional_relation
        self.transform_provenance = transform_provenance
        self.validation_status = validation_status
        self.operation_mapping_status = operation_mapping_status
        self.affine_validation_status = affine_validation_status
        self.affine_matched_operations = affine_matched_operations
        self.affine_total_operations = affine_total_operations
        self.affine_mismatch_count = affine_mismatch_count
        self.affine_missing_ingredients = affine_missing_ingredients
        self.affine_standard_setting_op_count = affine_standard_setting_op_count
        self.affine_operation_map = affine_operation_map
        self.affine_unmatched_parents = affine_unmatched_parents
        self.affine_unused_std = affine_unused_std
        self.affine_required_operation_ids = affine_required_operation_ids
        self.affine_required_op_count = affine_required_op_count
        self.operation_closure_validated = operation_closure_validated
        self.canonical_setting_status = canonical_setting_status
        self.canonical_setting_source = canonical_setting_source
        self.canonical_hall_numbers = canonical_hall_numbers
        self.canonical_candidate_hall_numbers = canonical_candidate_hall_numbers
        self.centering_coset_count = centering_coset_count
        self.primitive_conventional_index = primitive_conventional_index
        self.expanded_parent_operation_count = expanded_parent_operation_count
        self.matched_expanded_operations = matched_expanded_operations
        self.centered_affine_operation_map = centered_affine_operation_map
        self.affine_unmatched_centered_pairs = affine_unmatched_centered_pairs
        self.standard_operation_closure_validated = \
            standard_operation_closure_validated
        self._hash = hash((
            sg_number, sg_symbol,
            hall_number, hall_symbol,
            transform_key, origin_shift_key,
            centering_type, centering_vectors_key,
            primitive_conventional_relation,
            transform_provenance,
            validation_status,
            operation_mapping_status,
            affine_validation_status,
            affine_matched_operations,
            affine_total_operations,
            affine_mismatch_count,
            affine_missing_ingredients,
            affine_standard_setting_op_count,
            affine_operation_map,
            affine_unmatched_parents,
            affine_unused_std,
            affine_required_operation_ids,
            affine_required_op_count,
            operation_closure_validated,
            canonical_setting_status,
            canonical_setting_source,
            canonical_hall_numbers,
            canonical_candidate_hall_numbers,
            centering_coset_count,
            primitive_conventional_index,
            expanded_parent_operation_count,
            matched_expanded_operations,
            centered_affine_operation_map,
            affine_unmatched_centered_pairs,
            standard_operation_closure_validated,
        ))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _SettingIdentity):
            return NotImplemented
        return self._hash == other._hash and (
            self.sg_number == other.sg_number
            and self.sg_symbol == other.sg_symbol
            and self.hall_number == other.hall_number
            and self.hall_symbol == other.hall_symbol
            and self.transform_key == other.transform_key
            and self.origin_shift_key == other.origin_shift_key
            and self.centering_type == other.centering_type
            and self.centering_vectors_key == other.centering_vectors_key
            and self.primitive_conventional_relation == other.primitive_conventional_relation
            and self.transform_provenance == other.transform_provenance
            and self.validation_status == other.validation_status
            and self.operation_mapping_status == other.operation_mapping_status
            and self.affine_validation_status == other.affine_validation_status
            and self.affine_matched_operations == other.affine_matched_operations
            and self.affine_total_operations == other.affine_total_operations
            and self.affine_mismatch_count == other.affine_mismatch_count
            and self.affine_missing_ingredients == other.affine_missing_ingredients
            and self.affine_standard_setting_op_count == other.affine_standard_setting_op_count
            and self.affine_operation_map == other.affine_operation_map
            and self.affine_unmatched_parents == other.affine_unmatched_parents
            and self.affine_unused_std == other.affine_unused_std
            and self.affine_required_operation_ids == other.affine_required_operation_ids
            and self.affine_required_op_count == other.affine_required_op_count
            and self.operation_closure_validated == other.operation_closure_validated
            and self.canonical_setting_status == other.canonical_setting_status
            and self.canonical_setting_source == other.canonical_setting_source
            and self.canonical_hall_numbers == other.canonical_hall_numbers
            and self.canonical_candidate_hall_numbers == other.canonical_candidate_hall_numbers
            and self.centering_coset_count == other.centering_coset_count
            and self.primitive_conventional_index == other.primitive_conventional_index
            and self.expanded_parent_operation_count == other.expanded_parent_operation_count
            and self.matched_expanded_operations == other.matched_expanded_operations
            and self.centered_affine_operation_map == other.centered_affine_operation_map
            and self.affine_unmatched_centered_pairs == other.affine_unmatched_centered_pairs
            and self.standard_operation_closure_validated == other.standard_operation_closure_validated
        )


def _normalize_transform(
    matrix: object,
) -> tuple[tuple[float, float, float],
           tuple[float, float, float],
           tuple[float, float, float]] | None:
    """Normalize a 3×3 transform matrix to a hashable tuple.

    Rounds to tolerance, replaces -0.0 with 0.0.  Returns None for
    non-finite or non-3×3 input.
    """
    if not isinstance(matrix, (list, tuple)):
        return None
    if len(matrix) != 3:
        return None
    rows: list[tuple[float, float, float]] = []
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        r: list[float] = []
        for v in row:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(f):
                return None
            f = round(f / _CERT_TOL) * _CERT_TOL
            if f == 0.0:
                f = 0.0  # normalize -0.0
            r.append(f)
        rows.append((r[0], r[1], r[2]))
    return (rows[0], rows[1], rows[2])


def _normalize_origin_shift(
    vector: object,
) -> tuple[float, float, float] | None:
    """Normalize a 3-vector origin shift to a hashable tuple."""
    if not isinstance(vector, (list, tuple)) or len(vector) != 3:
        return None
    comps: list[float] = []
    for v in vector:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(f):
            return None
        # Modulo lattice: shift into [0, 1).
        f = f - np.floor(f)
        f = round(f / _CERT_TOL) * _CERT_TOL
        if f == 0.0:
            f = 0.0
        # Remap 1.0 (from rounding) back to 0.0.
        if f >= 1.0 - _CERT_TOL:
            f = 0.0
        comps.append(f)
    return (comps[0], comps[1], comps[2])


def _normalize_centering_vectors(
    vectors: object,
) -> tuple[tuple[float, float, float], ...] | None:
    """Normalize vectors without changing their map-index ordering."""
    if not isinstance(vectors, list):
        return None
    normed: list[tuple[float, float, float]] = []
    for v in vectors:
        normalized = _normalize_origin_shift(v)
        if normalized is None:
            return None
        normed.append(normalized)
    return tuple(normed)


def _normalize_strict_int_list(
    value: object,
    *,
    unique: bool = False,
) -> tuple[int, ...] | None:
    """Normalize an exact runtime ``list[int]`` without coercion."""
    if not isinstance(value, list):
        return None
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        return None
    if unique and len(value) != len(set(value)):
        return None
    return tuple(sorted(value))


def _normalize_strict_string_list(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    if any(not isinstance(item, str) for item in value):
        return None
    return tuple(sorted(value))


def _normalize_operation_id_key(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = int(value)
    except ValueError:
        return None
    if str(normalized) != value:
        return None
    return normalized


def _normalize_affine_operation_map(
    value: object,
) -> tuple[tuple[int, int], ...] | None:
    """Normalize an operation map and reject key aliases after coercion."""
    if not isinstance(value, dict):
        return None
    normalized: dict[int, int] = {}
    for raw_key, raw_value in value.items():
        key = _normalize_operation_id_key(raw_key)
        if key is None or key in normalized:
            return None
        if not isinstance(raw_value, int) or isinstance(raw_value, bool):
            return None
        normalized[key] = raw_value
    return tuple(sorted(normalized.items()))


def _normalize_unmatched_parent_operations(
    value: object,
) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        return None
    operation_ids: list[int] = []
    for row in value:
        if not isinstance(row, dict):
            return None
        operation_id = row.get("operation_id")
        if not isinstance(operation_id, int) or isinstance(operation_id, bool):
            return None
        operation_ids.append(operation_id)
    if len(operation_ids) != len(set(operation_ids)):
        return None
    return tuple(sorted(operation_ids))


def _normalize_centered_affine_operation_map(
    value: object,
) -> tuple[tuple[int, int, int], ...] | None:
    if not isinstance(value, list):
        return None
    normalized: list[tuple[int, int, int]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "parent_operation_id",
            "centering_coset_index",
            "standard_operation_index",
        }:
            return None
        fields = (
            row["parent_operation_id"],
            row["centering_coset_index"],
            row["standard_operation_index"],
        )
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in fields
        ):
            return None
        normalized.append(fields)
    if len(normalized) != len(set(normalized)):
        return None
    return tuple(sorted(normalized))


def _normalize_unmatched_centered_pairs(
    value: object,
) -> tuple[tuple[int, int], ...] | None:
    if not isinstance(value, list):
        return None
    normalized: list[tuple[int, int]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "parent_operation_id", "centering_coset_index",
        }:
            return None
        fields = (
            row["parent_operation_id"], row["centering_coset_index"],
        )
        if any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in fields
        ):
            return None
        normalized.append(fields)
    if len(normalized) != len(set(normalized)):
        return None
    return tuple(sorted(normalized))


def _certificate_fingerprint(candidate: dict[str, object]) -> _SettingIdentity:
    """Extract a normalized setting identity from one candidate."""
    prov = candidate.get("irrep_source_provenance")
    cert: dict[str, object] = {}
    if isinstance(prov, dict):
        kmap = prov.get("standard_setting_hsp_mapping")
        if isinstance(kmap, dict):
            c = kmap.get("standard_setting_certificate")
            if isinstance(c, dict):
                cert = c

    sg_number = 0
    sg_symbol = ""
    hall_number = 0
    hall_symbol = ""
    centering_type = ""
    primitive_conventional_relation = ""
    transform_provenance = ""
    validation_status = "not_evaluated"
    operation_mapping_status = "not_attempted"
    affine_validation_status = "not_attempted"

    # Also check subspace_space_group for SG identity.
    ssg = candidate.get("subspace_space_group")
    if isinstance(ssg, dict):
        sn = ssg.get("candidate_space_group_number")
        if isinstance(sn, int) and not isinstance(sn, bool):
            sg_number = int(sn)
        sy = ssg.get("candidate_space_group_symbol")
        if isinstance(sy, str) and sy:
            sg_symbol = str(sy)

    hn = cert.get("hall_number")
    if isinstance(hn, int) and not isinstance(hn, bool):
        hall_number = int(hn)
    hs = cert.get("hall_symbol")
    if isinstance(hs, str) and hs:
        hall_symbol = str(hs)
    vs = cert.get("validation_status")
    if isinstance(vs, str) and vs:
        validation_status = str(vs)
    ct = cert.get("centering_type")
    if isinstance(ct, str) and ct:
        centering_type = str(ct)
    pcr = cert.get("primitive_conventional_relation")
    if isinstance(pcr, str) and pcr:
        primitive_conventional_relation = str(pcr)
    tp = cert.get("transform_provenance")
    if isinstance(tp, str) and tp:
        transform_provenance = str(tp)
    oms = cert.get("operation_mapping_status")
    if isinstance(oms, str) and oms:
        operation_mapping_status = str(oms)
    avs = cert.get("translation_validation_status")
    if isinstance(avs, str) and avs:
        affine_validation_status = str(avs)

    def _opt_int(value: object) -> int | None:
        return int(value) if isinstance(value, int) \
            and not isinstance(value, bool) else None

    affine_matched_operations = _opt_int(cert.get("matched_affine_operations"))
    affine_total_operations = _opt_int(cert.get("total_parent_operations"))
    affine_mismatch_count = _opt_int(cert.get("mismatched_translation_count"))
    # Distinguish missing/absent from empty: None = evidence absent.
    # The list may still be empty — that is explicit evidence (no ingredients
    # missing).  A missing key / None is unknown.
    missing = cert.get("missing_affine_ingredients", _SENTINEL_MISSING)
    affine_missing_ingredients = (
        None
        if missing is _SENTINEL_MISSING
        else _normalize_strict_string_list(missing)
    )

    closure = cert.get("operation_closure_validated")
    operation_closure_validated = closure if isinstance(closure, bool) else None
    standard_closure = cert.get("standard_operation_closure_validated")
    standard_operation_closure_validated = (
        standard_closure if isinstance(standard_closure, bool) else None
    )

    std_op_count = _opt_int(cert.get("standard_setting_operation_count"))
    req_op_count = _opt_int(cert.get("required_operation_id_count"))
    affine_operation_map = _normalize_affine_operation_map(
        cert.get("affine_operation_map", _SENTINEL_MISSING)
    )
    affine_unmatched_parents = _normalize_unmatched_parent_operations(
        cert.get("unmatched_parent_operations", _SENTINEL_MISSING)
    )
    affine_unused_std = _normalize_strict_int_list(
        cert.get("unused_standard_operation_indices", _SENTINEL_MISSING),
        unique=True,
    )
    affine_required_operation_ids = _normalize_strict_int_list(
        cert.get("parent_basis_operation_ids", _SENTINEL_MISSING),
        unique=True,
    )
    canonical_setting_status = (
        str(cert.get("canonical_setting_status"))
        if isinstance(cert.get("canonical_setting_status"), str)
        else "not_evaluated"
    )
    canonical_setting_source = (
        str(cert.get("canonical_setting_source"))
        if isinstance(cert.get("canonical_setting_source"), str)
        else ""
    )
    canonical_hall_numbers = _normalize_strict_int_list(
        cert.get("canonical_hall_numbers", _SENTINEL_MISSING), unique=True,
    )
    canonical_candidate_hall_numbers = _normalize_strict_int_list(
        cert.get("canonical_candidate_hall_numbers", _SENTINEL_MISSING),
        unique=True,
    )
    centering_coset_count = _opt_int(cert.get("centering_coset_count"))
    primitive_conventional_index = _opt_int(
        cert.get("primitive_conventional_index")
    )
    expanded_parent_operation_count = _opt_int(
        cert.get("expanded_parent_operation_count")
    )
    matched_expanded_operations = _opt_int(
        cert.get("matched_expanded_operations")
    )
    centered_affine_operation_map = _normalize_centered_affine_operation_map(
        cert.get("centered_affine_operation_map", _SENTINEL_MISSING)
    )
    affine_unmatched_centered_pairs = _normalize_unmatched_centered_pairs(
        cert.get("unmatched_centered_operation_pairs", _SENTINEL_MISSING)
    )

    transform_key = _normalize_transform(
        cert.get("parent_to_standard_direct_transform")
    )
    origin_shift_key = _normalize_origin_shift(
        cert.get("origin_shift_fractional")
    )
    centering_vectors_key = _normalize_centering_vectors(
        cert.get("centering_vectors")
    )

    return _SettingIdentity(
        sg_number=sg_number,
        sg_symbol=sg_symbol,
        hall_number=hall_number,
        hall_symbol=hall_symbol,
        transform_key=transform_key,
        origin_shift_key=origin_shift_key,
        centering_type=centering_type,
        centering_vectors_key=centering_vectors_key,
        primitive_conventional_relation=primitive_conventional_relation,
        transform_provenance=transform_provenance,
        validation_status=validation_status,
        operation_mapping_status=operation_mapping_status,
        affine_validation_status=affine_validation_status,
        affine_matched_operations=affine_matched_operations,
        affine_total_operations=affine_total_operations,
        affine_mismatch_count=affine_mismatch_count,
        affine_missing_ingredients=affine_missing_ingredients,
        affine_standard_setting_op_count=std_op_count,
        affine_operation_map=affine_operation_map,
        affine_unmatched_parents=affine_unmatched_parents,
        affine_unused_std=affine_unused_std,
        affine_required_operation_ids=affine_required_operation_ids,
        affine_required_op_count=req_op_count,
        operation_closure_validated=operation_closure_validated,
        canonical_setting_status=canonical_setting_status,
        canonical_setting_source=canonical_setting_source,
        canonical_hall_numbers=canonical_hall_numbers,
        canonical_candidate_hall_numbers=canonical_candidate_hall_numbers,
        centering_coset_count=centering_coset_count,
        primitive_conventional_index=primitive_conventional_index,
        expanded_parent_operation_count=expanded_parent_operation_count,
        matched_expanded_operations=matched_expanded_operations,
        centered_affine_operation_map=centered_affine_operation_map,
        affine_unmatched_centered_pairs=affine_unmatched_centered_pairs,
        standard_operation_closure_validated=(
            standard_operation_closure_validated
        ),
    )


def _certificate_identity(
    cands: list[dict[str, object]],
) -> dict[str, object]:
    """Build certificate-identity dict from merged candidates.

    All candidates in one group share the same ``_SettingIdentity``, so the
    canonical identity is taken from the first candidate.  The complete
    setting-level fields are serialized so that the promotion function can
    validate affine evidence.
    """
    fps = [_certificate_fingerprint(c) for c in cands]
    fp0 = fps[0] if fps else None
    hall_numbers = sorted({fp.hall_number for fp in fps if fp.hall_number})
    hall_symbols = sorted({fp.hall_symbol for fp in fps if fp.hall_symbol})
    validation_statuses = sorted({fp.validation_status for fp in fps})
    centering_types = sorted({fp.centering_type for fp in fps if fp.centering_type})
    distinct = len({fp._hash for fp in fps})

    result: dict[str, object] = {
        "hall_numbers": hall_numbers,
        "hall_symbols": hall_symbols,
        "centering_types": centering_types,
        "certificate_validation_statuses": validation_statuses,
        "any_unresolved": (
            "unresolved" in validation_statuses
            or "not_evaluated" in validation_statuses
            or "rejected" in validation_statuses
        ),
        "distinct_setting_identities": distinct,
    }

    # Serialize the canonical _SettingIdentity fields.
    if fp0 is not None:
        result["sg_number"] = fp0.sg_number
        result["sg_symbol"] = fp0.sg_symbol
        result["hall_number"] = fp0.hall_number
        result["hall_symbol"] = fp0.hall_symbol
        result["centering_type"] = fp0.centering_type
        result["primitive_conventional_relation"] = (
            fp0.primitive_conventional_relation
        )
        result["transform_provenance"] = fp0.transform_provenance
        result["validation_status"] = fp0.validation_status
        result["operation_mapping_status"] = fp0.operation_mapping_status
        result["affine_validation_status"] = fp0.affine_validation_status
        result["affine_matched_operations"] = fp0.affine_matched_operations
        result["affine_total_operations"] = fp0.affine_total_operations
        result["affine_mismatch_count"] = fp0.affine_mismatch_count
        result["affine_missing_ingredients"] = (
            list(fp0.affine_missing_ingredients)
            if fp0.affine_missing_ingredients is not None else None
        )
        result["affine_standard_setting_op_count"] = \
            fp0.affine_standard_setting_op_count
        result["affine_operation_map"] = (
            {str(key): value for key, value in fp0.affine_operation_map}
            if fp0.affine_operation_map is not None else None
        )
        result["affine_required_operation_ids"] = (
            list(fp0.affine_required_operation_ids)
            if fp0.affine_required_operation_ids is not None else None
        )
        result["affine_required_op_count"] = fp0.affine_required_op_count
        result["affine_unmatched_parent_operations"] = (
            list(fp0.affine_unmatched_parents)
            if fp0.affine_unmatched_parents is not None else None
        )
        result["affine_unused_standard_operation_indices"] = (
            list(fp0.affine_unused_std)
            if fp0.affine_unused_std is not None else None
        )
        result["operation_closure_validated"] = fp0.operation_closure_validated
        result["canonical_setting_status"] = fp0.canonical_setting_status
        result["canonical_setting_source"] = fp0.canonical_setting_source
        result["canonical_hall_numbers"] = (
            list(fp0.canonical_hall_numbers)
            if fp0.canonical_hall_numbers is not None else None
        )
        result["canonical_candidate_hall_numbers"] = (
            list(fp0.canonical_candidate_hall_numbers)
            if fp0.canonical_candidate_hall_numbers is not None else None
        )
        result["centering_coset_count"] = fp0.centering_coset_count
        result["primitive_conventional_index"] = (
            fp0.primitive_conventional_index
        )
        result["expanded_parent_operation_count"] = (
            fp0.expanded_parent_operation_count
        )
        result["matched_expanded_operations"] = fp0.matched_expanded_operations
        result["centered_affine_operation_map"] = (
            [
                {
                    "parent_operation_id": parent_id,
                    "centering_coset_index": coset_index,
                    "standard_operation_index": standard_index,
                }
                for parent_id, coset_index, standard_index
                in fp0.centered_affine_operation_map
            ]
            if fp0.centered_affine_operation_map is not None else None
        )
        result["affine_unmatched_centered_operation_pairs"] = (
            [
                {
                    "parent_operation_id": parent_id,
                    "centering_coset_index": coset_index,
                }
                for parent_id, coset_index
                in fp0.affine_unmatched_centered_pairs
            ]
            if fp0.affine_unmatched_centered_pairs is not None else None
        )
        result["standard_operation_closure_validated"] = (
            fp0.standard_operation_closure_validated
        )
        if fp0.transform_key is not None:
            result["normalized_direct_transform"] = list(
                list(row) for row in fp0.transform_key
            )
        if fp0.origin_shift_key is not None:
            result["normalized_origin_shift"] = list(fp0.origin_shift_key)
        if fp0.centering_vectors_key is not None:
            result["normalized_centering_vectors"] = [
                list(v) for v in fp0.centering_vectors_key
            ]

    return result
