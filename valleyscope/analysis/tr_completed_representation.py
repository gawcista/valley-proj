"""Producer-owned C-prime evidence for time-reversal-completed HSP rows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from valleyscope.analysis.scoped_representation_evidence import (
    build_scoped_representation_evidence,
)
from valleyscope.symmetry.plane_wave_action import reciprocal_grid_identity


def attach_tr_completed_representation_evidence(
    *,
    time_reversal_orbit_report: dict[str, object],
    cprime_validation_context: dict[str, object],
) -> dict[str, object]:
    """Attach recomputable ``tr_completed`` C-prime identities to TR rows.

    The target HSP basis is constructed as the exact time-reversed image of
    observed source data.  No separately sampled time-reversed HSP arm is
    required or claimed.
    """
    report = deepcopy(time_reversal_orbit_report)
    raw_orbits = report.get("valley_orbits")
    valley_mapping = report.get("time_reversal_valley_mapping")
    if not isinstance(raw_orbits, list) or not isinstance(
        valley_mapping, Mapping
    ):
        return report

    for orbit in raw_orbits:
        if not isinstance(orbit, dict):
            continue
        blockers = [
            str(value) for value in orbit.get("blockers", [])
            if isinstance(value, str) and value
        ]
        if orbit.get("mapping_type") != "exchanged":
            orbit["cprime_scope_status"] = "not_constructed"
            continue
        members = orbit.get("members")
        records_by_valley = orbit.get(
            "unitary_valley_irrep_completion_records"
        )
        full_hsps = orbit.get("full_unitary_source_hsp_labels")
        independent_hsps = orbit.get(
            "independent_time_reversal_hsp_labels"
        )
        hsp_mapping = _hsp_mapping(
            orbit.get("time_reversal_hsp_orbits")
        )
        if (
            not isinstance(members, list)
            or len(members) != 2
            or not all(isinstance(value, str) and value for value in members)
            or not isinstance(records_by_valley, Mapping)
            or not isinstance(full_hsps, list)
            or not isinstance(independent_hsps, list)
            or set(full_hsps) != set(hsp_mapping)
        ):
            blockers.append("tr_completed_scope_inputs_malformed")
            _block_orbit(orbit, blockers)
            continue

        links_by_hsp: dict[str, dict[str, object]] = {}
        built_pairs: set[frozenset[str]] = set()
        for source_hsp in full_hsps:
            target_hsp = hsp_mapping.get(source_hsp)
            if not isinstance(target_hsp, str):
                blockers.append(
                    f"tr_completed_hsp_partner_missing:{source_hsp}"
                )
                continue
            pair = frozenset((source_hsp, target_hsp))
            if pair in built_pairs:
                continue
            built_pairs.add(pair)
            local_contexts = _local_contexts_for_hsp_pair(
                records_by_valley=records_by_valley,
                hsp_pair=pair,
                cprime_validation_context=cprime_validation_context,
            )
            merged_inputs = _merge_local_contexts(
                local_contexts=local_contexts,
                valley_members=members,
            )
            if merged_inputs is None:
                blockers.append(
                    f"tr_completed_local_context_missing:{source_hsp}"
                )
                continue
            antiunitary = _antiunitary_inputs(
                local_inputs=merged_inputs,
                source_hsp=source_hsp,
                target_hsp=target_hsp,
                hsp_mapping=hsp_mapping,
                source_valley=members[0],
                target_valley=str(valley_mapping.get(members[0], "")),
            )
            if antiunitary is None:
                blockers.append(
                    f"tr_completed_antiunitary_inputs_malformed:{source_hsp}"
                )
                continue
            scoped_inputs = dict(merged_inputs)
            scoped_inputs.update(
                {
                    "kpoint_label": (
                        f"TR:{source_hsp}->{target_hsp}"
                    ),
                    "scope_kind": "tr_completed",
                    "source_valleys": tuple(members),
                    "valley_orbit": tuple(members),
                    "antiunitary_evidence": antiunitary,
                }
            )
            scoped = build_scoped_representation_evidence(
                **scoped_inputs
            ).to_record()
            if scoped.get("status") != "passed":
                blockers.extend(
                    f"tr_completed_scope_failed:{source_hsp}:{reason}"
                    for reason in scoped.get("reason_codes", [])
                    if isinstance(reason, str)
                )
                continue
            identity = scoped["evidence_identity"]
            cprime_validation_context[str(identity)] = {
                "record": scoped,
                "raw_inputs": scoped_inputs,
            }
            links = {
                "spinor_source_basis_certificate_identity": scoped.get(
                    "source_basis_certificate_identity"
                ),
                "double_space_group_lift_certificate_identity": scoped.get(
                    "double_space_group_lift_certificate_identity"
                ),
                "scoped_representation_evidence_identity": identity,
            }
            for hsp in pair:
                links_by_hsp[hsp] = dict(links)

        if set(links_by_hsp) != set(full_hsps):
            blockers.append(
                "tr_completed_scoped_representation_evidence_missing"
            )
            _block_orbit(orbit, blockers)
            continue
        _attach_record_links(
            records_by_valley=records_by_valley,
            links_by_hsp=links_by_hsp,
        )
        orbit["tr_completed_cprime_identity_by_hsp"] = {
            hsp: dict(links_by_hsp[hsp]) for hsp in full_hsps
        }
        orbit["joint_cprime_identity_by_hsp"] = {
            hsp: dict(links_by_hsp[hsp]) for hsp in independent_hsps
        }
        orbit["cprime_scope_status"] = "passed"
        orbit["readiness_blockers"] = _deduplicate(
            value for value in orbit.get("readiness_blockers", [])
            if isinstance(value, str)
        )
        orbit["blockers"] = _deduplicate(blockers)
        orbit["status"] = (
            "validated" if not orbit["blockers"] else "blocked"
        )

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


def _hsp_mapping(raw_orbits: object) -> dict[str, str]:
    if not isinstance(raw_orbits, list):
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
        ):
            return {}
        if len(members) == 1:
            mapping[members[0]] = members[0]
        else:
            mapping[members[0]] = members[1]
            mapping[members[1]] = members[0]
    return mapping


def _local_contexts_for_hsp_pair(
    *,
    records_by_valley: Mapping[object, object],
    hsp_pair: frozenset[str],
    cprime_validation_context: Mapping[str, object],
) -> list[dict[str, object]]:
    contexts: dict[str, dict[str, object]] = {}
    for raw_by_hsp in records_by_valley.values():
        if not isinstance(raw_by_hsp, Mapping):
            continue
        for hsp in hsp_pair:
            records = raw_by_hsp.get(hsp, [])
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                candidate = record.get("source_candidate_provenance")
                provenance = (
                    candidate.get("irrep_source_provenance")
                    if isinstance(candidate, Mapping) else None
                )
                cprime = (
                    provenance.get("cprime")
                    if isinstance(provenance, Mapping) else None
                )
                identity = (
                    cprime.get(
                        "scoped_representation_evidence_identity"
                    )
                    if isinstance(cprime, Mapping) else None
                )
                context = cprime_validation_context.get(str(identity))
                if isinstance(identity, str) and isinstance(context, Mapping):
                    contexts[identity] = dict(context)
    return list(contexts.values())


def _merge_local_contexts(
    *,
    local_contexts: list[dict[str, object]],
    valley_members: list[str],
) -> dict[str, object] | None:
    raw_inputs = [
        context.get("raw_inputs")
        for context in local_contexts
        if isinstance(context.get("raw_inputs"), Mapping)
    ]
    by_valley: dict[str, Mapping[str, object]] = {}
    for raw in raw_inputs:
        source_valleys = raw.get("source_valleys")
        if (
            isinstance(source_valleys, (list, tuple))
            and len(source_valleys) == 1
            and source_valleys[0] in valley_members
        ):
            by_valley[str(source_valleys[0])] = raw
    if set(by_valley) != set(valley_members):
        return None
    base = by_valley[valley_members[0]]
    if any(
        not _same_local_unitary_inputs(base, by_valley[valley])
        for valley in valley_members[1:]
    ):
        return None
    merged = dict(base)
    projectors: dict[str, np.ndarray] = {}
    bases: dict[str, np.ndarray] = {}
    for valley in valley_members:
        raw = by_valley[valley]
        raw_projectors = raw.get("projectors")
        raw_bases = raw.get("valley_bases")
        if (
            not isinstance(raw_projectors, Mapping)
            or valley not in raw_projectors
            or not isinstance(raw_bases, Mapping)
            or valley not in raw_bases
        ):
            return None
        projectors[valley] = np.asarray(
            raw_projectors[valley], dtype=np.complex128
        )
        bases[valley] = np.asarray(
            raw_bases[valley], dtype=np.complex128
        )
    merged["projectors"] = projectors
    merged["valley_bases"] = bases
    return merged


def _same_local_unitary_inputs(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> bool:
    scalar_keys = (
        "extracted_wavefunction_payload_identity",
        "kpoint_label",
        "required_operation_ids",
        "source_basis_record",
        "lift_record",
        "lift_validation_inputs",
        "valley_mappings",
    )
    if any(
        not _equal_nested(left.get(key), right.get(key))
        for key in scalar_keys
    ):
        return False
    array_keys = ("kpoint_frac", "target_coefficients")
    if any(
        not np.array_equal(np.asarray(left.get(key)), np.asarray(right.get(key)))
        for key in array_keys
    ):
        return False
    for key in ("representations", "plane_wave_evidence"):
        left_map = left.get(key)
        right_map = right.get(key)
        if (
            not isinstance(left_map, Mapping)
            or not isinstance(right_map, Mapping)
            or set(left_map) != set(right_map)
            or any(
                not _equal_nested(
                    left_map[operation_id], right_map[operation_id]
                )
                for operation_id in left_map
            )
        ):
            return False
    return True


def _equal_nested(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right))
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _equal_nested(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(
        right, (list, tuple)
    ):
        return len(left) == len(right) and all(
            _equal_nested(a, b) for a, b in zip(left, right)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _antiunitary_inputs(
    *,
    local_inputs: Mapping[str, object],
    source_hsp: str,
    target_hsp: str,
    hsp_mapping: Mapping[str, str],
    source_valley: str,
    target_valley: str,
) -> dict[str, object] | None:
    representations = local_inputs.get("representations")
    plane_wave = local_inputs.get("plane_wave_evidence")
    if (
        not target_valley
        or not isinstance(representations, Mapping)
        or not representations
        or not isinstance(plane_wave, Mapping)
        or not plane_wave
    ):
        return None
    first_matrix = np.asarray(
        next(iter(representations.values())), dtype=np.complex128
    )
    first_plane_wave = next(iter(plane_wave.values()))
    if (
        first_matrix.ndim != 2
        or first_matrix.shape[0] != first_matrix.shape[1]
        or not isinstance(first_plane_wave, Mapping)
    ):
        return None
    try:
        source_q = np.asarray(first_plane_wave["q_cart"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if (
        source_q.ndim != 2
        or source_q.shape[1:] != (3,)
        or not np.all(np.isfinite(source_q))
    ):
        return None
    target_q = -source_q
    forward = np.eye(first_matrix.shape[0], dtype=np.complex128)
    reverse = -np.eye(first_matrix.shape[0], dtype=np.complex128)
    source_representations = {
        int(operation_id): np.asarray(matrix, dtype=np.complex128)
        for operation_id, matrix in representations.items()
    }
    target_representations = {
        operation_id: (
            forward @ matrix.conj() @ forward.conj().T
        )
        for operation_id, matrix in source_representations.items()
    }
    return {
        "source_valley": source_valley,
        "target_valley": target_valley,
        "source_hsp_label": source_hsp,
        "target_hsp_label": target_hsp,
        "time_reversal_hsp_mapping": dict(hsp_mapping),
        "construction_kind": "observed_to_inferred",
        "source_reciprocal_grid_vectors_cart": source_q,
        "target_reciprocal_grid_vectors_cart": target_q,
        "source_reciprocal_grid_identity": reciprocal_grid_identity(
            source_q
        ),
        "target_reciprocal_grid_identity": reciprocal_grid_identity(
            target_q
        ),
        "source_to_target_grid_map": list(range(len(source_q))),
        "forward_sewing_matrix": forward,
        "reverse_sewing_matrix": reverse,
        "expected_square_sign": -1,
        "source_unitary_representations": source_representations,
        "target_unitary_representations": target_representations,
    }


def _attach_record_links(
    *,
    records_by_valley: Mapping[object, object],
    links_by_hsp: Mapping[str, dict[str, object]],
) -> None:
    for raw_by_hsp in records_by_valley.values():
        if not isinstance(raw_by_hsp, Mapping):
            continue
        for hsp, records in raw_by_hsp.items():
            links = links_by_hsp.get(str(hsp))
            if links is None or not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                candidate = record.get("source_candidate_provenance")
                provenance = (
                    candidate.get("irrep_source_provenance")
                    if isinstance(candidate, dict) else None
                )
                if isinstance(provenance, dict):
                    provenance["cprime"] = dict(links)


def _block_orbit(orbit: dict[str, object], blockers: list[str]) -> None:
    combined = _deduplicate(blockers)
    orbit["cprime_scope_status"] = "blocked"
    orbit["readiness_blockers"] = _deduplicate(
        [
            *(
                value for value in orbit.get("readiness_blockers", [])
                if isinstance(value, str)
            ),
            *combined,
        ]
    )
    orbit["blockers"] = combined
    orbit["status"] = "blocked"


def _deduplicate(values) -> list[str]:
    return list(dict.fromkeys(values))
