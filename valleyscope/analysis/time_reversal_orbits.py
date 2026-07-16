"""Center-derived time-reversal mapping of valley-projected subspaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np

from valleyscope.analysis.reduced_ebr_solver import (
    derive_coefficient_bounds,
    search_nonnegative_witnesses,
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
) -> dict[str, object]:
    """Complete trusted sampled rows on exchanged TR valley orbits."""
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
        blockers: list[str] = []
        mapping_type = str(raw_orbit.get("mapping_type", ""))
        if mapping_type != "exchanged" or len(members) != 2:
            blockers.append(
                "antiunitary_corepresentation_required_not_proven"
            )

        source_reports = [
            source_irrep_orbits_by_valley.get(member, {})
            for member in members
        ]
        grey_reports = [
            grey_source_by_valley.get(member, {}) for member in members
        ]
        if any(report.get("status") != "validated" for report in source_reports):
            blockers.append("time_reversal_source_irrep_orbits_not_validated")
        if any(report.get("status") != "validated" for report in grey_reports):
            blockers.append("grey_group_time_reversal_source_not_validated")
        if source_reports and any(
            report.get("time_reversal_hsp_mapping")
            != source_reports[0].get("time_reversal_hsp_mapping")
            or report.get("irrep_partner_by_label")
            != source_reports[0].get("irrep_partner_by_label")
            for report in source_reports[1:]
        ):
            blockers.append("valley_source_time_reversal_models_disagree")
        if grey_reports and any(
            report.get("grey_bns_number")
            != grey_reports[0].get("grey_bns_number")
            or report.get("grey_unitary_restriction_by_irrep")
            != grey_reports[0].get("grey_unitary_restriction_by_irrep")
            for report in grey_reports[1:]
        ):
            blockers.append("valley_grey_source_models_disagree")

        source_report = source_reports[0] if source_reports else {}
        grey_report = grey_reports[0] if grey_reports else {}
        hsp_mapping = source_report.get("time_reversal_hsp_mapping", {})
        irrep_mapping = source_report.get("irrep_partner_by_label", {})
        independent_hsps = source_report.get("independent_hsp_labels", [])
        if not isinstance(hsp_mapping, Mapping) or not isinstance(
            irrep_mapping, Mapping
        ) or not isinstance(independent_hsps, list):
            blockers.append("time_reversal_source_mapping_malformed")
            hsp_mapping = {}
            irrep_mapping = {}
            independent_hsps = []

        actual = _candidate_unitary_counts(candidates, members)
        for member in members:
            for hsp in independent_hsps:
                if not actual.get(member, {}).get(str(hsp)):
                    blockers.append(
                        f"missing_trusted_independent_hsp:{member}:{hsp}"
                    )

        inferred: dict[str, dict[str, dict[str, int]]] = {
            member: {} for member in members
        }
        if len(members) == 2:
            partner_valley = {members[0]: members[1], members[1]: members[0]}
            for valley, by_hsp in actual.items():
                for hsp, counts in by_hsp.items():
                    partner_hsp = hsp_mapping.get(hsp)
                    if not isinstance(partner_hsp, str):
                        blockers.append(
                            f"missing_time_reversal_hsp_partner:{hsp}"
                        )
                        continue
                    target = inferred[partner_valley[valley]].setdefault(
                        partner_hsp, {}
                    )
                    for irrep, multiplicity in counts.items():
                        partner_irrep = irrep_mapping.get(irrep)
                        if not isinstance(partner_irrep, str):
                            blockers.append(
                                f"missing_time_reversal_irrep_partner:{irrep}"
                            )
                            continue
                        target[partner_irrep] = (
                            target.get(partner_irrep, 0) + multiplicity
                        )

        completed: dict[str, dict[str, dict[str, int]]] = {
            member: {
                hsp: dict(counts) for hsp, counts in actual.get(member, {}).items()
            }
            for member in members
        }
        for member in members:
            for hsp, inferred_counts in inferred.get(member, {}).items():
                actual_counts = actual.get(member, {}).get(hsp)
                if actual_counts is not None and actual_counts != inferred_counts:
                    blockers.append(
                        "time_reversal_multiplicity_or_irrep_mismatch:"
                        f"{member}:{hsp}:actual={actual_counts}:"
                        f"inferred={inferred_counts}"
                    )
                    continue
                completed[member].setdefault(hsp, dict(inferred_counts))

        full_hsps = list(hsp_mapping)
        for member in members:
            for hsp in full_hsps:
                if not completed.get(member, {}).get(hsp):
                    blockers.append(
                        f"incomplete_time_reversal_component:{member}:{hsp}"
                    )

        combined: dict[str, dict[str, int]] = {}
        for member in members:
            for hsp, counts in completed.get(member, {}).items():
                target = combined.setdefault(hsp, {})
                for irrep, multiplicity in counts.items():
                    target[irrep] = target.get(irrep, 0) + multiplicity

        grey_counts: dict[str, dict[str, int]] = {}
        if not blockers:
            grey_counts, decomposition_blockers = _decompose_grey_counts(
                unitary_counts_by_hsp=combined,
                grey_restrictions=grey_report.get(
                    "grey_unitary_restriction_by_irrep", {}
                ),
                grey_hsp_by_irrep=grey_report.get(
                    "grey_source_hsp_by_irrep", {}
                ),
                unitary_hsp_by_irrep=grey_report.get(
                    "unitary_source_hsp_by_irrep", {}
                ),
            )
            blockers.extend(decomposition_blockers)

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
            "representative": raw_orbit.get("representative", members[0]),
            "members": members,
            "mapping_type": mapping_type,
            "status": "validated" if not blockers else "blocked",
            "unitary_valley_irreps": actual,
            "time_reversal_completed_unitary_valley_irreps": completed,
            "time_reversal_hsp_orbits": source_report.get(
                "time_reversal_hsp_orbits", []
            ),
            "time_reversal_irrep_pairing": dict(irrep_mapping),
            "full_unitary_source_hsp_labels": full_hsps,
            "independent_time_reversal_hsp_labels": independent_hsps,
            "grey_bns_number": grey_report.get("grey_bns_number"),
            "grey_irrep_multiplicities_by_hsp": grey_counts,
            "irreps_by_kpoint": irreps_by_kpoint,
            "expected_hsps": list(independent_hsps),
            "blockers": _deduplicate(blockers),
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
        "valley_orbits": orbit_rows,
        "blockers": _deduplicate([
            blocker for row in orbit_rows for blocker in row["blockers"]
        ]),
    }


def _candidate_unitary_counts(
    candidates: Sequence[object],
    members: Sequence[str],
) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {
        member: {} for member in members
    }
    for raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        valley = raw.get("valley")
        irrep = raw.get("matched_irrep")
        multiplicity = raw.get("irrep_multiplicity", 1)
        provenance = raw.get("irrep_source_provenance", {})
        classification = raw.get("projected_hsp_classification", {})
        source_hsp = (
            classification.get("source_hsp_label")
            if isinstance(classification, Mapping) else None
        ) or (
            provenance.get("source_hsp_label")
            if isinstance(provenance, Mapping) else None
        )
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
        counts = out[str(valley)].setdefault(source_hsp, {})
        counts[irrep] = counts.get(irrep, 0) + multiplicity
    return out


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
