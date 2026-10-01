"""Conditional Chern residues of valley subspaces from physical rotations.

Only local, producer-validated representation matrices are used.  This module
does not establish a smooth, constant-rank valley bundle over the whole mBZ.
The convention is active positive rotations about Cartesian +z and
A = i <u|d u>, with positively oriented (kx, ky).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from valleyscope.analysis.rotation_chern_math import (
    rotation_chern_residue,
    rotation_fixed_point_orbits,
)
from valleyscope.analysis.scoped_representation_evidence import (
    validate_scoped_representation_evidence_record,
)
from valleyscope.analysis.tr_irrep_completion import (
    validate_tr_irrep_completion_certificate,
)
from valleyscope.geometry.lattice import cart_rotation_from_fractional
from valleyscope.symmetry.double_space_group_lift import spin_lift_from_orthogonal


GEOMETRY_TOLERANCE = 1.0e-7
ROTATION_PHASE_TOLERANCE = 1.0e-5
INTERPRETATION = (
    "Rotation-eigenvalue constraint, conditional on a globally defined, smooth "
    "constant-rank valley subspace with Bloch boundary sewing and the stated "
    "rotation symmetry. Not a calculation of the full integer Chern number."
)


def _matrix(value: object, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError("rotation_geometry_malformed")
    return result


def _mapping(value: object) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError("rotation_evidence_mapping_malformed")
    return value


def find_valley_rotations(
    operations: Sequence[Mapping[str, Any]], valley: str,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Find actual positive Cnz operations, not space-group-name prototypes."""
    result = []
    seen_ids = set()
    for operation in operations:
        operation = _mapping(operation)
        operation_id = operation.get("operation_id")
        if (not isinstance(operation_id, int) or isinstance(operation_id, bool)
                or operation_id in seen_ids):
            raise ValueError("rotation_operation_ids_malformed")
        seen_ids.add(operation_id)
        if _mapping(operation.get("sector_mapping", {})).get(valley) != valley:
            continue
        rotation = _matrix(operation.get("rotation_cart"), (3, 3))
        if not np.allclose(rotation.T @ rotation, np.eye(3),
                           atol=GEOMETRY_TOLERANCE, rtol=0):
            raise ValueError("rotation_cart_not_orthogonal")
        if (not np.isclose(np.linalg.det(rotation), 1, atol=GEOMETRY_TOLERANCE, rtol=0)
                or not np.allclose(rotation[:, 2], [0, 0, 1],
                                   atol=GEOMETRY_TOLERANCE, rtol=0)):
            continue
        angle = float(np.arctan2(rotation[1, 0], rotation[0, 0]) % (2 * np.pi))
        for order in (6, 4, 3, 2):
            if abs(angle - 2 * np.pi / order) <= GEOMETRY_TOLERANCE:
                full_r, full_t = _affine_power(operation, order)
                # A screw is not Cnz. Its finite-order powers may nevertheless
                # supply an eligible pure rotation elsewhere in the inventory.
                if (np.array_equal(full_r, np.eye(3))
                        and np.allclose(full_t, 0, atol=GEOMETRY_TOLERANCE, rtol=0)):
                    result.append((order, operation))
                break
    return sorted(result, key=lambda row: (-row[0], row[1]["operation_id"]))


def _affine_power(operation: Mapping[str, Any], power: int):
    rotation = _matrix(operation.get("rotation_frac"), (3, 3))
    if not np.array_equal(rotation, np.rint(rotation)):
        raise ValueError("rotation_fractional_not_integer")
    translation = _matrix(operation.get("translation_frac"), (3,))
    r, t = np.eye(3), np.zeros(3)
    for _ in range(power):
        t = r @ translation + t
        r = r @ rotation
    return r, t


def _equivalent(k: np.ndarray, points: list, tolerance: float) -> bool:
    if abs(k[2] - round(float(k[2]))) > tolerance:
        return False
    return any(np.max(np.abs(k[:2] - point - np.rint(k[:2] - point))) <= tolerance
               for point in points)


def _source_context(kpoint, valley, matches, contexts, cache, *, spinful):
    key = (kpoint, valley)
    if key in cache:
        return cache[key]
    match = _mapping(_mapping(matches.get(kpoint, {})).get(valley, {}))
    if (match.get("matching_status") != "matched"
            or match.get("readiness_level") != "trusted"
            or match.get("diagnostic_only") is not False):
        raise ValueError(f"untrusted_rotation_hsp:{kpoint}:{valley}")
    provenance = _mapping(match.get("source_payload_provenance", {}))
    identities = _mapping(provenance.get("cprime", {}))
    identity = identities.get("scoped_representation_evidence_identity")
    context = _mapping(_mapping(contexts).get(identity, {}))
    record = _mapping(context.get("record", {}))
    raw = _mapping(context.get("raw_inputs", {}))
    scope = _mapping(record.get("scope", {}))
    if (not raw or record.get("evidence_identity") != identity
            or scope.get("scope_kind") != "local_irrep"
            or scope.get("kpoint_label") != kpoint
            or scope.get("source_valleys") != [valley]
            or record.get("source_basis_certificate_identity")
            != identities.get("spinor_source_basis_certificate_identity")
            or record.get("double_space_group_lift_certificate_identity")
            != identities.get("double_space_group_lift_certificate_identity")):
        raise ValueError(f"rotation_cprime_scope_mismatch:{kpoint}:{valley}")
    validation = validate_scoped_representation_evidence_record(record, **raw)
    if validation.status != "passed":
        raise ValueError(f"rotation_cprime_revalidation_failed:{kpoint}:{valley}")
    source = _mapping(raw.get("source_basis_record"))
    nspinor = _mapping(source.get("coefficient_layout")).get("nspinor")
    if not isinstance(spinful, bool) or nspinor != (2 if spinful else 1):
        raise ValueError(f"rotation_spinful_source_layout_mismatch:{kpoint}:{valley}")
    cache[key] = (raw, identity, match)
    return cache[key]


def _physical_determinant(raw, valley, generator, power, *, spinful):
    """Use the exact Seitz power, including its central sign and Bloch phase."""
    target_r, target_t = _affine_power(generator, power)
    generator_spin = spin_lift_from_orthogonal(generator["rotation_cart"])
    target_spin = np.linalg.matrix_power(generator_spin, power)
    inventory = raw["lift_validation_inputs"]["expected_operations"]
    required = raw["required_operation_ids"]
    candidates = []
    for operation in inventory:
        op_id = operation["operation_id"]
        if op_id not in required:
            continue
        if not np.array_equal(operation["rotation_frac"], target_r):
            continue
        shift = target_t - np.asarray(operation["translation_frac"])
        if not np.allclose(shift, np.rint(shift), atol=GEOMETRY_TOLERANCE, rtol=0):
            continue
        if raw["valley_mappings"].get(op_id, {}).get(valley) != valley:
            continue
        candidates.append((op_id, operation, np.rint(shift)))
    if not candidates:
        raise ValueError("rotation_power_representation_missing")
    op_id, operation, shift = min(candidates, key=lambda x: x[0])
    basis = np.asarray(raw["valley_bases"][valley], dtype=complex)
    block = basis.conj().T @ raw["representations"][op_id] @ basis
    rank = int(block.shape[0])
    if rank < 1 or block.shape != (rank, rank) or not np.all(np.isfinite(block)):
        raise ValueError("rotation_valley_block_malformed")
    sign = 1
    if spinful:
        actual_spin = spin_lift_from_orthogonal(operation["rotation_cart"])
        if np.allclose(target_spin, actual_spin, atol=GEOMETRY_TOLERANCE, rtol=0):
            sign = 1
        elif np.allclose(target_spin, -actual_spin, atol=GEOMETRY_TOLERANCE, rtol=0):
            sign = -1
        else:
            raise ValueError("rotation_power_spin_lift_mismatch")
    kpoint = np.asarray(raw["kpoint_frac"], dtype=float)
    phase = sign * np.exp(-2j * np.pi * np.dot(kpoint, shift))
    determinant = complex(np.linalg.det(block) * phase ** rank)
    return determinant, rank, op_id


def _inferred_sources(report, target_valley):
    """Yield complete inferred HSP multiplets, never individual band fragments."""
    for orbit in _mapping(report).get("valley_orbits", []):
        orbit = _mapping(orbit)
        if (orbit.get("mapping_type") != "exchanged"
                or orbit.get("unitary_completion_status") != "validated"
                or orbit.get("unitary_completion_blockers") != []
                or orbit.get("tr_irrep_completion_status") != "passed"):
            continue
        by_hsp = _mapping(_mapping(orbit.get(
            "unitary_valley_irrep_completion_records", {}
        )).get(target_valley, {}))
        for hsp, records in by_hsp.items():
            if records and all(_mapping(r).get("completion_kind") == "inferred_by_time_reversal"
                               for r in records):
                if any(r.get("target_valley") != target_valley
                       or r.get("target_source_hsp_label") != hsp for r in records):
                    raise ValueError("rotation_tr_target_binding_mismatch")
                yield hsp, records, orbit


def _validate_inference(records, orbit, report, contexts, source_match):
    source_counts = {}
    for record in records:
        if record.get("readiness_status") != "trusted" or record.get("blockers") != []:
            raise ValueError("rotation_tr_completion_untrusted")
        certificate = record.get("tr_irrep_completion_certificate", {})
        reviewed = certificate.get("reviewed_time_reversal", {})
        if not validate_tr_irrep_completion_certificate(
            certificate,
            completion_record=record,
            valley_mapping=report.get("time_reversal_valley_mapping", {}),
            hsp_mapping=reviewed.get("hsp_involution", {}),
            irrep_pairing=orbit.get("time_reversal_irrep_pairing", {}),
            reviewed_source_identity=orbit.get("reviewed_time_reversal_source_identity", {}),
            reviewed_source_context=orbit.get("reviewed_time_reversal_source_context", {}),
            cprime_validation_context=contexts,
        ):
            raise ValueError("rotation_tr_completion_validation_failed")
        label = record.get("reviewed_time_reversal_relation", {}).get("evidence_irrep")
        count = record.get("multiplicity")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("rotation_tr_multiplicity_invalid")
        source_counts[label] = source_counts.get(label, 0) + count
    if source_counts != source_match.get("irrep_multiplicities"):
        raise ValueError("rotation_tr_incomplete_source_multiplet")


def _point_evidence(*, points, power, valley, generator, coordinates, matches,
                    contexts, cache, tr_report, spinful, k_tolerance):
    observed = sorted(kpoint for kpoint, k in coordinates.items()
                      if _equivalent(_matrix(k, (3,)), points, k_tolerance))
    candidates = []
    for kpoint in observed:
        raw, identity, match = _source_context(
            kpoint, valley, matches, contexts, cache, spinful=spinful,
        )
        if not np.array_equal(raw["kpoint_frac"], coordinates[kpoint]):
            raise ValueError(f"rotation_hsp_coordinate_binding_mismatch:{kpoint}:{valley}")
        det, rank, op_id = _physical_determinant(raw, valley, generator, power, spinful=spinful)
        candidates.append({
            "completion_kind": "observed_at_sampled_kpoint", "sampled_kpoint": kpoint,
            "source_hsp_label": match.get("projected_hsp_classification", {}).get("source_hsp_label"),
            "k_frac": np.asarray(raw["kpoint_frac"]).tolist(),
            "operation_id": op_id, "rotation_power": power, "subspace_rank": rank,
            "scoped_representation_evidence_identity": identity,
            "determinant": {"real": det.real, "imag": det.imag},
        })
    if not observed:
        for hsp, records, orbit in _inferred_sources(tr_report, valley):
            bindings = {(r.get("evidence_sampled_kpoint"), r.get("evidence_valley"))
                        for r in records}
            if len(bindings) != 1:
                continue
            kpoint, source_valley = next(iter(bindings))
            if kpoint not in coordinates:
                continue
            target_k = -_matrix(coordinates[kpoint], (3,))
            if not _equivalent(target_k, points, k_tolerance):
                continue
            raw, identity, match = _source_context(
                kpoint, source_valley, matches, contexts, cache, spinful=spinful,
            )
            if not np.array_equal(raw["kpoint_frac"], coordinates[kpoint]):
                raise ValueError(f"rotation_hsp_coordinate_binding_mismatch:{kpoint}:{source_valley}")
            _validate_inference(records, orbit, tr_report, contexts, match)
            det, rank, op_id = _physical_determinant(raw, source_valley, generator, power, spinful=spinful)
            det = det.conjugate()
            candidates.append({
                "completion_kind": "inferred_by_time_reversal", "source_hsp_label": hsp,
                "evidence_sampled_kpoint": kpoint, "evidence_valley": source_valley,
                "k_frac": target_k.tolist(), "operation_id": op_id,
                "rotation_power": power, "subspace_rank": rank,
                "scoped_representation_evidence_identity": identity,
                "completion_certificate_identities": [r["tr_irrep_completion_certificate"]["certificate_identity"]
                                                      for r in records],
                "determinant": {"real": det.real, "imag": det.imag},
            })
    if not candidates:
        raise ValueError(f"missing_rotation_hsp:power={power}:orbit={points}")
    first = candidates[0]
    determinant = complex(**first["determinant"])
    if any(c["subspace_rank"] != first["subspace_rank"]
           or abs(complex(**c["determinant"]) - determinant) > ROTATION_PHASE_TOLERANCE
           for c in candidates[1:]):
        raise ValueError("rotation_star_evidence_conflict")
    return first


def build_valley_chern_report(
    *, symmetry_payload: Mapping[str, Any], valley_names: Sequence[str],
    valley_irrep_matching: Mapping[str, Any], cprime_validation_context: Mapping[str, Any],
    kpoint_frac_by_name: Mapping[str, Any], time_reversal_orbit_report: Mapping[str, Any] | None,
    spinful: bool, use_2d_momentum_only: bool, k_tolerance: float,
) -> dict[str, Any]:
    """Compute the strongest available Cnz residue without changing trust gates."""
    rows = []
    cache = {}
    for valley in valley_names:
        row = {"valley": valley, "modulus": None, "residue": None,
               "status": "blocked", "subspace_rank": None,
               "rotation_operation_id": None, "eigenvalue_evidence": [], "blocking_reasons": []}
        rows.append(row)
        try:
            operations = _mapping(symmetry_payload).get("detected_operations", [])
            matches = _mapping(_mapping(valley_irrep_matching).get("generic_matches_by_kpoint", {}))
            if symmetry_payload.get("status") != "ok" or not operations:
                raise ValueError("rotation_symmetry_not_available")
            if not use_2d_momentum_only:
                raise ValueError("rotation_chern_requires_two_dimensional_scope")
            if (isinstance(k_tolerance, bool) or not np.isfinite(k_tolerance)
                    or k_tolerance <= 0 or k_tolerance >= 0.1):
                raise ValueError("rotation_hsp_tolerance_invalid")
            rotations = find_valley_rotations(operations, valley)
            if not rotations:
                row.update(status="not_applicable", blocking_reasons=["no_valley_preserving_Cnz"])
                continue
            order, generator = rotations[0]
            row.update(modulus=order, rotation_operation_id=generator["operation_id"])
            direct = _matrix(symmetry_payload.get("lattice_direct_cart"), (3, 3))
            calculated = cart_rotation_from_fractional(generator["rotation_frac"], direct)
            if not np.allclose(calculated, generator["rotation_cart"], atol=GEOMETRY_TOLERANCE, rtol=0):
                raise ValueError("rotation_cart_fractional_mismatch")
            full_r, full_t = _affine_power(generator, order)
            if (not np.array_equal(full_r, np.eye(3))
                    or not np.allclose(full_t, 0, atol=GEOMETRY_TOLERANCE, rtol=0)):
                raise ValueError("rotation_not_finite_order_Seitz_operation")
            # Finite order gives the exact integer inverse R^(n-1), without
            # rounding a floating-point inverse into a crystallographic map.
            fractional = _matrix(generator["rotation_frac"], (3, 3)).astype(int)
            reciprocal = np.linalg.matrix_power(fractional, order - 1).T
            if not np.allclose(reciprocal[2, :2], 0, atol=GEOMETRY_TOLERANCE, rtol=0):
                raise ValueError("rotation_does_not_preserve_reciprocal_plane")
            # The generator is defined by Cartesian +z, not the orientation
            # of the primitive basis used to enumerate invariant momenta.
            point_orbits = rotation_fixed_point_orbits(reciprocal[:2, :2], order)
            for orbit in point_orbits:
                row["eigenvalue_evidence"].append(_point_evidence(
                    points=orbit["points"], power=orbit["power"], valley=valley,
                    generator=generator, coordinates=kpoint_frac_by_name, matches=matches,
                    contexts=cprime_validation_context, cache=cache,
                    tr_report=time_reversal_orbit_report or {}, spinful=spinful,
                    k_tolerance=k_tolerance,
                ))
            ranks = {e["subspace_rank"] for e in row["eigenvalue_evidence"]}
            if len(ranks) != 1:
                raise ValueError("rotation_hsp_subspace_rank_mismatch")
            row["subspace_rank"] = ranks.pop()
            row["residue"] = rotation_chern_residue(
                order, [complex(**e["determinant"]) for e in row["eigenvalue_evidence"]],
                row["subspace_rank"], spinful, tolerance=ROTATION_PHASE_TOLERANCE,
            )
            row["status"] = "conditional"
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            row["blocking_reasons"].append(str(exc))
    statuses = {r["status"] for r in rows}
    status = ("conditional" if statuses <= {"conditional", "not_applicable"} and "conditional" in statuses
              else "partial" if "conditional" in statuses
              else "not_applicable" if statuses == {"not_applicable"} else "blocked")
    return {"status": status, "global_valley_subspace_status": "not_evaluated",
            "interpretation": INTERPRETATION,
            "convention": "active_positive_Cnz; A=i<u|du>; Cartesian_kx_ky",
            "rotation_phase_tolerance": ROTATION_PHASE_TOLERANCE,
            "hsp_fractional_k_tolerance": k_tolerance, "rows": rows}
