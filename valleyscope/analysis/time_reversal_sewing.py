"""Numerical time-reversal sewing evidence in the selected band subspace."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib

import numpy as np


_TOL = 1e-3
_K_TOL = 5e-6


_PROJECTOR_KIND_BY_WORKFLOW = {
    "direct_qcut": "fixed_center_seed",
    "symmetry_adapted": "symmetry_adapted",
}

_PROJECTOR_PROVENANCE_KEYS = frozenset({
    "workflow_path",
    "projector_kind",
    "projector_shape",
    "projector_fingerprint",
})

_PROJECTOR_COVARIANCE_KEYS = frozenset({
    "partner_valley",
    "status",
    "covariance_residual",
    "source_projector_provenance",
    "target_projector_provenance",
})


def select_trusted_valley_projectors(
    *,
    workflow_decisions: Mapping[str, object] | None,
    seed_projectors_by_kpoint: Mapping[str, Mapping[str, np.ndarray]],
    symmetry_adapted_projectors_by_kpoint: Mapping[
        str, Mapping[str, np.ndarray]
    ],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, dict[str, object]]],
    list[str],
]:
    """Select the exact trusted projector named by each workflow decision."""
    selected: dict[str, dict[str, np.ndarray]] = {}
    provenance: dict[str, dict[str, dict[str, object]]] = {}
    blockers: list[str] = []
    by_kpoint = (
        workflow_decisions.get("by_kpoint", {})
        if isinstance(workflow_decisions, Mapping) else {}
    )
    if not isinstance(by_kpoint, Mapping):
        return {}, {}, ["trusted_projector_workflow_decisions_malformed"]

    for kpoint, raw_by_valley in by_kpoint.items():
        if not isinstance(kpoint, str) or not kpoint or not isinstance(
            raw_by_valley, Mapping
        ):
            blockers.append("trusted_projector_workflow_decisions_malformed")
            continue
        for valley, raw_decision in raw_by_valley.items():
            if not isinstance(valley, str) or not valley or not isinstance(
                raw_decision, Mapping
            ):
                blockers.append(
                    f"trusted_projector_workflow_decision_malformed:{kpoint}"
                )
                continue
            workflow_path = raw_decision.get("workflow_path")
            if (
                raw_decision.get("readiness_level") != "trusted"
                or not isinstance(workflow_path, str)
                or workflow_path not in _PROJECTOR_KIND_BY_WORKFLOW
            ):
                blockers.append(
                    f"trusted_projector_workflow_blocked:{kpoint}:{valley}"
                )
                continue
            source_map = (
                seed_projectors_by_kpoint
                if workflow_path == "direct_qcut"
                else symmetry_adapted_projectors_by_kpoint
            )
            projectors_at_kpoint = source_map.get(kpoint, {})
            raw_projector = (
                projectors_at_kpoint.get(valley)
                if isinstance(projectors_at_kpoint, Mapping) else None
            )
            if raw_projector is None:
                blockers.append(
                    f"trusted_projector_missing:{kpoint}:{valley}:"
                    f"{workflow_path}"
                )
                continue
            try:
                projector = np.asarray(raw_projector, dtype=complex)
            except (TypeError, ValueError):
                projector = np.asarray([])
            if (
                projector.ndim != 2
                or projector.shape[0] != projector.shape[1]
                or not np.all(np.isfinite(projector))
            ):
                blockers.append(
                    f"trusted_projector_malformed:{kpoint}:{valley}:"
                    f"{workflow_path}"
                )
                continue
            selected.setdefault(kpoint, {})[valley] = raw_projector
            provenance.setdefault(kpoint, {})[valley] = (
                build_projector_provenance(
                    workflow_path=workflow_path,
                    projector=projector,
                )
            )
    return selected, provenance, _deduplicate(blockers)


def build_projector_provenance(
    *,
    workflow_path: str,
    projector: object,
) -> dict[str, object]:
    """Return compact provenance bound to canonical projector bytes."""
    if (
        not isinstance(workflow_path, str)
        or workflow_path not in _PROJECTOR_KIND_BY_WORKFLOW
    ):
        raise ValueError(f"unsupported projector workflow: {workflow_path}")
    identity = _projector_identity(projector)
    if identity is None:
        raise ValueError("projector must be a finite square matrix")
    shape, fingerprint = identity
    return {
        "workflow_path": workflow_path,
        "projector_kind": _PROJECTOR_KIND_BY_WORKFLOW[workflow_path],
        "projector_shape": shape,
        "projector_fingerprint": fingerprint,
    }


def build_time_reversal_sewing_report(
    *,
    kpoint_frac_by_name: Mapping[str, np.ndarray],
    g_vectors_frac_by_kpoint: Mapping[str, np.ndarray],
    coefficients_by_kpoint: Mapping[str, np.ndarray],
    band_indices_by_kpoint: Mapping[str, np.ndarray],
    valley_projectors_by_kpoint: Mapping[
        str, Mapping[str, np.ndarray]
    ],
    valley_projector_provenance_by_kpoint: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ],
    projector_selection_blockers: Sequence[str],
    time_reversal_valley_mapping: Mapping[str, str],
    spinor: bool,
    spinor_convention_verified: bool,
    tolerance: float = _TOL,
) -> dict[str, object]:
    """Build ``B_Theta(-k) B_Theta(k)^*`` and projector-covariance tests.

    Coefficients use the HDF5 convention ``[band, spinor, G]``.  If
    ``k_partner = -k + H``, the plane-wave map is
    ``G_partner = -G - H``.  Spinors use ``i sigma_y K``.
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    names = list(kpoint_frac_by_name)
    blockers = [
        str(value) for value in projector_selection_blockers
        if isinstance(value, str) and value
    ]
    if not names or len(set(names)) != len(names):
        blockers.append("time_reversal_kpoint_inventory_missing_or_duplicate")
    required_maps = (
        g_vectors_frac_by_kpoint,
        coefficients_by_kpoint,
        band_indices_by_kpoint,
    )
    if any(set(mapping) != set(names) for mapping in required_maps):
        blockers.append("time_reversal_wavefunction_inventory_incomplete")
    if spinor and not spinor_convention_verified:
        blockers.append("spinor_convention_unverified_for_time_reversal")

    valley_names = set(time_reversal_valley_mapping)
    if (
        not valley_names
        or set(time_reversal_valley_mapping.values()) != valley_names
    ):
        blockers.append(
            "incomplete_or_nonbijective_time_reversal_valley_mapping"
        )
    if any(
        time_reversal_valley_mapping.get(
            time_reversal_valley_mapping.get(valley, ""), ""
        ) != valley
        for valley in valley_names
    ):
        blockers.append("non_involutive_time_reversal_valley_mapping")

    kpoint_mapping: dict[str, str] = {}
    reciprocal_shifts: dict[str, list[int]] = {}
    for name in names:
        source_k = _vector3(kpoint_frac_by_name.get(name))
        if source_k is None:
            blockers.append(f"malformed_time_reversal_kpoint:{name}")
            continue
        candidates: list[tuple[str, np.ndarray]] = []
        for partner in names:
            partner_k = _vector3(kpoint_frac_by_name.get(partner))
            if partner_k is None:
                continue
            shift = source_k + partner_k
            rounded = np.rint(shift)
            if np.linalg.norm(shift - rounded) <= _K_TOL:
                candidates.append((partner, rounded.astype(int)))
        if len(candidates) != 1:
            blockers.append(
                f"ambiguous_time_reversal_kpoint_partner:{name}:"
                f"{[candidate for candidate, _ in candidates]}"
            )
            continue
        partner, shift = candidates[0]
        kpoint_mapping[name] = partner
        reciprocal_shifts[name] = shift.astype(int).tolist()

    if set(kpoint_mapping) != set(names) or set(kpoint_mapping.values()) != set(
        names
    ):
        blockers.append("incomplete_or_nonbijective_time_reversal_kpoint_mapping")
    if any(
        kpoint_mapping.get(kpoint_mapping.get(name, ""), "") != name
        for name in names
    ):
        blockers.append("non_involutive_time_reversal_kpoint_mapping")

    raw_rows: dict[str, dict[str, object]] = {}
    sewing_matrices: dict[str, np.ndarray] = {}
    for name in names:
        partner = kpoint_mapping.get(name)
        if partner is None or any(
            name not in mapping or partner not in mapping
            for mapping in required_maps
        ):
            continue
        row, sewing = _build_sewing_row(
            name=name,
            partner=partner,
            reciprocal_shift=np.asarray(reciprocal_shifts[name], dtype=int),
            source_g=g_vectors_frac_by_kpoint[name],
            partner_g=g_vectors_frac_by_kpoint[partner],
            source_coefficients=coefficients_by_kpoint[name],
            partner_coefficients=coefficients_by_kpoint[partner],
            source_band_indices=band_indices_by_kpoint[name],
            partner_band_indices=band_indices_by_kpoint[partner],
            spinor=spinor,
            tolerance=tolerance,
        )
        raw_rows[name] = row
        if sewing is not None:
            sewing_matrices[name] = sewing

    theta_square = -1 if spinor else 1
    rows: list[dict[str, object]] = []
    for name in names:
        if name not in raw_rows:
            continue
        row = raw_rows[name]
        partner = str(row["target_kpoint"])
        row_blockers = list(row["blockers"])
        sewing = sewing_matrices.get(name)
        reverse = sewing_matrices.get(partner)
        if sewing is None or reverse is None:
            theta_square_residual = None
            row_blockers.append(f"time_reversal_sewing_pair_incomplete:{name}")
        elif reverse.shape[1] != sewing.shape[0]:
            theta_square_residual = None
            row_blockers.append(
                f"time_reversal_sewing_composition_shape_mismatch:{name}"
            )
        else:
            composition = reverse @ sewing.conj()
            theta_square_residual = float(
                np.linalg.norm(
                    composition
                    - theta_square * np.eye(composition.shape[0])
                )
            )
            if theta_square_residual > tolerance:
                row_blockers.append(
                    "time_reversal_theta_square_failed:"
                    f"{name}:{theta_square_residual:.6e}"
                )
        row["theta_square"] = theta_square
        row["theta_square_residual"] = theta_square_residual
        numerical_blockers = _deduplicate(row_blockers)
        covariance_blockers: list[str] = []
        row["projector_covariance"] = _projector_covariance(
            name=name,
            partner=partner,
            sewing=sewing,
            source_projectors=valley_projectors_by_kpoint.get(name),
            partner_projectors=valley_projectors_by_kpoint.get(partner),
            source_provenance=(
                valley_projector_provenance_by_kpoint.get(name)
            ),
            partner_provenance=(
                valley_projector_provenance_by_kpoint.get(partner)
            ),
            valley_mapping=time_reversal_valley_mapping,
            tolerance=tolerance,
            blockers=covariance_blockers,
        )
        row["blockers"] = numerical_blockers
        row["status"] = "validated" if not row["blockers"] else "blocked"
        blockers.extend(row["blockers"])
        blockers.extend(covariance_blockers)
        rows.append(row)

    blockers = _deduplicate(blockers)
    return {
        "status": "validated" if rows and not blockers else "blocked",
        "theta_square": theta_square,
        "spin_convention": (
            "spinful_i_sigma_y_K" if spinor else "scalar_complex_conjugation"
        ),
        "spinor_convention_verified": bool(spinor_convention_verified),
        "time_reversal_kpoint_mapping": kpoint_mapping,
        "sampled_kpoint_frac_by_name": {
            name: vector.tolist()
            for name in names
            if (vector := _vector3(kpoint_frac_by_name.get(name))) is not None
        },
        "reciprocal_shifts_by_kpoint": reciprocal_shifts,
        "rows": rows,
        "blockers": blockers,
    }


def validate_time_reversal_sewing_report(
    evidence: Mapping[str, object] | None,
    *,
    valley_members: list[str],
    theta_square: object,
    required_kpoints: list[str] | None = None,
    required_projector_workflows: Mapping[
        str, Mapping[str, str]
    ] | None = None,
    required_projector_provenance: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ] | None = None,
) -> bool:
    """Validate serialized sewing evidence on the smallest TR-closed scope."""
    if not isinstance(evidence, Mapping):
        return False
    if required_kpoints is None and (
        evidence.get("status") != "validated"
        or evidence.get("blockers") != []
    ):
        return False
    if theta_square not in (-1, 1) or evidence.get("theta_square") != theta_square:
        return False
    if evidence.get("spin_convention") != (
        "spinful_i_sigma_y_K"
        if theta_square == -1 else "scalar_complex_conjugation"
    ):
        return False
    spinor_verified = evidence.get("spinor_convention_verified")
    if not isinstance(spinor_verified, bool) or (
        theta_square == -1 and not spinor_verified
    ):
        return False
    kpoint_mapping = evidence.get("time_reversal_kpoint_mapping")
    if not isinstance(kpoint_mapping, Mapping) or not kpoint_mapping:
        return False
    if any(
        not isinstance(source, str)
        or not source
        or not isinstance(target, str)
        or not target
        for source, target in kpoint_mapping.items()
    ):
        return False
    if required_kpoints is None:
        scope = set(kpoint_mapping)
    else:
        if (
            not required_kpoints
            or any(not isinstance(name, str) or not name for name in required_kpoints)
            or len(set(required_kpoints)) != len(required_kpoints)
        ):
            return False
        scope = set(required_kpoints)
        for source in required_kpoints:
            target = kpoint_mapping.get(source)
            if not isinstance(target, str) or not target:
                return False
            scope.add(target)
    if (
        not scope
        or any(source not in kpoint_mapping for source in scope)
        or {kpoint_mapping[source] for source in scope} != scope
        or any(
            kpoint_mapping.get(kpoint_mapping[source]) != source
            for source in scope
        )
    ):
        return False
    reciprocal_shifts = evidence.get("reciprocal_shifts_by_kpoint")
    if not isinstance(reciprocal_shifts, Mapping) or any(
        source not in reciprocal_shifts for source in scope
    ):
        return False
    if any(
        not _is_integer_list(reciprocal_shifts[source], length=3)
        for source in scope
    ):
        return False
    if any(
        reciprocal_shifts[source] != reciprocal_shifts[target]
        for source in scope
        for target in [kpoint_mapping[source]]
    ):
        return False
    sampled_kpoints = evidence.get("sampled_kpoint_frac_by_name")
    if not isinstance(sampled_kpoints, Mapping):
        return False
    sampled_vectors = {
        source: _vector3(sampled_kpoints.get(source)) for source in scope
    }
    if any(vector is None for vector in sampled_vectors.values()):
        return False
    if any(
        np.linalg.norm(
            sampled_vectors[source]
            + sampled_vectors[kpoint_mapping[source]]
            - np.asarray(reciprocal_shifts[source], dtype=float)
        ) > _K_TOL
        for source in scope
    ):
        return False
    rows = evidence.get("rows")
    if not isinstance(rows, list):
        return False
    scoped_rows = [
        row for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("source_kpoint"), str)
        and row.get("source_kpoint") in scope
    ]
    rows_by_source = {row.get("source_kpoint"): row for row in scoped_rows}
    if len(rows_by_source) != len(scoped_rows) or set(rows_by_source) != scope:
        return False
    if required_projector_workflows is not None and (
        not isinstance(required_projector_workflows, Mapping)
        or not set(required_projector_workflows).issubset(scope)
        or (
            required_kpoints is not None
            and not set(required_kpoints).issubset(
                required_projector_workflows
            )
        )
    ):
        return False
    if required_projector_provenance is not None and (
        not isinstance(required_projector_provenance, Mapping)
        or not set(required_projector_provenance).issubset(scope)
        or (
            required_kpoints is not None
            and not set(required_kpoints).issubset(
                required_projector_provenance
            )
        )
    ):
        return False
    for source in scope:
        target = kpoint_mapping[source]
        row = rows_by_source[source]
        if (
            row.get("status") != "validated"
            or row.get("blockers") != []
            or row.get("target_kpoint") != target
            or row.get("reciprocal_shift") != reciprocal_shifts[source]
            or row.get("mapping_miss_count") != 0
            or row.get("theta_square") != theta_square
        ):
            return False
        source_bands = row.get("source_band_indices_vasp")
        target_bands = row.get("target_band_indices_vasp")
        if (
            not _is_integer_list(source_bands, positive=True)
            or not _is_integer_list(target_bands, positive=True)
            or len(set(source_bands)) != len(source_bands)
            or len(set(target_bands)) != len(target_bands)
            or source_bands != rows_by_source[target].get(
                "target_band_indices_vasp"
            )
            or target_bands != rows_by_source[target].get(
                "source_band_indices_vasp"
            )
            or (theta_square == -1 and source == target and len(source_bands) % 2)
        ):
            return False
        for key in (
            "source_orthonormality_residual",
            "target_orthonormality_residual",
            "target_subspace_closure_residual",
            "sewing_unitarity_residual",
            "theta_square_residual",
        ):
            value = row.get(key)
            if not _is_residual(value):
                return False
        singular_values = row.get("overlap_singular_values")
        if (
            not isinstance(singular_values, list)
            or len(singular_values) != len(source_bands)
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or abs(float(value) - 1.0) > _TOL
                for value in singular_values
            )
        ):
            return False
        covariance = row.get("projector_covariance")
        if (
            not isinstance(covariance, Mapping)
            or set(covariance) != set(valley_members)
        ):
            return False
        for valley in valley_members:
            entry = covariance.get(valley)
            if (
                not isinstance(entry, Mapping)
                or set(entry) != _PROJECTOR_COVARIANCE_KEYS
                or entry.get("status") != "validated"
                or entry.get("partner_valley") != valley
                or not _is_residual(entry.get("covariance_residual"))
            ):
                return False
            expected_source_workflow = _expected_projector_workflow(
                required_projector_workflows, source, valley
            )
            expected_target_workflow = _expected_projector_workflow(
                required_projector_workflows, target, valley
            )
            if required_projector_workflows is not None and (
                (source in required_projector_workflows
                 and expected_source_workflow is None)
                or (target in required_projector_workflows
                    and expected_target_workflow is None)
            ):
                return False
            if not _valid_projector_provenance(
                entry.get("source_projector_provenance"),
                expected_workflow=expected_source_workflow,
            ) or not _valid_projector_provenance(
                entry.get("target_projector_provenance"),
                expected_workflow=expected_target_workflow,
            ):
                return False
            expected_source_provenance = _expected_projector_provenance(
                required_projector_provenance, source, valley
            )
            expected_target_provenance = _expected_projector_provenance(
                required_projector_provenance, target, valley
            )
            if required_projector_provenance is not None and (
                (source in required_projector_provenance
                 and expected_source_provenance is None)
                or (target in required_projector_provenance
                    and expected_target_provenance is None)
            ):
                return False
            if (
                expected_source_provenance is not None
                and _compact_projector_provenance(
                    entry.get("source_projector_provenance")
                ) != expected_source_provenance
            ) or (
                expected_target_provenance is not None
                and _compact_projector_provenance(
                    entry.get("target_projector_provenance")
                ) != expected_target_provenance
            ):
                return False
            reverse_covariance = rows_by_source[target].get(
                "projector_covariance"
            )
            reverse_entry = (
                reverse_covariance.get(valley)
                if isinstance(reverse_covariance, Mapping) else None
            )
            if not isinstance(reverse_entry, Mapping) or (
                _compact_projector_provenance(
                    entry.get("source_projector_provenance")
                )
                != _compact_projector_provenance(
                    reverse_entry.get("target_projector_provenance")
                )
                or _compact_projector_provenance(
                    entry.get("target_projector_provenance")
                )
                != _compact_projector_provenance(
                    reverse_entry.get("source_projector_provenance")
                )
            ):
                return False
    return True


def _expected_projector_workflow(
    expected: Mapping[str, Mapping[str, str]] | None,
    kpoint: str,
    valley: str,
) -> str | None:
    if expected is None:
        return None
    by_valley = expected.get(kpoint)
    if not isinstance(by_valley, Mapping):
        return None
    value = by_valley.get(valley)
    return (
        value
        if isinstance(value, str) and value in _PROJECTOR_KIND_BY_WORKFLOW
        else None
    )


def _valid_projector_provenance(
    value: object,
    *,
    expected_workflow: str | None,
) -> bool:
    compact = _compact_projector_provenance(value)
    return compact is not None and (
        expected_workflow is None
        or compact["workflow_path"] == expected_workflow
    )


def _expected_projector_provenance(
    expected: Mapping[
        str, Mapping[str, Mapping[str, object]]
    ] | None,
    kpoint: str,
    valley: str,
) -> dict[str, object] | None:
    if expected is None:
        return None
    by_valley = expected.get(kpoint)
    if not isinstance(by_valley, Mapping):
        return None
    return _compact_projector_provenance(by_valley.get(valley))


def _is_integer_list(
    value: object,
    *,
    length: int | None = None,
    positive: bool = False,
) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and (length is None or len(value) == length)
        and all(
            isinstance(item, int)
            and not isinstance(item, bool)
            and (not positive or item > 0)
            for item in value
        )
    )


def _is_residual(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and 0.0 <= float(value) <= _TOL
    )


def _build_sewing_row(
    *,
    name: str,
    partner: str,
    reciprocal_shift: np.ndarray,
    source_g: object,
    partner_g: object,
    source_coefficients: object,
    partner_coefficients: object,
    source_band_indices: object,
    partner_band_indices: object,
    spinor: bool,
    tolerance: float,
) -> tuple[dict[str, object], np.ndarray | None]:
    blockers: list[str] = []
    source_g_int = _integer_g_vectors(source_g)
    partner_g_int = _integer_g_vectors(partner_g)
    source = np.asarray(source_coefficients, dtype=complex)
    target = np.asarray(partner_coefficients, dtype=complex)
    expected_spin = 2 if spinor else 1
    if source_g_int is None or partner_g_int is None:
        blockers.append(f"malformed_time_reversal_g_vectors:{name}")
    if (
        source.ndim != 3
        or target.ndim != 3
        or source.shape[1] != expected_spin
        or target.shape[1] != expected_spin
        or source_g_int is None
        or partner_g_int is None
        or source.shape[2] != len(source_g_int)
        or target.shape[2] != len(partner_g_int)
    ):
        blockers.append(f"malformed_time_reversal_coefficients:{name}")

    source_bands = _integer_vector(source_band_indices)
    target_bands = _integer_vector(partner_band_indices)
    if (
        source_bands is None
        or target_bands is None
        or source_bands.size != source.shape[0]
        or target_bands.size != target.shape[0]
    ):
        blockers.append(f"malformed_time_reversal_band_indices:{name}")

    row: dict[str, object] = {
        "source_kpoint": name,
        "target_kpoint": partner,
        "reciprocal_shift": reciprocal_shift.astype(int).tolist(),
        "source_band_indices_vasp": (
            source_bands.tolist() if source_bands is not None else []
        ),
        "target_band_indices_vasp": (
            target_bands.tolist() if target_bands is not None else []
        ),
        "mapping_miss_count": None,
        "overlap_singular_values": [],
        "target_subspace_closure_residual": None,
        "sewing_unitarity_residual": None,
        "source_orthonormality_residual": None,
        "target_orthonormality_residual": None,
        "blockers": blockers,
    }
    if blockers or source_g_int is None or partner_g_int is None:
        return row, None

    lookup: dict[tuple[int, int, int], int] = {}
    for index, g_vector in enumerate(partner_g_int):
        key = tuple(int(value) for value in g_vector)
        if key in lookup:
            blockers.append(f"duplicate_time_reversal_partner_g_vector:{name}")
        lookup[key] = index
    mapped = np.asarray([
        lookup.get(
            tuple(int(value) for value in (-g_vector - reciprocal_shift)), -1
        )
        for g_vector in source_g_int
    ], dtype=int)
    miss_count = int(np.count_nonzero(mapped < 0))
    row["mapping_miss_count"] = miss_count
    if miss_count:
        blockers.append(f"incomplete_time_reversal_g_mapping:{name}:{miss_count}")

    transformed = np.zeros(
        (source.shape[0], expected_spin, target.shape[2]), dtype=complex
    )
    source_positions = np.flatnonzero(mapped >= 0)
    target_positions = mapped[source_positions]
    if spinor:
        transformed[:, 0, target_positions] = source[
            :, 1, source_positions
        ].conj()
        transformed[:, 1, target_positions] = -source[
            :, 0, source_positions
        ].conj()
    else:
        transformed[:, 0, target_positions] = source[
            :, 0, source_positions
        ].conj()

    source_flat = source.reshape(source.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    transformed_flat = transformed.reshape(source.shape[0], -1)
    source_gram_error = float(
        np.linalg.norm(
            source_flat @ source_flat.conj().T - np.eye(source.shape[0])
        )
    )
    target_gram_error = float(
        np.linalg.norm(
            target_flat @ target_flat.conj().T - np.eye(target.shape[0])
        )
    )
    row["source_orthonormality_residual"] = source_gram_error
    row["target_orthonormality_residual"] = target_gram_error
    if source_gram_error > tolerance:
        blockers.append(
            f"time_reversal_source_states_not_orthonormal:{name}:"
            f"{source_gram_error:.6e}"
        )
    if target_gram_error > tolerance:
        blockers.append(
            f"time_reversal_target_states_not_orthonormal:{name}:"
            f"{target_gram_error:.6e}"
        )

    sewing = target_flat.conj() @ transformed_flat.T
    singular_values = np.linalg.svd(sewing, compute_uv=False)
    row["overlap_singular_values"] = [float(value) for value in singular_values]
    transformed_norm = max(float(np.linalg.norm(transformed_flat)), 1e-30)
    closure_residual = float(
        np.linalg.norm(transformed_flat - sewing.T @ target_flat)
        / transformed_norm
    )
    row["target_subspace_closure_residual"] = closure_residual
    if closure_residual > tolerance:
        blockers.append(
            f"time_reversal_target_subspace_not_closed:{name}:"
            f"{closure_residual:.6e}"
        )
    if source.shape[0] != target.shape[0]:
        blockers.append(
            f"time_reversal_target_dimension_mismatch:{name}:"
            f"{source.shape[0]}:{target.shape[0]}"
        )
        unitary_residual = None
    else:
        unitary_residual = float(max(
            np.linalg.norm(
                sewing.conj().T @ sewing - np.eye(source.shape[0])
            ),
            np.linalg.norm(
                sewing @ sewing.conj().T - np.eye(target.shape[0])
            ),
        ))
        if unitary_residual > tolerance:
            blockers.append(
                f"time_reversal_sewing_not_unitary:{name}:"
                f"{unitary_residual:.6e}"
            )
    row["sewing_unitarity_residual"] = unitary_residual
    if spinor and name == partner and source.shape[0] % 2:
        blockers.append(f"incomplete_kramers_subspace_odd_dimension:{name}")
    row["blockers"] = blockers
    return row, sewing


def _projector_covariance(
    *,
    name: str,
    partner: str,
    sewing: np.ndarray | None,
    source_projectors: object,
    partner_projectors: object,
    source_provenance: object,
    partner_provenance: object,
    valley_mapping: Mapping[str, str],
    tolerance: float,
    blockers: list[str],
) -> dict[str, dict[str, object]]:
    if sewing is None:
        blockers.append(f"time_reversal_projector_covariance_not_evaluated:{name}")
        return {}
    if (
        not isinstance(source_projectors, Mapping)
        or not isinstance(partner_projectors, Mapping)
        or not isinstance(source_provenance, Mapping)
        or not isinstance(partner_provenance, Mapping)
    ):
        blockers.append(f"time_reversal_trusted_projectors_missing:{name}:{partner}")
        return {}
    result: dict[str, dict[str, object]] = {}
    for valley, partner_valley in valley_mapping.items():
        source = source_projectors.get(valley)
        target = partner_projectors.get(partner_valley)
        source_projector_provenance = source_provenance.get(valley)
        target_projector_provenance = partner_provenance.get(partner_valley)
        try:
            source_matrix = np.asarray(source, dtype=complex)
            target_matrix = np.asarray(target, dtype=complex)
        except (TypeError, ValueError):
            source_matrix = np.asarray([])
            target_matrix = np.asarray([])
        compact_source = _compact_projector_provenance(
            source_projector_provenance,
            projector=source_matrix,
        )
        compact_target = _compact_projector_provenance(
            target_projector_provenance,
            projector=target_matrix,
        )
        if compact_source is None or compact_target is None:
            blockers.append(
                f"time_reversal_projector_provenance_missing:{name}:{valley}"
            )
        if (
            source_matrix.shape != (sewing.shape[1], sewing.shape[1])
            or target_matrix.shape != (sewing.shape[0], sewing.shape[0])
        ):
            blockers.append(
                f"time_reversal_projector_shape_mismatch:{name}:{valley}"
            )
            result[valley] = {
                "partner_valley": partner_valley,
                "status": "blocked",
                "covariance_residual": None,
                "source_projector_provenance": compact_source,
                "target_projector_provenance": compact_target,
            }
            continue
        transformed = sewing @ source_matrix.conj() @ sewing.conj().T
        residual = float(
            np.linalg.norm(transformed - target_matrix)
            / max(float(np.linalg.norm(target_matrix)), 1e-30)
        )
        status = (
            "validated"
            if residual <= tolerance
            and compact_source is not None
            and compact_target is not None
            else "blocked"
        )
        if residual > tolerance:
            blockers.append(
                f"time_reversal_projector_covariance_failed:{name}:{valley}:"
                f"{residual:.6e}"
            )
        result[valley] = {
            "partner_valley": partner_valley,
            "status": status,
            "covariance_residual": residual,
            "source_projector_provenance": compact_source,
            "target_projector_provenance": compact_target,
        }
    return result


def _compact_projector_provenance(
    value: object,
    *,
    projector: object | None = None,
) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    keys = set(value)
    if keys - _PROJECTOR_PROVENANCE_KEYS or (
        projector is None and keys != _PROJECTOR_PROVENANCE_KEYS
    ):
        return None
    workflow_path = value.get("workflow_path")
    projector_kind = value.get("projector_kind")
    if (
        not isinstance(workflow_path, str)
        or workflow_path not in _PROJECTOR_KIND_BY_WORKFLOW
        or projector_kind != _PROJECTOR_KIND_BY_WORKFLOW[workflow_path]
    ):
        return None
    identity = _projector_identity(projector) if projector is not None else None
    declared_shape = value.get("projector_shape")
    declared_fingerprint = value.get("projector_fingerprint")
    if identity is not None:
        shape, fingerprint = identity
        if declared_shape is not None and (
            not isinstance(declared_shape, list)
            or declared_shape != shape
        ):
            return None
        if (
            declared_fingerprint is not None
            and declared_fingerprint != fingerprint
        ):
            return None
    else:
        shape = declared_shape
        fingerprint = declared_fingerprint
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item <= 0
            for item in shape
        )
        or shape[0] != shape[1]
        or not isinstance(fingerprint, str)
        or not fingerprint.startswith("sha256:")
        or len(fingerprint) != 71
        or any(
            character not in "0123456789abcdef"
            for character in fingerprint[7:]
        )
    ):
        return None
    return {
        "workflow_path": str(workflow_path),
        "projector_kind": str(projector_kind),
        "projector_shape": list(shape),
        "projector_fingerprint": fingerprint,
    }


def _projector_identity(
    projector: object,
) -> tuple[list[int], str] | None:
    try:
        matrix = np.ascontiguousarray(
            np.asarray(projector, dtype=np.dtype("<c16"))
        )
    except (TypeError, ValueError):
        return None
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] <= 0
        or not np.all(np.isfinite(matrix))
    ):
        return None
    shape = [int(value) for value in matrix.shape]
    shape_bytes = np.asarray(shape, dtype=np.dtype("<i8")).tobytes()
    digest = hashlib.sha256(shape_bytes + matrix.tobytes(order="C")).hexdigest()
    return shape, f"sha256:{digest}"


def _vector3(value: object) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _integer_g_vectors(value: object) -> np.ndarray | None:
    try:
        vectors = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        vectors.ndim != 2
        or vectors.shape[1] != 3
        or not np.all(np.isfinite(vectors))
        or not np.allclose(vectors, np.rint(vectors), atol=1e-8, rtol=0.0)
    ):
        return None
    return np.rint(vectors).astype(int)


def _integer_vector(value: object) -> np.ndarray | None:
    try:
        vector = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.integer):
        return None
    return vector.astype(int)


def _deduplicate(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out
