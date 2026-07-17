"""Numerical time-reversal sewing evidence in the selected band subspace."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_TOL = 1e-3
_K_TOL = 5e-6


def build_time_reversal_sewing_report(
    *,
    kpoint_frac_by_name: Mapping[str, np.ndarray],
    g_vectors_frac_by_kpoint: Mapping[str, np.ndarray],
    coefficients_by_kpoint: Mapping[str, np.ndarray],
    band_indices_by_kpoint: Mapping[str, np.ndarray],
    valley_projectors_by_kpoint: Mapping[
        str, Mapping[str, np.ndarray]
    ],
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
    blockers: list[str] = []
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
        row["projector_covariance"] = _projector_covariance(
            name=name,
            partner=partner,
            sewing=sewing,
            source_projectors=valley_projectors_by_kpoint.get(name),
            partner_projectors=valley_projectors_by_kpoint.get(partner),
            valley_mapping=time_reversal_valley_mapping,
            tolerance=tolerance,
            blockers=row_blockers,
        )
        row["blockers"] = _deduplicate(row_blockers)
        row["status"] = "validated" if not row["blockers"] else "blocked"
        blockers.extend(row["blockers"])
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
        "reciprocal_shifts_by_kpoint": reciprocal_shifts,
        "rows": rows,
        "blockers": blockers,
    }


def validate_time_reversal_sewing_report(
    evidence: Mapping[str, object] | None,
    *,
    valley_members: list[str],
    theta_square: object,
) -> bool:
    """Validate serialized sewing evidence before readiness promotion."""
    if not isinstance(evidence, Mapping) or evidence.get("status") != "validated":
        return False
    if theta_square not in (-1, 1) or evidence.get("theta_square") != theta_square:
        return False
    kpoint_mapping = evidence.get("time_reversal_kpoint_mapping")
    if not isinstance(kpoint_mapping, Mapping) or not kpoint_mapping:
        return False
    if set(kpoint_mapping) != set(kpoint_mapping.values()) or any(
        kpoint_mapping.get(kpoint_mapping.get(name, "")) != name
        for name in kpoint_mapping
    ):
        return False
    rows = evidence.get("rows")
    if not isinstance(rows, list) or len(rows) != len(kpoint_mapping):
        return False
    rows_by_source = {
        row.get("source_kpoint"): row
        for row in rows if isinstance(row, Mapping)
    }
    if set(rows_by_source) != set(kpoint_mapping):
        return False
    for source, target in kpoint_mapping.items():
        row = rows_by_source[source]
        if (
            row.get("status") != "validated"
            or row.get("target_kpoint") != target
            or row.get("theta_square") != theta_square
            or row.get("blockers") != []
        ):
            return False
        for key in (
            "target_subspace_closure_residual",
            "sewing_unitarity_residual",
            "theta_square_residual",
        ):
            value = row.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value > _TOL
            ):
                return False
        singular_values = row.get("overlap_singular_values")
        if (
            not isinstance(singular_values, list)
            or not singular_values
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or abs(float(value) - 1.0) > _TOL
                for value in singular_values
            )
        ):
            return False
        covariance = row.get("projector_covariance")
        if not isinstance(covariance, Mapping) or any(
            not isinstance(covariance.get(valley), Mapping)
            or covariance[valley].get("status") != "validated"
            for valley in valley_members
        ):
            return False
    return True


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
    valley_mapping: Mapping[str, str],
    tolerance: float,
    blockers: list[str],
) -> dict[str, dict[str, object]]:
    if sewing is None:
        blockers.append(f"time_reversal_projector_covariance_not_evaluated:{name}")
        return {}
    if not isinstance(source_projectors, Mapping) or not isinstance(
        partner_projectors, Mapping
    ):
        blockers.append(f"time_reversal_seed_projectors_missing:{name}:{partner}")
        return {}
    result: dict[str, dict[str, object]] = {}
    for valley, partner_valley in valley_mapping.items():
        source = source_projectors.get(valley)
        target = partner_projectors.get(partner_valley)
        try:
            source_matrix = np.asarray(source, dtype=complex)
            target_matrix = np.asarray(target, dtype=complex)
        except (TypeError, ValueError):
            source_matrix = np.asarray([])
            target_matrix = np.asarray([])
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
            }
            continue
        transformed = sewing @ source_matrix.conj() @ sewing.conj().T
        residual = float(
            np.linalg.norm(transformed - target_matrix)
            / max(float(np.linalg.norm(target_matrix)), 1e-30)
        )
        status = "validated" if residual <= tolerance else "blocked"
        if status == "blocked":
            blockers.append(
                f"time_reversal_projector_covariance_failed:{name}:{valley}:"
                f"{residual:.6e}"
            )
        result[valley] = {
            "partner_valley": partner_valley,
            "status": status,
            "covariance_residual": residual,
        }
    return result


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
