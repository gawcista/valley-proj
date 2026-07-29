"""Producer-owned canonical row target-frame construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from valleyscope.io.wavefunction_convention import canonical_identity


TARGET_FRAME_SCHEMA_VERSION = "1.0.0"
_RTAG_COMPLEX_DTYPES = {
    45200: np.dtype(np.complex64),
    45210: np.dtype(np.complex128),
    53300: np.dtype(np.complex128),
    53310: np.dtype(np.complex128),
}


@dataclass(frozen=True)
class TargetFrameResult:
    coefficients: np.ndarray | None
    transform: np.ndarray | None
    record: dict[str, object]
    reason_codes: tuple[str, ...]

    @property
    def status(self) -> str:
        return "passed" if not self.reason_codes else "blocked"


@dataclass(frozen=True)
class TargetFrameValidation:
    status: str
    reason_codes: tuple[str, ...]


def build_target_frame(
    coefficients: np.ndarray,
    *,
    wavecar_rtag: int | None,
) -> TargetFrameResult:
    """Build a canonical orthonormal frame from row-oriented coefficients.

    A non-identity normalization is accepted only when it is a near-identity
    correction supported by the WAVECAR source precision.  The correction
    limit is ``sqrt(source epsilon)`` so its second-order contribution remains
    below source machine precision.
    """
    reasons: list[str] = []
    try:
        source = np.asarray(coefficients)
        array = np.asarray(coefficients, dtype=np.complex128)
    except (TypeError, ValueError):
        return _blocked_malformed()
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] not in (1, 2)
        or array.shape[2] < 1
        or not np.all(np.isfinite(array))
    ):
        return _blocked_malformed()

    dimension = int(array.shape[0])
    flattened_dimension = int(array.shape[1] * array.shape[2])
    flattened = array.reshape(dimension, flattened_dimension)
    gram = flattened @ flattened.conj().T
    identity = np.eye(dimension, dtype=np.complex128)
    gram_delta = gram - identity
    gram_raw_fro = float(np.linalg.norm(gram_delta, ord="fro"))
    gram_normalization = float(np.sqrt(dimension))
    gram_error = gram_raw_fro / gram_normalization
    exact_identity = bool(np.array_equal(gram, identity))

    try:
        eigenvalues = np.linalg.eigvalsh(gram)
    except np.linalg.LinAlgError:
        return _blocked_malformed()
    if not np.all(np.isfinite(eigenvalues)):
        return _blocked_malformed()
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    condition_number: float | None = (
        float(maximum / minimum)
        if minimum > 0.0
        else None
    )

    source_precision = _source_precision_record(
        wavecar_rtag,
        source.dtype,
        exact_identity=exact_identity,
        reasons=reasons,
    )
    epsilon = source_precision.get("machine_epsilon")
    correction_limit = (
        float(np.sqrt(float(epsilon)))
        if isinstance(epsilon, float)
        else None
    )
    rank_floor = (
        float(float(epsilon) * dimension * max(maximum, 1.0))
        if isinstance(epsilon, float)
        else None
    )
    condition_limit = (
        float(1.0 / np.sqrt(float(epsilon)))
        if isinstance(epsilon, float)
        else None
    )

    transform: np.ndarray | None = None
    canonical: np.ndarray | None = None
    correction_raw_fro: float | None = None
    correction_size: float | None = None
    coefficient_correction_size: float | None = None
    post_gram_error: float | None = None
    method = "identity"
    applied = False

    if exact_identity:
        transform = identity
        canonical = array.copy()
        correction_raw_fro = 0.0
        correction_size = 0.0
        coefficient_correction_size = 0.0
        post_gram_error = 0.0
    elif isinstance(epsilon, float):
        if minimum <= float(rank_floor):
            reasons.append("target_frame_rank_deficient")
        elif (
            condition_number is not None
            and condition_number > float(condition_limit)
        ):
            reasons.append("target_frame_ill_conditioned")
        else:
            inverse_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
            _, eigenvectors = np.linalg.eigh(gram)
            candidate_transform = (
                eigenvectors @ inverse_sqrt @ eigenvectors.conj().T
            )
            correction_raw_fro = float(
                np.linalg.norm(candidate_transform - identity, ord="fro")
            )
            correction_size = correction_raw_fro / gram_normalization
            if correction_size > float(correction_limit):
                reasons.append(
                    "target_frame_correction_exceeds_precision_bound"
                )
            else:
                candidate = candidate_transform @ flattened
                post_gram = candidate @ candidate.conj().T
                post_gram_error = float(
                    np.linalg.norm(post_gram - identity, ord="fro")
                    / gram_normalization
                )
                working_limit = float(
                    np.sqrt(np.finfo(np.float64).eps)
                )
                if (
                    not np.all(np.isfinite(candidate_transform))
                    or not np.all(np.isfinite(candidate))
                    or not np.isfinite(post_gram_error)
                    or post_gram_error > working_limit
                ):
                    reasons.append("target_frame_normalization_failed")
                else:
                    transform = candidate_transform
                    canonical = candidate.reshape(array.shape)
                    coefficient_correction_size = float(
                        np.linalg.norm(candidate - flattened)
                        / max(np.linalg.norm(flattened), np.finfo(float).tiny)
                    )
                    method = "symmetric_inverse_square_root"
                    applied = True

    record_content: dict[str, object] = {
        "schema_version": TARGET_FRAME_SCHEMA_VERSION,
        "source_precision": source_precision,
        "shape": {
            "bands": dimension,
            "spinor_components": int(array.shape[1]),
            "reciprocal_grid_points": int(array.shape[2]),
            "flattened_dimension": flattened_dimension,
        },
        "source_coefficients_identity": _array_identity(array),
        "gram_convention": "C @ C.conj().T for row C[band,spinor,G]",
        "original_gram": {
            "spectrum": [float(value) for value in eigenvalues],
            "minimum_eigenvalue": minimum,
            "maximum_eigenvalue": maximum,
            "condition_number": condition_number,
            "raw_fro_error": gram_raw_fro,
            "normalization": gram_normalization,
            "normalized_error": gram_error,
        },
        "admissibility": {
            "rank_eigenvalue_floor": rank_floor,
            "condition_number_limit": condition_limit,
            "correction_size_limit": correction_limit,
            "correction_policy": (
                "normalized ||A-I||_F <= sqrt(source machine epsilon)"
            ),
        },
        "canonicalization": {
            "method": method,
            "applied": applied,
            "transform": (
                _complex_matrix_record(transform)
                if transform is not None
                else None
            ),
            "transform_identity": (
                _array_identity(transform)
                if transform is not None
                else None
            ),
            "correction_raw_fro": correction_raw_fro,
            "correction_normalization": gram_normalization,
            "correction_size": correction_size,
            "coefficient_relative_correction": coefficient_correction_size,
            "post_gram_error": post_gram_error,
            "canonical_coefficients_identity": (
                _array_identity(canonical)
                if canonical is not None
                else None
            ),
        },
        "status": "passed" if not reasons else "blocked",
        "reason_codes": _unique(reasons),
    }
    record_content["contract_identity"] = canonical_identity(record_content)
    return TargetFrameResult(
        coefficients=canonical,
        transform=transform,
        record=record_content,
        reason_codes=tuple(record_content["reason_codes"]),
    )


def validate_target_frame_record(
    record: object,
    coefficients: np.ndarray,
    *,
    wavecar_rtag: int | None,
) -> TargetFrameValidation:
    """Recompute a target-frame record from producer inputs."""
    if not isinstance(record, dict):
        return TargetFrameValidation(
            "blocked",
            ("target_frame_record_malformed",),
        )
    recomputed = build_target_frame(
        coefficients,
        wavecar_rtag=wavecar_rtag,
    )
    if record != recomputed.record:
        return TargetFrameValidation(
            "blocked",
            ("target_frame_recomputation_mismatch",),
        )
    return TargetFrameValidation(
        recomputed.status,
        recomputed.reason_codes,
    )


def _source_precision_record(
    wavecar_rtag: int | None,
    storage_dtype: np.dtype,
    *,
    exact_identity: bool,
    reasons: list[str],
) -> dict[str, object]:
    if wavecar_rtag is None:
        if not exact_identity:
            reasons.append("target_frame_source_precision_missing")
        return {
            "wavecar_rtag": None,
            "complex_dtype": None,
            "real_component_dtype": None,
            "machine_epsilon": None,
            "hdf5_storage_dtype": str(storage_dtype),
            "required_for_correction": not exact_identity,
        }
    if isinstance(wavecar_rtag, bool) or wavecar_rtag not in _RTAG_COMPLEX_DTYPES:
        reasons.append("target_frame_source_precision_unsupported")
        return {
            "wavecar_rtag": wavecar_rtag,
            "complex_dtype": None,
            "real_component_dtype": None,
            "machine_epsilon": None,
            "hdf5_storage_dtype": str(storage_dtype),
            "required_for_correction": not exact_identity,
        }
    complex_dtype = _RTAG_COMPLEX_DTYPES[wavecar_rtag]
    real_dtype = (
        np.dtype(np.float32)
        if complex_dtype == np.dtype(np.complex64)
        else np.dtype(np.float64)
    )
    return {
        "wavecar_rtag": int(wavecar_rtag),
        "complex_dtype": complex_dtype.name,
        "real_component_dtype": real_dtype.name,
        "machine_epsilon": float(np.finfo(real_dtype).eps),
        "hdf5_storage_dtype": str(storage_dtype),
        "required_for_correction": not exact_identity,
    }


def _blocked_malformed() -> TargetFrameResult:
    record: dict[str, object] = {
        "schema_version": TARGET_FRAME_SCHEMA_VERSION,
        "status": "blocked",
        "reason_codes": ["target_coefficients_malformed"],
    }
    record["contract_identity"] = canonical_identity(record)
    return TargetFrameResult(
        coefficients=None,
        transform=None,
        record=record,
        reason_codes=("target_coefficients_malformed",),
    )


def _array_identity(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(
        np.asarray(array, dtype=np.complex128)
    )
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(normalized.dtype.str.encode("ascii"))
    digest.update(normalized.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _complex_matrix_record(matrix: np.ndarray) -> list[list[list[float]]]:
    return [
        [
            [float(value.real), float(value.imag)]
            for value in row
        ]
        for row in np.asarray(matrix, dtype=np.complex128)
    ]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
