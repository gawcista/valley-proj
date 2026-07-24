"""Producer-owned numerical evidence for one exact representation scope."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from valleyscope.io.spinor_source_basis import (
    validate_spinor_source_basis_record,
)
from valleyscope.io.wavefunction_convention import (
    canonical_identity,
    valid_sha256_identity,
)
from valleyscope.symmetry.double_space_group_lift import (
    validate_double_space_group_lift_record,
)


SCOPED_REPRESENTATION_EVIDENCE_SCHEMA_VERSION = "1.0.0"
SUPPORTED_SCOPE_KINDS = frozenset(
    {"local_irrep", "valley_sewing", "tr_completed"}
)
DEFAULT_NUMERICAL_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class ScopedEvidenceValidation:
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ScopedRepresentationEvidence:
    _content: dict[str, object]
    _reason_codes: tuple[str, ...]

    @property
    def status(self) -> str:
        return "passed" if not self._reason_codes else "blocked"

    def to_record(self) -> dict[str, object]:
        record = deepcopy(self._content)
        record["status"] = self.status
        record["reason_codes"] = list(self._reason_codes)
        record["evidence_identity"] = canonical_identity(self._content)
        return record


def build_scoped_representation_evidence(
    *,
    source_basis_record: Mapping[str, object],
    lift_record: Mapping[str, object],
    lift_validation_inputs: Mapping[str, object],
    extracted_wavefunction_payload_identity: str,
    kpoint_label: str,
    kpoint_frac: np.ndarray,
    scope_kind: str,
    source_valleys: Sequence[str],
    valley_orbit: Sequence[str],
    required_operation_ids: Sequence[int],
    representations: Mapping[int, np.ndarray],
    plane_wave_evidence: Mapping[int, Mapping[str, object]],
    target_coefficients: np.ndarray,
    projectors: Mapping[str, np.ndarray],
    valley_bases: Mapping[str, np.ndarray],
    valley_mappings: Mapping[int, Mapping[str, str]],
    antiunitary_evidence: Mapping[str, object] | None = None,
    numerical_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
) -> ScopedRepresentationEvidence:
    """Derive one immutable scope record from raw numerical inputs.

    No positive validation status is accepted as input.  The source and lift
    records are revalidated against their producer inputs before any numerical
    representation result can pass.
    """
    reasons: list[str] = []
    tolerance = _positive_tolerance(numerical_tolerance, reasons)

    source_validation = validate_spinor_source_basis_record(source_basis_record)
    if source_validation.status != "passed":
        reasons.append("source_basis_certificate_not_passed")
    source_identity = source_basis_record.get("certificate_identity")
    source_payload_identity = source_basis_record.get(
        "extracted_wavefunction_payload_identity"
    )
    if (
        not valid_sha256_identity(extracted_wavefunction_payload_identity)
        or extracted_wavefunction_payload_identity != source_payload_identity
    ):
        reasons.append("extracted_payload_identity_mismatch")

    lift_validation = _validate_lift(
        lift_record=lift_record,
        source_basis_record=source_basis_record,
        lift_validation_inputs=lift_validation_inputs,
        required_operation_ids=required_operation_ids,
    )
    if lift_validation.status != "passed":
        reasons.append("double_space_group_lift_not_passed")
        reasons.extend(lift_validation.reason_codes)
    lift_identity = lift_record.get("certificate_identity")

    normalized_scope_kind = str(scope_kind)
    if normalized_scope_kind not in SUPPORTED_SCOPE_KINDS:
        reasons.append("scope_kind_unsupported")
    normalized_kpoint = _kpoint_record(kpoint_frac, reasons)
    normalized_source_valleys = _unique_strings(
        source_valleys, "source_valley_malformed", reasons
    )
    normalized_orbit = _unique_strings(
        valley_orbit, "valley_orbit_malformed", reasons
    )
    if not normalized_source_valleys:
        reasons.append("source_valley_missing")
    if not set(normalized_source_valleys).issubset(set(normalized_orbit)):
        reasons.append("source_valley_outside_orbit")
    operation_ids = _opaque_operation_ids(required_operation_ids, reasons)

    lift_operation_ids = lift_record.get("operation_ids")
    if not isinstance(lift_operation_ids, list) or not set(operation_ids).issubset(
        set(lift_operation_ids)
    ):
        reasons.append("required_operation_lift_evidence_missing")

    matrices, target_dimension = _representation_matrices(
        operation_ids, representations, reasons
    )
    coefficient_gram_error = _coefficient_gram_error(
        target_coefficients, target_dimension, reasons
    )
    closure_rows = _target_subspace_rows(
        operation_ids, matrices, tolerance, reasons
    )
    plane_wave_rows = _plane_wave_rows(
        operation_ids, plane_wave_evidence, tolerance, reasons
    )

    preserving_ids, changing_ids = _classify_operation_scope(
        operation_ids=operation_ids,
        source_valleys=normalized_source_valleys,
        orbit=normalized_orbit,
        valley_mappings=valley_mappings,
        reasons=reasons,
    )
    _validate_scope_semantics(
        scope_kind=normalized_scope_kind,
        preserving_ids=preserving_ids,
        changing_ids=changing_ids,
        reasons=reasons,
    )

    group_law_rows = _projected_group_law_rows(
        operation_ids=operation_ids,
        representations=matrices,
        lift_record=lift_record,
        kpoint_frac=np.asarray(normalized_kpoint, dtype=float),
        tolerance=tolerance,
        reasons=reasons,
    )
    covariance_rows = _projector_covariance_rows(
        operation_ids=operation_ids,
        source_valleys=normalized_source_valleys,
        representations=matrices,
        projectors=projectors,
        valley_mappings=valley_mappings,
        tolerance=tolerance,
        reasons=reasons,
    )
    block_rows = _valley_block_rows(
        operation_ids=operation_ids,
        source_valleys=normalized_source_valleys,
        representations=matrices,
        valley_bases=valley_bases,
        valley_mappings=valley_mappings,
        tolerance=tolerance,
        reasons=reasons,
    )
    antiunitary_record = _antiunitary_record(
        required=normalized_scope_kind == "tr_completed",
        evidence=antiunitary_evidence,
        valley_orbit=normalized_orbit,
        tolerance=tolerance,
        reasons=reasons,
    )

    content: dict[str, object] = {
        "schema_version": SCOPED_REPRESENTATION_EVIDENCE_SCHEMA_VERSION,
        "source_basis_certificate_identity": source_identity,
        "double_space_group_lift_certificate_identity": lift_identity,
        "extracted_wavefunction_payload_identity": (
            extracted_wavefunction_payload_identity
        ),
        "scope": {
            "scope_kind": normalized_scope_kind,
            "kpoint_label": str(kpoint_label),
            "kpoint_frac": normalized_kpoint,
            "source_valleys": normalized_source_valleys,
            "valley_orbit": normalized_orbit,
            "required_operation_ids": operation_ids,
            "valley_preserving_operation_ids": preserving_ids,
            "valley_changing_operation_ids": changing_ids,
        },
        "numerical_tolerance": tolerance,
        "target_subspace": {
            "dimension": target_dimension,
            "coefficient_gram_error": coefficient_gram_error,
            "coefficient_gram_passed": (
                coefficient_gram_error is not None
                and coefficient_gram_error <= tolerance
            ),
            "operation_rows": closure_rows,
        },
        "plane_wave_mapping": {"operation_rows": plane_wave_rows},
        "projected_representation_group_law": {
            "pair_rows": group_law_rows
        },
        "projector_covariance": {"operation_rows": covariance_rows},
        "valley_block_quality": {"operation_rows": block_rows},
        "antiunitary_evidence": antiunitary_record,
        "grey_group_matching_allowed": False,
    }
    return ScopedRepresentationEvidence(
        _content=content,
        _reason_codes=tuple(_unique(reasons)),
    )


def validate_scoped_representation_evidence_record(
    record: Mapping[str, object],
    **raw_inputs: object,
) -> ScopedEvidenceValidation:
    """Recompute a scoped record from raw evidence and require exact equality."""
    if not isinstance(record, Mapping):
        return ScopedEvidenceValidation("blocked", ("record_malformed",))
    reasons: list[str] = []
    try:
        recomputed = build_scoped_representation_evidence(
            **raw_inputs
        ).to_record()
    except (KeyError, TypeError, ValueError):
        recomputed = None
        reasons.append("evidence_recomputation_failed")

    if recomputed is not None:
        if record.get("status") != recomputed.get("status"):
            reasons.append("derived_status_mismatch")
        if record.get("reason_codes") != recomputed.get("reason_codes"):
            reasons.append("derived_reason_codes_mismatch")
        if dict(record) != recomputed:
            reasons.append("recomputed_evidence_mismatch")

    content = {
        key: deepcopy(value)
        for key, value in record.items()
        if key not in {"status", "reason_codes", "evidence_identity"}
    }
    try:
        expected_identity = canonical_identity(content)
    except (TypeError, ValueError):
        expected_identity = None
    if (
        expected_identity is None
        or record.get("evidence_identity") != expected_identity
        or not valid_sha256_identity(record.get("evidence_identity"))
    ):
        reasons.append("evidence_identity_mismatch")

    if recomputed is None:
        expected_status = "blocked"
        expected_reasons: tuple[str, ...] = ()
    else:
        expected_status = str(recomputed["status"])
        serialized = recomputed.get("reason_codes", [])
        expected_reasons = tuple(
            value for value in serialized if isinstance(value, str)
        )
    reasons = _unique(reasons)
    if reasons:
        return ScopedEvidenceValidation("blocked", tuple(reasons))
    return ScopedEvidenceValidation(expected_status, expected_reasons)


def validate_scoped_representation_evidence_link(
    record: Mapping[str, object],
    *,
    expected_evidence_identity: str,
    expected_source_basis_certificate_identity: str,
    expected_lift_certificate_identity: str,
    expected_payload_identity: str,
) -> ScopedEvidenceValidation:
    """Validate a downstream identity link without accepting a self-signed hash."""
    reasons: list[str] = []
    if record.get("evidence_identity") != expected_evidence_identity:
        reasons.append("evidence_identity_link_mismatch")
    if (
        record.get("source_basis_certificate_identity")
        != expected_source_basis_certificate_identity
    ):
        reasons.append("source_basis_identity_link_mismatch")
    if (
        record.get("double_space_group_lift_certificate_identity")
        != expected_lift_certificate_identity
    ):
        reasons.append("lift_identity_link_mismatch")
    if (
        record.get("extracted_wavefunction_payload_identity")
        != expected_payload_identity
    ):
        reasons.append("payload_identity_link_mismatch")
    content = {
        key: deepcopy(value)
        for key, value in record.items()
        if key not in {"status", "reason_codes", "evidence_identity"}
    }
    try:
        derived_identity = canonical_identity(content)
    except (TypeError, ValueError):
        derived_identity = None
    if derived_identity != expected_evidence_identity:
        reasons.append("evidence_content_identity_mismatch")
    if record.get("status") not in {"passed", "blocked"}:
        reasons.append("derived_status_malformed")
    if not isinstance(record.get("reason_codes"), list):
        reasons.append("reason_codes_malformed")
    reasons = _unique(reasons)
    if reasons:
        return ScopedEvidenceValidation("blocked", tuple(reasons))
    return ScopedEvidenceValidation(
        str(record["status"]),
        tuple(str(value) for value in record["reason_codes"]),
    )


def _validate_lift(
    *,
    lift_record: Mapping[str, object],
    source_basis_record: Mapping[str, object],
    lift_validation_inputs: Mapping[str, object],
    required_operation_ids: Sequence[int],
):
    return validate_double_space_group_lift_record(
        lift_record,
        source_basis_record=source_basis_record,
        source_table_identity=lift_validation_inputs["source_table_identity"],
        standard_setting_identity=lift_validation_inputs[
            "standard_setting_identity"
        ],
        direct_lattice_cart=np.asarray(
            lift_validation_inputs["direct_lattice_cart"], dtype=float
        ),
        expected_operations=lift_validation_inputs["expected_operations"],
        required_operation_ids=required_operation_ids,
    )


def _positive_tolerance(value: float, reasons: list[str]) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError):
        reasons.append("numerical_tolerance_malformed")
        return DEFAULT_NUMERICAL_TOLERANCE
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        reasons.append("numerical_tolerance_malformed")
        return DEFAULT_NUMERICAL_TOLERANCE
    return tolerance


def _kpoint_record(value: np.ndarray, reasons: list[str]) -> list[float]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        array = np.zeros(3)
        reasons.append("kpoint_malformed")
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        array = np.zeros(3)
        reasons.append("kpoint_malformed")
    return [float(item) for item in array]


def _unique_strings(
    values: Sequence[str], reason: str, reasons: list[str]
) -> list[str]:
    result: list[str] = []
    try:
        candidates = list(values)
    except TypeError:
        reasons.append(reason)
        return result
    for value in candidates:
        if not isinstance(value, str) or not value:
            reasons.append(reason)
            continue
        if value in result:
            reasons.append(reason)
            continue
        result.append(value)
    return result


def _opaque_operation_ids(
    values: Sequence[int], reasons: list[str]
) -> list[int]:
    result: list[int] = []
    try:
        candidates = list(values)
    except TypeError:
        reasons.append("required_operation_id_malformed")
        return result
    for value in candidates:
        if not isinstance(value, int) or isinstance(value, bool):
            reasons.append("required_operation_id_malformed")
            continue
        if value in result:
            reasons.append("required_operation_id_duplicate")
            continue
        result.append(value)
    if not result:
        reasons.append("required_operation_evidence_missing")
    return result


def _representation_matrices(
    operation_ids: Sequence[int],
    representations: Mapping[int, np.ndarray],
    reasons: list[str],
) -> tuple[dict[int, np.ndarray], int]:
    matrices: dict[int, np.ndarray] = {}
    dimension = 0
    for operation_id in operation_ids:
        value = representations.get(operation_id)
        try:
            matrix = np.asarray(value, dtype=np.complex128)
        except (TypeError, ValueError):
            matrix = np.empty((0, 0), dtype=np.complex128)
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 1
            or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))
        ):
            reasons.append("representation_matrix_malformed")
            continue
        if dimension == 0:
            dimension = int(matrix.shape[0])
        elif matrix.shape != (dimension, dimension):
            reasons.append("representation_dimension_mismatch")
            continue
        matrices[operation_id] = matrix
    if set(matrices) != set(operation_ids):
        reasons.append("required_representation_missing")
    return matrices, dimension


def _coefficient_gram_error(
    coefficients: np.ndarray,
    dimension: int,
    reasons: list[str],
) -> float | None:
    try:
        array = np.asarray(coefficients, dtype=np.complex128)
    except (TypeError, ValueError):
        reasons.append("target_coefficients_malformed")
        return None
    if array.ndim < 2 or array.shape[0] != dimension or dimension < 1:
        reasons.append("target_coefficients_malformed")
        return None
    flattened = array.reshape(dimension, -1)
    gram_error = float(
        np.linalg.norm(
            flattened @ flattened.conj().T - np.eye(dimension), ord="fro"
        )
    )
    if not np.isfinite(gram_error):
        reasons.append("target_coefficients_malformed")
        return None
    return gram_error


def _target_subspace_rows(
    operation_ids: Sequence[int],
    matrices: Mapping[int, np.ndarray],
    tolerance: float,
    reasons: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for operation_id in operation_ids:
        matrix = matrices.get(operation_id)
        if matrix is None:
            continue
        residual = float(
            np.linalg.norm(
                matrix.conj().T @ matrix - np.eye(matrix.shape[0]), ord="fro"
            )
        )
        passed = residual <= tolerance
        if not passed:
            reasons.append("target_subspace_closure_failed")
        rows.append(
            {
                "operation_id": operation_id,
                "unitarity_residual": residual,
                "passed": passed,
            }
        )
    return rows


def _plane_wave_rows(
    operation_ids: Sequence[int],
    evidence: Mapping[int, Mapping[str, object]],
    tolerance: float,
    reasons: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for operation_id in operation_ids:
        raw = evidence.get(operation_id)
        if not isinstance(raw, Mapping):
            reasons.append("plane_wave_mapping_evidence_missing")
            continue
        miss_count = raw.get("mapping_miss_count")
        residual = raw.get("norm_preservation_residual")
        if (
            not isinstance(miss_count, int)
            or isinstance(miss_count, bool)
            or miss_count < 0
        ):
            reasons.append("plane_wave_mapping_evidence_malformed")
            continue
        try:
            norm_residual = float(residual)
        except (TypeError, ValueError):
            reasons.append("plane_wave_mapping_evidence_malformed")
            continue
        passed = (
            miss_count == 0
            and np.isfinite(norm_residual)
            and norm_residual <= tolerance
        )
        if not passed:
            reasons.append("plane_wave_mapping_failed")
        rows.append(
            {
                "operation_id": operation_id,
                "mapping_miss_count": miss_count,
                "norm_preservation_residual": norm_residual,
                "passed": bool(passed),
            }
        )
    return rows


def _classify_operation_scope(
    *,
    operation_ids: Sequence[int],
    source_valleys: Sequence[str],
    orbit: Sequence[str],
    valley_mappings: Mapping[int, Mapping[str, str]],
    reasons: list[str],
) -> tuple[list[int], list[int]]:
    preserving: list[int] = []
    changing: list[int] = []
    orbit_set = set(orbit)
    for operation_id in operation_ids:
        mapping = valley_mappings.get(operation_id)
        if not isinstance(mapping, Mapping):
            reasons.append("valley_mapping_missing")
            continue
        operation_is_preserving = True
        operation_is_changing = False
        for valley in source_valleys:
            mapped = mapping.get(valley)
            if not isinstance(mapped, str) or mapped not in orbit_set:
                reasons.append("valley_mapping_incomplete")
                operation_is_preserving = False
                continue
            if mapped != valley:
                operation_is_preserving = False
                operation_is_changing = True
        if operation_is_preserving:
            preserving.append(operation_id)
        elif operation_is_changing:
            changing.append(operation_id)
    return preserving, changing


def _validate_scope_semantics(
    *,
    scope_kind: str,
    preserving_ids: Sequence[int],
    changing_ids: Sequence[int],
    reasons: list[str],
) -> None:
    if scope_kind == "local_irrep" and changing_ids:
        reasons.append("local_irrep_scope_contains_valley_changing_operation")
    if scope_kind in {"valley_sewing", "tr_completed"} and not changing_ids:
        reasons.append("valley_changing_scope_evidence_missing")
    if scope_kind == "local_irrep" and not preserving_ids:
        reasons.append("valley_preserving_scope_evidence_missing")


def _projected_group_law_rows(
    *,
    operation_ids: Sequence[int],
    representations: Mapping[int, np.ndarray],
    lift_record: Mapping[str, object],
    kpoint_frac: np.ndarray,
    tolerance: float,
    reasons: list[str],
) -> list[dict[str, object]]:
    pairwise = lift_record.get("pairwise_products")
    inventory = lift_record.get("operation_inventory")
    if not isinstance(pairwise, Mapping) or not isinstance(inventory, list):
        reasons.append("lift_pairwise_evidence_missing")
        return []
    rotations: dict[int, np.ndarray] = {}
    for operation in inventory:
        if not isinstance(operation, Mapping):
            continue
        operation_id = operation.get("operation_id")
        if not isinstance(operation_id, int) or isinstance(operation_id, bool):
            continue
        try:
            rotations[operation_id] = np.asarray(
                operation["rotation_frac"], dtype=float
            )
        except (KeyError, TypeError, ValueError):
            continue

    required = set(operation_ids)
    rows: list[dict[str, object]] = []
    for left_id in operation_ids:
        for right_id in operation_ids:
            pair = pairwise.get(f"{left_id},{right_id}")
            if not isinstance(pair, Mapping):
                reasons.append("projected_representation_product_missing")
                continue
            product_id = pair.get("product_operation_id")
            if product_id not in required:
                reasons.append("required_operation_scope_not_closed")
                continue
            left = representations.get(left_id)
            right = representations.get(right_id)
            product = representations.get(product_id)
            rotation = rotations.get(product_id)
            if (
                left is None
                or right is None
                or product is None
                or rotation is None
            ):
                continue
            lattice_translation = np.asarray(
                pair.get("lattice_translation_frac"), dtype=float
            )
            if lattice_translation.shape != (3,):
                reasons.append("projected_representation_product_malformed")
                continue
            transformed_k = np.linalg.inv(rotation).T @ kpoint_frac
            phase = np.exp(
                -2.0j * np.pi * float(transformed_k @ lattice_translation)
            )
            sign = pair.get("cocycle_sign")
            if sign not in (-1, 1):
                reasons.append("projected_representation_product_malformed")
                continue
            residual = float(
                np.linalg.norm(left @ right - sign * phase * product, ord="fro")
            )
            passed = residual <= tolerance
            if not passed:
                reasons.append("projected_representation_group_law_failed")
            rows.append(
                {
                    "left_operation_id": left_id,
                    "right_operation_id": right_id,
                    "product_operation_id": product_id,
                    "cocycle_sign": sign,
                    "bloch_factor": [float(phase.real), float(phase.imag)],
                    "residual": residual,
                    "passed": passed,
                }
            )
    return rows


def _projector_covariance_rows(
    *,
    operation_ids: Sequence[int],
    source_valleys: Sequence[str],
    representations: Mapping[int, np.ndarray],
    projectors: Mapping[str, np.ndarray],
    valley_mappings: Mapping[int, Mapping[str, str]],
    tolerance: float,
    reasons: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for operation_id in operation_ids:
        matrix = representations.get(operation_id)
        mapping = valley_mappings.get(operation_id)
        if matrix is None or not isinstance(mapping, Mapping):
            continue
        for valley in source_valleys:
            mapped = mapping.get(valley)
            try:
                source_projector = np.asarray(
                    projectors[valley], dtype=np.complex128
                )
                target_projector = np.asarray(
                    projectors[mapped], dtype=np.complex128
                )
            except (KeyError, TypeError, ValueError):
                reasons.append("projector_covariance_input_missing")
                continue
            if (
                source_projector.shape != matrix.shape
                or target_projector.shape != matrix.shape
            ):
                reasons.append("projector_covariance_input_malformed")
                continue
            denominator = max(
                float(np.linalg.norm(source_projector, ord="fro")), 1.0e-14
            )
            residual = float(
                np.linalg.norm(
                    matrix @ source_projector @ matrix.conj().T
                    - target_projector,
                    ord="fro",
                )
                / denominator
            )
            passed = residual <= tolerance
            if not passed:
                reasons.append("projector_covariance_failed")
            rows.append(
                {
                    "operation_id": operation_id,
                    "source_valley": valley,
                    "mapped_valley": mapped,
                    "residual": residual,
                    "passed": passed,
                }
            )
    return rows


def _valley_block_rows(
    *,
    operation_ids: Sequence[int],
    source_valleys: Sequence[str],
    representations: Mapping[int, np.ndarray],
    valley_bases: Mapping[str, np.ndarray],
    valley_mappings: Mapping[int, Mapping[str, str]],
    tolerance: float,
    reasons: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for operation_id in operation_ids:
        matrix = representations.get(operation_id)
        mapping = valley_mappings.get(operation_id)
        if matrix is None or not isinstance(mapping, Mapping):
            continue
        for valley in source_valleys:
            mapped = mapping.get(valley)
            try:
                source_basis = np.asarray(
                    valley_bases[valley], dtype=np.complex128
                )
                target_basis = np.asarray(
                    valley_bases[mapped], dtype=np.complex128
                )
            except (KeyError, TypeError, ValueError):
                reasons.append("valley_block_basis_missing")
                continue
            if (
                source_basis.ndim != 2
                or target_basis.ndim != 2
                or source_basis.shape[0] != matrix.shape[0]
                or target_basis.shape[0] != matrix.shape[0]
            ):
                reasons.append("valley_block_basis_malformed")
                continue
            source_ortho = float(
                np.linalg.norm(
                    source_basis.conj().T @ source_basis
                    - np.eye(source_basis.shape[1]),
                    ord="fro",
                )
            )
            target_ortho = float(
                np.linalg.norm(
                    target_basis.conj().T @ target_basis
                    - np.eye(target_basis.shape[1]),
                    ord="fro",
                )
            )
            block = target_basis.conj().T @ matrix @ source_basis
            if block.shape[0] != block.shape[1]:
                block_unitarity = float("inf")
            else:
                block_unitarity = float(
                    np.linalg.norm(
                        block.conj().T @ block - np.eye(block.shape[1]),
                        ord="fro",
                    )
                )
            transformed = matrix @ source_basis
            denominator = max(float(np.linalg.norm(transformed, ord="fro")), 1.0e-14)
            leakage = float(
                np.linalg.norm(
                    transformed - target_basis @ block, ord="fro"
                )
                / denominator
            )
            passed = max(
                source_ortho, target_ortho, block_unitarity, leakage
            ) <= tolerance
            if not passed:
                reasons.append("valley_block_quality_failed")
            rows.append(
                {
                    "operation_id": operation_id,
                    "source_valley": valley,
                    "mapped_valley": mapped,
                    "source_basis_orthonormality_residual": source_ortho,
                    "target_basis_orthonormality_residual": target_ortho,
                    "block_unitarity_residual": block_unitarity,
                    "block_leakage_residual": leakage,
                    "passed": bool(passed),
                }
            )
    return rows


def _antiunitary_record(
    *,
    required: bool,
    evidence: Mapping[str, object] | None,
    valley_orbit: Sequence[str],
    tolerance: float,
    reasons: list[str],
) -> dict[str, object]:
    if not required:
        return {"required": False, "evaluated": False}
    if not isinstance(evidence, Mapping):
        reasons.append("antiunitary_evidence_missing")
        return {"required": True, "evaluated": False}
    if any(key in evidence for key in ("status", "passed", "validated")):
        reasons.append("positive_validation_status_not_accepted")
    source = evidence.get("source_valley")
    target = evidence.get("target_valley")
    if source not in valley_orbit or target not in valley_orbit or source == target:
        reasons.append("antiunitary_valley_mapping_invalid")
    try:
        forward = np.asarray(
            evidence["forward_sewing_matrix"], dtype=np.complex128
        )
        reverse = np.asarray(
            evidence["reverse_sewing_matrix"], dtype=np.complex128
        )
    except (KeyError, TypeError, ValueError):
        reasons.append("antiunitary_sewing_matrix_malformed")
        return {"required": True, "evaluated": False}
    if (
        forward.ndim != 2
        or reverse.ndim != 2
        or forward.shape[0] != forward.shape[1]
        or reverse.shape != forward.shape
        or forward.shape[0] < 1
    ):
        reasons.append("antiunitary_sewing_matrix_malformed")
        return {"required": True, "evaluated": False}
    expected_sign = evidence.get("expected_square_sign")
    if expected_sign not in (-1, 1):
        reasons.append("antiunitary_square_sign_malformed")
        expected_sign = -1
    forward_unitarity = float(
        np.linalg.norm(
            forward.conj().T @ forward - np.eye(forward.shape[0]), ord="fro"
        )
    )
    reverse_unitarity = float(
        np.linalg.norm(
            reverse.conj().T @ reverse - np.eye(reverse.shape[0]), ord="fro"
        )
    )
    square_residual = float(
        np.linalg.norm(
            reverse @ forward.conj()
            - expected_sign * np.eye(forward.shape[0]),
            ord="fro",
        )
    )
    passed = max(
        forward_unitarity, reverse_unitarity, square_residual
    ) <= tolerance
    if not passed:
        reasons.append("antiunitary_evidence_failed")
    return {
        "required": True,
        "evaluated": True,
        "source_valley": source,
        "target_valley": target,
        "expected_square_sign": expected_sign,
        "forward_unitarity_residual": forward_unitarity,
        "reverse_unitarity_residual": reverse_unitarity,
        "square_residual": square_residual,
        "passed": bool(passed),
    }


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
