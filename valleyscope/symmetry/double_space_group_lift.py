from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from valleyscope.io.spinor_source_basis import (
    validate_spinor_source_basis_record,
)
from valleyscope.io.wavefunction_convention import (
    canonical_identity,
    valid_sha256_identity,
)
from valleyscope.geometry.lattice import (
    cart_rotation_from_fractional,
    cart_translation_from_fractional,
)


DOUBLE_SPACE_GROUP_LIFT_SCHEMA_VERSION = "1.0.0"
_TOLERANCE = 1.0e-8
_SOURCE_SPIN_TOLERANCE = 5.0e-5


@dataclass(frozen=True)
class LiftValidation:
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class DoubleSpaceGroupLiftCertificate:
    _content: dict[str, object]
    _reason_codes: tuple[str, ...]

    @property
    def status(self) -> str:
        return "passed" if not self._reason_codes else "blocked"

    def to_record(self) -> dict[str, object]:
        record = deepcopy(self._content)
        record["status"] = self.status
        record["reason_codes"] = list(self._reason_codes)
        record["certificate_identity"] = canonical_identity(self._content)
        return record


def axial_spin_rotation(rotation_cart: np.ndarray) -> np.ndarray:
    rotation = _orthogonal_matrix(rotation_cart)
    determinant = float(np.linalg.det(rotation))
    sign = 1.0 if determinant > 0.0 else -1.0
    axial = sign * rotation
    if not np.allclose(axial.T @ axial, np.eye(3), atol=_TOLERANCE):
        raise ValueError("axial spin rotation is not orthogonal")
    if not np.isclose(np.linalg.det(axial), 1.0, atol=_TOLERANCE):
        raise ValueError("axial spin rotation is not proper")
    return axial


def spin_lift_from_orthogonal(rotation_cart: np.ndarray) -> np.ndarray:
    axial = axial_spin_rotation(rotation_cart)
    quaternion = _canonical_quaternion(axial)
    w, x, y, z = quaternion
    return np.array(
        [
            [w - 1.0j * z, -y - 1.0j * x],
            [y - 1.0j * x, w + 1.0j * z],
        ],
        dtype=np.complex128,
    )


def seitz_product(
    left_rotation: np.ndarray,
    left_translation: np.ndarray,
    right_rotation: np.ndarray,
    right_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_r = np.asarray(left_rotation)
    right_r = np.asarray(right_rotation)
    left_t = np.asarray(left_translation, dtype=float)
    right_t = np.asarray(right_translation, dtype=float)
    return left_r @ right_r, left_t + left_r @ right_t


def build_double_space_group_lift_certificate(
    source_basis_record: Mapping[str, object],
    operations: Sequence[Mapping[str, object]],
    *,
    source_table_identity: Mapping[str, object],
    standard_setting_identity: Mapping[str, object],
    direct_lattice_cart: np.ndarray,
) -> DoubleSpaceGroupLiftCertificate:
    reasons: list[str] = []
    source_validation = validate_spinor_source_basis_record(source_basis_record)
    if source_validation.status != "passed":
        reasons.append("source_basis_certificate_not_passed")

    try:
        direct_lattice = _direct_lattice(direct_lattice_cart)
    except ValueError:
        direct_lattice = np.eye(3)
        reasons.append("direct_lattice_malformed")

    normalized_operations: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    for operation in operations:
        normalized, operation_reasons = _normalize_operation(
            operation,
            direct_lattice,
        )
        reasons.extend(operation_reasons)
        if normalized is None:
            continue
        operation_id = normalized["operation_id"]
        assert isinstance(operation_id, int)
        if operation_id in seen_ids:
            reasons.append("operation_id_duplicate")
            continue
        seen_ids.add(operation_id)
        normalized_operations.append(normalized)
    normalized_operations.sort(key=lambda item: item["operation_id"])

    if not normalized_operations:
        reasons.append("operation_inventory_empty")

    operation_records: list[dict[str, object]] = []
    lifts: dict[int, np.ndarray] = {}
    for operation in normalized_operations:
        operation_id = int(operation["operation_id"])
        try:
            axial = axial_spin_rotation(
                np.asarray(operation["rotation_cart"], dtype=float)
            )
            lift = spin_lift_from_orthogonal(
                np.asarray(operation["rotation_cart"], dtype=float)
            )
        except ValueError:
            reasons.append("operation_spin_lift_invalid")
            continue
        lifts[operation_id] = lift
        operation_records.append(
            {
                **operation,
                "axial_spin_rotation": _real_matrix_record(axial),
                "su2_lift": _complex_matrix_record(lift),
                "su2_unitarity_residual": float(
                    np.linalg.norm(lift.conj().T @ lift - np.eye(2))
                ),
                "su2_determinant_residual": float(
                    abs(np.linalg.det(lift) - 1.0)
                ),
            }
        )

    pairwise_products: dict[str, object] = {}
    for left in normalized_operations:
        for right in normalized_operations:
            pair, pair_reasons = _pairwise_product(
                left,
                right,
                normalized_operations,
                lifts,
            )
            reasons.extend(pair_reasons)
            key = f"{left['operation_id']},{right['operation_id']}"
            if pair is not None:
                pairwise_products[key] = pair

    if not _identity_operation_present(normalized_operations):
        reasons.append("identity_operation_missing")

    operation_inventory_content = {"operations": operation_records}
    operation_ids = [
        int(operation["operation_id"]) for operation in normalized_operations
    ]
    (
        source_table,
        standard_setting,
        source_operation_signs,
        source_setting_reasons,
    ) = _derive_source_and_setting_identities(
        source_table_evidence=source_table_identity,
        standard_setting_evidence=standard_setting_identity,
        operations=normalized_operations,
        lifts=lifts,
    )
    reasons.extend(source_setting_reasons)

    content: dict[str, object] = {
        "schema_version": DOUBLE_SPACE_GROUP_LIFT_SCHEMA_VERSION,
        "source_basis_certificate_identity": source_basis_record.get(
            "certificate_identity"
        ),
        "operation_ids": operation_ids,
        "operation_inventory": operation_records,
        "operation_inventory_identity": canonical_identity(
            operation_inventory_content
        ),
        "direct_lattice_cart": _real_matrix_record(direct_lattice),
        "direct_lattice_identity": canonical_identity(
            {"direct_lattice_cart": _real_matrix_record(direct_lattice)}
        ),
        "source_table_identity": source_table,
        "source_table_identity_hash": canonical_identity(source_table),
        "standard_setting_identity": standard_setting,
        "standard_setting_identity_hash": canonical_identity(standard_setting),
        "spin_basis": {
            "basis_identity": "vasp_saxis_cartesian_spinor_v1",
            "common_basis_transform": _complex_matrix_record(
                np.eye(2, dtype=np.complex128)
            ),
            "saxis_cart": [0.0, 0.0, 1.0],
            "source_operation_signs": source_operation_signs,
        },
        "central_element": {
            "label": "-E",
            "matrix": _complex_matrix_record(
                -np.eye(2, dtype=np.complex128)
            ),
        },
        "seitz_product_convention": "active_left_after_right",
        "reciprocal_action_convention": (
            "q_prime=R_cart@q; phase=exp(-i*q_prime_dot_translation)"
        ),
        "pairwise_products": pairwise_products,
        "max_spin_residual": _max_pair_residual(
            pairwise_products, "spin_residual"
        ),
        "max_bloch_phase_composition_residual": _max_pair_residual(
            pairwise_products, "bloch_phase_composition_residual"
        ),
    }
    return DoubleSpaceGroupLiftCertificate(
        _content=content,
        _reason_codes=tuple(dict.fromkeys(reasons)),
    )


def validate_double_space_group_lift_record(
    record: Mapping[str, object],
    *,
    source_basis_record: Mapping[str, object],
    source_table_identity: Mapping[str, object],
    standard_setting_identity: Mapping[str, object],
    direct_lattice_cart: np.ndarray,
    expected_operations: Sequence[Mapping[str, object]],
    required_operation_ids: Sequence[int] | None = None,
) -> LiftValidation:
    if not isinstance(record, Mapping):
        return LiftValidation("blocked", ("record_malformed",))
    reasons: list[str] = []

    source_validation = validate_spinor_source_basis_record(source_basis_record)
    if source_validation.status != "passed":
        reasons.append("source_basis_certificate_not_passed")
    if record.get("source_basis_certificate_identity") != source_basis_record.get(
        "certificate_identity"
    ):
        reasons.append("source_basis_identity_mismatch")
    try:
        expected_direct_lattice = _direct_lattice(direct_lattice_cart)
    except ValueError:
        expected_direct_lattice = None
        reasons.append("direct_lattice_malformed")
    if (
        expected_direct_lattice is None
        or record.get("direct_lattice_cart")
        != _real_matrix_record(expected_direct_lattice)
    ):
        reasons.append("direct_lattice_identity_mismatch")

    operation_inventory = record.get("operation_inventory")
    operation_ids = record.get("operation_ids")
    if not isinstance(operation_inventory, list):
        reasons.append("operation_inventory_malformed")
        operation_inventory = []
    derived_ids: list[int] = []
    for operation in operation_inventory:
        if not isinstance(operation, Mapping):
            reasons.append("operation_inventory_malformed")
            continue
        operation_id = operation.get("operation_id")
        if not _exact_int(operation_id):
            reasons.append("operation_id_malformed")
            continue
        derived_ids.append(operation_id)
    if operation_ids != derived_ids:
        reasons.append("operation_inventory_identity_mismatch")
    try:
        expected_inventory_identity = canonical_identity(
            {"operations": operation_inventory}
        )
    except (TypeError, ValueError):
        expected_inventory_identity = None
    if (
        expected_inventory_identity is None
        or record.get("operation_inventory_identity")
        != expected_inventory_identity
    ):
        reasons.append("operation_inventory_identity_mismatch")

    if required_operation_ids is not None:
        required: list[int] = []
        for operation_id in required_operation_ids:
            if not _exact_int(operation_id):
                reasons.append("required_operation_id_malformed")
            else:
                required.append(operation_id)
        if not set(required).issubset(set(derived_ids)):
            reasons.append("required_operation_evidence_missing")

    serialized_reasons = record.get("reason_codes")
    if not isinstance(serialized_reasons, list) or not all(
        isinstance(value, str) for value in serialized_reasons
    ):
        reasons.append("reason_codes_malformed")
        serialized_reasons = []
    recomputed_record: dict[str, object] | None = None
    if expected_direct_lattice is not None:
        try:
            recomputed_record = build_double_space_group_lift_certificate(
                source_basis_record,
                expected_operations,
                source_table_identity=source_table_identity,
                standard_setting_identity=standard_setting_identity,
                direct_lattice_cart=expected_direct_lattice,
            ).to_record()
        except (TypeError, ValueError):
            recomputed_record = None
    if recomputed_record is None:
        reasons.append("certificate_recomputation_failed")
        expected_status = "blocked"
    else:
        expected_status = str(recomputed_record["status"])
        if record.get("status") != expected_status:
            reasons.append("derived_status_mismatch")
        if record.get("reason_codes") != recomputed_record.get("reason_codes"):
            reasons.append("derived_reason_codes_mismatch")
        if dict(record) != recomputed_record:
            reasons.append("recomputed_certificate_mismatch")

    content = {
        key: deepcopy(value)
        for key, value in record.items()
        if key not in {"status", "reason_codes", "certificate_identity"}
    }
    try:
        expected_identity = canonical_identity(content)
    except (TypeError, ValueError):
        expected_identity = None
    if (
        expected_identity is None
        or record.get("certificate_identity") != expected_identity
        or not valid_sha256_identity(record.get("certificate_identity"))
    ):
        reasons.append("certificate_identity_mismatch")

    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return LiftValidation("blocked", tuple(reasons))
    return LiftValidation(expected_status, tuple(serialized_reasons))


def _pairwise_product(
    left: Mapping[str, object],
    right: Mapping[str, object],
    operations: Sequence[Mapping[str, object]],
    lifts: Mapping[int, np.ndarray],
) -> tuple[dict[str, object] | None, list[str]]:
    reasons: list[str] = []
    product_rotation, product_translation = seitz_product(
        np.asarray(left["rotation_frac"], dtype=int),
        np.asarray(left["translation_frac"], dtype=float),
        np.asarray(right["rotation_frac"], dtype=int),
        np.asarray(right["translation_frac"], dtype=float),
    )
    matches: list[tuple[Mapping[str, object], np.ndarray]] = []
    for candidate in operations:
        if not np.array_equal(
            product_rotation,
            np.asarray(candidate["rotation_frac"], dtype=int),
        ):
            continue
        difference = product_translation - np.asarray(
            candidate["translation_frac"], dtype=float
        )
        if np.allclose(difference, np.rint(difference), atol=_TOLERANCE):
            matches.append((candidate, np.rint(difference).astype(int)))
    if len(matches) != 1:
        reasons.append(
            "operation_inventory_not_closed"
            if len(matches) == 0
            else "operation_product_ambiguous"
        )
        return None, reasons

    candidate, lattice_translation = matches[0]
    left_id = int(left["operation_id"])
    right_id = int(right["operation_id"])
    product_id = int(candidate["operation_id"])
    if not all(operation_id in lifts for operation_id in (left_id, right_id, product_id)):
        reasons.append("operation_spin_lift_invalid")
        return None, reasons
    spin_product = lifts[left_id] @ lifts[right_id]
    plus_residual = float(np.linalg.norm(spin_product - lifts[product_id]))
    minus_residual = float(np.linalg.norm(spin_product + lifts[product_id]))
    cocycle_sign = 1 if plus_residual <= minus_residual else -1
    spin_residual = min(plus_residual, minus_residual)
    if spin_residual > _TOLERANCE:
        reasons.append("spin_cocycle_inconsistent")

    permutation_passed, phase_residual = _reciprocal_composition_residual(
        np.asarray(left["rotation_frac"], dtype=int),
        np.asarray(left["translation_frac"], dtype=float),
        np.asarray(right["rotation_frac"], dtype=int),
        np.asarray(right["translation_frac"], dtype=float),
        product_rotation,
        product_translation,
    )
    if not permutation_passed:
        reasons.append("reciprocal_permutation_composition_failed")
    if phase_residual > _TOLERANCE:
        reasons.append("bloch_phase_composition_failed")
    return {
        "left_operation_id": left_id,
        "right_operation_id": right_id,
        "product_operation_id": product_id,
        "lattice_translation_frac": lattice_translation.tolist(),
        "cocycle_sign": cocycle_sign,
        "spin_residual": spin_residual,
        "reciprocal_permutation_composition_passed": permutation_passed,
        "bloch_phase_composition_residual": phase_residual,
        "factor_system_phase_convention": (
            "exp(-2pii*((R_left R_right)^-T k)_dot_L)"
        ),
        "factor_system_phase_samples": _factor_system_phase_samples(
            product_rotation,
            lattice_translation,
        ),
    }, reasons


def _reciprocal_composition_residual(
    left_rotation: np.ndarray,
    left_translation: np.ndarray,
    right_rotation: np.ndarray,
    right_translation: np.ndarray,
    product_rotation: np.ndarray,
    product_translation: np.ndarray,
) -> tuple[bool, float]:
    left_reciprocal = np.linalg.inv(left_rotation).T
    right_reciprocal = np.linalg.inv(right_rotation).T
    product_reciprocal = np.linalg.inv(product_rotation).T
    composed_reciprocal = left_reciprocal @ right_reciprocal
    permutation_passed = bool(
        np.allclose(
            composed_reciprocal,
            product_reciprocal,
            atol=_TOLERANCE,
        )
    )
    probes = (
        np.zeros(3),
        np.array([0.125, 0.25, -0.375]),
        np.array([1.0, -2.0, 3.0]),
    )
    residual = 0.0
    for source_q in probes:
        right_q = right_reciprocal @ source_q
        product_q = left_reciprocal @ right_q
        right_phase = np.exp(
            -2.0j * np.pi * float(right_q @ right_translation)
        )
        left_phase = np.exp(
            -2.0j * np.pi * float(product_q @ left_translation)
        )
        product_phase = np.exp(
            -2.0j * np.pi * float(product_q @ product_translation)
        )
        residual = max(
            residual,
            float(abs(left_phase * right_phase - product_phase)),
        )
    return permutation_passed, residual


def _factor_system_phase_samples(
    product_rotation: np.ndarray,
    lattice_translation: np.ndarray,
) -> list[dict[str, object]]:
    product_reciprocal = np.linalg.inv(product_rotation).T
    probes = (
        np.zeros(3),
        np.array([0.125, 0.25, -0.375]),
        np.array([0.0, 0.0, 0.25]),
    )
    samples: list[dict[str, object]] = []
    for k_frac in probes:
        transformed_k = product_reciprocal @ k_frac
        phase = np.exp(
            -2.0j
            * np.pi
            * float(transformed_k @ np.asarray(lattice_translation, dtype=float))
        )
        phase_parts = np.array([phase.real, phase.imag], dtype=float)
        phase_parts[np.abs(phase_parts) < _TOLERANCE] = 0.0
        samples.append(
            {
                "k_frac": [float(value) for value in k_frac],
                "transformed_k_frac": [
                    float(value) for value in transformed_k
                ],
                "phase": [float(value) for value in phase_parts],
            }
        )
    return samples


def _normalize_operation(
    operation: Mapping[str, object],
    direct_lattice_cart: np.ndarray,
) -> tuple[dict[str, object] | None, list[str]]:
    reasons: list[str] = []
    if not isinstance(operation, Mapping):
        return None, ["operation_record_malformed"]
    operation_id = operation.get("operation_id")
    if not _exact_int(operation_id):
        return None, ["operation_id_malformed"]
    try:
        rotation_frac_raw = np.asarray(operation.get("rotation_frac"))
        rotation_cart = np.asarray(operation.get("rotation_cart"), dtype=float)
        translation_frac = np.asarray(
            operation.get("translation_frac"), dtype=float
        )
        translation_cart = np.asarray(
            operation.get("translation_cart"), dtype=float
        )
    except (TypeError, ValueError):
        return None, ["operation_record_malformed"]
    if (
        rotation_frac_raw.shape != (3, 3)
        or not np.issubdtype(rotation_frac_raw.dtype, np.integer)
    ):
        reasons.append("rotation_frac_malformed")
        return None, reasons
    rotation_frac = rotation_frac_raw.astype(int)
    if rotation_cart.shape != (3, 3):
        return None, ["rotation_cart_malformed"]
    if translation_frac.shape != (3,) or translation_cart.shape != (3,):
        return None, ["translation_malformed"]
    try:
        _orthogonal_matrix(rotation_cart)
    except ValueError:
        return None, ["rotation_cart_not_orthogonal"]
    expected_rotation_cart = cart_rotation_from_fractional(
        rotation_frac,
        direct_lattice_cart,
    )
    expected_translation_cart = cart_translation_from_fractional(
        translation_frac,
        direct_lattice_cart,
    )
    if not np.allclose(
        rotation_cart,
        expected_rotation_cart,
        atol=_TOLERANCE,
    ) or not np.allclose(
        translation_cart,
        expected_translation_cart,
        atol=_TOLERANCE,
    ):
        reasons.append("cartesian_affine_operation_mismatch")
    return {
        "operation_id": operation_id,
        "rotation_frac": rotation_frac.tolist(),
        "translation_frac": _normalized_float_vector(translation_frac),
        "rotation_cart": _real_matrix_record(rotation_cart),
        "translation_cart": _normalized_float_vector(translation_cart),
    }, reasons


def _direct_lattice(value: np.ndarray) -> np.ndarray:
    direct = np.asarray(value, dtype=float)
    if direct.shape != (3, 3) or not np.all(np.isfinite(direct)):
        raise ValueError("direct_lattice_cart must be finite with shape [3,3]")
    if abs(float(np.linalg.det(direct))) <= _TOLERANCE:
        raise ValueError("direct_lattice_cart must be invertible")
    return direct


def _derive_source_and_setting_identities(
    *,
    source_table_evidence: Mapping[str, object],
    standard_setting_evidence: Mapping[str, object],
    operations: Sequence[Mapping[str, object]],
    lifts: Mapping[int, np.ndarray],
) -> tuple[dict[str, object], dict[str, object], dict[str, int], list[str]]:
    source_reasons: list[str] = []
    setting_reasons: list[str] = []
    operation_ids = [int(operation["operation_id"]) for operation in operations]
    source_raw = _json_value(source_table_evidence)
    setting_raw = _json_value(standard_setting_evidence)
    if not isinstance(source_raw, dict):
        source_raw = {}
        source_reasons.append("source_table_evidence_malformed")
    if not isinstance(setting_raw, dict):
        setting_raw = {}
        setting_reasons.append("standard_setting_evidence_malformed")
    positive_keys = {
        "status",
        "validation_status",
        "operation_mapping_status",
        "affine_validation_status",
        "common_spin_basis_status",
    }
    if positive_keys.intersection(source_raw):
        source_reasons.append("positive_validation_status_not_accepted")
    if positive_keys.intersection(setting_raw):
        setting_reasons.append("positive_validation_status_not_accepted")

    if source_raw.get("schema_version") != "1.0.0":
        source_reasons.append("source_table_schema_mismatch")
    if source_raw.get("provider") != "irreptables":
        source_reasons.append("source_table_provider_mismatch")
    if source_raw.get("data_source") != "irreptables.StandardIrrepTable":
        source_reasons.append("source_table_data_source_mismatch")
    if source_raw.get("spinor") is not True:
        source_reasons.append("source_table_spinor_mismatch")
    space_group_number = source_raw.get("space_group_number")
    if not _exact_int(space_group_number) or space_group_number <= 0:
        source_reasons.append("source_table_space_group_malformed")

    source_operations_raw = source_raw.get("operations")
    source_operations: dict[int, dict[str, object]] = {}
    if not isinstance(source_operations_raw, list):
        source_reasons.append("source_operation_inventory_malformed")
        source_operations_raw = []
    for raw_operation in source_operations_raw:
        normalized = _normalize_source_operation(raw_operation)
        if normalized is None:
            source_reasons.append("source_operation_inventory_malformed")
            continue
        table_index = int(normalized["table_index"])
        if table_index in source_operations:
            source_reasons.append("source_operation_index_duplicate")
            continue
        source_operations[table_index] = normalized

    if setting_raw.get("schema_version") != "1.0.0":
        setting_reasons.append("standard_setting_schema_mismatch")
    transform = _matrix3(
        setting_raw.get("parent_to_standard_direct_transform")
    )
    if transform is None or abs(float(np.linalg.det(transform))) <= _TOLERANCE:
        setting_reasons.append("standard_setting_transform_malformed")
        transform = np.eye(3)
    origin = _vector3(setting_raw.get("origin_shift_fractional"))
    if origin is None:
        setting_reasons.append("standard_setting_origin_malformed")
        origin = np.zeros(3)
    operation_map_raw = setting_raw.get("parent_to_standard_operation_map")
    operation_map: dict[int, int] = {}
    if not isinstance(operation_map_raw, Mapping):
        setting_reasons.append("standard_setting_operation_map_malformed")
        operation_map_raw = {}
    for parent_id_raw, table_index in operation_map_raw.items():
        try:
            parent_id = int(parent_id_raw)
        except (TypeError, ValueError):
            setting_reasons.append("standard_setting_operation_map_malformed")
            continue
        if str(parent_id) != str(parent_id_raw) or not _exact_int(table_index):
            setting_reasons.append("standard_setting_operation_map_malformed")
            continue
        operation_map[parent_id] = table_index
    if set(operation_map) != set(operation_ids):
        setting_reasons.append("standard_setting_operation_coverage_incomplete")
    if len(set(operation_map.values())) != len(operation_map):
        setting_reasons.append("standard_setting_operation_map_not_bijective")
    if set(operation_map.values()) != set(source_operations):
        setting_reasons.append("source_operation_coverage_incomplete")

    transform_inverse = np.linalg.inv(transform)
    common_spin_basis = _derive_common_spin_basis_transform(
        operations=operations,
        lifts=lifts,
        operation_map=operation_map,
        source_operations=source_operations,
    )
    if common_spin_basis is None:
        source_reasons.append("source_spin_common_basis_failed")
        common_spin_basis = np.eye(2, dtype=np.complex128)
    source_operation_signs: dict[str, int] = {}
    spatial_mapping_rows: list[dict[str, object]] = []
    spin_mapping_rows: list[dict[str, object]] = []
    for parent in operations:
        parent_id = int(parent["operation_id"])
        table_index = operation_map.get(parent_id)
        source_operation = source_operations.get(table_index) \
            if table_index is not None else None
        if source_operation is None:
            continue
        parent_rotation = np.asarray(parent["rotation_frac"], dtype=float)
        parent_translation = np.asarray(
            parent["translation_frac"], dtype=float
        )
        standard_rotation_float = (
            transform @ parent_rotation @ transform_inverse
        )
        standard_rotation = np.rint(standard_rotation_float).astype(int)
        if not np.allclose(
            standard_rotation_float,
            standard_rotation,
            atol=_TOLERANCE,
        ):
            setting_reasons.append("standard_setting_rotation_not_integral")
            continue
        standard_translation = (
            transform @ parent_translation
            + origin
            - standard_rotation @ origin
        )
        source_rotation = np.asarray(
            source_operation["rotation_frac"], dtype=int
        )
        source_translation = np.asarray(
            source_operation["translation_frac"], dtype=float
        )
        translation_difference = standard_translation - source_translation
        spatial_passed = bool(
            np.array_equal(standard_rotation, source_rotation)
            and np.allclose(
                translation_difference,
                np.rint(translation_difference),
                atol=_TOLERANCE,
            )
        )
        if not spatial_passed:
            setting_reasons.append("standard_setting_affine_mapping_failed")
        spatial_mapping_rows.append(
            {
                "parent_operation_id": parent_id,
                "source_table_index": table_index,
                "passed": spatial_passed,
                "lattice_translation_standard_frac": (
                    np.rint(translation_difference).astype(int).tolist()
                    if spatial_passed else None
                ),
            }
        )

        source_spin = _complex_matrix_from_record(
            source_operation["spin_rotation"]
        )
        parent_spin = lifts.get(parent_id)
        if source_spin is None or parent_spin is None:
            source_reasons.append("source_spin_rotation_malformed")
            continue
        transformed_parent_spin = (
            common_spin_basis
            @ parent_spin
            @ common_spin_basis.conj().T
        )
        plus_residual = float(
            np.linalg.norm(source_spin - transformed_parent_spin)
        )
        minus_residual = float(
            np.linalg.norm(source_spin + transformed_parent_spin)
        )
        sign = 1 if plus_residual <= minus_residual else -1
        residual = min(plus_residual, minus_residual)
        if residual > _SOURCE_SPIN_TOLERANCE:
            source_reasons.append("source_spin_common_basis_failed")
        source_operation_signs[str(parent_id)] = sign
        spin_mapping_rows.append(
            {
                "parent_operation_id": parent_id,
                "source_table_index": table_index,
                "central_sign": sign,
                "residual": residual,
            }
        )

    source_reasons = list(dict.fromkeys(source_reasons))
    setting_reasons = list(dict.fromkeys(setting_reasons))
    source_content: dict[str, object] = {
        "schema_version": "1.0.0",
        "provider": source_raw.get("provider"),
        "data_source": source_raw.get("data_source"),
        "space_group_number": space_group_number,
        "spinor": source_raw.get("spinor"),
        "source_operation_inventory": [
            source_operations[index] for index in sorted(source_operations)
        ],
        "source_operation_inventory_identity": canonical_identity(
            {
                "operations": [
                    source_operations[index]
                    for index in sorted(source_operations)
                ]
            }
        ),
        "common_spin_basis_transform": _complex_matrix_record(
            common_spin_basis
        ),
        "spin_mapping_rows": spin_mapping_rows,
        "status": "passed" if not source_reasons else "blocked",
        "reason_codes": source_reasons,
    }
    source_content["identity"] = canonical_identity(source_content)

    setting_content: dict[str, object] = {
        "schema_version": "1.0.0",
        "parent_to_standard_direct_transform": _real_matrix_record(transform),
        "origin_shift_fractional": _normalized_float_vector(origin),
        "parent_to_standard_operation_map": {
            str(parent_id): operation_map[parent_id]
            for parent_id in sorted(operation_map)
        },
        "required_parent_operation_ids": operation_ids,
        "spatial_mapping_rows": spatial_mapping_rows,
        "status": "passed" if not setting_reasons else "blocked",
        "reason_codes": setting_reasons,
    }
    setting_content["identity"] = canonical_identity(setting_content)

    reasons: list[str] = []
    if source_reasons:
        reasons.append("source_table_convention_not_validated")
    if setting_reasons:
        reasons.append("standard_setting_not_validated")
    return (
        source_content,
        setting_content,
        source_operation_signs,
        reasons,
    )


def _derive_common_spin_basis_transform(
    *,
    operations: Sequence[Mapping[str, object]],
    lifts: Mapping[int, np.ndarray],
    operation_map: Mapping[int, int],
    source_operations: Mapping[int, Mapping[str, object]],
) -> np.ndarray | None:
    """Solve one SU(2) basis transform for the complete mapped inventory."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    axes: list[tuple[np.ndarray, np.ndarray]] = []
    for operation in operations:
        operation_id = int(operation["operation_id"])
        table_index = operation_map.get(operation_id)
        source_operation = (
            source_operations.get(table_index)
            if table_index is not None
            else None
        )
        parent_spin = lifts.get(operation_id)
        source_spin = (
            _complex_matrix_from_record(source_operation["spin_rotation"])
            if isinstance(source_operation, Mapping)
            else None
        )
        if parent_spin is None or source_spin is None:
            continue
        parent_unitary = _nearest_unitary(parent_spin)
        source_unitary = _nearest_unitary(source_spin)
        if parent_unitary is None or source_unitary is None:
            continue
        pairs.append((parent_unitary, source_unitary))
        parent_axis = _proper_rotation_axis(_su2_adjoint(parent_unitary))
        source_axis = _proper_rotation_axis(_su2_adjoint(source_unitary))
        if parent_axis is not None and source_axis is not None:
            axes.append((parent_axis, source_axis))
    if not pairs:
        return None

    rotation_candidates: list[np.ndarray] = [np.eye(3)]
    for parent_axis, source_axis in axes:
        for sign in (-1.0, 1.0):
            candidate = _align_unit_vectors(
                parent_axis, sign * source_axis
            )
            if candidate is not None:
                rotation_candidates.append(candidate)

    independent_axes: list[tuple[np.ndarray, np.ndarray]] = []
    for pair in axes:
        rank = np.linalg.matrix_rank(
            np.stack(
                [item[0] for item in independent_axes] + [pair[0]],
                axis=1,
            ),
            tol=1.0e-7,
        )
        if rank > len(independent_axes):
            independent_axes.append(pair)
        if len(independent_axes) == 3:
            break
    if len(independent_axes) >= 2:
        parent_vectors = np.stack(
            [item[0] for item in independent_axes], axis=1
        )
        source_vectors = np.stack(
            [item[1] for item in independent_axes], axis=1
        )
        for signs in product((-1.0, 1.0), repeat=len(independent_axes)):
            target_vectors = source_vectors * np.asarray(signs)[None, :]
            correlation = target_vectors @ parent_vectors.T
            left, _, right_t = np.linalg.svd(correlation)
            orientation = np.eye(3)
            orientation[-1, -1] = np.linalg.det(left @ right_t)
            rotation_candidates.append(left @ orientation @ right_t)

    best_basis: np.ndarray | None = None
    best_residual = float("inf")
    for rotation in rotation_candidates:
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-7)
        ):
            continue
        basis = spin_lift_from_orthogonal(rotation)
        residual = max(
            min(
                float(np.linalg.norm(
                    source - basis @ parent @ basis.conj().T
                )),
                float(np.linalg.norm(
                    source + basis @ parent @ basis.conj().T
                )),
            )
            for parent, source in pairs
        )
        if residual < best_residual:
            best_residual = residual
            best_basis = basis
    if best_basis is None or best_residual > _SOURCE_SPIN_TOLERANCE:
        return None
    return best_basis


def _nearest_unitary(matrix: np.ndarray) -> np.ndarray | None:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (2, 2) or not np.all(np.isfinite(value)):
        return None
    left, singular_values, right_h = np.linalg.svd(value)
    if np.min(singular_values) <= 0.0:
        return None
    unitary = left @ right_h
    determinant = np.linalg.det(unitary)
    if abs(determinant) <= _TOLERANCE:
        return None
    return unitary / np.sqrt(determinant)


def _su2_adjoint(matrix: np.ndarray) -> np.ndarray:
    sigma = (
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    )
    unitary = np.asarray(matrix, dtype=np.complex128)
    return np.asarray(
        [
            [
                0.5
                * np.trace(
                    sigma[row]
                    @ unitary
                    @ sigma[column]
                    @ unitary.conj().T
                ).real
                for column in range(3)
            ]
            for row in range(3)
        ],
        dtype=float,
    )


def _proper_rotation_axis(rotation: np.ndarray) -> np.ndarray | None:
    matrix = np.asarray(rotation, dtype=float)
    if np.linalg.norm(matrix - np.eye(3)) <= 1.0e-7:
        return None
    _, _, right_h = np.linalg.svd(matrix - np.eye(3))
    axis = right_h[-1]
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-10:
        return None
    return axis / norm


def _align_unit_vectors(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray | None:
    left = np.asarray(source, dtype=float)
    right = np.asarray(target, dtype=float)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1.0e-10 or right_norm <= 1.0e-10:
        return None
    left = left / left_norm
    right = right / right_norm
    cross = np.cross(left, right)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if sine <= 1.0e-10:
        if cosine > 0.0:
            return np.eye(3)
        basis_vectors = np.eye(3)
        perpendicular = min(
            basis_vectors,
            key=lambda vector: abs(float(np.dot(vector, left))),
        )
        axis = np.cross(left, perpendicular)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / sine
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + sine * skew
        + (1.0 - cosine) * (skew @ skew)
    )


def _normalize_source_operation(
    value: object,
) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    table_index = value.get("table_index")
    if not _exact_int(table_index):
        return None
    rotation_raw = np.asarray(value.get("rotation_frac"))
    translation = _vector3(value.get("translation_frac"))
    spin_rotation = _complex_matrix_from_record(value.get("spin_rotation"))
    if (
        rotation_raw.shape != (3, 3)
        or not np.issubdtype(rotation_raw.dtype, np.integer)
        or translation is None
        or spin_rotation is None
    ):
        return None
    normalized_spin_rotation = _nearest_unitary(spin_rotation)
    if (
        normalized_spin_rotation is None
        or np.linalg.norm(
            spin_rotation - normalized_spin_rotation
        ) > _SOURCE_SPIN_TOLERANCE
    ):
        return None
    return {
        "table_index": table_index,
        "rotation_frac": rotation_raw.astype(int).tolist(),
        "translation_frac": np.mod(translation, 1.0).tolist(),
        "spin_rotation": _complex_matrix_record(
            normalized_spin_rotation
        ),
    }


def _complex_matrix_from_record(value: object) -> np.ndarray | None:
    try:
        raw = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if raw.shape != (2, 2, 2) or not np.all(np.isfinite(raw)):
        return None
    return raw[:, :, 0] + 1.0j * raw[:, :, 1]


def _matrix3(value: object) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _vector3(value: object) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _identity_operation_present(
    operations: Sequence[Mapping[str, object]],
) -> bool:
    for operation in operations:
        rotation = np.asarray(operation["rotation_frac"], dtype=int)
        translation = np.asarray(operation["translation_frac"], dtype=float)
        if np.array_equal(rotation, np.eye(3, dtype=int)) and np.allclose(
            translation,
            np.rint(translation),
            atol=_TOLERANCE,
        ):
            return True
    return False


def _orthogonal_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape [3,3]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be finite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=_TOLERANCE):
        raise ValueError("rotation must be orthogonal")
    determinant = float(np.linalg.det(matrix))
    if not np.isclose(abs(determinant), 1.0, atol=_TOLERANCE):
        raise ValueError("rotation determinant must be +/-1")
    return matrix


def _canonical_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    for component in quaternion:
        if abs(component) <= _TOLERANCE:
            continue
        if component < 0.0:
            quaternion = -quaternion
        break
    quaternion[np.abs(quaternion) < _TOLERANCE] = 0.0
    return quaternion


def _real_matrix_record(matrix: np.ndarray) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in np.asarray(matrix, dtype=float)
    ]


def _complex_matrix_record(
    matrix: np.ndarray,
) -> list[list[list[float]]]:
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in np.asarray(matrix, dtype=np.complex128)
    ]


def _normalized_float_vector(vector: np.ndarray) -> list[float]:
    result = np.asarray(vector, dtype=float).copy()
    result[np.abs(result) < _TOLERANCE] = 0.0
    return [float(value) for value in result]


def _max_pair_residual(
    pairwise_products: Mapping[str, object],
    key: str,
) -> float | None:
    values = [
        float(pair[key])
        for pair in pairwise_products.values()
        if isinstance(pair, Mapping) and pair.get(key) is not None
    ]
    return max(values) if values else None


def _exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
