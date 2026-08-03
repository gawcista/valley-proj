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
from valleyscope.analysis.target_frame import (
    build_target_frame,
    validate_target_frame_record,
)
from valleyscope.symmetry.double_space_group_lift import (
    spin_lift_from_orthogonal,
    validate_double_space_group_lift_record,
)
from valleyscope.symmetry.plane_wave_action import (
    RECIPROCAL_GRID_ACTION_CONVENTION,
    apply_plane_wave_action,
    build_plane_wave_representation,
    build_reciprocal_grid_map,
    reciprocal_grid_identity,
    validate_reciprocal_grid_permutation,
)


SCOPED_REPRESENTATION_EVIDENCE_SCHEMA_VERSION = "1.4.0"
SUPPORTED_SCOPE_KINDS = frozenset({"local_irrep"})
DEFAULT_GROUP_LAW_TOLERANCE = 1.0e-2
DEFAULT_PLANE_WAVE_NORM_TOLERANCE = 1.0e-8
DEFAULT_TARGET_SUBSPACE_TOLERANCE = 1.0e-2
DEFAULT_PROJECTOR_COVARIANCE_TOLERANCE = 1.0e-2
DEFAULT_VALLEY_BLOCK_TOLERANCE = 1.0e-2


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
    source_target_coefficients: np.ndarray | None = None,
    wavecar_rtag: int | None = None,
    target_frame_record: Mapping[str, object] | None = None,
) -> ScopedRepresentationEvidence:
    """Derive one immutable scope record from raw numerical inputs.

    No positive validation status is accepted as input.  The source and lift
    records are revalidated against their producer inputs before any numerical
    representation result can pass.
    """
    reasons: list[str] = []
    group_law_tol = DEFAULT_GROUP_LAW_TOLERANCE
    plane_wave_norm_tol = DEFAULT_PLANE_WAVE_NORM_TOLERANCE
    target_tol = DEFAULT_TARGET_SUBSPACE_TOLERANCE
    projector_tol = DEFAULT_PROJECTOR_COVARIANCE_TOLERANCE
    block_tol = DEFAULT_VALLEY_BLOCK_TOLERANCE

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
    frame_source = (
        target_coefficients
        if source_target_coefficients is None
        else source_target_coefficients
    )
    target_frame = build_target_frame(
        frame_source,
        wavecar_rtag=wavecar_rtag,
    )
    reasons.extend(target_frame.reason_codes)
    frame_shape = target_frame.record.get("shape")
    if (
        isinstance(frame_shape, Mapping)
        and frame_shape.get("bands") != target_dimension
    ):
        reasons.append("target_frame_dimension_mismatch")
    if target_frame_record is not None:
        frame_validation = validate_target_frame_record(
            dict(target_frame_record),
            frame_source,
            wavecar_rtag=wavecar_rtag,
        )
        if frame_validation.status == "blocked":
            reasons.extend(frame_validation.reason_codes)
    try:
        supplied_target = np.asarray(
            target_coefficients,
            dtype=np.complex128,
        )
    except (TypeError, ValueError):
        supplied_target = None
    if (
        target_frame.coefficients is not None
        and (
            supplied_target is None
            or not np.array_equal(
                supplied_target,
                target_frame.coefficients,
            )
        )
    ):
        reasons.append("target_frame_coefficients_mismatch")
    closure_rows = _target_subspace_rows(
        operation_ids, matrices, target_tol, reasons
    )
    plane_wave_rows, plane_wave_maps = _plane_wave_rows(
        operation_ids, plane_wave_evidence, plane_wave_norm_tol, reasons
    )
    plane_wave_composition_rows = _plane_wave_composition_rows(
        operation_ids=operation_ids,
        maps=plane_wave_maps,
        lift_record=lift_record,
        reasons=reasons,
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
        tolerance=group_law_tol,
        reasons=reasons,
    )
    covariance_rows = _projector_covariance_rows(
        operation_ids=operation_ids,
        source_valleys=normalized_source_valleys,
        representations=matrices,
        projectors=projectors,
        valley_mappings=valley_mappings,
        tolerance=projector_tol,
        reasons=reasons,
    )
    block_rows = _valley_block_rows(
        operation_ids=operation_ids,
        source_valleys=normalized_source_valleys,
        representations=matrices,
        valley_bases=valley_bases,
        valley_mappings=valley_mappings,
        tolerance=block_tol,
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
        "tolerances": {
            "group_law": group_law_tol,
            "plane_wave_norm": plane_wave_norm_tol,
            "target_subspace": target_tol,
            "projector_covariance": projector_tol,
            "valley_block": block_tol,
        },
        "target_subspace": {
            "dimension": target_dimension,
            "operation_rows": closure_rows,
        },
        "target_frame": deepcopy(target_frame.record),
        "plane_wave_mapping": {
            "operation_rows": plane_wave_rows,
            "composition_rows": plane_wave_composition_rows,
        },
        "projected_representation_group_law": {
            "pair_rows": group_law_rows
        },
        "projector_covariance": {"operation_rows": covariance_rows},
        "valley_block_quality": {"operation_rows": block_rows},
        "grey_group_matching_allowed": False,
    }
    return ScopedRepresentationEvidence(
        _content=content,
        _reason_codes=tuple(dict.fromkeys(reasons)),
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
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        return ScopedEvidenceValidation("blocked", tuple(reasons))
    return ScopedEvidenceValidation(expected_status, expected_reasons)


def build_directed_valley_sewing_evidence(
    **raw_inputs: object,
) -> ScopedRepresentationEvidence:
    """Recompute one source-HSP to target-HSP unitary sewing scope."""
    reasons: list[str] = []
    source_basis = raw_inputs.get("source_basis_record")
    lift_record = raw_inputs.get("lift_record")
    lift_inputs = raw_inputs.get("lift_validation_inputs")
    if not isinstance(source_basis, Mapping):
        reasons.append("source_basis_certificate_malformed")
        source_basis = {}
    if validate_spinor_source_basis_record(source_basis).status != "passed":
        reasons.append("source_basis_certificate_not_passed")
    payload_identity = raw_inputs.get("extracted_wavefunction_payload_identity")
    if (
        not valid_sha256_identity(payload_identity)
        or payload_identity
        != source_basis.get("extracted_wavefunction_payload_identity")
    ):
        reasons.append("extracted_payload_identity_mismatch")

    operation_rows = raw_inputs.get("detected_operations")
    operations = {
        row.get("operation_id"): row
        for row in operation_rows
        if isinstance(row, Mapping)
        and isinstance(row.get("operation_id"), int)
        and not isinstance(row.get("operation_id"), bool)
    } if isinstance(operation_rows, Sequence) else {}
    sewing_id = raw_inputs.get("sewing_operation_id")
    source_ids = _opaque_operation_ids(
        raw_inputs.get("source_little_group_operation_ids", ()), reasons
    )
    target_ids = _opaque_operation_ids(
        raw_inputs.get("target_little_group_operation_ids", ()), reasons
    )
    required_ids = list(dict.fromkeys([*source_ids, *target_ids, sewing_id]))
    if (
        not isinstance(lift_record, Mapping)
        or not isinstance(lift_inputs, Mapping)
        or not isinstance(sewing_id, int)
        or isinstance(sewing_id, bool)
        or sewing_id not in operations
    ):
        reasons.append("directed_sewing_lift_context_malformed")
    else:
        lift_validation = _validate_lift(
            lift_record=lift_record,
            source_basis_record=source_basis,
            lift_validation_inputs=lift_inputs,
            required_operation_ids=required_ids,
        )
        if lift_validation.status != "passed":
            reasons.extend(
                ["double_space_group_lift_not_passed", *lift_validation.reason_codes]
            )

    source_k = np.asarray(
        _kpoint_record(raw_inputs.get("source_k_frac"), reasons), dtype=float
    )
    target_k = np.asarray(
        _kpoint_record(raw_inputs.get("target_k_frac"), reasons), dtype=float
    )
    source_valley = raw_inputs.get("source_valley")
    target_valley = raw_inputs.get("target_valley")
    sewing = operations.get(sewing_id, {})
    mapping = sewing.get("sector_mapping") if isinstance(sewing, Mapping) else None
    reviewed_labels = list(mapping) if isinstance(mapping, Mapping) else []
    reviewed_label_set = set(reviewed_labels)
    mappings = [
        row.get("sector_mapping") for row in operations.values()
        if isinstance(row, Mapping)
    ]
    mappings_are_bijections = bool(reviewed_labels) and all(
        isinstance(item, Mapping)
        and set(item) == reviewed_label_set
        and set(item.values()) == reviewed_label_set
        for item in mappings
    )
    full_orbit = {source_valley} if isinstance(source_valley, str) else set()
    frontier = list(full_orbit)
    while frontier and mappings_are_bijections:
        current = frontier.pop()
        for item in mappings:
            target = item[current]
            if target not in full_orbit:
                full_orbit.add(target)
                frontier.append(target)
    orbit = [label for label in reviewed_labels if label in full_orbit]
    selected_cycle, current = [], source_valley
    while (
        isinstance(mapping, Mapping)
        and isinstance(current, str)
        and current not in selected_cycle
        and len(selected_cycle) <= len(mapping)
    ):
        selected_cycle.append(current)
        current = mapping.get(current)
    if (
        not isinstance(source_valley, str)
        or not isinstance(target_valley, str)
        or source_valley not in orbit
        or target_valley not in orbit
        or not mappings_are_bijections
        or not isinstance(mapping, Mapping)
        or mapping.get(source_valley) != target_valley
        or current != source_valley
    ):
        reasons.append("directed_valley_mapping_failed")
    reciprocal_shift = _directed_hsp_shift(sewing, source_k, target_k)
    if reciprocal_shift is None:
        reasons.append("directed_hsp_mapping_failed")
    for side, ids, kpoint, valley in (
        ("source", source_ids, source_k, source_valley),
        ("target", target_ids, target_k, target_valley),
    ):
        if any(
            not _operation_preserves(
                operations.get(operation_id, {}), kpoint, valley
            )
            for operation_id in ids
        ):
            reasons.append(f"{side}_little_group_scope_failed")

    source_coefficients = np.asarray(
        raw_inputs.get("source_coefficients"), dtype=np.complex128
    )
    wavecar_rtag = raw_inputs.get("wavecar_rtag")
    source_frame = build_target_frame(source_coefficients, wavecar_rtag=wavecar_rtag)
    reasons.extend(source_frame.reason_codes)
    source_q = np.asarray(raw_inputs.get("source_q_cart"), dtype=float)
    try:
        sewing_spin = spin_lift_from_orthogonal(sewing["rotation_cart"])
        rotated_q = source_q @ np.asarray(
            sewing["rotation_cart"], dtype=float
        ).T
        order = np.lexsort((rotated_q[:, 2], rotated_q[:, 1], rotated_q[:, 0]))
        target_q = rotated_q[order]
        directed_action = apply_plane_wave_action(
            source_coefficients,
            source_q,
            sewing["rotation_cart"],
            sewing["translation_cart"],
            spin_rotation=sewing_spin,
            target_q_cart=target_q,
        )
        target_coefficients = directed_action.transformed_coefficients
        cross = build_plane_wave_representation(
            source_coefficients,
            source_q,
            sewing["rotation_cart"],
            sewing["translation_cart"],
            spin_rotation=sewing_spin,
            target_coefficients=target_coefficients,
            target_q_cart=target_q,
            action=directed_action,
        )
    except (KeyError, TypeError, ValueError):
        sewing_spin = None
        cross = None
        target_q = np.empty((0, 3))
        target_coefficients = np.empty((0, 0, 0), dtype=np.complex128)
        reasons.append("directed_plane_wave_action_failed")
    target_frame = build_target_frame(target_coefficients, wavecar_rtag=wavecar_rtag)
    reasons.extend(target_frame.reason_codes)
    permutation_status = "blocked"
    if cross is not None:
        permutation = validate_reciprocal_grid_permutation(
            cross.mapping, dimension=len(target_q)
        )
        permutation_status = permutation.status
        if permutation.status != "passed":
            reasons.extend(permutation.reason_codes)
        if cross.relative_norm_preservation_residual > DEFAULT_PLANE_WAVE_NORM_TOLERANCE:
            reasons.append("plane_wave_mapping_failed")
        if cross.relative_target_subspace_residual > DEFAULT_TARGET_SUBSPACE_TOLERANCE:
            reasons.append("target_subspace_closure_failed")

    source_projector = np.asarray(
        raw_inputs.get("source_projector"), dtype=np.complex128
    )
    full_sewing = (
        cross.matrix if cross is not None else np.empty((0, 0), dtype=np.complex128)
    )
    source_valley_basis = np.asarray(
        raw_inputs.get("source_valley_basis"), dtype=np.complex128
    )
    target_projector = (
        full_sewing @ source_projector @ full_sewing.conj().T
        if cross is not None else np.empty((0, 0), dtype=np.complex128)
    )
    target_valley_basis = (
        full_sewing @ source_valley_basis
        if cross is not None else np.empty((0, 0), dtype=np.complex128)
    )
    rank = (
        source_valley_basis.shape[1]
        if source_valley_basis.ndim == 2 else 0
    )
    shape_ok = (
        rank > 0
        and full_sewing.shape[0] == full_sewing.shape[1]
        and source_projector.shape == full_sewing.shape
        and target_projector.shape == full_sewing.shape
        and source_valley_basis.shape == (full_sewing.shape[0], rank)
        and target_valley_basis.shape == (full_sewing.shape[0], rank)
    )
    if shape_ok:
        sewing_matrix = (
            target_valley_basis.conj().T
            @ full_sewing @ source_valley_basis
        )
        normalization = float(np.sqrt(rank))
        unitarity = float(
            np.linalg.norm(sewing_matrix.conj().T @ sewing_matrix - np.eye(rank))
            / normalization
        )
        covariance = float(
            np.linalg.norm(
                full_sewing @ source_projector @ full_sewing.conj().T
                - target_projector
            ) / normalization
        )
        source_block = full_sewing @ source_valley_basis
        target_range = (
            target_valley_basis @ target_valley_basis.conj().T
        )
        block = float(
            np.linalg.norm(source_block - target_range @ source_block)
            / max(np.linalg.norm(source_block), np.finfo(float).tiny)
        )
        basis_residual = max(
            _valley_basis_residual(source_projector, source_valley_basis),
            _valley_basis_residual(target_projector, target_valley_basis),
            float(
                np.linalg.norm(
                    target_valley_basis
                    - full_sewing @ source_valley_basis
                ) / normalization
            ),
        )
    else:
        sewing_matrix = np.empty((0, 0), dtype=np.complex128)
        unitarity = covariance = block = float("inf")
        basis_residual = float("inf")
        reasons.append("directed_sewing_matrix_shape_mismatch")
    if basis_residual > DEFAULT_TARGET_SUBSPACE_TOLERANCE:
        reasons.append("valley_basis_range_failed")
    if unitarity > DEFAULT_TARGET_SUBSPACE_TOLERANCE:
        reasons.append("valley_sewing_unitarity_failed")
    if covariance > DEFAULT_PROJECTOR_COVARIANCE_TOLERANCE:
        reasons.append("projector_covariance_failed")
    if block > DEFAULT_VALLEY_BLOCK_TOLERANCE:
        reasons.append("valley_block_failed")

    conjugation_rows = []
    for source_id in source_ids:
        row = _directed_conjugation_row(
            source_id=source_id,
            target_ids=target_ids,
            sewing=sewing,
            operations=operations,
            source_k=source_k,
            target_k=target_k,
            source_q=source_q,
            target_q=target_q,
            source_coefficients=source_coefficients,
            target_coefficients=target_coefficients,
            sewing_matrix=sewing_matrix,
            source_valley_basis=source_valley_basis,
            target_valley_basis=target_valley_basis,
            sewing_map=cross.mapping if cross is not None else np.empty(0, dtype=int),
        )
        conjugation_rows.append(row)
        if not row.get("passed"):
            reasons.extend(row.get("reason_codes", []))
    if {row.get("target_operation_id") for row in conjugation_rows} != set(target_ids):
        reasons.append("little_group_conjugation_not_bijective")

    content = {
        "schema_version": SCOPED_REPRESENTATION_EVIDENCE_SCHEMA_VERSION,
        "source_basis_certificate_identity": source_basis.get("certificate_identity"),
        "double_space_group_lift_certificate_identity": (
            lift_record.get("certificate_identity")
            if isinstance(lift_record, Mapping) else None
        ),
        "extracted_wavefunction_payload_identity": payload_identity,
        "scope": {
            "scope_kind": "valley_sewing",
            "source_kpoint_label": str(raw_inputs.get("source_kpoint_label", "")),
            "target_kpoint_label": str(raw_inputs.get("target_kpoint_label", "")),
            "source_k_frac": source_k.tolist(),
            "target_k_frac": target_k.tolist(),
            "source_valley": source_valley,
            "target_valley": target_valley,
            "full_valley_orbit": orbit,
            "sewing_operation_valley_permutation": dict(mapping or {}),
            "source_operation_cycle": selected_cycle,
            "sewing_operation_id": sewing_id,
            "source_little_group_operation_ids": source_ids,
            "target_little_group_operation_ids": target_ids,
            "reciprocal_lattice_shift": reciprocal_shift,
        },
        "source_target_frames": {
            "source": deepcopy(source_frame.record),
            "target": deepcopy(target_frame.record),
            "target_basis_kind": "canonical_unitary_transport",
            "independent_target_numerical_evidence": False,
        },
        "reciprocal_grid": {
            "source_identity": reciprocal_grid_identity(source_q),
            "target_identity": reciprocal_grid_identity(target_q),
            "source_to_target_map": (
                cross.mapping.tolist() if cross is not None else []
            ),
            "permutation_status": permutation_status,
            "relative_norm_residual": (
                _finite_scalar(cross.relative_norm_preservation_residual)
                if cross is not None else None
            ),
            "relative_target_subspace_residual": (
                _finite_scalar(cross.relative_target_subspace_residual)
                if cross is not None else None
            ),
        },
        "valley_sewing_matrix": {
            "matrix": _complex_matrix(sewing_matrix),
            "spin_rotation": _complex_matrix(sewing_spin),
            "rank_normalized_unitarity_residual": _finite_scalar(unitarity),
            "valley_basis_range_residual": _finite_scalar(basis_residual),
        },
        "directed_group_law": {"rows": conjugation_rows},
        "intertwining": {"rows": conjugation_rows},
        "projector_covariance": {
            "rank_normalized_residual": _finite_scalar(covariance),
            "passed": covariance <= DEFAULT_PROJECTOR_COVARIANCE_TOLERANCE,
        },
        "valley_block_quality": {
            "relative_residual": _finite_scalar(block),
            "passed": block <= DEFAULT_VALLEY_BLOCK_TOLERANCE,
        },
        "grey_group_matching_allowed": False,
    }
    return ScopedRepresentationEvidence(content, tuple(dict.fromkeys(reasons)))


def validate_directed_valley_sewing_evidence_record(
    record: Mapping[str, object], **raw_inputs: object
) -> ScopedEvidenceValidation:
    """Recompute a directed unitary sewing scope and require exact equality."""
    try:
        recomputed = build_directed_valley_sewing_evidence(**raw_inputs).to_record()
    except (KeyError, TypeError, ValueError):
        return ScopedEvidenceValidation("blocked", ("evidence_recomputation_failed",))
    if dict(record) != recomputed:
        return ScopedEvidenceValidation("blocked", ("recomputed_evidence_mismatch",))
    return ScopedEvidenceValidation(
        str(record.get("status")),
        tuple(str(value) for value in record.get("reason_codes", [])),
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


def _directed_conjugation_row(
    *,
    source_id,
    target_ids,
    sewing,
    operations,
    source_k,
    target_k,
    source_q,
    target_q,
    source_coefficients,
    target_coefficients,
    sewing_matrix,
    source_valley_basis,
    target_valley_basis,
    sewing_map,
):
    reasons: list[str] = []
    target_id = None
    lattice_translation = np.zeros(3, dtype=int)
    try:
        source = operations[source_id]
        sewing_r = np.asarray(sewing["rotation_frac"], dtype=float)
        sewing_t = np.asarray(sewing["translation_frac"], dtype=float)
        inverse_r = np.linalg.inv(sewing_r)
        inverse_t = -inverse_r @ sewing_t
        rotation, translation = _compose_affine(
            sewing_r,
            sewing_t,
            *_compose_affine(
                np.asarray(source["rotation_frac"], dtype=float),
                np.asarray(source["translation_frac"], dtype=float),
                inverse_r,
                inverse_t,
            ),
        )
        matches = []
        for candidate_id in target_ids:
            candidate = operations[candidate_id]
            delta = translation - np.asarray(
                candidate["translation_frac"], dtype=float
            )
            lattice = np.rint(delta).astype(int)
            if np.allclose(
                rotation,
                candidate["rotation_frac"],
                atol=1.0e-8,
                rtol=0.0,
            ) and np.linalg.norm(delta - lattice) <= 1.0e-8:
                matches.append((candidate_id, lattice))
        if len(matches) != 1:
            raise ValueError
        target_id, lattice_translation = matches[0]
        target = operations[target_id]
        source_spin = spin_lift_from_orthogonal(source["rotation_cart"])
        target_spin = spin_lift_from_orthogonal(target["rotation_cart"])
        sewing_spin = spin_lift_from_orthogonal(sewing["rotation_cart"])
        conjugated_spin = sewing_spin @ source_spin @ np.linalg.inv(sewing_spin)
        if np.linalg.norm(conjugated_spin - target_spin) <= 1.0e-8:
            spin_factor = 1
        elif np.linalg.norm(conjugated_spin + target_spin) <= 1.0e-8:
            spin_factor = -1
        else:
            raise ValueError
        source_rep = build_plane_wave_representation(
            source_coefficients,
            source_q,
            source["rotation_cart"],
            source["translation_cart"],
            spin_rotation=source_spin,
        )
        target_rep = build_plane_wave_representation(
            target_coefficients,
            target_q,
            target["rotation_cart"],
            target["translation_cart"],
            spin_rotation=target_spin,
        )
        source_matrix = (
            source_valley_basis.conj().T
            @ source_rep.matrix @ source_valley_basis
        )
        target_matrix = (
            target_valley_basis.conj().T
            @ target_rep.matrix @ target_valley_basis
        )
        target_character = complex(np.trace(target_matrix))
        source_permutation = validate_reciprocal_grid_permutation(
            source_rep.mapping, dimension=len(source_q)
        )
        target_permutation = validate_reciprocal_grid_permutation(
            target_rep.mapping, dimension=len(target_q)
        )
        left_map = target_rep.mapping[np.asarray(sewing_map, dtype=int)]
        right_map = np.asarray(sewing_map, dtype=int)[source_rep.mapping]
        map_composition_passed = bool(
            source_permutation.status == "passed"
            and target_permutation.status == "passed"
            and np.array_equal(left_map, right_map)
        )
        if not map_composition_passed:
            reasons.append("directed_plane_wave_composition_failed")
        transformed_k = (
            np.linalg.inv(
                np.asarray(target["rotation_frac"], dtype=float)
            ).T @ target_k
        )
        phase = np.exp(
            -2.0j * np.pi * float(transformed_k @ lattice_translation)
        )
        factor = spin_factor * phase
        rank = sewing_matrix.shape[1]
        residual = float(
            np.linalg.norm(
                target_matrix @ sewing_matrix
                - factor * sewing_matrix @ source_matrix
            ) / np.sqrt(rank)
        )
        if residual > DEFAULT_GROUP_LAW_TOLERANCE:
            reasons.append("rank_normalized_intertwining_failed")
    except (KeyError, TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        spin_factor = phase = residual = target_character = None
        map_composition_passed = False
        reasons.append("directed_little_group_conjugation_failed")
    return {
        "source_operation_id": source_id,
        "target_operation_id": target_id,
        "affine_lattice_translation": lattice_translation.tolist(),
        "bloch_phase": (
            [float(phase.real), float(phase.imag)] if phase is not None else None
        ),
        "double_group_cocycle_factor": spin_factor,
        "source_to_target_map_composition_passed": map_composition_passed,
        "rank_normalized_residual": residual,
        "target_character": (
            [float(target_character.real), float(target_character.imag)]
            if target_character is not None else None
        ),
        "passed": not reasons,
        "reason_codes": reasons,
    }


def _directed_hsp_shift(sewing, source_k, target_k):
    try:
        mapped = np.linalg.inv(
            np.asarray(sewing["rotation_frac"], dtype=float)
        ).T @ source_k
        delta = mapped - target_k
        shift = np.rint(delta).astype(int)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
        return None
    return shift.tolist() if np.linalg.norm(delta - shift) <= 1.0e-8 else None


def _operation_preserves(operation, kpoint, valley):
    mapping = operation.get("sector_mapping")
    return bool(
        isinstance(mapping, Mapping)
        and mapping.get(valley) == valley
        and _directed_hsp_shift(operation, kpoint, kpoint) is not None
    )


def _valley_basis_residual(projector, basis):
    rank = basis.shape[1]
    hermitian = 0.5 * (projector + projector.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    selected = eigenvalues > 0.5
    if int(np.count_nonzero(selected)) != rank:
        return float("inf")
    spectral = eigenvectors[:, selected] @ eigenvectors[:, selected].conj().T
    supplied = basis @ basis.conj().T
    orthogonality = np.linalg.norm(basis.conj().T @ basis - np.eye(rank))
    return float(max(
        orthogonality, np.linalg.norm(spectral - supplied)
    ) / np.sqrt(rank))


def _compose_affine(left_r, left_t, right_r, right_t):
    return left_r @ right_r, left_r @ right_t + left_t


def _complex_matrix(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.complex128)
    return [
        [[float(item.real), float(item.imag)] for item in row]
        for row in array
    ]


def _finite_scalar(value):
    return float(value) if np.isfinite(value) else None


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
        raw_fro = float(
            np.linalg.norm(
                matrix.conj().T @ matrix - np.eye(matrix.shape[0]), ord="fro"
            )
        )
        normalization = float(np.sqrt(matrix.shape[0]))
        residual = raw_fro / normalization
        passed = residual <= tolerance
        if not passed:
            reasons.append("target_subspace_closure_failed")
        rows.append(
            {
                "operation_id": operation_id,
                "unitarity_residual_raw_fro": raw_fro,
                "unitarity_residual_normalization": normalization,
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
) -> tuple[list[dict[str, object]], dict[int, list[int]]]:
    rows: list[dict[str, object]] = []
    maps: dict[int, list[int]] = {}
    for operation_id in operation_ids:
        raw = evidence.get(operation_id)
        if not isinstance(raw, Mapping):
            reasons.append("plane_wave_mapping_evidence_missing")
            continue
        convention = raw.get("action_convention")
        grid_identity = raw.get("reciprocal_grid_identity")
        dimension = raw.get("reciprocal_grid_dimension")
        q_cart = raw.get("q_cart")
        rotation_cart = raw.get("rotation_cart")
        map_tolerance = raw.get("mapping_tolerance")
        serialized_map = raw.get("source_to_target_map")
        miss_count = raw.get("mapping_miss_count")
        residual = raw.get("relative_norm_residual")
        if (
            convention != RECIPROCAL_GRID_ACTION_CONVENTION
            or not valid_sha256_identity(grid_identity)
            or not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 1
            or not isinstance(miss_count, int)
            or isinstance(miss_count, bool)
            or miss_count < 0
        ):
            reasons.append("plane_wave_mapping_evidence_malformed")
            continue
        try:
            q = np.asarray(q_cart, dtype=float)
            rotation = np.asarray(rotation_cart, dtype=float)
            effective_map_tolerance = float(map_tolerance)
            norm_residual = float(residual)
        except (TypeError, ValueError):
            reasons.append("plane_wave_mapping_evidence_malformed")
            continue
        if (
            q.shape != (dimension, 3)
            or rotation.shape != (3, 3)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(rotation))
            or not np.isfinite(effective_map_tolerance)
            or effective_map_tolerance <= 0.0
            or not np.isfinite(norm_residual)
            or norm_residual < 0.0
        ):
            reasons.append("plane_wave_mapping_evidence_malformed")
            continue

        permutation_validation = validate_reciprocal_grid_permutation(
            serialized_map,
            dimension=dimension,
        )
        if permutation_validation.status != "passed":
            if any(
                reason in permutation_validation.reason_codes
                for reason in (
                    "mapping_collection_malformed",
                    "mapping_index_malformed",
                )
            ):
                reasons.append("plane_wave_mapping_evidence_malformed")
            else:
                reasons.append("plane_wave_mapping_not_bijective")
        try:
            recomputed = build_reciprocal_grid_map(
                q,
                rotation,
                tolerance=effective_map_tolerance,
            )
            recomputed_map = recomputed.mapping.tolist()
            expected_grid_identity = reciprocal_grid_identity(q)
        except ValueError:
            reasons.append("plane_wave_mapping_evidence_malformed")
            continue
        if grid_identity != expected_grid_identity:
            reasons.append("plane_wave_grid_identity_mismatch")
        if not isinstance(serialized_map, (list, tuple)):
            normalized_map: list[object] = []
        else:
            normalized_map = list(serialized_map)
        if normalized_map != recomputed_map:
            reasons.append("plane_wave_mapping_recomputation_mismatch")
        recomputed_validation = validate_reciprocal_grid_permutation(
            recomputed.mapping,
            dimension=dimension,
        )
        if recomputed_validation.status != "passed":
            reasons.append("plane_wave_mapping_not_bijective")
        if miss_count != recomputed.mapping_miss_count:
            reasons.append("plane_wave_mapping_miss_count_mismatch")
        passed = (
            miss_count == 0
            and permutation_validation.status == "passed"
            and recomputed_validation.status == "passed"
            and normalized_map == recomputed_map
            and grid_identity == expected_grid_identity
            and norm_residual <= tolerance
        )
        if not passed:
            reasons.append("plane_wave_mapping_failed")
        if (
            permutation_validation.status == "passed"
            and normalized_map == recomputed_map
        ):
            maps[operation_id] = [int(value) for value in normalized_map]
        rows.append(
            {
                "operation_id": operation_id,
                "action_convention": convention,
                "reciprocal_grid_identity": grid_identity,
                "reciprocal_grid_dimension": dimension,
                "source_to_target_map": normalized_map,
                "mapping_miss_count": miss_count,
                "mapping_tolerance": effective_map_tolerance,
                "relative_norm_residual": norm_residual,
                "reciprocal_grid_permutation_passed": (
                    permutation_validation.status == "passed"
                    and recomputed_validation.status == "passed"
                ),
                "passed": bool(passed),
            }
        )
    return rows, maps


def _plane_wave_composition_rows(
    *,
    operation_ids: Sequence[int],
    maps: Mapping[int, Sequence[int]],
    lift_record: Mapping[str, object],
    reasons: list[str],
) -> list[dict[str, object]]:
    pairwise = lift_record.get("pairwise_products")
    if not isinstance(pairwise, Mapping):
        reasons.append("plane_wave_map_composition_evidence_missing")
        return []
    required = set(operation_ids)
    rows: list[dict[str, object]] = []
    for left_id in operation_ids:
        for right_id in operation_ids:
            pair = pairwise.get(f"{left_id},{right_id}")
            if not isinstance(pair, Mapping):
                reasons.append("plane_wave_map_composition_evidence_missing")
                continue
            product_id = pair.get("product_operation_id")
            if product_id not in required:
                reasons.append("required_operation_scope_not_closed")
                continue
            left = maps.get(left_id)
            right = maps.get(right_id)
            product = maps.get(product_id)
            if left is None or right is None or product is None:
                reasons.append("plane_wave_map_composition_evidence_missing")
                continue
            if not (len(left) == len(right) == len(product)):
                reasons.append("plane_wave_map_composition_failed")
                continue
            try:
                composed = [int(left[int(right[index])]) for index in range(len(right))]
            except (IndexError, TypeError, ValueError):
                reasons.append("plane_wave_map_composition_failed")
                continue
            passed = composed == list(product)
            if not passed:
                reasons.append("plane_wave_map_composition_failed")
            rows.append(
                {
                    "left_operation_id": left_id,
                    "right_operation_id": right_id,
                    "product_operation_id": product_id,
                    "composed_source_to_target_map": composed,
                    "passed": passed,
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
            raw_fro = float(
                np.linalg.norm(left @ right - sign * phase * product, ord="fro")
            )
            normalization = float(np.sqrt(left.shape[0]))
            residual = raw_fro / normalization
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
                    "residual_raw_fro": raw_fro,
                    "residual_normalization": normalization,
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
            raw_fro = float(
                np.linalg.norm(
                    matrix @ source_projector @ matrix.conj().T
                    - target_projector,
                    ord="fro",
                )
            )
            residual = raw_fro / denominator
            passed = residual <= tolerance
            if not passed:
                reasons.append("projector_covariance_failed")
            rows.append(
                {
                    "operation_id": operation_id,
                    "source_valley": valley,
                    "mapped_valley": mapped,
                    "residual_raw_fro": raw_fro,
                    "residual_normalization": denominator,
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
            source_ortho_raw = float(
                np.linalg.norm(
                    source_basis.conj().T @ source_basis
                    - np.eye(source_basis.shape[1]),
                    ord="fro",
                )
            )
            source_ortho_normalization = float(
                np.sqrt(source_basis.shape[1])
            )
            source_ortho = (
                source_ortho_raw / source_ortho_normalization
            )
            target_ortho_raw = float(
                np.linalg.norm(
                    target_basis.conj().T @ target_basis
                    - np.eye(target_basis.shape[1]),
                    ord="fro",
                )
            )
            target_ortho_normalization = float(
                np.sqrt(target_basis.shape[1])
            )
            target_ortho = (
                target_ortho_raw / target_ortho_normalization
            )
            block = target_basis.conj().T @ matrix @ source_basis
            if block.shape[0] != block.shape[1]:
                block_unitarity_raw = float("inf")
                block_unitarity_normalization = float(
                    np.sqrt(max(block.shape[1], 1))
                )
                block_unitarity = float("inf")
            else:
                block_unitarity_raw = float(
                    np.linalg.norm(
                        block.conj().T @ block - np.eye(block.shape[1]),
                        ord="fro",
                    )
                )
                block_unitarity_normalization = float(
                    np.sqrt(block.shape[1])
                )
                block_unitarity = (
                    block_unitarity_raw
                    / block_unitarity_normalization
                )
            transformed = matrix @ source_basis
            denominator = max(float(np.linalg.norm(transformed, ord="fro")), 1.0e-14)
            leakage_raw = float(
                np.linalg.norm(
                    transformed - target_basis @ block, ord="fro"
                )
            )
            leakage = leakage_raw / denominator
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
                    "source_basis_orthonormality_raw_fro": (
                        source_ortho_raw
                    ),
                    "source_basis_orthonormality_normalization": (
                        source_ortho_normalization
                    ),
                    "source_basis_orthonormality_residual": source_ortho,
                    "target_basis_orthonormality_raw_fro": (
                        target_ortho_raw
                    ),
                    "target_basis_orthonormality_normalization": (
                        target_ortho_normalization
                    ),
                    "target_basis_orthonormality_residual": target_ortho,
                    "block_unitarity_raw_fro": block_unitarity_raw,
                    "block_unitarity_normalization": (
                        block_unitarity_normalization
                    ),
                    "block_unitarity_residual": block_unitarity,
                    "block_leakage_raw_fro": leakage_raw,
                    "block_leakage_normalization": denominator,
                    "block_leakage_residual": leakage,
                    "passed": bool(passed),
                }
            )
    return rows
