"""Tracked-only producer-chain regression for exact TR irrep completion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
from valleyscope.analysis.ebr_problem_instances import (
    build_ebr_problem_instances,
)
from valleyscope.analysis.database_ingestion_record import (
    build_database_ingestion_record,
)
from valleyscope.analysis.irreptables_runtime_table_builder import (
    build_auto_canonical_reduced_ebr_table,
)
from valleyscope.analysis.reduced_ebr_mapping import (
    build_reduced_ebr_mapping,
)
from valleyscope.analysis.scoped_representation_evidence import (
    build_scoped_representation_evidence,
)
from valleyscope.analysis.time_reversal_orbits import (
    build_time_reversal_valley_orbit_report,
)
from valleyscope.analysis.tr_irrep_completion import (
    attach_tr_irrep_completion_certificates,
    validate_tr_irrep_completion_certificate,
)
from valleyscope.analysis.unitary_provenance import (
    validate_tr_completed_unitary_bundle,
)
from valleyscope.geometry.lattice import (
    cart_rotation_from_fractional,
    cart_translation_from_fractional,
)
from valleyscope.io.h5_reader import read_wavefunction_h5
from valleyscope.io.spinor_source_basis import (
    build_spinor_source_basis_certificate,
)
from valleyscope.io.wavefunction_convention import canonical_identity
from valleyscope.irreps.ebr_data_adapter import load_ebr_source_data
from valleyscope.irreps.tables import (
    build_spinful_source_table_evidence,
    load_standard_irrep_table,
    resolve_ebr_source_irrep_label_evidence,
)
from valleyscope.irreps.time_reversal_source import (
    derive_time_reversal_source_irrep_orbits,
    validate_reviewed_time_reversal_source_context,
)
from valleyscope.symmetry.double_space_group_lift import (
    build_double_space_group_lift_certificate,
)
from valleyscope.symmetry.plane_wave_action import (
    RECIPROCAL_GRID_ACTION_CONVENTION,
    reciprocal_grid_identity,
)
from tests.reduced_ebr_promo_helpers import (
    real_primitive_certificate_dict,
)


def _write_spinor_payload(path: Path, direct_lattice: np.ndarray) -> None:
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct_lattice).T
    with h5py.File(path, "w") as h5:
        metadata = h5.create_group("metadata")
        lattice = metadata.create_group("lattice")
        lattice["direct_cart"] = direct_lattice
        lattice["reciprocal_cart"] = reciprocal
        metadata["spinor"] = True
        metadata["source"] = "portable_producer_chain"
        metadata["vasp_band_index_base"] = 1
        kpoints = h5.create_group("kpoints")
        for index, label in enumerate(("K_left", "K_right")):
            kpoint = kpoints.create_group(str(index))
            kpoint["name"] = label
            kpoint["frac"] = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
            kpoint["cart"] = np.zeros(3)
            kpoint["g_vectors_frac"] = np.zeros((1, 3), dtype=int)
            kpoint["g_vectors_cart"] = np.zeros((1, 3))
            kpoint["coefficients"] = np.eye(
                2, dtype=np.complex128
            ).reshape(2, 2, 1)
            kpoint["energies_eV"] = np.zeros(2)
            kpoint["band_indices_vasp"] = np.array([1, 2])


def _producer_contexts(tmp_path: Path):
    direct_lattice = np.array([
        [1.0, 0.0, 0.0],
        [-0.5, np.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, 8.0],
    ])
    payload = tmp_path / "spinor_wavefunctions.h5"
    _write_spinor_payload(payload, direct_lattice)
    source = build_spinor_source_basis_certificate(
        read_wavefunction_h5(payload)
    ).to_record()

    table = load_standard_irrep_table(143, spinor=True)
    standard_certificate = real_primitive_certificate_dict(
        143, "P3", spinor=True
    )
    assert standard_certificate is not None
    standard_certificate["parent_basis_operation_ids"] = [1, 2, 3]
    standard_certificate["affine_operation_map"] = {
        "1": 0,
        "2": 1,
        "3": 2,
    }
    operation_indices = [
        operation.table_index for operation in table.operations
    ]
    source_table = build_spinful_source_table_evidence(
        table,
        required_operation_indices=operation_indices,
    )
    operations = [{
        "operation_id": operation.table_index,
        "rotation_frac": operation.rotation_frac,
        "translation_frac": operation.translation_frac,
        "rotation_cart": cart_rotation_from_fractional(
            operation.rotation_frac, direct_lattice
        ),
        "translation_cart": cart_translation_from_fractional(
            operation.translation_frac, direct_lattice
        ),
    } for operation in table.operations]
    setting = {
        "schema_version": "1.0.0",
        "parent_to_standard_direct_transform": np.eye(3).tolist(),
        "origin_shift_fractional": [0.0, 0.0, 0.0],
        "parent_to_standard_operation_map": {
            str(index): index for index in operation_indices
        },
    }
    lift_inputs = {
        "expected_operations": operations,
        "source_table_identity": source_table,
        "standard_setting_identity": setting,
        "direct_lattice_cart": direct_lattice,
    }
    lift = build_double_space_group_lift_certificate(
        source,
        operations,
        source_table_identity=source_table,
        standard_setting_identity=setting,
        direct_lattice_cart=direct_lattice,
    ).to_record()
    identity_operation = next(
        operation.table_index
        for operation in table.operations
        if np.array_equal(
            operation.rotation_frac, np.eye(3, dtype=int)
        )
        and np.allclose(operation.translation_frac, 0.0)
    )
    q_cart = np.zeros((1, 3))
    context: dict[str, object] = {}
    cprime_by_valley: dict[str, dict[str, object]] = {}
    for valley, sampled_kpoint in (
        ("left", "K_left"),
        ("right", "K_right"),
    ):
        raw_inputs = {
            "source_basis_record": source,
            "lift_record": lift,
            "lift_validation_inputs": lift_inputs,
            "extracted_wavefunction_payload_identity": source[
                "extracted_wavefunction_payload_identity"
            ],
            "kpoint_label": sampled_kpoint,
            "kpoint_frac": np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
            "scope_kind": "local_irrep",
            "source_valleys": (valley,),
            "valley_orbit": (valley,),
            "required_operation_ids": (identity_operation,),
            "representations": {
                identity_operation: np.eye(2, dtype=np.complex128),
            },
            "plane_wave_evidence": {
                identity_operation: {
                    "action_convention": (
                        RECIPROCAL_GRID_ACTION_CONVENTION
                    ),
                    "reciprocal_grid_identity": reciprocal_grid_identity(
                        q_cart
                    ),
                    "reciprocal_grid_dimension": 1,
                    "q_cart": q_cart,
                    "rotation_cart": np.eye(3),
                    "mapping_tolerance": 1.0e-6,
                    "source_to_target_map": [0],
                    "mapping_miss_count": 0,
                    "relative_norm_residual": 0.0,
                },
            },
            "target_coefficients": np.eye(
                2, dtype=np.complex128
            ).reshape(2, 1, 2),
            "projectors": {
                valley: np.diag([1.0, 0.0]).astype(np.complex128),
            },
            "valley_bases": {
                valley: np.array(
                    [[1.0], [0.0]], dtype=np.complex128
                ),
            },
            "valley_mappings": {
                identity_operation: {valley: valley},
            },
        }
        record = build_scoped_representation_evidence(
            **raw_inputs
        ).to_record()
        assert source["status"] == lift["status"] == record["status"] == (
            "passed"
        )
        context[record["evidence_identity"]] = {
            "record": record,
            "raw_inputs": raw_inputs,
            "standard_setting_certificate": standard_certificate,
        }
        cprime_by_valley[valley] = {
            "spinor_source_basis_certificate_identity": source[
                "certificate_identity"
            ],
            "double_space_group_lift_certificate_identity": lift[
                "certificate_identity"
            ],
            "scoped_representation_evidence_identity": record[
                "evidence_identity"
            ],
        }
    return table, context, cprime_by_valley, standard_certificate


def _candidate(
    *,
    valley: str,
    sampled_kpoint: str,
    irrep: str,
    cprime: dict[str, object],
    standard_setting_certificate: dict[str, object],
) -> dict[str, object]:
    return {
        "valley": valley,
        "matched_irrep": irrep,
        "irrep_multiplicity": 1,
        "kpoint": sampled_kpoint,
        "workflow_path": "direct_qcut",
        "readiness_level": "trusted",
        "source": f"portable/{valley}/K",
        "subspace_group_candidate": "P3",
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_number": 143,
            "candidate_space_group_symbol": "P3",
        },
        "irrep_source_provenance": {
            "matching_strategy": "bilbao_restricted_character",
            "subspace_space_group_number": 143,
            "subspace_space_group_symbol": "P3",
            "source_table_sg_number": 143,
            "source_table_name": "P3",
            "source_table_spinor": True,
            "source_hsp_label": "K",
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": (
                    standard_setting_certificate
                ),
            },
            "cprime": cprime,
        },
        "ready_for_ebr_input": True,
    }


def _portable_orbit_inputs(tmp_path: Path):
    table, context, cprime, standard_certificate = _producer_contexts(
        tmp_path
    )
    ebr_source = load_ebr_source_data(143, True)
    reviewed = resolve_ebr_source_irrep_label_evidence(
        table=table,
        source_basis_labels=ebr_source["source_basis_labels"],
    )
    in_plane_rows = [
        row for row in reviewed["reviewed_rows"]
        if row.kpoint_label in {"K", "KA"}
    ]
    source_table_identity = {
        "space_group_number": table.number,
        "space_group_symbol": table.name,
        "source_table_name": table.name,
        "source_table_provenance": "irreptables.StandardIrrepTable",
        "spinor": table.spinor,
    }
    parent_affine_operations = [
        {
            "operation_id": operation.table_index,
            "rotation_frac": operation.rotation_frac,
            "translation_frac": operation.translation_frac,
        }
        for operation in table.operations
    ]
    parent_affine_lift_record = next(iter(context.values()))[
        "raw_inputs"
    ]["lift_record"]
    tr_source = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=in_plane_rows,
        centering_vectors=[[0.0, 0.0, 0.0]],
        source_table_identity=source_table_identity,
        standard_setting_certificate=standard_certificate,
        parent_affine_operations=parent_affine_operations,
        parent_affine_lift_record=parent_affine_lift_record,
    )
    left = _candidate(
        valley="left",
        sampled_kpoint="K_left",
        irrep="-K4",
        cprime=cprime["left"],
        standard_setting_certificate=standard_certificate,
    )
    right = _candidate(
        valley="right",
        sampled_kpoint="K_right",
        irrep="-K4",
        cprime=cprime["right"],
        standard_setting_certificate=standard_certificate,
    )
    return {
        "table": table,
        "context": context,
        "standard_certificate": standard_certificate,
        "reviewed": reviewed,
        "in_plane_rows": in_plane_rows,
        "source_table_identity": source_table_identity,
        "parent_affine_operations": parent_affine_operations,
        "parent_affine_lift_record": parent_affine_lift_record,
        "tr_source": tr_source,
        "candidates": [left, right],
    }


def _build_portable_orbit_report(inputs, *, sources=None):
    source = inputs["tr_source"]
    return build_time_reversal_valley_orbit_report(
        valley_mapping_report={
            "status": "validated",
            "enabled": True,
            "theta_square": -1,
            "time_reversal_valley_mapping": {
                "left": "right",
                "right": "left",
            },
            "valley_orbits": [{
                "representative": "left",
                "members": ["left", "right"],
                "mapping_type": "exchanged",
            }],
        },
        source_irrep_orbits_by_valley=(
            sources
            if sources is not None
            else {"left": source, "right": source}
        ),
        grey_source_by_valley={},
        ebr_input_candidates={
            "status": "has_candidates",
            "candidate_count": 2,
            "candidates": inputs["candidates"],
        },
    )


def _complete_and_export_portable_orbit(inputs, report):
    completed = attach_tr_irrep_completion_certificates(
        time_reversal_orbit_report=report,
        cprime_validation_context=inputs["context"],
    )
    problems = build_ebr_problem_instances(
        ebr_input_candidates={
            "status": "has_candidates",
            "candidate_count": 2,
            "candidates": inputs["candidates"],
        },
        time_reversal_orbit_report=completed,
    )
    export = build_ebr_export_bundle(
        ebr_problem_instances=problems,
    )
    return completed, problems, export


def test_tracked_only_exact_completion_uses_real_irreptables_source(
    tmp_path,
):
    inputs = _portable_orbit_inputs(tmp_path)
    table = inputs["table"]
    context = inputs["context"]
    reviewed = inputs["reviewed"]
    tr_source = inputs["tr_source"]
    left, right = inputs["candidates"]
    assert reviewed["status"] == tr_source["status"] == "validated"
    assert tr_source["irrep_partner_by_label"]["-K4"] == "-KA4"
    assert tr_source["reviewed_source_context"]["reviewed_rows"]
    assert tr_source["reviewed_source_context"]["normalized_centering_vectors"] == [
        [0.0, 0.0, 0.0],
    ]

    report = _build_portable_orbit_report(inputs)
    assert report["status"] == "blocked"
    orbit = report["valley_orbits"][0]
    assert orbit["unitary_completion_status"] == "validated"
    assert orbit["unitary_completion_blockers"] == []
    assert orbit["joint_corepresentation_status"] == "blocked"
    assert "grey_group_time_reversal_source_not_validated" in orbit[
        "joint_corepresentation_blockers"
    ]
    assert orbit["unitary_valley_irrep_completion_records"]["left"]["KA"][0][
        "completion_kind"
    ] == "inferred_by_time_reversal"
    completed = attach_tr_irrep_completion_certificates(
        time_reversal_orbit_report=report,
        cprime_validation_context=context,
    )
    assert completed["status"] == "blocked", completed
    assert completed["valley_orbits"][0]["unitary_completion_status"] == (
        "validated"
    )
    problems = build_ebr_problem_instances(
        ebr_input_candidates={
            "status": "has_candidates",
            "candidate_count": 2,
            "candidates": [left, right],
        },
        time_reversal_orbit_report=completed,
    )
    unitary = [
        instance for instance in problems["instances"]
        if instance["problem_kind"] == "unitary_valley_reduced_ebr"
    ]
    joint = next(
        instance for instance in problems["instances"]
        if instance["problem_kind"] == "valley_orbit_reduced_ebr"
    )
    assert len(unitary) == 2
    assert all(
        instance["canonical_hsp_vector_ready"] is True
        for instance in unitary
    ), unitary
    assert joint["canonical_hsp_vector_ready"] is False
    assert "joint_time_reversal_corepresentation_not_certified" in joint[
        "blocked_by"
    ]

    export = build_ebr_export_bundle(
        ebr_problem_instances=problems
    )
    assert export["bundle_count"] == 2
    assert all(
        bundle["time_reversal"]["reviewed_time_reversal_source_context"][
            "reviewed_rows"
        ]
        for bundle in export["bundles"]
    )
    assert all(
        validate_tr_completed_unitary_bundle(bundle)
        for bundle in export["bundles"]
    )
    inferred = export["bundles"][0][
        "unitary_irrep_completion_records_by_hsp"
    ]["KA"][0]
    assert inferred["tr_irrep_completion_certificate"][
        "certificate_kind"
    ] == "exact_tr_irrep_completion"

    first_bundle = export["bundles"][0]
    reduced_table = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=143,
        spinor=True,
        bundle_irreps_by_kpoint=first_bundle["irreps_by_kpoint"],
        expected_hsps=first_bundle["expected_hsps"],
        subspace_group_candidate="P3",
    )
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=export,
        table=reduced_table,
        reduced_ebr_input={"source": "portable_irreptables_runtime"},
        cprime_validation_context=context,
    )
    assert len(mapping["solutions"]) == 2, mapping
    source_basis_identity = next(iter(
        first_bundle["cprime_identity_by_kpoint"].values()
    ))["spinor_source_basis_certificate_identity"]
    acceptance_matrix = [
        {
            "kpoint": hsp,
            "valley": bundle["valley"],
            "double_space_group_lift_status": "passed",
            "double_space_group_lift_identity": links[
                "double_space_group_lift_certificate_identity"
            ],
            "scoped_representation_status": "passed",
            "scoped_representation_evidence_identity": links[
                "scoped_representation_evidence_identity"
            ],
        }
        for bundle in export["bundles"]
        for hsp, links in bundle["cprime_identity_by_kpoint"].items()
    ]
    ingestion = build_database_ingestion_record(
        valley_summary={
            "schema_version": "2.0.0",
            "target_kpoints": ["K_left", "K_right"],
            "iband": [1, 2],
            "input": {},
            "cprime": {
                "spinor_source_basis": {
                    "status": "passed",
                    "identity": source_basis_identity,
                    "blockers": [],
                },
                "acceptance_matrix": acceptance_matrix,
            },
        },
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )
    assert ingestion["final_reduced_ebr_result_count"] == 2, ingestion

    tampered = deepcopy(export["bundles"][0])
    tampered["unitary_irrep_completion_records_by_hsp"]["KA"][0].pop(
        "tr_irrep_completion_certificate"
    )
    assert not validate_tr_completed_unitary_bundle(tampered)


def test_portable_unitary_completion_rejects_disagreeing_reviewed_models(
    tmp_path,
):
    inputs = _portable_orbit_inputs(tmp_path)
    rows = inputs["in_plane_rows"]
    ka5 = next(row for row in rows if row.label == "-KA5")
    ka4 = next(row for row in rows if row.label == "-KA4")
    disagreeing_rows = [
        replace(row, characters=ka4.characters)
        if row.label == "-KA5"
        else replace(row, characters=ka5.characters)
        if row.label == "-KA4"
        else row
        for row in rows
    ]
    disagreeing = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=disagreeing_rows,
        centering_vectors=[[0.0, 0.0, 0.0]],
        source_table_identity=inputs["source_table_identity"],
        standard_setting_certificate=inputs["standard_certificate"],
        parent_affine_operations=inputs["parent_affine_operations"],
        parent_affine_lift_record=inputs["parent_affine_lift_record"],
    )
    assert disagreeing["status"] == "validated"
    assert disagreeing["irrep_partner_by_label"]["-K4"] == "-KA5"

    report = _build_portable_orbit_report(
        inputs,
        sources={
            "left": inputs["tr_source"],
            "right": disagreeing,
        },
    )
    orbit = report["valley_orbits"][0]
    assert orbit["unitary_completion_status"] == "blocked"
    assert "valley_source_time_reversal_models_disagree" in orbit[
        "unitary_completion_blockers"
    ]
    completed, problems, export = _complete_and_export_portable_orbit(
        inputs, report
    )
    assert all(
        "tr_irrep_completion_certificate" not in record
        for by_hsp in completed["valley_orbits"][0][
            "unitary_valley_irrep_completion_records"
        ].values()
        for records in by_hsp.values()
        for record in records
        if record["completion_kind"] == "inferred_by_time_reversal"
    )
    assert not any(
        instance["canonical_hsp_vector_ready"]
        for instance in problems["instances"]
        if instance["problem_kind"] == "unitary_valley_reduced_ebr"
    )
    assert export["bundle_count"] == 0


def test_portable_unitary_completion_rejects_missing_or_malformed_raw_context(
    tmp_path,
):
    inputs = _portable_orbit_inputs(tmp_path)
    for mutation in ("missing", "malformed", "centering_mismatch"):
        source = deepcopy(inputs["tr_source"])
        if mutation == "missing":
            source.pop("reviewed_source_context")
        else:
            context = source["reviewed_source_context"]
            if mutation == "malformed":
                context["reviewed_rows"] = []
            else:
                context["normalized_centering_vectors"] = [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                ]
            content = {
                key: deepcopy(value)
                for key, value in context.items()
                if key not in {"status", "context_identity", "blockers"}
            }
            context["context_identity"] = canonical_identity(content)
        report = _build_portable_orbit_report(
            inputs,
            sources={"left": source, "right": source},
        )
        orbit = report["valley_orbits"][0]
        assert orbit["unitary_completion_status"] == "blocked"
        assert "time_reversal_source_irrep_orbits_not_validated" in orbit[
            "unitary_completion_blockers"
        ]
        _, problems, export = _complete_and_export_portable_orbit(
            inputs, report
        )
        assert not any(
            instance["canonical_hsp_vector_ready"]
            for instance in problems["instances"]
            if instance["problem_kind"] == "unitary_valley_reduced_ebr"
        )
        assert export["bundle_count"] == 0


@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("hall_symbol", "FORGED HALL"),
        ("origin_shift_fractional", [0.125, 0.0, 0.0]),
        (
            "parent_to_standard_direct_transform",
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
    ],
)
def test_portable_completion_cross_binds_raw_affine_setting_to_local_irrep(
    tmp_path,
    field,
    forged_value,
):
    inputs = _portable_orbit_inputs(tmp_path)
    forged_source = deepcopy(inputs["tr_source"])
    context = forged_source["reviewed_source_context"]
    context["standard_setting_certificate"][field] = forged_value
    content = {
        key: deepcopy(value)
        for key, value in context.items()
        if key not in {"status", "context_identity", "blockers"}
    }
    context["context_identity"] = canonical_identity(content)

    report = _build_portable_orbit_report(
        inputs,
        sources={"left": forged_source, "right": forged_source},
    )
    orbit = report["valley_orbits"][0]
    assert orbit["unitary_completion_status"] == "blocked"

    completed, problems, export = _complete_and_export_portable_orbit(
        inputs, report
    )
    completed_orbit = completed["valley_orbits"][0]
    assert completed_orbit["unitary_completion_status"] == "blocked"
    expected_blocker = "time_reversal_source_irrep_orbits_not_validated"
    assert any(
        expected_blocker in blocker
        for blocker in completed_orbit["unitary_completion_blockers"]
    )
    assert all(
        "tr_irrep_completion_certificate" not in record
        for by_hsp in completed_orbit[
            "unitary_valley_irrep_completion_records"
        ].values()
        for records in by_hsp.values()
        for record in records
        if record["completion_kind"] == "inferred_by_time_reversal"
    )
    assert not any(
        instance["canonical_hsp_vector_ready"]
        for instance in problems["instances"]
        if instance["problem_kind"] == "unitary_valley_reduced_ebr"
    )
    assert export["bundle_count"] == 0


@pytest.mark.parametrize(
    ("setting_field", "bundle_field", "forged_value"),
    [
        ("hall_symbol", "hall_symbol", "FORGED HALL"),
        (
            "parent_to_standard_direct_transform",
            "normalized_direct_transform",
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        (
            "origin_shift_fractional",
            "normalized_origin_shift",
            [0.125, 0.0, 0.0],
        ),
    ],
)
def test_portable_unitary_validator_rejects_coordinated_affine_substitution(
    tmp_path,
    setting_field,
    bundle_field,
    forged_value,
):
    inputs = _portable_orbit_inputs(tmp_path)
    _, _, export = _complete_and_export_portable_orbit(
        inputs,
        _build_portable_orbit_report(inputs),
    )
    forged = deepcopy(export["bundles"][0])
    time_reversal = forged["time_reversal"]
    source_context = time_reversal[
        "reviewed_time_reversal_source_context"
    ]
    source_context["standard_setting_certificate"][setting_field] = (
        forged_value
    )
    if setting_field == "parent_to_standard_direct_transform":
        transform = np.asarray(forged_value, dtype=float)
        inverse = np.linalg.inv(transform)
        for operation in source_context["parent_affine_operations"]:
            rotation = np.asarray(
                operation["rotation_frac"], dtype=float
            )
            translation = np.asarray(
                operation["translation_frac"], dtype=float
            )
            operation["rotation_frac"] = (
                inverse @ rotation @ transform
            ).tolist()
            operation["translation_frac"] = (
                inverse @ translation
            ).tolist()
    elif setting_field == "origin_shift_fractional":
        origin = np.asarray(forged_value, dtype=float)
        for operation in source_context["parent_affine_operations"]:
            rotation = np.asarray(
                operation["rotation_frac"], dtype=float
            )
            translation = np.asarray(
                operation["translation_frac"], dtype=float
            )
            operation["translation_frac"] = (
                translation - origin + rotation @ origin
            ).tolist()
    source_context["context_identity"] = canonical_identity({
        key: deepcopy(value)
        for key, value in source_context.items()
        if key not in {"status", "context_identity", "blockers"}
    })
    forged["certificate_identity"][bundle_field] = forged_value
    if setting_field == "hall_symbol":
        forged["certificate_identity"]["hall_symbols"] = [forged_value]
    for records in forged[
        "unitary_irrep_completion_records_by_hsp"
    ].values():
        for record in records:
            source_irrep = record["source_candidate_provenance"][
                "irrep_source_provenance"
            ]
            local_setting = source_irrep["standard_setting_hsp_mapping"][
                "standard_setting_certificate"
            ]
            local_setting[setting_field] = forged_value
            certificate = record.get(
                "tr_irrep_completion_certificate"
            )
            if not isinstance(certificate, dict):
                continue
            certificate["standard_setting_certificate_identity"] = (
                canonical_identity(local_setting)
            )
            certificate["reviewed_time_reversal"][
                "source_context_identity"
            ] = source_context["context_identity"]
            certificate["certificate_identity"] = canonical_identity({
                key: deepcopy(value)
                for key, value in certificate.items()
                if key != "certificate_identity"
            })

    assert forged["certificate_identity"][bundle_field] == forged_value
    assert not validate_tr_completed_unitary_bundle(forged)


def test_portable_problem_export_rederive_raw_source_pairing(
    tmp_path,
):
    inputs = _portable_orbit_inputs(tmp_path)
    report = _build_portable_orbit_report(inputs)
    completed, _, _ = _complete_and_export_portable_orbit(inputs, report)
    forged = deepcopy(completed)
    orbit = forged["valley_orbits"][0]
    pairing = dict(orbit["time_reversal_irrep_pairing"])
    pairing.update({
        "-K4": "-KA5",
        "-KA5": "-K4",
        "-K5": "-KA4",
        "-KA4": "-K5",
    })
    orbit["time_reversal_irrep_pairing"] = pairing
    source_identity = dict(
        orbit["reviewed_time_reversal_source_identity"]
    )
    source_identity["irrep_pairing"] = pairing
    source_identity["identity"] = canonical_identity({
        key: value
        for key, value in source_identity.items()
        if key != "identity"
    })
    orbit["reviewed_time_reversal_source_identity"] = source_identity

    rederived_source = validate_reviewed_time_reversal_source_context(
        orbit["reviewed_time_reversal_source_context"]
    )
    assert rederived_source["irrep_partner_by_label"]["-K4"] == "-KA4"
    assert pairing["-K4"] == "-KA5"

    inferred_records = []
    for valley in orbit[
        "time_reversal_completed_unitary_valley_irreps"
    ]:
        orbit["time_reversal_completed_unitary_valley_irreps"][valley][
            "KA"
        ] = {"-KA5": 1}
        inferred = orbit[
            "unitary_valley_irrep_completion_records"
        ][valley]["KA"][0]
        inferred["irrep"] = "-KA5"
        inferred["reviewed_time_reversal_relation"][
            "target_irrep"
        ] = "-KA5"
        certificate = inferred["tr_irrep_completion_certificate"]
        certificate["inferred_target"]["irrep"] = "-KA5"
        certificate["reviewed_time_reversal"]["irrep_pairing"] = pairing
        certificate["reviewed_time_reversal"][
            "source_model_identity"
        ] = source_identity
        certificate["certificate_identity"] = canonical_identity({
            key: value
            for key, value in certificate.items()
            if key != "certificate_identity"
        })
        inferred_records.append(inferred)

    assert all(
        not validate_tr_irrep_completion_certificate(
            record["tr_irrep_completion_certificate"],
            completion_record=record,
            valley_mapping=forged["time_reversal_valley_mapping"],
            hsp_mapping={"K": "KA", "KA": "K"},
            irrep_pairing=pairing,
            reviewed_source_identity=source_identity,
            reviewed_source_context=orbit[
                "reviewed_time_reversal_source_context"
            ],
        )
        for record in inferred_records
    )
    problems = build_ebr_problem_instances(
        ebr_input_candidates={
            "status": "has_candidates",
            "candidate_count": 2,
            "candidates": inputs["candidates"],
        },
        time_reversal_orbit_report=forged,
    )
    assert sum(
        instance["canonical_hsp_vector_ready"]
        for instance in problems["instances"]
        if instance["problem_kind"] == "unitary_valley_reduced_ebr"
    ) == 0
    assert build_ebr_export_bundle(
        ebr_problem_instances=problems
    )["bundle_count"] == 0


def test_portable_completion_validator_rejects_malformed_inputs(
    tmp_path,
):
    inputs = _portable_orbit_inputs(tmp_path)
    completed = attach_tr_irrep_completion_certificates(
        time_reversal_orbit_report=_build_portable_orbit_report(inputs),
        cprime_validation_context=inputs["context"],
    )
    orbit = completed["valley_orbits"][0]
    record = orbit[
        "unitary_valley_irrep_completion_records"
    ]["left"]["KA"][0]
    kwargs = {
        "certificate": record["tr_irrep_completion_certificate"],
        "completion_record": record,
        "valley_mapping": completed["time_reversal_valley_mapping"],
        "hsp_mapping": {"K": "KA", "KA": "K"},
        "irrep_pairing": orbit["time_reversal_irrep_pairing"],
        "reviewed_source_identity": orbit[
            "reviewed_time_reversal_source_identity"
        ],
        "reviewed_source_context": orbit[
            "reviewed_time_reversal_source_context"
        ],
    }
    assert validate_tr_irrep_completion_certificate(**kwargs)
    for field in (
        "completion_record",
        "valley_mapping",
        "hsp_mapping",
        "irrep_pairing",
        "cprime_validation_context",
    ):
        malformed = dict(kwargs)
        malformed[field] = None if field != "cprime_validation_context" else []
        assert not validate_tr_irrep_completion_certificate(**malformed)


def test_portable_promotion_rederives_raw_source_against_coordinated_substitution(
    tmp_path,
):
    inputs = _portable_orbit_inputs(tmp_path)
    report = _build_portable_orbit_report(inputs)
    _, _, export = _complete_and_export_portable_orbit(inputs, report)
    forged = deepcopy(export["bundles"][0])
    time_reversal = forged["time_reversal"]
    pairing = dict(time_reversal["time_reversal_irrep_pairing"])
    pairing.update({
        "-K4": "-KA5",
        "-KA5": "-K4",
        "-K5": "-KA4",
        "-KA4": "-K5",
    })
    time_reversal["time_reversal_irrep_pairing"] = pairing
    source_identity = dict(
        time_reversal["reviewed_time_reversal_source_identity"]
    )
    source_identity["irrep_pairing"] = pairing
    source_identity["identity"] = canonical_identity({
        key: value
        for key, value in source_identity.items()
        if key != "identity"
    })
    time_reversal["reviewed_time_reversal_source_identity"] = source_identity
    forged["irreps_by_kpoint"]["KA"] = ["-KA5"]
    inferred = forged["unitary_irrep_completion_records_by_hsp"]["KA"][0]
    inferred["irrep"] = "-KA5"
    inferred["reviewed_time_reversal_relation"]["target_irrep"] = "-KA5"
    certificate = inferred["tr_irrep_completion_certificate"]
    certificate["inferred_target"]["irrep"] = "-KA5"
    certificate["reviewed_time_reversal"]["irrep_pairing"] = pairing
    certificate["reviewed_time_reversal"][
        "source_model_identity"
    ] = source_identity
    certificate["certificate_identity"] = canonical_identity({
        key: value
        for key, value in certificate.items()
        if key != "certificate_identity"
    })

    assert not validate_tr_completed_unitary_bundle(forged)
    forged_export = deepcopy(export)
    forged_export["bundles"] = [forged]
    forged_export["bundle_count"] = 1
    reduced_table = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=143,
        spinor=True,
        bundle_irreps_by_kpoint=forged["irreps_by_kpoint"],
        expected_hsps=forged["expected_hsps"],
        subspace_group_candidate="P3",
    )
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=forged_export,
        table=reduced_table,
        reduced_ebr_input={"source": "portable_irreptables_runtime"},
        cprime_validation_context=inputs["context"],
    )
    assert mapping["solutions"] == []


def test_unitary_problem_builder_cannot_bypass_orbit_source_blocker(
    tmp_path,
):
    inputs = _portable_orbit_inputs(tmp_path)
    report = _build_portable_orbit_report(inputs)
    completed, _, _ = _complete_and_export_portable_orbit(inputs, report)
    forged = deepcopy(completed)
    orbit = forged["valley_orbits"][0]
    orbit["unitary_completion_status"] = "blocked"
    orbit["unitary_completion_blockers"] = [
        "valley_source_time_reversal_models_disagree"
    ]
    assert any(
        "tr_irrep_completion_certificate" in record
        for by_hsp in orbit[
            "unitary_valley_irrep_completion_records"
        ].values()
        for records in by_hsp.values()
        for record in records
        if record["completion_kind"] == "inferred_by_time_reversal"
    )

    problems = build_ebr_problem_instances(
        ebr_input_candidates={
            "status": "has_candidates",
            "candidate_count": 2,
            "candidates": inputs["candidates"],
        },
        time_reversal_orbit_report=forged,
    )
    assert not any(
        instance["canonical_hsp_vector_ready"]
        for instance in problems["instances"]
        if instance["problem_kind"] == "unitary_valley_reduced_ebr"
    )
