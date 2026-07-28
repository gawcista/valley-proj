"""Tracked-only producer-chain regression for exact TR irrep completion."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np

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
from valleyscope.analysis.tr_irrep_completion import (
    attach_tr_irrep_completion_certificates,
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
            "target_coefficients": np.eye(2, dtype=np.complex128),
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
            "source_table_name": "irreptables.StandardIrrepTable",
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


def _identity(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "source": candidate["source"],
        "workflow_path": candidate["workflow_path"],
        "valley": candidate["valley"],
        "source_hsp_label": "K",
        "sampled_kpoint": candidate["kpoint"],
        "irrep": candidate["matched_irrep"],
        "multiplicity": 1,
    }


def _completion_record(
    *,
    target_valley: str,
    target_hsp: str,
    target_irrep: str,
    source: dict[str, object],
    inferred: bool,
) -> dict[str, object]:
    record = {
        "completion_kind": (
            "inferred_by_time_reversal"
            if inferred
            else "observed_at_sampled_kpoint"
        ),
        "target_valley": target_valley,
        "target_source_hsp_label": target_hsp,
        "irrep": target_irrep,
        "multiplicity": 1,
        "evidence_valley": source["valley"],
        "evidence_source_hsp_label": "K",
        "evidence_sampled_kpoint": source["kpoint"],
        "source_candidate_identity": _identity(source),
        "source_candidate_provenance": {
            "source": source["source"],
            "workflow_path": source["workflow_path"],
            "irrep_source_provenance": deepcopy(
                source["irrep_source_provenance"]
            ),
        },
        "structural_status": "validated",
        "readiness_status": "trusted",
        "blockers": [],
    }
    if inferred:
        record["reviewed_time_reversal_relation"] = {
            "evidence_valley": source["valley"],
            "target_valley": target_valley,
            "evidence_source_hsp_label": "K",
            "target_source_hsp_label": target_hsp,
            "evidence_irrep": source["matched_irrep"],
            "target_irrep": target_irrep,
        }
    else:
        record["sampled_kpoint"] = source["kpoint"]
    return record


def test_tracked_only_exact_completion_uses_real_irreptables_source(
    tmp_path,
):
    table, context, cprime, standard_certificate = _producer_contexts(
        tmp_path
    )
    ebr_source = load_ebr_source_data(143, True)
    reviewed = resolve_ebr_source_irrep_label_evidence(
        table=table,
        source_basis_labels=ebr_source["source_basis_labels"],
    )
    tr_source = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=reviewed["reviewed_rows"],
        centering_vectors=[[0.0, 0.0, 0.0]],
    )
    assert reviewed["status"] == tr_source["status"] == "validated"
    assert tr_source["irrep_partner_by_label"]["-K4"] == "-KA4"
    reviewed_source_content = {
        "operation_inventory_identity": tr_source[
            "operation_inventory_identity"
        ],
        "spin_convention": tr_source["spin_convention"],
        "hsp_involution": tr_source["time_reversal_hsp_mapping"],
        "irrep_pairing": tr_source["irrep_partner_by_label"],
    }
    reviewed_source_identity = {
        **reviewed_source_content,
        "identity": canonical_identity(reviewed_source_content),
    }

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
    records = {
        "left": {
            "K": [_completion_record(
                target_valley="left",
                target_hsp="K",
                target_irrep="-K4",
                source=left,
                inferred=False,
            )],
            "KA": [_completion_record(
                target_valley="left",
                target_hsp="KA",
                target_irrep="-KA4",
                source=right,
                inferred=True,
            )],
        },
        "right": {
            "K": [_completion_record(
                target_valley="right",
                target_hsp="K",
                target_irrep="-K4",
                source=right,
                inferred=False,
            )],
            "KA": [_completion_record(
                target_valley="right",
                target_hsp="KA",
                target_irrep="-KA4",
                source=left,
                inferred=True,
            )],
        },
    }
    report = {
        "enabled": True,
        "status": "validated",
        "theta_square": -1,
        "time_reversal_valley_mapping": {
            "left": "right",
            "right": "left",
        },
        "valley_orbits": [{
            "orbit_id": "portable_orbit",
            "representative": "left",
            "members": ["left", "right"],
            "mapping_type": "exchanged",
            "status": "validated",
            "blockers": [],
            "expected_hsps": ["K"],
            "irreps_by_kpoint": {"K": ["portable_corep"]},
            "full_unitary_source_hsp_labels": ["K", "KA"],
            "independent_time_reversal_hsp_labels": ["K"],
            "time_reversal_hsp_orbits": [{
                "representative": "K",
                "members": ["K", "KA"],
                "self_mapped": False,
            }],
            "time_reversal_irrep_pairing": tr_source[
                "irrep_partner_by_label"
            ],
            "reviewed_time_reversal_source_identity": (
                reviewed_source_identity
            ),
            "time_reversal_completed_unitary_valley_irreps": {
                "left": {"K": {"-K4": 1}, "KA": {"-KA4": 1}},
                "right": {"K": {"-K4": 1}, "KA": {"-KA4": 1}},
            },
            "unitary_valley_irrep_completion_records": records,
            "independent_source_hsp_to_sampled_kpoint_by_valley": {
                "left": {"K": "K_left"},
                "right": {"K": "K_right"},
            },
            "observed_source_hsp_to_sampled_kpoint_by_valley": {
                "left": {"K": "K_left"},
                "right": {"K": "K_right"},
            },
        }],
    }
    completed = attach_tr_irrep_completion_certificates(
        time_reversal_orbit_report=report,
        cprime_validation_context=context,
    )
    assert completed["status"] == "validated", completed
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
