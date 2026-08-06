"""Shared portable producer-chain construction for tracked tests and the
installed-artifact release gate.

The source suite (`tests/test_portable_production_cprime_chain.py`) and the
installed-wheel acceptance (`scripts/release_gate_installed_check.py`) exercise
the same minimal genuine producer path, so the gate never duplicates fixture
construction.  Imports stay limited to the installed `valleyscope` package
plus this tests package; nothing here may reference the repository root.
"""

from __future__ import annotations

import tempfile
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
from valleyscope.analysis.time_reversal_orbits import (
    build_time_reversal_valley_orbit_report,
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

SOURCE_SPACE_GROUP = 143
SOURCE_TABLE_NAME = "P3"


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

    table = load_standard_irrep_table(SOURCE_SPACE_GROUP, spinor=True)
    standard_certificate = real_primitive_certificate_dict(
        SOURCE_SPACE_GROUP, SOURCE_TABLE_NAME, spinor=True
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
        "subspace_group_candidate": SOURCE_TABLE_NAME,
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_number": SOURCE_SPACE_GROUP,
            "candidate_space_group_symbol": SOURCE_TABLE_NAME,
        },
        "irrep_source_provenance": {
            "matching_strategy": "bilbao_restricted_character",
            "subspace_space_group_number": SOURCE_SPACE_GROUP,
            "subspace_space_group_symbol": SOURCE_TABLE_NAME,
            "source_table_sg_number": SOURCE_SPACE_GROUP,
            "source_table_name": SOURCE_TABLE_NAME,
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
    ebr_source = load_ebr_source_data(SOURCE_SPACE_GROUP, True)
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


def _portable_reduced_table(first_bundle):
    return build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=SOURCE_SPACE_GROUP,
        spinor=True,
        bundle_irreps_by_kpoint=first_bundle["irreps_by_kpoint"],
        expected_hsps=first_bundle["expected_hsps"],
        subspace_group_candidate=SOURCE_TABLE_NAME,
    )


def _cprime_acceptance_matrix(export):
    """Build the scope-keyed summary acceptance matrix from an export."""
    matrix = []
    seen_scopes: set[tuple[str, str]] = set()
    for bundle in export["bundles"]:
        for hsp, links in bundle["cprime_identity_by_kpoint"].items():
            scope = bundle["cprime_scope_metadata"][hsp]
            scope_key = (scope["sampled_kpoint"], scope["evidence_valley"])
            if scope_key in seen_scopes:
                continue
            seen_scopes.add(scope_key)
            matrix.append({
                "kpoint": scope["sampled_kpoint"],
                "valley": scope["evidence_valley"],
                "double_space_group_lift_status": "passed",
                "double_space_group_lift_identity": links[
                    "double_space_group_lift_certificate_identity"
                ],
                "scoped_representation_status": "passed",
                "scoped_representation_evidence_identity": links[
                    "scoped_representation_evidence_identity"
                ],
            })
    return matrix


def run_installed_portable_acceptance(
    workdir: Path | None = None,
) -> dict[str, object]:
    """Run the full portable production chain and return its evidence.

    Raises AssertionError with details when any physical step fails.  The
    returned summary is JSON-serializable for the release gate report.  A
    caller-supplied workdir is kept; otherwise a temporary directory is
    created and removed on exit.
    """
    if workdir is not None:
        return _run_portable_acceptance_in(Path(workdir))
    with tempfile.TemporaryDirectory(prefix="valleyscope_acceptance_") as tmp:
        return _run_portable_acceptance_in(Path(tmp))


def _run_portable_acceptance_in(root: Path) -> dict[str, object]:
    inputs = _portable_orbit_inputs(root)
    report = _build_portable_orbit_report(inputs)
    completed, problems, export = _complete_and_export_portable_orbit(
        inputs, report
    )
    assert completed["valley_orbits"][0][
        "unitary_completion_status"
    ] == "validated"
    assert export["bundle_count"] == 2, export["bundle_count"]
    assert all(
        validate_tr_completed_unitary_bundle(bundle)
        for bundle in export["bundles"]
    )
    first_bundle = export["bundles"][0]
    reduced_table = _portable_reduced_table(first_bundle)
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=export,
        table=reduced_table,
        reduced_ebr_input={"source": "portable_irreptables_runtime"},
        cprime_validation_context=inputs["context"],
    )
    assert len(mapping["solutions"]) == 2, mapping["solutions"]
    source_basis_identity = next(iter(
        first_bundle["cprime_identity_by_kpoint"].values()
    ))["spinor_source_basis_certificate_identity"]
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
                "acceptance_matrix": _cprime_acceptance_matrix(export),
            },
        },
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )
    assert ingestion["validation_errors"] == [], ingestion
    assert ingestion["final_reduced_ebr_result_count"] == 2, ingestion
    return {
        "source_space_group": SOURCE_SPACE_GROUP,
        "source_table_name": SOURCE_TABLE_NAME,
        "spinor": True,
        "bundles_ready": export["bundle_count"],
        "mapping_solution_count": len(mapping["solutions"]),
        "database_irrep_record_count": len(ingestion["valley_irrep_records"]),
        "final_reduced_ebr_result_count": ingestion[
            "final_reduced_ebr_result_count"
        ],
        "validation_errors": [],
    }
