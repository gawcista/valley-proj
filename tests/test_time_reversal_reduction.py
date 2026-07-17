from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
from valleyscope.analysis.irreptables_runtime_table_builder import (
    build_auto_time_reversal_reduced_ebr_table,
)
from valleyscope.analysis.reduced_ebr_mapping import (
    _joint_bundle_time_reversal_evidence_valid,
    promote_bundle_for_solve,
)

from valleyscope.analysis.time_reversal_orbits import (
    _decompose_grey_counts,
    build_time_reversal_valley_orbit_report,
    derive_time_reversal_valley_mapping,
    validate_time_reversal_valley_mapping,
)
from valleyscope.analysis.time_reversal_sewing import (
    build_time_reversal_sewing_report,
)
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.irreps.tables import (
    StandardIrrep,
    StandardIrrepTable,
    StandardTableOperation,
    resolve_ebr_source_irrep_label_evidence,
    load_standard_irrep_table,
)
from valleyscope.irreps.ebr_data_adapter import load_ebr_source_data
from valleyscope.irreps.time_reversal_ebr import (
    validate_grey_group_time_reversal_source,
)
from valleyscope.irreps.time_reversal_source import (
    derive_time_reversal_source_irrep_orbits,
)
from tests.reduced_ebr_promo_helpers import attach_real_certificate


def _operation(index: int, rotation: list[list[int]]) -> StandardTableOperation:
    return StandardTableOperation(
        table_index=index,
        rotation_frac=np.asarray(rotation, dtype=int),
        translation_frac=np.zeros(3),
        spin_rotation=np.eye(2, dtype=complex),
        time_reversal=False,
    )


def _irrep(
    label: str,
    hsp: str,
    k_frac: list[float],
    characters: dict[int, complex],
) -> StandardIrrep:
    return StandardIrrep(
        label=label,
        kpoint_label=hsp,
        k_frac=np.asarray(k_frac, dtype=float),
        dimension=1,
        characters=characters,
    )


def _reviewed_rows(table: StandardIrrepTable, labels: list[str]):
    evidence = resolve_ebr_source_irrep_label_evidence(
        table=table,
        source_basis_labels=labels,
    )
    assert evidence["status"] == "validated"
    return evidence["reviewed_rows"]


def test_primitive_hsp_and_scalar_irrep_time_reversal_orbits_use_characters():
    identity = _operation(1, np.eye(3, dtype=int).tolist())
    generator = _operation(2, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    table = StandardIrrepTable(
        number=75,
        name="P4",
        spinor=False,
        operations=(identity, generator),
        irreps=(
            _irrep("Q1", "Q", [0.25, 0.0, 0.0], {1: 1, 2: 1j}),
            _irrep("QA1", "QA", [-0.25, 0.0, 0.0], {1: 1, 2: -1j}),
        ),
    )

    report = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=_reviewed_rows(table, ["Q1", "QA1"]),
        centering_vectors=[[0.0, 0.0, 0.0]],
    )

    assert report["status"] == "validated"
    assert report["independent_hsp_labels"] == ["Q"]
    assert report["time_reversal_hsp_orbits"] == [{
        "representative": "Q",
        "members": ["Q", "QA"],
        "self_mapped": False,
    }]
    assert report["irrep_partner_by_label"] == {
        "Q1": "QA1",
        "QA1": "Q1",
    }


def test_centered_trim_orbit_uses_certified_reciprocal_lattice():
    table = StandardIrrepTable(
        number=5,
        name="C2",
        spinor=True,
        operations=(_operation(1, np.eye(3, dtype=int).tolist()),),
        irreps=(
            _irrep("-V2", "V", [0.5, 0.5, 0.0], {1: 1}),
        ),
    )

    report = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=_reviewed_rows(table, ["-V2"]),
        centering_vectors=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
    )

    assert report["status"] == "validated"
    assert report["time_reversal_hsp_orbits"] == [{
        "representative": "V",
        "members": ["V"],
        "self_mapped": True,
    }]
    assert report["irrep_partner_by_label"] == {"-V2": "-V2"}


def _center(name: str, q: float, *, layer: str = "top") -> ValleyCenter:
    return ValleyCenter(
        name=name,
        cart=np.asarray([q, 0.0, 0.0]),
        layer=layer,
        reciprocal_cart=np.eye(3),
    )


def test_valley_mapping_is_center_derived_bijective_and_involutive():
    report = derive_time_reversal_valley_mapping(
        enabled=True,
        centers=[_center("c1", 0.25), _center("c2", -0.25)],
        valley_subspaces=[
            ValleySector("left", ["c1"]),
            ValleySector("right", ["c2"]),
        ],
        spinor=True,
    )

    assert report["status"] == "validated"
    assert report["theta_square"] == -1
    assert report["time_reversal_valley_mapping"] == {
        "left": "right",
        "right": "left",
    }
    assert report["valley_orbits"] == [{
        "representative": "left",
        "members": ["left", "right"],
        "mapping_type": "exchanged",
    }]


def test_self_mapped_valley_is_explicit_corepresentation_case():
    report = derive_time_reversal_valley_mapping(
        enabled=True,
        centers=[_center("m", 0.5)],
        valley_subspaces=[ValleySector("m_valley", ["m"])],
        spinor=False,
    )

    assert report["status"] == "validated"
    assert report["theta_square"] == 1
    assert report["valley_orbits"][0]["mapping_type"] == "self_mapped"
    assert report["valley_orbits"][0]["antiunitary_corepresentation_status"] == (
        "required_not_proven"
    )


def test_ambiguous_center_partner_blocks_time_reversal_mapping():
    report = derive_time_reversal_valley_mapping(
        enabled=True,
        centers=[
            _center("source", 0.25),
            _center("partner_a", -0.25),
            _center("partner_b", -0.25),
        ],
        valley_subspaces=[
            ValleySector("source_valley", ["source"]),
            ValleySector("partner_valley", ["partner_a", "partner_b"]),
        ],
        spinor=False,
    )

    assert report["status"] == "blocked"
    assert "ambiguous_time_reversal_center_partner" in report["blockers"][0]


def test_non_involutive_explicit_mapping_validation_fails_closed():
    validation = validate_time_reversal_valley_mapping(
        mapping={"a": "b", "b": "c", "c": "a"},
        valley_names=["a", "b", "c"],
    )

    assert validation["status"] == "blocked"
    assert "non_involutive_time_reversal_valley_mapping" in validation[
        "blockers"
    ]


def test_spinful_sg143_grey_source_proves_unitary_pair_closure():
    table = load_standard_irrep_table(143, spinor=True)
    source = load_ebr_source_data(143, True)
    rows = _reviewed_rows(table, source["source_basis_labels"])
    orbits = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=rows,
        centering_vectors=[[0.0, 0.0, 0.0]],
    )

    report = validate_grey_group_time_reversal_source(
        unitary_table=table,
        reviewed_rows=rows,
        unitary_source_data=source,
        irrep_partner_by_label=orbits["irrep_partner_by_label"],
        centering_vectors=[[0.0, 0.0, 0.0]],
    )

    assert report["status"] == "validated"
    assert report["grey_bns_number"] == "143.2"
    assert report["grey_unitary_restriction_case_by_irrep"]["-GM4GM4"] == (
        "quaternionic"
    )
    assert report["grey_unitary_restriction_by_irrep"]["-GM4GM4"] == {
        "-GM4": 2,
    }


def test_scalar_real_grey_source_restricts_once_without_column_doubling():
    table = load_standard_irrep_table(143, spinor=False)
    source = load_ebr_source_data(143, False)
    rows = _reviewed_rows(table, source["source_basis_labels"])
    orbits = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=rows,
        centering_vectors=[[0.0, 0.0, 0.0]],
    )
    report = validate_grey_group_time_reversal_source(
        unitary_table=table,
        reviewed_rows=rows,
        unitary_source_data=source,
        irrep_partner_by_label=orbits["irrep_partner_by_label"],
        centering_vectors=[[0.0, 0.0, 0.0]],
    )

    assert report["status"] == "validated"
    assert report["grey_unitary_restriction_by_irrep"]["GM1"] == {"GM1": 1}
    assert report["grey_unitary_restriction_case_by_irrep"]["GM1"] == "real"


def test_scalar_complex_grey_source_restricts_to_conjugate_pair():
    table = load_standard_irrep_table(143, spinor=False)
    source = load_ebr_source_data(143, False)
    rows = _reviewed_rows(table, source["source_basis_labels"])
    orbits = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=rows,
        centering_vectors=[[0.0, 0.0, 0.0]],
    )

    report = validate_grey_group_time_reversal_source(
        unitary_table=table,
        reviewed_rows=rows,
        unitary_source_data=source,
        irrep_partner_by_label=orbits["irrep_partner_by_label"],
        centering_vectors=[[0.0, 0.0, 0.0]],
    )

    assert report["status"] == "validated"
    assert report["grey_unitary_restriction_by_irrep"]["GM2GM3"] == {
        "GM2": 1,
        "GM3": 1,
    }
    assert report["grey_unitary_restriction_case_by_irrep"]["GM2GM3"] == (
        "complex_paired"
    )


def test_grey_source_rejects_nonbijective_unitary_irrep_involution():
    table = load_standard_irrep_table(143, spinor=False)
    source = load_ebr_source_data(143, False)
    rows = _reviewed_rows(table, source["source_basis_labels"])
    orbits = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=rows,
        centering_vectors=[[0.0, 0.0, 0.0]],
    )
    corrupted = dict(orbits["irrep_partner_by_label"])
    corrupted["H1"] = "GM1"

    report = validate_grey_group_time_reversal_source(
        unitary_table=table,
        reviewed_rows=rows,
        unitary_source_data=source,
        irrep_partner_by_label=corrupted,
        centering_vectors=[[0.0, 0.0, 0.0]],
    )

    assert report["status"] == "blocked"
    assert "incomplete_or_nonbijective_time_reversal_irrep_row_mapping" in (
        report["blockers"]
    )


def _reviewed_joint_bundle_and_table():
    table = build_auto_time_reversal_reduced_ebr_table(
        unitary_space_group_number=143,
        grey_bns_number="143.2",
        spinor=True,
        bundle_irreps_by_kpoint={"GM": ["-GM4GM4"]},
        expected_hsps=["GM"],
        subspace_group_candidate="P3",
        subspace_space_group={
            "status": "resolved",
            "candidate_space_group_number": 143,
            "candidate_space_group_symbol": "P3",
        },
    )
    bundle = {
        "bundle_id": "b_tr",
        "problem_kind": "valley_orbit_reduced_ebr",
        "valley": "",
        "valley_orbit": ["left", "right"],
        "unitary_valley_irreps": {
            "left": {"GM": {"-GM4": 1}},
            "right": {"GM": {"-GM4": 1}},
        },
        "time_reversal": {
            "theta_square": -1,
            "time_reversal_valley_mapping": {
                "left": "right", "right": "left",
            },
            "time_reversal_hsp_orbits": [{
                "representative": "GM",
                "members": ["GM"],
                "self_mapped": True,
            }],
            "full_unitary_source_hsp_labels": ["GM"],
            "time_reversal_irrep_pairing": {"-GM4": "-GM4"},
            "grey_bns_number": "143.2",
        },
        "subspace_group_candidate": "P3",
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_number": 143,
            "candidate_space_group_symbol": "P3",
        },
        "subspace_sg_number": 143,
        "spinor": True,
        "expected_hsps": ["GM"],
        "irreps_by_kpoint": {"GM": ["-GM4GM4"]},
        "source_hsp_to_sampled_kpoint": {"GM": "GM"},
                "ready_for_reduced_table_validation": True,
    }
    export = {"bundles": [bundle]}
    assert attach_real_certificate(export, table) is not None
    return export["bundles"][0], table


def test_scalar_grey_reduced_table_uses_authoritative_columns_directly():
    table = build_auto_time_reversal_reduced_ebr_table(
        unitary_space_group_number=143,
        grey_bns_number="143.2",
        spinor=False,
        bundle_irreps_by_kpoint={"GM": ["GM1"]},
        expected_hsps=["GM"],
        subspace_group_candidate="P3",
        subspace_space_group={
            "status": "resolved",
            "candidate_space_group_number": 143,
            "candidate_space_group_symbol": "P3",
        },
    )

    assert table["irreps"] == ["GM:GM1", "GM:GM2GM3"]
    scalar_real_columns = [
        row["vector"] for row in table["ebrs"]
        if str(row["label"]).startswith("A1")
    ]
    assert scalar_real_columns == [[1, 0], [1, 0], [1, 0]]


def _blocker_codes(promotion):
    return {row["code"] for row in promotion["blocker_reasons"]}


def _projector_provenance_from_sewing(report):
    return {
        row["source_kpoint"]: {
            valley: dict(entry["source_projector_provenance"])
            for valley, entry in row["projector_covariance"].items()
        }
        for row in report["rows"]
    }


def test_joint_problem_promotion_requires_matching_type_ii_grey_provenance():
    bundle, table = _reviewed_joint_bundle_and_table()

    assert promote_bundle_for_solve(bundle=bundle, table=table)["promoted"]

    missing = dict(table)
    missing["provenance"] = dict(table["provenance"])
    missing["provenance"].pop("time_reversal_source")
    missing["provenance"].pop("time_reversal_grey_bns_number")
    promotion = promote_bundle_for_solve(bundle=bundle, table=missing)
    assert promotion["promoted"] is False
    assert "time_reversal_table_provenance_missing" in _blocker_codes(
        promotion
    )

    wrong = dict(table)
    wrong["provenance"] = dict(table["provenance"])
    wrong["provenance"]["time_reversal_grey_bns_number"] = "142.2"
    promotion = promote_bundle_for_solve(bundle=bundle, table=wrong)
    assert promotion["promoted"] is False
    assert "time_reversal_grey_bns_mismatch" in _blocker_codes(promotion)


def test_problem_kind_compatibility_rejects_grey_table_for_unitary_bundle():
    bundle, table = _reviewed_joint_bundle_and_table()
    unitary = dict(bundle)
    unitary.update({
        "problem_kind": "unitary_valley_reduced_ebr",
        "valley": "left",
        "valley_orbit": [],
    })
    unitary.pop("time_reversal")
    unitary.pop("unitary_valley_irreps")

    promotion = promote_bundle_for_solve(bundle=unitary, table=table)

    assert promotion["promoted"] is False
    assert "unitary_problem_rejects_grey_table" in _blocker_codes(promotion)


def test_joint_problem_requires_complete_bundle_time_reversal_evidence():
    bundle, table = _reviewed_joint_bundle_and_table()
    incomplete = dict(bundle)
    incomplete["time_reversal"] = dict(bundle["time_reversal"])
    incomplete["time_reversal"].pop("grey_bns_number")

    promotion = promote_bundle_for_solve(bundle=incomplete, table=table)

    assert promotion["promoted"] is False
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promotion
    )


def test_joint_problem_rejects_incomplete_hsp_and_irrep_involutions():
    bundle, table = _reviewed_joint_bundle_and_table()
    malformed_hsp = dict(bundle)
    malformed_hsp["time_reversal"] = dict(bundle["time_reversal"])
    malformed_hsp["time_reversal"]["time_reversal_hsp_orbits"] = [{
        "representative": "GM",
        "members": ["GM", "GM"],
        "self_mapped": False,
    }]
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promote_bundle_for_solve(bundle=malformed_hsp, table=table)
    )

    malformed_irrep = dict(bundle)
    malformed_irrep["time_reversal"] = dict(bundle["time_reversal"])
    malformed_irrep["time_reversal"]["time_reversal_irrep_pairing"] = {
        "-GM4": "missing",
    }
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promote_bundle_for_solve(bundle=malformed_irrep, table=table)
    )


def test_joint_problem_rejects_component_hsp_outside_declared_inventory():
    bundle, table = _reviewed_joint_bundle_and_table()
    malformed = deepcopy(bundle)
    malformed["unitary_valley_irreps"]["left"] = {
        "X": {"-GM4": 1},
    }

    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promote_bundle_for_solve(bundle=malformed, table=table)
    )


def test_self_mapped_joint_problem_requires_serialized_sewing_evidence():
    bundle, table = _reviewed_joint_bundle_and_table()
    self_mapped = dict(bundle)
    self_mapped["valley_orbit"] = ["v"]
    self_mapped["unitary_valley_irreps"] = {
        "v": {"GM": {"-GM4": 2}},
    }
    self_mapped["time_reversal"] = dict(bundle["time_reversal"])
    self_mapped["time_reversal"]["time_reversal_valley_mapping"] = {"v": "v"}
    coefficients = np.asarray([
        [[1.0 + 0.0j], [0.0 + 0.0j]],
        [[0.0 + 0.0j], [1.0 + 0.0j]],
    ])
    sewing = build_time_reversal_sewing_report(
        kpoint_frac_by_name={"GM": np.zeros(3)},
        g_vectors_frac_by_kpoint={"GM": np.zeros((1, 3), dtype=int)},
        coefficients_by_kpoint={"GM": coefficients},
        band_indices_by_kpoint={"GM": np.asarray([1, 2])},
        valley_projectors_by_kpoint={"GM": {"v": np.eye(2)}},
        valley_projector_provenance_by_kpoint={
            "GM": {"v": {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }},
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "v"},
        spinor=True,
        spinor_convention_verified=True,
    )
    self_mapped["time_reversal"]["antiunitary_sewing_evidence"] = sewing
    self_mapped["time_reversal"][
        "projector_workflow_by_sampled_kpoint"
    ] = {"GM": {"v": "direct_qcut"}}
    self_mapped["time_reversal"][
        "projector_provenance_by_sampled_kpoint"
    ] = _projector_provenance_from_sewing(sewing)
    self_mapped["time_reversal"][
        "source_hsp_binding_by_sampled_kpoint"
    ] = {"GM": {"v": {
        "source_hsp_label": "GM",
        "classification": "representative",
        "validation_status": "validated",
        "parent_k_frac": [0.0, 0.0, 0.0],
        "standard_k_frac": [0.0, 0.0, 0.0],
        "source_hsp_representative_k_frac": [0.0, 0.0, 0.0],
        "standard_operation_index": None,
    }}}

    assert promote_bundle_for_solve(bundle=self_mapped, table=table)["promoted"]

    copied_row = deepcopy(self_mapped)
    copied_row["source_hsp_to_sampled_kpoint"] = {"GM": "other"}
    copied_time_reversal = copied_row["time_reversal"]
    copied_time_reversal["projector_workflow_by_sampled_kpoint"] = {
        "other": copied_time_reversal[
            "projector_workflow_by_sampled_kpoint"
        ]["GM"]
    }
    copied_time_reversal["projector_provenance_by_sampled_kpoint"] = {
        "other": copied_time_reversal[
            "projector_provenance_by_sampled_kpoint"
        ]["GM"]
    }
    copied_binding = deepcopy(
        copied_time_reversal["source_hsp_binding_by_sampled_kpoint"]["GM"]
    )
    for binding in copied_binding.values():
        binding["parent_k_frac"] = [0.5, 0.0, 0.0]
        binding["standard_k_frac"] = [0.5, 0.0, 0.0]
        binding["source_hsp_representative_k_frac"] = [0.5, 0.0, 0.0]
    copied_time_reversal["source_hsp_binding_by_sampled_kpoint"] = {
        "other": copied_binding
    }
    copied_sewing = copied_time_reversal["antiunitary_sewing_evidence"]
    copied_sewing["time_reversal_kpoint_mapping"]["other"] = "other"
    copied_sewing["reciprocal_shifts_by_kpoint"]["other"] = [1, 0, 0]
    copied_sewing["sampled_kpoint_frac_by_name"]["other"] = [0.5, 0.0, 0.0]
    copied_sewing_row = deepcopy(copied_sewing["rows"][0])
    copied_sewing_row.update({
        "source_kpoint": "other",
        "target_kpoint": "other",
        "reciprocal_shift": [1, 0, 0],
    })
    copied_sewing["rows"].append(copied_sewing_row)
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promote_bundle_for_solve(bundle=copied_row, table=table)
    )

    for location in (
        "projector_provenance_by_sampled_kpoint",
        "covariance_projector_provenance",
        "source_hsp_binding_by_sampled_kpoint",
    ):
        leaked = deepcopy(self_mapped)
        leaked_time_reversal = leaked["time_reversal"]
        if location == "projector_provenance_by_sampled_kpoint":
            target = leaked_time_reversal[location]["GM"]["v"]
        elif location == "covariance_projector_provenance":
            target = leaked_time_reversal["antiunitary_sewing_evidence"][
                "rows"
            ][0]["projector_covariance"]["v"][
                "source_projector_provenance"
            ]
        else:
            target = leaked_time_reversal[location]["GM"]["v"]
        target["raw_projector"] = [[1.0, 0.0], [0.0, 1.0]]
        assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
            promote_bundle_for_solve(bundle=leaked, table=table)
        )

    scoped = deepcopy(self_mapped)
    scoped_evidence = scoped["time_reversal"]["antiunitary_sewing_evidence"]
    scoped_evidence["status"] = "blocked"
    scoped_evidence["blockers"] = ["unrelated_sample_failed"]
    scoped_evidence["time_reversal_kpoint_mapping"]["other"] = "other"
    scoped_evidence["reciprocal_shifts_by_kpoint"]["other"] = [1, 0, 0]
    unrelated_row = deepcopy(scoped_evidence["rows"][0])
    unrelated_row.update({
        "source_kpoint": "other",
        "target_kpoint": "other",
        "reciprocal_shift": [1, 0, 0],
        "status": "blocked",
        "blockers": ["unrelated_sample_failed"],
        "target_subspace_closure_residual": 9.0,
    })
    scoped_evidence["rows"].append(unrelated_row)
    assert promote_bundle_for_solve(bundle=scoped, table=table)["promoted"]

    substituted_seed = deepcopy(self_mapped)
    substituted_seed["time_reversal"][
        "projector_workflow_by_sampled_kpoint"
    ] = {"GM": {"v": "symmetry_adapted"}}
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promote_bundle_for_solve(bundle=substituted_seed, table=table)
    )

    adapted = deepcopy(substituted_seed)
    covariance = adapted["time_reversal"]["antiunitary_sewing_evidence"][
        "rows"
    ][0]["projector_covariance"]["v"]
    for field in ("source_projector_provenance", "target_projector_provenance"):
        covariance[field] = {
            "workflow_path": "symmetry_adapted",
            "projector_kind": "symmetry_adapted",
        }
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promote_bundle_for_solve(bundle=adapted, table=table)
    )

    adapted_sewing = build_time_reversal_sewing_report(
        kpoint_frac_by_name={"GM": np.zeros(3)},
        g_vectors_frac_by_kpoint={"GM": np.zeros((1, 3), dtype=int)},
        coefficients_by_kpoint={"GM": coefficients},
        band_indices_by_kpoint={"GM": np.asarray([1, 2])},
        valley_projectors_by_kpoint={"GM": {"v": np.zeros((2, 2))}},
        valley_projector_provenance_by_kpoint={
            "GM": {"v": {
                "workflow_path": "symmetry_adapted",
                "projector_kind": "symmetry_adapted",
            }},
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "v"},
        spinor=True,
        spinor_convention_verified=True,
    )
    adapted_exact = deepcopy(self_mapped)
    adapted_exact["time_reversal"][
        "antiunitary_sewing_evidence"
    ] = adapted_sewing
    adapted_exact["time_reversal"][
        "projector_workflow_by_sampled_kpoint"
    ] = {"GM": {"v": "symmetry_adapted"}}
    adapted_exact["time_reversal"][
        "projector_provenance_by_sampled_kpoint"
    ] = _projector_provenance_from_sewing(adapted_sewing)
    assert promote_bundle_for_solve(
        bundle=adapted_exact, table=table
    )["promoted"]

    tampered = deepcopy(adapted_exact)
    tampered["time_reversal"]["antiunitary_sewing_evidence"]["rows"][0][
        "projector_covariance"
    ]["v"]["source_projector_provenance"][
        "projector_fingerprint"
    ] = "sha256:" + "0" * 64
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promote_bundle_for_solve(bundle=tampered, table=table)
    )

    unrelated = deepcopy(self_mapped)
    unrelated["source_hsp_to_sampled_kpoint"] = {"GM": "other"}
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promote_bundle_for_solve(bundle=unrelated, table=table)
    )

    missing = dict(self_mapped)
    missing["time_reversal"] = dict(self_mapped["time_reversal"])
    missing["time_reversal"].pop("antiunitary_sewing_evidence")
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promote_bundle_for_solve(bundle=missing, table=table)
    )

    malformed_counts = dict(bundle)
    malformed_counts["unitary_valley_irreps"] = {
        valley: {
            hsp: dict(counts) for hsp, counts in by_hsp.items()
        }
        for valley, by_hsp in bundle["unitary_valley_irreps"].items()
    }
    malformed_counts["unitary_valley_irreps"]["left"]["GM"]["-GM4"] = 0
    promotion = promote_bundle_for_solve(
        bundle=malformed_counts,
        table=table,
    )

    assert promotion["promoted"] is False
    assert "time_reversal_bundle_evidence_invalid" in _blocker_codes(
        promotion
    )


def _synthetic_source_orbit_report():
    return {
        "status": "validated",
        "time_reversal_hsp_mapping": {
            "G": "G", "Q": "QA", "QA": "Q", "M": "M",
        },
        "time_reversal_hsp_orbits": [
            {"representative": "G", "members": ["G"], "self_mapped": True},
            {"representative": "Q", "members": ["Q", "QA"], "self_mapped": False},
            {"representative": "M", "members": ["M"], "self_mapped": True},
        ],
        "independent_hsp_labels": ["G", "Q", "M"],
        "irrep_partner_by_label": {
            "g": "g", "q1": "qa1", "qa1": "q1",
            "q2": "qa2", "qa2": "q2", "m": "m",
        },
    }


def _synthetic_grey_report():
    return {
        "status": "validated",
        "grey_bns_number": "75.2",
        "grey_unitary_restriction_by_irrep": {
            "g_corep": {"g": 2},
            "q1_corep": {"q1": 1},
            "q2_corep": {"q2": 1},
            "qa1_corep": {"qa1": 1},
            "qa2_corep": {"qa2": 1},
            "m_corep": {"m": 2},
        },
        "grey_source_hsp_by_irrep": {
            "g_corep": "G", "q1_corep": "Q", "q2_corep": "Q",
            "qa1_corep": "QA", "qa2_corep": "QA", "m_corep": "M",
        },
        "unitary_source_hsp_by_irrep": {
            "g": "G", "q1": "Q", "q2": "Q", "qa1": "QA",
            "qa2": "QA", "m": "M",
        },
    }


def _self_mapped_source_report():
    return {
        "status": "validated",
        "time_reversal_hsp_mapping": {"G": "G"},
        "time_reversal_hsp_orbits": [{
            "representative": "G",
            "members": ["G"],
            "self_mapped": True,
        }],
        "independent_hsp_labels": ["G"],
        "irrep_partner_by_label": {"g": "g"},
    }


def _self_mapped_grey_report(*, multiplicity: int):
    return {
        "status": "validated",
        "grey_bns_number": "1.2",
        "grey_unitary_restriction_by_irrep": {
            "g_corep": {"g": multiplicity},
        },
        "grey_source_hsp_by_irrep": {"g_corep": "G"},
        "unitary_source_hsp_by_irrep": {"g": "G"},
    }


def _self_mapped_candidates(*, multiplicity: int):
    return {"candidates": [{
        "valley": "v",
        "matched_irrep": "g",
        "irrep_multiplicity": multiplicity,
        "irrep_source_provenance": {"source_hsp_label": "G"},
        "projected_hsp_classification": {
            "source_hsp_label": "G",
            "classification": "representative",
            "source_hsp_membership": True,
            "validation_status": "validated",
            "parent_k_frac": [0.0, 0.0, 0.0],
            "standard_k_frac": [0.0, 0.0, 0.0],
            "source_hsp_representative_k_frac": [0.0, 0.0, 0.0],
        },
        "kpoint": "G",
        "workflow_path": "direct_qcut",
        "ready_for_ebr_input": True,
    }]}


def _scalar_self_mapped_sewing_report():
    return build_time_reversal_sewing_report(
        kpoint_frac_by_name={"G": np.zeros(3)},
        g_vectors_frac_by_kpoint={"G": np.zeros((1, 3), dtype=int)},
        coefficients_by_kpoint={"G": np.asarray([[[1.0 + 0.0j]]])},
        band_indices_by_kpoint={"G": np.asarray([1])},
        valley_projectors_by_kpoint={"G": {"v": np.eye(1)}},
        valley_projector_provenance_by_kpoint={
            "G": {"v": {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }},
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "v"},
        spinor=False,
        spinor_convention_verified=True,
    )


def test_self_mapped_valley_promotes_only_with_numerical_antiunitary_evidence():
    source = _self_mapped_source_report()
    grey = _self_mapped_grey_report(multiplicity=1)
    sewing = _scalar_self_mapped_sewing_report()

    report = build_time_reversal_valley_orbit_report(
        valley_mapping_report={
            "status": "validated",
            "theta_square": 1,
            "time_reversal_valley_mapping": {"v": "v"},
            "valley_orbits": [{
                "representative": "v",
                "members": ["v"],
                "mapping_type": "self_mapped",
            }],
        },
        source_irrep_orbits_by_valley={"v": source},
        grey_source_by_valley={"v": grey},
        ebr_input_candidates=_self_mapped_candidates(multiplicity=1),
        antiunitary_sewing_report=sewing,
        trusted_projector_provenance_by_kpoint=(
            _projector_provenance_from_sewing(sewing)
        ),
    )

    assert report["status"] == "validated"
    orbit = report["valley_orbits"][0]
    assert orbit["mapping_type"] == "self_mapped"
    assert orbit["irreps_by_kpoint"] == {"G": ["g_corep"]}
    assert orbit["source_hsp_to_sampled_kpoint"] == {"G": "G"}
    assert orbit["antiunitary_corepresentation_status"] == "validated"


def test_self_mapped_valley_rejects_malformed_or_blocked_sewing_evidence():
    source = _self_mapped_source_report()
    grey = _self_mapped_grey_report(multiplicity=2)
    blocked_sewing = build_time_reversal_sewing_report(
        kpoint_frac_by_name={"G": np.zeros(3)},
        g_vectors_frac_by_kpoint={"G": np.zeros((1, 3), dtype=int)},
        coefficients_by_kpoint={
            "G": np.asarray([[[1.0 + 0.0j], [0.0 + 0.0j]]]),
        },
        band_indices_by_kpoint={"G": np.asarray([1])},
        valley_projectors_by_kpoint={"G": {"v": np.eye(1)}},
        valley_projector_provenance_by_kpoint={
            "G": {"v": {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }},
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "v"},
        spinor=True,
        spinor_convention_verified=True,
    )

    report = build_time_reversal_valley_orbit_report(
        valley_mapping_report={
            "status": "validated",
            "theta_square": -1,
            "time_reversal_valley_mapping": {"v": "v"},
            "valley_orbits": [{
                "representative": "v",
                "members": ["v"],
                "mapping_type": "self_mapped",
            }],
        },
        source_irrep_orbits_by_valley={"v": source},
        grey_source_by_valley={"v": grey},
        ebr_input_candidates=_self_mapped_candidates(multiplicity=2),
        antiunitary_sewing_report=blocked_sewing,
        trusted_projector_provenance_by_kpoint=(
            _projector_provenance_from_sewing(blocked_sewing)
        ),
    )

    assert report["status"] == "blocked"
    assert "antiunitary_corepresentation_sewing_not_validated" in report[
        "blockers"
    ]


def _self_mapped_nontrim_source_report():
    return {
        "status": "validated",
        "time_reversal_hsp_mapping": {"Q": "QA", "QA": "Q"},
        "time_reversal_hsp_orbits": [{
            "representative": "Q",
            "members": ["Q", "QA"],
            "self_mapped": False,
        }],
        "independent_hsp_labels": ["Q"],
        "irrep_partner_by_label": {"q": "qa", "qa": "q"},
    }


def _self_mapped_nontrim_grey_report():
    return {
        "status": "validated",
        "grey_bns_number": "1.2",
        "grey_unitary_restriction_by_irrep": {
            "q_corep": {"q": 1},
            "qa_corep": {"qa": 1},
        },
        "grey_source_hsp_by_irrep": {
            "q_corep": "Q", "qa_corep": "QA",
        },
        "unitary_source_hsp_by_irrep": {"q": "Q", "qa": "QA"},
    }


def _self_mapped_nontrim_sewing_report():
    coefficients = np.asarray([[[1.0 + 0.0j]]])
    return build_time_reversal_sewing_report(
        kpoint_frac_by_name={
            "Q_sample": np.asarray([0.25, 0.0, 0.0]),
            "QA_sample": np.asarray([0.75, 0.0, 0.0]),
        },
        g_vectors_frac_by_kpoint={
            "Q_sample": np.asarray([[0, 0, 0]]),
            "QA_sample": np.asarray([[-1, 0, 0]]),
        },
        coefficients_by_kpoint={
            "Q_sample": coefficients, "QA_sample": coefficients,
        },
        band_indices_by_kpoint={
            "Q_sample": np.asarray([1]), "QA_sample": np.asarray([1]),
        },
        valley_projectors_by_kpoint={
            name: {"v": np.eye(1)} for name in ("Q_sample", "QA_sample")
        },
        valley_projector_provenance_by_kpoint={
            name: {"v": {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }}
            for name in ("Q_sample", "QA_sample")
        },
        projector_selection_blockers=[],
        time_reversal_valley_mapping={"v": "v"},
        spinor=False,
        spinor_convention_verified=True,
    )


def _self_mapped_nontrim_reviewed_source_model():
    return {
        "source_hsp_representative_k_frac_by_label": {
            "Q": [0.25, 0.0, 0.0],
            "QA": [0.75, 0.0, 0.0],
        },
        "standard_operation_rotation_frac_by_index": {
            1: np.eye(3, dtype=int).tolist(),
            2: np.diag([-1, 1, 1]).tolist(),
        },
        "normalized_centering_vectors": [[0.0, 0.0, 0.0]],
    }


def test_redundant_dependent_hsp_candidate_is_checked_but_not_independent_map():
    candidates = {"candidates": [
        {
            "valley": "v",
            "matched_irrep": "q",
            "irrep_multiplicity": 1,
            "irrep_source_provenance": {"source_hsp_label": "Q"},
            "projected_hsp_classification": {
                "source_hsp_label": "Q",
                "classification": "representative",
                "source_hsp_membership": True,
                "validation_status": "validated",
                "parent_k_frac": [0.25, 0.0, 0.0],
                "standard_k_frac": [0.25, 0.0, 0.0],
                "source_hsp_representative_k_frac": [0.25, 0.0, 0.0],
            },
            "kpoint": "Q_sample",
            "workflow_path": "direct_qcut",
            "ready_for_ebr_input": True,
        },
        {
            "valley": "v",
            "matched_irrep": "qa",
            "irrep_multiplicity": 1,
            "irrep_source_provenance": {"source_hsp_label": "QA"},
            "projected_hsp_classification": {
                "source_hsp_label": "QA",
                "classification": "representative",
                "source_hsp_membership": True,
                "validation_status": "validated",
                "parent_k_frac": [0.75, 0.0, 0.0],
                "standard_k_frac": [0.75, 0.0, 0.0],
                "source_hsp_representative_k_frac": [0.75, 0.0, 0.0],
            },
            "kpoint": "QA_sample",
            "workflow_path": "direct_qcut",
            "ready_for_ebr_input": True,
        },
    ]}
    kwargs = {
        "valley_mapping_report": {
            "status": "validated",
            "theta_square": 1,
            "time_reversal_valley_mapping": {"v": "v"},
            "valley_orbits": [{
                "representative": "v",
                "members": ["v"],
                "mapping_type": "self_mapped",
            }],
        },
        "source_irrep_orbits_by_valley": {
            "v": _self_mapped_nontrim_source_report(),
        },
        "grey_source_by_valley": {"v": _self_mapped_nontrim_grey_report()},
        "antiunitary_sewing_report": _self_mapped_nontrim_sewing_report(),
    }
    kwargs["trusted_projector_provenance_by_kpoint"] = (
        _projector_provenance_from_sewing(
            kwargs["antiunitary_sewing_report"]
        )
    )

    report = build_time_reversal_valley_orbit_report(
        ebr_input_candidates=candidates,
        **kwargs,
    )

    assert report["status"] == "validated"
    orbit = report["valley_orbits"][0]
    assert orbit["source_hsp_to_sampled_kpoint"] == {"Q": "Q_sample"}
    assert orbit["projector_workflow_by_sampled_kpoint"] == {
        "Q_sample": {"v": "direct_qcut"},
        "QA_sample": {"v": "direct_qcut"},
    }
    bundle = {
        "valley": "",
        "valley_orbit": ["v"],
        "unitary_valley_irreps": orbit["unitary_valley_irreps"],
        "expected_hsps": ["Q"],
        "irreps_by_kpoint": {"Q": ["q_corep"]},
        "source_hsp_to_sampled_kpoint": {"Q": "Q_sample"},
        "time_reversal": {
            "theta_square": 1,
            "time_reversal_valley_mapping": {"v": "v"},
            "time_reversal_hsp_orbits": orbit["time_reversal_hsp_orbits"],
            "full_unitary_source_hsp_labels": ["Q", "QA"],
            "time_reversal_irrep_pairing": orbit[
                "time_reversal_irrep_pairing"
            ],
            "projector_workflow_by_sampled_kpoint": orbit[
                "projector_workflow_by_sampled_kpoint"
            ],
            "projector_provenance_by_sampled_kpoint": orbit[
                "projector_provenance_by_sampled_kpoint"
            ],
            "source_hsp_binding_by_sampled_kpoint": orbit[
                "source_hsp_binding_by_sampled_kpoint"
            ],
            "antiunitary_sewing_evidence": kwargs[
                "antiunitary_sewing_report"
            ],
            "grey_bns_number": "1.2",
        },
    }
    assert _joint_bundle_time_reversal_evidence_valid(
        bundle=bundle,
        table_spinful=False,
        expected_bns_number="1.2",
        reviewed_source_model=_self_mapped_nontrim_reviewed_source_model(),
    )

    star_equivalent = deepcopy(bundle)
    star_binding = star_equivalent["time_reversal"][
        "source_hsp_binding_by_sampled_kpoint"
    ]["QA_sample"]["v"]
    star_binding["classification"] = "star_equivalent"
    star_binding["standard_k_frac"] = [0.25, 0.0, 0.0]
    star_binding["standard_operation_index"] = 2
    assert _joint_bundle_time_reversal_evidence_valid(
        bundle=star_equivalent,
        table_spinful=False,
        expected_bns_number="1.2",
        reviewed_source_model=_self_mapped_nontrim_reviewed_source_model(),
    )
    star_binding["standard_operation_index"] = 1
    assert not _joint_bundle_time_reversal_evidence_valid(
        bundle=star_equivalent,
        table_spinful=False,
        expected_bns_number="1.2",
        reviewed_source_model=_self_mapped_nontrim_reviewed_source_model(),
    )

    dependent_misbound = deepcopy(bundle)
    dependent_misbound["time_reversal"][
        "source_hsp_binding_by_sampled_kpoint"
    ]["QA_sample"]["v"]["source_hsp_label"] = "Q"
    assert not _joint_bundle_time_reversal_evidence_valid(
        bundle=dependent_misbound,
        table_spinful=False,
        expected_bns_number="1.2",
        reviewed_source_model=_self_mapped_nontrim_reviewed_source_model(),
    )

    dependent_missing = deepcopy(bundle)
    dependent_missing["time_reversal"][
        "source_hsp_binding_by_sampled_kpoint"
    ].pop("QA_sample")
    assert not _joint_bundle_time_reversal_evidence_valid(
        bundle=dependent_missing,
        table_spinful=False,
        expected_bns_number="1.2",
        reviewed_source_model=_self_mapped_nontrim_reviewed_source_model(),
    )

    misbound = deepcopy(bundle)
    misbound["source_hsp_to_sampled_kpoint"] = {"Q": "QA_sample"}
    assert not _joint_bundle_time_reversal_evidence_valid(
        bundle=misbound,
        table_spinful=False,
        expected_bns_number="1.2",
        reviewed_source_model=_self_mapped_nontrim_reviewed_source_model(),
    )

    conflicting = deepcopy(candidates)
    conflicting["candidates"][1]["matched_irrep"] = "q"
    blocked = build_time_reversal_valley_orbit_report(
        ebr_input_candidates=conflicting,
        **kwargs,
    )
    assert blocked["status"] == "blocked"
    assert any(
        value.startswith("time_reversal_multiplicity_or_irrep_mismatch")
        for value in blocked["blockers"]
    )
    conflicting_bundle = deepcopy(bundle)
    conflicting_bundle["unitary_valley_irreps"]["v"]["QA"] = {"q": 1}
    assert not _joint_bundle_time_reversal_evidence_valid(
        bundle=conflicting_bundle,
        table_spinful=False,
        expected_bns_number="1.2",
        reviewed_source_model=_self_mapped_nontrim_reviewed_source_model(),
    )


def _orbit_candidates(*, mismatched_g: bool = False):
    rows = []
    for valley, values in {
        "left": [("G", "g"), ("Q", "q1"), ("M", "m")],
        "right": [
            ("G", "q1" if mismatched_g else "g"),
            ("Q", "q2"), ("M", "m"),
        ],
    }.items():
        for hsp, irrep in values:
            rows.append({
                "valley": valley,
                "matched_irrep": irrep,
                "irrep_multiplicity": 1,
                "irrep_source_provenance": {"source_hsp_label": hsp},
                "ready_for_ebr_input": True,
            })
    return {"candidates": rows}


def test_cross_valley_tr_completion_needs_no_separately_sampled_minus_k_row():
    source = _synthetic_source_orbit_report()
    grey = _synthetic_grey_report()
    report = build_time_reversal_valley_orbit_report(
        valley_mapping_report={
            "status": "validated", "theta_square": -1,
            "time_reversal_valley_mapping": {"left": "right", "right": "left"},
            "valley_orbits": [{
                "representative": "left", "members": ["left", "right"],
                "mapping_type": "exchanged",
            }],
        },
        source_irrep_orbits_by_valley={"left": source, "right": source},
        grey_source_by_valley={"left": grey, "right": grey},
        ebr_input_candidates=_orbit_candidates(),
    )

    assert report["status"] == "validated"
    orbit = report["valley_orbits"][0]
    assert orbit["independent_time_reversal_hsp_labels"] == ["G", "Q", "M"]
    assert orbit["time_reversal_completed_unitary_valley_irreps"]["left"][
        "QA"
    ] == {"qa2": 1}
    assert orbit["time_reversal_completed_unitary_valley_irreps"]["right"][
        "QA"
    ] == {"qa1": 1}
    assert orbit["irreps_by_kpoint"] == {
        "G": ["g_corep"],
        "Q": ["q1_corep", "q2_corep"],
        "M": ["m_corep"],
    }


def test_cross_valley_tr_multiplicity_or_irrep_mismatch_blocks_completion():
    source = _synthetic_source_orbit_report()
    grey = _synthetic_grey_report()
    report = build_time_reversal_valley_orbit_report(
        valley_mapping_report={
            "status": "validated", "theta_square": -1,
            "time_reversal_valley_mapping": {"left": "right", "right": "left"},
            "valley_orbits": [{
                "representative": "left", "members": ["left", "right"],
                "mapping_type": "exchanged",
            }],
        },
        source_irrep_orbits_by_valley={"left": source, "right": source},
        grey_source_by_valley={"left": grey, "right": grey},
        ebr_input_candidates=_orbit_candidates(mismatched_g=True),
    )

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("time_reversal_multiplicity_or_irrep_mismatch")
        for blocker in report["blockers"]
    )


def test_cross_valley_tr_completion_rejects_not_ready_candidate_rows():
    source = _synthetic_source_orbit_report()
    grey = _synthetic_grey_report()
    candidates = _orbit_candidates()
    candidates["candidates"][0]["ready_for_ebr_input"] = False

    report = build_time_reversal_valley_orbit_report(
        valley_mapping_report={
            "status": "validated", "theta_square": -1,
            "time_reversal_valley_mapping": {"left": "right", "right": "left"},
            "valley_orbits": [{
                "representative": "left", "members": ["left", "right"],
                "mapping_type": "exchanged",
            }],
        },
        source_irrep_orbits_by_valley={"left": source, "right": source},
        grey_source_by_valley={"left": grey, "right": grey},
        ebr_input_candidates=candidates,
    )

    assert report["status"] == "blocked"
    assert "missing_trusted_independent_hsp:left:G" in report["blockers"]


@pytest.mark.parametrize(
    ("target", "unitary_hsps", "grey_restrictions", "grey_hsps", "blocker"),
    [
        (
            {"g": 2, "rogue": 1},
            {"g": "G"},
            {"g_corep": {"g": 2}},
            {"g_corep": "G"},
            "unknown_unitary_irrep_in_grey_target:G:rogue",
        ),
        (
            {"g": 2, "q": 1},
            {"g": "G", "q": "Q"},
            {"g_corep": {"g": 2}},
            {"g_corep": "G"},
            "wrong_hsp_unitary_irrep_in_grey_target:G:q:Q",
        ),
        (
            {"g": 2},
            {"g": "G"},
            {},
            {},
            "missing_grey_irrep_basis_for_hsp:G",
        ),
    ],
)
def test_grey_decomposition_rejects_unknown_wrong_hsp_and_empty_basis(
    target, unitary_hsps, grey_restrictions, grey_hsps, blocker,
):
    result, blockers = _decompose_grey_counts(
        unitary_counts_by_hsp={"G": target},
        grey_restrictions=grey_restrictions,
        grey_hsp_by_irrep=grey_hsps,
        unitary_hsp_by_irrep=unitary_hsps,
    )

    assert result == {}
    assert blocker in blockers


def test_joint_valley_orbit_problem_and_export_replace_one_valley_claims():
    candidates = _orbit_candidates()["candidates"]
    for candidate in candidates:
        candidate["irrep_source_provenance"]["source_table_spinor"] = True
        candidate.update({
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "subspace_group_candidate": "P4",
            "subspace_space_group": {
                "status": "resolved",
                "candidate_space_group_number": 75,
                "candidate_space_group_symbol": "P4",
            },
        })
    orbit_report = {
        "status": "validated",
        "enabled": True,
        "theta_square": -1,
        "time_reversal_valley_mapping": {"left": "right", "right": "left"},
        "valley_orbits": [{
            "status": "validated",
            "members": ["left", "right"],
            "full_unitary_source_hsp_labels": ["G", "Q", "QA", "M"],
            "expected_hsps": ["G", "Q", "M"],
            "irreps_by_kpoint": {
                "G": ["g_corep"], "Q": ["q1_corep", "q2_corep"],
                "M": ["m_corep"],
            },
            "unitary_valley_irreps": {"left": {}, "right": {}},
            "time_reversal_hsp_orbits": [],
            "time_reversal_irrep_pairing": {},
            "grey_bns_number": "75.2",
            "blockers": [],
        }],
    }

    problems = build_ebr_problem_instances(
        ebr_input_candidates={"candidates": candidates},
        time_reversal_orbit_report=orbit_report,
    )

    assert problems["instance_count"] == 1
    instance = problems["instances"][0]
    assert instance["problem_kind"] == "valley_orbit_reduced_ebr"
    assert instance["valley_orbit"] == ["left", "right"]
    assert instance["valley"] == ""
    assert instance["canonical_hsp_vector_complete"] is True
    assert "ready_for_reduced_table_validation" not in instance

    export = build_ebr_export_bundle(ebr_problem_instances=problems)
    assert export["schema_version"] == "1.5.0"
    assert export["bundle_count"] == 1
    assert export["bundles"][0]["problem_kind"] == (
        "valley_orbit_reduced_ebr"
    )
    assert export["bundles"][0]["valley_orbit"] == ["left", "right"]
    assert not any(
        bundle.get("valley") in {"left", "right"}
        for bundle in export["bundles"]
    )
