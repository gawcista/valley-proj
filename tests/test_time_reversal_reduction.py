from __future__ import annotations

import numpy as np
import pytest

from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
from valleyscope.analysis.irreptables_runtime_table_builder import (
    build_auto_time_reversal_reduced_ebr_table,
)
from valleyscope.analysis.reduced_ebr_mapping import promote_bundle_for_solve

from valleyscope.analysis.time_reversal_orbits import (
    _decompose_grey_counts,
    build_time_reversal_valley_orbit_report,
    derive_time_reversal_valley_mapping,
    validate_time_reversal_valley_mapping,
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
    derive_time_reversal_ebr_column_pairing,
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


def test_ebr_pairing_uses_complete_vectors_and_preserves_ambiguity():
    report = derive_time_reversal_ebr_column_pairing(
        source_basis_labels=["q1", "qa1", "r1", "ra1"],
        source_ebrs=[
            {"ebr_label": "x", "vector": [1, 0, 1, 0]},
            {"ebr_label": "y", "vector": [0, 1, 0, 1]},
            {"ebr_label": "y_duplicate", "vector": [0, 1, 0, 1]},
        ],
        irrep_partner_by_label={
            "q1": "qa1", "qa1": "q1", "r1": "ra1", "ra1": "r1",
        },
    )

    assert report["status"] == "ambiguous"
    assert report["partner_candidates_by_ebr_label"]["x"] == [
        "y", "y_duplicate",
    ]
    assert report["partner_candidates_by_ebr_label"]["y"] == ["x"]


def test_time_reversal_ebr_pairing_rejects_empty_source_columns():
    report = derive_time_reversal_ebr_column_pairing(
        source_basis_labels=["g"],
        source_ebrs=[],
        irrep_partner_by_label={"g": "g"},
    )

    assert report["status"] == "blocked"
    assert "time_reversal_ebr_source_columns_missing" in report["blockers"]


def test_ebr_pairing_does_not_pair_columns_equal_only_after_row_deletion():
    report = derive_time_reversal_ebr_column_pairing(
        source_basis_labels=["q1", "qa1", "r1", "ra1"],
        source_ebrs=[
            {"ebr_label": "x", "vector": [1, 0, 1, 0]},
            {"ebr_label": "false_reduced_partner", "vector": [0, 1, 1, 0]},
        ],
        irrep_partner_by_label={
            "q1": "qa1", "qa1": "q1", "r1": "ra1", "ra1": "r1",
        },
    )

    assert report["status"] == "blocked"
    assert report["partner_candidates_by_ebr_label"]["x"] == []
    assert "missing_time_reversal_ebr_partner:x" in report["blockers"]


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
    assert report["unitary_ebr_partner_candidates"] == {
        source["source_ebrs"][0]["ebr_label"]: [
            source["source_ebrs"][1]["ebr_label"]
        ],
        source["source_ebrs"][1]["ebr_label"]: [
            source["source_ebrs"][0]["ebr_label"]
        ],
        source["source_ebrs"][2]["ebr_label"]: [
            source["source_ebrs"][2]["ebr_label"]
        ],
        source["source_ebrs"][3]["ebr_label"]: [
            source["source_ebrs"][4]["ebr_label"]
        ],
        source["source_ebrs"][4]["ebr_label"]: [
            source["source_ebrs"][3]["ebr_label"]
        ],
        source["source_ebrs"][5]["ebr_label"]: [
            source["source_ebrs"][5]["ebr_label"]
        ],
        source["source_ebrs"][6]["ebr_label"]: [
            source["source_ebrs"][7]["ebr_label"]
        ],
        source["source_ebrs"][7]["ebr_label"]: [
            source["source_ebrs"][6]["ebr_label"]
        ],
        source["source_ebrs"][8]["ebr_label"]: [
            source["source_ebrs"][8]["ebr_label"]
        ],
    }
    assert set(report["grey_ebr_unitary_column_candidates"].values()) == {
        (0, 1), (2, 2), (3, 4), (5, 5), (6, 7), (8, 8),
    }
    assert set(report["unitary_ebr_column_orbits"]) == {
        (0, 1), (2, 2), (3, 4), (5, 5), (6, 7), (8, 8),
    }


def test_grey_source_must_cover_every_unitary_ebr_column_orbit():
    table = load_standard_irrep_table(143, spinor=True)
    source = load_ebr_source_data(143, True)
    rows = _reviewed_rows(table, source["source_basis_labels"])
    orbits = derive_time_reversal_source_irrep_orbits(
        reviewed_rows=rows,
        centering_vectors=[[0.0, 0.0, 0.0]],
    )
    incomplete_grey = dict(load_ebr_source_data("143.2", True))
    incomplete_grey["source_ebrs"] = list(
        incomplete_grey["source_ebrs"][:-1]
    )

    report = validate_grey_group_time_reversal_source(
        unitary_table=table,
        reviewed_rows=rows,
        unitary_source_data=source,
        irrep_partner_by_label=orbits["irrep_partner_by_label"],
        centering_vectors=[[0.0, 0.0, 0.0]],
        grey_source_loader=lambda _number, _spinor: incomplete_grey,
    )

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("incomplete_grey_ebr_unitary_orbit_coverage:")
        for blocker in report["blockers"]
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
        "ready_for_external_solver": True,
    }
    export = {"bundles": [bundle]}
    assert attach_real_certificate(export, table) is not None
    return export["bundles"][0], table


def _blocker_codes(promotion):
    return {row["code"] for row in promotion["blocker_reasons"]}


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
    assert instance["ready_for_reduced_table_validation"] is True

    export = build_ebr_export_bundle(ebr_problem_instances=problems)
    assert export["schema_version"] == "1.4.0"
    assert export["bundle_count"] == 1
    assert export["bundles"][0]["problem_kind"] == (
        "valley_orbit_reduced_ebr"
    )
    assert export["bundles"][0]["valley_orbit"] == ["left", "right"]
    assert not any(
        bundle.get("valley") in {"left", "right"}
        for bundle in export["bundles"]
    )
