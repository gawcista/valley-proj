from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from valleyscope.analysis.ebr_export_bundle import (
    build_ebr_export_bundle as _build_ebr_export_bundle,
)
from valleyscope.analysis.ebr_problem_instances import (
    build_ebr_problem_instances as _build_ebr_problem_instances,
)
from valleyscope.analysis.irreptables_runtime_table_builder import (
    build_auto_time_reversal_reduced_ebr_table,
)
from valleyscope.analysis.reduced_ebr_mapping import (
    _joint_bundle_time_reversal_evidence_valid,
    build_reduced_ebr_mapping as _build_reduced_ebr_mapping,
    promote_bundle_for_solve as _promote_bundle_for_solve,
    validate_joint_grey_bundle_provenance,
)
from valleyscope.analysis.unitary_provenance import (
    validate_tr_completed_unitary_bundle as _unitary_bundle_completion_evidence_valid,
)

from valleyscope.analysis.time_reversal_orbits import (
    _candidate_source_hsp_to_sampled_kpoint,
    _decompose_grey_counts,
    build_time_reversal_valley_orbit_report,
    derive_time_reversal_valley_mapping,
    validate_time_reversal_valley_mapping,
)
from valleyscope.analysis.time_reversal_sewing import (
    build_time_reversal_sewing_report,
)
from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
from valleyscope.io.wavefunction_convention import canonical_identity
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
from tests.reduced_ebr_promo_helpers import (
    attach_real_certificate,
    attach_cprime_fixture_to_candidates,
    attach_cprime_fixture_contract,
    cprime_summary_for_export,
    cprime_validation_context_for_export,
    real_primitive_certificate_identity,
)


def build_ebr_problem_instances(**kwargs):
    candidates = kwargs.get("ebr_input_candidates")
    if isinstance(candidates, dict):
        attach_cprime_fixture_to_candidates(candidates)
    orbit_report = kwargs.get("time_reversal_orbit_report")
    if isinstance(orbit_report, dict):
        for orbit in orbit_report.get("valley_orbits", []):
            if not isinstance(orbit, dict):
                continue
            full_hsps = {
                str(value)
                for value in orbit.get(
                    "full_unitary_source_hsp_labels", []
                )
                if isinstance(value, str) and value
            }
            joint_hsps = {
                str(value)
                for value in orbit.get("expected_hsps", [])
                if isinstance(value, str) and value
            }

            def identities(hsps):
                return {
                    hsp: {
                        "spinor_source_basis_certificate_identity": (
                            canonical_identity({
                                "fixture": "tr_source_basis",
                            })
                        ),
                        "double_space_group_lift_certificate_identity": (
                            canonical_identity({
                                "fixture": "tr_lift", "hsp": hsp,
                            })
                        ),
                        "scoped_representation_evidence_identity": (
                            canonical_identity({
                                "fixture": "tr_completed_scope",
                                "hsp": hsp,
                            })
                        ),
                    }
                    for hsp in hsps
                }

            orbit["tr_completed_cprime_identity_by_hsp"] = identities(
                full_hsps
            )
            orbit["joint_cprime_identity_by_hsp"] = identities(
                joint_hsps
            )
    return _build_ebr_problem_instances(**kwargs)


def build_ebr_export_bundle(**kwargs):
    export = _build_ebr_export_bundle(**kwargs)
    attach_cprime_fixture_contract(export)
    return export


def _fixture_cprime_context(export):
    context = cprime_validation_context_for_export(export)
    return dict(context["_by_identity"])


def build_reduced_ebr_mapping(**kwargs):
    export = kwargs.get("ebr_export_bundle")
    if isinstance(export, dict):
        kwargs.setdefault(
            "cprime_validation_context",
            _fixture_cprime_context(export),
        )
    return _build_reduced_ebr_mapping(**kwargs)


def promote_bundle_for_solve(*, bundle, table):
    return _promote_bundle_for_solve(
        bundle=bundle,
        table=table,
        cprime_validation_context=_fixture_cprime_context(
            {"bundles": [bundle]}
        ),
    )
    return export


def test_tr_completed_problem_requires_scoped_cprime_identity_inventory():
    candidates = _orbit_candidates()
    attach_cprime_fixture_to_candidates(candidates)
    problems = _build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=_exchanged_orbit_report(),
    )

    assert problems["ready_instance_count"] == 0
    assert all(
        any(
            str(blocker).startswith(
                "tr_completed_scoped_representation_evidence_missing"
            )
            for blocker in instance["blocked_by"]
        )
        for instance in problems["instances"]
    )


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
    def observed_completion(valley):
        source = f"fixture/{valley}/GM"
        identity = {
            "source": source,
            "workflow_path": "direct_qcut",
            "valley": valley,
            "source_hsp_label": "GM",
            "sampled_kpoint": "GM",
            "irrep": "-GM4",
            "multiplicity": 1,
        }
        return {
            "GM": [{
                "completion_kind": "observed_at_sampled_kpoint",
                "target_valley": valley,
                "target_source_hsp_label": "GM",
                "irrep": "-GM4",
                "multiplicity": 1,
                "sampled_kpoint": "GM",
                "source_candidate_identity": identity,
                "source_candidate_provenance": {
                    "source": source,
                    "workflow_path": "direct_qcut",
                    "irrep_source_provenance": {
                        "source_hsp_label": "GM",
                        "source_table_spinor": True,
                    },
                },
                "structural_status": "validated",
                "readiness_status": "trusted",
                "blockers": [],
            }],
        }

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
            "representative_valley": "left",
            "time_reversal_valley_mapping": {
                "left": "right", "right": "left",
            },
            "source_hsp_to_sampled_kpoint_by_valley": {
                "left": {"GM": "GM"},
                "right": {"GM": "GM"},
            },
            "observed_source_hsp_to_sampled_kpoint_by_valley": {
                "left": {"GM": "GM"},
                "right": {"GM": "GM"},
            },
            "unitary_valley_irrep_completion_records": {
                "left": observed_completion("left"),
                "right": observed_completion("right"),
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

    promotion = promote_bundle_for_solve(bundle=bundle, table=table)
    assert promotion["promoted"]
    assert validate_joint_grey_bundle_provenance(
        bundle,
        promotion["table_provenance"],
    )
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle={"bundles": [bundle]},
        table=table,
    )
    assert mapping["solutions"][0]["time_reversal"][
        "source_hsp_to_sampled_kpoint_by_valley"
    ] == {
        "left": {"GM": "GM"},
        "right": {"GM": "GM"},
    }

    missing_component_binding = deepcopy(bundle)
    missing_component_binding["time_reversal"] = deepcopy(
        bundle["time_reversal"]
    )
    missing_component_binding["time_reversal"][
        "source_hsp_to_sampled_kpoint_by_valley"
    ]["right"] = {}
    assert not promote_bundle_for_solve(
        bundle=missing_component_binding,
        table=table,
    )["promoted"]

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


def test_joint_grey_mapping_is_authoritative_for_ingestion():
    from valleyscope.analysis.database_ingestion_record import (
        build_database_ingestion_record,
    )

    bundle, table = _reviewed_joint_bundle_and_table()
    export = {"bundles": [bundle]}
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=export,
        table=table,
        reduced_ebr_input={"source": "auto_time_reversal_grey"},
    )
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(
            export, target_kpoints=["GM"], iband=[1]
        ),
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )

    assert record["final_reduced_ebr_result_count"] == 1
    assert record["validation_errors"] == []
    assert record["reduced_ebr_records"][0]["table_source"] == (
        "auto_time_reversal_grey"
    )


@pytest.mark.parametrize(
    "mutation",
    ["valley_mapping", "hsp_orbit", "irrep_pairing"],
)
def test_ingestion_revalidates_coordinated_joint_grey_mutation(mutation):
    from valleyscope.analysis.database_ingestion_record import (
        build_database_ingestion_record,
    )

    bundle, table = _reviewed_joint_bundle_and_table()
    export = {"bundles": [bundle]}
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=export,
        table=table,
        reduced_ebr_input={"source": "auto_time_reversal_grey"},
    )
    solution = mapping["solutions"][0]
    if mutation == "valley_mapping":
        forged = {
            "left": "left",
            "right": "right",
        }
        field = "time_reversal_valley_mapping"
    elif mutation == "hsp_orbit":
        forged = [{
            "representative": "GM",
            "members": ["GM", "forged_hsp"],
            "self_mapped": False,
        }]
        field = "time_reversal_hsp_orbits"
    else:
        forged = {"-GM4": "forged_irrep"}
        field = "time_reversal_irrep_pairing"
    bundle["time_reversal"][field] = deepcopy(forged)
    solution["time_reversal"][field] = deepcopy(forged)

    canonical_bundle = {
        key: value
        for key, value in bundle.items()
        if key != "promotion_provenance"
    }
    digest = hashlib.sha256(json.dumps(
        canonical_bundle,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    solution["promotion_provenance"]["promotion_input_identity"] = {
        "schema_version": "1.0.0",
        "algorithm": "sha256",
        "digest": digest,
    }
    solution["promotion_provenance"]["irrep_vector"] = deepcopy(
        solution["irrep_vector"]
    )

    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(
            export, target_kpoints=["GM"], iband=[1]
        ),
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )

    assert record["final_reduced_ebr_result_count"] == 0
    assert record["validation_errors"] == [
        "mapping solution b_tr: current joint grey provenance is invalid"
    ]


def test_problem_kind_compatibility_rejects_grey_table_for_unitary_bundle():
    bundle, table = _reviewed_joint_bundle_and_table()
    unitary = dict(bundle)
    unitary.update({
        "problem_kind": "unitary_valley_reduced_ebr",
        "physical_object_kind": "unitary_valley_projected_subspace",
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
    self_mapped["time_reversal"]["representative_valley"] = "v"
    self_mapped["time_reversal"][
        "source_hsp_to_sampled_kpoint_by_valley"
    ] = {"v": {"GM": "GM"}}
    observed = deepcopy(
        bundle["time_reversal"][
            "unitary_valley_irrep_completion_records"
        ]["left"]["GM"][0]
    )
    observed["target_valley"] = "v"
    observed["multiplicity"] = 2
    observed["source_candidate_identity"].update({
        "source": "fixture/v/GM",
        "valley": "v",
        "multiplicity": 2,
    })
    observed["source_candidate_provenance"]["source"] = "fixture/v/GM"
    self_mapped["time_reversal"][
        "observed_source_hsp_to_sampled_kpoint_by_valley"
    ] = {"v": {"GM": "GM"}}
    self_mapped["time_reversal"][
        "unitary_valley_irrep_completion_records"
    ] = {"v": {"GM": [observed]}}
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
    attach_cprime_fixture_contract({"bundles": [self_mapped]})

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
        "irrep_source_provenance": {
            "source_hsp_label": "G",
            "source_table_spinor": False,
        },
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
        "source": "fixture/v/G",
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


def _self_mapped_nontrim_pipeline_candidates(*, include_dependent: bool):
    rows = [{
        "valley": "v",
        "matched_irrep": "q",
        "irrep_multiplicity": 1,
        "irrep_source_provenance": {
            "source_hsp_label": "Q",
            "source_table_spinor": False,
        },
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
        "source": "fixture/v/Q",
        "readiness_level": "trusted",
        "subspace_group_candidate": "P1",
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_number": 1,
            "candidate_space_group_symbol": "P1",
        },
        "ready_for_ebr_input": True,
    }]
    if include_dependent:
        rows.append({
            **deepcopy(rows[0]),
            "matched_irrep": "qa",
            "irrep_source_provenance": {
                "source_hsp_label": "QA",
                "source_table_spinor": False,
            },
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
            "source": "fixture/v/QA",
        })
    return {"candidates": rows}


def _self_mapped_nontrim_pipeline(
    *, include_dependent: bool, include_sewing: bool,
):
    candidates = _self_mapped_nontrim_pipeline_candidates(
        include_dependent=include_dependent,
    )
    sewing = _self_mapped_nontrim_sewing_report() if include_sewing else None
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
        source_irrep_orbits_by_valley={
            "v": _self_mapped_nontrim_source_report(),
        },
        grey_source_by_valley={"v": _self_mapped_nontrim_grey_report()},
        ebr_input_candidates=candidates,
        antiunitary_sewing_report=sewing,
        trusted_projector_provenance_by_kpoint=(
            _projector_provenance_from_sewing(sewing)
            if sewing is not None else None
        ),
    )
    problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=report,
    )
    export = build_ebr_export_bundle(ebr_problem_instances=problems)
    unitary = next((
        bundle for bundle in export["bundles"]
        if bundle["problem_kind"] == "unitary_valley_reduced_ebr"
    ), None)
    return report, unitary


def test_self_mapped_unitary_final_validation_rechecks_sewing_and_theta():
    report, bundle = _self_mapped_nontrim_pipeline(
        include_dependent=False,
        include_sewing=True,
    )
    assert report["status"] == "validated"
    assert bundle is not None
    assert _unitary_bundle_completion_evidence_valid(bundle)

    missing_sewing = deepcopy(bundle)
    missing_sewing["time_reversal"].pop("antiunitary_sewing_evidence")
    assert not _unitary_bundle_completion_evidence_valid(missing_sewing)

    blocked_sewing = deepcopy(bundle)
    blocked_sewing["time_reversal"]["antiunitary_sewing_evidence"][
        "status"
    ] = "blocked"
    assert not _unitary_bundle_completion_evidence_valid(blocked_sewing)

    wrong_sewing_orbit = deepcopy(bundle)
    wrong_sewing_orbit["time_reversal"][
        "antiunitary_sewing_evidence"
    ]["time_reversal_kpoint_mapping"]["Q_sample"] = "Q_sample"
    assert not _unitary_bundle_completion_evidence_valid(
        wrong_sewing_orbit
    )

    wrong_theta = deepcopy(bundle)
    wrong_theta["time_reversal"]["theta_square"] = -1
    assert not _unitary_bundle_completion_evidence_valid(wrong_theta)

    wrong_projector = deepcopy(bundle)
    wrong_projector["time_reversal"][
        "projector_workflow_by_sampled_kpoint"
    ]["Q_sample"]["v"] = "symmetry_adapted"
    assert not _unitary_bundle_completion_evidence_valid(wrong_projector)

    wrong_evidence_k = deepcopy(bundle)
    inferred = wrong_evidence_k[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    inferred["evidence_sampled_kpoint"] = "QA_sample"
    inferred["source_candidate_identity"]["sampled_kpoint"] = "QA_sample"
    assert not _unitary_bundle_completion_evidence_valid(wrong_evidence_k)


def test_self_mapped_fully_observed_vector_does_not_require_sewing():
    report, bundle = _self_mapped_nontrim_pipeline(
        include_dependent=True,
        include_sewing=False,
    )

    assert report["status"] == "blocked"
    assert "antiunitary_corepresentation_sewing_not_validated" in report[
        "blockers"
    ]
    assert bundle is not None
    assert all(
        record["completion_kind"] == "observed_at_sampled_kpoint"
        for records in bundle[
            "unitary_irrep_completion_records_by_hsp"
        ].values()
        for record in records
    )
    assert _unitary_bundle_completion_evidence_valid(bundle)


def test_redundant_dependent_hsp_candidate_is_checked_but_not_independent_map():
    candidates = {"candidates": [
        {
            "valley": "v",
            "matched_irrep": "q",
            "irrep_multiplicity": 1,
            "irrep_source_provenance": {
                "source_hsp_label": "Q",
                "source_table_spinor": False,
            },
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
            "source": "fixture/v/Q",
            "ready_for_ebr_input": True,
        },
        {
            "valley": "v",
            "matched_irrep": "qa",
            "irrep_multiplicity": 1,
            "irrep_source_provenance": {
                "source_hsp_label": "QA",
                "source_table_spinor": False,
            },
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
            "source": "fixture/v/QA",
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
            "representative_valley": "v",
            "time_reversal_valley_mapping": {"v": "v"},
            "source_hsp_to_sampled_kpoint_by_valley": orbit[
                "source_hsp_to_sampled_kpoint_by_valley"
            ],
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
                "irrep_source_provenance": {
                    "source_hsp_label": hsp,
                    "source_table_spinor": True,
                },
                "kpoint": f"{hsp}_{valley}",
                "workflow_path": "direct_qcut",
                "source": f"fixture/{valley}/{hsp}",
                "subspace_group_candidate": "P4",
                "subspace_space_group": {
                    "status": "resolved",
                    "candidate_space_group_number": 75,
                    "candidate_space_group_symbol": "P4",
                },
                "ready_for_ebr_input": True,
            })
    return {"candidates": rows}


def _exchanged_orbit_report(candidates=None):
    source = _synthetic_source_orbit_report()
    grey = _synthetic_grey_report()
    return build_time_reversal_valley_orbit_report(
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
        ebr_input_candidates=candidates or _orbit_candidates(),
    )


def test_cross_valley_completion_records_observed_and_inferred_provenance():
    report = _exchanged_orbit_report()

    assert report["status"] == "validated"
    orbit = report["valley_orbits"][0]
    assert orbit["independent_time_reversal_hsp_labels"] == ["G", "Q", "M"]
    assert orbit["time_reversal_completed_unitary_valley_irreps"]["left"][
        "QA"
    ] == {"qa2": 1}
    assert orbit["time_reversal_completed_unitary_valley_irreps"]["right"][
        "QA"
    ] == {"qa1": 1}
    assert orbit["source_hsp_to_sampled_kpoint"] == {
        "G": "G_left", "Q": "Q_left", "M": "M_left",
    }
    assert orbit["source_hsp_to_sampled_kpoint_by_valley"] == {
        "left": {"G": "G_left", "Q": "Q_left", "M": "M_left"},
        "right": {"G": "G_right", "Q": "Q_right", "M": "M_right"},
    }
    records = orbit[
        "unitary_valley_irrep_completion_records"
    ]
    observed = records["left"]["Q"][0]
    assert observed["completion_kind"] == "observed_at_sampled_kpoint"
    assert observed["target_valley"] == "left"
    assert observed["target_source_hsp_label"] == "Q"
    assert observed["irrep"] == "q1"
    assert observed["multiplicity"] == 1
    assert observed["sampled_kpoint"] == "Q_left"
    assert observed["evidence_sampled_kpoint"] == "Q_left"
    assert observed["source_candidate_identity"]["source"] == (
        "fixture/left/Q"
    )
    assert observed["structural_status"] == "validated"
    assert observed["readiness_status"] == "trusted"

    inferred = records["left"]["QA"][0]
    assert inferred["completion_kind"] == "inferred_by_time_reversal"
    assert inferred["target_valley"] == "left"
    assert inferred["target_source_hsp_label"] == "QA"
    assert inferred["irrep"] == "qa2"
    assert inferred["multiplicity"] == 1
    assert "sampled_kpoint" not in inferred
    assert inferred["evidence_valley"] == "right"
    assert inferred["evidence_source_hsp_label"] == "Q"
    assert inferred["evidence_sampled_kpoint"] == "Q_right"
    assert inferred["reviewed_time_reversal_relation"] == {
        "evidence_valley": "right",
        "target_valley": "left",
        "evidence_source_hsp_label": "Q",
        "target_source_hsp_label": "QA",
        "evidence_irrep": "q2",
        "target_irrep": "qa2",
    }
    assert inferred["source_candidate_identity"]["source"] == (
        "fixture/right/Q"
    )
    assert inferred["structural_status"] == "validated"
    assert inferred["readiness_status"] == "trusted"


def test_missing_source_hsp_blockers_preserve_reviewed_basis_order():
    _, _, blockers = _candidate_source_hsp_to_sampled_kpoint(
        [],
        ["left"],
        independent_hsps=["GM", "V", "Y"],
        representative="left",
    )

    assert blockers == [
        "source_hsp_sampled_kpoint_mapping_incomplete:left:GM",
        "source_hsp_sampled_kpoint_mapping_incomplete:left:V",
        "source_hsp_sampled_kpoint_mapping_incomplete:left:Y",
    ]


def test_tr_enabled_problem_builder_emits_unitary_components_and_joint_grey():
    report = _exchanged_orbit_report()

    problems = build_ebr_problem_instances(
        ebr_input_candidates=_orbit_candidates(),
        time_reversal_orbit_report=report,
    )

    assert problems["instance_count"] == 3
    assert len({row["instance_id"] for row in problems["instances"]}) == 3
    unitary = {
        row["valley"]: row for row in problems["instances"]
        if row["problem_kind"] == "unitary_valley_reduced_ebr"
    }
    joint = [
        row for row in problems["instances"]
        if row["problem_kind"] == "valley_orbit_reduced_ebr"
    ]
    assert set(unitary) == {"left", "right"}
    assert len(joint) == 1
    assert unitary["left"]["physical_object_kind"] == (
        "unitary_valley_projected_subspace"
    )
    assert unitary["left"]["irreps_by_kpoint"] == {
        "G": ["g"], "M": ["m"], "Q": ["q1"], "QA": ["qa2"],
    }
    assert unitary["right"]["irreps_by_kpoint"] == {
        "G": ["g"], "M": ["m"], "Q": ["q2"], "QA": ["qa1"],
    }
    assert unitary["left"]["expected_hsps"] == ["G", "Q", "QA", "M"]
    assert unitary["left"]["source_hsp_to_sampled_kpoint"] == {
        "G": "G_left", "Q": "Q_left", "M": "M_left",
    }
    assert unitary["left"][
        "independent_source_hsp_to_sampled_kpoint"
    ] == {"G": "G_left", "Q": "Q_left", "M": "M_left"}
    assert unitary["left"][
        "observed_source_hsp_to_sampled_kpoint"
    ] == {"G": "G_left", "Q": "Q_left", "M": "M_left"}
    assert "QA" not in unitary["left"]["source_hsp_to_sampled_kpoint"]
    assert unitary["left"]["unitary_vector_construction"] == {
        "kind": "time_reversal_completed_unitary_rows",
        "source": "validated_time_reversal_valley_orbit",
        "orbit_id": "time_reversal_valley_orbit_001",
    }
    assert unitary["left"]["canonical_hsp_vector_ready"] is True
    assert unitary["left"][
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]["completion_kind"] == "inferred_by_time_reversal"
    assert joint[0]["physical_object_kind"] == (
        "joint_time_reversal_valley_orbit"
    )
    assert joint[0]["irreps_by_kpoint"] == {
        "G": ["g_corep"],
        "Q": ["q1_corep", "q2_corep"],
        "M": ["m_corep"],
    }
    orbit = report["valley_orbits"][0]
    assert joint[0]["canonical_hsp_vector_ready"] is True
    assert joint[0]["time_reversal"][
        "source_hsp_to_sampled_kpoint_by_valley"
    ] == orbit["source_hsp_to_sampled_kpoint_by_valley"]


def test_tr_unitary_completion_provenance_survives_export_without_fake_sample():
    report = _exchanged_orbit_report()
    problems = build_ebr_problem_instances(
        ebr_input_candidates=_orbit_candidates(),
        time_reversal_orbit_report=report,
    )

    export = build_ebr_export_bundle(ebr_problem_instances=problems)

    assert export["bundle_count"] == 3
    left = next(
        bundle for bundle in export["bundles"]
        if bundle["problem_kind"] == "unitary_valley_reduced_ebr"
        and bundle["valley"] == "left"
    )
    assert left["physical_object_kind"] == (
        "unitary_valley_projected_subspace"
    )
    assert left["expected_hsps"] == ["G", "Q", "QA", "M"]
    assert left["source_hsp_to_sampled_kpoint"] == {
        "G": "G_left", "Q": "Q_left", "M": "M_left",
    }
    inferred = left["unitary_irrep_completion_records_by_hsp"]["QA"][0]
    assert inferred["completion_kind"] == "inferred_by_time_reversal"
    assert "sampled_kpoint" not in inferred
    assert inferred["evidence_sampled_kpoint"] == "Q_right"
    joint = next(
        bundle for bundle in export["bundles"]
        if bundle["problem_kind"] == "valley_orbit_reduced_ebr"
    )
    assert joint["physical_object_kind"] == (
        "joint_time_reversal_valley_orbit"
    )
    assert joint["time_reversal"][
        "unitary_valley_irrep_completion_records"
    ]["left"]["QA"][0]["evidence_sampled_kpoint"] == "Q_right"
    assert joint["time_reversal"][
        "source_hsp_to_sampled_kpoint_by_valley"
    ] == report["valley_orbits"][0][
        "source_hsp_to_sampled_kpoint_by_valley"
    ]


def test_tr_unitary_promotion_revalidates_row_level_completion_provenance():
    export = build_ebr_export_bundle(
        ebr_problem_instances=build_ebr_problem_instances(
            ebr_input_candidates=_orbit_candidates(),
            time_reversal_orbit_report=_exchanged_orbit_report(),
        )
    )
    left = next(
        bundle for bundle in export["bundles"]
        if bundle["problem_kind"] == "unitary_valley_reduced_ebr"
        and bundle["valley"] == "left"
    )
    assert _unitary_bundle_completion_evidence_valid(left)

    fake_sample = deepcopy(left)
    fake_sample["unitary_irrep_completion_records_by_hsp"]["QA"][0][
        "sampled_kpoint"
    ] = "Q_right"
    assert not _unitary_bundle_completion_evidence_valid(fake_sample)

    missing_evidence = deepcopy(left)
    missing_evidence[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0].pop("evidence_sampled_kpoint")
    assert not _unitary_bundle_completion_evidence_valid(missing_evidence)

    representative_fallback = deepcopy(left)
    representative_fallback["source_hsp_to_sampled_kpoint"]["QA"] = (
        "Q_left"
    )
    assert not _unitary_bundle_completion_evidence_valid(
        representative_fallback
    )

    mismatched_candidate = deepcopy(left)
    mismatched_candidate[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]["source_candidate_identity"]["valley"] = "left"
    assert not _unitary_bundle_completion_evidence_valid(
        mismatched_candidate
    )

    forged_partner = deepcopy(left)
    forged = forged_partner[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    forged["irrep"] = "qa1"
    forged["reviewed_time_reversal_relation"]["target_irrep"] = "qa1"
    forged_partner["irreps_by_kpoint"]["QA"] = ["qa1"]
    assert not _unitary_bundle_completion_evidence_valid(forged_partner)

    forged_evidence = deepcopy(left)
    forged = forged_evidence[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    forged["evidence_valley"] = "left"
    forged["reviewed_time_reversal_relation"]["evidence_valley"] = "left"
    forged["source_candidate_identity"]["valley"] = "left"
    assert not _unitary_bundle_completion_evidence_valid(forged_evidence)

    missing_construction = deepcopy(left)
    missing_construction.pop("unitary_vector_construction")
    assert not _unitary_bundle_completion_evidence_valid(
        missing_construction
    )

    empty_source = deepcopy(left)
    record = empty_source[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    record["source_candidate_identity"]["source"] = ""
    record["source_candidate_provenance"]["source"] = ""
    assert not _unitary_bundle_completion_evidence_valid(empty_source)

    contradictory_spin = deepcopy(left)
    contradictory_spin[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]["source_candidate_provenance"][
        "irrep_source_provenance"
    ]["source_table_spinor"] = False
    assert not _unitary_bundle_completion_evidence_valid(
        contradictory_spin
    )


def test_generated_tr_unitary_adversarial_mutations_block_promotion():
    export = build_ebr_export_bundle(
        ebr_problem_instances=build_ebr_problem_instances(
            ebr_input_candidates=_orbit_candidates(),
            time_reversal_orbit_report=_exchanged_orbit_report(),
        )
    )
    left = next(
        bundle for bundle in export["bundles"]
        if bundle["problem_kind"] == "unitary_valley_reduced_ebr"
        and bundle["valley"] == "left"
    )
    left["certificate_identity"] = real_primitive_certificate_identity(
        75, "P4", spinor=True
    )
    left["subspace_sg_number"] = 75
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["G", "Q", "QA", "M"],
        "irreps": ["G:g", "Q:q1", "QA:qa2", "M:m"],
        "ebrs": [{"label": "E", "vector": [1, 1, 1, 1]}],
        "provenance": {
            "data_source": "irreptables",
            "package": "irreptables",
            "package_version": "0.0.test",
            "space_group_number": 75,
            "spinful": True,
            "valleyscope_reduction": "sampled_hsp_valley_preserving",
        },
    }
    assert promote_bundle_for_solve(bundle=left, table=table)["promoted"]

    forged_partner = deepcopy(left)
    forged = forged_partner[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    forged["irrep"] = "qa1"
    forged["reviewed_time_reversal_relation"]["target_irrep"] = "qa1"
    forged_partner["irreps_by_kpoint"]["QA"] = ["qa1"]

    forged_evidence = deepcopy(left)
    forged = forged_evidence[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    forged["evidence_valley"] = "left"
    forged["reviewed_time_reversal_relation"]["evidence_valley"] = "left"
    forged["source_candidate_identity"]["valley"] = "left"

    missing_construction = deepcopy(left)
    missing_construction.pop("unitary_vector_construction")

    empty_candidate_source = deepcopy(left)
    source_record = empty_candidate_source[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    source_record["source_candidate_identity"]["source"] = ""
    source_record["source_candidate_provenance"]["source"] = ""

    invalid_candidate_workflow = deepcopy(left)
    workflow_record = invalid_candidate_workflow[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    workflow_record["source_candidate_provenance"]["workflow_path"] = (
        "time_reversal_valley_orbit"
    )

    forged_direct = deepcopy(left)
    forged_direct["workflow_path"] = "direct_qcut"
    forged_direct["valley_orbit"] = []
    forged_direct["time_reversal"] = {}
    forged_direct["unitary_irrep_completion_records_by_hsp"] = {}
    forged_direct["unitary_vector_construction"] = {
        "kind": "direct_observed_unitary_rows",
        "source": "trusted_ebr_input_candidates",
    }
    forged_direct["source_hsp_to_sampled_kpoint"] = {
        hsp: hsp for hsp in forged_direct["expected_hsps"]
    }
    forged_direct["irrep_records_by_kpoint"] = {
        hsp: [{
            "valley": "left",
            "matched_irrep": labels[0],
            "irrep_multiplicity": 1,
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "source": f"forged/{hsp}",
            "irrep_source_provenance": {
                "source_hsp_label": hsp,
                "source_table_spinor": True,
            },
        }]
        for hsp, labels in forged_direct["irreps_by_kpoint"].items()
    }

    for mutated in (
        forged_partner,
        forged_evidence,
        missing_construction,
        empty_candidate_source,
        invalid_candidate_workflow,
        forged_direct,
    ):
        promotion = promote_bundle_for_solve(bundle=mutated, table=table)
        assert not promotion["promoted"]
        assert _blocker_codes(promotion).intersection({
            "unitary_completion_provenance_invalid",
            "unitary_construction_provenance_invalid",
        })


@pytest.mark.parametrize(
    ("problem_kind", "physical_object_kind"),
    [
        (
            "unitary_valley_reduced_ebr",
            "joint_time_reversal_valley_orbit",
        ),
        (
            "valley_orbit_reduced_ebr",
            "unitary_valley_projected_subspace",
        ),
    ],
)
def test_problem_kind_and_physical_object_kind_must_match(
    problem_kind, physical_object_kind,
):
    bundle, table = _reviewed_joint_bundle_and_table()
    forged = deepcopy(bundle)
    forged["problem_kind"] = problem_kind
    forged["physical_object_kind"] = physical_object_kind

    promotion = promote_bundle_for_solve(bundle=forged, table=table)

    assert not promotion["promoted"]
    assert "problem_physical_object_kind_mismatch" in _blocker_codes(
        promotion
    )


def test_directly_observed_dependent_hsp_uses_observed_unitary_binding():
    candidates = _orbit_candidates()
    candidates["candidates"].extend([
        {
            **deepcopy(next(
                row for row in candidates["candidates"]
                if row["valley"] == "left"
                and row["irrep_source_provenance"]["source_hsp_label"] == "Q"
            )),
            "matched_irrep": "qa2",
            "irrep_source_provenance": {
                "source_hsp_label": "QA",
                "source_table_spinor": True,
            },
            "kpoint": "QA_left",
            "source": "fixture/left/QA",
        },
        {
            **deepcopy(next(
                row for row in candidates["candidates"]
                if row["valley"] == "right"
                and row["irrep_source_provenance"]["source_hsp_label"] == "Q"
            )),
            "matched_irrep": "qa1",
            "irrep_source_provenance": {
                "source_hsp_label": "QA",
                "source_table_spinor": True,
            },
            "kpoint": "QA_right",
            "source": "fixture/right/QA",
        },
    ])

    report = _exchanged_orbit_report(candidates)
    problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=report,
    )
    export = build_ebr_export_bundle(ebr_problem_instances=problems)
    unitary = {
        row["valley"]: row for row in export["bundles"]
        if row["problem_kind"] == "unitary_valley_reduced_ebr"
    }

    assert report["status"] == "validated"
    assert len(unitary) == 2
    assert unitary["left"][
        "independent_source_hsp_to_sampled_kpoint"
    ] == {"G": "G_left", "Q": "Q_left", "M": "M_left"}
    assert unitary["left"][
        "observed_source_hsp_to_sampled_kpoint"
    ] == {
        "G": "G_left", "Q": "Q_left", "QA": "QA_left", "M": "M_left",
    }
    dependent = unitary["left"][
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    assert dependent["completion_kind"] == "observed_at_sampled_kpoint"
    assert dependent["sampled_kpoint"] == "QA_left"
    assert dependent["time_reversal_consistency"][
        "evidence_sampled_kpoint"
    ] == "Q_right"
    assert all(
        _unitary_bundle_completion_evidence_valid(row)
        for row in unitary.values()
    )

    fully_observed_without_consistency = deepcopy(unitary["left"])
    fully_observed_without_consistency[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0].pop("time_reversal_consistency")
    assert _unitary_bundle_completion_evidence_valid(
        fully_observed_without_consistency
    )

    independent_binding_mismatch = deepcopy(unitary["left"])
    independent_binding_mismatch[
        "independent_source_hsp_to_sampled_kpoint"
    ]["Q"] = "QA_left"
    independent_binding_mismatch[
        "source_hsp_to_sampled_kpoint"
    ]["Q"] = "QA_left"
    assert not _unitary_bundle_completion_evidence_valid(
        independent_binding_mismatch
    )

    noninjective_consumer = deepcopy(unitary["left"])
    noninjective_consumer[
        "observed_source_hsp_to_sampled_kpoint"
    ]["QA"] = "Q_left"
    dependent = noninjective_consumer[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]
    dependent["sampled_kpoint"] = "Q_left"
    dependent["source_candidate_identity"]["sampled_kpoint"] = "Q_left"
    assert not _unitary_bundle_completion_evidence_valid(
        noninjective_consumer
    )

    noninjective_candidates = deepcopy(candidates)
    for row in noninjective_candidates["candidates"]:
        source_hsp = row["irrep_source_provenance"]["source_hsp_label"]
        if source_hsp == "QA":
            row["kpoint"] = (
                "Q_left" if row["valley"] == "left" else "Q_right"
            )
    noninjective_report = _exchanged_orbit_report(
        noninjective_candidates
    )
    assert noninjective_report["status"] == "blocked"
    assert any(
        blocker.startswith(
            "observed_source_hsp_sampled_kpoint_mapping_noninjective"
        )
        for blocker in noninjective_report["blockers"]
    )


def test_redundant_observed_dependent_hsp_disagreement_blocks_all_dependents():
    candidates = _orbit_candidates()
    dependent = deepcopy(next(
        row for row in candidates["candidates"]
        if row["valley"] == "left"
        and row["irrep_source_provenance"]["source_hsp_label"] == "Q"
    ))
    dependent.update({
        "matched_irrep": "qa1",
        "irrep_source_provenance": {
            "source_hsp_label": "QA",
            "source_table_spinor": True,
        },
        "kpoint": "QA_left",
        "source": "fixture/left/QA",
    })
    candidates["candidates"].append(dependent)

    report = _exchanged_orbit_report(candidates)
    problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=report,
    )

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("time_reversal_multiplicity_or_irrep_mismatch")
        for blocker in report["blockers"]
    )
    assert not any(
        row["canonical_hsp_vector_ready"]
        for row in problems["instances"]
    )
    assert build_ebr_export_bundle(
        ebr_problem_instances=problems
    )["bundle_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_source",
        "missing_workflow_path",
        "invalid_workflow_path",
        "missing_spin_provenance",
    ],
)
def test_candidate_provenance_is_required_for_source_and_dependent_rows(
    mutation,
):
    candidates = _orbit_candidates()
    source = next(
        row for row in candidates["candidates"]
        if row["valley"] == "left"
        and row["irrep_source_provenance"]["source_hsp_label"] == "Q"
    )
    if mutation == "missing_source":
        source.pop("source")
    elif mutation == "missing_workflow_path":
        source.pop("workflow_path")
    elif mutation == "invalid_workflow_path":
        source["workflow_path"] = "time_reversal_valley_orbit"
    else:
        source["irrep_source_provenance"].pop("source_table_spinor")

    report = _exchanged_orbit_report(candidates)
    problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=report,
    )

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("source_candidate_provenance_incomplete:left:Q")
        for blocker in report["blockers"]
    )
    assert not any(
        row["canonical_hsp_vector_ready"]
        for row in problems["instances"]
    )


def test_missing_sampled_evidence_blocks_source_and_dependent_unitary_objects():
    candidates = _orbit_candidates()
    left_q = next(
        row for row in candidates["candidates"]
        if row["valley"] == "left"
        and row["irrep_source_provenance"]["source_hsp_label"] == "Q"
    )
    left_q.pop("kpoint")
    report = _exchanged_orbit_report(candidates)

    problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=report,
    )

    assert report["status"] == "blocked"
    by_kind_and_valley = {
        (row["problem_kind"], row["valley"]): row
        for row in problems["instances"]
    }
    left = by_kind_and_valley[("unitary_valley_reduced_ebr", "left")]
    right = by_kind_and_valley[("unitary_valley_reduced_ebr", "right")]
    joint = by_kind_and_valley[("valley_orbit_reduced_ebr", "")]
    assert left["canonical_hsp_vector_ready"] is False
    assert right["canonical_hsp_vector_ready"] is False
    assert joint["canonical_hsp_vector_ready"] is False
    assert right["unitary_irrep_completion_records_by_hsp"]["QA"][0][
        "readiness_status"
    ] == "blocked"
    assert "source_hsp_sampled_kpoint_missing:left:Q" in right[
        "unitary_irrep_completion_records_by_hsp"
    ]["QA"][0]["blockers"]
    export = build_ebr_export_bundle(ebr_problem_instances=problems)
    assert export["bundle_count"] == 0
    assert export["excluded_count"] == 3


def test_independent_unitary_component_remains_ready_when_dependency_is_absent():
    candidates = _orbit_candidates()
    left_g = next(
        row for row in candidates["candidates"]
        if row["valley"] == "left"
        and row["irrep_source_provenance"]["source_hsp_label"] == "G"
    )
    left_g.pop("kpoint")
    report = _exchanged_orbit_report(candidates)

    problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=report,
    )

    unitary = {
        row["valley"]: row for row in problems["instances"]
        if row["problem_kind"] == "unitary_valley_reduced_ebr"
    }
    joint = next(
        row for row in problems["instances"]
        if row["problem_kind"] == "valley_orbit_reduced_ebr"
    )
    assert unitary["left"]["canonical_hsp_vector_ready"] is False
    assert unitary["right"]["canonical_hsp_vector_ready"] is True
    assert joint["canonical_hsp_vector_ready"] is False
    export = build_ebr_export_bundle(ebr_problem_instances=problems)
    assert [
        (bundle["problem_kind"], bundle["valley"])
        for bundle in export["bundles"]
    ] == [("unitary_valley_reduced_ebr", "right")]


def test_self_mapped_inferred_unitary_arm_requires_antiunitary_sewing():
    candidates = {"candidates": [{
        "valley": "v",
        "matched_irrep": "q",
        "irrep_multiplicity": 1,
        "irrep_source_provenance": {
            "source_hsp_label": "Q",
            "source_table_spinor": False,
        },
        "projected_hsp_classification": {
            "source_hsp_label": "Q",
            "classification": "representative",
            "source_hsp_membership": True,
            "validation_status": "validated",
        },
        "kpoint": "Q_sample",
        "workflow_path": "direct_qcut",
        "source": "fixture/v/Q",
        "subspace_group_candidate": "P1",
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_number": 1,
            "candidate_space_group_symbol": "P1",
        },
        "ready_for_ebr_input": True,
    }]}
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
        source_irrep_orbits_by_valley={
            "v": _self_mapped_nontrim_source_report(),
        },
        grey_source_by_valley={"v": _self_mapped_nontrim_grey_report()},
        ebr_input_candidates=candidates,
        antiunitary_sewing_report=None,
        trusted_projector_provenance_by_kpoint=None,
    )

    orbit = report["valley_orbits"][0]
    inferred = orbit["unitary_valley_irrep_completion_records"]["v"][
        "QA"
    ][0]
    assert orbit["time_reversal_completed_unitary_valley_irreps"]["v"] == {
        "Q": {"q": 1}, "QA": {"qa": 1},
    }
    assert inferred["completion_kind"] == "inferred_by_time_reversal"
    assert inferred["structural_status"] == "validated"
    assert inferred["readiness_status"] == "blocked"
    assert "antiunitary_corepresentation_sewing_not_validated" in inferred[
        "blockers"
    ]
    problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=report,
    )
    unitary = next(
        row for row in problems["instances"]
        if row["problem_kind"] == "unitary_valley_reduced_ebr"
    )
    assert unitary["canonical_hsp_vector_complete"] is True
    assert unitary["canonical_hsp_vector_ready"] is False


def test_cross_valley_tr_completion_rejects_missing_sampled_kpoint_binding():
    source = _synthetic_source_orbit_report()
    grey = _synthetic_grey_report()
    candidates = _orbit_candidates()
    candidates["candidates"][0].pop("kpoint")

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
    assert "source_hsp_sampled_kpoint_missing:left:G" in report["blockers"]
    assert "source_hsp_sampled_kpoint_mapping_incomplete:left:G" in (
        report["blockers"]
    )


def test_cross_valley_tr_completion_rejects_conflicting_component_binding():
    source = _synthetic_source_orbit_report()
    grey = _synthetic_grey_report()
    candidates = _orbit_candidates()
    conflicting = deepcopy(candidates["candidates"][0])
    conflicting["kpoint"] = "G_left_conflict"
    candidates["candidates"].append(conflicting)

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
    assert (
        "source_hsp_sampled_kpoint_mapping_ambiguous:"
        "left:G:G_left:G_left_conflict"
    ) in report["blockers"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time_reversal_hsp_mapping", {}),
        ("irrep_partner_by_label", {}),
        ("independent_hsp_labels", []),
        ("independent_hsp_labels", ["G", ""]),
    ],
)
def test_cross_valley_tr_completion_rejects_empty_or_malformed_source_basis(
    field, value,
):
    source = _synthetic_source_orbit_report()
    source[field] = value
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

    assert report["status"] == "blocked"
    assert "time_reversal_source_mapping_malformed" in report["blockers"]
    assert report["valley_orbits"][0]["irreps_by_kpoint"] == {}


def test_cross_valley_tr_completion_rejects_empty_grey_source_mapping():
    source = _synthetic_source_orbit_report()
    grey = _synthetic_grey_report()
    grey["grey_unitary_restriction_by_irrep"] = {}

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

    assert report["status"] == "blocked"
    assert "grey_group_time_reversal_source_mapping_malformed" in (
        report["blockers"]
    )
    assert report["valley_orbits"][0]["irreps_by_kpoint"] == {}


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


def _self_mapped_pipeline_candidates():
    candidates = _self_mapped_candidates(multiplicity=1)
    candidate = candidates["candidates"][0]
    candidate["irrep_source_provenance"]["source_table_spinor"] = False
    candidate.update({
        "readiness_level": "trusted",
        "subspace_group_candidate": "P1",
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_number": 1,
            "candidate_space_group_symbol": "P1",
        },
    })
    return candidates


def _self_mapped_pipeline_orbit_report(
    *, candidates, sewing_report, projector_provenance,
):
    return build_time_reversal_valley_orbit_report(
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
        source_irrep_orbits_by_valley={"v": _self_mapped_source_report()},
        grey_source_by_valley={
            "v": _self_mapped_grey_report(multiplicity=1),
        },
        ebr_input_candidates=candidates,
        antiunitary_sewing_report=sewing_report,
        trusted_projector_provenance_by_kpoint=projector_provenance,
    )


def test_tr_producer_preserves_complete_untrusted_vector_but_export_blocks_it():
    candidates = _self_mapped_pipeline_candidates()
    sewing = _scalar_self_mapped_sewing_report()
    projector_provenance = _projector_provenance_from_sewing(sewing)

    validated_report = _self_mapped_pipeline_orbit_report(
        candidates=candidates,
        sewing_report=sewing,
        projector_provenance=projector_provenance,
    )
    validated_problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=validated_report,
    )
    validated_instance = validated_problems["instances"][0]
    assert validated_instance["canonical_hsp_vector_complete"] is True
    assert validated_instance["canonical_hsp_vector_ready"] is True
    validated_export = build_ebr_export_bundle(
        ebr_problem_instances=validated_problems,
    )
    assert validated_export["bundle_count"] == 2
    assert {row["problem_kind"] for row in validated_export["bundles"]} == {
        "unitary_valley_reduced_ebr", "valley_orbit_reduced_ebr",
    }

    untrusted_report = _self_mapped_pipeline_orbit_report(
        candidates=candidates,
        sewing_report=None,
        projector_provenance=projector_provenance,
    )
    untrusted_orbit = untrusted_report["valley_orbits"][0]
    assert untrusted_report["status"] == "blocked"
    assert untrusted_orbit["structural_blockers"] == []
    assert untrusted_orbit["readiness_blockers"] == [
        "antiunitary_corepresentation_sewing_not_validated",
    ]
    assert untrusted_orbit["grey_irrep_multiplicities_by_hsp"] == {
        "G": {"g_corep": 1},
    }
    assert untrusted_orbit["irreps_by_kpoint"] == {"G": ["g_corep"]}
    assert "antiunitary_corepresentation_sewing_not_validated" in (
        untrusted_orbit["blockers"]
    )

    untrusted_problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=untrusted_report,
    )
    untrusted_instance = untrusted_problems["instances"][0]
    assert untrusted_instance["status"] == (
        "canonical_hsp_vector_complete_but_untrusted"
    )
    assert untrusted_instance["canonical_hsp_vector_complete"] is True
    assert untrusted_instance["canonical_hsp_vector_ready"] is False
    untrusted_export = build_ebr_export_bundle(
        ebr_problem_instances=untrusted_problems,
    )
    assert untrusted_export["bundle_count"] == 1
    assert untrusted_export["bundles"][0]["problem_kind"] == (
        "unitary_valley_reduced_ebr"
    )
    assert untrusted_export["excluded_instances"][0][
        "problem_kind"
    ] == "valley_orbit_reduced_ebr"
    assert untrusted_export["excluded_instances"][0][
        "canonical_hsp_vector_complete"
    ] is True


def test_tr_producer_keeps_structurally_incomplete_vector_incomplete():
    candidates = _self_mapped_pipeline_candidates()
    candidates["candidates"][0]["matched_irrep"] = ""
    sewing = _scalar_self_mapped_sewing_report()
    report = _self_mapped_pipeline_orbit_report(
        candidates=candidates,
        sewing_report=sewing,
        projector_provenance=_projector_provenance_from_sewing(sewing),
    )

    orbit = report["valley_orbits"][0]
    assert "missing_trusted_independent_hsp:v:G" in orbit[
        "structural_blockers"
    ]
    assert orbit["grey_irrep_multiplicities_by_hsp"] == {}
    assert orbit["irreps_by_kpoint"] == {}
    assert "missing_trusted_independent_hsp:v:G" in orbit["blockers"]
    problems = build_ebr_problem_instances(
        ebr_input_candidates=candidates,
        time_reversal_orbit_report=report,
    )
    instance = problems["instances"][0]
    assert instance["status"] == "incomplete_canonical_hsp_vector"
    assert instance["canonical_hsp_vector_complete"] is False
    assert instance["canonical_hsp_vector_ready"] is False
    assert build_ebr_export_bundle(
        ebr_problem_instances=problems,
    )["bundle_count"] == 0
