from __future__ import annotations

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

import valleyscope.irreps.time_reversal_source as _tr_source_module
import valleyscope.analysis.tr_irrep_completion as _tr_completion_module
import valleyscope.analysis.ebr_problem_instances as _problem_module
from valleyscope.analysis.scoped_representation_evidence import (
    build_scoped_representation_evidence,
)
from valleyscope.analysis.ebr_problem_instances import (
    build_ebr_problem_instances as _build_ebr_problem_instances,
)
from valleyscope.analysis.tr_irrep_completion import (
    attach_tr_irrep_completion_certificates as _attach_tr_irrep_completion_certificates,
    validate_tr_irrep_completion_certificate as _validate_tr_irrep_completion_certificate,
)
from valleyscope.analysis.unitary_provenance import (
    validate_tr_completed_unitary_bundle,
)
from valleyscope.io.wavefunction_convention import canonical_identity
from valleyscope.irreps.tables import ReviewedSourceIrrep
from valleyscope.irreps.time_reversal_source import (
    derive_time_reversal_source_irrep_orbits,
    validate_reviewed_time_reversal_source_context as _validate_reviewed_time_reversal_source_context,
)
from tests.test_scoped_representation_evidence import _raw_inputs
from tests.test_database_ingestion import _tr_completed_unitary_bundle


def _fixture_source_context_validator(context, **_kwargs):
    return _validate_reviewed_time_reversal_source_context(
        context,
        require_reviewed_table=False,
    )


def validate_tr_irrep_completion_certificate(*args, **kwargs):
    """Validate structural fixtures through an explicit test-only source path."""
    with patch.object(
        _tr_source_module,
        "validate_reviewed_time_reversal_source_context",
        _fixture_source_context_validator,
    ), patch.object(
        _tr_completion_module,
        "_source_context_matches_source_irrep",
        lambda _context, _source_irrep: True,
    ):
        return _validate_tr_irrep_completion_certificate(*args, **kwargs)


def attach_tr_irrep_completion_certificates(**kwargs):
    """Exercise structural synthetic rows without weakening production trust."""

    with patch.object(
        _tr_source_module,
        "validate_reviewed_time_reversal_source_context",
        _fixture_source_context_validator,
    ), patch.object(
        _tr_completion_module,
        "_source_context_matches_source_irrep",
        lambda _context, _source_irrep: True,
    ):
        return _attach_tr_irrep_completion_certificates(**kwargs)


def build_ebr_problem_instances(**kwargs):
    """Keep synthetic structural fixtures outside production trust claims."""

    def fixture_certificate_validator(*args, **call_kwargs):
        return validate_tr_irrep_completion_certificate(
            *args,
            **call_kwargs,
        )

    with patch.object(
        _problem_module,
        "validate_tr_irrep_completion_certificate",
        fixture_certificate_validator,
    ):
        return _build_ebr_problem_instances(**kwargs)


def _setting_certificate() -> dict[str, object]:
    return {
        "schema_version": "test",
        "validation_status": "validated",
        "hall_number": 1,
        "hall_symbol": "P 1",
        "centering_vectors": [[0.0, 0.0, 0.0]],
    }


def _reviewed_source_report() -> dict[str, object]:
    inventory = canonical_identity({
        "fixture": "reviewed_tr_source_operations",
    })
    rows = [
        ReviewedSourceIrrep(
            label=label,
            kpoint_label=hsp,
            k_frac=np.asarray(k_frac, dtype=float),
            dimension=1,
            characters={1: character},
            operation_indices=(1,),
            operation_inventory_identity=inventory,
            spinor=True,
            spin_convention="double_group_spinor",
            source_table="fixture_standard_irrep_table",
            source_table_status="reviewed_fixture",
            source_provenance="fixture.StandardIrrepTable",
        )
        for label, hsp, k_frac, character in (
            ("K1", "K", [0.25, 0.0, 0.0], 0.0 + 1.0j),
            ("K2", "K", [0.25, 0.0, 0.0], 0.5 + 0.5j),
            ("KA1", "KA", [-0.25, 0.0, 0.0], 0.0 - 1.0j),
            ("KA2", "KA", [-0.25, 0.0, 0.0], 0.5 - 0.5j),
        )
    ]
    return derive_time_reversal_source_irrep_orbits(
        reviewed_rows=rows,
        centering_vectors=[[0.0, 0.0, 0.0]],
        source_table_identity={
            "space_group_number": 1,
            "space_group_symbol": "P1",
            "source_table_name": "P1",
            "source_table_provenance": "fixture.StandardIrrepTable",
            "spinor": True,
        },
        standard_setting_certificate=_setting_certificate(),
    )


def _reviewed_source_identity() -> dict[str, object]:
    report = _reviewed_source_report()
    content = {
        "operation_inventory_identity": report[
            "operation_inventory_identity"
        ],
        "spin_convention": report["spin_convention"],
        "hsp_involution": report["time_reversal_hsp_mapping"],
        "irrep_pairing": report["irrep_partner_by_label"],
    }
    return {**content, "identity": canonical_identity(content)}


def _local_context(
    *,
    valley: str,
    sampled_kpoint: str,
) -> tuple[dict[str, object], dict[str, object]]:
    raw_inputs = _raw_inputs(
        scope_kind="local_irrep",
        required_operation_ids=(2,),
    )
    raw_inputs["kpoint_label"] = sampled_kpoint
    raw_inputs["source_valleys"] = (valley,)
    raw_inputs["valley_orbit"] = ("left", "right")
    raw_inputs["projectors"] = {
        "left": raw_inputs["projectors"]["v0"],
        "right": raw_inputs["projectors"]["v1"],
    }
    raw_inputs["valley_bases"] = {
        "left": raw_inputs["valley_bases"]["v0"],
        "right": raw_inputs["valley_bases"]["v1"],
    }
    raw_inputs["valley_mappings"] = {
        2: {"left": "left", "right": "right"},
        5: {"left": "right", "right": "left"},
    }
    record = build_scoped_representation_evidence(
        **raw_inputs
    ).to_record()
    assert record["status"] == "passed"
    links = {
        "spinor_source_basis_certificate_identity": record[
            "source_basis_certificate_identity"
        ],
        "double_space_group_lift_certificate_identity": record[
            "double_space_group_lift_certificate_identity"
        ],
        "scoped_representation_evidence_identity": record[
            "evidence_identity"
        ],
    }
    return {
        "record": record,
        "raw_inputs": raw_inputs,
        "standard_setting_certificate": _setting_certificate(),
    }, links


def _candidate_provenance(
    *,
    valley: str,
    source_hsp: str,
    sampled_kpoint: str,
    irrep: str,
    cprime: dict[str, object],
) -> dict[str, object]:
    setting_certificate = _setting_certificate()
    return {
        "source": f"fixture/{valley}/{source_hsp}",
        "workflow_path": "direct_qcut",
        "irrep_source_provenance": {
            "matching_strategy": "bilbao_restricted_character",
            "subspace_space_group_number": 1,
            "subspace_space_group_symbol": "P1",
            "source_table_sg_number": 1,
            "source_table_name": "P1",
            "source_table_spinor": True,
            "source_hsp_label": source_hsp,
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": setting_certificate,
            },
            "cprime": deepcopy(cprime),
        },
    }


def _observed_record(
    *,
    valley: str,
    source_hsp: str,
    sampled_kpoint: str,
    irrep: str,
    cprime: dict[str, object],
) -> dict[str, object]:
    source = f"fixture/{valley}/{source_hsp}"
    return {
        "completion_kind": "observed_at_sampled_kpoint",
        "target_valley": valley,
        "target_source_hsp_label": source_hsp,
        "irrep": irrep,
        "multiplicity": 1,
        "evidence_valley": valley,
        "evidence_source_hsp_label": source_hsp,
        "evidence_sampled_kpoint": sampled_kpoint,
        "sampled_kpoint": sampled_kpoint,
        "source_candidate_identity": {
            "source": source,
            "workflow_path": "direct_qcut",
            "valley": valley,
            "source_hsp_label": source_hsp,
            "sampled_kpoint": sampled_kpoint,
            "irrep": irrep,
            "multiplicity": 1,
        },
        "source_candidate_provenance": _candidate_provenance(
            valley=valley,
            source_hsp=source_hsp,
            sampled_kpoint=sampled_kpoint,
            irrep=irrep,
            cprime=cprime,
        ),
        "structural_status": "validated",
        "readiness_status": "trusted",
        "blockers": [],
    }


def _inferred_record(
    *,
    observed: dict[str, object],
    target_valley: str,
    target_hsp: str,
    target_irrep: str,
) -> dict[str, object]:
    return {
        "completion_kind": "inferred_by_time_reversal",
        "target_valley": target_valley,
        "target_source_hsp_label": target_hsp,
        "irrep": target_irrep,
        "multiplicity": 1,
        "evidence_valley": observed["target_valley"],
        "evidence_source_hsp_label": observed[
            "target_source_hsp_label"
        ],
        "evidence_sampled_kpoint": observed["sampled_kpoint"],
        "reviewed_time_reversal_relation": {
            "evidence_valley": observed["target_valley"],
            "target_valley": target_valley,
            "evidence_source_hsp_label": observed[
                "target_source_hsp_label"
            ],
            "target_source_hsp_label": target_hsp,
            "evidence_irrep": observed["irrep"],
            "target_irrep": target_irrep,
        },
        "source_candidate_identity": deepcopy(
            observed["source_candidate_identity"]
        ),
        "source_candidate_provenance": deepcopy(
            observed["source_candidate_provenance"]
        ),
        "structural_status": "validated",
        "readiness_status": "trusted",
        "blockers": [],
    }


def _orbit_and_context():
    reviewed_source = _reviewed_source_report()
    left_context, left_cprime = _local_context(
        valley="left",
        sampled_kpoint="K_left",
    )
    right_context, right_cprime = _local_context(
        valley="right",
        sampled_kpoint="K_right",
    )
    left_observed = _observed_record(
        valley="left",
        source_hsp="K",
        sampled_kpoint="K_left",
        irrep="K1",
        cprime=left_cprime,
    )
    right_observed = _observed_record(
        valley="right",
        source_hsp="K",
        sampled_kpoint="K_right",
        irrep="K2",
        cprime=right_cprime,
    )
    records = {
        "left": {
            "K": [left_observed],
            "KA": [
                _inferred_record(
                    observed=right_observed,
                    target_valley="left",
                    target_hsp="KA",
                    target_irrep="KA2",
                )
            ],
        },
        "right": {
            "K": [right_observed],
            "KA": [
                _inferred_record(
                    observed=left_observed,
                    target_valley="right",
                    target_hsp="KA",
                    target_irrep="KA1",
                )
            ],
        },
    }
    report = {
        "status": "validated",
        "enabled": True,
        "theta_square": -1,
        "time_reversal_valley_mapping": {
            "left": "right",
            "right": "left",
        },
        "valley_orbits": [
            {
                "orbit_id": "time_reversal_valley_orbit_001",
                "representative": "left",
                "members": ["left", "right"],
                "mapping_type": "exchanged",
                "status": "validated",
                "unitary_completion_status": "validated",
                "unitary_completion_blockers": [],
                "joint_corepresentation_status": "blocked",
                "joint_corepresentation_blockers": [
                    "joint_time_reversal_corepresentation_not_certified"
                ],
                "unitary_valley_irrep_completion_records": records,
                "time_reversal_hsp_orbits": [
                    {
                        "representative": "K",
                        "members": ["K", "KA"],
                        "self_mapped": False,
                    }
                ],
                "time_reversal_irrep_pairing": {
                    "K1": "KA1",
                    "KA1": "K1",
                    "K2": "KA2",
                    "KA2": "K2",
                },
                "reviewed_time_reversal_source_identity": (
                    _reviewed_source_identity()
                ),
                "reviewed_time_reversal_source_context": reviewed_source[
                    "reviewed_source_context"
                ],
                "time_reversal_completed_unitary_valley_irreps": {
                    "left": {
                        "K": {"K1": 1},
                        "KA": {"KA2": 1},
                    },
                    "right": {
                        "K": {"K2": 1},
                        "KA": {"KA1": 1},
                    },
                },
                "full_unitary_source_hsp_labels": ["K", "KA"],
                "independent_time_reversal_hsp_labels": ["K"],
                "expected_hsps": ["K"],
                "irreps_by_kpoint": {"K": ["joint"]},
                "source_hsp_to_sampled_kpoint": {"K": "K_left"},
                "source_hsp_to_sampled_kpoint_by_valley": {
                    "left": {"K": "K_left"},
                    "right": {"K": "K_right"},
                },
                "independent_source_hsp_to_sampled_kpoint_by_valley": {
                    "left": {"K": "K_left"},
                    "right": {"K": "K_right"},
                },
                "observed_source_hsp_to_sampled_kpoint_by_valley": {
                    "left": {"K": "K_left"},
                    "right": {"K": "K_right"},
                },
                "readiness_blockers": [],
                "blockers": [],
            }
        ],
        "blockers": [],
    }
    context = {
        left_cprime["scoped_representation_evidence_identity"]:
            left_context,
        right_cprime["scoped_representation_evidence_identity"]:
            right_context,
    }
    return report, context, left_cprime, right_cprime


def _input_candidates(report: dict[str, object]) -> dict[str, object]:
    records = report["valley_orbits"][0][
        "unitary_valley_irrep_completion_records"
    ]
    candidates = []
    for valley in ("left", "right"):
        observed = records[valley]["K"][0]
        source_provenance = observed["source_candidate_provenance"][
            "irrep_source_provenance"
        ]
        candidates.append({
            "kpoint": observed["sampled_kpoint"],
            "valley": valley,
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "subspace_group_candidate": "P1",
            "subspace_space_group": {
                "status": "resolved",
                "candidate_space_group_number": 1,
                "candidate_space_group_symbol": "P1",
            },
            "matching_strategy": "bilbao_restricted_character",
            "matched_irrep": observed["irrep"],
            "irrep_multiplicity": 1,
            "irrep_source_provenance": deepcopy(source_provenance),
            "source": observed["source_candidate_identity"]["source"],
            "ready_for_ebr_input": True,
        })
    return {
        "status": "has_candidates",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def test_completion_keeps_observed_local_cprime_and_certifies_only_inferred_rows():
    report, context, left_cprime, right_cprime = _orbit_and_context()

    completed = attach_tr_irrep_completion_certificates(
        time_reversal_orbit_report=report,
        cprime_validation_context=context,
    )

    orbit = completed["valley_orbits"][0]
    records = orbit["unitary_valley_irrep_completion_records"]
    assert records["left"]["K"][0]["source_candidate_provenance"][
        "irrep_source_provenance"
    ]["cprime"] == left_cprime
    assert records["right"]["K"][0]["source_candidate_provenance"][
        "irrep_source_provenance"
    ]["cprime"] == right_cprime
    assert "tr_irrep_completion_certificate" not in records["left"]["K"][0]
    assert "tr_irrep_completion_certificate" not in records["right"]["K"][0]

    left_inferred = records["left"]["KA"][0]
    right_inferred = records["right"]["KA"][0]
    for inferred, source_cprime in (
        (left_inferred, right_cprime),
        (right_inferred, left_cprime),
    ):
        certificate = inferred["tr_irrep_completion_certificate"]
        content = {
            key: value
            for key, value in certificate.items()
            if key != "certificate_identity"
        }
        assert certificate["certificate_kind"] == (
            "exact_tr_irrep_completion"
        )
        assert certificate["status"] == "passed"
        assert certificate["observed_source"]["local_cprime_identity"] == (
            source_cprime
        )
        assert certificate["certificate_identity"] == canonical_identity(
            content
        )

    assert all(
        entry["record"]["scope"]["scope_kind"] == "local_irrep"
        for entry in context.values()
    )


def _set_nested(
    target: dict[str, object],
    path: tuple[str, ...],
    value: object,
) -> None:
    cursor = target
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "tampered"),
        (("certificate_kind",), "numerical_sewing"),
        (("status",), "blocked"),
        (("observed_source", "candidate_identity", "source"), "tampered"),
        (
            (
                "observed_source",
                "local_cprime_identity",
                "scoped_representation_evidence_identity",
            ),
            "sha256:" + "0" * 64,
        ),
        (("observed_source", "valley"), "left"),
        (("observed_source", "source_hsp_label"), "GM"),
        (("observed_source", "sampled_kpoint"), "other"),
        (("observed_source", "irrep"), "K9"),
        (("observed_source", "multiplicity"), 2),
        (("inferred_target", "valley"), "right"),
        (("inferred_target", "source_hsp_label"), "GM"),
        (("inferred_target", "irrep"), "KA9"),
        (("inferred_target", "multiplicity"), 2),
        (
            ("reviewed_time_reversal", "valley_involution", "left"),
            "left",
        ),
        (
            ("reviewed_time_reversal", "hsp_involution", "K"),
            "K",
        ),
        (
            ("reviewed_time_reversal", "irrep_pairing", "K2"),
            "K2",
        ),
        (
            (
                "reviewed_time_reversal",
                "source_table_identity",
                "source_table_name",
            ),
            "tampered",
        ),
        (
            (
                "reviewed_time_reversal",
                "source_model_identity",
                "operation_inventory_identity",
            ),
            "sha256:" + "0" * 64,
        ),
        (
            (
                "reviewed_time_reversal",
                "source_model_identity",
                "spin_convention",
            ),
            "tampered",
        ),
        (
            (
                "reviewed_time_reversal",
                "source_model_identity",
                "hsp_involution",
                "K",
            ),
            "K",
        ),
        (
            (
                "reviewed_time_reversal",
                "source_model_identity",
                "irrep_pairing",
                "K2",
            ),
            "K2",
        ),
        (
            (
                "reviewed_time_reversal",
                "source_model_identity",
                "identity",
            ),
            "sha256:" + "0" * 64,
        ),
        (
            ("standard_setting_certificate_identity",),
            "sha256:" + "0" * 64,
        ),
        (("producer_context_identity",), "sha256:" + "0" * 64),
        (
            ("supported_parent_profile", "profile_identity"),
            "tampered",
        ),
        (
            (
                "supported_parent_profile",
                "profile_assumptions",
                "nonmagnetic",
            ),
            False,
        ),
        (
            ("supported_parent_profile", "profile_assumptions", "soc"),
            False,
        ),
        (
            (
                "supported_parent_profile",
                "profile_assumptions",
                "time_reversal",
            ),
            False,
        ),
        (
            (
                "supported_parent_profile",
                "profile_assumptions",
                "saxis_cart",
            ),
            [1.0, 0.0, 0.0],
        ),
        (
            (
                "supported_parent_profile",
                "spinor_source_basis_certificate_identity",
            ),
            "sha256:" + "0" * 64,
        ),
        (("certificate_identity",), "sha256:" + "0" * 64),
    ],
)
def test_completion_certificate_rejects_tampering_of_every_bound_field(
    path: tuple[str, ...],
    value: object,
):
    report, context, _, _ = _orbit_and_context()
    completed = attach_tr_irrep_completion_certificates(
        time_reversal_orbit_report=report,
        cprime_validation_context=context,
    )
    orbit = completed["valley_orbits"][0]
    record = orbit["unitary_valley_irrep_completion_records"]["left"][
        "KA"
    ][0]
    certificate = deepcopy(record["tr_irrep_completion_certificate"])
    assert validate_tr_irrep_completion_certificate(
        certificate,
        completion_record=record,
        valley_mapping=completed["time_reversal_valley_mapping"],
        hsp_mapping={"K": "KA", "KA": "K"},
        irrep_pairing=orbit["time_reversal_irrep_pairing"],
        reviewed_source_identity=(
            orbit["reviewed_time_reversal_source_identity"]
        ),
        reviewed_source_context=orbit[
            "reviewed_time_reversal_source_context"
        ],
        cprime_validation_context=context,
    )

    _set_nested(certificate, path, value)
    if path != ("certificate_identity",):
        content = {
            key: item
            for key, item in certificate.items()
            if key != "certificate_identity"
        }
        certificate["certificate_identity"] = canonical_identity(content)

    assert not validate_tr_irrep_completion_certificate(
        certificate,
        completion_record=record,
        valley_mapping=completed["time_reversal_valley_mapping"],
        hsp_mapping={"K": "KA", "KA": "K"},
        irrep_pairing=orbit["time_reversal_irrep_pairing"],
        reviewed_source_identity=(
            orbit["reviewed_time_reversal_source_identity"]
        ),
        reviewed_source_context=orbit[
            "reviewed_time_reversal_source_context"
        ],
        cprime_validation_context=context,
    )


@pytest.mark.parametrize("mutation", ["sample_binding", "setting_binding"])
def test_completion_certificate_rejects_coordinated_tampering_against_producer_context(
    mutation: str,
):
    report, context, _, _ = _orbit_and_context()
    completed = attach_tr_irrep_completion_certificates(
        time_reversal_orbit_report=report,
        cprime_validation_context=context,
    )
    orbit = completed["valley_orbits"][0]
    record = deepcopy(
        orbit["unitary_valley_irrep_completion_records"]["left"]["KA"][0]
    )
    certificate = record["tr_irrep_completion_certificate"]
    if mutation == "sample_binding":
        record["evidence_sampled_kpoint"] = "forged_sample"
        record["source_candidate_identity"]["sampled_kpoint"] = (
            "forged_sample"
        )
        certificate["observed_source"]["sampled_kpoint"] = "forged_sample"
        certificate["observed_source"]["candidate_identity"][
            "sampled_kpoint"
        ] = "forged_sample"
    else:
        setting = record["source_candidate_provenance"][
            "irrep_source_provenance"
        ]["standard_setting_hsp_mapping"][
            "standard_setting_certificate"
        ]
        setting["hall_symbol"] = "forged"
        certificate["standard_setting_certificate_identity"] = (
            canonical_identity(setting)
        )
    certificate["certificate_identity"] = canonical_identity({
        key: value
        for key, value in certificate.items()
        if key != "certificate_identity"
    })

    assert not validate_tr_irrep_completion_certificate(
        certificate,
        completion_record=record,
        valley_mapping=completed["time_reversal_valley_mapping"],
        hsp_mapping={"K": "KA", "KA": "K"},
        irrep_pairing=orbit["time_reversal_irrep_pairing"],
        reviewed_source_identity=(
            orbit["reviewed_time_reversal_source_identity"]
        ),
        reviewed_source_context=orbit[
            "reviewed_time_reversal_source_context"
        ],
        cprime_validation_context=context,
    )


def test_unitary_bundle_rejects_inferred_row_without_exact_completion_certificate():
    bundle = _tr_completed_unitary_bundle()

    assert bundle["unitary_irrep_completion_records_by_hsp"]["KA"][0][
        "completion_kind"
    ] == "inferred_by_time_reversal"
    assert "tr_irrep_completion_certificate" not in bundle[
        "unitary_irrep_completion_records_by_hsp"
    ]["KA"][0]
    assert not validate_tr_completed_unitary_bundle(bundle)


def test_problem_builder_promotes_only_certified_unitary_components():
    report, context, left_cprime, right_cprime = _orbit_and_context()
    completed = attach_tr_irrep_completion_certificates(
        time_reversal_orbit_report=report,
        cprime_validation_context=context,
    )

    problems = build_ebr_problem_instances(
        ebr_input_candidates=_input_candidates(completed),
        time_reversal_orbit_report=completed,
    )

    unitary = [
        row
        for row in problems["instances"]
        if row["problem_kind"] == "unitary_valley_reduced_ebr"
    ]
    joint = [
        row
        for row in problems["instances"]
        if row["problem_kind"] == "valley_orbit_reduced_ebr"
    ]
    assert len(unitary) == 2
    assert all(row["canonical_hsp_vector_ready"] is True for row in unitary)
    assert len(joint) == 1
    assert joint[0]["canonical_hsp_vector_ready"] is False
    assert "joint_time_reversal_corepresentation_not_certified" in joint[0][
        "blocked_by"
    ]
    by_valley = {row["valley"]: row for row in unitary}
    assert by_valley["left"]["cprime_identity_by_kpoint"] == {
        "K": left_cprime,
        "KA": right_cprime,
    }
    assert by_valley["right"]["cprime_identity_by_kpoint"] == {
        "K": right_cprime,
        "KA": left_cprime,
    }
