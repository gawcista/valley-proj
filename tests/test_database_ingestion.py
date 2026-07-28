import json
import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.io.wavefunction_convention import canonical_identity
from valleyscope.workflows.analyze_hsp import analyze_hsp

import valleyscope.analysis.database_ingestion_record as _ingestion_module
import valleyscope.analysis.tr_irrep_completion as _tr_completion_module
import valleyscope.irreps.time_reversal_source as _tr_source_module
from valleyscope.analysis.database_ingestion_record import (
    build_database_ingestion_record as _build_database_ingestion_record,
    load_database_ingestion_record_from_directory,
)
from valleyscope.analysis.tr_irrep_completion import (
    attach_tr_irrep_completion_certificates as _attach_tr_irrep_completion_certificates,
)
from valleyscope.analysis.unitary_provenance import (
    unitary_bundle_claims_time_reversal_completion,
    validate_tr_completed_unitary_bundle as _validate_tr_completed_unitary_bundle,
)
from valleyscope.irreps.tables import ReviewedSourceIrrep
from valleyscope.irreps.time_reversal_source import (
    derive_time_reversal_source_irrep_orbits,
    validate_reviewed_time_reversal_source_context as _validate_reviewed_time_reversal_source_context,
)

from tests.helpers_io_workflow import write_fixture, write_config
from tests.reduced_ebr_promo_helpers import (
    attach_cprime_fixture_contract,
    cprime_validation_context_for_export,
    cprime_summary_for_export,
)


def _fixture_source_context_validator(context, **_kwargs):
    return _validate_reviewed_time_reversal_source_context(
        context,
        require_reviewed_table=False,
    )


def build_database_ingestion_record(**kwargs):
    production_validator = (
        _ingestion_module.validate_unitary_bundle_provenance
    )

    def fixture_validator(bundle):
        if unitary_bundle_claims_time_reversal_completion(bundle):
            return _validate_tr_completed_unitary_bundle(
                bundle,
                require_reviewed_table=False,
            )
        return production_validator(bundle)

    with patch.object(
        _ingestion_module,
        "validate_unitary_bundle_provenance",
        fixture_validator,
    ), patch.object(
        _tr_source_module,
        "validate_reviewed_time_reversal_source_context",
        _fixture_source_context_validator,
    ), patch.object(
        _tr_completion_module,
        "_source_context_matches_source_irrep",
        lambda _context, _source_irrep: True,
    ):
        return _build_database_ingestion_record(**kwargs)


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


def _reviewed_tr_source_report() -> dict[str, object]:
    inventory = canonical_identity({
        "fixture": "database_ingestion_reviewed_operations",
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
            source_provenance="irreptables.StandardIrrepTable",
        )
        for label, hsp, k_frac, character in (
            ("A", "GM", [0.0, 0.0, 0.0], 1.0 + 0.0j),
            ("B", "K", [1.0 / 3.0, 1.0 / 3.0, 0.0], 0.0 + 1.0j),
            ("Bp", "KA", [-1.0 / 3.0, -1.0 / 3.0, 0.0], 0.0 - 1.0j),
        )
    ]
    return derive_time_reversal_source_irrep_orbits(
        reviewed_rows=rows,
        centering_vectors=[[0.0, 0.0, 0.0]],
        source_table_identity={
            "space_group_number": 143,
            "space_group_symbol": "P3",
            "source_table_name": "P3",
            "source_table_provenance": "irreptables.StandardIrrepTable",
            "spinor": True,
        },
        standard_setting_certificate={
            "schema_version": "1.0.0",
            "validation_status": "validated",
            "fixture": "database_ingestion_unitary",
            "centering_vectors": [[0.0, 0.0, 0.0]],
        },
    )


def _reviewed_tr_source_identity() -> dict[str, object]:
    report = _reviewed_tr_source_report()
    content = {
        "operation_inventory_identity": report[
            "operation_inventory_identity"
        ],
        "spin_convention": report["spin_convention"],
        "hsp_involution": report["time_reversal_hsp_mapping"],
        "irrep_pairing": report["irrep_partner_by_label"],
    }
    return {**content, "identity": canonical_identity(content)}


def _cprime_export(*bundles):
    export = {"bundles": list(bundles)}
    attach_cprime_fixture_contract(export)
    context = cprime_validation_context_for_export(export)
    for bundle in export["bundles"]:
        construction = bundle.get("unitary_vector_construction")
        if (
            not isinstance(construction, dict)
            or construction.get("kind")
            != "time_reversal_completed_unitary_rows"
        ):
            continue
        evidence = bundle.get("time_reversal")
        records = bundle.get(
            "unitary_irrep_completion_records_by_hsp"
        )
        valley = bundle.get("valley")
        if (
            not isinstance(evidence, dict)
            or not isinstance(records, dict)
            or not isinstance(valley, str)
        ):
            continue
        report = {
            "enabled": True,
            "status": "validated",
            "time_reversal_valley_mapping": evidence.get(
                "time_reversal_valley_mapping", {}
            ),
            "valley_orbits": [{
                "status": "validated",
                "unitary_completion_status": "validated",
                "unitary_completion_blockers": [],
                "blockers": [],
                "mapping_type": evidence.get("mapping_type"),
                "time_reversal_hsp_orbits": evidence.get(
                    "time_reversal_hsp_orbits", []
                ),
                "time_reversal_irrep_pairing": evidence.get(
                    "time_reversal_irrep_pairing", {}
                ),
                "reviewed_time_reversal_source_identity": evidence.get(
                    "reviewed_time_reversal_source_identity", {}
                ),
                "reviewed_time_reversal_source_context": evidence.get(
                    "reviewed_time_reversal_source_context", {}
                ),
                "unitary_valley_irrep_completion_records": {
                    valley: records,
                },
            }],
        }
        completed = attach_tr_irrep_completion_certificates(
            time_reversal_orbit_report=report,
            cprime_validation_context=dict(context["_by_identity"]),
        )
        bundle["unitary_irrep_completion_records_by_hsp"] = (
            completed["valley_orbits"][0][
                "unitary_valley_irrep_completion_records"
            ][valley]
        )
    return export

# Database ingestion record tests
# -----------------------------------------------------------------------

def test_ingestion_record_requires_summary():
    """Missing valley_summary.json produces invalid record."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    record = build_database_ingestion_record()
    assert record["record_status"] == "invalid_missing_summary"
    assert len(record["validation_errors"]) > 0


def test_ingestion_uses_stage_owned_counts_and_final_result_status():
    export, mapping = _authoritative_unitary_ingestion_payload(
        bundle_id="b_candidate",
        solution_over={
            "status": "no_exact_solution",
            "classification": "in_integer_span_no_nonnegative_witness",
            "nonnegative_solution_status": "no_nonnegative_solution",
        },
    )
    export.update({
        "status": "partial_export",
        "excluded_instances": [{
            "source_instance_id": "i_blocked",
            "status": "canonical_hsp_vector_complete_but_untrusted",
            "canonical_hsp_vector_complete": True,
            "canonical_hsp_vector_ready": False,
            "exclusion_reasons": ["source_hsp_coverage_not_ready"],
        }],
    })
    mapping.update({
        "status": "partial",
        "excluded_bundles": [{
            "bundle_id": "b_other",
            "reason": "validation blocked",
            "blocker_reasons": [{"code": "certificate_unresolved"}],
        }],
    })

    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(
            export, target_kpoints=["GammaM"], iband=[1]
        ),
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )

    assert record["record_status"] == "has_final_reduced_ebr_results"
    assert record["reduced_table_validation_candidate_bundle_count"] == 1
    assert record["final_reduced_ebr_result_count"] == 1
    assert record["final_mapping_excluded_bundle_count"] == 1
    assert record["input_excluded_instance_count"] == 1
    assert len(record["valley_irrep_records"]) == 3
    assert record["reduced_ebr_records"][0]["status"] == "no_exact_solution"
    assert record["input_excluded_ebr_records"][0][
        "canonical_hsp_vector_complete"
    ] is True
    assert record["input_excluded_ebr_records"][0][
        "canonical_hsp_vector_ready"
    ] is False
    assert record["final_mapping_excluded_records"][0]["bundle_id"] == (
        "b_other"
    )
    for removed in (
        "ready_bundle_count",
        "validation_candidate_count",
        "decomposition_ready_count",
        "excluded_bundle_count",
        "excluded_ebr_records",
    ):
        assert removed not in record


def test_ingestion_candidate_without_mapping_has_candidate_status():
    export = _cprime_export({
        "bundle_id": "b",
        "valley": "K",
        "ready_for_reduced_table_validation": True,
        "irreps_by_kpoint": {},
        "irrep_records_by_kpoint": {},
    })
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(export),
        valley_ebr_export_bundle=export,
    )

    assert record["record_status"] == (
        "has_reduced_table_validation_candidates"
    )
    assert record["reduced_table_validation_candidate_bundle_count"] == 1
    assert record["final_reduced_ebr_result_count"] == 0


@pytest.mark.parametrize(
    ("solution_status", "classification", "search_status"),
    [
        (
            "no_exact_solution",
            "in_integer_span_no_nonnegative_witness",
            None,
        ),
        (
            "indeterminate_truncated",
            "indeterminate_truncated",
            "truncated_by_max_coefficient",
        ),
    ],
)
def test_evaluated_nonexact_solution_is_a_final_result(
    solution_status, classification, search_status,
):
    solution = {
        "bundle_id": "b",
        "status": solution_status,
        "classification": classification,
    }
    if search_status is not None:
        solution["search_status"] = search_status
    export, mapping = _authoritative_unitary_ingestion_payload(
        bundle_id="b",
        solution_over=solution,
    )
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(export),
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )

    assert record["record_status"] == "has_final_reduced_ebr_results"
    assert record["final_reduced_ebr_result_count"] == 1
    counts = record["reduced_ebr_classification_counts"]
    assert counts[classification] == 1
    assert sum(counts.values()) == record["final_reduced_ebr_result_count"]
    if search_status is not None:
        assert record["reduced_ebr_records"][0]["search_status"] == (
            search_status
        )


def test_ingestion_rejects_mapping_solution_without_current_export_bundle():
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": [], "iband": [], "input": {}},
        valley_reduced_ebr_mapping={
            "status": "solved_exact",
            "table_status": "loaded",
            "solutions": [{
                "bundle_id": "stale_bundle",
                "classification": "atomic-compatible-candidate",
            }],
            "excluded_bundles": [],
        },
    )

    assert record["record_status"] == "no_reduced_ebr_input"
    assert record["final_reduced_ebr_result_count"] == 0
    assert record["reduced_ebr_records"] == []
    assert record["reduced_ebr_classification_counts"] == {
        "atomic_compatible": 0,
        "in_integer_span_no_nonnegative_witness": 0,
        "outside_integer_span": 0,
        "indeterminate_truncated": 0,
    }
    assert record["validation_errors"] == [
        "mapping solution stale_bundle: no matching ready export bundle"
    ]


def test_ingestion_rejects_matching_solution_without_promotion_evidence():
    export = _cprime_export({
        "bundle_id": "b",
        "valley": "K",
        "ready_for_reduced_table_validation": True,
        "irreps_by_kpoint": {},
        "irrep_records_by_kpoint": {},
    })
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(export),
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping={
            "status": "solved_exact",
            "table_status": "loaded",
            "solutions": [{
                "bundle_id": "b",
                "classification": "atomic-compatible-candidate",
            }],
            "excluded_bundles": [],
        },
    )

    assert record["record_status"] == (
        "has_reduced_table_validation_candidates"
    )
    assert record["final_reduced_ebr_result_count"] == 0
    assert record["reduced_ebr_records"] == []
    assert record["validation_errors"] == [
        "mapping solution b: missing passed promotion provenance"
    ]


def test_ingestion_rejects_promoted_solution_changed_after_export():
    export, mapping = _authoritative_unitary_ingestion_payload(
        bundle_id="b",
    )
    mapping["solutions"][0]["time_reversal"] = {
        **mapping["solutions"][0]["time_reversal"],
        "theta_square": 1,
    }

    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(export),
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )

    assert record["record_status"] == (
        "has_reduced_table_validation_candidates"
    )
    assert record["final_reduced_ebr_result_count"] == 0
    assert record["reduced_ebr_records"] == []
    assert record["validation_errors"] == [
        "mapping solution b: promotion provenance does not match current "
        "export bundle"
    ]


def _tr_validation_candidate_bundle():
    return {
        "bundle_id": "tr",
        "source_instance_id": "orbit",
        "problem_kind": "valley_orbit_reduced_ebr",
        "subspace_group_candidate": "P3",
        "valley": "K",
        "spinor": True,
        "ready_for_reduced_table_validation": True,
        "workflow_path": "time_reversal_valley_orbit",
        "readiness_level": "trusted",
        "source_hsp_to_sampled_kpoint": {
            "GM": "GammaM", "K": "KM",
        },
        "time_reversal": {
            "representative_valley": "K",
            "source_hsp_to_sampled_kpoint_by_valley": {
                "K": {"GM": "GammaM", "K": "KM"},
                "Kp": {"GM": "GammaM_Kp", "K": "KM_Kp"},
            },
        },
        "unitary_valley_irreps": {
            "K": {"GM": {"A": 1}, "K": {"B": 2}},
            "Kp": {"GM": {"A": 1}},
        },
        "irreps_by_kpoint": {
            "GammaM": ["A"],
            "KM": ["B", "B"],
            "GammaM_Kp": ["A"],
        },
        "irrep_records_by_kpoint": {},
    }


def test_tr_validation_candidate_unitary_irreps_survive_without_mapping():
    export = _cprime_export(_tr_validation_candidate_bundle())
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(export),
        valley_ebr_export_bundle=export,
    )

    assert record["record_status"] == (
        "has_reduced_table_validation_candidates"
    )
    assert len(record["valley_irrep_records"]) == 3
    assert record["valley_irrep_records"] == [
        {
            "kpoint": "GammaM",
            "source_hsp_label": "GM",
            "valley": "K",
            "subspace_group_candidate": "P3",
            "matched_irrep": "A",
            "irrep_multiplicity": 1,
            "workflow_path": "time_reversal_valley_orbit",
            "readiness_level": "trusted",
            "source": "unitary_valley_irreps",
            "source_bundle_id": "tr",
            "source_instance_id": "orbit",
            "certificate_identity": {},
        },
        {
            "kpoint": "KM",
            "source_hsp_label": "K",
            "valley": "K",
            "subspace_group_candidate": "P3",
            "matched_irrep": "B",
            "irrep_multiplicity": 2,
            "workflow_path": "time_reversal_valley_orbit",
            "readiness_level": "trusted",
            "source": "unitary_valley_irreps",
            "source_bundle_id": "tr",
            "source_instance_id": "orbit",
            "certificate_identity": {},
        },
        {
            "kpoint": "GammaM_Kp",
            "source_hsp_label": "GM",
            "valley": "Kp",
            "subspace_group_candidate": "P3",
            "matched_irrep": "A",
            "irrep_multiplicity": 1,
            "workflow_path": "time_reversal_valley_orbit",
            "readiness_level": "trusted",
            "source": "unitary_valley_irreps",
            "source_bundle_id": "tr",
            "source_instance_id": "orbit",
            "certificate_identity": {},
        },
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_valley_resolved_contract",
        "malformed_valley_resolved_contract",
        "missing_nonrepresentative_component",
        "missing_nonrepresentative_source_hsp",
        "representative_flat_map_conflict",
    ],
)
def test_tr_ingestion_fallback_requires_complete_valley_resolved_binding(
    mutation,
):
    from valleyscope.analysis.database_index import build_database_index

    bundle = _tr_validation_candidate_bundle()
    if mutation == "missing_valley_resolved_contract":
        bundle["time_reversal"].pop(
            "source_hsp_to_sampled_kpoint_by_valley"
        )
    elif mutation == "malformed_valley_resolved_contract":
        bundle["time_reversal"][
            "source_hsp_to_sampled_kpoint_by_valley"
        ] = []
    elif mutation == "missing_nonrepresentative_component":
        bundle["time_reversal"][
            "source_hsp_to_sampled_kpoint_by_valley"
        ].pop("Kp")
    elif mutation == "missing_nonrepresentative_source_hsp":
        bundle["time_reversal"][
            "source_hsp_to_sampled_kpoint_by_valley"
        ]["Kp"].pop("GM")
    else:
        bundle["source_hsp_to_sampled_kpoint"]["GM"] = "wrong_GammaM"

    export = _cprime_export(bundle)
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(export),
        valley_ebr_export_bundle=export,
    )

    assert record["valley_irrep_records"] == []
    assert len(record["validation_errors"]) == 1
    assert "tr" in record["validation_errors"][0]
    assert "source-HSP/sample binding" in record["validation_errors"][0]
    index = build_database_index([record])
    assert index["valley_irrep_records"] == []


def _tr_completed_unitary_bundle():
    reviewed_source = _reviewed_tr_source_report()
    observed_identity = {
        "source": "fixture/K/GM",
        "workflow_path": "direct_qcut",
        "valley": "K",
        "source_hsp_label": "GM",
        "sampled_kpoint": "GammaM",
        "irrep": "A",
        "multiplicity": 1,
    }
    provenance = {
        "source": "fixture/K/GM",
        "workflow_path": "direct_qcut",
        "irrep_source_provenance": {
            "matching_strategy": "bilbao_restricted_character",
            "subspace_space_group_number": 143,
            "subspace_space_group_symbol": "P3",
            "source_table_sg_number": 143,
            "source_table_name": "P3",
            "source_hsp_label": "GM",
            "source_table_spinor": True,
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": {
                    "schema_version": "1.0.0",
                    "validation_status": "validated",
                    "fixture": "database_ingestion_unitary",
                    "centering_vectors": [[0.0, 0.0, 0.0]],
                },
            },
        },
    }
    return {
        "bundle_id": "unitary_K",
        "source_instance_id": "unitary_K_instance",
        "problem_kind": "unitary_valley_reduced_ebr",
        "physical_object_kind": "unitary_valley_projected_subspace",
        "valley": "K",
        "valley_orbit": ["K", "Kp"],
        "subspace_group_candidate": "P3",
        "spinor": True,
        "workflow_path": "time_reversal_completed_unitary_valley",
        "unitary_vector_construction": {
            "kind": "time_reversal_completed_unitary_rows",
            "source": "validated_time_reversal_valley_orbit",
            "orbit_id": "time_reversal_valley_orbit_001",
        },
        "readiness_level": "trusted",
        "ready_for_reduced_table_validation": True,
        "expected_hsps": ["GM", "K", "KA"],
        "irreps_by_kpoint": {"GM": ["A"], "K": ["B"], "KA": ["B"]},
        "source_hsp_to_sampled_kpoint": {
            "GM": "GammaM",
            "K": "KM_K",
        },
        "independent_source_hsp_to_sampled_kpoint": {
            "GM": "GammaM",
            "K": "KM_K",
        },
        "observed_source_hsp_to_sampled_kpoint": {
            "GM": "GammaM",
            "K": "KM_K",
        },
        "time_reversal": {
            "theta_square": -1,
            "mapping_type": "exchanged",
            "valley_orbit": ["K", "Kp"],
            "time_reversal_valley_mapping": {"K": "Kp", "Kp": "K"},
            "time_reversal_hsp_orbits": [
                {
                    "representative": "GM",
                    "members": ["GM"],
                    "self_mapped": True,
                },
                {
                    "representative": "K",
                    "members": ["K", "KA"],
                    "self_mapped": False,
                },
            ],
            "full_unitary_source_hsp_labels": ["GM", "K", "KA"],
            "independent_time_reversal_hsp_labels": ["GM", "K"],
            "time_reversal_irrep_pairing": {
                "A": "A",
                "B": "Bp",
                "Bp": "B",
            },
            "reviewed_time_reversal_source_identity": (
                _reviewed_tr_source_identity()
            ),
            "reviewed_time_reversal_source_context": reviewed_source[
                "reviewed_source_context"
            ],
        },
        "irrep_records_by_kpoint": {},
        "unitary_irrep_completion_records_by_hsp": {
            "GM": [{
                "completion_kind": "observed_at_sampled_kpoint",
                "target_valley": "K",
                "target_source_hsp_label": "GM",
                "irrep": "A",
                "multiplicity": 1,
                "sampled_kpoint": "GammaM",
                "source_candidate_identity": observed_identity,
                "source_candidate_provenance": provenance,
                "structural_status": "validated",
                "readiness_status": "trusted",
                "blockers": [],
            }],
            "K": [{
                "completion_kind": "observed_at_sampled_kpoint",
                "target_valley": "K",
                "target_source_hsp_label": "K",
                "irrep": "B",
                "multiplicity": 1,
                "sampled_kpoint": "KM_K",
                "source_candidate_identity": {
                    **observed_identity,
                    "source": "fixture/K/K",
                    "source_hsp_label": "K",
                    "sampled_kpoint": "KM_K",
                    "irrep": "B",
                },
                "source_candidate_provenance": {
                    **provenance,
                    "source": "fixture/K/K",
                    "irrep_source_provenance": {
                        **provenance["irrep_source_provenance"],
                        "source_hsp_label": "K",
                    },
                },
                "structural_status": "validated",
                "readiness_status": "trusted",
                "blockers": [],
            }],
            "KA": [{
                "completion_kind": "inferred_by_time_reversal",
                "target_valley": "K",
                "target_source_hsp_label": "KA",
                "irrep": "B",
                "multiplicity": 1,
                "evidence_valley": "Kp",
                "evidence_source_hsp_label": "K",
                "evidence_sampled_kpoint": "KM_Kp",
                "reviewed_time_reversal_relation": {
                    "evidence_valley": "Kp",
                    "target_valley": "K",
                    "evidence_source_hsp_label": "K",
                    "target_source_hsp_label": "KA",
                    "evidence_irrep": "Bp",
                    "target_irrep": "B",
                },
                "source_candidate_identity": {
                    **observed_identity,
                    "source": "fixture/Kp/K",
                    "valley": "Kp",
                    "source_hsp_label": "K",
                    "sampled_kpoint": "KM_Kp",
                    "irrep": "Bp",
                },
                "source_candidate_provenance": {
                    **provenance,
                    "source": "fixture/Kp/K",
                    "irrep_source_provenance": {
                        **provenance["irrep_source_provenance"],
                        "source_hsp_label": "K",
                    },
                },
                "structural_status": "validated",
                "readiness_status": "trusted",
                "blockers": [],
            }],
        },
    }


def _authoritative_unitary_ingestion_payload(
    *,
    bundle_id="unitary_K",
    solution_over=None,
    table_provenance=None,
):
    """Current export/promotion pair for compact-ingestion unit tests."""
    bundle = _tr_completed_unitary_bundle()
    bundle["bundle_id"] = bundle_id
    bundle["certificate_identity"] = {"fixture_certificate": bundle_id}
    bundle["subspace_space_group"] = {
        "candidate_space_group_symbol": "P3",
    }
    export = _cprime_export(bundle)
    report = {
        check: "passed"
        for check in (
            "table_provenance_check",
            "table_setting_check",
            "sg_symbol_check",
            "sg_number_check",
            "certificate_check",
            "cprime_identity_check",
            "certificate_consistency_check",
            "cert_sg_consistency_check",
            "affine_setting_check",
            "hall_setting_check",
            "spin_convention_check",
            "problem_kind_check",
            "hsp_basis_check",
            "irrep_basis_check",
            "completion_provenance_check",
        )
    }
    promoted_table = deepcopy(
        table_provenance
        if table_provenance is not None
        else {
            "source": "synthetic_ingestion_fixture",
            "data_source": "irreptables",
            "package": "irreptables",
            "package_version": "3.1.0",
        }
    )
    identity_fields = (
        "problem_kind",
        "physical_object_kind",
        "valley",
        "valley_orbit",
        "subspace_group_candidate",
        "subspace_space_group",
        "expected_hsps",
        "required_source_hsp_labels",
        "covered_source_hsp_labels",
        "source_hsp_to_sampled_kpoint",
        "independent_source_hsp_to_sampled_kpoint",
        "observed_source_hsp_to_sampled_kpoint",
        "unitary_vector_construction",
        "unitary_irrep_completion_records_by_hsp",
        "unitary_valley_irreps",
        "time_reversal",
        "certificate_identity",
        "cprime_identity_by_kpoint",
    )
    solution = {
        "bundle_id": bundle_id,
        **{
            field: deepcopy(bundle.get(field))
            for field in identity_fields
        },
        "status": "solved_exact",
        "classification": "atomic-compatible-candidate",
        "integer_span_status": "in_integer_span",
        "nonnegative_solution_status": "solved_exact",
        "irrep_vector": [1],
        "table_provenance": deepcopy(promoted_table),
        "table_status": "loaded",
        "validation_report": deepcopy(report),
        "promotion_provenance": {
            "source": "promote_bundle_for_solve",
            "validation_report": deepcopy(report),
            "table_provenance": deepcopy(promoted_table),
            "certificate_identity": deepcopy(
                bundle["certificate_identity"]
            ),
        },
    }
    solution["required_source_hsp_labels"] = bundle.get(
        "required_source_hsp_labels", []
    )
    solution["covered_source_hsp_labels"] = bundle.get(
        "covered_source_hsp_labels", []
    )
    solution["unitary_valley_irreps"] = bundle.get(
        "unitary_valley_irreps", {}
    )
    if solution_over:
        solution.update(deepcopy(solution_over))
    from valleyscope.analysis.promotion_identity import (
        build_promotion_input_identity,
    )
    solution["promotion_provenance"]["promotion_input_identity"] = (
        build_promotion_input_identity(bundle)
    )
    solution["promotion_provenance"]["irrep_vector"] = deepcopy(
        solution["irrep_vector"]
    )
    return (
        export,
        {
            "status": solution["status"],
            "table_status": "loaded",
            "solutions": [solution],
            "excluded_bundles": [],
        },
    )


def test_tr_unitary_ingestion_preserves_observed_and_inferred_rows():
    unitary = _tr_completed_unitary_bundle()
    legacy_joint = _tr_validation_candidate_bundle()
    export = _cprime_export(unitary, legacy_joint)
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(export),
        valley_ebr_export_bundle=export,
    )

    assert len(record["valley_irrep_records"]) == 3
    observed, observed_k, inferred = record["valley_irrep_records"]
    assert observed["completion_kind"] == "observed_at_sampled_kpoint"
    assert observed["kpoint"] == "GammaM"
    assert observed_k["completion_kind"] == "observed_at_sampled_kpoint"
    assert observed_k["kpoint"] == "KM_K"
    assert inferred["completion_kind"] == "inferred_by_time_reversal"
    assert "kpoint" not in inferred
    assert inferred["evidence_sampled_kpoint"] == "KM_Kp"
    assert inferred["source_bundle_id"] == "unitary_K"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_evidence",
        "missing_tr_metadata",
        "missing_construction",
        "changed_workflow_and_missing_completion",
        "forged_evidence_hsp",
    ],
)
def test_tr_unitary_ingestion_never_uses_joint_representative_fallback(
    mutation,
):
    unitary = _tr_completed_unitary_bundle()
    if mutation == "missing_evidence":
        unitary["unitary_irrep_completion_records_by_hsp"]["KA"][0].pop(
            "evidence_sampled_kpoint"
        )
    elif mutation == "missing_tr_metadata":
        unitary.pop("time_reversal")
    elif mutation == "missing_construction":
        unitary.pop("unitary_vector_construction")
    elif mutation == "changed_workflow_and_missing_completion":
        unitary["workflow_path"] = "direct_qcut"
        unitary.pop("unitary_vector_construction")
        unitary.pop("unitary_irrep_completion_records_by_hsp")
    else:
        inferred = unitary[
            "unitary_irrep_completion_records_by_hsp"
        ]["KA"][0]
        inferred["evidence_source_hsp_label"] = "GM"
        inferred["reviewed_time_reversal_relation"][
            "evidence_source_hsp_label"
        ] = "GM"
        inferred["source_candidate_identity"]["source_hsp_label"] = "GM"
        inferred["source_candidate_provenance"][
            "irrep_source_provenance"
        ]["source_hsp_label"] = "GM"
    export = _cprime_export(unitary, _tr_validation_candidate_bundle())
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(export),
        valley_ebr_export_bundle=export,
    )

    assert record["valley_irrep_records"] == []
    assert record["validation_errors"] == [
        "bundle unitary_K: invalid TR-completed unitary provenance"
    ]


def test_ingestion_record_with_ready_bundle():
    """Ready export bundle produces trusted irrep records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    bundle = {
        "bundles": [{
            "bundle_id": "b_001", "source_instance_id": "ebr_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
                "ready_for_reduced_table_validation": True,
            "irreps_by_kpoint": {
                "GammaM": ["C3_spinor_phase_+1/2"],
            },
            "irrep_records_by_kpoint": {
                "GammaM": [{"valley": "K_valley", "operation_id": 1,
                            "operation_order": 3,
                            "matched_irrep": "C3_spinor_phase_+1/2",
                            "eigenphases": [0.5],
                            "workflow_path": "direct_qcut",
                            "readiness_level": "trusted",
                            "source": "valley_irrep_matching/GammaM/K_valley"}],
            },
        }],
    }
    attach_cprime_fixture_contract(bundle)
    summary = cprime_summary_for_export(
        bundle, target_kpoints=["GammaM"], iband=[1]
    )
    summary["symmetry_analysis"] = {
        "international": "P321", "spacegroup_number": 150
    }

    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle)

    assert record["record_status"] == (
        "has_reduced_table_validation_candidates"
    )
    assert record["final_reduced_ebr_result_count"] == 0
    assert record["reduced_table_validation_candidate_bundle_count"] == 1
    assert record["space_group_international"] == "P321"
    records = record["valley_irrep_records"]
    assert len(records) == 1
    r = records[0]
    assert r["kpoint"] == "GammaM"
    assert r["valley"] == "K_valley"
    assert r["subspace_group_candidate"] == "P3"
    assert r["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert r["source_bundle_id"] == "b_001"


def test_ingestion_record_excludes_non_ready_bundles():
    """Non-ready bundles do not contribute trusted irrep records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    bundle = {
        "bundles": [{
            "bundle_id": "b_001",
                "ready_for_reduced_table_validation": False,
            "irrep_records_by_kpoint": {},
        }],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle)
    assert record["reduced_table_validation_candidate_bundle_count"] == 0
    assert record["valley_irrep_records"] == []


def test_ingestion_record_with_reduced_ebr_mapping():
    """Reduced EBR mapping adds status and classification counts."""
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    payloads = [
        _authoritative_unitary_ingestion_payload(
            bundle_id=f"b_{index}",
            solution_over={"classification": classification},
        )
        for index, classification in enumerate((
            "atomic-compatible-candidate",
            "atomic-compatible-candidate",
            "in_integer_span_no_nonnegative_witness",
        ))
    ]
    export = {
        "bundles": [
            payload[0]["bundles"][0]
            for payload in payloads
        ],
    }
    mapping = {
        "status": "solved_exact",
        "table_status": "loaded",
        "solutions": [
            payload[1]["solutions"][0]
            for payload in payloads
        ],
    }
    record = build_database_ingestion_record(
        valley_summary=summary,
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )
    assert record["reduced_ebr_mapping_status"] == "solved_exact"
    counts = record["reduced_ebr_classification_counts"]
    assert counts["atomic_compatible"] == 2
    assert counts["in_integer_span_no_nonnegative_witness"] == 1
    assert counts["outside_integer_span"] == 0


def test_ingestion_record_missing_reduced_ebr_is_not_an_error():
    """Missing reduced EBR mapping is not an error."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record

    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(
        valley_summary=summary, valley_reduced_ebr_mapping=None)
    assert record["reduced_ebr_mapping_status"] == "not_available"
    assert record["reduced_ebr_classification_counts"] == {
        "atomic_compatible": 0,
        "in_integer_span_no_nonnegative_witness": 0,
        "outside_integer_span": 0,
        "indeterminate_truncated": 0,
    }
    assert record["reduced_ebr_records"] == []
    assert len(record["validation_errors"]) == 0


def test_ingestion_record_from_directory(tmp_path):
    """load_database_ingestion_record_from_directory reads files from dir."""
    from valleyscope.analysis.database_ingestion_record import (
        load_database_ingestion_record_from_directory,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": False}}
    (run_dir / "valley_summary.json").write_text(json.dumps(summary))

    record = load_database_ingestion_record_from_directory(str(run_dir))
    assert record["summary_status"] == "present"
    assert record["final_reduced_ebr_result_count"] == 0


def test_cli_collect_database_record(tmp_path, capsys):
    """CLI writes ingestion record to requested output path."""
    from valleyscope.cli import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {"target_kpoints": ["GammaM"], "iband": [1],
               "input": {"spinor_convention_verified": False}}
    (run_dir / "valley_summary.json").write_text(json.dumps(summary))

    out_path = tmp_path / "nested" / "record.json"
    rc = main(["collect-database-record", str(run_dir), "-o", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["summary_status"] == "present"
    captured = capsys.readouterr().out
    assert "no_reduced_ebr_input" in captured
    assert "validation candidates:" in captured
    assert "final EBR results:" in captured


def test_cli_collect_database_record_returns_nonzero_for_invalid_record(tmp_path):
    """CLI writes invalid record but exits nonzero when summary is missing."""
    from valleyscope.cli import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_path = tmp_path / "record.json"

    rc = main(["collect-database-record", str(run_dir), "-o", str(out_path)])
    assert rc == 1
    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["record_status"] == "invalid_missing_summary"
    assert record["validation_errors"]


def test_schema_doc_documents_database_ingestion_record():
    """Public schema documents the explicit offline ingestion-record CLI."""
    schema = Path("docs/schema.md").read_text(encoding="utf-8")
    assert "database_ingestion_record.json" in schema
    assert "collect-database-record" in schema
    assert "valley_irrep_records" in schema
    assert "not a default `analyze-hsp` output" in schema


def test_ingestion_record_no_material_names():
    """Ingestion record module must not contain real material names."""
    src = Path("valleyscope/analysis/database_ingestion_record.py").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src, f"database_ingestion_record.py must not contain {name!r}"


def test_ingestion_record_from_public_outputs_with_reduced_ebr_mapping(tmp_path):
    """Legacy C3-like solutions remain candidates but are not final results."""
    from valleyscope.analysis.database_ingestion_record import (
        load_database_ingestion_record_from_directory,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    c3_records = {
        "GammaM": [
            {"valley": "K_valley", "operation_id": "C3", "operation_order": 3,
             "matched_irrep": "C3_spinor_phase_+1/2", "eigenphases": [0.5],
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "source": "valley_irrep_matching/GammaM/K_valley/C3"},
            {"valley": "K_valley", "operation_id": "C3^2", "operation_order": 3,
             "matched_irrep": "C3_spinor_phase_+1/2", "eigenphases": [0.5],
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "source": "valley_irrep_matching/GammaM/K_valley/C3^2"},
        ],
        "KM": [
            {"valley": "K_valley", "operation_id": "C3", "operation_order": 3,
             "matched_irrep": "C3_spinor_phase_+1/6", "eigenphases": [1 / 6],
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "source": "valley_irrep_matching/KM/K_valley/C3"},
            {"valley": "K_valley", "operation_id": "C3^2", "operation_order": 3,
             "matched_irrep": "C3_spinor_phase_-1/6", "eigenphases": [-1 / 6],
             "workflow_path": "direct_qcut", "readiness_level": "trusted",
             "source": "valley_irrep_matching/KM/K_valley/C3^2"},
        ],
    }
    c3p_records = {
        kpoint: [
            {**record, "valley": "Kp_valley",
             "source": record["source"].replace("K_valley", "Kp_valley")}
            for record in records
        ]
        for kpoint, records in c3_records.items()
    }
    bundle = {
        "status": "ready_for_reduced_table_validation",
        "bundle_count": 2,
        "excluded_count": 0,
        "bundles": [
            {
                "bundle_id": "b_001", "source_instance_id": "ebr_001",
                "valley": "K_valley",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                "ready_for_reduced_table_validation": True,
                "irreps_by_kpoint": {
                    "GammaM": ["C3_spinor_phase_+1/2"],
                    "KM": [
                        "C3_spinor_phase_+1/6",
                        "C3_spinor_phase_-1/6",
                    ],
                },
                "irrep_records_by_kpoint": c3_records,
            },
            {
                "bundle_id": "b_002", "source_instance_id": "ebr_002",
                "valley": "Kp_valley",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                "ready_for_reduced_table_validation": True,
                "irreps_by_kpoint": {
                    "GammaM": ["C3_spinor_phase_+1/2"],
                    "KM": [
                        "C3_spinor_phase_+1/6",
                        "C3_spinor_phase_-1/6",
                    ],
                },
                "irrep_records_by_kpoint": c3p_records,
            },
        ],
        "excluded_instances": [],
    }
    summary = cprime_summary_for_export(
        bundle,
        target_kpoints=["GammaM", "KM"],
        iband=[101, 102],
    )
    mapping = {
        "status": "solved_exact", "table_status": "loaded",
        "solutions": [
            {
                "bundle_id": "b_001", "valley": "K_valley",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                "status": "solved_exact",
                "classification": "atomic-compatible-candidate",
                "integer_span_status": "in_integer_span",
                "nonnegative_solution_status": "solved_exact",
                "irrep_vector": [0, 2, 0, 1, 0, 1],
                "ebr_decomposition": [{"label": "-E↑G(2)", "coefficient": 1}],
            },
            {
                "bundle_id": "b_002", "valley": "Kp_valley",
                "subspace_group_candidate": "P3",
                "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                "status": "solved_exact",
                "classification": "atomic-compatible-candidate",
                "integer_span_status": "in_integer_span",
                "nonnegative_solution_status": "solved_exact",
                "irrep_vector": [0, 2, 0, 1, 0, 1],
                "ebr_decomposition": [{"label": "-E↑G(2)", "coefficient": 1}],
            },
        ],
    }
    (run_dir / "valley_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "valley_ebr_export_bundle.json").write_text(
        json.dumps(bundle), encoding="utf-8"
    )
    (run_dir / "valley_reduced_ebr_mapping.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )

    record = load_database_ingestion_record_from_directory(run_dir)

    assert record["schema_version"] == "2.0.0"
    assert record["record_status"] == (
        "has_reduced_table_validation_candidates"
    )
    assert record["reduced_table_validation_candidate_bundle_count"] == 2
    assert record["final_reduced_ebr_result_count"] == 0
    assert len(record["valley_irrep_records"]) == 8
    assert record["valley_irrep_records"][0]["valley"] == "K_valley"
    assert record["valley_irrep_records"][0]["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert record["valley_irrep_records"][0]["source_bundle_id"] == "b_001"
    assert record["reduced_ebr_mapping_status"] == "solved_exact"
    assert record["reduced_ebr_table_status"] == "loaded"
    counts = record["reduced_ebr_classification_counts"]
    assert counts["atomic_compatible"] == 0
    assert counts["in_integer_span_no_nonnegative_witness"] == 0
    assert counts["outside_integer_span"] == 0
    assert set(record["source_files"]) == {
        "valley_summary",
        "valley_ebr_export_bundle",
        "valley_reduced_ebr_mapping",
    }
    assert all(Path(path).is_absolute() for path in record["source_files"].values())
    assert record["reduced_ebr_records"] == []
    assert record["validation_errors"] == [
        "mapping solution b_001: missing passed promotion provenance",
        "mapping solution b_002: missing passed promotion provenance",
    ]


def _make_ingestion_record(
    status="has_final_reduced_ebr_results", run_id="run_0000"
):
    return {
        "schema_version": "1.3.0",
        "record_status": status,
        "space_group_international": "P321",
        "space_group_number": 150,
        "reduced_table_validation_candidate_bundle_count": 2,
        "final_reduced_ebr_result_count": 1,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 0,
        "input_excluded_ebr_records": [],
        "final_mapping_excluded_records": [],
        "valley_irrep_records": [
            {"kpoint": "GammaM", "valley": "K_valley",
             "subspace_group_candidate": "P3"},
        ],
        "reduced_ebr_records": [
            {"bundle_id": "b_001", "valley": "K_valley",
             "subspace_group_candidate": "P3",
             "status": "solved_exact",
             "classification": "atomic-compatible-candidate",
             "irrep_vector": [0, 2, 0, 1, 0, 1],
             "ebr_decomposition": [{"label": "-E↑G(2)", "coefficient": 1}]},
        ],
        "reduced_ebr_classification_counts": {
            "atomic_compatible": 1, "in_integer_span_no_nonnegative_witness": 0, "outside_integer_span": 0,
        },
        "reduced_ebr_mapping_status": "solved_exact",
        "reduced_ebr_table_status": "loaded",
        "validation_errors": [],
    }


def _write_blocked_public_run(run_dir: Path) -> Path:
    run_dir.mkdir()
    (run_dir / "valley_summary.json").write_text(
        json.dumps({
            "target_kpoints": ["GammaM"],
            "iband": [1],
            "input": {"spinor_convention_verified": False},
            "symmetry_analysis": {
                "international": "P1",
                "spacegroup_number": 1,
            },
        }),
        encoding="utf-8",
    )
    (run_dir / "valley_ebr_export_bundle.json").write_text(
        json.dumps({
            "status": "no_bundles",
            "bundles": [],
            "excluded_instances": [{
                "source_instance_id": "blocked_001",
                "valley": "valley_a",
                "status": "blocked",
                "canonical_hsp_vector_complete": True,
                "canonical_hsp_vector_ready": False,
                "exclusion_reasons": ["spinor_convention_unverified"],
            }],
        }),
        encoding="utf-8",
    )
    return run_dir


def test_database_index_builder_two_records():
    """Pure builder with final-result and no-input records."""
    from valleyscope.analysis.database_index import build_database_index
    rec1 = _make_ingestion_record("has_final_reduced_ebr_results")
    rec2 = _make_ingestion_record("no_reduced_ebr_input")
    index = build_database_index(
        [rec1, rec2],
        source_files=[
            "/tmp/run_a/database_ingestion_record.json",
            "/tmp/run_b/database_ingestion_record.json",
        ],
    )
    assert index["record_count"] == 2
    assert index["status_counts"]["has_final_reduced_ebr_results"] == 1
    assert index["status_counts"]["no_reduced_ebr_input"] == 1
    assert index[
        "reduced_table_validation_candidate_bundle_count_total"
    ] == 4
    assert index["final_reduced_ebr_result_count_total"] == 2
    assert index["reduced_ebr_classification_counts_total"]["atomic_compatible"] == 2
    # Flattened records have run_id provenance.
    assert index["runs"][0]["run_id"] == "run_0000"
    assert index["runs"][1]["run_id"] == "run_0001"
    assert index["runs"][0]["source"].endswith("/run_a/database_ingestion_record.json")
    assert index["source_inputs"] == [
        {
            "kind": "ingestion_record_file",
            "path": "/tmp/run_a/database_ingestion_record.json",
        },
        {
            "kind": "ingestion_record_file",
            "path": "/tmp/run_b/database_ingestion_record.json",
        },
    ]
    assert index["runs"][0]["source_input"] == index["source_inputs"][0]
    for ir in index["valley_irrep_records"]:
        assert "run_id" in ir
        assert "source_record" in ir
        assert ir["source_input"]["kind"] == "ingestion_record_file"
    for rr in index["reduced_ebr_records"]:
        assert "run_id" in rr
        assert "source_record" in rr
        assert rr["source_input"]["kind"] == "ingestion_record_file"
    assert len(index["reduced_ebr_records"]) == 2


def test_database_index_aggregates_indeterminate_truncated_classification():
    from valleyscope.analysis.database_index import build_database_index

    record = _make_ingestion_record()
    record["final_reduced_ebr_result_count"] = 2
    record["reduced_ebr_classification_counts"] = {
        "atomic_compatible": 1,
        "in_integer_span_no_nonnegative_witness": 0,
        "outside_integer_span": 0,
        "indeterminate_truncated": 1,
    }

    index = build_database_index([record])

    assert index["final_reduced_ebr_result_count_total"] == 2
    assert index["reduced_ebr_classification_counts_total"] == {
        "atomic_compatible": 1,
        "in_integer_span_no_nonnegative_witness": 0,
        "outside_integer_span": 0,
        "indeterminate_truncated": 1,
    }


def test_database_index_uses_stage_owned_aggregates_only():
    from valleyscope.analysis.database_index import build_database_index

    record = {
        "schema_version": "1.8.0",
        "record_status": "has_final_reduced_ebr_results",
        "reduced_table_validation_candidate_bundle_count": 2,
        "final_reduced_ebr_result_count": 1,
        "final_mapping_excluded_bundle_count": 1,
        "input_excluded_instance_count": 3,
        "valley_irrep_records": [],
        "reduced_ebr_records": [{"bundle_id": "b"}],
        "input_excluded_ebr_records": [{"source_instance_id": "i"}],
        "final_mapping_excluded_records": [{"bundle_id": "blocked"}],
        "reduced_ebr_classification_counts": {},
        "validation_errors": [],
    }

    index = build_database_index([record])

    assert index["schema_version"] == "1.2.0"
    assert index["status_counts"]["has_final_reduced_ebr_results"] == 1
    assert index[
        "reduced_table_validation_candidate_bundle_count_total"
    ] == 2
    assert index["final_reduced_ebr_result_count_total"] == 1
    assert index["final_mapping_excluded_bundle_count_total"] == 1
    assert index["input_excluded_instance_count_total"] == 3
    assert index["input_excluded_ebr_records"][0]["run_id"] == "run_0000"
    assert index["final_mapping_excluded_records"][0]["run_id"] == (
        "run_0000"
    )
    assert "ready_bundle_count_total" not in index
    assert "excluded_ebr_records" not in index


def test_database_index_cli_mixed_inputs_write_fail_closed_index(tmp_path):
    from valleyscope.cli import main

    record_path = tmp_path / "success.json"
    record_path.write_text(
        json.dumps(_make_ingestion_record()),
        encoding="utf-8",
    )
    run_dir = _write_blocked_public_run(tmp_path / "blocked_run")
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        json.dumps({"hello": "world"}),
        encoding="utf-8",
    )
    output = tmp_path / "index.json"

    rc = main([
        "collect-database-index",
        str(record_path),
        str(run_dir),
        str(invalid_path),
        "--output",
        str(output),
    ])

    assert rc == 1
    index = json.loads(output.read_text(encoding="utf-8"))
    assert index["schema_version"] == "1.2.0"
    assert index["record_count"] == 3
    assert index["final_reduced_ebr_result_count_total"] == 1
    assert [run["record_status"] for run in index["runs"]] == [
        "has_final_reduced_ebr_results",
        "no_reduced_ebr_input",
        "invalid_missing_summary",
    ]
    assert [source["kind"] for source in index["source_inputs"]] == [
        "ingestion_record_file",
        "analyze_output_directory",
        "ingestion_record_file",
    ]
    assert "schema_version must be a nonempty string" in (
        index["validation_errors"][0]
    )


def test_database_index_cli_help_describes_mixed_inputs(capsys):
    from valleyscope.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["collect-database-index", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "database_ingestion_record.json" in help_text
    assert "analyze-hsp output directory" in help_text


def test_database_collector_cli_does_not_import_physics_workflows(tmp_path):
    run_dir = _write_blocked_public_run(tmp_path / "blocked_run")
    output = tmp_path / "index.json"
    script = """
import importlib.abc
import sys

blocked = {
    "valleyscope.workflows.analyze_hsp",
    "valleyscope.workflows.extract_wavecar",
    "valleyscope.analysis.reduced_ebr_mapping",
}

class BlockPhysicsImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise RuntimeError(f"unexpected physics import: {fullname}")
        return None

sys.meta_path.insert(0, BlockPhysicsImports())
from valleyscope.cli import main

try:
    main(["--help"])
except SystemExit as exc:
    assert exc.code == 0

raise SystemExit(main([
    "collect-database-index",
    sys.argv[1],
    "--output",
    sys.argv[2],
]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(run_dir), str(output)],
        cwd=Path(__file__).parent.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["record_count"] == 1


def test_database_index_loader_accepts_run_directory(tmp_path):
    from valleyscope.analysis.database_index import (
        load_database_index_from_inputs,
    )

    run_dir = _write_blocked_public_run(tmp_path / "blocked_run")
    index = load_database_index_from_inputs([str(run_dir)])

    resolved = str(run_dir.resolve())
    assert index["record_count"] == 1
    assert index["schema_version"] == "1.2.0"
    assert index["source_files"] == []
    assert index["source_inputs"] == [{
        "kind": "analyze_output_directory",
        "path": resolved,
    }]
    assert index["runs"] == [{
        "run_id": "run_0000",
        "record_status": "no_reduced_ebr_input",
        "space_group_international": "P1",
        "space_group_number": 1,
        "reduced_table_validation_candidate_bundle_count": 0,
        "final_reduced_ebr_result_count": 0,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 1,
        "valley_irrep_record_count": 0,
        "reduced_ebr_mapping_status": "not_available",
        "reduced_ebr_table_status": "not_available",
        "ebr_export_status": "no_bundles",
        "source_input": {
            "kind": "analyze_output_directory",
            "path": resolved,
        },
    }]
    assert index["validation_errors"] == []
    excluded = index["input_excluded_ebr_records"][0]
    assert excluded["source_input"] == index["source_inputs"][0]
    assert "source_record" not in excluded


def test_database_index_loader_mixes_record_file_and_run_directory_in_order(
    tmp_path,
):
    from valleyscope.analysis.database_index import (
        load_database_index_from_inputs,
    )

    blocked_dir = _write_blocked_public_run(tmp_path / "blocked_run")
    success_path = tmp_path / "success_record.json"
    success_path.write_text(
        json.dumps(_make_ingestion_record()),
        encoding="utf-8",
    )

    index = load_database_index_from_inputs([
        str(blocked_dir),
        str(success_path),
    ])

    assert index["record_count"] == 2
    assert [run["run_id"] for run in index["runs"]] == [
        "run_0000", "run_0001",
    ]
    assert index["source_inputs"] == [
        {
            "kind": "analyze_output_directory",
            "path": str(blocked_dir.resolve()),
        },
        {
            "kind": "ingestion_record_file",
            "path": str(success_path.resolve()),
        },
    ]
    assert index["source_files"] == [str(success_path.resolve())]
    assert [run["source_input"] for run in index["runs"]] == (
        index["source_inputs"]
    )
    assert "source" not in index["runs"][0]
    assert index["runs"][1]["source"] == str(success_path.resolve())
    assert index["runs"][0]["record_status"] == "no_reduced_ebr_input"
    assert index["runs"][0]["final_reduced_ebr_result_count"] == 0
    assert index["runs"][1]["record_status"] == (
        "has_final_reduced_ebr_results"
    )
    assert index["runs"][1]["final_reduced_ebr_result_count"] == 1
    assert index["status_counts"]["no_reduced_ebr_input"] == 1
    assert index["status_counts"]["has_final_reduced_ebr_results"] == 1
    assert index["final_reduced_ebr_result_count_total"] == 1
    assert index["validation_errors"] == []


def test_database_index_loader_rejects_duplicate_resolved_input(tmp_path):
    from valleyscope.analysis.database_index import (
        load_database_index_from_inputs,
    )

    run_dir = _write_blocked_public_run(tmp_path / "blocked_run")
    index = load_database_index_from_inputs([str(run_dir), str(run_dir)])

    assert index["record_count"] == 1
    assert index["source_files"] == []
    assert index["source_inputs"] == [{
        "kind": "analyze_output_directory",
        "path": str(run_dir.resolve()),
    }]
    assert index["input_excluded_instance_count_total"] == 1
    assert index["validation_errors"] == [
        f"duplicate resolved input: {run_dir.resolve()}"
    ]


@pytest.mark.parametrize(
    ("summary_text", "expected_error"),
    [
        (None, "valley_summary.json is missing"),
        ("{not-json", "JSONDecodeError"),
    ],
)
def test_database_index_loader_directory_errors_fail_closed(
    tmp_path,
    summary_text,
    expected_error,
):
    from valleyscope.analysis.database_index import (
        load_database_index_from_inputs,
    )

    run_dir = tmp_path / "invalid_run"
    run_dir.mkdir()
    if summary_text is not None:
        (run_dir / "valley_summary.json").write_text(
            summary_text,
            encoding="utf-8",
        )

    index = load_database_index_from_inputs([str(run_dir)])

    assert index["record_count"] == 1
    assert index["status_counts"]["invalid_missing_summary"] == 1
    assert index["runs"][0]["source_input"] == {
        "kind": "analyze_output_directory",
        "path": str(run_dir.resolve()),
    }
    assert "source" not in index["runs"][0]
    assert any(
        expected_error in error for error in index["validation_errors"]
    )


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("valley_summary.json", []),
        ("valley_ebr_export_bundle.json", []),
        ("valley_reduced_ebr_mapping.json", []),
    ],
)
def test_database_index_loader_rejects_non_object_public_payloads(
    tmp_path,
    filename,
    payload,
):
    from valleyscope.analysis.database_index import (
        load_database_index_from_inputs,
    )

    run_dir = _write_blocked_public_run(tmp_path / "invalid_run")
    (run_dir / filename).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    index = load_database_index_from_inputs([str(run_dir)])

    assert index["record_count"] == 1
    assert index["runs"][0]["record_status"] == "invalid_missing_summary"
    assert index["status_counts"]["invalid_missing_summary"] == 1
    assert any(
        f"{filename} must contain a JSON object" in error
        for error in index["validation_errors"]
    )


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_error"),
    [
        (None, None, "schema_version must be a nonempty string"),
        (
            "final_reduced_ebr_result_count",
            "three",
            "final_reduced_ebr_result_count must be a nonnegative integer",
        ),
        (
            "final_reduced_ebr_result_count",
            True,
            "final_reduced_ebr_result_count must be a nonnegative integer",
        ),
        (
            "valley_irrep_records",
            {},
            "valley_irrep_records must be a list",
        ),
        (
            "validation_errors",
            "not-a-list",
            "validation_errors must be a list",
        ),
        (
            "reduced_ebr_classification_counts",
            {"outside_integer_span": True},
            "outside_integer_span must be a nonnegative integer",
        ),
        (
            "ebr_export_status",
            [],
            "ebr_export_status must be a string",
        ),
    ],
)
def test_database_index_loader_rejects_semantically_invalid_record_files(
    tmp_path,
    field,
    invalid_value,
    expected_error,
):
    from valleyscope.analysis.database_index import (
        load_database_index_from_inputs,
    )

    record = {"hello": "world"} if field is None else _make_ingestion_record()
    if field is not None:
        record[field] = invalid_value
    record_path = tmp_path / "invalid_record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    index = load_database_index_from_inputs([str(record_path)])

    assert index["record_count"] == 1
    assert index["runs"][0]["record_status"] == "invalid_missing_summary"
    assert index["status_counts"]["invalid_missing_summary"] == 1
    assert any(
        expected_error in error for error in index["validation_errors"]
    )


def test_database_index_loader_preserves_all_flattened_source_provenance(
    tmp_path,
):
    from valleyscope.analysis.database_index import (
        load_database_index_from_inputs,
    )

    record = _make_ingestion_record()
    record["input_excluded_instance_count"] = 1
    record["input_excluded_ebr_records"] = [{
        "source_instance_id": "input_blocked",
    }]
    record["final_mapping_excluded_bundle_count"] = 1
    record["final_mapping_excluded_records"] = [{
        "bundle_id": "mapping_blocked",
    }]
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    index = load_database_index_from_inputs([str(record_path)])
    resolved = str(record_path.resolve())

    for collection in (
        "valley_irrep_records",
        "reduced_ebr_records",
        "input_excluded_ebr_records",
        "final_mapping_excluded_records",
    ):
        assert index[collection]
        assert all(
            row["run_id"] == "run_0000"
            and row["source_record"] == resolved
            and row["source_input"] == {
                "kind": "ingestion_record_file",
                "path": resolved,
            }
            for row in index[collection]
        )


def test_database_index_directory_provenance_removes_source_record_alias():
    from valleyscope.analysis.database_index import build_database_index

    record = _make_ingestion_record()
    record["input_excluded_instance_count"] = 1
    record["input_excluded_ebr_records"] = [{"source_record": "/forged"}]
    record["final_mapping_excluded_bundle_count"] = 1
    record["final_mapping_excluded_records"] = [{"source_record": "/forged"}]
    for collection in (
        "valley_irrep_records",
        "reduced_ebr_records",
    ):
        record[collection][0]["source_record"] = "/forged"
    source_input = {
        "kind": "analyze_output_directory",
        "path": "/tmp/valley_analysis",
    }

    index = build_database_index(
        [record],
        source_inputs=[source_input],
    )

    for collection in (
        "valley_irrep_records",
        "reduced_ebr_records",
        "input_excluded_ebr_records",
        "final_mapping_excluded_records",
    ):
        assert index[collection]
        assert all(
            row["source_input"] == source_input
            and "source_record" not in row
            for row in index[collection]
        )


@pytest.mark.parametrize(
    "source_inputs",
    [
        [{}],
        [{"kind": "unknown", "path": "/tmp/input"}],
        [{"kind": "ingestion_record_file"}],
        [],
    ],
)
def test_database_index_builder_rejects_invalid_typed_source_inputs(
    source_inputs,
):
    from valleyscope.analysis.database_index import build_database_index

    with pytest.raises(ValueError):
        build_database_index(
            [_make_ingestion_record()],
            source_inputs=source_inputs,
        )


def test_database_index_loader_is_byte_deterministic_across_hash_seeds(
    tmp_path,
):
    blocked_dir = _write_blocked_public_run(tmp_path / "blocked_run")
    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps(_make_ingestion_record()),
        encoding="utf-8",
    )
    outputs = []
    for seed in ("1", "41"):
        output = tmp_path / f"index_{seed}.json"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "valleyscope.cli",
                "collect-database-index",
                str(blocked_dir),
                str(record_path),
                "--output",
                str(output),
            ],
            cwd=Path(__file__).parent.parent,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(output.read_bytes())

    assert outputs[0] == outputs[1]


def test_database_index_module_no_material_names():
    """Index module must not contain real material names."""
    src = Path("valleyscope/analysis/database_index.py").read_text(encoding="utf-8")
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in src


def test_ingestion_record_includes_input_excluded_ebr_records():
    """Export exclusions remain distinct input-stage records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    bundle = {
        "status": "partial_export",
        "interpretation": "one blocked instance",
        "bundles": [],
        "excluded_instances": [
            {
                "source_instance_id": "ebr_001", "valley": "M3_valley",
                "subspace_group_candidate": "P2",
                "subspace_space_group": {"candidate_space_group_symbol": "P2"},
                "status": "blocked",
                "canonical_hsp_vector_complete": False,
                "exclusion_reasons": [
                    "spinor_convention_unverified",
                    "low_seed_projector_symmetry",
                ],
            },
        ],
    }
    record = build_database_ingestion_record(
        valley_summary=summary, valley_ebr_export_bundle=bundle,
    )
    assert record["ebr_export_status"] == "partial_export"
    assert record["ebr_export_interpretation"] == "one blocked instance"
    exclude = record["input_excluded_ebr_records"]
    assert len(exclude) == 1
    assert exclude[0]["source_instance_id"] == "ebr_001"
    assert exclude[0]["valley"] == "M3_valley"
    assert exclude[0]["subspace_space_group"] == {
        "candidate_space_group_symbol": "P2"
    }
    assert exclude[0]["exclusion_reasons"] == [
        "spinor_convention_unverified", "low_seed_projector_symmetry",
    ]


def test_input_excluded_ebr_records_empty_when_not_present():
    """Missing bundle gives empty input_excluded_ebr_records."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(valley_summary=summary)
    assert record["input_excluded_ebr_records"] == []
    assert record["ebr_export_status"] == "not_available"


def test_database_index_input_excluded_records_aggregated():
    """Index aggregates input exclusions with run provenance."""
    from valleyscope.analysis.database_index import build_database_index
    rec = {
        "schema_version": "1.3.0",
        "record_status": "no_reduced_ebr_input",
        "reduced_table_validation_candidate_bundle_count": 0,
        "final_reduced_ebr_result_count": 0,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 1,
        "valley_irrep_records": [],
        "reduced_ebr_records": [],
        "reduced_ebr_classification_counts": {},
        "reduced_ebr_mapping_status": "?",
        "reduced_ebr_table_status": "?",
        "ebr_export_status": "partial_export",
        "final_mapping_excluded_records": [],
        "input_excluded_ebr_records": [
            {"source_instance_id": "ebr_001", "valley": "M3_valley",
             "exclusion_reasons": ["spinor_convention_unverified"]},
        ],
        "validation_errors": [],
    }
    idx = build_database_index([rec])
    assert idx["input_excluded_instance_count_total"] == 1
    assert idx["ebr_export_status_counts"]["partial_export"] == 1
    assert idx["input_excluded_ebr_records"][0]["run_id"] == "run_0000"


def test_database_index_input_exclusions_have_source_record():
    """Input exclusions carry source_record when source_files are provided."""
    from valleyscope.analysis.database_index import build_database_index
    rec = {
        "schema_version": "1.8.0",
        "record_status": "no_reduced_ebr_input",
        "reduced_table_validation_candidate_bundle_count": 0,
        "final_reduced_ebr_result_count": 0,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 1,
        "valley_irrep_records": [],
        "reduced_ebr_records": [],
        "reduced_ebr_classification_counts": {},
        "reduced_ebr_mapping_status": "?",
        "reduced_ebr_table_status": "?",
        "ebr_export_status": "no_bundles",
        "final_mapping_excluded_records": [],
        "input_excluded_ebr_records": [
            {"source_instance_id": "ebr_x", "valley": "M1_valley",
             "exclusion_reasons": ["low_seed_overlap"]},
        ],
        "validation_errors": [],
    }
    idx = build_database_index([rec], source_files=["/tmp/rec.json"])
    er = idx["input_excluded_ebr_records"][0]
    assert er["run_id"] == "run_0000"
    assert er["source_record"] == "/tmp/rec.json"
    assert er["source_input"] == {
        "kind": "ingestion_record_file",
        "path": "/tmp/rec.json",
    }


def test_ingestion_record_schema_version_is_2_0_0():
    """Ingestion record schema reflects the C-prime breaking reset."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    summary = {"target_kpoints": [], "iband": [], "input": {}}
    record = build_database_ingestion_record(valley_summary=summary)
    assert record["schema_version"] == "2.0.0"


# -----------------------------------------------------------------------


def test_irrep_records_preserve_generic_fields():
    """Generic irrep provenance fields survive ingestion flattening."""
    from valleyscope.analysis.database_ingestion_record import build_database_ingestion_record
    bundle = {
        "bundles": [{
            "bundle_id": "b_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
                "ready_for_reduced_table_validation": True,
            "irreps_by_kpoint": {"GammaM": ["-GM5", "-GM5"]},
            "irrep_records_by_kpoint": {
                "GammaM": [{
                    "valley": "K_valley",
                    "operation_id": 2,
                    "operation_order": 3,
                    "matched_irrep": "-GM5",
                    "eigenphases": [0.5],
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                    "source": "generic/GammaM/K_valley",
                    "irrep_multiplicity": 2,
                    "matching_strategy": "bilbao_restricted_character",
                    "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                    "legacy_subspace_group_candidate": "C3_like",
                    "valley_preserving_operation_ids": [0, 2, 3],
                    "source_operation_map": {0: 1, 2: 2, 3: 3},
                    "irrep_source_provenance": {
                        "source_hsp_label": "GM",
                        "source_table_sg_number": 143,
                        "standard_setting_hsp_mapping": {
                            "standard_setting_certificate": {
                                "validation_status": "validated",
                                "subspace_sg_number": 143,
                                "resolved_hsp_label": "GM",
                                "centering_status": "primitive_direct_match",
                            },
                        },
                    },
                }],
            },
        }],
    }
    attach_cprime_fixture_contract(bundle)
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(
            bundle, target_kpoints=["GammaM"], iband=[1]
        ),
        valley_ebr_export_bundle=bundle,
    )
    recs = record["valley_irrep_records"]
    assert len(recs) == 1
    r = recs[0]
    assert r["irrep_multiplicity"] == 2
    assert r["matching_strategy"] == "bilbao_restricted_character"
    assert r["subspace_space_group"] == {"candidate_space_group_symbol": "P3"}
    assert r["valley_preserving_operation_ids"] == [0, 2, 3]
    assert r["source_operation_map"] == {0: 1, 2: 2, 3: 3}
    cert = (
        r["irrep_source_provenance"]
        ["standard_setting_hsp_mapping"]
        ["standard_setting_certificate"]
    )
    assert cert["validation_status"] == "validated"
    assert cert["subspace_sg_number"] == 143
    assert cert["resolved_hsp_label"] == "GM"


def test_ingestion_preserves_centered_certificate_identity_from_bundle():
    from valleyscope.analysis.database_ingestion_record import (
        build_database_ingestion_record,
    )

    centered_map = [{
        "parent_operation_id": -3,
        "centering_coset_index": 0,
        "standard_operation_index": 0,
    }, {
        "parent_operation_id": -3,
        "centering_coset_index": 1,
        "standard_operation_index": 1,
    }]
    certificate_identity = {
        "sg_number": 5,
        "hall_number": 9,
        "centering_type": "C",
        "centered_affine_operation_map": centered_map,
        "affine_unmatched_centered_operation_pairs": [],
    }
    bundle = {
        "bundles": [{
            "bundle_id": "b_centered",
            "source_instance_id": "i_centered",
            "valley": "K_valley",
            "subspace_group_candidate": "C2",
                "ready_for_reduced_table_validation": True,
            "certificate_identity": certificate_identity,
            "irreps_by_kpoint": {"GM": ["GM1"]},
            "irrep_records_by_kpoint": {
                "GM": [{
                    "valley": "K_valley",
                    "operation_id": -3,
                    "matched_irrep": "GM1",
                }],
            },
        }],
    }
    attach_cprime_fixture_contract(bundle)
    record = build_database_ingestion_record(
        valley_summary=cprime_summary_for_export(
            bundle, target_kpoints=["GM"], iband=[1]
        ),
        valley_ebr_export_bundle=bundle,
    )

    assert record["schema_version"] == "2.0.0"
    assert record["valley_irrep_records"][0]["certificate_identity"] == (
        certificate_identity
    )


def test_database_index_preserves_generic_irrep_fields_with_run_provenance():
    """Database index keeps generic irrep fields with run provenance."""
    from valleyscope.analysis.database_index import build_database_index

    record = {
        "schema_version": "1.3.0",
        "record_status": "has_reduced_table_validation_candidates",
        "reduced_table_validation_candidate_bundle_count": 1,
        "final_reduced_ebr_result_count": 0,
        "final_mapping_excluded_bundle_count": 0,
        "input_excluded_instance_count": 0,
        "input_excluded_ebr_records": [],
        "final_mapping_excluded_records": [],
        "valley_irrep_records": [{
            "kpoint": "GammaM",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "matched_irrep": "-GM5",
            "irrep_multiplicity": 2,
            "matching_strategy": "bilbao_restricted_character",
            "subspace_space_group": {"candidate_space_group_symbol": "P3"},
            "legacy_subspace_group_candidate": "C3_like",
            "valley_preserving_operation_ids": [0, 2, 3],
            "source_operation_map": {0: 1, 2: 2, 3: 3},
        }],
        "reduced_ebr_records": [],
        "reduced_ebr_classification_counts": {
            "atomic_compatible": 0,
            "in_integer_span_no_nonnegative_witness": 0,
            "outside_integer_span": 0,
        },
        "reduced_ebr_mapping_status": "not_available",
        "reduced_ebr_table_status": "not_available",
        "validation_errors": [],
    }

    index = build_database_index(
        [record],
        source_files=["/tmp/database_ingestion_record.json"],
    )
    ir = index["valley_irrep_records"][0]
    assert ir["run_id"] == "run_0000"
    assert ir["source_record"] == "/tmp/database_ingestion_record.json"
    assert ir["source_input"] == {
        "kind": "ingestion_record_file",
        "path": "/tmp/database_ingestion_record.json",
    }
    assert ir["irrep_multiplicity"] == 2
    assert ir["matching_strategy"] == "bilbao_restricted_character"
    assert ir["subspace_space_group"] == {"candidate_space_group_symbol": "P3"}
    assert ir["legacy_subspace_group_candidate"] == "C3_like"
    assert ir["valley_preserving_operation_ids"] == [0, 2, 3]
    assert ir["source_operation_map"] == {0: 1, 2: 2, 3: 3}


# ---------------------------------------------------------------------------
# Compact reduced EBR table provenance in ingestion records
# ---------------------------------------------------------------------------

def _auto_table_provenance():
    """Minimal auto-canonical table_provenance dict."""
    return {
        "source": "auto_canonical",
        "auto_canonical": True,
        "subspace_group_candidate": "P3",
        "space_group_number": 143,
        "spinful": True,
        "data_source": "irreptables",
        "package": "irreptables",
        "package_version": "3.1.0",
        "expected_hsps": ["GammaM", "KM"],
        "valleyscope_reduction": "sampled_hsp_valley_preserving",
        "source_basis_count": 20,
        "reduction_basis_count": 6,
        "dropped_source_row_count": 14,
        "dropped_source_rows": ["label1", "label2"],
    }


def test_reduced_ebr_records_pick_up_table_provenance():
    """Compact ingestion records carry table_provenance fields when present."""
    export, mapping = _authoritative_unitary_ingestion_payload(
        bundle_id="b_001",
        table_provenance=_auto_table_provenance(),
        solution_over={
            "status": "solved_exact",
            "classification": "atomic-compatible-candidate",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "solved_exact",
            "irrep_vector": [1, 0],
            "ebr_decomposition": [{"label": "E@1a", "coefficient": 1}],
        },
    )
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": ["GammaM", "KM"], "iband": [1, 2],
                        "input": {}},
        valley_ebr_export_bundle=export,
        valley_reduced_ebr_mapping=mapping,
    )
    recs = record["reduced_ebr_records"]
    assert len(recs) == 1
    r = recs[0]
    assert r["table_source"] == "auto_canonical"
    assert r["data_source"] == "irreptables"
    assert r["package"] == "irreptables"
    assert r["package_version"] == "3.1.0"
    assert r["space_group_number"] == 143
    assert r["spinful"] is True
    assert r["expected_hsps"] == ["GammaM", "KM"]
    assert r["valleyscope_reduction"] == "sampled_hsp_valley_preserving"
    assert r["source_basis_count"] == 20
    assert r["reduction_basis_count"] == 6
    assert r["dropped_source_row_count"] == 14
    assert r["table_status"] == "loaded"
    assert r["dropped_source_rows"] == ["label1", "label2"]
    assert r["filtered_zero_vector_ebr_count"] == 0
    assert r["filtered_zero_vector_ebrs"] == []
    assert "auto_canonical" not in r


def test_reduced_ebr_records_preserve_joint_valley_orbit_identity():
    """Unpromoted legacy joint TR solutions remain fail-closed."""
    time_reversal = {
        "theta_square": -1,
        "time_reversal_valley_mapping": {
            "valley_a": "valley_b",
            "valley_b": "valley_a",
        },
        "representative_valley": "valley_a",
        "source_hsp_to_sampled_kpoint_by_valley": {
            "valley_a": {"K": "K_a"},
            "valley_b": {"K": "K_b"},
        },
    }
    unitary_valley_irreps = {
        "valley_a": {"K": {"rho_a": 1}},
        "valley_b": {"K": {"rho_b": 1}},
    }
    mapping = {
        "status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_orbit",
            "valley": "",
            "problem_kind": "valley_orbit_reduced_ebr",
            "valley_orbit": ["valley_a", "valley_b"],
            "unitary_valley_irreps": unitary_valley_irreps,
            "time_reversal": time_reversal,
            "subspace_group_candidate": "P1",
            "status": "no_exact_solution",
            "classification": "in_integer_span_no_nonnegative_witness",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "no_nonnegative_solution",
            "irrep_vector": [1],
        }],
    }

    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": ["K"], "iband": [1], "input": {}},
        valley_reduced_ebr_mapping=mapping,
    )

    assert record["reduced_ebr_records"] == []
    assert record["final_reduced_ebr_result_count"] == 0
    assert record["validation_errors"] == [
        "mapping solution b_orbit: no matching ready export bundle"
    ]


def test_reduced_ebr_records_without_table_provenance_are_rejected():
    """A mapping record without promotion/table evidence is not final."""
    mapping = {
        "status": "solved_exact", "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "classification": "atomic-compatible-candidate",
            "status": "solved_exact",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "solved_exact",
            "irrep_vector": [1],
        }],
    }
    record = build_database_ingestion_record(
        valley_summary={"target_kpoints": [], "iband": [], "input": {}},
        valley_reduced_ebr_mapping=mapping,
    )
    assert record["reduced_ebr_records"] == []
    assert record["final_reduced_ebr_result_count"] == 0
    assert record["validation_errors"] == [
        "mapping solution b_001: no matching ready export bundle"
    ]


def test_summary_only_ingestion_does_not_import_optional_irrep_runtime(
    tmp_path,
):
    fixture = tmp_path / "summary_only"
    fixture.mkdir()
    (fixture / "valley_summary.json").write_text(
        json.dumps({"target_kpoints": [], "iband": []}),
        encoding="utf-8",
    )
    script = f"""
import builtins
from pathlib import Path

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "irrep" or name.startswith("irrep."):
        raise AssertionError(f"optional irrep runtime imported: {{name}}")
    if name == "valleyscope.irreps.tables":
        raise AssertionError(f"heavy irrep tables imported: {{name}}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from valleyscope.analysis.database_ingestion_record import (
    load_database_ingestion_record_from_directory,
)
record = load_database_ingestion_record_from_directory(Path({str(fixture)!r}))
assert record["record_status"] == "no_reduced_ebr_input"
assert record["final_reduced_ebr_result_count"] == 0
assert record["ebr_export_status"] == "not_available"
assert record["input_excluded_instance_count"] == 0
assert record["reduced_ebr_mapping_status"] == "not_available"
assert record["validation_errors"] == []
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parent.parent,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
