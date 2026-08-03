"""Affine standard-setting certificate fail-closed promotion tests.

Positive certificate evidence is produced by the real resolver from a
``StandardIrrepTable`` and complete spglib operations.  ``_table`` is only a
synthetic low-level validator fixture; it does not prove EBR data provenance.
The SG 79 centered exact-solve test separately uses the default installed
irreptables EBR loader and the runtime auto-canonical table builder.
"""

import copy

import numpy as np
import pytest

from valleyscope.analysis.reduced_ebr_mapping import (
    promote_bundle_for_solve as _promote_bundle_for_solve,
    build_reduced_ebr_mapping as _build_reduced_ebr_mapping,
    _validate_primitive_affine_setting,
)
from valleyscope.analysis.standard_setting_kmap import (
    resolve_standard_setting_hsp_label,
)
from valleyscope.analysis.ebr_problem_instances import (
    build_ebr_problem_instances as _build_ebr_problem_instances,
    _certificate_identity,
)
from valleyscope.analysis.ebr_export_bundle import (
    build_ebr_export_bundle as _build_ebr_export_bundle,
)
from valleyscope.analysis.irreptables_runtime_table_builder import (
    build_auto_canonical_reduced_ebr_table,
)
from valleyscope.irreps.tables import load_standard_irrep_table

from tests.reduced_ebr_promo_helpers import (
    real_primitive_certificate_identity,
    real_primitive_certificate_dict,
    add_real_certificate_to_candidates,
    attach_cprime_fixture_to_candidates,
    attach_cprime_fixture_contract,
    attach_real_certificate,
    cprime_validation_context_for_export,
    _detected_standard_operations,
)


def _build_problem_instances_with_explicit_low_level_cprime_fixture(**kwargs):
    candidates = kwargs.get("ebr_input_candidates")
    if isinstance(candidates, dict):
        attach_cprime_fixture_to_candidates(candidates)
    return _build_ebr_problem_instances(**kwargs)


def _build_export_with_explicit_low_level_cprime_fixture(**kwargs):
    export = _build_ebr_export_bundle(**kwargs)
    attach_cprime_fixture_contract(export)
    return export


def _fixture_cprime_context(export):
    context = cprime_validation_context_for_export(export)
    return dict(context["_by_identity"])


def _map_with_explicit_low_level_cprime_fixture(**kwargs):
    export = kwargs.get("ebr_export_bundle")
    if isinstance(export, dict):
        kwargs.setdefault(
            "cprime_validation_context",
            _fixture_cprime_context(export),
        )
    return _build_reduced_ebr_mapping(**kwargs)


def _promote_with_explicit_low_level_cprime_fixture(*, bundle, table):
    return _promote_bundle_for_solve(
        bundle=bundle,
        table=table,
        cprime_validation_context=_fixture_cprime_context(
            {"bundles": [bundle]}
        ),
    )


# Real affine primitive identity for SG 143 P3 (Hall 430), no post-mutation.
_PRIMITIVE_IDENTITY = real_primitive_certificate_identity(143, "P3")


def _centered_identity(**over):
    """Isolated validator-contract centered identity (real vocabulary, valid
    spglib SG 79 I4 / Hall 353).  NOT a resolver product; Phase E only."""
    cert = {
        "hall_numbers": [353], "hall_symbols": ["I 4"], "centering_types": ["I"],
        "certificate_validation_statuses": ["validated"],
        "any_unresolved": False, "distinct_setting_identities": 1,
        "sg_number": 79, "sg_symbol": "I4",
        "hall_number": 353, "hall_symbol": "I 4", "centering_type": "I",
        "primitive_conventional_relation": "explicit_transform",
        "transform_provenance": "explicit_transform",
        "validation_status": "validated",
        "operation_mapping_status": "operation_basis_verification_passed",
        "affine_validation_status": "passed",
        "normalized_direct_transform": [[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]],
        "normalized_centering_vectors": [[0.5, 0.5, 0.5]],
    }
    cert.update(over)
    return cert


def _real_centered_certificate_dict():
    """Resolver-produced SG 79 I4 centered certificate with opaque IDs."""
    table = load_standard_irrep_table(79, spinor=False)
    transform = np.array([[-0.5, 0.5, 0.5],
                          [0.5, -0.5, 0.5],
                          [0.5, 0.5, -0.5]])
    transform_inv = np.linalg.inv(transform)
    operation_ids = [-7, 0, 4, 11]
    detected = []
    for operation_id, operation in zip(operation_ids, table.operations):
        parent_rotation = transform_inv @ operation.rotation_frac @ transform
        detected.append({
            "operation_id": operation_id,
            "rotation_frac": np.rint(parent_rotation).astype(int).tolist(),
            "translation_frac": (
                transform_inv @ operation.translation_frac
            ).tolist(),
        })
    label, blocker, provenance = resolve_standard_setting_hsp_label(
        k_frac=np.zeros(3), table=table,
        standard_match={
            "number": 79, "international_short": "I4",
            "hall_number": 353, "hall_symbol": "I 4",
            "operation_ids": operation_ids,
        },
        detected_operations=detected,
        parent_to_standard_direct_transform=transform,
        origin_shift_fractional=np.zeros(3),
        transform_provenance="reviewed_test_primitive_to_conventional",
    )
    assert label == "GM"
    assert blocker is None
    return provenance["standard_setting_certificate"]


def _real_centered_identity():
    candidate = {
        "subspace_space_group": {
            "candidate_space_group_number": 79,
            "candidate_space_group_symbol": "I4",
        },
        "irrep_source_provenance": {
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": _real_centered_certificate_dict(),
            },
        },
    }
    return _certificate_identity([candidate])


def _identity(**over):
    cert = copy.deepcopy(_PRIMITIVE_IDENTITY)
    cert.update(over)
    return cert


def _direct_record(*, sampled, source_hsp, irrep, multiplicity, spinful, cert):
    source = f"fixture/K/{source_hsp}/{irrep}"
    identity = {
        "source": source,
        "workflow_path": "direct_qcut",
        "valley": "K",
        "source_hsp_label": source_hsp,
        "sampled_kpoint": sampled,
        "irrep": irrep,
        "multiplicity": multiplicity,
    }
    irrep_provenance = {
        "source_hsp_label": source_hsp,
        "source_table_spinor": spinful,
    }
    return {
        "matched_irrep": irrep,
        "irrep_multiplicity": multiplicity,
        "valley": "K",
        "sampled_kpoint": sampled,
        "source_hsp_label": source_hsp,
        "workflow_path": "direct_qcut",
        "readiness_level": "trusted",
        "source": source,
        "certificate_identity": cert,
        "irrep_source_provenance": irrep_provenance,
        "source_candidate_identity": identity,
        "source_candidate_provenance": {
            "source": source,
            "workflow_path": "direct_qcut",
            "irrep_source_provenance": irrep_provenance,
        },
    }


def _spin_records(spinful, cert):
    return {
        "GammaM": [_direct_record(
            sampled="GammaM", source_hsp="GammaM", irrep="A",
            multiplicity=2, spinful=spinful, cert=cert,
        )],
        "KM": [
            _direct_record(
                sampled="KM", source_hsp="KM", irrep="A",
                multiplicity=1, spinful=spinful, cert=cert,
            ),
            _direct_record(
                sampled="KM", source_hsp="KM", irrep="B",
                multiplicity=1, spinful=spinful, cert=cert,
            ),
        ],
    }


def _complete_coverage(mapping, *, valley="K_valley"):
    required = list(mapping)
    return {
        "by_valley": {
            valley: {
                "required_source_hsp_labels": required,
                "covered_source_hsp_labels": required,
                "missing_source_hsp_labels": [],
                "trusted_matched_source_hsp_labels": required,
                "trusted_missing_source_hsp_labels": [],
                "source_hsp_to_sampled_kpoint": dict(mapping),
                "complete": True,
                "ready_for_ebr_promotion": True,
            }
        }
    }


def _table(*, sg_number=143, symbol="P3", spinful=False, **over):
    """Synthetic table for isolated promotion-validator unit tests only."""
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": symbol,
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:A", "KM:A", "KM:B"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0, 1]},
                 {"label": "EBR_B", "vector": [1, 1, 0]}],
        "provenance": {
            "data_source": "irreptables", "package": "irreptables",
            "package_version": "0.7.1", "space_group_number": sg_number,
            "spinful": spinful,
            "valleyscope_reduction": "sampled_hsp_valley_preserving",
        },
    }
    if over:
        table = copy.deepcopy(table)
        table.update(over)
    return table


def _bundle(*, sg_number=143, symbol="P3", spinful=False, cert=None, **over):
    certificate = cert if cert is not None else _identity()
    bundle = {
        "bundle_id": "b1", "source_instance_id": "i1", "valley": "K",
        "problem_kind": "unitary_valley_reduced_ebr",
        "physical_object_kind": "unitary_valley_projected_subspace",
        "subspace_group_candidate": symbol, "subspace_sg_number": sg_number,
        "subspace_space_group": {"status": "resolved",
                                 "candidate_space_group_number": sg_number,
                                 "candidate_space_group_symbol": symbol},
        "spinor": spinful,
        "workflow_path": "direct_qcut",
        "readiness_level": "trusted",
        "valley_orbit": [],
        "time_reversal": {},
        "unitary_irrep_completion_records_by_hsp": {},
        "unitary_vector_construction": {
            "kind": "direct_observed_unitary_rows",
            "source": "trusted_ebr_input_candidates",
        },
        "ready_for_reduced_table_validation": True,
        "expected_hsps": ["GammaM", "KM"],
        "required_source_hsp_labels": ["GammaM", "KM"],
        "source_hsp_to_sampled_kpoint": {
            "GammaM": "GammaM",
            "KM": "KM",
        },
        "irreps_by_kpoint": {"GammaM": ["A", "A"], "KM": ["A", "B"]},
        "irrep_records_by_kpoint": _spin_records(spinful, certificate),
        "certificate_identity": certificate,
    }
    bundle.update(over)
    attach_cprime_fixture_contract({"bundles": [bundle]})
    return bundle


def _promote(bundle, table):
    return _promote_with_explicit_low_level_cprime_fixture(bundle=bundle, table=table)


def _codes(result):
    return {b["code"] for b in result["blocker_reasons"]}


# ---------------------------------------------------------------------------
# Resolver: primitive direct match requires real affine equivalence
# ---------------------------------------------------------------------------

def test_resolver_primitive_needs_affine_operations():
    """Without detected operations the primitive coordinate match is diagnostic
    only; the certificate is unresolved and no HSP label is trusted."""
    table = load_standard_irrep_table(143, spinor=False)
    sm = {"number": 143, "international_short": "P3", "hall_number": 430,
          "hall_symbol": "P 3", "operation_ids": [0, 1, 2]}
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]), table=table, standard_match=sm)
    assert label is None
    assert blocker is not None
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "unresolved"
    assert cert["operation_mapping_status"] == "not_attempted"


def test_resolver_primitive_with_affine_operations_validates():
    table = load_standard_irrep_table(143, spinor=False)
    detected = _detected_standard_operations(430)
    sm = {"number": 143, "international_short": "P3", "hall_number": 430,
          "hall_symbol": "P 3", "operation_ids": [op["operation_id"]
                                                   for op in detected]}
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]), table=table, standard_match=sm,
        detected_operations=detected)
    assert blocker is None and label == "GM"
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "validated"
    assert cert["operation_mapping_status"] == "operation_basis_verification_passed"
    assert cert["translation_validation_status"] == "passed"
    assert cert["matched_affine_operations"] == cert["total_parent_operations"]
    assert cert["missing_affine_ingredients"] == []
    assert cert["unmatched_parent_operations"] == []
    assert cert["unused_standard_operation_indices"] == []


def _p2_relabelled_certificate_identity():
    """Real Hall-3 P2 affine operations with opaque parent IDs [0, 4]."""
    sym = _detected_standard_operations(3)
    assert sym is not None and len(sym) == 2
    detected = [dict(sym[0]), dict(sym[1])]
    detected[0]["operation_id"] = 0
    detected[1]["operation_id"] = 4
    table = load_standard_irrep_table(3, spinor=False)
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=table,
        standard_match={
            "number": 3,
            "international_short": "P2",
            "hall_number": 3,
            "hall_symbol": "P 2y",
            "operation_ids": [0, 4],
        },
        detected_operations=detected,
    )
    assert label == "GM" and blocker is None
    cert = prov["standard_setting_certificate"]
    candidate = {
        "subspace_space_group": {
            "candidate_space_group_number": 3,
            "candidate_space_group_symbol": "P2",
        },
        "irrep_source_provenance": {
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": cert,
            },
        },
    }
    return _certificate_identity([candidate])


def test_noncontiguous_required_operation_ids_pass_promotion_affine_gate():
    cert_id = _p2_relabelled_certificate_identity()
    assert cert_id["affine_required_operation_ids"] == [0, 4]
    assert cert_id["affine_operation_map"] == {"0": 0, "4": 1}
    blockers = []
    report = {}
    _validate_primitive_affine_setting(
        cert_id,
        cert_id["validation_status"],
        cert_id["primitive_conventional_relation"],
        blockers,
        report,
    )
    assert blockers == []
    assert report["affine_setting_check"] == "passed"


def test_dense_map_keys_rejected_for_noncontiguous_required_operation_ids():
    cert_id = _p2_relabelled_certificate_identity()
    cert_id["affine_operation_map"] = {"0": 0, "1": 1}
    blockers = []
    report = {}
    _validate_primitive_affine_setting(
        cert_id,
        cert_id["validation_status"],
        cert_id["primitive_conventional_relation"],
        blockers,
        report,
    )
    assert len(blockers) == 1
    assert blockers[0]["code"] == "primitive_affine_evidence_invalid"
    assert "keys_do_not_match_required_ids" in blockers[0]["detail"]


def test_legacy_export_without_required_operation_ids_remains_fail_closed():
    cert_id = _identity()
    cert_id.pop("affine_required_operation_ids")
    legacy_export = {
        "schema_version": "1.0.0",
        "bundles": [_bundle(cert=cert_id)],
    }
    result = _map_with_explicit_low_level_cprime_fixture(
        ebr_export_bundle=legacy_export,
        table=_table(),
    )
    assert result["status"] == "blocked"
    assert result["solutions"] == []
    assert len(result["excluded_bundles"]) == 1
    assert "primitive_affine_evidence_invalid" in result["excluded_bundles"][0]["reason"]


# ---------------------------------------------------------------------------
# Positive promotion from real affine evidence (no mutation)
# ---------------------------------------------------------------------------

def test_resolver_primitive_identity_passes_low_level_promotion_validator():
    ci = real_primitive_certificate_identity(143, "P3")
    assert ci is not None
    assert ci["operation_mapping_status"] == "operation_basis_verification_passed"
    assert ci["affine_validation_status"] == "passed"
    frozen = copy.deepcopy(ci)
    r = _promote(_bundle(cert=ci), _table())
    assert r["promoted"] is True
    assert r["validation_report"]["affine_setting_check"] == "passed"
    assert ci == frozen  # promotion did not mutate the identity


def test_absent_raw_audit_fields_remain_none_and_block_promotion():
    raw = real_primitive_certificate_dict(143, "P3")
    assert raw is not None
    raw.pop("unmatched_parent_operations")
    raw.pop("unused_standard_operation_indices")
    candidate = {
        "subspace_space_group": {
            "candidate_space_group_number": 143,
            "candidate_space_group_symbol": "P3",
        },
        "irrep_source_provenance": {
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": raw,
            },
        },
    }
    cert_id = _certificate_identity([candidate])
    assert cert_id["affine_unmatched_parent_operations"] is None
    assert cert_id["affine_unused_standard_operation_indices"] is None
    out = _promote(_bundle(cert=cert_id), _table())
    assert out["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(out)


@pytest.mark.parametrize("field,value", [
    ("affine_unmatched_parent_operations", "bad"),
    ("affine_unmatched_parent_operations", {}),
    ("affine_unmatched_parent_operations", ()),
    ("affine_unmatched_parent_operations", False),
    ("affine_unmatched_parent_operations", [{"operation_id": 0}]),
    ("affine_unused_standard_operation_indices", "bad"),
    ("affine_unused_standard_operation_indices", {}),
    ("affine_unused_standard_operation_indices", ()),
    ("affine_unused_standard_operation_indices", False),
    ("affine_unused_standard_operation_indices", ["0"]),
    ("affine_missing_ingredients", "bad"),
    ("affine_missing_ingredients", {}),
    ("affine_missing_ingredients", ()),
    ("affine_missing_ingredients", False),
    ("affine_missing_ingredients", [0]),
    ("affine_required_operation_ids", "bad"),
    ("affine_required_operation_ids", {}),
    ("affine_required_operation_ids", (0, 1, 2)),
    ("affine_required_operation_ids", False),
    ("affine_required_operation_ids", [0, True, 2]),
    ("affine_operation_map", "bad"),
    ("affine_operation_map", []),
    ("affine_operation_map", False),
    ("affine_operation_map", {"0": 0, "1": "1", "2": 2}),
])
def test_malformed_affine_audit_evidence_blocks(field, value):
    cert = _identity()
    cert[field] = value
    out = _promote(_bundle(cert=cert), _table())
    assert out["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(out)


def test_operation_map_key_aliases_are_rejected():
    cert = _identity()
    cert["affine_operation_map"] = {0: 0, "0": 1, "2": 2}
    out = _promote(_bundle(cert=cert), _table())
    assert out["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(out)


def test_resolver_spinful_identity_passes_low_level_promotion_validator():
    ci = real_primitive_certificate_identity(75, "P4", spinor=True)
    assert ci is not None
    r = _promote(_bundle(sg_number=75, symbol="P4", spinful=True, cert=ci),
                 _table(sg_number=75, symbol="P4", spinful=True))
    assert r["promoted"] is True


def test_legacy_centered_validator_contract_without_expansion_map_fails_closed():
    r = _promote(_bundle(sg_number=79, symbol="I4", cert=_centered_identity()),
                 _table(sg_number=79, symbol="I4"))
    assert r["promoted"] is False
    assert "centered_affine_evidence_invalid" in _codes(r)


def test_resolver_centered_identity_passes_low_level_promotion_validator():
    cert = _real_centered_identity()
    r = _promote(
        _bundle(sg_number=79, symbol="I4", cert=cert),
        _table(sg_number=79, symbol="I4"),
    )
    assert r["promoted"] is True
    assert r["validation_report"]["affine_setting_check"] == "passed"
    assert r["validation_report"]["hall_setting_check"] == "passed"


def test_sg79_centered_resolver_to_exact_solve_uses_default_irreptables_table():
    raw_certificate = _real_centered_certificate_dict()

    standard_table = load_standard_irrep_table(79, spinor=False)
    gamma_labels = [
        irrep.label for irrep in standard_table.irreps
        if irrep.kpoint_label == "GM"
    ]
    assert gamma_labels == ["GM1", "GM2", "GM3", "GM4"]

    subspace_group = {
        "candidate_space_group_number": 79,
        "candidate_space_group_symbol": "I4",
        "status": "resolved",
    }
    generated_table = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=79,
        spinor=False,
        bundle_irreps_by_kpoint={"GammaM": [gamma_labels[0]]},
        expected_hsps=["GammaM"],
        subspace_group_candidate="I4",
        subspace_space_group=subspace_group,
    )
    setting_identity = generated_table["provenance"][
        "standard_setting_identity"
    ]
    assert generated_table["provenance"]["data_source"] == "irreptables"
    assert generated_table["provenance"]["package"] == "irreptables"
    assert generated_table["provenance"]["package_version"]
    assert generated_table["provenance"]["auto_canonical"] is True
    assert setting_identity["status"] == "unique_match"
    assert setting_identity["hall_number"] == 353
    assert setting_identity["centering_type"] == "I"

    selected_ebr = next(
        ebr for ebr in generated_table["ebrs"] if any(ebr["vector"])
    )
    target_rows = [
        (irrep_key.split(":", 1)[0], irrep_key.split(":", 1)[1], multiplicity)
        for irrep_key, multiplicity
        in zip(generated_table["irreps"], selected_ebr["vector"])
        if multiplicity
    ]
    assert target_rows

    def _provenance():
        return {
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": copy.deepcopy(raw_certificate),
            },
            "source_table_spinor": False,
            "source_table_sg_number": 79,
            "source_table_name": "I4",
            "source_hsp_label": "GM",
        }

    candidates = [{
        "ready_for_ebr_input": True,
        "valley": "K_valley",
        "kpoint": kpoint,
        "matched_irrep": irrep,
        "irrep_multiplicity": multiplicity,
        "operation_id": operation_id,
        "workflow_path": "direct_qcut",
        "readiness_level": "trusted",
        "source": f"fixture/K_valley/{kpoint}/{irrep}",
        "subspace_group_candidate": "I4",
        "subspace_space_group": dict(subspace_group),
        "irrep_source_provenance": _provenance(),
    } for operation_id, (kpoint, irrep, multiplicity) in enumerate(target_rows)]

    instances = _build_problem_instances_with_explicit_low_level_cprime_fixture(
        ebr_input_candidates={"candidates": candidates},
        projected_hsp_coverage=_complete_coverage({"GM": "GammaM"}),
    )
    export = _build_export_with_explicit_low_level_cprime_fixture(ebr_problem_instances=instances)
    exported_identity = export["bundles"][0]["certificate_identity"]
    expected_identity = _real_centered_identity()
    assert exported_identity["centered_affine_operation_map"] == (
        expected_identity["centered_affine_operation_map"]
    )
    assert exported_identity["affine_required_operation_ids"] == [-7, 0, 4, 11]

    result = _map_with_explicit_low_level_cprime_fixture(
        ebr_export_bundle=export,
        table=generated_table,
    )
    assert result["status"] == "solved_exact"
    solution = result["solutions"][0]
    assert solution["irrep_vector"] == selected_ebr["vector"]
    assert solution["ebr_decomposition"] == [{
        "label": selected_ebr["label"], "coefficient": 1,
    }]
    assert set(solution["validation_report"].values()) == {"passed"}
    promotion_table = solution["promotion_provenance"]["table_provenance"]
    assert promotion_table["data_source"] == "irreptables"
    assert promotion_table["package_version"] == (
        generated_table["provenance"]["package_version"]
    )
    assert promotion_table["independent_setting_identity"]["hall_number"] == 353
    solution_identity = solution["certificate_identity"]
    assert solution_identity["centered_affine_operation_map"] == (
        expected_identity["centered_affine_operation_map"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("centered_affine_operation_map", []),
        ("centered_affine_operation_map", [{
            "parent_operation_id": -7,
            "centering_coset_index": 0,
            "standard_operation_index": 0,
        }]),
        ("affine_unmatched_centered_operation_pairs", None),
        ("centering_coset_count", 1),
        ("primitive_conventional_index", 1),
        ("expanded_parent_operation_count", 7),
        ("matched_expanded_operations", 7),
        ("standard_operation_closure_validated", False),
        ("primitive_conventional_relation",
         "spglib_affine_subgroup_standardization"),
    ],
)
def test_malformed_or_incomplete_centered_affine_evidence_blocks(field, value):
    cert = _real_centered_identity()
    cert[field] = value
    r = _promote(
        _bundle(sg_number=79, symbol="I4", cert=cert),
        _table(sg_number=79, symbol="I4"),
    )
    assert r["promoted"] is False
    assert "centered_affine_evidence_invalid" in _codes(r)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_coset",
        "duplicate_coset",
        "wrong_coset",
        "reordered_cosets",
        "reused_standard",
    ],
)
def test_nonbijective_centered_affine_evidence_blocks(mutation):
    cert = _real_centered_identity()
    if mutation == "missing_coset":
        cert["normalized_centering_vectors"] = [[0.0, 0.0, 0.0]]
    elif mutation == "duplicate_coset":
        cert["normalized_centering_vectors"] = [
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ]
    elif mutation == "wrong_coset":
        cert["normalized_centering_vectors"] = [
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
        ]
    elif mutation == "reordered_cosets":
        cert["normalized_centering_vectors"] = [
            [0.5, 0.5, 0.5], [0.0, 0.0, 0.0],
        ]
    else:
        cert["centered_affine_operation_map"][1]["standard_operation_index"] = 0

    result = _promote(
        _bundle(sg_number=79, symbol="I4", cert=cert),
        _table(sg_number=79, symbol="I4"),
    )
    assert result["promoted"] is False
    assert "centered_affine_evidence_invalid" in _codes(result)


def test_injected_primitive_certificate_with_synthetic_table_solves_plumbing():
    """Resolver certificate -> plumbing -> synthetic validator table solve.

    This is not production EBR provenance evidence.  It checks only that the
    resolver-produced affine identity survives the plumbing layers.
    """
    cert = real_primitive_certificate_dict(143, "P3")
    assert cert is not None

    def _prov(source_hsp):
        return {"standard_setting_hsp_mapping": {
                    "standard_setting_certificate": dict(cert)},
                "source_table_spinor": False,
                "source_hsp_label": source_hsp}

    ssg = {"candidate_space_group_number": 143,
           "candidate_space_group_symbol": "P3", "status": "resolved"}
    rows = [("GammaM", "A", 2), ("KM", "A", 1), ("KM", "B", 1)]
    candidates = [{
        "ready_for_ebr_input": True, "valley": "K_valley",
        "kpoint": kp, "matched_irrep": irr, "irrep_multiplicity": mult,
        "workflow_path": "direct_qcut", "readiness_level": "trusted",
        "source": f"fixture/K_valley/{kp}/{irr}",
        "operation_id": i, "subspace_group_candidate": "P3",
        "subspace_space_group": dict(ssg),
        "irrep_source_provenance": _prov("GM" if kp == "GammaM" else "K"),
    } for i, (kp, irr, mult) in enumerate(rows)]

    instances = _build_problem_instances_with_explicit_low_level_cprime_fixture(
        ebr_input_candidates={"candidates": candidates},
        projected_hsp_coverage=_complete_coverage(
            {"GM": "GammaM", "K": "KM"}
        ),
    )
    export = _build_export_with_explicit_low_level_cprime_fixture(ebr_problem_instances=instances)
    r = _map_with_explicit_low_level_cprime_fixture(ebr_export_bundle=export, table=_table())
    assert r["status"] == "solved_exact"
    sol = r["solutions"][0]
    assert sol["certificate_identity"]["operation_mapping_status"] == \
        "operation_basis_verification_passed"


# ---------------------------------------------------------------------------
# Required affine negative tests (handoff section D)
# ---------------------------------------------------------------------------

def test_no_detected_operations_blocks_primitive_promotion():
    # An identity whose affine evidence is not_attempted must not promote.
    cert = _identity(operation_mapping_status="not_attempted",
                     affine_validation_status="not_attempted",
                     affine_matched_operations=None,
                     affine_total_operations=None)
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(r)


def test_operation_mapping_not_attempted_blocks():
    r = _promote(_bundle(cert=_identity(
        operation_mapping_status="not_attempted")), _table())
    assert r["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(r)


def test_affine_validation_not_attempted_blocks():
    r = _promote(_bundle(cert=_identity(
        affine_validation_status="not_attempted")), _table())
    assert r["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(r)


@pytest.mark.parametrize("matched,total", [
    (None, 3), (3, None), (0, 0), (2, 3), (3, 2),
])
def test_bad_operation_counts_block(matched, total):
    r = _promote(_bundle(cert=_identity(
        affine_matched_operations=matched,
        affine_total_operations=total)), _table())
    assert r["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(r)


def test_nonzero_translation_mismatch_blocks():
    r = _promote(_bundle(cert=_identity(affine_mismatch_count=2)), _table())
    assert r["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(r)


def test_nonempty_missing_ingredients_blocks():
    r = _promote(_bundle(cert=_identity(
        affine_missing_ingredients=["parent_translation_frac"])), _table())
    assert r["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(r)


@pytest.mark.parametrize("transform", [
    None,
    [[1.0, 0.0], [0.0, 1.0]],                       # wrong shape
    [[float("inf"), 0, 0], [0, 1, 0], [0, 0, 1]],   # non-finite
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],              # singular
])
def test_bad_direct_transform_blocks(transform):
    r = _promote(_bundle(cert=_identity(
        normalized_direct_transform=transform)), _table())
    assert r["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(r)


def test_operation_closure_false_blocks():
    r = _promote(_bundle(cert=_identity(
        operation_closure_validated=False)), _table())
    assert r["promoted"] is False
    assert "primitive_affine_evidence_invalid" in _codes(r)


# ---------------------------------------------------------------------------
# Retained prior fail-open reproductions (identity integrity)
# ---------------------------------------------------------------------------

def test_repro_certificate_symbol_conflict():
    r = _promote(_bundle(cert=_identity(sg_symbol="P4")), _table())
    assert r["promoted"] is False
    assert "certificate_symbol_conflict" in _codes(r)


def test_repro_bundle_table_symbol_conflict_with_sg():
    r = _promote(_bundle(symbol="P4"), _table(symbol="P4"))
    assert r["promoted"] is False
    assert {"bundle_symbol_conflict", "table_symbol_conflict"} & _codes(r)


def test_repro_certificate_sg_missing():
    cert = _identity()
    cert.pop("sg_number")
    cert.pop("sg_symbol")
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_sg_number_missing" in _codes(r)


def test_repro_producer_identity_collections_missing():
    cert = _identity()
    for k in ("hall_numbers", "hall_symbols", "centering_types",
              "certificate_validation_statuses"):
        cert.pop(k)
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_field_inconsistent" in _codes(r)


def test_repro_zero_distinct_setting_identities():
    r = _promote(_bundle(cert=_identity(distinct_setting_identities=0)), _table())
    assert r["promoted"] is False
    assert "certificate_ambiguous_setting" in _codes(r)


def test_sg_143_with_hall_1_rejected():
    cert = _identity(hall_number=1, hall_numbers=[1], hall_symbol="P 1",
                     hall_symbols=["P 1"])
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert {"hall_sg_inconsistent", "setting_identity_mismatch"} & _codes(r)


def test_centered_table_setting_is_independently_resolved_before_cert_conflict():
    r = _promote(_bundle(sg_number=5, symbol="C2"),
                 _table(sg_number=5, symbol="C2"))
    assert r["promoted"] is False
    assert r["validation_report"]["table_setting_check"] == "passed"
    assert "certificate_sg_conflict" in _codes(r)


@pytest.mark.parametrize("mutate,code", [
    ({"data_source": "invented"}, "table_data_source_invalid"),
    ({"package": "not-irreptables"}, "table_package_invalid"),
    ({"valleyscope_reduction": "raw_3d"}, "table_reduction_provenance_invalid"),
    ({"spinful": "yes"}, "table_spinful_missing"),
    ({"space_group_number": 0}, "table_sg_number_missing"),
])
def test_table_provenance_fields_block(mutate, code):
    table = _table()
    table["provenance"].update(mutate)
    r = _promote(_bundle(), table)
    assert r["promoted"] is False
    assert code in _codes(r)


# ---------------------------------------------------------------------------
# Spin / HSP / irrep / production path
# ---------------------------------------------------------------------------

def test_spin_evidence_conflict_blocks():
    b = _bundle(spinful=False)
    b["irrep_records_by_kpoint"]["KM"][0][
        "irrep_source_provenance"]["source_table_spinor"] = True
    r = _promote(b, _table(spinful=False))
    assert r["promoted"] is False
    assert "spin_evidence_conflict" in _codes(r)


def test_hsp_mismatch_blocks():
    b = _bundle(expected_hsps=["GammaM"],
                irreps_by_kpoint={"GammaM": ["A", "A"]})
    r = _promote(b, _table())
    assert r["promoted"] is False
    assert "hsp_basis_mismatch" in _codes(r)


def test_unresolved_irrep_key_blocks():
    b = _bundle(irreps_by_kpoint={"GammaM": ["A", "A"], "KM": ["A", "ZZ"]})
    r = _promote(b, _table())
    assert r["promoted"] is False
    assert "irrep_key_unresolved" in _codes(r)


def test_missing_certificate_blocks():
    b = _bundle()
    del b["certificate_identity"]
    r = _promote(b, _table())
    assert r["promoted"] is False
    assert "certificate_missing" in _codes(r)


def test_arbitrary_minimal_table_rejected():
    minimal = {
        "schema_version": "1.0.0", "subspace_group_candidate": "P3",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:A", "KM:A", "KM:B"],
        "ebrs": [{"label": "EBR_A", "vector": [1, 0, 1]},
                 {"label": "EBR_B", "vector": [1, 1, 0]}],
    }
    r = _map_with_explicit_low_level_cprime_fixture(ebr_export_bundle={"bundles": [_bundle()]},
                                  table=minimal)
    assert r["status"] != "solved_exact"


def test_attach_resolver_certificate_to_synthetic_validator_fixture_solves():
    table = _table()
    export = attach_real_certificate({"bundles": [_bundle()]}, table)
    assert export is not None
    attach_cprime_fixture_contract(export)
    assert "setting_identity" not in table["provenance"]
    r = _map_with_explicit_low_level_cprime_fixture(ebr_export_bundle=export, table=table)
    assert r["status"] == "solved_exact"


# ---------------------------------------------------------------------------
# E1: P321 parent ops with P3 subgroup → validated
# ---------------------------------------------------------------------------

def test_p321_parent_p3_subgroup_passes():
    """Full generic P321 parent operations with selected P3 subgroup validate."""
    import spglib
    sym = spglib.get_symmetry_from_database(439)  # P321
    ops = [{"operation_id": i, "rotation_frac": np.asarray(r, float).tolist(),
            "translation_frac": np.asarray(t, float).tolist()}
           for i, (r, t) in enumerate(zip(sym["rotations"], sym["translations"]))]
    # P321 parent ops with P3 subgroup IDs [0,1,2]: use P3 Hall 430
    # for the selected subgroup standard match.
    table = load_standard_irrep_table(143, spinor=False)
    _, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=table,
        standard_match={"number": 143, "international_short": "P3",
                        "hall_number": 430, "hall_symbol": "P 3",
                        "operation_ids": [0, 1, 2]},
        detected_operations=ops,
        parent_to_standard_direct_transform=np.eye(3),
    )
    assert blocker is None
    cert = prov["standard_setting_certificate"]
    assert cert["validation_status"] == "validated"


# ---------------------------------------------------------------------------
# E2: Extra non-required parent operations do not block subgroup validation
# ---------------------------------------------------------------------------

def test_extra_non_required_ops_do_not_block():
    """Extra parent operations outside the required ID set are silently filtered."""
    ident = {"operation_id": 0, "rotation_frac": np.eye(3).tolist(),
             "translation_frac": [0., 0., 0.]}
    extra = {"operation_id": 99, "rotation_frac": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
             "translation_frac": [0.3, 0.3, 0.0]}
    ops = [ident, extra]
    _, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=load_standard_irrep_table(1, spinor=False),
        standard_match={"number": 1, "international_short": "P1",
                        "hall_number": 1, "hall_symbol": "P 1",
                        "operation_ids": [0]},
        detected_operations=ops,
    )
    assert blocker is None
    assert prov["standard_setting_certificate"]["validation_status"] == "validated"


# ---------------------------------------------------------------------------
# E3-4: Closure failure blocks at resolver level (not only promotion)
# ---------------------------------------------------------------------------

def test_closure_failure_blocks_resolver_hsp_label():
    """A non-closed parent operation set must not return a trusted HSP label."""
    ops = [
        {"operation_id": 0, "rotation_frac": np.eye(3).tolist(),
         "translation_frac": [0., 0., 0.]},
        {"operation_id": 1, "rotation_frac": [[0, -1, 0], [1, -1, 0], [0, 0, 1]],
         "translation_frac": [0., 0., 0.]},
        # Missing inverse of op 1 (op 2) → not closed.
    ]
    label, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=load_standard_irrep_table(143, spinor=False),
        standard_match={"number": 143, "international_short": "P3",
                        "hall_number": 430, "hall_symbol": "P 3",
                        "operation_ids": [0, 1]},
        detected_operations=ops,
    )
    assert label is None
    assert blocker is not None
    assert "parent_standard_group_order_mismatch" in str(prov)


# ---------------------------------------------------------------------------
# E5-6: Map-integrity negatives for promotion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,patch,expected_code", [
    ("short_map", {"affine_operation_map": {"0": 0}},
     "primitive_affine_evidence_invalid"),
    ("duplicate_targets", {"affine_operation_map": {"0": 0, "1": 0, "2": 0}},
     "primitive_affine_evidence_invalid"),
    ("missing_key", {"affine_operation_map": {"0": 0, "1": 1}},
     "primitive_affine_evidence_invalid"),
    ("extra_key", {"affine_operation_map": {"0": 0, "1": 1, "2": 2, "3": 0}},
     "primitive_affine_evidence_invalid"),
    ("out_of_range_target", {"affine_operation_map": {"0": 0, "1": 1, "2": 99}},
     "primitive_affine_evidence_invalid"),
    ("non_integer_target", {"affine_operation_map": {"0": 0, "1": 1, "2": "0"}},
     "primitive_affine_evidence_invalid"),
    ("nonempty_unmatched",
     {"affine_operation_map": {"0": 0, "1": 1, "2": 2},
      "affine_unmatched_parent_operations": ["anything"]},
     "primitive_affine_evidence_invalid"),
    ("nonempty_unused",
     {"affine_operation_map": {"0": 0, "1": 1, "2": 2},
      "affine_unused_standard_operation_indices": [4]},
     "primitive_affine_evidence_invalid"),
    ("absent_unmatched",
     {"affine_operation_map": {"0": 0, "1": 1, "2": 2},
      "affine_unmatched_parent_operations": None},
     "primitive_affine_evidence_invalid"),
    ("absent_unused",
     {"affine_operation_map": {"0": 0, "1": 1, "2": 2},
      "affine_unused_standard_operation_indices": None},
     "primitive_affine_evidence_invalid"),
])
def test_map_integrity_blocks_promotion(name, patch, expected_code):
    base = real_primitive_certificate_identity(143, "P3")
    cert = copy.deepcopy(base)
    for key, val in patch.items():
        if isinstance(val, dict):
            cert[key] = dict(val)
        else:
            cert[key] = val
    out = _promote(_bundle(cert=cert), _table())
    assert out["promoted"] is False, name
    assert expected_code in _codes(out)


# ---------------------------------------------------------------------------
# E7: Real P3 and P4 identity promotes with exact map keys/values
# ---------------------------------------------------------------------------

def test_resolver_p3_identity_has_exact_map_in_low_level_validator():
    ci = real_primitive_certificate_identity(143, "P3")
    assert ci is not None
    assert ci["affine_operation_map"] == {"0": 0, "1": 1, "2": 2}
    assert ci["affine_unmatched_parent_operations"] == []
    assert ci["affine_unused_standard_operation_indices"] == []
    out = _promote(_bundle(cert=ci), _table())
    assert out["promoted"] is True


def test_resolver_p4_identity_passes_low_level_promotion_validator():
    ci = real_primitive_certificate_identity(75, "P4", spinor=True)
    assert ci is not None
    assert len(ci["affine_operation_map"]) == 4  # P4 has 4 ops
    out = _promote(_bundle(sg_number=75, symbol="P4", spinful=True, cert=ci),
                   _table(sg_number=75, symbol="P4", spinful=True))
    assert out["promoted"] is True


# ---------------------------------------------------------------------------
# E8: Certificate-injection test accurately labelled (lower-level plumbing)
# ---------------------------------------------------------------------------

def test_injected_certificate_to_solve_is_plumbing_not_production():
    """A certificate built by the real resolver, injected at candidate level,
    reaches solved_exact.  This is a lower-level plumbing integration test,
    not a full production workflow test (the real workflow currently
    produces an unresolved certificate; Phase E)."""
    cert = real_primitive_certificate_dict(143, "P3")
    assert cert is not None

    def _prov(source_hsp):
        return {"standard_setting_hsp_mapping": {
                    "standard_setting_certificate": dict(cert)},
                "source_table_spinor": False,
                "source_hsp_label": source_hsp}

    ssg = {"candidate_space_group_number": 143,
           "candidate_space_group_symbol": "P3", "status": "resolved"}
    rows = [("GammaM", "A", 2), ("KM", "A", 1), ("KM", "B", 1)]
    candidates = [{
        "ready_for_ebr_input": True, "valley": "K_valley",
        "kpoint": kp, "matched_irrep": irr, "irrep_multiplicity": mult,
        "workflow_path": "direct_qcut", "readiness_level": "trusted",
        "source": f"fixture/K_valley/{kp}/{irr}",
        "operation_id": i, "subspace_group_candidate": "P3",
        "subspace_space_group": dict(ssg),
        "irrep_source_provenance": _prov("GM" if kp == "GammaM" else "K"),
    } for i, (kp, irr, mult) in enumerate(rows)]

    instances = _build_problem_instances_with_explicit_low_level_cprime_fixture(
        ebr_input_candidates={"candidates": candidates},
        projected_hsp_coverage=_complete_coverage(
            {"GM": "GammaM", "K": "KM"}
        ),
    )
    export = _build_export_with_explicit_low_level_cprime_fixture(ebr_problem_instances=instances)
    r = _map_with_explicit_low_level_cprime_fixture(ebr_export_bundle=export, table=_table())
    assert r["status"] == "solved_exact"
