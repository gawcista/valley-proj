"""Primitive affine standard-setting certificate — fail-closed promotion.

Positive primitive evidence is produced by the REAL resolver fed a real
``StandardIrrepTable`` and a complete spglib detected-operation set, so the
generic affine ``{R | tau}`` equivalence gate actually runs.  Nothing is
mutated after the resolver returns.  Negative tests flip exactly one affine
field of the resolver-produced identity (isolated validator-unit tests).  The
centered case is an isolated validator contract only (Phase E).
"""

import copy

import numpy as np
import pytest

from valleyscope.analysis.reduced_ebr_mapping import (
    promote_bundle_for_solve,
    build_reduced_ebr_mapping,
)
from valleyscope.analysis.standard_setting_kmap import (
    resolve_standard_setting_hsp_label,
)
from valleyscope.analysis.ebr_problem_instances import (
    build_ebr_problem_instances,
)
from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
from valleyscope.irreps.tables import load_standard_irrep_table
from tests.reduced_ebr_promo_helpers import (
    real_primitive_certificate_identity,
    real_primitive_certificate_dict,
    add_real_certificate_to_candidates,
    attach_real_certificate,
    _detected_standard_operations,
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


def _identity(**over):
    cert = copy.deepcopy(_PRIMITIVE_IDENTITY)
    cert.update(over)
    return cert


def _spin_records(spinful):
    return {
        "GammaM": [{"matched_irrep": "A", "irrep_multiplicity": 1,
                    "irrep_source_provenance": {"source_table_spinor": spinful}}],
        "KM": [{"matched_irrep": "A", "irrep_multiplicity": 1,
                "irrep_source_provenance": {"source_table_spinor": spinful}}],
    }


def _table(*, sg_number=143, symbol="P3", spinful=False, **over):
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
    bundle = {
        "bundle_id": "b1", "valley": "K",
        "subspace_group_candidate": symbol, "subspace_sg_number": sg_number,
        "subspace_space_group": {"status": "resolved",
                                 "candidate_space_group_number": sg_number,
                                 "candidate_space_group_symbol": symbol},
        "ready_for_external_solver": True,
        "ready_for_reduced_table_validation": True,
        "expected_hsps": ["GammaM", "KM"],
        "irreps_by_kpoint": {"GammaM": ["A", "A"], "KM": ["A", "B"]},
        "irrep_records_by_kpoint": _spin_records(spinful),
        "certificate_identity": cert if cert is not None else _identity(),
    }
    bundle.update(over)
    return bundle


def _promote(bundle, table):
    return promote_bundle_for_solve(bundle=bundle, table=table)


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


# ---------------------------------------------------------------------------
# Positive promotion from real affine evidence (no mutation)
# ---------------------------------------------------------------------------

def test_real_affine_primitive_promotes():
    ci = real_primitive_certificate_identity(143, "P3")
    assert ci is not None
    assert ci["operation_mapping_status"] == "operation_basis_verification_passed"
    assert ci["affine_validation_status"] == "passed"
    frozen = copy.deepcopy(ci)
    r = _promote(_bundle(cert=ci), _table())
    assert r["promoted"] is True
    assert r["validation_report"]["affine_setting_check"] == "passed"
    assert ci == frozen  # promotion did not mutate the identity


def test_real_affine_primitive_spinful_promotes():
    ci = real_primitive_certificate_identity(75, "P4", spinor=True)
    assert ci is not None
    r = _promote(_bundle(sg_number=75, symbol="P4", spinful=True, cert=ci),
                 _table(sg_number=75, symbol="P4", spinful=True))
    assert r["promoted"] is True


def test_isolated_centered_validator_contract_passes():
    r = _promote(_bundle(sg_number=79, symbol="I4", cert=_centered_identity()),
                 _table(sg_number=79, symbol="I4"))
    assert r["promoted"] is True
    assert r["validation_report"]["affine_setting_check"] == "passed"


def test_full_chain_real_certificate_to_solved():
    """Real resolver certificate -> candidates -> instances -> export -> solve.

    Readiness comes from the real workflow builders; the certificate is the
    real affine resolver product, injected at the candidate level only.
    """
    cert = real_primitive_certificate_dict(143, "P3")
    assert cert is not None

    def _prov():
        return {"standard_setting_hsp_mapping": {
                    "standard_setting_certificate": dict(cert)},
                "source_table_spinor": False}

    ssg = {"candidate_space_group_number": 143,
           "candidate_space_group_symbol": "P3", "status": "resolved"}
    rows = [("GammaM", "A", 2), ("KM", "A", 1), ("KM", "B", 1)]
    candidates = [{
        "ready_for_ebr_input": True, "valley": "K_valley",
        "kpoint": kp, "matched_irrep": irr, "irrep_multiplicity": mult,
        "operation_id": i, "subspace_group_candidate": "P3",
        "subspace_space_group": dict(ssg),
        "irrep_source_provenance": _prov(),
    } for i, (kp, irr, mult) in enumerate(rows)]

    instances = build_ebr_problem_instances(
        ebr_input_candidates={"candidates": candidates})
    export = build_ebr_export_bundle(ebr_problem_instances=instances)
    r = build_reduced_ebr_mapping(ebr_export_bundle=export, table=_table())
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


def test_table_setting_unresolved_blocks():
    r = _promote(_bundle(sg_number=5, symbol="C2"),
                 _table(sg_number=5, symbol="C2"))
    assert r["promoted"] is False
    assert "table_standard_setting_unresolved" in _codes(r)


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
    r = build_reduced_ebr_mapping(ebr_export_bundle={"bundles": [_bundle()]},
                                  table=minimal)
    assert r["status"] != "solved_exact"


def test_attach_real_certificate_contract_fixture_solves():
    table = _table()
    export = attach_real_certificate({"bundles": [_bundle()]}, table)
    assert export is not None
    assert "setting_identity" not in table["provenance"]
    r = build_reduced_ebr_mapping(ebr_export_bundle=export, table=table)
    assert r["status"] == "solved_exact"
