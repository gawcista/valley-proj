"""Certificate identity integrity — fail-closed promotion validator.

Positive promotion evidence is produced by the REAL resolver
(``resolve_standard_setting_hsp_label`` via ``resolver_certificate_identity``)
and never mutated afterwards.  Negative tests are isolated validator-unit
tests that flip exactly one field of the resolver-produced identity.  The
centered positive case is an isolated validator contract only (Phase E does
not yet emit a centered affine certificate) and is labelled as such.
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
    _certificate_identity,
)
from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
from tests.reduced_ebr_promo_helpers import (
    resolver_certificate_identity,
    apply_resolver_certificate,
    _GammaCoordinateTable,
)

# Real resolver output for SG 143 P3 (Hall 430 "P 3"), never mutated.
_PRIMITIVE_IDENTITY = resolver_certificate_identity(143, "P3")


def _centered_identity(**over):
    """Isolated validator-contract centered identity (real vocabulary, valid
    spglib SG 79 I4 / Hall 353).  NOT produced by a resolver; Phase E only."""
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
        "normalized_origin_shift": [0.0, 0.0, 0.0],
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
# Resolver produces a valid identity with no post-mutation
# ---------------------------------------------------------------------------

def test_resolver_identity_is_complete_and_consistent():
    ci = resolver_certificate_identity(143, "P3")
    assert ci["validation_status"] == "validated"
    assert ci["primitive_conventional_relation"] == "direct_coordinate_match"
    assert ci["hall_numbers"] == [ci["hall_number"]] == [430]
    assert ci["hall_symbols"] == [ci["hall_symbol"]] == ["P 3"]
    assert ci["centering_types"] == [ci["centering_type"]] == ["P"]
    assert ci["certificate_validation_statuses"] == ["validated"]
    assert ci["distinct_setting_identities"] == 1
    assert ci["any_unresolved"] is False


def test_reviewed_primitive_spinless_passes():
    r = _promote(_bundle(), _table())
    assert r["promoted"] is True
    assert r["blocker_reasons"] == []
    assert r["canonical_state"] == "validated_basis"


def test_reviewed_primitive_spinful_passes():
    r = _promote(_bundle(spinful=True), _table(spinful=True))
    assert r["promoted"] is True
    assert r["validation_report"]["spin_convention_check"] == "passed"


def test_isolated_centered_validator_contract_passes():
    # Isolated validator contract only (Phase E does not yet emit this).
    r = _promote(_bundle(sg_number=79, symbol="I4", cert=_centered_identity()),
                 _table(sg_number=79, symbol="I4"))
    assert r["promoted"] is True
    assert r["validation_report"]["affine_setting_check"] == "passed"


# ---------------------------------------------------------------------------
# Real resolver -> identity -> promotion (no post-processing)
# ---------------------------------------------------------------------------

def test_real_resolver_identity_promotes_without_mutation():
    ci = resolver_certificate_identity(143, "P3")
    frozen = copy.deepcopy(ci)
    r = _promote(_bundle(cert=ci), _table())
    assert r["promoted"] is True
    assert ci == frozen  # promotion did not modify the identity


def test_full_chain_resolver_to_solved():
    """resolver certificate -> candidates -> instances -> export -> solve.

    Readiness is set by the real workflow builders; the certificate comes from
    the resolver and is not modified.
    """
    _, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=_GammaCoordinateTable(143, "P3"),
        standard_match={"number": 143, "international_short": "P3",
                        "hall_number": 430, "hall_symbol": "P 3",
                        "operation_ids": [0, 1, 2]},
    )
    assert blocker is None
    cert = prov["standard_setting_certificate"]

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
    assert r["solutions"][0]["certificate_identity"]["validation_status"] == \
        "validated"


# ---------------------------------------------------------------------------
# Codex reproductions (must block with specific structured codes)
# ---------------------------------------------------------------------------

def test_repro_certificate_symbol_conflict():
    r = _promote(_bundle(cert=_identity(sg_symbol="P4")), _table())
    assert r["promoted"] is False
    assert "certificate_symbol_conflict" in _codes(r)


def test_repro_bundle_table_symbol_conflict_with_sg():
    # Bundle and table both claim P4 while SG number/Hall/certificate are P3.
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
    assert "certificate_sg_symbol_missing" in _codes(r)


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


# ---------------------------------------------------------------------------
# Required negative tests (handoff section D)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", [
    "hall_numbers", "hall_symbols", "centering_types",
    "certificate_validation_statuses",
])
def test_missing_singular_plural_field_blocks(field):
    cert = _identity()
    cert.pop(field)
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_field_inconsistent" in _codes(r)


@pytest.mark.parametrize("field,value", [
    ("hall_numbers", [430, 430]),
    ("hall_numbers", [430, 431]),
    ("hall_symbols", ["P 3", "P 3"]),
    ("centering_types", ["P", "P"]),
    ("certificate_validation_statuses", ["validated", "validated"]),
])
def test_extra_or_duplicate_plural_blocks(field, value):
    r = _promote(_bundle(cert=_identity(**{field: value})), _table())
    assert r["promoted"] is False
    assert "certificate_field_inconsistent" in _codes(r)


@pytest.mark.parametrize("value", [None, 0, True, False, 2, "1", 1.0])
def test_malformed_distinct_setting_identities_blocks(value):
    r = _promote(_bundle(cert=_identity(distinct_setting_identities=value)),
                 _table())
    assert r["promoted"] is False
    assert "certificate_ambiguous_setting" in _codes(r)


def test_missing_hall_number_blocks():
    cert = _identity()
    cert.pop("hall_number")
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_hall_number_missing" in _codes(r)


def test_any_unresolved_not_false_blocks():
    r = _promote(_bundle(cert=_identity(any_unresolved=True,
                                        validation_status="unresolved",
                                        certificate_validation_statuses=["unresolved"])),
                 _table())
    assert r["promoted"] is False
    assert "certificate_unresolved" in _codes(r)


# ---------------------------------------------------------------------------
# Crystallographic / spglib consistency
# ---------------------------------------------------------------------------

def test_sg_143_with_hall_1_rejected():
    cert = _identity(hall_number=1, hall_numbers=[1], hall_symbol="P 1",
                     hall_symbols=["P 1"])
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert {"hall_sg_inconsistent", "setting_identity_mismatch"} & _codes(r)


def test_centering_inconsistent_with_hall_blocks():
    cert = _identity(centering_type="C", centering_types=["C"])
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert {"centering_hall_inconsistent", "setting_identity_mismatch"} & _codes(r)


def test_table_setting_unresolved_blocks():
    # SG 5 has multiple spglib Hall settings -> unresolved.
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
# Spin / HSP / irrep
# ---------------------------------------------------------------------------

def test_spin_evidence_conflict_blocks():
    b = _bundle(spinful=False)
    b["irrep_records_by_kpoint"]["KM"][0][
        "irrep_source_provenance"]["source_table_spinor"] = True
    r = _promote(b, _table(spinful=False))
    assert r["promoted"] is False
    assert "spin_evidence_conflict" in _codes(r)


def test_spin_mismatch_blocks():
    r = _promote(_bundle(spinful=True), _table(spinful=False))
    assert r["promoted"] is False
    assert "spin_convention_mismatch" in _codes(r)


def test_spin_missing_blocks():
    b = _bundle()
    b["irrep_records_by_kpoint"] = {}
    r = _promote(b, _table())
    assert r["promoted"] is False
    assert "spin_convention_missing" in _codes(r)


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


# ---------------------------------------------------------------------------
# Production mapping path (no injected pass)
# ---------------------------------------------------------------------------

def test_premarked_ready_revalidated_and_blocked():
    cert = _identity(any_unresolved=True, validation_status="not_evaluated",
                     certificate_validation_statuses=["not_evaluated"])
    b = _bundle(cert=cert)
    r = build_reduced_ebr_mapping(ebr_export_bundle={"bundles": [b]},
                                  table=_table())
    assert r["status"] != "solved_exact"
    assert not r["solutions"]
    assert len(r["excluded_bundles"]) == 1


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
    codes = {bl["code"] for bl in r["excluded_bundles"][0]["blocker_reasons"]}
    assert "table_sg_number_missing" in codes


def test_apply_resolver_certificate_does_not_copy_setting():
    table = _table()
    export = apply_resolver_certificate({"bundles": [_bundle()]}, table)
    assert export is not None
    assert "setting_identity" not in table["provenance"]
    r = build_reduced_ebr_mapping(ebr_export_bundle=export, table=table)
    assert r["status"] == "solved_exact"
