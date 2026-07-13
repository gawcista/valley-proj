"""Trusted reduced-EBR promotion validator — fail-closed contract.

Each negative test flips exactly one physical field from a shared valid
fixture and asserts the single structured blocker code that must fire.  The
positive tests exercise the same validator and state transition used by the
production mapping path; there is no ``require_reviewed_table`` bypass.
"""

import copy

import pytest

from valleyscope.analysis.reduced_ebr_mapping import (
    promote_bundle_for_solve,
    build_reduced_ebr_mapping,
)


# ---------------------------------------------------------------------------
# Shared valid fixture builders
# ---------------------------------------------------------------------------

def _valid_cert(**over):
    """A validated primitive standard-setting certificate identity."""
    cert = {
        "hall_numbers": [430],
        "hall_symbols": ["P 3"],
        "centering_types": ["P"],
        "certificate_validation_statuses": ["validated"],
        "any_unresolved": False,
        "distinct_setting_identities": 1,
        "sg_number": 143,
        "sg_symbol": "P3",
        "hall_number": 430,
        "hall_symbol": "P 3",
        "centering_type": "P",
        "primitive_conventional_relation": "identity",
        "transform_provenance": "derived_affine_equivalence",
        "validation_status": "validated",
        "operation_mapping_status": "validated",
        "affine_validation_status": "validated",
    }
    cert.update(over)
    return cert


def _centered_cert(**over):
    """A validated C-centered certificate with full affine evidence."""
    cert = {
        "hall_numbers": [300],
        "hall_symbols": ["C 2y"],
        "centering_types": ["C"],
        "certificate_validation_statuses": ["validated"],
        "any_unresolved": False,
        "distinct_setting_identities": 1,
        "sg_number": 5,
        "sg_symbol": "C2",
        "hall_number": 300,
        "hall_symbol": "C 2y",
        "centering_type": "C",
        "primitive_conventional_relation": "c_centered_conventional",
        "transform_provenance": "derived_affine_equivalence",
        "validation_status": "validated",
        "operation_mapping_status": "validated",
        "affine_validation_status": "validated",
        "normalized_direct_transform": [[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]],
        "normalized_origin_shift": [0.0, 0.0, 0.0],
        "normalized_centering_vectors": [[0.5, 0.5, 0.0]],
    }
    cert.update(over)
    return cert


def _valid_table(*, spinful=False, centered=False, **over):
    setting = {
        "hall_number": 300 if centered else 430,
        "hall_symbol": "C 2y" if centered else "P 3",
        "centering_type": "C" if centered else "P",
        "space_group_symbol": "C2" if centered else "P3",
    }
    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C2" if centered else "P3",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:A", "KM:A", "KM:B"],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0, 1]},
            {"label": "EBR_B", "vector": [1, 1, 0]},
        ],
        "provenance": {
            "data_source": "irreptables",
            "package": "irreptables",
            "package_version": "0.7.1",
            "space_group_number": 5 if centered else 143,
            "spinful": spinful,
            "valleyscope_reduction": "sampled_hsp_valley_preserving",
            "setting_identity": setting,
        },
    }
    if over:
        table = copy.deepcopy(table)
        for key, value in over.items():
            table[key] = value
    return table


def _valid_bundle(*, spinor=False, centered=False, cert=None, **over):
    bundle = {
        "bundle_id": "bundle_ebr_instance_001",
        "valley": "K_valley",
        "subspace_group_candidate": "C2" if centered else "P3",
        "subspace_sg_number": 5 if centered else 143,
        "subspace_space_group": {
            "status": "resolved",
            "candidate_space_group_number": 5 if centered else 143,
            "candidate_space_group_symbol": "C2" if centered else "P3",
        },
        "ready_for_external_solver": True,
        "ready_for_reduced_table_validation": True,
        "expected_hsps": ["GammaM", "KM"],
        "irreps_by_kpoint": {"GammaM": ["A", "A"], "KM": ["A", "B"]},
        "spinor": spinor,
        "certificate_identity": cert if cert is not None else (
            _centered_cert() if centered else _valid_cert()
        ),
    }
    bundle.update(over)
    return bundle


def _promote(bundle, table):
    return promote_bundle_for_solve(bundle=bundle, table=table)


def _codes(result):
    return {b["code"] for b in result["blocker_reasons"]}


# ---------------------------------------------------------------------------
# 1-3. Positive: reviewed primitive/spinful/centered certificates pass
# ---------------------------------------------------------------------------

def test_reviewed_primitive_spinless_passes():
    r = _promote(_valid_bundle(), _valid_table())
    assert r["promoted"] is True
    assert r["canonical_state"] == "validated_basis"
    assert r["blocker_reasons"] == []
    assert all(v == "passed" for v in r["validation_report"].values())


def test_reviewed_primitive_spinful_passes():
    r = _promote(_valid_bundle(spinor=True),
                 _valid_table(spinful=True))
    assert r["promoted"] is True
    assert r["validation_report"]["spin_convention_check"] == "passed"


def test_reviewed_centered_valid_affine_passes():
    r = _promote(_valid_bundle(centered=True),
                 _valid_table(centered=True))
    assert r["promoted"] is True
    assert r["validation_report"]["affine_setting_check"] == "passed"
    assert r["validation_report"]["setting_identity_check"] == "passed"


# ---------------------------------------------------------------------------
# 4-5. Certificate presence / resolution
# ---------------------------------------------------------------------------

def test_missing_certificate_blocks():
    b = _valid_bundle()
    del b["certificate_identity"]
    r = _promote(b, _valid_table())
    assert r["promoted"] is False
    assert "certificate_missing" in _codes(r)


def test_unresolved_certificate_blocks():
    cert = _valid_cert(
        any_unresolved=True,
        validation_status="unresolved",
        certificate_validation_statuses=["unresolved"],
    )
    r = _promote(_valid_bundle(cert=cert), _valid_table())
    assert r["promoted"] is False
    assert "certificate_unresolved" in _codes(r)


def test_rejected_certificate_blocks():
    cert = _valid_cert(
        certificate_validation_statuses=["rejected"],
        validation_status="rejected",
        any_unresolved=True,
    )
    r = _promote(_valid_bundle(cert=cert), _valid_table())
    assert r["promoted"] is False
    assert "certificate_rejected" in _codes(r)


def test_certificate_not_validated_alone_blocks():
    # Fully resolved (any_unresolved=False) but not validated → still blocks.
    cert = _valid_cert(
        any_unresolved=False,
        validation_status="not_evaluated",
        certificate_validation_statuses=["not_evaluated"],
    )
    r = _promote(_valid_bundle(cert=cert), _valid_table())
    assert r["promoted"] is False
    assert "certificate_not_validated" in _codes(r)


def test_ambiguous_setting_identity_blocks():
    cert = _valid_cert(distinct_setting_identities=2)
    r = _promote(_valid_bundle(cert=cert), _valid_table())
    assert r["promoted"] is False
    assert "certificate_ambiguous_setting" in _codes(r)


# ---------------------------------------------------------------------------
# 6-8. Centering / affine evidence
# ---------------------------------------------------------------------------

def test_missing_centering_blocks():
    cert = _valid_cert(centering_type="", centering_types=[])
    r = _promote(_valid_bundle(cert=cert), _valid_table())
    assert r["promoted"] is False
    assert "centering_missing" in _codes(r)


def test_centered_missing_transform_blocks():
    cert = _centered_cert()
    del cert["normalized_direct_transform"]
    r = _promote(_valid_bundle(centered=True, cert=cert),
                 _valid_table(centered=True))
    assert r["promoted"] is False
    assert "centered_affine_evidence_invalid" in _codes(r)


def test_failed_affine_validation_blocks():
    cert = _centered_cert(affine_validation_status="failed")
    r = _promote(_valid_bundle(centered=True, cert=cert),
                 _valid_table(centered=True))
    assert r["promoted"] is False
    assert "centered_affine_evidence_invalid" in _codes(r)


def test_failed_operation_mapping_blocks():
    cert = _centered_cert(operation_mapping_status="failed")
    r = _promote(_valid_bundle(centered=True, cert=cert),
                 _valid_table(centered=True))
    assert r["promoted"] is False
    assert "centered_affine_evidence_invalid" in _codes(r)


def test_primitive_relation_not_declared_blocks():
    cert = _valid_cert(primitive_conventional_relation="")
    r = _promote(_valid_bundle(cert=cert), _valid_table())
    assert r["promoted"] is False
    assert "primitive_relation_not_declared" in _codes(r)


# ---------------------------------------------------------------------------
# 9-11. SG / Hall / spin identity
# ---------------------------------------------------------------------------

def test_sg_symbol_mismatch_blocks():
    r = _promote(_valid_bundle(subspace_group_candidate="P4"), _valid_table())
    assert r["promoted"] is False
    assert "sg_symbol_mismatch" in _codes(r)


def test_sg_number_mismatch_blocks():
    r = _promote(_valid_bundle(subspace_sg_number=75), _valid_table())
    assert r["promoted"] is False
    assert "sg_number_mismatch" in _codes(r)


def test_hall_setting_mismatch_blocks():
    cert = _valid_cert(hall_number=431, hall_symbol="P 3 alt")
    r = _promote(_valid_bundle(cert=cert), _valid_table())
    assert r["promoted"] is False
    assert "setting_identity_mismatch" in _codes(r)


def test_spin_mismatch_blocks():
    r = _promote(_valid_bundle(spinor=True), _valid_table(spinful=False))
    assert r["promoted"] is False
    assert "spin_convention_mismatch" in _codes(r)


def test_spin_convention_missing_blocks():
    b = _valid_bundle()
    del b["spinor"]  # no spinor/spinful anywhere and no source records
    r = _promote(b, _valid_table())
    assert r["promoted"] is False
    assert "spin_convention_missing" in _codes(r)


def test_none_centering_blocks_as_missing():
    cert = _valid_cert(centering_type=None, centering_types=[])
    r = _promote(_valid_bundle(cert=cert), _valid_table())
    assert r["promoted"] is False
    assert "centering_missing" in _codes(r)


def test_sg_number_missing_blocks():
    b = _valid_bundle()
    del b["subspace_sg_number"]
    r = _promote(b, _valid_table())
    assert r["promoted"] is False
    assert "sg_number_missing" in _codes(r)


# ---------------------------------------------------------------------------
# 12. Table provenance fail-closed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate,code", [
    ({"data_source": "invented"}, "table_data_source_invalid"),
    ({"package": "not-irreptables"}, "table_package_invalid"),
    ({"package_version": ""}, "table_package_invalid"),
    ({"valleyscope_reduction": "raw_3d"}, "table_reduction_provenance_invalid"),
    ({"spinful": "yes"}, "table_spinful_missing"),
    ({"space_group_number": 0}, "table_sg_number_missing"),
])
def test_table_provenance_fields_block(mutate, code):
    table = _valid_table()
    table["provenance"].update(mutate)
    r = _promote(_valid_bundle(), table)
    assert r["promoted"] is False
    assert code in _codes(r)


def test_table_setting_identity_missing_blocks():
    table = _valid_table()
    del table["provenance"]["setting_identity"]
    r = _promote(_valid_bundle(), table)
    assert r["promoted"] is False
    assert "table_setting_identity_missing" in _codes(r)


# ---------------------------------------------------------------------------
# 13-14. HSP / irrep basis
# ---------------------------------------------------------------------------

def test_hsp_mismatch_blocks():
    b = _valid_bundle(expected_hsps=["GammaM"],
                      irreps_by_kpoint={"GammaM": ["A", "A"]})
    r = _promote(b, _valid_table())
    assert r["promoted"] is False
    assert "hsp_basis_mismatch" in _codes(r)


def test_unresolved_irrep_key_blocks():
    b = _valid_bundle(
        irreps_by_kpoint={"GammaM": ["A", "A"], "KM": ["A", "ZZ_unknown"]})
    r = _promote(b, _valid_table())
    assert r["promoted"] is False
    assert "irrep_key_unresolved" in _codes(r)


# ---------------------------------------------------------------------------
# 15-17. Production mapping path (no bypass)
# ---------------------------------------------------------------------------

def test_premarked_solver_ready_mismatch_revalidated_and_blocked():
    # Pre-marked ready_for_external_solver=True must NOT bypass validation.
    cert = _valid_cert(
        any_unresolved=True,
        validation_status="not_evaluated",
        certificate_validation_statuses=["not_evaluated"],
        centering_type="",
        centering_types=[],
    )
    b = _valid_bundle(cert=cert)
    r = build_reduced_ebr_mapping(
        ebr_export_bundle={"bundles": [b]}, table=_valid_table())
    assert r["status"] != "solved_exact"
    assert not r["solutions"]
    assert len(r["excluded_bundles"]) == 1
    excl = r["excluded_bundles"][0]
    assert "validation blocked" in excl["reason"]
    assert {bl["code"] for bl in excl["blocker_reasons"]} & {
        "certificate_unresolved", "centering_missing"}


def test_validation_report_survives_into_solution_output():
    r = build_reduced_ebr_mapping(
        ebr_export_bundle={"bundles": [_valid_bundle()]}, table=_valid_table())
    assert r["status"] == "solved_exact"
    sol = r["solutions"][0]
    assert sol["validation_report"]["certificate_check"] == "passed"
    assert sol["table_provenance"]["data_source"] == "irreptables"
    assert sol["table_provenance"]["setting_identity"]["hall_number"] == 430
    assert sol["certificate_identity"]["validation_status"] == "validated"


def test_arbitrary_minimal_table_rejected_by_production_mapping():
    # A minimal hand-written table with no trusted provenance must never
    # promote a bundle for solving.
    minimal_table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:A", "KM:A", "KM:B"],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0, 1]},
            {"label": "EBR_B", "vector": [1, 1, 0]},
        ],
    }
    r = build_reduced_ebr_mapping(
        ebr_export_bundle={"bundles": [_valid_bundle()]}, table=minimal_table)
    assert r["status"] != "solved_exact"
    assert not r["solutions"]
    assert len(r["excluded_bundles"]) == 1
    codes = {bl["code"] for bl in r["excluded_bundles"][0]["blocker_reasons"]}
    assert "table_setting_identity_missing" in codes
