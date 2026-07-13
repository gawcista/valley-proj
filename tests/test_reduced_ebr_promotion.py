"""Independent convention trust chain — fail-closed promotion validator.

Fixtures use the EXACT producer vocabulary emitted by
``StandardSettingCertificate`` / ``_certificate_identity`` and
crystallographically valid SG/Hall values (spglib).  The table standard
setting is derived independently by the validator (spglib); it is never
supplied by these fixtures.  Direct ``certificate_identity`` dicts appear only
in isolated validator-unit tests; the production integration test obtains the
identity through the real producer + serializer.
"""

import copy

import pytest

from valleyscope.analysis.reduced_ebr_mapping import (
    promote_bundle_for_solve,
    build_reduced_ebr_mapping,
)
from tests.reduced_ebr_promo_helpers import (
    real_certificate_identity,
    real_promotion,
)

# Crystallographically valid settings (spglib canonical Hall):
#   SG 143 P3  -> Hall 430 "P 3" (primitive)
#   SG 79  I4  -> Hall 353 "I 4" (I-centered)


def _primitive_cert(**over):
    cert = {
        "hall_numbers": [430], "hall_symbols": ["P 3"], "centering_types": ["P"],
        "certificate_validation_statuses": ["validated"],
        "any_unresolved": False, "distinct_setting_identities": 1,
        "sg_number": 143, "sg_symbol": "P3",
        "hall_number": 430, "hall_symbol": "P 3", "centering_type": "P",
        "primitive_conventional_relation": "direct_coordinate_match",
        "transform_provenance": "spglib.per_valley_standard_matches",
        "validation_status": "validated",
        "operation_mapping_status": "not_attempted",
        "affine_validation_status": "passed",
    }
    cert.update(over)
    return cert


def _centered_cert(**over):
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
        "certificate_identity": cert if cert is not None else _primitive_cert(),
    }
    bundle.update(over)
    return bundle


def _promote(bundle, table):
    return promote_bundle_for_solve(bundle=bundle, table=table)


def _codes(result):
    return {b["code"] for b in result["blocker_reasons"]}


# ---------------------------------------------------------------------------
# Positive: producer vocabulary passes (isolated validator units)
# ---------------------------------------------------------------------------

def test_reviewed_primitive_spinless_passes():
    r = _promote(_bundle(), _table())
    assert r["promoted"] is True
    assert r["canonical_state"] == "validated_basis"
    assert r["blocker_reasons"] == []


def test_reviewed_primitive_spinful_passes():
    r = _promote(_bundle(spinful=True), _table(spinful=True))
    assert r["promoted"] is True
    assert r["validation_report"]["spin_convention_check"] == "passed"


def test_reviewed_centered_valid_affine_passes():
    r = _promote(_bundle(sg_number=79, symbol="I4", cert=_centered_cert()),
                 _table(sg_number=79, symbol="I4"))
    assert r["promoted"] is True
    assert r["validation_report"]["affine_setting_check"] == "passed"
    assert r["validation_report"]["hall_setting_check"] == "passed"


# ---------------------------------------------------------------------------
# Real producer -> certificate identity -> promotion integration
# ---------------------------------------------------------------------------

def test_real_producer_certificate_identity_promotes():
    cert_identity = real_certificate_identity(143, "P3")
    assert cert_identity is not None
    # Not hand-written: exact producer vocabulary.
    assert cert_identity["primitive_conventional_relation"] == \
        "direct_coordinate_match"
    assert cert_identity["validation_status"] == "validated"
    r = _promote(_bundle(cert=cert_identity), _table())
    assert r["promoted"] is True
    assert r["blocker_reasons"] == []


def test_real_producer_full_mapping_solves():
    export = {"bundles": [_bundle(irreps_by_kpoint={
        "GammaM": ["A", "A"], "KM": ["A", "B"]})]}
    export["bundles"][0]["certificate_identity"] = \
        real_certificate_identity(143, "P3")
    r = build_reduced_ebr_mapping(ebr_export_bundle=export, table=_table())
    assert r["status"] == "solved_exact"
    sol = r["solutions"][0]
    assert sol["validation_report"]["certificate_check"] == "passed"
    assert sol["table_provenance"]["independent_setting_identity"][
        "hall_number"] == 430


def test_full_chain_candidates_to_solved_without_manual_readiness():
    """Producer certificate -> instances -> export bundle -> promotion -> solve.

    Readiness flags and the certificate identity are produced by the real
    workflow builders; nothing is set by hand except the input candidate rows.
    """
    from valleyscope.analysis.standard_setting_kmap import (
        build_standard_setting_certificate,
    )
    from valleyscope.analysis.ebr_problem_instances import (
        build_ebr_problem_instances,
    )
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle

    cert = build_standard_setting_certificate(
        standard_match={"number": 143, "international_short": "P3",
                        "hall_number": 430, "hall_symbol": "P 3",
                        "operation_ids": [0, 1, 2]},
        validation_status="validated",
        parent_basis_operation_ids=[0, 1, 2],
        parent_k_frac=[0.0, 0.0, 0.0], resolved_hsp_label="GM1",
    )
    cert.operation_mapping_status = "not_attempted"
    cert.centering_status = "primitive_direct_match"
    cert.primitive_conventional_relation = "direct_coordinate_match"
    cert.translation_validation_status = "passed"
    cert_dict = cert.to_dict()

    def _prov():
        return {"standard_setting_hsp_mapping": {
                    "standard_setting_certificate": dict(cert_dict)},
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
# Vocabulary: fictitious vocabulary is rejected
# ---------------------------------------------------------------------------

def test_fictitious_primitive_relation_blocks():
    # The previous fictitious vocabulary "identity" must not pass.
    cert = _primitive_cert(primitive_conventional_relation="identity")
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "primitive_relation_not_declared" in _codes(r)


def test_fictitious_centered_op_status_blocks():
    cert = _centered_cert(operation_mapping_status="validated")  # not real vocab
    r = _promote(_bundle(sg_number=79, symbol="I4", cert=cert),
                 _table(sg_number=79, symbol="I4"))
    assert r["promoted"] is False
    assert "centered_affine_evidence_invalid" in _codes(r)


# ---------------------------------------------------------------------------
# Certificate presence / status
# ---------------------------------------------------------------------------

def test_missing_certificate_blocks():
    b = _bundle()
    del b["certificate_identity"]
    r = _promote(b, _table())
    assert r["promoted"] is False
    assert "certificate_missing" in _codes(r)


def test_unresolved_certificate_blocks():
    cert = _primitive_cert(any_unresolved=True, validation_status="unresolved",
                           certificate_validation_statuses=["unresolved"])
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_unresolved" in _codes(r)


def test_rejected_certificate_blocks():
    cert = _primitive_cert(certificate_validation_statuses=["rejected"],
                           validation_status="rejected", any_unresolved=True)
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_rejected" in _codes(r)


def test_not_validated_alone_blocks():
    cert = _primitive_cert(any_unresolved=False, validation_status="not_evaluated",
                           certificate_validation_statuses=["not_evaluated"])
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_not_validated" in _codes(r)


def test_ambiguous_setting_blocks():
    cert = _primitive_cert(distinct_setting_identities=2)
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_ambiguous_setting" in _codes(r)


# ---------------------------------------------------------------------------
# Internal consistency (Codex findings 1, 2 + singular/plural)
# ---------------------------------------------------------------------------

def test_certificate_sg_conflict_blocks():
    # Certificate claims SG 75 while bundle/table are SG 143.
    cert = _primitive_cert(sg_number=75, sg_symbol="P4")
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_sg_conflict" in _codes(r)


def test_spin_evidence_conflict_blocks():
    # A per-record source_table_spinor contradicts the others.
    b = _bundle(spinful=False)
    b["irrep_records_by_kpoint"]["KM"][0][
        "irrep_source_provenance"]["source_table_spinor"] = True
    r = _promote(b, _table(spinful=False))
    assert r["promoted"] is False
    assert "spin_evidence_conflict" in _codes(r)


def test_singular_plural_field_conflict_blocks():
    cert = _primitive_cert(hall_number=430, hall_numbers=[431])
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "certificate_field_inconsistent" in _codes(r)


# ---------------------------------------------------------------------------
# Hall / SG / centering crystallographic consistency (spglib)
# ---------------------------------------------------------------------------

def test_sg_143_with_hall_1_rejected():
    # Hall 1 belongs to SG 1, not SG 143.
    cert = _primitive_cert(hall_number=1, hall_numbers=[1], hall_symbol="P 1")
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "hall_sg_inconsistent" in _codes(r) \
        or "setting_identity_mismatch" in _codes(r)


def test_centering_inconsistent_with_hall_blocks():
    # Hall "P 3" implies centering P, certificate claims C.
    cert = _primitive_cert(centering_type="C", centering_types=["C"])
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "centering_hall_inconsistent" in _codes(r) \
        or "setting_identity_mismatch" in _codes(r)


def test_hall_mismatch_with_table_blocks():
    # Certificate Hall for a different (valid) SG than the table.
    cert = _primitive_cert(hall_number=349, hall_numbers=[349], hall_symbol="P 4",
                           sg_number=143)  # 349 belongs to SG 75, not 143
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "hall_sg_inconsistent" in _codes(r) \
        or "setting_identity_mismatch" in _codes(r)


# ---------------------------------------------------------------------------
# Independent table standard setting
# ---------------------------------------------------------------------------

def test_table_setting_unresolved_blocks():
    # SG 5 (monoclinic C2) has multiple spglib Hall settings -> unresolved.
    r = _promote(_bundle(sg_number=5, symbol="C2"),
                 _table(sg_number=5, symbol="C2"))
    assert r["promoted"] is False
    assert "table_standard_setting_unresolved" in _codes(r)


def test_table_provenance_missing_sg_number_blocks():
    table = _table()
    del table["provenance"]["space_group_number"]
    r = _promote(_bundle(), table)
    assert r["promoted"] is False
    assert "table_sg_number_missing" in _codes(r)


@pytest.mark.parametrize("mutate,code", [
    ({"data_source": "invented"}, "table_data_source_invalid"),
    ({"package": "not-irreptables"}, "table_package_invalid"),
    ({"valleyscope_reduction": "raw_3d"}, "table_reduction_provenance_invalid"),
    ({"spinful": "yes"}, "table_spinful_missing"),
])
def test_table_provenance_fields_block(mutate, code):
    table = _table()
    table["provenance"].update(mutate)
    r = _promote(_bundle(), table)
    assert r["promoted"] is False
    assert code in _codes(r)


# ---------------------------------------------------------------------------
# Centering / spin / SG absence
# ---------------------------------------------------------------------------

def test_missing_centering_blocks():
    cert = _primitive_cert(centering_type="", centering_types=[])
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "centering_missing" in _codes(r)


def test_none_centering_blocks_as_missing():
    cert = _primitive_cert(centering_type=None, centering_types=[])
    r = _promote(_bundle(cert=cert), _table())
    assert r["promoted"] is False
    assert "centering_missing" in _codes(r)


def test_spin_missing_blocks():
    b = _bundle()
    b["irrep_records_by_kpoint"] = {}
    r = _promote(b, _table())
    assert r["promoted"] is False
    assert "spin_convention_missing" in _codes(r)


def test_spin_mismatch_blocks():
    r = _promote(_bundle(spinful=True), _table(spinful=False))
    assert r["promoted"] is False
    assert "spin_convention_mismatch" in _codes(r)


def test_sg_symbol_mismatch_blocks():
    r = _promote(_bundle(symbol="P4"), _table())
    assert r["promoted"] is False
    assert "sg_symbol_mismatch" in _codes(r)


def test_sg_number_missing_blocks():
    b = _bundle()
    del b["subspace_sg_number"]
    r = _promote(b, _table())
    assert r["promoted"] is False
    assert "sg_number_missing" in _codes(r)


# ---------------------------------------------------------------------------
# HSP / irrep basis
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Production mapping path (no bypass)
# ---------------------------------------------------------------------------

def test_premarked_ready_revalidated_and_blocked():
    cert = _primitive_cert(any_unresolved=True, validation_status="not_evaluated",
                           certificate_validation_statuses=["not_evaluated"])
    b = _bundle(cert=cert)
    r = build_reduced_ebr_mapping(ebr_export_bundle={"bundles": [b]},
                                  table=_table())
    assert r["status"] != "solved_exact"
    assert not r["solutions"]
    assert len(r["excluded_bundles"]) == 1
    assert "certificate_unresolved" in {
        bl["code"] for bl in r["excluded_bundles"][0]["blocker_reasons"]}


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


def test_no_helper_copies_setting_into_table():
    # real_promotion must not write a setting_identity into table provenance.
    table = _table()
    export = real_promotion({"bundles": [_bundle()]}, table)
    assert export is not None
    assert "setting_identity" not in table["provenance"]
    r = build_reduced_ebr_mapping(ebr_export_bundle=export, table=table)
    assert r["status"] == "solved_exact"
