"""Auto-canonical reduced EBR table builder tests — contract coverage."""

import json, pytest, yaml
from pathlib import Path

from valleyscope.io.config import load_config
from valleyscope.workflows.analyze_hsp import _build_auto_canonical_mapping, analyze_hsp
from valleyscope.analysis.irreptables_runtime_table_builder import (
    build_auto_canonical_reduced_ebr_table,
    _extract_kvector_token,
    _validate_expected_hsps,
)
from valleyscope.analysis.reduced_ebr_mapping import (
    build_reduced_ebr_mapping, load_reduced_ebr_table,
)
from valleyscope.analysis.reduced_ebr_solver import classify_bundle
from tests.helpers_io_workflow import write_fixture, write_config
from tests.reduced_ebr_promo_helpers import (
    apply_resolver_certificate, resolver_certificate_identity,
)

# ---------------------------------------------------------------------------
# Shared compact factories
# ---------------------------------------------------------------------------

_P4_GM = ["-GM5", "-GM6", "-GM7", "-GM8"]
_P4_X  = ["-X3", "-X4"]

def _p4_loader(gm=None, x=None):
    gm = gm or _P4_GM; x = x or _P4_X; labels = list(gm) + list(x)
    h = len(gm) // 2 if len(gm) >= 2 else 1
    def _f(sg, spin):
        assert sg == 75 and spin is True
        return {"basis": {"irrep_labels": labels}, "ebrs": [
            {"ebr_name": "EBR_GM_A", "vector": [1 if l in gm[:h] else 0 for l in labels]},
            {"ebr_name": "EBR_GM_B", "vector": [1 if l in gm[h:] else 0 for l in labels]},
            {"ebr_name": "EBR_X",   "vector": [1 if l in x else 0 for l in labels]},
        ]}
    return _f

def _spin_records(irreps, spinful=True):
    return {kp: [{"matched_irrep": (labs[0] if labs else ""),
                  "irrep_multiplicity": 1,
                  "irrep_source_provenance": {"source_table_spinor": spinful}}]
            for kp, labs in irreps.items()}

def _mk_bundle(bid, sg_num, symbol, hsps, irreps, ready=True):
    # Real producer-built primitive certificate for a spglib-unique SG; None
    # for unresolved/invalid SGs so the validator blocks.
    cert = resolver_certificate_identity(sg_num, symbol)
    return {"bundle_id": bid, "valley": "K_valley",
            "subspace_group_candidate": symbol,
            "subspace_sg_number": sg_num,
            "subspace_space_group": {"status": "resolved",
                                     "candidate_space_group_number": sg_num,
                                     "candidate_space_group_symbol": symbol},
            "ready_for_external_solver": ready,
            "ready_for_reduced_table_validation": ready,
            "expected_hsps": hsps, "irreps_by_kpoint": irreps,
            "irrep_records_by_kpoint": _spin_records(irreps, spinful=True),
            "certificate_identity": cert if cert is not None else {}}

def _bm(*bundles): return _build_auto_canonical_mapping(
    ebr_export_bundle={"bundles": list(bundles)}, spinor_wf=True)

def _auto_table(sg, hsps, irreps, loader=None):
    return build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=sg, spinor=True, bundle_irreps_by_kpoint=irreps,
        expected_hsps=hsps, subspace_group_candidate=symbol_for(sg),
        source_loader=loader or _p4_loader())

def symbol_for(sg): return {75: "P4", 143: "P3"}.get(sg, f"SG{sg}")

# ---------------------------------------------------------------------------
# Workflow-level: _build_auto_canonical_mapping
# ---------------------------------------------------------------------------

def test_none_or_empty_returns_none():
    assert _build_auto_canonical_mapping(ebr_export_bundle=None, spinor_wf=True) is None
    assert _build_auto_canonical_mapping(ebr_export_bundle={"bundles": []}, spinor_wf=True) is None

def test_malformed_ready_excluded():
    r = _bm({** _mk_bundle("b", 75, "P4", ["GM"], {"GM": ["-GM5"]}),
             "subspace_space_group": None})
    assert r and len(r["excluded_bundles"]) == 1 and "subspace_space_group" in r["excluded_bundles"][0]["reason"]

@pytest.mark.parametrize("label,desc", [
    ("solved_only", "solved_exact"),
    ("solved_plus_blocked", "partial"),
    ("non_ready_excluded", "solved_exact"),
])
def test_per_bundle_aggregation(label, desc):
    if label == "solved_only":
        r = _bm(_mk_bundle("b1", 75, "P4", ["GammaM"], {"GammaM": ["-GM5"]}))
        assert r["mapping_status"] == "solved_exact"
    elif label == "solved_plus_blocked":
        r = _bm(_mk_bundle("b1", 75, "P4", ["GammaM"], {"GammaM": ["-GM5"]}),
                {**_mk_bundle("b2", 143, "P3", ["GammaM"], {"GammaM": ["-GM4"]}),
                 "subspace_space_group": None})
        assert r["mapping_status"] == "partial"
    elif label == "non_ready_excluded":
        r = _bm(_mk_bundle("b1", 75, "P4", ["GammaM"], {"GammaM": ["-GM5"]}),
                {**_mk_bundle("b2", 75, "P4", ["GammaM"], {"GammaM": ["-GM4"]}),
                 "ready_for_external_solver": False,
                 "ready_for_reduced_table_validation": False})
        assert r["mapping_status"] == "solved_exact" and len(r["excluded_bundles"]) == 1

def test_no_ready_bundles_not_evaluated():
    r = _bm({"_id": "b", "ready_for_external_solver": False})
    assert r and r["status"] == "not_evaluated" and r["reduced_ebr_input"]["ready_bundle_count"] == 0

def test_no_buildable_blocked():
    r = _bm({** _mk_bundle("b", 75, "P4", ["GammaM"], {"GammaM": ["-GM5"]}),
             "subspace_space_group": None})
    assert r and r["status"] == "blocked"

def test_source_failure_blocked_not_physical():
    r = _bm(_mk_bundle("b", 9999, "XX", ["GammaM"], {"GammaM": ["-GM4"]}))
    assert r and r["status"] == "blocked"

# ---------------------------------------------------------------------------
# Auto table builder (unit + integration)
# ---------------------------------------------------------------------------

def test_auto_table_p4_gm():
    t = _auto_table(75, ["GammaM"], {"GammaM": ["-GM5", "-GM6"]})
    assert t["subspace_group_candidate"] == "P4" and t["expected_hsps"] == ["GammaM"]
    assert len(t["irreps"]) == len(_P4_GM) and len(t["ebrs"]) == 2
    assert t["provenance"]["auto_canonical"] is True
    assert t["provenance"]["sampled_bilbao_hsps"] == ["GM"]

def test_auto_table_p4_gm_xm():
    t = _auto_table(75, ["GammaM", "XM"], {"GammaM": ["-GM5", "-GM6"], "XM": ["-X3"]})
    gm = [k for k in t["irreps"] if k.startswith("GammaM:")]; xm = [k for k in t["irreps"] if k.startswith("XM:")]
    assert len(gm) == len(_P4_GM) and len(xm) == len(_P4_X) and t["expected_hsps"] == ["GammaM", "XM"]

def test_auto_table_e2e_solve(tmp_path):
    t = _auto_table(75, ["GammaM"], {"GammaM": ["-GM5", "-GM6"]})
    p = tmp_path / "t.json"; p.write_text(json.dumps(t))
    eb = {"bundles": [_mk_bundle("b", 75, "P4", ["GammaM"], {"GammaM": ["-GM5", "-GM6"]})]}
    r = build_reduced_ebr_mapping(ebr_export_bundle=eb, table=load_reduced_ebr_table(p))
    assert r["mapping_status"] == "solved_exact" and r["solutions"][0]["classification"] == "atomic-compatible-candidate"

def test_auto_table_hsp_mismatch(tmp_path):
    t = _auto_table(75, ["GammaM"], {"GammaM": ["-GM5"]})
    p = tmp_path / "t.json"; p.write_text(json.dumps(t))
    eb = {"bundles": [_mk_bundle("b", 75, "P4", ["GammaM", "XM"], {"GammaM": ["-GM5"], "XM": ["-X3"]})]}
    r = build_reduced_ebr_mapping(ebr_export_bundle=eb, table=load_reduced_ebr_table(p))
    assert len(r["excluded_bundles"]) == 1 and "expected_hsps mismatch" in r["excluded_bundles"][0]["reason"]

def test_subspace_sg_not_parent():
    called = []
    def _ld(sg, spin): called.append(sg); return {"basis": {"irrep_labels": ["-GM4","-GM5","-GM6","-K4","-K5","-K6"]}, "ebrs": [{"ebr_name": "E", "vector": [1,0,0,0,0,0]}]}
    build_auto_canonical_reduced_ebr_table(subspace_sg_number=143, spinor=True, bundle_irreps_by_kpoint={"GammaM": ["-GM4"], "KM": ["-K5"]}, expected_hsps=["GammaM", "KM"], subspace_group_candidate="P3", source_loader=_ld)
    assert called == [143]

@pytest.mark.parametrize("spin,labels,sg", [(True, ["-GM4"], 143), (False, ["GM1"], 143)])
def test_spinor_flag(spin, labels, sg):
    called = []
    def _ld(s, sp):
        called.append(sp)
        return {"basis": {"irrep_labels": labels}, "ebrs": [{"ebr_name": "E", "vector": [1]}]}
    build_auto_canonical_reduced_ebr_table(subspace_sg_number=sg, spinor=spin, bundle_irreps_by_kpoint={"GammaM": labels}, expected_hsps=["GammaM"], subspace_group_candidate=("P3" if sg==143 else "P4"), source_loader=_ld)
    assert called == [spin]

def test_ordered_unique_irrep_keys():
    t = _auto_table(75, ["GammaM"], {"GammaM": ["-GM5"]})
    assert t["irreps"] == sorted(t["irreps"]) and len(t["irreps"]) == len(set(t["irreps"]))

def test_hsp_mapping_bijective():
    with pytest.raises(ValueError, match="maps to multiple Bilbao HSPs"):
        _auto_table(75, ["GammaM"], {"GammaM": ["-GM5", "-X3"]})

def test_hsp_key_mismatch_blocks():
    with pytest.raises(ValueError, match="do not match expected_hsps"):
        _auto_table(75, ["XM"], {"GammaM": ["-GM5"]})

def test_expected_hsps_and_hsp_validation():
    for bad in ([], ["A","A"], ["","B"], "not_a_list"):
        with pytest.raises(ValueError): _validate_expected_hsps(bad)
    with pytest.raises(ValueError, match="must be a non-empty list"):
        _auto_table(75, ["GammaM"], {"GammaM": []})

# ---------------------------------------------------------------------------
# Strict label resolution
# ---------------------------------------------------------------------------

def test_unknown_bundle_label_blocks():
    with pytest.raises(ValueError, match="not found in irreptables irrep table"):
        _auto_table(75, ["GammaM"], {"GammaM": ["-GM5", "-ZZ99"]})

def test_no_numeric_index_blocks():
    with pytest.raises(ValueError, match="invalid k-vector token"):
        _auto_table(75, ["GammaM"], {"GammaM": ["-GM5"]},
                    loader=lambda sg,spin: {"basis": {"irrep_labels": ["-GM5","-UNKNOWN"]}, "ebrs": [{"ebr_name":"E","vector":[1,0]}]})

def test_unknown_irreptable_label_blocks():
    with pytest.raises(ValueError, match="not found in irreptables irrep table"):
        _auto_table(75, ["GammaM"], {"GammaM": ["-ZZ99"]},
                    loader=lambda sg,spin: {"basis": {"irrep_labels": ["-ZZ99"]}, "ebrs": [{"ebr_name":"E","vector":[1]}]})

def test_conflicting_hsp_raises():
    with pytest.raises(ValueError, match="conflicting HSP mapping"):
        _auto_table(75, ["GammaM", "XM"], {"GammaM": ["-GM5"], "XM": ["-GM5"]})

def test_dropped_rows_provenance():
    t = _auto_table(75, ["GammaM"], {"GammaM": ["-GM5"]})
    p = t["provenance"]; assert "GM" in p["sampled_bilbao_hsps"]
    assert any("token=X" in d for d in p.get("dropped_source_rows", []))

# ---------------------------------------------------------------------------
# K-vector token parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("-GM4","GM"),("-GM5","GM"),("-KA4","KA"),("-HA4","HA"),
    ("WA1","WA"),("GM1","GM"),("-K5","K"),("-UNKNOWN",None),("ABC",None),("999",None),("-",None),("",None),
])
def test_kvector_parser(label, expected):
    if expected is not None: assert _extract_kvector_token(label) == expected
    else:
        with pytest.raises(ValueError): _extract_kvector_token(label)

def test_sg143_ka_ha_dropped():
    """KA/HA rows from real SG143 EBR data are dropped (not sampled)."""
    from irreptables.ebrs import load_ebr_data
    t = build_auto_canonical_reduced_ebr_table(
        subspace_sg_number=143, spinor=True,
        bundle_irreps_by_kpoint={"GammaM": ["-GM4"], "KM": ["-K5"]},
        expected_hsps=["GammaM", "KM"], subspace_group_candidate="P3",
        source_loader=lambda sg,spin: load_ebr_data(sg, spin))
    dropped = [d for d in t["provenance"].get("dropped_source_rows", []) if "KA" in d or "HA" in d]
    assert len(dropped) > 0 and all("token=" in d for d in dropped)

# ---------------------------------------------------------------------------
# Solver uniqueness and truncation
# ---------------------------------------------------------------------------

def test_unique_exact():
    r = classify_bundle([1], [[1], [2]], ["A", "B"], 6)
    assert r["decomposition_uniqueness"] == "unique" and r["classification"] == "atomic-compatible-candidate"

def test_non_unique_witnesses():
    r = classify_bundle([1], [[1], [1]], ["A", "B"], 6)
    assert r["decomposition_uniqueness"] == "non_unique" and len(r.get("decomposition_witnesses", [])) >= 2

@pytest.mark.parametrize("target,vecs,cap,exp_uniq,exp_class", [
    ([100], [[1]], 0, None, "indeterminate_truncated"),
    ([2], [[1], [1]], 1, "unknown_truncated", "atomic-compatible-candidate"),
    ([10], [[1], [2]], 5, "non_unique", "atomic-compatible-candidate"),
])
def test_truncated_states(target, vecs, cap, exp_uniq, exp_class):
    r = classify_bundle(target, vecs, [f"E{i}" for i in range(len(vecs))], cap)
    assert r["classification"] == exp_class
    if exp_uniq is not None:
        assert r["decomposition_uniqueness"] == exp_uniq
    else:
        assert "decomposition_uniqueness" not in r

# ---------------------------------------------------------------------------
# Irrep-key regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("irreps,valid", [
    (["GammaM:-GM5","GammaM:-GM6"], True), (["KM:-K5","KM:-K6"], True),
    (["GammaM:GM1"], True), (["GammaM:-GM5:op1"], True),
    (["GammaM:1GM"], False), (["GammaM:+GM"], False), (["GammaM:/GM"], False),
    (["GammaM:-1"], False),
])
def test_irrep_key_regex(irreps, valid):
    import tempfile, os
    t = {"schema_version":"1.0.0","subspace_group_candidate":"P4","expected_hsps":["GammaM"],"irreps":irreps,"ebrs":[{"label":"E","vector":[1]*len(irreps)}]}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        json.dump(t, f); f.close()
        if valid: load_reduced_ebr_table(f.name)
        else:
            with pytest.raises(ValueError, match="invalid irrep key format"): load_reduced_ebr_table(f.name)
    finally: os.unlink(f.name)

# ---------------------------------------------------------------------------
# Disabled default / explicit table & spec regression / no-Cn
# ---------------------------------------------------------------------------

def test_disabled_no_artifacts(tmp_path):
    h5 = tmp_path / "w.h5"; out = tmp_path / "out"
    write_fixture(h5); cfg = tmp_path / "c.yaml"; write_config(cfg, h5, out)
    outputs = analyze_hsp(cfg)
    assert not (out / "valley_reduced_ebr_mapping.json").exists()

def test_explicit_table_and_spec_regression(tmp_path):
    # table_file
    td = {"schema_version":"1.0.0","subspace_group_candidate":"P4","expected_hsps":["GammaM"],"irreps":["GammaM:-GM5","GammaM:-GM6"],"ebrs":[{"label":"E","vector":[1,0]}],"provenance":{"space_group_number":75,"spinful":True}}
    tp = tmp_path / "t.json"; tp.write_text(json.dumps(td))
    eb = {"bundles": [_mk_bundle("b",75,"P4",["GammaM"],{"GammaM":["-GM5"]})]}
    tt = load_reduced_ebr_table(tp); apply_resolver_certificate(eb, tt)
    r = build_reduced_ebr_mapping(ebr_export_bundle=eb, table=tt, reduced_ebr_input={"source":"table_file"})
    assert r["mapping_status"] == "solved_exact"
    # spec_file — use a loader that returns exactly the labels declared in the spec
    from valleyscope.analysis.irreptables_runtime_table_builder import build_reduced_table_from_spec_file
    sp = tmp_path / "s.json"; sp.write_text(json.dumps({"schema_version":"1.1.0","data_source":"irreptables","space_group_number":75,"spinful":True,"source_hsp_by_irrep":{"-GM5":"GammaM","-GM6":"GammaM"},"valleyscope_irrep_multiplicity_by_source_irrep":{"-GM5":{"GammaM:-GM5":1},"-GM6":{"GammaM:-GM6":1}},"expected_hsps":["GammaM"],"allowed_irrep_keys":["GammaM:-GM5","GammaM:-GM6"],"subspace_group_candidate":"P4"}))
    def _spec_ld(sg,spin): return {"basis":{"irrep_labels":["-GM5","-GM6"]},"ebrs":[{"ebr_name":"E","vector":[1,0]}]}
    t2 = build_reduced_table_from_spec_file(str(sp), source_loader=_spec_ld)
    eb2 = {"bundles": [_mk_bundle("b",75,"P4",["GammaM"],{"GammaM":["-GM5"]})]}; apply_resolver_certificate(eb2, t2)
    r2 = build_reduced_ebr_mapping(ebr_export_bundle=eb2, table=t2, reduced_ebr_input={"source":"spec_file"})
    assert r2["mapping_status"] == "solved_exact"

def test_no_cn_like_labels():
    t = _auto_table(75, ["GammaM"], {"GammaM": ["-GM5"]}, loader=_p4_loader(gm=["-GM5","-GM6"], x=[]))
    raw = json.dumps(t)
    for cn in ("C2_like","C3_like","C4_like"): assert cn not in raw
    assert "P4" in raw
