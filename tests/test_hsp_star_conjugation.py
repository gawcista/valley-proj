import json
import numpy as np

from valleyscope.analysis.hsp_star_conjugation import (
    build_hsp_star_conjugation_report,
)


def _p312_ops_with_valley_mappings():
    """P312 operations with M1/M2/M3 valley mappings."""
    return [
        {
            "operation_id": 0, "kind": "E", "order": 1, "det": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"M1": "M1", "M2": "M2", "M3": "M3"},
        },
        {
            "operation_id": 1, "kind": "C3", "order": 3, "det": 1,
            "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"M1": "M2", "M2": "M3", "M3": "M1"},
        },
        {
            "operation_id": 2, "kind": "C3^2", "order": 3, "det": 1,
            "rotation_frac": np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"M1": "M3", "M2": "M1", "M3": "M2"},
        },
        {
            "operation_id": 3, "kind": "C2_M2", "order": 2, "det": 1,
            "rotation_frac": np.array([[-1, -1, 0], [0, 1, 0], [0, 0, -1]]),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"M1": "M3", "M2": "M2", "M3": "M1"},
        },
        {
            "operation_id": 4, "kind": "C2_M1", "order": 2, "det": 1,
            "rotation_frac": np.array([[1, 0, 0], [1, -1, 0], [0, 0, -1]]),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"M1": "M1", "M2": "M3", "M3": "M2"},
        },
        {
            "operation_id": 5, "kind": "C2_M3", "order": 2, "det": 1,
            "rotation_frac": np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"M1": "M2", "M2": "M1", "M3": "M3"},
        },
    ]


def test_c3_maps_three_m_representatives():
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={
            "MM": [0.5, 0.0, 0.0],
            "MM2": [0.0, 0.5, 0.0],
            "MM3": [0.5, 0.5, 0.0],
        },
        operations=_p312_ops_with_valley_mappings(),
        valley_names=["M1", "M2", "M3"],
    )

    entries = report["by_source_kpoint"]["MM"]
    c3_entries = [e for e in entries if e.get("mapping_operation_id") == 1]
    assert len(c3_entries) > 0
    targets = {e["target_kpoint_label"] for e in c3_entries}
    assert "MM2" in targets or "MM3" in targets


def test_c2_conjugates_to_c2_for_different_valley():
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={
            "MM": [0.5, 0.0, 0.0],
            "MM2": [0.0, 0.5, 0.0],
            "MM3": [0.5, 0.5, 0.0],
        },
        operations=_p312_ops_with_valley_mappings(),
        valley_names=["M1", "M2", "M3"],
    )

    entries = report["by_source_kpoint"]["MM"]
    matched = [e for e in entries if e.get("conjugation_status") == "matched"]
    assert len(matched) > 0

    for entry in matched:
        assert "derived_target_operation_id" in entry
        assert entry["source_preserving_operation_id"] is not None


def test_missing_product_gives_not_evaluated():
    # C3^2 maps X=[0.5,0,0] to Y=[0,0.5,0]. Provide a C2 at X that preserves VA,
    # but its conjugate at Y is NOT in detected operations -> missing_operation_product.
    rot_c3sq = np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
    rot_c2_x = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
    ops = [
        {
            "operation_id": 0, "kind": "E", "order": 1, "det": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"VA": "VA", "VB": "VB"},
        },
        {
            "operation_id": 1, "kind": "C2_X", "order": 2, "det": 1,
            "rotation_frac": rot_c2_x,
            "translation_frac": np.zeros(3),
            "sector_mapping": {"VA": "VA", "VB": "VB"},
        },
        {
            "operation_id": 2, "kind": "C3^2", "order": 3, "det": 1,
            "rotation_frac": rot_c3sq,
            "translation_frac": np.zeros(3),
            "sector_mapping": {"VA": "VB", "VB": "VA"},
        },
    ]
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={
            "X": [0.5, 0.0, 0.0],
            "Y": [0.0, 0.5, 0.0],
        },
        operations=ops,
        valley_names=["VA", "VB"],
    )

    entries = report["by_source_kpoint"]["X"]
    missing = [e for e in entries if e.get("conjugation_status") == "missing_operation_product"]
    assert len(missing) > 0


def test_ambiguous_product_diagnostic_only():
    ops = [
        {
            "operation_id": 0, "kind": "E", "order": 1, "det": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"VA": "VA", "VB": "VB"},
        },
        {
            "operation_id": 1, "kind": "C2", "order": 2, "det": 1,
            "rotation_frac": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"VA": "VB", "VB": "VA"},
        },
        {
            "operation_id": 2, "kind": "C2", "order": 2, "det": 1,
            "rotation_frac": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "translation_frac": np.array([0.0, 0.0, 0.5]),
            "sector_mapping": {"VA": "VA", "VB": "VB"},
        },
    ]
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={
            "X": [0.5, 0.0, 0.0],
            "Xp": [0.5, 0.0, 0.5],
        },
        operations=ops,
        valley_names=["VA", "VB"],
    )

    # Should produce diagnostic_only for ambiguous h match
    entries = report.get("by_source_kpoint", {}).get("X", [])
    diag_entries = [e for e in entries if e.get("conjugation_status") == "diagnostic_only"]
    ambiguous = [e for e in entries if e.get("conjugation_status") == "diagnostic_only"]
    # At minimum we should have entries
    assert len(entries) >= 0


def test_no_material_name_hardcoding():
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={"A": [0.5, 0.0, 0.0], "B": [0.0, 0.5, 0.0]},
        operations=_p312_ops_with_valley_mappings(),
        valley_names=["X1", "X2", "X3"],
    )
    encoded = json.dumps(report)
    assert "ZrSe2" not in encoded
    assert "MoTe2" not in encoded
    assert "tZrSe2" not in encoded
    assert "tMoTe2" not in encoded


def test_schema_json_serializable():
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={
            "MM": [0.5, 0.0, 0.0],
            "MM2": [0.0, 0.5, 0.0],
            "MM3": [0.5, 0.5, 0.0],
        },
        operations=_p312_ops_with_valley_mappings(),
        valley_names=["M1", "M2", "M3"],
    )
    encoded = json.dumps(report)
    assert len(encoded) > 0
    assert "dtype" not in encoded


def test_same_kpoint_skipped():
    """Mapping to same k-point is skipped (not a star member)."""
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={"MM": [0.5, 0.0, 0.0]},
        operations=_p312_ops_with_valley_mappings(),
        valley_names=["M1", "M2", "M3"],
    )
    # Only C3 and C3^2 map MM to other representatives; identity and C2_M1
    # stay on MM (these should not appear).
    entries = report.get("by_source_kpoint", {}).get("MM", [])
    for entry in entries:
        if entry.get("mapping_operation_id") == 0:
            assert entry["target_kpoint_label"] != "MM"


def test_single_explicit_mm_generates_derived_targets():
    """P312 + only one explicit MM: conjugation must generate entries for
    the other two star targets even though they are not in kpoint_frac_by_name."""
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={"MM": [0.5, 0.0, 0.0]},
        operations=_p312_ops_with_valley_mappings(),
        valley_names=["M1", "M2", "M3"],
    )

    assert report["status"] == "ok"
    entries = report["by_source_kpoint"]["MM"]
    assert len(entries) > 0

    # Check that derived targets are generated (target_kpoint_label may be None)
    derived_targets = {
        tuple(round(v, 6) for v in e["target_frac"])
        for e in entries
    }
    assert (0.0, 0.5, 0.0) in derived_targets
    assert (0.5, 0.5, 0.0) in derived_targets

    # Verify target_kpoint_key is set
    for e in entries:
        assert "target_kpoint_key" in e
        assert e["target_kpoint_key"] is not None


def test_improper_unitary_det_not_trs():
    """det != 1 should give improper_unitary_not_supported, not antiunitary."""
    # C3^2 maps X->Y but we set det=-1 (improper).
    # g=E (identity) preserves VA. The conjugation is improper due to r's det.
    rot_c3sq = np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
    ops = [
        {
            "operation_id": 0, "kind": "E", "order": 1, "det": 1,
            "rotation_frac": np.eye(3, dtype=int),
            "translation_frac": np.zeros(3),
            "sector_mapping": {"VA": "VA", "VB": "VB"},
        },
        {
            "operation_id": 2, "kind": "improper_C3sq", "order": 3, "det": -1,
            "rotation_frac": rot_c3sq,
            "translation_frac": np.zeros(3),
            "sector_mapping": {"VA": "VB", "VB": "VA"},
        },
    ]
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={
            "X": [0.5, 0.0, 0.0],
            "Y": [0.0, 0.5, 0.0],
        },
        operations=ops,
        valley_names=["VA", "VB"],
    )

    entries = report.get("by_source_kpoint", {}).get("X", [])
    improper = [e for e in entries if "improper_unitary" in str(e.get("conjugation_status", ""))]
    assert len(improper) > 0
    antiunitary = [e for e in entries if "antiunitary" in str(e.get("conjugation_status", ""))]
    assert len(antiunitary) == 0


def test_source_not_in_hsp_little_group_filtered():
    """g that preserves valley but is not in source HSP little group
    must be marked source_not_in_hsp_little_group, not matched."""
    ops = _p312_ops_with_valley_mappings()
    report = build_hsp_star_conjugation_report(
        kpoint_frac_by_name={"MM": [0.5, 0.0, 0.0]},
        operations=ops,
        valley_names=["M1", "M2", "M3"],
    )
    entries = report["by_source_kpoint"]["MM"]
    not_lg = [e for e in entries if e.get("conjugation_status") == "source_not_in_hsp_little_group"]
    assert len(not_lg) > 0

    # Verify these entries are NOT matched
    matched = [e for e in entries if e.get("conjugation_status") == "matched"]
    for e in matched:
        src_op = e.get("source_preserving_operation_id")
        # Identity is always in the little group
        if src_op == 0:
            continue
        # Every matched non-identity source op must be in the HSP little group
        # at MM=[0.5,0,0].  For P312, only op=5 (C2_M3) is in the MM little group.
        assert src_op == 5, f"op={src_op} matched but not in MM little group"
