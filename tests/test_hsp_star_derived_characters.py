import json
import numpy as np

from valleyscope.analysis.hsp_star_derived_characters import (
    build_hsp_star_derived_characters,
    collect_derived_characters_by_target,
    has_derived_character_for,
)


def _make_conjugation_report():
    return {
        "status": "ok",
        "by_source_kpoint": {
            "MM": [
                {
                    "source_kpoint": "MM",
                    "target_kpoint_label": "MM2",
                    "target_frac": [0.0, 0.5, 0.0],
                    "mapping_operation_id": 2,
                    "source_valley": "M1",
                    "target_valley": "M1",
                    "source_preserving_operation_id": 4,
                    "derived_target_operation_id": 7,
                    "conjugation_status": "matched",
                    "reason": "",
                },
                {
                    "source_kpoint": "MM",
                    "target_kpoint_label": "MM3",
                    "target_frac": [0.5, 0.5, 0.0],
                    "mapping_operation_id": 1,
                    "source_valley": "M3",
                    "target_valley": "M3",
                    "source_preserving_operation_id": 5,
                    "derived_target_operation_id": 9,
                    "conjugation_status": "matched",
                    "reason": "",
                },
                {
                    "source_kpoint": "MM",
                    "target_kpoint_label": "MM2",
                    "target_frac": [0.0, 0.5, 0.0],
                    "mapping_operation_id": 2,
                    "source_valley": "M2",
                    "target_valley": "M2",
                    "source_preserving_operation_id": 3,
                    "conjugation_status": "missing_operation_product",
                    "reason": "h not in detected operations",
                },
            ],
        },
    }


def _make_source_char_diagnostics():
    return {
        "MM": {
            "status": "ok",
            "local_irrep_ready": True,
            "diagnostic_only": False,
            "per_valley": {
                "M1": [
                    {
                        "operation_id": 4,
                        "valley": "M1",
                        "character": {"real": 0.0, "imag": 0.0},
                        "eigenphases": [-0.25, 0.25],
                        "representation_unitarity_error": 1e-6,
                    },
                ],
                "M3": [
                    {
                        "operation_id": 5,
                        "valley": "M3",
                        "character": {"real": 2.0, "imag": 0.0},
                        "eigenphases": [0.25, 0.25],
                        "representation_unitarity_error": 1e-6,
                    },
                ],
            },
        },
    }


def test_explicit_mm_has_m3_c2_derived_m1_m2_c2():
    report = build_hsp_star_derived_characters(
        conjugation_report=_make_conjugation_report(),
        source_character_diagnostics=_make_source_char_diagnostics(),
    )

    entries = report["entries"]
    derived = [e for e in entries if e.get("status") == "derived"]
    assert len(derived) == 2

    m1_entry = next((e for e in derived if e.get("source_valley") == "M1"), None)
    assert m1_entry is not None
    assert m1_entry["trusted_for_ebr_input"] is True
    assert m1_entry["target_valley"] == "M1"
    assert m1_entry["target_kpoint_label"] == "MM2"

    missing_entry = next((e for e in entries if e.get("status") == "missing_operation_product"), None)
    assert missing_entry is not None
    assert missing_entry["trusted_for_ebr_input"] is False


def test_source_diagnostic_only_derived_diagnostic_only():
    conj_report = {
        "status": "ok",
        "by_source_kpoint": {
            "MM": [
                {
                    "source_kpoint": "MM",
                    "target_kpoint_label": "MM2",
                    "target_frac": [0.0, 0.5, 0.0],
                    "mapping_operation_id": 2,
                    "source_valley": "M1",
                    "target_valley": "M1",
                    "source_preserving_operation_id": 4,
                    "derived_target_operation_id": 7,
                    "conjugation_status": "matched",
                    "reason": "",
                },
            ],
        },
    }
    source_chars = {
        "MM": {
            "status": "diagnostic_only",
            "local_irrep_ready": False,
            "diagnostic_only": True,
            "per_valley": {
                "M1": [
                    {
                        "operation_id": 4,
                        "valley": "M1",
                        "character": {"real": 0.0, "imag": 0.0},
                        "eigenphases": [-0.25, 0.25],
                        "representation_unitarity_error": 1e-6,
                    },
                ],
            },
        },
    }

    report = build_hsp_star_derived_characters(
        conjugation_report=conj_report,
        source_character_diagnostics=source_chars,
    )

    entries = report["entries"]
    diag = [e for e in entries if e.get("status") == "diagnostic_only"]
    assert len(diag) == 1
    assert diag[0]["trusted_for_ebr_input"] is False


def test_unitary_character_copied_exactly():
    conj_report = {
        "status": "ok",
        "by_source_kpoint": {
            "MM": [
                {
                    "source_kpoint": "MM",
                    "target_kpoint_label": "MM2",
                    "target_frac": [0.0, 0.5, 0.0],
                    "mapping_operation_id": 2,
                    "source_valley": "M3",
                    "target_valley": "M3",
                    "source_preserving_operation_id": 5,
                    "derived_target_operation_id": 9,
                    "conjugation_status": "matched",
                    "reason": "",
                },
            ],
        },
    }
    source_chars = _make_source_char_diagnostics()

    report = build_hsp_star_derived_characters(
        conjugation_report=conj_report,
        source_character_diagnostics=source_chars,
    )

    derived = [e for e in report["entries"] if e.get("status") == "derived"]
    assert len(derived) == 1
    assert derived[0]["character"] == {"real": 2.0, "imag": 0.0}
    assert derived[0]["eigenphases"] == [0.25, 0.25]


def test_schema_json_serializable():
    report = build_hsp_star_derived_characters(
        conjugation_report=_make_conjugation_report(),
        source_character_diagnostics=_make_source_char_diagnostics(),
    )
    encoded = json.dumps(report)
    assert len(encoded) > 0
    assert "dtype" not in encoded
    assert "default=str" not in encoded


def test_collect_derived_characters_by_target():
    report = build_hsp_star_derived_characters(
        conjugation_report=_make_conjugation_report(),
        source_character_diagnostics=_make_source_char_diagnostics(),
    )
    by_target = collect_derived_characters_by_target(report)

    assert "MM2" in by_target
    assert "M1" in by_target["MM2"]
    assert 7 in by_target["MM2"]["M1"]
    assert "MM3" in by_target
    assert "M3" in by_target["MM3"]
    assert 9 in by_target["MM3"]["M3"]


def test_has_derived_character_for():
    report = build_hsp_star_derived_characters(
        conjugation_report=_make_conjugation_report(),
        source_character_diagnostics=_make_source_char_diagnostics(),
    )

    assert has_derived_character_for(report, "MM2", "M1", 7) is True
    assert has_derived_character_for(report, "MM3", "M3", 9) is True
    assert has_derived_character_for(report, "MM2", "M2", 3) is False
    assert has_derived_character_for(report, "XX", "M1", 7) is False


def test_closure_failed_blocks_derived():
    report = build_hsp_star_derived_characters(
        conjugation_report=_make_conjugation_report(),
        source_character_diagnostics=_make_source_char_diagnostics(),
        target_subspace_closure_blockers=["target_subspace_closure_failed"],
    )

    blocked = [e for e in report["entries"] if e.get("status") == "blocked_by_source_closure"]
    assert len(blocked) == 2
    assert len(report["blocked_sources"]) == 2
