import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.workflows.analyze_hsp import analyze_hsp

from valleyscope.reports.analysis_outputs import write_analysis_outputs
from valleyscope.reports.summary_report import build_summary_payload, render_summary_text
from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table, build_reduced_ebr_mapping
from tests.helpers_io_workflow import write_fixture, write_config
from tests.helpers_io_workflow import _E2E_SAMPLE_TABLE, e2e_write_table

_DEBUG_ONLY_FILES = frozenset({
    "valley_subspace.json", "symmetry_report.json", "symmetry_eigenvalues.csv",
    "diagnostics.h5", "valley_basis_transform.h5",
    "projector_symmetry_report.json", "symmetry_adapted_valley_analysis.json",
    "target_subspace_closure.json", "hsp_star_conjugation.json",
    "hsp_star_derived_characters.json", "subspace_representation_quality.json",
    "irrep_workflow_decisions.json", "valley_irrep_matching.json",
    "valley_ebr_input_candidates.json", "valley_ebr_problem_instances.json",
    "folded_center_report.json", "sampled_k_coverage.json",
})

# Reduced EBR classification E2E smoke tests
# -----------------------------------------------------------------------


def test_reduced_ebr_e2e_config_path_writes_mapping_and_embeds_in_summary(tmp_path):
    """E2E: analyze_hsp with analysis.reduced_ebr.enabled writes mapping JSON
    and embeds it in valley_summary.json."""
    h5_path = tmp_path / "wf.h5"
    table_path = tmp_path / "table.json"
    out_dir = tmp_path / "out"
    e2e_write_table(table_path, _E2E_SAMPLE_TABLE)
    write_fixture(h5_path)
    config_path = tmp_path / "cfg.yaml"
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["analysis"]["reduced_ebr"] = {"enabled": True, "table_file": str(table_path)}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    mapping_path = out_dir / "valley_reduced_ebr_mapping.json"
    assert mapping_path.exists(), "valley_reduced_ebr_mapping.json must be written"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping["table_status"] == "loaded"

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "valley_reduced_ebr_mapping" in summary
    assert summary["valley_reduced_ebr_mapping"] == mapping
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "Reduced EBR mapping" in summary_text
    assert "table: loaded" in summary_text


def test_reduced_ebr_disabled_does_not_write_mapping(tmp_path):
    """E2E: without analysis.reduced_ebr.enabled, no mapping JSON is written."""
    h5_path = tmp_path / "wf.h5"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    config_path = tmp_path / "cfg.yaml"
    write_config(config_path, h5_path, out_dir)

    outputs = analyze_hsp(config_path)

    assert not (out_dir / "valley_reduced_ebr_mapping.json").exists()
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "valley_reduced_ebr_mapping" not in summary


def test_summary_text_surfaces_atomic_fragile_stable_classification():
    """E2E: summary text surfaces atomic, fragile, stable classifications."""
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text
    from valleyscope.io.config import load_config

    mapping = {
        "status": "no_exact_solution",
        "mapping_status": "no_exact_solution",
        "reduced_ebr_decomposition_status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [
            {"bundle_id": "b_atom", "valley": "K", "status": "solved_exact",
             "classification": "atomic-compatible-candidate",
             "integer_span_status": "in_integer_span",
             "nonnegative_solution_status": "solved_exact",
             "ebr_decomposition": [
                 {"label": "EBR_A", "coefficient": 1},
                 {"label": "EBR_B", "coefficient": 2},
             ]},
            {"bundle_id": "b_frag", "valley": "K", "status": "no_exact_solution",
             "classification": "fragile-topology-candidate",
             "integer_span_status": "in_integer_span",
             "nonnegative_solution_status": "no_nonnegative_solution",
             "integer_solution": [
                 {"label": "EBR_A", "coefficient": -1},
                 {"label": "EBR_B", "coefficient": 1},
             ]},
            {"bundle_id": "b_stab", "valley": "K", "status": "no_exact_solution",
             "classification": "stable-topology-candidate",
             "integer_span_status": "outside_integer_span",
             "nonnegative_solution_status": "no_nonnegative_solution"},
        ],
        "excluded_bundles": [],
        "solver": "smith_normal_form_plus_bounded_nonnegative_search",
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        h5 = d / "wf.h5"
        write_fixture(h5)
        cfg_path = d / "cfg.yaml"
        write_config(cfg_path, h5, d / "out")
        config = load_config(cfg_path)

        payload = build_summary_payload(
            config=config, qcut=0.5,
            subspace_payload={"kpoints": {}},
            symmetry_payload={"status": "skipped", "reason": "toy",
                              "detected_operations": [], "candidate_rotations": [],
                              "little_group_check": {"status": "not_run"},
                              "valley_preservation_check": {"status": "not_run"}},
            symmetry_rows=[], output_paths={},
            reduced_ebr_mapping=mapping,
        )
        text = render_summary_text(payload)

    assert "atomic-compatible=1" in text
    assert "fragile-topology=1" in text
    assert "stable-topology=1" in text
    assert "EBR_A x 1" in text
    assert "EBR_B x 2" in text
    assert "signed witness" in text
    assert "EBR_A: -1" in text
    assert "EBR_B: 1" in text
    assert "outside integer span" in text


def test_summary_text_truncated_search_surfaced():
    """E2E: truncated_by_max_coefficient search status appears in summary."""
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text
    from valleyscope.io.config import load_config

    mapping = {
        "status": "no_exact_solution",
        "mapping_status": "no_exact_solution",
        "reduced_ebr_decomposition_status": "no_exact_solution",
        "table_status": "loaded",
        "solutions": [{
            "bundle_id": "b_001", "valley": "K", "status": "no_exact_solution",
            "classification": "fragile-topology-candidate",
            "integer_span_status": "in_integer_span",
            "nonnegative_solution_status": "no_nonnegative_solution",
            "search_status": "truncated_by_max_coefficient",
            "integer_solution": [{"label": "EBR_A", "coefficient": -1}],
        }],
        "excluded_bundles": [],
        "solver": "smith_normal_form_plus_bounded_nonnegative_search",
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        h5 = d / "wf.h5"
        write_fixture(h5)
        cfg_path = d / "cfg.yaml"
        write_config(cfg_path, h5, d / "out")
        config = load_config(cfg_path)

        payload = build_summary_payload(
            config=config, qcut=0.5,
            subspace_payload={"kpoints": {}},
            symmetry_payload={"status": "skipped", "reason": "toy",
                              "detected_operations": [], "candidate_rotations": [],
                              "little_group_check": {"status": "not_run"},
                              "valley_preservation_check": {"status": "not_run"}},
            symmetry_rows=[], output_paths={},
            reduced_ebr_mapping=mapping,
        )
        text = render_summary_text(payload)

    assert "search_truncated=1" in text
    assert "truncated by max_coefficient" in text


def test_reduced_ebr_classifier_payload_written_consistently_to_public_outputs(tmp_path):
    """Classifier output is written consistently to mapping JSON and summaries."""
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping,
        load_reduced_ebr_table,
    )
    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    table_path = tmp_path / "table.json"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "C1",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:irrep_A", "GammaM:irrep_B", "GammaM:irrep_C"],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0, 0]},
            {"label": "EBR_B", "vector": [1, 1, 0]},
            {"label": "EBR_C", "vector": [0, 0, 2]},
        ],
    }
    e2e_write_table(table_path, table)
    loaded_table = load_reduced_ebr_table(table_path)
    export_bundle = {
        "bundles": [
            {
                "bundle_id": "b_atom",
                "valley": "K",
                "subspace_group_candidate": "C1",
                "ready_for_external_solver": True,
                "expected_hsps": ["GammaM"],
                "irreps_by_kpoint": {"GammaM": ["irrep_A", "irrep_A", "irrep_B"]},
            },
            {
                "bundle_id": "b_frag",
                "valley": "K",
                "subspace_group_candidate": "C1",
                "ready_for_external_solver": True,
                "expected_hsps": ["GammaM"],
                "irreps_by_kpoint": {"GammaM": ["irrep_B"]},
            },
            {
                "bundle_id": "b_stab",
                "valley": "K",
                "subspace_group_candidate": "C1",
                "ready_for_external_solver": True,
                "expected_hsps": ["GammaM"],
                "irreps_by_kpoint": {"GammaM": ["irrep_C"]},
            },
        ],
    }
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=export_bundle,
        table=loaded_table,
    )
    assert [s["classification"] for s in mapping["solutions"]] == [
        "atomic-compatible-candidate",
        "fragile-topology-candidate",
        "stable-topology-candidate",
    ]

    config = load_config(config_path)
    outputs = write_analysis_outputs(
        config=config,
        qcut=0.5,
        weight_rows=[],
        sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={"status": "skipped", "reason": "test",
                          "detected_operations": [], "candidate_rotations": [],
                          "little_group_check": {"status": "not_run"},
                          "valley_preservation_check": {"status": "not_run"}},
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
        reduced_ebr_mapping=mapping,
    )

    mapping_json = json.loads(outputs["valley_reduced_ebr_mapping_json"].read_text(encoding="utf-8"))
    summary_json = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert mapping_json == mapping
    assert summary_json["valley_reduced_ebr_mapping"] == mapping
    assert "classifications: atomic-compatible=1, fragile-topology=1, stable-topology=1" in summary_text
    assert "b_atom K: atomic-compatible" in summary_text
    assert "b_frag K: fragile-topology" in summary_text
    assert "b_stab K: stable-topology (outside integer span)" in summary_text


def test_e2e_smoke_fixture_table_is_material_agnostic():
    """E2E smoke fixture data must not name real validation materials."""
    fixture_text = json.dumps(_E2E_SAMPLE_TABLE)
    for name in ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]:
        assert name not in fixture_text


# -----------------------------------------------------------------------

# Provenance propagation smoke (output-writer fallback)
# -----------------------------------------------------------------------
# Direct analyze_hsp smoke is skipped because it requires a real HDF5
# with symmetry operations to produce trusted EBR candidates.  The
# output-writer path exercises the same builder chain
# (ebr_input_candidates -> ebr_problem_instances -> ebr_export_bundle
# -> write_analysis_outputs) that analyze_hsp uses at lines 452-468,
# so this is a valid end-to-end provenance pipeline test.

def test_provenance_survives_through_output_writer_to_export_bundle(tmp_path):
    """Provenance records survive through the full output-writer pipeline."""
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.reports.analysis_outputs import write_analysis_outputs
    from valleyscope.io.config import load_config

    # Trusted irrep matching with character data.
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
            "KM": {
                "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            },
        },
    }
    matching = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [0.5], "diagnostic_only": False},
                    "2": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/2",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [-0.5], "diagnostic_only": True},
                },
            },
            "KM": {
                "K_valley": {
                    "1": {"matching_status": "matched", "matched_irrep": "C3_spinor_phase_+1/6",
                          "subspace_group_candidate": "C3_like", "operation_order": 3,
                          "eigenphases": [0.166667], "diagnostic_only": False},
                },
            },
        },
    }
    # Character lookup via symmetry-adapted report.
    sa_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [{
                                "operation_id": 1,
                                "character": {"real": -1.0, "imag": 0.0},
                            }],
                        },
                    },
                }],
            },
            "KM": {
                "valley_preserving_subspaces": [{
                    "orbit": ["K_valley"],
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [{
                                "operation_id": 1,
                                "character": {"real": 0.5, "imag": 0.866025},
                            }],
                        },
                    },
                }],
            },
        },
    }

    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
        symmetry_adapted_valley_report=sa_report,
    )
    assert candidates["candidate_count"] == 2
    assert candidates["blocked_count"] == 1

    instances = build_ebr_problem_instances(ebr_input_candidates=candidates)
    assert instances["instance_count"] == 1
    inst = instances["instances"][0]
    assert inst["ready_for_ebr_decomposition"] is True

    export_bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    assert export_bundle["bundle_count"] == 1

    # Write through the output writer.
    h5_path = tmp_path / "wf.h5"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    config_path = tmp_path / "cfg.yaml"
    write_config(config_path, h5_path, out_dir)
    # Override to standard profile for smoke.
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)

    outputs = write_analysis_outputs(
        config=config, qcut=0.5, weight_rows=[], sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={"status": "skipped", "reason": "toy",
                          "detected_operations": [], "candidate_rotations": [],
                          "little_group_check": {"status": "not_run"},
                          "valley_preservation_check": {"status": "not_run"}},
        symmetry_rows=[], projectors_by_kpoint={}, qcut_scan_payload={},
        symmetry_representation_payload={}, basis_transforms={},
        ebr_export_bundle=export_bundle,
    )

    # Assertions on export bundle.
    assert outputs["valley_ebr_export_bundle_json"].exists()
    bundle_json = json.loads(outputs["valley_ebr_export_bundle_json"].read_text(encoding="utf-8"))
    bundle = bundle_json["bundles"][0]
    assert "irrep_records_by_kpoint" in bundle
    records = bundle["irrep_records_by_kpoint"]
    assert "GammaM" in records and "KM" in records

    gamma_rec = records["GammaM"][0]
    assert gamma_rec["valley"] == "K_valley"
    assert gamma_rec["operation_id"] == "1"
    assert gamma_rec["operation_order"] == 3
    assert gamma_rec["matched_irrep"] == "C3_spinor_phase_+1/2"
    assert gamma_rec["eigenphases"] == [0.5]
    assert gamma_rec["workflow_path"] == "direct_qcut"
    assert gamma_rec["readiness_level"] == "trusted"
    assert "valley_irrep_matching" in gamma_rec["source"]
    # Character preserved from SA report.
    assert gamma_rec["character"] is not None
    assert gamma_rec["character"]["real"] == -1.0
    assert all(rec["operation_id"] != "2" for recs in records.values() for rec in recs)

    km_rec = records["KM"][0]
    assert km_rec["operation_id"] == "1"
    assert km_rec["matched_irrep"] == "C3_spinor_phase_+1/6"
    assert km_rec["character"] is not None

    # irreps_by_kpoint still present for reduced EBR matching.
    assert "irreps_by_kpoint" in bundle
    assert bundle["irreps_by_kpoint"]["GammaM"] == ["C3_spinor_phase_+1/2"]

    # Summary embeds the export bundle with provenance.
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    embedded = summary["valley_ebr_export_bundle"]["bundles"][0]
    assert embedded["irrep_records_by_kpoint"] == records

    # Standard profile: no debug/detail files.
    written = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert not (written & _DEBUG_ONLY_FILES)


# -----------------------------------------------------------------------
# Reviewed C3 package-table E2E via table_name
# -----------------------------------------------------------------------


def test_analyze_hsp_uses_reviewed_package_table_by_name(tmp_path):
    """analyze_hsp loads the reviewed C3 package table by table_name
    and writes valley_reduced_ebr_mapping.json with table_status=loaded."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["analysis"]["reduced_ebr"] = {
        "enabled": True,
        "table_name": "P321_C3_like_GammaM_KM_spinful_v1",
    }
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    assert outputs["valley_ebr_export_bundle_json"].exists()
    assert outputs["valley_summary_json"].exists()
    mapping_path = outputs.get("valley_reduced_ebr_mapping_json")
    assert mapping_path is not None and mapping_path.exists()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping["table_status"] == "loaded"
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary["valley_reduced_ebr_mapping"] == mapping

    # Standard profile: exactly the public files, no debug/detail files.
    expected_public = {
        "valley_summary.txt",
        "valley_summary.json",
        "valley_weights.csv",
        "valley_ebr_export_bundle.json",
        "valley_reduced_ebr_mapping.json",
    }
    written = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert written == expected_public
    assert not (written & _DEBUG_ONLY_FILES)


# -----------------------------------------------------------------------


def test_p3_reviewed_table_by_name_e2e():
    """Load reviewed P3 reduced EBR table by manifest name, verify identity,
    and reach exact solver from a generic P3 export bundle shape."""
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping
    from valleyscope.data.reduced_ebr.catalog import (
        load_reduced_ebr_manifest,
        load_reviewed_reduced_ebr_table,
    )

    # 1. Load reviewed table by P3 physical manifest name.
    manifest = load_reduced_ebr_manifest()
    p3_entry = next(
        entry for entry in manifest["tables"]
        if entry["name"] == "P3_GammaM_KM_spinful_v1"
    )
    p3_source = p3_entry["source_reference"].lower()
    assert "sg150" in p3_source
    assert "restricted to the p3 valley-preserving subgroup" in p3_source
    assert "sg143" not in p3_source

    table = load_reviewed_reduced_ebr_table("P3_GammaM_KM_spinful_v1")
    assert table["subspace_group_candidate"] == "P3"
    assert table["expected_hsps"] == ["GammaM", "KM"]
    assert "provenance" in table
    prov = table["provenance"]
    assert prov.get("subspace_group_candidate") == "P3"
    assert prov.get("data_source") == "irreptables"
    assert prov.get("valleyscope_reduction") == "sampled_hsp_valley_preserving"
    assert prov.get("space_group_number") == 150
    # No C3_like label as physical table identity.
    assert table.get("subspace_group_candidate") != "C3_like"

    # 2. Build a generic P3 export bundle shape matching the table's HSPs.
    bundle = {
        "bundles": [{
            "bundle_id": "bundle_test_p3",
            "source_instance_id": "inst_001",
            "valley": "K_valley",
            "subspace_group_candidate": "P3",
            "workflow_path": "direct_qcut",
            "readiness_level": "trusted",
            "irreps_by_kpoint": {
                "GammaM": ["C3_spinor_phase_+1/2"],
                "KM": ["C3_spinor_phase_+1/2"],
            },
            "operations_by_kpoint": {
                "GammaM": [1],
                "KM": [1],
            },
            "expected_hsps": ["GammaM", "KM"],
            "optional_hsps": [],
            "missing_optional_hsps": [],
            "ready_for_external_solver": True,
        }],
        "excluded_instances": [],
    }

    # 3. Solve — must reach exact integer mapping.
    result = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle, table=table,
    )
    assert result["mapping_status"] == "solved_exact"
    solution = result["solutions"][0]
    assert solution["subspace_group_candidate"] == "P3"
    assert solution["classification"] == "atomic-compatible-candidate"

    # 4. Legacy compatibility: same table loads by legacy name.
    legacy_table = load_reviewed_reduced_ebr_table(
        "P321_C3_like_GammaM_KM_spinful_v1"
    )
    assert legacy_table["subspace_group_candidate"] == "P3"
