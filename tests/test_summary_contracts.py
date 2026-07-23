"""Summary rendering contracts: human-readable text, output-file labels,
schema-doc coverage, and summary-payload field contracts."""

import ast
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.workflows.analyze_hsp import analyze_hsp

from tests.helpers_io_workflow import (
    write_fixture,
    write_config,
    write_simple_poscar,
)


# ---------------------------------------------------------------------------
# helper shared within this file
# ---------------------------------------------------------------------------

def _analysis_output_file_keys() -> set[str]:
    """Derive file-output keys from analysis_outputs AST.
    Excludes non-file return keys (summary_text, summary_stdout)."""
    from valleyscope.reports import analysis_outputs

    tree = ast.parse(inspect.getsource(analysis_outputs))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in ("outputs", "summary_path_plan"):
            continue
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            keys.add(node.slice.value)
    return keys - {"summary_text", "summary_stdout"}


# ---------------------------------------------------------------------------
# summary rendering
# ---------------------------------------------------------------------------

def _concise_summary_contract_payload() -> dict[str, object]:
    return {
        "input": {
            "wavefunction_h5": "fixture.h5",
            "operation_structure_file": "structure.vasp",
            "operation_detection_backend": "spglib",
            "spinor_convention": "vasp",
            "spinor_convention_verified": True,
            "spinor_benchmark": "reviewed",
        },
        "target_kpoints": ["H0"],
        "iband": [7],
        "valley_subspaces": [
            {"label": "alpha", "centers": ["A"]},
            {"label": "beta", "centers": ["B"]},
        ],
        "qcut": {
            "projector_mode": "fixed_center",
            "mode": "relative_min_valley_distance",
            "value_Ainv": 0.125,
            "fraction": 0.25,
        },
        "valley_projection_summary": [{
            "kpoint": "H0",
            "band_vasp": 7,
            "valley_weights": {"alpha": 0.7, "beta": 0.2},
            "W_val": 0.9,
            "P_v": 0.7 / 0.9,
            "W_overlap": 0.0,
            "W_res": 0.1,
            "status": "mixed",
        }],
        "valley_subspace_analysis": [],
        "valley_projector_quality": [{"kpoint": "H0"}],
        "symmetry_analysis": {
            "status": "ok",
            "international": "parent",
            "spacegroup_number": 99,
            "detected_operations": [{"operation_id": 0, "kind": "identity"}],
            "by_kpoint": {},
        },
        "symmetry_eigenvalues": [{
            "kpoint": "H0",
            "target_valley": "alpha",
            "operation_id": 0,
            "order": 1,
            "state_index": 0,
            "phase_2pi": 0.0,
            "nearest_root_of_unity": "1",
            "root_deviation": 0.0,
        }],
        "projector_symmetry": {"status": "ok", "by_kpoint": {}},
        "symmetry_adapted_valley_analysis": {"status": "not_needed"},
        "target_subspace_closure": {"status": "ok", "by_kpoint": {}},
        "hsp_star_conjugation": {"status": "ok", "by_source_kpoint": {}},
        "hsp_star_derived_characters": {"status": "ok", "entries": []},
        "irrep_workflow_decisions": {"workflow_paths": [], "readiness_levels": []},
        "valley_ebr_input_candidates": {
            "status": "ready",
            "candidate_count": 1,
            "blocked": [{"reason": "source HSP coverage incomplete"}],
        },
        "valley_ebr_problem_instances": {
            "status": "ready",
            "instance_count": 1,
            "instances": [{
                "status": "complete_but_blocked",
                "blocked_by": ["canonical HSP vector not ready"],
            }],
        },
        "valley_resolved_irreps": {
            "status": "ok",
            "matched_count": 1,
            "blocked_count": 1,
            "diagnostic_count": 0,
            "non_source_count": 0,
            "rows": [{
                "kpoint": "H0",
                "valley": "alpha",
                "subspace_space_group": "SgX",
                "hsp_little_group_operation_ids": [0, 4],
                "valley_preserving_operation_ids": [0],
                "matching_status": "matched",
                "irrep_multiplicities": {"rho": 1},
                "readiness_level": "trusted",
                "diagnostic_only": False,
                "reason": "",
            }, {
                "kpoint": "H0",
                "valley": "beta",
                "subspace_space_group": "SgX",
                "hsp_little_group_operation_ids": [0, 4],
                "valley_preserving_operation_ids": [0],
                "matching_status": "blocked",
                "irrep_multiplicities": {},
                "readiness_level": "blocked",
                "diagnostic_only": True,
                "reason": "seed projector symmetry-consistency failed",
            }],
        },
        "valley_projected_representations": {
            "trusted_representation_count": 1,
            "blocked_representation_count": 1,
            "diagnostic_only_count": 1,
            "subspace_space_group_counts": {"SgX": 2},
            "rows": [{
                "kpoint": "H0",
                "valley": "beta",
                "blocking_reasons": [
                    "seed projector symmetry-consistency failed",
                ],
            }],
        },
        "valley_ebr_export_bundle": {
            "status": "ready_for_reduced_table_validation",
            "bundle_count": 1,
            "excluded_count": 0,
            "bundles": [],
            "excluded_instances": [],
        },
        "valley_reduced_ebr_mapping": {
            "status": "solved_exact",
            "table_status": "loaded",
            "reduced_ebr_input": {
                "table_input_provenance_by_bundle": {
                    "bundle-1": {
                        "data_source": "reviewed-source",
                        "package_version": "1.2.3",
                    },
                },
            },
            "solutions": [{
                "bundle_id": "bundle-1",
                "physical_object_kind": "unitary_valley_projected_subspace",
                "valley": "alpha",
                "valley_orbit": ["alpha"],
                "classification": "atomic-compatible-candidate",
                "ebr_decomposition": [{"label": "E@1a", "coefficient": 1}],
            }],
            "excluded_bundles": [],
        },
        "warnings": ["projection overlap requires review"],
        "output_profile": "standard",
        "output_files": {
            "valley_summary_txt": "out/valley_summary.txt",
            "valley_summary_json": "out/valley_summary.json",
            "valley_reduced_ebr_mapping_json":
                "out/valley_reduced_ebr_mapping.json",
        },
    }


def test_standard_summary_is_concise_result_first_and_preserves_payload():
    from valleyscope.reports.summary_report import render_summary_text

    summary = _concise_summary_contract_payload()
    before = json.dumps(summary, sort_keys=True)

    text = render_summary_text(summary)

    assert json.dumps(summary, sort_keys=True) == before
    required_in_order = [
        "Run and projection context",
        "qcut value: 0.125 A^-1",
        "qcut fraction: 0.25",
        "Valley projection by sampled state",
        "alpha",
        "beta",
        "Valley-projected subspace space group and trusted HSP irreps",
        "SgX",
        "rho:1",
        "Authoritative reduced EBR results",
        "unitary_valley_projected_subspace",
        "reviewed-source 1.2.3",
        "atomic-compatible-candidate",
        "E@1a x 1",
        "Readiness blockers and warnings",
        "canonical HSP vector not ready",
        "seed projector symmetry-consistency failed",
        "source HSP coverage incomplete",
        "Public output files",
    ]
    positions = [text.index(value) for value in required_in_order]
    assert positions == sorted(positions)

    detailed_sections = [
        "Detected operations:",
        "Symmetry eigenvalues",
        "Projected q-cut seed projector quality",
        "Projector symmetry-consistency",
        "Symmetry-adapted valley analysis",
        "Target-subspace symmetry closure",
        "HSP-star conjugation",
        "HSP-star derived characters",
        "Irrep workflow decisions",
        "EBR input candidates",
        "EBR problem instances",
    ]
    assert not any(section in text for section in detailed_sections)


def test_debug_summary_retains_detailed_diagnostic_report():
    from valleyscope.reports.summary_report import render_summary_text

    summary = _concise_summary_contract_payload()
    summary["output_profile"] = "debug"

    text = render_summary_text(summary)

    assert "Detected operations:" in text
    assert "Symmetry eigenvalues" in text
    assert "Projected q-cut seed projector quality" in text
    assert "Projector symmetry-consistency" in text
    assert "Target-subspace symmetry closure" in text
    assert "HSP-star conjugation" in text
    assert "HSP-star derived characters" in text
    assert "Irrep workflow decisions" in text
    assert "EBR input candidates" in text
    assert "EBR problem instances" in text


def test_summary_distinguishes_unitary_and_joint_tr_ebr_objects():
    from valleyscope.reports.summary_report import _render_ebr_problem_instances

    report = {
            "status": "ready",
            "instance_count": 2,
            "instances": [{
                "instance_id": "u",
                "physical_object_kind": "unitary_valley_projected_subspace",
                "valley": "K",
                "status": "ready",
                "canonical_hsp_vector_complete": True,
                "canonical_hsp_vector_ready": True,
                "unitary_irrep_completion_records_by_hsp": {
                    "GM": [{"completion_kind": "observed_at_sampled_kpoint"}],
                    "KA": [{"completion_kind": "inferred_by_time_reversal"}],
                },
            }, {
                "instance_id": "j",
                "physical_object_kind": "joint_time_reversal_valley_orbit",
                "valley_orbit": ["K", "Kp"],
                "status": "ready",
                "canonical_hsp_vector_complete": True,
                "canonical_hsp_vector_ready": True,
            }],
        }

    lines = []
    _render_ebr_problem_instances(lines, report)
    text = "\n".join(lines)
    assert "unitary_valley_projected_subspace" in text
    assert "joint_time_reversal_valley_orbit" in text
    assert "observed=1, inferred=1" in text

def test_summary_text_renders_qcut_fraction_for_relative_mode(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["projection"].pop("qcut_Ainv", None)
    raw["projection"]["qcut_mode"] = "relative_min_valley_distance"
    raw["projection"]["qcut_fraction"] = 0.2
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.034,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "skipped",
            "reason": "no structure",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert summary["schema_version"] == "1.8.0"
    assert summary["qcut"]["fraction"] == pytest.approx(0.2)
    text = render_summary_text(summary)
    assert "qcut mode: relative_min_valley_distance" in text
    assert "qcut value: 0.034 A^-1" in text
    assert "qcut fraction: 0.2" in text


def test_summary_warns_about_skipped_rotation_representation(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "GammaM": {
                    "weights": [],
                    "warnings": [],
                    "valley_adapted_subspace": {"status": "single_band"},
                }
            }
        },
        symmetry_payload={
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "CONTCAR",
            "detected_operation_count": 1,
            "candidate_rotations": [0],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
            "detected_operations": [
                {
                    "operation_id": 0,
                    "kind": "C2",
                    "candidate_rotation": True,
                    "representation_quality": {
                        "GammaM": {"skipped_reason": "unsupported nspinor=3"}
                    },
                }
            ],
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert any("unsupported nspinor=3" in warning for warning in summary["warnings"])
    assert "unsupported nspinor=3" in render_summary_text(summary)


def test_summary_text_renders_symmetry_adapted_valley_subspaces(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "GammaM": {
                    "weights": [],
                    "warnings": [],
                    "valley_adapted_subspace": {"status": "single_band"},
                }
            }
        },
        symmetry_payload={
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "CONTCAR",
            "detected_operation_count": 0,
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
            "detected_operations": [],
        },
        symmetry_rows=[],
        output_paths={},
        symmetry_adapted_valley_report={
            "space_group_valley_orbits": [["M1_valley", "M2_valley", "M3_valley"]],
            "by_kpoint": {
                "GammaM": {
                    "status": "warn",
                    "reason": "projector warning",
                    "feature_status": "formal",
                    "workflow_integration_status": "integrated",
                    "local_irrep_ready": True,
                    "diagnostic_only": False,
                    "irrep_matching_input_ready": False,
                    "irrep_matching_input_reason": "spinor convention unverified",
                    "orbits": [
                        {
                            "orbit": ["M1_valley", "M2_valley", "M3_valley"],
                            "status": "diagnostic_only",
                            "local_irrep_ready": False,
                            "irrep_matching_input_ready": False,
                            "reason": "HSP-local orbit diagnostic",
                            "symmetry_adapted_projectors": {
                                "selected_rank": 0,
                                "status": "failed",
                                "max_projector_symmetry_error": 0.0,
                            },
                        }
                    ],
                    "valley_preserving_subspaces": [
                        {
                            "orbit": ["M1_valley"],
                            "analysis_scope": "valley_preserving_subspace",
                            "hsp_preserving_operation_ids": [0, 4],
                            "status": "warn",
                            "local_irrep_ready": True,
                            "irrep_matching_input_ready": False,
                            "irrep_matching_input_reason": "spinor convention unverified",
                            "subspace_space_group": {
                                "candidate_space_group_symbol": "P2",
                                "valley_preserving_operation_ids": [0, 4],
                            },
                            "subspace_group": {
                                "subspace_group_candidate": "P2",
                            },
                            "symmetry_adapted_projectors": {
                                "selected_rank": 2,
                                "rank_source": "user_specified",
                                "seed_overlap": {"M1_valley": 0.7385595},
                                "max_projector_symmetry_error": 1.1e-5,
                                "status": "warn",
                            },
                            "valley_preserving_character_diagnostics": {
                                "per_valley": {
                                    "M1_valley": [
                                        {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                        {"operation_id": 4, "eigenphases": [-0.25, 0.25]},
                                    ]
                                }
                            },
                        }
                    ],
                }
            }
        },
    )

    text = render_summary_text(summary)

    assert "Symmetry-adapted valley analysis" in text
    assert "Symmetry-adapted valley analysis (experimental)" not in text
    assert "space-group valley orbits: [M1_valley, M2_valley, M3_valley]" in text
    assert "HSP-local valley-orbit reports" in text
    assert "full valley-orbit reports" not in text
    assert "valley-preserving subspaces" in text
    assert "GammaM" in text
    assert "M1_valley" in text
    assert "P2" in text
    assert "[0, 4]" in text
    assert "-0.25, 0.25" in text
    assert "spinor convention unverified" in text


def test_summary_payload_renders_valley_projected_representations(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    report = {
        "rows": [
            {
                "kpoint": "GammaM",
                "valley": "M1_valley",
                "operation_id": 4,
                "operation_order": 2,
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P2",
                    "valley_preserving_operation_ids": [0, 4],
                },
                "hsp_little_group_operation_ids": [0, 4],
                "valley_preserving_operation_ids": [0, 4],
                "readiness_level": "trusted",
                "workflow_path": "direct_qcut",
                "blocking_reasons": [],
            },
        ],
        "representation_records": [
            {
                "kpoint": "GammaM",
                "valley": "M1_valley",
                "subspace_space_group": {
                    "candidate_space_group_symbol": "P2",
                    "valley_preserving_operation_ids": [0, 4],
                },
                "hsp_little_group_operation_ids": [0, 4],
                "valley_preserving_operation_ids": [0, 4],
                "valley_changing_operation_ids": [],
                "valley_preserving_operations": [
                    {
                        "operation_id": 4,
                        "operation_order": 2,
                        "diagnostic_only": False,
                        "topology_input_ready": True,
                    }
                ],
                "readiness_level": "trusted",
                "workflow_path": "direct_qcut",
                "blocking_reasons": [],
                "irrep_matching": None,
            },
        ],
        "grouped_record_count": 1,
        "subspace_space_group_counts": {"P2": 1},
        "trusted_representation_count": 1,
        "blocked_representation_count": 0,
        "diagnostic_only_count": 0,
    }

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "skipped",
            "reason": "unit test",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
        valley_projected_representation=report,
    )

    assert "P2" in json.dumps(report)
    assert "C2_like" not in json.dumps(summary)
    assert summary["valley_projected_representations"]["subspace_space_group_counts"] == {"P2": 1}
    text = render_summary_text(summary)
    assert "Valley-projected representations" in text
    assert "subspace space groups: P2=1" in text
    assert "GammaM" in text
    assert "M1_valley" in text
    assert "P2" in text
    assert "C2_like" not in text


def test_summary_text_renders_generic_irrep_matching_without_legacy_rows(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    matching = {
        "status": "ok",
        "matching_mode": "generic",
        "by_kpoint": {},
        "generic_matches_by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "matching_strategy": "bilbao_restricted_character",
                    "matching_status": "matched",
                    "irrep_multiplicities": {"-GM5": 1},
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P3",
                    },
                    "valley_preserving_operation_ids": [0, 2],
                    "reason": "restricted character match",
                    "readiness_level": "trusted",
                    "diagnostic_only": False,
                },
            },
        },
    }
    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "skipped",
            "reason": "unit test",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
        valley_irrep_matching=matching,
    )

    assert "valley_resolved_irreps" in summary
    resolved = summary["valley_resolved_irreps"]
    assert resolved["status"] == "ok"
    assert resolved["rows"][0]["subspace_space_group"] == "P3"
    assert resolved["rows"][0]["diagnostic_only"] is False

    text = render_summary_text(summary)
    assert "Valley-resolved irreps" in text
    assert "Valley irrep matching" in text
    assert "mode: generic" in text
    assert "legacy phase tables" not in text
    assert "generic restricted-character matches:" in text
    assert "GammaM" in text
    assert "K_valley" in text
    assert "bilbao_restricted_character" in text
    assert "-GM5:1" in text
    assert "P3" in text
    assert "tables implemented:" not in text


def test_summary_text_renders_projected_seed_projector_quality(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "GammaM": {
                    "weights": [],
                    "warnings": [],
                    "valley_adapted_subspace": {
                        "status": "valley_separable",
                        "s_min": 0.98,
                        "s_max": 0.99,
                        "assigned_valleys": ["M1_valley", "M1_valley"],
                        "min_valley_concentration": 0.999,
                        "projector_quality": {
                            "expected_rank": 2,
                            "rank_threshold": 0.5,
                            "per_valley": {
                                "M1_valley": {
                                    "rank_estimate": 2,
                                    "rank_gap": 0.97,
                                }
                            },
                            "sum_projector": {
                                "identity_deviation_fro": 0.04,
                                "idempotency_deviation_fro": 0.02,
                            },
                            "max_idempotency_deviation": 0.02,
                            "max_trace_overlap": 1.0e-4,
                            "max_commutator_norm": 2.0e-4,
                        },
                    },
                }
            }
        },
        symmetry_payload={
            "status": "skipped",
            "reason": "test",
            "detected_operations": [],
        },
        symmetry_rows=[],
        output_paths={},
    )

    text = render_summary_text(summary)

    assert "Projected q-cut seed projector quality" in text
    assert "rank_estimates" in text
    assert "M1_valley=2" in text
    assert "M1_valley=0.97" in text


def test_summary_text_renders_hsp_star_coverage(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "MM": {
                    "weights": [],
                    "warnings": [],
                    "valley_adapted_subspace": {"status": "single_band"},
                }
            }
        },
        symmetry_payload={
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "CONTCAR",
            "detected_operation_count": 0,
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
            "detected_operations": [],
            "hsp_star_report": {
                "status": "symmetry_derivable",
                "by_kpoint": {
                    "MM": {
                        "status": "symmetry_derivable",
                        "star_size": 3,
                        "explicit_count": 1,
                        "symmetry_derivable_count": 2,
                        "requires_additional_dft": False,
                        "symmetry_derivable_representatives": [
                            {
                                "canonical_frac": [0.0, 0.5, 0.0],
                                "generated_by_operation_ids": [2],
                            }
                        ],
                    }
                },
            },
        },
        symmetry_rows=[],
        output_paths={},
    )

    text = render_summary_text(summary)

    assert "HSP-star coverage" in text
    assert "symmetry_derivable" in text
    assert "False" in text
    assert "[0, 0.5, 0] via ops [2]" in text


def test_summary_marks_spinor_rotation_as_diagnostic_only(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "GammaM": {
                    "weights": [],
                    "warnings": [],
                    "valley_adapted_subspace": {"status": "two_valley_adapted"},
                }
            }
        },
        symmetry_payload={
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "CONTCAR",
            "detected_operation_count": 1,
            "candidate_rotations": [0],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
            "detected_operations": [],
        },
        symmetry_rows=[
            {
                "kpoint": "GammaM",
                "operation_id": 0,
                "order": 2,
                "basis": "valley_adapted",
                "state_index": 0,
                "phase_2pi": 0.5,
                "nearest_root_of_unity": "exp(2pii*1/2)",
                "root_deviation": 0.0,
                "rotation_ready": True,
                "topology_input_ready": False,
                "topology_ready": False,
                "spinor_rotation_applied": True,
                "spinor_convention_verified": False,
                "diagnostic_only": True,
                "D_valley_offdiag_norm": 0.0,
            }
        ],
        output_paths={},
    )

    assert any("Spinor rotation is applied" in warning for warning in summary["warnings"])
    text = render_summary_text(summary)
    assert "topology_input_ready" in text
    assert "diagnostic-only" in text


def test_summary_output_files_use_human_readable_labels(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "skipped",
            "reason": "no structure",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={
            "symmetry_eigenvalues_csv": out_dir / "symmetry_eigenvalues.csv",
            "valley_summary_txt": out_dir / "valley_summary.txt",
        },
    )

    text = render_summary_text(summary)

    assert "Symmetry eigenvalues:" in text
    assert "Human-readable summary:" in text
    assert "symmetry_eigenvalues_csv:" not in text
    assert "valley_summary_txt:" not in text


def test_summary_subspace_polarization_uses_min_eta_score(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "GammaM": {
                    "weights": [],
                    "warnings": [],
                    "polarization_score": 0.2,
                    "subspace_valley_status": "valley_mixed_subspace",
                    "valley_adapted_subspace": {
                        "status": "two_valley_adapted",
                        "eta": [0.99, 0.2],
                        "max_abs_eta": 0.99,
                        "s_min": 0.98,
                        "s_max": 1.0,
                    },
                }
            }
        },
        symmetry_payload={
            "status": "skipped",
            "reason": "no structure",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert summary["valley_subspace_analysis"][0]["P_v_min"] == pytest.approx(0.6)


def test_summary_status_keeps_not_derived_and_unreliable_distinct(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "GammaM": {
                    "weights": [
                        {"band_vasp": 101, "valley_status": "not_valley_derived", "W_val": 0.2},
                        {"band_vasp": 102, "valley_status": "projector_unreliable", "W_overlap": 0.2},
                    ],
                    "warnings": [],
                    "subspace_valley_status": "projector_unreliable",
                    "valley_adapted_subspace": {"status": "two_valley_adapted"},
                },
                "KM": {
                    "weights": [],
                    "warnings": [],
                    "valley_adapted_subspace": {"status": "single_band"},
                },
            }
        },
        symmetry_payload={
            "status": "skipped",
            "reason": "no structure",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert [row["status"] for row in summary["valley_projection_summary"]] == ["not_derived", "unreliable"]
    assert summary["valley_subspace_analysis"][0]["status"] == "unreliable"
    assert summary["valley_subspace_analysis"][1]["status"] == "n/a"


def test_valley_projection_summary_contains_per_valley_weights(tmp_path):
    """valley_projection_summary rows include valley_weights dict."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={
            "kpoints": {
                "GammaM": {
                    "weights": [{
                        "band_vasp": 1,
                        "valley_weights": {"K_valley": 0.8, "Kp_valley": 0.1},
                        "W_val": 0.9, "P_v": 0.89, "eta": 0.78,
                        "W_overlap": 0.05, "W_res": 0.05,
                        "valley_status": "clean",
                    }],
                    "warnings": [],
                    "valley_adapted_subspace": {"status": "two_valley_adapted"},
                },
            },
        },
        symmetry_payload={
            "status": "skipped", "reason": "no structure",
            "detected_operations": [], "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
        symmetry_rows=[],
        output_paths={},
    )
    row = summary["valley_projection_summary"][0]
    assert row["valley_weights"] == {"K_valley": 0.8, "Kp_valley": 0.1}
    assert row["W_val"] == 0.9
    assert row["P_v"] == 0.89


def test_symmetry_analysis_distinguishes_computed_from_diagnostic_only(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[
            {
                "kpoint": "KM",
                "operation_id": 1,
                "order": 3,
                "basis": "valley_adapted",
                "state_index": 0,
                "phase_2pi": 1.0 / 6.0,
                "nearest_root_of_unity": "exp(2pii*1/6)",
                "root_deviation": 0.0,
                "rotation_ready": True,
                "topology_input_ready": False,
                "diagnostic_only": True,
                "D_valley_offdiag_norm": 0.01,
                "reason": "two-valley D_valley offdiag diagnostic too large",
            }
        ],
        output_paths={},
    )

    text = render_summary_text(summary)
    assert "two-valley D_valley offdiag diagnostic too large" in text
    assert "exp(i*pi/3)" in text
    assert "exp(2pii*1/6)" not in text
    assert "valley_adapted" not in text
    assert "skipped (two-valley D_valley offdiag diagnostic too large)" not in text


def test_symmetry_summary_orders_hsp_and_labels_valley_exchanging(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["analysis"]["kpoints"] = ["GammaM", "KM"]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [
                {
                    "operation_id": 2,
                    "kind": "C2",
                    "order": 2,
                    "det": 1,
                    "rotation_frac": np.eye(3),
                    "translation_frac": np.zeros(3),
                    "little_group_by_kpoint": {"KM": True, "GammaM": True},
                    "rejection_reason_by_kpoint": {"KM": "valley-exchanging", "GammaM": "valley-exchanging"},
                }
            ],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    rejected = summary["symmetry_analysis"]["rejected_operations"]
    assert [row["kpoint"] for row in rejected] == ["GammaM", "KM"]
    text = render_summary_text(summary)
    assert text.index("GammaM: HSP little group") < text.index("KM: HSP little group")
    assert "valley-exchanging" in text


def test_summary_rejected_operations_are_per_valley_when_inventory_available(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    symmetry_payload = {
        "status": "ok",
        "detected_operations": [
            {
                "operation_id": 3,
                "kind": "C2",
                "order": 2,
                "det": 1,
                "rotation_frac": np.eye(3),
                "translation_frac": np.zeros(3),
                "little_group_by_kpoint": {"GammaM": True},
                "rejection_reason_by_kpoint": {"GammaM": "valley-exchanging"},
            }
        ],
        "candidate_rotations": [],
        "per_valley_preserving_operation_inventory": {
            "GammaM": {
                "M1_valley": [
                    {
                        "operation_id": 3,
                        "kind": "C2",
                        "order": 2,
                        "little_group_passed": True,
                        "target_valley": "M1_valley",
                        "mapped_valley": "M1_valley",
                        "valley_preserving": True,
                        "allowed_for_valley_preserving_representation": True,
                        "reason": "",
                    }
                ],
                "M2_valley": [
                    {
                        "operation_id": 3,
                        "kind": "C2",
                        "order": 2,
                        "little_group_passed": True,
                        "target_valley": "M2_valley",
                        "mapped_valley": "M3_valley",
                        "valley_preserving": False,
                        "allowed_for_valley_preserving_representation": False,
                        "reason": "valley-changing (maps to M3_valley)",
                    }
                ],
            }
        },
        "little_group_check": {"status": "evaluated_per_kpoint"},
        "valley_preservation_check": {"status": "completed"},
    }

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload=symmetry_payload,
        symmetry_rows=[],
        output_paths={},
    )

    rejected = summary["symmetry_analysis"]["rejected_operations"]
    assert rejected == [
        {
            "kpoint": "GammaM",
            "target_valley": "M2_valley",
            "operation_id": 3,
            "order": 2,
            "kind": "C2",
            "reason": "valley-changing (maps to M3_valley)",
        }
    ]
    text = render_summary_text(summary)
    assert "M2_valley" in text
    assert "M1_valley" not in text.split("rejected operations:", 1)[1]


def test_summary_preserves_hsp_little_group_inventory(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload

    inventory = {
        "KM": [
            {
                "operation_id": 7,
                "kind": "C3",
                "order": 3,
                "little_group_passed": True,
                "valley_preserving": True,
                "valley_exchanging": False,
                "allowed_for_valley_preserving_representation": True,
                "reason": "",
            }
        ]
    }
    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "hsp_little_group_inventory": inventory,
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert summary["symmetry_analysis"]["hsp_little_group_inventory"] == inventory


def test_summary_preserves_valley_preserving_subgroup_report(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload, render_summary_text

    subgroup_report = {
        "status": "standard_group_matched",
        "global_operation_set": {
            "operation_set_label": "G_tau",
            "allowed_operation_ids": [0, 1, 2],
            "closure_status": "closed",
        },
        "standard_group_match": {
            "number": 143,
            "international_short": "P3",
            "source": "spglib.get_spacegroup_type_from_symmetry",
            "operation_ids": [0, 1, 2],
        },
        "standard_group_match_status": "matched",
        "by_kpoint": {
            "KM": {
                "operation_set_label": "G_tau,k(KM)",
                "allowed_operation_ids": [0, 1, 2],
                "closure_status": "closed",
                "missing_products": [],
            }
        },
    }
    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "valley_preserving_subgroup_report": subgroup_report,
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    assert summary["symmetry_analysis"]["valley_preserving_subgroup_report"] == subgroup_report
    text = render_summary_text(summary)
    assert "valley-preserving subgroup: P3 (143)" in text


def test_summary_exposes_symmetry_characters_as_first_class_rows(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload

    symmetry_rows = [
        {
            "kpoint": "KM",
            "operation_id": 7,
            "kind": "C3",
            "order": 3,
            "basis": "valley_adapted",
            "little_group_passed": True,
            "valley_preserving": True,
            "character_raw": "1.000000+0.000000j",
            "character_valley": "0.500000+0.866025j",
            "topology_input_ready": True,
            "diagnostic_only": False,
        },
        {
            "kpoint": "KM",
            "operation_id": 7,
            "kind": "C3",
            "order": 3,
            "basis": "valley_adapted",
            "little_group_passed": True,
            "valley_preserving": True,
            "character_raw": "",
            "character_valley": "",
            "topology_input_ready": True,
            "diagnostic_only": False,
        },
        {
            "kpoint": "KM",
            "operation_id": 8,
            "kind": "C2",
            "order": 2,
            "basis": "valley_adapted",
            "little_group_passed": True,
            "valley_preserving": False,
            "character_raw": "0.000000+0.000000j",
            "character_valley": "0.000000+0.000000j",
            "topology_input_ready": False,
            "diagnostic_only": True,
        },
    ]
    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=symmetry_rows,
        output_paths={},
    )

    assert summary["symmetry_characters"] == [
        {
            "kpoint": "KM",
            "target_valley": "",
            "operation_id": 7,
            "kind": "C3",
            "order": 3,
            "basis": "valley_adapted",
            "character_raw": "1.000000+0.000000j",
            "character_valley": "0.500000+0.866025j",
            "topology_input_ready": True,
            "diagnostic_only": False,
            "accepted_for_valley_preserving_representation": True,
        }
    ]


def test_summary_exposes_rotation_readiness_thresholds(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["rotation"] = {"readiness_preset": "normal"}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)
    from valleyscope.reports.summary_report import build_summary_payload

    summary = build_summary_payload(
        config=config,
        qcut=0.5,
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        output_paths={},
    )

    thresholds = summary["rotation_readiness_thresholds"]
    assert thresholds["readiness_preset"] == "normal"
    assert thresholds["root_deviation_tol"] == pytest.approx(1.0e-5)
    assert thresholds["D_valley_offdiag_tol"] == pytest.approx(1.0e-3)
    assert thresholds["irrep_weight_tol"] == pytest.approx(5.0e-5)
    assert "not universal physical constants" in thresholds["interpretation"]
    assert "do not loosen" in thresholds["recommended_action"]


# ---------------------------------------------------------------------------
# README doc contracts
# ---------------------------------------------------------------------------

def test_readme_symmetry_example_uses_parser_schema(tmp_path):
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "symmetry:\n  operations:" in readme
    assert "valley_subspaces:" in readme
    assert "valley_manifolds" not in readme
    assert "valley_sectors" not in readme
    assert "target_bands_vasp" not in readme
    assert "raw_valley_clean" not in readme
    assert "valley_separable_subspace" not in readme
    assert "relative_min_sector_distance" not in readme
    assert "overlap_cross_sector" not in readme
    assert "valley_sectors" not in readme
    assert "valley_manifolds" not in readme
    assert "valley_manifolds" not in readme.lower()
    assert not re.search(r"(?<!\w)V[123](?!\.\d)", readme)
    assert "input:\n  wavefunction_h5: ./wave.h5\n\n  # Moire/bilayer POSCAR" not in readme
    assert "symmetry:\n  source: spglib" not in readme
    assert "Example screen summary" in readme
    assert "Valley projection summary" in readme
    assert "Valley subspace analysis" in readme
    assert "Two-valley subspace" not in readme
    assert "S_min" in readme and "minimum target-valley-subspace weight" in readme
    assert "min_concentration" in readme
    assert "assigned_valleys" in readme
    assert "qcut mode:" in readme
    assert "`not_derived`" in readme
    assert "`unreliable`" in readme
    assert "valley-preserving irrep" in readme
    assert "valley-preserving subgroup" in readme
    assert "irrep_multiplicities" in readme
    assert "valley_resolved_irreps" in readme
    assert "P321 No.150" not in readme
    assert "P3 No.143" not in readme
    assert "Benchmark:" not in readme
    assert "double-valued" in readme
    assert "`root_deviation_tol`, `D_valley_offdiag_tol`, and `irrep_weight_tol` are numerical readiness thresholds" in readme
    assert "`strict`, `normal`, and `loose`" in readme

    match = re.search(
        r"For a `generate_hexagonal_210\(9, 5, \.\.\.\)` style cell:\n\n```yaml\n(.*?)\n```",
        readme,
        flags=re.DOTALL,
    )
    assert match is not None

    h5_path = tmp_path / "wave.h5"
    mono = tmp_path / "2dm-5370.vasp"
    structure = tmp_path / "2dm-5370-7.34.vasp"
    out_dir = tmp_path / "valley_analysis"
    write_fixture(h5_path)
    write_simple_poscar(mono)
    write_simple_poscar(structure)

    yaml_text = match.group(1)
    # Material-specific benchmark names must not appear in example YAML blocks.
    # Physics descriptions in prose may mention tMoTe2/tZrSe2 as benchmarks.
    assert "tMoTe2" not in yaml_text
    yaml_text = yaml_text.replace("./wave.h5", str(h5_path))
    yaml_text = yaml_text.replace("./2dm-5370.vasp", str(mono))
    yaml_text = yaml_text.replace("./2dm-5370-7.34.vasp", str(structure))
    yaml_text = yaml_text.replace("./valley_analysis", str(out_dir))
    config_path = tmp_path / "readme_example.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    config = load_config(config_path)

    assert config.symmetry.operations.structure_file == structure
    assert config.symmetry.operations.backend == "spglib"
    assert config.symmetry.tolerance.symprec == pytest.approx(1.0e-3)
    assert config.symmetry.filters.allowed_orders == [2, 3, 4, 6]
    assert config.symmetry.filters.rotation_order == "auto"


def test_chinese_readme_uses_public_valley_vocabulary():
    readme = Path("README.zh.md").read_text(encoding="utf-8")
    assert "valley_subspaces:" in readme
    assert "valley_manifolds" not in readme
    assert "valley_sectors" not in readme
    assert "target_bands_vasp" not in readme
    assert "raw_valley_clean" not in readme
    assert "valley_separable_subspace" not in readme
    assert "relative_min_sector_distance" not in readme
    assert "overlap_cross_sector" not in readme
    assert "扇区" not in readme
    assert not re.search(r"(?<!\w)V[123](?!\.\d)", readme)
    assert "Valley subspaces" in readme
    assert "Valley subspace analysis" in readme
    assert "Two-valley subspace" not in readme
    assert "S_min:              目标谷子空间权重下界" in readme
    assert "min_concentration:" in readme
    assert "assigned_valleys:" in readme
    assert "`not_derived`" in readme
    assert "`unreliable`" in readme
    assert "valley-preserving irrep" in readme
    assert "谷保持子群" in readme
    assert "irrep_multiplicities" in readme
    assert "valley_resolved_irreps" in readme
    # Material names may appear in physics benchmark descriptions.
    assert "P321 No.150" not in readme
    assert "P3 No.143" not in readme
    assert "double-valued" in readme
    assert "valley_weights_adapted" in readme
    assert "assigned_valleys" in readme
    assert "`root_deviation_tol`、`D_valley_offdiag_tol` 和 `irrep_weight_tol` 是 numerical readiness thresholds" in readme
    assert "`strict`、`normal`、`loose`" in readme
    assert "header-only" in readme


# ---------------------------------------------------------------------------
# output-file label contracts
# ---------------------------------------------------------------------------

def test_output_files_manifest_has_labels_for_all_output_keys():
    """Every file-output key must have a human-readable label in
    OUTPUT_FILE_LABELS.  No key may silently fall back to title-case."""
    from valleyscope.reports.summary_report import OUTPUT_FILE_LABELS, _output_file_label

    keys = _analysis_output_file_keys()
    assert keys, "AST-derived key set is empty - check analysis_outputs module"

    missing = keys - set(OUTPUT_FILE_LABELS)
    assert not missing, (
        f"output keys missing from OUTPUT_FILE_LABELS: {sorted(missing)}"
    )

    for key in keys:
        label = _output_file_label(key)
        fallback = key.replace("_", " ").title()
        assert label != fallback, (
            f"label for {key!r} is the title-case fallback; "
            f"add it to OUTPUT_FILE_LABELS"
        )


def test_output_file_labels_no_stale_legacy_names():
    """No public output file label may contain stale legacy terms."""
    from valleyscope.reports.summary_report import OUTPUT_FILE_LABELS, _output_file_label

    keys = _analysis_output_file_keys()
    stale_terms = [
        "valley_sectors", "target_bands_vasp",
        "rotation_eigenvalues", "little_group_eigenvalues",
        "little_group_representations",
    ]
    for key in sorted(keys):
        label = _output_file_label(key).lower()
        for term in stale_terms:
            assert term not in label, (
                f"stale term '{term}' in label for {key!r}: {label}"
            )


# ---------------------------------------------------------------------------
# schema-doc coverage
# ---------------------------------------------------------------------------

def test_schema_doc_covers_public_outputs_and_reduced_ebr_statuses():
    """docs/schema.md names the required public files and reduced-EBR statuses."""
    schema_text = Path("docs/schema.md").read_text(encoding="utf-8")

    # Public user and downstream EBR files must be named.
    for name in [
        "valley_summary.txt", "valley_summary.json",
        "valley_ebr_export_bundle.json", "valley_reduced_ebr_mapping.json",
    ]:
        assert name in schema_text, f"docs/schema.md must mention {name}"

    # Reduced-EBR public statuses must be listed in table or prose.
    for status in [
        "not_evaluated", "missing_table", "blocked", "partial",
        "indeterminate_truncated", "solved_exact", "no_exact_solution",
    ]:
        assert status in schema_text, (
            f"docs/schema.md must document reduced-EBR status '{status}'"
        )
    assert "| `indeterminate_truncated` | int |" in schema_text

    # Reduced-EBR public field names must appear.
    for field in [
        "schema_version", "table_status", "reduced_ebr_input",
        "not_applicable", "not_provided", "loaded", "partial",
    ]:
        assert field in schema_text, (
            f"docs/schema.md must document reduced-EBR field '{field}'"
        )

    # Stale phrases must not appear.
    stale_phrases = [
        "enabled with valid table",   # output presence != valid table
    ]
    for phrase in stale_phrases:
        assert phrase not in schema_text, (
            f"docs/schema.md contains stale phrase: '{phrase}'"
        )


def test_summary_valley_resolved_irreps():
    """valley_summary includes compact valley_resolved_irreps from generic matching."""
    from valleyscope.reports.summary_report import _build_valley_resolved_irreps
    matching = {
        "matching_mode": "generic",
        "generic_matches_by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "matching_status": "matched",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {"-GM4": 1},
                    "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                    "valley_preserving_operation_ids": [0, 1, 2],
                    "hsp_little_group_operation_ids": [0, 1, 2],
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                    "diagnostic_only": False,
                },
                "Kp_valley": {
                    "matching_status": "diagnostic_only",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {},
                    "subspace_space_group": {"candidate_space_group_symbol": "P3"},
                    "valley_preserving_operation_ids": [0],
                    "hsp_little_group_operation_ids": [0, 1, 2],
                    "readiness_level": "usable_with_caution",
                    "workflow_path": "symmetry_adapted",
                    "diagnostic_only": True,
                    "reason": "spinor_convention_unverified",
                },
                "generic_valley": {
                    "matching_status": "not_applicable",
                    "matching_strategy": "bilbao_restricted_character",
                    "irrep_multiplicities": {},
                    "subspace_space_group": {
                        "candidate_space_group_symbol": "P3"
                    },
                    "valley_preserving_operation_ids": [0],
                    "hsp_little_group_operation_ids": [0],
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                    "diagnostic_only": False,
                    "source_hsp_membership": False,
                    "projected_hsp_classification": {
                        "classification": "generic",
                        "source_hsp_membership": False,
                    },
                    "reason": "generic_projected_subspace_k",
                },
            },
        },
    }
    r = _build_valley_resolved_irreps(matching)
    assert r["status"] == "ok"
    assert r["matched_count"] == 1
    assert r["diagnostic_count"] == 1
    assert r["non_source_count"] == 1
    assert r["rows"][0]["subspace_space_group"] == "P3"
    assert r["rows"][0]["subspace_hsp_little_group_operation_ids"] == [0, 1, 2]
    assert r["rows"][0]["hsp_little_group_operation_ids"] == [0, 1, 2]
    assert r["rows"][0]["matching_strategy"] == "bilbao_restricted_character"
    assert r["rows"][0]["irrep_multiplicities"] == {"-GM4": 1}
    assert r["rows"][0]["diagnostic_only"] is False
    assert r["rows"][1]["subspace_hsp_little_group_operation_ids"] == [0]
    assert r["rows"][1]["hsp_little_group_operation_ids"] == [0]
    assert r["rows"][1]["valley_preserving_operation_ids"] == [0]
    assert r["rows"][1]["diagnostic_only"] is True
    assert r["rows"][1]["reason"] == "spinor_convention_unverified"
    assert r["rows"][2]["matching_status"] == "not_applicable"
    assert r["rows"][2]["source_hsp_membership"] is False
    assert r["rows"][2]["projected_hsp_classification"] == "generic"
    assert "C2_like" not in json.dumps(r)
    assert "C3_like" not in json.dumps(r)


def test_valley_resolved_irreps_no_data():
    """valley_resolved_irreps reports no_generic_irrep_data when empty."""
    from valleyscope.reports.summary_report import _build_valley_resolved_irreps
    r = _build_valley_resolved_irreps({"generic_matches_by_kpoint": {}})
    assert r["status"] == "no_generic_irrep_data"
    assert r["matched_count"] == 0
    assert r["rows"] == []


# ---------------------------------------------------------------------------
# Real-fixture public output validation (tMoTe2 P321 P3/SG143)
# ---------------------------------------------------------------------------

_FIXTURE_SUMMARY = Path(
    __file__
).parent.parent / "real_tests" / "tMoTe2" / "output" / "valley_analysis_wave" / "valley_summary.json"


def _read_fixture_summary():
    """Read tMoTe2 fixture summary JSON, skipping if not available."""
    if not _FIXTURE_SUMMARY.exists():
        import pytest
        pytest.skip(f"tMoTe2 fixture output not found at {_FIXTURE_SUMMARY}")
    return json.loads(_FIXTURE_SUMMARY.read_text(encoding="utf-8"))


def test_tmote2_valley_resolved_irreps_compact_public_rows():
    """tMoTe2 valley_resolved_irreps: compact rows, subspace-first semantics."""
    s = _read_fixture_summary()
    resolved = s.get("valley_resolved_irreps")
    assert resolved is not None, "valley_resolved_irreps must be present"
    assert resolved["status"] == "ok"
    assert resolved["matching_mode"] == "generic"

    rows_by_kp = {}
    for row in resolved["rows"]:
        rows_by_kp.setdefault(row["kpoint"], {})[row["valley"]] = row

    # --- GammaM/K_valley: trusted, P3, subspace = [0,1,2] ---
    gm_k = rows_by_kp["GammaM"]["K_valley"]
    assert gm_k["subspace_space_group"] == "P3"
    assert gm_k["subspace_hsp_little_group_operation_ids"] == [0, 1, 2]
    assert gm_k["hsp_little_group_operation_ids"] == [0, 1, 2]
    assert gm_k["valley_preserving_operation_ids"] == [0, 1, 2]
    assert gm_k["matching_strategy"] == "bilbao_restricted_character"
    assert gm_k["matching_status"] == "matched"
    assert gm_k["readiness_level"] == "trusted"
    assert gm_k["workflow_path"] == "direct_qcut"
    assert gm_k["diagnostic_only"] is False
    assert gm_k["irrep_multiplicities"] == {"-GM4": 1}

    # --- KM/K_valley: trusted, P3 ---
    km_k = rows_by_kp["KM"]["K_valley"]
    assert km_k["subspace_space_group"] == "P3"
    assert km_k["subspace_hsp_little_group_operation_ids"] == [0, 1, 2]
    assert km_k["matching_status"] == "matched"
    assert km_k["readiness_level"] == "trusted"

    # --- MM/K_valley: matched to -M2 via table-driven restricted-character matching ---
    # G_k^(a)={E} with resolved source HSP label 'M' (P3/SG143); the single
    # source irrep -M2 matches uniquely on the identity character chi_a(E)=1.
    mm_k = rows_by_kp["MM"]["K_valley"]
    assert mm_k["subspace_space_group"] == "P3"
    assert mm_k["subspace_hsp_little_group_operation_ids"] == [0]
    assert mm_k["matching_status"] == "matched"
    assert mm_k["matching_strategy"] == "bilbao_restricted_character"
    assert mm_k["irrep_multiplicities"] == {"-M2": 1}
    assert mm_k["readiness_level"] == "trusted"


def test_tmote2_representation_records_subspace_first():
    """tMoTe2 representation_records: all sampled HSPs, subspace first."""
    s = _read_fixture_summary()
    rep = s.get("valley_projected_representations")
    assert rep is not None
    assert rep["grouped_record_count"] >= 6
    assert "MM" in rep["kpoint_labels"]
    assert "GammaM" in rep["kpoint_labels"]
    assert "KM" in rep["kpoint_labels"]

    for rec in rep["representation_records"]:
        kp = rec["kpoint"]
        subspace = rec.get("subspace_hsp_little_group_operation_ids", [])
        parent = rec.get("parent_hsp_little_group_operation_ids", [])
        sewing = rec.get("valley_sewing_operation_ids", [])
        hsp_lg = rec.get("hsp_little_group_operation_ids", [])

        # Public hsp_little_group = subspace, not parent
        assert hsp_lg == subspace, (
            f"{kp}/{rec['valley']}: hsp_little_group must alias subspace"
        )
        # subspace ⊆ parent
        assert set(subspace).issubset(set(parent)), (
            f"{kp}/{rec['valley']}: subspace not subset of parent"
        )
        # sewing ∩ subspace = {identity} only
        identity = 0
        sewing_non_id = [op for op in sewing if op != identity]
        for op in sewing_non_id:
            assert op not in subspace, (
                f"{kp}/{rec['valley']}: sewing op {op} in subspace!"
            )

    # MM records: identity-only, not blocked
    mm_recs = [r for r in rep["representation_records"] if r["kpoint"] == "MM"]
    assert len(mm_recs) == 2
    for mm in mm_recs:
        assert mm["subspace_hsp_little_group_operation_ids"] == [0]
        assert mm["parent_hsp_little_group_operation_ids"] == [0, 5]
        assert mm["valley_sewing_operation_ids"] == [5]
        assert mm["workflow_path"] != "blocked"
        assert mm["valley_preserving_operations"] == []
        assert "workflow_blocked" not in mm.get("blocking_reasons", [])

    # GammaM/KM records: trusted, eigenvalue data
    for kp in ("GammaM", "KM"):
        for rec in rep["representation_records"]:
            if rec["kpoint"] != kp:
                continue
            assert rec["subspace_hsp_little_group_operation_ids"] == [0, 1, 2]
            assert rec["parent_hsp_little_group_operation_ids"] == [0, 1, 2, 3, 4, 5]
            assert rec["valley_sewing_operation_ids"] == [3, 4, 5]


def test_tmote2_projected_source_hsp_coverage_and_tr_orbit_are_explicit():
    s = _read_fixture_summary()
    coverage = s["sampled_k_coverage"][
        "projected_subspace_hsp_coverage"
    ]
    for valley in ("K_valley", "Kp_valley"):
        row = coverage["by_valley"][valley]
        assert row["required_source_hsp_labels"] == ["GM", "K", "KA", "M"]
        assert row["covered_source_hsp_labels"] == ["GM", "K", "M"]
        assert row["missing_source_hsp_labels"] == ["KA"]
        assert row["trusted_matched_source_hsp_labels"] == ["GM", "K", "M"]
        assert row["complete"] is False
        assert row["ready_for_ebr_promotion"] is False

        decision = s["irrep_workflow_decisions"]["by_kpoint"]["MM"][valley]
        assert decision["workflow_path"] == "direct_qcut"
        assert decision["readiness_level"] == "trusted"
        assert decision["uses_symmetry_adapted_projector"] is False
        assert decision["direct_qcut_allowed"] is True

    assert s["valley_ebr_input_candidates"]["candidate_count"] == 6
    assert s["valley_ebr_input_candidates"]["blocked_count"] == 0
    mm_candidates = [
        row for row in s["valley_ebr_input_candidates"]["candidates"]
        if row.get("kpoint") == "MM"
    ]
    assert len(mm_candidates) == 2
    assert all(
        row["matched_irrep"] == "-M2"
        and row["readiness_level"] == "trusted"
        for row in mm_candidates
    )
    time_reversal = coverage["time_reversal"]
    assert time_reversal["status"] == "validated"
    assert time_reversal["theta_square"] == -1
    assert time_reversal["time_reversal_valley_mapping"] == {
        "K_valley": "Kp_valley", "Kp_valley": "K_valley",
    }
    assert len(time_reversal["valley_orbits"]) == 1
    orbit = time_reversal["valley_orbits"][0]
    assert orbit["members"] == ["K_valley", "Kp_valley"]
    assert orbit["status"] == "validated"
    assert orbit["full_unitary_source_hsp_labels"] == ["GM", "K", "KA", "M"]
    assert orbit["independent_time_reversal_hsp_labels"] == ["GM", "K", "M"]
    assert orbit["grey_bns_number"] == "143.2"
    sewing = time_reversal["antiunitary_sewing_evidence"]
    assert sewing["status"] == "blocked"
    assert sewing["time_reversal_kpoint_mapping"] == {
        "GammaM": "GammaM", "MM": "MM",
    }
    assert "ambiguous_time_reversal_kpoint_partner:KM:[]" in sewing[
        "blockers"
    ]
    rows = {row["source_kpoint"]: row for row in sewing["rows"]}
    assert set(rows) == {"GammaM", "MM"}
    assert all(row["status"] == "validated" for row in rows.values())
    assert rows["GammaM"]["mapping_miss_count"] == 0
    assert rows["MM"]["mapping_miss_count"] == 0
    assert rows["GammaM"]["theta_square_residual"] < 2e-7
    assert rows["MM"]["theta_square_residual"] < 3e-8
    for row in rows.values():
        for covariance in row["projector_covariance"].values():
            source = covariance["source_projector_provenance"]
            target = covariance["target_projector_provenance"]
            assert {
                key: source[key]
                for key in ("workflow_path", "projector_kind")
            } == {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }
            assert {
                key: target[key]
                for key in ("workflow_path", "projector_kind")
            } == {
                "workflow_path": "direct_qcut",
                "projector_kind": "fixed_center_seed",
            }
            assert source["projector_shape"] == [2, 2]
            assert target["projector_shape"] == [2, 2]
            assert source["projector_fingerprint"].startswith("sha256:")
            assert target["projector_fingerprint"].startswith("sha256:")

    export = s["valley_ebr_export_bundle"]
    assert export["bundle_count"] == 3
    assert s["valley_ebr_export_bundle"]["schema_version"] == "1.8.0"
    by_kind_and_valley = {
        (bundle["problem_kind"], bundle["valley"]): bundle
        for bundle in export["bundles"]
    }
    joint = by_kind_and_valley[("valley_orbit_reduced_ebr", "")]
    assert joint["valley_orbit"] == ["K_valley", "Kp_valley"]
    for valley, expected_ka in (
        ("K_valley", "-KA5"), ("Kp_valley", "-KA6")
    ):
        unitary = by_kind_and_valley[
            ("unitary_valley_reduced_ebr", valley)
        ]
        assert unitary["expected_hsps"] == ["GM", "K", "KA", "M"]
        inferred = unitary[
            "unitary_irrep_completion_records_by_hsp"
        ]["KA"][0]
        assert inferred["irrep"] == expected_ka
        assert inferred["completion_kind"] == "inferred_by_time_reversal"
        assert "sampled_kpoint" not in inferred
        assert inferred["evidence_valley"] != valley
        assert inferred["evidence_sampled_kpoint"] == "KM"


def test_tmote2_public_output_no_cn_like_and_production_no_material_names():
    """tMoTe2 public summary avoids Cn-like labels; production code has no material branch."""
    s = _read_fixture_summary()
    raw = json.dumps(s)

    for cn in ("C2_like", "C3_like", "C4_like", "C6_like"):
        assert cn not in raw, f"{cn} must not appear in public summary output"

    # Material labels are allowed in validation fixtures, but never in
    # production logic or public schema names.
    production_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("valleyscope").rglob("*.py")
    )
    for material in ("tMoTe2", "tZrSe2"):
        assert material not in production_text


_CENTERED_FIXTURE_SUMMARY = Path(
    __file__
).parent.parent / "real_tests" / "tZrSe2" / "output" / "valley_analysis" / "valley_summary.json"


def _read_centered_fixture_summary():
    if not _CENTERED_FIXTURE_SUMMARY.exists():
        pytest.skip(
            f"centered fixture output not found at {_CENTERED_FIXTURE_SUMMARY}"
        )
    return json.loads(_CENTERED_FIXTURE_SUMMARY.read_text(encoding="utf-8"))


def test_centered_fixture_projected_hsp_coverage_is_per_valley():
    s = _read_centered_fixture_summary()
    coverage = s["sampled_k_coverage"][
        "projected_subspace_hsp_coverage"
    ]["by_valley"]
    expected = {
        "M1_valley": (["GM", "V"], ["Y"]),
        "M2_valley": (["GM", "V"], ["Y"]),
        "M3_valley": (["GM", "Y"], ["V"]),
    }
    for valley, (covered, missing) in expected.items():
        row = coverage[valley]
        assert row["required_source_hsp_labels"] == ["GM", "V", "Y"]
        assert row["covered_source_hsp_labels"] == covered
        assert row["missing_source_hsp_labels"] == missing
        assert row["complete"] is False
        assert row["ready_for_ebr_promotion"] is False
        assert [
            item["source_hsp_label"]
            for item in row["missing_source_hsp_representatives"]
        ] == missing
        assert all(
            len(item["inverse_parent_k_frac"]) == 3
            for item in row["missing_source_hsp_representatives"]
        )

    time_reversal = s["sampled_k_coverage"][
        "projected_subspace_hsp_coverage"
    ]["time_reversal"]
    assert time_reversal["status"] == "blocked"
    assert time_reversal["theta_square"] == -1
    assert time_reversal["time_reversal_valley_mapping"] == {
        "M1_valley": "M1_valley",
        "M2_valley": "M2_valley",
        "M3_valley": "M3_valley",
    }
    assert all(
        orbit["status"] == "blocked"
        and "antiunitary_corepresentation_sewing_not_validated"
        in orbit["blockers"]
        for orbit in time_reversal["valley_orbits"]
    )
    sewing = time_reversal["antiunitary_sewing_evidence"]
    assert sewing["status"] == "blocked"
    assert "spinor_convention_unverified_for_time_reversal" in sewing[
        "blockers"
    ]
    assert "ambiguous_time_reversal_kpoint_partner:KM:[]" in sewing[
        "blockers"
    ]
    assert "trusted_projector_workflow_blocked:GammaM:M1_valley" in sewing[
        "blockers"
    ]
    assert "time_reversal_trusted_projectors_missing:GammaM:GammaM" in sewing[
        "blockers"
    ]


def test_centered_fixture_star_and_generic_rows_do_not_become_false_blockers():
    s = _read_centered_fixture_summary()
    matches = s["valley_irrep_matching"]["generic_matches_by_kpoint"]

    m1 = matches["MM"]["M1_valley"]["projected_hsp_classification"]
    assert m1["classification"] == "star_equivalent"
    assert m1["source_hsp_label"] == "V"
    assert m1["standard_operation_witness"]["table_index"] == 2
    assert m1["representation_transport_status"] == "validated"

    for valley in ("M1_valley", "M2_valley", "M3_valley"):
        generic = matches["KM"][valley]
        assert generic["matching_status"] == "not_applicable"
        assert generic["diagnostic_only"] is False
        assert generic["reason"] == "generic_projected_subspace_k"
        assert generic["projected_hsp_classification"]["classification"] == "generic"

    candidates = s["valley_ebr_input_candidates"]
    assert candidates["candidate_count"] == 0
    assert candidates["blocked_count"] == 6
    assert candidates["non_source_count"] == 3
    assert len(candidates["non_source_rows"]) == 3
    assert s["valley_ebr_export_bundle"]["bundle_count"] == 0


# ---------------------------------------------------------------------------
# Auto-canonical reduced EBR provenance regression (tMoTe2)
# ---------------------------------------------------------------------------

_FIXTURE_REDUCED_EBR = Path(
    __file__
).parent.parent / "real_tests" / "tMoTe2" / "output" / "valley_analysis_wave" / "valley_reduced_ebr_mapping.json"


def _read_fixture_reduced_ebr():
    """Read tMoTe2 reduced EBR mapping JSON, skipping if not available."""
    if not _FIXTURE_REDUCED_EBR.exists():
        pytest.skip(f"tMoTe2 reduced EBR fixture not found at {_FIXTURE_REDUCED_EBR}")
    return json.loads(_FIXTURE_REDUCED_EBR.read_text(encoding="utf-8"))


def test_tmote2_reduced_ebr_auto_canonical_provenance():
    """The TR valley orbit uses the reviewed type-II grey source."""
    r = _read_fixture_reduced_ebr()

    # Top-level status
    assert r["status"] == "no_exact_solution"
    assert r["schema_version"] == "1.9.0"
    assert r["table_status"] == "loaded"

    # reduced_ebr_input self-auditing
    inp = r.get("reduced_ebr_input", {})
    assert inp["source"] == "auto_unitary_and_time_reversal"
    assert inp["spinful"] is True
    assert inp["reduced_table_validation_candidate_bundle_count"] == 3
    assert inp["final_reduced_ebr_result_count"] == 3
    assert inp["final_mapping_excluded_bundle_count"] == 0

    # auto_canonical_bundles
    bundles = r.get("auto_canonical_bundles", [])
    assert len(bundles) == 3
    for b in bundles:
        assert b["sg_number"] == 143
        assert b["status"] == "no_exact_solution"
        assert b["table_status"] == "loaded"

    # Solutions
    solutions = r.get("solutions", [])
    assert len(solutions) == 3
    joint = next(
        sol for sol in solutions
        if sol["problem_kind"] == "valley_orbit_reduced_ebr"
    )
    assert joint["valley_orbit"] == ["K_valley", "Kp_valley"]
    assert joint["classification"] == "in_integer_span_no_nonnegative_witness"
    assert joint["nonnegative_solution_status"] == "no_nonnegative_solution"
    unitary = {
        sol["valley"]: sol for sol in solutions
        if sol["problem_kind"] == "unitary_valley_reduced_ebr"
    }
    assert set(unitary) == {"K_valley", "Kp_valley"}
    assert all(
        sol["classification"] == "outside_integer_span"
        and sol["expected_hsps"] == ["GM", "K", "KA", "M"]
        and sol["table_provenance"]["source"] == "auto_canonical"
        for sol in unitary.values()
    )

    for sol in [joint]:

        # table_provenance injected per solution
        tp = sol.get("table_provenance", {})
        assert tp["source"] == "auto_time_reversal_grey"
        assert tp["space_group_number"] == 143
        assert tp["spinful"] is True
        assert tp["data_source"] == "irreptables"
        assert tp["package"] == "irreptables"
        assert isinstance(tp.get("package_version"), str) and tp["package_version"]
        assert tp["valleyscope_reduction"] == "sampled_hsp_valley_preserving"
        assert tp["source_basis_count"] > 0
        assert tp["reduction_basis_count"] > 0
        assert tp["source_basis_count"] > tp["reduction_basis_count"]
        assert tp["filtered_zero_vector_ebr_count"] == 0
        assert tp["filtered_zero_vector_ebrs"] == []
        assert tp["dropped_source_row_count"] == 0
        assert tp["dropped_source_rows"] == []
        assert tp["time_reversal_grey_bns_number"] == "143.2"
        assert tp["time_reversal_source"] == (
            "irreptables_type_ii_grey_group"
        )

        # subspace group
        assert sol["subspace_group_candidate"] == "P3"

        assert sol["time_reversal"]["grey_bns_number"] == "143.2"
        assert set(sol["unitary_valley_irreps"]) == {
            "K_valley", "Kp_valley",
        }
        assert sol["unitary_valley_irreps"]["K_valley"]["M"] == {
            "-M2": 1,
        }
        assert sol["unitary_valley_irreps"]["Kp_valley"]["M"] == {
            "-M2": 1,
        }
