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
                                "subspace_group_candidate": "C2_like",
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
        "legacy_tables_implemented": ["spinful_C3", "spinful_C2"],
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
    assert "legacy phase tables: spinful_C3, spinful_C2" in text
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
        "irrep_matching": {
            "status": "table_mapping_complete",
            "table_source": "irreptables",
            "label_matching": "matched",
            "character_matching_status": "matched",
            "irrep_results_by_kpoint": {
                "KM": {
                    "status": "matched",
                    "table_kpoint_label": "K",
                    "irrep_multiplicities": {"-K5": 1, "-K6": 1},
                    "failure_reasons": [],
                }
            },
        },
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
    assert "irrep matching: matched" in text
    assert "KM: -K5 x 1, -K6 x 1" in text


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
    assert "irrep_results_by_kpoint" in readme
    assert "irrep_multiplicities" in readme
    assert "state_irrep_results" in readme
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
    assert "irrep_results_by_kpoint" in readme
    assert "irrep_multiplicities" in readme
    assert "state_irrep_results" in readme
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
        "not_evaluated", "missing_table", "solved_exact", "no_exact_solution",
    ]:
        assert status in schema_text, (
            f"docs/schema.md must document reduced-EBR status '{status}'"
        )

    # Reduced-EBR public field names must appear.
    for field in [
        "mapping_status", "reduced_ebr_decomposition_status",
        "table_status", "not_applicable", "not_provided", "loaded",
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
            },
        },
    }
    r = _build_valley_resolved_irreps(matching)
    assert r["status"] == "ok"
    assert r["matched_count"] == 1
    assert r["diagnostic_count"] == 1
    assert r["rows"][0]["subspace_space_group"] == "P3"
    assert r["rows"][0]["matching_strategy"] == "bilbao_restricted_character"
    assert r["rows"][0]["irrep_multiplicities"] == {"-GM4": 1}
    assert r["rows"][0]["diagnostic_only"] is False
    assert r["rows"][1]["diagnostic_only"] is True
    assert r["rows"][1]["reason"] == "spinor_convention_unverified"
    assert "C2_like" not in json.dumps(r)
    assert "C3_like" not in json.dumps(r)


def test_valley_resolved_irreps_no_data():
    """valley_resolved_irreps reports no_generic_irrep_data when empty."""
    from valleyscope.reports.summary_report import _build_valley_resolved_irreps
    r = _build_valley_resolved_irreps({"generic_matches_by_kpoint": {}})
    assert r["status"] == "no_generic_irrep_data"
    assert r["matched_count"] == 0
    assert r["rows"] == []
