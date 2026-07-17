import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.workflows.analyze_hsp import analyze_hsp
from valleyscope.reports.analysis_outputs import write_analysis_outputs

from tests.helpers_io_workflow import write_fixture, write_config


def test_default_standard_profile_writes_only_public_outputs(tmp_path):
    """Default output.profile=standard emits only public files."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"].pop("profile", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    # Public outputs always present.
    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert outputs.get("valley_weights_csv") and outputs["valley_weights_csv"].exists()

    # Debug/detail files must NOT exist with standard profile.
    debug_files = [
        "valley_subspace.json", "symmetry_report.json", "symmetry_eigenvalues.csv",
        "diagnostics.h5", "valley_basis_transform.h5",
        "projector_symmetry_report.json", "symmetry_adapted_valley_analysis.json",
        "target_subspace_closure.json", "hsp_star_conjugation.json",
        "hsp_star_derived_characters.json", "subspace_representation_quality.json",
        "irrep_workflow_decisions.json", "valley_irrep_matching.json",
        "valley_ebr_input_candidates.json", "valley_ebr_problem_instances.json",
        "folded_center_report.json", "sampled_k_coverage.json",
    ]
    for fname in debug_files:
        assert not (out_dir / fname).exists(), f"{fname} must not exist in standard profile"

    # Summary mentions debug suppression.
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "Debug/detail outputs suppressed" in summary_text
    assert "output.profile: debug" in summary_text

    summary_json = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary_json.get("output_profile") == "standard"
    assert "valley_projected_representations" in summary_json
    assert "valley_irrep_matching" not in summary_json


def test_debug_profile_writes_all_detailed_files(tmp_path):
    """output.profile=debug emits the full current detailed file set."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "debug"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    # Public outputs.
    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert outputs["valley_weights_csv"].exists()
    # Detailed files.
    assert outputs["valley_subspace_json"].exists()
    assert outputs["symmetry_report_json"].exists()
    assert outputs["diagnostics_h5"].exists()
    # Summary must NOT mention suppression.
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "Debug/detail outputs suppressed" not in summary_text

    summary_json = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary_json.get("output_profile") == "debug"


def test_write_detailed_files_false_maps_to_standard_with_warning(tmp_path):
    """Legacy write_detailed_files: false maps to profile=standard."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"].pop("profile", None)
    raw["output"]["write_detailed_files"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="write_detailed_files"):
        outputs = analyze_hsp(config_path)

    assert outputs["valley_summary_txt"].exists()
    assert not (out_dir / "valley_subspace.json").exists()
    assert not (out_dir / "diagnostics.h5").exists()


def test_invalid_output_profile_rejected(tmp_path):
    """Invalid output.profile raises ValueError."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "invalid_profile"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="output.profile must be one of"):
        load_config(config_path)


def test_no_material_specific_strings_in_production_code():
    """Verify no real material names appear in valleyscope/ production modules.

    tMoTe2, tZrSe2, and future real materials are validation examples and
    regression fixtures only.  They must not appear in program logic, output
    strings, config keys, or file paths inside valleyscope/.

    This test does not guard docs/benchmarks/ or real_tests/.
    """
    valleyscope_dir = Path("valleyscope")
    forbidden = ["tMoTe2", "tZrSe2", "MoTe2", "ZrSe2"]
    failures: list[str] = []
    for py_file in sorted(valleyscope_dir.rglob("*.py")):
        lines = py_file.read_text(encoding="utf-8").split("\n")
        for i, line in enumerate(lines, start=1):
            for name in forbidden:
                if name in line:
                    failures.append(f"{py_file}:{i}: {line.strip()[:120]}")
    if failures:
        msg = (
            "Material names found in valleyscope/ production code:\n"
            + "\n".join(failures)
            + "\n\nReal materials are validation examples only; "
            "they must not appear in program logic, output strings, "
            "or config paths."
        )
        raise AssertionError(msg)


def test_config_profiles_accepted(tmp_path):
    """Both 'standard' and 'debug' profiles are accepted."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    for profile in ["standard", "debug"]:
        config = {
            "input": {"wavefunction_h5": str(h5_path)},
            "analysis": {"kpoints": ["GammaM"], "iband": [101]},
            "monolayer_lattices": {
                "default": {"reciprocal_cart": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]}
            },
            "valley_centers": {
                "coordinate_mode": "cart",
                "centers": [
                    {"name": "K", "cart": [0.0, 0.0, 0.0]},
                    {"name": "Kp", "cart": [5.0, 0.0, 0.0]},
                ],
            },
            "valley_subspaces": [
                {"name": "K_valley", "centers": ["K"]},
                {"name": "Kp_valley", "centers": ["Kp"]},
            ],
            "output": {"directory": str(out_dir), "profile": profile},
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        loaded = load_config(config_path)
        assert loaded.output.profile == profile


def test_standard_profile_always_writes_summary_even_with_flags_false(tmp_path):
    """Standard profile writes valley_summary.txt/json even when write flags are false."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"]["write_summary_txt"] = False
    raw["output"]["write_summary_json"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    # Main user entry must always be present in standard profile.
    assert outputs["valley_summary_txt"].exists(), (
        "valley_summary.txt must be written in standard profile even with write_summary_txt=false"
    )
    assert outputs["valley_summary_json"].exists(), (
        "valley_summary.json must be written in standard profile even with write_summary_json=false"
    )


def test_write_analysis_outputs_creates_standard_summary_directory(tmp_path):
    """Report writer creates output.directory for standard profile summaries."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"]["write_csv"] = False
    raw["output"]["write_summary_txt"] = False
    raw["output"]["write_summary_json"] = False
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(config_path)

    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    assert not out_dir.exists()

    outputs = write_analysis_outputs(
        config=config,
        qcut=0.1,
        weight_rows=[],
        sector_names=[],
        subspace_payload={},
        symmetry_payload={},
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
    )

    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "valley_summary.json",
        "valley_summary.txt",
    ]



_STANDARD_PUBLIC_FILES = frozenset({
    "valley_summary.txt",
    "valley_summary.json",
    "valley_weights.csv",
    "valley_ebr_export_bundle.json",
    "valley_reduced_ebr_mapping.json",
})

_DEBUG_ONLY_FILES = frozenset({
    "valley_subspace.json",
    "symmetry_report.json",
    "symmetry_eigenvalues.csv",
    "diagnostics.h5",
    "valley_basis_transform.h5",
    "projector_symmetry_report.json",
    "symmetry_adapted_valley_analysis.json",
    "target_subspace_closure.json",
    "hsp_star_conjugation.json",
    "hsp_star_derived_characters.json",
    "subspace_representation_quality.json",
    "irrep_workflow_decisions.json",
    "valley_irrep_matching.json",
    "valley_ebr_input_candidates.json",
    "valley_ebr_problem_instances.json",
    "folded_center_report.json",
    "sampled_k_coverage.json",
})


def _write_minimal_outputs(config, **payloads):
    return write_analysis_outputs(
        config=config,
        qcut=0.1,
        weight_rows=[],
        sector_names=[],
        subspace_payload={},
        symmetry_payload={},
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
        **payloads,
    )


def test_standard_rerun_removes_stale_debug_artifacts(tmp_path):
    """A debug-to-standard rerun leaves no managed debug/detail files."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    analyze_hsp(config_path)
    assert (out_dir / "diagnostics.h5").exists()
    assert (out_dir / "valley_subspace.json").exists()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    analyze_hsp(config_path)

    assert not ({path.name for path in out_dir.iterdir()} & _DEBUG_ONLY_FILES)


def test_optional_public_outputs_removed_when_payload_becomes_absent(tmp_path):
    """Disabled/absent EBR payloads remove files produced by the prior run."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["analysis"]["reduced_ebr"] = {"enabled": True}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    _write_minimal_outputs(
        load_config(config_path),
        ebr_export_bundle={"status": "has_bundles"},
        reduced_ebr_mapping={"status": "solved_exact"},
    )
    assert (out_dir / "valley_ebr_export_bundle.json").exists()
    assert (out_dir / "valley_reduced_ebr_mapping.json").exists()

    raw["analysis"]["reduced_ebr"] = {"enabled": False}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    _write_minimal_outputs(load_config(config_path))

    assert not (out_dir / "valley_ebr_export_bundle.json").exists()
    assert not (out_dir / "valley_reduced_ebr_mapping.json").exists()


def test_managed_output_cleanup_preserves_unrelated_user_file(tmp_path):
    """Cleanup is allowlisted and never removes an unrelated user file."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out_dir.mkdir()
    user_file = out_dir / "user-notes.txt"
    user_file.write_text("preserve me", encoding="utf-8")

    _write_minimal_outputs(load_config(config_path))

    assert user_file.read_text(encoding="utf-8") == "preserve me"


def test_managed_output_cleanup_preserves_configured_input_file(tmp_path):
    """A managed-looking filename is never removed when it is an input path."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    h5_path = out_dir / "diagnostics.h5"
    config_path = tmp_path / "config.yaml"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    analyze_hsp(config_path)

    assert h5_path.exists()


def test_summary_output_files_match_current_managed_files_after_rerun(tmp_path):
    """Summary paths describe every and only managed file from the current run."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    analyze_hsp(config_path)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    outputs = analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    declared = {Path(path).name for path in summary["output_files"].values()}
    managed = {
        path.name for path in out_dir.iterdir()
        if path.name in _STANDARD_PUBLIC_FILES | _DEBUG_ONLY_FILES
    }
    assert declared == managed


def test_standard_profile_output_files_are_only_public_set(tmp_path):
    """Standard profile writes only the contracted public output files."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    all_written = {p.name for p in out_dir.iterdir() if p.is_file()}
    # Every written file must be in the public set.
    unexpected = all_written - _STANDARD_PUBLIC_FILES
    assert not unexpected, f"Standard profile wrote non-public files: {unexpected}"
    # No debug-only file may exist.
    debug_found = all_written & _DEBUG_ONLY_FILES
    assert not debug_found, f"Standard profile wrote debug/detail files: {debug_found}"
    # Core public files must be present.
    assert "valley_summary.txt" in all_written
    assert "valley_summary.json" in all_written
    assert "valley_weights.csv" in all_written


def test_standard_profile_summary_output_files_excludes_debug_keys(tmp_path):
    """valley_summary.json output_files must not list debug/detail files in standard profile."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    output_keys = set(summary.get("output_files", {}).keys())
    # Must include the public files that were actually written.
    assert "valley_summary_txt" in output_keys
    assert "valley_summary_json" in output_keys
    assert "valley_weights_csv" in output_keys
    # Must not include debug-only file keys.
    debug_keys_in_summary = output_keys & {
        "valley_subspace_json", "symmetry_report_json", "symmetry_eigenvalues_csv",
        "diagnostics_h5", "valley_basis_transform_h5",
        "projector_symmetry_report_json", "symmetry_adapted_valley_analysis_json",
        "target_subspace_closure_json", "hsp_star_conjugation_json",
        "hsp_star_derived_characters_json", "subspace_representation_quality_json",
        "irrep_workflow_decisions_json", "valley_irrep_matching_json",
        "valley_ebr_input_candidates_json", "valley_ebr_problem_instances_json",
        "folded_center_report_json", "sampled_k_coverage_json",
    }
    assert not debug_keys_in_summary, (
        f"Standard profile summary output_files lists debug/detail keys: {debug_keys_in_summary}"
    )


def test_standard_profile_ebr_export_bundle_present_when_payload_exists():
    """valley_ebr_export_bundle.json is written in standard profile when payload exists."""
    from valleyscope.reports.analysis_outputs import write_analysis_outputs
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        h5_path = out_dir / "wf.h5"
        write_fixture(h5_path)
        config_path = out_dir / "cfg.yaml"
        write_config(config_path, h5_path, out_dir)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["output"]["profile"] = "standard"
        raw["output"].pop("write_detailed_files", None)
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        config = load_config(config_path)

        ebr_bundle = {"status": "ready_for_reduced_table_validation",
                      "bundle_count": 1, "excluded_count": 0,
                      "schema_version": "1.5.0",
                      "bundles": [], "excluded_instances": []}
        outputs = write_analysis_outputs(
            config=config, qcut=0.5, weight_rows=[], sector_names=["K_valley"],
            subspace_payload={"kpoints": {}},
            symmetry_payload={"status": "skipped", "reason": "test",
                              "detected_operations": [], "candidate_rotations": [],
                              "little_group_check": {"status": "not_run"},
                              "valley_preservation_check": {"status": "not_run"}},
            symmetry_rows=[], projectors_by_kpoint={}, qcut_scan_payload={},
            symmetry_representation_payload={}, basis_transforms={},
            ebr_export_bundle=ebr_bundle,
        )
        assert outputs["valley_ebr_export_bundle_json"].exists()
        assert (out_dir / "valley_ebr_export_bundle.json").exists()


def test_standard_profile_no_ebr_export_bundle_when_payload_none():
    """valley_ebr_export_bundle.json is NOT written when no EBR payload exists."""
    from valleyscope.reports.analysis_outputs import write_analysis_outputs
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        h5_path = out_dir / "wf.h5"
        write_fixture(h5_path)
        config_path = out_dir / "cfg.yaml"
        write_config(config_path, h5_path, out_dir)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["output"]["profile"] = "standard"
        raw["output"].pop("write_detailed_files", None)
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        config = load_config(config_path)

        outputs = write_analysis_outputs(
            config=config, qcut=0.5, weight_rows=[], sector_names=["K_valley"],
            subspace_payload={"kpoints": {}},
            symmetry_payload={"status": "skipped", "reason": "test",
                              "detected_operations": [], "candidate_rotations": [],
                              "little_group_check": {"status": "not_run"},
                              "valley_preservation_check": {"status": "not_run"}},
            symmetry_rows=[], projectors_by_kpoint={}, qcut_scan_payload={},
            symmetry_representation_payload={}, basis_transforms={},
            ebr_export_bundle=None,
        )
        assert "valley_ebr_export_bundle_json" not in outputs
        assert not (out_dir / "valley_ebr_export_bundle.json").exists()


def test_debug_profile_writes_all_expected_detail_files(tmp_path):
    """Debug profile writes public files AND all debug/detail files."""
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "debug"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    all_written = {p.name for p in out_dir.iterdir() if p.is_file()}
    # Public files must be present.
    assert "valley_summary.txt" in all_written
    assert "valley_summary.json" in all_written
    # Debug files that are always written with this fixture must be present.
    assert "diagnostics.h5" in all_written
    assert "valley_subspace.json" in all_written
    assert "symmetry_report.json" in all_written
    # Summary must NOT mention suppression.
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "Debug/detail outputs suppressed" not in summary_text
