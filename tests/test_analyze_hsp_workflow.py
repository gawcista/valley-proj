"""Analyze-HSP workflow integration tests: CLI, HDF5 diagnostics, qcut screen,
output-writer integration, and workflow-level subspace status contracts."""

import csv
import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from valleyscope.cli import main as cli_main
from valleyscope.io.config import load_config
from valleyscope.io.h5_reader import read_wavefunction_h5
from valleyscope.projection.sector_projectors import SectorProjectors
from valleyscope.workflows.analyze_hsp import analyze_hsp

from tests.helpers_io_workflow import (
    write_fixture,
    write_config,
    write_simple_poscar,
    write_square_poscar,
)


def test_analyze_hsp_writes_csv_json_and_diagnostics_h5(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    outputs = analyze_hsp(config_path)

    assert outputs["valley_weights_csv"].exists()
    assert outputs["valley_subspace_json"].exists()
    assert outputs["valley_summary_txt"].exists()
    assert outputs["valley_summary_json"].exists()
    assert outputs["diagnostics_h5"].exists()
    report = json.loads(outputs["symmetry_report_json"].read_text(encoding="utf-8"))
    assert report["status"] == "skipped"
    assert "symmetry.operations.structure_file" in report["reason"]
    csv_text = outputs["valley_weights_csv"].read_text(encoding="utf-8")
    csv_header = csv_text.splitlines()[0].split(",")
    assert csv_header == [
        "kpoint",
        "band_vasp",
        "energy_eV",
        "K_valley",
        "Kp_valley",
        "W_val",
        "P_v",
        "eta",
        "W_overlap",
        "W_res",
        "center_K",
        "center_Kp",
    ]
    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    weight = subspace["kpoints"]["GammaM"]["weights"][0]
    assert {
        "band_vasp", "sector_weights", "W_val", "P_v", "eta", "W_overlap", "W_res",
    }.issubset(set(weight))
    assert weight.get("analysis_level") == "raw_state"
    assert weight.get("valley_status") == "raw_valley_clean"
    with h5py.File(outputs["diagnostics_h5"], "r") as h5:
        projector_group = h5["projectors"]["GammaM"]
        assert "overlap_mask" in projector_group
        assert "ambiguous_mask" not in projector_group
        scan_group = h5["qcut_scan"]["GammaM"]
        assert "W_overlap" in scan_group
        assert "W_res" in scan_group
        assert "overlap_count" in scan_group
        assert "ambiguous_weight" not in scan_group
        assert "ambiguous_count" not in scan_group
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert {
        "input",
        "valley_projection_summary",
        "valley_subspace_analysis",
        "symmetry_analysis",
        "symmetry_eigenvalues",
        "warnings",
        "output_files",
        "legend",
    } <= set(summary)
    assert "valley_adapted_subspace" not in summary
    assert "rotation_eigenvalues" not in summary
    assert "allowed_valley_preserving_rotations" not in summary
    assert "two_valley_subspace" not in summary
    subspace_rows = summary["valley_subspace_analysis"]
    assert subspace_rows
    assert subspace_rows[0].get("status") in {"clean", "approx", "mixed", "not_derived", "unreliable", "n/a"}
    assert "derived_score" not in summary["valley_projection_summary"][0]
    assert "polarization_score" not in summary["valley_projection_summary"][0]
    assert "valley_status" not in summary["valley_projection_summary"][0]
    assert "derived_score" not in subspace_rows[0]
    assert "polarization_score" not in subspace_rows[0]
    assert "valley_status" not in subspace_rows[0]
    assert not any("target subspace is not valley-derived" in w for w in summary["warnings"])


def test_cli_prints_human_readable_summary(tmp_path, capsys):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    assert cli_main(["analyze-hsp", str(config_path)]) == 0

    out = capsys.readouterr().out
    assert "Input" in out
    assert "Valley projection summary" in out
    assert "W_val" in out
    assert "P_v" in out
    assert "derived" not in out
    assert " pol " not in out
    assert "status" in out
    assert "W_overlap" in out
    assert "W_res" in out
    assert "Valley subspace analysis" in out
    assert "Two-valley subspace" not in out
    assert "S_min:      minimum target-valley-subspace weight" in out
    assert "eta_adapted: signed valley polarization" in out
    assert "Symmetry analysis" in out
    assert "Symmetry diagnostics" not in out
    assert "Allowed valley-preserving rotations" not in out
    assert "Rotation eigenvalues" not in out
    assert "Valley subspaces" in out
    assert "K_valley" in out
    assert "K_sector" not in out
    assert "qcut mode: absolute" in out
    assert "qcut value" in out


def test_analyze_hsp_collects_overlap_warnings_without_raw_warning_output(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["projection"]["qcut_Ainv"] = 5.1
    raw["projection"]["qcut_scan"] = [5.1]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        outputs = analyze_hsp(config_path)

    assert not [item for item in captured if "overlap across valleys" in str(item.message)]
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert any("overlap across valleys" in item for item in summary["warnings"])


def test_analyze_hsp_writes_valley_subspace_analysis_transform_for_degenerate_pair(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3) * 10.0
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"
        kp["frac"] = np.array([0.0, 0.0, 0.0])
        kp["cart"] = np.array([0.0, 0.0, 0.0])
        kp["g_vectors_frac"] = np.array([[0, 0, 0], [1, 0, 0]])
        kp["g_vectors_cart"] = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        kp["coefficients"] = np.array(
            [
                [[inv_sqrt2 + 0.0j, inv_sqrt2 + 0.0j]],
                [[inv_sqrt2 + 0.0j, -inv_sqrt2 + 0.0j]],
            ]
        )
        kp["energies_eV"] = np.array([0.1, 0.1002])
        kp["band_indices_vasp"] = np.array([101, 102])
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["analysis"]["iband"] = [101, 102]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    with h5py.File(outputs["valley_basis_transform_h5"], "r") as h5:
        assert "GammaM" in h5
        assert h5["GammaM/transform"].shape == (2, 2)
        assert h5["GammaM/eta"].shape == (2,)
        assert h5["GammaM/v_matrix"].shape == (2, 2)
        assert [value.decode() for value in h5["GammaM/sectors"][()]] == ["K_valley", "Kp_valley"]
        assert [value.decode() for value in h5["GammaM/valleys"][()]] == ["K_valley", "Kp_valley"]
    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    diagnostic = subspace["kpoints"]["GammaM"]["valley_adapted_subspace"]
    assert diagnostic["status"] == "valley_separable"
    assert diagnostic["valid_valley_subspace"] is True
    assert diagnostic["s_min"] == pytest.approx(1.0)
    assert diagnostic["s_max"] == pytest.approx(1.0)
    projector_quality = diagnostic["projector_quality"]
    assert projector_quality["expected_rank"] == 1
    assert projector_quality["per_valley"]["K_valley"]["rank_estimate"] == 1
    assert projector_quality["per_valley"]["Kp_valley"]["rank_estimate"] == 1
    assert projector_quality["sum_projector"]["identity_deviation_fro"] == pytest.approx(0.0)
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary["valley_subspace_analysis"][0]["status"] == "clean"


def test_multivalley_subspace_status_uses_pv_thresholds_for_concentration():
    from valleyscope.workflows.analyze_hsp import _add_valley_subspace_diagnostic

    coefficients = np.zeros((2, 1, 3), dtype=np.complex128)
    coefficients[0, 0, 0] = 1.0
    coefficients[1, 0, 1] = np.sqrt(0.92)
    coefficients[1, 0, 2] = np.sqrt(0.08)
    projectors = SectorProjectors(
        sector_masks={
            "A_valley": np.array([True, False, False]),
            "B_valley": np.array([False, True, False]),
            "C_valley": np.array([False, False, True]),
        },
        center_masks={},
        overlap_mask=np.zeros(3, dtype=bool),
        qcut=0.5,
        warnings=[],
    )
    payload: dict[str, object] = {}
    basis_transforms: dict[str, dict[str, np.ndarray]] = {}

    _add_valley_subspace_diagnostic(
        payload,
        basis_transforms,
        "GammaM",
        np.array([101, 102]),
        np.array([0.1, 0.1001]),
        coefficients,
        projectors,
        degeneracy_tol_meV=1.0,
        thresholds={"W_val_min": 0.8, "P_v_clean": 0.95, "P_v_approx": 0.85},
    )

    assert payload["polarization_score"] == pytest.approx(0.92)
    assert payload["subspace_valley_status"] == "valley_approximately_separable_subspace"
    assert "GammaM" in basis_transforms
    assert bool(basis_transforms["GammaM"]["valid_valley_subspace"]) is False


def test_write_detailed_files_false_writes_only_summary_files(tmp_path):
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
    assert outputs["valley_summary_json"].exists()
    # valley_weights.csv is a quick-scan file in standard profile.
    assert outputs.get("valley_weights_csv", None) and outputs["valley_weights_csv"].exists()
    assert not (out_dir / "valley_subspace.json").exists()
    assert not (out_dir / "symmetry_report.json").exists()
    assert not (out_dir / "rotation_eigenvalues.csv").exists()
    assert not (out_dir / "little_group_eigenvalues.csv").exists()
    assert not (out_dir / "little_group_representations.json").exists()
    assert not (out_dir / "symmetry_eigenvalues.csv").exists()
    assert not (out_dir / "diagnostics.h5").exists()
    assert not (out_dir / "projector_symmetry_report.json").exists()
    assert not (out_dir / "hsp_star_conjugation.json").exists()
    assert not (out_dir / "hsp_star_derived_characters.json").exists()


def test_write_analysis_outputs_plumbs_hsp_star_reports_to_summary(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["output"]["profile"] = "standard"
    raw["output"].pop("write_detailed_files", None)
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)

    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    hsp_star_conjugation = {
        "status": "ok",
        "entries": [{"conjugation_status": "matched"}],
    }
    hsp_star_derived = {
        "status": "ok",
        "entries": [{"status": "derived"}],
    }
    irrep_workflow_decisions = {
        "status": "ok",
        "workflow_paths": ["direct_qcut", "symmetry_adapted", "blocked"],
        "readiness_levels": ["trusted", "usable_with_caution", "blocked"],
        "by_kpoint": {
            "Gamma": {
                "K_valley": {
                    "workflow_path": "direct_qcut",
                    "readiness_level": "trusted",
                    "reason": "test",
                    "uses_symmetry_adapted_projector": False,
                    "direct_qcut_allowed": True,
                },
            },
        },
    }
    valley_irrep_matching = {
        "status": "ok",
        "tables_implemented": ["spinful_C3"],
        "by_kpoint": {
            "Gamma": {
                "K_valley": {
                    "1": {
                        "workflow_path": "direct_qcut",
                        "readiness_level": "trusted",
                        "subspace_group_candidate": "C3_like",
                        "operation_id": 1,
                        "operation_order": 3,
                        "matched_irrep": "C3_spinor_phase_+1/2",
                        "matching_status": "matched",
                        "reason": "test",
                        "eigenphases": [0.5],
                    },
                },
            },
        },
    }
    ebr_input_candidates = {
        "status": "has_candidates",
        "candidate_count": 1,
        "blocked_count": 0,
        "candidates": [{"ready_for_ebr_input": True}],
        "blocked": [],
    }
    ebr_problem_instances = {
        "status": "has_instances",
        "instance_count": 1,
        "instances": [{"instance_id": "ebr_instance_001"}],
    }
    ebr_export_bundle = {
        "status": "ready_for_external_solver",
        "schema_version": "1.0.0",
        "bundle_count": 1,
        "excluded_count": 0,
        "reduced_ebr_decomposition_status": "not_implemented",
        "bundles": [
            {
                "bundle_id": "bundle_ebr_instance_001",
                "valley": "K_valley",
                "subspace_group_candidate": "C3_like",
                "workflow_path": "direct_qcut",
                "expected_hsps": ["GammaM", "KM"],
                "optional_hsps": ["MM"],
                "missing_optional_hsps": ["MM"],
                "ready_for_external_solver": True,
            },
        ],
        "excluded_instances": [],
    }

    outputs = write_analysis_outputs(
        config=config,
        qcut=0.5,
        weight_rows=[],
        sector_names=["K_valley"],
        subspace_payload={"kpoints": {}},
        symmetry_payload={
            "status": "ok",
            "detected_operations": [],
            "candidate_rotations": [],
            "little_group_check": {"status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"status": "completed"},
        },
        symmetry_rows=[],
        projectors_by_kpoint={},
        qcut_scan_payload={},
        symmetry_representation_payload={},
        basis_transforms={},
        hsp_star_conjugation_report=hsp_star_conjugation,
        hsp_star_derived_characters=hsp_star_derived,
        irrep_workflow_decisions=irrep_workflow_decisions,
        valley_irrep_matching=valley_irrep_matching,
        ebr_input_candidates=ebr_input_candidates,
        ebr_problem_instances=ebr_problem_instances,
        ebr_export_bundle=ebr_export_bundle,
    )

    payload = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert payload["hsp_star_conjugation"] == hsp_star_conjugation
    assert payload["hsp_star_derived_characters"] == hsp_star_derived
    assert payload["irrep_workflow_decisions"] == irrep_workflow_decisions
    assert payload["valley_irrep_matching"] == valley_irrep_matching
    assert payload["valley_ebr_input_candidates"] == ebr_input_candidates
    assert payload["valley_ebr_problem_instances"] == ebr_problem_instances
    assert payload["valley_ebr_export_bundle"] == ebr_export_bundle
    assert "Valley irrep matching" in outputs["summary_text"]
    assert "C3_spinor_phase_+1/2" in outputs["summary_text"]
    assert "EBR export bundle" in outputs["summary_text"]
    assert "bundle_ebr_instance_001" in outputs["summary_text"]


def test_single_band_and_not_degenerate_no_subspace_valley_status_mislabel(tmp_path):
    """P1-2: single_band / not_degenerate should not be labelled not_valley_derived."""
    cases = [
        ("single_band", np.array([0.1]), np.array([101])),
        ("not_degenerate", np.array([0.1, 0.105]), np.array([101, 102])),
    ]
    for label, energies, bands in cases:
        h5_path = tmp_path / f"wf_{label}.h5"
        with h5py.File(h5_path, "w") as h5:
            meta = h5.create_group("metadata")
            lattice = meta.create_group("lattice")
            lattice["direct_cart"] = np.eye(3)
            lattice["reciprocal_cart"] = np.eye(3) * 10.0
            meta["spinor"] = False
            meta["source"] = "toy"
            meta["vasp_band_index_base"] = 1
            kp = h5.create_group("kpoints").create_group("0")
            kp["name"] = "GammaM"
            kp["frac"] = np.array([0.0, 0.0, 0.0])
            kp["cart"] = np.array([0.0, 0.0, 0.0])
            nb = len(bands)
            kp["g_vectors_frac"] = np.array([[0, 0, 0], [1, 0, 0]])
            kp["g_vectors_cart"] = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
            coeffs = np.zeros((nb, 1, 2), dtype=np.complex128)
            coeffs[0, 0, 0] = 1.0 + 0.0j
            if nb == 2:
                coeffs[1, 0, 1] = 1.0 + 0.0j
            kp["coefficients"] = coeffs
            kp["energies_eV"] = energies
            kp["band_indices_vasp"] = bands

        config_path = tmp_path / f"config_{label}.yaml"
        out_dir = tmp_path / f"out_{label}"
        config = {
            "input": {"wavefunction_h5": str(h5_path)},
            "analysis": {"kpoints": ["GammaM"], "iband": bands.tolist(), "degeneracy_tol_meV": 1.0},
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
                {"name": "K_sector", "centers": ["K"]},
                {"name": "Kp_sector", "centers": ["Kp"]},
            ],
            "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.5, "overlap_policy": "warn_exclude"},
            "output": {"directory": str(out_dir), "profile": "debug"},
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        outputs = analyze_hsp(config_path)
        subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
        kp_data = subspace["kpoints"]["GammaM"]

        for w in kp_data["weights"]:
            assert w.get("valley_status") == "raw_valley_clean", f"raw band mislabeled in {label}"

        diag = kp_data.get("valley_adapted_subspace", {})
        assert diag.get("status") == label, f"subspace diagnostic status mismatch in {label}"
        assert kp_data.get("subspace_valley_status", "missing") not in ("not_valley_derived",), \
            f"subspace labeled not_valley_derived in {label}"

        summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
        assert not any("target subspace is not valley-derived" in w for w in summary["warnings"]), \
            f"false subspace warning in {label}"
        subspace_rows = summary["valley_subspace_analysis"]
        assert subspace_rows
        assert subspace_rows[0].get("basis_status") == label, \
            f"basis_status missing from summary JSON for {label}"
        summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
        assert "basis_status" not in summary_text


def test_rotation_order_none_yields_not_requested_symmetry_status(tmp_path):
    """P1-3: rotation_order: none should give symmetry_status = not_requested."""
    h5_path = tmp_path / "wf.h5"
    write_fixture(h5_path)
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    structure = tmp_path / "CONTCAR"
    write_square_poscar(structure)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["symmetry"]["filters"]["rotation_order"] = "none"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    kp_data = subspace["kpoints"]["GammaM"]
    assert kp_data.get("symmetry_status") == "not_requested"

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary["symmetry_analysis"]["symmetry_eigenvalue_enabled"] is False
    assert summary["symmetry_eigenvalues"] == []
    assert "symmetry_eigenvalues_csv" not in outputs
    assert not (out_dir / "symmetry_eigenvalues.csv").exists()


def test_subspace_projector_unreliable_when_band_overlap_exceeds_threshold(tmp_path):
    """P2-4: adapted subspace with band W_overlap > threshold -> projector_unreliable."""
    h5_path = tmp_path / "wf.h5"
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3) * 10.0
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"
        kp["frac"] = np.array([0.0, 0.0, 0.0])
        kp["cart"] = np.array([0.0, 0.0, 0.0])
        # G-vectors: [0,0,0] clean K, [5,0,0] clean Kp, [2.5,0,0] overlap
        q_cart = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        kp["g_vectors_frac"] = q_cart
        kp["g_vectors_cart"] = q_cart
        a = np.sqrt(0.9)
        b = np.sqrt(0.1)
        kp["coefficients"] = np.array(
            [
                [[a + 0.0j, 0.0 + 0.0j, b + 0.0j]],
                [[0.0 + 0.0j, a + 0.0j, b + 0.0j]],
            ],
            dtype=np.complex128,
        )
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0},
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
            {"name": "K_sector", "centers": ["K"]},
            {"name": "Kp_sector", "centers": ["Kp"]},
        ],
        "projection": {
            "qcut_mode": "absolute", "qcut_Ainv": 3.0, "overlap_policy": "warn_exclude",
            "thresholds": {"W_val_min": 0.8},
        },
        "output": {"directory": str(out_dir), "profile": "debug"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        outputs = analyze_hsp(config_path)

    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    kp_data = subspace["kpoints"]["GammaM"]
    assert kp_data.get("subspace_valley_status") == "projector_unreliable"


def test_h5_reader_rejects_duplicate_kpoint_names(tmp_path):
    """P2-6: duplicate k-point names in HDF5 should raise ValueError."""
    h5_path = tmp_path / "dup.h5"
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3) * 10.0
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kpts_grp = h5.create_group("kpoints")
        for idx in range(2):
            kp = kpts_grp.create_group(str(idx))
            kp["name"] = "GammaM"
            kp["frac"] = np.zeros(3)
            kp["cart"] = np.zeros(3)
            kp["g_vectors_frac"] = np.zeros((1, 3))
            kp["g_vectors_cart"] = np.zeros((1, 3))
            kp["coefficients"] = np.ones((1, 1, 1), dtype=np.complex128)
            kp["energies_eV"] = np.array([0.1])
            kp["band_indices_vasp"] = np.array([1])

    with pytest.raises(ValueError, match="Duplicate k-point name"):
        read_wavefunction_h5(h5_path)


def test_subspace_thresholds_inherit_user_config(tmp_path):
    """Fix 2: user thresholds (overlap_warn) affect subspace projector_unreliable."""
    h5_path = tmp_path / "wf.h5"
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3) * 10.0
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"
        kp["frac"] = np.array([0.0, 0.0, 0.0])
        kp["cart"] = np.array([0.0, 0.0, 0.0])
        q_cart = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        kp["g_vectors_frac"] = q_cart
        kp["g_vectors_cart"] = q_cart
        a = np.sqrt(0.9)
        b = np.sqrt(0.1)
        kp["coefficients"] = np.array(
            [[[a, 0.0, b]], [[0.0, a, b]]], dtype=np.complex128,
        )
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0},
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
            {"name": "K_sector", "centers": ["K"]},
            {"name": "Kp_sector", "centers": ["Kp"]},
        ],
        "projection": {
            "qcut_mode": "absolute", "qcut_Ainv": 3.0, "overlap_policy": "warn_exclude",
            "thresholds": {"W_val_min": 0.8, "overlap_warn": 0.15},
        },
        "output": {"directory": str(out_dir), "profile": "debug"},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        outputs = analyze_hsp(config_path)

    subspace = json.loads(outputs["valley_subspace_json"].read_text(encoding="utf-8"))
    kp_data = subspace["kpoints"]["GammaM"]
    # overlap_warn=0.15 > max_w_overlap=0.1 -> not projector_unreliable
    assert kp_data.get("subspace_valley_status") == "valley_separable_subspace"


def test_generic_irrep_source_disabled_by_default(tmp_path):
    """Default config has generic_irrep_source disabled."""
    from valleyscope.io.config import load_config
    import yaml, numpy as np
    config = {
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {"kpoints": ["GammaM"], "iband": [1]},
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": str(tmp_path)},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.generic_irrep_source.enabled is False
    assert cfg.generic_irrep_source.spacegroup_number is None


def test_generic_irrep_source_config_parses(tmp_path):
    """Config parses generic_irrep_source block correctly."""
    from valleyscope.io.config import load_config
    import yaml, numpy as np
    config = {
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [1],
            "generic_irrep_source": {
                "enabled": True,
                "spacegroup_number": 150,
                "spinor": True,
                "operation_match_tol": 1e-4,
                "source_hsp_labels": {
                    "GammaM": {"K_valley": "GM"},
                    "KM": {"K_valley": "K"},
                },
            },
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": str(tmp_path)},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    cfg = load_config(cfg_path)
    gis = cfg.generic_irrep_source
    assert gis.enabled is True
    assert gis.spacegroup_number == 150
    assert gis.spinor is True
    assert gis.operation_match_tol == 1e-4
    assert gis.source_hsp_labels == {
        "GammaM": {"K_valley": "GM"},
        "KM": {"K_valley": "K"},
    }


def test_generic_irrep_source_rejects_non_integer_sg(tmp_path):
    """Config rejects non-integer spacegroup_number."""
    from valleyscope.io.config import load_config
    import yaml, numpy as np
    for bad_sg in [1.5, "150", True, None]:
        config = {
            "input": {"wavefunction_h5": "wave.h5"},
            "analysis": {
                "kpoints": ["GammaM"], "iband": [1],
                "generic_irrep_source": {
                    "enabled": True,
                    "spacegroup_number": bad_sg,
                    "spinor": True,
                    "source_hsp_labels": {"GammaM": {"K_valley": "GM"}},
                },
            },
            "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
            "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
            "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
            "output": {"directory": str(tmp_path)},
        }
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="spacegroup_number"):
            load_config(cfg_path)


def test_generic_irrep_source_rejects_non_bool_spinor(tmp_path):
    """Config rejects non-boolean spinor."""
    from valleyscope.io.config import load_config
    import yaml, numpy as np
    config = {
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [1],
            "generic_irrep_source": {
                "enabled": True,
                "spacegroup_number": 150,
                "spinor": "true",
                "source_hsp_labels": {"GammaM": {"K_valley": "GM"}},
            },
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": str(tmp_path)},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="spinor"):
        load_config(cfg_path)


def test_generic_irrep_source_rejects_invalid_tol(tmp_path):
    """Config rejects non-positive operation_match_tol."""
    from valleyscope.io.config import load_config
    import yaml, numpy as np
    config = {
        "input": {"wavefunction_h5": "wave.h5"},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [1],
            "generic_irrep_source": {
                "enabled": True,
                "spacegroup_number": 150,
                "spinor": True,
                "operation_match_tol": 0.0,
                "source_hsp_labels": {"GammaM": {"K_valley": "GM"}},
            },
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": str(tmp_path)},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="operation_match_tol"):
        load_config(cfg_path)


def test_generic_irrep_source_blocked_negative_toy_fixture(tmp_path):
    """A mismatched source table reaches generic preflight but stays blocked."""
    from valleyscope.analysis.database_ingestion_record import (
        load_database_ingestion_record_from_directory,
    )

    h5_path = tmp_path / "wf.h5"
    structure = tmp_path / "CONTCAR"
    write_square_poscar(structure)
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3)
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"; kp["frac"] = np.zeros(3); kp["cart"] = np.zeros(3)
        kp["g_vectors_frac"] = np.array([[1, 0, 0], [-1, 0, 0]])
        kp["g_vectors_cart"] = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        kp["coefficients"] = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.complex128)
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    out_dir = tmp_path / "out"
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0,
            "generic_irrep_source": {
                "enabled": True, "spacegroup_number": 143, "spinor": False,
                "source_hsp_labels": {"GammaM": {"K_valley": "GM"}},
            },
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart",
            "centers": [{"name": "K", "cart": [1.0, 0.0, 0.0]},
                        {"name": "Kp", "cart": [-1.0, 0.0, 0.0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]},
                              {"name": "Kp_valley", "centers": ["Kp"]}],
        "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.25, "overlap_policy": "warn_exclude"},
        "symmetry": {
            "operations": {"mode": "auto", "structure_file": str(structure), "backend": "spglib"},
            "tolerance": {"symprec": 1.0e-5, "angle_tolerance": -1.0},
            "filters": {"rotation_order": 2},
        },
        "output": {"directory": str(out_dir), "profile": "standard"},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    # Standard: public files exist, no debug leaks.
    assert outputs["valley_summary_json"].exists()
    assert outputs["valley_ebr_export_bundle_json"].exists()
    written = {p.name for p in out_dir.iterdir() if p.is_file()}
    debug_only = {"valley_subspace.json", "diagnostics.h5", "symmetry_report.json",
                  "hsp_star_conjugation.json", "hsp_star_derived_characters.json"}
    assert not (written & debug_only), f"debug files leaked: {written & debug_only}"

    # Database ingestion from output directory.
    record = load_database_ingestion_record_from_directory(str(out_dir))
    assert record["summary_status"] == "present"
    assert "valley_irrep_records" in record

    # Negative assertion: the scalar square fixture exposes a P4/mmm-like
    # valley-preserving operation set, which must not be matched to the SG143
    # source table by convenience.
    summary = json.loads(outputs["valley_summary_json"].read_text())
    vm = summary.get("valley_irrep_matching", {})
    gm = vm.get("generic_matches_by_kpoint", {})
    blocked = gm["GammaM"]["K_valley"]
    assert blocked["matching_status"] == "blocked"
    assert blocked["matching_strategy"] == "bilbao_restricted_character"
    assert blocked["diagnostic_only"] is True
    assert "table_operation_matching_failed" in blocked["reason"]
    assert summary["valley_ebr_input_candidates"]["status"] == "no_candidates"
    assert summary["valley_ebr_input_candidates"]["candidate_count"] == 0
    assert summary["valley_ebr_problem_instances"]["status"] == "no_instances"
    assert summary["valley_ebr_export_bundle"]["status"] == "no_bundles"
    assert not (out_dir / "valley_reduced_ebr_mapping.json").exists()


def test_generic_irrep_positive_analyze_hsp_workflow_e2e(tmp_path, monkeypatch):
    """analyze_hsp wires a trusted generic match into reduced EBR mapping."""
    from valleyscope.analysis.database_ingestion_record import (
        load_database_ingestion_record_from_directory,
    )
    import valleyscope.workflows.analyze_hsp as workflow_mod

    h5_path = tmp_path / "wf.h5"
    structure = tmp_path / "CONTCAR"
    write_square_poscar(structure)
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3)
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"; kp["frac"] = np.zeros(3); kp["cart"] = np.zeros(3)
        kp["g_vectors_frac"] = np.array([[1, 0, 0], [-1, 0, 0]])
        kp["g_vectors_cart"] = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        kp["coefficients"] = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.complex128)
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    symmetry_adapted_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "reference_valley": "K_valley",
                    "orbit": ["K_valley"],
                    "hsp_preserving_operation_ids": [0, 4],
                    "subspace_space_group": {
                        "valley_preserving_operation_ids": [0, 4],
                        "candidate_space_group_symbol": "P4",
                    },
                    "subspace_group": {
                        "subspace_group_candidate": "C2_like",
                        "operation_orders": {"0": 1, "4": 2},
                    },
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                                {"operation_id": 4, "eigenphases": [0.0, 0.5]},
                            ],
                        },
                    },
                }],
            },
        },
    }
    workflow_decisions = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    source_chars = {
        "A": {1: 1 + 0j, 2: 1 + 0j},
        "B": {1: 1 + 0j, 2: -1 + 0j},
    }
    matching_table = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P4",
        "expected_hsps": ["GammaM"],
        "irreps": ["GammaM:A", "GammaM:B"],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0]},
            {"label": "EBR_B", "vector": [0, 1]},
        ],
    }
    bad_table = {
        **matching_table,
        "expected_hsps": ["GammaM", "KM"],
        "irreps": ["GammaM:A", "GammaM:B", "KM:X"],
        "ebrs": [
            {"label": "EBR_A", "vector": [1, 0, 0]},
            {"label": "EBR_B", "vector": [0, 1, 0]},
        ],
    }
    table_path = tmp_path / "reduced_ebr_table.json"
    bad_table_path = tmp_path / "bad_reduced_ebr_table.json"
    table_path.write_text(json.dumps(matching_table), encoding="utf-8")
    bad_table_path.write_text(json.dumps(bad_table), encoding="utf-8")

    monkeypatch.setattr(
        workflow_mod,
        "_build_symmetry_adapted_valley_report",
        lambda **_: symmetry_adapted_report,
    )
    monkeypatch.setattr(
        workflow_mod,
        "_build_hsp_star_derived_character_layer",
        lambda **_: (None, None),
    )
    monkeypatch.setattr(
        workflow_mod,
        "build_irrep_workflow_decisions",
        lambda **_: workflow_decisions,
    )
    monkeypatch.setattr(
        workflow_mod,
        "load_standard_irrep_table",
        lambda spacegroup_number, *, spinor: object(),
    )
    monkeypatch.setattr(
        workflow_mod,
        "build_source_payload_for_generic_matching",
        lambda **_: {
            "status": "ok",
            "source_irrep_characters": source_chars,
            "source_operation_map": {0: 1, 4: 2},
        },
    )

    def write_cfg(config_path: Path, out_dir: Path, reduced_table_path: Path) -> None:
        config = {
            "input": {"wavefunction_h5": str(h5_path)},
            "analysis": {
                "kpoints": ["GammaM"],
                "iband": [101, 102],
                "degeneracy_tol_meV": 1.0,
                "generic_irrep_source": {
                    "enabled": True,
                    "spacegroup_number": 143,
                    "spinor": False,
                    "source_hsp_labels": {"GammaM": {"K_valley": "GM"}},
                },
                "reduced_ebr": {
                    "enabled": True,
                    "table_file": str(reduced_table_path),
                },
            },
            "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
            "valley_centers": {
                "coordinate_mode": "cart",
                "centers": [
                    {"name": "K", "cart": [1.0, 0.0, 0.0]},
                    {"name": "Kp", "cart": [-1.0, 0.0, 0.0]},
                ],
            },
            "valley_subspaces": [
                {"name": "K_valley", "centers": ["K"]},
                {"name": "Kp_valley", "centers": ["Kp"]},
            ],
            "projection": {
                "qcut_mode": "absolute",
                "qcut_Ainv": 0.25,
                "overlap_policy": "warn_exclude",
            },
            "symmetry": {
                "operations": {
                    "mode": "auto",
                    "structure_file": str(structure),
                    "backend": "spglib",
                },
                "tolerance": {"symprec": 1.0e-5, "angle_tolerance": -1.0},
                "filters": {"rotation_order": 2},
            },
            "output": {"directory": str(out_dir), "profile": "standard"},
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    out_dir = tmp_path / "out"
    config_path = tmp_path / "cfg.yaml"
    write_cfg(config_path, out_dir, table_path)

    outputs = analyze_hsp(config_path)

    assert outputs["valley_summary_json"].exists()
    assert outputs["valley_reduced_ebr_mapping_json"].exists()
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    gm = summary["valley_irrep_matching"]["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "matched"
    assert gm["matching_strategy"] == "bilbao_restricted_character"
    assert gm["irrep_multiplicities"] == {"A": 1, "B": 1}
    assert gm["subspace_space_group"]["candidate_space_group_symbol"] == "P4"
    assert gm["subspace_space_group"]["candidate_space_group_symbol"] != "C2_like"
    assert summary["valley_ebr_input_candidates"]["candidate_count"] == 2
    assert summary["valley_ebr_problem_instances"]["instance_count"] == 1
    inst = summary["valley_ebr_problem_instances"]["instances"][0]
    assert inst["ready_for_ebr_decomposition"] is True
    assert inst["subspace_group_candidate"] == "P4"
    assert inst["legacy_subspace_group_candidate"] == "C2_like"
    assert inst["expected_hsps"] == ["GammaM"]
    assert summary["valley_ebr_export_bundle"]["bundle_count"] == 1
    b = summary["valley_ebr_export_bundle"]["bundles"][0]
    assert b["ready_for_external_solver"] is True
    result = summary["valley_reduced_ebr_mapping"]
    assert result["mapping_status"] == "solved_exact"
    assert result["solutions"][0]["ebr_decomposition"] == [
        {"label": "EBR_A", "coefficient": 1},
        {"label": "EBR_B", "coefficient": 1},
    ]
    record = load_database_ingestion_record_from_directory(str(out_dir))
    assert record["record_status"] == "has_ready_ebr_bundles"
    assert record["ready_bundle_count"] == 1
    assert record["reduced_ebr_mapping_status"] == "solved_exact"
    assert record["reduced_ebr_classification_counts"]["atomic_compatible"] == 1

    bad_out_dir = tmp_path / "bad_out"
    bad_config_path = tmp_path / "bad_cfg.yaml"
    write_cfg(bad_config_path, bad_out_dir, bad_table_path)
    bad_outputs = analyze_hsp(bad_config_path)
    bad_summary = json.loads(
        bad_outputs["valley_summary_json"].read_text(encoding="utf-8")
    )
    bad_mapping = bad_summary["valley_reduced_ebr_mapping"]
    assert bad_mapping["mapping_status"] == "not_evaluated"
    assert bad_mapping["excluded_bundles"]
    assert "expected_hsps" in bad_mapping["excluded_bundles"][0]["reason"]
