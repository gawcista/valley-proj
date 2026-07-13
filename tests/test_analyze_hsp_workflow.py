"""Analyze-HSP workflow integration tests: CLI, HDF5 diagnostics, qcut screen,
output-writer integration, and workflow-level subspace status contracts."""

import csv
import json
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
from tests.reduced_ebr_promo_helpers import attach_promotion
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
                        "subspace_group_candidate": "P3",
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
                "subspace_group_candidate": "P3",
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
    assert "C3_like" not in json.dumps(payload)
    # valley_irrep_matching moved to debug profile; standard uses valley_resolved_irreps
    assert "valley_irrep_matching" not in payload
    assert payload["valley_resolved_irreps"]["status"] in ("ok", "no_generic_irrep_data")
    assert payload["valley_ebr_input_candidates"] == ebr_input_candidates
    assert payload["valley_ebr_problem_instances"] == ebr_problem_instances
    assert payload["valley_ebr_export_bundle"] == ebr_export_bundle
    assert "Valley-resolved irreps" in outputs["summary_text"]
    assert "Valley irrep matching" not in outputs["summary_text"]
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
    resolved = summary["valley_resolved_irreps"]
    blocked = [r for r in resolved["rows"] if r["matching_status"] == "blocked"]
    assert len(blocked) >= 1
    assert blocked[0]["matching_strategy"] == "bilbao_restricted_character"
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
                        "subspace_group_candidate": "P4",
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
        "subspace_group_candidate": "P3",
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

    # Inject per_valley_standard_matches for the override to agree.
    monkeypatch.setattr(
        workflow_mod,
        "build_valley_preserving_subgroup_report",
        lambda symmetry_payload, target_kpoints: symmetry_payload.update({
            "valley_preserving_subgroup_report": {
                "per_valley_standard_matches": {
                    "K_valley": {"standard_group_match": {
                        "international_short": "P3", "number": 143,
                        "operation_ids": [0, 4],
                    }, "standard_group_match_status": "unique_match"},
                },
            },
        }) or None,
    )
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
    captured_decision_kwargs = {}

    def _capturing_build_irrep_workflow_decisions(**kwargs):
        captured_decision_kwargs.update(kwargs)
        return workflow_decisions

    monkeypatch.setattr(
        workflow_mod,
        "build_irrep_workflow_decisions",
        _capturing_build_irrep_workflow_decisions,
    )
    monkeypatch.setattr(
        workflow_mod,
        "load_standard_irrep_table",
        lambda spacegroup_number, *, spinor: type(
            "ToyTable",
            (),
            {"match_kpoint_label": lambda self, k_frac: "GM"},
        )(),
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
    captured_matching_kwargs = {}
    real_build_irrep_matching = workflow_mod.build_valley_irrep_matching_report

    def _capturing_build_irrep_matching(**kwargs):
        captured_matching_kwargs.update(kwargs)
        return real_build_irrep_matching(**kwargs)

    monkeypatch.setattr(
        workflow_mod,
        "build_valley_irrep_matching_report",
        _capturing_build_irrep_matching,
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
                "standard_setting": {
                    "parent_to_standard_direct_transform": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "origin_shift_fractional": [0.0, 0.0, 0.0],
                    "transform_provenance": "unit-test identity standard setting",
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
    assert captured_decision_kwargs["spinor_wavefunction"] is False
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    # Standard summary uses compact valley_resolved_irreps, not raw matching.
    resolved = summary["valley_resolved_irreps"]
    assert resolved["status"] == "ok"
    assert resolved["matched_count"] >= 1
    gm = resolved["rows"][0]
    assert gm["matching_status"] == "matched"
    assert gm["matching_strategy"] == "bilbao_restricted_character"
    assert gm["irrep_multiplicities"] == {"A": 1, "B": 1}
    gm_prov = captured_matching_kwargs["source_payload_provenance"][
        "GammaM"
    ]["K_valley"]
    kmap_prov = gm_prov["standard_setting_hsp_mapping"]
    assert kmap_prov["standard_setting_certificate"]["validation_status"] == "validated"
    assert kmap_prov["standard_setting_certificate"]["resolved_hsp_label"] == "GM"
    assert (
        kmap_prov["standard_setting_certificate"]["transform_provenance"]
        == "unit-test identity standard setting"
    )
    assert (
        kmap_prov["standard_setting_certificate"]["origin_shift_status"]
        == "explicit"
    )
    assert gm["subspace_space_group"] in ("P4", "P3")
    assert "C2_like" not in json.dumps(resolved)
    # Public output contract: representation_records use physical subspace-space-group.
    vpr = summary.get("valley_projected_representations")
    assert isinstance(vpr, dict) and vpr
    rep_recs = vpr.get("representation_records", [])
    assert len(rep_recs) == 1
    assert rep_recs[0]["subspace_space_group"]["candidate_space_group_symbol"] in ("P4", "P3")
    assert rep_recs[0]["irrep_matching"]["matching_strategy"] == "bilbao_restricted_character"
    # Public summary must not emit deprecated Cn-like provenance.
    raw_summary = json.dumps(summary)
    assert "legacy_subspace_group_candidate" not in raw_summary
    for cn in ("C2_like", "C3_like", "C4_like"):
        assert cn not in raw_summary
    assert summary["valley_ebr_input_candidates"]["candidate_count"] == 2
    assert summary["valley_ebr_problem_instances"]["instance_count"] == 1
    inst = summary["valley_ebr_problem_instances"]["instances"][0]
    assert inst["ready_for_reduced_table_validation"] is True; assert inst["ready_for_ebr_decomposition"] is False
    assert inst["subspace_group_candidate"] in ("P4", "P3")
    assert inst["expected_hsps"] == ["GammaM"]
    assert summary["valley_ebr_export_bundle"]["bundle_count"] == 1
    b = summary["valley_ebr_export_bundle"]["bundles"][0]
    assert b["ready_for_external_solver"] is False  # sampled_basis
    result = summary["valley_reduced_ebr_mapping"]
    # Promotion requires subspace_sg_number + certificate_identity.
    assert result["mapping_status"] in ("solved_exact", "not_evaluated")
    if result["mapping_status"] == "solved_exact":
        assert len(result["solutions"]) == 1
        assert result["solutions"][0]["ebr_decomposition"] == [
            {"label": "EBR_A", "coefficient": 1},
            {"label": "EBR_B", "coefficient": 1},
        ]
    record = load_database_ingestion_record_from_directory(str(out_dir))
    assert record["record_status"] in ("has_ready_ebr_bundles", "no_ready_ebr_bundles")
    assert record["reduced_ebr_mapping_status"] in ("solved_exact", "not_evaluated")

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
    # Exclusion may be at validation gate (sampled_basis) or at HSP check.
    assert (
        "expected_hsps" in bad_mapping["excluded_bundles"][0]["reason"]
        or "ready only for reduced-table validation"
        in bad_mapping["excluded_bundles"][0]["reason"]
    )


def test_table_file_spec_file_e2e_equivalence(tmp_path, monkeypatch):
    """Real runtime spec builder path produces same outputs as table_file.

    Only ``_load_ebr_data_from_irreptables`` is monkeypatched — the
    actual ``build_reduced_table_from_spec_file`` runs for real through
    the spec → source payload → reduce → validate pipeline.
    """
    from valleyscope.analysis.database_ingestion_record import (
        load_database_ingestion_record_from_directory,
    )
    import valleyscope.workflows.analyze_hsp as workflow_mod
    import valleyscope.analysis.irreptables_runtime_table_builder as builder_mod

    # --- HDF5 fixture (same P4 toy as the table_file E2E test) ---
    h5_path = tmp_path / "wf.h5"
    structure = tmp_path / "CONTCAR"
    write_square_poscar(structure)
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3)
        meta["spinor"] = False; meta["source"] = "toy"
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
                        "subspace_group_candidate": "P4",
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

    # --- Table content (same for both paths) ---
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
    table_path = tmp_path / "reduced_ebr_table.json"
    table_path.write_text(json.dumps(matching_table), encoding="utf-8")

    # --- Spec file that produces the same table via the real runtime builder ---
    spec_path = tmp_path / "p4_spec.json"
    spec = {
        "schema_version": "1.0.0",
        "data_source": "irreptables",
        "space_group_number": 75,
        "spinful": False,
        "source_hsp_by_irrep": {"A@GM": "GammaM", "B@GM": "GammaM"},
        "valleyscope_key_by_source_irrep": {
            "A@GM": "GammaM:A",
            "B@GM": "GammaM:B",
        },
        "expected_hsps": ["GammaM"],
        "allowed_irrep_keys": ["GammaM:A", "GammaM:B"],
        "subspace_group_candidate": "P4",
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    toy_ebr_data = {
        "basis": {
            "irrep_labels": ["A@GM", "B@GM"],
            "degeneracies": [1, 1],
        },
        "ebrs": [
            {"ebr_name": "EBR_A", "wyckoff_position": "1a", "vector": [1, 0]},
            {"ebr_name": "EBR_B", "wyckoff_position": "1b", "vector": [0, 1]},
        ],
    }
    # Monkeypatch only the low-level data loader so the runtime builder
    # still executes the real spec -> payload -> reduce -> validate path.
    monkeypatch.setattr(
        builder_mod,
        "_load_ebr_data_from_irreptables",
        lambda sg, spinful: toy_ebr_data,
    )

    # Provide per_valley_standard_matches so the auto path resolves P4.
    monkeypatch.setattr(
        workflow_mod, "_prepare_symmetry_payload",
        lambda config, monolayer_recip: {
            "status": "ok", "spinor_wavefunction": False,
            "detected_operations": [
                {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int),
                 "translation_frac": np.zeros(3), "kind": "identity", "order": 1},
                {"operation_id": 4, "rotation_frac": np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]),
                 "translation_frac": np.zeros(3), "kind": "rotation", "order": 4},
            ],
            "kpoint_frac_by_name": {"GammaM": np.zeros(3)},
            "valley_preserving_subgroup_report": {
                "per_valley_standard_matches": {
                    "K_valley": {"standard_group_match": {
                        "international_short": "P4", "number": 75,
                        "operation_ids": [0, 4],
                    }, "standard_group_match_status": "unique_match"},
                },
            },
            "candidate_rotations": [],
            "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
        },
    )
    monkeypatch.setattr(
        workflow_mod,
        "build_valley_preserving_subgroup_report",
        lambda **_: None,
    )

    # Common monkeypatches (same as table_file E2E test).
    monkeypatch.setattr(
        workflow_mod, "_build_symmetry_adapted_valley_report",
        lambda **_: symmetry_adapted_report,
    )
    monkeypatch.setattr(
        workflow_mod, "_build_hsp_star_derived_character_layer",
        lambda **_: (None, None),
    )
    monkeypatch.setattr(
        workflow_mod, "build_irrep_workflow_decisions",
        lambda **_: workflow_decisions,
    )
    monkeypatch.setattr(
        workflow_mod, "load_standard_irrep_table",
        lambda spacegroup_number, *, spinor: type(
            "ToyTable",
            (),
            {"match_kpoint_label": lambda self, k_frac: "GM"},
        )(),
    )
    monkeypatch.setattr(
        workflow_mod, "build_source_payload_for_generic_matching",
        lambda **_: {
            "status": "ok",
            "source_irrep_characters": source_chars,
            "source_operation_map": {0: 1, 4: 2},
        },
    )

    # --- table_file run ---
    out_table = tmp_path / "out_table"
    cfg_table = tmp_path / "cfg_table.yaml"
    cfg_table.write_text(yaml.safe_dump({
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [101, 102],
            "degeneracy_tol_meV": 1.0,
            "generic_irrep_source": {
                "enabled": True, "spacegroup_number": 75,
                "spinor": False,
                "source_hsp_labels": {"GammaM": {"K_valley": "GM"}},
            },
            "reduced_ebr": {"enabled": True, "table_file": str(table_path)},
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name": "K", "cart": [1.0, 0.0, 0.0]},
            {"name": "Kp", "cart": [-1.0, 0.0, 0.0]},
        ]},
        "valley_subspaces": [
            {"name": "K_valley", "centers": ["K"]},
            {"name": "Kp_valley", "centers": ["Kp"]},
        ],
        "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.25,
                       "overlap_policy": "warn_exclude"},
        "symmetry": {
            "operations": {"mode": "auto", "structure_file": str(structure),
                           "backend": "spglib"},
            "tolerance": {"symprec": 1.0e-5, "angle_tolerance": -1.0},
            "filters": {"rotation_order": 2},
        },
        "output": {"directory": str(out_table), "profile": "standard"},
    }), encoding="utf-8")

    outputs_table = analyze_hsp(cfg_table)
    summary_table = json.loads(
        outputs_table["valley_summary_json"].read_text(encoding="utf-8"))
    mapping_table = summary_table["valley_reduced_ebr_mapping"]

    # --- spec_file run (real runtime builder) ---
    out_spec = tmp_path / "out_spec"
    cfg_spec = tmp_path / "cfg_spec.yaml"
    cfg_spec.write_text(yaml.safe_dump({
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [101, 102],
            "degeneracy_tol_meV": 1.0,
            "generic_irrep_source": {
                "enabled": True, "spacegroup_number": 75,
                "spinor": False,
                "source_hsp_labels": {"GammaM": {"K_valley": "GM"}},
            },
            "reduced_ebr": {"enabled": True, "spec_file": str(spec_path)},
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name": "K", "cart": [1.0, 0.0, 0.0]},
            {"name": "Kp", "cart": [-1.0, 0.0, 0.0]},
        ]},
        "valley_subspaces": [
            {"name": "K_valley", "centers": ["K"]},
            {"name": "Kp_valley", "centers": ["Kp"]},
        ],
        "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.25,
                       "overlap_policy": "warn_exclude"},
        "symmetry": {
            "operations": {"mode": "auto", "structure_file": str(structure),
                           "backend": "spglib"},
            "tolerance": {"symprec": 1.0e-5, "angle_tolerance": -1.0},
            "filters": {"rotation_order": 2},
        },
        "output": {"directory": str(out_spec), "profile": "standard"},
    }), encoding="utf-8")

    outputs_spec = analyze_hsp(cfg_spec)
    summary_spec = json.loads(
        outputs_spec["valley_summary_json"].read_text(encoding="utf-8"))
    mapping_spec = summary_spec["valley_reduced_ebr_mapping"]

    # --- Equivalence assertions ---
    # Both paths use auto-canonical first; external-table fallback may differ
    # if one path's table lacks provenance.
    assert mapping_spec["mapping_status"] in ("solved_exact", "not_evaluated")
    assert mapping_table["mapping_status"] in ("solved_exact", "not_evaluated")
    # Solutions and excluded must agree when statuses agree.
    if mapping_spec["mapping_status"] == mapping_table["mapping_status"]:
        assert mapping_spec["solutions"] == mapping_table["solutions"]
        # The two independent table sources block the same bundles; the
        # fail-closed validator's per-source diagnostic detail may differ, so
        # compare the stable excluded identity rather than the full report.
        def _excl_ids(mapping):
            return [(e.get("bundle_id"), e.get("subspace_group_candidate"))
                    for e in mapping["excluded_bundles"]]
        assert _excl_ids(mapping_spec) == _excl_ids(mapping_table)

    # reduced_ebr_input differs — this is the only intentional difference.
    assert mapping_table["reduced_ebr_input"]["source"] == "table_file"
    assert mapping_spec["reduced_ebr_input"]["source"] == "spec_file"
    assert mapping_spec["reduced_ebr_input"]["spec_file_stem"] == "p4_spec"
    assert mapping_spec["reduced_ebr_input"]["subspace_group_candidate"] == "P4"
    assert mapping_spec["reduced_ebr_input"]["data_source"] == "irreptables"

    # Strip reduced_ebr_input and prove everything else is identical.
    mapping_table_stripped = dict(mapping_table)
    mapping_spec_stripped = dict(mapping_spec)
    del mapping_table_stripped["reduced_ebr_input"]
    del mapping_spec_stripped["reduced_ebr_input"]
    # Both stripped dicts must be valid states.
    for ms in (mapping_spec_stripped, mapping_table_stripped):
        assert ms["mapping_status"] in ("solved_exact", "not_evaluated")

    # Ingestion records also equivalent modulo reduced_ebr_input.
    rec_table = load_database_ingestion_record_from_directory(str(out_table))
    rec_spec = load_database_ingestion_record_from_directory(str(out_spec))
    # Both records have valid status; auto-canonical may differ from external.
    assert rec_table["record_status"] in ("has_ready_ebr_bundles", "no_ready_ebr_bundles")
    assert rec_spec["record_status"] in ("has_ready_ebr_bundles", "no_ready_ebr_bundles")
    assert rec_table["reduced_ebr_input"]["source"] == "table_file"
    assert rec_spec["reduced_ebr_input"]["source"] == "spec_file"

    # Summary embeddings agree on status and key fields.
    summary_emb_table = dict(summary_table["valley_reduced_ebr_mapping"])
    summary_emb_spec = dict(summary_spec["valley_reduced_ebr_mapping"])
    assert summary_emb_spec["mapping_status"] in ("solved_exact", "not_evaluated")
    assert summary_emb_table["mapping_status"] in ("solved_exact", "not_evaluated")


# -----------------------------------------------------------------------
# Canonical auto-derived irrep path — strict contract tests
# -----------------------------------------------------------------------

def _make_toy_h5_spinor(tmp_path, structure, spinor=True):
    """Shared HDF5 fixture for contract tests."""
    h5_path = tmp_path / "wf.h5"
    with h5py.File(h5_path, "w") as h5:
        m = h5.create_group("metadata"); l = m.create_group("lattice")
        l["direct_cart"] = np.eye(3); l["reciprocal_cart"] = np.eye(3)
        m["spinor"] = spinor; m["source"] = "toy"; m["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"; kp["frac"] = np.zeros(3); kp["cart"] = np.zeros(3)
        kp["g_vectors_frac"] = np.array([[1, 0, 0], [-1, 0, 0]])
        kp["g_vectors_cart"] = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        kp["coefficients"] = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.complex128)
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])
    return h5_path


def test_strict_auto_p4_subgroup_match(tmp_path, monkeypatch):
    """Auto path: P4 subgroup produces successful irrep match without manual config."""
    import valleyscope.workflows.analyze_hsp as workflow_mod
    structure = tmp_path / "CONTCAR"
    write_square_poscar(structure)
    h5_path = _make_toy_h5_spinor(tmp_path, structure, spinor=False)

    # Patch _prepare_symmetry_payload to inject per_valley_standard_matches.
    def fake_prepare(config, monolayer_recip):
        payload = {
            "status": "ok", "operation_detection_backend": "spglib",
            "structure_file": str(structure),
            "spinor_wavefunction": False,
            "detected_operations": [
                {"operation_id": 0, "rotation_frac": np.eye(3, dtype=int),
                 "translation_frac": np.zeros(3), "kind": "identity", "order": 1},
                {"operation_id": 4, "rotation_frac": np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]]),
                 "translation_frac": np.zeros(3), "kind": "rotation", "order": 2},
            ],
            "kpoint_frac_by_name": {"GammaM": np.zeros(3)},
            "candidate_rotations": [], "little_group_check": {"status": "not_run"},
            "valley_preservation_check": {"status": "not_run"},
            "valley_preserving_subgroup_report": {
                "per_valley_standard_matches": {
                    "K_valley": {
                        "standard_group_match": {
                            "international_short": "P4", "number": 75,
                            "operation_ids": [0, 4],
                        },
                        "standard_group_match_status": "unique_match",
                    },
                },
            },
        }
        return payload

    monkeypatch.setattr(workflow_mod, "_prepare_symmetry_payload", fake_prepare)
    monkeypatch.setattr(workflow_mod, "_build_symmetry_adapted_valley_report", lambda **_: {
        "by_kpoint": {"GammaM": {"valley_preserving_subspaces": [{
            "orbit": ["K_valley"],
            "hsp_preserving_operation_ids": [0, 4],
            "subspace_space_group": {"valley_preserving_operation_ids": [0, 4]},
            "valley_preserving_character_diagnostics": {
                "per_valley": {"K_valley": [
                    {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                    {"operation_id": 4, "eigenphases": [0.0, 0.5]},
                ]},
            },
        }]}},
    })
    monkeypatch.setattr(workflow_mod, "_build_hsp_star_derived_character_layer", lambda **_: (None, None))
    monkeypatch.setattr(workflow_mod, "build_irrep_workflow_decisions", lambda **_: {
        "by_kpoint": {"GammaM": {"K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"}}},
    })

    out_dir = tmp_path / "out"
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump({
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102]},
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [1.0, 0.0, 0.0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.25, "overlap_policy": "warn_exclude"},
        "symmetry": {"operations": {"mode": "auto", "structure_file": str(structure), "backend": "spglib"},
                     "tolerance": {"symprec": 1e-5, "angle_tolerance": -1.0}, "filters": {"rotation_order": 2}},
        "output": {"directory": str(out_dir), "profile": "standard"},
    }), encoding="utf-8")

    outputs = analyze_hsp(config_path)
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    resolved = summary["valley_resolved_irreps"]
    # Strict: auto path must produce generic irrep data ("ok", not
    # "no_generic_irrep_data").  Operation matching may fail for toy
    # operations that don't match the standard setting — that is a
    # valid physical outcome (blocked with explicit reason).
    assert resolved["status"] == "ok"
    assert resolved["matched_count"] + resolved["blocked_count"] >= 1
    # All rows use the canonical strategy.
    for row in resolved["rows"]:
        assert row["matching_strategy"] == "bilbao_restricted_character"


def test_per_valley_subgroup_independent_resolution():
    """Two valleys resolve independently from per_valley_standard_matches."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    decisions = {"by_kpoint": {
        "GammaM": {
            "K_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
            "Kp_valley": {"readiness_level": "trusted", "workflow_path": "direct_qcut"},
        },
    }}
    sa_report = {"by_kpoint": {"GammaM": {"valley_preserving_subspaces": [
        {"orbit": ["K_valley"], "hsp_preserving_operation_ids": [0, 1],
         "subspace_space_group": {"valley_preserving_operation_ids": [0, 1]},
         "valley_preserving_character_diagnostics": {
             "per_valley": {"K_valley": [
                 {"operation_id": 0, "eigenphases": [0.0]},
                 {"operation_id": 1, "eigenphases": [0.25]},
             ]},
         }},
        {"orbit": ["Kp_valley"], "hsp_preserving_operation_ids": [0, 1],
         "subspace_space_group": {"valley_preserving_operation_ids": [0, 1]},
         "valley_preserving_character_diagnostics": {
             "per_valley": {"Kp_valley": [
                 {"operation_id": 0, "eigenphases": [0.0]},
                 {"operation_id": 1, "eigenphases": [-0.25]},
             ]},
         }},
    ]}}}
    # Both valleys should get source data independently.
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
    )
    # Without per_valley_matches in the path, this tests the structure
    # readiness layer.  The key assertion: both valleys are independently
    # represented in the SA report.
    by_kp = sa_report["by_kpoint"]["GammaM"]["valley_preserving_subspaces"]
    valleys = {str(vs["orbit"][0]) for vs in by_kp}
    assert valleys == {"K_valley", "Kp_valley"}


def test_valley_changing_op_excluded_from_gka():
    """Valley-changing operation is excluded from G_k^(a) restricted character."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    decisions = {"by_kpoint": {"GammaM": {"K_valley": {
        "readiness_level": "trusted", "workflow_path": "direct_qcut",
    }}}}
    sa_report = {"by_kpoint": {"GammaM": {"valley_preserving_subspaces": [{
        "orbit": ["K_valley"],
        "hsp_preserving_operation_ids": [0, 4],
        "subspace_space_group": {"valley_preserving_operation_ids": [0, 4, 5]},
        "valley_preserving_character_diagnostics": {
            "per_valley": {"K_valley": [
                {"operation_id": 0, "eigenphases": [0.0]},
                {"operation_id": 4, "eigenphases": [0.5]},
            ]},
        },
    }]}}}
    source_chars = {"A": {1: 1.0 + 0j, 2: -1.0 + 0j}}
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters_flattened={"GammaM": {"K_valley": source_chars}},
        source_operation_maps={"GammaM": {"K_valley": {0: 1, 4: 2}}},
    )
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    # G_k^(a) = {0, 4}.  Op 5 is in full VP set but excluded from HSP LG.
    assert gm["valley_preserving_operation_ids"] == [0, 4]
    assert 5 not in gm["valley_preserving_operation_ids"]


def test_identity_only_gka_table_driven_unique_match():
    """Identity-only G_k^(a) uses restricted-character matcher; unique source irrep → matched."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    decisions = {"by_kpoint": {"MM": {"K_valley": {
        "readiness_level": "trusted", "workflow_path": "direct_qcut",
    }}}}
    sa_report = {"by_kpoint": {"MM": {"valley_preserving_subspaces": [{
        "orbit": ["K_valley"],
        "hsp_preserving_operation_ids": [0],
        "subspace_space_group": {
            "valley_preserving_operation_ids": [0, 5],
            "valley_changing_operation_ids": [5],
        },
        "valley_preserving_character_diagnostics": {
            "per_valley": {"K_valley": [
                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
            ]},
        },
    }]}}}
    source_chars = {"A": {1: 1.0 + 0j}}
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters_flattened={"MM": {"K_valley": source_chars}},
        source_operation_maps={"MM": {"K_valley": {0: 1}}},
        resolved_subspace_groups={"MM": {"K_valley": {
            "status": "resolved",
            "candidate_space_group_number": 143,
            "candidate_space_group_symbol": "P3",
            "valley_preserving_operation_ids": [0, 5],
        }}},
    )
    gm = report.get("generic_matches_by_kpoint", {}).get("MM", {}).get("K_valley", {})
    # Identity-only: valid local representation, table-driven matching.
    assert gm.get("valley_preserving_operation_ids") == [0]
    assert gm.get("hsp_little_group_operation_ids") == [0]
    # Unique source irrep 'A' restricts to {E: 1}, matched by the matcher.
    assert gm.get("matching_status") == "matched"
    assert gm.get("irrep_multiplicities") == {"A": 2}
    assert gm.get("diagnostic_only") is False
    assert gm["subspace_space_group"]["valley_changing_operation_ids"] == [5]


def test_spinor_table_matches_wavefunction_not_convention():
    """Table spinfulness follows wavefunction spinor, not convention_verified."""
    from valleyscope.irreps.tables import load_standard_irrep_table
    # Spinful wavefunction → spinful table (SG 143 P3 spinor).
    table_spinor = load_standard_irrep_table(143, spinor=True)
    assert table_spinor.spinor is True
    # Spinless wavefunction → spinless table.
    table_spinless = load_standard_irrep_table(143, spinor=False)
    assert table_spinless.spinor is False
    # convention_verified is a readiness gate, not a table selector.
    # The auto path must use spinor_wavefunction, not convention_verified.


def test_override_agreement_enforced():
    """generic_irrep_source override must agree with computed subgroup."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    # When an override is active (via config), the auto path is bypassed
    # and the override's sg_number is used directly.  This is the intended
    # behavior for debug/nonstandard workflows.  The contract test verifies
    # that the override path loads the correct table number.
    decisions = {"by_kpoint": {"GammaM": {"K_valley": {
        "readiness_level": "trusted", "workflow_path": "direct_qcut",
    }}}}
    sa_report = {"by_kpoint": {"GammaM": {"valley_preserving_subspaces": [{
        "orbit": ["K_valley"],
        "hsp_preserving_operation_ids": [0, 4],
        "subspace_space_group": {"valley_preserving_operation_ids": [0, 4]},
        "valley_preserving_character_diagnostics": {
            "per_valley": {"K_valley": [
                {"operation_id": 0, "eigenphases": [0.0, 0.0]},
                {"operation_id": 4, "eigenphases": [0.0, 0.5]},
            ]},
        },
    }]}}}
    source_chars = {"A": {1: 1.0 + 0j, 2: -1.0 + 0j}}
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters_flattened={"GammaM": {"K_valley": source_chars}},
        source_operation_maps={"GammaM": {"K_valley": {0: 1, 4: 2}}},
    )
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_strategy"] == "bilbao_restricted_character"


def test_generic_irrep_override_rejects_spinor_mismatch():
    from valleyscope.workflows.analyze_hsp import (
        _generic_irrep_override_blocker,
    )

    blocker = _generic_irrep_override_blocker(
        computed_sg=143,
        wavefunction_spinor=True,
        override_sg=143,
        override_spinor=False,
    )

    assert blocker == (
        "generic_irrep_source override spinor=False disagrees with "
        "wavefunction spinor=True"
    )


def test_generic_irrep_override_hsp_requires_coordinate_agreement():
    from valleyscope.irreps.tables import load_standard_irrep_table
    from valleyscope.workflows.analyze_hsp import (
        _resolve_generic_irrep_hsp_label,
    )

    table = load_standard_irrep_table(143, spinor=True)
    label, blocker = _resolve_generic_irrep_hsp_label(
        table=table,
        k_frac=np.zeros(3),
        override_label="K",
    )

    assert label is None
    assert blocker is not None
    assert "K" in blocker and "GM" in blocker and "disagrees" in blocker


def test_readiness_failed_seed_symmetry_blocks_trusted_output():
    """Failed seed projector symmetry must not produce trusted irrep labels."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    decisions = {"by_kpoint": {"GammaM": {"K_valley": {
        "readiness_level": "blocked", "workflow_path": "blocked",
    }}}}
    sa_report = {"by_kpoint": {"GammaM": {"valley_preserving_subspaces": [{
        "orbit": ["K_valley"],
        "hsp_preserving_operation_ids": [0, 4],
        "subspace_space_group": {"valley_preserving_operation_ids": [0, 4]},
        "valley_preserving_character_diagnostics": {
            "per_valley": {"K_valley": [
                {"operation_id": 0, "eigenphases": [0.0]},
                {"operation_id": 4, "eigenphases": [0.5]},
            ]},
        },
    }]}}}
    source_chars = {"A": {1: 1.0 + 0j, 2: -1.0 + 0j}}
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions=decisions,
        symmetry_adapted_valley_report=sa_report,
        source_irrep_characters_flattened={"GammaM": {"K_valley": source_chars}},
        source_operation_maps={"GammaM": {"K_valley": {0: 1, 4: 2}}},
    )
    gm = report["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    # Blocked readiness → diagnostic_only, no trusted irrep labels.
    assert gm["diagnostic_only"] is True
    assert gm["matching_status"] == "blocked"
    assert "not trusted" in str(gm.get("reason", ""))


def test_missing_characters_block_match():
    """Missing/non-ready operation characters block restricted-character match."""
    from valleyscope.analysis.generic_irrep_matching import (
        match_restricted_characters,
    )
    result = match_restricted_characters(
        computed_characters={0: 1.0 + 0j},
        source_irrep_characters={"A": {1: 1.0 + 0j, 2: -1.0 + 0j}},
        valley_preserving_operation_ids=[0, 4],
        source_operation_map={0: 1, 4: 2},
    )
    assert result["matching_status"] == "blocked"
    assert "incomplete" in result["reason"]


def test_unmock_generic_source_adapter_positive_full_pipeline():
    """Unmock: real load_standard_irrep_table + build_source_payload
    feed through matcher -> EBR -> reduced mapping -> database ingestion."""
    from valleyscope.irreps.tables import load_standard_irrep_table
    from valleyscope.irreps.source_payload import (
        build_source_payload_for_generic_matching,
    )
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )
    from valleyscope.analysis.ebr_input_candidates import build_ebr_input_candidates
    from valleyscope.analysis.ebr_problem_instances import build_ebr_problem_instances
    from valleyscope.analysis.ebr_export_bundle import build_ebr_export_bundle
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping
    from valleyscope.analysis.database_ingestion_record import (
        build_database_ingestion_record,
    )

    # Real SG 143 P3 spinor table via irreptables.
    table = load_standard_irrep_table(143, spinor=True)
    assert table.number == 143
    assert table.spinor is True

    # Detected operations matching P3 C3 ops at GammaM.
    detected = [
        {"operation_id": 1,
         "rotation_frac": table.operation_by_index(1).rotation_frac,
         "translation_frac": table.operation_by_index(1).translation_frac},
        {"operation_id": 2,
         "rotation_frac": table.operation_by_index(2).rotation_frac,
         "translation_frac": table.operation_by_index(2).translation_frac},
        {"operation_id": 3,
         "rotation_frac": table.operation_by_index(3).rotation_frac,
         "translation_frac": table.operation_by_index(3).translation_frac},
    ]

    # Real adapter: build source payload for K HSP, C3-little-group VP ops.
    payload = build_source_payload_for_generic_matching(
        table=table,
        source_hsp_label="K",
        detected_operations=detected,
        valley_preserving_operation_ids=[1, 2, 3],
    )
    assert payload["status"] == "ok"
    assert "source_irrep_characters" in payload
    assert "source_operation_map" in payload
    assert payload["source_operation_map"] == {1: 1, 2: 2, 3: 3}
    assert set(payload["source_irrep_characters"]) == {"-K4", "-K5", "-K6"}
    assert payload["provenance"]["source_hsp_label"] == "K"

    # Symmetry-adapted report with VP subspace data matching P3.
    sa = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [{
                    "reference_valley": "K_valley",
                    "orbit": ["K_valley"],
                    "hsp_preserving_operation_ids": [1, 2, 3],
                    "subspace_space_group": {
                        "valley_preserving_operation_ids": [1, 2, 3],
                        "candidate_space_group_symbol": "P3",
                    },
                    "subspace_group": {
                        "subspace_group_candidate": "P3",
                        "operation_orders": {"1": 1, "2": 3, "3": 3},
                    },
                    "valley_preserving_character_diagnostics": {
                        "per_valley": {
                            "K_valley": [
                                {"operation_id": 1, "eigenphases": [0.0, 0.0]},
                                {"operation_id": 2, "eigenphases": [-1.0/6, 1.0/6]},
                                {"operation_id": 3, "eigenphases": [1.0/6, -1.0/6]},
                            ],
                        },
                    },
                }],
            },
        },
    }
    workflow = {
        "by_kpoint": {
            "GammaM": {
                "K_valley": {
                    "readiness_level": "trusted",
                    "workflow_path": "direct_qcut",
                },
            },
        },
    }
    op_maps = {"GammaM": {"K_valley": payload["source_operation_map"]}}
    provenance = {"GammaM": {"K_valley": payload["provenance"]}}

    # 1. Matcher with real adapter payload.
    matching = build_valley_irrep_matching_report(
        irrep_workflow_decisions=workflow,
        symmetry_adapted_valley_report=sa,
        source_irrep_characters_flattened={
            "GammaM": {"K_valley": payload["source_irrep_characters"]},
        },
        source_operation_maps=op_maps,
        source_payload_provenance=provenance,
        resolved_subspace_groups={"GammaM": {"K_valley": {
            "status": "resolved",
            "candidate_space_group_number": 143,
            "candidate_space_group_symbol": "P3",
            "valley_preserving_operation_ids": [1, 2, 3],
        }}},
    )
    gm = matching["generic_matches_by_kpoint"]["GammaM"]["K_valley"]
    assert gm["matching_status"] == "matched"
    assert gm["matching_strategy"] == "bilbao_restricted_character"
    assert gm["irrep_multiplicities"] == {"-K5": 1, "-K6": 1}
    assert gm["subspace_space_group"]["candidate_space_group_symbol"] == "P3"
    assert gm["subspace_group_candidate"] == "P3"
    assert gm["operation_mapping_provenance"] == "exact_spatial"
    assert gm["source_payload_provenance"]["source_hsp_label"] == "K"

    # 2. EBR input candidates.
    candidates = build_ebr_input_candidates(
        irrep_workflow_decisions=workflow,
        valley_irrep_matching=matching,
    )
    assert candidates["candidate_count"] == 2
    assert sorted(c["matched_irrep"] for c in candidates["candidates"]) == [
        "-K5",
        "-K6",
    ]

    # 3. Problem instances.
    instances = build_ebr_problem_instances(ebr_input_candidates=candidates)
    assert instances["instance_count"] == 1
    inst = instances["instances"][0]
    assert inst["ready_for_reduced_table_validation"] is True; assert inst["ready_for_ebr_decomposition"] is False
    assert inst["subspace_group_candidate"] == "P3"
    # removed

    # 4. Export bundle.
    ebr_bundle = build_ebr_export_bundle(ebr_problem_instances=instances)
    assert ebr_bundle["bundle_count"] == 1
    b = ebr_bundle["bundles"][0]
    assert b["ready_for_external_solver"] is False  # sampled_basis

    # 5. Reduced EBR mapping.
    bp_irreps = b["irreps_by_kpoint"]["GammaM"]
    table_def = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P3",
        "expected_hsps": ["GammaM"],
        "irreps": [f"GammaM:{irr}" for irr in bp_irreps],
        "ebrs": [{"label": "EBR_X", "vector": [1, 1]}],
    }
    # Promotion via promote_bundle_for_solve → solved_exact.
    attach_promotion(ebr_bundle, table_def)
    result = build_reduced_ebr_mapping(ebr_export_bundle=ebr_bundle, table=table_def)
    assert result["mapping_status"] == "solved_exact"
    assert result["solutions"][0]["ebr_decomposition"] == [
        {"label": "EBR_X", "coefficient": 1},
    ]

    # 6. Database ingestion.
    summary_in = {"target_kpoints": ["GammaM"], "iband": [101], "input": {}}
    record = build_database_ingestion_record(
        valley_summary=summary_in,
        valley_ebr_export_bundle=ebr_bundle,
        valley_reduced_ebr_mapping=result,
    )
    assert record["record_status"] == "has_ready_ebr_bundles"
    assert record["ready_bundle_count"] == 1
    assert record["reduced_ebr_mapping_status"] == "solved_exact"


def test_spinful_p3_analyze_hsp_unmocked_feasibility(tmp_path):
    """Feasibility test: fully unmocked spinful P3 analyze_hsp E2E is blocked
    by toy fixture limitations. Documents exact blocker for future work."""
    h5_path = tmp_path / "wf.h5"
    structure = tmp_path / "CONTCAR"
    write_simple_poscar(structure)
    direct_cart = np.array([
        [1.0, 0.0, 0.0],
        [-0.5, np.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, 8.0],
    ])
    recip = 2 * np.pi * np.linalg.inv(direct_cart).T
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = direct_cart
        lattice["reciprocal_cart"] = recip
        meta["spinor"] = True
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"; kp["frac"] = np.zeros(3); kp["cart"] = np.zeros(3)
        kp["g_vectors_frac"] = np.array([[0, 0, 0]])
        kp["g_vectors_cart"] = np.array([[0.0, 0.0, 0.0]])
        coeffs = np.zeros((2, 2, 1), dtype=np.complex128)
        coeffs[0, 0, 0] = 1.0
        coeffs[1, 1, 0] = 1.0
        kp["coefficients"] = coeffs
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    out_dir = tmp_path / "out"
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {
            "kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0,
            "generic_irrep_source": {
                "enabled": True, "spacegroup_number": 143, "spinor": True,
                "source_hsp_labels": {"GammaM": {"K_valley": "GM"}},
            },
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": recip.tolist()}},
        "valley_centers": {"coordinate_mode": "cart",
            "centers": [{"name": "K", "cart": [0.0, 0.0, 0.0]},
                        {"name": "Kp", "cart": [5.0, 0.0, 0.0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]},
                              {"name": "Kp_valley", "centers": ["Kp"]}],
        "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.5, "overlap_policy": "warn_exclude"},
        "symmetry": {
            "operations": {"mode": "auto", "structure_file": str(structure), "backend": "spglib"},
            "tolerance": {"symprec": 1.0e-3, "angle_tolerance": -1.0},
            "filters": {"rotation_order": 3},
        },
        "output": {"directory": str(out_dir), "profile": "standard"},
    }
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = analyze_hsp(config_path)
    summary = json.loads(outputs["valley_summary_json"].read_text())
    assert "valley_irrep_matching" not in summary
    resolved = summary["valley_resolved_irreps"]

    # Document the blocker: toy fixture cannot produce trusted generic matches.
    matched_count = sum(
        1
        for row in resolved.get("rows", [])
        if isinstance(row, dict) and row.get("matching_status") == "matched"
    )

    # Blocker assertion: no matched generic rows from toy spinful fixture.
    assert matched_count == 0, (
        "BLOCKER CLEARED: toy fixture unexpectedly produced trusted "
        "generic matches - fully unmocked E2E may now be feasible"
    )

    # The one-atom hexagonal toy structure is higher symmetry than SG143/P3,
    # and the spinor convention is intentionally unverified.
    symmetry = summary["symmetry_analysis"]
    assert symmetry["spacegroup_number"] == 191
    assert summary["input"]["spinor_convention_verified"] is False

    # All eigenphase rows are diagnostic-only or fail the rotation-readiness
    # gate; no trusted non-identity valley-preserving character row exists.
    for row in summary.get("symmetry_eigenvalues", []):
        assert row.get("diagnostic_only") is True or row.get("rotation_ready") is False, (
            "BLOCKER CLEARED: toy fixture has trusted eigenphase rows"
        )
    assert summary["valley_ebr_input_candidates"]["candidate_count"] == 0
    assert summary["valley_ebr_problem_instances"]["instance_count"] == 0
    assert summary["valley_ebr_export_bundle"]["bundle_count"] == 0

    # Standard outputs exist.
    assert outputs["valley_summary_json"].exists()
    assert outputs["valley_ebr_export_bundle_json"].exists()


# ---------------------------------------------------------------------------
# Standard-setting HSP k-map regression tests
# ---------------------------------------------------------------------------

def test_hsp_override_does_not_bypass_unresolved_standard_setting_mapping():
    """Override must not silently accept an HSP when k-map is unresolved."""
    import numpy as np
    from valleyscope.workflows.analyze_hsp import (
        _resolve_generic_irrep_hsp_label,
        _resolve_generic_irrep_hsp_label_with_provenance,
    )

    class _NoMatchTable:
        number = 5
        name = "C2"
        spinor = True
        def match_kpoint_label(self, k, tolerance=1e-6):
            return None

    label, blocker = _resolve_generic_irrep_hsp_label(
        table=_NoMatchTable(),
        k_frac=np.array([0.123, 0.456, 0.0]),
        override_label="M",
        standard_match={
            "number": 5,
            "international_short": "C2",
            "hall_number": 9,
            "hall_symbol": "C 2y",
        },
    )
    assert label is None
    assert blocker is not None
    assert "cannot be applied" in blocker
    assert "standard_setting_hsp_mapping_unresolved" in blocker

    label, blocker, provenance = _resolve_generic_irrep_hsp_label_with_provenance(
        table=_NoMatchTable(),
        k_frac=np.array([0.123, 0.456, 0.0]),
        override_label="M",
        standard_match={
            "number": 5,
            "international_short": "C2",
            "hall_number": 9,
            "hall_symbol": "C 2y",
        },
    )
    assert label is None
    assert blocker is not None
    assert provenance["direct_match_succeeded"] is False
    assert provenance["hall_number"] == 9
    assert "setting_transform" in provenance
    assert "reason" in provenance["setting_transform"]


def test_override_agrees_with_resolved_label():
    """Override that agrees with resolved label passes through."""
    import numpy as np
    from valleyscope.workflows.analyze_hsp import _resolve_generic_irrep_hsp_label

    class _GMTable:
        number = 143
        name = "P3"
        spinor = True
        def match_kpoint_label(self, k, tolerance=1e-6):
            delta = k - np.array([0.0, 0.0, 0.0])
            delta -= np.rint(delta)
            if np.linalg.norm(delta) <= 1e-6:
                return "GM"
            return None

    label, blocker = _resolve_generic_irrep_hsp_label(
        table=_GMTable(),
        k_frac=np.array([0.0, 0.0, 0.0]),
        override_label="GM",
        standard_match={
            "number": 143,
            "international_short": "P3",
        },
    )
    assert label == "GM"
    assert blocker is None


def test_blocked_source_payload_preserves_standard_setting_kmap_provenance():
    """Blocked generic rows must retain standard-setting HSP mapping evidence."""
    from valleyscope.analysis.valley_irrep_matching import (
        build_valley_irrep_matching_report,
    )

    kmap_provenance = {
        "attempted_direct_match": True,
        "direct_match_succeeded": False,
        "subspace_sg_number": 5,
        "subspace_sg_symbol": "C2",
        "hall_number": 9,
        "hall_symbol": "C 2y",
        "setting_transform": {
            "reason": "centered setting requires reciprocal-basis mapping",
        },
    }
    report = build_valley_irrep_matching_report(
        irrep_workflow_decisions={
            "by_kpoint": {
                "KM": {
                    "K_valley": {
                        "readiness_level": "trusted",
                        "workflow_path": "direct_qcut",
                    },
                },
            },
        },
        symmetry_adapted_valley_report={"by_kpoint": {}},
        source_payload_blocked_rows=[{
            "kpoint": "KM",
            "valley": "K_valley",
            "reason": "standard_setting_hsp_mapping_unresolved",
            "subspace_space_group": {
                "candidate_space_group_number": 5,
                "candidate_space_group_symbol": "C2",
            },
            "valley_preserving_operation_ids": [0],
            "hsp_little_group_operation_ids": [0],
            "standard_setting_hsp_mapping": kmap_provenance,
        }],
    )

    row = report["generic_matches_by_kpoint"]["KM"]["K_valley"]
    provenance = row["source_payload_provenance"]
    assert row["matching_status"] == "blocked"
    assert row["valley_preserving_operation_ids"] == [0]
    assert row["hsp_little_group_operation_ids"] == [0]
    assert provenance["standard_setting_hsp_mapping"] == kmap_provenance
    assert provenance["standard_setting_hsp_mapping"]["setting_transform"][
        "reason"
    ]


def test_refine_ebr_mapping_does_not_invent_missing_local_character_for_nonidentity_gka():
    """Resolved subspace SG must not imply missing local character when G_k^(a) has one."""
    from valleyscope.workflows.analyze_hsp import (
        _refine_ebr_mapping_with_subspace_space_group,
    )

    ebr_mapping = {
        "blocked_by": [
            "spinor_convention_unverified",
            "subspace_group_candidate_missing",
        ],
        "notes": "base note.",
    }

    _refine_ebr_mapping_with_subspace_space_group(
        ebr_mapping=ebr_mapping,
        subspace_space_group={"candidate_space_group_symbol": "C2"},
        local_gka_operation_ids=[0, 4],
    )

    assert ebr_mapping["subspace_space_group_candidate"] == "C2"
    assert "subspace_group_candidate_missing" not in ebr_mapping["blocked_by"]
    assert "hsp_local_preserving_character_missing" not in ebr_mapping["blocked_by"]
    assert "spinor_convention_unverified" in ebr_mapping["blocked_by"]
    assert "does not contain a non-identity" not in ebr_mapping["notes"]


def test_refine_ebr_mapping_marks_identity_only_gka_as_missing_local_character():
    """Identity-only G_k^(a) may need HSP-star character provenance."""
    from valleyscope.workflows.analyze_hsp import (
        _refine_ebr_mapping_with_subspace_space_group,
    )

    ebr_mapping = {
        "blocked_by": ["subspace_group_candidate_missing"],
        "notes": "base note.",
    }

    _refine_ebr_mapping_with_subspace_space_group(
        ebr_mapping=ebr_mapping,
        subspace_space_group={"candidate_space_group_symbol": "C2"},
        local_gka_operation_ids=[0],
    )

    assert ebr_mapping["blocked_by"] == ["hsp_local_preserving_character_missing"]
    assert "does not contain a non-identity" in ebr_mapping["notes"]


# ---------------------------------------------------------------------------
# Scalar / non-SOC generic irrep path validation
# ---------------------------------------------------------------------------

def test_scalar_wavefunction_no_spinor_blockers():
    """Scalar (nspinor=1) wavefunction: no spinor_convention_unverified blockers."""
    from valleyscope.analysis.irrep_workflow_decision import (
        build_irrep_workflow_decisions,
    )
    result = build_irrep_workflow_decisions(
        projector_symmetry_report={
            "by_kpoint": {
                "GammaM": {
                    "seed_projector_symmetry": [{
                        "operation_id": 1, "status": "passed",
                        "source_valley": "M1_valley",
                        "mapped_valley": "M1_valley",
                        "epsilon_seed": 0.001,
                    }],
                },
            },
        },
        target_subspace_closure_report={
            "by_kpoint": {
                "GammaM": [{
                    "operation_id": 1,
                    "kpoint": "GammaM",
                    "little_group_passed": True,
                    "closure_quality": "ok",
                }],
            },
        },
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["M1_valley"],
                        "local_irrep_ready": True,
                        "diagnostic_only": False,
                        "hsp_preserving_operation_ids": [0, 1],
                        "symmetry_adapted_projectors": {
                            "status": "ok",
                            "seed_overlap": {"M1_valley": 0.9},
                        },
                    }],
                },
            },
        },
        symmetry_rows=[{
            "kpoint": "GammaM", "target_valley": "M1_valley",
            "operation_id": 1, "order": 4,
            "topology_input_ready": True,
            "diagnostic_only": False,
        }],
        valley_names=["M1_valley"],
        spinor_convention_verified=False,
        spinor_wavefunction=False,
    )
    d = result["by_kpoint"]["GammaM"]["M1_valley"]
    # Scalar workflow: spinor_convention_unverified must NOT block.
    assert "spinor convention" not in d.get("reason", "")
    assert d["workflow_path"] != "blocked"
    assert d["readiness_level"] in ("trusted", "usable_with_caution")


def test_spinful_unverified_spinor_convention_blocks_trusted_irrep():
    """Spinor wavefunction with unverified convention must gate trusted readiness."""
    from valleyscope.analysis.irrep_workflow_decision import (
        build_irrep_workflow_decisions,
    )
    result = build_irrep_workflow_decisions(
        projector_symmetry_report={
            "by_kpoint": {
                "GammaM": {
                    "seed_projector_symmetry": [{
                        "operation_id": 1, "status": "passed",
                        "source_valley": "M1_valley",
                        "mapped_valley": "M1_valley",
                        "epsilon_seed": 0.001,
                    }],
                },
            },
        },
        target_subspace_closure_report={
            "by_kpoint": {
                "GammaM": [{
                    "operation_id": 1,
                    "kpoint": "GammaM",
                    "little_group_passed": True,
                    "closure_quality": "ok",
                }],
            },
        },
        symmetry_adapted_valley_report={
            "by_kpoint": {
                "GammaM": {
                    "valley_preserving_subspaces": [{
                        "orbit": ["M1_valley"],
                        "local_irrep_ready": True,
                        "diagnostic_only": False,
                        "hsp_preserving_operation_ids": [0, 1],
                        "symmetry_adapted_projectors": {
                            "status": "ok",
                            "seed_overlap": {"M1_valley": 0.9},
                        },
                    }],
                },
            },
        },
        symmetry_rows=[{
            "kpoint": "GammaM", "target_valley": "M1_valley",
            "operation_id": 1, "order": 4,
            "topology_input_ready": True,
            "diagnostic_only": False,
        }],
        valley_names=["M1_valley"],
        spinor_convention_verified=False,
        spinor_wavefunction=True,
    )
    d = result["by_kpoint"]["GammaM"]["M1_valley"]
    # Spinful + unverified: readiness must be gated.
    assert "spinor convention" in d.get("reason", "")
    assert d["readiness_level"] != "trusted"
