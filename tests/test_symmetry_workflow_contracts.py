"""Symmetry workflow contracts: symmetry operation detection, symmetry
eigenvalues, HSP little group, valley-preserving subgroup, valley-changing
operation inventory, and subspace representation quality workflow tests."""

import csv
import importlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from valleyscope.io.config import load_config
from valleyscope.workflows.analyze_hsp import analyze_hsp

from tests.helpers_io_workflow import (
    write_fixture,
    write_config,
    write_simple_poscar,
    write_square_poscar,
    _p3_fake_symmetry_payload,
)


def test_analyze_hsp_writes_symmetry_operation_detection_report(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    structure = tmp_path / "CONTCAR"
    write_fixture(h5_path)
    write_simple_poscar(structure)
    write_config(config_path, h5_path, out_dir)

    outputs = analyze_hsp(config_path)
    report = json.loads(outputs["symmetry_report_json"].read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    assert report["operation_detection_backend"] == "spglib"
    assert report["structure_file"] == str(structure)
    assert report["symprec"] == pytest.approx(1.0e-3)
    assert report["angle_tolerance"] == pytest.approx(-1.0)
    assert "symprec_scan_summary" in report
    assert {"symprec", "spacegroup_number", "international", "n_operations", "n_candidate_rotations", "order_counts"} <= set(
        report["symprec_scan_summary"][0]
    )
    assert "detected_operations" in report
    assert "candidate_rotations" in report
    assert "operations" not in report
    assert report["little_group_check"]["required"] is True
    assert report["valley_preservation_check"]["required"] is True


def test_symmetry_eigenvalues_use_valley_adapted_basis_and_write_diagnostics(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    structure = tmp_path / "CONTCAR"
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3)
        meta["spinor"] = False
        meta["source"] = "toy"
        meta["vasp_band_index_base"] = 1
        kp = h5.create_group("kpoints").create_group("0")
        kp["name"] = "GammaM"
        kp["frac"] = np.array([0.0, 0.0, 0.0])
        kp["cart"] = np.array([0.0, 0.0, 0.0])
        q_cart = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        kp["g_vectors_frac"] = q_cart
        kp["g_vectors_cart"] = q_cart
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        k_state = np.array([inv_sqrt2, inv_sqrt2, 0.0, 0.0])
        kp_state = np.array([0.0, 0.0, inv_sqrt2, inv_sqrt2])
        kp["coefficients"] = np.array(
            [
                [[*(inv_sqrt2 * (k_state + kp_state))]],
                [[*(inv_sqrt2 * (k_state - kp_state))]],
            ],
            dtype=np.complex128,
        )
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])
    write_square_poscar(structure)

    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0},
        "monolayer_lattices": {
            "default": {"reciprocal_cart": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]}
        },
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [
                {"name": "K_plus", "cart": [1.0, 0.0, 0.0]},
                {"name": "K_minus", "cart": [-1.0, 0.0, 0.0]},
                {"name": "Kp_plus", "cart": [0.0, 1.0, 0.0]},
                {"name": "Kp_minus", "cart": [0.0, -1.0, 0.0]},
            ],
        },
        "valley_subspaces": [
            {"name": "K_sector", "centers": ["K_plus", "K_minus"]},
            {"name": "Kp_sector", "centers": ["Kp_plus", "Kp_minus"]},
        ],
        "projection": {
            "use_2d_momentum_only": True,
            "qcut_mode": "absolute",
            "qcut_Ainv": 0.25,
            "overlap_policy": "warn_exclude",
            "thresholds": {"W_val_min": 0.8, "P_v_clean": 0.95, "P_v_approx": 0.85},
        },
        "symmetry": {
            "operations": {"mode": "auto", "structure_file": str(structure), "backend": "spglib"},
            "tolerance": {"symprec": 1.0e-5, "angle_tolerance": -1.0},
            "filters": {"proper_rotations_only": True, "allowed_orders": [2, 4], "rotation_order": 2},
        },
        "output": {"directory": str(out_dir), "profile": "debug", "write_json": True, "write_csv": True, "write_hdf5_basis_transform": True},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = analyze_hsp(config_path)

    assert "rotation_eigenvalues_csv" not in outputs
    assert "little_group_eigenvalues_csv" not in outputs
    assert "little_group_representations_json" not in outputs
    assert not (out_dir / "rotation_eigenvalues.csv").exists()
    assert not (out_dir / "little_group_eigenvalues.csv").exists()
    assert not (out_dir / "little_group_representations.json").exists()
    with outputs["symmetry_eigenvalues_csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {
        "basis",
        "nearest_root_of_unity",
        "root_deviation",
        "rotation_ready",
        "topology_input_ready",
        "topology_ready",
        "spinor_rotation_applied",
        "spinor_convention_verified",
        "spinor_convention",
        "spinor_benchmark",
        "diagnostic_only",
        "D_valley_offdiag_norm",
        "reason",
        "valley_eta",
    } <= set(rows[0])
    assert any(row["basis"] == "valley_adapted" for row in rows)
    assert all(row["basis"] != "raw_vasp_final" for row in rows)

    with h5py.File(outputs["diagnostics_h5"], "r") as h5:
        assert "rotation" not in h5
        assert "symmetry_representations/GammaM" in h5
        operation_groups = list(h5["symmetry_representations/GammaM"].values())
        assert operation_groups
        assert any("D_valley" in group for group in operation_groups)
        assert all("D_raw" in group for group in operation_groups)
        assert all("operation_order" in group for group in operation_groups)
        assert all("rotation_cart" in group for group in operation_groups)
        assert all("translation_cart" in group for group in operation_groups)
        assert all("basis" in group.attrs for group in operation_groups)
        assert all("spinor_rotation_applied" in group for group in operation_groups)
        assert all("spinor_convention_verified" in group for group in operation_groups)
        assert all("spinor_convention" in group.attrs for group in operation_groups)
        assert all("spinor_benchmark" in group.attrs for group in operation_groups)
        assert all("rotation_ready" in group for group in operation_groups)
        assert all("topology_input_ready" in group for group in operation_groups)
        assert all("diagnostic_only" in group for group in operation_groups)
        assert all("D_valley_offdiag_norm" in group for group in operation_groups)
        assert all("root_deviation" in group for group in operation_groups)
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "rejected" in summary_text
    assert (
        "not in little group" in summary_text
        or "valley-changing" in summary_text
        or "not valley preserving" in summary_text
    )
    assert "topology_input_ready" in summary_text
    assert "Symmetry eigenvalues" in summary_text
    assert "Rotation eigenvalues" not in summary_text
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert any("topology_input_ready" in row for row in summary["symmetry_eigenvalues"])
    assert "spinor_convention" in summary["input"]
    assert "spinor_benchmark" in summary["symmetry_eigenvalues"][0]
    assert outputs["symmetry_eigenvalues_csv"].exists()
    assert "symmetry_eigenvalues_csv" in summary["output_files"]


def test_symmetry_eigenvalues_csv_is_header_only_when_no_rows(tmp_path, monkeypatch):
    importlib_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    def fake_prepare_symmetry_payload(config, monolayer_recip):
        return {
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "fake-CONTCAR",
            "spacegroup_number": 150,
            "international": "P321",
            "symmetry_eigenvalue_enabled": True,
            "requested_rotation_order": "auto",
            "resolved_rotation_order": 3,
            "detected_operation_count": 0,
            "detected_operations": [],
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
        }

    def fake_symmetry_diagnostic(**kwargs):
        return []

    monkeypatch.setattr(importlib_module, "_prepare_symmetry_payload", fake_prepare_symmetry_payload)
    monkeypatch.setattr(importlib_module, "symmetry_eigenvalue_diagnostics_for_kpoint", fake_symmetry_diagnostic)

    outputs = importlib_module.analyze_hsp(config_path)

    assert outputs["symmetry_eigenvalues_csv"].exists()
    with outputs["symmetry_eigenvalues_csv"].open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert {"kpoint", "operation_id", "root_deviation", "D_valley_offdiag_norm"} <= set(reader.fieldnames or [])
        assert list(reader) == []
    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "symmetry_eigenvalues_csv" in summary["output_files"]
    subgroup_report = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]
    assert subgroup_report["status"] in ("per_valley_preserving_subgroups_computed", "operation_set_only")
    if subgroup_report["status"] == "per_valley_preserving_subgroups_computed":
        assert subgroup_report["all_valley_intersection"]["operation_count"] == 0
    else:
        assert subgroup_report["standard_group_match_status"] == "not_attempted"
        assert subgroup_report["by_kpoint"]["GammaM"]["closure_status"] == "empty"


def test_subspace_representation_quality_standalone_json_default_off(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    quality_row = {
        "target_valley": "K_valley",
        "operation_id": 0,
        "diagnosis": "ok",
        "local_unitarity_error": 1.0e-8,
    }
    symmetry_adapted_valley_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [
                    {
                        "reference_valley": "K_valley",
                        "subspace_representation_quality": {"rows": [quality_row]},
                    }
                ]
            }
        }
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
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
    )

    assert config.symmetry_adapted_valley.write_subspace_representation_quality is False
    assert "subspace_representation_quality_json" not in outputs
    assert not (out_dir / "subspace_representation_quality.json").exists()
    assert outputs["symmetry_adapted_valley_analysis_json"].exists()
    analysis_payload = json.loads(outputs["symmetry_adapted_valley_analysis_json"].read_text(encoding="utf-8"))
    assert analysis_payload["by_kpoint"]["GammaM"]["valley_preserving_subspaces"][0][
        "subspace_representation_quality"
    ]["rows"] == [quality_row]
    summary_payload = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "subspace_representation_quality_json" not in summary_payload["output_files"]
    assert summary_payload["symmetry_adapted_valley_analysis"] == symmetry_adapted_valley_report


def test_subspace_representation_quality_standalone_json_opt_in(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw.setdefault("analysis", {})["symmetry_adapted_valley"] = {
        "write_subspace_representation_quality": True,
    }
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    from valleyscope.reports.analysis_outputs import write_analysis_outputs

    quality_row = {
        "target_valley": "K_valley",
        "operation_id": 0,
        "diagnosis": "unitarity_failed",
        "local_unitarity_error": 2.0e-2,
    }
    symmetry_adapted_valley_report = {
        "by_kpoint": {
            "GammaM": {
                "valley_preserving_subspaces": [
                    {
                        "reference_valley": "K_valley",
                        "subspace_representation_quality": {"rows": [quality_row]},
                    }
                ]
            }
        }
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
        symmetry_adapted_valley_report=symmetry_adapted_valley_report,
    )

    assert config.symmetry_adapted_valley.write_subspace_representation_quality is True
    assert outputs["subspace_representation_quality_json"] == out_dir / "subspace_representation_quality.json"
    quality_payload = json.loads(outputs["subspace_representation_quality_json"].read_text(encoding="utf-8"))
    assert quality_payload["status"] == "quality_issues_detected"
    assert quality_payload["rows"] == [{**quality_row, "kpoint": "GammaM"}]
    summary_payload = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert "subspace_representation_quality_json" in summary_payload["output_files"]


def test_workflow_requests_all_valley_preserving_little_group_operations(tmp_path, monkeypatch):
    importlib_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    calls: list[bool] = []

    def fake_prepare_symmetry_payload(config, monolayer_recip):
        return {
            "status": "ok",
            "operation_detection_backend": "spglib",
            "structure_file": "fake-CONTCAR",
            "spacegroup_number": 150,
            "international": "P321",
            "symmetry_eigenvalue_enabled": True,
            "requested_rotation_order": "auto",
            "resolved_rotation_order": 3,
            "detected_operation_count": 0,
            "detected_operations": [],
            "candidate_rotations": [],
            "symprec_scan_summary": [],
            "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
            "valley_preservation_check": {"required": True, "status": "completed"},
        }

    def fake_symmetry_diagnostic(**kwargs):
        calls.append(bool(kwargs.get("generators_only", False)))
        return []

    monkeypatch.setattr(importlib_module, "_prepare_symmetry_payload", fake_prepare_symmetry_payload)
    monkeypatch.setattr(importlib_module, "symmetry_eigenvalue_diagnostics_for_kpoint", fake_symmetry_diagnostic)

    importlib_module.analyze_hsp(config_path)

    assert calls == [False]


def test_workflow_rejects_hdf5_spinor_flag_coefficient_shape_conflict(tmp_path):
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    with pytest.raises(
        ValueError,
        match="metadata/spinor conflicts with coefficient nspinor",
    ):
        analyze_hsp(config_path)


def test_workflow_writes_symmetry_consistency_report_when_no_seed_data(tmp_path, monkeypatch):
    importlib_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(importlib_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(
        importlib_module,
        "symmetry_eigenvalue_diagnostics_for_kpoint",
        lambda **kwargs: [],
    )

    outputs = importlib_module.analyze_hsp(config_path)

    cov_path = outputs["projector_symmetry_report_json"]
    symmetry_consistency = json.loads(cov_path.read_text(encoding="utf-8"))
    assert symmetry_consistency["status"] == "no_data"

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    assert summary["projector_symmetry"]["status"] == "no_data"
