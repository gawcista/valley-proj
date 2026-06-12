"""Irrep workflow readiness: matching decisions, readiness gates,
diagnostic-only rows, off-diagonal valley mixing, seed projector
symmetry-consistency failure, and rotation threshold plumbing."""

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
    write_square_poscar,
    _p3_fake_symmetry_payload,
)


# ---------------------------------------------------------------------------
# helpers shared within this file (only used by irrep-readiness tests)
# ---------------------------------------------------------------------------

def _p3_d_valley_matrices(*, operation_2_offdiag: float = 0.0) -> dict[int, np.ndarray]:
    """Return diagonal D_valley matrices at GM for P3 irrep matching test.

    In the valley-adapted basis:
      state 0: C3 eigenvalue = exp(-i*pi/3),  C3^2 eigenvalue = exp(+i*pi/3)
      state 1: C3 eigenvalue = exp(+i*pi/3),  C3^2 eigenvalue = exp(-i*pi/3)

    These match -GM5 and -GM6 respectively.
    """
    matrices = {
        0: np.eye(2, dtype=np.complex128),  # identity
        1: np.diag([np.exp(-1j * np.pi / 3.0), np.exp(+1j * np.pi / 3.0)]),  # C3
        2: np.diag([np.exp(+1j * np.pi / 3.0), np.exp(-1j * np.pi / 3.0)]),  # C3^2
    }
    if operation_2_offdiag > 0.0:
        matrices[2] = matrices[2].copy()
        matrices[2][0, 1] = operation_2_offdiag
        matrices[2][1, 0] = operation_2_offdiag * 0.7
    return matrices


def _ready_character_rows(kpoint: str, *, operation_2_state_1_ready: bool = True) -> list[dict]:
    """Return aggregate (trace-level) character rows for P3 irrep matching.

    State-level characters are now collected from D_valley diagonal via
    the representation_payload, not from eigenvalue ordering in these rows.
    The eigenvalue_real/imag fields here are deliberately scrambled to
    verify that they are NOT used for state irrep assignment.
    """
    rows = []
    for operation_id in [1, 2]:
        for state_index in [0, 1]:
            ready = operation_id != 2 or state_index != 1 or operation_2_state_1_ready
            # Deliberately wrong eigenvalue ordering to catch any regression
            # that would read eigenvalues instead of D_valley diagonal
            wrong_eigenvalue = np.exp(1j * np.pi * (state_index + operation_id) / 7.0)
            rows.append(
                {
                    "kpoint": kpoint,
                    "operation_id": operation_id,
                    "kind": "C3",
                    "order": 3,
                    "basis": "valley_adapted",
                    "state_index": state_index,
                    "phase_2pi": 0.0,
                    "nearest_root_of_unity": "1",
                    "root_deviation": 0.0,
                    "rotation_ready": ready,
                    "D_valley_offdiag_norm": 0.0,
                    "eigenvalue_real": float(wrong_eigenvalue.real),
                    "eigenvalue_imag": float(wrong_eigenvalue.imag),
                    "character_valley": "1.000000+0.000000j" if state_index == 0 else "",
                    "character_raw": "",
                    "little_group_passed": True,
                    "valley_preserving": True,
                    "topology_input_ready": ready,
                    "diagnostic_only": not ready,
                    "reason": "" if ready else "root deviation too large",
                }
            )
    return rows


def _fake_diagnostics_with_dvalley(kpoint_name, representation_payload, **kwargs):
    """Mock symmetry_eigenvalue_diagnostics_for_kpoint that populates
    representation_payload with D_valley and returns aggregate rows."""
    d_valleys = _p3_d_valley_matrices()
    kp_repr = representation_payload.setdefault(kpoint_name, {})
    for op_id, d_valley in d_valleys.items():
        kp_repr[f"operation_{op_id}"] = {
            "D_valley": d_valley,
            "unitarity_deviation": 0.0,
            "mapping_miss_count": 0,
        }
    return _ready_character_rows(kpoint_name)


def _fake_diagnostics_with_dvalley_incomplete(kpoint_name, representation_payload, **kwargs):
    """Same as _fake_diagnostics_with_dvalley but operation 2 has one non-ready row."""
    d_valleys = _p3_d_valley_matrices()
    kp_repr = representation_payload.setdefault(kpoint_name, {})
    for op_id, d_valley in d_valleys.items():
        kp_repr[f"operation_{op_id}"] = {
            "D_valley": d_valley,
            "unitarity_deviation": 0.0,
            "mapping_miss_count": 0,
        }
    return _ready_character_rows(kpoint_name, operation_2_state_1_ready=False)


def _fake_diagnostics_dvalley_mixed(kpoint_name, representation_payload, **kwargs):
    """D_valley for operation 2 has large off-diagonal mixing."""
    d_valleys = _p3_d_valley_matrices(operation_2_offdiag=0.5)
    kp_repr = representation_payload.setdefault(kpoint_name, {})
    for op_id, d_valley in d_valleys.items():
        kp_repr[f"operation_{op_id}"] = {
            "D_valley": d_valley,
            "unitarity_deviation": 0.0,
            "mapping_miss_count": 0,
        }
    return _ready_character_rows(kpoint_name)


# ---------------------------------------------------------------------------
# irrep matching workflow
# ---------------------------------------------------------------------------

def test_workflow_passes_irrep_weight_tol_to_matching(tmp_path, monkeypatch):
    importlib_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    write_config(config_path, h5_path, out_dir)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["rotation"] = {"irrep_weight_tol": 2.5e-4}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    symmetry_payload = {
        "status": "ok",
        "operation_detection_backend": "spglib",
        "structure_file": "fake-CONTCAR",
        "symmetry_eigenvalue_enabled": True,
        "detected_operations": [],
        "candidate_rotations": [],
        "little_group_check": {"required": True, "status": "evaluated_per_kpoint"},
        "valley_preservation_check": {"required": True, "status": "completed"},
    }
    captured: list[float] = []

    def fake_add_valley_irrep_results(**kwargs):
        captured.append(kwargs["tolerance"])
        return {}

    monkeypatch.setattr(importlib_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: dict(symmetry_payload))
    monkeypatch.setattr(importlib_module, "symmetry_eigenvalue_diagnostics_for_kpoint", lambda **kwargs: [])
    monkeypatch.setattr(importlib_module, "add_valley_irrep_results", fake_add_valley_irrep_results)

    importlib_module.analyze_hsp(config_path)

    assert captured == [pytest.approx(2.5e-4)]


def test_workflow_writes_irrep_results_when_characters_are_ready(tmp_path, monkeypatch):
    importlib_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(importlib_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(
        importlib_module,
        "symmetry_eigenvalue_diagnostics_for_kpoint",
        _fake_diagnostics_with_dvalley,
    )

    outputs = importlib_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    matching = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]
    assert matching["character_matching_status"] == "matched"
    result = matching["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    assert result["status"] == "matched"
    assert result["table_kpoint_label"] == "GM"
    assert result["irrep_multiplicities"] == {"-GM5": 1, "-GM6": 1}
    assert result["state_irrep_assignment_status"] == "matched"
    assert result["state_irrep_results"] == [
        {
            "state_index": 0,
            "status": "matched",
            "irrep_label": "-GM5",
            "computed_characters": {
                "1": "1.000000+0.000000j",
                "2": "0.500000-0.866025j",
                "3": "0.500000+0.866025j",
            },
            "irrep_multiplicities": {"-GM5": 1},
            "missing_table_operation_indices": [],
            "failure_reasons": [],
        },
        {
            "state_index": 1,
            "status": "matched",
            "irrep_label": "-GM6",
            "computed_characters": {
                "1": "1.000000+0.000000j",
                "2": "0.500000+0.866025j",
                "3": "0.500000-0.866025j",
            },
            "irrep_multiplicities": {"-GM6": 1},
            "missing_table_operation_indices": [],
            "failure_reasons": [],
        },
    ]
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "GammaM/K_valley: -GM5 x 1, -GM6 x 1" in summary_text
    assert "GammaM/K_valley state irreps: state 0 -> -GM5, state 1 -> -GM6" in summary_text


def test_workflow_keeps_irrep_results_incomplete_when_an_operation_has_non_ready_rows(tmp_path, monkeypatch):
    importlib_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(importlib_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(
        importlib_module,
        "symmetry_eigenvalue_diagnostics_for_kpoint",
        _fake_diagnostics_with_dvalley_incomplete,
    )

    outputs = importlib_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    matching = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]
    assert matching["character_matching_status"] == "incomplete"
    result = matching["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    assert result["status"] == "missing_characters"
    assert result["irrep_multiplicities"] == {}
    assert result["missing_table_operation_indices"] == [3]
    assert result["state_irrep_assignment_status"] == "incomplete"


def test_workflow_keeps_irrep_results_incomplete_when_symmetry_consistency_fails(tmp_path, monkeypatch):
    importlib_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    symmetry_consistency_report = {
        "status": "symmetry_consistency_failures_detected",
        "warn_tol": 0.01,
        "fail_tol": 0.1,
        "by_kpoint": {
            "GammaM": {
                "seed_projector_symmetry": [
                    {
                        "operation_id": 1,
                        "source_valley": "K_valley",
                        "mapped_valley": "K_valley",
                        "epsilon_seed": 0.5,
                        "little_group_passed": True,
                        "status": "failed",
                        "reason": "",
                    },
                    {
                        "operation_id": 2,
                        "source_valley": "K_valley",
                        "mapped_valley": "K_valley",
                        "epsilon_seed": 0.4,
                        "little_group_passed": True,
                        "status": "failed",
                        "reason": "",
                    },
                ]
            }
        },
    }

    def fake_diagnostics_with_target_valleys(kpoint_name, representation_payload, **kwargs):
        rows = _fake_diagnostics_with_dvalley(kpoint_name, representation_payload, **kwargs)
        return [
            {**row, "target_valley": valley_name}
            for valley_name in ("K_valley", "Kp_valley")
            for row in rows
        ]

    monkeypatch.setattr(importlib_module, "_prepare_symmetry_payload", lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(
        importlib_module,
        "symmetry_eigenvalue_diagnostics_for_kpoint",
        fake_diagnostics_with_target_valleys,
    )
    monkeypatch.setattr(
        importlib_module,
        "build_projector_symmetry_report",
        lambda **kwargs: symmetry_consistency_report,
    )

    outputs = importlib_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    matching = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]
    assert matching["character_matching_status"] == "incomplete"
    result = matching["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    assert result["status"] == "missing_characters"
    rows = [
        row for row in summary["symmetry_eigenvalues"]
        if row["operation_id"] == 1 and row["target_valley"] == "K_valley"
    ]
    assert rows
    assert all(row["diagnostic_only"] is True for row in rows)
    assert all(row["topology_input_ready"] is False for row in rows)
    assert all(row["projector_symmetry_status"] == "failed" for row in rows)


# ---------------------------------------------------------------------------
# off-diagonal valley mixing gate
# ---------------------------------------------------------------------------

def test_state_irrep_rejected_when_dvalley_has_offdiagonal_mixing(tmp_path, monkeypatch):
    """Mixing gate: D_valley with large off-diagonal -> no state label."""
    importlib_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(importlib_module, "_prepare_symmetry_payload",
                        lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(importlib_module, "symmetry_eigenvalue_diagnostics_for_kpoint",
                        _fake_diagnostics_dvalley_mixed)

    outputs = importlib_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    matching = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]
    result = matching["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    # Aggregate characters still pass (trace = sum of eigenvalues, even with off-diagonal)
    assert result["status"] == "matched"
    # But state-level is incomplete because C3^2 D_valley has mixing
    assert result["state_irrep_assignment_status"] == "incomplete"
    assert len(result["state_irrep_results"]) > 0
    # At least one state result should not be "matched"
    assert any(s.get("status") != "matched" for s in result["state_irrep_results"])


def test_state_irrep_ignores_eigenvalue_ordering_uses_dvalley_diagonal(tmp_path, monkeypatch):
    """Regression test: eigenvalue rows contain deliberately wrong values,
    but state irrep follows D_valley diagonal entries correctly."""
    importlib_module = importlib.import_module("valleyscope.workflows.analyze_hsp")
    h5_path = tmp_path / "wf.h5"
    config_path = tmp_path / "config.yaml"
    out_dir = tmp_path / "out"
    write_fixture(h5_path)
    with h5py.File(h5_path, "r+") as h5:
        h5["metadata/spinor"][()] = True
    write_config(config_path, h5_path, out_dir)

    monkeypatch.setattr(importlib_module, "_prepare_symmetry_payload",
                        lambda config, monolayer_recip: _p3_fake_symmetry_payload())
    monkeypatch.setattr(importlib_module, "symmetry_eigenvalue_diagnostics_for_kpoint",
                        _fake_diagnostics_with_dvalley)

    outputs = importlib_module.analyze_hsp(config_path)

    summary = json.loads(outputs["valley_summary_json"].read_text(encoding="utf-8"))
    result = summary["symmetry_analysis"]["valley_preserving_subgroup_report"]["irrep_matching"]["irrep_results_by_kpoint"]["GammaM"]["K_valley"]
    assert result["state_irrep_assignment_status"] == "matched"
    # State 0 characters must equal D_valley diagonal: identity=1, C3=exp(-i*pi/3), C3^2=exp(+i*pi/3)
    s0 = result["state_irrep_results"][0]
    assert s0["irrep_label"] == "-GM5"
    assert s0["status"] == "matched"
    # The eigenvalue rows contain wrong values; state assignment uses D_valley diagonal
    summary_text = outputs["valley_summary_txt"].read_text(encoding="utf-8")
    assert "state 0 -> -GM5" in summary_text
    assert "state 1 -> -GM6" in summary_text


# ---------------------------------------------------------------------------
# rotation threshold plumbing
# ---------------------------------------------------------------------------

def test_rotation_thresholds_from_config_parsed_and_applied(tmp_path):
    """Fix 3: rotation thresholds in YAML config are parsed and used."""
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
        kp["name"] = "GammaM"
        kp["frac"] = np.array([0.0, 0.0, 0.0])
        kp["cart"] = np.array([0.0, 0.0, 0.0])
        q_cart = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        kp["g_vectors_frac"] = q_cart
        kp["g_vectors_cart"] = q_cart
        kp["coefficients"] = np.array(
            [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.complex128,
        )
        kp["energies_eV"] = np.array([0.1, 0.1001])
        kp["band_indices_vasp"] = np.array([101, 102])

    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": ["GammaM"], "iband": [101, 102], "degeneracy_tol_meV": 1.0},
        "monolayer_lattices": {
            "default": {"reciprocal_cart": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]]}
        },
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [
                {"name": "K", "cart": [1.0, 0.0, 0.0]},
                {"name": "Kp", "cart": [-1.0, 0.0, 0.0]},
            ],
        },
        "valley_subspaces": [
            {"name": "K_sector", "centers": ["K"]},
            {"name": "Kp_sector", "centers": ["Kp"]},
        ],
        "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.25, "overlap_policy": "warn_exclude"},
        "symmetry": {
            "operations": {"mode": "auto", "structure_file": str(structure), "backend": "spglib"},
            "tolerance": {"symprec": 1.0e-5, "angle_tolerance": -1.0},
            "filters": {"rotation_order": 2},
        },
        "rotation": {"unitarity_tol": 1.0e-4, "root_deviation_tol": 1.0e-6, "D_valley_offdiag_tol": 1.0e-6},
        "output": {"directory": str(tmp_path / "out"), "profile": "debug"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    outputs = analyze_hsp(config_path)
    # Verify config is parsed
    app_config = load_config(config_path)
    assert app_config.rotation.unitarity_tol == pytest.approx(1.0e-4)
    assert app_config.rotation.root_deviation_tol == pytest.approx(1.0e-6)
    assert app_config.rotation.D_valley_offdiag_tol == pytest.approx(1.0e-6)

    # Verify symmetry eigenvalues were produced (thresholds don't break output)
    with outputs["symmetry_eigenvalues_csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 0

    # Default config (all fields omitted) still works
    config2 = dict(config)
    del config2["rotation"]
    config_path2 = tmp_path / "config2.yaml"
    config_path2.write_text(yaml.safe_dump(config2), encoding="utf-8")
    outputs2 = analyze_hsp(config_path2)
    with outputs2["symmetry_eigenvalues_csv"].open(encoding="utf-8") as handle:
        rows2 = list(csv.DictReader(handle))
    assert len(rows2) > 0
