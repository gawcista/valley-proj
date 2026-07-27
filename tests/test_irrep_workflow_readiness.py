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
                    "plane_wave_mapping_complete": ready,
                    "D_block_leakage_norm": 0.0,
                    "eigenvalue_real": float(wrong_eigenvalue.real),
                    "eigenvalue_imag": float(wrong_eigenvalue.imag),
                    "character_valley": "1.000000+0.000000j" if state_index == 0 else "",
                    "character_raw": "",
                    "little_group_passed": True,
                    "valley_preserving": True,
                    "local_irrep_ready": ready,
                    "diagnostic_only": not ready,
                    "reason": "" if ready else "representation evidence blocked",
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




def test_removed_rotation_thresholds_are_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    raw = {
        "input": {"wavefunction_h5": "w.h5"},
        "analysis": {"kpoints": ["G"], "iband": [1]},
        "rotation": {"readiness_preset": "strict"},
        "monolayer_lattices": {"default": {"reciprocal_cart": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [{"name": "K", "cart": [0, 0, 0]}]},
        "valley_subspaces": [{"name": "K_valley", "centers": ["K"]}],
        "output": {"directory": "out"},
    }
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="rotation config block has been removed"):
        load_config(config_path)
