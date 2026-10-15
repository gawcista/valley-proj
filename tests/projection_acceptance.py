"""Small independently specified projection-only CLI acceptance fixture."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import yaml


def make_projection_inputs(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    h5_path = directory / "selected.h5"
    with h5py.File(h5_path, "w") as h5:
        meta = h5.create_group("metadata")
        lattice = meta.create_group("lattice")
        reciprocal = np.diag([0.5, 0.5, 1.0])
        lattice["reciprocal_cart"] = reciprocal
        lattice["direct_cart"] = 2 * np.pi * np.linalg.inv(reciprocal).T
        meta["spinor"] = False
        meta["source"] = "generated-projection-acceptance"
        meta["vasp_band_index_base"] = 1
        meta["wavecar_nspin"] = 1
        g = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        for index, name in enumerate(["start", "finish"]):
            kp = h5.create_group(f"kpoints/{index}")
            frac = np.array([0.04 * index, 0, 0])
            kp["name"] = name
            kp["frac"] = frac
            kp["cart"] = frac @ reciprocal
            kp["g_vectors_frac"] = g
            kp["g_vectors_cart"] = g @ reciprocal
            kp["band_indices_vasp"] = [10, 11]
            kp["energies_eV"] = [1.0 + 0.1 * index, 1.2 + 0.1 * index]
            kp["coefficients"] = np.array([
                [[np.sqrt(0.1), np.sqrt(0.6), np.sqrt(0.2), np.sqrt(0.1)]],
                [[0, 0, 1, 0]],
            ], dtype=complex)
    config = {
        "input": {"wavefunction_h5": str(h5_path)},
        "monolayer_lattices": {"default": {"reciprocal_cart": np.eye(3).tolist()}},
        "analysis": {"kpoints": ["start", "finish"], "iband": [11, 10]},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name": "Gamma", "cart": [0, 0, 0]},
            {"name": "X", "cart": [0.5, 0, 0]},
            {"name": "Y", "cart": [0, 0.5, 0]},
        ]},
        "valley_subspaces": [{"name": name, "centers": [name]} for name in ["Gamma", "X", "Y"]],
        "projection": {"qcut_mode": "absolute", "qcut_Ainv": 0.1},
        "path": {"ticks": [{"index": 0, "label": "A"}, {"index": 1, "label": "B"}]},
        "plot": {"energy_reference_eV": 1.0},
        "output": {"directory": str(directory / "output")},
    }
    path = directory / "projection.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def verify_projection_result(path: Path) -> dict:
    result = json.loads(path.read_text())
    assert result["document_kind"] == "valley_projection"
    assert result["schema_version"] == "1.0.0"
    assert result["band_indices_vasp"] == [11, 10]
    assert len(result["rows"]) == 4
    first, mixed = result["rows"][:2]
    assert (first["path_index"], first["band_vasp"]) == (0, 11)
    assert first["energy_eV"] == 1.2
    assert first["normalized_valley_weights"] == {"Gamma": 0., "X": 0., "Y": 1.}
    assert mixed["band_vasp"] == 10
    np.testing.assert_allclose(
        [mixed["normalized_valley_weights"][name] for name in ["Gamma", "X", "Y"]],
        [0.1, 0.6, 0.2], rtol=0, atol=1e-14,
    )
    assert abs(mixed["normalized_W_res"] - 0.1) < 1e-14
    assert abs(result["path"]["kpoints"][1]["distance_Ainv"] - 0.02) < 1e-14
    assert result["stages"]["irrep"] == "not_requested"
    assert result["stages"]["ebr"] == "not_requested"
    assert not (path.parent / "valley_summary.json").exists()
    assert (path.parent / "valley_weights.csv").is_file()
    assert (path.parent / "valley_projection.txt").is_file()
    return {"rows": 4, "valleys": 3, "raw_mixed_weights": [0.1, 0.6, 0.2],
            "residual_weight": 0.1, "validation_errors": []}


def run_installed_projection_acceptance() -> dict:
    # The gate checks installed import/console provenance independently. Keep
    # numerical acceptance in the caller's current package import context so
    # the same production fixture also runs in a clean source test environment.
    from valleyscope.cli import main

    with tempfile.TemporaryDirectory(prefix="valleyscope-projection-installed-") as name:
        directory = Path(name)
        config = make_projection_inputs(directory)
        if main(["project-bands", str(config)]) != 0:
            raise AssertionError("Projection CLI acceptance failed")
        return verify_projection_result(directory / "output/valley_projection.json")
