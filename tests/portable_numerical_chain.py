"""Generated spinful numerical acceptance, with no injected trust records.

The P3 structure and six-plane-wave payload are symmetry-consistent test
data, not a DFT calculation or a material model. Two parent valleys fold to
moire Gamma. Each valley carries an equal-amplitude three-wave orbit times
the two spin states; its spatial character is trivial and its spin character
is 2 at identity and 1 at either threefold operation.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

from valleyscope.analysis.database_ingestion_record import (
    load_database_ingestion_record_from_directory,
)
from valleyscope.workflows.analyze_hsp import analyze_hsp


# Independently specified reciprocal stars, not obtained from irrep tables
# or the plane-wave action under test. Rows 0:3 and 3:6 are opposite valleys.
STARS = {
    "GM": ([0, 0, 0], [[1, 1, 0], [-2, 1, 0], [1, -2, 0],
                       [-1, -1, 0], [2, -1, 0], [-1, 2, 0]]),
    "K": ([1 / 3, 1 / 3, 0], [[1, 1, 0], [-3, 1, 0], [1, -3, 0],
                              [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]),
    "KA": ([-1 / 3, -1 / 3, 0], [[1, 1, 0], [-1, 1, 0], [1, -1, 0],
                                 [-1, -1, 0], [3, -1, 0], [-1, 3, 0]]),
    # Only identity preserves M in P3. The opposite valley uses
    # G' = -G - (1,0,0), so the four-state target is TR-closed at this TRIM.
    "M": ([0.5, 0, 0], [[1, 1, 0], [-2, 1, 0], [1, -2, 0],
                         [-2, -1, 0], [1, -1, 0], [-2, 2, 0]]),
}


def write_numerical_fixture(
    root: Path, *, broken_coefficients: bool = False, profile: str = "debug",
) -> Path:
    """Write only physical inputs and ordinary config; return its path."""
    root.mkdir(parents=True, exist_ok=True)
    direct = np.array([[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 8]])
    reciprocal = 2 * np.pi * np.linalg.inv(direct).T
    structure = root / "POSCAR"
    structure.write_text(
        "Generated P3 symmetry fixture\n1.0\n"
        "1 0 0\n-0.5 0.8660254037844386 0\n0 0 8\n"
        "H He\n3 3\nDirect\n"
        "0.173 0.287 0.119\n0.713 0.886 0.119\n0.114 0.827 0.119\n"
        "0.329 0.071 0.437\n0.929 0.258 0.437\n0.742 0.671 0.437\n",
        encoding="utf-8",
    )
    payload = root / "wavefunctions.h5"
    with h5py.File(payload, "w") as h5:
        metadata = h5.create_group("metadata")
        lattice = metadata.create_group("lattice")
        lattice["direct_cart"] = direct
        lattice["reciprocal_cart"] = reciprocal
        metadata["spinor"] = True
        metadata["source"] = "generated_numerical_acceptance"
        # Synthetic RTAG-compatible precision, not extractor provenance.
        metadata["wavecar_rtag"] = 45210
        metadata["vasp_band_index_base"] = 1
        kpoints = h5.create_group("kpoints")
        for index, (label, (frac, grid)) in enumerate(STARS.items()):
            kpoint = kpoints.create_group(str(index))
            kpoint["name"] = label
            kpoint["frac"] = frac
            kpoint["cart"] = np.array(frac) @ reciprocal
            kpoint["g_vectors_frac"] = np.array(grid, dtype=int)
            kpoint["g_vectors_cart"] = np.array(grid) @ reciprocal
            coefficients = np.zeros((4, 2, 6), dtype=complex)
            for valley in range(2):
                for spin in range(2):
                    coefficients[2 * valley + spin, spin, 3 * valley:3 * valley + 3] = (
                        1 / np.sqrt(3)
                    )
            if broken_coefficients:
                # Remain normalized, orthogonal and valley-pure, but the
                # target space no longer closes under either nonidentity op.
                coefficients[0, 0, :3] = [1 / np.sqrt(2), 1 / np.sqrt(2), 0]
            kpoint["coefficients"] = coefficients
            kpoint["energies_eV"] = np.zeros(4)
            kpoint["band_indices_vasp"] = np.arange(1, 5)
    center = np.array([1, 1, 0]) @ reciprocal
    config = {
        "input": {"wavefunction_h5": str(payload)},
        "analysis": {
            "kpoints": list(STARS), "iband": [1, 2, 3, 4],
            "reduced_ebr": {"enabled": True},
        },
        "monolayer_lattices": {"default": {"reciprocal_cart": (3 * reciprocal).tolist()}},
        "valley_centers": {
            "coordinate_mode": "cart",
            "centers": [{"name": "plus", "cart": center.tolist()},
                        {"name": "minus", "cart": (-center).tolist()}],
        },
        "valley_subspaces": [
            {"name": name, "centers": [name]} for name in ("plus", "minus")
        ],
        "projection": {
            "use_2d_momentum_only": True,
            "qcut_mode": "absolute", "qcut_Ainv": 5.0,
        },
        "symmetry": {"operations": {"structure_file": str(structure)}},
        "output": {"directory": str(root / "out"), "profile": profile, "summary_stdout": False},
    }
    path = root / "analyze.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def run_numerical_workflow(root: Path, **kwargs) -> dict[str, object]:
    """Execute the public workflow and read its actual serialized outputs."""
    outputs = analyze_hsp(write_numerical_fixture(root, **kwargs))
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in outputs.items()
        if isinstance(path, Path) and path.suffix == ".json"
    }
    return {
        "outputs": outputs,
        "reports": reports,
        "ingestion": load_database_ingestion_record_from_directory(
            root / "out", output_profile=kwargs.get("profile", "debug"),
        ),
    }


def assert_numerical_positive(result: dict[str, object]) -> dict[str, object]:
    """Check numerical scope, reviewed irreps and exact downstream results."""
    reports = result["reports"]
    symmetry = reports["symmetry_report_json"]
    assert symmetry["spacegroup_number"] == 143
    assert symmetry["detected_operation_count"] == 3
    expected_irreps = {
        "GM": {"-GM5": 1, "-GM6": 1},
        "K": {"-K5": 1, "-K6": 1},
        "KA": {"-KA5": 1, "-KA6": 1},
        "M": {"-M2": 2},
    }
    matches = reports["valley_irrep_matching_json"]["generic_matches_by_kpoint"]
    assert set(matches) == set(expected_irreps)
    for hsp, valleys in matches.items():
        assert set(valleys) == {"plus", "minus"}
        for row in valleys.values():
            assert row["matching_status"] == "matched", row
            assert row["irrep_multiplicities"] == expected_irreps[hsp], row
            assert row["workflow_path"] == "direct_qcut"
            assert row["readiness_level"] == "trusted"
            assert row["diagnostic_only"] is False

    required_counts = {"GM": 3, "K": 3, "KA": 3, "M": 1}
    expected_maps = {(0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3), (2, 0, 1, 5, 3, 4)}
    with h5py.File(result["outputs"]["diagnostics_h5"], "r") as h5:
        source = json.loads(h5["cprime/spinor_source_basis_certificate"][()])
        assert source["status"] == "passed"
        assert source["extractor_provenance"] is None
        evidence = json.loads(h5["cprime/scoped_representation_evidence"][()])
        assert set(evidence) == set(expected_irreps)
        for hsp, valleys in evidence.items():
            assert set(valleys) == {"plus", "minus"}
            for row in valleys.values():
                assert row["status"] == "passed", row
                assert len(row["scope"]["required_operation_ids"]) == required_counts[hsp]
                mapping = row["plane_wave_mapping"]
                maps = {tuple(op["source_to_target_map"]) for op in mapping["operation_rows"]}
                assert maps == (expected_maps if hsp != "M" else {(0, 1, 2, 3, 4, 5)})
                assert all(op["reciprocal_grid_permutation_passed"] for op in mapping["operation_rows"])
                assert len(mapping["composition_rows"]) == required_counts[hsp] ** 2
                assert all(op["passed"] for op in mapping["composition_rows"])
            groups = h5[f"symmetry_representations/{hsp}"]
            checked = 0
            scopes = set()
            for group in groups.values():
                checked += 1
                # These numerical matrices precede C-prime promotion; the
                # formal trust assertions above use producer evidence.
                scopes.add((group.attrs["target_valley"], group.attrs["source_operation_key"]))
                matrix = group["D_valley"][()]
                assert matrix.shape == (2, 2)
                identity = np.array_equal(group["rotation_frac"][()], np.eye(3))
                np.testing.assert_allclose(np.trace(matrix), 2 if identity else 1, atol=1e-12)
                if not identity:
                    np.testing.assert_allclose(
                        np.sort(np.angle(np.linalg.eigvals(matrix)) / (2 * np.pi)),
                        [-1 / 6, 1 / 6], atol=1e-12,
                    )
            assert checked == 2 * required_counts[hsp]
            assert scopes == {
                (valley, f"operation_{op}")
                for valley in ("plus", "minus")
                for op in evidence[hsp][valley]["scope"]["required_operation_ids"]
            }

    export = reports["valley_ebr_export_bundle_json"]
    assert export["bundle_count"] == 2
    for bundle in export["bundles"]:
        assert bundle["expected_hsps"] == ["GM", "K", "KA", "M"]
        assert bundle["unitary_vector_construction"]["kind"] == "direct_observed_unitary_rows"
    solutions = reports["valley_reduced_ebr_mapping_json"]["solutions"]
    assert len(solutions) == 2
    for solution in solutions:
        assert solution["status"] == "solved_exact", solution
        assert solution["classification"] == "atomic-compatible-candidate"
        assert solution["ebr_decomposition"] == [
            {"label": "-1E↑G(1) @ 1a(3,3)", "coefficient": 1},
            {"label": "-2E↑G(1) @ 1a(3,3)", "coefficient": 1},
        ]
    ingestion = result["ingestion"]
    assert ingestion["validation_errors"] == [], ingestion
    assert ingestion["final_reduced_ebr_result_count"] == 2, ingestion
    return {
        "source_space_group": 143, "spinor": True,
        "observed_hsps": ["GM", "K", "KA", "M"],
        "required_operations_by_hsp": required_counts,
        "plane_waves_per_kpoint": 6, "bands": 4,
        "final_reduced_ebr_result_count": 2,
        "validation_errors": [],
    }


def assert_broken_coefficients_blocked(result: dict[str, object]) -> None:
    """High valley purity and a good Gram matrix cannot replace closure."""
    reports = result["reports"]
    with result["outputs"]["valley_weights_csv"].open(encoding="utf-8") as handle:
        weights = list(csv.DictReader(handle))
    assert len(weights) == 16
    np.testing.assert_allclose([float(row["W_val"]) for row in weights], 1, atol=1e-12)
    np.testing.assert_allclose([float(row["P_v"]) for row in weights], 1, atol=1e-12)
    closure = reports["target_subspace_closure_json"]["by_kpoint"]
    for hsp in ("GM", "K", "KA"):
        rows = closure[hsp]
        assert all(row["target_frame_status"] == "passed" for row in rows)
        failed = [row for row in rows if row["classification"] == "target_subspace_not_closed"]
        assert len(failed) == 2
        assert all(row["closure_quality"] == "blocked" for row in failed)
    decisions = reports["irrep_workflow_decisions_json"]["by_kpoint"]
    for hsp in ("GM", "K", "KA"):
        assert all(row["readiness_level"] == "blocked" for row in decisions[hsp].values())
    # A different exact local scope is still valid, but cannot complete EBR.
    assert all(row["readiness_level"] == "trusted" for row in decisions["M"].values())
    assert result["ingestion"]["final_reduced_ebr_result_count"] == 0
    assert result["ingestion"]["validation_errors"] == []
    summary = reports["valley_summary_json"]
    assert summary["valley_ebr_export_bundle"]["bundle_count"] == 0
    assert summary["valley_ebr_export_bundle"]["bundles"] == []
    assert summary["valley_reduced_ebr_mapping"]["solutions"] == []


def run_installed_numerical_acceptance(workdir: Path | None = None) -> dict[str, object]:
    """Run positive and physical negative controls using installed code."""
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="valleyscope_numerical_") as tmp:
            return run_installed_numerical_acceptance(Path(tmp))
    summary = assert_numerical_positive(run_numerical_workflow(workdir / "positive"))
    assert_broken_coefficients_blocked(
        run_numerical_workflow(workdir / "broken", broken_coefficients=True)
    )
    summary["broken_coefficients_final_result_count"] = 0
    return summary
