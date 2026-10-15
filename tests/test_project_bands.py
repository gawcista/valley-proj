"""Generated projection-only production tests; no representation certificates."""
from __future__ import annotations

import csv
import gc
import json
import os
import weakref
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml


def _write_input(tmp_path, *, spinor=False, scale=1.0):
    path = tmp_path / "states.h5"
    reciprocal = np.diag([2., 2., 2.])
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("metadata")
        meta["lattice/direct_cart"] = np.eye(3) * np.pi
        meta["lattice/reciprocal_cart"] = reciprocal
        meta["spinor"] = spinor
        meta["source"] = "generated"
        meta["vasp_band_index_base"] = 1
        for idx, name in enumerate(["first", "second", "unused"]):
            group = h5.create_group(f"kpoints/{idx}")
            group["name"] = name
            group["frac"] = [idx / 4., 0., 0.]
            group["cart"] = [idx / 2., 0., 0.]
            group["g_vectors_frac"] = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
            group["g_vectors_cart"] = [[0., 0., 0.], [2., 0., 0.], [4., 0., 0.], [6., 0., 0.]]
            scalar = scale * np.sqrt([[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 1., 1., 1.]])
            coefficients = scalar[:, None, :].astype(complex)
            if spinor:
                coefficients = np.concatenate([coefficients, 1j * coefficients], axis=1) / np.sqrt(2.)
            group["coefficients"] = coefficients
            group["energies_eV"] = [idx + .1, idx + .2, idx + .3]
            group["band_indices_vasp"] = [13, 7, 22]
    return path


def _write_config(tmp_path, h5_path, *, kpoints=None, bands=None, **overrides):
    raw = {
        "input": {"wavefunction_h5": str(h5_path)},
        "analysis": {"kpoints": kpoints or ["first", "second"], "iband": bands or [7, 13]},
        "monolayer_lattices": {"default": {"reciprocal_cart": np.diag([20., 20., 20.]).tolist()}},
        "valley_centers": {"coordinate_mode": "cart", "centers": [
            {"name": "a", "cart": [0., 0., 0.]},
            {"name": "a_copy", "cart": [0., 0., 0.]},
            {"name": "b", "cart": [2., 0., 0.]},
            {"name": "c", "cart": [4., 0., 0.]},
        ]},
        "valley_subspaces": [
            {"name": "red", "centers": ["a", "a_copy"]},
            {"name": "green", "centers": ["b"]},
            {"name": "blue", "centers": ["c"]},
        ],
        "projection": {"qcut_Ainv": .75},
        "path": {"break_before": [], "ticks": [{"index": 0, "label": "start"}]},
        "plot": {"energy_reference_eV": -.25},
        "output": {"directory": str(tmp_path / "out")},
    }
    raw.update(overrides)
    path = tmp_path / "projection.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _project(path):
    from valleyscope.workflows.project_bands import project_bands
    return project_bands(path)


def test_scalar_soc_generic_weights_normalization_and_serialized_roundtrip(tmp_path):
    results = []
    for spinor, scale in [(False, 1.), (True, 1.), (False, 3.)]:
        case = tmp_path / f"case-{spinor}-{scale}"
        case.mkdir()
        result = _project(_write_config(case, _write_input(case, spinor=spinor, scale=scale)))
        payload = json.loads(result["result_path"].read_text())
        assert payload == result["result"]
        assert payload["document_kind"] == "valley_projection"
        assert payload["schema_version"] == "1.0.0"
        assert payload["valleys"] == ["red", "green", "blue"]
        assert payload["band_indices_vasp"] == [7, 13]
        assert [(r["path_index"], r["kpoint"], r["band_vasp"]) for r in payload["rows"]] == [
            (0, "first", 7), (0, "first", 13), (1, "second", 7), (1, "second", 13)]
        row = payload["rows"][0]
        assert row["energy_eV"] == .2
        assert row["norm"] == pytest.approx(10 * scale**2)
        assert row["valley_weights"] == pytest.approx({"red": 4 * scale**2, "green": 3 * scale**2, "blue": 2 * scale**2})
        assert row["normalized_valley_weights"] == pytest.approx({"red": .4, "green": .3, "blue": .2})
        assert row["W_val"] + row["W_overlap"] + row["W_res"] == pytest.approx(row["norm"])
        assert row["normalized_W_res"] == pytest.approx(.1)
        assert row["center_weights"]["a"] == pytest.approx(row["center_weights"]["a_copy"])
        assert payload["energy_reference_eV"] == -.25
        assert payload["path"]["kpoints"][1]["distance_Ainv"] == pytest.approx(.5)
        assert set(payload["stages"].values()) == {"not_requested"}
        assert payload["input"]["identity_kind"] == "stat_snapshot"
        assert "selected" in payload["input"]["validation_scope"]
        assert "hdf5_payload_identity" not in payload["input"]
        with result["csv_path"].open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == 4
        assert float(csv_rows[0]["normalized_W_res"]) == pytest.approx(.1)
        assert "full-state" in result["summary_text"]
        results.append(payload)
    for payload in results[1:]:
        assert payload["rows"][0]["normalized_valley_weights"] == pytest.approx(results[0]["rows"][0]["normalized_valley_weights"])


def test_reader_reads_only_selected_coefficient_rows_in_requested_order(tmp_path, monkeypatch):
    from valleyscope.io.projection_reader import ProjectionReader
    path = _write_input(tmp_path)
    with h5py.File(path, "r+") as h5:
        h5["kpoints/0/coefficients"][2] = np.nan
        h5["kpoints/2/coefficients"][...] = np.nan
        h5["kpoints/2/energies_eV"][...] = np.nan
    original = h5py.Dataset.__getitem__
    reads = []
    def record(dataset, key):
        if dataset.name.endswith("/coefficients"):
            reads.append((dataset.name, key))
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", record)
    with ProjectionReader(path) as reader:
        assert reads == []
        point = reader.read_kpoint("first", [7, 13])
        assert point.band_indices_vasp.tolist() == [7, 13]
        assert point.energies_eV.tolist() == [.2, .1]
    assert len(reads) == 2
    assert all(name == "/kpoints/0/coefficients" for name, _ in reads)
    assert [key[0] for _, key in reads] == [1, 0]


def test_workflow_does_not_retain_coefficients_from_previous_kpoint(tmp_path, monkeypatch):
    from valleyscope.io.projection_reader import ProjectionReader
    original = ProjectionReader.read_kpoint
    previous = []
    def bounded(reader, *args):
        gc.collect()
        assert not previous or previous[-1]() is None, "previous k-point coefficients retained"
        point = original(reader, *args)
        previous.append(weakref.ref(point.coefficients))
        return point
    monkeypatch.setattr(ProjectionReader, "read_kpoint", bounded)
    _project(_write_config(tmp_path, _write_input(tmp_path)))


@pytest.mark.parametrize("failure", ["nan", "zero", "missing_band", "bad_energy", "bad_grid"])
@pytest.mark.parametrize("prior", [False, True])
def test_late_selected_data_failure_preserves_all_outputs(tmp_path, failure, prior):
    path = _write_input(tmp_path)
    config = _write_config(tmp_path, path)
    output = tmp_path / "out"
    previous = {}
    if prior:
        _project(config)
        (output / "unrelated.bin").write_bytes(b"keep")
        previous = {f.name: f.read_bytes() for f in output.iterdir()}
    with h5py.File(path, "r+") as h5:
        group = h5["kpoints/1"]
        if failure == "nan": group["coefficients"][1, 0, 0] = np.nan
        if failure == "zero": group["coefficients"][1] = 0
        if failure == "missing_band": group["band_indices_vasp"][1] = 8
        if failure == "bad_energy": group["energies_eV"][1] = np.inf
        if failure == "bad_grid": group["g_vectors_cart"][1, 0] = np.nan
    with pytest.raises(ValueError):
        _project(config)
    actual = {f.name: f.read_bytes() for f in output.iterdir()} if output.exists() else {}
    assert actual == previous


@pytest.mark.parametrize("failure", ["duplicate_name", "duplicate_band", "bad_layout", "inconsistent_spinor", "collinear"])
def test_all_file_structure_validation_includes_unselected_kpoints(tmp_path, failure):
    path = _write_input(tmp_path)
    with h5py.File(path, "r+") as h5:
        group = h5["kpoints/2"]
        if failure == "duplicate_name": group["name"][()] = "first"
        if failure == "duplicate_band": group["band_indices_vasp"][1] = 13
        if failure == "bad_layout":
            del group["g_vectors_frac"]
            group["g_vectors_frac"] = np.zeros((4, 2))
        if failure == "inconsistent_spinor":
            del group["coefficients"]
            group["coefficients"] = np.ones((3, 2, 4))
        if failure == "collinear": h5["metadata/wavecar_nspin"] = 2
    with pytest.raises(ValueError):
        _project(_write_config(tmp_path, path))


def test_source_snapshot_change_blocks_publication(tmp_path, monkeypatch):
    from valleyscope.io.projection_reader import ProjectionReader
    path = _write_input(tmp_path)
    config = _write_config(tmp_path, path)
    original = ProjectionReader.read_kpoint
    def change_stat(reader, *args):
        point = original(reader, *args)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        return point
    monkeypatch.setattr(ProjectionReader, "read_kpoint", change_stat)
    with pytest.raises(ValueError, match="changed"):
        _project(config)
    assert not (tmp_path / "out/valley_projection.json").exists()


def test_projection_never_calls_analysis_or_eager_identity_producers(tmp_path, monkeypatch):
    import valleyscope.io.config as configuration
    import valleyscope.io.h5_reader as eager
    import valleyscope.io.wavefunction_convention as convention
    import valleyscope.analysis.target_frame as frames
    import valleyscope.irreps.tables as tables
    def forbidden(*args, **kwargs):
        raise AssertionError("pure projection invoked an analysis producer")
    monkeypatch.setattr(configuration, "load_config", forbidden)
    monkeypatch.setattr(eager, "read_wavefunction_h5", forbidden)
    monkeypatch.setattr(convention, "file_payload_identity", forbidden)
    monkeypatch.setattr(frames, "build_target_frame", forbidden)
    monkeypatch.setattr(tables, "load_standard_irrep_table", forbidden)
    _project(_write_config(tmp_path, _write_input(tmp_path)))


def test_dynamic_overlap_is_reported_and_excluded_without_valley_merging(tmp_path):
    config = _write_config(tmp_path, _write_input(tmp_path), projection={"qcut_Ainv": .75, "projector_mode": "k_resolved_parent_valley"})
    raw = yaml.safe_load(config.read_text())
    raw["valley_centers"]["centers"][2]["cart"] = [.4, 0., 0.]
    config.write_text(yaml.safe_dump(raw))
    result = _project(config)["result"]
    row = result["rows"][0]
    assert result["valleys"] == ["red", "green", "blue"]
    assert row["W_overlap"] == pytest.approx(4.)
    assert row["valley_weights"] == pytest.approx({"red": 0., "green": 0., "blue": 2.})
    assert row["W_res"] == pytest.approx(4.)
    assert result["warnings"]
    raw["projection"]["overlap_policy"] = "error"
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="overlap"):
        _project(config)


@pytest.mark.parametrize("projection", [{"qcut_Ainv": .75, "overlap_policy": "include"}, {"qcut_Ainv": float("nan")}, {"qcut_Ainv": 0.}])
def test_invalid_projection_options_fail_clearly(tmp_path, projection):
    config = _write_config(tmp_path, _write_input(tmp_path), projection=projection)
    with pytest.raises(ValueError):
        _project(config)


def test_path_occurrences_repeated_endpoints_breaks_and_band_range(tmp_path):
    config = _write_config(tmp_path, _write_input(tmp_path), kpoints=["second", "first", "first", "second"], bands="7-7",
        path={"break_before": [3], "ticks": [{"index": 1, "label": "repeat"}, {"index": 3, "label": "restart"}]})
    result = _project(config)["result"]
    assert result["band_indices_vasp"] == [7]
    assert [p["name"] for p in result["path"]["kpoints"]] == ["second", "first", "first", "second"]
    assert [p["distance_Ainv"] for p in result["path"]["kpoints"]] == pytest.approx([0., .5, .5, .5])
    assert result["path"]["break_before"] == [3]
    assert [r["path_index"] for r in result["rows"]] == [0, 1, 2, 3]


@pytest.mark.parametrize("updates", [
    {"path": {"break_before": [0]}}, {"path": {"break_before": [2]}},
    {"path": {"ticks": [{"index": 0, "label": ""}]}},
    {"plot": {"energy_reference_eV": float("inf")}},
    {"valley_subspaces": [{"name": "red", "centers": ["a"]}, {"name": "red", "centers": ["b"]}]},
])
def test_invalid_metadata_is_rejected(tmp_path, updates):
    with pytest.raises(ValueError):
        _project(_write_config(tmp_path, _write_input(tmp_path), **updates))


def test_shared_core_strict_boundary_and_skew_parent_metric():
    from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
    from valleyscope.projection.kpoint import project_kpoint
    basis = np.array([[1., 0., 0.], [.9, .3, 0.], [0., 0., 1.]])
    q = np.array([[.651, .147, 0.], [.5, 0., 0.]])
    # Complete finite image enumeration is an independent metric reference.
    d = min(np.linalg.norm(q[0, :2] - np.array([i, j]) @ basis[:2, :2]) for i in range(-5, 6) for j in range(-5, 6))
    projectors, weights = project_kpoint(q, np.ones((1, 1, 2)), [ValleyCenter("a", np.zeros(3), reciprocal_cart=basis)],
        [ValleySector("one", ["a"])], basis, (d + .5) / 2.)
    assert projectors.center_masks["a"][0]
    _, boundary = project_kpoint(np.array([[.5, 0., 0.]]), np.ones((1, 1, 1)), [ValleyCenter("a", np.zeros(3))],
        [ValleySector("one", ["a"])], np.eye(3) * 20, .5)
    assert boundary[0].w_val == 0.


def test_shared_reporting_preserves_fixed_dynamic_and_legacy_include_behavior():
    from valleyscope.geometry.valley_centers import ValleyCenter, ValleySector
    from valleyscope.projection.kpoint import project_kpoint
    from valleyscope.projection.sector_projectors import adjust_centers_for_parent_valley, build_sector_projectors
    from valleyscope.projection.weights import compute_valley_weights
    reciprocal = np.eye(3) * 20.
    moire_reciprocal = np.eye(3) * 2.
    centers = [ValleyCenter("a", np.array([.2, 0., 0.])), ValleyCenter("b", np.array([.4, 0., 0.]))]
    valleys = [ValleySector("a", ["a"]), ValleySector("b", ["b"])]
    q = np.array([[.3, 0., 0.], [2., 0., 0.], [6., 0., 0.]])
    coefficients = np.array([[[1., 2., 3.]], [[2j, 3., 4.]]])
    for dynamic in (False, True):
        reporting = adjust_centers_for_parent_valley(centers, np.array([.3, 0., 0.]), moire_reciprocal) if dynamic else centers
        for policy in ("warn_exclude", "include"):
            baseline = build_sector_projectors(q, reporting, valleys, reciprocal, .5, overlap_policy=policy, emit_warnings=False)
            masks, weights = project_kpoint(q, coefficients, reporting, valleys, reciprocal, .5, overlap_policy=policy)
            assert weights == compute_valley_weights(coefficients, baseline)
            for name in baseline.sector_names:
                np.testing.assert_array_equal(masks.sector_masks[name], baseline.sector_masks[name])
            if policy == "include":
                assert weights[0].w_val == 2.


@pytest.mark.parametrize("qcut_options,expected", [
    ({"qcut_mode": "moire_shell", "qcut_shell": .3}, .6),
    ({"qcut_fraction": .3}, .6),
])
def test_projection_reuses_existing_qcut_definitions(tmp_path, qcut_options, expected):
    result = _project(_write_config(tmp_path, _write_input(tmp_path), projection=qcut_options))["result"]
    assert result["projection"]["qcut_Ainv"] == pytest.approx(expected)


def test_parent_poscar_layer_fractional_centers_and_transforms_are_reused(tmp_path):
    parent = tmp_path / "parent.vasp"
    parent.write_text("parent\n1.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nX\n1\nDirect\n0 0 0\n")
    path = _write_input(tmp_path)
    config = _write_config(tmp_path, path,
        input={"wavefunction_h5": str(path), "monolayer_poscars": {"layer": str(parent)}},
        monolayer_lattices={}, layer_transforms={"layer": {"rotation_deg": 90.}},
        valley_centers={"coordinate_mode": "layer_frac", "centers": [{"name": "p", "frac": [.25, 0., 0.], "layer": "layer"}]},
        valley_subspaces=[{"name": "parent", "centers": ["p"]}])
    result = _project(config)["result"]
    center = result["parent_geometry"]["centers"][0]
    assert center["cart_Ainv"] == pytest.approx([0., np.pi / 2, 0.], abs=1.e-14)
    assert center["layer"] == "layer"


@pytest.mark.parametrize("failure", ["numeric_name", "float_band_ids", "complex_grid", "string_coefficients", "spin_channel_attribute"])
def test_malformed_layout_types_and_explicit_spin_provenance(tmp_path, failure):
    path = _write_input(tmp_path)
    with h5py.File(path, "r+") as h5:
        group = h5["kpoints/0"]
        if failure == "numeric_name":
            del group["name"]
            group["name"] = 4
        if failure == "float_band_ids":
            del group["band_indices_vasp"]
            group["band_indices_vasp"] = [13., 7., 22.]
        if failure == "complex_grid":
            del group["g_vectors_cart"]
            group["g_vectors_cart"] = np.ones((4, 3), dtype=complex)
        if failure == "string_coefficients":
            del group["coefficients"]
            group["coefficients"] = np.full((3, 1, 4), b"1")
        if failure == "spin_channel_attribute":
            h5["metadata"].attrs["wavecar_nspin"] = 2
    with pytest.raises(ValueError):
        _project(_write_config(tmp_path, path))


@pytest.mark.parametrize("failure", ["kcart", "gcart", "fractional_g", "duplicate_g", "lattice", "qcut_scan"])
def test_coordinate_and_grid_input_consistency_failures(tmp_path, failure):
    path = _write_input(tmp_path)
    config = _write_config(tmp_path, path)
    if failure == "qcut_scan":
        raw = yaml.safe_load(config.read_text())
        raw["projection"]["qcut_scan"] = [.5, .75]
        config.write_text(yaml.safe_dump(raw))
    else:
        with h5py.File(path, "r+") as h5:
            group = h5["kpoints/1"]
            if failure == "kcart": group["cart"][0] = .6
            if failure == "gcart": group["g_vectors_cart"][1, 0] = 2.1
            if failure == "fractional_g":
                del group["g_vectors_frac"]
                group["g_vectors_frac"] = [[0., 0., 0.], [1.1, 0., 0.], [2., 0., 0.], [3., 0., 0.]]
                group["g_vectors_cart"][1, 0] = 2.2
            if failure == "duplicate_g":
                group["g_vectors_frac"][1] = group["g_vectors_frac"][0]
                group["g_vectors_cart"][1] = group["g_vectors_cart"][0]
            if failure == "lattice": h5["metadata/lattice/direct_cart"][0, 0] = 1.
    with pytest.raises(ValueError):
        _project(config)
    assert not (tmp_path / "out/valley_projection.json").exists()


def test_distinct_gz_rows_remain_distinct_full_state_probabilities_in_2d_mode(tmp_path):
    path = _write_input(tmp_path)
    with h5py.File(path, "r+") as h5:
        group = h5["kpoints/0"]
        group["g_vectors_frac"][1] = [0, 0, 1]
        group["g_vectors_cart"][1] = [0., 0., 2.]
    result = _project(_write_config(tmp_path, path, kpoints=["first"]))["result"]
    row = result["rows"][0]
    assert row["valley_weights"]["red"] == pytest.approx(7.)
    assert row["norm"] == pytest.approx(10.)


def test_input_coordinate_serialization_precision_is_accepted(tmp_path):
    path = _write_input(tmp_path)
    with h5py.File(path, "r+") as h5:
        h5["kpoints/1/cart"][0] += 1.e-9
        h5["kpoints/1/g_vectors_cart"][1, 0] += 1.e-9
    _project(_write_config(tmp_path, path))
