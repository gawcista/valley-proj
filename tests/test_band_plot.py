"""Serialized projection plotting; fixtures do not use wavefunction producers."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest


def projection_document():
    rows = []
    for i, name in enumerate(["A", "B", "A", "A", "C"]):
        for band, energy in [(17, 3.0 + i / 10), (3, 2.0 - i / 10)]:
            rows.append({
                "path_index": i, "kpoint": name, "band_vasp": band,
                "energy_eV": energy, "norm": 4.0,
                "valley_weights": {"arbitrary": 1.0, "second": 0.0, "third": 1.0},
                "normalized_valley_weights": {"arbitrary": 0.25, "second": 0.0, "third": 0.25},
                "center_weights": {"window_1": 1.0, "window_2": 1.0},
                "W_val": 2.0, "W_overlap": 0.5, "W_res": 1.5,
                "normalized_W_overlap": 0.125, "normalized_W_res": 0.375,
            })
    return {
        "document_kind": "valley_projection", "schema_version": "1.0.0",
        "valleys": ["arbitrary", "second", "third"], "band_indices_vasp": [17, 3],
        "energy_reference_eV": 2.0,
        "projection": {"projector_mode": "fixed_center", "qcut_Ainv": 0.1,
                       "overlap_policy": "warn_exclude", "use_2d_momentum_only": True},
        "path": {
            "kpoints": [{"name": name, "frac": [x, 0., 0.], "cart_Ainv": [x, 0., 0.],
                         "distance_Ainv": distance}
                        for name, x, distance in [("A", 0, 0), ("B", 1, 1), ("A", 0, 2),
                                                   ("A", 0, 2), ("C", 1, 3)]],
            "break_before": [3], "ticks": [{"index": 0, "label": "A"},
                                               {"index": 2, "label": "A"},
                                               {"index": 4, "label": "C"}],
        },
        "rows": rows,
    }


def write_document(tmp_path, document):
    path = tmp_path / "valley_projection.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def plotter():
    return importlib.import_module("valleyscope.reports.band_plot").plot_valley_bands


def test_replot_preserves_occurrence_breaks_full_state_scale_and_original_bands(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    observations = []
    savefig = Figure.savefig

    def record_and_save(figure, *args, **kwargs):
        if not observations:
            for ax in figure.axes:
                observations.append({
                    "segments": [line.get_xdata().tolist() for line in ax.lines
                                 if len(line.get_xdata()) > 1],
                    "labels": [item.get_text() for item in ax.texts],
                    "ylim": ax.get_ylim(),
                    "sizes": [item.get_sizes().tolist() for item in ax.collections],
                    "offsets": [item.get_offsets().copy() for item in ax.collections],
                })
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record_and_save)
    result = plotter()(write_document(tmp_path, projection_document()), label_bands=True)
    for extension, signature in [("png", b"\x89PNG"), ("pdf", b"%PDF")]:
        path = Path(result[extension])
        assert path.read_bytes().startswith(signature)
        assert path.stat().st_size > 1000
    assert len(observations) == 4
    for panel in observations:
        # Repeated A at index 2 belongs to the first segment; index 3 starts the next.
        assert panel["segments"][:4] == [[0., 1., 2.], [2., 3.], [0., 1., 2.], [2., 3.]]
        assert "17" in panel["labels"]
        assert "3" in panel["labels"]
    assert observations[0]["ylim"] == observations[-1]["ylim"]
    assert observations[0]["sizes"][0] == [25.] * 10
    assert observations[1]["sizes"][0] == [0.] * 10
    assert observations[-1]["sizes"][0] == [37.5] * 10
    assert observations[-1]["sizes"][1] == [12.5] * 10
    assert observations[0]["offsets"][0][0, 1] == 1.0
    metadata = json.loads(Path(result["metadata"]).read_text())
    assert metadata["energy_reference_eV"] == 2.0
    assert metadata["break_before"] == [3]
    assert metadata["band_indices_vasp"] == [17, 3]
    assert metadata["weight_normalization"] == "full_state_norm"


def test_replot_explicit_energy_reference_output_and_uniform_coefficient_scale(tmp_path):
    pytest.importorskip("matplotlib")
    doc = projection_document()
    # Uniform coefficient scaling by 3 changes raw probabilities by 9.
    for row in doc["rows"]:
        for key in ["norm", "W_val", "W_overlap", "W_res"]:
            row[key] *= 9
        for key in ["valley_weights", "center_weights"]:
            row[key] = {label: weight * 9 for label, weight in row[key].items()}
    result = plotter()(write_document(tmp_path, doc), output_directory=tmp_path / "figures",
                       energy_reference_eV=-0.25)
    metadata = json.loads(Path(result["metadata"]).read_text())
    assert metadata["energy_reference_eV"] == -0.25
    assert metadata["marker_area_points2_per_unit_weight"] == 100.0
    assert Path(result["png"]).parent == tmp_path / "figures"


@pytest.mark.parametrize("change", [
    lambda d: d.update(document_kind="valley_summary"),
    lambda d: d.update(schema_version="2.2.0"),
    lambda d: d.update(valleys=["arbitrary", "arbitrary"]),
    lambda d: d.update(band_indices_vasp=[17, True]),
    lambda d: d["path"].update(break_before=["A"]),
    lambda d: d["path"].update(break_before=[0]),
    lambda d: d["path"].update(break_before=[3, 3]),
    lambda d: d["path"]["ticks"].append({"index": 5, "label": "bad"}),
    lambda d: d["path"]["kpoints"][1].update(cart_Ainv=[float("nan"), 0, 0]),
    lambda d: d["path"]["kpoints"][1].update(frac=[0, 0]),
    lambda d: d["path"]["kpoints"][2].update(distance_Ainv=-1),
    lambda d: d["path"]["kpoints"][2].update(distance_Ainv=1.5),
    lambda d: d["rows"].pop(),
    lambda d: d["rows"].append(copy.deepcopy(d["rows"][0])),
    lambda d: d["rows"][0].update(path_index=True),
    lambda d: d["rows"][0].update(kpoint="wrong"),
    lambda d: d["rows"][0].update(energy_eV=float("inf")),
    lambda d: d["rows"][0].update(norm=0),
    lambda d: d["rows"][0].update(norm=-1),
    lambda d: d["rows"][0]["valley_weights"].update(arbitrary=-1),
    lambda d: d["rows"][0]["valley_weights"].update(extra=1),
    lambda d: d["rows"][0]["normalized_valley_weights"].update(arbitrary=0.5),
    lambda d: d["rows"][0]["center_weights"].update(window_1=5),
    lambda d: d["rows"][0].update(W_val=3),
    lambda d: d["rows"][0].update(W_res=2),
    lambda d: d["rows"][0].update(normalized_W_overlap=0.5),
    lambda d: d.update(energy_reference_eV="2"),
    lambda d: d["projection"].update(overlap_policy="include"),
    lambda d: d["projection"].update(qcut_Ainv=0),
])
def test_malformed_result_is_rejected_without_outputs_or_replacing_prior_figures(tmp_path, change):
    doc = projection_document()
    change(doc)
    destination = tmp_path / "figures"
    destination.mkdir()
    prior = destination / "valley_fat_bands.png"
    prior.write_bytes(b"prior figure")
    with pytest.raises(ValueError):
        plotter()(write_document(tmp_path, doc), output_directory=destination)
    assert prior.read_bytes() == b"prior figure"
    assert sorted(p.name for p in destination.iterdir()) == ["valley_fat_bands.png"]


def test_missing_matplotlib_explains_optional_extra_and_writes_nothing(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    with pytest.raises(RuntimeError, match=r"valleyscope\[plot\]"):
        plotter()(write_document(tmp_path, projection_document()), output_directory=tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_duplicate_json_fields_are_rejected(tmp_path):
    encoded = json.dumps(projection_document())
    path = tmp_path / "duplicate.json"
    path.write_text(encoded.replace('"norm": 4.0', '"norm": 0, "norm": 4.0'), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON"):
        plotter()(path)
    assert list(tmp_path.iterdir()) == [path]


def test_second_render_failure_preserves_previous_output_set(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    savefig = Figure.savefig

    def fail_pdf(figure, filename, *args, **kwargs):
        if Path(filename).suffix == ".pdf":
            raise OSError("simulated PDF writer failure")
        return savefig(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", fail_pdf)
    destination = tmp_path / "figures"
    destination.mkdir()
    for name in ["valley_fat_bands.png", "valley_fat_bands.pdf", "valley_fat_bands.metadata.json"]:
        (destination / name).write_bytes(b"prior artifact")
    with pytest.raises(OSError, match="PDF writer"):
        plotter()(write_document(tmp_path, projection_document()), output_directory=destination)
    assert len(list(destination.iterdir())) == 3
    assert all(path.read_bytes() == b"prior artifact" for path in destination.iterdir())


def test_zero_weight_single_point_segments_keep_a_visible_base_state(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    document = projection_document()
    document["path"]["kpoints"] = document["path"]["kpoints"][:1]
    document["path"]["break_before"] = []
    document["path"]["ticks"] = [{"index": 0, "label": "A"}]
    document["rows"] = document["rows"][:2]
    visible = []
    savefig = Figure.savefig

    def observe(figure, *args, **kwargs):
        visible.extend(line.get_marker() not in [None, "None", "", " "]
                       for line in figure.axes[1].lines)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", observe)
    plotter()(write_document(tmp_path, document))
    assert visible and all(visible)
