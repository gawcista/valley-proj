"""Fat-band figures from the standalone serialized projection document."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


def _number(value, field, *, nonnegative=False, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(result) or (nonnegative and result < 0) or (positive and result <= 0):
        raise ValueError(f"{field} has an invalid value")
    return result


def _index(value, field, size):
    if type(value) is not int or not 0 <= value < size:
        raise ValueError(f"{field} must be an occurrence index in [0, {size})")
    return value


def _list(value, field, *, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{field} must be {'a nonempty' if nonempty else 'a'} list")
    return value


def _object(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _close(first, second, field):
    if not math.isclose(first, second, rel_tol=1e-8, abs_tol=1e-10):
        raise ValueError(f"{field} is inconsistent with full-state normalization")


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _validate(document):
    document = _object(document, "document")
    if document.get("document_kind") != "valley_projection" or document.get("schema_version") != "1.0.0":
        raise ValueError("Expected valley_projection schema_version 1.0.0")
    valleys = _list(document.get("valleys"), "valleys", nonempty=True)
    for value in valleys:
        _text(value, "valleys entry")
    if len(set(valleys)) != len(valleys):
        raise ValueError("valleys must be unique")
    bands = _list(document.get("band_indices_vasp"), "band_indices_vasp", nonempty=True)
    if any(type(band) is not int or band <= 0 for band in bands) or len(set(bands)) != len(bands):
        raise ValueError("band_indices_vasp must contain unique positive integer band IDs")
    projection = _object(document.get("projection"), "projection")
    if projection.get("projector_mode") not in {"fixed_center", "k_resolved_parent_valley"}:
        raise ValueError("Invalid projector_mode")
    if projection.get("overlap_policy") not in {"warn_exclude", "error"}:
        raise ValueError("Projection plotting requires excluded or rejected overlap")
    _number(projection.get("qcut_Ainv"), "qcut_Ainv", positive=True)
    if type(projection.get("use_2d_momentum_only")) is not bool:
        raise ValueError("use_2d_momentum_only must be boolean")
    _number(document.get("energy_reference_eV"), "energy_reference_eV")
    path = _object(document.get("path"), "path")
    points = _list(path.get("kpoints"), "path.kpoints", nonempty=True)
    for point in points:
        _object(point, "kpoint")
        _text(point.get("name"), "kpoint.name")
        for key in ["frac", "cart_Ainv"]:
            vector = _list(point.get(key), f"kpoint.{key}")
            if len(vector) != 3:
                raise ValueError(f"kpoint.{key} must have three components")
            for value in vector:
                _number(value, f"kpoint.{key}")
        _number(point.get("distance_Ainv"), "kpoint.distance_Ainv", nonnegative=True)
    distances = [point["distance_Ainv"] for point in points]
    if any(right < left for left, right in zip(distances, distances[1:])):
        raise ValueError("Path distances must be nondecreasing in occurrence order")
    breaks = _list(path.get("break_before"), "path.break_before")
    for value in breaks:
        _index(value, "break_before", len(points))
        if value == 0:
            raise ValueError("break_before cannot contain the first occurrence")
    if len(set(breaks)) != len(breaks):
        raise ValueError("break_before must contain unique occurrence indices")
    _close(distances[0], 0.0, "first path distance")
    for i in range(1, len(points)):
        step = 0.0 if i in breaks else math.dist(points[i]["cart_Ainv"], points[i - 1]["cart_Ainv"])
        _close(distances[i] - distances[i - 1], step, "Cartesian path distance")
    for tick in _list(path.get("ticks"), "path.ticks"):
        _object(tick, "tick")
        _index(tick.get("index"), "tick.index", len(points))
        _text(tick.get("label"), "tick.label")
    rows = _list(document.get("rows"), "rows", nonempty=True)
    by_state = {}
    valley_set = set(valleys)
    for row in rows:
        _object(row, "row")
        i = _index(row.get("path_index"), "row.path_index", len(points))
        band = row.get("band_vasp")
        if type(band) is not int or band not in bands:
            raise ValueError("row.band_vasp is outside band_indices_vasp")
        if row.get("kpoint") != points[i]["name"]:
            raise ValueError("row.kpoint disagrees with its path occurrence")
        if (i, band) in by_state:
            raise ValueError("Duplicate (path_index, band_vasp) row")
        by_state[i, band] = row
        _number(row.get("energy_eV"), "row.energy_eV")
        norm = _number(row.get("norm"), "row.norm", positive=True)
        raw = _object(row.get("valley_weights"), "valley_weights")
        normalized = _object(row.get("normalized_valley_weights"), "normalized_valley_weights")
        if set(raw) != valley_set or set(normalized) != valley_set:
            raise ValueError("Every weight row must have exactly the configured valleys")
        fractions = []
        for label in valleys:
            fraction = _number(raw[label], "valley_weights", nonnegative=True) / norm
            value = _number(normalized[label], "normalized_valley_weights", nonnegative=True)
            if fraction > 1.0 + 1e-8 or value > 1.0 + 1e-8:
                raise ValueError("Valley weight exceeds full-state norm")
            _close(value, fraction, "normalized_valley_weights")
            fractions.append(fraction)
        centers = _object(row.get("center_weights"), "center_weights")
        for label, value in centers.items():
            _text(label, "center label")
            if _number(value, "center_weights", nonnegative=True) / norm > 1.0 + 1e-8:
                raise ValueError("Center window weight exceeds full-state norm")
        w_val = _number(row.get("W_val"), "W_val", nonnegative=True) / norm
        overlap = _number(row.get("W_overlap"), "W_overlap", nonnegative=True) / norm
        residual = _number(row.get("W_res"), "W_res", nonnegative=True) / norm
        _close(w_val, math.fsum(fractions), "W_val")
        _close(w_val + overlap + residual, 1.0, "weight accounting")
        for key, expected in [("normalized_W_overlap", overlap), ("normalized_W_res", residual)]:
            _close(_number(row.get(key), key, nonnegative=True), expected, key)
    if len(by_state) != len(points) * len(bands):
        raise ValueError("Rows do not cover every requested path occurrence and band")
    return by_state


def plot_valley_bands(result_path, *, output_directory=None, energy_reference_eV=None,
                      label_bands=False) -> dict:
    """Validate a projection JSON and render absolute full-state-normalized weights.

    All figure files are rendered in a temporary directory before publication.
    Replotting reads only this JSON, including when original inputs are absent.
    """
    source = Path(result_path).resolve()
    try:
        encoded = source.read_text(encoding="utf-8")
        document = json.loads(encoded, object_pairs_hook=_unique_fields)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid projection JSON: {exc}") from exc
    states = _validate(document)
    reference = _number(document["energy_reference_eV"] if energy_reference_eV is None
                        else energy_reference_eV, "energy_reference_eV")
    if type(label_bands) is not bool:
        raise ValueError("label_bands must be boolean")
    destination = Path(output_directory).resolve() if output_directory is not None else source.parent
    names = {"png": "valley_fat_bands.png", "pdf": "valley_fat_bands.pdf",
             "metadata": "valley_fat_bands.metadata.json"}
    for filename in names.values():
        target = destination / filename
        if target.resolve() == source:
            raise ValueError(f"Plot output would replace an input: {target}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Plot output destination is not a file: {target}")
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError('Plotting requires matplotlib; install "valleyscope[plot]".') from exc

    points = document["path"]["kpoints"]
    valleys = document["valleys"]
    bands = document["band_indices_vasp"]
    x = np.asarray([point["distance_Ainv"] for point in points], dtype=float)
    energies = np.asarray([[states[i, band]["energy_eV"] - reference for i in range(len(points))]
                           for band in bands])
    if not np.all(np.isfinite(energies)):
        raise ValueError("Energy-reference subtraction must remain finite")
    boundaries = [0, *sorted(document["path"]["break_before"]), len(points)]
    segments = [slice(start, stop) for start, stop in zip(boundaries, boundaries[1:])]
    area_scale = 100.0
    fig = Figure(figsize=(8.5, max(5.0, 2.2 * (len(valleys) + 1))), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(valleys) + 1, 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    ordered_rows = [states[i, band] for band in bands for i in range(len(points))]
    scatter_x = np.tile(x, len(bands))
    scatter_y = energies.ravel()
    ylo, yhi = float(np.min(energies)), float(np.max(energies))
    padding = max(0.05, (yhi - ylo) * 0.06)
    colors = ["#1976a3", "#d56a20", "#39804c", "#9156a3", "#bb3f68"]
    try:
        for panel, ax in enumerate(axes):
            for position, band in enumerate(bands):
                for segment in segments:
                    ax.plot(x[segment], energies[position, segment], color="0.7", linewidth=0.75,
                            marker="." if segment.stop - segment.start == 1 else None,
                            markersize=2, zorder=1)
                if label_bands:
                    ax.annotate(str(band), (x[-1], energies[position, -1]), xytext=(4, 0),
                                textcoords="offset points", fontsize=7, va="center")
            if panel < len(valleys):
                label = valleys[panel]
                sizes = [area_scale * row["normalized_valley_weights"][label] for row in ordered_rows]
                ax.scatter(scatter_x, scatter_y, s=sizes, color=colors[panel % len(colors)],
                           linewidths=0, zorder=2)
                ax.set_title(label, loc="left", fontsize=10)
            else:
                ax.scatter(scatter_x, scatter_y,
                           s=[area_scale * row["normalized_W_res"] for row in ordered_rows],
                           color="0.25", linewidths=0, label="Residual", zorder=2)
                ax.scatter(scatter_x, scatter_y,
                           s=[area_scale * row["normalized_W_overlap"] for row in ordered_rows],
                           facecolors="none", edgecolors="#bd3b34", linewidths=0.8,
                           label="Overlap", zorder=3)
                ax.set_title("Residual and overlap", loc="left", fontsize=10)
                ax.legend(loc="best", fontsize=8)
            ax.set_ylim(ylo - padding, yhi + padding)
            ax.set_ylabel("Energy − reference (eV)")
            ax.grid(axis="y", color="0.92", linewidth=0.5)
            ax.tick_params(labelsize=8)
        ticks = document["path"]["ticks"]
        if ticks:
            # Distinct labeled occurrences may share an endpoint distance.
            tick_labels = {}
            for tick in ticks:
                location = float(x[tick["index"]])
                labels = tick_labels.setdefault(location, [])
                if tick["label"] not in labels:
                    labels.append(tick["label"])
            axes[-1].set_xticks(list(tick_labels), [" | ".join(v) for v in tick_labels.values()])
        axes[-1].set_xlabel("Path distance (Å⁻¹)")
        fig.suptitle(f"{document['projection']['projector_mode']} · reference {reference:g} eV\n"
                     "Marker area = absolute weight / full-state norm (100 pt² = 1)", fontsize=11)
        metadata = {
            "document_kind": "valley_projection_plot", "schema_version": "1.0.0",
            "source_result": str(source), "energy_reference_eV": reference,
            "source_result_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "projector_mode": document["projection"]["projector_mode"],
            "qcut_Ainv": document["projection"]["qcut_Ainv"],
            "weight_normalization": "full_state_norm", "marker_area_points2_per_unit_weight": area_scale,
            "valley_panels": valleys, "band_indices_vasp": bands, "label_bands": label_bands,
            "break_before": document["path"]["break_before"], "ticks": ticks,
        }
        # Validation and both renderings finish before touching prior outputs.
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".valley-fat-bands-", dir=destination) as staging:
            stage = Path(staging)
            fig.savefig(stage / names["png"], dpi=180)
            fig.savefig(stage / names["pdf"])
            (stage / names["metadata"]).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")
            previous = {}
            for filename in names.values():
                target = destination / filename
                if target.exists() or target.is_symlink():
                    backup = stage / f"previous-{filename}"
                    shutil.copy2(target, backup, follow_symlinks=False)
                    previous[filename] = backup
            published = []
            try:
                for filename in names.values():
                    os.replace(stage / filename, destination / filename)
                    published.append(filename)
            except BaseException:
                for filename in reversed(published):
                    if filename in previous:
                        os.replace(previous[filename], destination / filename)
                    else:
                        (destination / filename).unlink()
                raise
        return {key: str(destination / filename) for key, filename in names.items()}
    finally:
        fig.clear()
