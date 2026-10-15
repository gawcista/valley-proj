"""Projection-only valley diagnostics with one selected k point in memory."""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from valleyscope.geometry.reciprocal import RunLocalPeriodicDistanceCache
from valleyscope.io.projection_config import load_projection_config, resolve_projection_qcut
from valleyscope.io.projection_reader import ProjectionReader
from valleyscope.projection.kpoint import project_kpoint
from valleyscope.projection.sector_projectors import adjust_centers_for_parent_valley

PROJECTION_DOCUMENT_KIND = "valley_projection"
PROJECTION_SCHEMA_VERSION = "1.0.0"


def _projection_row(path_index, point, band, result):
    norm = result.norm
    return {"path_index": path_index, "kpoint": point.name, "band_vasp": int(band),
            "energy_eV": float(point.energies_eV[result.band_position]), "norm": norm,
            "valley_weights": result.sector_weights,
            "normalized_valley_weights": {name: value / norm for name, value in result.sector_weights.items()},
            "center_weights": result.center_weights, "W_val": result.w_val,
            "W_overlap": result.overlap_weight, "W_res": result.residual_weight,
            "normalized_W_overlap": result.overlap_weight / norm,
            "normalized_W_res": result.residual_weight / norm}


def _summary(payload):
    projection = payload["projection"]
    lines = ["Valley momentum projection", f"Projector mode: {projection['projector_mode']}",
             f"qcut: {projection['qcut_Ainv']:.10g} A^-1; strict distance < qcut",
             "Normalization: absolute full-state probability raw/N, N=sum(|C|^2) over all G and spinor components.",
             "Scalar states carry no spin factor two. Center windows are momentum diagnostics.",
             f"Selected path occurrences: {len(payload['path']['kpoints'])}; original VASP bands: {payload['band_indices_vasp']}"]
    for name in payload["valleys"]:
        values = [row["normalized_valley_weights"][name] for row in payload["rows"]]
        lines.append(f"{name}: full-state weight min={min(values):.6g}, max={max(values):.6g}, mean={np.mean(values):.6g}")
    for name in ("normalized_W_overlap", "normalized_W_res"):
        values = [row[name] for row in payload["rows"]]
        lines.append(f"{name}: min={min(values):.6g}, max={max(values):.6g}, mean={np.mean(values):.6g}")
    lines.append("Subspace, irrep, reduced EBR and valley C mod n stages: not_requested.")
    if payload["warnings"]:
        lines.extend(f"Warning: {warning}" for warning in payload["warnings"])
    return "\n".join(lines) + "\n"


def _write_csv(path, rows, valleys, center_names):
    fixed = ["path_index", "kpoint", "band_vasp", "energy_eV", "norm", "W_val", "W_overlap", "W_res", "normalized_W_overlap", "normalized_W_res"]
    fields = fixed + [f"weight:{name}" for name in valleys] + [f"normalized_weight:{name}" for name in valleys] + [f"center_weight:{name}" for name in center_names]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {key: row[key] for key in fixed}
            flattened.update({f"weight:{name}": row["valley_weights"][name] for name in valleys})
            flattened.update({f"normalized_weight:{name}": row["normalized_valley_weights"][name] for name in valleys})
            flattened.update({f"center_weight:{name}": row["center_weights"][name] for name in center_names})
            writer.writerow(flattened)


def _publish(output, payload, summary, centers, protected_paths):
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ["valley_projection.json", "valley_weights.csv", "valley_projection.txt"]
    for name in names:
        destination = output / name
        if destination.resolve() in protected_paths:
            raise ValueError(f"Projection output would replace an input: {destination}")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"Projection output destination is not a file: {destination}")
    # All selected states have passed before this directory is created. Stage
    # complete serializations on the destination filesystem before replacement.
    with tempfile.TemporaryDirectory(prefix=".valley-projection-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / names[0]).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_csv(stage / names[1], payload["rows"], payload["valleys"], centers)
        (stage / names[2]).write_text(summary, encoding="utf-8")
        previous = {}
        for name in names:
            destination = output / name
            if destination.exists():
                backup = stage / f"previous-{name}"
                shutil.copy2(destination, backup)
                previous[name] = backup
        output.mkdir(parents=True, exist_ok=True)
        published = []
        try:
            for name in names:
                os.replace(stage / name, output / name)
                published.append(name)
        except BaseException:
            for name in published:
                if name in previous:
                    os.replace(previous[name], output / name)
                else:
                    (output / name).unlink()
            raise


def project_bands(config_path: str | Path) -> dict:
    config = load_projection_config(config_path)
    rows, path_points, warnings = [], [], []
    distance = 0.
    previous_cart = None
    with ProjectionReader(config.input.wavefunction_h5) as reader:
        metadata = reader.metadata
        monolayer = config.default_monolayer_reciprocal()
        qcut = resolve_projection_qcut(config, metadata.lattice.reciprocal_cart)
        provenance = reader.provenance()
        for index, name in enumerate(config.analysis.kpoints):
            point = reader.read_kpoint(name, config.analysis.iband)
            cart = np.asarray(point.cart, dtype=float)
            if previous_cart is not None and index not in config.break_before:
                distance += float(np.linalg.norm(cart - previous_cart))
            path_points.append({"name": name, "frac": point.frac.tolist(), "cart_Ainv": cart.tolist(), "distance_Ainv": distance})
            previous_cart = cart.copy()
            centers = config.valley_centers
            if config.projection.projector_mode == "k_resolved_parent_valley":
                centers = adjust_centers_for_parent_valley(centers, cart,
                    metadata.lattice.reciprocal_cart, use_2d=config.projection.use_2d_momentum_only)
            projectors, weights = project_kpoint(
                point.cart.reshape(1, 3) + point.g_vectors_cart, point.coefficients,
                centers, config.valley_subspaces, monolayer, qcut,
                use_2d=config.projection.use_2d_momentum_only,
                overlap_policy=config.projection.overlap_policy,
                distance_cache=RunLocalPeriodicDistanceCache(),
            )
            warnings.extend(f"path[{index}] {name}: {message}" for message in projectors.warnings)
            rows.extend(_projection_row(index, point, band, weight)
                        for band, weight in zip(point.band_indices_vasp, weights, strict=True))
            # Retain only small scalar results and k coordinates between reads.
            del point, projectors, weights, centers, cart
        reader.verify_unchanged()
    payload = {"document_kind": PROJECTION_DOCUMENT_KIND, "schema_version": PROJECTION_SCHEMA_VERSION,
        "valleys": [valley.name for valley in config.valley_subspaces],
        "band_indices_vasp": config.analysis.iband,
        "projection": {"projector_mode": config.projection.projector_mode, "qcut_Ainv": qcut,
            "qcut_mode": config.projection.qcut_mode, "qcut_shell": config.projection.qcut_shell,
            "qcut_fraction": config.projection.qcut_fraction,
            "overlap_policy": config.projection.overlap_policy,
            "use_2d_momentum_only": config.projection.use_2d_momentum_only},
        "normalization": {"norm": "sum(abs(C)**2) over all G and spinor components",
            "normalized_weights": "raw_weight / full_state_norm", "scalar_spin_factor": 1},
        "parent_geometry": {"centers": [{"name": center.name, "layer": center.layer,
             "cart_Ainv": center.cart.tolist(), "reciprocal_cart_Ainv": center.reciprocal_cart.tolist()}
             for center in config.valley_centers],
             "valley_subspaces": [{"name": valley.name, "centers": valley.centers} for valley in config.valley_subspaces],
             "layer_transforms": config.layer_transforms},
        "path": {"kpoints": path_points, "break_before": config.break_before, "ticks": config.ticks},
        "energy_reference_eV": config.energy_reference_eV, "rows": rows, "input": provenance,
        "warnings": warnings,
        "stages": {name: "not_requested" for name in ("subspace", "irrep", "ebr", "valley_chern_mod")}}
    summary = _summary(payload)
    protected = {config.input.wavefunction_h5.resolve(), Path(config_path).resolve()}
    protected.update(path.resolve() for path in config.input.monolayer_poscars.values())
    _publish(config.output.directory, payload, summary, [center.name for center in config.valley_centers], protected)
    return {"result": payload, "result_path": config.output.directory / "valley_projection.json",
            "csv_path": config.output.directory / "valley_weights.csv",
            "summary_path": config.output.directory / "valley_projection.txt", "summary_text": summary}
