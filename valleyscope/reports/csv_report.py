from __future__ import annotations

import csv
from pathlib import Path

from valleyscope.projection.weights import ValleyWeightResult


def write_valley_weights_csv(path: str | Path, rows: list[dict[str, object]], sector_names: list[str]) -> Path:
    out = Path(path)
    fieldnames = ["kpoint", "band_vasp", "energy_eV", *sector_names, "W_val", "P_v", "eta", "W_overlap", "W_res"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def write_rotation_eigenvalues_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out = Path(path)
    fieldnames = [
        "kpoint",
        "operation_id",
        "order",
        "basis",
        "state_index",
        "eigenvalue_real",
        "eigenvalue_imag",
        "phase_2pi",
        "modulus_deviation",
        "unitarity_deviation",
        "rotation_ready",
        "topology_input_ready",
        "topology_ready",
        "spinor_rotation_applied",
        "spinor_convention_verified",
        "spinor_convention",
        "spinor_benchmark",
        "diagnostic_only",
        "D_valley_offdiag_norm",
        "nearest_root_of_unity",
        "root_deviation",
        "reason",
        "valley_eta",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def weight_row(
    *,
    kpoint: str,
    band_vasp: int,
    energy_eV: float,
    result: ValleyWeightResult,
    sector_names: list[str],
) -> dict[str, object]:
    row: dict[str, object] = {
        "kpoint": kpoint,
        "band_vasp": band_vasp,
        "energy_eV": energy_eV,
        "W_val": result.w_val,
        "P_v": result.purity,
        "eta": "" if result.eta is None else result.eta,
        "W_overlap": result.overlap_weight,
        "W_res": result.residual_weight,
    }
    for sector in sector_names:
        row[sector] = result.sector_weights.get(sector, 0.0)
    return row
