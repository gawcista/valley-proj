from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import yaml

from valley_proj.io.wavecar import WavecarReader


def extract_wavecar_to_h5(config_path: str | Path) -> Path:
    path = Path(config_path)
    base = path.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    input_raw = raw.get("input", {})
    extract_raw = raw.get("extract", {})
    output_raw = raw.get("output", {})
    if "wavecar" not in input_raw:
        raise ValueError("input.wavecar is required")
    if "wavefunction_h5" not in output_raw:
        raise ValueError("output.wavefunction_h5 is required")
    wavecar_path = _resolve(base, input_raw["wavecar"])
    output_h5 = _resolve(base, output_raw["wavefunction_h5"])
    kpoints = extract_raw.get("kpoints", [])
    bands_vasp = [int(value) for value in extract_raw.get("bands_vasp", [])]
    spin_index = int(extract_raw.get("spin_index", 1)) - 1
    if not kpoints:
        raise ValueError("extract.kpoints must not be empty")
    if not bands_vasp:
        raise ValueError("extract.bands_vasp must not be empty")
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with WavecarReader(wavecar_path) as reader, h5py.File(output_h5, "w") as h5:
        metadata = h5.create_group("metadata")
        lattice_group = metadata.create_group("lattice")
        lattice_group["direct_cart"] = reader.header.lattice.direct_cart
        lattice_group["reciprocal_cart"] = reader.header.lattice.reciprocal_cart
        metadata["source"] = f"WAVECAR:{wavecar_path}"
        metadata["vasp_band_index_base"] = 1
        metadata["wavecar_rtag"] = reader.header.rtag
        metadata["wavecar_record_length"] = reader.header.record_length
        metadata["wavecar_nspin"] = reader.header.nspin
        kpoints_group = h5.create_group("kpoints")

        spinor_seen = False
        for out_index, item in enumerate(kpoints):
            name = str(item["name"])
            kpoint_index = int(item["vasp_index"]) - 1
            header = reader.read_band_header(kpoint_index, spin_index=spin_index)
            g_frac = reader.generate_g_vectors_frac(header.k_frac, header.nplane)
            g_cart = g_frac @ reader.header.lattice.reciprocal_cart
            coeffs = []
            energies = []
            for band_vasp in bands_vasp:
                band_index = band_vasp - 1
                band = reader.read_band_coefficients(
                    kpoint_index,
                    band_index,
                    header.nplane,
                    spin_index=spin_index,
                )
                spinor_seen = spinor_seen or band.nspinor == 2
                coeffs.append(band.coefficients)
                energies.append(header.energies_eV[band_index])
            coefficients = _stack_coefficients(coeffs)
            group = kpoints_group.create_group(str(out_index))
            group["name"] = name
            group["frac"] = header.k_frac
            group["cart"] = header.k_frac @ reader.header.lattice.reciprocal_cart
            group["g_vectors_frac"] = g_frac
            group["g_vectors_cart"] = g_cart
            group["coefficients"] = coefficients
            group["energies_eV"] = np.asarray(energies, dtype=float)
            group["band_indices_vasp"] = np.asarray(bands_vasp, dtype=int)
            group["norms"] = np.sum(np.abs(coefficients) ** 2, axis=(1, 2))
        metadata["spinor"] = bool(spinor_seen)
    return output_h5


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _stack_coefficients(coefficients: list[np.ndarray]) -> np.ndarray:
    nspinors = {item.shape[0] for item in coefficients}
    if len(nspinors) != 1:
        raise ValueError("Selected bands mix spinor and scalar coefficient records")
    return np.stack(coefficients, axis=0)
