from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import yaml

from valleyscope.io import resolve_config_path
from valleyscope.io.config import _expand_iband_item
from valleyscope.io.wavecar import WavecarReader


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
    wavecar_path = resolve_config_path(base, input_raw["wavecar"])
    if wavecar_path is None:
        raise ValueError("input.wavecar path resolution failed")
    output_h5 = resolve_config_path(base, output_raw["wavefunction_h5"])
    if output_h5 is None:
        raise ValueError("output.wavefunction_h5 path resolution failed")
    kpoints = extract_raw.get("kpoints", [])
    bands_raw = extract_raw.get("bands_vasp", [])
    bands_vasp = _parse_bands_vasp(bands_raw, path="extract.bands_vasp")
    spin_index = int(extract_raw.get("spin_index", 1)) - 1
    ecut_adjust_tol = float(extract_raw.get("ecut_adjust_tol", 0.0))
    if ecut_adjust_tol < 0.0:
        raise ValueError("extract.ecut_adjust_tol must be >= 0.0")
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
        metadata["g_vector_order"] = "vasp_z_y_x"
        metadata["original_encut_eV"] = reader.header.encut_eV
        metadata["ecut_adjust_tol_eV"] = ecut_adjust_tol
        kpoints_group = h5.create_group("kpoints")

        g_list_mode = "exact"
        representative_delta = 0.0
        representative_reconstruction_encut = reader.header.encut_eV
        spinor_seen = False
        for out_index, item in enumerate(kpoints):
            name = str(item["name"])
            kpoint_index = int(item["vasp_index"]) - 1
            header = reader.read_band_header(kpoint_index, spin_index=spin_index)
            g_frac, adjust = reader.generate_g_vectors_frac(
                header.k_frac, header.nplane_record, ecut_adjust_tol=ecut_adjust_tol
            )
            g_cart = g_frac @ reader.header.lattice.reciprocal_cart

            if adjust is not None:
                g_list_mode = "ecut_adjusted"
                if abs(adjust.delta_eV) > abs(representative_delta):
                    representative_delta = adjust.delta_eV
                    representative_reconstruction_encut = adjust.reconstruction_encut_eV
                _print_adjustment(name, header.nplane_record, adjust)

            coeffs = []
            energies = []
            for band_vasp in bands_vasp:
                band_index = band_vasp - 1
                band = reader.read_band_coefficients(
                    kpoint_index,
                    band_index,
                    len(g_frac),
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

            # Per-kpoint G-list reconstruction metadata
            group["nplane_record"] = header.nplane_record
            if adjust is not None:
                group["target_g_count"] = adjust.target_g_count
                group["generated_g_count_at_header_encut"] = adjust.generated_at_header_encut
                group["generated_g_count_final"] = adjust.generated_at_recon_encut
                group["ecut_adjust_delta_eV"] = adjust.delta_eV
            else:
                group["target_g_count"] = len(g_frac)
                group["generated_g_count_at_header_encut"] = len(g_frac)
                group["generated_g_count_final"] = len(g_frac)
                group["ecut_adjust_delta_eV"] = 0.0

        metadata["spinor"] = bool(spinor_seen)
        metadata["g_list_reconstruction_mode"] = g_list_mode
        metadata["reconstruction_encut_eV"] = representative_reconstruction_encut
        metadata["ecut_adjust_delta_eV"] = representative_delta
    return output_h5


def _print_adjustment(kpoint_name: str, nplane_record: int, adjust) -> None:
    print(
        "WAVECAR G-list reconstruction:\n"
        f"  kpoint: {kpoint_name}\n"
        f"  nplane_record: {nplane_record}\n"
        f"  original ENCUT: {adjust.original_encut_eV:.6f} eV\n"
        f"  adjusted ENCUT: {adjust.reconstruction_encut_eV:.6f} eV\n"
        f"  target G count: {adjust.target_g_count}\n"
        f"  generated at header ENCUT: {adjust.generated_at_header_encut}\n"
        f"  delta_Ecut: {adjust.delta_eV:+.6f} eV\n"
        f"  final G count: {adjust.generated_at_recon_encut}"
    )


def _stack_coefficients(coefficients: list[np.ndarray]) -> np.ndarray:
    nspinors = {item.shape[0] for item in coefficients}
    if len(nspinors) != 1:
        raise ValueError("Selected bands mix spinor and scalar coefficient records")
    return np.stack(coefficients, axis=0)


def _parse_bands_vasp(raw: object, *, path: str) -> list[int]:
    """Parse VASP band indices from the same compact syntax as analysis.iband.

    Accepts a flat integer list, an inclusive range mapping
    ``{start: N, end: M}``, a range string ``"N-M"`` or ``"N..M"``,
    or a mixed list of these forms.  Duplicates are deduplicated
    preserving order.
    """
    values = _expand_iband_item(raw, path=path)
    out: list[int] = []
    seen: set[int] = set()
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out
