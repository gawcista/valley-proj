from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager, nullcontext
from multiprocessing import get_context
import os
from pathlib import Path
import tempfile
from time import perf_counter

import h5py
import numpy as np
import yaml

from valleyscope.io import resolve_config_path
from valleyscope.io.config import _expand_iband_item
from valleyscope.io.g_vectors import ordered_bounded_map
from valleyscope.io.wavecar import WavecarReader
from valleyscope.io.wavefunction_convention import (
    COEFFICIENT_SHAPE_ORDER,
    WAVECAR_H5_EXTRACTOR_IDENTITY,
    spinor_component_order,
)
from valleyscope.workflows.extract_resources import plan_extraction


def _stat_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _check_source(reader, identity):
    if (_stat_identity(reader.path.stat()) != identity or
            _stat_identity(os.fstat(reader._handle.fileno())) != identity):
        raise ValueError("WAVECAR changed during extraction; output was not published")


def _read_band_task(task):
    path, kpoint, band, nplane, spin, identity = task
    with WavecarReader(path) as reader:
        _check_source(reader, identity)
        result = reader.read_band_coefficients(kpoint, band, nplane, spin_index=spin)
        _check_source(reader, identity)
        return result


@contextmanager
def _worker_thread_limits(enabled):
    """Spawned workers use one native thread each, inside the process budget."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if enabled:
            os.environ.update(dict.fromkeys(keys, "1"))
        yield
    finally:
        if enabled:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def extract_wavecar_to_h5(config_path: str | Path) -> Path:
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    input_raw, extract_raw, output_raw = (raw.get(key, {}) for key in ("input", "extract", "output"))
    if "wavecar" not in input_raw:
        raise ValueError("input.wavecar is required")
    if "wavefunction_h5" not in output_raw:
        raise ValueError("output.wavefunction_h5 is required")
    wavecar_path = resolve_config_path(path.parent, input_raw["wavecar"])
    output_h5 = resolve_config_path(path.parent, output_raw["wavefunction_h5"])
    if wavecar_path is None or output_h5 is None:
        raise ValueError("WAVECAR or output path resolution failed")
    if wavecar_path.resolve() == output_h5.resolve() or (output_h5.exists() and os.path.samefile(wavecar_path, output_h5)):
        raise ValueError("output.wavefunction_h5 must differ from input.wavecar")
    kpoints = extract_raw.get("kpoints", [])
    bands_vasp = _parse_bands_vasp(extract_raw.get("bands_vasp", []), path="extract.bands_vasp")
    spin_index = int(extract_raw.get("spin_index", 1)) - 1
    ecut_adjust_tol = float(extract_raw.get("ecut_adjust_tol", 0.0))
    if not np.isfinite(ecut_adjust_tol) or ecut_adjust_tol < 0:
        raise ValueError("extract.ecut_adjust_tol must be finite and >= 0.0")
    if not kpoints or not bands_vasp:
        raise ValueError("extract.kpoints and extract.bands_vasp must not be empty")
    if len({str(item["name"]) for item in kpoints}) != len(kpoints):
        raise ValueError("extract.kpoints names must be unique")

    with WavecarReader(wavecar_path) as reader:
        identity = _stat_identity(os.fstat(reader._handle.fileno()))
        if min(bands_vasp) < 1 or max(bands_vasp) > reader.header.nbands:
            raise ValueError("band_index out of range in extract.bands_vasp")
        max_nplane = max(reader.read_band_header(int(item["vasp_index"]) - 1, spin_index=spin_index).nplane_record
                         for item in kpoints)
        plan = plan_extraction(extract_raw, record_length=reader.header.record_length, max_nplane=max_nplane)
        print(f"WAVECAR extraction: workers={plan.workers}, memory budget={plan.memory_bytes / 1024**3:.3f} GiB, "
              f"G chunk={plan.chunk_size} candidates (estimated working-set limits)")
        output_h5.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output_h5.name}.", suffix=".tmp", dir=output_h5.parent)
        os.close(descriptor)
        try:
            with _worker_thread_limits(plan.workers > 1):
                executor = (ProcessPoolExecutor(plan.workers, mp_context=get_context("spawn"))
                            if plan.workers > 1 else nullcontext(None))
                with executor as pool, h5py.File(temporary, "w") as h5:
                    reader.g_chunk_size = plan.chunk_size
                    reader.executor = pool
                    reader.max_in_flight = plan.workers
                    _write_wavefunctions(h5, reader, kpoints, bands_vasp, spin_index,
                                         ecut_adjust_tol, identity, pool, plan.workers)
            _check_source(reader, identity)
            os.replace(temporary, output_h5)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return output_h5


def _write_wavefunctions(h5, reader, kpoints, bands_vasp, spin_index, tolerance, identity, pool, workers):
    metadata = h5.create_group("metadata")
    lattice_group = metadata.create_group("lattice")
    lattice_group["direct_cart"] = reader.header.lattice.direct_cart
    lattice_group["reciprocal_cart"] = reader.header.lattice.reciprocal_cart
    for key, value in {
        "source": f"WAVECAR:{reader.path}", "vasp_band_index_base": 1,
        "wavecar_rtag": reader.header.rtag, "wavecar_record_length": reader.header.record_length,
        "wavecar_nspin": reader.header.nspin, "g_vector_order": "vasp_z_y_x",
        "extractor_identity": WAVECAR_H5_EXTRACTOR_IDENTITY,
        "coefficient_shape_order": ",".join(COEFFICIENT_SHAPE_ORDER),
        "original_encut_eV": reader.header.encut_eV, "ecut_adjust_tol_eV": tolerance,
    }.items():
        metadata[key] = value
    kpoints_group = h5.create_group("kpoints")
    mode, delta, reconstruction_encut, nspinor = "exact", 0., reader.header.encut_eV, None
    for out_index, item in enumerate(kpoints):
        start = perf_counter()
        name, kpoint_index = str(item["name"]), int(item["vasp_index"]) - 1
        header = reader.read_band_header(kpoint_index, spin_index=spin_index)
        g_frac, adjust = reader.generate_g_vectors_frac(header.k_frac, header.nplane_record, ecut_adjust_tol=tolerance)
        g_seconds = perf_counter() - start
        if adjust is not None:
            mode = "ecut_adjusted"
            if abs(adjust.delta_eV) > abs(delta):
                delta, reconstruction_encut = adjust.delta_eV, adjust.reconstruction_encut_eV
            _print_adjustment(name, header.nplane_record, adjust)
        group = kpoints_group.create_group(str(out_index))
        reciprocal = reader.header.lattice.reciprocal_cart
        for key, value in {
            "name": name, "frac": header.k_frac, "cart": header.k_frac @ reciprocal,
            "g_vectors_frac": g_frac, "g_vectors_cart": g_frac @ reciprocal,
            "energies_eV": header.energies_eV[np.asarray(bands_vasp) - 1],
            "band_indices_vasp": np.asarray(bands_vasp, dtype=int), "nplane_record": header.nplane_record,
            "target_g_count": adjust.target_g_count if adjust else len(g_frac),
            "generated_g_count_at_header_encut": adjust.generated_at_header_encut if adjust else len(g_frac),
            "generated_g_count_final": adjust.generated_at_recon_encut if adjust else len(g_frac),
            "ecut_adjust_delta_eV": adjust.delta_eV if adjust else 0.,
        }.items():
            group[key] = value
        _check_source(reader, identity)
        io_start = perf_counter()
        point_nspinor = _write_bands(group, reader, kpoint_index, len(g_frac), bands_vasp,
                                    spin_index, identity, pool, workers)
        if nspinor is not None and point_nspinor != nspinor:
            raise ValueError("Selected k-points mix spinor and scalar coefficient records")
        nspinor = point_nspinor
        print(f"  {name}: nG={len(g_frac)}, bands={len(bands_vasp)}, G reconstruction={g_seconds:.3f}s, "
              f"coefficient read/normalize/write={perf_counter() - io_start:.3f}s")
        del g_frac
    for key, value in {
        "spinor": nspinor == 2, "spinor_component_order": ",".join(spinor_component_order(nspinor)),
        "g_list_reconstruction_mode": mode, "reconstruction_encut_eV": reconstruction_encut,
        "ecut_adjust_delta_eV": delta,
    }.items():
        metadata[key] = value


def _write_bands(group, reader, kpoint_index, nplane, bands, spin_index, identity, pool, workers):
    if pool is None:
        results = (reader.read_band_coefficients(kpoint_index, band - 1, nplane, spin_index=spin_index)
                   for band in bands)
    else:
        tasks = ((str(reader.path), kpoint_index, band - 1, nplane, spin_index, identity) for band in bands)
        results = ordered_bounded_map(pool, _read_band_task, tasks, limit=workers)
    coefficients, nspinor = None, None
    norms = np.empty(len(bands), dtype=float)
    for index, band in enumerate(results):
        if coefficients is None:
            nspinor = band.nspinor
            coefficients = group.create_dataset("coefficients", shape=(len(bands), nspinor, nplane), dtype=np.complex128)
        elif nspinor != band.nspinor:
            raise ValueError("Selected bands mix spinor and scalar coefficient records")
        coefficients[index] = band.coefficients
        norms[index] = np.sum(np.abs(band.coefficients) ** 2)
        del band
    group["norms"] = norms
    return nspinor


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


def _parse_bands_vasp(raw: object, *, path: str) -> list[int]:
    """Parse inclusive VASP band ranges, deduplicating in the requested order."""
    return list(dict.fromkeys(_expand_iband_item(raw, path=path)))
