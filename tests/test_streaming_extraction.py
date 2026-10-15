"""Portable extraction regressions using independently constructed WAVECAR records.

The fixtures cover coefficient transport and normalization, not independent
validation of VASP's binary format or the physical validity of a valley frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import h5py
import numpy as np
import pytest
import yaml

from valleyscope.io.wavecar import WavecarReader
from valleyscope.workflows import extract_wavecar as extraction


@dataclass(frozen=True)
class WavecarFixture:
    path: Path
    direct: np.ndarray
    reciprocal: np.ndarray
    kpoints: tuple[np.ndarray, ...]
    g_vectors: tuple[np.ndarray, ...]
    coefficients: tuple[tuple[np.ndarray, ...], ...]
    energies: np.ndarray
    record_length: int
    rtag: int
    nspinor: int


def _reference_g_vectors(kpoint, reciprocal, encut):
    # Independent bounded reference: all admitted vectors lie strictly inside
    # this box. The explicit axis inventory specifies the WAVECAR FFT order.
    axis = (0, 1, 2, 3, 4, -4, -3, -2, -1)
    selected = []
    for z, y, x in product(axis, repeat=3):
        vector = np.array([x, y, z], dtype=int)
        q_cart = (vector + kpoint) @ reciprocal
        if 3.80998212 * np.dot(q_cart, q_cart) < encut:
            assert np.max(np.abs(vector)) < 4
            selected.append(vector)
    return np.asarray(selected, dtype=int)


def _write_wavecar(path: Path, *, rtag=45200, nspinor=2):
    """Write two non-Gamma k points and four nontrivial complex bands."""
    direct = np.array([[18.0, 0.0, 0.0], [3.0, 19.0, 0.0], [1.0, 2.0, 17.0]])
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct).T
    kpoints = (np.array([0.12, -0.18, 0.08]), np.array([-0.2, 0.11, 0.0]))
    encut = 0.8
    g_vectors = tuple(_reference_g_vectors(k, reciprocal, encut) for k in kpoints)
    dtype = np.complex64 if rtag == 45200 else np.complex128
    record_length = 4096
    nbands = 4
    energies = np.array([[-0.83, -0.12, 0.31, 1.07], [-0.73, -0.04, 0.49, 1.19]])
    coefficients = []

    with path.open("wb") as handle:
        def write_header(values):
            record = np.zeros(record_length // 8, dtype=np.float64)
            record[: len(values)] = values
            handle.write(record.tobytes())

        write_header([record_length, 1, rtag])
        write_header([len(kpoints), nbands, encut, *direct.reshape(-1)])
        for ik, (kpoint, vectors) in enumerate(zip(kpoints, g_vectors)):
            rows = []
            for energy in energies[ik]:
                rows.extend([energy, 0.0, float(energy < 0)])
            write_header([nspinor * len(vectors), *kpoint, *rows])
            bands = []
            for ib in range(nbands):
                index = np.arange(1, nspinor * len(vectors) + 1, dtype=float)
                values = (
                    0.071 * index + 0.13 * (ib + 1) + 0.19 * ik
                    + 1j * (0.037 * index**2 - 0.11 * (ib + 1) + 0.17 * ik)
                ).astype(dtype).reshape(nspinor, len(vectors))
                # A nonzero final element avoids conflating this performance
                # regression with the existing trailing-zero layout issue.
                assert values.reshape(-1)[-1] != 0
                record = np.zeros(record_length // np.dtype(dtype).itemsize, dtype=dtype)
                record[: values.size] = values.reshape(-1)
                handle.write(record.tobytes())
                bands.append(values)
            coefficients.append(tuple(bands))
    return WavecarFixture(
        path, direct, reciprocal, kpoints, g_vectors, tuple(coefficients),
        energies, record_length, rtag, nspinor,
    )


def _write_config(tmp_path, fixture, *, name="extract", bands=None, **options):
    config = {
        "input": {"wavecar": str(fixture.path)},
        "extract": {
            "kpoints": [
                {"name": "second", "vasp_index": 2},
                {"name": "first", "vasp_index": 1},
            ],
            "bands_vasp": [4, 2, 1] if bands is None else bands,
            **options,
        },
        "output": {"wavefunction_h5": str(tmp_path / f"{name}.h5")},
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, Path(config["output"]["wavefunction_h5"])


def _source_normalized(values):
    # Preserve the existing source-dtype reduction/division order, including
    # complex64 rounding, before the final complex128 conversion.
    norm = np.sqrt(float(np.sum(np.abs(values) ** 2)))
    return (values / norm).astype(np.complex128)


def _assert_complete_values(path, fixture, bands=(4, 2, 1)):
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5["metadata/lattice/direct_cart"][()], fixture.direct)
        np.testing.assert_array_equal(h5["metadata/lattice/reciprocal_cart"][()], fixture.reciprocal)
        assert h5["metadata/wavecar_rtag"][()] == fixture.rtag
        assert bool(h5["metadata/spinor"][()]) == (fixture.nspinor == 2)
        for output_index, source_index in enumerate((1, 0)):
            group = h5[f"kpoints/{output_index}"]
            assert group["name"][()].decode() == ("second", "first")[output_index]
            np.testing.assert_array_equal(group["frac"][()], fixture.kpoints[source_index])
            np.testing.assert_array_equal(group["cart"][()], fixture.kpoints[source_index] @ fixture.reciprocal)
            np.testing.assert_array_equal(group["g_vectors_frac"][()], fixture.g_vectors[source_index])
            np.testing.assert_array_equal(group["g_vectors_cart"][()], fixture.g_vectors[source_index] @ fixture.reciprocal)
            expected = np.stack([
                _source_normalized(fixture.coefficients[source_index][band - 1])
                for band in bands
            ])
            assert group["coefficients"].dtype == np.dtype(np.complex128)
            np.testing.assert_array_equal(group["coefficients"][()], expected)
            np.testing.assert_array_equal(group["norms"][()], np.sum(np.abs(expected) ** 2, axis=(1, 2)))
            np.testing.assert_array_equal(group["energies_eV"][()], fixture.energies[source_index, np.asarray(bands) - 1])
            np.testing.assert_array_equal(group["band_indices_vasp"][()], bands)
            assert group["nplane_record"][()] == fixture.nspinor * len(fixture.g_vectors[source_index])
            for key in ("target_g_count", "generated_g_count_at_header_encut", "generated_g_count_final"):
                assert group[key][()] == len(fixture.g_vectors[source_index])


def _datasets(path):
    result = {}
    with h5py.File(path, "r") as h5:
        def collect(name, obj):
            if isinstance(obj, h5py.Dataset) and name != "metadata/source":
                result[name] = obj[()]
        h5.visititems(collect)
    return result


@pytest.mark.parametrize("rtag", [45200, 45210])
@pytest.mark.parametrize("nspinor", [1, 2])
def test_serial_extraction_preserves_complete_source_values_and_requested_order(tmp_path, rtag, nspinor):
    fixture = _write_wavecar(tmp_path / "WAVECAR", rtag=rtag, nspinor=nspinor)
    config, output = _write_config(tmp_path, fixture, workers=1, g_chunk_size=3)

    assert extraction.extract_wavecar_to_h5(config) == output

    _assert_complete_values(output, fixture)


@pytest.mark.parametrize("rtag,nspinor", [(45200, 2), (45210, 1)])
def test_spawned_cli_workers_match_serial_datasets_exactly(tmp_path, rtag, nspinor):
    from valleyscope.workflows.extract_resources import plan_extraction

    fixture = _write_wavecar(tmp_path / "WAVECAR", rtag=rtag, nspinor=nspinor)
    options = {"workers": 2, "memory_gb": 2, "g_chunk_size": 3}
    try:
        plan = plan_extraction(
            options, record_length=fixture.record_length,
            max_nplane=max(fixture.nspinor * len(vectors) for vectors in fixture.g_vectors),
        )
    except ValueError as exc:
        if str(exc) == "Memory budget cannot fit one extraction worker":
            pytest.skip(f"Actual runtime resource allowance is insufficient: {exc}")
        raise
    if plan.workers < 2:
        pytest.skip("Actual CPU/memory allowance permits fewer than two extraction workers")
    serial_config, serial_output = _write_config(tmp_path, fixture, name="serial", workers=1)
    parallel_config, parallel_output = _write_config(
        tmp_path, fixture, name="parallel", **options,
    )
    extraction.extract_wavecar_to_h5(serial_config)
    # Spawn imports this module as __mp_main__. The probe observes real reader
    # calls without replacing coefficients, headers, G generation, or results.
    probe = tmp_path / "cli_spawn_probe.py"
    pid_log = tmp_path / "reader_processes.jsonl"
    probe.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from valleyscope.io.wavecar import WavecarReader\n"
        "from valleyscope.cli import main\n"
        "original = WavecarReader.read_band_coefficients\n"
        "def observed(self, *args, **kwargs):\n"
        "    with Path(sys.argv[2]).open('a', encoding='utf-8') as handle:\n"
        "        handle.write(json.dumps({'pid': os.getpid(), 'module': __name__}) + '\\n')\n"
        "    return original(self, *args, **kwargs)\n"
        "WavecarReader.read_band_coefficients = observed\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main(['extract-wavecar', sys.argv[1]]))\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    project_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(project_root) + os.pathsep + environment.get("PYTHONPATH", "")

    completed = subprocess.run(
        [sys.executable, str(probe), str(parallel_config), str(pid_log)],
        cwd=project_root, env=environment, capture_output=True, text=True, timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    _assert_complete_values(parallel_output, fixture)
    serial, parallel = _datasets(serial_output), _datasets(parallel_output)
    assert serial.keys() == parallel.keys()
    for name in serial:
        np.testing.assert_array_equal(parallel[name], serial[name], err_msg=name)
    observations = [json.loads(line) for line in pid_log.read_text(encoding="utf-8").splitlines()]
    assert any(item["module"] == "__mp_main__" for item in observations), observations


def test_streaming_releases_coefficients_before_accumulating_selected_bands(tmp_path, monkeypatch):
    fixture = _write_wavecar(tmp_path / "WAVECAR")
    config, output = _write_config(tmp_path, fixture, bands=[4, 2, 1, 3], workers=1)
    references = []
    live_counts = []
    original = WavecarReader.read_band_coefficients

    def observed(self, *args, **kwargs):
        band = original(self, *args, **kwargs)
        gc.collect()
        references.append(weakref.ref(band.coefficients))
        live_counts.append(sum(reference() is not None for reference in references))
        return band

    monkeypatch.setattr(WavecarReader, "read_band_coefficients", observed)

    extraction.extract_wavecar_to_h5(config)

    _assert_complete_values(output, fixture, bands=(4, 2, 1, 3))
    assert len(references) == 8
    assert max(live_counts) <= 2, f"retained coefficient arrays per read: {live_counts}"
    gc.collect()
    assert all(reference() is None for reference in references)


def test_streaming_does_not_build_a_full_band_stack(tmp_path, monkeypatch):
    fixture = _write_wavecar(tmp_path / "WAVECAR")
    config, output = _write_config(tmp_path, fixture, workers=1)

    def prohibited(*args, **kwargs):
        pytest.fail("Extraction accumulated a full selected-band coefficient stack")

    if hasattr(extraction, "_stack_coefficients"):
        monkeypatch.setattr(extraction, "_stack_coefficients", prohibited)

    extraction.extract_wavecar_to_h5(config)

    _assert_complete_values(output, fixture)


@pytest.mark.parametrize("workers", [1, 2])
def test_truncated_record_keeps_existing_output_and_removes_temporary_files(tmp_path, workers):
    fixture = _write_wavecar(tmp_path / "WAVECAR")
    with fixture.path.open("r+b") as handle:
        handle.truncate(fixture.path.stat().st_size - 23)
    config, output = _write_config(tmp_path, fixture, bands=[1, 2, 4], workers=workers)
    with h5py.File(output, "w") as h5:
        h5["previous_successful_payload"] = np.array([2.0, 7.0, 11.0])
    original_bytes = output.read_bytes()
    original_entries = set(tmp_path.iterdir())

    with pytest.raises(ValueError, match="WAVECAR record"):
        extraction.extract_wavecar_to_h5(config)

    assert output.read_bytes() == original_bytes
    assert set(tmp_path.iterdir()) == original_entries


@pytest.mark.parametrize("key,value", [
    ("workers", 0), ("workers", -1), ("workers", 1.5), ("workers", True),
    ("workers", "2"), ("workers", "invalid"), ("workers", None),
    ("memory_gb", 0), ("memory_gb", -0.5), ("memory_gb", float("inf")),
    ("memory_gb", float("nan")), ("memory_gb", True), ("memory_gb", "1"),
    ("g_chunk_size", 0), ("g_chunk_size", -1), ("g_chunk_size", 1.5),
    ("g_chunk_size", True), ("g_chunk_size", "3"), ("g_chunk_size", None),
])
def test_invalid_resource_config_does_not_modify_existing_output(tmp_path, key, value):
    fixture = _write_wavecar(tmp_path / "WAVECAR")
    config, output = _write_config(tmp_path, fixture, **{key: value})
    output.write_bytes(b"previous successful output")
    original_entries = set(tmp_path.iterdir())

    with pytest.raises(ValueError, match=rf"extract\.{key}"):
        extraction.extract_wavecar_to_h5(config)

    assert output.read_bytes() == b"previous successful output"
    assert set(tmp_path.iterdir()) == original_entries


def test_changed_source_does_not_publish_partial_extraction(tmp_path, monkeypatch):
    fixture = _write_wavecar(tmp_path / "WAVECAR")
    config, output = _write_config(tmp_path, fixture)
    output.write_bytes(b"previous output")
    original = WavecarReader.read_band_coefficients

    def changed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        stat = fixture.path.stat()
        os.utime(fixture.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        return result

    monkeypatch.setattr(WavecarReader, "read_band_coefficients", changed)
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="WAVECAR changed"):
        extraction.extract_wavecar_to_h5(config)
    assert output.read_bytes() == b"previous output"
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("alias", ["same_path", "hardlink"])
def test_input_cannot_be_used_as_output(tmp_path, alias):
    fixture = _write_wavecar(tmp_path / "WAVECAR")
    config, output = _write_config(tmp_path, fixture)
    if alias == "same_path":
        output = fixture.path
    else:
        os.link(fixture.path, output)
    raw = yaml.safe_load(config.read_text())
    raw["output"]["wavefunction_h5"] = str(output)
    config.write_text(yaml.safe_dump(raw))
    original = fixture.path.read_bytes()
    with pytest.raises(ValueError, match="must differ"):
        extraction.extract_wavecar_to_h5(config)
    assert fixture.path.read_bytes() == original


def test_parallel_extraction_restores_native_thread_environment(tmp_path, monkeypatch):
    fixture = _write_wavecar(tmp_path / "WAVECAR")
    config, output = _write_config(tmp_path, fixture, workers=2, memory_gb=2)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("BLIS_NUM_THREADS", raising=False)
    extraction.extract_wavecar_to_h5(config)
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert "BLIS_NUM_THREADS" not in os.environ
    _assert_complete_values(output, fixture)
