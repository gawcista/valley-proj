from pathlib import Path

import h5py
import numpy as np
import yaml

from valley_proj.workflows.extract_wavecar import extract_wavecar_to_h5
from valley_proj.cli import main


def write_synthetic_wavecar(path: Path, *, header_nplane: int = 1, coeffs=None, encut: float = 0.01):
    recl = 256
    nspin = 1
    rtag = 45200
    nkpts = 1
    nbands = 1
    lattice = np.eye(3) * 20.0
    kvec = np.array([0.0, 0.0, 0.0])
    if coeffs is None:
        coeffs = np.array([1.0 + 0.0j], dtype=np.complex64)
    else:
        coeffs = np.asarray(coeffs, dtype=np.complex64)

    with path.open("wb") as handle:
        record = np.zeros(recl // 8, dtype=np.float64)
        record[:3] = [recl, nspin, rtag]
        handle.write(record.tobytes())

        record = np.zeros(recl // 8, dtype=np.float64)
        record[:3] = [nkpts, nbands, encut]
        record[3:12] = lattice.reshape(-1)
        handle.write(record.tobytes())

        record = np.zeros(recl // 8, dtype=np.float64)
        record[:4] = [header_nplane, *kvec]
        record[4:7] = [0.25, 0.0, 1.0]
        handle.write(record.tobytes())

        coeff_record = np.zeros(recl // np.dtype(np.complex64).itemsize, dtype=np.complex64)
        coeff_record[: len(coeffs)] = coeffs
        handle.write(coeff_record.tobytes())


def test_extract_wavecar_writes_v1_hdf5_schema(tmp_path):
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "selected_wavefunctions.h5"
    config_path = tmp_path / "extract.yaml"
    write_synthetic_wavecar(wavecar)
    config = {
        "input": {"wavecar": str(wavecar)},
        "extract": {
            "kpoints": [{"name": "GammaM", "vasp_index": 1}],
            "bands_vasp": [1],
        },
        "output": {"wavefunction_h5": str(output_h5)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = extract_wavecar_to_h5(config_path)

    assert result == output_h5
    with h5py.File(output_h5, "r") as h5:
        assert h5["metadata/spinor"][()] == np.bool_(False)
        assert h5["metadata/vasp_band_index_base"][()] == 1
        kp = h5["kpoints/0"]
        assert kp["name"][()].decode("utf-8") == "GammaM"
        assert kp["coefficients"].shape == (1, 1, 1)
        assert kp["band_indices_vasp"][()].tolist() == [1]
        assert kp["energies_eV"][()].tolist() == [0.25]
        assert np.isclose(np.sum(np.abs(kp["coefficients"][()]) ** 2), 1.0)


def test_extract_wavecar_cli_writes_hdf5(tmp_path):
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "selected_wavefunctions.h5"
    config_path = tmp_path / "extract.yaml"
    write_synthetic_wavecar(wavecar)
    config = {
        "input": {"wavecar": str(wavecar)},
        "extract": {
            "kpoints": [{"name": "GammaM", "vasp_index": 1}],
            "bands_vasp": [1],
        },
        "output": {"wavefunction_h5": str(output_h5)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    exit_code = main(["extract-wavecar", str(config_path)])

    assert exit_code == 0
    assert output_h5.exists()


def test_extract_wavecar_accepts_spinor_count_in_nplane_record(tmp_path):
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "selected_spinor_wavefunctions.h5"
    config_path = tmp_path / "extract.yaml"
    write_synthetic_wavecar(
        wavecar,
        header_nplane=2,
        coeffs=np.array([1.0 + 0.0j, 1.0j], dtype=np.complex64),
    )
    config = {
        "input": {"wavecar": str(wavecar)},
        "extract": {
            "kpoints": [{"name": "GammaM", "vasp_index": 1}],
            "bands_vasp": [1],
        },
        "output": {"wavefunction_h5": str(output_h5)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    extract_wavecar_to_h5(config_path)

    with h5py.File(output_h5, "r") as h5:
        assert h5["metadata/spinor"][()] == np.bool_(True)
        kp = h5["kpoints/0"]
        assert kp["coefficients"].shape == (1, 2, 1)
        assert np.isclose(np.sum(np.abs(kp["coefficients"][()]) ** 2), 1.0)


def test_wavecar_g_vectors_follow_vasp_record_order(tmp_path):
    wavecar = tmp_path / "WAVECAR"
    write_synthetic_wavecar(wavecar, header_nplane=7, encut=0.5)

    from valley_proj.io.wavecar import WavecarReader

    with WavecarReader(wavecar) as reader:
        header = reader.read_band_header(0)
        gvecs = reader.generate_g_vectors_frac(header.k_frac, header.nplane_record)

    raw = np.where(gvecs >= 0, gvecs, gvecs + 3)
    order_keys = list(zip(raw[:, 2], raw[:, 1], raw[:, 0]))

    assert order_keys == sorted(order_keys)
