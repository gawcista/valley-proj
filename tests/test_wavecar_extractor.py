from pathlib import Path

import h5py
import numpy as np
import yaml

from valley_proj.workflows.extract_wavecar import extract_wavecar_to_h5
from valley_proj.cli import main


def write_synthetic_wavecar(path: Path):
    recl = 256
    nspin = 1
    rtag = 45200
    nkpts = 1
    nbands = 1
    encut = 0.01
    lattice = np.eye(3) * 20.0
    kvec = np.array([0.0, 0.0, 0.0])
    coeffs = np.array([1.0 + 0.0j], dtype=np.complex64)

    with path.open("wb") as handle:
        record = np.zeros(recl // 8, dtype=np.float64)
        record[:3] = [recl, nspin, rtag]
        handle.write(record.tobytes())

        record = np.zeros(recl // 8, dtype=np.float64)
        record[:3] = [nkpts, nbands, encut]
        record[3:12] = lattice.reshape(-1)
        handle.write(record.tobytes())

        record = np.zeros(recl // 8, dtype=np.float64)
        record[:4] = [1, *kvec]
        record[4:7] = [0.25, 0.0, 1.0]
        handle.write(record.tobytes())

        coeff_record = np.zeros(recl // np.dtype(np.complex64).itemsize, dtype=np.complex64)
        coeff_record[:1] = coeffs
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
        assert np.sum(np.abs(kp["coefficients"][()]) ** 2) == 1.0


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
