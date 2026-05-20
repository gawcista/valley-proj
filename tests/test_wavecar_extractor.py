from io import StringIO
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import yaml

from valleyscope.workflows.extract_wavecar import extract_wavecar_to_h5
from valleyscope.cli import main


def write_synthetic_wavecar(path: Path, *, header_nplane: int = 1, coeffs=None, encut: float = 0.01,
                            lattice=None):
    recl = 256
    nspin = 1
    rtag = 45200
    nkpts = 1
    nbands = 1
    if lattice is None:
        lattice = np.eye(3) * 20.0
    else:
        lattice = np.asarray(lattice, dtype=float)
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


# -----------------------------------------------------------------------
# A. Default ecut_adjust_tol=0.0 — behavior unchanged
# -----------------------------------------------------------------------

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
        assert h5["metadata/g_list_reconstruction_mode"][()].decode() == "exact"
        assert h5["metadata/original_encut_eV"][()] == 0.01
        assert h5["metadata/ecut_adjust_tol_eV"][()] == 0.0
        assert h5["metadata/ecut_adjust_delta_eV"][()] == 0.0
        kp = h5["kpoints/0"]
        assert kp["name"][()].decode("utf-8") == "GammaM"
        assert kp["coefficients"].shape == (1, 1, 1)
        assert kp["band_indices_vasp"][()].tolist() == [1]
        assert kp["energies_eV"][()].tolist() == [0.25]
        assert kp["ecut_adjust_delta_eV"][()] == 0.0
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


# -----------------------------------------------------------------------
# B. Exact match — no adjustment, metadata mode = exact
# -----------------------------------------------------------------------

def test_extract_wavecar_exact_match_no_adjustment(tmp_path):
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "selected_wavefunctions.h5"
    config_path = tmp_path / "extract.yaml"
    # encut=0.5, lattice 20A → generates 7 G-vectors, header_nplane=7 matches
    write_synthetic_wavecar(wavecar, header_nplane=7, encut=0.5,
                            coeffs=np.array([1.0]*7, dtype=np.complex64))
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
        assert h5["metadata/g_list_reconstruction_mode"][()].decode() == "exact"
        kp = h5["kpoints/0"]
        assert kp["ecut_adjust_delta_eV"][()] == 0.0
        assert kp["target_g_count"][()] == 7


# -----------------------------------------------------------------------
# C. Spinor-count target: nplane_record = 2 * nG
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# D. Over-generate at header ENCUT → negative delta_Ecut adjustment
# -----------------------------------------------------------------------

def test_ecut_adjust_overgenerate_negative_delta(tmp_path):
    """Header ENCUT over-generates; tol allows negative delta."""
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "wf.h5"
    config_path = tmp_path / "extract.yaml"

    # Cubic lattice 20A, encut=0.5 → 7 vectors (0,0,0 + 6 nearest).
    # header_nplane=1 → need encut_recon near 0 → delta≈-0.5
    write_synthetic_wavecar(wavecar, header_nplane=1, encut=0.5,
                            coeffs=np.array([1.0], dtype=np.complex64))
    config = {
        "input": {"wavecar": str(wavecar)},
        "extract": {
            "kpoints": [{"name": "GammaM", "vasp_index": 1}],
            "bands_vasp": [1],
            "ecut_adjust_tol": 1.0,
        },
        "output": {"wavefunction_h5": str(output_h5)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    extract_wavecar_to_h5(config_path)

    with h5py.File(output_h5, "r") as h5:
        assert h5["metadata/g_list_reconstruction_mode"][()].decode() == "ecut_adjusted"
        kp = h5["kpoints/0"]
        delta = kp["ecut_adjust_delta_eV"][()]
        assert delta < 0.0, f"Expected negative delta, got {delta}"
        assert kp["g_vectors_frac"].shape[0] == 1
        assert kp["generated_g_count_at_header_encut"][()] == 7
        assert kp["generated_g_count_final"][()] == 1
        assert kp["target_g_count"][()] == 1


# -----------------------------------------------------------------------
# E. Under-generate at header ENCUT → positive delta_Ecut adjustment
# -----------------------------------------------------------------------

def test_ecut_adjust_undergenerate_positive_delta(tmp_path):
    """Header ENCUT under-generates; tol allows positive delta."""
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "wf.h5"
    config_path = tmp_path / "extract.yaml"

    # Cubic lattice 20A, encut=0.5 → 7 vectors (shell 0:1, shell 1:6).
    # Next shell (|G|=√2): 12 vectors at E≈0.752 eV → 19 total.
    # target=19 needs encut_recon≈0.753 → delta≈+0.253
    write_synthetic_wavecar(wavecar, header_nplane=19, encut=0.5,
                            coeffs=np.array([1.0]*19, dtype=np.complex64))
    config = {
        "input": {"wavecar": str(wavecar)},
        "extract": {
            "kpoints": [{"name": "GammaM", "vasp_index": 1}],
            "bands_vasp": [1],
            "ecut_adjust_tol": 5.0,
        },
        "output": {"wavefunction_h5": str(output_h5)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    extract_wavecar_to_h5(config_path)

    with h5py.File(output_h5, "r") as h5:
        assert h5["metadata/g_list_reconstruction_mode"][()].decode() == "ecut_adjusted"
        kp = h5["kpoints/0"]
        delta = kp["ecut_adjust_delta_eV"][()]
        assert delta > 0.0, f"Expected positive delta, got {delta}"
        assert kp["g_vectors_frac"].shape[0] == 19
        assert kp["generated_g_count_at_header_encut"][()] == 7
        assert kp["generated_g_count_final"][()] == 19


# -----------------------------------------------------------------------
# F. ecut_adjust_tol insufficient → ValueError
# -----------------------------------------------------------------------

def test_ecut_adjust_tol_insufficient_raises_error(tmp_path):
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "wf.h5"
    config_path = tmp_path / "extract.yaml"

    write_synthetic_wavecar(wavecar, header_nplane=5, encut=0.5,
                            coeffs=np.array([1.0]*5, dtype=np.complex64))
    config = {
        "input": {"wavecar": str(wavecar)},
        "extract": {
            "kpoints": [{"name": "GammaM", "vasp_index": 1}],
            "bands_vasp": [1],
            "ecut_adjust_tol": 1e-9,
        },
        "output": {"wavefunction_h5": str(output_h5)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with np.testing.assert_raises(ValueError):
        extract_wavecar_to_h5(config_path)


# -----------------------------------------------------------------------
# G. HDF5 metadata records: original_encut, reconstruction_encut, delta
# -----------------------------------------------------------------------

def test_hdf5_metadata_records_ecut_adjustment_fields(tmp_path):
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "wf.h5"
    config_path = tmp_path / "extract.yaml"

    write_synthetic_wavecar(wavecar, header_nplane=1, encut=0.5,
                            coeffs=np.array([1.0], dtype=np.complex64))
    config = {
        "input": {"wavecar": str(wavecar)},
        "extract": {
            "kpoints": [{"name": "GammaM", "vasp_index": 1}],
            "bands_vasp": [1],
            "ecut_adjust_tol": 1.0,
        },
        "output": {"wavefunction_h5": str(output_h5)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    extract_wavecar_to_h5(config_path)

    with h5py.File(output_h5, "r") as h5:
        assert "metadata/original_encut_eV" in h5
        assert "metadata/reconstruction_encut_eV" in h5
        assert "metadata/ecut_adjust_tol_eV" in h5
        assert "metadata/ecut_adjust_delta_eV" in h5
        assert "metadata/g_list_reconstruction_mode" in h5
        kp = h5["kpoints/0"]
        assert "nplane_record" in kp
        assert "target_g_count" in kp
        assert "generated_g_count_at_header_encut" in kp
        assert "generated_g_count_final" in kp
        assert "ecut_adjust_delta_eV" in kp


# -----------------------------------------------------------------------
# H. stdout reports delta_Ecut and final G count when adjustment occurs
# -----------------------------------------------------------------------

def test_stdout_reports_ecut_adjustment(tmp_path, capsys):
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "wf.h5"
    config_path = tmp_path / "extract.yaml"

    write_synthetic_wavecar(wavecar, header_nplane=1, encut=0.5,
                            coeffs=np.array([1.0], dtype=np.complex64))
    config = {
        "input": {"wavecar": str(wavecar)},
        "extract": {
            "kpoints": [{"name": "GammaM", "vasp_index": 1}],
            "bands_vasp": [1],
            "ecut_adjust_tol": 1.0,
        },
        "output": {"wavefunction_h5": str(output_h5)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    extract_wavecar_to_h5(config_path)

    captured = capsys.readouterr().out
    assert "ENCUT adjustment:" in captured
    assert "final G count:" in captured
    assert "delta" not in captured.lower() or "eV" in captured


# -----------------------------------------------------------------------
# I. Strict mode (tol=0.0) raises ValueError on mismatch
# -----------------------------------------------------------------------

def test_default_tol_zero_raises_on_mismatch(tmp_path):
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "wf.h5"
    config_path = tmp_path / "extract.yaml"

    write_synthetic_wavecar(wavecar, header_nplane=5, encut=0.5,
                            coeffs=np.array([1.0]*5, dtype=np.complex64))
    config = {
        "input": {"wavecar": str(wavecar)},
        "extract": {
            "kpoints": [{"name": "GammaM", "vasp_index": 1}],
            "bands_vasp": [1],
        },
        "output": {"wavefunction_h5": str(output_h5)},
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with np.testing.assert_raises(ValueError):
        extract_wavecar_to_h5(config_path)


# -----------------------------------------------------------------------
# J. G-vector VASP order preserved
# -----------------------------------------------------------------------

def test_wavecar_g_vectors_follow_vasp_record_order(tmp_path):
    wavecar = tmp_path / "WAVECAR"
    write_synthetic_wavecar(wavecar, header_nplane=7, encut=0.5)

    from valleyscope.io.wavecar import WavecarReader

    with WavecarReader(wavecar) as reader:
        header = reader.read_band_header(0)
        gvecs, adj = reader.generate_g_vectors_frac(header.k_frac, header.nplane_record)
        assert adj is None  # exact match

    raw = np.where(gvecs >= 0, gvecs, gvecs + 3)
    order_keys = list(zip(raw[:, 2], raw[:, 1], raw[:, 0]))

    assert order_keys == sorted(order_keys)


# -----------------------------------------------------------------------
# K. No adjustment stdout for exact match
# -----------------------------------------------------------------------

def test_no_stdout_adjustment_for_exact_match(tmp_path, capsys):
    wavecar = tmp_path / "WAVECAR"
    output_h5 = tmp_path / "wf.h5"
    config_path = tmp_path / "extract.yaml"

    write_synthetic_wavecar(wavecar, header_nplane=7, encut=0.5,
                            coeffs=np.array([1.0]*7, dtype=np.complex64))
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

    captured = capsys.readouterr().out
    assert "ENCUT adjustment" not in captured
