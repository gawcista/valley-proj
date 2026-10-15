"""Exact G-count cutoff intervals with the configured budget unchanged."""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from tests.test_wavecar_extractor import write_synthetic_wavecar
from valleyscope.geometry.lattice import reciprocal_from_direct
from valleyscope.io.g_vectors import enumerate_g_vectors
from valleyscope.io.wavecar import HBAR2_OVER_2M_EV_A2, WavecarReader


_DIRECT = np.diag([20., 21., 23.])
_KPOINT = np.array([.173, .291, .087])
_TARGET = 5


def _energies():
    _, energies = enumerate_g_vectors(_DIRECT, reciprocal_from_direct(_DIRECT), _KPOINT, 3.)
    return np.sort(energies)


def _write_reader_input(path, header_encut, *, nplane=_TARGET, direct=_DIRECT, kpoint=_KPOINT):
    write_synthetic_wavecar(path, header_nplane=nplane, encut=header_encut,
                            lattice=direct, coeffs=np.ones(nplane, dtype=np.complex64))
    with path.open("r+b") as stream:
        stream.seek(2 * 256 + 8)
        stream.write(np.asarray(kpoint, dtype=np.float64).tobytes())


def _reference_vectors(cutoff):
    reciprocal = reciprocal_from_direct(_DIRECT)
    # Independent scalar arithmetic and VASP z-y-x index order over a grid
    # larger than the small physical cutoff ellipsoid.
    order = [0, 1, 2, 3, 4, -4, -3, -2, -1]
    vectors = []
    for gz, gy, gx in product(order, repeat=3):
        vector = np.array([gx, gy, gz])
        q = (vector + _KPOINT) @ reciprocal
        if HBAR2_OVER_2M_EV_A2 * float(q @ q) < cutoff:
            vectors.append(vector)
    return np.asarray(vectors)


@pytest.mark.parametrize("nplane", [_TARGET, 2 * _TARGET], ids=["scalar-count", "spinor-count"])
def test_exact_interval_intersects_budget_when_midpoint_is_outside(tmp_path, nplane):
    energies = _energies()
    lower, upper = energies[_TARGET - 1:_TARGET + 1]
    gap = upper - lower
    header, tolerance = upper + gap / 4., gap / 2.
    assert abs((lower + upper) / 2. - header) > tolerance
    assert lower < header - tolerance < upper
    path = tmp_path / "WAVECAR"
    _write_reader_input(path, header, nplane=nplane)
    with WavecarReader(path, g_chunk_size=7) as reader:
        point = reader.read_band_header(0)
        vectors, adjustment = reader.generate_g_vectors_frac(point.k_frac, point.nplane_record,
                                                            ecut_adjust_tol=tolerance)
    assert adjustment is not None
    assert adjustment.generated_at_header_encut != _TARGET
    assert lower < adjustment.reconstruction_encut_eV <= upper
    assert abs(adjustment.delta_eV) <= tolerance
    assert len(vectors) == adjustment.target_g_count == adjustment.generated_at_recon_encut == _TARGET
    np.testing.assert_array_equal(vectors, _reference_vectors(adjustment.reconstruction_encut_eV))


def test_existing_midpoint_is_preserved_when_inside_budget(tmp_path):
    energies = _energies()
    lower, upper = energies[_TARGET - 1:_TARGET + 1]
    gap = upper - lower
    header, tolerance = upper + gap / 4., gap
    path = tmp_path / "WAVECAR"
    _write_reader_input(path, header)
    with WavecarReader(path) as reader:
        vectors, adjustment = reader.generate_g_vectors_frac(_KPOINT, _TARGET, ecut_adjust_tol=tolerance)
    assert adjustment.reconstruction_encut_eV == (lower + upper) / 2.
    np.testing.assert_array_equal(vectors, _reference_vectors(adjustment.reconstruction_encut_eV))


def test_strict_included_energy_uses_next_float_within_one_ulp_budget(tmp_path):
    lower = _energies()[_TARGET - 1]
    tolerance = np.spacing(lower)
    path = tmp_path / "WAVECAR"
    _write_reader_input(path, lower)
    with WavecarReader(path) as reader:
        vectors, adjustment = reader.generate_g_vectors_frac(_KPOINT, _TARGET, ecut_adjust_tol=tolerance)
    assert adjustment.reconstruction_encut_eV == np.nextafter(lower, np.inf)
    assert abs(adjustment.delta_eV) <= tolerance
    assert len(vectors) == _TARGET
    np.testing.assert_array_equal(vectors, _reference_vectors(adjustment.reconstruction_encut_eV))


def test_empty_cutoff_budget_intersection_still_fails(tmp_path):
    energies = _energies()
    lower, upper = energies[_TARGET - 1:_TARGET + 1]
    gap = upper - lower
    header, tolerance = upper + gap / 4., gap / 8.
    assert header - tolerance > upper
    path = tmp_path / "WAVECAR"
    _write_reader_input(path, header)
    with WavecarReader(path) as reader, pytest.raises(ValueError):
        reader.generate_g_vectors_frac(_KPOINT, _TARGET, ecut_adjust_tol=tolerance)


def test_degenerate_boundary_cannot_split_a_g_shell(tmp_path):
    path = tmp_path / "WAVECAR"
    _write_reader_input(path, .5, nplane=2, direct=np.eye(3) * 20., kpoint=np.zeros(3))
    with WavecarReader(path) as reader, pytest.raises(ValueError, match="degenerate"):
        reader.generate_g_vectors_frac(np.zeros(3), 2, ecut_adjust_tol=1.)
