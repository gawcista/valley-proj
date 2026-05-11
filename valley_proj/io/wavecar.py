from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from valley_proj.geometry.lattice import Lattice, reciprocal_from_direct

HBAR2_OVER_2M_EV_A2 = 3.80998212


@dataclass(frozen=True)
class WavecarHeader:
    record_length: int
    nspin: int
    rtag: int
    nkpts: int
    nbands: int
    encut_eV: float
    lattice: Lattice


@dataclass(frozen=True)
class WavecarBandHeader:
    nplane_record: int
    k_frac: np.ndarray
    energies_eV: np.ndarray
    occupations: np.ndarray


@dataclass(frozen=True)
class WavecarBandData:
    coefficients: np.ndarray
    nspinor: int


class WavecarReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = self.path.open("rb")
        first_raw = self._handle.read(3 * np.dtype(np.float64).itemsize)
        if len(first_raw) != 3 * np.dtype(np.float64).itemsize:
            raise ValueError("Could not read WAVECAR header")
        first = np.frombuffer(first_raw, dtype=np.float64)
        self.record_length = int(round(first[0]))
        self.nspin = int(round(first[1]))
        self.rtag = int(round(first[2]))
        if self.record_length <= 0:
            raise ValueError("Invalid WAVECAR record length")
        if self.rtag == 45200:
            self.coeff_dtype = np.complex64
        elif self.rtag in (45210, 53300, 53310):
            self.coeff_dtype = np.complex128
        else:
            raise ValueError(f"Unsupported WAVECAR RTAG {self.rtag}; supported: 45200, 45210, 53300, 53310")
        second = self._read_record_raw(1, 12, np.float64)
        nkpts = int(round(second[0]))
        nbands = int(round(second[1]))
        encut = float(second[2])
        direct = np.asarray(second[3:12], dtype=float).reshape(3, 3)
        self.header = WavecarHeader(
            record_length=self.record_length,
            nspin=self.nspin,
            rtag=self.rtag,
            nkpts=nkpts,
            nbands=nbands,
            encut_eV=encut,
            lattice=Lattice(direct_cart=direct, reciprocal_cart=reciprocal_from_direct(direct)),
        )

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "WavecarReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _read_record_raw(self, record_index: int, count: int, dtype) -> np.ndarray:
        self._handle.seek(record_index * self.record_length)
        raw = self._handle.read(self.record_length)
        if len(raw) != self.record_length:
            raise ValueError(f"Could not read WAVECAR record {record_index + 1}")
        arr = np.frombuffer(raw, dtype=dtype)
        if count > len(arr):
            raise ValueError(f"WAVECAR record {record_index + 1} is too short for {count} values")
        return arr[:count].copy()

    def _read_record_all(self, record_index: int, dtype) -> np.ndarray:
        self._handle.seek(record_index * self.record_length)
        raw = self._handle.read(self.record_length)
        if len(raw) != self.record_length:
            raise ValueError(f"Could not read WAVECAR record {record_index + 1}")
        return np.frombuffer(raw, dtype=dtype).copy()

    def _record_index(self, spin_index: int, kpoint_index: int, band_offset: int | None = None) -> int:
        if spin_index < 0 or spin_index >= self.header.nspin:
            raise ValueError(f"spin_index out of range: {spin_index + 1}")
        if kpoint_index < 0 or kpoint_index >= self.header.nkpts:
            raise ValueError(f"kpoint_index out of range: {kpoint_index + 1}")
        base = 2 + spin_index * self.header.nkpts * (self.header.nbands + 1)
        base += kpoint_index * (self.header.nbands + 1)
        if band_offset is None:
            return base
        return base + 1 + band_offset

    def read_band_header(self, kpoint_index: int, *, spin_index: int = 0) -> WavecarBandHeader:
        record_index = self._record_index(spin_index, kpoint_index)
        values = self._read_record_raw(record_index, 4 + 3 * self.header.nbands, np.float64)
        bands = values[4:].reshape(self.header.nbands, 3)
        return WavecarBandHeader(
            nplane_record=int(round(values[0])),
            k_frac=np.asarray(values[1:4], dtype=float),
            energies_eV=np.asarray(bands[:, 0], dtype=float),
            occupations=np.asarray(bands[:, 2], dtype=float),
        )

    def read_band_coefficients(
        self,
        kpoint_index: int,
        band_index: int,
        nplane: int,
        *,
        spin_index: int = 0,
        normalize: bool = True,
    ) -> WavecarBandData:
        if band_index < 0 or band_index >= self.header.nbands:
            raise ValueError(f"band_index out of range: {band_index + 1}")
        record_index = self._record_index(spin_index, kpoint_index, band_index)
        coeffs = self._read_record_all(record_index, self.coeff_dtype)
        nonzero_len = _trim_complex_record(coeffs)
        if nonzero_len == nplane:
            nspinor = 1
            data = coeffs[:nplane].reshape(1, nplane)
        elif nonzero_len == 2 * nplane:
            nspinor = 2
            data = coeffs[: 2 * nplane].reshape(2, nplane)
        else:
            raise ValueError(
                f"Coefficient record length {nonzero_len} is incompatible with nplane={nplane}; "
                "this WAVECAR layout is unsupported or the G-list reconstruction is wrong"
            )
        if normalize:
            norm = np.sqrt(float(np.sum(np.abs(data) ** 2)))
            if norm > 0.0:
                data = data / norm
        return WavecarBandData(coefficients=data.astype(np.complex128), nspinor=nspinor)

    def generate_g_vectors_frac(self, k_frac: np.ndarray, nplane_record: int) -> np.ndarray:
        reciprocal = self.header.lattice.reciprocal_cart
        direct = self.header.lattice.direct_cart
        gcut = np.sqrt(self.header.encut_eV / HBAR2_OVER_2M_EV_A2)
        max_indices = np.ceil(gcut * np.linalg.norm(direct, axis=1) / (2.0 * np.pi)).astype(int) + 1
        vectors: list[list[int]] = []
        for i_raw in range(2 * max_indices[0] + 1):
            i = _wrap_fft_index(i_raw, max_indices[0])
            for j_raw in range(2 * max_indices[1] + 1):
                j = _wrap_fft_index(j_raw, max_indices[1])
                for k_raw in range(2 * max_indices[2] + 1):
                    k = _wrap_fft_index(k_raw, max_indices[2])
                    g_frac = np.array([i, j, k], dtype=float)
                    q_cart = (g_frac + k_frac) @ reciprocal
                    energy = HBAR2_OVER_2M_EV_A2 * float(q_cart @ q_cart)
                    if energy < self.header.encut_eV:
                        vectors.append([i, j, k])
        arr = np.asarray(vectors, dtype=int)
        if len(arr) == nplane_record:
            return arr
        if nplane_record % 2 == 0 and len(arr) == nplane_record // 2:
            return arr
        expected_text = f"{nplane_record} or {nplane_record // 2} for spinor-count records"
        if nplane_record % 2 != 0:
            expected_text = str(nplane_record)
        if nplane_record % 2 == 0:
            raise ValueError(
                f"Generated {len(arr)} G-vectors but WAVECAR reports {nplane_record} "
                f"({expected_text}). Unsupported WAVECAR variant or cutoff/G-list convention mismatch."
            )
        raise ValueError(
            f"Generated {len(arr)} G-vectors but WAVECAR reports {nplane_record}. "
            "Unsupported WAVECAR variant or cutoff/G-list convention mismatch."
        )


def _wrap_fft_index(raw: int, max_index: int) -> int:
    return raw if raw <= max_index else raw - (2 * max_index + 1)


def _trim_complex_record(values: np.ndarray, tolerance: float = 0.0) -> int:
    nonzero = np.where(np.abs(values) > tolerance)[0]
    if len(nonzero) == 0:
        return 0
    return int(nonzero[-1] + 1)
