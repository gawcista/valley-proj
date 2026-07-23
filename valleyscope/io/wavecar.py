from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from valleyscope.geometry.lattice import Lattice, reciprocal_from_direct

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


@dataclass
class GVectorAdjustment:
    """Record of an ECUT adjustment applied during G-list reconstruction."""
    original_encut_eV: float
    reconstruction_encut_eV: float
    delta_eV: float
    target_g_count: int
    generated_at_header_encut: int
    generated_at_recon_encut: int


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
        second = self._read_record(1, np.float64, count=12)
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

    def _read_record(self, record_index: int, dtype, *, count: int | None = None) -> np.ndarray:
        self._handle.seek(record_index * self.record_length)
        raw = self._handle.read(self.record_length)
        if len(raw) != self.record_length:
            raise ValueError(f"Could not read WAVECAR record {record_index + 1}")
        arr = np.frombuffer(raw, dtype=dtype)
        if count is not None:
            if count > len(arr):
                raise ValueError(f"WAVECAR record {record_index + 1} is too short for {count} values")
            return arr[:count].copy()
        return arr.copy()

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
        values = self._read_record(record_index, np.float64, count=4 + 3 * self.header.nbands)
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
        coeffs = self._read_record(record_index, self.coeff_dtype)
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

    def generate_g_vectors_frac(
        self,
        k_frac: np.ndarray,
        nplane_record: int,
        *,
        ecut_adjust_tol: float = 0.0,
    ) -> tuple[np.ndarray, GVectorAdjustment | None]:
        """Generate fractional G-vectors matching VASP's WAVECAR record ordering.

        Parameters
        ----------
        k_frac : fractional k-point coordinate
        nplane_record : expected plane-wave count from the WAVECAR band header
        ecut_adjust_tol : maximum allowed |delta_Ecut| in eV for automatic
            cutoff adjustment (default 0.0 = strict exact-match only).

        Returns
        -------
        g_vectors : fractional G-vector array [nG, 3]
        adjustment : GVectorAdjustment if ECUT was adjusted, else None
        """
        header_encut = self.header.encut_eV
        reciprocal = self.header.lattice.reciprocal_cart

        # 1. Try header ENCUT first
        arr = self._generate_g_vectors_with_encut(k_frac, header_encut, reciprocal)
        count_at_header = len(arr)

        # Exact match at header ENCUT — no adjustment needed
        target = _resolve_target_count(nplane_record, count_at_header)
        if target is not None:
            return arr, None

        # 2. Strict mode: no tolerance — raise immediately
        if ecut_adjust_tol <= 0.0:
            targets = _target_candidates(nplane_record, count_at_header)
            hint = _format_hint(nplane_record, targets)
            raise ValueError(
                f"G-vector count mismatch: generated {count_at_header} at header ENCUT "
                f"({self.header.encut_eV:.4f} eV), but WAVECAR reports {nplane_record} "
                f"(expected target G count {hint}). "
                f"This is a strict exact-match failure. "
                f"To attempt automatic ENCUT adjustment, set extract.ecut_adjust_tol "
                f"to a small positive value (start from 0.005 eV or 0.01 eV)."
            )

        # 3. Automatic adjustment
        return self._adjust_encut(
            k_frac, nplane_record, count_at_header, header_encut, ecut_adjust_tol, reciprocal
        )

    def _generate_g_vectors_with_encut(
        self, k_frac: np.ndarray, encut: float, reciprocal: np.ndarray
    ) -> np.ndarray:
        """Generate G-vectors for a specific ENCUT in VASP loop order."""
        direct = self.header.lattice.direct_cart
        gcut = np.sqrt(encut / HBAR2_OVER_2M_EV_A2)
        max_indices = np.ceil(gcut * np.linalg.norm(direct, axis=1) / (2.0 * np.pi)).astype(int) + 1
        vectors: list[list[int]] = []
        for k_raw in range(2 * max_indices[2] + 1):
            k = _wrap_fft_index(k_raw, max_indices[2])
            for j_raw in range(2 * max_indices[1] + 1):
                j = _wrap_fft_index(j_raw, max_indices[1])
                for i_raw in range(2 * max_indices[0] + 1):
                    i = _wrap_fft_index(i_raw, max_indices[0])
                    g_frac = np.array([i, j, k], dtype=float)
                    q_cart = (g_frac + k_frac) @ reciprocal
                    energy = HBAR2_OVER_2M_EV_A2 * float(q_cart @ q_cart)
                    if energy < encut:
                        vectors.append([i, j, k])
        return np.asarray(vectors, dtype=int)

    def _generate_g_vectors_with_energies(
        self, k_frac: np.ndarray, encut: float, reciprocal: np.ndarray
    ) -> tuple[list[list[int]], list[float]]:
        """Generate G-vectors and their kinetic energies in VASP loop order."""
        direct = self.header.lattice.direct_cart
        gcut = np.sqrt(encut / HBAR2_OVER_2M_EV_A2)
        max_indices = np.ceil(gcut * np.linalg.norm(direct, axis=1) / (2.0 * np.pi)).astype(int) + 1
        vectors: list[list[int]] = []
        energies: list[float] = []
        for k_raw in range(2 * max_indices[2] + 1):
            k = _wrap_fft_index(k_raw, max_indices[2])
            for j_raw in range(2 * max_indices[1] + 1):
                j = _wrap_fft_index(j_raw, max_indices[1])
                for i_raw in range(2 * max_indices[0] + 1):
                    i = _wrap_fft_index(i_raw, max_indices[0])
                    g_frac = np.array([i, j, k], dtype=float)
                    q_cart = (g_frac + k_frac) @ reciprocal
                    energy = HBAR2_OVER_2M_EV_A2 * float(q_cart @ q_cart)
                    if energy < encut:
                        vectors.append([i, j, k])
                        energies.append(energy)
        return vectors, energies

    def _adjust_encut(
        self,
        k_frac: np.ndarray,
        nplane_record: int,
        count_at_header: int,
        header_encut: float,
        ecut_adjust_tol: float,
        reciprocal: np.ndarray,
    ) -> tuple[np.ndarray, GVectorAdjustment | None]:
        """Search for an adjusted ENCUT within tolerance that yields exact target count."""
        # Determine target count
        target_candidates = _target_candidates(nplane_record, count_at_header)
        target_count = _choose_target(target_candidates, count_at_header)

        # Collect candidate G-vectors up to header_encut + ecut_adjust_tol
        max_encut = header_encut + ecut_adjust_tol
        raw_vectors, raw_energies = self._generate_g_vectors_with_energies(
            k_frac, max_encut, reciprocal
        )

        if target_count > len(raw_energies):
            raise ValueError(
                f"Cannot reach target G count {target_count}: only {len(raw_energies)} "
                f"G-vector candidates within [{header_encut:.6f}, {max_encut:.6f}] eV.\n"
                f"  Generated at header ENCUT: {count_at_header} G-vectors\n"
                f"  Header ENCUT: {header_encut:.4f} eV\n"
                f"  ecut_adjust_tol: {ecut_adjust_tol:.4f} eV\n"
                f"  Suggested action: increase ecut_adjust_tol if this is a genuine "
                f"cutoff-boundary issue, or verify WAVECAR variant / lattice convention."
            )

        # Sort by energy to find the boundary
        sorted_idx = np.argsort(raw_energies)
        sorted_energies = np.asarray(raw_energies)[sorted_idx]

        # ecut_recon is placed between the target_count-th and (target_count+1)-th
        if target_count < len(sorted_energies):
            ecut_recon = (sorted_energies[target_count - 1] + sorted_energies[target_count]) / 2.0
        else:
            ecut_recon = sorted_energies[-1] + _TINY_ENERGY_EV

        delta = ecut_recon - header_encut

        if abs(delta) > ecut_adjust_tol + 1e-10:
            closest_achievable = len(sorted_energies)
            if target_count < len(sorted_energies):
                lower_boundary_delta = sorted_energies[target_count - 1] - header_encut
                upper_boundary_delta = sorted_energies[target_count] - header_encut
            else:
                lower_boundary_delta = float("inf")
                upper_boundary_delta = float("inf")
            raise ValueError(
                f"Reconstruction cutoff delta {delta:+.6f} eV (midpoint) exceeds "
                f"ecut_adjust_tol={ecut_adjust_tol:.4f} eV.\n"
                f"  Header ENCUT: {header_encut:.4f} eV\n"
                f"  Generated at header ENCUT: {count_at_header} G-vectors\n"
                f"  Target G count: {target_count}\n"
                f"  Closest achievable count within tolerance: {closest_achievable}\n"
                f"  Lower boundary delta_Ecut: {lower_boundary_delta:+.6f} eV "
                f"(energy of last included G-vector minus header ENCUT)\n"
                f"  Upper boundary delta_Ecut: {upper_boundary_delta:+.6f} eV "
                f"(energy of first excluded G-vector minus header ENCUT)\n"
                f"  Suggested action: if the mismatch is a genuine cutoff-boundary "
                f"reconstruction issue, increase ecut_adjust_tol. "
                f"Otherwise check WAVECAR variant, lattice convention, G-list ordering, "
                f"and k-point convention."
            )

        # Regenerate with adjusted ENCUT
        arr = self._generate_g_vectors_with_encut(k_frac, ecut_recon, reciprocal)

        if len(arr) != target_count:
            raise ValueError(
                f"ENCUT adjustment failed: generated {len(arr)} vectors at "
                f"reconstruction ENCUT={ecut_recon:.6f} eV, expected {target_count}. "
                f"This may indicate degenerate energies at the cutoff boundary."
            )

        return arr, GVectorAdjustment(
            original_encut_eV=header_encut,
            reconstruction_encut_eV=ecut_recon,
            delta_eV=delta,
            target_g_count=target_count,
            generated_at_header_encut=count_at_header,
            generated_at_recon_encut=len(arr),
        )


_TINY_ENERGY_EV = 1e-9


def _resolve_target_count(nplane_record: int, count_at_header: int) -> int | None:
    """Return the matching target count or None if no exact match.

    Handles both scalar (nplane_record = nG) and spinor (nplane_record = 2*nG).
    """
    if count_at_header == nplane_record:
        return nplane_record
    if nplane_record % 2 == 0 and count_at_header == nplane_record // 2:
        return nplane_record // 2
    return None


def _target_candidates(nplane_record: int, count_at_header: int) -> list[int]:
    """Return list of plausible target G counts."""
    cand = [nplane_record]
    if nplane_record % 2 == 0:
        cand.append(nplane_record // 2)
    return sorted(set(cand))


def _choose_target(candidates: list[int], count_at_header: int) -> int:
    """Choose the best target count among candidates derived from nplane_record.

    Excludes count_at_header itself (which is what we already have) and
    picks the remaining candidate closest to it. This ensures the target
    comes from the WAVECAR's actual record, not from our reconstruction.
    """
    primary = [c for c in candidates if c != count_at_header]
    if primary:
        return min(primary, key=lambda t: abs(t - count_at_header))
    return count_at_header


def _format_hint(nplane_record: int, targets: list[int]) -> str:
    if len(targets) == 1:
        return str(targets[0])
    return f"{targets[0]} or {targets[1]} for spinor-count records"


def _wrap_fft_index(raw: int, max_index: int) -> int:
    return raw if raw <= max_index else raw - (2 * max_index + 1)


def _trim_complex_record(values: np.ndarray, tolerance: float = 0.0) -> int:
    nonzero = np.where(np.abs(values) > tolerance)[0]
    if len(nonzero) == 0:
        return 0
    return int(nonzero[-1] + 1)
