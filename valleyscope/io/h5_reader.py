from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from valleyscope.geometry.lattice import Lattice


@dataclass(frozen=True)
class WavefunctionMetadata:
    lattice: Lattice
    spinor: bool
    source: str
    vasp_band_index_base: int


@dataclass(frozen=True)
class KPointData:
    name: str
    frac: np.ndarray
    cart: np.ndarray
    g_vectors_frac: np.ndarray
    g_vectors_cart: np.ndarray
    coefficients: np.ndarray
    energies_eV: np.ndarray
    band_indices_vasp: np.ndarray


@dataclass(frozen=True)
class WavefunctionData:
    metadata: WavefunctionMetadata
    kpoints: list[KPointData]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_kpoint_by_name", {kp.name: kp for kp in self.kpoints})

    def find_kpoint(self, name: str) -> KPointData:
        try:
            return self._kpoint_by_name[name]
        except KeyError:
            raise KeyError(f"HDF5 does not contain target k-point: {name}")


def _read_string(dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_wavefunction_h5(path: str | Path) -> WavefunctionData:
    h5_path = Path(path)
    with h5py.File(h5_path, "r") as h5:
        for required in ["metadata/lattice/direct_cart", "metadata/lattice/reciprocal_cart", "kpoints"]:
            if required not in h5:
                raise ValueError(f"HDF5 is missing required dataset/group: {required}")
        metadata = WavefunctionMetadata(
            lattice=Lattice(
                direct_cart=h5["metadata/lattice/direct_cart"][()],
                reciprocal_cart=h5["metadata/lattice/reciprocal_cart"][()],
            ),
            spinor=bool(h5["metadata/spinor"][()]),
            source=_read_string(h5["metadata/source"]),
            vasp_band_index_base=int(h5["metadata/vasp_band_index_base"][()]),
        )
        kpoints: list[KPointData] = []
        for key in sorted(h5["kpoints"], key=lambda item: int(item)):
            group = h5["kpoints"][key]
            for name in [
                "name",
                "frac",
                "cart",
                "g_vectors_frac",
                "g_vectors_cart",
                "coefficients",
                "energies_eV",
                "band_indices_vasp",
            ]:
                if name not in group:
                    raise ValueError(f"HDF5 k-point {key} is missing {name}")
            coefficients = group["coefficients"][()]
            g_vectors_cart = group["g_vectors_cart"][()]
            energies = group["energies_eV"][()]
            bands = group["band_indices_vasp"][()]
            if coefficients.ndim != 3:
                raise ValueError(f"coefficients for k-point {key} must have shape [nb,nspinor,nG]")
            if coefficients.shape[2] != g_vectors_cart.shape[0]:
                raise ValueError(f"coefficients nG does not match g_vectors_cart for k-point {key}")
            if coefficients.shape[0] != len(energies) or coefficients.shape[0] != len(bands):
                raise ValueError(f"coefficients nb does not match energies/bands for k-point {key}")
            kpoints.append(
                KPointData(
                    name=_read_string(group["name"]),
                    frac=group["frac"][()],
                    cart=group["cart"][()],
                    g_vectors_frac=group["g_vectors_frac"][()],
                    g_vectors_cart=g_vectors_cart,
                    coefficients=coefficients,
                    energies_eV=energies,
                    band_indices_vasp=bands,
                )
            )
    seen_names: set[str] = set()
    for kp in kpoints:
        if kp.name in seen_names:
            raise ValueError(f"Duplicate k-point name in HDF5: {kp.name}")
        seen_names.add(kp.name)
    return WavefunctionData(metadata=metadata, kpoints=kpoints)
