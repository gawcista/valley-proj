from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from valleyscope.geometry.lattice import Lattice
from valleyscope.io.wavefunction_convention import (
    COEFFICIENT_SHAPE_ORDER,
    H5_LAYOUT_IDENTITY,
    H5_PARSER_IDENTITY,
    file_payload_identity,
    spinor_component_order,
)


@dataclass(frozen=True)
class WavefunctionMetadata:
    lattice: Lattice
    spinor: bool
    nspinor: int
    source: str
    vasp_band_index_base: int
    coefficient_shape_order: tuple[str, ...]
    spinor_component_order: tuple[str, ...]
    parser_identity: str
    hdf5_layout_identity: str
    extractor_provenance: str | None
    hdf5_payload_identity: str


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
    payload_identity = file_payload_identity(h5_path)
    with h5py.File(h5_path, "r") as h5:
        for required in ["metadata/lattice/direct_cart", "metadata/lattice/reciprocal_cart", "kpoints"]:
            if required not in h5:
                raise ValueError(f"HDF5 is missing required dataset/group: {required}")
        spinor_metadata = bool(h5["metadata/spinor"][()])
        kpoints: list[KPointData] = []
        nspinor_values: set[int] = set()
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
            nspinor_values.add(int(coefficients.shape[1]))
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
        if len(nspinor_values) != 1:
            raise ValueError("HDF5 k-points contain inconsistent coefficient nspinor values")
        nspinor = next(iter(nspinor_values))
        if nspinor not in (1, 2):
            raise ValueError(f"HDF5 coefficient layout has unsupported nspinor={nspinor}")
        if spinor_metadata != (nspinor == 2):
            raise ValueError("metadata/spinor conflicts with coefficient nspinor")
        extractor_provenance = (
            _read_string(h5["metadata/extractor_identity"])
            if "metadata/extractor_identity" in h5
            else None
        )
        metadata = WavefunctionMetadata(
            lattice=Lattice(
                direct_cart=h5["metadata/lattice/direct_cart"][()],
                reciprocal_cart=h5["metadata/lattice/reciprocal_cart"][()],
            ),
            spinor=spinor_metadata,
            nspinor=nspinor,
            source=_read_string(h5["metadata/source"]),
            vasp_band_index_base=int(h5["metadata/vasp_band_index_base"][()]),
            coefficient_shape_order=COEFFICIENT_SHAPE_ORDER,
            spinor_component_order=spinor_component_order(nspinor),
            parser_identity=H5_PARSER_IDENTITY,
            hdf5_layout_identity=H5_LAYOUT_IDENTITY,
            extractor_provenance=extractor_provenance,
            hdf5_payload_identity=payload_identity,
        )
    seen_names: set[str] = set()
    for kp in kpoints:
        if kp.name in seen_names:
            raise ValueError(f"Duplicate k-point name in HDF5: {kp.name}")
        seen_names.add(kp.name)
    return WavefunctionData(metadata=metadata, kpoints=kpoints)
