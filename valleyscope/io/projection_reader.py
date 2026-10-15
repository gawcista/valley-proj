"""Bounded diagnostic HDF5 access, without a full-payload trust certificate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from valleyscope.geometry.lattice import Lattice
from valleyscope.io.h5_reader import KPointData, _read_string

# Serialization consistency bounds, separate from representation trust gates.
GEOMETRY_SERIALIZATION_RTOL = 1.e-7
GEOMETRY_SERIALIZATION_ATOL_AINV = 1.e-8
INTEGER_GRID_ATOL = 1.e-8


def _snapshot(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def _finite(value: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains nonfinite values")
    return value


@dataclass(frozen=True)
class ProjectionMetadata:
    lattice: Lattice
    spinor: bool
    nspinor: int
    source: str | None
    extractor_provenance: str | None
    wavecar_rtag: int | None


class ProjectionReader:
    """Validate every layout; load numerical PW data for one selected k only.

    Band IDs and names are inspected over the complete file to establish unique
    selection. Coefficients and energies are read only for requested band rows.
    stat_snapshot detects ordinary file changes; it is not a payload hash or
    independent representation evidence.
    """
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._h5 = None
        self._points: dict[str, tuple[str, dict[int, int]]] = {}

    def __enter__(self) -> "ProjectionReader":
        self.snapshot = _snapshot(self.path)
        self._h5 = h5py.File(self.path, "r")
        try:
            self._validate_structure()
            self.verify_unchanged()
        except BaseException:
            self._h5.close()
            self._h5 = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.verify_unchanged()
        finally:
            self._h5.close()
            self._h5 = None

    def verify_unchanged(self) -> None:
        if _snapshot(self.path) != self.snapshot:
            raise ValueError("HDF5 source stat_snapshot changed during projection")

    @staticmethod
    def _dataset(group, key, shape=None, *, numeric=True):
        if key not in group or not isinstance(group[key], h5py.Dataset):
            raise ValueError(f"HDF5 is missing required dataset: {group.name}/{key}")
        dataset = group[key]
        if shape is not None and dataset.shape != shape:
            raise ValueError(f"{dataset.name} must have shape {shape}, got {dataset.shape}")
        if numeric and dataset.dtype.kind not in "biufc":
            raise ValueError(f"{dataset.name} must contain numeric values")
        return dataset

    def _validate_structure(self):
        h5 = self._h5
        direct = self._dataset(h5, "metadata/lattice/direct_cart", (3, 3))[()]
        reciprocal = self._dataset(h5, "metadata/lattice/reciprocal_cart", (3, 3))[()]
        for name, matrix in [("direct_cart", direct), ("reciprocal_cart", reciprocal)]:
            _finite(matrix, name)
            if np.iscomplexobj(matrix) or np.linalg.det(matrix) == 0:
                raise ValueError(f"{name} must be a real nonsingular lattice")
        if not np.allclose(reciprocal, 2. * np.pi * np.linalg.inv(direct).T,
                           rtol=GEOMETRY_SERIALIZATION_RTOL, atol=GEOMETRY_SERIALIZATION_ATOL_AINV):
            raise ValueError("HDF5 direct/reciprocal lattices conflict with B=2*pi*inv(A).T")
        spinor = self._dataset(h5, "metadata/spinor", ())[()]
        if spinor not in (False, True):
            raise ValueError("metadata/spinor must be Boolean")
        if "metadata/vasp_band_index_base" in h5:
            if self._dataset(h5, "metadata/vasp_band_index_base", ())[()] != 1:
                raise ValueError("projection requires original 1-based VASP band identities")
        for key in ("wavecar_nspin", "nspin", "ispin", "ISPIN"):
            full = f"metadata/{key}"
            if full in h5 and self._dataset(h5, full, ())[()] != 1:
                raise ValueError(f"{full}: two collinear spin channels are outside projection scope")
            if key in h5["metadata"].attrs:
                declared = np.asarray(h5["metadata"].attrs[key])
                if declared.shape != () or declared.item() != 1:
                    raise ValueError(f"metadata attribute {key}: two collinear spin channels are outside projection scope")
        if "kpoints" not in h5 or not isinstance(h5["kpoints"], h5py.Group) or not h5["kpoints"]:
            raise ValueError("HDF5 must contain a nonempty kpoints group")
        nspinors = set()
        for key, group in h5["kpoints"].items():
            if not isinstance(group, h5py.Group):
                raise ValueError(f"HDF5 kpoints/{key} must be a group")
            name_dataset = self._dataset(group, "name", (), numeric=False)
            if name_dataset.dtype.kind not in "OSU":
                raise ValueError(f"{name_dataset.name} must contain a string name")
            name = _read_string(name_dataset)
            if not name or name in self._points:
                raise ValueError(f"Duplicate or empty k-point name in HDF5: {name!r}")
            self._dataset(group, "frac", (3,))
            self._dataset(group, "cart", (3,))
            coefficients = self._dataset(group, "coefficients")
            if coefficients.ndim != 3:
                raise ValueError(f"{coefficients.name} must have shape [nb,nspinor,nG]")
            nb, nspinor, ng = coefficients.shape
            if nb < 1 or ng < 1 or nspinor not in (1, 2):
                raise ValueError(f"{coefficients.name} has unsupported or empty layout")
            nspinors.add(nspinor)
            self._dataset(group, "g_vectors_frac", (ng, 3))
            self._dataset(group, "g_vectors_cart", (ng, 3))
            self._dataset(group, "energies_eV", (nb,))
            bands_dataset = self._dataset(group, "band_indices_vasp", (nb,))
            if bands_dataset.dtype.kind not in "iu":
                raise ValueError(f"{bands_dataset.name} must contain integer VASP band identities")
            bands = bands_dataset[()]
            if np.any(bands <= 0) or len(np.unique(bands)) != nb:
                raise ValueError(f"{bands_dataset.name} has duplicate or invalid VASP band identities")
            self._points[name] = (key, {int(band): position for position, band in enumerate(bands)})
        if len(nspinors) != 1:
            raise ValueError("HDF5 k-points contain inconsistent coefficient nspinor values")
        nspinor = nspinors.pop()
        if bool(spinor) != (nspinor == 2):
            raise ValueError("metadata/spinor conflicts with coefficient nspinor")
        optional_text = lambda key: _read_string(self._dataset(h5, key, (), numeric=False)) if key in h5 else None
        self.metadata = ProjectionMetadata(
            lattice=Lattice(direct, reciprocal), spinor=bool(spinor), nspinor=nspinor,
            source=optional_text("metadata/source"),
            extractor_provenance=optional_text("metadata/extractor_identity"),
            wavecar_rtag=int(self._dataset(h5, "metadata/wavecar_rtag", ())[()]) if "metadata/wavecar_rtag" in h5 else None)

    def read_kpoint(self, name: str, bands: list[int]) -> KPointData:
        self.verify_unchanged()
        if name not in self._points:
            raise ValueError(f"HDF5 does not contain target k-point: {name}")
        key, band_positions = self._points[name]
        missing = [band for band in bands if band not in band_positions]
        if missing:
            raise ValueError(f"HDF5 k-point {name} is missing target VASP band indices: {missing}")
        positions = [band_positions[band] for band in bands]
        group = self._h5[f"kpoints/{key}"]
        # Individual HDF5 row selections preserve requested order, including
        # descending original band IDs, without selecting any unrequested row.
        coefficients = _finite(np.stack([group["coefficients"][position, :, :] for position in positions]), f"coefficients at {name}")
        energies = _finite(np.asarray([group["energies_eV"][position] for position in positions]), f"energies at {name}")
        if np.iscomplexobj(energies):
            raise ValueError(f"energies at {name} must be real")
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.sum(np.abs(coefficients) ** 2, axis=(1, 2))
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
            raise ValueError(f"coefficients at {name} require finite strictly positive full-state norms")
        values = {}
        for field in ("frac", "cart", "g_vectors_frac", "g_vectors_cart"):
            values[field] = _finite(group[field][()], f"{field} at {name}")
            if np.iscomplexobj(values[field]):
                raise ValueError(f"{field} at {name} must be real")
        reciprocal = self.metadata.lattice.reciprocal_cart
        for fraction, cartesian in (("frac", "cart"), ("g_vectors_frac", "g_vectors_cart")):
            if not np.allclose(values[cartesian], values[fraction] @ reciprocal,
                               rtol=GEOMETRY_SERIALIZATION_RTOL, atol=GEOMETRY_SERIALIZATION_ATOL_AINV):
                raise ValueError(f"{cartesian} at {name} conflicts with {fraction} @ reciprocal_cart")
        grid = values["g_vectors_frac"]
        integer_grid = np.rint(grid)
        if np.any(np.abs(grid - integer_grid) > INTEGER_GRID_ATOL):
            raise ValueError(f"g_vectors_frac at {name} must be an integer reciprocal grid")
        if len(np.unique(integer_grid, axis=0)) != len(grid):
            raise ValueError(f"g_vectors_frac at {name} contains duplicate full-3D G rows")
        self.verify_unchanged()
        return KPointData(name=name, coefficients=coefficients, energies_eV=energies,
                         band_indices_vasp=np.asarray(bands, dtype=int), **values)

    def provenance(self) -> dict:
        return {"wavefunction_h5": str(self.path), "identity_kind": "stat_snapshot",
                "stat_snapshot": dict(self.snapshot),
                "validation_scope": "all_file_structure_and_unique_ids; selected_kpoints_and_bands_numerical_values",
                "spinor": self.metadata.spinor, "nspinor": self.metadata.nspinor,
                "source": self.metadata.source,
                "extractor_provenance": self.metadata.extractor_provenance,
                "wavecar_rtag": self.metadata.wavecar_rtag}
