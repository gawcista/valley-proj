from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib


@dataclass(frozen=True)
class SpglibDatasetSummary:
    spacegroup_number: int
    international: str
    rotations: list[np.ndarray]
    translations: list[np.ndarray]


def find_symmetry_operations(cell, symprec: float = 1e-3, angle_tolerance: float = -1.0) -> SpglibDatasetSummary:
    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=angle_tolerance)
    if dataset is None:
        raise ValueError("spglib did not find a symmetry dataset")
    rotations = [np.asarray(rot, dtype=int) for rot in dataset.rotations]
    translations = [np.asarray(trans, dtype=float) for trans in dataset.translations]
    return SpglibDatasetSummary(
        spacegroup_number=int(dataset.number),
        international=str(dataset.international),
        rotations=rotations,
        translations=translations,
    )
