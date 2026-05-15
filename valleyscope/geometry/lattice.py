from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Lattice:
    direct_cart: np.ndarray
    reciprocal_cart: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "direct_cart", np.asarray(self.direct_cart, dtype=float))
        object.__setattr__(self, "reciprocal_cart", np.asarray(self.reciprocal_cart, dtype=float))
        if self.direct_cart.shape != (3, 3):
            raise ValueError("direct_cart must have shape [3,3]")
        if self.reciprocal_cart.shape != (3, 3):
            raise ValueError("reciprocal_cart must have shape [3,3]")


def reciprocal_from_direct(direct_cart: np.ndarray) -> np.ndarray:
    direct = np.asarray(direct_cart, dtype=float)
    if direct.shape != (3, 3):
        raise ValueError("direct_cart must have shape [3,3]")
    return 2.0 * np.pi * np.linalg.inv(direct).T


def read_poscar_lattice(path: str) -> Lattice:
    lines = _read_poscar_lines(path)
    if len(lines) < 5:
        raise ValueError(f"POSCAR file is too short: {path}")
    scale = float(lines[1].split()[0])
    direct = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float)
    direct *= scale
    return Lattice(direct_cart=direct, reciprocal_cart=reciprocal_from_direct(direct))


def _read_poscar_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        lines = [line.rstrip() for line in handle]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def cart_rotation_from_fractional(
    rotation_frac: np.ndarray, direct_cart: np.ndarray, *, inv_direct_T: np.ndarray | None = None
) -> np.ndarray:
    """Convert spglib x' = W x to Cartesian column action r' = R r."""
    rotation = np.asarray(rotation_frac, dtype=float)
    if inv_direct_T is None:
        inv_direct_T = np.linalg.inv(np.asarray(direct_cart, dtype=float).T)
    else:
        inv_direct_T = np.asarray(inv_direct_T, dtype=float)
    return np.asarray(direct_cart, dtype=float).T @ rotation @ inv_direct_T


def cart_translation_from_fractional(translation_frac: np.ndarray, direct_cart: np.ndarray) -> np.ndarray:
    """Convert fractional translation to Cartesian column components."""
    return np.asarray(direct_cart, dtype=float).T @ np.asarray(translation_frac, dtype=float)


def read_poscar_cell(path: str):
    lines = _read_poscar_lines(path)
    if len(lines) < 8:
        raise ValueError(f"POSCAR file is too short: {path}")
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float) * scale
    species_or_counts = lines[5].split()
    if all(token.replace("-", "", 1).isdigit() for token in species_or_counts):
        counts = [int(token) for token in species_or_counts]
        coord_line = 6
    else:
        counts = [int(token) for token in lines[6].split()]
        coord_line = 7
    if lines[coord_line].lower().startswith("s"):
        coord_line += 1
    coord_mode = lines[coord_line].strip().lower()
    start = coord_line + 1
    total = sum(counts)
    coords = np.array([[float(x) for x in lines[start + i].split()[:3]] for i in range(total)], dtype=float)
    if coord_mode.startswith("c") or coord_mode.startswith("k"):
        positions = coords @ np.linalg.inv(lattice)
    else:
        positions = coords
    numbers = []
    for species_idx, count in enumerate(counts, start=1):
        numbers.extend([species_idx] * count)
    return lattice, positions, np.asarray(numbers, dtype=int)
