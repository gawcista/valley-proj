"""Exact planar rotation geometry and conditional Chern-residue arithmetic.

These functions do not establish a globally defined valley subspace or certify
the representation evidence passed to them.
"""

from __future__ import annotations

import cmath
from fractions import Fraction
import math
from numbers import Integral, Number, Real

import numpy as np


_ORBIT_SIZES = {
    2: ((1, 1, 4),),
    3: ((1, 1, 3),),
    4: ((1, 1, 2), (2, 2, 1)),
    6: ((1, 1, 1), (2, 2, 1), (3, 3, 1)),
}
_IDENTITY = ((1, 0), (0, 1))


def _valid_order(order: object) -> int:
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, Integral):
        raise ValueError("rotation order must be an integer in {2, 3, 4, 6}")
    order = int(order)
    if order not in _ORBIT_SIZES:
        raise ValueError("rotation order must be in {2, 3, 4, 6}")
    return order


def _matrix_product(left, right):
    return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(2))
                       for j in range(2)) for i in range(2))


def _matrix_power(matrix, power):
    result = _IDENTITY
    for _ in range(power):
        result = _matrix_product(result, matrix)
    return result


def _determinant(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def _exact_rotation(matrix: object, order: int):
    raw = np.asarray(matrix, dtype=object)
    if raw.shape != (2, 2) or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in raw.flat
    ):
        raise ValueError("reciprocal rotation must be an exact integer 2x2 matrix")
    rotation = tuple(tuple(int(raw[i, j]) for j in range(2)) for i in range(2))
    if _determinant(rotation) != 1 or _matrix_power(rotation, order) != _IDENTITY:
        raise ValueError("reciprocal rotation must have determinant +1 and given order")
    if any(_matrix_power(rotation, divisor) == _IDENTITY
           for divisor in range(1, order)):
        raise ValueError("reciprocal rotation has smaller order than requested")
    return rotation


def _fixed_points(rotation, power):
    """Enumerate Z² / (R^power - I)Z² with exact rational coordinates."""
    powered = _matrix_power(rotation, power)
    a, b = powered[0][0] - 1, powered[0][1]
    c, d = powered[1][0], powered[1][1] - 1
    determinant = a * d - b * c
    if determinant == 0:
        raise ValueError("rotation power has a nonisolated fixed set")
    modulus = abs(determinant)
    sign = 1 if determinant > 0 else -1
    points = {
        (
            Fraction((sign * (d * x - b * y)) % modulus, modulus),
            Fraction((sign * (-c * x + a * y)) % modulus, modulus),
        )
        for x in range(modulus) for y in range(modulus)
    }
    if len(points) != modulus:
        raise ValueError("fixed-point quotient enumeration is inconsistent")
    return points


def _rotate_point(rotation, point):
    return tuple(
        sum(rotation[i][j] * point[j] for j in range(2)) % 1
        for i in range(2)
    )


def rotation_fixed_point_orbits(
    rotation_reciprocal_2d: np.ndarray, order: int,
) -> list[dict]:
    """Return required fixed-point orbits on the 2D reciprocal torus.

    ``power`` selects R**power. Each ``points`` list is the entire orbit
    under R, not just the representative used by a Cn indicator formula.
    """
    order = _valid_order(order)
    rotation = _exact_rotation(rotation_reciprocal_2d, order)
    rows = []
    for power, orbit_size, expected_count in _ORBIT_SIZES[order]:
        fixed = _fixed_points(rotation, power)
        seen = set()
        orbits = []
        for point in sorted(fixed):
            if point in seen:
                continue
            orbit = set()
            current = point
            while current not in orbit:
                orbit.add(current)
                current = _rotate_point(rotation, current)
            if current != point or not orbit <= fixed:
                raise ValueError("rotation orbit is inconsistent with fixed set")
            seen.update(orbit)
            if len(orbit) == orbit_size:
                orbits.append(sorted(orbit))
        if len(orbits) != expected_count:
            raise ValueError("unexpected rotation fixed-point orbit structure")
        rows.extend({"power": power, "points": [[float(x), float(y)]
                                               for x, y in orbit]}
                    for orbit in sorted(orbits))
    return rows


def rotation_chern_residue(
    order: int,
    determinants: list[complex],
    rank: int,
    spinful: bool,
    *,
    tolerance: float = 1e-6,
) -> int:
    """Return the unique C mod n compatible with trusted HSP determinants.

    Input order follows the respective C2/C3/C4/C6 rotation formula. This
    routine checks arithmetic only; it cannot establish HSP provenance.
    """
    order = _valid_order(order)
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, Integral) or rank <= 0:
        raise ValueError("subspace rank must be a positive integer")
    if not isinstance(spinful, bool):
        raise ValueError("spinful must be a Boolean")
    if (isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, Real)
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be positive and finite")
    expected_length = 4 if order == 2 else 3
    if not isinstance(determinants, list) or len(determinants) != expected_length:
        raise ValueError("incorrect number of rotation determinants")
    product = 1 + 0j
    for raw in determinants:
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Number):
            raise ValueError("each determinant must be a finite complex number")
        value = complex(raw)
        if (not math.isfinite(value.real) or not math.isfinite(value.imag)
                or abs(abs(value) - 1) > tolerance):
            raise ValueError("rotation determinant is not unit modulus")
        product *= value
    if order != 2 and spinful and rank % 2:
        product = -product
    matches = [residue for residue in range(order)
               if abs(product - cmath.exp(2j * math.pi * residue / order))
               <= tolerance]
    if len(matches) != 1:
        raise ValueError("rotation product is not near a unique nth root of unity")
    return matches[0]
