"""Standard-setting evidence fixtures for lift and promotion tests.

Hall numbers and conventional centering cosets are read from the spglib Hall
database entry of the declared space group, so a fixture can only declare a
setting that the reviewed database supports.  The transform defaults to the
identity (the parent cell is the conventional cell); callers that pin a
primitive parent cell must pass the ``primitive_transform`` of that space
group so the declared transform really rebases the standard lattice.
"""

from __future__ import annotations

import numpy as np

from valleyscope.analysis.standard_setting_kmap import (
    _centering_cosets_from_hall_database,
    _standard_translation_lattice_basis,
)


def standard_hall_number(space_group_number: int) -> int:
    """First spglib Hall number of a space group (its standard setting)."""
    import spglib

    for hall_number in range(1, 531):
        space_group_type = spglib.get_spacegroup_type(hall_number)
        if (
            space_group_type is not None
            and int(space_group_type.number) == int(space_group_number)
        ):
            return hall_number
    raise AssertionError(
        f"space group {space_group_number} has no Hall database entry"
    )


def centering_cosets(space_group_number: int) -> list[list[float]]:
    """Reviewed conventional centering cosets of a space group."""
    evidence = _centering_cosets_from_hall_database(
        standard_hall_number(space_group_number)
    )
    assert evidence.get("status") == "passed", evidence
    return [
        [float(value) for value in vector]
        for vector in evidence["centering_cosets"]
    ]


def primitive_transform(space_group_number: int) -> np.ndarray:
    """Primitive basis of the standard-setting lattice (column convention)."""
    evidence = _standard_translation_lattice_basis(
        centering_cosets(space_group_number)
    )
    assert evidence.get("status") == "passed", evidence
    return np.asarray(evidence["basis"], dtype=float)


def standard_setting_identity(
    space_group_number: int,
    *,
    transform: object = None,
    origin: object = None,
    operation_map: dict[object, object] | None = None,
) -> dict[str, object]:
    """Serialized standard-setting evidence for a declared space group."""
    return {
        "schema_version": "1.1.0",
        "parent_to_standard_direct_transform": (
            np.eye(3).tolist()
            if transform is None
            else np.asarray(transform, dtype=float).tolist()
        ),
        "origin_shift_fractional": (
            [0.0, 0.0, 0.0]
            if origin is None
            else [float(value) for value in np.asarray(origin, dtype=float)]
        ),
        "parent_to_standard_operation_map": (
            {}
            if operation_map is None
            else {
                str(key): int(value) for key, value in operation_map.items()
            }
        ),
        "hall_number": standard_hall_number(space_group_number),
        "normalized_centering_vectors": centering_cosets(space_group_number),
    }
