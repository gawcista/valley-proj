"""Generic magnetic space-group identities derived from spglib."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def derive_type_ii_bns_number(unitary_spacegroup_number: int) -> str:
    """Return the unique type-II BNS number for a unitary space group."""
    if (
        not isinstance(unitary_spacegroup_number, int)
        or isinstance(unitary_spacegroup_number, bool)
        or not 1 <= unitary_spacegroup_number <= 230
    ):
        raise ValueError(
            "unitary_spacegroup_number must be an integer in [1, 230]"
        )

    import spglib

    matches: list[str] = []
    for uni_number in range(1, 1652):
        row = spglib.get_magnetic_spacegroup_type(uni_number)
        if row is None:
            continue
        if (
            int(row.number) == unitary_spacegroup_number
            and int(row.type) == 2
        ):
            matches.append(str(row.bns_number))
    if len(matches) != 1:
        raise RuntimeError(
            "type-II magnetic space-group number is not unique for "
            f"SG {unitary_spacegroup_number}: {matches}"
        )
    return matches[0]
