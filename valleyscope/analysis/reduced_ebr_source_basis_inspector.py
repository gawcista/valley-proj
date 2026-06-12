"""Offline source-basis inspector for ``irreptables`` EBR data.

Helps a human author inspect the public source basis for a given
projected-subspace / moire space group before writing a canonical reduced
EBR mapping spec.  This is an authoring aid, not a decomposition tool.

No HSP inference, no ValleyScope irrep-key inference, no ``analyze-hsp``
wiring, no raw 3D decomposition, no built-in tables.
"""

from __future__ import annotations

from typing import Any


def inspect_source_basis(
    space_group_number: int,
    *,
    spinful: bool = True,
) -> dict[str, Any]:
    """Return the source basis labels and EBR names for a space group.

    Parameters
    ----------
    space_group_number : int
        Projected-subspace / moire space group number (e.g. 150 for P321).
    spinful : bool
        Whether to load double-valued (spinor) data.

    Returns
    -------
    dict
        With keys ``space_group_number``, ``spinful``, ``irrep_labels``
        (the complete list of source basis labels), ``degeneracies``,
        ``ebr_count`` (number of EBR definitions in the source),
        ``ebr_names`` (first 20 EBR labels for preview), and ``source``
        (package provenance).

    Raises
    ------
    RuntimeError
        If the ``irreptables`` package cannot be loaded or the data is
        unreadable.
    """
    try:
        from irreptables.ebrs import load_ebr_data
        ebr_data = load_ebr_data(int(space_group_number), bool(spinful))
    except ImportError as exc:
        raise RuntimeError(
            "irreptables package is not available; cannot inspect source basis"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"failed to load irreptables EBR data for "
            f"space_group_number={space_group_number}, spinful={spinful}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    basis = ebr_data.get("basis", {})
    irrep_labels = list(basis.get("irrep_labels", []))
    degeneracies = list(basis.get("degeneracies", []))
    ebrs = list(ebr_data.get("ebrs", []))

    return {
        "space_group_number": int(space_group_number),
        "spinful": bool(spinful),
        "irrep_labels": irrep_labels,
        "degeneracies": degeneracies,
        "irrep_count": len(irrep_labels),
        "ebr_count": len(ebrs),
        "ebr_names": [
            e.get("ebr_name", "?") for e in ebrs[:20] if isinstance(e, dict)
        ],
        "source": {"package": "irreptables"},
    }
