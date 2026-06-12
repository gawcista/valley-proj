"""Offline source-basis inspector for ``irreptables`` EBR data.

Helps a human author inspect the public source basis for a given
projected-subspace / moire space group before writing a canonical reduced
EBR mapping spec.  This is an authoring aid, not a decomposition tool.

Supports ``source_loader`` injection so tests do not require the real
installed ``irreptables`` package or any specific space group data.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


_SCHEMA_VERSION = "1.0.0"
_DATA_SOURCE = "irreptables"


def inspect_irreptables_source_basis(
    space_group_number: int,
    *,
    spinful: bool = True,
    source_loader: Callable[[int, bool], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a canonical source-basis inspection payload.

    Parameters
    ----------
    space_group_number : int
        Projected-subspace / moire space group number (e.g. 150 for P321).
    spinful : bool
        Use double-valued (spinor) data.
    source_loader : callable or None
        If None, imports and calls ``irreptables.ebrs.load_ebr_data``.
        Inject a fake loader for testing.

    Returns
    -------
    dict
        Canonical payload with ``schema_version``, ``data_source``,
        ``space_group_number``, ``spinful``, ``source_basis``,
        ``source_ebr_count``, ``provenance``.

    Raises
    ------
    RuntimeError
        If the source package cannot be loaded or the data is unreadable.
    ValueError
        If basis labels and degeneracies have mismatched lengths.
    """
    loader = source_loader or _default_irreptables_loader

    try:
        ebr_data = loader(int(space_group_number), bool(spinful))
    except Exception as exc:
        raise RuntimeError(
            f"failed to load irreptables EBR data for "
            f"space_group_number={space_group_number}, spinful={spinful}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(ebr_data, Mapping):
        raise RuntimeError("source_loader must return a mapping")

    basis_raw = ebr_data.get("basis", {})
    irrep_labels = list(basis_raw.get("irrep_labels", []))
    degeneracies = list(basis_raw.get("degeneracies", []))

    if len(degeneracies) != len(irrep_labels):
        raise ValueError(
            f"degeneracies length {len(degeneracies)} != "
            f"irrep_labels length {len(irrep_labels)}"
        )

    source_basis: list[dict[str, Any]] = []
    for i, label in enumerate(irrep_labels):
        if not isinstance(label, str) or not label:
            raise ValueError(f"irrep_labels[{i}] must be a non-empty string")
        source_basis.append({
            "source_label": label,
            "degeneracy": int(degeneracies[i]),
        })

    ebrs_raw = list(ebr_data.get("ebrs", []))
    ebr_count = len(ebrs_raw)

    provenance: dict[str, Any] = {
        "package": _DATA_SOURCE,
        "package_version": _resolve_package_version(),
    }

    return {
        "schema_version": _SCHEMA_VERSION,
        "data_source": _DATA_SOURCE,
        "space_group_number": int(space_group_number),
        "spinful": bool(spinful),
        "source_basis": source_basis,
        "source_basis_count": len(source_basis),
        "source_ebr_count": ebr_count,
        "provenance": provenance,
    }


def inspect_irreptables_source_basis_from_args(
    space_group_number: int,
    *,
    spinful: bool = True,
    source_loader: Callable[[int, bool], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for CLI — same as inspect_irreptables_source_basis."""
    return inspect_irreptables_source_basis(
        space_group_number=space_group_number,
        spinful=spinful,
        source_loader=source_loader,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _default_irreptables_loader(sg: int, spinful: bool) -> Mapping[str, Any]:
    from irreptables.ebrs import load_ebr_data
    return load_ebr_data(sg, spinful)


def _resolve_package_version() -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version("irreptables")
    except Exception:
        return None
