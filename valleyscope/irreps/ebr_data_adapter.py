"""Irreptables EBR source data adapter skeleton.

Provides a thin, testable wrapper around ``irreptables.ebrs.load_ebr_data``
that exposes normalized data needed for ValleyScope's sampled-HSP
valley-preserving reduced-dimensional EBR reduction.

This adapter does NOT invoke raw 3D EBR decomposition, does NOT require
OR-Tools, and does NOT report raw 3D EBR results as a ValleyScope output.
It provides only source data; the reduction and exact integer solving remain
ValleyScope's responsibility.
"""

from __future__ import annotations

from typing import Any


def load_ebr_source_data(
    space_group_number: int | str,
    spinful: bool,
) -> dict[str, Any]:
    """Load normalized irreptables EBR source data for a space group.

    Returns a dict with these keys:

    - ``source_basis_labels``: list of source irrep labels from
      ``ebr_data["basis"]["irrep_labels"]``.
    - ``source_ebrs``: list of dicts with ``ebr_label``, ``wyckoff_position``,
      ``vector``.
    - ``data_source``: ``"irreptables"``.
    - ``space_group_number``: the input space group number.
    - ``spinful``: whether spinful (double-group) data was requested.
    - ``source_basis_count``: number of source irrep labels.

    Raises ``RuntimeError`` if the ``irreptables`` package is not
    installed or ``load_ebr_data`` fails.  Raises ``ValueError`` for
    malformed source data.
    """
    try:
        from irreptables.ebrs import load_ebr_data
    except ImportError as exc:
        raise RuntimeError(
            "irreptables package is required for EBR source data. "
            f"ImportError: {exc}"
        ) from exc

    try:
        raw = load_ebr_data(space_group_number, spinful)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load irreptables EBR data for "
            f"space_group_number={space_group_number!r}, "
            f"spinful={spinful}: {type(exc).__name__}: {exc}"
        ) from exc

    return _normalize_ebr_data(raw, space_group_number, spinful)


def _normalize_ebr_data(
    raw: Any,
    space_group_number: int | str,
    spinful: bool,
) -> dict[str, Any]:
    """Normalize irreptables EBR data dict into a stable ValleyScope payload."""
    if not isinstance(raw, dict):
        raise ValueError("irreptables EBR data must be a dict")

    basis = raw.get("basis")
    if not isinstance(basis, dict):
        raise ValueError("irreptables EBR data must have a 'basis' dict")

    irrep_labels = basis.get("irrep_labels")
    if not isinstance(irrep_labels, list) or not irrep_labels:
        raise ValueError(
            "irreptables EBR data 'basis.irrep_labels' must be a non-empty list"
        )
    if not all(isinstance(lbl, str) and lbl for lbl in irrep_labels):
        raise ValueError(
            "irreptables EBR data 'basis.irrep_labels' entries must be "
            "non-empty strings"
        )

    ebrs_raw = raw.get("ebrs")
    if not isinstance(ebrs_raw, list) or not ebrs_raw:
        raise ValueError(
            "irreptables EBR data 'ebrs' must be a non-empty list"
        )

    source_ebrs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, entry in enumerate(ebrs_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"ebrs[{i}] must be a dict")
        ebr_label = entry.get("ebr_name")
        if not isinstance(ebr_label, str) or not ebr_label:
            raise ValueError(f"ebrs[{i}] must define a non-empty ebr_name")
        wp = entry.get("wyckoff_position", "")
        qualified = f"{ebr_label} @ {wp}" if wp else ebr_label
        if qualified in seen:
            raise ValueError(f"duplicate EBR label {qualified!r}")
        seen.add(qualified)
        vector = entry.get("vector")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"EBR {ebr_label!r} must have a non-empty vector")
        if len(vector) != len(irrep_labels):
            raise ValueError(
                f"EBR {ebr_label!r} vector length {len(vector)} != "
                f"basis length {len(irrep_labels)}"
            )
        source_ebrs.append({
            "ebr_label": qualified,
            "wyckoff_position": wp if wp else None,
            "vector": list(vector),
        })

    return {
        "source_basis_labels": list(irrep_labels),
        "source_basis_count": len(irrep_labels),
        "source_ebrs": source_ebrs,
        "data_source": "irreptables",
        "space_group_number": space_group_number,
        "spinful": spinful,
    }
