"""irrep package-style 3D ebr_data -> ValleyScope runtime source payload normalizer.

Converts the native ``irrep`` package EBR data dict (``basis`` with
``irrep_labels`` + ``degeneracies``, ``ebrs`` with ``ebr_name`` +
``wyckoff_position`` + ``vector``) into the runtime source payload
consumed by ``build_reduced_table_from_runtime_source``.

This module imports the ``irrep`` Python package as a public dependency
only if it is available; availability is probed at import time for
diagnostic use.
"""

from __future__ import annotations

from typing import Any


def normalize_irrep_ebr_data_to_source_payload(
    ebr_data: dict[str, Any],
    *,
    hsp_name_map: dict[str, str] | None = None,
    irrep_key_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert irrep package-style ebr_data into a runtime source payload.

    Parameters
    ----------
    ebr_data : dict
        Native ``irrep`` EBR data with ``basis`` (``irrep_labels``,
        ``degeneracies``) and ``ebrs`` (list of ``ebr_name``,
        ``wyckoff_position``, ``vector``).
    hsp_name_map : dict or None
        Optional mapping from irrep HSP label convention to ValleyScope
        HSP labels (e.g. ``{"GM": "GammaM", "K": "KM"}``).  If None,
        labels are used as-is.
    irrep_key_map : dict or None
        Optional mapping from source irrep labels to ValleyScope
        valley-preserving irrep keys.  If None, keys are built as
        ``<hsp>:<irrep_label>``.

    Returns
    -------
    dict
        A source payload dict with ``basis`` and ``ebrs`` keys suitable
        for ``build_reduced_table_from_runtime_source()``.

    Raises
    ------
    ValueError
        If ebr_data is missing required fields or has malformed entries.
    """
    if not isinstance(ebr_data, dict):
        raise ValueError("ebr_data must be a dict")

    # --- basis ---
    basis_raw = ebr_data.get("basis")
    if not isinstance(basis_raw, dict):
        raise ValueError("ebr_data['basis'] must be a dict")
    irrep_labels = list(basis_raw.get("irrep_labels", []))
    degeneracies = list(basis_raw.get("degeneracies", []))
    if not irrep_labels:
        raise ValueError("ebr_data['basis']['irrep_labels'] must be a non-empty list")
    if len(degeneracies) != len(irrep_labels):
        raise ValueError(
            f"degeneracies length {len(degeneracies)} != "
            f"irrep_labels length {len(irrep_labels)}"
        )

    # --- ebrs ---
    ebrs_raw = list(ebr_data.get("ebrs", []))
    if not ebrs_raw:
        raise ValueError("ebr_data['ebrs'] must be a non-empty list")

    hsp_map = dict(hsp_name_map) if hsp_name_map else {}
    key_map = dict(irrep_key_map) if irrep_key_map else {}

    # Build source basis entries.
    basis_entries: list[dict[str, Any]] = []
    for i, label in enumerate(irrep_labels):
        if not isinstance(label, str) or not label:
            raise ValueError(f"irrep_labels[{i}] must be a non-empty string")
        hsp, irrep = _split_hsp_irrep(label)
        hsp_valley = hsp_map.get(hsp, hsp)
        if key_map:
            valleyscope_key = key_map.get(label)
            if valleyscope_key is None:
                raise ValueError(
                    f"no irrep_key_map entry for source label {label!r}"
                )
        else:
            valleyscope_key = f"{hsp_valley}:{irrep}"
        basis_entries.append({
            "source_label": label,
            "hsp": hsp_valley,
            "valleyscope_irrep_key": valleyscope_key,
        })

    # Build source EBR entries.
    ebr_entries: list[dict[str, Any]] = []
    for j, ebr in enumerate(ebrs_raw):
        if not isinstance(ebr, dict):
            raise ValueError(f"ebrs[{j}] must be a dict")
        ebr_label = ebr.get("ebr_name")
        if not isinstance(ebr_label, str) or not ebr_label:
            raise ValueError(f"ebrs[{j}] must have non-empty 'ebr_name'")
        vector = list(ebr.get("vector", []))
        entry: dict[str, Any] = {
            "label": ebr_label,
            "vector": vector,
        }
        if "wyckoff_position" in ebr:
            entry["wyckoff_position"] = ebr["wyckoff_position"]
        ebr_entries.append(entry)

    return {"basis": basis_entries, "ebrs": ebr_entries}


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _split_hsp_irrep(label: str) -> tuple[str, str]:
    """Split 'GM:K5' -> ('GM', 'K5')."""
    if ":" not in label:
        raise ValueError(
            f"irrep label {label!r} must contain ':' separating HSP and irrep"
        )
    hsp, irrep = label.split(":", 1)
    if not hsp or not irrep:
        raise ValueError(
            f"irrep label {label!r}: HSP and irrep parts must be non-empty"
        )
    return hsp, irrep
