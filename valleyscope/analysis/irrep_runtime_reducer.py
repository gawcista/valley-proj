"""Irrep runtime source -> ValleyScope reduced-dimensional external table reducer.

Transforms a full/runtime irrep/EBR source payload into the
ValleyScope external reduced EBR table format consumed by
``load_reduced_ebr_table`` and ``build_reduced_ebr_mapping``.

This is a pure-Python contract layer.  It does not import the external
``irrep`` package or the private ``irrep2`` repository.  Actual package
wiring belongs in a separate adapter module.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_reduced_table_from_runtime_source(
    *,
    source_payload: Mapping[str, Any],
    expected_hsps: Sequence[str],
    allowed_irrep_keys: Sequence[str],
    subspace_group_candidate: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ValleyScope reduced external table from a runtime source payload.

    Parameters
    ----------
    source_payload : Mapping
        Must contain ``basis`` (list of dicts with ``source_label``, ``hsp``,
        ``valleyscope_irrep_key``) and ``ebrs`` (list of dicts with ``label``
        and ``vector`` aligned to ``basis``).
    expected_hsps : Sequence[str]
        Sampled moire HSP labels.  Only basis entries whose ``hsp`` field is
        in this set are included in the reduced output.
    allowed_irrep_keys : Sequence[str]
        ValleyScope trusted valley-preserving irrep keys.  Only basis entries
        whose ``valleyscope_irrep_key`` is in this set are included.
    subspace_group_candidate : str
        The ``C{order}_like`` group label for the reduced table.
    provenance : Mapping or None
        Optional provenance dict (package name, version, detected space group,
        etc.) attached to the output.

    Returns
    -------
    dict
        A dict compatible with ``load_reduced_ebr_table()``.

    Raises
    ------
    ValueError
        If basis mappings are missing, ambiguous, or EBR vectors are
        misaligned.
    """
    allowed_set = set(allowed_irrep_keys)
    hsps_set = set(expected_hsps)
    basis = list(source_payload.get("basis", []))
    ebrs = list(source_payload.get("ebrs", []))

    # --- Validate basis ---
    if not basis:
        raise ValueError("source_payload['basis'] must be a non-empty list")
    if not ebrs:
        raise ValueError("source_payload['ebrs'] must be a non-empty list")

    # Check each basis entry and collect reduced indices.
    reduced_indices: list[int] = []
    seen_keys: set[str] = set()
    reduced_irreps: list[str] = []
    for i, entry in enumerate(basis):
        if not isinstance(entry, dict):
            raise ValueError(f"basis[{i}] must be a dict")
        hsp = str(entry.get("hsp", ""))
        key = str(entry.get("valleyscope_irrep_key", ""))
        if not hsp or not key:
            raise ValueError(
                f"basis[{i}] must define non-empty 'hsp' and 'valleyscope_irrep_key'"
            )
        if hsp not in hsps_set:
            continue  # Not in sampled HSP set — skip.
        if key not in allowed_set:
            continue  # Not an allowed valley-preserving irrep key — skip.
        if key in seen_keys:
            raise ValueError(
                f"duplicate valleyscope_irrep_key {key!r} in reduced basis "
                f"(basis entries {seen_keys} and {i})"
            )
        seen_keys.add(key)
        reduced_indices.append(i)
        reduced_irreps.append(key)

    if not reduced_indices:
        raise ValueError(
            "no source basis entries match expected_hsps and allowed_irrep_keys"
        )

    # --- Validate and reduce EBR vectors ---
    n_source = len(basis)
    reduced_ebrs: list[dict[str, Any]] = []
    ebr_labels: list[str] = []
    for j, ebr in enumerate(ebrs):
        if not isinstance(ebr, dict):
            raise ValueError(f"ebrs[{j}] must be a dict")
        label = ebr.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"ebrs[{j}] must have a non-empty 'label'")
        if label in ebr_labels:
            raise ValueError(f"duplicate EBR label {label!r}")
        ebr_labels.append(label)

        vector = list(ebr.get("vector", []))
        if len(vector) != n_source:
            raise ValueError(
                f"EBR '{label}' vector length {len(vector)} != "
                f"source basis length {n_source}"
            )
        if not all(isinstance(v, int) and v >= 0 for v in vector):
            raise ValueError(
                f"EBR '{label}' vector must be nonnegative integers"
            )

        # Reduce vector to sampled HSPs only.
        reduced_vector = [vector[i] for i in reduced_indices]
        if not any(v > 0 for v in reduced_vector):
            raise ValueError(
                f"EBR '{label}' reduced vector is all-zero "
                f"(no weight in sampled HSPs); consider filtering upstream"
            )

        entry: dict[str, Any] = {"label": label, "vector": reduced_vector}
        # Preserve optional fields if present.
        for opt in ("wyckoff_position", "site_symmetry"):
            if opt in ebr:
                entry[opt] = ebr[opt]
        reduced_ebrs.append(entry)

    if not reduced_ebrs:
        raise ValueError("no EBR vectors remain after reduction")

    # --- Build output ---
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": subspace_group_candidate,
        "expected_hsps": sorted(expected_hsps),
        "irreps": reduced_irreps,
        "ebrs": reduced_ebrs,
    }
    if provenance is not None:
        result["provenance"] = dict(provenance)

    return result
