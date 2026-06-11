"""Irrep runtime source -> ValleyScope reduced-dimensional external table reducer.

Transforms a full/runtime irrep/EBR source payload into the
ValleyScope external reduced EBR table format consumed by
``load_reduced_ebr_table`` and ``build_reduced_ebr_mapping``.

This is a pure-Python contract layer.  It does not import the external
``irrep`` package or the private ``irrep2`` repository.  Actual package
wiring belongs in a separate adapter module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _unique_nonempty_strings(values: Sequence[str], *, field: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of non-empty strings")
    out: list[str] = []
    seen: set[str] = set()
    for i, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field}[{i}] must be a non-empty string")
        if value in seen:
            raise ValueError(f"{field} must contain unique entries")
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError(f"{field} must be non-empty")
    return out


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
    expected_hsps_list = _unique_nonempty_strings(
        expected_hsps, field="expected_hsps"
    )
    allowed_irrep_key_list = _unique_nonempty_strings(
        allowed_irrep_keys, field="allowed_irrep_keys"
    )
    if not isinstance(subspace_group_candidate, str) or not subspace_group_candidate:
        raise ValueError("subspace_group_candidate must be a non-empty string")

    allowed_set = set(allowed_irrep_key_list)
    hsps_set = set(expected_hsps_list)
    basis = list(source_payload.get("basis", []))
    ebrs = list(source_payload.get("ebrs", []))

    # --- Validate basis ---
    if not basis:
        raise ValueError("source_payload['basis'] must be a non-empty list")
    if not ebrs:
        raise ValueError("source_payload['ebrs'] must be a non-empty list")

    # Check each basis entry and collect the source index for each allowed
    # ValleyScope key.  The output basis order is controlled by
    # allowed_irrep_keys, not by the external source package's basis order.
    key_to_source_index: dict[str, int] = {}
    for i, entry in enumerate(basis):
        if not isinstance(entry, dict):
            raise ValueError(f"basis[{i}] must be a dict")
        hsp = entry.get("hsp")
        key = entry.get("valleyscope_irrep_key")
        if not isinstance(hsp, str) or not hsp:
            raise ValueError(f"basis[{i}] must define non-empty string 'hsp'")
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"basis[{i}] must define non-empty string 'valleyscope_irrep_key'"
            )
        if hsp not in hsps_set:
            continue  # Not in sampled HSP set — skip.
        if key not in allowed_set:
            continue  # Not an allowed valley-preserving irrep key — skip.
        if key in key_to_source_index:
            raise ValueError(
                f"duplicate valleyscope_irrep_key {key!r} in reduced basis "
                f"(basis entries {key_to_source_index[key]} and {i})"
            )
        key_to_source_index[key] = i

    if not key_to_source_index:
        raise ValueError(
            "no source basis entries match expected_hsps and allowed_irrep_keys"
        )
    missing_keys = [
        key for key in allowed_irrep_key_list if key not in key_to_source_index
    ]
    if missing_keys:
        raise ValueError(
            "missing source basis mapping for allowed_irrep_keys: "
            f"{missing_keys}"
        )
    reduced_indices = [key_to_source_index[key] for key in allowed_irrep_key_list]
    reduced_irreps = list(allowed_irrep_key_list)

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
        "expected_hsps": expected_hsps_list,
        "irreps": reduced_irreps,
        "ebrs": reduced_ebrs,
    }
    result["provenance"] = _build_provenance(
        source_payload=source_payload,
        explicit_provenance=provenance,
        expected_hsps=expected_hsps_list,
        source_basis_count=n_source,
        reduction_basis_count=len(reduced_irreps),
    )

    return result


def _build_provenance(
    *,
    source_payload: Mapping[str, Any],
    explicit_provenance: Mapping[str, Any] | None,
    expected_hsps: list[str],
    source_basis_count: int,
    reduction_basis_count: int,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    source = source_payload.get("source")
    if isinstance(source, Mapping):
        provenance.update(source)
    if explicit_provenance is not None:
        provenance.update(dict(explicit_provenance))
    provenance.update({
        "expected_hsps": list(expected_hsps),
        "source_basis_count": source_basis_count,
        "reduction_basis_count": reduction_basis_count,
    })
    return provenance
