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
    subspace_space_group: Mapping[str, Any] | None = None,
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
        The physical valley-projected subspace-space-group symbol
        (e.g. ``"P3"``, ``"P4"``, ``"P2"``) — required scalar key for the
        reduced external-table interface.
    provenance : Mapping or None
        Optional provenance dict (package name, version, detected space group,
        etc.) attached to the output.
    subspace_space_group : Mapping or None
        Optional structured subgroup provenance (e.g. with
        ``candidate_space_group_symbol``, ``valley_preserving_operation_ids``).
        When provided, it is stored alongside the scalar key so downstream
        reduced-EBR consumers can access the canonical physical object.

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

    n_source = _source_basis_count(
        source_payload=source_payload,
        basis=basis,
        ebrs=ebrs,
    )
    _validate_ebr_vector_lengths(ebrs=ebrs, source_basis_count=n_source)

    # Collect multiplicity-weighted source-index contributions for each
    # allowed ValleyScope key.  The output basis order is controlled by
    # allowed_irrep_keys, not by the external source package's basis order.
    # Many source entries can contribute to the same key (many-to-one
    # aggregation) and one source label can contribute to multiple keys
    # (one-to-many decomposition).
    key_to_contributions: dict[str, list[dict[str, int]]] = {}
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
            continue
        if key not in allowed_set:
            continue
        # source_index: use explicit field if present, else fall back to
        # the basis entry index (backward compat with legacy one-to-one
        # normalizer output).
        source_index_raw = entry.get("source_index", i)
        if not isinstance(source_index_raw, int) or isinstance(source_index_raw, bool):
            raise ValueError(
                f"basis[{i}] 'source_index' must be an integer, "
                f"got {source_index_raw!r}"
            )
        mult_raw = entry.get("multiplicity", 1)
        if not isinstance(mult_raw, int) or isinstance(mult_raw, bool):
            raise ValueError(f"basis[{i}] multiplicity must be an integer")
        if mult_raw <= 0:
            raise ValueError(f"basis[{i}] multiplicity must be positive")
        if source_index_raw < 0 or source_index_raw >= n_source:
            raise ValueError(
                f"basis[{i}] source_index {source_index_raw} out of range "
                f"[0, {n_source})"
            )
        key_to_contributions.setdefault(key, []).append({
            "source_index": source_index_raw,
            "multiplicity": mult_raw,
        })

    if not key_to_contributions:
        raise ValueError(
            "no source basis entries match expected_hsps and allowed_irrep_keys"
        )
    missing_keys = [
        key for key in allowed_irrep_key_list if key not in key_to_contributions
    ]
    if missing_keys:
        raise ValueError(
            "missing source basis mapping for allowed_irrep_keys: "
            f"{missing_keys}"
        )

    # Build per-key reduction: for each allowed key, collect all
    # (source_index, multiplicity) contributions.
    reduced_indices: list[list[dict[str, int]]] = [
        list(key_to_contributions[key]) for key in allowed_irrep_key_list
    ]
    reduced_irreps = list(allowed_irrep_key_list)

    # --- Validate and reduce EBR vectors ---
    reduced_ebrs: list[dict[str, Any]] = []
    skipped_zero_vector_ebrs: list[str] = []
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

        # Reduce vector to sampled HSPs with multiplicity-weighted aggregation.
        reduced_vector: list[int] = []
        for contributions in reduced_indices:
            val: int = 0
            for contrib in contributions:
                si = contrib["source_index"]
                mult = contrib["multiplicity"]
                val += vector[si] * mult
            reduced_vector.append(val)
        if not any(v > 0 for v in reduced_vector):
            skipped_zero_vector_ebrs.append(label)
            continue

        entry: dict[str, Any] = {"label": label, "vector": reduced_vector}
        # Preserve optional fields if present.
        for opt in ("wyckoff_position", "site_symmetry"):
            if opt in ebr:
                entry[opt] = ebr[opt]
        reduced_ebrs.append(entry)

    if not reduced_ebrs:
        raise ValueError("no nonzero EBR vectors remain after reduction")

    # --- Build output ---
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "subspace_group_candidate": subspace_group_candidate,
        "expected_hsps": expected_hsps_list,
        "irreps": reduced_irreps,
        "ebrs": reduced_ebrs,
    }
    if subspace_space_group is not None and isinstance(subspace_space_group, Mapping):
        result["subspace_space_group"] = dict(subspace_space_group)
    result["provenance"] = _build_provenance(
        source_payload=source_payload,
        explicit_provenance=provenance,
        expected_hsps=expected_hsps_list,
        source_basis_count=n_source,
        reduction_basis_count=len(reduced_irreps),
        filtered_zero_vector_ebrs=skipped_zero_vector_ebrs,
    )

    return result


def _build_provenance(
    *,
    source_payload: Mapping[str, Any],
    explicit_provenance: Mapping[str, Any] | None,
    expected_hsps: list[str],
    source_basis_count: int,
    reduction_basis_count: int,
    filtered_zero_vector_ebrs: list[str],
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
        "filtered_zero_vector_ebr_count": len(filtered_zero_vector_ebrs),
    })
    if filtered_zero_vector_ebrs:
        provenance["filtered_zero_vector_ebrs"] = list(filtered_zero_vector_ebrs)
    return provenance


def _source_basis_count(
    *,
    source_payload: Mapping[str, Any],
    basis: list[Any],
    ebrs: list[Any],
) -> int:
    raw_count = source_payload.get("source_basis_count")
    if raw_count is not None:
        if (
            not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
            or raw_count <= 0
        ):
            raise ValueError(
                "source_payload['source_basis_count'] must be a positive integer"
            )
        return raw_count

    if not any(
        isinstance(entry, Mapping) and "source_index" in entry
        for entry in basis
    ):
        return len(basis)

    first = ebrs[0]
    if not isinstance(first, Mapping):
        raise ValueError("ebrs[0] must be a dict")
    vector = first.get("vector", [])
    if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
        raise ValueError("EBR vector must be a sequence")
    if len(vector) == 0:
        raise ValueError(
            "first EBR vector is empty — cannot determine source basis size"
        )
    return len(vector)


def _validate_ebr_vector_lengths(
    *,
    ebrs: list[Any],
    source_basis_count: int,
) -> None:
    for j, ebr in enumerate(ebrs):
        if not isinstance(ebr, Mapping):
            raise ValueError(f"ebrs[{j}] must be a dict")
        label = ebr.get("label")
        display_label = label if isinstance(label, str) and label else f"#{j}"
        vector = ebr.get("vector", [])
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise ValueError(f"EBR '{display_label}' vector must be a sequence")
        if len(vector) != source_basis_count:
            raise ValueError(
                f"EBR '{display_label}' vector length {len(vector)} != "
                f"source basis length {source_basis_count}"
            )
