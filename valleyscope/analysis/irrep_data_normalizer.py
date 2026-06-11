"""Normalize package-style 3D EBR data into a ValleyScope runtime source payload.

The input shape follows the ``irreptables.ebrs.load_ebr_data`` dictionary used
by the public ``irrep`` package.  This module does not import ``irrep``,
``irreptables``, or private ``irrep2``; it only normalizes an already-loaded
data dictionary into the explicit source payload consumed by
``build_reduced_table_from_runtime_source``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_runtime_source_payload_from_ebr_data(
    *,
    ebr_data: Mapping[str, object],
    source_hsp_by_irrep: Mapping[str, str],
    valleyscope_key_by_source_irrep: Mapping[str, str],
    source: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Convert package-style 3D EBR data to a runtime source payload.

    Every mapping from a source irrep label to a sampled moire HSP and to a
    ValleyScope valley-preserving irrep key must be supplied explicitly.  This
    prevents hidden assumptions such as deriving ``GammaM`` from ``GM`` or
    inferring spinful valley phases from source irrep labels.
    """
    basis_labels = _basis_irrep_labels(ebr_data)
    hsp_map = _string_mapping(source_hsp_by_irrep, field="source_hsp_by_irrep")
    key_map = _string_mapping(
        valleyscope_key_by_source_irrep,
        field="valleyscope_key_by_source_irrep",
    )

    basis_entries: list[dict[str, str]] = []
    for label in basis_labels:
        if label not in hsp_map:
            raise ValueError(f"missing source_hsp_by_irrep entry for {label!r}")
        if label not in key_map:
            raise ValueError(
                f"missing valleyscope_key_by_source_irrep entry for {label!r}"
            )
        basis_entries.append({
            "source_label": label,
            "hsp": hsp_map[label],
            "valleyscope_irrep_key": key_map[label],
        })

    ebr_entries = _ebr_entries(ebr_data, source_basis_length=len(basis_labels))
    payload: dict[str, Any] = {"basis": basis_entries, "ebrs": ebr_entries}
    if source is not None:
        payload["source"] = dict(source)
    elif isinstance(ebr_data.get("source"), Mapping):
        payload["source"] = dict(ebr_data["source"])  # type: ignore[index]
    return payload


def normalize_irrep_ebr_data_to_source_payload(
    ebr_data: Mapping[str, object],
    *,
    hsp_name_map: Mapping[str, str] | None = None,
    irrep_key_map: Mapping[str, str] | None = None,
    source: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the initial normalizer name.

    New code should call ``build_runtime_source_payload_from_ebr_data`` and use
    explicit full-source-label maps.  This wrapper keeps the old import working
    while still requiring explicit maps.
    """
    if hsp_name_map is None:
        raise ValueError("hsp_name_map is required; implicit HSP mapping is forbidden")
    if irrep_key_map is None:
        raise ValueError("irrep_key_map is required; implicit irrep-key mapping is forbidden")
    return build_runtime_source_payload_from_ebr_data(
        ebr_data=ebr_data,
        source_hsp_by_irrep=hsp_name_map,
        valleyscope_key_by_source_irrep=irrep_key_map,
        source=source,
    )


def _basis_irrep_labels(ebr_data: Mapping[str, object]) -> list[str]:
    if not isinstance(ebr_data, Mapping):
        raise ValueError("ebr_data must be a mapping")
    basis = ebr_data.get("basis")
    if not isinstance(basis, Mapping):
        raise ValueError("ebr_data['basis'] must be a mapping")

    labels_raw = basis.get("irrep_labels")
    if not isinstance(labels_raw, Sequence) or isinstance(labels_raw, (str, bytes)):
        raise ValueError("basis.irrep_labels must be a non-empty list")
    labels: list[str] = []
    seen: set[str] = set()
    for i, label in enumerate(labels_raw):
        if not isinstance(label, str) or not label:
            raise ValueError(f"basis.irrep_labels[{i}] must be a non-empty string")
        if label in seen:
            raise ValueError(f"duplicate source irrep label {label!r}")
        seen.add(label)
        labels.append(label)
    if not labels:
        raise ValueError("basis.irrep_labels must be a non-empty list")

    degeneracies = basis.get("degeneracies")
    if degeneracies is not None:
        if (
            not isinstance(degeneracies, Sequence)
            or isinstance(degeneracies, (str, bytes))
        ):
            raise ValueError("basis.degeneracies must be a list when present")
        if len(degeneracies) != len(labels):
            raise ValueError(
                f"degeneracies length {len(degeneracies)} != "
                f"irrep_labels length {len(labels)}"
            )
    return labels


def _string_mapping(mapping: Mapping[str, str], *, field: str) -> dict[str, str]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{field} must be a mapping")
    out: dict[str, str] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{field} keys must be non-empty strings")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field}[{key!r}] must be a non-empty string")
        out[key] = value
    return out


def _ebr_entries(
    ebr_data: Mapping[str, object],
    *,
    source_basis_length: int,
) -> list[dict[str, Any]]:
    ebrs_raw = ebr_data.get("ebrs")
    if not isinstance(ebrs_raw, Sequence) or isinstance(ebrs_raw, (str, bytes)):
        raise ValueError("ebr_data['ebrs'] must be a non-empty list")
    if not ebrs_raw:
        raise ValueError("ebr_data['ebrs'] must be a non-empty list")

    entries: list[dict[str, Any]] = []
    labels: set[str] = set()
    for i, ebr in enumerate(ebrs_raw):
        if not isinstance(ebr, Mapping):
            raise ValueError(f"ebrs[{i}] must be a mapping")
        label = ebr.get("ebr_name")
        if not isinstance(label, str) or not label:
            raise ValueError(f"ebrs[{i}] must define a non-empty ebr_name")
        if label in labels:
            raise ValueError(f"duplicate EBR label {label!r}")
        labels.add(label)

        vector_raw = ebr.get("vector")
        if not isinstance(vector_raw, Sequence) or isinstance(vector_raw, (str, bytes)):
            raise ValueError(f"EBR {label!r} vector must be a list")
        if len(vector_raw) != source_basis_length:
            raise ValueError(
                f"EBR {label!r} vector length {len(vector_raw)} != "
                f"source basis length {source_basis_length}"
            )
        entry: dict[str, Any] = {
            "label": label,
            "vector": [_exact_int(v, label=label, index=j) for j, v in enumerate(vector_raw)],
        }
        if "wyckoff_position" in ebr:
            entry["wyckoff_position"] = ebr["wyckoff_position"]
        if "site_symmetry" in ebr:
            entry["site_symmetry"] = ebr["site_symmetry"]
        entries.append(entry)
    return entries


def _exact_int(value: object, *, label: str, index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"EBR {label!r} vector[{index}] must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"EBR {label!r} vector[{index}] is not an integer")
