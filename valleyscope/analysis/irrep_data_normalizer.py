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
    valleyscope_key_by_source_irrep: Mapping[str, str] | None = None,
    valleyscope_irrep_multiplicity_by_source_irrep: (
        Mapping[str, Mapping[str, int]] | None
    ) = None,
    expected_hsps: Sequence[str] | None = None,
    source: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Convert package-style 3D EBR data to a runtime source payload.

    Every mapping from a source irrep label to a sampled moire HSP and to a
    ValleyScope valley-preserving irrep key must be supplied explicitly.  This
    prevents hidden assumptions such as deriving ``GammaM`` from ``GM`` or
    inferring spinful valley phases from source irrep labels.

    Two mapping styles are supported (mutually exclusive):

    - **legacy one-to-one**: ``valleyscope_key_by_source_irrep`` is a
      ``dict[str, str]`` mapping each source label to a single ValleyScope irrep
      key.  Internally converted to multiplicity 1.
    - **multiplicity-aware**: ``valleyscope_irrep_multiplicity_by_source_irrep``
      is a ``dict[str, dict[str, int]]`` mapping each source label to one or more
      ValleyScope irrep keys with positive integer multiplicities.  Supports
      many-to-one aggregation and one-to-many decomposition.

    When ``expected_hsps`` is provided with multiplicity-aware maps, source
    labels outside those sampled HSPs may omit multiplicity entries because they
    are filtered out before the valley-preserving reduced basis is formed.
    """
    basis_labels = _basis_irrep_labels(ebr_data)
    hsp_map = _string_mapping(source_hsp_by_irrep, field="source_hsp_by_irrep")
    expected_hsp_set = _optional_string_set(expected_hsps, field="expected_hsps")

    legacy_key_map = valleyscope_key_by_source_irrep
    mult_map = valleyscope_irrep_multiplicity_by_source_irrep

    _has_legacy = legacy_key_map is not None
    _has_mult = mult_map is not None

    if _has_legacy and _has_mult:
        raise ValueError(
            "provide only one of valleyscope_key_by_source_irrep or "
            "valleyscope_irrep_multiplicity_by_source_irrep"
        )
    if not _has_legacy and not _has_mult:
        raise ValueError(
            "either valleyscope_key_by_source_irrep or "
            "valleyscope_irrep_multiplicity_by_source_irrep is required"
        )

    # Build a source-label -> source-index lookup.
    label_to_index: dict[str, int] = {
        label: idx for idx, label in enumerate(basis_labels)
    }

    basis_entries: list[dict[str, object]] = []
    if _has_mult:
        basis_entries = _basis_from_multiplicities(
            basis_labels=basis_labels,
            hsp_map=hsp_map,
            mult_map=mult_map,
            label_to_index=label_to_index,
            expected_hsp_set=expected_hsp_set,
        )
    else:
        basis_entries = _basis_from_legacy_key_map(
            basis_labels=basis_labels,
            hsp_map=hsp_map,
            key_map=_string_mapping(
                legacy_key_map, field="valleyscope_key_by_source_irrep",
            ),
            label_to_index=label_to_index,
        )

    ebr_entries = _ebr_entries(ebr_data, source_basis_length=len(basis_labels))
    payload: dict[str, Any] = {
        "basis": basis_entries,
        "ebrs": ebr_entries,
        "source_basis_count": len(basis_labels),
    }
    if source is not None:
        payload["source"] = dict(source)
    elif isinstance(ebr_data.get("source"), Mapping):
        payload["source"] = dict(ebr_data["source"])
    return payload


def _basis_from_legacy_key_map(
    *,
    basis_labels: list[str],
    hsp_map: dict[str, str],
    key_map: dict[str, str],
    label_to_index: dict[str, int],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for label in basis_labels:
        if label not in hsp_map:
            raise ValueError(f"missing source_hsp_by_irrep entry for {label!r}")
        if label not in key_map:
            raise ValueError(
                f"missing valleyscope_key_by_source_irrep entry for {label!r}"
            )
        entries.append({
            "source_label": label,
            "source_index": label_to_index[label],
            "hsp": hsp_map[label],
            "valleyscope_irrep_key": key_map[label],
            "multiplicity": 1,
        })
    return entries


def _basis_from_multiplicities(
    *,
    basis_labels: list[str],
    hsp_map: dict[str, str],
    mult_map: Mapping[str, Mapping[str, int]],
    label_to_index: dict[str, int],
    expected_hsp_set: set[str] | None,
) -> list[dict[str, object]]:
    _validate_mult_map(mult_map)
    entries: list[dict[str, object]] = []
    for label in basis_labels:
        if label not in hsp_map:
            raise ValueError(f"missing source_hsp_by_irrep entry for {label!r}")
        hsp = hsp_map[label]
        if label not in mult_map:
            if expected_hsp_set is not None and hsp not in expected_hsp_set:
                continue
            raise ValueError(
                f"missing valleyscope_irrep_multiplicity_by_source_irrep entry "
                f"for {label!r}"
            )
        key_mult_map = mult_map[label]
        if not key_mult_map:
            raise ValueError(
                f"valleyscope_irrep_multiplicity_by_source_irrep[{label!r}] "
                f"must be a non-empty dict"
            )
        for key, mult in sorted(key_mult_map.items()):
            entries.append({
                "source_label": label,
                "source_index": label_to_index[label],
                "hsp": hsp,
                "valleyscope_irrep_key": key,
                "multiplicity": mult,
            })
    return entries


def _validate_mult_map(mult_map: object) -> None:
    if not isinstance(mult_map, Mapping):
        raise ValueError(
            "valleyscope_irrep_multiplicity_by_source_irrep must be a mapping"
        )
    for key, submap in mult_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                "valleyscope_irrep_multiplicity_by_source_irrep keys must be "
                "non-empty strings"
            )
        if not isinstance(submap, Mapping):
            raise ValueError(
                f"valleyscope_irrep_multiplicity_by_source_irrep[{key!r}] "
                f"must be a mapping (dict[str, int])"
            )
        for irrep_key, mult_val in submap.items():
            if not isinstance(irrep_key, str) or not irrep_key:
                raise ValueError(
                    f"valleyscope_irrep_multiplicity_by_source_irrep[{key!r}] "
                    f"keys must be non-empty strings"
                )
            if not isinstance(mult_val, int) or isinstance(mult_val, bool):
                raise ValueError(
                    f"valleyscope_irrep_multiplicity_by_source_irrep[{key!r}]"
                    f"[{irrep_key!r}] must be an integer, got {mult_val!r}"
                )
            if mult_val <= 0:
                raise ValueError(
                    f"valleyscope_irrep_multiplicity_by_source_irrep[{key!r}]"
                    f"[{irrep_key!r}] must be a positive integer"
                )


def _optional_string_set(
    values: Sequence[str] | None,
    *,
    field: str,
) -> set[str] | None:
    if values is None:
        return None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of non-empty strings")
    out: set[str] = set()
    for i, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field}[{i}] must be a non-empty string")
        out.add(value)
    return out


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

    raw_label_counts: dict[str, int] = {}
    for i, ebr in enumerate(ebrs_raw):
        if not isinstance(ebr, Mapping):
            raise ValueError(f"ebrs[{i}] must be a mapping")
        label = ebr.get("ebr_name")
        if not isinstance(label, str) or not label:
            raise ValueError(f"ebrs[{i}] must define a non-empty ebr_name")
        raw_label_counts[label] = raw_label_counts.get(label, 0) + 1

    entries: list[dict[str, Any]] = []
    labels: set[str] = set()
    for i, ebr in enumerate(ebrs_raw):
        label = ebr.get("ebr_name")
        assert isinstance(label, str)
        wp = ebr.get("wyckoff_position", "")
        if raw_label_counts[label] > 1:
            if not isinstance(wp, str) or not wp:
                raise ValueError(
                    f"duplicate EBR label {label!r} requires wyckoff_position"
                )
            qualified = f"{label} @ {wp}"
        else:
            qualified = label
        if qualified in labels:
            raise ValueError(f"duplicate EBR label {qualified!r}")
        labels.add(qualified)

        vector_raw = ebr.get("vector")
        if not isinstance(vector_raw, Sequence) or isinstance(vector_raw, (str, bytes)):
            raise ValueError(f"EBR {label!r} vector must be a list")
        if len(vector_raw) != source_basis_length:
            raise ValueError(
                f"EBR {label!r} vector length {len(vector_raw)} != "
                f"source basis length {source_basis_length}"
            )
        entry: dict[str, Any] = {
            "label": qualified,
            "vector": [_exact_int(v, label=qualified, index=j) for j, v in enumerate(vector_raw)],
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
