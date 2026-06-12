"""Build a ValleyScope reduced table from public ``irreptables`` EBR data.

This is an offline/library adapter.  It uses the projected-subspace / moire
space group supplied by ValleyScope, loads public package-style 3D EBR data,
then applies ValleyScope's explicit sampled-HSP and valley-preserving reduction.
It is not wired into ``analyze_hsp.py`` and never calls raw 3D decomposition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib.metadata
import json
from pathlib import Path
import tempfile
from typing import Any

from valleyscope.analysis.irrep_data_normalizer import (
    build_runtime_source_payload_from_ebr_data,
)
from valleyscope.analysis.irrep_runtime_reducer import (
    build_reduced_table_from_runtime_source,
)


def build_reduced_table_from_irreptables(
    *,
    space_group_number: int | str | None = None,
    spinful: bool | None = None,
    source_hsp_by_irrep: Mapping[str, str],
    valleyscope_key_by_source_irrep: Mapping[str, str],
    expected_hsps: Sequence[str],
    allowed_irrep_keys: Sequence[str],
    subspace_group_candidate: str,
    source_loader: Callable[[int | str, bool], Mapping[str, object]] | None = None,
    provenance: Mapping[str, object] | None = None,
    sg_number: int | str | None = None,
    spinor: bool | None = None,
    provenance_extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a reduced external table from public ``irreptables`` EBR data.

    ``sg_number`` / ``spinor`` and ``provenance_extra`` are accepted as legacy
    aliases for the first implementation pass.  New callers should use
    ``space_group_number``, ``spinful``, and ``provenance``.
    """
    resolved_sg = _resolve_space_group_number(space_group_number, sg_number)
    resolved_spinful = _resolve_spinful(spinful, spinor)
    loader = source_loader or _load_ebr_data_from_irreptables

    try:
        ebr_data = loader(resolved_sg, resolved_spinful)
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            "failed to load public irreptables EBR data for "
            f"space_group_number={resolved_sg!r}, spinful={resolved_spinful}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    source_provenance = _builder_provenance(
        space_group_number=resolved_sg,
        spinful=resolved_spinful,
        expected_hsps=expected_hsps,
        subspace_group_candidate=subspace_group_candidate,
        provenance=provenance,
        provenance_extra=provenance_extra,
    )
    source_payload = build_runtime_source_payload_from_ebr_data(
        ebr_data=ebr_data,
        source_hsp_by_irrep=source_hsp_by_irrep,
        valleyscope_key_by_source_irrep=valleyscope_key_by_source_irrep,
        source=source_provenance,
    )

    return build_reduced_table_from_runtime_source(
        source_payload=source_payload,
        expected_hsps=expected_hsps,
        allowed_irrep_keys=allowed_irrep_keys,
        subspace_group_candidate=subspace_group_candidate,
        provenance=source_provenance,
    )


def _load_ebr_data_from_irreptables(
    space_group_number: int | str,
    spinful: bool,
) -> Mapping[str, object]:
    try:
        from irreptables.ebrs import load_ebr_data
    except Exception as exc:
        raise RuntimeError(
            "cannot import public irreptables.ebrs.load_ebr_data: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return load_ebr_data(space_group_number, spinful)


def _resolve_space_group_number(
    space_group_number: int | str | None,
    sg_number: int | str | None,
) -> int | str:
    if space_group_number is not None and sg_number is not None:
        raise ValueError("provide only one of space_group_number or sg_number")
    resolved = space_group_number if space_group_number is not None else sg_number
    if resolved is None:
        raise ValueError("space_group_number is required")
    if (
        not isinstance(resolved, (int, str))
        or isinstance(resolved, bool)
        or resolved == ""
    ):
        raise ValueError("space_group_number must be a non-empty int or string")
    return resolved


def _resolve_spinful(spinful: bool | None, spinor: bool | None) -> bool:
    if spinful is not None and spinor is not None:
        raise ValueError("provide only one of spinful or spinor")
    resolved = spinful if spinful is not None else spinor
    if not isinstance(resolved, bool):
        raise ValueError("spinful is required and must be bool")
    return resolved


def _builder_provenance(
    *,
    space_group_number: int | str,
    spinful: bool,
    expected_hsps: Sequence[str],
    subspace_group_candidate: str,
    provenance: Mapping[str, object] | None,
    provenance_extra: Mapping[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "data_source": "irreptables",
        "package": "irreptables",
        "package_version": _package_version("irreptables"),
        "space_group_number": space_group_number,
        "spinful": spinful,
        "expected_hsps": list(expected_hsps),
        "subspace_group_candidate": subspace_group_candidate,
        "valleyscope_reduction": "sampled_hsp_valley_preserving",
    }
    if provenance_extra:
        payload.update(dict(provenance_extra))
    if provenance:
        payload.update(dict(provenance))
    return payload


_SPEC_SCHEMA_VERSION = "1.0.0"
_SPEC_DATA_SOURCE = "irreptables"


def build_reduced_table_from_spec_file(
    spec_path: str | Path,
    *,
    source_loader: Callable[[int | str, bool], Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Build a ValleyScope reduced table from a JSON mapping spec file.

    The spec must contain:
    - ``schema_version`` (``"1.0.0"``)
    - ``data_source`` (``"irreptables"``)
    - ``space_group_number`` (int or non-empty string)
    - ``spinful`` (bool)
    - ``source_hsp_by_irrep`` (dict)
    - ``valleyscope_key_by_source_irrep`` (dict)
    - ``expected_hsps`` (list[str])
    - ``allowed_irrep_keys`` (list[str])
    - ``subspace_group_candidate`` (str)

    The output is validated through ``load_reduced_ebr_table`` before
    being returned.

    Raises ValueError if spec is missing required fields or the output
    fails validation.
    """
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be a JSON object")
    required = [
        "schema_version",
        "data_source",
        "space_group_number",
        "spinful",
        "source_hsp_by_irrep",
        "valleyscope_key_by_source_irrep",
        "expected_hsps",
        "allowed_irrep_keys",
        "subspace_group_candidate",
    ]
    missing = [k for k in required if k not in spec]
    if missing:
        raise ValueError(f"spec missing required keys: {missing}")
    if spec["schema_version"] != _SPEC_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be "
            f"{_SPEC_SCHEMA_VERSION!r}, got {spec['schema_version']!r}"
        )
    if spec["data_source"] != _SPEC_DATA_SOURCE:
        raise ValueError(
            "data_source must be "
            f"{_SPEC_DATA_SOURCE!r}, got {spec['data_source']!r}"
        )

    space_group_number = _resolve_space_group_number(spec["space_group_number"], None)
    spinful = _resolve_spinful(spec["spinful"], None)
    subspace_group_candidate = _required_nonempty_string(
        spec,
        "subspace_group_candidate",
    )
    provenance = _optional_mapping(spec, "provenance")

    table = build_reduced_table_from_irreptables(
        space_group_number=space_group_number,
        spinful=spinful,
        source_loader=source_loader,
        source_hsp_by_irrep=_required_string_mapping(spec, "source_hsp_by_irrep"),
        valleyscope_key_by_source_irrep=_required_string_mapping(
            spec,
            "valleyscope_key_by_source_irrep",
        ),
        expected_hsps=_required_string_sequence(spec, "expected_hsps"),
        allowed_irrep_keys=_required_string_sequence(spec, "allowed_irrep_keys"),
        subspace_group_candidate=subspace_group_candidate,
        provenance=provenance,
    )
    _validate_reduced_table_dict(table)
    return table


def _required_string_mapping(spec: Mapping[str, object], key: str) -> dict[str, str]:
    value = spec[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    result: dict[str, str] = {}
    for raw_k, raw_v in value.items():
        if not isinstance(raw_k, str) or not isinstance(raw_v, str):
            raise ValueError(f"{key} entries must be string -> string")
        result[raw_k] = raw_v
    return result


def _required_string_sequence(spec: Mapping[str, object], key: str) -> list[str]:
    value = spec[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a sequence of strings")
    result = list(value)
    if not result or not all(isinstance(item, str) for item in result):
        raise ValueError(f"{key} must be a non-empty sequence of strings")
    return result


def _required_nonempty_string(spec: Mapping[str, object], key: str) -> str:
    value = spec[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_mapping(spec: Mapping[str, object], key: str) -> dict[str, object] | None:
    if key not in spec or spec[key] is None:
        return None
    value = spec[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def _validate_reduced_table_dict(table: Mapping[str, object]) -> None:
    from valleyscope.analysis.reduced_ebr_mapping import load_reduced_ebr_table

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    )
    try:
        json.dump(table, tmp)
        tmp.close()
        load_reduced_ebr_table(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
