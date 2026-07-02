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
    valleyscope_key_by_source_irrep: Mapping[str, str] | None = None,
    valleyscope_irrep_multiplicity_by_source_irrep: (
        Mapping[str, Mapping[str, int]] | None
    ) = None,
    expected_hsps: Sequence[str],
    allowed_irrep_keys: Sequence[str],
    subspace_group_candidate: str,
    source_loader: Callable[[int | str, bool], Mapping[str, object]] | None = None,
    provenance: Mapping[str, object] | None = None,
    subspace_space_group: Mapping[str, object] | None = None,
    sg_number: int | str | None = None,
    spinor: bool | None = None,
    provenance_extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build a reduced external table from public ``irreptables`` EBR data.

    ``sg_number`` / ``spinor`` and ``provenance_extra`` are accepted as legacy
    aliases for the first implementation pass.  New callers should use
    ``space_group_number``, ``spinful``, and ``provenance``.

    ``valleyscope_irrep_multiplicity_by_source_irrep`` accepts the new
    multiplicity-aware mapping alongside the legacy one-to-one
    ``valleyscope_key_by_source_irrep``.  Only one of the two may be
    provided.
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
        valleyscope_irrep_multiplicity_by_source_irrep=(
            valleyscope_irrep_multiplicity_by_source_irrep
        ),
        expected_hsps=expected_hsps,
        source=source_provenance,
    )

    return build_reduced_table_from_runtime_source(
        source_payload=source_payload,
        expected_hsps=expected_hsps,
        allowed_irrep_keys=allowed_irrep_keys,
        subspace_group_candidate=subspace_group_candidate,
        provenance=source_provenance,
        subspace_space_group=subspace_space_group,
    )


def _load_ebr_data_from_irreptables(
    space_group_number: int | str,
    spinful: bool,
) -> Mapping[str, object]:
    from valleyscope.irreps.ebr_data_adapter import load_raw_ebr_data
    return load_raw_ebr_data(space_group_number, spinful)


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


_SPEC_V1_SCHEMA_VERSION = "1.0.0"
_SPEC_V1_1_SCHEMA_VERSION = "1.1.0"
_SUPPORTED_SCHEMA_VERSIONS = frozenset([
    _SPEC_V1_SCHEMA_VERSION,
    _SPEC_V1_1_SCHEMA_VERSION,
])
_SPEC_DATA_SOURCE = "irreptables"

_V1_REQUIRED = [
    "schema_version", "data_source", "space_group_number", "spinful",
    "source_hsp_by_irrep", "valleyscope_key_by_source_irrep",
    "expected_hsps", "allowed_irrep_keys", "subspace_group_candidate",
]

_V1_1_REQUIRED = [
    "schema_version", "data_source", "space_group_number", "spinful",
    "source_hsp_by_irrep", "valleyscope_irrep_multiplicity_by_source_irrep",
    "expected_hsps", "allowed_irrep_keys", "subspace_group_candidate",
]


def build_reduced_table_from_spec_file(
    spec_path: str | Path,
    *,
    source_loader: Callable[[int | str, bool], Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Build a ValleyScope reduced table from a JSON mapping spec file.

    Supports two schema versions:

    - ``"1.0.0"`` (legacy): uses ``valleyscope_key_by_source_irrep``
      (dict[str, str]) for one-to-one mapping.
    - ``"1.1.0"`` (multiplicity-aware): uses
      ``valleyscope_irrep_multiplicity_by_source_irrep``
      (dict[str, dict[str, int]]) for many-to-one aggregation and
      one-to-many decomposition.

    The output is validated through ``load_reduced_ebr_table`` before
    being returned.
    """
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be a JSON object")

    schema_version = spec.get("schema_version")
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported schema_version "
            f"{schema_version!r}; supported: "
            f"{sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )
    required = (
        _V1_REQUIRED
        if schema_version == _SPEC_V1_SCHEMA_VERSION
        else _V1_1_REQUIRED
    )
    missing = [k for k in required if k not in spec]
    if missing:
        raise ValueError(f"spec missing required keys: {missing}")

    if spec["data_source"] != _SPEC_DATA_SOURCE:
        raise ValueError(
            "data_source must be "
            f"{_SPEC_DATA_SOURCE!r}, got {spec['data_source']!r}"
        )

    if schema_version == _SPEC_V1_SCHEMA_VERSION:
        return _build_from_v1_spec(spec, source_loader=source_loader)
    else:
        return _build_from_v1_1_spec(spec, source_loader=source_loader)


def _build_from_v1_spec(spec: Mapping[str, object], **kwargs: Any) -> dict[str, Any]:
    return _build_common(
        spec=spec,
        valleyscope_key_by_source_irrep=_required_string_mapping(
            spec, "valleyscope_key_by_source_irrep",
        ),
        valleyscope_irrep_multiplicity_by_source_irrep=None,
        **kwargs,
    )


def _build_from_v1_1_spec(spec: Mapping[str, object], **kwargs: Any) -> dict[str, Any]:
    mult_raw = spec["valleyscope_irrep_multiplicity_by_source_irrep"]
    if not isinstance(mult_raw, Mapping):
        raise ValueError(
            "valleyscope_irrep_multiplicity_by_source_irrep must be a mapping"
        )
    mult_map: dict[str, dict[str, int]] = {}
    for k, v in mult_raw.items():
        if not isinstance(k, str) or not isinstance(v, Mapping):
            raise ValueError(
                "valleyscope_irrep_multiplicity_by_source_irrep must be "
                "dict[str, dict[str, int]]"
            )
        sub: dict[str, int] = {}
        for sk, sv in v.items():
            if not isinstance(sv, int) or isinstance(sv, bool):
                raise ValueError(
                    f"multiplicities must be integers, got {sv!r} "
                    f"for {k!r}[{sk!r}]"
                )
            sub[sk] = sv
        mult_map[k] = sub
    return _build_common(
        spec=spec,
        valleyscope_key_by_source_irrep=None,
        valleyscope_irrep_multiplicity_by_source_irrep=mult_map,
        **kwargs,
    )


def _build_common(
    *,
    spec: Mapping[str, object],
    valleyscope_key_by_source_irrep: Mapping[str, str] | None,
    valleyscope_irrep_multiplicity_by_source_irrep: (
        Mapping[str, Mapping[str, int]] | None
    ),
    source_loader: Callable[[int | str, bool], Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    space_group_number = _resolve_space_group_number(spec["space_group_number"], None)
    spinful = _resolve_spinful(spec["spinful"], None)
    subspace_group_candidate = _required_nonempty_string(spec, "subspace_group_candidate")
    provenance = _optional_mapping(spec, "provenance")
    expected_hsps = _required_string_sequence(spec, "expected_hsps")
    allowed_irrep_keys = _required_string_sequence(spec, "allowed_irrep_keys")
    subspace_space_group_raw = spec.get("subspace_space_group")

    table = build_reduced_table_from_irreptables(
        space_group_number=space_group_number,
        spinful=spinful,
        source_loader=source_loader,
        source_hsp_by_irrep=_required_string_mapping(spec, "source_hsp_by_irrep"),
        valleyscope_key_by_source_irrep=valleyscope_key_by_source_irrep,
        valleyscope_irrep_multiplicity_by_source_irrep=(
            valleyscope_irrep_multiplicity_by_source_irrep
        ),
        expected_hsps=expected_hsps,
        allowed_irrep_keys=allowed_irrep_keys,
        subspace_group_candidate=subspace_group_candidate,
        provenance=provenance,
        subspace_space_group=(
            dict(subspace_space_group_raw)
            if isinstance(subspace_space_group_raw, Mapping)
            else None
        ),
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


# ---------------------------------------------------------------------------
# Auto-canonical EBR table builder
# ---------------------------------------------------------------------------

def build_auto_canonical_reduced_ebr_table(
    *,
    subspace_sg_number: int,
    spinor: bool,
    bundle_irreps_by_kpoint: Mapping[str, Sequence[str]],
    expected_hsps: Sequence[str],
    subspace_group_candidate: str,
    subspace_space_group: Mapping[str, object] | None = None,
    source_loader: Callable[[int | str, bool], Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Build a reduced EBR table automatically from canonical irrep data.

    Derives the source-label → ValleyScope-key mapping directly from
    canonical irreptables irrep labels without a user-supplied spec file
    or hand-written table.  Both the canonical irrep labels and the
    irreptables EBR basis labels come from the same irreptables source,
    so the mapping is identity: ``-GM4`` → ``GammaM:-GM4``.

    Parameters
    ----------
    subspace_sg_number : int
        Valley-projected subspace space-group number (e.g. 143 for P3).
    spinor : bool
        Whether to load spinful (double-group) EBR data.
    bundle_irreps_by_kpoint : Mapping[str, Sequence[str]]
        Canonical irrep labels grouped by ValleyScope HSP, as carried by
        the EBR export bundle (``irreps_by_kpoint`` field).
    expected_hsps : Sequence[str]
        Sampled ValleyScope HSP labels to include in the reduced basis.
    subspace_group_candidate : str
        Subspace-group symbol for the reduced table key (e.g. ``"P3"``).
    subspace_space_group : Mapping or None
        Optional structured subspace-space-group provenance.
    source_loader : Callable or None
        Injectable loader for irreptables EBR data (test seam).

    Returns
    -------
    dict
        A validated reduced EBR table dict compatible with
        ``load_reduced_ebr_table`` and ``build_reduced_ebr_mapping``.

    Raises
    ------
    ValueError
        If canonical labels cannot be resolved to irreptables kpoints,
        expected_hsps is empty, or the reduced table is empty.
    RuntimeError
        If irreptables EBR data cannot be loaded.
    """
    from valleyscope.irreps.tables import load_standard_irrep_table

    if not expected_hsps:
        raise ValueError("expected_hsps must be non-empty")

    # --- 1. Load irreptables irrep table to build label → Bilbao kpoint ---
    irrep_table = load_standard_irrep_table(subspace_sg_number, spinor=spinor)
    label_to_bilbao_kp: dict[str, str] = {}
    bilbao_kp_set: set[str] = set()
    for irrep in irrep_table.irreps:
        label_to_bilbao_kp[irrep.label] = irrep.kpoint_label
        bilbao_kp_set.add(irrep.kpoint_label)

    # --- 2a. Validate every canonical bundle label resolves ---
    unresolved_bundle_labels: list[str] = []
    for vs_hsp, labels in bundle_irreps_by_kpoint.items():
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            continue
        for label in labels:
            lbl = str(label)
            if lbl not in label_to_bilbao_kp:
                unresolved_bundle_labels.append(
                    f"{lbl} (declared HSP: {vs_hsp})"
                )
    if unresolved_bundle_labels:
        raise ValueError(
            "canonical bundle labels not found in irreptables irrep table "
            f"for SG {subspace_sg_number} spinor={spinor}: "
            f"{unresolved_bundle_labels}"
        )

    # --- 2b. Build Bilbao kpoint → ValleyScope HSP from canonical data ---
    bilbao_to_valleyscope: dict[str, str] = {}
    for vs_hsp, labels in bundle_irreps_by_kpoint.items():
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            continue
        for label in labels:
            bilbao_kp = label_to_bilbao_kp[str(label)]
            if (
                bilbao_kp in bilbao_to_valleyscope
                and bilbao_to_valleyscope[bilbao_kp] != vs_hsp
            ):
                raise ValueError(
                    f"conflicting HSP mapping for Bilbao kpoint "
                    f"{bilbao_kp!r}: {bilbao_to_valleyscope[bilbao_kp]!r} "
                    f"vs {vs_hsp!r}"
                )
            bilbao_to_valleyscope[bilbao_kp] = vs_hsp

    if not bilbao_to_valleyscope:
        raise ValueError(
            "could not derive Bilbao→ValleyScope HSP mapping from "
            "canonical irrep labels"
        )

    # --- 3. Load irreptables EBR data (once, cached for downstream) ---
    loader = source_loader or _load_ebr_data_from_irreptables
    try:
        ebr_data = loader(subspace_sg_number, spinor)
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            "failed to load irreptables EBR data for "
            f"space_group_number={subspace_sg_number!r}, "
            f"spinful={spinor}: {type(exc).__name__}: {exc}"
        ) from exc

    ebr_basis_labels = _extract_ebr_basis_labels(ebr_data)

    # Wrap loader so downstream build_reduced_table_from_irreptables
    # reuses the already-loaded data.
    def _cached_loader(_sg: int | str, _spin: bool) -> Mapping[str, object]:
        return ebr_data

    # --- 4a. Validate and derive source_hsp_by_irrep ---
    # Every EBR basis label must resolve to a known Bilbao kpoint.
    # Labels from the IrrepTable resolve directly.  Compatibility labels
    # (e.g. -HA4, -KA4) that only appear in the EBR data are resolved
    # by matching the longest known kpoint prefix.
    unresolved_ebr_labels: list[str] = []
    unsampled_hsp_labels: list[str] = []
    source_hsp_by_irrep: dict[str, str] = {}

    kpoint_labels_sorted = sorted(bilbao_kp_set, key=len, reverse=True)

    def _resolve_ebr_bilbao_kp(label: str) -> str | None:
        """Resolve a Bilbao kpoint for an EBR basis label.

        Labels known to the IrrepTable resolve directly.  Unknown labels
        are matched against the longest known kpoint prefix.

        Returns the kpoint string, or ``None`` if the label cannot be
        resolved at all (no known kpoint prefix match).
        """
        if label in label_to_bilbao_kp:
            return label_to_bilbao_kp[label]
        stripped = label[1:] if label.startswith("-") else label
        for kp in kpoint_labels_sorted:
            if stripped.startswith(kp):
                return kp
        return None

    def _is_compatibility_label(label: str, bilbao_kp: str) -> bool:
        """Check whether an EBR label is a compatibility (composite) label.

        A compatibility label like ``-KA4`` bridges two kpoints (K and A).
        After stripping the matched kpoint prefix, the remainder starts
        with another known kpoint label.
        """
        if label in label_to_bilbao_kp:
            return False  # known to IrrepTable — trusted
        stripped = label[1:] if label.startswith("-") else label
        remainder = stripped[len(bilbao_kp):]
        for kp2 in kpoint_labels_sorted:
            if kp2 != bilbao_kp and remainder.startswith(kp2):
                return True
        return False

    compatibility_labels: list[str] = []
    for label in ebr_basis_labels:
        bilbao_kp = _resolve_ebr_bilbao_kp(label)
        if bilbao_kp is None:
            unresolved_ebr_labels.append(label)
            continue
        if _is_compatibility_label(label, bilbao_kp):
            # Composite label bridging two kpoints (e.g. K↔A).
            # Map to a composite HSP name so it is filtered out.
            compatibility_labels.append(label)
            source_hsp_by_irrep[label] = f"_{bilbao_kp}_compat"
            continue
        if bilbao_kp in bilbao_to_valleyscope:
            source_hsp_by_irrep[label] = bilbao_to_valleyscope[bilbao_kp]
        else:
            source_hsp_by_irrep[label] = bilbao_kp
            unsampled_hsp_labels.append(f"{label} (Bilbao {bilbao_kp})")

    if unresolved_ebr_labels:
        raise ValueError(
            "EBR basis labels could not be resolved to a Bilbao kpoint "
            f"for SG {subspace_sg_number} spinor={spinor}: "
            f"{unresolved_ebr_labels}"
        )

    # --- 4b. Auto-derive valleyscope_irrep_multiplicity_by_source_irrep ---
    hsps_set = set(expected_hsps)
    valleyscope_irrep_multiplicity_by_source_irrep: dict[str, dict[str, int]] = {}
    for label in ebr_basis_labels:
        bilbao_kp = _resolve_ebr_bilbao_kp(label)
        vs_hsp = bilbao_to_valleyscope.get(bilbao_kp) if bilbao_kp else None
        if vs_hsp and vs_hsp in hsps_set:
            valleyscope_irrep_multiplicity_by_source_irrep[label] = {
                f"{vs_hsp}:{label}": 1
            }

    # --- 4c. Build allowed_irrep_keys: all irrep keys at sampled HSPs ---
    allowed_irrep_keys: list[str] = []
    for label in ebr_basis_labels:
        vs_hsp = source_hsp_by_irrep.get(label, "")
        if vs_hsp in hsps_set:
            key = f"{vs_hsp}:{label}"
            if key not in allowed_irrep_keys:
                allowed_irrep_keys.append(key)

    if not allowed_irrep_keys:
        raise ValueError(
            "no irreptables EBR basis labels map to expected_hsps "
            f"{sorted(hsps_set)}"
        )

    # --- 5. Build provenance ---
    provenance: dict[str, object] = {
        "data_source": "irreptables",
        "package": "irreptables",
        "package_version": _package_version("irreptables"),
        "space_group_number": subspace_sg_number,
        "spinful": spinor,
        "expected_hsps": list(expected_hsps),
        "subspace_group_candidate": subspace_group_candidate,
        "valleyscope_reduction": "sampled_hsp_valley_preserving",
        "auto_canonical": True,
        "bilbao_to_valleyscope_hsp": dict(bilbao_to_valleyscope),
    }
    if unsampled_hsp_labels:
        provenance["unsampled_hsp_labels"] = unsampled_hsp_labels
        provenance["unsampled_hsp_count"] = len(unsampled_hsp_labels)
    if compatibility_labels:
        provenance["compatibility_labels"] = compatibility_labels
        provenance["compatibility_label_count"] = len(compatibility_labels)

    # --- 6. Build reduced table ---
    table = build_reduced_table_from_irreptables(
        space_group_number=subspace_sg_number,
        spinful=spinor,
        source_loader=_cached_loader,
        source_hsp_by_irrep=source_hsp_by_irrep,
        valleyscope_irrep_multiplicity_by_source_irrep=(
            valleyscope_irrep_multiplicity_by_source_irrep
        ),
        expected_hsps=expected_hsps,
        allowed_irrep_keys=allowed_irrep_keys,
        subspace_group_candidate=subspace_group_candidate,
        provenance=provenance,
        subspace_space_group=(
            dict(subspace_space_group)
            if isinstance(subspace_space_group, Mapping) else None
        ),
    )

    _validate_reduced_table_dict(table)
    return table


def _extract_ebr_basis_labels(ebr_data: Mapping[str, object]) -> list[str]:
    """Extract basis irrep labels from irreptables EBR data dict."""
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
    return labels
