from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


V1_PROFILE_IDENTITY = "vasp_nonmagnetic_soc_default_saxis_v1"
V1_EVIDENCE_ORIGIN = "workflow_scope_contract"
V1_PROFILE_ASSUMPTIONS: dict[str, object] = {
    "nonmagnetic": True,
    "soc": True,
    "time_reversal": True,
    "saxis_cart": [0.0, 0.0, 1.0],
}
H5_PARSER_IDENTITY = "valleyscope_h5_reader_v1"
H5_LAYOUT_IDENTITY = "valleyscope_wavefunction_h5_layout_v1"
WAVECAR_H5_EXTRACTOR_IDENTITY = "valleyscope_extract_wavecar_v1"
COEFFICIENT_SHAPE_ORDER = (
    "band",
    "spinor_component",
    "reciprocal_grid",
)


def spinor_component_order(nspinor: int) -> tuple[str, ...]:
    if nspinor == 1:
        return ("scalar_component_0",)
    if nspinor == 2:
        return (
            "vasp_spinor_component_0",
            "vasp_spinor_component_1",
        )
    raise ValueError(f"unsupported nspinor={nspinor}")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_identity(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"sha256:{digest}"


def file_payload_identity(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def valid_sha256_identity(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
