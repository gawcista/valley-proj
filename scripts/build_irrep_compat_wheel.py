#!/usr/bin/env python3
"""Build the reviewed Python 3.10 syntax repair of the official irrep 3.3.0 wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile


UPSTREAM_VERSION = "3.3.0"
LOCAL_VERSION = "3.3.0+valleyscope.1"
UPSTREAM_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/75/01/"
    "815ff9a8eb5d1897e3114753892372d5b440b7ef564e8209b31fb3f15d8c/"
    "irrep-3.3.0-py3-none-any.whl"
)
UPSTREAM_WHEEL_SHA256 = "cf23d8f77471dcf7d2efe6029f47f17ddcd29597ec332ec77747001414b0f54f"
UPSTREAM_SOURCE_URL = (
    "https://files.pythonhosted.org/packages/40/8d/"
    "fa99aed4825e6c67103f7752b7746a635f1925486067e989bc736599051a/"
    "irrep-3.3.0.tar.gz"
)
UPSTREAM_SOURCE_SHA256 = "9a812a27baf20f625d3f261b159e124f307ffecd7b29c12ff4dbf2de80a2f5d1"
PATCHED_FILE = "irrep/storage.py"
_ORIGINAL_LINE = b'''            name_loc = f"{name}{f"_{i}" if i > 0 else ''}.npy"\n'''
_PATCHED_LINE = b'''            name_loc = f"{name}{f'_{i}' if i > 0 else ''}.npy"\n'''


def build_compat_wheel(source_wheel: Path | None, output_dir: Path) -> Path:
    """Verify the fixed upstream input and write or reuse a deterministic wheel."""
    if source_wheel is None:
        with urlopen(UPSTREAM_WHEEL_URL, timeout=30) as response:
            source_bytes = response.read()
    else:
        source_bytes = Path(source_wheel).read_bytes()
    actual_digest = hashlib.sha256(source_bytes).hexdigest()
    if actual_digest != UPSTREAM_WHEEL_SHA256:
        raise ValueError(
            f"upstream wheel SHA256 mismatch: expected {UPSTREAM_WHEEL_SHA256}, "
            f"got {actual_digest}"
        )

    old_info = f"irrep-{UPSTREAM_VERSION}.dist-info/"
    new_info = f"irrep-{LOCAL_VERSION}.dist-info/"
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as upstream:
        names = upstream.namelist()
        if len(names) != len(set(names)):
            raise ValueError("upstream wheel contains duplicate member paths")
        members = {}
        for name in names:
            if name in {old_info + suffix for suffix in ("RECORD", "RECORD.jws", "RECORD.p7s")}:
                continue
            destination = new_info + name[len(old_info):] if name.startswith(old_info) else name
            members[destination] = upstream.read(name)

    original = members[PATCHED_FILE]
    if original.count(_ORIGINAL_LINE) != 1:
        raise ValueError("expected storage.py source line must occur exactly once")
    patched = original.replace(_ORIGINAL_LINE, _PATCHED_LINE, 1)
    members[PATCHED_FILE] = patched

    metadata_path = new_info + "METADATA"
    header, separator, body = members[metadata_path].partition(b"\n\n")
    lines = header.split(b"\n")
    version_lines = [line for line in lines if line.startswith(b"Version:")]
    if not separator or version_lines != [b"Version: 3.3.0"]:
        raise ValueError("unexpected upstream distribution version metadata")
    lines = [
        f"Version: {LOCAL_VERSION}".encode("ascii") if line == b"Version: 3.3.0" else line
        for line in lines
    ]
    lines = [line for line in lines if not line.startswith(b"Requires-Python:")]
    lines.append(b"Requires-Python: >=3.10")
    members[metadata_path] = b"\n".join(lines) + separator + body

    provenance = {
        "upstream_version": UPSTREAM_VERSION,
        "upstream_wheel_sha256": actual_digest,
        "upstream_wheel_url": UPSTREAM_WHEEL_URL,
        "upstream_source_url": UPSTREAM_SOURCE_URL,
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
        "local_version": LOCAL_VERSION,
        "patched_file": PATCHED_FILE,
        "original_file_sha256": hashlib.sha256(original).hexdigest(),
        "patched_file_sha256": hashlib.sha256(patched).hexdigest(),
        "patch_date": "2026-10-15",
        "modified_by": "ValleyScope contributors",
        "description": (
            "Use distinct f-string quotes in Storage.generate_name "
            "for Python 3.10 compatibility."
        ),
    }
    members[new_info + "valleyscope_patch.json"] = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    record_path = new_info + "RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name in sorted(members):
        data = members[name]
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow((name, "sha256=" + digest.decode("ascii"), len(data)))
    writer.writerow((record_path, "", ""))
    members[record_path] = record.getvalue().encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            wheel.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    wheel_bytes = buffer.getvalue()

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"irrep-{LOCAL_VERSION}-py3-none-any.whl"
    try:
        with output.open("xb") as destination:
            destination.write(wheel_bytes)
    except FileExistsError:
        if output.read_bytes() != wheel_bytes:
            raise FileExistsError(f"refusing to overwrite different existing artifact: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-wheel", type=Path,
        help="official irrep 3.3.0 wheel for offline input; otherwise download the fixed URL",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dist/compat"))
    args = parser.parse_args(argv)
    try:
        output = build_compat_wheel(args.source_wheel, args.output_dir)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
