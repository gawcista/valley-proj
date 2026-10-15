"""Unit tests with synthetic wheels; these fixtures are not physics evidence."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from email.parser import BytesParser
from pathlib import Path
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_irrep_compat_wheel.py"
OLD_INFO = "irrep-3.3.0.dist-info"
NEW_INFO = "irrep-3.3.0+valleyscope.1.dist-info"
WHEEL_NAME = "irrep-3.3.0+valleyscope.1-py3-none-any.whl"
ORIGINAL_LINE = b'''            name_loc = f"{name}{f"_{i}" if i > 0 else ''}.npy"\n'''
PATCHED_LINE = b'''            name_loc = f"{name}{f'_{i}' if i > 0 else ''}.npy"\n'''
UNIT_STORAGE = b'''class Storage:
    def __init__(self, files):
        self.files = files

    def generate_name(self, name="array"):
        existing_files = set(self.files)
        i = 0
        while True:
            name_loc = f"{name}{f"_{i}" if i > 0 else ''}.npy"
            if name_loc not in existing_files:
                return name_loc
            i += 1
'''
EXPECTED_STORAGE = b'''class Storage:
    def __init__(self, files):
        self.files = files

    def generate_name(self, name="array"):
        existing_files = set(self.files)
        i = 0
        while True:
            name_loc = f"{name}{f'_{i}' if i > 0 else ''}.npy"
            if name_loc not in existing_files:
                return name_loc
            i += 1
'''


def load_builder():
    assert SCRIPT.is_file(), "compatibility wheel builder is not implemented"
    spec = importlib.util.spec_from_file_location("irrep_compat_wheel", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit_wheel(tmp_path, monkeypatch, builder, storage=UNIT_STORAGE):
    """Replace only the fixed input digest for an explicitly synthetic wheel."""
    members = {
        "irrep/storage.py": storage,
        "irrep/__init__.py": b"from ._version import __version__\n",
        "irrep/_version.py": b'__version__ = "3.3.0"\n',
        "irrep/data/example.data": b"synthetic table\x00\xff\n",
        f"{OLD_INFO}/METADATA": (
            b"Metadata-Version: 2.4\nName: irrep\nVersion: 3.3.0\n"
            b"Requires-Dist: numpy\nLicense: GPL-3.0\n\nSynthetic package.\n"
        ),
        f"{OLD_INFO}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{OLD_INFO}/entry_points.txt": b"[console_scripts]\nirrep = irrep.cli:cli\n",
        f"{OLD_INFO}/licenses/LICENSE": b"Synthetic license fixture\n",
        f"{OLD_INFO}/RECORD": b"old,sha256=stale,1\n",
        f"{OLD_INFO}/RECORD.jws": b"obsolete signature",
        f"{OLD_INFO}/RECORD.p7s": b"obsolete signature",
    }
    source = tmp_path / "irrep-3.3.0-py3-none-any.whl"
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    monkeypatch.setattr(
        builder, "UPSTREAM_WHEEL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    return source, members


def test_rejects_wrong_upstream_digest_before_creating_output(tmp_path):
    builder = load_builder()
    source = tmp_path / "untrusted.whl"
    source.write_bytes(b"not the official wheel")
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="SHA256"):
        builder.build_compat_wheel(source, output_dir)

    assert not output_dir.exists()


@pytest.mark.parametrize("matches", [0, 2])
def test_requires_exactly_one_expected_source_line(tmp_path, monkeypatch, matches):
    builder = load_builder()
    storage = UNIT_STORAGE.replace(ORIGINAL_LINE, ORIGINAL_LINE * matches)
    source, _ = unit_wheel(tmp_path, monkeypatch, builder, storage)
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="exactly once"):
        builder.build_compat_wheel(source, output_dir)

    assert not output_dir.exists()


def test_updates_distribution_metadata_record_and_patch_provenance(tmp_path, monkeypatch):
    builder = load_builder()
    source, members = unit_wheel(tmp_path, monkeypatch, builder)
    output = builder.build_compat_wheel(source, tmp_path / "output")

    assert output.name == WHEEL_NAME
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert not any(name.startswith(f"{OLD_INFO}/") for name in names)
        assert f"{NEW_INFO}/RECORD.jws" not in names
        assert f"{NEW_INFO}/RECORD.p7s" not in names
        metadata = BytesParser().parsebytes(archive.read(f"{NEW_INFO}/METADATA"))
        assert metadata["Name"] == "irrep"
        assert metadata["Version"] == "3.3.0+valleyscope.1"
        assert metadata["Requires-Python"] == ">=3.10"
        assert metadata.get_all("Requires-Dist") == ["numpy"]
        assert metadata.get_payload() == "Synthetic package.\n"

        record_path = f"{NEW_INFO}/RECORD"
        rows = list(csv.reader(io.StringIO(archive.read(record_path).decode("utf-8"))))
        assert len(rows) == len(names)
        assert {row[0] for row in rows} == set(names)
        for name, digest, size in rows:
            if name == record_path:
                assert (digest, size) == ("", "")
            else:
                data = archive.read(name)
                encoded_digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                assert digest == "sha256=" + encoded_digest.rstrip(b"=").decode("ascii")
                assert size == str(len(data))

        provenance = json.loads(archive.read(f"{NEW_INFO}/valleyscope_patch.json"))
        assert provenance == {
            "upstream_version": "3.3.0",
            "upstream_wheel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "upstream_wheel_url": (
                "https://files.pythonhosted.org/packages/75/01/"
                "815ff9a8eb5d1897e3114753892372d5b440b7ef564e8209b31fb3f15d8c/"
                "irrep-3.3.0-py3-none-any.whl"
            ),
            "upstream_source_url": (
                "https://files.pythonhosted.org/packages/40/8d/"
                "fa99aed4825e6c67103f7752b7746a635f1925486067e989bc736599051a/"
                "irrep-3.3.0.tar.gz"
            ),
            "upstream_source_sha256": (
                "9a812a27baf20f625d3f261b159e124f307ffecd7b29c12ff4dbf2de80a2f5d1"
            ),
            "local_version": "3.3.0+valleyscope.1",
            "patched_file": "irrep/storage.py",
            "original_file_sha256": hashlib.sha256(members["irrep/storage.py"]).hexdigest(),
            "patched_file_sha256": hashlib.sha256(EXPECTED_STORAGE).hexdigest(),
            "patch_date": "2026-10-15",
            "modified_by": "ValleyScope contributors",
            "description": (
                "Use distinct f-string quotes in Storage.generate_name "
                "for Python 3.10 compatibility."
            ),
        }


def test_changes_only_storage_line_and_preserves_other_file_bytes(tmp_path, monkeypatch):
    builder = load_builder()
    source, members = unit_wheel(tmp_path, monkeypatch, builder)
    output = builder.build_compat_wheel(source, tmp_path / "output")

    with zipfile.ZipFile(output) as archive:
        assert archive.read("irrep/storage.py") == EXPECTED_STORAGE
        changed = {"irrep/storage.py", f"{OLD_INFO}/METADATA", f"{OLD_INFO}/RECORD"}
        signatures = {f"{OLD_INFO}/RECORD.jws", f"{OLD_INFO}/RECORD.p7s"}
        expected_names = {
            name.replace(f"{OLD_INFO}/", f"{NEW_INFO}/", 1)
            for name in members
            if name not in signatures
        } | {f"{NEW_INFO}/valleyscope_patch.json"}
        assert set(archive.namelist()) == expected_names
        for name, data in members.items():
            if name not in changed | signatures:
                destination = name.replace(f"{OLD_INFO}/", f"{NEW_INFO}/", 1)
                assert archive.read(destination) == data

        namespace = {}
        exec(compile(archive.read("irrep/storage.py"), "irrep/storage.py", "exec"), namespace)
        storage = namespace["Storage"]([])
        assert storage.generate_name("sample") == "sample.npy"
        storage.files.extend(["sample.npy", "sample_1.npy"])
        assert storage.generate_name("sample") == "sample_2.npy"


def test_repeated_builds_are_identical_and_reuse_existing_artifact(tmp_path, monkeypatch):
    builder = load_builder()
    source, _ = unit_wheel(tmp_path, monkeypatch, builder)
    output = builder.build_compat_wheel(source, tmp_path / "output")
    original = output.read_bytes()
    os.utime(output, ns=(1_000_000_000, 1_000_000_000))

    assert builder.build_compat_wheel(source, output.parent) == output
    assert output.stat().st_mtime_ns == 1_000_000_000
    separate_output = builder.build_compat_wheel(source, tmp_path / "separate-output")
    assert output.read_bytes() == separate_output.read_bytes() == original


def test_rejects_conflicting_output_without_overwriting_it(tmp_path, monkeypatch):
    builder = load_builder()
    source, _ = unit_wheel(tmp_path, monkeypatch, builder)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    conflicting = output_dir / WHEEL_NAME
    conflicting.write_bytes(b"existing different artifact")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder.build_compat_wheel(source, output_dir)

    assert conflicting.read_bytes() == b"existing different artifact"


def test_downloaded_input_uses_the_same_hash_checked_build(tmp_path, monkeypatch):
    builder = load_builder()
    source, _ = unit_wheel(tmp_path, monkeypatch, builder)
    explicit_output = builder.build_compat_wheel(source, tmp_path / "explicit")

    def fake_urlopen(url, *, timeout):
        assert url == (
            "https://files.pythonhosted.org/packages/75/01/"
            "815ff9a8eb5d1897e3114753892372d5b440b7ef564e8209b31fb3f15d8c/"
            "irrep-3.3.0-py3-none-any.whl"
        )
        return io.BytesIO(source.read_bytes())

    monkeypatch.setattr(builder, "urlopen", fake_urlopen)
    downloaded_output = builder.build_compat_wheel(None, tmp_path / "downloaded")
    assert downloaded_output.read_bytes() == explicit_output.read_bytes()

    monkeypatch.setattr(builder, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="SHA256"):
        builder.build_compat_wheel(None, tmp_path / "untrusted-download")
    assert not (tmp_path / "untrusted-download").exists()


def test_cli_accepts_offline_input_and_prints_output_path(tmp_path, monkeypatch, capsys):
    builder = load_builder()
    source, _ = unit_wheel(tmp_path, monkeypatch, builder)
    output_dir = tmp_path / "output"

    assert builder.main(["--source-wheel", str(source), "--output-dir", str(output_dir)]) == 0

    output = Path(capsys.readouterr().out.strip())
    assert output == output_dir / WHEEL_NAME
    assert zipfile.is_zipfile(output)


def test_cli_defaults_to_dist_compat(tmp_path, monkeypatch, capsys):
    builder = load_builder()
    source, _ = unit_wheel(tmp_path, monkeypatch, builder)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builder, "urlopen", lambda *args, **kwargs: io.BytesIO(source.read_bytes()))

    assert builder.main([]) == 0

    output = Path(capsys.readouterr().out.strip())
    assert output == tmp_path / "dist/compat" / WHEEL_NAME
    assert zipfile.is_zipfile(output)
