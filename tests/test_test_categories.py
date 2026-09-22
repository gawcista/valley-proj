"""The local development-doc opt-in must never hide portable tests."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("opt_in", [None, "0", "1"])
def test_dev_docs_opt_in_preserves_unmarked_tests(tmp_path, opt_in):
    """Explicit markers alone control skipping, regardless of source text."""
    shutil.copyfile(Path(__file__).with_name("conftest.py"), tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    dev_docs: local development documents\n",
        encoding="utf-8",
    )
    (tmp_path / "test_examples.py").write_text(
        'from pathlib import Path\n'
        'import pytest\n\n'
        '@pytest.mark.dev_docs\n'
        'def test_development_document_contract():\n'
        '    Path("dev-doc-ran").touch()\n\n'
        'def test_portable_contract_with_document_path():\n'
        '    assert Path("docs/optional.md").name == "optional.md"\n'
        '    Path("portable-ran").touch()\n',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("VALLEYSCOPE_RUN_DEV_DOC_TESTS", None)
    if opt_in is not None:
        env["VALLEYSCOPE_RUN_DEV_DOC_TESTS"] = opt_in
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "portable-ran").exists(), result.stdout
    assert (tmp_path / "dev-doc-ran").exists() is (opt_in == "1"), result.stdout
