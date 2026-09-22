from __future__ import annotations

import os

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip development-document contract tests by default.

    Public clones intentionally track only README.md and README.zh.md. The
    internal Markdown notes may exist in a developer workspace, but tests must
    not require them unless explicitly requested.
    """
    if (
        item.get_closest_marker("dev_docs") is not None
        and os.environ.get("VALLEYSCOPE_RUN_DEV_DOC_TESTS") != "1"
    ):
        pytest.skip(
            "development Markdown docs are not tracked in the public repo; "
            "set VALLEYSCOPE_RUN_DEV_DOC_TESTS=1 to run these checks"
        )
