from __future__ import annotations

import inspect
import os
import re

import pytest


_DEV_MARKDOWN_PATH_RE = re.compile(
    r"Path\([\"']("
    r"docs/[^\"']+\.md|"
    r"AGENTS\.md|"
    r"CLAUDE\.md|"
    r"PLAN\.md|"
    r"\.codex_cc_handoff\.template\.md|"
    r"valleyscope/data/reduced_ebr/README\.md"
    r")[\"']\)"
)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip development-document contract tests by default.

    Public clones intentionally track only README.md and README.zh.md. The
    internal Markdown notes may exist in a developer workspace, but tests must
    not require them unless explicitly requested.
    """
    if os.environ.get("VALLEYSCOPE_RUN_DEV_DOC_TESTS") == "1":
        return

    try:
        source = inspect.getsource(item.obj)
    except (OSError, TypeError):
        source = ""

    if _DEV_MARKDOWN_PATH_RE.search(source):
        pytest.skip(
            "development Markdown docs are not tracked in the public repo; "
            "set VALLEYSCOPE_RUN_DEV_DOC_TESTS=1 to run these checks"
        )
