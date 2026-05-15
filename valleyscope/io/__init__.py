"""Input and configuration parsing."""

from __future__ import annotations

from pathlib import Path


def resolve_config_path(base: Path, value: str | None) -> Path | None:
    """Resolve a config path: absolute paths pass through, relative paths join base."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path
