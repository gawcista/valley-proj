"""Diagnostic-only availability probe for the external ``irrep`` package.

Reports whether the ``irrep`` package and its key submodules can be
imported without actually calling any decomposition routines.  This is
a diagnostic tool, not a runtime dependency gate.
"""

from __future__ import annotations

from typing import Any


def probe_irrep_availability() -> dict[str, Any]:
    """Probe irrep package availability without calling decomposition.

    Returns a dict with availability flags and diagnostic messages.
    Does NOT import the private ``irrep2`` repository.
    """
    result: dict[str, Any] = {
        "package": "irrep",
        "irrep_available": False,
        "irrep_version": None,
        "irrep_path": None,
        "spacegroup_irreps_available": False,
        "ebrs_available": False,
        "errors": [],
    }

    # Top-level package.
    try:
        import irrep
        result["irrep_available"] = True
        result["irrep_version"] = getattr(irrep, "__version__", None)
        result["irrep_path"] = getattr(irrep, "__file__", None)
    except ImportError as exc:
        result["errors"].append(f"cannot import irrep: {exc}")
        return result

    # spacegroup_irreps — needed for irrep table lookup.
    try:
        import irrep.spacegroup_irreps  # noqa: F401
        result["spacegroup_irreps_available"] = True
    except ImportError as exc:
        result["errors"].append(f"cannot import irrep.spacegroup_irreps: {exc}")

    # ebrs — needed for EBR matrix / Smith form.
    try:
        import irrep.ebrs  # noqa: F401
        result["ebrs_available"] = True
    except ImportError as exc:
        result["errors"].append(f"cannot import irrep.ebrs: {exc}")

    return result


def probe_irrep_is_importable() -> bool:
    """Return True if the irrep top-level package can be imported."""
    info = probe_irrep_availability()
    return bool(info["irrep_available"])
