"""Offline builder: irreptables.ebrs.load_ebr_data -> reduced external table.

Lazily loads 3D EBR data from the public ``irreptables`` package for a given
projected-subspace/moire space group, then normalizes and reduces it through
the existing ValleyScope normalizer + reducer pipeline to produce a
ValleyScope reduced-dimensional external table.

This is a library builder, not wired into ``analyze_hsp.py``.  All source-irrep
to ValleyScope label mappings must be supplied explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_reduced_table_from_irreptables(
    *,
    sg_number: int,
    spinor: bool,
    source_hsp_by_irrep: Mapping[str, str],
    valleyscope_key_by_source_irrep: Mapping[str, str],
    expected_hsps: Sequence[str],
    allowed_irrep_keys: Sequence[str],
    subspace_group_candidate: str,
    provenance_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a ValleyScope reduced external table from irreptables EBR data.

    Parameters
    ----------
    sg_number : int
        Projected-subspace / moire space group number (e.g. 150 for P321).
    spinor : bool
        Whether to use double-valued (spinor) irreps.
    source_hsp_by_irrep : Mapping
        Map from source irrep label (e.g. ``"-GM5"``) to sampled moire HSP
        label (e.g. ``"GammaM"``).  All labels in the source basis must have
        an entry; missing entries raise ValueError.
    valleyscope_key_by_source_irrep : Mapping
        Map from source irrep label to ValleyScope valley-preserving irrep key
        (e.g. ``"GammaM:C3_spinor_phase_+1/2"``).  All labels in the source
        basis must have an entry.
    expected_hsps : Sequence[str]
        Sampled moire HSP labels.  Only irreps whose HSP is in this set are
        included in the reduced output.
    allowed_irrep_keys : Sequence[str]
        ValleyScope trusted valley-preserving irrep keys.  Must be a subset
        of the values in ``valleyscope_key_by_source_irrep``.
    subspace_group_candidate : str
        The ``C{order}_like`` group label.
    provenance_extra : Mapping or None
        Extra provenance fields attached to the output.

    Returns
    -------
    dict
        A dict compatible with ``load_reduced_ebr_table()``.

    Raises
    ------
    ValueError
        If mapping is incomplete or the irreptables package is unavailable.
    """
    from irreptables.ebrs import load_ebr_data

    from valleyscope.analysis.irrep_data_normalizer import (
        build_runtime_source_payload_from_ebr_data,
    )
    from valleyscope.analysis.irrep_runtime_reducer import (
        build_reduced_table_from_runtime_source,
    )

    # Load source 3D EBR data.
    ebr_data = load_ebr_data(int(sg_number), bool(spinor))
    source_pkg = {"package": "irreptables", "sg_number": int(sg_number),
                   "spinor": bool(spinor)}

    # Normalize to runtime source payload.
    source_payload = build_runtime_source_payload_from_ebr_data(
        ebr_data=ebr_data,
        source_hsp_by_irrep=source_hsp_by_irrep,
        valleyscope_key_by_source_irrep=valleyscope_key_by_source_irrep,
        source=source_pkg,
    )

    # Build provenance.
    provenance: dict[str, Any] = dict(source_pkg)
    if provenance_extra:
        provenance.update(provenance_extra)

    # Reduce to sampled HSPs.
    return build_reduced_table_from_runtime_source(
        source_payload=source_payload,
        expected_hsps=expected_hsps,
        allowed_irrep_keys=allowed_irrep_keys,
        subspace_group_candidate=subspace_group_candidate,
        provenance=provenance,
    )
