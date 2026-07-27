"""Real-affine primitive certificate helpers for reduced-EBR tests.

Primitive standard-setting certificates are produced by the ACTUAL resolver
(``resolve_standard_setting_hsp_label``) fed a REAL ``StandardIrrepTable`` and a
complete detected-operation set built from spglib standard operations, so the
generic affine ``{R | tau}`` operation-equivalence gate actually runs.  Nothing
is mutated after the resolver returns; there is no Gamma-only coordinate stub.

Only primitive space groups with a unique spglib Hall setting yield a validated
certificate here.  Centered/multiple-Hall settings remain Phase E and produce
None (the caller must then expect a fail-closed blocker).
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np

from valleyscope.analysis.standard_setting_kmap import (
    resolve_standard_setting_hsp_label,
)
from valleyscope.analysis.ebr_problem_instances import _certificate_identity
from valleyscope.analysis.reduced_ebr_mapping import (
    _derive_table_standard_setting,
)
from valleyscope.io.wavefunction_convention import canonical_identity
from valleyscope.io.spinor_source_basis import SpinorSourceBasisCertificate
from valleyscope.analysis.scoped_representation_evidence import (
    build_scoped_representation_evidence,
)
from valleyscope.symmetry.double_space_group_lift import (
    build_double_space_group_lift_certificate,
    spin_lift_from_orthogonal,
)


def attach_cprime_fixture_contract(export_bundle: dict) -> dict:
    """Attach identities derived from recomputable producer fixtures."""
    context = cprime_validation_context_for_export(export_bundle)
    source_identity = _cprime_fixture_source_record()[
        "certificate_identity"
    ]
    for bundle in export_bundle.get("bundles", []):
        if not isinstance(bundle, dict):
            continue
        links_by_kpoint: dict[str, dict[str, str]] = {}
        irreps = bundle.get("irreps_by_kpoint", {})
        if not isinstance(irreps, dict):
            continue
        records = bundle.get("irrep_records_by_kpoint", {})
        if not isinstance(records, dict):
            records = {}
        bundle["irrep_records_by_kpoint"] = records
        construction = bundle.get("unitary_vector_construction")
        tr_completed = (
            isinstance(construction, dict)
            and construction.get("kind")
            == "time_reversal_completed_unitary_rows"
        )
        joint_problem = (
            bundle.get("problem_kind") == "valley_orbit_reduced_ebr"
        )
        completion_records = bundle.get(
            "unitary_irrep_completion_records_by_hsp", {}
        )
        if not isinstance(completion_records, dict):
            completion_records = {}
        for kpoint in irreps:
            links = {
                "spinor_source_basis_certificate_identity": source_identity,
                "double_space_group_lift_certificate_identity": "",
                "scoped_representation_evidence_identity": "",
            }
            context_entry = context[(id(bundle), str(kpoint))]
            scoped_record = context_entry["record"]
            links["double_space_group_lift_certificate_identity"] = (
                scoped_record[
                    "double_space_group_lift_certificate_identity"
                ]
            )
            links["scoped_representation_evidence_identity"] = (
                scoped_record["evidence_identity"]
            )
            links_by_kpoint[str(kpoint)] = links
            if (
                not tr_completed
                and not joint_problem
                and (
                    not isinstance(records.get(kpoint), list)
                    or not records.get(kpoint)
                )
            ):
                records[kpoint] = [
                    {
                        "matched_irrep": str(label),
                        "irrep_source_provenance": {},
                    }
                    for label in irreps.get(kpoint, [])
                ]
            for row in records.get(kpoint, []):
                if not isinstance(row, dict):
                    continue
                provenance = row.get("irrep_source_provenance")
                if not isinstance(provenance, dict):
                    provenance = {}
                    row["irrep_source_provenance"] = provenance
                provenance["cprime"] = dict(links)
            if tr_completed:
                for completion in completion_records.get(kpoint, []):
                    if not isinstance(completion, dict):
                        continue
                    candidate_provenance = completion.get(
                        "source_candidate_provenance"
                    )
                    if not isinstance(candidate_provenance, dict):
                        continue
                    candidate_provenance = deepcopy(candidate_provenance)
                    completion["source_candidate_provenance"] = (
                        candidate_provenance
                    )
                    irrep_provenance = candidate_provenance.get(
                        "irrep_source_provenance"
                    )
                    if not isinstance(irrep_provenance, dict):
                        irrep_provenance = {}
                        candidate_provenance[
                            "irrep_source_provenance"
                        ] = irrep_provenance
                    irrep_provenance["cprime"] = dict(links)
        bundle["cprime_identity_by_kpoint"] = links_by_kpoint
    return export_bundle


def cprime_validation_context_for_export(
    export_bundle: dict,
) -> dict[str, object]:
    """Build genuine, fully recomputable C-prime contexts for test bundles."""
    entries: dict[object, object] = {}
    by_identity: dict[str, object] = {}
    for bundle in export_bundle.get("bundles", []):
        if not isinstance(bundle, dict):
            continue
        irreps = bundle.get("irreps_by_kpoint", {})
        if not isinstance(irreps, dict):
            continue
        for kpoint in irreps:
            record, raw_inputs = _cprime_fixture_scope(
                bundle=bundle,
                kpoint=str(kpoint),
            )
            entry = {"record": record, "raw_inputs": raw_inputs}
            entries[(id(bundle), str(kpoint))] = entry
            by_identity[str(record["evidence_identity"])] = entry
    entries["_by_identity"] = by_identity
    return entries


def mapping_cprime_context(export_bundle: dict) -> dict[str, object]:
    """Attach fixture links and return the explicit solver context."""
    context = cprime_validation_context_for_export(export_bundle)
    attach_cprime_fixture_contract(export_bundle)
    rebuilt = cprime_validation_context_for_export(export_bundle)
    return dict(rebuilt["_by_identity"])


def _cprime_fixture_source_record() -> dict[str, object]:
    return SpinorSourceBasisCertificate(
        extracted_wavefunction_payload_identity="sha256:" + "a" * 64,
        nspinor=2,
        parser_identity="valleyscope_h5_reader_v1",
        extractor_identity="valleyscope_wavecar_h5_layout_v1",
    ).to_record()


def _cprime_fixture_operation(
    operation_id: int,
    rotation: np.ndarray,
) -> dict[str, object]:
    matrix = np.asarray(rotation, dtype=int)
    return {
        "operation_id": operation_id,
        "rotation_frac": matrix,
        "translation_frac": np.zeros(3),
        "rotation_cart": matrix.astype(float),
        "translation_cart": np.zeros(3),
    }


def _complex_matrix_record(matrix: np.ndarray) -> list:
    return [
        [[float(value.real), float(value.imag)] for value in row]
        for row in np.asarray(matrix, dtype=np.complex128)
    ]


def _cprime_fixture_lift_inputs() -> dict[str, object]:
    operations = [
        _cprime_fixture_operation(2, np.eye(3, dtype=int)),
        _cprime_fixture_operation(5, np.diag([1, -1, -1])),
    ]
    source_table = {
        "schema_version": "1.0.0",
        "provider": "irreptables",
        "data_source": "irreptables.StandardIrrepTable",
        "space_group_number": 1,
        "spinor": True,
        "operations": [
            {
                "table_index": index,
                "rotation_frac": operation["rotation_frac"].tolist(),
                "translation_frac": [0.0, 0.0, 0.0],
                "spin_rotation": _complex_matrix_record(
                    spin_lift_from_orthogonal(
                        operation["rotation_cart"]
                    )
                ),
            }
            for index, operation in enumerate(operations)
        ],
    }
    return {
        "expected_operations": operations,
        "source_table_identity": source_table,
        "standard_setting_identity": {
            "schema_version": "1.0.0",
            "parent_to_standard_direct_transform": np.eye(3).tolist(),
            "origin_shift_fractional": [0.0, 0.0, 0.0],
            "parent_to_standard_operation_map": {"2": 0, "5": 1},
        },
        "direct_lattice_cart": np.eye(3),
    }


def _cprime_fixture_scope(
    *,
    bundle: dict,
    kpoint: str,
) -> tuple[dict[str, object], dict[str, object]]:
    source = _cprime_fixture_source_record()
    lift_inputs = _cprime_fixture_lift_inputs()
    lift = build_double_space_group_lift_certificate(
        source,
        lift_inputs["expected_operations"],
        source_table_identity=lift_inputs["source_table_identity"],
        standard_setting_identity=lift_inputs[
            "standard_setting_identity"
        ],
        direct_lattice_cart=lift_inputs["direct_lattice_cart"],
    ).to_record()
    construction = bundle.get("unitary_vector_construction")
    tr_completed = (
        bundle.get("problem_kind") == "valley_orbit_reduced_ebr"
        or (
            isinstance(construction, dict)
            and construction.get("kind")
            == "time_reversal_completed_unitary_rows"
        )
    )
    declared_orbit = [
        value
        for value in bundle.get("valley_orbit", [])
        if isinstance(value, str) and value
    ]
    source_valley = str(bundle.get("valley") or "fixture_v0")
    orbit = declared_orbit or [source_valley]
    if source_valley not in orbit:
        source_valley = orbit[0]
    exchanged = len(orbit) > 1
    required_ids = (2, 5) if exchanged else (2,)
    representations = {2: np.eye(2, dtype=np.complex128)}
    mappings = {2: {valley: valley for valley in orbit}}
    if exchanged:
        representations[5] = np.array(
            [[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128
        )
        mappings[5] = {
            orbit[0]: orbit[1],
            orbit[1]: orbit[0],
        }
    projectors = {
        valley: np.diag(
            [1.0, 0.0] if index == 0 else [0.0, 1.0]
        )
        for index, valley in enumerate(orbit[:2])
    }
    bases = {
        valley: np.array(
            [[1.0], [0.0]]
            if index == 0
            else [[0.0], [1.0]],
            dtype=np.complex128,
        )
        for index, valley in enumerate(orbit[:2])
    }
    antiunitary = None
    if tr_completed:
        target = orbit[1] if exchanged else source_valley
        antiunitary = {
            "source_valley": source_valley,
            "target_valley": target,
            "forward_sewing_matrix": np.array([[1.0]]),
            "reverse_sewing_matrix": np.array([[-1.0]]),
            "expected_square_sign": -1,
        }
    raw_inputs: dict[str, object] = {
        "source_basis_record": source,
        "lift_record": lift,
        "lift_validation_inputs": lift_inputs,
        "extracted_wavefunction_payload_identity": (
            source["extracted_wavefunction_payload_identity"]
        ),
        "kpoint_label": kpoint,
        "kpoint_frac": np.zeros(3),
        "scope_kind": "tr_completed" if tr_completed else "local_irrep",
        "source_valleys": (source_valley,),
        "valley_orbit": tuple(orbit),
        "required_operation_ids": required_ids,
        "representations": representations,
        "plane_wave_evidence": {
            operation_id: {
                "mapping_miss_count": 0,
                "norm_preservation_residual": 0.0,
            }
            for operation_id in required_ids
        },
        "target_coefficients": np.eye(2, dtype=np.complex128),
        "projectors": projectors,
        "valley_bases": bases,
        "valley_mappings": mappings,
        "antiunitary_evidence": antiunitary,
    }
    return (
        build_scoped_representation_evidence(**raw_inputs).to_record(),
        raw_inputs,
    )


def cprime_summary_for_export(
    export_bundle: dict,
    *,
    target_kpoints: list[str] | None = None,
    iband: list[int] | None = None,
) -> dict:
    """Build the compact public C-prime view consumed by ingestion tests."""
    attach_cprime_fixture_contract(export_bundle)
    source_identity = _cprime_fixture_source_record()[
        "certificate_identity"
    ]
    matrix: list[dict[str, object]] = []
    for bundle in export_bundle.get("bundles", []):
        if not isinstance(bundle, dict):
            continue
        valley = str(bundle.get("valley", ""))
        for kpoint, links in bundle.get(
            "cprime_identity_by_kpoint", {}
        ).items():
            matrix.append({
                "kpoint": str(kpoint),
                "valley": valley,
                "double_space_group_lift_status": "passed",
                "double_space_group_lift_identity": links[
                    "double_space_group_lift_certificate_identity"
                ],
                "scoped_representation_status": "passed",
                "scoped_representation_evidence_identity": links[
                    "scoped_representation_evidence_identity"
                ],
            })
    return {
        "schema_version": "2.0.0",
        "target_kpoints": list(target_kpoints or []),
        "iband": list(iband or []),
        "input": {},
        "cprime": {
            "spinor_source_basis": {
                "status": "passed",
                "identity": source_identity,
                "blockers": [],
            },
            "acceptance_matrix": matrix,
        },
    }


def _detected_standard_operations(hall_number: int):
    """Complete detected-operation set from spglib standard operations."""
    import spglib
    sym = spglib.get_symmetry_from_database(int(hall_number))
    if sym is None:
        return None
    rots = sym["rotations"]
    trans = sym["translations"]
    return [{
        "operation_id": i,
        "rotation_frac": np.asarray(rots[i], dtype=float).tolist(),
        "translation_frac": np.asarray(trans[i], dtype=float).tolist(),
    } for i in range(len(rots))]


def real_primitive_certificate_dict(sg_number: int, sg_symbol: str,
                                    *, spinor: bool = False) -> dict | None:
    """Raw resolver certificate for a primitive SG via a real table + real
    spglib affine operations, with no post-mutation."""
    setting = _derive_table_standard_setting(int(sg_number))
    if setting is None or setting["centering_type"] != "P":
        return None
    from valleyscope.irreps.tables import load_standard_irrep_table
    try:
        table = load_standard_irrep_table(int(sg_number), spinor=bool(spinor))
    except Exception:
        return None
    detected = _detected_standard_operations(setting["hall_number"])
    if not detected:
        return None
    standard_match = {
        "number": int(sg_number),
        "international_short": sg_symbol,
        "hall_number": setting["hall_number"],
        "hall_symbol": setting["hall_symbol"],
        "operation_ids": [op["operation_id"] for op in detected],
    }
    _, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=table,
        standard_match=standard_match,
        detected_operations=detected,
    )
    if blocker is not None:
        return None
    cert = prov.get("standard_setting_certificate")
    return dict(cert) if isinstance(cert, dict) else None


def real_primitive_certificate_identity(sg_number: int, sg_symbol: str,
                                        *, spinor: bool = False) -> dict | None:
    """Serialized certificate identity from a real affine primitive cert."""
    cert = real_primitive_certificate_dict(sg_number, sg_symbol, spinor=spinor)
    if cert is None:
        return None
    candidate = {
        "subspace_space_group": {
            "candidate_space_group_number": int(sg_number),
            "candidate_space_group_symbol": sg_symbol,
        },
        "irrep_source_provenance": {
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": cert,
            },
        },
    }
    return _certificate_identity([candidate])


def add_real_certificate_to_candidates(ebr_input_candidates: dict,
                                       sg_number: int, sg_symbol: str,
                                       *, spinor: bool = False) -> dict:
    """Inject a REAL affine primitive certificate + spin into each input
    candidate so the workflow serializes a validated bundle identity naturally
    (no bundle injection)."""
    cert = real_primitive_certificate_dict(sg_number, sg_symbol, spinor=spinor)
    if cert is None:
        return ebr_input_candidates
    for c in ebr_input_candidates.get("candidates", []):
        if not isinstance(c, dict):
            continue
        prov = c.get("irrep_source_provenance")
        if not isinstance(prov, dict):
            prov = {}
            c["irrep_source_provenance"] = prov
        classification = c.get("projected_hsp_classification")
        source_hsp = (
            classification.get("source_hsp_label")
            if isinstance(classification, dict) else None
        )
        if not isinstance(source_hsp, str) or not source_hsp:
            source_hsp = c.get("kpoint")
        if isinstance(source_hsp, str) and source_hsp:
            if not isinstance(
                prov.get("source_hsp_label"), str
            ) or not prov.get("source_hsp_label"):
                prov["source_hsp_label"] = source_hsp
        c.setdefault(
            "source",
            "fixture/"
            f"{c.get('valley', 'unknown')}/{source_hsp or 'unknown'}",
        )
        kmap = prov.get("standard_setting_hsp_mapping")
        if not isinstance(kmap, dict):
            kmap = {}
            prov["standard_setting_hsp_mapping"] = kmap
        kmap["standard_setting_certificate"] = dict(cert)
        prov["source_table_spinor"] = bool(spinor)
        prov["cprime"] = {
            "spinor_source_basis_certificate_identity": canonical_identity({
                "fixture": "spinor_source_basis",
                "profile": "vasp_nonmagnetic_soc_default_saxis_v1",
            }),
            "double_space_group_lift_certificate_identity": (
                canonical_identity({
                    "fixture": "double_space_group_lift",
                    "kpoint": str(c.get("kpoint", "")),
                })
            ),
            "scoped_representation_evidence_identity": (
                canonical_identity({
                    "fixture": "scoped_representation_evidence",
                    "kpoint": str(c.get("kpoint", "")),
                    "valley": str(c.get("valley", "")),
                })
            ),
        }
    return ebr_input_candidates


def attach_cprime_fixture_to_candidates(
    ebr_input_candidates: dict,
) -> dict:
    """Attach producer-like C-prime identity links without changing setting."""
    for candidate in ebr_input_candidates.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        provenance = candidate.get("irrep_source_provenance")
        if not isinstance(provenance, dict):
            provenance = {}
            candidate["irrep_source_provenance"] = provenance
        provenance["cprime"] = {
            "spinor_source_basis_certificate_identity": canonical_identity({
                "fixture": "spinor_source_basis",
                "profile": "vasp_nonmagnetic_soc_default_saxis_v1",
            }),
            "double_space_group_lift_certificate_identity": (
                canonical_identity({
                    "fixture": "double_space_group_lift",
                    "kpoint": str(candidate.get("kpoint", "")),
                })
            ),
            "scoped_representation_evidence_identity": (
                canonical_identity({
                    "fixture": "scoped_representation_evidence",
                    "kpoint": str(candidate.get("kpoint", "")),
                    "valley": str(candidate.get("valley", "")),
                })
            ),
        }
    return ebr_input_candidates


def attach_real_certificate(export_bundle: dict, table: dict) -> dict | None:
    """Attach a REAL affine primitive certificate + spin to each bundle for an
    explicitly-labelled validated-affine mapping-contract fixture.

    The table must already declare its ``space_group_number``; the standard
    setting is NOT copied from the bundle (the validator re-derives it via
    spglib).  Returns None when the SG is not a unique primitive setting."""
    prov = table.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
        table["provenance"] = prov
    sg = prov.get("space_group_number")
    if not (isinstance(sg, int) and not isinstance(sg, bool) and sg > 0):
        return None
    prov.setdefault("data_source", "irreptables")
    prov.setdefault("package", "irreptables")
    prov.setdefault("package_version", "0.0.test")
    prov.setdefault("valleyscope_reduction", "sampled_hsp_valley_preserving")
    spinful = prov.get("spinful")
    if not isinstance(spinful, bool):
        spinful = False
        prov["spinful"] = spinful
    symbol = str(table.get("subspace_group_candidate", ""))
    cert_identity = real_primitive_certificate_identity(
        int(sg), symbol, spinor=spinful)
    if cert_identity is None:
        return None
    for b in export_bundle.get("bundles", []):
        if isinstance(b, dict):
            b["certificate_identity"] = dict(cert_identity)
            b["subspace_sg_number"] = int(sg)
            b.setdefault(
                "source_instance_id",
                f"fixture_instance_{b.get('bundle_id', 'unknown')}",
            )
            problem_kind = b.setdefault(
                "problem_kind", "unitary_valley_reduced_ebr"
            )
            if problem_kind == "unitary_valley_reduced_ebr":
                _inject_direct_unitary_contract(
                    b,
                    spinful=spinful,
                    certificate_identity=cert_identity,
                )
            elif problem_kind == "valley_orbit_reduced_ebr":
                b.setdefault(
                    "physical_object_kind",
                    "joint_time_reversal_valley_orbit",
                )
    attach_cprime_fixture_contract(export_bundle)
    return export_bundle


def complete_table_provenance(table: dict, sg_number: int | None = None,
                              spinful: bool | None = None) -> dict:
    """Fill the conventional irreptables provenance constants on a table.

    ``sg_number`` / ``spinful`` are declared explicitly by the caller (no
    fabricated symbol->SG lookup).  Certificates and crystallographic setting
    evidence are untouched (the validator derives setting from spglib)."""
    prov = table.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
        table["provenance"] = prov
    if sg_number is not None:
        prov["space_group_number"] = int(sg_number)
    prov.setdefault("data_source", "irreptables")
    prov.setdefault("package", "irreptables")
    prov.setdefault("package_version", "0.0.test")
    prov.setdefault("valleyscope_reduction", "sampled_hsp_valley_preserving")
    if spinful is not None:
        prov["spinful"] = bool(spinful)
    elif not isinstance(prov.get("spinful"), bool):
        prov["spinful"] = False
    return table


def _inject_direct_unitary_contract(
    bundle: dict,
    *,
    spinful: bool,
    certificate_identity: dict,
) -> None:
    """Upgrade a synthetic unitary fixture to the current producer contract."""
    irreps = bundle.get("irreps_by_kpoint", {})
    if not isinstance(irreps, dict):
        return
    original_records = bundle.get("irrep_records_by_kpoint")
    if not isinstance(original_records, dict):
        original_records = {}
    valley = bundle.get("valley")
    source_to_sample: dict[str, str] = {}
    rebuilt_records: dict[str, list[dict]] = {}
    for sampled, labels in irreps.items():
        existing = original_records.get(sampled)
        existing = existing if isinstance(existing, list) else []
        first = next(
            (row for row in existing if isinstance(row, dict)),
            {},
        )
        first_provenance = first.get("irrep_source_provenance")
        source_hsp = (
            first_provenance.get("source_hsp_label")
            if isinstance(first_provenance, dict)
            else None
        )
        if (
            not isinstance(source_hsp, str)
            or not source_hsp
            or source_hsp in source_to_sample
        ):
            source_hsp = sampled
        source_to_sample[source_hsp] = sampled
        rows: list[dict] = []
        for irrep, multiplicity in Counter(labels).items():
            template = next(
                (
                    row for row in existing
                    if isinstance(row, dict)
                    and row.get("matched_irrep") == irrep
                ),
                first,
            )
            row = deepcopy(template)
            source = row.get("source")
            if not isinstance(source, str) or not source:
                source = f"fixture/{valley}/{source_hsp}/{irrep}"
            irrep_provenance = row.get("irrep_source_provenance")
            if not isinstance(irrep_provenance, dict):
                irrep_provenance = {}
            irrep_provenance = dict(irrep_provenance)
            irrep_provenance["source_hsp_label"] = source_hsp
            irrep_provenance["source_table_spinor"] = spinful
            identity = {
                "source": source,
                "workflow_path": "direct_qcut",
                "valley": valley,
                "source_hsp_label": source_hsp,
                "sampled_kpoint": sampled,
                "irrep": irrep,
                "multiplicity": multiplicity,
            }
            row.update({
                "matched_irrep": irrep,
                "irrep_multiplicity": multiplicity,
                "valley": valley,
                "sampled_kpoint": sampled,
                "source_hsp_label": source_hsp,
                "workflow_path": "direct_qcut",
                "readiness_level": "trusted",
                "source": source,
                "certificate_identity": dict(certificate_identity),
                "irrep_source_provenance": irrep_provenance,
                "source_candidate_identity": identity,
                "source_candidate_provenance": {
                    "source": source,
                    "workflow_path": "direct_qcut",
                    "irrep_source_provenance": irrep_provenance,
                },
            })
            rows.append(row)
        rebuilt_records[sampled] = rows

    bundle.update({
        "physical_object_kind": "unitary_valley_projected_subspace",
        "spinor": spinful,
        "workflow_path": "direct_qcut",
        "readiness_level": "trusted",
        "valley_orbit": [],
        "time_reversal": {},
        "unitary_irrep_completion_records_by_hsp": {},
        "unitary_vector_construction": {
            "kind": "direct_observed_unitary_rows",
            "source": "trusted_ebr_input_candidates",
        },
        "expected_hsps": bundle.get(
            "expected_hsps", list(irreps)
        ),
        "required_source_hsp_labels": list(source_to_sample),
        "source_hsp_to_sampled_kpoint": source_to_sample,
        "irrep_records_by_kpoint": rebuilt_records,
    })
