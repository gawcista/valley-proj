"""Report-boundary tests; supplied residues are not physical certification."""

import json

import pytest
import yaml

from tests.helpers_io_workflow import write_config
from valleyscope.io.config import load_config
from valleyscope.reports.analysis_outputs import write_analysis_outputs


def _write_summary(tmp_path, profile, **kwargs):
    config_path = tmp_path / "config.yaml"
    write_config(config_path, tmp_path / "unused.h5", tmp_path / "out")
    raw = yaml.safe_load(config_path.read_text())
    raw["output"]["profile"] = profile
    config_path.write_text(yaml.safe_dump(raw))
    outputs = write_analysis_outputs(
        config=load_config(config_path), qcut=0.5,
        weight_rows=[], sector_names=[], subspace_payload={"kpoints": {}},
        symmetry_payload={"status": "skipped", "detected_operations": []},
        symmetry_rows=[], projectors_by_kpoint={}, qcut_scan_payload={},
        symmetry_representation_payload={}, basis_transforms={},
        valley_irrep_matching={"generic_matches_by_kpoint": {}},
        reduced_ebr_mapping={"status": "blocked", "solutions": [],
                             "excluded_bundles": [{"reason": "existing EBR blocker"}]},
        **kwargs,
    )
    return json.loads(outputs["valley_summary_json"].read_text()), outputs["summary_text"]


@pytest.mark.parametrize("profile", ["standard", "debug"])
def test_absent_chern_report_is_not_evaluated_without_a_number(tmp_path, profile):
    summary, text = _write_summary(tmp_path, profile)
    report = summary["valley_chern_mod"]
    assert summary["schema_version"] == "2.2.0"
    assert report["status"] == "not_evaluated"
    assert report["global_valley_subspace_status"] == "not_evaluated"
    assert report["rows"] == []
    assert "conditional on a globally defined valley subspace" in report["interpretation"]
    assert "Valley Chern residues from rotation eigenvalues" in text
    assert "C = " not in text


@pytest.mark.parametrize("profile", ["standard", "debug"])
@pytest.mark.parametrize("status,residue,reasons", [
    ("conditional", 2, []),
    ("blocked", None, ["missing rotation-invariant momentum"]),
    ("not_applicable", None, ["rotation changes valley"]),
])
def test_chern_report_is_preserved_and_rendered_without_promoting_ebr(
    tmp_path, profile, status, residue, reasons,
):
    report = {
        "status": status,
        "global_valley_subspace_status": "not_evaluated",
        "interpretation": "Total-subspace residue, not an integer Chern number.",
        "rows": [{
            "valley": "K_valley", "modulus": 3, "residue": residue,
            "status": status, "subspace_rank": 2, "rotation_operation_id": 4,
            "blocking_reasons": reasons,
            "eigenvalue_evidence": [{"kpoint": "GM", "operation_id": 4}],
        }],
    }
    summary, text = _write_summary(tmp_path, profile, valley_chern_mod=report)
    assert summary["valley_chern_mod"] == report
    assert "Total-subspace residue, not an integer Chern number." in text
    assert "K_valley" in text
    if status == "conditional":
        assert "C = 2 (mod 3)" in text
        assert "conditional on a globally defined valley subspace" in text
    else:
        assert "C = " not in text
        assert reasons[0] in text
    irrep_title = ("Valley-resolved irreps" if profile == "debug" else
                   "Valley-projected subspace space group and trusted HSP irreps")
    ebr_title = ("Reduced EBR mapping" if profile == "debug" else
                 "Authoritative reduced EBR results")
    assert text.index(irrep_title) < text.index(
        "Valley Chern residues from rotation eigenvalues"
    ) < text.index(ebr_title)
    if profile == "standard":
        assert summary["reduced_ebr_summary"]["mapping_status"] == "blocked"
        assert "existing EBR blocker" in str(summary["readiness_blocker_summary"])
    else:
        assert summary["valley_reduced_ebr_mapping"]["status"] == "blocked"
        assert summary["valley_reduced_ebr_mapping"]["excluded_bundles"] == [
            {"reason": "existing EBR blocker"}
        ]
