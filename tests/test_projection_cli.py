"""Exercise CLI selection against real generated wavefunctions."""
from valleyscope.cli import main
from tests.projection_acceptance import (
    make_projection_inputs, run_installed_projection_acceptance, verify_projection_result,
)


def test_project_bands_cli_preserves_original_bands_and_absolute_weights(tmp_path):
    config = make_projection_inputs(tmp_path)
    assert main(["project-bands", str(config)]) == 0
    verify_projection_result(tmp_path / "output/valley_projection.json")


def test_projection_acceptance_runs_in_current_source_import_context():
    """The numerical acceptance helper also runs before source installation.

    The release gate verifies installation and console provenance separately;
    this helper must exercise the production CLI imported by this interpreter.
    """
    assert run_installed_projection_acceptance() == {
        "rows": 4, "valleys": 3, "raw_mixed_weights": [0.1, 0.6, 0.2],
        "residual_weight": 0.1, "validation_errors": [],
    }
