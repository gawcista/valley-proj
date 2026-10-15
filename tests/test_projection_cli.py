"""Exercise CLI selection against real generated wavefunctions."""
from valleyscope.cli import main
from tests.projection_acceptance import make_projection_inputs, verify_projection_result


def test_project_bands_cli_preserves_original_bands_and_absolute_weights(tmp_path):
    config = make_projection_inputs(tmp_path)
    assert main(["project-bands", str(config)]) == 0
    verify_projection_result(tmp_path / "output/valley_projection.json")
