def test_valleyscope_package_and_cli_name(capsys):
    import pytest

    import valleyscope
    from valleyscope.cli import main

    assert valleyscope.__name__ == "valleyscope"
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "valleyscope" in capsys.readouterr().out
