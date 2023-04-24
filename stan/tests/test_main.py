import subprocess


def test_main_include_path() -> None:
    output = subprocess.check_output(["python", "-m", "gptools.stan"], text=True)
    assert output.startswith("The include path is")
