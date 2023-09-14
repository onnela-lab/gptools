import subprocess
from gptools.stan import set_cmdstanpy_log_level


def test_main_include_path() -> None:
    output = subprocess.check_output(["python", "-m", "gptools.stan"], text=True)
    assert output.startswith("The include path is")


def test_set_cmdstanpy_log_level() -> None:
    set_cmdstanpy_log_level("INFO")
