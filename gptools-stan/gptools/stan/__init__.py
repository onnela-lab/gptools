import cmdstanpy
from cmdstanpy.model import EXTENSION, OptionalPath
import functools as ft
import os
from pathlib import Path
import re
import typing


def get_include() -> str:
    """
    Get the include directory for the graph Gaussian process library.
    """
    return os.path.dirname(__file__)


def _needs_compilation(stan_file: Path, reference: float, include_paths: list[str]) -> bool:
    """
    Determine whether the stan file needs to be compiled.

    Args:
        stan_file: File that may need compilation.
        reference: Time at which the program was last compiled.
        include_paths: List of paths to look up included files.
    """
    if stan_file.stat().st_mtime > reference:
        return True

    with open(stan_file) as fp:
        for path in re.findall(r"#include\s+(.*)", fp.read()):
            discovered = False
            for include_path in include_paths:
                qualified_path = Path(include_path) / path
                if not qualified_path.is_file():
                    continue
                discovered = True
                if _needs_compilation(qualified_path, reference, include_paths):
                    return True
            if not discovered:
                raise FileNotFoundError(path)


@ft.wraps(cmdstanpy.CmdStanModel)
def compile_model(
        model_name: typing.Optional[str] = None, stan_file: OptionalPath = None,
        exe_file: OptionalPath = None, compile: typing.Union[bool, str] = True,
        stanc_options: typing.Optional[dict[str, typing.Any]] = None,
        cpp_options: typing.Optional[dict[str, typing.Any]] = None,
        user_header: OptionalPath = None) -> cmdstanpy.CmdStanModel:
    # Add gptools includes by default.
    stanc_options = stanc_options or {}
    stanc_options.setdefault("include-paths", []).append(get_include())
    # Determine whether we need to compile the model.
    if compile is True:
        stan_file = Path(stan_file).expanduser().absolute()
        _exe_file = Path(exe_file).expanduser().absolute() if exe_file else \
            stan_file.with_suffix(EXTENSION)
        include_paths = [stan_file.parent] + stanc_options["include-paths"]
        if _exe_file.is_file() and _needs_compilation(stan_file, _exe_file.stat().st_mtime,
                                                      include_paths):
            compile = "force"
    return cmdstanpy.CmdStanModel(model_name, stan_file, exe_file, compile, stanc_options,
                                  cpp_options, user_header)


if __name__ == "__main__":
    print(get_include())
