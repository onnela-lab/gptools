import cmdstanpy
from cmdstanpy.model import OptionalPath
import functools as ft
import os
from typing import Any, Optional, Union


def get_include() -> str:
    """
    Get the include directory for the graph Gaussian process library.
    """
    return os.path.dirname(__file__)


@ft.wraps(cmdstanpy.CmdStanModel)
def compile_model(
        model_name: Optional[str] = None, stan_file: OptionalPath = None,
        exe_file: OptionalPath = None, compile: Union[bool, str] = True,
        stanc_options: Optional[dict[str, Any]] = None,
        cpp_options: Optional[dict[str, Any]] = None,
        user_header: OptionalPath = None, **kwargs) -> cmdstanpy.CmdStanModel:
    # Add gptools includes by default.
    stanc_options = stanc_options or {}
    stanc_options.setdefault("include-paths", []).append(get_include())
    return cmdstanpy.CmdStanModel(
        model_name=model_name, stan_file=stan_file, exe_file=exe_file, compile=compile,
        stanc_options=stanc_options, cpp_options=cpp_options, user_header=user_header, **kwargs,
    )


if __name__ == "__main__":
    print(get_include())
