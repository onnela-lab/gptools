import cmdstanpy
from cmdstanpy.model import OptionalPath
import logging
import os
from typing import Any, Dict, Optional, Type, Union


def get_include() -> str:
    """
    Get the include directory for the library.
    """
    version = cmdstanpy.cmdstan_version()
    if version and version < (2, 36):  # pragma: no cover
        logging.warning(
            "cmdstan<2.36.0 had a bug in the evaluation of Matern 3/2 kernels "
            "with different length scales for different dimensions; see "
            "https://github.com/stan-dev/math/pull/3084 for details. Your "
            "model may yield unexpected results or crash if you use "
            "nearest-neighbor Gaussian processes with Matern 3/2 kernels. Your "
            "cmdstan version is %d.%d; consider upgrading.",
            *version,
        )
    return os.path.dirname(__file__)


def set_cmdstanpy_log_level(level: Optional[Union[str, int]] = None) -> None:
    """
    Set the log level of cmdstanpy.

    Args:
        level: Log level (defaults to `WARNING`).
    """
    cmdstanpy_logger = cmdstanpy.utils.get_logger()
    for handler in cmdstanpy_logger.handlers:
        handler.setLevel(level or logging.WARNING)


def compile_model(
    model_name: Optional[str] = None,
    stan_file: OptionalPath = None,
    exe_file: OptionalPath = None,
    compile: Union[bool, str] = True,
    stanc_options: Optional[Dict[str, Any]] = None,
    cpp_options: Optional[Dict[str, Any]] = None,
    user_header: OptionalPath = None,
    cls: Optional[Type] = None,
    **kwargs
) -> cmdstanpy.CmdStanModel:
    """
    Create a :class:`cmdstanpy.CmdStanModel` and configure include paths for gptools.

    Args:
        model_name: Model name used for output file names (defaults to base name of
            :code:`stan_file`).
        stan_file: Path to Stan model file.
        exe_file: Path to compiled executable file.  Optional unless :code:`stan_file`
            is not given. If both :code:`stan_file` and :code:`exe_file` are given, the
            base names must match.
        compile: Whether or not to compile the model (defaults to :code:`True`). If
            :code:`"force"`, the model will compile even if a newer executable is found.
        stanc_options: Stanc3 compiler options (see
            `here <https://mc-stan.org/docs/stan-users-guide/stanc-args.html>`__ for
            details).
        cpp_options: C++ compiler options.
        user_header: Path to a C++ header file to include during compilation.
        cls: Subclass of :class:`cmdstanpy.CmdStanModel` if a custom implementation
            should be used.
        **kwargs: Keyword arguments passed to the :class:`cmdstanpy.CmdStanModel`
            constructor.

    Returns:
        model: Model instance with include paths configured for gptools.
    """
    # Add gptools includes by default.
    stanc_options = stanc_options or {}
    stanc_options.setdefault("include-paths", []).append(get_include())
    cls = cls or cmdstanpy.CmdStanModel
    return cls(
        model_name=model_name,
        stan_file=stan_file,
        exe_file=exe_file,
        compile=compile,
        stanc_options=stanc_options,
        cpp_options=cpp_options,
        user_header=user_header,
        **kwargs,
    )


# Reduce logging verbosity if we are compiling documentation on readthedocs. We cannot
# easily achieve the same effect in the sphinx `conf.py`` because each example runs in
# its own kernel.
if "READTHEDOCS" in os.environ:  # pragma: no cover
    set_cmdstanpy_log_level("ERROR")
