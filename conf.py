import cmdstanpy
from sphinx.application import Sphinx

project = "gptools"
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.stan",
]

napoleon_custom_sections = [("Returns", "params_style")]
plot_formats = [
    ("png", 144),
]
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/onnela-lab/gptools",
    "use_repository_button": True,
}
exclude_patterns = [
    ".pytest_cache",
    "**.ipynb_checkpoints",
    "**/cran-comments.md",
    "**/LICENSE.md",
    "**/NEWS.md",
    "**/README.md",
    "docs/_build",
    "docs/jupyter_execute",
    "figures",
    "playground",
    "README.md",
    "venv",
]

# Configure autodoc to avoid excessively long fully-qualified names.
add_module_names = False
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "cmdstanpy": (
        f"https://cmdstanpy.readthedocs.io/en/v{cmdstanpy.__version__}",
        None,
    ),
}

source_suffix = {
    ".rst": "restructuredtext",
}

nb_execution_mode = "cache"
nb_execution_timeout = 300  # Some examples take a while to compile and sample.
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]
myst_dmath_double_inline = True

mathjax3_config = {
    "tex": {
        "macros": {
            "BigO": r"\mathcal{O}",
            "braces": [r"\left\{#1\right\}", 1],
            "cov": r"\operatorname{cov}",
            "dist": r"\sim",
            "E": [r"\mathbb{E}\parenth{#1}", 1],
            "imag": r"\mathrm{i}",
            "mat": [r"\mathbf{#1}", 1],
            "parenth": [r"\left(#1\right)", 1],
            "pred": r"\mathcal{P}",
            "proba": [r"p\parenth{#1}", 1],
        }
    }
}


def setup(app: Sphinx) -> None:
    # Ignore .ipynb and .html files (cf.
    # https://github.com/executablebooks/MyST-NB/issues/363).
    app.registry.source_suffix.pop(".ipynb", None)
    app.registry.source_suffix.pop(".html", None)
