import cmdstanpy
from docutils import nodes
from docutils.statemachine import ViewList
import logging
import pathlib
import re
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles


master_doc = "README"
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]
project = "gptools"
napoleon_custom_sections = [("Returns", "params_style")]
plot_formats = [
    ("png", 144),
]
html_theme = "sphinx_rtd_theme"
html_sidebars = {}
exclude_patterns = ["playground"]

# Configure autodoc to avoid excessively long fully-qualified names.
add_module_names = False
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


cmdstanpy_logger = cmdstanpy.utils.get_logger()
for handler in cmdstanpy_logger.handlers:
    handler.setLevel(logging.WARNING)


class StanDocDirective(SphinxDirective):
    """
    Generate documentation for stan source files.
    """
    required_arguments = 1

    def run(self):
        # Load the stan file.
        stan_file, = self.arguments
        with open(pathlib.Path(self.state.document.current_source).parent / stan_file) as fp:
            text = fp.read()

        # Parse the function descriptions.
        state = "inactive"
        current = [""]
        lines = []
        for line in text.splitlines():
            if line.startswith("/**"):
                state = "active"
            elif line.startswith("*/"):
                state = "signature"
            elif state == "active":
                current.append(f"    {line}")
            elif state == "signature":
                line = re.sub(r"\s*{\s*$", "", line)
                current.insert(0, f".. cpp:function:: {line}")
                current.append("")
                lines.extend(current)
                current = [""]
                state = "inactive"

        # Render the content.
        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, ViewList(lines), node)
        return node.children


def setup(app: Sphinx) -> None:
    app.add_directive("standoc", StanDocDirective)
