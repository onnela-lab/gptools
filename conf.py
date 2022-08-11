master_doc = "README"
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]
project = "TEMPLATE_NAME"
napoleon_custom_sections = [("Returns", "params_style")]
plot_formats = [
    ("png", 144),
]
html_theme = "nature"

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
