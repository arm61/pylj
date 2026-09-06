"""Sphinx configuration for the pylj documentation."""

from pylj import __version__

project = "pylj"
copyright = "2018, Andrew R. McCluskey"
author = "Andrew R. McCluskey"
version = __version__
release = version
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
master_doc = "index"

# Chapters are executed when the site is built. "cache" re-runs a chapter
# only when its source changes, so after changing pylj itself run `make clean`
# first; an error in any cell fails the build.
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_merge_streams = True
myst_enable_extensions = ["dollarmath", "colon_fence"]

autodoc_member_order = "bysource"
napoleon_google_docstring = True

html_theme = "sphinx_book_theme"
html_title = "pylj"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["my_styles.css"]
html_theme_options = {
    "repository_url": "https://github.com/arm61/pylj",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/source",
    "show_toc_level": 2,
}
