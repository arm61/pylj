# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'pylj'
author = 'Andrew R. McCluskey'
release = '1.5.2'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']
