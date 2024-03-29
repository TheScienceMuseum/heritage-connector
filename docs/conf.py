# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

master_doc = "index"


# -- Project information -----------------------------------------------------

project = "Heritage Connector"
copyright = "2021, Science Museum Group"
author = "Science Museum Group"

links = {
    "GitHub": "https://github.com/TheScienceMuseum/heritage-connector",
    "Project site": "https://www.sciencemuseumgroup.org.uk/project/heritage-connector/",
    "Blog": "https://thesciencemuseum.github.io/heritageconnector/",
}


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    # "sphinxcontrib.bibtex",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
]

extensions.append("autoapi.extension")
autoapi_type = "python"
# autoapi_add_toctree_entry = False
# autoapi_keep_files = True
autoapi_dirs = ["../heritageconnector"]


def autoapi_skipper(app, what, name, obj, skip, options):
    if name == "logger":
        return True
    return False


[extensions]
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["custom.css"]

html_theme_options = {
    # "logo": "header.jpg",
    # "logo_name": True,
    # "page_width": "50%",
    # "github_user": "theScienceMuseum",
    # "github_repo": "heritage-connector",
    # "github_type": "star",
    # "github_count": False,
    # "sidebar_collapse": True,
    # "extra_nav_links": links,
    # "sidebar_includehidden": True,
}


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", autoapi_skipper)
