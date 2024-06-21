import os
import sys
import pydata_sphinx_theme
#sys.path.insert(0, os.path.abspath('../../ivers'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(1, os.path.abspath('../../test'))

# -- Project information -----------------------------------------------------

project = 'ivers'
author = 'Philip Ivers Ohlsson'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = '.rst'
master_doc = 'index'

language = 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'


html_static_path = ['_static']

# -- Options for HTMLHelp output ---------------------------------------------


latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'project_name.tex', 'project\\_name Documentation',
     'author_name', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'project_name', 'project_name Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'project_name', 'project_name Documentation',
     author, 'project_name', 'One line description of project.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------

epub_title = project

epub_exclude_files = ['search.html']
