Installation
============

The library needs Python 3.8 or newer and depends only on NumPy.

.. code-block:: console

   pip install fxtensor-salmon

The DisCoPy bridge is optional and needs Python 3.10 or newer:

.. code-block:: console

   pip install "fxtensor-salmon[discopy]"

Import the class from the top-level package:

.. code-block:: python

   from fxtensor_salmon import FXTensor

Building these pages
--------------------

The documentation is not part of the installed wheel. Clone the repository
and, on Python 3.10 or newer, install the ``docs`` extra. That extra installs
Sphinx, Furo, and (on Python 3.10+) DisCoPy, which autodoc imports while it
reads ``fxtensor_salmon.discopy``.

.. code-block:: console

   pip install -e ".[docs]"
   python docs/check_examples.py
   sphinx-build -W --keep-going -b html docs/en docs/_build/html/en
   sphinx-build -W --keep-going -b html docs/ja docs/_build/html/ja
   cp docs/index.html docs/_build/html/index.html

``docs/build.sh`` runs the example check and both Sphinx builds, then copies
the site root. The same third-party packages are listed in
``docs/requirements.txt``.

``sphinx-build -W`` treats warnings as errors. The example scripts under
``docs/examples/`` are the code blocks in the guide; ``docs/check_examples.py``
executes them, and ``pytest`` runs the same scripts (the DisCoPy example is
skipped when DisCoPy is not installed).

Published HTML is at https://salmonsnare.github.io/fxtensor-salmon/
(English under ``en/``, Japanese under ``ja/``). GitHub Pages has to use
**GitHub Actions** as its source; the workflow cannot switch that setting.
