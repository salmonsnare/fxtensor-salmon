Contributing figures
====================

When an issue is resolved, add a corresponding TikZ figure to the docs.

Draw it with the sources in ``docs/diagrams/`` (shared style
``fritzstrings.sty``). Rebuild the SVG with ``docs/diagrams/build.sh``, then
copy that SVG into ``docs/_static/figures/``. Put the figure on the relevant
page in both English and Japanese, with a caption in each language. The same
steps are written in ``docs/_static/figures/README.md``.

The eight diagrams shipped with the README are already in that folder:
composition, tensor product, the three copy laws, naturality of discard,
Bayesian inversion, and almost-sure equality.
