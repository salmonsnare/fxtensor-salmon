Contributing figures
====================

When an issue is resolved, add a corresponding TikZ figure to the docs.

Draw it with the sources in ``docs/diagrams/`` (shared style
``fritzstrings.sty``). Rebuild the SVG with ``docs/diagrams/build.sh``, then
copy that SVG into ``docs/_static/figures/``. Put the figure on the relevant
page in both English and Japanese, with a caption in each language. The same
steps are written in ``docs/_static/figures/README.md``.

The diagrams shipped with the README are already in that folder:
composition, tensor product, the three copy laws, naturality of discard,
Bayesian inversion, almost-sure equality, exchangeability, and Kolmogorov
consistency.

``conditional_independence.tex`` is a LuaLaTeX figure, so
``docs/diagrams/build.sh`` skips it. Its SVG and PNG are already in
``docs/_static/figures/``. The figure is read bottom to top.
