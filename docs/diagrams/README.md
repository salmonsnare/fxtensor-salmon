# String diagrams

TikZ sources for the README figures. Shared styles: `fritzstrings.sty` (original implementation, drawing conventions referenced from T. Fritz, arXiv:1908.07021). Those diagrams read top to bottom.

`conditional_independence.tex` is a separate LuaLaTeX figure (Japanese labels, read bottom to top). `build.sh` skips it. The published SVG and PNG are `docs/_static/figures/conditional_independence.svg` and `docs/_static/figures/conditional_independence.png`; a copy of the SVG also sits next to the source.

Regenerate (requires TeX Live with tikz/standalone and dvisvgm):

```sh
./docs/diagrams/build.sh            # .tex -> .svg (dvisvgm --no-fonts)
PREVIEW_DIR=/tmp/png ./docs/diagrams/build.sh   # also PNG previews (rsvg-convert)
```
