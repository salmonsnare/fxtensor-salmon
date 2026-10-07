# String diagrams

TikZ sources for the README figures. Shared styles: `fritzstrings.sty` (after Fritz, arXiv:1908.07021).

Regenerate (requires TeX Live with tikz/standalone and dvisvgm):

```sh
./docs/diagrams/build.sh            # .tex -> .svg (dvisvgm --no-fonts)
PREVIEW_DIR=/tmp/png ./docs/diagrams/build.sh   # also PNG previews (rsvg-convert)
```
