# String diagrams

TikZ sources for the README figures. Shared styles: `fritzstrings.sty` (original implementation, drawing conventions referenced from T. Fritz, arXiv:1908.07021). Those diagrams read top to bottom.

`conditional_independence.tex` is a separate LuaLaTeX figure (Japanese labels, read bottom to top). `build.sh` skips it. The published SVG and PNG are `docs/_static/figures/conditional_independence.svg` and `docs/_static/figures/conditional_independence.png`; a copy of the SVG also sits next to the source.

`exchangeability.tex` and `kolmogorov_consistency.tex` use `fritzstrings.sty` and are built by `build.sh`. Both are read top to bottom. `exchangeability.tex` is `p ; (id^{⊗(k-1)} ⊗ swap(X,X) ⊗ id^{⊗(n-k-1)}) = p`. `kolmogorov_consistency.tex` is `p_n ; (id^{⊗(n-1)} ⊗ !) = p_{n-1}`. `named_marginal.tex` is `p ; (! ⊗ id_B ⊗ !) = p_B`. `named_permute.tex` reorders wires `A, B, C` into `C, A, B`. SVG and PNG copies are in `docs/_static/figures/`.

Regenerate (requires TeX Live with tikz/standalone and dvisvgm):

```sh
./docs/diagrams/build.sh            # .tex -> .svg (dvisvgm --no-fonts)
PREVIEW_DIR=/tmp/png ./docs/diagrams/build.sh   # also PNG previews (rsvg-convert)
```
