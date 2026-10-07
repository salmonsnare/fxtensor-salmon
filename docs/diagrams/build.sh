#!/bin/sh
# Regenerate SVGs from the TikZ sources. Requires TeX Live (latex, tikz,
# standalone) and dvisvgm. Optional: rsvg-convert for PNG previews (PREVIEW_DIR).
set -eu
cd "$(dirname "$0")"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for tex in *.tex; do
  name=${tex%.tex}
  TEXINPUTS=.: latex -interaction=nonstopmode -halt-on-error -output-directory="$tmp" "$tex" >/dev/null
  dvisvgm --no-fonts --exact-bbox -o "$name.svg" "$tmp/$name.dvi" 2>/dev/null
  echo "built $name.svg"
  if [ -n "${PREVIEW_DIR:-}" ]; then
    mkdir -p "$PREVIEW_DIR"; rsvg-convert -z 3 -o "$PREVIEW_DIR/$name.png" "$name.svg"
  fi
done
