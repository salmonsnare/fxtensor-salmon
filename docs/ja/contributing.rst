図の追加
========

Issue が解決したら、対応する TikZ の図をドキュメントに追加します。

図は ``docs/diagrams/`` のソース（共通スタイル ``fritzstrings.sty``）で描きます。
``docs/diagrams/build.sh`` で SVG を作り、``docs/_static/figures/`` にコピーします。
英語と日本語の該当ページに、それぞれの言語のキャプションを付けて置きます。
同じ手順は ``docs/_static/figures/README.md`` にもあります。

README にある図（合成、テンソル積、copy の3つの法則、discard の自然性、
ベイズ反転、ほとんど確実な等価、交換可能性、Kolmogorov 整合性）は、
すでにそのフォルダに入っています。

``conditional_independence.tex`` は LuaLaTeX の図なので、
``docs/diagrams/build.sh`` はこれをスキップします。SVG と PNG は
``docs/_static/figures/`` にあります。この図は下から上に読みます。
