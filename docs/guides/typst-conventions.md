# Typst conventions

Rules for tables this package emits. Typst 0.13 reads `data.json`. Python
writes the numbers. The template does not invent them.

## Table cells

A data row is a parenthesised tuple of one value per cell:

```typst
let rows = (
  ("a1", "b1", "c1"),
  ("a2", "b2", "c2"),
)
table(
  columns: 3,
  table.header([A], [B], [C]),
  ..rows.flatten(),
)
```

A row written as one content block, `([a, b, c]),` or `[a, b, c],`,
renders as a single full-width cell. The datasheet templates push one
cell at a time and then spread `..cells` for the same reason.

`@` starts a label reference. Escape it as `\@` inside a string, or
rephrase the cell. A comma at the top level of a row splits cells, so a
label that needs a comma uses a semicolon.

`str()` accepts strings and numbers. It rejects a boolean. JSON that the
template prints with `str()` stores `true` and `false` as those words.
`#if` still receives real booleans for `draft`, `accepted`, and `ok`.

## Curvature labels

Every curvature, in a table, a tick, or a note, is the signed radius:

`k = 5e-5 1/mm (R = 20 m)`

`R` in metres is `1 / (1000 * k)` with `k` in 1/mm, and it keeps the sign
of `k`. `k = 0` reads `0 (flat)`. Use `curvature_tick` in Python. Do not
hand-format a second label.

## Compile check

```bash
typst compile in.typ out.pdf
pdfinfo out.pdf | grep Pages
pdfimages -list out.pdf
pdftotext -layout out.pdf - | head -60
```

`pdfimages -list` shows one image and one soft mask per embedded figure.
A text session cannot see the pictures. Say that a person still has to
look at them.
