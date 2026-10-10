// One-page sheet. Every number is read from data.json.
#let data = json("data.json")
#set page(paper: "a4", margin: 14mm)
#set text(size: 10pt)

#align(center)[
  #text(size: 16pt, weight: "bold")[#data.title]
  #if data.draft [ #text(fill: rgb("#a33"))[draft] ]
]

#data.prose.summary

#text(size: 9pt)[
  Curvature #data.kx_tick, #data.ky_tick.
  #data.version.b3_core · #data.version.generated_utc
]

#{
  let cells = ()
  for (key, value) in data.properties.pairs() {
    cells.push([#key])
    cells.push([#str(value)])
  }
  if cells.len() > 0 {
    table(
      columns: (1fr, 1fr),
      table.header([Quantity], [Value]),
      ..cells,
    )
  }
}

#if data.footnote != "" [
  #text(size: 8pt, fill: rgb("#444"))[#data.footnote]
]
