// Full sheet. Figures, kerf convention, provenance, acceptance stamp.
// Table cells are pushed one at a time so a row cannot collapse into one cell.
#let data = json("data.json")
#set page(paper: "a4", margin: 14mm)
#set text(size: 9pt)

#align(center)[
  #text(size: 16pt, weight: "bold")[#data.title]
  #linebreak()
  #if data.draft [ #text(fill: rgb("#a33"))[DRAFT — not accepted] ] else [ #text(fill: rgb("#174"))[accepted] ]
]

== Use
#data.prose.summary

*Intended use.* #data.prose.intended_use

*Limitations.* #data.prose.limitations

#if data.prose.notes != "" [ #data.prose.notes ]

== Properties
Curvature #data.kx_tick and #data.ky_tick.

#{
  let cells = ()
  for (key, value) in data.properties.pairs() {
    cells.push([#key])
    cells.push([#str(value)])
  }
  if cells.len() > 0 {
    table(
      columns: (1fr, 2fr),
      table.header([Quantity], [Value]),
      ..cells,
    )
  }
}

== Convention
#data.kx_tick means the mould face is the shorter fibre when k is positive.
Each kerf row names the sign that opens it.

#{
  let cells = ()
  for row in data.convention {
    cells.push([#row.axis])
    cells.push([#row.mouth])
    cells.push([opens for #row.opens_for])
  }
  if cells.len() > 0 {
    table(
      columns: (auto, auto, 1fr),
      table.header([Axis], [Mouth], [Opens]),
      ..cells,
    )
  }
}

== Figures
#{
  let names = ("props_curvature", "sweep_heatmap", "rho_curvature", "kerf_ranges_wedge")
  for name in names {
    let path = data.figures.at(name, default: "")
    if path != "" and path != none {
      [#heading(level: 3)[#name] #image(path, width: 100%)]
    }
  }
}

== Provenance
#{
  let cells = ()
  for (key, value) in data.provenance.pairs() {
    cells.push([#key])
    cells.push([#str(value)])
  }
  for (key, value) in data.version.pairs() {
    cells.push([#key])
    cells.push([#str(value)])
  }
  if cells.len() > 0 {
    table(
      columns: (auto, 1fr),
      table.header([Field], [Value]),
      ..cells,
    )
  }
}

== Acceptance
#if data.accepted [ Stamp ok. ] else [ Stamp not ok. ]
#{
  let cells = ()
  for row in data.acceptance.checks {
    cells.push([#row.id])
    cells.push([#if row.ok [ok] else [fail]])
    cells.push([#str(row.value)])
  }
  if cells.len() > 0 {
    table(
      columns: (auto, auto, 1fr),
      table.header([Check], [Result], [Value]),
      ..cells,
    )
  }
}

#if data.footnote != "" [
  #text(size: 8pt)[#data.footnote]
]
