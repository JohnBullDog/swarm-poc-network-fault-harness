# presentation

A Quarto reveal.js deck for a research audience: why this work matters, what
was measured, and what the results do and do not support.

Every figure and every number is computed from the published CSVs in
`../datasets/` **at render time**. Nothing is typed in, so the deck cannot
drift from the data.

## Open it

```
presentation/swarm_poc.qmd
```

Open in RStudio and click **Render**, or from a shell:

```bash
quarto render swarm_poc.qmd            # -> swarm_poc.html
```

RStudio ships Quarto, so no separate install is needed. Requires R with
`readr`, `dplyr`, `tidyr`, `ggplot2` and `scales`.

## Files

| file | what |
|---|---|
| `swarm_poc.qmd` | the deck: narrative, R chunks, takeaways |
| `_setup.R` | palette, ggplot theme, data loading, the delay-sweep regression |
| `custom.scss` | slide styling, matched to the figures' palette |
| `swarm_poc.html` | rendered output (self-contained apart from `_files/`) |

## How it is built

**Narrative first.** Five sections — why it matters, the claim under test, how
to read the numbers, the four findings, and what the data will and will not
carry. Each slide ends in one highlighted takeaway, so skimming only those
still gives the argument.

**Every data point is defined.** A slide early in the deck states what one
point *is* (one complete run), and every figure caption repeats the
configuration behind it and the units.

**The negative result is given equal weight.** The mode check — synthetic mode
is 77–86% wrong on the tier table's nominal bandwidth, and still degrades with
concurrency once calibrated — gets its own section, because it bounds what
every other synthetic number can be used for.

**Colour is deliberate.** The palette is the same validated categorical set
used in `docs/figures/`, assigned in fixed order. Every series is also
distinguished by direct label or dash pattern, so identity never rests on
colour alone.
