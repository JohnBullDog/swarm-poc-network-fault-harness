# paper

The findings written up as a paper, for internal circulation.

Same data and same narrative as `../presentation/`, in a form that can be read
without a presenter: abstract, numbered sections, numbered and captioned
figures and tables, threats to validity, and what the data does and does not
support.

Every figure and every statistic is computed from the published CSVs in
`../datasets/` **at render time**. Nothing is transcribed, so the paper cannot
drift from the data.

## Render it

Open `swarm_poc_paper.qmd` in RStudio and click **Render**, or:

```bash
quarto render swarm_poc_paper.qmd --to html    # -> swarm_poc_paper.html
quarto render swarm_poc_paper.qmd --to typst   # -> swarm_poc_paper.pdf (11 pp)
```

The PDF goes through Typst, which Quarto bundles, so no LaTeX install is
needed.

## Files

| file | what |
|---|---|
| `swarm_poc_paper.qmd` | the paper |
| `paper.scss` | reading styles: measured column, serif body, sans headings |
| `swarm_poc_paper.html` | rendered, with a left-hand table of contents |
| `swarm_poc_paper.pdf` | rendered, A4, 11 pages |

Shared R setup (palette, theme, data loading, the delay-sweep regression)
lives in `../R/swarm_setup.R` and is used by the paper and the deck alike.

## Structure

1. **Introduction** — the promise, the structural constraint, what was unknown
2. **The cost model under test** — the four expressions, with symbols defined
3. **Method** — the harness, two engines, two compute modes, and the three
   measurement decisions that do most of the work for credibility
4. **Results** — delay, concurrency, scale, failover, mode check, engine
   agreement
5. **Threats to validity** — including the six instrumentation defects found
   during the work, four of which would have produced plausible wrong results
6. **What the data supports** — supported and not-supported, stated plainly
7. **Future work** — what would close each open question
8. **Data and code availability**
