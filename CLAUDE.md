# ACIQ — Master's Thesis Project Guidance

**Thesis:** *Statistical Analysis of Weight Discretization in Deep Learning* — Master's Final Degree Project at KTU, 2026.
**Manuscript:** `./docs/thesis.pdf` (authored in Word; no `.tex` in repo).
**Current focus:** Results chapter (§3.1–3.8). Lit Review (§1) and Methodology (§2) are already written.

# Writing results prose

The centre of these instructions. Voice: formal, third-person, observational. Calibration target: §3.8 (Bias and variance correction) of `./docs/thesis.pdf`.

## Rules

1. **One idea per sentence.** Split on "and", "so", "but", ", which", or run-on commas.
2. **Be concise.** Cut every word that does not add information. Pick the shorter phrasing.
3. **No editorial filler.** Avoid "notably", "interestingly", "comprehensive", "robust", "carefully", "we will see".
4. **Plain words.** Use the everyday word when the meaning is the same.
5. **No first-person.** No "we", "our", "this thesis argues".
6. **No author names in prose.** Let the citation carry the attribution.
7. **No parenthetical glosses on established acronyms.** Spell out on first use; bare thereafter.
8. **US spelling.** "optimization", not "optimisation".
9. **State what is measured before reporting numbers.** Metric, sample, unit — then the number.
10. **Observation before interpretation.** Report the value; then state what it implies. Separate sentences.
11. **Number precision.** 3 significant figures by default; integer counts; scientific notation outside `[10⁻², 10³]`. Don't mix precision across related quantities. Ranges as `[lo, hi]` — never "approximately X", "around X", "roughly Y".
12. **Figures.** `\label{fig:...}` + `Figure~\ref{fig:...}` for every figure. Caption: one sentence with axis units, ending in a period.
13. **Cite methods, not findings.** Cite on first use of an imported method, an external baseline, or a named concept. Empirical claims from this work need no citation. Most results paragraphs end with the observation, not `[N]`.

## Word swaps

| Avoid | Use | Avoid | Use |
|---|---|---|---|
| It can be seen that X | X | governs | affects |
| It is worth noting that X | (say X) | magnitude | size |
| In order to | to | attenuates | shrinks |
| Due to the fact that | because | methodology | method |
| A large number of | many | facilitate | help |
| There exists | there is | demonstrate | show |
| Utilize | use | leverage | use |
| As shown in Figure 1, X | X (Figure~\ref{...}) | approximately | (pick precision) |

## Citation hygiene

1. **ISO 690 numeric; never fabricate.** Cite the original paper; confirm the claim matches what it reports.
2. **Peer-reviewed preferred.** Skip blogs and lecture notes unless no alternative.
3. **Hard-code `[N]` in prose.** Pandoc does not expand `\cite{}`. Number from 1 in order of appearance.

# Mechanics

Each writing request ships as `docs/<topic>/` containing `<topic>.tex`, `Makefile` (copy of `docs/Makefile.example`, `TEX :=` updated), and the figures referenced. When the request names an existing thesis section, reuse the **exact heading text** for `\subsection*{...}`.

```sh
cd docs/<topic> && make pdf docx
```

`tectonic` at `~/.local/bin/tectonic`; `pandoc` 3.x at `~/.local/bin/pandoc` (Ubuntu's 2.9 is too buggy on math).

```latex
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}\usepackage{graphicx}\usepackage{hyperref}
```

Pandoc gotchas: no `\tag{...}` (use `\setcounter{equation}{N-1}` + `\begin{equation}`); no `\;` / `\!` / thin-space macros in math (use plain space or `\,`); ignore the `.docx` round-trip math-conversion warning — Word renders correctly.

# Workflow

- **No auto-commit.** Wait for explicit "commit" instruction.
- **One section at a time.** Don't pre-emptively draft adjacent sections.
- **Edit, don't rewrite.** Focus on flow and conciseness when iterating.
