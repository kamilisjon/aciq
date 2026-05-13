# ACIQ — Master's Thesis Project Guidance

## Project context

**Thesis:** *Statistical Analysis of Weight Discretization in Deep Learning* — Master's Final Degree Project at KTU, Faculty of Mathematics and Natural Sciences, 2026.
**Manuscript:** `./docs/thesis.pdf` (authored in Word; no `.tex` in repo).
**Current focus:** Results chapter (§3.1–3.8). Literature Review (§1) and Methodology (§2) are already written.

# Writing results prose

The centre of these instructions. Everything below this section is plumbing.

## Voice

Formal, third-person, observational. Calibration target: §3.8 (Bias and variance correction) of `./docs/thesis.pdf`.

## Rules

1. **One idea per sentence.** Split on "and", "so", "but", ", which", or run-on commas.
2. **Be concise.** Cut every word that does not add information. Pick the shorter phrasing.
3. **No editorial filler.** Avoid "notably", "interestingly", "comprehensive", "robust", "carefully", "we will see".
4. **Plain words.** Use the everyday word when meaning is the same.
5. **No first-person voice.** No "we", "our", "this thesis argues".
6. **No author names in prose.** Let the citation carry the attribution.
7. **No parenthetical glosses on established acronyms.** Spell out once on first use; bare thereafter.
8. **US spelling.** "optimization", not "optimisation".
9. **State what is measured before reporting numbers.** Metric, sample, unit — then the number.
10. **Observation before interpretation.** Report the value; then state what it implies. Separate sentences.
11. **Number precision matches signal.** See §Numbers.
12. **Figures by `\ref{}`, never by hard-coded number.** See §Figures.
13. **Cite methods, not findings.** See §Citations in results.

## Anti-patterns

| Avoid | Use |
|---|---|
| It can be seen that X | X |
| It is worth noting that X | (just say X) |
| In order to | to |
| Due to the fact that | because |
| A large number of | many |
| There exists | there is |
| Utilize | use |
| As shown in Figure 1, X | X (Figure~\ref{...}) |

## Plain-word swaps

| Avoid | Use |
|---|---|
| governs | affects |
| magnitude | size |
| attenuates | shrinks |
| methodology | method |
| facilitate | help |
| demonstrate | show |
| leverage | use |
| approximately | (pick a precision) |

## Numbers

- 3 significant figures by default; counts as integers; scientific notation outside `[10⁻², 10³]`.
- Don't mix precision across related quantities in one paragraph.
- Ranges as `[lo, hi]`. Never "approximately X", "around X", or "roughly Y".
- Don't report digits the measurement can't support.

## Figures

- Every figure has a `\label{fig:...}` and is referenced with `Figure~\ref{fig:...}`.
- Captions: one sentence with axis units, ending in a period.
- Figures live alongside the `.tex` in `docs/<topic>/`.

## Citations in results

Most results paragraphs end with the empirical observation, not with `[N]`. Cite only on first use of an imported method, an external numeric baseline, or a named concept.

1. **ISO 690 numeric; never fabricate.** Cite the original paper; confirm the claim matches what it reports.
2. **Peer-reviewed sources preferred.** Skip blogs and lecture notes unless no alternative exists.
3. **Hard-code `[N]` in prose.** Pandoc does not expand `\cite{}`. Number from 1 in order of appearance.

# Mechanics

## Layout

Each writing request ships as `docs/<topic>/`:
- `<topic>.tex` — standalone LaTeX source.
- `Makefile` — copy from `docs/Makefile.example`; update `TEX :=`.
- Figures live in this directory; relative paths in `\includegraphics{...}`.

When the request names an existing thesis section, reuse the **exact heading text** for `\subsection*{...}`.

## Build

```sh
cd docs/<topic> && make pdf docx
```

`tectonic` at `~/.local/bin/tectonic`; `pandoc` 3.x at `~/.local/bin/pandoc` (Ubuntu's 2.9 is too buggy on math).

## Preamble

```latex
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}\usepackage{graphicx}\usepackage{hyperref}
```

## Pandoc gotchas

- No `\tag{...}`; use `\setcounter{equation}{N-1}` then `\begin{equation}` for manual numbers.
- No `\;`, `\!`, or thin-space macros inside math. Use a plain space or `\,`.
- Ignore the `.docx` round-trip math-conversion warning. Word renders correctly.

# Workflow

- **No auto-commit.** Wait for explicit "commit" instruction.
- **One section at a time.** Wait for an explicit request naming the section.
- **Edit, don't rewrite.** Focus on flow and conciseness when iterating.
