# ACIQ — Master's Thesis Project Guidance

## Project context

**Thesis:** *Statistical Analysis of Weight Discretization in Deep Learning* — Master's Final Degree Project at Kaunas University of Technology (KTU), Faculty of Mathematics and Natural Sciences, 2026.

**Current state of the thesis:** `./docs/thesis.pdf`. The manuscript itself is authored in Word/Pages — no `.tex` source exists in the repo.

**Current focus:** Results chapter (§3.1–3.8). The Literature Review (§1) and Methodology (§2) are already written in `./docs/thesis.pdf`; new prose contributions are §3 subsections.

The repo also holds the ACIQ research code under `aciq/`, experiments under `examples/` and `results/`.

## Writing style

### Voice

Use a formal, third-person, observational voice. The calibration target is §3.8 (Bias and variance correction) of `./docs/thesis.pdf` — re-read when in doubt.

### Rules

1. **One idea per sentence.** Split sentences that join two distinct ideas with "and", "so", "but", ", which", or a comma.
2. **Be concise.** Cut every word that does not add information. If two phrasings convey the same meaning, use the shorter one. A short paragraph that carries one idea is better than a long paragraph that pads it.
3. **No editorial filler.** Avoid: "notably", "interestingly", "the approach is notable because", "a key advantage", "comprehensive", "robust", "carefully", "we will see".
4. **Plain words.** Prefer the everyday word over its Latinate or technical-sounding synonym when both mean the same thing — "affects" not "governs", "size" not "magnitude". Keep technical terms with no plain equivalent.
5. **No first-person voice.** No "we", "our", "this thesis argues".
6. **No author names in prose.** Describe the work and let the citation carry the attribution.
7. **No parenthetical glosses on established acronyms.** Spell out once on first use; use bare thereafter.
8. **US spelling.** "optimization", not "optimisation"; "behavior", not "behaviour".
9. **State what is measured before reporting numbers.** Define the metric, the sample, and the unit; then give the number. A bare statistic without its definition is unreadable.
10. **Observation before interpretation.** Report the empirical value first; then state what it implies. Keep the two roles in separate sentences.
11. **Number precision matches signal.** Three or four significant digits for most quantities; precision must not exceed measurement resolution. Report ranges as `[lo, hi]`, not "around X".
12. **Refer to figures with `\ref{}` and `\label{}`.** Never hard-code "Figure 1" in prose.
13. **Cite methods, not findings.** Cite the literature paper on first use of an imported method (e.g. "ACIQ [N]", "Kolmogorov-Smirnov test [N]"). Empirical claims from this work need no citation — the script and figure are the source. External numeric baselines compared against do need a citation. Most results paragraphs end with the empirical observation, not with `[N]`.

## Citations

1. **ISO 690 numeric format. Never fabricate** — fetch the source page or ask. Cite the original paper that introduced a concept, not a later restatement. Read each cited paper enough to confirm the claim matches what the paper actually reports.
2. **Peer-reviewed sources preferred.** Journal articles, peer-reviewed conference proceedings, widely-cited arXiv preprints. Skip blogs, lecture notes, course slides, and unpublished technical reports unless no alternative exists.
3. **Hard-code `[N]` in prose.** Pandoc does not expand `\cite{}` against an inline bibliography. Numbering starts at 1 in the order references first appear in the deliverable; the author renumbers against the master bibliography when pasting into `./docs/thesis.pdf`.

## Deliverables

### Files

All requested writing ships as a self-contained directory `docs/<topic>/` containing:
- `<topic>.tex` — LaTeX source
- `<topic>.pdf` — built via `make pdf` (for visual review)
- `<topic>.docx` — built via `make docx` (the paste-into-manuscript artefact; equations render as native Word OMML)
- `Makefile` — copy from `docs/Makefile.example`

Section requests come **one at a time**. When asked to write a thesis section (e.g. "§3.3. Statistical properties of layer weights distributions"), open `./docs/thesis.pdf` and reuse the **exact heading text** — number and title — for the LaTeX `\subsection*{...}`. Do not paraphrase or shorten.

Do not produce prose in Markdown, plain text, or chat-only form unless explicitly asked.

### Build

- `tectonic` (at `~/.local/bin/tectonic`) builds the PDF.
- `pandoc` 3.x (at `~/.local/bin/pandoc`) builds the DOCX. Ubuntu's 2.9 is too buggy on math.
- `docs/Makefile.example` is the working reference — targets are `pdf`, `docx`, `all`, `clean`. After copying, update the `TEX :=` line to point at the new `<topic>.tex`.

### Math + pandoc gotchas

- **Avoid `\tag{...}`.** Pandoc's math parser rejects it. For a manual equation number `(N)`, use `\setcounter{equation}{N-1}` then `\begin{equation}...\end{equation}` — both engines honor this.
- **Avoid `\;`, `\!`, and other thin-space macros inside math.** They survive in PDF but pandoc strips them and can miss-parse the result. Use a plain space or `\,`.
- **Ignore the `.docx` round-trip warning.** `pandoc file.docx -t plain` warns "Could not convert TeX math…", but `word/document.xml` contains valid `<m:oMath>` and Word renders the equations natively.

### Document layout

- `\documentclass[11pt,a4paper]{article}`. Preamble in this order: `geometry` with `margin=2.5cm`, `amsmath`, `amssymb`, `bm`, `graphicx`, `hyperref`.
- Thesis-section deliverables (prose meant to be pasted into `./docs/thesis.pdf`) start with the starred heading immediately inside `\begin{document}`. No `\title`, `\author`, `\date`, or `\maketitle`.
- Figures live alongside the `.tex` in the same `docs/<topic>/` directory; reference them with relative paths in `\includegraphics{...}`.

## Workflow

- **Do not auto-commit.** Spec docs, plans, and thesis-deliverable files are left untracked. Wait for an explicit "commit" instruction before running `git add` / `git commit`.
- **One section at a time.** Wait for an explicit request naming the section to write; do not pre-emptively draft adjacent sections.
- **Editing and revision:** focus on logical flow and conciseness; edit rather than rewrite when the original is sound.
