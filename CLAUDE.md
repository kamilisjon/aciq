# ACIQ — Master's Thesis Project Guidance

## Project context

**Thesis:** *Statistical Analysis of Weight Discretization in Deep Learning* — Master's Final Degree Project at Kaunas University of Technology (KTU), Faculty of Mathematics and Natural Sciences, 2026.
- **Current state of the thesis:** `./docs/thesis.pdf`. The manuscript itself is authored in Word/Pages — no `.tex` source exists in the repo.

The repo also holds the ACIQ research code under `aciq/`, experiments under `examples/` and `results/`.

## Writing style

### Voice

Use a formal, third-person, observational voice. The calibration target is §1.2.6 (Batch normalization) of `./docs/thesis.pdf` — re-read when in doubt.

### Rules

These rules are non-negotiable for new prose contributions:

1. **One idea per sentence.** If a sentence joins two distinct ideas with "and", "so", "but", ", which", or a comma, split it.
2. **Be concise.** Cut every word that does not add information. Delete sentences that restate the previous one. If two phrasings convey the same meaning, use the shorter one. A short paragraph that carries one idea is better than a long paragraph that pads it.
3. **No editorial filler.** Avoid: "notably", "interestingly", "the approach is notable because", "a key advantage", "comprehensive", "robust", "carefully", "we will see".
4. **Use plain words.** Prefer the everyday word over its Latinate or technical-sounding synonym when both mean the same thing — "affects" not "governs", "size" not "magnitude", "unchanged" not "without attenuation", "problem" not "pathology", "fixes" not "removes the pathology". Keep technical terms that have no plain equivalent.
5. **No first-person voice.** No "we", "our", "this thesis argues".
6. **Do not refer to authors by name in prose.** Describe the work and let the citation carry the attribution. Write *"Deep Compression reduces model size by an order of magnitude with negligible accuracy loss [N]"*, not *"Han et al.\ demonstrate ... [N]"*.
7. **No parenthetical glosses on established acronyms.** Spell out once on first use (e.g. "Post-Training Quantization (PTQ)"); use bare thereafter. Do not add "(PTQ, a technique that…)".
8. **US spelling.** "optimization", not "optimisation"; "behavior", not "behaviour".
9. **Reason from first principles.** Before showing a method or formula, state the underlying problem it solves. §1.2.6. motivates BN as the response to internal covariate shift before introducing the equation.
10. **Encapsulate every claim with a citation.** Group consecutive sentences drawn from the same source into a block and close the block with `[N]` (rendered from numeric `\cite{key}`). The closing `[N]` covers every sentence in the block back to the previous citation or paragraph start. No claim sits outside such a block. Every paragraph ends with a citation. Trailing sentences after the last `[N]` are not allowed — either extend the block with a citation or cut the sentence. Do not fabricate references — if context is missing, ask.

## Deliverables

### Files

All requested writing ships as a self-contained directory `docs/<topic>/` containing:
- `<topic>.tex` — LaTeX source
- `<topic>.pdf` — built via `make pdf` (for visual review)
- `<topic>.docx` — built via `make docx` (the paste-into-manuscript artefact; equations render as native Word OMML)
- `Makefile` — copy from `docs/Makefile.example`

Section requests come **one at a time**. When asked to write a thesis section (e.g. "§1.1.1. ImageNet dataset"), open `./docs/thesis.pdf` and reuse the **exact heading text** — number and title — for the LaTeX `\subsection*{...}`. Do not paraphrase or shorten.

Do not produce prose in Markdown, plain text, or chat-only form unless explicitly asked.

### Build

- `tectonic` (at `~/.local/bin/tectonic`) builds the PDF.
- `pandoc` 3.x (at `~/.local/bin/pandoc`) builds the DOCX. Ubuntu's 2.9 is too buggy on math.
- `docs/Makefile.example` is the working reference — targets are `pdf`, `docx`, `all`, `clean`. After copying, update the `TEX :=` line to point at the new `<topic>.tex`.

### Math + pandoc gotchas

- **Avoid `\tag{...}`.** Pandoc's math parser rejects it. For a manual equation number `(N)`, use `\setcounter{equation}{N-1}` then `\begin{equation}...\end{equation}` — both engines honor this.
- **Avoid `\;`, `\!`, and other thin-space macros inside math.** They survive in PDF but pandoc strips them and can mis-parse the result. Use a plain space or `\,`.
- **Hard-code citations for pandoc.** Pandoc does not expand `\cite{key}` against an inline `thebibliography` block. Write the deliverable's own `[N]` directly in prose, numbered from 1 in the order references first appear. The author renumbers against the master bibliography by hand when pasting into `./docs/thesis.pdf`.
- **Ignore the `.docx` round-trip warning.** `pandoc file.docx -t plain` warns "Could not convert TeX math…", but `word/document.xml` contains valid `<m:oMath>` and Word renders the equations natively.

### Document layout

- `\documentclass[11pt,a4paper]{article}`. Preamble in this order: `geometry` with `margin=2.5cm`, `amsmath`, `amssymb`, `bm`, `hyperref`.
- Thesis-section deliverables (prose meant to be pasted into `./docs/thesis.pdf`) start with the starred heading immediately inside `\begin{document}`. No `\title`, `\author`, `\date`, or `\maketitle`.
- Standalone working documents (e.g. `docs/bias_variance_correction/`) may use `\maketitle`.

## Citations

1. **ISO 690 numeric format.** Never fabricate — fetch the arxiv page or ask.
2. **Peer-reviewed sources only.** Journal articles, peer-reviewed conference proceedings, and widely-cited arxiv preprints. No books, textbooks, monographs, blog posts, lecture notes, course slides, or unpublished technical reports. When the obvious source is a textbook, find the underlying paper.
3. **Prefer canonical, original papers.** Cite the paper that introduced a concept, not a later paper that uses or restates it. A survey supplements but does not replace the primary source.
4. **Read each paper before citing it.** The cited claim must match what the paper actually reports — not the title, not the abstract, not a downstream paraphrase. The dataset, model, and metric named in the prose must match the paper. If the paper does not support the claim, change the claim or change the citation.
5. **Paraphrase, never copy.** Re-express every borrowed sentence in the thesis's own voice. Five-or-more consecutive words shared with the source = rewrite. Named technical terms ("internal covariate shift", "dying ReLU") are the exception. Plagiarism risk is non-negotiable; the Declaration of Academic Integrity binds the author to this.

## Workflow

- **Do not auto-commit.** Spec docs, plans, and thesis-deliverable files are left untracked. Wait for an explicit "commit" instruction before running `git add` / `git commit`.
- **One section at a time.** Wait for an explicit request naming the section to write; do not pre-emptively draft adjacent sections.
- **Editing and revision:** focus on logical flow and conciseness; edit rather than rewrite when the original is sound.
