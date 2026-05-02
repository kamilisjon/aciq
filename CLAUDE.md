# ACIQ — Master's Thesis Project Guidance

## Project context

**Thesis:** *Statistical Analysis of Weight Discretization in Deep Learning* — Master's Final Degree Project at Kaunas University of Technology (KTU), Faculty of Mathematics and Natural Sciences, 2026.
- **Author:** Kamilis Jonkus
- **Supervisor:** Assoc. prof., dr. Tomas Iešmantas
- **Active branch:** `bias_correction`
- **Current state of the thesis:** `./docs/thesis.pdf` at the repo root. The thesis itself is authored in Word/Pages — no `.tex` source for the manuscript exists in the repo.

The repo also holds the ACIQ research code under `aciq/`, experiments under `examples/` and `results/`.

## Writing style — voice

Use a formal, third-person, observational voice. Match the voice of the **already-finished** thesis sections.

When in doubt, re-read those pages. They are the calibration target.

## Writing style — concrete rules

These rules are non-negotiable for new prose contributions:

1. **One idea per sentence.** If a sentence joins two distinct ideas with "and", "so", "but", ", which", or a comma, split it.
2. **No editorial filler.** Avoid: "notably", "interestingly", "the approach is notable because", "a key advantage", "comprehensive", "robust", "carefully", "we will see".
3. **No first-person voice.** No "we", "our", "this thesis argues".
4. **Do not refer to authors by name in prose.** Describe the work and let the citation carry the attribution. Write *"Deep Compression reduces model size by an order of magnitude with negligible accuracy loss [N]"*, not *"Han et al.\ demonstrate in Deep Compression that ... [N]"*. The bibliography entry already names the authors.
5. **No parenthetical glosses on established acronyms.** Spell out once on first use (e.g. "Post-Training Quantization (PTQ)"); use bare thereafter. Do not add "(PTQ, a technique that…)".
6. **US spelling.** "optimization", not "optimisation"; "behavior", not "behaviour".
7. **Reason from first principles.** Before showing a method or formula, state the underlying problem it solves. §1.2.4 BN motivates BN as the response to internal covariate shift before introducing the equation. Parent sections (e.g. §1.1 above §1.1.1 and §1.1.2) motivate the underlying problem the children answer; they do not summarize the children. §1.1 explains why benchmark datasets matter before §1.1.1 and §1.1.2 describe specific datasets.
8. **Cite on the first non-trivial claim.** Use numeric `\cite{key}` rendered as `[N]`. Do not fabricate references — if context is missing, ask.
9. **Figure references use the manuscript phrasing.** Write *"illustrated in figure Fig. N"* (matching §1.1.1's pattern), not *"see Figure N"* or *"as shown in Fig. N"*. Keep figure numbers literal to match the manual numbering in `./docs/thesis.pdf`.

## Output format — always LaTeX + PDF

**All requested writing must be delivered as a LaTeX `.tex` source plus a tectonic-built `.pdf`** under a self-contained directory `docs/<topic>/`. Do not produce prose in Markdown, plain text, chat-only, or Word-paste-ready form unless explicitly asked.

Section requests come **one at a time**. When asked to write a particular thesis section (e.g. "§1.1.1. ImageNet dataset"), open `./docs/thesis.pdf` and reuse the **exact heading text** as it appears in the manuscript — number and title — for the LaTeX `\subsection*{...}`. Do not paraphrase, retitle, or shorten the heading.

Every new writing artefact gets:
- `docs/<topic>/<topic>.tex`
- `docs/<topic>/<topic>.pdf` (built via `make pdf`)
- `docs/<topic>/Makefile` (copy from `docs/bias_variance_correction/Makefile`)

## LaTeX conventions

**Build:**
- Engine: `tectonic` (installed at `~/.local/bin/tectonic`).
- Per-doc `Makefile` with `pdf` and `clean` targets — the `Makefile` at `docs/bias_variance_correction/Makefile` is the working reference.

**Document setup:**
- `\documentclass[11pt,a4paper]{article}`
- Preamble in this order: `geometry` with `margin=2.5cm`, `amsmath`, `amssymb`, `bm`, `hyperref`.

**Numbering:**
- Use `\section*{N. Heading}`, `\subsection*{N.M. Subheading}`, and `\subsubsection*{N.M.O. Subsubheading}` (e.g. `1.1.2. MNIST dataset`) — the **starred** forms. The numbers are written into the heading text literally so they match the thesis manuscript's manual numbering. If sections are reordered, update the numbers by hand.
- The heading text after the number must match `./docs/thesis.pdf` verbatim.

**Document layout for thesis-section deliverables:**
- Thesis-section deliverables (prose meant to be pasted into `./docs/thesis.pdf`) start with the starred heading immediately inside `\begin{document}`. No `\title`, `\author`, `\date`, or `\maketitle`. The PDF exists for review, not as a self-contained document.
- Derivation / working documents (e.g. `docs/bias_variance_correction/`) may use `\maketitle` since they stand alone.

**Bibliography (ISO 690, numeric / citation-order):**
- Inline `\begin{thebibliography}{N}` block; no `.bib` files.
- Citation key convention: `firstauthor_lastnameYEAR` lowercase (e.g. `nagel2019`, `gholami2021`). Suffix `a`/`b` for multiple papers from the same author and year.
- ISO 690 entry format — surnames in uppercase, given-name initials, "and" before the last author, italicised venue, page range or arxiv URL:
  ```
  \bibitem{key}
  SURNAME, A. and SURNAME, B. Title of the work. In: \emph{Venue Name}, Year, p. XXX--YYY.
  ```
  For arxiv preprints (no formal venue):
  ```
  \bibitem{key}
  SURNAME, A. and SURNAME, B. Title of the work [online]. arXiv preprint arXiv:NNNN.NNNNN, Year. Available from: \url{https://arxiv.org/abs/NNNN.NNNNN}
  ```

## Citation policy

Use ISO 690 numeric format throughout. Never fabricate references — if uncertain, fetch the arxiv abstract page or ask.

**Cite only peer-reviewed scientific papers and conference proceedings.** Do not cite books, textbooks, monographs, blog posts, lecture notes, course slides, technical reports without peer review, or other unpublished material. Acceptable venues: journal articles (e.g. *Nature*, *JMLR*, *IEEE TPAMI*), peer-reviewed conference proceedings (e.g. NeurIPS, ICML, ICLR, CVPR), and arxiv preprints when the work is widely cited and the preprint is the primary venue. When the obvious reference is a textbook, find the underlying scientific paper instead — e.g. cite Rumelhart, Hinton, Williams (1986, *Nature*) for the multilayer feedforward network, not the Goodfellow–Bengio–Courville textbook.

**Prefer canonical papers over secondary or derivative ones.** Cite the original paper that introduced a concept, method, or architecture, not a later paper that merely uses or restates it. Examples: Rumelhart, Hinton, Williams (1986) for backpropagation and multilayer feedforward networks; LeCun et al.\ (1998, *Proc. IEEE*) for convolutional networks and MNIST; Krizhevsky, Sutskever, Hinton (2012, NeurIPS) for AlexNet and the modern ImageNet result; He et al.\ (2016, CVPR) for ResNet; Ioffe \& Szegedy (2015, ICML) for batch normalization; Russakovsky et al.\ (2015, *IJCV*) for the ImageNet benchmark. If a survey is cited, it should be in addition to the canonical primary source, not as a substitute for it.

## Workflow

- **Do not auto-commit.** Spec docs (`docs/superpowers/specs/`), implementation plans (`docs/superpowers/plans/`), and the thesis-deliverable LaTeX/PDF files are left untracked. Wait for an explicit "commit" instruction before running `git add` / `git commit`.
- **One section at a time.** Wait for an explicit request naming the section to write; do not pre-emptively draft adjacent sections.
- **Editing and revision:** focus on logical flow and conciseness; edit rather than rewrite when the original is sound.
- **Structure:** follow IMRaD (Introduction, Methods, Results, Discussion) for original-research sections where applicable. Match the conventions already established in `./docs/thesis.pdf`.
