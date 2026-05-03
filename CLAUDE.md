# ACIQ — Master's Thesis Project Guidance

## Project context

**Thesis:** *Statistical Analysis of Weight Discretization in Deep Learning* — Master's Final Degree Project at Kaunas University of Technology (KTU), Faculty of Mathematics and Natural Sciences, 2026.
- **Current state of the thesis:** `./docs/thesis.pdf` at the repo root. The thesis itself is authored in Word/Pages — no `.tex` source for the manuscript exists in the repo.

The repo also holds the ACIQ research code under `aciq/`, experiments under `examples/` and `results/`.

## Writing style — voice

Use a formal, third-person, observational voice. Match the voice of the already-finished thesis section: Batch normalization.

When in doubt, re-read those pages. They are the calibration target.

## Writing style — concrete rules

These rules are non-negotiable for new prose contributions:

1. **One idea per sentence.** If a sentence joins two distinct ideas with "and", "so", "but", ", which", or a comma, split it.
2. **No editorial filler.** Avoid: "notably", "interestingly", "the approach is notable because", "a key advantage", "comprehensive", "robust", "carefully", "we will see".
3. **Use plain words.** Prefer the everyday word over its Latinate or technical-sounding synonym when both mean the same thing — "affects" not "governs", "size" not "magnitude", "unchanged" not "without attenuation", "problem" not "pathology", "any desired accuracy" not "arbitrary accuracy", "fixes" not "removes the pathology". Keep technical terms that have no plain equivalent (back-propagation, universal approximation, affine map, ReLU, gradient).
3. **No first-person voice.** No "we", "our", "this thesis argues".
4. **Do not refer to authors by name in prose.** Describe the work and let the citation carry the attribution. Write *"Deep Compression reduces model size by an order of magnitude with negligible accuracy loss [N]"*, not *"Han et al.\ demonstrate in Deep Compression that ... [N]"*. The bibliography entry already names the authors.
5. **No parenthetical glosses on established acronyms.** Spell out once on first use (e.g. "Post-Training Quantization (PTQ)"); use bare thereafter. Do not add "(PTQ, a technique that…)".
6. **US spelling.** "optimization", not "optimisation"; "behavior", not "behaviour".
7. **Reason from first principles.** Before showing a method or formula, state the underlying problem it solves. §1.2.4 BN motivates BN as the response to internal covariate shift before introducing the equation.
8. **Cite on the first non-trivial claim.** Use numeric `\cite{key}` rendered as `[N]`. Do not fabricate references — if context is missing, ask.
9. **Figure references use the manuscript phrasing.** Write *"illustrated in figure Fig. N"* (matching §1.1.1's pattern), not *"see Figure N"* or *"as shown in Fig. N"*. Keep figure numbers literal to match the manual numbering in `./docs/thesis.pdf`.
10. Each claim needs to be backed by reference. There must be at least one reference in a paragraph. If there is 0 references this means that either paragraph scope is too small or that it is claimed without a reference, where both situations are bad.

## Output format — always LaTeX + PDF

**All requested writing must be delivered as a LaTeX `.tex` source, a tectonic-built `.pdf`, and a pandoc-built `.docx`** under a self-contained directory `docs/<topic>/`. Do not produce prose in Markdown, plain text, or chat-only form unless explicitly asked. The `.pdf` is for visual review; the `.docx` is what gets copy-pasted into the manuscript (paragraph structure preserved, equations rendered as native Word OMML).

Section requests come **one at a time**. When asked to write a particular thesis section (e.g. "§1.1.1. ImageNet dataset"), open `./docs/thesis.pdf` and reuse the **exact heading text** as it appears in the manuscript — number and title — for the LaTeX `\subsection*{...}`. Do not paraphrase, retitle, or shorten the heading.

Every new writing artefact gets:
- `docs/<topic>/<topic>.tex`
- `docs/<topic>/<topic>.pdf` (built via `make pdf`)
- `docs/<topic>/<topic>.docx` (built via `make docx`)
- `docs/<topic>/Makefile` (copy from `docs/bias_variance_correction/Makefile`)

## LaTeX conventions

**Build:**
- Engines: `tectonic` (installed at `~/.local/bin/tectonic`) for PDF; `pandoc` (`~/.local/bin/pandoc`, version 3.x — Ubuntu's 2.9 is too buggy on math) for DOCX.
- Per-doc `Makefile` with `pdf`, `docx`, `all`, and `clean` targets — the `Makefile` at `docs/bias_variance_correction/Makefile` is the working reference.

**Math compatibility for `.docx` (pandoc):**
- Avoid `\tag{...}`. Pandoc's math parser rejects it. For a manual equation number `(N)`, use `\setcounter{equation}{N-1}` followed by `\begin{equation}...\end{equation}` — both tectonic and pandoc honor this.
- Avoid `\;`, `\!`, and other thin-space macros inside math; they survive in PDF but pandoc strips them and can mis-parse the result. Use a plain space or `\,` if spacing is needed.
- Citations: pandoc does not expand `\cite{key}` against an inline `thebibliography` block. Hard-code the **deliverable's own** `[N]` directly in prose, numbered from 1 in the order the references first appear in the section. The deliverable then reads as a self-contained document; the author renumbers against the master manuscript bibliography by hand when pasting into `./docs/thesis.pdf`.
- The `.docx` will appear to fail a pandoc round-trip (`pandoc file.docx -t plain` warns "Could not convert TeX math…"); ignore that. The actual `word/document.xml` contains valid `<m:oMath>` and Word renders the equation as a native, editable formula.

**Document setup:**
- `\documentclass[11pt,a4paper]{article}`
- Preamble in this order: `geometry` with `margin=2.5cm`, `amsmath`, `amssymb`, `bm`, `hyperref`.

**Document layout for thesis-section deliverables:**
- Thesis-section deliverables (prose meant to be pasted into `./docs/thesis.pdf`) start with the starred heading immediately inside `\begin{document}`. No `\title`, `\author`, `\date`, or `\maketitle`. The PDF exists for review, not as a self-contained document.
- Derivation / working documents (e.g. `docs/bias_variance_correction/`) may use `\maketitle` since they stand alone.


## Citation policy

Use ISO 690 numeric format throughout. Never fabricate references — if uncertain, fetch the arxiv abstract page or ask.

**Cite only peer-reviewed scientific papers and conference proceedings.** Do not cite books, textbooks, monographs, blog posts, lecture notes, course slides, technical reports without peer review, or other unpublished material. Acceptable venues: journal articles (e.g. *Nature*, *JMLR*, *IEEE TPAMI*), peer-reviewed conference proceedings (e.g. NeurIPS, ICML, ICLR, CVPR), and arxiv preprints when the work is widely cited and the preprint is the primary venue. When the obvious reference is a textbook, find the underlying scientific paper instead — e.g. cite Rumelhart, Hinton, Williams (1986, *Nature*) for the multilayer feedforward network, not the Goodfellow–Bengio–Courville textbook.

**Prefer canonical papers over secondary or derivative ones.** Cite the original paper that introduced a concept, method, or architecture, not a later paper that merely uses or restates it. Examples: Rumelhart, Hinton, Williams (1986) for backpropagation and multilayer feedforward networks; LeCun et al.\ (1998, *Proc. IEEE*) for convolutional networks and MNIST; Krizhevsky, Sutskever, Hinton (2012, NeurIPS) for AlexNet and the modern ImageNet result; He et al.\ (2016, CVPR) for ResNet; Ioffe \& Szegedy (2015, ICML) for batch normalization; Russakovsky et al.\ (2015, *IJCV*) for the ImageNet benchmark. If a survey is cited, it should be in addition to the canonical primary source, not as a substitute for it.

**Read each cited paper before citing it.** Spend the time required to actually understand the paper, not just its title or abstract. Fetch the arxiv abstract page, the published PDF, or another primary source. Read the introduction, the relevant method section, and the results that the prose claim depends on. The bar to clear before writing `[N]` after a sentence: the cited paper genuinely supports that specific sentence, the dataset / model / metric named in the prose matches what the paper actually reports, and the framing of the contribution matches the paper's own framing rather than a downstream paraphrase. Common failure modes to avoid — citing a paper because it is famously associated with a topic rather than because it supports the specific claim, attributing a result to a survey when the survey itself is citing the primary work, and inheriting wording from a secondary source that subtly misstates the original. If the paper turns out not to support the claim, change the claim or change the citation; do not leave the mismatch in.

**Paraphrase, do not copy.** Every sentence drawn from a cited paper must be re-expressed in the thesis's own voice and sentence structure. Do not lift phrases, definitions, or formulations verbatim from the source — even short distinctive phrases ("internal covariate shift", "dying ReLU") are acceptable as named technical terms but the surrounding explanation must be the author's own. Read the relevant passage, close the paper, then write the sentence from understanding rather than from the page. A useful self-check: if a sentence in the deliverable shares a clause of five or more consecutive words with the source, rewrite it. The same applies to figures and equations re-expressed as prose — describe them in the author's own words, not the source's caption text. Plagiarism risk is the reason this rule is non-negotiable; the Declaration of Academic Integrity at the front of the manuscript binds the author to it explicitly.

## Workflow

- **Do not auto-commit.** Spec docs (`docs/superpowers/specs/`), implementation plans (`docs/superpowers/plans/`), and the thesis-deliverable LaTeX/PDF files are left untracked. Wait for an explicit "commit" instruction before running `git add` / `git commit`.
- **One section at a time.** Wait for an explicit request naming the section to write; do not pre-emptively draft adjacent sections.
- **Editing and revision:** focus on logical flow and conciseness; edit rather than rewrite when the original is sound.
