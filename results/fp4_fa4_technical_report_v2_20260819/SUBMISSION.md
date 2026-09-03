# Submission packaging

This directory contains the authoritative manuscript. The entry point is
`main.tex`. Each generated `submission/verification.json` records the Git
commit from which its archives were built. The audited local build uses
pdfLaTeX and BibTeX from TeX Live 2023.

Build both upload archives from the manuscript directory:

```bash
python3 prepare_submission_archives.py
```

The command creates and clean-builds these gitignored outputs:

- `submission/hardware-aware-fp4-fa4-arxiv.tar.gz`
- `submission/hardware-aware-fp4-fa4-overleaf.zip`
- `submission/SHA256SUMS`
- `submission/arxiv-manifest.sha256`
- `submission/overleaf-manifest.sha256`
- `submission/verification.json`

The build is fail-closed. It vendors the exact generated tables and figures
used by LaTeX, rejects unlisted files and unsafe names, regenerates the
bibliography, disables shell escape, builds both archives after clean
extraction, and checks path closure, page count, extracted text, embedded
fonts, JavaScript, and archive size.

## Archive differences

The arXiv archive is a minimal compile-only source bundle. Its 47-file
allowlist includes `main.tex`, `main.bib`, the matching BibTeX-generated
`main.bbl`, the custom style, section and appendix sources, 14 generated table
snippets, 11 final scientific figure files, and the one used logo. It excludes
`main.pdf`, auxiliary build files, READMEs, raw receipts, data-fetch scripts,
plotting scripts, unused plots and tables, caches, legacy classes, and unused
font files.

The Overleaf archive has the same source closure but omits `main.bbl`, so
Overleaf regenerates it from `main.bib`, and adds a short build README. Select
`main.tex` as the main document, pdfLaTeX as the compiler, and TeX Live 2023.
TeX Live 2025 is arXiv's current default, but this manuscript has been locally
reproduced in the still-supported TeX Live 2023 environment; do not claim a
TeX Live 2025 reproduction until it has actually been run.

The relevant service documentation is:

- [arXiv TeX submission guidance](https://info.arxiv.org/help/submit_tex.html)
- [TeX Live versions at arXiv](https://info.arxiv.org/help/faq/texlive.html)
- [arXiv common processing mistakes](https://info.arxiv.org/help/faq/mistakes.html)
- [Overleaf's arXiv checklist](https://docs.overleaf.com/troubleshooting-and-support/checklist-for-arxiv-submissions)
- [Overleaf project upload documentation](https://docs.overleaf.com/managing-projects-and-files/uploading-a-project)

## Human gates before submission

Packaging success is not submission approval. The submitter must still:

1. Confirm the final title, abstract, author list and order, affiliations,
   acknowledgements, and contact details in both the source and arXiv metadata.
2. Confirm the primary category and every cross-list with all authors. Do not
   infer a category from the repository or experiments.
3. Select the arXiv license deliberately and confirm that the manuscript,
   citations, figures, logo, and linked code can be distributed under it.
4. Upload the arXiv archive, explicitly select pdfLaTeX and TeX Live 2023, and
   inspect the complete processing log rather than accepting compilation alone.
5. Inspect all 56 pages of the upload-time preview, especially the title page,
   plots, table boundaries, equations, citations, hyperlinks, references, and
   the final appendix page. Compare the preview with the locally verified PDF.
6. Confirm that no figure has been substituted, downsampled, cropped, or
   reordered and that the bibliography is populated from `main.bbl`.
7. Re-run the packager after every manuscript, bibliography, table, or figure
   change. Upload only archives whose checksums match the latest
   `submission/SHA256SUMS` and retain `verification.json` with the release
   record.

Do not upload the whole manuscript working directory. arXiv makes submission
sources public, and that directory intentionally contains development and
provenance material outside the minimal audited source closure.
