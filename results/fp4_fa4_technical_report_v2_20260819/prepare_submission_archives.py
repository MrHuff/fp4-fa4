#!/usr/bin/env python3
"""Build and verify minimal arXiv and Overleaf source archives.

The manuscript consumes generated tables and figures from sibling result trees.
This script vendors only the files reached by the verified LaTeX dependency
graph, rewrites those paths inside temporary staging trees, and rejects any
archive that does not build in isolation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Iterable
import zipfile


PAPER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PAPER_DIR.parent
OUTPUT_DIR = PAPER_DIR / "submission"
SOURCE_DATE_EPOCH = 1_788_436_743
EXPECTED_PAGES = 56
ARXIV_MAX_ARCHIVE_BYTES = 50_000_000
ARXIV_NAME = "hardware-aware-fp4-fa4-arxiv.tar.gz"
OVERLEAF_NAME = "hardware-aware-fp4-fa4-overleaf.zip"

SECTIONS = (
    "sections/abstract.tex",
    "sections/01_summary.tex",
    "sections/02_problem_formats.tex",
    "sections/03_architecture.tex",
    "sections/04_retained_kernel.tex",
    "sections/05_method.tex",
    "sections/06_results.tex",
    "sections/07_profile.tex",
    "sections/08_causal_training.tex",
    "sections/09_conclusion.tex",
)
APPENDICES = (
    "appendices/a_nvnv_history.tex",
    "appendices/b_rejected_experiments.tex",
    "appendices/c_full_format_matrix.tex",
    "appendices/c_reproduction.tex",
    "appendices/d_accuracy_control.tex",
    "appendices/e_causal_training_provenance.tex",
    "appendices/f_causal_training_history.tex",
)
LOCAL_FIGURES = (
    "figures/causal_combined_forward_backward.pdf",
    "figures/causal_isolated_backward.pdf",
    "figures/llama8b_b4_matched_throughput.pdf",
    "figures/llama8b_b4_matched_training_curves.pdf",
    "figures/llama8b_b4_mxfp4_failure.pdf",
    "figures/llama8b_e2e_batch_scaling.pdf",
    "figures/llama8b_mxfp4_divergence.pdf",
    "figures/llama8b_training_curves.pdf",
)

# Source paths are relative to results/; destinations are archive-root relative.
VENDORED_FILES = (
    (
        "fp4_fa4_b300_tuning_20260802/tables/accuracy_matched_rows.tex",
        "generated/accuracy_matched_rows.tex",
    ),
    (
        "fp4_fa4_b300_tuning_20260802/tables/b300_d64_rows.tex",
        "generated/b300_d64_rows.tex",
    ),
    (
        "fp4_fa4_b300_tuning_20260802/tables/b300_tuning_macros.tex",
        "generated/b300_tuning_macros.tex",
    ),
    (
        "fp4_fa4_b300_tuning_20260802/tables/b300_tuning_rows.tex",
        "generated/b300_tuning_rows.tex",
    ),
    (
        "fp4_fa4_b300_tuning_20260802/tables/primary_cross_generation_d128_rows.tex",
        "generated/primary_cross_generation_d128_rows.tex",
    ),
    (
        "fp4_fa4_hao_table_gb200_20260802/tables/hao_grid_macros.tex",
        "generated/hao_grid_macros.tex",
    ),
    (
        "fp4_fa4_reconstruction_20260805/tables/reconstruction_rows.tex",
        "generated/reconstruction_rows.tex",
    ),
    (
        "fp4_fa4_unified_20260801/tables/downstream_main_rows.tex",
        "generated/downstream_main_rows.tex",
    ),
    (
        "fp4_fa4_unified_20260801/tables/downstream_margin_macros.tex",
        "generated/downstream_margin_macros.tex",
    ),
    (
        "fp4_fa4_unified_20260801/tables/downstream_nvnv_failure_rows.tex",
        "generated/downstream_nvnv_failure_rows.tex",
    ),
    (
        "fp4_fa4_unified_20260801/tables/full_format_rows.tex",
        "generated/full_format_rows.tex",
    ),
    (
        "fp4_fa4_unified_20260801/tables/p_range_rows.tex",
        "generated/p_range_rows.tex",
    ),
    (
        "fp4_fa4_unified_20260801/tables/unified_macros.tex",
        "generated/unified_macros.tex",
    ),
    (
        "fp4_fa4_wan_cute_bf16_20260806/tables/wan_quality_speed_rows.tex",
        "generated/wan_quality_speed_rows.tex",
    ),
    (
        "fp4_fa4_unified_20260801/figures/headline_pareto.pdf",
        "figures/headline_pareto.pdf",
    ),
    (
        "fp4_fa4_unified_20260801/figures/cross_shape_speed_accuracy.pdf",
        "figures/cross_shape_speed_accuracy.pdf",
    ),
    (
        "fp4_fa4_reconstruction_20260805/nvmx_fast_report.png",
        "figures/nvmx_fast_report.png",
    ),
)

PATH_REWRITES = (
    ("../fp4_fa4_unified_20260801/tables/", "generated/"),
    ("../fp4_fa4_hao_table_gb200_20260802/tables/", "generated/"),
    ("../fp4_fa4_b300_tuning_20260802/tables/", "generated/"),
    ("../fp4_fa4_reconstruction_20260805/tables/", "generated/"),
    ("../fp4_fa4_wan_cute_bf16_20260806/tables/", "generated/"),
    ("../fp4_fa4_unified_20260801/figures/", "figures/"),
    (
        "../fp4_fa4_reconstruction_20260805/nvmx_fast_report.png",
        "figures/nvmx_fast_report.png",
    ),
)

OVERLEAF_README = """# Overleaf build

- Main document: `main.tex`
- Compiler: pdfLaTeX
- TeX Live: 2023

The bibliography is `main.bib`; Overleaf will run BibTeX. Generated table
snippets and final figure assets are already vendored, so no shell escape,
Python step, external file, or network access is required.
"""

SAFE_MEMBER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
FORBIDDEN_SOURCE = re.compile(
    r"shell-escape|\\write18|\\immediate\s*\\write18|"
    r"\\usepackage(?:\[[^]]*\])?\{minted\}|\\inputminted",
    re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"aws_secret_access_key", re.IGNORECASE),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], cwd: Path, *, env: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        tail = completed.stdout[-12_000:]
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{tail}"
        )
    return completed.stdout


def _tool(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"required build tool is unavailable: {name}")
    return resolved


def _build_env(texmf_home: Path) -> dict[str, str]:
    texmf_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "FORCE_SOURCE_DATE": "1",
            "SOURCE_DATE_EPOCH": str(SOURCE_DATE_EPOCH),
            "TZ": "UTC",
            "TEXMFHOME": str(texmf_home),
            "openin_any": "p",
        }
    )
    return env


def _copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"required regular file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _materialize_common(destination: Path) -> None:
    local_files = (
        "main.tex",
        "main.bib",
        "graphcore_report.sty",
        "assets/graphcore-symbol.png",
        *SECTIONS,
        *APPENDICES,
        *LOCAL_FIGURES,
    )
    for relative in local_files:
        _copy_regular(PAPER_DIR / relative, destination / relative)
    for source, relative in VENDORED_FILES:
        _copy_regular(RESULTS_DIR / source, destination / relative)

    for source in destination.rglob("*"):
        if source.suffix not in {".tex", ".sty"} or not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        for old, new in PATH_REWRITES:
            text = text.replace(old, new)
        source.write_text(text, encoding="utf-8", newline="\n")

    unresolved = []
    for source in destination.rglob("*"):
        if source.suffix not in {".tex", ".sty"} or not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        if "../fp4_fa4_" in text:
            unresolved.append(source.relative_to(destination).as_posix())
    if unresolved:
        raise RuntimeError(f"unresolved sibling dependencies: {unresolved}")


def _source_entries(root: Path) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for source in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = source.relative_to(root).as_posix()
        entries[relative] = source.read_bytes()
    return entries


def _check_members(entries: dict[str, bytes], *, kind: str) -> None:
    expected_count = 47
    if len(entries) != expected_count:
        raise RuntimeError(
            f"{kind}: expected {expected_count} files, found {len(entries)}"
        )
    for name, payload in entries.items():
        pure = PurePosixPath(name)
        if (
            not SAFE_MEMBER.fullmatch(name)
            or pure.is_absolute()
            or ".." in pure.parts
            or any(part.startswith(".") for part in pure.parts)
            or len(name) > 160
            or any(len(part) > 80 for part in pure.parts)
        ):
            raise RuntimeError(f"{kind}: unsafe archive member name: {name}")
        if not payload:
            raise RuntimeError(f"{kind}: empty archive member: {name}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(payload):
                raise RuntimeError(f"{kind}: secret-like content in {name}")

    text = "\n".join(
        payload.decode("utf-8")
        for name, payload in entries.items()
        if Path(name).suffix in {".tex", ".sty", ".bbl", ".bib"}
    )
    match = FORBIDDEN_SOURCE.search(text)
    if match:
        raise RuntimeError(
            f"{kind}: forbidden shell-escape source token: {match.group(0)}"
        )


def _generate_bbl(root: Path, env: dict[str, str]) -> bytes:
    _run(
        [
            _tool("pdflatex"),
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-recorder",
            "main.tex",
        ],
        root,
        env=env,
    )
    bibtex_output = _run([_tool("bibtex"), "main"], root, env=env)
    if "Warning--" in bibtex_output:
        raise RuntimeError(
            f"BibTeX warning while generating main.bbl:\n{bibtex_output}"
        )
    bbl = root / "main.bbl"
    if not bbl.is_file() or not bbl.stat().st_size:
        raise RuntimeError("BibTeX did not create main.bbl")
    canonical_bbl = PAPER_DIR / "main.bbl"
    if canonical_bbl.exists() and canonical_bbl.read_bytes() != bbl.read_bytes():
        raise RuntimeError(
            "generated main.bbl does not match the canonical build output"
        )
    return bbl.read_bytes()


def _write_tar_gz(path: Path, entries: dict[str, bytes]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            fileobj=raw,
            mode="wb",
            compresslevel=9,
            mtime=SOURCE_DATE_EPOCH,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                for name in sorted(entries):
                    payload = entries[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    info.mtime = SOURCE_DATE_EPOCH
                    archive.addfile(info, io.BytesIO(payload))
    temporary.replace(path)


def _write_zip(path: Path, entries: dict[str, bytes]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    timestamp = dt.datetime.fromtimestamp(SOURCE_DATE_EPOCH, tz=dt.timezone.utc)
    date_time = (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=date_time)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, entries[name], compresslevel=9)
    temporary.replace(path)


def _extract_tar(path: Path, destination: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if not member.isfile():
                raise RuntimeError(
                    f"arXiv archive has non-regular member: {member.name}"
                )
        archive.extractall(destination, filter="data")
        return sorted(member.name for member in members)


def _extract_zip(path: Path, destination: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        archive.extractall(destination)
        return sorted(names)


def _compile(root: Path, *, with_bibtex: bool, env: dict[str, str]) -> Path:
    command = [
        _tool("pdflatex"),
        "-no-shell-escape",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-recorder",
        "main.tex",
    ]
    _run(command, root, env=env)
    if with_bibtex:
        output = _run([_tool("bibtex"), "main"], root, env=env)
        if "Warning--" in output:
            raise RuntimeError(f"BibTeX warning in clean build:\n{output}")
    _run(command, root, env=env)
    _run(command, root, env=env)
    log = (root / "main.log").read_text(encoding="utf-8", errors="replace")
    failures = (
        "Warning:",
        "Overfull \\hbox",
        "Overfull \\vbox",
        "There were undefined references",
        "Citation `",
        "Reference `",
        "Missing character:",
    )
    found = [token for token in failures if token in log]
    if found:
        raise RuntimeError(f"clean build has forbidden log diagnostics: {found}")
    return root / "main.pdf"


def _system_tex_roots(env: dict[str, str]) -> tuple[Path, ...]:
    roots = {
        Path("/etc/texmf"),
        Path("/usr/share/fonts"),
        Path("/usr/share/texlive"),
        Path("/usr/share/texmf"),
        Path("/var/lib/texmf"),
    }
    for variable in (
        "TEXMFROOT",
        "TEXMFDIST",
        "TEXMFLOCAL",
        "TEXMFSYSCONFIG",
        "TEXMFSYSVAR",
    ):
        value = _run([_tool("kpsewhich"), f"-var-value={variable}"], PAPER_DIR, env=env)
        value = value.strip()
        if value.startswith("/"):
            roots.add(Path(value).resolve())
    return tuple(sorted(roots, key=lambda item: str(item)))


def _is_beneath(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _check_fls(root: Path, env: dict[str, str]) -> None:
    system_roots = _system_tex_roots(env)
    fls = root / "main.fls"
    if not fls.is_file():
        raise RuntimeError("pdflatex did not produce main.fls")
    outside = []
    for line in fls.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("INPUT "):
            continue
        value = line.removeprefix("INPUT ")
        candidate = Path(value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if not _is_beneath(resolved, (root.resolve(), *system_roots)):
                outside.append(value)
        elif ".." in candidate.parts:
            outside.append(value)
    if outside:
        raise RuntimeError(
            f"clean build reached outside archive/system roots: {outside}"
        )


def _pdfinfo(path: Path) -> str:
    return _run([_tool("pdfinfo"), str(path)], path.parent, env=os.environ.copy())


def _check_no_javascript(path: Path) -> None:
    info = _pdfinfo(path)
    match = re.search(r"^JavaScript:\s+(\S+)", info, flags=re.MULTILINE)
    if not match or match.group(1).lower() != "no":
        raise RuntimeError(f"could not verify JavaScript-free PDF: {path}")


def _check_pdf(
    path: Path, *, reference_text: bytes, expected_pages: int
) -> dict[str, object]:
    info = _pdfinfo(path)
    page_match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
    if not page_match or int(page_match.group(1)) != expected_pages:
        raise RuntimeError(f"unexpected page count in {path}")
    _check_no_javascript(path)
    font_output = _run(
        [_tool("pdffonts"), str(path)], path.parent, env=os.environ.copy()
    )
    font_rows = [line for line in font_output.splitlines()[2:] if line.strip()]
    unembedded = []
    type3 = 0
    for row in font_rows:
        fields = row.split()
        if len(fields) < 6 or fields[-5].lower() != "yes":
            unembedded.append(row)
        if "Type 3" in row:
            type3 += 1
    if unembedded:
        raise RuntimeError(f"unembedded fonts in {path}: {unembedded}")
    if type3:
        raise RuntimeError(f"Type-3 fonts remain in {path}: {type3} font rows")
    text = subprocess.run(
        [_tool("pdftotext"), str(path), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    if text != reference_text:
        raise RuntimeError(f"PDF text differs from canonical main.pdf: {path}")
    return {
        "pages": expected_pages,
        "pdf_sha256": _sha256_file(path),
        "text_sha256": _sha256_bytes(text),
        "font_rows": len(font_rows),
        "type3_font_rows": type3,
        "all_fonts_embedded": True,
        "javascript": False,
    }


def _validate_figure_pdfs(entries: dict[str, bytes], scratch: Path) -> None:
    figure_root = scratch / "figure-pdf-checks"
    figure_root.mkdir()
    for name, payload in entries.items():
        if not name.startswith("figures/") or not name.endswith(".pdf"):
            continue
        target = figure_root / Path(name).name
        target.write_bytes(payload)
        _check_no_javascript(target)


def _archive_manifest(entries: dict[str, bytes]) -> str:
    return "".join(
        f"{_sha256_bytes(entries[name])}  {name}\n" for name in sorted(entries)
    )


def _git_head() -> str:
    return _run(["git", "rev-parse", "HEAD"], PAPER_DIR, env=os.environ.copy()).strip()


def build() -> dict[str, object]:
    for tool in ("bibtex", "kpsewhich", "pdffonts", "pdfinfo", "pdflatex", "pdftotext"):
        _tool(tool)
    reference_pdf = PAPER_DIR / "main.pdf"
    if not reference_pdf.is_file():
        raise RuntimeError(
            "canonical main.pdf is missing; run the manuscript build first"
        )
    reference_info = _pdfinfo(reference_pdf)
    if not re.search(
        rf"^Pages:\s+{EXPECTED_PAGES}$", reference_info, flags=re.MULTILINE
    ):
        raise RuntimeError("canonical main.pdf is not the expected 56-page manuscript")
    reference_text = subprocess.run(
        [_tool("pdftotext"), str(reference_pdf), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fa4-submission-") as temporary:
        scratch = Path(temporary)
        common = scratch / "common"
        common.mkdir()
        _materialize_common(common)
        env = _build_env(scratch / "texmf-home")
        bbl = _generate_bbl(common, env)

        common_names = {
            "main.tex",
            "main.bib",
            "graphcore_report.sty",
            "assets/graphcore-symbol.png",
            *SECTIONS,
            *APPENDICES,
            *LOCAL_FIGURES,
            *(destination for _, destination in VENDORED_FILES),
        }
        staged_entries = _source_entries(common)
        missing = sorted(common_names - staged_entries.keys())
        if missing:
            raise RuntimeError(
                f"verified common allowlist files are missing: {missing}"
            )
        # Select by exact name so bibliography-generation products such as
        # main.pdf cannot enter either archive merely because their type is valid.
        common_entries = {name: staged_entries[name] for name in common_names}
        if len(common_entries) != 46:
            raise RuntimeError(
                f"verified common allowlist drifted: expected 46, found {len(common_entries)}"
            )
        arxiv_entries = {**common_entries, "main.bbl": bbl}
        overleaf_entries = {
            **common_entries,
            "README.md": OVERLEAF_README.encode("utf-8"),
        }
        _check_members(arxiv_entries, kind="arXiv")
        _check_members(overleaf_entries, kind="Overleaf")
        if "main.pdf" in arxiv_entries or "main.pdf" in overleaf_entries:
            raise RuntimeError("compiled main.pdf must not enter either source archive")
        if "main.bbl" not in arxiv_entries or "main.bbl" in overleaf_entries:
            raise RuntimeError("archive-specific bibliography policy was violated")
        if "README.md" in arxiv_entries or "README.md" not in overleaf_entries:
            raise RuntimeError("archive-specific README policy was violated")
        _validate_figure_pdfs(arxiv_entries, scratch)

        arxiv_path = OUTPUT_DIR / ARXIV_NAME
        overleaf_path = OUTPUT_DIR / OVERLEAF_NAME
        _write_tar_gz(arxiv_path, arxiv_entries)
        _write_zip(overleaf_path, overleaf_entries)
        if arxiv_path.stat().st_size > ARXIV_MAX_ARCHIVE_BYTES:
            raise RuntimeError(
                f"arXiv archive exceeds {ARXIV_MAX_ARCHIVE_BYTES} bytes: "
                f"{arxiv_path.stat().st_size}"
            )

        checks: dict[str, object] = {}
        for kind, archive_path, expected, extractor, with_bibtex in (
            ("arxiv", arxiv_path, arxiv_entries, _extract_tar, False),
            ("overleaf", overleaf_path, overleaf_entries, _extract_zip, True),
        ):
            extracted = scratch / f"extract-{kind}"
            extracted.mkdir()
            names = extractor(archive_path, extracted)
            if names != sorted(expected):
                raise RuntimeError(f"{kind}: archive members differ from allowlist")
            build_env = _build_env(scratch / f"texmf-home-{kind}")
            pdf = _compile(extracted, with_bibtex=with_bibtex, env=build_env)
            _check_fls(extracted, build_env)
            checks[kind] = _check_pdf(
                pdf, reference_text=reference_text, expected_pages=EXPECTED_PAGES
            )

        tex_version = _run(
            [_tool("pdflatex"), "--version"], PAPER_DIR, env=os.environ.copy()
        ).splitlines()[0]
        if "TeX Live 2023" not in _run(
            [_tool("pdflatex"), "--version"], PAPER_DIR, env=os.environ.copy()
        ):
            raise RuntimeError(
                "submission archives must be validated with the documented TeX Live 2023"
            )
        receipt = {
            "schema_version": 1,
            "canonical_git_head": _git_head(),
            "source_date_epoch": SOURCE_DATE_EPOCH,
            "tex_engine": tex_version,
            "expected_pages": EXPECTED_PAGES,
            "canonical_pdf_sha256": _sha256_file(reference_pdf),
            "canonical_text_sha256": _sha256_bytes(reference_text),
            "archives": {
                "arxiv": {
                    "path": ARXIV_NAME,
                    "bytes": arxiv_path.stat().st_size,
                    "sha256": _sha256_file(arxiv_path),
                    "members": len(arxiv_entries),
                    "contains_main_bib": True,
                    "contains_main_bbl": True,
                    "contains_main_pdf": False,
                    "contains_readme": False,
                    "clean_build": checks["arxiv"],
                },
                "overleaf": {
                    "path": OVERLEAF_NAME,
                    "bytes": overleaf_path.stat().st_size,
                    "sha256": _sha256_file(overleaf_path),
                    "members": len(overleaf_entries),
                    "contains_main_bib": True,
                    "contains_main_bbl": False,
                    "contains_main_pdf": False,
                    "contains_readme": True,
                    "clean_build": checks["overleaf"],
                },
            },
        }
        (OUTPUT_DIR / "arxiv-manifest.sha256").write_text(
            _archive_manifest(arxiv_entries), encoding="utf-8", newline="\n"
        )
        (OUTPUT_DIR / "overleaf-manifest.sha256").write_text(
            _archive_manifest(overleaf_entries), encoding="utf-8", newline="\n"
        )
        (OUTPUT_DIR / "verification.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (OUTPUT_DIR / "SHA256SUMS").write_text(
            f"{_sha256_file(arxiv_path)}  {ARXIV_NAME}\n"
            f"{_sha256_file(overleaf_path)}  {OVERLEAF_NAME}\n",
            encoding="utf-8",
            newline="\n",
        )
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    receipt = build()
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
