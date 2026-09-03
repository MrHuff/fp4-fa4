#!/usr/bin/env python3
"""Insert PTX 9.3's mma_throughput pragma into selected entry functions."""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_entry(lines: list[str], symbol: str) -> None:
    entry = f".visible .entry {symbol}("
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(entry))
    except StopIteration as exc:
        raise RuntimeError(f"PTX entry not found: {symbol}") from exc

    for index in range(start + 1, len(lines)):
        if lines[index].strip() == "{":
            pragma = '\t.pragma "mma_throughput";\n'
            if index + 1 < len(lines) and lines[index + 1] == pragma:
                return
            lines.insert(index + 1, pragma)
            return
        if lines[index].startswith(".visible .entry "):
            break
    raise RuntimeError(f"opening brace not found for PTX entry: {symbol}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("symbols", nargs="+")
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    for symbol in args.symbols:
        patch_entry(lines, symbol)
    args.output.write_text("".join(lines))


if __name__ == "__main__":
    main()
