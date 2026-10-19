#!/usr/bin/env python3
"""Cell-level editor for .ipynb files that keeps the file's line endings.

Programmatic use (a scratch script per chapter):

    from nbedit import load, save, md, code, sub, rename, dump

    nb = load("polars_guide/005_polars_vs_pandas.ipynb")
    md(nb, 3, "# 新しい見出し")          # replace whole cell source
    sub(nb, 7, "以下の", "次の")          # must match exactly once
    code(nb, 12, "df = pl.DataFrame(...)")  # replace code cell
    rename(nb, {"alist": "values", "ndarr": "array"})
    save(nb, path)

CLI:
    python tools/nbedit.py show <nb> <idx>
    python tools/nbedit.py list <nb>
    python tools/nbedit.py set  <nb> <idx> --file <txt>
    python tools/nbedit.py sub  <nb> <idx> --old <a> --new <b>
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat


class NB:
    def __init__(self, path: Path, data, crlf: bool):
        self.path, self.data, self.crlf = path, data, crlf

    @property
    def cells(self):
        return self.data.cells


def load(path) -> NB:
    path = Path(path)
    raw = path.read_bytes()
    nb = nbformat.read(path, as_version=4)
    return NB(path, nb, b"\r\n" in raw)


def save(nb: NB) -> None:
    text = nbformat.writes(nb.data, version=4)
    if text.endswith("\n"):
        text = text[:-1]
    if nb.crlf:
        nb.path.write_bytes(text.replace("\n", "\r\n").encode("utf-8") + b"\r\n")
    else:
        nb.path.write_bytes(text.encode("utf-8") + b"\n")


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _set(cell, text: str) -> None:
    cell["source"] = text


def md(nb: NB, idx: int, text: str) -> None:
    c = nb.cells[idx]
    assert c["cell_type"] == "markdown", f"cell {idx} is {c['cell_type']}"
    _set(c, text)


def code(nb: NB, idx: int, text: str) -> None:
    c = nb.cells[idx]
    assert c["cell_type"] == "code", f"cell {idx} is {c['cell_type']}"
    _set(c, text)


def set_any(nb: NB, idx: int, text: str) -> None:
    _set(nb.cells[idx], text)


def src(nb: NB, idx: int) -> str:
    return _src(nb.cells[idx])


def sub(nb: NB, idx: int, old: str, new: str, count: int = 1) -> None:
    c = nb.cells[idx]
    text = _src(c)
    n = text.count(old)
    if n != count:
        raise AssertionError(f"cell {idx}: expected {count} occurrence(s) of {old!r}, found {n}")
    _set(c, text.replace(old, new))


def sub_all(nb: NB, old: str, new: str) -> int:
    """Replace across every cell; returns the number of cells touched."""
    touched = 0
    for c in nb.cells:
        text = _src(c)
        if old in text:
            _set(c, text.replace(old, new))
            touched += 1
    return touched


def insert_md(nb: NB, idx: int, text: str) -> None:
    nb.cells.insert(idx, {"cell_type": "markdown", "metadata": {}, "source": text})


def insert_code(nb: NB, idx: int, text: str) -> None:
    nb.cells.insert(idx, {"cell_type": "code", "metadata": {}, "source": text,
                          "outputs": [], "execution_count": None})


def delete(nb: NB, idx: int) -> None:
    del nb.cells[idx]


def rename(nb: NB, mapping: dict[str, str], types=("code",)) -> int:
    """Rename identifiers across cell sources with word-boundary regex."""
    import re

    touched = 0
    for c in nb.cells:
        if c["cell_type"] not in types:
            continue
        text = _src(c)
        new = text
        for old, rep in mapping.items():
            new = re.sub(rf"\b{re.escape(old)}\b", rep, new)
        if new != text:
            _set(c, new)
            touched += 1
    return touched


def stats(nb: NB) -> dict:
    code_cells = [c for c in nb.cells if c["cell_type"] == "code"]
    lines = comments = 0
    for c in code_cells:
        for line in _src(c).split("\n"):
            s = line.strip()
            if not s:
                continue
            lines += 1
            if s.startswith("#"):
                comments += 1
    return {
        "cells": len(nb.cells),
        "code_cells": len(code_cells),
        "code_lines": lines,
        "comment_lines": comments,
        "comment_pct": round(100 * comments / max(lines, 1), 1),
    }


if __name__ == "__main__":
    cmd = sys.argv[1]
    nb = load(sys.argv[2])
    if cmd == "list":
        for i, c in enumerate(nb.cells):
            head = _src(c).splitlines()
            print(f"[{i:3}] {c['cell_type']:8} {head[0][:90] if head else ''}")
    elif cmd == "show":
        print(_src(nb.cells[int(sys.argv[3])]))
    elif cmd == "set":
        text = Path(sys.argv[sys.argv.index("--file") + 1]).read_text(encoding="utf-8")
        set_any(nb, int(sys.argv[3]), text.rstrip("\n"))
        save(nb)
        print("ok")
    elif cmd == "sub":
        old = sys.argv[sys.argv.index("--old") + 1]
        new = sys.argv[sys.argv.index("--new") + 1]
        sub(nb, int(sys.argv[3]), old, new)
        save(nb)
        print("ok")
    else:
        raise SystemExit(f"unknown command {cmd}")
