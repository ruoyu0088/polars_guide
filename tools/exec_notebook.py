#!/usr/bin/env python3
"""polars_guide 用：在本地执行 notebook 并把输出写回。

用法:
    python tools/exec_notebook.py polars_guide/005_polars_vs_pandas.ipynb [more ...]

必须用 C:\\pixi_env 的 python（polars 2.0 / hvplot / geoviews / pyarrow 都在里面）。

写回规则（与 scipybook 一致，避免无意义 diff）:
- 删掉 nbclient 塞进 cell metadata 的 execution 时间戳
- 保持文件原来的行尾（CRLF / LF）
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

_ENV_BIN = Path(sys.executable).resolve().parent / "Library" / "bin"
if _ENV_BIN.exists():
    import os
    os.environ["PATH"] = str(_ENV_BIN) + os.pathsep + os.environ.get("PATH", "")


def _write_like(original_bytes: bytes, nb, path: Path) -> None:
    text = nbformat.writes(nb, version=4)
    if text.endswith("\n"):
        text = text[:-1]
    if b"\r\n" in original_bytes:
        path.write_bytes(text.replace("\n", "\r\n").encode("utf-8") + b"\r\n")
    else:
        path.write_bytes(text.encode("utf-8") + b"\n")


def run(path: Path) -> bool:
    original = path.read_bytes()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=True,
    )
    client.execute()

    for cell in nb.cells:
        cell.get("metadata", {}).pop("execution", None)
    _write_like(original, nb, path)

    failed = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                failed += 1
                head = "".join(cell.source).splitlines()
                head = head[0][:60] if head else ""
                print(f"  [{path.name}] cell {i}: {out.get('ename')}: {out.get('evalue')}  <- {head}")
    print(f"{path.name}: {'OK' if not failed else f'{failed} ERROR cell(s)'}")
    return failed == 0


if __name__ == "__main__":
    ok = True
    for arg in sys.argv[1:]:
        ok &= run(Path(arg))
    sys.exit(0 if ok else 1)
