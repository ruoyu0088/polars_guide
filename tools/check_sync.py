#!/usr/bin/env python3
"""日⇄中チェック（polars_guide / polars_guide_cn）.

構造（セル数・セル種別の並び・コードセル数）が一致しているか、
中国語版に仮名が残っていないか、未実行セルや error 出力がないかを見る。

usage:
    python tools/check_sync.py                 # 全章
    python tools/check_sync.py 005_polars_vs_pandas ...
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbformat

ROOT = Path(__file__).resolve().parent.parent
JA = ROOT / "polars_guide"
CN = ROOT / "polars_guide_cn"
KANA = re.compile(r"[\u3040-\u30ff]")


def src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def check(name: str) -> bool:
    ja_path, cn_path = JA / f"{name}.ipynb", CN / f"{name}.ipynb"
    if not ja_path.exists() or not cn_path.exists():
        print(f"{name:46} MISSING ({'ja' if not ja_path.exists() else 'cn'})")
        return False
    ja = nbformat.read(ja_path, as_version=4)
    cn = nbformat.read(cn_path, as_version=4)

    ja_types = [c["cell_type"] for c in ja.cells]
    cn_types = [c["cell_type"] for c in cn.cells]
    problems = []
    if len(ja.cells) != len(cn.cells):
        problems.append(f"cells {len(ja.cells)}!={len(cn.cells)}")
    if ja_types != cn_types:
        problems.append("cell-type sequence differs")

    kana_md = sum(len(KANA.findall(src(c))) for c in cn.cells if c["cell_type"] == "markdown")
    if kana_md:
        problems.append(f"kana(md)={kana_md}")

    # コードセル内の仮名はデータ値（都道府県名など）として残ることがあるので、警告に留める
    kana_code_cells = [i for i, c in enumerate(cn.cells)
                       if c["cell_type"] == "code" and KANA.search(src(c))]

    unexec = [i for i, c in enumerate(cn.cells)
              if c["cell_type"] == "code" and src(c).strip() and c.get("execution_count") is None]
    if unexec:
        problems.append(f"unexecuted={unexec[:5]}")

    errors = []
    for i, c in enumerate(cn.cells):
        if c["cell_type"] != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors.append((i, o.get("ename")))
    if errors:
        problems.append(f"errors={errors[:5]}")

    ja_code = sum(1 for c in ja.cells if c["cell_type"] == "code")
    cn_code = sum(1 for c in cn.cells if c["cell_type"] == "code")
    if ja_code != cn_code:
        problems.append(f"code cells {ja_code}!={cn_code}")

    status = "OK" if not problems else "!! " + "; ".join(problems)
    if kana_code_cells:
        status += f"  (kana in code cells: {kana_code_cells})"
    print(f"{name:46} {status}")
    return not problems


if __name__ == "__main__":
    names = sys.argv[1:]
    if not names:
        toc = (JA / "_toc.yml").read_text(encoding="utf-8")
        names = [m.strip() for m in re.findall(r"-\s*file:\s*(\S+)", toc)]
    ok = all([check(n) for n in names])
    sys.exit(0 if ok else 1)
