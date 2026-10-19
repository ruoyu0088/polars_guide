#!/usr/bin/env python3
"""打印 zh_style_lint 实际统计的句子及其汉字长度，用于精确调平。"""
import json
import re
import sys
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
HERE = Path(__file__).resolve().parent.parent
CN = HERE / "polars_guide_cn"


def sents(name):
    nb = json.loads((CN / f"{name}.ipynb").read_text(encoding="utf-8"))
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        cell = "".join(c["source"])
        cell = re.sub(r"```.*?```", "", cell, flags=re.S)
        cur = []
        for line in cell.split("\n"):
            t = line.strip()
            if not t or t[0] in "#|>-*!<$`" or re.match(r"^\d+\.", t):
                if cur:
                    out.append((i, " ".join(cur)))
                    cur = []
                continue
            cur.append(t)
        if cur:
            out.append((i, " ".join(cur)))
    res = []
    for i, p in out:
        if len(CJK.findall(p)) < 15:
            continue
        for s in re.split(r"(?<=[。！？])", p):
            if len(CJK.findall(s)) >= 6:
                res.append((i, len(CJK.findall(s)), s.strip()))
    return res


for name in sys.argv[1:]:
    rs = sents(name)
    lens = [r[1] for r in rs]
    print(f"=== {name}  句数 {len(lens)}  均值 {sum(lens)/len(lens):.1f}")
    for i, n, s in rs:
        flag = "L" if n >= 45 else ("s" if n <= 12 else " ")
        print(f"  [{flag}] cell{i:3d} {n:3d}  {s[:90]}")
