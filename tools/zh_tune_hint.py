#!/usr/bin/env python3
"""紧凑版：只列短句、最长句、口语/人称/问号命中，便于快速决定合并与拆分。"""
import json
import re
import sys
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]")
HERE = Path(__file__).resolve().parent.parent
CN = HERE / "polars_guide_cn"


def cells(name):
    nb = json.loads((CN / f"{name}.ipynb").read_text(encoding="utf-8"))
    return [("".join(c["source"]), i) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"]


def sents(name):
    out = []
    for text, i in cells(name):
        t = re.sub(r"```.*?```", "", text, flags=re.S)
        cur = []
        for line in t.split("\n"):
            s = line.strip()
            if not s or s[0] in "#|>-*!<$`" or re.match(r"^\d+\.", s):
                if cur:
                    out.append((i, " ".join(cur)))
                    cur = []
                continue
            cur.append(s)
        if cur:
            out.append((i, " ".join(cur)))
    rs = []
    for i, p in out:
        if len(CJK.findall(p)) < 15:
            continue
        for s in re.split(r"(?<=[。！？])", p):
            n = len(CJK.findall(s))
            if n >= 6:
                rs.append((i, n, s.strip()))
    return rs


for name in sys.argv[1:]:
    rs = sents(name)
    lens = [r[1] for r in rs]
    shorts = [r for r in rs if r[1] <= 12]
    longs = [r for r in rs if r[1] >= 45]
    mids = sorted(rs, key=lambda r: -r[1])[:3]
    print(f"=== {name}: {len(lens)} 句 均值 {sum(lens)/len(lens):.1f} 短 {len(shorts)} 长 {len(longs)}")
    for i, n, s in shorts:
        print(f"  短 c{i} {n:2d} | {s[:60]}")
    for i, n, s in mids:
        print(f"  长 c{i} {n:3d} | {s[:70]}")
