#!/usr/bin/env python3
"""把「读起来不顺」的可疑句挑出来并排序，供人工逐句复核。

不判断好坏，只按可疑度打分：每命中一条规则加权重，最后按分数排序。
用法：python tools/zh_flow_lint.py [章节名 ...]   不给参数则检查全书
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CN = re.compile(r"[\u4e00-\u9fff]")
SENT = re.compile(r"[^。！？\n]+[。！？]?")
CODE = re.compile(r"`[^`]*`")
MASK = "▮"

COLLOQUIAL = ["东西", "省事", "顺手", "搞定", "一眼", "一堆", "咋", "啥", "老是", "挺"]
MACHINE = ["使得", "从而", "以便", "由于其", "将会", "能够", "某种程度上", "大致上"]
VERBISH = set("是有用把让会能要可需应不没为在对从使得做算给取放跑写读变成等及则都也还就才加减除乘决定列指说明称含记录引带保换选")
JARGON = re.compile(r"(多写|多用|多一次|多一步)[^，。]{0,6}(join|连接|map|filter|over|cast)")

RULES = {
    "的字堆叠": 3,      # 汉字里「的」占比过高
    "无谓语句": 3,      # 只有名词堆叠，没有谓语性成分
    "中英缺空格": 2,    # Polars的底层代码
    "口语词": 2,
    "机翻腔": 2,
    "逗号过多": 2,      # ≥5 个逗号，从句套太多
    "顿号过多": 2,      # ≥6 个顿号，读成流水账
    "连接无对象": 3,    # 说 join 却不说连哪两张表
    "术语当动词": 3,    # 「多写一次 join」这类
    "括号补注": 2,      # 正文里 (...) 塞整句
    "连续碎句": 2,      # 连续 3 句都 <10 汉字
    "同词重复": 1,      # 同一实词在一句里出现 ≥3 次
}


def strip_code(text: str) -> str:
    """去掉代码块、表格、列表项、标题，正文里保留行内代码（判定时再掩掉）。"""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^\s*\|.*$", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+] .*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\. .*$", "", text, flags=re.M)
    text = re.sub(r"^\s*>.*$", "", text, flags=re.M)
    text = re.sub(r"^\s*\$.*$", "", text, flags=re.M)
    text = re.sub(r"^\s*#+ .*$", "", text, flags=re.M)
    text = "\n".join(line for line in text.split("\n") if "\\" not in line)
    return text


def sentences(text: str):
    out = []
    for m in SENT.finditer(text):
        s = m.group(0).strip()
        if len(CN.findall(CODE.sub("", s))) >= 4:
            out.append(s)
    return out


def check(raw: str, prev_han: int, prev2_han: int):
    """raw 保留行内代码，便于人读；判定用掩掉代码的版本。"""
    s = CODE.sub(MASK, raw)
    hits = []
    han = max(len(CN.findall(s)), 1)
    if han >= 16 and s.count("的") >= 4 and s.count("的") / han > 0.14:
        hits.append("的字堆叠")
    if han >= 14 and "：" not in s and not (set(s) & VERBISH):
        hits.append("无谓语句")
    if re.search(r"[\u4e00-\u9fff][A-Za-z]|[A-Za-z][\u4e00-\u9fff]", s):
        hits.append("中英缺空格")
    if any(w in s for w in COLLOQUIAL):
        hits.append("口语词")
    if any(w in s for w in MACHINE):
        hits.append("机翻腔")
    if s.count("，") >= 5:
        hits.append("逗号过多")
    if s.count("、") >= 6:
        hits.append("顿号过多")
    if ("join" in s or "连接" in s) and not (set(s) & set("表键列行")):
        hits.append("连接无对象")
    if JARGON.search(s):
        hits.append("术语当动词")
    for m in re.finditer(r"\(([^)]*)\)", s):
        if len(CN.findall(m.group(1))) >= 10:
            hits.append("括号补注")
    if han < 10 and prev_han < 10 and prev2_han < 10:
        hits.append("连续碎句")
    words = [w for w in re.findall(r"[\u4e00-\u9fff]{2,4}", s)]
    for w in set(words):
        if words.count(w) >= 3:
            hits.append("同词重复")
            break
    return hits


def lint(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for idx, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        body = strip_code("".join(c["source"]))
        ss = sentences(body)
        hans = [len(CN.findall(CODE.sub("", s))) for s in ss]
        for k, s in enumerate(ss):
            prev_han = hans[k - 1] if k >= 1 else 99
            prev2_han = hans[k - 2] if k >= 2 else 99
            hits = check(s, prev_han, prev2_han)
            if hits:
                score = sum(RULES[h] for h in hits)
                rows.append((score, idx, hits, s))
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    root = Path(__file__).resolve().parent.parent / "polars_guide_cn"
    names = args or sorted(p.stem for p in root.glob("*.ipynb"))
    total = 0
    for name in names:
        rows = lint(root / f"{name}.ipynb")
        total += len(rows)
        if not rows:
            print(f"{name:44} 0 句可疑")
            continue
        rows.sort(key=lambda r: -r[0])
        print(f"\n=== {name}  可疑 {len(rows)} 句，最高分 {rows[0][0]}")
        for score, idx, hits, s in (rows if verbose else rows[:4]):
            print(f"  [{score}] cell{idx} {'/'.join(sorted(set(hits)))}")
            print(f"      {s[:110]}")
    print(f"\n合计可疑句：{total}")


if __name__ == "__main__":
    main()
