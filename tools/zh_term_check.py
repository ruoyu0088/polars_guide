#!/usr/bin/env python3
"""术语一致性检查：同一概念在全书里只允许一种写法。

每组里第一个是推荐写法，其余是必须清零的异形写法。
用法：python tools/zh_term_check.py        全章扫描
      python tools/zh_term_check.py --fix  自动把异形改成推荐写法
"""
import json
import sys
from pathlib import Path

CN = Path(__file__).resolve().parent.parent / "polars_guide_cn"

GROUPS = [
    ("数据框", ["数据帧", "数据表单", "数据表框"]),
    ("合并", ["结合"]),                      # 040 的 concat 用法统一到合并
    ("键列", ["连接键", "键字段"]),
    ("即时执行", ["立即执行"]),
    ("表达式", ["运算式", "演算式"]),
    ("分组", ["分群"]),
    ("缺失值", ["空值", "缺值"]),
    ("数据类型", ["数据型"]),
    ("流式处理", ["流处理", "流式运算"]),
    ("上位类型", ["父类型", "超类型"]),
]

# 「结合代码与输出」这类合法用法，替换时跳过
WHITELIST = ["结合代码", "结合输出", "结合实例", "结合例子"]


import re as _re
_FLAT = _re.compile(r"[ 	]*[；。][ 	]*(?:\*|-|\d+\.)[ 	]+")


def scan_flat():
    """正文行内出现列表标记，说明列表被并成了一段。"""
    out = []
    for p in sorted(CN.glob("*.ipynb")):
        if p.stem.startswith("2"):
            continue
        nb = json.loads(p.read_text(encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "markdown":
                continue
            if _FLAT.search("".join(c["source"])):
                out.append((p.stem, i))
    return out


def scan_capture():
    """%%capture_except 只该用在会报错的演示格；不报错还包着就是噪音。"""
    out = []
    for book in ("polars_guide_cn", "polars_guide"):
        for p in sorted((CN.parent / book).glob("*.ipynb")):
            if p.stem.startswith("2"):
                continue
            nb = json.loads(p.read_text(encoding="utf-8"))
            for i, c in enumerate(nb["cells"]):
                if not "".join(c["source"]).startswith("%%capture_except"):
                    continue
                txt = ""
                for o in (c.get("outputs") or []):
                    t = o.get("text")
                    txt += "".join(t) if isinstance(t, list) else (t or "")
                if not _re.match(r"\w*Error:", txt.strip()):
                    out.append((book, p.stem, i))
    return out


def scan(fix=False):
    hits = []
    for p in sorted(CN.glob("*.ipynb")):
        if p.stem[:3].isdigit() and p.stem.startswith("2"):
            continue   # 100knocks 不在这次范围内
        raw = p.read_text(encoding="utf-8")
        crlf = "\r\n" in raw
        nb = json.loads(raw)
        changed = False
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "markdown":
                continue
            t = "".join(c["source"])
            orig = t
            for good, bads in GROUPS:
                for bad in bads:
                    if bad not in t:
                        continue
                    n = t.count(bad) - sum(t.count(wl) for wl in WHITELIST if bad in wl)
                    if n <= 0:
                        continue
                    hits.append((p.stem, i, bad, good, n))
                    if fix:
                        for wl in WHITELIST:
                            t = t.replace(wl, "\x00" + wl)
                        t = t.replace(bad, good)
                        t = t.replace("\x00", "")
            if fix and t != orig:
                c["source"] = t
                changed = True
        if changed:
            out = json.dumps(nb, ensure_ascii=False, indent=1)
            if crlf:
                out = out.replace("\n", "\r\n")
            p.write_bytes((out + ("\r\n" if crlf else "\n")).encode("utf-8"))
    return hits


def main():
    fix = "--fix" in sys.argv
    hits = scan(fix)
    cap = scan_capture()
    for book, ch, i in cap:
        print(f"{ch:40} cell{i:3d}  %%capture_except 包着不报错的代码 ({book})")
    flat = scan_flat()
    for ch, i in flat:
        print(f"{ch:40} cell{i:3d}  列表被压平成一段")
    if not hits and not flat and not cap:
        print("术语一致，且没有压平的列表")
        return
    for ch, i, bad, good, n in hits:
        print(f"{ch:40} cell{i:3d}  {bad} x{n} -> {good}")
    print(f"\n合计 {sum(h[4] for h in hits)} 处异形写法" + ("，已替换" if fix else ""))


if __name__ == "__main__":
    main()
