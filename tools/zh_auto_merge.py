#!/usr/bin/env python3
"""把相邻两句用「；」合成一句，直到长句比例进带。

只做一件事：在同一段里挑两句最短、合计仍不超过 62 个汉字的相邻句，
把中间的句号换成「；」。合成后若均值超上限就停手。
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("zl", TOOLS / "zh_style_lint.py")
zl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zl)

CJK = re.compile(r"[\u4e00-\u9fff]")
CN = TOOLS.parent / "polars_guide_cn"


def paragraphs(nb):
    """返回 [(cell_idx, [sentence, ...]), ...]，与 lint 的口径一致。"""
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        cell = re.sub(r"```.*?```", "", "".join(c["source"]), flags=re.S)
        cur = []
        for line in cell.split("\n"):
            s = line.strip()
            if not s or s[0] in "#|>-*!<$`" or re.match(r"^\d+\.", s):
                if cur:
                    out.append((i, cur))
                    cur = []
                continue
            cur.append(s)
        if cur:
            out.append((i, cur))
    return out


def sentence_list(paras):
    rs = []
    for _, lines in paras:
        text = " ".join(lines)
        for s in re.split(r"(?<=[。！？])", text):
            n = len(CJK.findall(s))
            if n >= 6:
                rs.append(n)
    return rs


def try_merge(nb, want_short=False):
    """找一对相邻短句合并，成功返回 (cell_idx, 旧文本片段, 新文本片段)，否则 None。"""
    best = None
    for i, lines in paragraphs(nb):
        text = " ".join(lines)
        parts = re.split(r"(?<=[。])", text)
        for k in range(len(parts) - 1):
            a, b = parts[k].strip(), parts[k + 1].strip()
            na, nb_ = len(CJK.findall(a)), len(CJK.findall(b))
            if na < 6 or nb_ < 6:
                continue
            total = na + nb_
            if want_short:
                # 优先消灭短句：至少一句是短句，合起来不超过 80 字
                if na > 12 and nb_ > 12:
                    continue
                if total > 80:
                    continue
                score = min(na, nb_)
            else:
                if na >= 45 or nb_ >= 45 or total > 72 or total < 45:
                    continue
                score = abs(total - 52)
            if best is None or score < best[0]:
                best = (score, i, a + b, a + "；" + b)
    if not best:
        return None
    _, i, old, new = best
    cell = nb["cells"][i]
    src = "".join(cell["source"])
    # 原文里两句之间可能是换行，统一用正则找
    pat = re.escape(old)
    pat = pat.replace(re.escape("。"), r"。\s*")
    m = re.search(pat, src)
    if not m:
        return None
    merged = old.replace("。", "；", 1) if old.count("。") == 1 else old[: old.rfind("。")] + "；" + old[old.rfind("。") + 1 :]
    cell["source"] = src[: m.start()] + merged + src[m.end():]
    return i, old, merged


def main():
    for name in sys.argv[1:]:
        p = CN / f"{name}.ipynb"
        raw = p.read_text(encoding="utf-8")
        crlf = "\r\n" in raw
        nb = json.loads(raw)
        for step in range(14):
            ls = sentence_list(paragraphs(nb))
            long_ratio = sum(1 for x in ls if x >= 45) / len(ls)
            short_ratio = sum(1 for x in ls if x <= 12) / len(ls)
            mean = sum(ls) / len(ls)
            want_short = short_ratio > 0.10
            if not want_short and long_ratio >= 0.11:
                break
            if mean > 32.0:
                print(f"{name}: 停手，均值已达 {mean:.1f}")
                break
            # 预估合并后的均值：句数减一、总字数不变
            after = sum(ls) / (len(ls) - 1) if len(ls) > 1 else mean
            if after > 32.6:
                print(f"{name}: 停手，再合并均值会到 {after:.1f}")
                break
            got = try_merge(nb, want_short)
            if not got:
                print(f"{name}: 找不到可合并的相邻句")
                break
            print(f"{name}: 合并 cell{got[0]}  {len(CJK.findall(got[1]))} 字")
        out = json.dumps(nb, ensure_ascii=False, indent=1)
        if crlf:
            out = out.replace("\n", "\r\n")
        p.write_bytes((out + ("\r\n" if crlf else "\n")).encode("utf-8"))


if __name__ == "__main__":
    main()
