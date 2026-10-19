#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文版文风门禁，指标与 scipybook 的 tools/zh_style_lint.py 一致。

    python tools/zh_style_lint.py --all            所有章节，句长最小者在前
    python tools/zh_style_lint.py 040_polars_combine_tables ...
    python tools/zh_style_lint.py 040_polars_combine_tables --diff   打印越界句

参考值来自作者自己的语料（scipybook3，1238 段 / 3003 句）：
句长 30.1、短句 0.06、长句 0.14、问号约 0、不用「你/您」、全角括号 0、口语词 0。
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

CJK = re.compile(r"[\u4e00-\u9fff]")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CN = os.path.join(HERE, "polars_guide_cn")

BAND = {
    "sent_mean": (27.0, 33.0),
    "short": (0.0, 0.10),          # <=12 汉字
    "long": (0.10, 0.18),          # >=45 汉字
    "para": (55.0, None),          # 纯正文段落均长，仅参考
    "q_per_10k": (0.0, 2.0),       # 问号
    "ni": (0.0, 0.0),              # 你 / 您
    "fw_paren_per_10k": (0.0, 3.0),
    "colloq": (0.0, 0.0),
}
ADVISORY = {"para"}

SOFT_MIN = {"由于": 10, "因此": 20, "下面": 25, "注意": 3, "例如": 8, "其": 30, "表示": 18}
SOFT_MAX = {"所以": 6, "于是": 4, "接下来": 3}

COLLOQ = ["脾气", "掏", "塞", "收工", "门道", "写岔", "搞定", "压不住", "瞅", "说白了",
          "其实就是", "炸了", "玩意儿", "活儿", "别急", "咱", "呢？", "吧。",
          "摆出来", "摆出", "收上来", "第一路", "第二路", "套路", "乱撞", "冒出来",
          "钉成", "钉在", "砍掉", "顺手", "一眼", "摊开", "走过", "跑一遍", "最笨",
          "省事", "东西", "弄混", "撞到",
          # 本书另外抓到的口语／评述式说法
          "这件事", "这一点", "麻烦", "一句话", "往下游", "看一遍", "走一遍", "自己留",
          "不管", "现实里", "到底", "顺手", "没事", "搞定"]


def md_cells(path):
    nb = json.load(open(path, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]


def measure(path):
    mds = md_cells(path)
    txt = "\n".join(mds)
    n = max(1, len(CJK.findall(txt)))
    paras = []
    for cell in mds:
        cell = re.sub(r"```.*?```", "", cell, flags=re.S)
        cur = []
        for line in cell.split("\n"):
            t = line.strip()
            if not t or t[0] in "#|>-*!<$`" or re.match(r"^\d+\.", t):
                if cur:
                    paras.append(" ".join(cur))
                    cur = []
                continue
            cur.append(t)
        if cur:
            paras.append(" ".join(cur))
    paras = [p for p in paras if len(CJK.findall(p)) >= 15]
    sents = [s for p in paras for s in re.split(r"(?<=[。！？])", p)
             if len(CJK.findall(s)) >= 6] or [""]
    lens = [len(CJK.findall(s)) for s in sents] or [0]
    plain = txt
    plain = re.sub(r"(?m)^[ \t]*`{3,}(?!\{)[^\n]*\n.*?^[ \t]*`{3,}[ \t]*$", "", plain, flags=re.S)
    plain = re.sub(r"\$\$.*?\$\$", "", plain, flags=re.S)
    plain = re.sub(r"\$[^$\n]*\$", "", plain)
    plain = re.sub(r"`[^`\n]*`", "", plain)
    plain = re.sub(r"(?m)^[ \t]*>.*$", "", plain)
    return {
        "name": os.path.basename(path)[:-6],
        "chars": n,
        "sent_mean": round(st.mean(lens), 1),
        "short": round(sum(1 for x in lens if x <= 12) / len(lens), 2),
        "long": round(sum(1 for x in lens if x >= 45) / len(lens), 2),
        "para": round(st.mean([len(CJK.findall(p)) for p in paras]), 1) if paras else 0.0,
        "q_per_10k": round(plain.count("？") / n * 10000, 1),
        "ni": plain.count("你") + plain.count("您"),
        "fw_paren_per_10k": round(plain.count("（") / n * 10000, 1),
        "colloq": sum(txt.count(w) for w in COLLOQ),
        "_txt": txt,
        "_sents": sents,
    }


def verdict(m):
    fails = []
    for k, (lo, hi) in BAND.items():
        v = m[k]
        if v < lo or (hi is not None and v > hi):
            fails.append(("NOTE" if k in ADVISORY else "FAIL", f"{k}={v} (band {lo}~{hi})"))
    return fails


def soft_report(m):
    n = m["chars"]
    out = []
    for w, lo in SOFT_MIN.items():
        v = m["_txt"].count(w) / n * 10000
        if v < lo:
            out.append(f"{w}={v:.0f}/万字 < {lo}")
    for w, hi in SOFT_MAX.items():
        v = m["_txt"].count(w) / n * 10000
        if v > hi:
            out.append(f"{w}={v:.0f}/万字 > {hi}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--diff", action="store_true")
    a = ap.parse_args()

    if a.all or not a.names:
        paths = sorted(glob.glob(os.path.join(CN, "*.ipynb")))
    else:
        paths = [os.path.join(CN, nm if nm.endswith(".ipynb") else nm + ".ipynb") for nm in a.names]

    bad = 0
    rows = [measure(p) for p in paths if os.path.exists(p)]
    rows.sort(key=lambda r: r["sent_mean"])
    print(f"{'chapter':40s}{'字':>7s}{'句长':>6s}{'短句':>6s}{'长句':>6s}{'段长':>6s}{'问号':>6s}{'你':>4s}{'全角(':>7s}{'口语':>5s}")
    for m in rows:
        print(f"{m['name']:40s}{m['chars']:7d}{m['sent_mean']:6.1f}{m['short']:6.2f}{m['long']:6.2f}"
              f"{m['para']:6.1f}{m['q_per_10k']:6.1f}{m['ni']:4d}{m['fw_paren_per_10k']:7.1f}{m['colloq']:5d}")
        fails = verdict(m)
        soft = soft_report(m)
        if fails:
            print("    " + "; ".join(f"{t} {msg}" for t, msg in fails))
            bad += sum(1 for t, _ in fails if t == "FAIL")
        if soft:
            print("    lexical: " + "; ".join(soft))
        if a.diff and (fails or soft):
            for s in m["_sents"]:
                if (any(w in s for w in COLLOQ) or "你" in s or "您" in s
                        or "？" in s or "（" in s or len(CJK.findall(s)) <= 12
                        or len(CJK.findall(s)) >= 45):
                    print("      - " + s.strip())
    print(f"\n{len(rows)} chapters; band: 句长 27-33, 短句 <=0.10, 长句 0.10-0.18, 问号 <=2/万字, 你 0, 全角( <=3/万字, 口语 0")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
