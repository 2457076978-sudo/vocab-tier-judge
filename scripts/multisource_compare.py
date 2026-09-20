# -*- coding: utf-8 -*-
"""多源户口对比（2026-09-20）：课标 × genkin × 人教教材分册 × 高中必修 × 四级书 × 六级书
在 416 词队列上做源间一致性 + 共识档 vs 模型读数对比。
产物：data_v6/多源对比_416.tsv
"""
import csv
import os
from collections import Counter, defaultdict

PROJ = "/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/初中单词判定器"
DD = "/tmp/dictdata/LinXueyuanStdio-DictionaryData-032c3cd"

# ---- 书目 id ----
BOOKS = {
    "人教七年级": {"e3796ae838835da0b6f6ea37", "92c8c96e4c37100777c7190b"},      # 七上+七下
    "人教八年级": {"6a9aeddfc689c1d0e3b9ccc3", "db8e1af0cb3aca1ae2d00186"},      # 八上+八下
    "人教九年级": {"63923f49e5241343aa7acb6a"},                                   # 九年级全册
    "高中必修": {"d645920e395fedad7bbbed0e", "3416a75f4cea9109507cacd8",
               "a1d0c8e83f027327d8461063", "17e62166fc8586dfa4d1bc0e",
               "f7177163c833dff4b38fc8d2"},                                       # 必修1-5
    "四级书": {"38b3eff8baf56627478ec76a"},                                        # 新东方四级正序版4450
}
# 六级书 id 动态找
for line in open(f"{DD}/book.csv", encoding="utf-8"):
    p = line.rstrip("\n").split(">")
    if len(p) > 5 and "六级" in p[4] and p[1] == "0" and "专业" not in p[4]:
        BOOKS["六级书"] = {p[0]}
        print("六级书:", p[4], p[5])
        break

# ---- 词表加载 ----
def load_sets():
    vocab = {}          # vc_id -> word
    for line in open(f"{DD}/word.csv", encoding="utf-8"):
        p = line.rstrip("\n").split(">")
        w = p[1].strip().lower()
        if w.isascii() and " " not in w and "'" not in w and len(w) >= 2:
            vocab[p[0]] = w
    want = set().union(*BOOKS.values())
    sets = defaultdict(set)   # 书名 -> set(word)
    with open(f"{DD}/relation_book_word.csv", encoding="utf-8") as f:
        for line in f:
            p = line.split(">")
            if p[1] in want and p[2] in vocab:
                for name, ids in BOOKS.items():
                    if p[1] in ids:
                        sets[name].add(vocab[p[2]])
    return sets

def load_genkin():
    g = {}
    for line in open(f"{PROJ}/data_v6/学段户口_genkin.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if p[0] != "词":
            g[p[0]] = p[1]
    return g

def load_kebiao():
    k = set()
    for line in open(f"{PROJ}/data/eval_课标.tsv", encoding="utf-8"):
        w = line.split("\t")[0].strip().lower()
        if w.isascii():
            k.add(w)
    return k

# ---- 档位序数（用于共识） ----
ORD = {"七年级": 1, "八年级": 2, "九年级": 3, "初中": 2.5, "高中": 4, "四级": 4.3, "六级": 4.6}
BAND_OF = {1: "七", 2: "八", 3: "九", 2.5: "初中", 4: "高中", 4.3: "四级", 4.6: "六级"}

def sources_of(w, kebiao, genkin, sets):
    """返回 {源名: 档序数}，最早学段优先取舍"""
    s = {}
    if w in kebiao:
        s["课标"] = ORD["初中"]
    if w in genkin:
        s["genkin"] = ORD[genkin[w]]
    for name in ("人教七年级", "人教八年级", "人教九年级"):
        if w in sets[name]:
            s[name] = ORD[name[2:]]
    if w in sets["高中必修"]:
        s["高中教材"] = ORD["高中"]
    if w in sets["四级书"]:
        s["四级书"] = ORD["四级"]
    if w in sets["六级书"]:
        s["六级书"] = ORD["六级"]
    return s

def deinflect(w):
    outs = {w}
    if w.endswith("ies") and len(w) > 4: outs.add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 4: outs.add(w[:-2])
    if w.endswith("s") and len(w) > 3: outs.add(w[:-1])
    if w.endswith("ed") and len(w) > 4:
        outs.update({w[:-2], w[:-1]})
        if w.endswith("ied"): outs.add(w[:-3] + "y")
        stem = w[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        outs.update({stem, stem + "e"})
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
    return outs - {w}

def consensus(sources):
    """多数投票 + 最早学段加权：取中位序数；源数<2 不定共识"""
    if len(sources) < 2:
        return None, 0
    vals = sorted(sources.values())
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    # 中位若落在初中未细分带(2.25-2.75)且有细分票，向下归到细分档
    return med, n

def main():
    sets = load_sets()
    genkin, kebiao = load_genkin(), load_kebiao()
    for k, v in sorted(sets.items()):
        print(f"{k}: {len(v)} 词")

    words, es = [], {}
    for line in open(f"{PROJ}/data_v4/v6全量读数.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if p[0] != "词":
            words.append(p[0]); es[p[0]] = float(p[1])

    out = ["词\tv6E\t课标\tgenkin\t人教七\t人教八\t人教九\t高中教材\t四级书\t六级书\t共识档\t源数\t一致\tE落档"]
    stats = Counter(); mid_rows = []
    for w in words:
        src = sources_of(w, kebiao, genkin, sets)
        if not src:
            for cand in deinflect(w):
                src = sources_of(cand, kebiao, genkin, sets)
                if src:
                    src = {f"{k}*" : v for k, v in src.items()}
                    break
        med, n = consensus(src)
        agree = "-" if med is None else ("同" if len(set(src.values())) == 1 else "分")
        e = es[w]
        e_band = ("初中域" if e < 3.5 else "超纲域")
        c_band = "-" if med is None else ("初中域" if med <= 3 else "超纲域")
        hit = "-" if med is None else ("=" if e_band == c_band else "≠")
        stats[f"源数{n}"] += 1
        if med is not None:
            stats[f"模型{'合' if hit=='=' else '不合'}"] += 1
            stats[f"源间{'同' if agree=='同' else '分'}"] += 1
        if 3.0 <= e < 3.5 and med is not None:
            mid_rows.append((w, e, med, hit))
        def cell(k):
            for key in (k, k + "*"):
                if key in src:
                    v = src[key]
                    for bname, bordin in ORD.items():
                        if bordin == v:
                            return bname
                    return str(v)
            return "-"
        row = [w, f"{e:.2f}"] + [cell(k) for k in
               ["课标", "genkin", "人教七年级", "人教八年级", "人教九年级", "高中教材", "四级书", "六级书"]]
        row += [BAND_OF.get(med, "-") if med is not None else "-", str(n), agree, hit]
        out.append("\t".join(row))

    with open(f"{PROJ}/data_v6/多源对比_416.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print("\n==== 统计 ====")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"\n骑线带(3.0-3.5)且有多源共识: {len(mid_rows)} 词")
    ok = sum(1 for r in mid_rows if r[3] == "=")
    print(f"  其中模型与共识一致: {ok}，冲突: {len(mid_rows)-ok}")
    print("  冲突例:", ", ".join(f"{w} E={e:.2f} 共识={BAND_OF.get(m,'?')}" for w, e, m, h in mid_rows if h == "≠")[:400])
    print("\n已写出 data_v6/多源对比_416.tsv")

if __name__ == "__main__":
    main()
