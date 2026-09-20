# -*- coding: utf-8 -*-
"""学段户口表 v2 构建器（2026-09-20，多源最早侧规则）
源与序（最早学段优先，序小者胜）：
  1.0 人教七年级(七上∪七下)   2.0 人教八年级(八上∪八下)   3.0 人教九年级(九全)
  2.5 课标1673                2.6 genkin初中
  4.0 高中必修1-5             4.05 genkin高中
  4.3 新东方四级正序版        4.35 genkin四级
  4.6 六级词汇正序版          4.65 genkin六级
产出 data_v6/学段户口_v2.tsv：词\t锚档\t序\t源列表
"""
import os

PROJ = "/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/初中单词判定器"
DD = "/tmp/dictdata/LinXueyuanStdio-DictionaryData-032c3cd"
OUT = PROJ + "/data_v6/学段户口_v2.tsv"

DD_BOOKS = {
    "人教七年级": (1.0, {"e3796ae838835da0b6f6ea37", "92c8c96e4c37100777c7190b"}),
    "人教八年级": (2.0, {"6a9aeddfc689c1d0e3b9ccc3", "db8e1af0cb3aca1ae2d00186"}),
    "人教九年级": (3.0, {"63923f49e5241343aa7acb6a"}),
    "高中必修": (4.0, {"d645920e395fedad7bbbed0e", "3416a75f4cea9109507cacd8",
                   "a1d0c6e83f027327d8461063", "17e62166fc8586dfa4d1bc0e",
                   "f7177163c833dff4b38fc8d2"}),
    "四级书": (4.3, {"38b3eff8baf56627478ec76a"}),
    "六级书": (4.6, {"7647966b7343c29048673252"}),
}
GENKIN_ORD = {"初中": 2.6, "高中": 4.05, "四级": 4.35, "六级": 4.65}


def main():
    entries = {}  # word -> [best_ord, [(源,序)...]]

    def put(w, src, ordv):
        w = w.strip().lower()
        if not w or not w.isascii() or " " in w or "'" in w or len(w) < 2:
            return
        ent = entries.setdefault(w, [99.0, []])
        ent[1].append(src)
        if ordv < ent[0]:
            ent[0] = ordv

    # DictionaryData 教材/词书
    vocab = {}
    for line in open(f"{DD}/word.csv", encoding="utf-8"):
        p = line.rstrip("\n").split(">")
        w = p[1].strip().lower()
        if w.isascii() and " " not in w and "'" not in w and len(w) >= 2:
            vocab[p[0]] = w
    want = {bid: (name, ordv) for name, (ordv, ids) in DD_BOOKS.items() for bid in ids}
    with open(f"{DD}/relation_book_word.csv", encoding="utf-8") as f:
        for line in f:
            p = line.split(">")
            hit = want.get(p[1])
            if hit and p[2] in vocab:
                put(vocab[p[2]], hit[0], hit[1])

    # 课标 1673
    for line in open(f"{PROJ}/data/eval_课标.tsv", encoding="utf-8"):
        w = line.split("\t")[0].strip().lower()
        if w.isascii():
            put(w, "课标", 2.5)

    # genkin v1 户口
    for line in open(f"{PROJ}/data_v6/学段户口_genkin.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if p[0] != "词":
            put(p[0], "genkin" + p[1], GENKIN_ORD[p[1]])

    NAME = {1.0: "七年级", 2.0: "八年级", 3.0: "九年级", 2.5: "课标初中", 2.6: "初中(genkin)",
            4.0: "高中教材", 4.05: "高中(genkin)", 4.3: "四级(书)", 4.35: "四级(genkin)",
            4.6: "六级(书)", 4.65: "六级(genkin)"}
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("词\t锚档\t序\t源\n")
        for w in sorted(entries):
            o, srcs = entries[w]
            f.write(f"{w}\t{NAME[o]}\t{o}\t{','.join(sorted(set(srcs)))}\n")
    print(f"户口v2: {len(entries)} 词 → {OUT}")
    from collections import Counter
    print(Counter(NAME[e[0]] for e in entries.values()))


if __name__ == "__main__":
    main()
