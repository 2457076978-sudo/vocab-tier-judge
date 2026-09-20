# -*- coding: utf-8 -*-
"""初中单词判定器 v0 — 数据集构建
正例：词频表 v1.1 口径=课标内·已学 ∪ 课标 1673
负例：词频表 口径=超纲/变形 ∪ AF超纲高频词 ∪ AF注释词典（取未见于正例者）
排除：专名剔除清单 84 + AF专名 23 + 教材词·未学 + 待定；单字母/非字母词
输出：train.jsonl / valid.jsonl（分层 9:1）+ 考卷_课标.txt / 超纲负例池.tsv + 构建报告
"""
import csv
import json
import random
import re
import os
import sys

BASE = "/Users/wayne/Desktop/工作文档库"
KF = BASE + "/01-教学工作/名著阅读工作区_AnimalFarm/知识文件"
CURR = BASE + "/05-网站与AI工作区/LayerText/assets/wordlists"
OUT = BASE + "/05-网站与AI工作区/初中单词判定器/data"

WORD_RE = re.compile(r"^[a-z][a-z'-]*$")

def load_wordfreq():
    pos, neg, skipped = [], [], {}
    with open(KF + "/材料词频表_v1.1.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            w = (row["词"] or "").strip().lower()
            if not w:
                continue
            kou = (row["口径"] or "").strip()
            if kou.startswith("课标内·已学"):
                pos.append(w)
            elif kou == "超纲/变形":
                neg.append(w)
            else:
                skipped[kou] = skipped.get(kou, 0) + 1
    return pos, neg, skipped

def load_curriculum():
    words = []
    with open(CURR + "/curriculum_2022_level3_1600.txt", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#"):
                words.append(w)
    return words

def load_csv_words(path, col="词"):
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            w = (row.get(col) or "").strip().lower()
            if w:
                out.append(w)
    return out

def load_af_names():
    out = []
    with open(KF + "/专名_AnimalFarm.txt", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#"):
                out.append(w)
    return out

IRREGULAR = {
    "gone": "go", "went": "go", "made": "make", "said": "say", "took": "take",
    "taken": "take", "came": "come", "saw": "see", "seen": "see", "got": "get",
    "gotten": "get", "known": "know", "knew": "know", "told": "tell", "found": "find",
    "felt": "feel", "left": "leave", "kept": "keep", "held": "hold", "brought": "bring",
    "began": "begin", "begun": "begin", "spoken": "speak", "spoke": "speak",
    "broken": "break", "broke": "break", "chosen": "choose", "chose": "choose",
    "driven": "drive", "drove": "drive", "eaten": "eat", "ate": "eat", "given": "give",
    "gave": "give", "written": "write", "wrote": "write", "fallen": "fall", "fell": "fall",
    "grown": "grow", "grew": "grow", "thrown": "throw", "threw": "throw", "worn": "wear",
    "wore": "wear", "built": "build", "bought": "buy", "caught": "catch", "taught": "teach",
    "thought": "think", "sold": "sell", "sent": "send", "spent": "spend", "lost": "lose",
    "won": "win", "met": "meet", "led": "lead", "paid": "pay", "laid": "lay",
    "stood": "stand", "understood": "understand", "became": "become", "ran": "run",
    "sang": "sing", "sung": "sing", "swam": "swim", "swum": "swim", "drew": "draw",
    "drawn": "draw", "been": "be", "had": "have", "did": "do", "done": "do",
}

def deinflect(w):
    """去后缀还原候选（查正例用，宁滥勿漏，误还原由灰区文件人工复核）"""
    outs = set()
    if w in IRREGULAR:
        outs.add(IRREGULAR[w])
    if w.endswith("ies") and len(w) > 4: outs.add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 4: outs.add(w[:-2])
    if w.endswith("s") and len(w) > 3: outs.add(w[:-1])
    if w.endswith("ed") and len(w) > 4:
        outs.add(w[:-2]); outs.add(w[:-1])
        if w.endswith("ied"): outs.add(w[:-3] + "y")
        stem = w[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])  # stopped->stop
    if w.endswith("ing") and len(w) > 5:
        outs.add(w[:-3]); outs.add(w[:-3] + "e")
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])  # running->run
        if w.endswith("ying"): outs.add(w[:-4] + "ie")
    if w.endswith("er") and len(w) > 4: outs.add(w[:-2]); outs.add(w[:-1])
    if w.endswith("est") and len(w) > 5: outs.add(w[:-3]); outs.add(w[:-3] + "e")
    if w.endswith("ly") and len(w) > 4: outs.add(w[:-2])
    return outs

def main():
    random.seed(42)

    freq_pos, freq_neg, skipped = load_wordfreq()
    curr = load_curriculum()
    af_super = load_csv_words(KF + "/AF超纲高频词_v0.1.csv")
    af_dict = load_csv_words(KF + "/AF注释词典_v1.csv")
    proper = set(load_csv_words(KF + "/分档允许表_v0/专名剔除清单_v0.csv")) | set(load_af_names())

    # 断言：源数据规模与勘察一致，防静默漏读
    assert 2500 < len(freq_pos) < 2700, f"课标内·已学 {len(freq_pos)} 异常"
    assert 8400 < len(freq_neg) < 8700, f"超纲/变形 {len(freq_neg)} 异常"
    assert len(curr) == 1673, f"课标 {len(curr)} != 1673"
    assert 2100 < len(af_super) < 2300, f"AF超纲 {len(af_super)} 异常"
    assert 1500 < len(af_dict) < 1600, f"AF词典 {len(af_dict)} 异常"
    print(f"源数据核对通过：已学{len(freq_pos)} 超纲{len(freq_neg)} 课标{len(curr)} "
          f"AF超纲{len(af_super)} AF词典{len(af_dict)} 专名{len(proper)}")

    def clean(ws):
        return [w for w in ws if len(w) >= 2 and WORD_RE.match(w)]

    positives = list(dict.fromkeys(clean(freq_pos) + clean(curr)))
    neg_pool_raw = clean(freq_neg) + clean(af_super) + clean(af_dict)

    pos_set = set(positives)
    # 变形误标剔除：负例候选若能还原成正例（player->play / running->run / gone->go），进灰区不训练
    # 含撇号的负例（缩略词 didn't/possessives）属语法形态非词汇判定，一并进灰区
    gray_set = {w for w in dict.fromkeys(neg_pool_raw)
                if (w not in pos_set and w not in proper
                    and (("'" in w) or (deinflect(w) & pos_set)))}
    negatives = []
    conflict = 0
    for w in dict.fromkeys(neg_pool_raw):
        if w in pos_set:
            conflict += 1
            continue
        if w in proper or w in gray_set:
            continue
        negatives.append(w)
    negatives = list(dict.fromkeys(negatives))

    print(f"正例 {len(positives)}，负例 {len(negatives)}，冲突剔除 {conflict}，灰区(变形/缩略) {len(gray_set)}")
    print("词频表灰区跳过分布：", skipped)

    PROMPT = "判断英文单词是否属于中国初中英语教学词汇（课标与教材范围内）。\n单词：%s\n/no_think"

    def make(w, label):
        return {"prompt": PROMPT % w, "completion": label}

    # 正例×3 过采样对冲负例多数（1:3.6 → 约1:1.2），判定任务需要平衡先验
    data = [make(w, "是") for w in positives] * 3 + [make(w, "否") for w in negatives]
    random.shuffle(data)
    n_valid = len(data) // 10
    valid, train = data[:n_valid], data[n_valid:]

    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(OUT + "/" + name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} 条")

    with open(OUT + "/考卷_课标.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(dict.fromkeys(clean(curr))) + "\n")
    with open(OUT + "/灰区_变形误标.tsv", "w", encoding="utf-8") as f:
        f.write("词\t判定依据（去后缀还原为正例）\n")
        for w in sorted(gray_set):
            hits = sorted(deinflect(w) & pos_set)
            f.write(f"{w}\t{'/'.join(hits)}\n")
    neg_set = set(negatives)
    with open(OUT + "/超纲负例池.tsv", "w", encoding="utf-8") as f:
        f.write("词\t来源\n")
        for w in dict.fromkeys(clean(af_super)):
            if w in neg_set:
                f.write(f"{w}\tAF超纲高频词\n")
        for w in dict.fromkeys(clean(af_dict)):
            if w in neg_set:
                f.write(f"{w}\tAF注释词典\n")
        for w in dict.fromkeys(clean(freq_neg)):
            if w in neg_set:
                f.write(f"{w}\t词频表超纲\n")

    train_pos = sum(1 for r in train if r["completion"] == "是")
    valid_pos = sum(1 for r in valid if r["completion"] == "是")
    print(f"分布：train 正{train_pos}/负{len(train)-train_pos}，valid 正{valid_pos}/负{len(valid)-valid_pos}")
    print("DONE")

if __name__ == "__main__":
    main()
