# -*- coding: utf-8 -*-
"""初中单词判定器 v2 — 置信度分级数据集
标签层级（按置信度加权复制）：
  是·官方 ×3   ：课标1673 ∪ 已学口径（"应该掌握"的金标准）
  是·材料 ×2   ：不在官方表，但 材料类覆盖≥5 且在词频表有户口（跨语料确认，收编词表漏收）
  否     ×1   ：词频表超纲(材料覆盖≤1) ∪ AF超纲 ∪ AF词典，剔除专名/变形/缩略/正例
  灰（不训练） ：材料覆盖1-4无户口；超纲但材料覆盖≥2（教材边缘）；变形/缩略；材料高频但零户口（进审计清单）
"""
import csv
import json
import random
import re

BASE = "/Users/wayne/Desktop/工作文档库"
KF = BASE + "/01-教学工作/名著阅读工作区_AnimalFarm/知识文件"
CURR = BASE + "/05-网站与AI工作区/LayerText/assets/wordlists"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"
OUT = PROJ + "/data"

WORD_RE = re.compile(r"^[a-z][a-z'-]*$")

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
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
    if w.endswith("ing") and len(w) > 5:
        outs.add(w[:-3]); outs.add(w[:-3] + "e")
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
        if w.endswith("ying"): outs.add(w[:-4] + "ie")
    if w.endswith("er") and len(w) > 4: outs.add(w[:-2]); outs.add(w[:-1])
    if w.endswith("est") and len(w) > 5: outs.add(w[:-3]); outs.add(w[:-3] + "e")
    if w.endswith("ly") and len(w) > 4: outs.add(w[:-2])
    return outs


def load_csv(path, col="词"):
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            w = (row.get(col) or "").strip().lower()
            if w:
                out.append(w)
    return out


def main():
    random.seed(42)

    # 1) 官方正例
    curr = set()
    for line in open(CURR + "/curriculum_2022_level3_1600.txt", encoding="utf-8"):
        w = line.strip().lower()
        if w and not w.startswith("#"):
            curr.add(w)
    learned, chaogang = set(), set()
    for row in csv.DictReader(open(KF + "/材料词频表_v1.1.csv", encoding="utf-8-sig")):
        kou = (row["口径"] or "").strip()
        w = row["词"].strip().lower()
        if kou.startswith("课标内·已学"):
            learned.add(w)
        elif kou == "超纲/变形":
            chaogang.add(w)
    official = {w for w in (curr | learned) if len(w) >= 2 and WORD_RE.match(w)}
    assert 2700 < len(official) < 3000, f"官方正例 {len(official)} 异常"

    # 2) 全量提词结果
    mat_cov, list_cov, corpus_words = {}, {}, set()
    for row in csv.DictReader(open(OUT + "/词汇总表_教学工作区.csv", encoding="utf-8-sig")):
        w = row["词"].strip().lower()
        mat_cov[w] = int(row["材料类覆盖数"])
        list_cov[w] = int(row["词表类覆盖数"])
        corpus_words.add(w)

    # 词频表全表 11202 = 跨语料户口（任何词在 Downloads 材料出现过即算有户）
    freq户口 = set()
    for row in csv.DictReader(open(KF + "/材料词频表_v1.1.csv", encoding="utf-8-sig")):
        freq户口.add(row["词"].strip().lower())

    af_super = set(load_csv(KF + "/AF超纲高频词_v0.1.csv"))
    af_dict = set(load_csv(KF + "/AF注释词典_v1.csv"))
    proper = set(load_csv(KF + "/分档允许表_v0/专名剔除清单_v0.csv"))
    for line in open(KF + "/专名_AnimalFarm.txt", encoding="utf-8"):
        w = line.strip().lower()
        if w and not w.startswith("#"):
            proper.add(w)

    # 3) 材料强证据正例（收编漏收）：不在官方表、材料覆盖≥5、有跨语料户口
    mat_pos = {w for w in corpus_words
               if w not in official and w not in proper and "'" not in w
               and WORD_RE.match(w) and mat_cov.get(w, 0) >= 5 and w in freq户口}
    # 反向确认 v0 发现的漏收词确实被收编
    for probe in ("length", "skin", "mood", "blame", "belief", "okay"):
        print(f"  漏收探针 {probe}: 材料覆盖={mat_cov.get(probe,0)} → {'收编' if probe in mat_pos else '未收编'}")

    pos_set = official | mat_pos

    # 4) 负例与灰区
    neg_pool = {w for w in (chaogang | af_super | af_dict)
                if len(w) >= 2 and WORD_RE.match(w)}
    gray = {}
    negatives = []
    for w in sorted(neg_pool):
        if w in pos_set:
            continue
        if w in proper:
            continue
        if "'" in w or (deinflect(w) & pos_set):
            gray[w] = "变形/缩略"
            continue
        cov = mat_cov.get(w, 0)
        if cov >= 5 and w in freq户口:
            continue  # 上面已收编；防御性兜底
        if cov >= 2:
            gray[w] = f"教材边缘(材料覆盖{cov})"
            continue
        if cov == 1 and w in freq户口 is False:
            pass
        negatives.append(w)

    # 材料高频但零户口 → 审计清单（可能是反爬噪音或真新词，交人工）
    audit_nohukou = [w for w in corpus_words
                     if w not in pos_set and w not in proper and "'" not in w
                     and WORD_RE.match(w) and mat_cov.get(w, 0) >= 5 and w not in freq户口]

    print(f"官方正例 {len(official)}，材料强证据 {len(mat_pos)}，负例 {len(negatives)}，灰区 {len(gray)}，零户口审计 {len(audit_nohukou)}")

    PROMPT = "判断英文单词是否属于中国初中英语教学词汇（课标与教材范围内）。\n单词：%s\n/no_think"

    def mk(w, label):
        return {"prompt": PROMPT % w, "completion": label}

    data = ([mk(w, "是") for w in sorted(official)] * 3
            + [mk(w, "是") for w in sorted(mat_pos)] * 2
            + [mk(w, "否") for w in negatives])
    random.shuffle(data)
    n_valid = len(data) // 10
    valid, train = data[:n_valid], data[n_valid:]

    import os
    os.makedirs(OUT + "/v0档", exist_ok=True)
    for name in ("train.jsonl", "valid.jsonl"):
        os.replace(OUT + "/" + name, OUT + "/v0档/" + name)
        with open(OUT + "/" + name, "w", encoding="utf-8") as f:
            rows = train if name == "train.jsonl" else valid
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} 条")

    with open(OUT + "/灰区_v2.tsv", "w", encoding="utf-8") as f:
        f.write("词\t原因\n")
        for w in sorted(gray):
            f.write(f"{w}\t{gray[w]}\n")
    with open(OUT + "/审计_零户口高频词.tsv", "w", encoding="utf-8") as f:
        f.write("词\t材料覆盖数\t词表类覆盖数\n")
        for w in sorted(audit_nohukou, key=lambda x: -mat_cov[x]):
            f.write(f"{w}\t{mat_cov[w]}\t{list_cov.get(w,0)}\n")
    with open(OUT + "/收编_材料强证据.tsv", "w", encoding="utf-8") as f:
        f.write("词\t材料覆盖数\t词表类覆盖数\n")
        for w in sorted(mat_pos, key=lambda x: -mat_cov[x]):
            f.write(f"{w}\t{mat_cov[w]}\t{list_cov.get(w,0)}\n")

    tp = sum(1 for r in train if r["completion"] == "是")
    vp = sum(1 for r in valid if r["completion"] == "是")
    print(f"分布：train 正{tp}/负{len(train)-tp}，valid 正{vp}/负{len(valid)-vp}")
    print("V2_DATA_DONE")


if __name__ == "__main__":
    main()
