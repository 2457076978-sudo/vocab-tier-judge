# -*- coding: utf-8 -*-
"""初中单词判定器 v6 — 软标签版（自制 RLCD L2）
核心：锚点冲突不再是"最早获胜"，而是变成目标分布——45.9% 的分歧 = 概率质量。
证据源与权重：教材单元/课标 w=3；收编 w=2；3500doc w=2；AF域(超纲/词典) w=2；
             CEFR w=1（已知系统性错位，只作弱证据）；边缘定档 w=1；低频超纲 w=1
实现：分布 → 采样 6 个硬标签（+邻档平滑 15%）→ 走原生 mlx_lm 管线（期望=软标签CE）
产物：data_v6/train|valid.jsonl + 软标签分布表.tsv（审计：每词的证据分布）
"""
import csv
import json
import os
import random
import re
import subprocess
from collections import defaultdict

BASE = "/Users/wayne/Desktop/工作文档库"
KF = BASE + "/01-教学工作/名著阅读工作区_AnimalFarm/知识文件"
CURR = BASE + "/05-网站与AI工作区/LayerText/assets/wordlists"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"
OUT = PROJ + "/data_v6"
DOC3500 = BASE + "/01-教学工作/教材词典/工作文档___英语40篇英语短文搞定高考3500词记忆本册单词背诵本带翻译.doc"

WORD_RE = re.compile(r"^[a-z][a-z'-]*$")
TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")
VOWEL_RE = re.compile(r"[aeiou]")
RUN_RE = re.compile(r"([a-z])\1{3,}")
SAMPLES_PER_WORD = 6
NEIGHBOR_SMOOTH = 0.15

NAMES = {
    "mike", "mary", "tom", "lucy", "helen", "adam", "alice", "bob", "john",
    "jane", "kate", "peter", "paul", "sam", "ben", "bill", "david", "eric",
    "frank", "grace", "henry", "jack", "jim", "judy", "karen", "larry",
    "lily", "linda", "lisa", "mark", "nancy", "nick", "rose", "roy",
    "sara", "susan", "ted", "tina", "tony", "anna", "emma", "leo", "alan",
    "carl", "cindy", "dale", "dan", "dean", "gary", "jason", "jerry",
    "kevin", "laura", "martin", "mona", "rita", "roger", "ron", "scott",
}


def sane(w):
    return 3 <= len(w) <= 15 and VOWEL_RE.search(w) and not RUN_RE.search(w)


def deinflect(w):
    IRREG = {"gone": "go", "went": "go", "made": "make", "said": "say", "took": "take",
             "came": "come", "saw": "see", "seen": "see", "got": "get", "known": "know",
             "told": "tell", "found": "find", "left": "leave", "kept": "keep", "held": "hold",
             "brought": "bring", "began": "begin", "spoken": "speak", "broken": "break",
             "chosen": "choose", "driven": "drive", "eaten": "eat", "given": "give",
             "written": "write", "grown": "grow", "thrown": "throw", "worn": "wear",
             "built": "build", "bought": "buy", "caught": "catch", "taught": "teach",
             "thought": "think", "sold": "sell", "sent": "send", "spent": "spend",
             "won": "win", "met": "meet", "led": "lead", "paid": "pay", "ran": "run",
             "stood": "stand", "understood": "understand", "became": "become", "tore": "tear"}
    outs = set()
    if w in IRREG:
        outs.add(IRREG[w])
    if w.endswith("ies") and len(w) > 4: outs.add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 4: outs.add(w[:-2])
    if w.endswith("s") and len(w) > 3: outs.add(w[:-1])
    if w.endswith("ed") and len(w) > 4:
        outs.add(w[:-2]); outs.add(w[:-1])
        stem = w[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
    if w.endswith("ing") and len(w) > 5:
        outs.add(w[:-3]); outs.add(w[:-3] + "e")
        stem = w[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
    if w.endswith("er") and len(w) > 4: outs.add(w[:-2]); outs.add(w[:-1])
    return outs


def main():
    rng = random.Random(42)

    # ---- 证据收集：word -> [(tier, weight, source)] ----
    ev = defaultdict(list)

    tj = json.load(open(KF + "/教材单元_人教版.json", encoding="utf-8"))

    def ok(w):
        return len(w) >= 2 and WORD_RE.match(w) and "'" not in w

    for w in tj["base"]:
        if ok(w) and w not in NAMES:
            ev[w].append((1, 3.0, "教材base"))
    for bk, tier in (("八上", 2), ("八下", 2), ("九上", 3)):
        for u in tj["books"][bk].values():
            for w in u.get("词", []):
                w = w.strip().lower()
                if ok(w) and w not in NAMES:
                    ev[w].append((tier, 3.0, f"教材{bk}"))

    curr = set()
    for line in open(CURR + "/curriculum_2022_level3_1600.txt", encoding="utf-8"):
        w = line.strip().lower()
        if w and not w.startswith("#"):
            curr.add(w)
    for w in curr:
        if ok(w) and w not in NAMES:
            ev[w].append((3, 3.0, "课标"))

    shoubian = set()
    for row in csv.DictReader(open(PROJ + "/data/收编_材料强证据.tsv", encoding="utf-8-sig"), delimiter="\t"):
        w = row["词"].strip().lower()
        if len(w) >= 4 and sane(w):
            shoubian.add(w)
    for w in shoubian:
        ev[w].append((3, 2.0, "材料收编"))

    r = subprocess.run(["textutil", "-convert", "txt", "-stdout", DOC3500],
                       capture_output=True, timeout=120)
    doc3500 = {t.lower() for t in TOKEN.findall(r.stdout.decode("utf-8", "ignore")) if sane(t)}
    for w in doc3500:
        ev[w].append((4, 2.0, "3500doc"))

    cefr = {}
    for line in open(CURR + "/cefrj_levels.txt", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            p = line.split()
            if len(p) == 2:
                cefr[p[0].lower()] = p[1]
    CM = {"A1": 1, "A2": 2, "B1": 4, "B2": 4, "C1": 5, "C2": 5}
    for w, lv in cefr.items():
        if ok(w) and sane(w):
            ev[w].append((CM[lv], 1.0, f"CEFR{lv}"))  # 弱证据：已知与国内学制错位

    af_super, af_dict, chaogang, freq_cnt = set(), set(), set(), {}
    for row in csv.DictReader(open(KF + "/材料词频表_v1.1.csv", encoding="utf-8-sig")):
        w = row["词"].strip().lower()
        try:
            freq_cnt[w] = int(str(row["词频"]).strip() or 0)
        except ValueError:
            freq_cnt[w] = 0
        if (row["口径"] or "").strip() == "超纲/变形":
            chaogang.add(w)
    for row in csv.DictReader(open(KF + "/AF超纲高频词_v0.1.csv", encoding="utf-8-sig")):
        af_super.add(row["词"].strip().lower())
    for row in csv.DictReader(open(KF + "/AF注释词典_v1.csv", encoding="utf-8-sig")):
        af_dict.add(row["词"].strip().lower())
    for w in af_super | af_dict:
        if ok(w) and sane(w):
            ev[w].append((5, 2.0, "AF域"))

    mat_cov = {}
    for row in csv.DictReader(open(PROJ + "/data/词汇总表_教学工作区.csv", encoding="utf-8-sig")):
        mat_cov[row["词"].strip().lower()] = int(row["材料类覆盖数"] or 0)
    for w in chaogang:
        if not ok(w) or not sane(w) or w in NAMES:
            continue
        cov = mat_cov.get(w, 0)
        if cov >= 8:
            ev[w].append((3, 1.0, "边缘高覆盖"))
        elif cov >= 2:
            ev[w].append((4, 1.0, "边缘中覆盖"))
        elif w in af_super or w in af_dict or w in cefr:
            pass  # 已有更强证据
        elif freq_cnt.get(w, 0) >= 3:
            ev[w].append((5, 1.0, "低频超纲"))
        else:
            ev[w].append((5, 0.5, "碎片超纲"))  # 半信半疑

    # 变形级联：无证据的词从基础形借分布
    words = sorted(ev)
    ev_map = dict(ev)
    for w in words:
        for b in deinflect(w):
            if b in ev_map and b != w:
                pass  # 保留自身证据；基础形另有样本
    for w in sorted({x for x in chaogang | af_super | af_dict if ok(x) and sane(x)} | set(ev_map)):
        if w not in ev_map:
            for b in deinflect(w):
                if b in ev_map:
                    ev_map[w] = [(t, wt * 0.8, f"{s}·借形") for t, wt, s in ev_map[b]]
                    break

    print(f"有证据词 {len(ev_map)}")
    multi = sum(1 for v in ev_map.values() if len({t for t, _, _ in v}) > 1)
    print(f"多源多档词（软标签真正发挥作用的）: {multi} ({multi/len(ev_map):.1%})")

    # ---- 分布 + 邻档平滑 + 采样 ----
    PROMPT = ("给英文单词的学段难度定档（按中国学生普通进度，取最早学段）。\n单词：%s\n"
              "选项：一=七年级 二=八年级 三=九年级（初中毕业线） 四=高中（高考3500内） 五=大学毕业以上\n/no_think")

    data = []
    dist_rows = []
    for w, evs in sorted(ev_map.items()):
        raw = defaultdict(float)
        for t, wt, _s in evs:
            raw[t] += wt
        tot = sum(raw.values())
        p = {t: v / tot for t, v in raw.items()}
        # 邻档平滑：档位有序，相邻档各分走 15%（除以两侧存在的邻档数）
        sm = defaultdict(float)
        for t, v in p.items():
            sm[t] += v * (1 - NEIGHBOR_SMOOTH)
            nbrs = [t + 1] if t == 1 else ([t - 1] if t == 5 else [t - 1, t + 1])
            for nb in nbrs:
                if 1 <= nb <= 5:
                    sm[nb] += v * NEIGHBOR_SMOOTH / len([x for x in nbrs if 1 <= x <= 5])
        tot2 = sum(sm.values())
        sm = {t: v / tot2 for t, v in sm.items() if v > 1e-6}
        dist_rows.append((w, " ".join(f"{t}:{sm[t]:.2f}" for t in sorted(sm))))
        labels = list(sm.keys())
        probs = [sm[k] for k in labels]
        n_samples = 5 if len({t for t, _, _ in evs}) > 1 else 2  # 多档词多采承载分布，单档词省算力
        lab_str = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五"}
        for lab in rng.choices(labels, weights=probs, k=n_samples):
            data.append({"prompt": PROMPT % w, "completion": lab_str[lab]})

    rng.shuffle(data)
    n_valid = len(data) // 10
    valid, train = data[:n_valid], data[n_valid:]
    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(OUT + "/" + name, "w", encoding="utf-8") as f:
            for r_ in rows:
                f.write(json.dumps(r_, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} 条")
    with open(OUT + "/软标签分布表.tsv", "w", encoding="utf-8") as f:
        f.write("词\t分布(档:概率)\n")
        for w, d in dist_rows:
            f.write(f"{w}\t{d}\n")
    print("V6_DATA_DONE")


if __name__ == "__main__":
    main()
