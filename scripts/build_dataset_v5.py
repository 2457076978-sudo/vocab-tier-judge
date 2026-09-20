# -*- coding: utf-8 -*-
"""初中单词判定器 v3 — 旋钮版（学段五档有序标签）
刻度（最早锚点优先，变形词级联归并到基础形的档）：
  一=七年级   教材单元库 base
  二=八年级   八上+八下单元词
  三=九年级   九上 ∪ 课标1673 ∪ v2材料收编词（≥4字母+ sane）＝初中毕业线
  四=高中     高考3500短文doc ∪ CEFR B1/B2∩词频户口（sane）
  五=大学+    AF超纲 ∪ AF词典 ∪ 词频表超纲 ∪ CEFR C1/C2（sane 且材料覆盖≤1）
灰区（不训练）：变形/缩略/噪音串/教材边缘（覆盖≥2）/CEFR A1A2 无国内户口
产物：data_v3/train|valid.jsonl + 学段锚点统计.tsv + 灰区_v3.tsv
"""
import csv
import json
import os
import random
import re
import subprocess

BASE = "/Users/wayne/Desktop/工作文档库"
KF = BASE + "/01-教学工作/名著阅读工作区_AnimalFarm/知识文件"
CURR = BASE + "/05-网站与AI工作区/LayerText/assets/wordlists"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"
OUT = PROJ + "/data_v5"
DOC3500 = BASE + "/01-教学工作/教材词典/工作文档___英语40篇英语短文搞定高考3500词记忆本册单词背诵本带翻译.doc"

WORD_RE = re.compile(r"^[a-z][a-z'-]*$")
TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")
VOWEL_RE = re.compile(r"[aeiou]")
RUN_RE = re.compile(r"([a-z])\1{3,}")

IRREGULAR = {
    "gone": "go", "went": "go", "made": "make", "said": "say", "took": "take",
    "taken": "take", "came": "come", "saw": "see", "seen": "see", "got": "get",
    "known": "know", "knew": "know", "told": "tell", "found": "find", "felt": "feel",
    "left": "leave", "kept": "keep", "held": "hold", "brought": "bring",
    "began": "begin", "begun": "begin", "spoken": "speak", "spoke": "speak",
    "broken": "break", "broke": "break", "chosen": "choose", "chose": "choose",
    "driven": "drive", "drove": "drive", "eaten": "eat", "ate": "eat", "given": "give",
    "gave": "give", "written": "write", "wrote": "write", "fallen": "fall", "fell": "fall",
    "grown": "grow", "grew": "grow", "thrown": "throw", "threw": "throw", "worn": "wear",
    "built": "build", "bought": "buy", "caught": "catch", "taught": "teach",
    "thought": "think", "sold": "sell", "sent": "send", "spent": "spend", "lost": "lose",
    "won": "win", "met": "meet", "led": "lead", "paid": "pay", "ran": "run",
    "stood": "stand", "understood": "understand", "became": "become",
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


def ok(w):
    return len(w) >= 2 and WORD_RE.match(w) and "'" not in w


def sane(w):
    """噪音滤网：拒绝 base64 碎片/选项字母串——含元音、无 4 连同字母、长度 3-15"""
    return 3 <= len(w) <= 15 and VOWEL_RE.search(w) and not RUN_RE.search(w)


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

    tj = json.load(open(KF + "/教材单元_人教版.json", encoding="utf-8"))
    base = {w for w in tj["base"] if ok(w)}

    def unit_words(book):
        ws = []
        for u in tj["books"][book].values():
            for w in u.get("词", []):
                w = w.strip().lower()
                if ok(w):
                    ws.append(w)
        return set(ws)

    g8 = unit_words("八上") | unit_words("八下")
    g9 = unit_words("九上")
    curr = set()
    for line in open(CURR + "/curriculum_2022_level3_1600.txt", encoding="utf-8"):
        w = line.strip().lower()
        if w and not w.startswith("#"):
            curr.add(w)
    shoubian = set()
    for row in csv.DictReader(open(PROJ + "/data/收编_材料强证据.tsv", encoding="utf-8-sig"), delimiter="\t"):
        w = row["词"].strip().lower()
        if len(w) >= 4 and sane(w):
            shoubian.add(w)

    # ---- 逐层分配（最早锚点优先） ----
    tier_of = {}
    for pool, t in ((base, 1), (g8, 2), (g9 | curr | shoubian, 3)):
        for w in pool:
            if w not in tier_of:
                tier_of[w] = t

    # ---- L4 高中：3500doc + CEFR B1/B2 ----
    r = subprocess.run(["textutil", "-convert", "txt", "-stdout", DOC3500],
                       capture_output=True, timeout=120)
    doc3500 = {t.lower() for t in TOKEN.findall(r.stdout.decode("utf-8", "ignore"))}
    cefr = {}
    for line in open(CURR + "/cefrj_levels.txt", encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            cefr.setdefault(parts[0].lower(), parts[1])
    freq_hukou, chaogang, freq_cnt = set(), set(), {}
    for row in csv.DictReader(open(KF + "/材料词频表_v1.1.csv", encoding="utf-8-sig")):
        w = row["词"].strip().lower()
        freq_hukou.add(w)
        try:
            freq_cnt[w] = int(str(row["词频"]).strip() or 0)
        except ValueError:
            freq_cnt[w] = 0  # 个别行列错位（词频列出现'功'等），按0计走灰区
        if (row["口径"] or "").strip() == "超纲/变形":
            chaogang.add(w)
    cefr_b12 = {w for w, lv in cefr.items() if lv in ("B1", "B2") and ok(w) and sane(w)}  # v5: 免户口直进高中锚
    for w in ({x for x in doc3500 if sane(x)} | cefr_b12):
        if w not in tier_of:
            tier_of[w] = 4

    # ---- L5 大学+ ----
    af_super = set(load_csv(KF + "/AF超纲高频词_v0.1.csv"))
    af_dict = set(load_csv(KF + "/AF注释词典_v1.csv"))
    cefr_c = {w for w, lv in cefr.items() if lv in ("C1", "C2")}
    cefr_a12 = {w for w, lv in cefr.items() if lv in ("A1", "A2")}
    mat_cov, mat_freq = {}, {}
    for row in csv.DictReader(open(PROJ + "/data/词汇总表_教学工作区.csv", encoding="utf-8-sig")):
        w0 = row["词"].strip().lower()
        mat_cov[w0] = int(row["材料类覆盖数"] or 0)
        mat_freq[w0] = int(row["频次"] or 0)
    proper = set(load_csv(KF + "/分档允许表_v0/专名剔除清单_v0.csv"))
    for line in open(KF + "/专名_AnimalFarm.txt", encoding="utf-8"):
        w = line.strip().lower()
        if w and not w.startswith("#"):
            proper.add(w)
    af_pools = af_super | af_dict | chaogang | cefr_c
    edge_words = set()
    for w in af_pools:
        if w in tier_of or w in proper or not sane(w):
            continue
        if w in chaogang and w not in (af_super | af_dict | cefr_c) and freq_cnt.get(w, 0) < 3:
            continue  # 词频表超纲且自身频次<3：OCR/编码碎片，进灰区
        if mat_cov.get(w, 0) >= 2:
            edge_words.add(w)  # 教材边缘：词频强化①——按覆盖分位定档（级联后处理）
            continue
        tier_of[w] = 5

    # ---- 变形级联：activities→activity 所在档 ----
    for _ in range(3):
        moved = 0
        for w in list(tier_of):
            for b in deinflect(w):
                if b in tier_of and tier_of[b] < tier_of[w]:
                    tier_of[w] = tier_of[b]
                    moved += 1
                    break
        if not moved:
            break

    # ---- 词频强化①：教材边缘词按材料覆盖分位定档（覆盖≥8=材料大量见证→九年级档；2-7=练习册渗透→高中档）----
    edge_assigned = {}
    for w in edge_words:
        t = 3 if mat_cov.get(w, 0) >= 12 else 4
        tier_of[w] = t
        edge_assigned[w] = (mat_cov.get(w, 0), t)

    # ---- 人名黑名单（v2 实锤污染源：教材对话人名非词汇）----
    NAMES = {
        "mike", "mary", "tom", "lucy", "helen", "adam", "alice", "bob", "john",
        "jane", "kate", "peter", "paul", "sam", "ben", "bill", "david", "eric",
        "frank", "grace", "henry", "jack", "jim", "judy", "karen", "larry",
        "lily", "linda", "lisa", "mark", "nancy", "nick", "rose", "roy",
        "sara", "susan", "ted", "tina", "tony", "anna", "emma", "leo", "alan",
        "carl", "cindy", "dale", "dan", "dean", "gary", "jason", "jerry",
        "kevin", "laura", "martin", "mona", "rita", "roger", "ron", "scott",
    }
    dropped_names = sorted(w for w in NAMES if w in tier_of)
    for w in dropped_names:
        tier_of.pop(w)

    L = {i: {w for w, t in tier_of.items() if t == i} for i in range(1, 6)}
    print(f"L1七年级 {len(L[1])} | L2八年级 {len(L[2])} | L3九年级 {len(L[3])} | "
          f"L4高中 {len(L[4])} | L5大学+ {len(L[5])}")
    assert len(L[1]) > 1400 and len(L[2]) > 500 and len(L[3]) > 2000 and len(L[4]) > 500 and len(L[5]) > 2000

    # ---- 灰区 ----
    gray = {}
    known = set(tier_of)
    for w in sorted(af_pools | cefr_a12):
        if w in known or w in proper or not ok(w):
            continue
        if "'" in w:
            gray[w] = "缩略"
        elif not sane(w):
            gray[w] = "噪音串"
        elif deinflect(w) & known:
            gray[w] = "变形"
        elif mat_cov.get(w, 0) >= 2:
            gray[w] = f"教材边缘(覆盖{mat_cov.get(w, 0)})"
        elif w in cefr_a12 and w not in freq_hukou:
            gray[w] = "CEFR基础词无国内户口"
        elif w in chaogang and w not in (af_super | af_dict | cefr_c) and freq_cnt.get(w, 0) < 3:
            gray[w] = "低频碎片(词频表频次<3)"
    print(f"灰区 {len(gray)}")

    PROMPT = ("给英文单词的学段难度定档（按中国学生普通进度，取最早学段）。\n单词：%s\n"
              "选项：一=七年级 二=八年级 三=九年级（初中毕业线） 四=高中（高考3500内） 五=大学毕业以上\n/no_think")

    def mk(w, lab):
        return {"prompt": PROMPT % w, "completion": lab}

    tiers = [("一", sorted(L[1])), ("二", sorted(L[2])), ("三", sorted(L[3])),
             ("四", sorted(L[4])), ("五", sorted(L[5]))]
    data = []
    import statistics
    for lab, ws in tiers:
        rep = max(1, round(4000 / len(ws)))
        freqs = {w: mat_freq.get(w, 0) + freq_cnt.get(w, 0) for w in ws}
        med = statistics.median(freqs.values()) if freqs else 0
        tail = [w for w in ws if freqs[w] <= med]  # 低频半：长尾词（hostess/mutton 类）加固
        data += [mk(w, lab) for w in ws] * rep
        tier_idx = {"一":1,"二":2,"三":3,"四":4,"五":5}[lab]
        if tier_idx >= 3:  # v5: 尾巴加固只给高档位，防整体先验下移
            data += [mk(w, lab) for w in tail]
        print(f"  档{lab}: {len(ws)} 词 × {rep} + 低频尾 {len(tail)} × 1")
    print(f"人名黑名单剔除 {len(dropped_names)}: {' '.join(dropped_names[:20])}")
    print(f"教材边缘定档 {len(edge_assigned)}（≥8→L3: {sum(1 for c, t in edge_assigned.values() if t == 3)}，2-7→L4: {sum(1 for c, t in edge_assigned.values() if t == 4)}）")
    random.shuffle(data)
    n_valid = len(data) // 10
    valid, train = data[:n_valid], data[n_valid:]

    os.makedirs(OUT, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        with open(OUT + "/" + name, "w", encoding="utf-8") as f:
            for r_ in rows:
                f.write(json.dumps(r_, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} 条")

    with open(OUT + "/学段锚点统计.tsv", "w", encoding="utf-8") as f:
        f.write("档\t词数\t样例\n")
        for (lab, ws) in tiers:
            f.write(f"{lab}\t{len(ws)}\t{' '.join(ws[:15])}\n")
    with open(OUT + "/灰区_v5.tsv", "w", encoding="utf-8") as f:
        f.write("词\t原因\n")
        for w in sorted(gray):
            f.write(f"{w}\t{gray[w]}\n")
    with open(OUT + "/边缘定档.tsv", "w", encoding="utf-8") as f:
        f.write("词\t材料覆盖数\t定档\n")
        for w, (c, t) in sorted(edge_assigned.items(), key=lambda x: -x[1][0]):
            f.write(f"{w}\t{c}\t{t}\n")
    print("V5_DATA_DONE")


if __name__ == "__main__":
    main()
