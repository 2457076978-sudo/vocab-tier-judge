# -*- coding: utf-8 -*-
"""agent 判定入库：对账 v4 旋钮读数 + 审计教材声明 + 产分歧清单
产物：data_v4/agent判定_对账.tsv
"""
import csv
import json
import re

BASE = "/Users/wayne/Desktop/工作文档库"
KF = BASE + "/01-教学工作/名著阅读工作区_AnimalFarm/知识文件"
CURR = BASE + "/05-网站与AI工作区/LayerText/assets/wordlists"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"

# ---- 载入 ----
agent = []
for row in csv.DictReader(open(PROJ + "/data_v4/agent判定.tsv", encoding="utf-8"), delimiter="\t"):
    if row["词"].startswith("#") or not row.get("判定"):
        continue
    agent.append((row["词"], row["判定"], row["依据"]))
assert len(agent) == 416, f"行数 {len(agent)} != 416"
assert all(v in ("是", "否") for _, v, _ in agent)
words_agent = [w for w, _, _ in agent]
assert len(set(words_agent)) == 416, "有重复词"

queue = {}
for row in csv.DictReader(open(PROJ + "/data_v4/人工校对_词单.tsv", encoding="utf-8"), delimiter="\t"):
    queue[row["词"]] = (row["v4读数"], row["来源"])
assert set(queue) == set(words_agent) or len(set(queue) & set(words_agent)) > 400, "词单与判定集不齐"

# ---- 审计"教材"声明：19 词应能在 课标∪教材单元库 找到 ----
def okw(w): return len(w) >= 2 and re.match(r"^[a-z][a-z'-]*$", w)
curr = set()
for line in open(CURR + "/curriculum_2022_level3_1600.txt", encoding="utf-8"):
    w = line.strip().lower()
    if w and not w.startswith("#"):
        curr.add(w)
tj = json.load(open(KF + "/教材单元_人教版.json", encoding="utf-8"))
textbook = {w for w in tj["base"] if okw(w)}
for bk in tj["books"].values():
    for u in bk.values():
        for w in u.get("词", []):
            w = w.strip().lower()
            if okw(w):
                textbook.add(w)

def deinflect(w):
    outs = set()
    if w.endswith("ies") and len(w) > 4: outs.add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 4: outs.add(w[:-2])
    if w.endswith("s") and len(w) > 3: outs.add(w[:-1])
    if w.endswith("ed") and len(w) > 4: outs.add(w[:-2]); outs.add(w[:-1])
    if w.endswith("ing") and len(w) > 5: outs.add(w[:-3]); outs.add(w[:-3] + "e")
    if w.endswith("er") and len(w) > 4: outs.add(w[:-2]); outs.add(w[:-1])
    if w.endswith("ly") and len(w) > 4: outs.add(w[:-2])
    return outs

teach_claims = [(w, v, b) for w, v, b in agent if b == "教材"]
audit_fail = []
for w, v, b in teach_claims:
    if w not in curr and w not in textbook and not (deinflect(w) & (curr | textbook)):
        audit_fail.append(w)
print(f"[教材声明审计] {len(teach_claims)} 条中 {len(teach_claims)-len(audit_fail)} 条本地可证")
if audit_fail:
    print("  本地查无（agent 可能有超纲教材知识，人工复核）：", " ".join(audit_fail))

# ---- 对账 v4 旋钮 ----
def band(e):
    e = float(e)
    return "超纲带≥3.5" if e >= 3.5 else ("边界带2.5-3.5" if e >= 2.5 else "初中带<2.5")

rows_out = []
agree = disagree_hard = 0
for w, v, b in agent:
    e, src = queue.get(w, ("?", "?"))
    e_f = float(e) if e != "?" else None
    # 一致性：agent是 vs 旋钮<3.0；agent否 vs 旋钮≥3.5（边界带2.5-3.5不算分歧）
    d = ""
    if e_f is not None:
        if v == "是" and e_f < 3.0:
            agree += 1
        elif v == "否" and e_f >= 3.5:
            agree += 1
        elif 2.5 <= e_f < 3.5:
            d = "边界带(不计)"
        else:
            d = "分歧"
            disagree_hard += 1
    rows_out.append((w, e, src, v, b, band(e_f) if e_f is not None else "?", d))

with open(PROJ + "/data_v4/agent判定_对账.tsv", "w", encoding="utf-8") as f:
    f.write("词\tv4读数\t来源\tagent判定\t依据\t旋钮带\t一致性\n")
    for r in sorted(rows_out, key=lambda x: -(float(x[1]) if x[1] != "?" else 0)):
        f.write("\t".join(r) + "\n")

hard = [r for r in rows_out if r[6] == "分歧"]
print(f"\n[对账] 一致 {agree} / 硬分歧 {disagree_hard} / 边界带不计 {len(rows_out)-agree-disagree_hard}")
print("硬分歧=旋钮与agent反向（agent是但旋钮≥3.5，或agent否但旋钮<2.5）")
for r in sorted(hard, key=lambda x: -float(x[1]))[:20]:
    print(f"  {r[0]:<18} 旋钮{r[1]}  agent={r[3]}({r[4]})  [{r[2]}]")

from collections import Counter
by_v = Counter(r[3] for r in rows_out)
print(f"\n入库完成：agent判定_对账.tsv（{len(rows_out)} 行）判定分布 {dict(by_v)}")
