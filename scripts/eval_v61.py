# -*- coding: utf-8 -*-
"""v6.1 评估（配套新 prompt：选项表前置、单词后置贴答案位）
① 416 子组画像（对照 v6 基线）② 教材组召回 @3.0/3.5 ③ 十词卷
"""
import csv
import math
import os
from collections import defaultdict

import mlx.core as mx
from mlx_lm import load

PROJ = "/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/初中单词判定器"
PROMPT = ("给英文单词的学段难度定档（按中国学生普通进度，取最早学段）。\n"
          "选项：一=七年级 二=八年级 三=九年级（初中毕业线） 四=高中（高考3500内） 五=大学毕业以上\n单词：%s\n/no_think")

gold = []
for row in csv.DictReader(open(PROJ + "/data_v4/agent判定.tsv", encoding="utf-8"), delimiter="\t"):
    if not row["词"].startswith("#") and row.get("判定"):
        gold.append((row["词"], row["判定"], row["依据"]))

model, tok = load(os.path.expanduser("~/.omlx/models/Qwen3-0.6B-bf16"), adapter_path=PROJ + "/adapters/v6.1")
lids = [tok.encode(l, add_special_tokens=False)[0] for l in ("一", "二", "三", "四", "五")]
think = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)


def score(ws):
    seqs = []
    for w in ws:
        ids = tok.apply_chat_template([{"role": "user", "content": PROMPT % w}], tokenize=True, add_generation_prompt=True)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        seqs.append(list(ids) + think)
    L = max(len(s) for s in seqs)
    arr = mx.zeros((len(seqs), L), dtype=mx.int32)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = mx.array(s)
    out = model(arr)
    res = []
    for i, s in enumerate(seqs):
        xs = [float(out[i, len(s) - 1, k]) for k in lids]
        m = max(xs)
        es = [math.exp(x - m) for x in xs]
        ps = [e / sum(es) for e in es]
        res.append(sum((j + 1) * p for j, p in enumerate(ps)))
    return res


words = sorted({w for w, _, _ in gold} | {"achieve", "ancient", "challenge", "courage", "environment",
                                          "ambiguous", "phenomenon", "inevitable", "sophisticated",
                                          "deteriorate", "bother", "the", "book", "sallow", "tyranny"})
sc = {}
B = 32
for i in range(0, len(words), B):
    for w, e in zip(words[i:i + B], score(words[i:i + B])):
        sc[w] = e

groups = defaultdict(list)
for g in gold:
    groups[g[2]].append(g)
print(f"{'组':<10}{'n':>5}  v6基线@3.0  v6.1@3.0")
base = {"变形": "79.6%", "边缘": "94.4%", "复合词": "15.7%", "专名": "96.7%", "教材": "15.8%", "非词": "100.0%"}
for name in ("教材", "变形", "边缘", "复合词", "专名", "非词"):
    grp = groups.get(name, [])
    if not grp:
        continue
    a = sum(1 for w, lab, _ in grp if (sc[w] < 3.0) == (lab == "是")) / len(grp)
    print(f"{name:<10}{len(grp):>5}  {base[name]:>9}  {a:.1%}")
all6 = sum(1 for w, lab, _ in gold if (sc[w] < 3.0) == (lab == "是")) / len(gold)
print(f"{'全量':<10}{len(gold):>5}  {'76.2%':>9}  {all6:.1%}")
print("\n教材组召回 @阈值: " + " | ".join(
    f"@{t}:{sum(1 for w,_,_ in groups['教材'] if sc[w] < t)/len(groups['教材']):.0%}" for t in (3.0, 3.5, 4.0)))
print("边缘组特异 @阈值: " + " | ".join(
    f"@{t}:{sum(1 for w,_,_ in groups['边缘'] if sc[w] >= t)/len(groups['边缘']):.0%}" for t in (3.0, 3.5, 4.0)))
print("\n十词卷（v6.1）:")
for w in ("achieve", "ancient", "challenge", "courage", "environment",
          "ambiguous", "phenomenon", "inevitable", "sophisticated", "deteriorate", "bother"):
    print(f"  {w:<15} {sc[w]:.2f}")
