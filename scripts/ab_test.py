# -*- coding: utf-8 -*-
"""0.6B(v4) vs 0.8B(08b) A/B 对分：416 词 agent 金标签做考卷
指标：AUC（阈值无关）+ 最优阈值准确率；分组=全部 / 纯单词（剔除复合词/专名/非词）
0.6B 读数直接取自词单文件；0.8B 现场打分（enable_thinking=False 前缀，Qwen3.5 模板自闭合空思考块）
"""
import csv
import math
import os

import mlx.core as mx
from mlx_lm import load

PROJ = "/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/初中单词判定器"
MODEL_08B = "/Users/wayne/models/Qwen3.5-0.8B-MLX"

PROMPT = ("给英文单词的学段难度定档（按中国学生普通进度，取最早学段）。\n单词：%s\n"
          "选项：一=七年级 二=八年级 三=九年级（初中毕业线） 四=高中（高考3500内） 五=大学毕业以上\n/no_think")

# ---- 金标签 + 0.6B 读数 ----
gold = {}
basis = {}
for row in csv.DictReader(open(PROJ + "/data_v4/agent判定.tsv", encoding="utf-8"), delimiter="\t"):
    if row["词"].startswith("#"):
        continue
    gold[row["词"]] = 1 if row["判定"] == "是" else 0
    basis[row["词"]] = row["依据"]
read6 = {}
for row in csv.DictReader(open(PROJ + "/data_v4/人工校对_词单.tsv", encoding="utf-8"), delimiter="\t"):
    read6[row["词"]] = float(row["v4读数"])
words = [w for w in gold if w in read6]
assert len(words) >= 410

# ---- 0.8B 现场打分 ----
model, tok = load(MODEL_08B, adapter_path=PROJ + "/adapters/08b")
lids = [tok.encode(l, add_special_tokens=False)[0] for l in ("一", "二", "三", "四", "五")]
assert len(set(lids)) == 5


def score8(word):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT % word}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    logits = model(mx.array(ids)[None])[0, -1]
    xs = [float(logits[k]) for k in lids]
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    ps = [e / sum(es) for e in es]
    return sum((j + 1) * p for j, p in enumerate(ps))


read8 = {}
for i, w in enumerate(words, 1):
    read8[w] = score8(w)
    if i % 50 == 0:
        print(f"  ... {i}/{len(words)}", flush=True)


# ---- 指标 ----
def auc(pairs):
    pos = sorted(s for s, y in pairs if y == 1)
    neg = sorted(s for s, y in pairs if y == 0)
    if not pos or not neg:
        return float("nan")
    import bisect
    wins = ties = 0
    for p in pos:
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        wins += lo
        ties += hi - lo
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def best_acc(pairs):
    thr = sorted({s for s, _ in pairs})
    best = 0
    for t in thr:
        acc = sum(1 for s, y in pairs if (s < t) == (y == 1)) / len(pairs)
        best = max(best, acc)
    return best


def report(name, rd):
    pairs_all = [(rd[w], gold[w]) for w in words]
    pairs_pure = [(rd[w], gold[w]) for w in words if basis[w] in ("教材", "变形", "边缘", "课标")]
    a1, a2 = auc(pairs_all), auc(pairs_pure)
    print(f"\n[{name}] 全部{len(pairs_all)}词: AUC={a1:.3f} 最优阈值准确率={best_acc(pairs_all):.1%}")
    print(f"        纯单词{len(pairs_pure)}词: AUC={a2:.3f} 最优阈值准确率={best_acc(pairs_pure):.1%}")
    return a1, a2


print(f"A/B 考卷：{len(words)} 词（agent 金标 是{sum(gold.values())}/否{len(words)-sum(gold.values())}）")
report("0.6B+v4 (旋钮E)", read6)
report("0.8B+08b (旋钮E)", read8)

with open(PROJ + "/data_v4/AB对分_416金标.tsv", "w", encoding="utf-8") as f:
    f.write("词\t依据\t金标\t0.6B_E\t0.8B_E\n")
    for w in sorted(words, key=lambda x: -read6[x]):
        f.write(f"{w}\t{basis[w]}\t{'是' if gold[w] else '否'}\t{read6[w]:.2f}\t{read8[w]:.2f}\n")
print("\n明细 → data_v4/AB对分_416金标.tsv")
