# -*- coding: utf-8 -*-
"""初中单词判定器 v3 — 旋钮版评估
读分：五档 token（一二三四五）logits → softmax → 期望学段 E∈[1,5] → 旋钮读数 (E-1)/4 ∈ [0,1]
考卷：① valid 留出集分档准确率 + 有序指标（平均档差/within-1） ② 阶梯手感词 ③ 追踪词（bother 等）
"""
import json
import math
import os

import mlx.core as mx
from mlx_lm import load

BASE = "/Users/wayne/Desktop/工作文档库"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"
MODEL = os.path.expanduser("~/.omlx/models/Qwen3-0.6B-bf16")
ADAPTER = PROJ + "/adapters/" + os.environ.get("JUDGE_ADAPTER", "v6")

PROMPT = ("给英文单词的学段难度定档（按中国学生普通进度，取最早学段）。\n单词：%s\n"
          "选项：一=七年级 二=八年级 三=九年级（初中毕业线） 四=高中（高考3500内） 五=大学毕业以上\n/no_think")
LAB2TIER = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
TIER2NAME = {1: "七年级", 2: "八年级", 3: "九年级", 4: "高中", 5: "大学+"}

model, tokenizer = load(MODEL, adapter_path=ADAPTER)
LIDS = []
for lab in LAB2TIER:
    ids = tokenizer.encode(lab, add_special_tokens=False)
    assert len(ids) == 1, f"{lab} 非单token: {ids}"
    LIDS.append(ids[0])
assert len(set(LIDS)) == 5, f"档位token冲突: {LIDS}"
THINK = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
print("档位 token ids:", LIDS)


def rheostat(word):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT % word}],
        tokenize=True, add_generation_prompt=True)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    logits = model(mx.array(list(ids) + THINK)[None])[0, -1]
    xs = [float(logits[i]) for i in LIDS]
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    ps = [e / sum(es) for e in es]
    e_level = sum((i + 1) * p for i, p in enumerate(ps))
    return e_level, ps


def main():
    # ---- ① valid 留出集 ----
    DV = PROJ + "/" + os.environ.get("JUDGE_DATA", "data_v3")
    valid = [json.loads(l) for l in open(DV + "/valid.jsonl", encoding="utf-8")]
    seen = {}
    for r in valid:
        w = r["prompt"].split("单词：")[1].split("\n")[0]
        seen[w] = LAB2TIER[r["completion"]]
    words = sorted(seen)
    hits, dsum, w1 = 0, 0.0, 0
    rows = []
    for w in words:
        e, _ = rheostat(w)
        t = seen[w]
        d = e - t
        dsum += abs(d)
        w1 += 1 if abs(d) <= 1 else 0
        hits += 1 if round(e) == t else 0
        rows.append((w, t, e))
    n = len(words)
    print(f"[valid 留出 {n} 词] 分档准确 {hits/n:.1%} | 平均档差 {dsum/n:.2f} | within-1 {w1/n:.1%}")

    from collections import defaultdict
    by_t = defaultdict(list)
    for w, t, e in rows:
        by_t[t].append(e)
    for t in sorted(by_t):
        es = by_t[t]
        print(f"  档{t}({TIER2NAME[t]}, n={len(es)}): 期望学段均值 {sum(es)/len(es):.2f}")

    with open(DV + "/eval_留出分档.tsv", "w", encoding="utf-8") as f:
        f.write("词\t真档\t期望学段\n")
        for w, t, e in rows:
            f.write(f"{w}\t{t}\t{e:.3f}\n")

    # ---- ② 阶梯手感 + ③ 追踪词 ----
    showcase = ["the", "book", "apple", "school", "receive", "environment",
                "length", "skin", "mood", "curtain", "abandon", "teenager",
                "windmill", "tyranny", "execution", "rebellion", "bother",
                "cucumber", "empire", "photosynthesis", "smartphone", "harvest",
                "mike", "mary"]
    print("\n[旋钮读数] (1=七年级 … 5=大学+)")
    for w in showcase:
        e, ps = rheostat(w)
        dist = "".join(f"{LAB2TIER[l]}:{p:.2f} " for l, p in zip(LAB2TIER, ps))
        print(f"  {w:<15} E={e:.2f}  {dist}")
    print("RHEOSTAT_EVAL_DONE")


if __name__ == "__main__":
    main()
