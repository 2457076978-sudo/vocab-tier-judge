# -*- coding: utf-8 -*-
"""初中单词判定器 v0 — 评估：逐词读 P(是)-P(否)，输出分数与粗指标
考卷：① 课标 1673（在训练集内，乐观指标）② valid 留出集正/负（诚实指标）
     ③ 灰区变形词分布 ④ 词表外新词手感样例
"""
import json
import math
import os
import random
import sys

import mlx.core as mx
from mlx_lm import load

BASE = "/Users/wayne/Desktop/工作文档库"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"
MODEL = os.path.expanduser("~/.omlx/models/Qwen3-0.6B-bf16")
ADAPTER = PROJ + "/adapters/" + os.environ.get("JUDGE_ADAPTER", "v1")

PROMPT = "判断英文单词是否属于中国初中英语教学词汇（课标与教材范围内）。\n单词：%s\n/no_think"

model, tokenizer = load(MODEL, adapter_path=ADAPTER)

ids_yes = tokenizer.encode("是", add_special_tokens=False)
ids_no = tokenizer.encode("否", add_special_tokens=False)
assert len(ids_yes) == 1 and len(ids_no) == 1, f"是/否非单token: {ids_yes} {ids_no}"
IY, IN = ids_yes[0], ids_no[0]
# 训练时 chat template 在 assistant 起始处自动插入空思考块，评估前缀必须复刻，否则读分位置错位
THINK_BLOCK = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
print(f"token ids: 是={IY} 否={IN} 思考块长度={len(THINK_BLOCK)}")


def score(word):
    text_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT % word}],
        tokenize=True, add_generation_prompt=True,
    )
    if isinstance(text_ids, list) and text_ids and isinstance(text_ids[0], list):
        text_ids = text_ids[0]
    ids = list(text_ids) + THINK_BLOCK
    logits = model(mx.array(ids)[None])
    ly, ln = float(logits[0, -1, IY]), float(logits[0, -1, IN])
    return 2.0 / (1.0 + math.exp(-(ly - ln))) - 1.0  # P(是)-P(否) ∈ [-1,1]


def load_words(p):
    return [w.strip() for w in open(p, encoding="utf-8") if w.strip() and not w.startswith("#")]


def main():
    random.seed(7)
    curr = load_words(PROJ + "/data/考卷_课标.txt")
    valid = [json.loads(l) for l in open(PROJ + "/data/valid.jsonl", encoding="utf-8")]
    vpos = sorted({m["prompt"].split("单词：")[1].split("\\n")[0].split("\n")[0]
                   for m in valid if m["completion"] == "是"})
    vneg = sorted({m["prompt"].split("单词：")[1].split("\\n")[0].split("\n")[0]
                   for m in valid if m["completion"] == "否"})
    gray = [l.split("\t")[0] for l in open(PROJ + "/data/灰区_变形误标.tsv", encoding="utf-8").readlines()[1:]]
    novel = ["cucumber", "photosynthesis", "smartphone", "rebellion", "hay",
             "aeroplane", "blockchain", "umbrella", "vocabulary", "tyranny",
             "hooves", "fortnight", "algebra", "dinosaur", "microscope",
             "homework", "cafeteria", "empire", "harvest", "slogan"]

    results = {}

    def run(name, words):
        sc = [(w, score(w)) for w in words]
        results[name] = sc
        with open(PROJ + f"/data/eval_{name}.tsv", "w", encoding="utf-8") as f:
            f.write("词\t分数\n")
            for w, s in sc:
                f.write(f"{w}\t{s:.4f}\n")
        return sc

    sc = run("课标", curr)
    pos_hit = sum(1 for _, s in sc if s > 0)
    print(f"[课标1673·训练内乐观] 召回 {pos_hit}/{len(sc)} = {pos_hit/len(sc):.1%}")
    lo = sorted(sc, key=lambda x: x[1])[:10]
    print("  分数最低10词:", [(w, round(s, 2)) for w, s in lo])

    sc = run("留出正例", vpos)
    hit = sum(1 for _, s in sc if s > 0)
    print(f"[留出正例] 召回 {hit}/{len(sc)} = {hit/len(sc):.1%}")

    sc = run("留出负例", vneg)
    tn = sum(1 for _, s in sc if s < 0)
    print(f"[留出负例] 特异度 {tn}/{len(sc)} = {tn/len(sc):.1%}")
    hi = sorted(sc, key=lambda x: -x[1])[:10]
    print("  分数最高10词(误判):", [(w, round(s, 2)) for w, s in hi])

    sc = run("灰区变形", gray)
    mid = sum(1 for _, s in sc if -0.5 < s < 0.5)
    print(f"[灰区306] 中间带(-0.5,0.5)占比 {mid}/{len(sc)} = {mid/len(sc):.1%}")

    sc = run("新词样例", novel)
    print("[词表外新词手感]")
    for w, s in sorted(sc, key=lambda x: -x[1]):
        tag = "是" if s > 0 else "否"
        print(f"  {w:<15} {s:+.2f}  {tag}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
