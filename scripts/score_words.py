# -*- coding: utf-8 -*-
"""初中单词判定器 — 逐词打分 CLI
用法：
  echo homework | ~/venvs/mlx/bin/python scripts/score_words.py
  ~/venvs/mlx/bin/python scripts/score_words.py word1 word2 ...
  ~/venvs/mlx/bin/python scripts/score_words.py -f 词表.txt -o 出分.tsv
分数 = P(是)-P(否) ∈ [-1,1]，>0 判"是"（初中教学词汇口径=课标∪教材已学）
"""
import argparse
import math
import os
import sys

import mlx.core as mx
from mlx_lm import load

BASE = "/Users/wayne/Desktop/工作文档库"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"
MODEL = os.path.expanduser("~/.omlx/models/Qwen3-0.6B-bf16")
ADAPTER = PROJ + "/adapters/" + os.environ.get("JUDGE_ADAPTER", "v1")

PROMPT = "判断英文单词是否属于中国初中英语教学词汇（课标与教材范围内）。\n单词：%s\n/no_think"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*", help="直接给词")
    ap.add_argument("-f", "--file", help="词表文件，一行一词")
    ap.add_argument("-o", "--out", help="输出 tsv 路径（默认打印到屏幕）")
    args = ap.parse_args()

    words = [w.strip().lower() for w in args.words if w.strip()]
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            words += [l.strip().lower() for l in fh if l.strip() and not l.startswith("#")]
    if not words:
        print(__doc__)
        sys.exit(1)

    model, tokenizer = load(MODEL, adapter_path=ADAPTER)
    iy = tokenizer.encode("是", add_special_tokens=False)[0]
    ino = tokenizer.encode("否", add_special_tokens=False)[0]
    think = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)

    rows = []
    for i, w in enumerate(words, 1):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT % w}],
            tokenize=True, add_generation_prompt=True)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        logits = model(mx.array(list(ids) + think)[None])
        ly, ln = float(logits[0, -1, iy]), float(logits[0, -1, ino])
        s = 2.0 / (1.0 + math.exp(-(ly - ln))) - 1.0
        rows.append((w, s))
        print(f"[{i}/{len(words)}] {w}\t{s:+.4f}\t{'是' if s > 0 else '否'}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("词\t分数\t判定\n")
            for w, s in rows:
                f.write(f"{w}\t{s:.4f}\t{'是' if s > 0 else '否'}\n")
        print(f"已写出: {args.out}")


if __name__ == "__main__":
    main()
