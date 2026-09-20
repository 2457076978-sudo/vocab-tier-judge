# -*- coding: utf-8 -*-
"""Laya 0.4B 零样本对照实验：416 金标签同卷考（vs 0.6B v4 / 0.8B）
问法① choice：是否中国初中英语教学词汇（是/否）
问法② score：学段五档（七年级~大学毕业+）
产出：data_v4/Laya对照_416金标.tsv + AUC/准确率汇总
"""
import csv
import json
import sys
import time

sys.path.insert(0, "/Users/wayne/models/laya-0.4b")
from rl_agent_api import RLAgent

PROJ = "/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/初中单词判定器"
MODEL_DIR = "/Users/wayne/models/laya-0.4b"

gold, basis = {}, {}
for row in csv.DictReader(open(PROJ + "/data_v4/agent判定.tsv", encoding="utf-8"), delimiter="\t"):
    if row["词"].startswith("#"):
        continue
    gold[row["词"]] = 1 if row["判定"] == "是" else 0
    basis[row["词"]] = row["依据"]
words = sorted(gold)
assert len(words) == 416

import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"设备: {device}")
agent = RLAgent(MODEL_DIR, device=device)

Q_CHOICE = {
    "type": "choice",
    "instructions": "Is this English word part of China's junior-high school English curriculum (grades 7-9; the 2022 national standard ~1600 words plus the PEP textbook vocabulary)? Regular inflections of taught words also count as yes.",
    "criteria": {"yes": "taught in Chinese junior high school", "no": "beyond junior high (senior high, college, or not a real word)"},
}
Q_SCORE = {
    "type": "score",
    "instructions": "Grade this English word by the earliest stage at which Chinese students normally learn it.",
    "criteria": ["grade 7", "grade 8", "grade 9 (junior-high exit)", "senior high (gaokao)", "college and beyond"],
}

rows = []
t0 = time.time()
for i, w in enumerate(words, 1):
    state = f"English word: {w}"
    try:
        r = agent.system_one(state, {"bin": Q_CHOICE, "grade": Q_SCORE})
        p_yes = r["answers"]["bin"]["probabilities"]["yes"]
        conf = r["answers"]["bin"]["confidence"]
        sc = r["answers"]["grade"]["score"]  # 0-4
        rows.append((w, p_yes, conf, sc + 1))
    except Exception as e:
        rows.append((w, None, None, None))
        print(f"  [{w}] 失败: {str(e)[:80]}")
    if i % 50 == 0:
        print(f"  ... {i}/416  ({(time.time()-t0)/i*1000:.0f} ms/词)", flush=True)
dt = (time.time() - t0) / len(words) * 1000
print(f"平均 {dt:.0f} ms/词")

ok = [r for r in rows if r[1] is not None]
print(f"成功 {len(ok)}/416")


def auc(pairs):
    pos = sorted(s for s, y in pairs if y == 1)
    neg = sorted(s for s, y in pairs if y == 0)
    import bisect
    wins = ties = 0
    for p in pos:
        lo = bisect.bisect_left(neg, p)
        hi = bisect.bisect_right(neg, p)
        wins += lo
        ties += hi - lo
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def best_acc(pairs):
    best = 0
    for t in sorted({s for s, _ in pairs}):
        acc = sum(1 for s, y in pairs if (s > t) == (y == 1)) / len(pairs)
        best = max(best, acc)
    return best


pairs_all = [(r[1], gold[r[0]]) for r in ok]
pairs_pure = [(r[1], gold[r[0]]) for r in ok if basis[r[0]] in ("教材", "变形", "边缘", "课标")]
print(f"\n[Laya 零样本·choice] 全部{len(pairs_all)}词: AUC={auc(pairs_all):.3f} 最优阈值准确率={best_acc(pairs_all):.1%}   (0.6B v4: 0.651/77.3%)")
print(f"                    纯单词{len(pairs_pure)}词: AUC={auc(pairs_pure):.3f} 最优阈值准确率={best_acc(pairs_pure):.1%}   (0.6B v4: 0.569/84.7%)")

cal_pos = sum(r[2] for r in ok if gold[r[0]] == 1) / max(1, sum(1 for r in ok if gold[r[0]] == 1))
cal_neg = sum(r[2] for r in ok if gold[r[0]] == 0) / max(1, sum(1 for r in ok if gold[r[0]] == 0))
print(f"校准粗查：金标=是 的平均置信 {cal_pos:.3f} vs 金标=否 的平均置信 {cal_neg:.3f}（差越大校准越有效）")

with open(PROJ + "/data_v4/Laya对照_416金标.tsv", "w", encoding="utf-8") as f:
    f.write("词\t依据\t金标\tLaya_P是\tLaya置信\tLaya学段E\n")
    for w, p, c, sc in sorted(rows, key=lambda r: -(r[1] or 0)):
        f.write(f"{w}\t{basis[w]}\t{'是' if gold[w] else '否'}\t{p if p is None else round(p,4)}\t{c if c is None else round(c,4)}\t{sc if sc is None else round(sc,2)}\n")
print("明细 → data_v4/Laya对照_416金标.tsv")
