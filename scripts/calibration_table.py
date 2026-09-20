# -*- coding: utf-8 -*-
"""初中单词判定器 — 校准表（RLCD 土法平替）
输入：eval_留出分档.tsv（词/真档/期望学段）
输出：E 读数分桶 → 真档分布 / 峰值档命中率 / within-1 命中率
用途：让旋钮读数带上经验置信语义——"读 3.8 的词，历史上 62% 真档是 4"
"""
import csv
import os
import sys
from collections import Counter

PROJ = "/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/初中单词判定器"
DV = PROJ + "/" + (sys.argv[1] if len(sys.argv) > 1 else "data_v4")
BIN_W = 0.25

rows = []
with open(DV + "/eval_留出分档.tsv", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rows.append((r["词"], int(r["真档"]), float(r["期望学段"])))
assert rows, "空输入"

bins = {}
for w, t, e in rows:
    b = min(int((e - 1.0) / BIN_W), int((5.0 - 1.0) / BIN_W))  # [1,5] 共 16 桶
    bins.setdefault(b, []).append((t, e))

out = [["读数区间", "词数", "真档分布(1-5)", "峰值档命中", "within-1命中"]]
for b in sorted(bins):
    items = bins[b]
    lo = 1.0 + b * BIN_W
    hi = lo + BIN_W
    cnt = Counter(t for t, _ in items)
    n = len(items)
    peak = max(range(1, 6), key=lambda t: cnt.get(t, 0))
    # 峰值档命中率：真档=该桶众数档 的比例
    hit_peak = cnt.get(peak, 0) / n
    hit_w1 = sum(1 for t, _ in items if abs(t - (lo + hi) / 2) <= 1.05) / n
    dist = " ".join(f"{t}:{cnt.get(t, 0)}" for t in range(1, 6))
    out.append([f"[{lo:.2f},{hi:.2f})", str(n), dist, f"{hit_peak:.0%}", f"{hit_w1:.0%}"])

dst = DV + "/校准表.tsv"
with open(dst, "w", encoding="utf-8") as f:
    f.write("\t".join(out[0]) + "\n")
    for r in out[1:]:
        f.write("\t".join(r) + "\n")
print("\t".join(out[0]))
for r in out[1:]:
    print("\t".join(r))
print(f"CALIB_DONE -> {dst}（共 {len(rows)} 词）")
