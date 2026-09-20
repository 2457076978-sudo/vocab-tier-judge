# -*- coding: utf-8 -*-
"""扫描 LayerText 最新 A 层十章，旋钮找超纲词
提词：剥（中文注）/[P01]标记/章节标题；剔 AF 专名表
打分：v4 旋钮批量前向；超纲带 E>=3.5，骑线带 3.0<=E<3.5
对照：AF注释词典1561词 → 分"已注/旋钮新发现"
产物：data_v4/A层超纲扫描_日期.tsv
"""
import csv
import glob
import math
import os
import re
from collections import Counter

import mlx.core as mx
from mlx_lm import load

BASE = "/Users/wayne/Desktop/工作文档库"
KF = BASE + "/01-教学工作/名著阅读工作区_AnimalFarm/知识文件"
AFDIR = BASE + "/01-教学工作/名著阅读工作区_AnimalFarm/调适工作区/重制三版"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"
MODEL = os.path.expanduser("~/.omlx/models/Qwen3-0.6B-bf16")
ADAPTER = PROJ + "/adapters/" + os.environ.get("JUDGE_ADAPTER", "v4")

PROMPT = ("给英文单词的学段难度定档（按中国学生普通进度，取最早学段）。\n单词：%s\n"
          "选项：一=七年级 二=八年级 三=九年级（初中毕业线） 四=高中（高考3500内） 五=大学毕业以上\n/no_think")

CJK_ANNO = re.compile(r"（[^）]*[\u4e00-\u9fff][^）]*）")
PP_MARK = re.compile(r"\[P+P?\d+\]")
TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")

# ---- 提词 ----
files = sorted(glob.glob(AFDIR + "/第*/原文_A层85_*_工序化.md"))
files = [f for f in files if not f.endswith(".bak")]
assert len(files) == 10, f"A层文件 {len(files)} != 10"
freq = Counter()
for fp in files:
    for line in open(fp, encoding="utf-8"):
        if line.startswith("#"):
            continue
        line = CJK_ANNO.sub(" ", line)
        line = PP_MARK.sub(" ", line)
        line = line.replace("--", " ")  # 管线编辑残留的双连字符，防跨词粘连
        for t in TOKEN.findall(line):
            t = t.lower().rstrip("'")   # 撇号/所有格尾巴
            if t.endswith("s") and t[:-1] in ():
                pass
            if len(t) >= 2:
                freq[t] += 1
print(f"A层十章提词：{len(freq)} 词种，总 token {sum(freq.values())}")

proper = set()
for line in open(KF + "/专名_AnimalFarm.txt", encoding="utf-8"):
    w = line.strip().lower()
    if w and not w.startswith("#"):
        proper.add(w)
annotated = set()
for row in csv.DictReader(open(KF + "/AF注释词典_v1.csv", encoding="utf-8-sig")):
    w = (row.get("词") or "").strip().lower()
    if w:
        annotated.add(w)
print(f"专名剔除 {len(proper)}，注释词典 {len(annotated)}")

# 变形归并：token 可还原成语料内另一词（milked→milk）则记基础形
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
    "wore": "wear", "built": "build", "bought": "buy", "caught": "catch", "taught": "teach",
    "thought": "think", "sold": "sell", "sent": "send", "spent": "spend", "lost": "lose",
    "won": "win", "met": "meet", "led": "lead", "paid": "pay", "ran": "run",
    "stood": "stand", "understood": "understand", "became": "become", "tore": "tear",
    "hissed": "hiss", "whips": "whip", "milked": "milk",
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
    return outs

vocab = set(freq)
def base_form(w):
    for b in deinflect(w):
        if b in vocab and b != w:
            return b
    return w

words = []
base_map = {}
for w in freq:
    if w in proper or "'" in w:
        continue
    b = base_form(w)
    base_map[w] = b
    if b == w:
        words.append(w)
print(f"待打分 {len(words)} 词（变形已归并到基础形 {sum(1 for w in base_map if base_map[w] != w)} 个）")

# ---- 批量旋钮 ----
model, tokenizer = load(MODEL, adapter_path=ADAPTER)
lids = []
for lab in ("一", "二", "三", "四", "五"):
    ids = tokenizer.encode(lab, add_special_tokens=False)
    assert len(ids) == 1
    lids.append(ids[0])
think = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)

def batch_read(batch):
    seqs = []
    for w in batch:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT % w}],
            tokenize=True, add_generation_prompt=True)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        seqs.append(list(ids) + think)
    maxlen = max(len(s) for s in seqs)
    arr = mx.zeros((len(seqs), maxlen), dtype=mx.int32)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = mx.array(s)
    logits = model(arr)
    out = []
    for i, s in enumerate(seqs):
        xs = [float(logits[i, len(s) - 1, k]) for k in lids]
        m = max(xs)
        es = [math.exp(x - m) for x in xs]
        ps = [e / sum(es) for e in es]
        out.append(sum((j + 1) * p for j, p in enumerate(ps)))
    return out

scores = {}
B = 32
for i in range(0, len(words), B):
    chunk = words[i:i + B]
    for w, e in zip(chunk, batch_read(chunk)):
        scores[w] = e
    if (i // B) % 10 == 0:
        print(f"  ... {min(i + B, len(words))}/{len(words)}", flush=True)

# ---- 汇总（计数并入变形，注释按词族查）----
cluster_ann = {}
cluster_cnt = Counter()
for w, b in base_map.items():
    cluster_cnt[b] += freq[w]
    if w in annotated:
        cluster_ann[b] = True

over = [(w, cluster_cnt[w], scores[w]) for w in words if scores[w] >= 3.5]
line_ = [(w, cluster_cnt[w], scores[w]) for w in words if 3.0 <= scores[w] < 3.5]
over.sort(key=lambda x: -x[2])
line_.sort(key=lambda x: -x[2])

dst = PROJ + "/data_v4/A层超纲扫描.tsv"
with open(dst, "w", encoding="utf-8") as f:
    f.write("词\tA层出现次数(含变形)\t旋钮读数\t带\t已有注释\n")
    for w, c, e in over:
        f.write(f"{w}\t{c}\t{e:.2f}\t超纲(≥3.5)\t{'是' if cluster_ann.get(w) else '否'}\n")
    for w, c, e in line_:
        f.write(f"{w}\t{c}\t{e:.2f}\t骑线(3.0-3.5)\t{'是' if cluster_ann.get(w) else '否'}\n")

print(f"\n=== 超纲带 E≥3.5：{len(over)} 词（已注 {sum(1 for w,_,_ in over if cluster_ann.get(w))} / 新发现 {sum(1 for w,_,_ in over if not cluster_ann.get(w))}）===")
print("新发现（旋钮认为超纲但词典没收）：")
for w, c, e in over:
    if not cluster_ann.get(w):
        print(f"  {w:<16} 出现{c}次  E={e:.2f}")
print(f"\n骑线带 3.0-3.5：{len(line_)} 词（已注 {sum(1 for w,_,_ in line_ if cluster_ann.get(w))}）")
print("TOP15:", [(w, round(e, 2)) for w, c, e in line_[:15]])
print(f"SCAN_DONE -> {dst}")
