# -*- coding: utf-8 -*-
"""章节学段读数生成器：工序化 md → <基名>_学段读数.json（App 审校台着色数据源）
解析与 App reader 同口径：## 头跳过、（中文注）剥除、[P01]/[PP02] 剥除、-- 断词、撇号尾剥。
词归并到基础形（该文件词集内回退），批量 v6 旋钮打分 + 锚点层 v0.3 户口纠偏，键=小写基础形。
用法：~/venvs/mlx/bin/python scripts/scan_chapter_grades.py <md文件...>
"""
import json
import math
import os
import re
import sys

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anchor_layer import anchor_e

PROJ = "/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/初中单词判定器"
MODEL = os.path.expanduser(os.environ.get("JUDGE_MODEL", "/Users/wayne/models/MiniCPM5-1B-mlx-bf16"))
ADAPTER = PROJ + "/adapters/" + os.environ.get("JUDGE_ADAPTER", "mcpm5_v7")

PROMPT = ("给英文单词的学段难度定档（按中国学生普通进度，取最早学段）。\n单词：%s\n"
          "选项：一=七年级 二=八年级 三=九年级（初中毕业线） 四=高中（高考3500内） 五=大学毕业以上\n/no_think")
CJK_ANNO = re.compile(r"（[^）]*[\u4e00-\u9fff][^）]*）")
PP_MARK = re.compile(r"\[P+P?\d+\]")
TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")

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


def words_of(path):
    """返回 (小写token序列, 专名候选集)。专名判定=确定性组合闸：
    ①句中位置出现过大写形、且全文从未出现小写形 ②且不在课标1673（防句首偶现普通词误伤）。"""
    toks = []
    cap = {}      # 小写形 -> {'mid': 句中大写次数, 'low': 小写出现次数}
    line_start = True
    for line in open(path, encoding="utf-8"):
        if line.startswith("#"):
            continue
        line = CJK_ANNO.sub(" ", line)
        line = PP_MARK.sub(" ", line)
        line = line.replace("--", " ")
        for t in TOKEN.findall(line):
            raw = t.rstrip("'")
            lo = raw.lower()
            if len(lo) < 2:
                line_start = raw in ".!?"
                continue
            st = cap.setdefault(lo, {"mid": 0, "start": 0, "low": 0})
            if raw[0].isupper():
                st["mid" if not line_start else "start"] += 1
            else:
                st["low"] += 1
            toks.append(lo)
            line_start = t[-1] in ".!?" or raw[-1] in ".!?"
    if not hasattr(words_of, "_kebiao"):
        kb = set()
        for line in open(PROJ + "/data/eval_课标.tsv", encoding="utf-8").readlines()[1:]:
            w = line.split("\t")[0].strip().lower()
            if w.isascii():
                kb.add(w)
        words_of._kebiao = kb
    propers = {w for w, s in cap.items()
               if s["low"] == 0 and len(w) >= 2 and w not in words_of._kebiao
               and (s["mid"] >= 1 or s["start"] >= 2)}
    return toks, propers


def main():
    files = sys.argv[1:]
    assert files, "用法：scan_chapter_grades.py <md文件...>"

    model, tokenizer = load(MODEL, adapter_path=ADAPTER)
    lids = []
    for lab in ("一", "二", "三", "四", "五"):
        ids = tokenizer.encode(lab, add_special_tokens=False)
        assert len(ids) == 1
        lids.append(ids[0])
    think = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False) if "Qwen" in MODEL else []

    def batch_read(ws):
        seqs = []
        for w in ws:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT % w}],
                tokenize=True, add_generation_prompt=True)
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            seqs.append(list(ids) + think)
        L = max(len(s) for s in seqs)
        arr = mx.zeros((len(seqs), L), dtype=mx.int32)
        for i, s in enumerate(seqs):
            arr[i, :len(s)] = mx.array(s)
        logits = model(arr)
        out = []
        for i, s in enumerate(seqs):
            xs = [float(logits[i, len(s) - 1, k]) for k in lids]
            m = max(xs)
            es = [math.exp(x - m) for x in xs]
            ps = [e / sum(es) for e in es]
            e = sum((j + 1) * p for j, p in enumerate(ps))
            # 置信度=期望档±1 档的概率质量（within-1 口径，配套校准表语义）
            conf = sum(p for j, p in enumerate(ps) if abs((j + 1) - e) <= 1)
            out.append((e, conf))
        return out

    # 全部文件词集合并打分（去重+归并基础形）；专名候选不进打分与着色，单列 proper 通道
    per_file = {}
    proper_of = {}
    vocab = set()
    for f in files:
        toks, propers = words_of(f)
        toks = [t for t in toks if t not in propers]
        vocab.update(toks)
        per_file[f] = toks
        proper_of[f] = propers
    base_map = {}
    bases = set()
    for w in vocab:
        b = w
        for cand in deinflect(w):
            if cand in vocab and cand != w:
                b = cand
                break
        base_map[w] = b
        bases.add(b)
    print(f"共 {len(files)} 文件，词种 {len(vocab)}，归并后待打分 {len(bases)}")

    scores = {}
    confs = {}
    anchored_names = []
    held_names = []
    B = 32
    bl = sorted(bases)
    for i in range(0, len(bl), B):
        chunk = bl[i:i + B]
        for w, (e, c) in zip(chunk, batch_read(chunk)):
            ae, st, act = anchor_e(w, e)
            scores[w] = ae
            confs[w] = c
            if act in ("封顶", "托底"):
                anchored_names.append((w, e, ae, st, act))
            elif act == "正典缺口·留审":
                held_names.append((w, e, st))
        if (i // B) % 20 == 0:
            print(f"  ... {min(i + B, len(bl))}/{len(bl)}", flush=True)

    moved = {w for w, e, ae, st, act in anchored_names if abs(ae - e) > 1e-9}
    if held_names:
        with open(os.path.join(PROJ, "data_v6", "正典缺口_留审.tsv"), "w", encoding="utf-8") as hf:
            hf.write("词\t模型E\t超纲侧户口\n")
            for w, e, st in sorted(set(held_names)):
                hf.write(f"{w}\t{e:.3f}\t{st}\n")
        print(f"正典缺口留审 {len(set(held_names))} 词 → data_v6/正典缺口_留审.tsv")
    for f in files:
        fmap = {}
        cmap = {}
        for w in per_file[f]:
            fmap[base_map[w]] = round(scores[base_map[w]], 3)
            cmap[base_map[w]] = round(confs[base_map[w]], 3)
        dst = re.sub(r"\.md$", "", f, flags=re.I) + "_学段读数.json"
        with open(dst, "w", encoding="utf-8") as out:
            json.dump({
                "schemaVersion": 1,
                "generator": "初中单词判定器 scan_chapter_grades.py",
                "adapter": os.environ.get("JUDGE_ADAPTER", "mcpm5_v7"),
                "anchor": "学段户口v0.3·最早侧封顶托底",
                "generated": "2026-09-20",
                "words": dict(sorted(fmap.items(), key=lambda kv: -kv[1])),
                "conf": cmap,
                "proper": sorted(proper_of[f]),
            }, out, ensure_ascii=False, indent=0)
        hot = sum(1 for e in fmap.values() if e >= 3.5)
        nmoved = len(moved & set(fmap))
        print(f"{os.path.basename(f)} → {os.path.basename(dst)}（{len(fmap)} 词，超纲带 {hot}，锚定纠偏 {nmoved}）")
    print("SCAN_GRADES_DONE")


if __name__ == "__main__":
    main()
