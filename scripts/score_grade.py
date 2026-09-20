# -*- coding: utf-8 -*-
"""初中单词判定器 v3 — 学段旋钮 CLI（滑动变阻器版）
用法：
  ~/venvs/mlx/bin/python scripts/score_grade.py bother tyranny
  ~/venvs/mlx/bin/python scripts/score_grade.py -f 词表.txt -o 出分.tsv
读数：五档期望学段 E∈[1,5]（1=七年级 2=八年级 3=九年级 4=高中 5=大学毕业以上）
"""
import argparse
import math
import os
import sys

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anchor_layer import anchor_e

BASE = "/Users/wayne/Desktop/工作文档库"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"
MODEL = os.path.expanduser("~/.omlx/models/Qwen3-0.6B-bf16")
ADAPTER = PROJ + "/adapters/" + os.environ.get("JUDGE_ADAPTER", "v6")

PROMPT = ("给英文单词的学段难度定档（按中国学生普通进度，取最早学段）。\n单词：%s\n"
          "选项：一=七年级 二=八年级 三=九年级（初中毕业线） 四=高中（高考3500内） 五=大学毕业以上\n/no_think")
TIER2NAME = {1: "七年级", 2: "八年级", 3: "九年级", 4: "高中", 5: "大学+"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words", nargs="*")
    ap.add_argument("-f", "--file")
    ap.add_argument("-o", "--out")
    args = ap.parse_args()

    words = [w.strip().lower() for w in args.words if w.strip()]
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            words += [l.strip().lower() for l in fh if l.strip() and not l.startswith("#")]
    if not words:
        print(__doc__)
        sys.exit(1)

    model, tokenizer = load(MODEL, adapter_path=ADAPTER)
    lids = []
    for lab in ("一", "二", "三", "四", "五"):
        ids = tokenizer.encode(lab, add_special_tokens=False)
        assert len(ids) == 1
        lids.append(ids[0])
    think = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)

    # ---- 词频融合层 v0.1（确定性，2026-09-20）：模型管泛化，频率管已见证词 ----
    # 规则：fined 自身罕见，但基础形 fine 在课标正册且材料覆盖 185 → 可解码，融合分下拉。
    # 防假阳性：只认 s/es/ed/ing 规则缀（er/ly 太吵：bother≠both+er）；基础形须在课标∪教材正册。
    import csv as _csv
    DECODE_TAX = 0.3  # 规则变形的可解码税：融合分 = min(自身E, 基础形E + 0.3)
    cov, curriculum = {}, set()
    try:
        for row in _csv.DictReader(open(PROJ + "/data/词汇总表_教学工作区.csv", encoding="utf-8-sig")):
            cov[row["词"]] = int(row["材料类覆盖数"] or 0)
    except FileNotFoundError:
        pass
    KF = "/Users/wayne/Desktop/工作文档库/01-教学工作/名著阅读工作区_AnimalFarm/知识文件"
    for line in open("/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/LayerText/assets/wordlists/curriculum_2022_level3_1600.txt", encoding="utf-8"):
        w = line.strip().lower()
        if w and not w.startswith("#"):
            curriculum.add(w)
    import json as _json
    tj = _json.load(open(KF + "/教材单元_人教版.json", encoding="utf-8"))
    curriculum |= {w for w in tj["base"] if len(w) >= 2}
    for bk in tj["books"].values():
        for u in bk.values():
            for w in u.get("词", []):
                w = w.strip().lower()
                if len(w) >= 2 and " " not in w:
                    curriculum.add(w)
    for line in open(KF + "/已学词.txt", encoding="utf-8"):
        w = line.strip().lower()
        if w and not w.startswith("#"):
            curriculum.add(w)

    def deinflect(w):
        outs = set()
        if w.endswith("ies") and len(w) > 4: outs.add(w[:-3] + "y")
        if w.endswith("es") and len(w) > 4: outs.add(w[:-2])
        if w.endswith("s") and len(w) > 3: outs.add(w[:-1])
        if w.endswith("ed") and len(w) > 4:
            outs.add(w[:-2]); outs.add(w[:-1])
            stem = w[:-2]
            if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
        if w.endswith("ing") and len(w) > 5:
            outs.add(w[:-3]); outs.add(w[:-3] + "e")
            stem = w[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
        return outs

    def base_form(w):
        best = None
        for b in deinflect(w):
            if b in curriculum and (best is None or cov.get(b, 0) > cov.get(best, 0)):
                best = b
        return best

    def batch_rheostat(batch):
        """批量前向：右填充+因果注意力下读各自末位 logits（Jev 式并行读数）"""
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
        rows = []
        for i, s in enumerate(seqs):
            xs = [float(logits[i, len(s) - 1, k]) for k in lids]
            m = max(xs)
            es = [math.exp(x - m) for x in xs]
            ps = [e / sum(es) for e in es]
            e = sum((j + 1) * p for j, p in enumerate(ps))
            rows.append((e, ps))
        return rows

    rows = []
    B = 32
    for i in range(0, len(words), B):
        chunk = words[i:i + B]
        # 融合层需要的基础形一并打分（去重后随批前向）
        bases = {b for w in chunk if (b := base_form(w))}
        extra = [b for b in bases if b not in set(chunk)]
        scored = batch_rheostat(chunk + extra)
        for w, (e, ps) in zip(chunk, scored[:len(chunk)]):
            rows.append((w, e, ps, None))
        base_e = {b: scored[j][0] for j, b in enumerate(extra, start=len(chunk)) if j < len(scored)}
        for w, e, ps, _ in list(rows[-len(chunk):]):
            b = base_form(w)
            if b and b in base_e:
                fused = min(e, base_e[b] + DECODE_TAX)
                # 回填融合分
                for k, (ww, ee, pp, _) in enumerate(rows):
                    if ww == w:
                        rows[k] = (w, e, ps, fused)
                        break
        done = min(i + B, len(words))
        print(f"... {done}/{len(words)}", flush=True)
    # 锚点层 v0.3：词表户口封顶/托底（正典优先·最早学段侧），作用于最终读数
    rows = [(w, e, ps, fused, *anchor_e(w, fused if fused is not None else e))
            for (w, e, ps, fused) in rows]
    for i, (w, e, ps, fused, ae, st, act) in enumerate(rows, 1):
        top = max(range(5), key=lambda j: ps[j]) + 1
        extra = f"\t融合E={fused:.2f}" if fused is not None else ""
        mark = " <==" if act in ("封顶", "托底") else ""
        print(f"[{i}/{len(words)}] {w}\tE={e:.2f} ({TIER2NAME[top]}){extra}\t"
              f"锚定E={ae:.2f}[{st or '无户口'}·{act}]{mark}\t" +
              " ".join(f"{j+1}:{p:.2f}" for j, p in enumerate(ps)))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("词\t模型E\t融合E\t锚定E\t户口\t处置\t峰值档\tP1\tP2\tP3\tP4\tP5\n")
            for w, e, ps, fused, ae, st, act in rows:
                top = max(range(5), key=lambda j: ps[j]) + 1
                fu = f"{fused:.3f}" if fused is not None else ""
                f.write(f"{w}\t{e:.3f}\t{fu}\t{ae:.3f}\t{st or '-'}\t{act}\t{top}\t" +
                        "\t".join(f"{p:.3f}" for p in ps) + "\n")
        print(f"已写出: {args.out}")


if __name__ == "__main__":
    main()
