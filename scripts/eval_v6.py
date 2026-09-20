# -*- coding: utf-8 -*-
"""v6 评估 + L1 温度缩放（自制 RLCD 收尾件）
① 留出集指标（同 eval_rheostat 口径：分档准确/within-1/档均值）
② 温度拟合：在留出集上找 T 最小化真档 NLL → 校准前后的可靠性分桶对比
③ 产出：data_v6/eval_留出分档.tsv（含 logits）+ 校准表_v6.tsv + 最优 T
"""
import json
import math
import os
from collections import defaultdict

import mlx.core as mx
from mlx_lm import load

BASE = "/Users/wayne/Desktop/工作文档库"
PROJ = BASE + "/05-网站与AI工作区/初中单词判定器"
MODEL = os.path.expanduser("~/.omlx/models/Qwen3-0.6B-bf16")
ADAPTER = PROJ + "/adapters/" + os.environ.get("JUDGE_ADAPTER", "v6")
DV = PROJ + "/" + os.environ.get("JUDGE_DATA", "data_v6")

PROMPT = ("给英文单词的学段难度定档（按中国学生普通进度，取最早学段）。\n单词：%s\n"
          "选项：一=七年级 二=八年级 三=九年级（初中毕业线） 四=高中（高考3500内） 五=大学毕业以上\n/no_think")
LAB2TIER = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}

model, tokenizer = load(MODEL, adapter_path=ADAPTER)
LIDS = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LAB2TIER]
THINK = tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)


def logits_of(ws):
    seqs = []
    for w in ws:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT % w}],
            tokenize=True, add_generation_prompt=True)
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        seqs.append(list(ids) + THINK)
    L = max(len(s) for s in seqs)
    arr = mx.zeros((len(seqs), L), dtype=mx.int32)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = mx.array(s)
    out = model(arr)
    return [[float(out[i, len(s) - 1, k]) for k in LIDS] for i, s in enumerate(seqs)]


def main():
    valid = {}
    for line in open(DV + "/valid.jsonl", encoding="utf-8"):
        r = json.loads(line)
        w = r["prompt"].split("单词：")[1].split("\n")[0]
        valid.setdefault(w, defaultdict(int))[LAB2TIER[r["completion"]]] += 1
    words = sorted(valid)
    # 真档 = 留出采样的众数（软标签任务的评价锚）
    truth = {w: max(d, key=d.get) for w, d in valid.items()}

    raws = []
    B = 32
    for i in range(0, len(words), B):
        raws += logits_of(words[i:i + B])
        if (i // B) % 20 == 0:
            print(f"  ... {min(i + B, len(words))}/{len(words)}", flush=True)

    def softmax_T(z, T):
        m = max(v / T for v in z)
        es = [math.exp(v / T - m) for v in z]
        return [e / sum(es) for e in es]

    # ① 未校准指标（T=1）
    def metrics(T):
        hit = w1 = 0
        dsum = 0.0
        by_t = defaultdict(list)
        for w, z in zip(words, raws):
            ps = softmax_T(z, T)
            e = sum((j + 1) * p for j, p in enumerate(ps))
            t = truth[w]
            hit += 1 if round(e) == t else 0
            w1 += 1 if abs(e - t) <= 1 else 0
            dsum += abs(e - t)
            by_t[t].append(e)
        n = len(words)
        means = {t: sum(v) / len(v) for t, v in sorted(by_t.items())}
        return hit / n, w1 / n, dsum / n, means

    h1, w11, d1, m1 = metrics(1.0)
    print(f"\n[未校准 T=1] 准确 {h1:.1%} | within-1 {w11:.1%} | 平均档差 {d1:.2f}")
    print("  档均值:", {t: round(v, 2) for t, v in m1.items()})

    # ② 温度拟合：最小化真档 NLL（网格 T∈[0.5,5]）
    def nll(T):
        s = 0.0
        for w, z in zip(words, raws):
            ps = softmax_T(z, T)
            s -= math.log(max(ps[truth[w] - 1], 1e-9))
        return s / len(words)

    Ts = [0.5 + 0.05 * i for i in range(91)]
    best_T = min(Ts, key=nll)
    print(f"[温度拟合] 最优 T={best_T:.2f}（NLL {nll(1.0):.3f} → {nll(best_T):.3f}）")
    h2, w12, d2, m2 = metrics(best_T)
    print(f"[校准后 T={best_T:.2f}] 准确 {h2:.1%} | within-1 {w12:.1%} | 平均档差 {d2:.2f}")
    print("  档均值:", {t: round(v, 2) for t, v in m2.items()})

    # ③ 校准表 v6（用校准后概率）
    with open(DV + "/校准表_v6.tsv", "w", encoding="utf-8") as f:
        f.write("读数区间\t词数\t真档分布(1-5)\t峰值档命中\twithin-1命中\n")
        bins = defaultdict(list)
        for w, z in zip(words, raws):
            ps = softmax_T(z, best_T)
            e = sum((j + 1) * p for j, p in enumerate(ps))
            b = min(int((e - 1.0) / 0.25), 15)
            bins[b].append((truth[w], e))
        for b in sorted(bins):
            items = bins[b]
            lo = 1.0 + b * 0.25
            from collections import Counter
            cnt = Counter(t for t, _ in items)
            n = len(items)
            peak = max(range(1, 6), key=lambda t: cnt.get(t, 0))
            mid = lo + 0.125
            w1b = sum(1 for t, _ in items if abs(t - mid) <= 1.05) / n
            dist = " ".join(f"{t}:{cnt.get(t, 0)}" for t in range(1, 6))
            f.write(f"[{lo:.2f},{lo+0.25:.2f})\t{n}\t{dist}\t{cnt.get(peak,0)/n:.0%}\t{w1b:.0%}\n")
    with open(DV + "/eval_留出分档.tsv", "w", encoding="utf-8") as f:
        f.write("词\t真档\tE(T=1)\tE(校准)\n")
        for w, z in zip(words, raws):
            e1 = sum((j + 1) * p for j, p in enumerate(softmax_T(z, 1.0)))
            e2 = sum((j + 1) * p for j, p in enumerate(softmax_T(z, best_T)))
            f.write(f"{w}\t{truth[w]}\t{e1:.3f}\t{e2:.3f}\n")
    print(f"EVAL_V6_DONE（T*={best_T:.2f}，校准表_v6.tsv 已出）")


if __name__ == "__main__":
    main()
