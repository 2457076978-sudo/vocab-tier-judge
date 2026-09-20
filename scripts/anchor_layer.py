# -*- coding: utf-8 -*-
"""锚点层 v0.3.1（正典优先·最早学段侧，2026-09-20 上岗）
规则：词表为锚、模型兜底；户口只做封顶/托底，不重写档内排序。
  初中侧（人教分册/课标/genkin初中，最早者）→ 锚定E = min(E, 3.4)（无条件封顶）
  高中/四级/六级侧 → 锚定E = max(E, 档位下界)，但需模型让渡：
      E ≥ FLOOR_YIELD(3.0) 才托底。模型强读初中(<3.0)而正典无记录 = 正典缺口，
      不托底、留审补录（416 实证：<3.0 冲突仅 7 词且多为教材白体/专名噪音，
      [3.0,3.5) 冲突 37 词才是纠偏主体）。
  无户口 → 锚定E = E（词表外长尾，模型兜底）
查词顺序：词形直查 → 规则变形回退（ed/ing/es/s/ies），户口取最早学段。
依赖 data_v6/学段户口_v2.tsv（由 build_huji_v2.py 生成，本地数据不进开源仓）。
"""
import functools
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUJI = os.path.join(PROJ, "data_v6", "学段户口_v2.tsv")

CAP_FLOOR = {"七年级": ("min", 3.4), "八年级": ("min", 3.4), "九年级": ("min", 3.4),
             "课标初中": ("min", 3.4), "初中(genkin)": ("min", 3.4),
             "高中教材": ("max", 3.6), "高中(genkin)": ("max", 3.6),
             "四级(书)": ("max", 4.0), "四级(genkin)": ("max", 4.0),
             "六级(书)": ("max", 4.3), "六级(genkin)": ("max", 4.3)}

FLOOR_YIELD = 3.0  # 托底需模型至少让渡到骑线；低于此不托底（正典缺口留审）


@functools.lru_cache(maxsize=1)
def load():
    huji = {}
    if not os.path.exists(HUJI):
        return huji
    with open(HUJI, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                huji[p[0]] = (p[1], float(p[2]))
    return huji


def deinflect(w):
    outs = {w}
    if w.endswith("ies") and len(w) > 4: outs.add(w[:-3] + "y")
    if w.endswith("es") and len(w) > 4: outs.add(w[:-2])
    if w.endswith("s") and len(w) > 3: outs.add(w[:-1])
    if w.endswith("ed") and len(w) > 4:
        outs.update({w[:-2], w[:-1]})
        if w.endswith("ied"): outs.add(w[:-3] + "y")
        stem = w[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        outs.update({stem, stem + "e"})
        if len(stem) > 2 and stem[-1] == stem[-2]: outs.add(stem[:-1])
        if w.endswith("ying"): outs.add(w[:-4] + "ie")
    return outs


@functools.lru_cache(maxsize=100000)
def huji_of(w):
    """返回 (锚档名|None, 是否词形直查)"""
    huji = load()
    if not huji:
        return None, False
    if w in huji:
        return huji[w][0], True
    best = None
    for cand in deinflect(w):
        if cand in huji and (best is None or huji[cand][1] < best[1]):
            best = huji[cand]
    return (best[0], False) if best else (None, False)


def anchor_e(w, e):
    """返回 (锚定E, 锚档|None, 处置)"""
    st, _ = huji_of(w.lower())
    if st is None:
        return e, None, "无户口·模型兜底"
    kind, bound = CAP_FLOOR[st]
    if kind == "min":
        v = min(e, bound)
        act = "封顶" if e > bound + 1e-9 else "不动"
    else:
        if e < FLOOR_YIELD:
            return e, st, "正典缺口·留审"
        v = max(e, bound)
        act = "托底" if e < bound - 1e-9 else "不动"
    return v, st, act
