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
    """主户口表 + 留审补录表（补录优先，来源可审计可回滚）。"""
    huji = {}
    if not os.path.exists(HUJI):
        return huji
    with open(HUJI, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                huji[p[0]] = (p[1], float(p[2]))
    bu = os.path.join(os.path.dirname(HUJI), "户口_补录.tsv")
    if os.path.exists(bu):
        with open(bu, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    huji[p[0]] = (p[1], float(p[2]))   # 补录无条件覆盖（人工裁决最高优先）
    return huji


IRREGULAR = {"stolen": "steal", "sung": "sing", "frozen": "freeze", "torn": "tear",
             "woven": "weave", "sworn": "swear", "lain": "lie", "slain": "slay",
             "shown": "show", "hewn": "hew", "cloven": "cleave"}


def _spelling_variants(w):
    """英美拼写对齐（2026-09-20）：labour/practise/traveller 类英式形在词表常缺户口
    或挂较晚档，而美式孪生（labor/practice/traveler）有更早户口——按最早学段取孪生。"""
    outs = set()
    if w.endswith("our") and len(w) > 5:
        outs.add(w[:-3] + "or")            # labour→labor colour→color
    if w.endswith("ise") and len(w) > 5:
        outs.update({w[:-3] + "ize", w[:-3] + "ice"})   # realise→realize practise→practice
    if w.endswith("isation"):
        outs.add(w[:-7] + "ization")
    if w.endswith("re") and len(w) > 4:
        outs.add(w[:-2] + "er")            # centre→center theatre→theater
    if w.endswith("ogue") and len(w) > 5:
        outs.add(w[:-4] + "og")
    if w.endswith("ence") and len(w) > 5:
        outs.add(w[:-4] + "ense")          # defence→defense
    if w.endswith("ller") and len(w) > 5:
        outs.add(w[:-4] + "ler")           # traveller→traveler
    if w.endswith("lling") and len(w) > 6:
        outs.add(w[:-5] + "ling")
    return outs


def deinflect(w):
    outs = {w}
    if w in IRREGULAR:
        outs.add(IRREGULAR[w])
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
    """返回 (锚档名|None, 是否词形直查)。直接户口/变形回退/英美孪生同台竞争，取最早学段。"""
    huji = load()
    if not huji:
        return None, False
    cands = {w} | set(deinflect(w))
    spell = {v for base in cands for v in _spelling_variants(base)}
    best = None
    for c in cands | spell:
        if c in huji and (best is None or huji[c][1] < best[1]):
            best = huji[c]
    if w in huji and best is None:
        best = huji[w]
    return (best[0], best is not None and w in huji and huji[w] == best) if best else (None, False)


JUN_STAGES = {"七年级", "八年级", "九年级", "课标初中", "初中(genkin)"}


def _hyphen_gate(w):
    """部件闸（v7 2026-09-20）：连字符复合词无直接户口时拆件判定。
    部件全初中 → 封顶 3.4（平反）；任一部件超纲 → 托底 3.6；任一部件无户口 → 不动。"""
    if "-" not in w:
        return None
    parts = [p for p in w.split("-") if len(p) >= 2]
    if len(parts) < 2:
        return None
    huji = load()
    if not huji:
        return None
    stages = []
    for p in parts:
        if p in huji:
            stages.append(huji[p][0])
        else:
            bp, _ = huji_of(p)
            stages.append(bp)
    if any(s is None for s in stages):
        return None
    if all(s in JUN_STAGES for s in stages):
        return ("min", 3.4)
    if any(s not in JUN_STAGES for s in stages):
        return ("max", 3.6)
    return None


def _ly_gate(w):
    """-ly 派生闸（v7 2026-09-20）：副词形无户口时回溯形容词基础形
    （highly←high / simply←simple / happily←happy 三种拼写路径），
    基础形初中户口 → 封顶 3.4。误伤方向=把本来就容易的词封低，无害。"""
    if not w.endswith("ly") or len(w) <= 5 or w.endswith("lly") and len(w) <= 6:
        return None
    huji = load()
    if not huji:
        return None
    for stem in (w[:-2], w[:-2] + "e", w[:-3] + "y"):
        if stem in huji and huji[stem][0] in JUN_STAGES:
            return ("min", 3.4)
    return None


def anchor_e(w, e):
    """返回 (锚定E, 锚档|None, 处置)。学段证据取最早：透明派生(ly/连字符全初中件)
    视作初中学段，压过较晚的直接户口（词书累积性：四级书收录≠初中不可解码）。"""
    wl = w.lower()
    gate = _hyphen_gate(wl) or _ly_gate(wl)
    if gate and gate[0] == "min":
        bound = gate[1]
        v = min(e, bound)
        name = "部件闸" if "-" in wl else "ly派生闸"
        return v, name, "平反封顶" if e > bound + 1e-9 else "不动"
    st, _ = huji_of(wl)
    if st is not None:
        kind, bound = CAP_FLOOR[st]
        if kind == "min":
            v = min(e, bound)
            return v, st, "封顶" if e > bound + 1e-9 else "不动"
        if e >= FLOOR_YIELD:
            v = max(e, bound)
            return v, st, "托底" if e < bound - 1e-9 else "不动"
        return e, st, "正典缺口·留审"
    if gate and gate[0] == "max":
        v = max(e, gate[1])
        return v, "部件闸", "超纲托底" if e < gate[1] - 1e-9 else "不动"
    return e, None, "无户口·模型兜底"
