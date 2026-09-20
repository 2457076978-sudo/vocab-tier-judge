# -*- coding: utf-8 -*-
"""初中单词判定器 — 人工校对台（J=是 K=否）
启动即开始，每词一次按键，结果即时落盘（中断不丢）。
词单优先级：①A层超纲扫描的新发现+骑线未注词（你的待决清单）
           ②v4 读数中间带 2.5~3.5 的分层样本（校准金标签）
用法：~/venvs/mlx/bin/python scripts/label_words.py
"""
import csv
import json
import os
import random
import sys
import termios
import tty

PROJ = "/Users/wayne/Desktop/工作文档库/05-网站与AI工作区/初中单词判定器"
RESULTS = PROJ + "/data_v4/人工金标签.tsv"
QUEUE_CACHE = PROJ + "/data_v4/人工校对_词单.tsv"


def build_queue():
    """超纲扫描未注词优先，再补中间带分层样本"""
    queue = []  # (词, v4读数, 来源)
    seen = set()
    scan = PROJ + "/data_v4/A层超纲扫描.tsv"
    if os.path.exists(scan):
        for row in csv.DictReader(open(scan, encoding="utf-8"), delimiter="\t"):
            if row["已有注释"] == "否":
                w, e = row["词"], row["旋钮读数"]
                queue.append((w, e, "A层扫描新发现"))
                seen.add(w)
    evalf = PROJ + "/data_v4/eval_留出分档.tsv"
    if os.path.exists(evalf):
        mid = []
        for row in csv.DictReader(open(evalf, encoding="utf-8"), delimiter="\t"):
            w, e = row["词"], float(row["期望学段"])
            if w not in seen and 2.5 <= e <= 3.5:
                mid.append((w, f"{e:.2f}", "边界带金标"))
        random.Random(42).shuffle(mid)
        queue += mid[:300]
    with open(QUEUE_CACHE, "w", encoding="utf-8") as f:
        f.write("词\tv4读数\t来源\n")
        for w, e, s in queue:
            f.write(f"{w}\t{e}\t{s}\n")
    return queue


def load_state(queue):
    done = {}
    if os.path.exists(RESULTS):
        for row in csv.DictReader(open(RESULTS, encoding="utf-8"), delimiter="\t"):
            done[row["词"]] = row["人工判定"]
    return done


def append_result(word, reading, source, verdict):
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", encoding="utf-8") as f:
        if new:
            f.write("词\tv4读数\t来源\t人工判定\n")
        f.write(f"{word}\t{reading}\t{source}\t{verdict}\n")


def main():
    queue = build_queue()
    done = load_state(queue)
    pending = [(w, e, s) for w, e, s in queue if w not in done]
    undone = []  # 撤销栈
    print(f"词单 {len(queue)} 个，已完成 {len(done)}，本次待校 {len(pending)}")
    print("J = 是（初中教学词汇）   K = 否   U = 撤销上一条   Q/Ctrl-C = 保存退出\n")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd) if sys.stdin.isatty() else None
    n_yes = sum(1 for v in done.values() if v == "是")
    n_no = len(done) - n_yes
    try:
        if old is not None:
            tty.setcbreak(fd)

        def read_key():
            if old is not None:
                return sys.stdin.read(1).lower()
            line = sys.stdin.readline().strip().lower()
            return line[0] if line else "q"  # 管道模式：一行一键（自测/自动化用）

        idx = 0
        while idx < len(pending):
            w, e, s = pending[idx]
            sys.stdout.write(
                f"\r[{idx+1}/{len(pending)}]  {w:<22} 旋钮读数 {e}  ({s})   "
                f"[J=是 K=否] 是:{n_yes} 否:{n_no}  ")
            sys.stdout.flush()
            ch = read_key()
            if ch == "j":
                append_result(w, e, s, "是"); n_yes += 1; undone.append((w, e, s)); idx += 1
            elif ch == "k":
                append_result(w, e, s, "否"); n_no += 1; undone.append((w, e, s)); idx += 1
            elif ch == "u" and undone:
                w0, e0, s0 = undone.pop()
                lines = open(RESULTS, encoding="utf-8").readlines()
                for i in range(len(lines) - 1, 0, -1):
                    if lines[i].split("\t")[0] == w0:
                        del lines[i]
                        break
                open(RESULTS, "w", encoding="utf-8").writelines(lines)
                if w0 in done:
                    del done[w0]
                idx -= 1
                sys.stdout.write(f"\n  已撤销 {w0}\n")
            elif ch in ("q", "\x03"):
                break
        print(f"\n\n本次收工：累计 是:{n_yes} 否:{n_no}，结果在 data_v4/人工金标签.tsv")
        if idx >= len(pending):
            print("词单全部校完！")
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
