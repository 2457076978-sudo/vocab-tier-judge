# -*- coding: utf-8 -*-
"""初中单词判定器 v1 — 全量提词：扫 01-教学工作 全部教学资料
只读源文件，零修改零删除。解析：md/txt 直读、html 剥标签、docx/pptx(zip+XML正则)、
doc(textutil)、pdf(fitz·有文本层)、zip(内存解压·GBK文件名)。
排除：成绩数据库、班级画像、名著阅读工作区(AF域)、xls/xlsx、媒体、*.kdtmp。
统计：每词 频次 + 材料类覆盖数 + 词表类覆盖数（教材词典单独标记）。
去重：文件内容哈希（md5），docx+md 孪生/_1副本/师生版同文只计一次。
产物：data/词汇总表_教学工作区.csv + 扫描件清单 + 提取统计.json
"""
import csv
import hashlib
import html as htmllib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import defaultdict

BASE = "/Users/wayne/Desktop/工作文档库"
ROOT = BASE + "/01-教学工作"
OUT = BASE + "/05-网站与AI工作区/初中单词判定器/data"

PRUNE_DIRS = {"成绩数据库", "班级画像", "名著阅读工作区_AnimalFarm", ".git", "node_modules",
              "__pycache__", ".DS_Store"}
SKIP_EXT = {".xls", ".xlsx", ".mp3", ".mp4", ".jpg", ".jpeg", ".png", ".gif",
            ".kdtmp", ".downloading", ".zip.part"}
WORDLIST_DIR_MARK = "教材词典"  # 词表类：默写本/词表，覆盖数单独计

WT = re.compile(r"<w:t[^>]*>([^<]*)</w:t>")
AT = re.compile(r"<a:t>([^<]*)</a:t>")
TAG = re.compile(r"<[^>]+>")
TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*")

stats = defaultdict(int)
scanned_pdfs = []


def extract_doc(path):
    try:
        r = subprocess.run(["textutil", "-convert", "txt", "-stdout", path],
                           capture_output=True, timeout=60)
        return r.stdout.decode("utf-8", "ignore")
    except Exception:
        stats["doc失败"] += 1
        return ""


def extract_pdf(path):
    import fitz
    try:
        with fitz.open(path) as doc:
            text = "\n".join(p.get_text() for p in doc)
        if len(text.strip()) < 30:
            scanned_pdfs.append(path)
            stats["pdf无文本层"] += 1
            return ""
        return text
    except Exception:
        stats["pdf解析失败"] += 1
        return ""


def extract_zip(path):
    out = []
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                try:
                    name = info.filename.encode("cp437").decode("gbk", "ignore")
                except Exception:
                    name = info.filename
                ext = os.path.splitext(name)[1].lower()
                if info.is_dir() or ext not in {".md", ".txt", ".docx", ".pptx", ".html", ".htm"}:
                    continue
                if info.file_size > 30 * 1024 * 1024:
                    continue
                data = z.read(info)
                out.append((name, ext, data))
    except Exception:
        stats["zip失败"] += 1
    texts = []
    for name, ext, data in out:
        texts.append(parse_bytes(name, ext, data))
    return "\n".join(t for t in texts if t)


def parse_bytes(name, ext, data):
    try:
        if ext in {".md", ".txt"}:
            return data.decode("utf-8", "ignore")
        if ext in {".html", ".htm"}:
            return TAG.sub(" ", data.decode("utf-8", "ignore"))
        if ext == ".docx":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                xml = z.read("word/document.xml").decode("utf-8", "ignore")
            return " ".join(htmllib.unescape(m) for m in WT.findall(xml))
        if ext == ".pptx":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                parts = []
                for n in sorted(z.namelist()):
                    if re.match(r"ppt/slides/slide\d+\.xml$", n):
                        xml = z.read(n).decode("utf-8", "ignore")
                        parts.append(" ".join(htmllib.unescape(m) for m in AT.findall(xml)))
                return "\n".join(parts)
    except Exception:
        stats[f"{ext}内存解析失败"] += 1
    return ""


def extract_file(path, ext):
    if ext in {".md", ".txt"}:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in {".html", ".htm"}:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return TAG.sub(" ", f.read())
    if ext == ".docx":
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
        return " ".join(htmllib.unescape(m) for m in WT.findall(xml))
    if ext == ".pptx":
        with zipfile.ZipFile(path) as z:
            parts = []
            for n in sorted(z.namelist()):
                if re.match(r"ppt/slides/slide\d+\.xml$", n):
                    xml = z.read(n).decode("utf-8", "ignore")
                    parts.append(" ".join(htmllib.unescape(m) for m in AT.findall(xml)))
            return "\n".join(parts)
    if ext == ".doc":
        return extract_doc(path)
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".zip":
        return extract_zip(path)
    return ""


def main():
    freq = defaultdict(int)
    mat_cover = defaultdict(set)   # 材料类覆盖（文件id集合）
    list_cover = defaultdict(set)  # 词表类覆盖
    seen_hash = {}
    files_parsed = 0

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT or ext.startswith(".kd"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, ROOT)
            size = os.path.getsize(path)
            if size > 200 * 1024 * 1024:
                stats["超大跳过"] += 1
                continue
            stats[f"ext{ext}"] += 1
            text = extract_file(path, ext)
            if not text:
                continue
            toks = [t.lower() for t in TOKEN.findall(text)]
            toks = [t for t in toks if len(t) >= 2 and any(c.isalpha() for c in t)]
            if len(toks) < 10:
                stats[f"英文token不足10_{ext}"] += 1
            h = hashlib.md5(" ".join(toks).encode()).hexdigest()
            if h in seen_hash:
                stats["内容哈希去重"] += 1
                continue
            seen_hash[h] = rel
            files_parsed += 1
            is_list = WORDLIST_DIR_MARK in rel
            fid = len(seen_hash)
            for t in set(toks):
                (list_cover if is_list else mat_cover)[t].add(fid)
            for t in toks:
                freq[t] += 1

    os.makedirs(OUT, exist_ok=True)
    with open(OUT + "/词汇总表_教学工作区.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["词", "频次", "材料类覆盖数", "词表类覆盖数"])
        for word in sorted(freq, key=lambda x: -freq[x]):
            w.writerow([word, freq[word], len(mat_cover[word]), len(list_cover[word])])

    with open(OUT + "/扫描件清单.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(scanned_pdfs) + "\n")

    summary = {
        "解析文件数": files_parsed,
        "去重丢弃": stats["内容哈希去重"],
        "词种数": len(freq),
        "扫描pdf数": len(scanned_pdfs),
        "统计": dict(stats),
    }
    with open(OUT + "/提取统计.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    total_seen = files_parsed + stats["内容哈希去重"]
    assert 3500 < total_seen < 7000, f"解析+去重总数 {total_seen} 异常"
    assert len(freq) > 8000, f"词种数 {len(freq)} 异常"
    print("EXTRACT_DONE")


if __name__ == "__main__":
    main()
