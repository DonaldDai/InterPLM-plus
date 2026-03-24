"""
patch_add_uniprot_idx.py — 为 rsa_per_residue.tsv 补充 UniProt 序列位置
=========================================================================

功能:
  1. 下载 SIFTS 片段映射数据 (如已下载则跳过)
  2. 读取 step5 生成的 rsa_per_residue.tsv
  3. 通过 SIFTS 将每行的 (pdb_id, chain, resi_pdb) 映射到 UniProt 序列位置
  4. 生成新文件 rsa_per_residue_mapped.tsv (原文件不动)

日志:
  mapping_log.tsv  — 每个残基的映射详情 (独立文件)
  process_log.tsv  — 每个蛋白的处理摘要 (独立文件)

断点续跑:
  读取已有输出文件中已完成的 uniprot_id, 跳过已处理的蛋白

用法:
  python patch_add_uniprot_idx.py
  python patch_add_uniprot_idx.py --flush-every 200
"""

import gzip
import csv
import os
import sys
import time
import argparse
import json
import requests
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# 配置
# ============================================================
DATA_DIR = Path("cusdata/01_raw")
RSA_DIR = Path("cusdata/05_rsa")

SIFTS_FILE = DATA_DIR / "sifts_pdb_chain_uniprot.csv.gz"
SIFTS_URL = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz"

INPUT_TSV = RSA_DIR / "rsa_per_residue.tsv"
OUTPUT_TSV = RSA_DIR / "rsa_per_residue_mapped.tsv"
MAPPING_LOG = RSA_DIR / "sifts_mapping_log.tsv"
PROCESS_LOG = RSA_DIR / "sifts_process_log.tsv"

FLUSH_EVERY = 100  # 每处理多少个蛋白写一次盘 (命令行可覆盖)


# ============================================================
# 1. 下载 SIFTS
# ============================================================
def download_sifts():
    """下载 SIFTS 片段映射文件 (已存在则跳过)"""
    if SIFTS_FILE.exists():
        size_mb = SIFTS_FILE.stat().st_size / (1024 * 1024)
        print(f"[跳过] SIFTS已存在: {SIFTS_FILE} ({size_mb:.1f} MB)")
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"下载SIFTS: {SIFTS_URL}")

    try:
        resp = requests.get(SIFTS_URL, stream=True, timeout=180)
        resp.raise_for_status()
        with open(SIFTS_FILE, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        size_mb = SIFTS_FILE.stat().st_size / (1024 * 1024)
        print(f"  下载完成: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  下载失败: {e}")
        print(f"  手动下载: wget {SIFTS_URL} -O {SIFTS_FILE}")
        if SIFTS_FILE.exists():
            SIFTS_FILE.unlink()
        return False


# ============================================================
# 2. 加载 SIFTS 查找表
# ============================================================
def load_sifts() -> Dict[Tuple[str, str], List[Tuple[int, int, int, int, str]]]:
    """
    构建查找表: {(pdb_id, chain_id): [(pdb_beg, pdb_end, sp_beg, sp_end, uniprot_acc), ...]}

    SIFTS csv 格式 (跳过 # 注释行后):
      PDB,CHAIN,SP_PRIMARY,RES_BEG,RES_END,PDB_BEG,PDB_END,SP_BEG,SP_END
    """
    print(f"加载SIFTS查找表: {SIFTS_FILE}")
    t0 = time.time()

    lookup = defaultdict(list)
    n_rows = 0

    with gzip.open(SIFTS_FILE, "rt") as f:
        # 跳过 # 注释行, 找到真正的 header
        header_line = ""
        for line in f:
            if not line.startswith("#"):
                header_line = line.strip()
                break

        if not header_line:
            print("  ERROR: SIFTS文件为空或格式异常")
            return {}

        fields = header_line.split(",")
        reader = csv.DictReader(f, fieldnames=fields)

        for row in reader:
            try:
                pdb_id = row["PDB"].strip().lower()
                chain = row["CHAIN"].strip()
                sp = row["SP_PRIMARY"].strip()
                pdb_beg = int(row["PDB_BEG"])
                pdb_end = int(row["PDB_END"])
                sp_beg = int(row["SP_BEG"])
                sp_end = int(row["SP_END"])
                lookup[(pdb_id, chain)].append((pdb_beg, pdb_end, sp_beg, sp_end, sp))
                n_rows += 1
            except (ValueError, KeyError):
                continue

    elapsed = time.time() - t0
    print(f"  加载完成: {n_rows} 条区段, {len(lookup)} 个 (pdb, chain) 对, "
          f"耗时 {elapsed:.1f}s")
    return dict(lookup)


# ============================================================
# 3. 单残基映射
# ============================================================
def map_residue(
    lookup: Dict,
    pdb_id: str,
    chain: str,
    resi_pdb: str,
    target_uniprot: str,
) -> Tuple[str, str]:
    """
    返回 (uniprot_idx, status)
      uniprot_idx: UniProt序列位置 (str), 映射失败为空字符串
      status: "mapped" / "no_segment" / "parse_error"
    """
    try:
        resi_int = int(resi_pdb)
    except (ValueError, TypeError):
        return "", "parse_error"

    segments = lookup.get((pdb_id.lower(), chain), [])
    if not segments:
        return "", "no_segment"

    # 优先匹配目标UniProt ID
    for pdb_beg, pdb_end, sp_beg, sp_end, sp_acc in segments:
        if sp_acc == target_uniprot and pdb_beg <= resi_int <= pdb_end:
            return str(sp_beg + (resi_int - pdb_beg)), "mapped"

    # 放宽: 不限定UniProt ID
    # for pdb_beg, pdb_end, sp_beg, sp_end, sp_acc in segments:
    #     if pdb_beg <= resi_int <= pdb_end:
    #         return str(sp_beg + (resi_int - pdb_beg)), "mapped"

    return "", "no_segment"


# ============================================================
# 4. 断点恢复: 读取已完成的 uniprot_id
# ============================================================
def load_checkpoint() -> Tuple[Set[str], int]:
    """
    从已有输出文件中读取已完成的 uniprot_id 集合和已写行数

    返回 (done_uids, n_lines_written)
    """
    done = set()
    n_lines = 0

    if OUTPUT_TSV.exists():
        with open(OUTPUT_TSV) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.split("\t", 2)
                if parts:
                    done.add(parts[0])
                n_lines += 1

    return done, n_lines


# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="为RSA数据补充UniProt序列位置")
    parser.add_argument("--flush-every", type=int, default=FLUSH_EVERY,
                        help=f"每处理N个蛋白写一次盘 (默认: {FLUSH_EVERY})")
    args = parser.parse_args()
    flush_every = args.flush_every

    print("=" * 60)
    print("Patch: 为 rsa_per_residue.tsv 添加 uniprot_idx 列")
    print("=" * 60)

    # ---- 5.1 下载SIFTS ----
    if not download_sifts():
        return

    # ---- 5.2 加载SIFTS ----
    lookup = load_sifts()
    if not lookup:
        print("ERROR: SIFTS查找表为空")
        return

    # ---- 5.3 检查输入 ----
    if not INPUT_TSV.exists():
        print(f"ERROR: 输入文件不存在: {INPUT_TSV}")
        return
    total_input_lines = sum(1 for _ in open(INPUT_TSV)) - 1
    print(f"\n输入: {INPUT_TSV} ({total_input_lines} 行)")

    # ---- 5.4 断点恢复 ----
    done_uids, n_done_lines = load_checkpoint()
    print(f"断点: 已完成 {len(done_uids)} 个UniProt ({n_done_lines} 行)")

    # ---- 5.5 读取输入, 按 uniprot_id 分组 ----
    print(f"\n读取输入数据...")
    groups = defaultdict(list)  # {uniprot_id: [(line_idx, row_dict), ...]}
    with open(INPUT_TSV) as f:
        header_line = f.readline().strip()
        input_columns = header_line.split("\t")
        for line_idx, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < len(input_columns):
                print(f'ERROR line_idx {line_idx} not do meet input_columns')
                continue
            row = dict(zip(input_columns, parts))
            uid = row.get("uniprot_id", "")
            groups[uid].append(row)

    total_proteins = len(groups)
    todo_proteins = {uid: rows for uid, rows in groups.items() if uid not in done_uids}
    n_skip = total_proteins - len(todo_proteins)
    print(f"  蛋白总数: {total_proteins}, 跳过(已完成): {n_skip}, 待处理: {len(todo_proteins)}")

    if not todo_proteins:
        print("\n全部已完成!")
        return

    # ---- 5.6 准备输出和日志 ----
    RSA_DIR.mkdir(parents=True, exist_ok=True)

    # 输出文件: 新建 or 追加
    output_header = header_line + "\tuniprot_idx\n"
    if not OUTPUT_TSV.exists():
        with open(OUTPUT_TSV, "w") as f:
            f.write(output_header)

    # 映射日志 (每个残基一行)
    mapping_log_header = "uniprot_id\tpdb_id\tchain\tresi\tuniprot_idx\tstatus\n"
    if not MAPPING_LOG.exists():
        with open(MAPPING_LOG, "w") as f:
            f.write(mapping_log_header)

    # 处理日志 (每个蛋白一行)
    process_log_header = "timestamp\tuniprot_id\tpdb_id\tn_residues\tn_mapped\tn_unmapped\tstatus\n"
    if not PROCESS_LOG.exists():
        with open(PROCESS_LOG, "w") as f:
            f.write(process_log_header)

    # ---- 5.7 处理 ----
    print(f"\n开始映射 (flush_every={flush_every})...\n")
    t0 = time.time()

    output_buf: List[str] = []         # 输出行缓冲
    mapping_buf: List[str] = []        # 映射日志缓冲
    process_buf: List[str] = []        # 处理日志缓冲
    stats = Counter()
    done_count = 0

    sorted_uids = sorted(todo_proteins.keys())

    for uid in sorted_uids:
        rows = todo_proteins[uid]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        n_mapped = 0
        n_unmapped = 0
        pdb_id_used = ""

        for row in rows:
            pdb_id = row.get("pdb_id", "")
            chain = row.get("chain", "")
            resi_pdb = row.get("resi", "")
            pdb_id_used = pdb_id

            # 映射
            uniprot_idx, map_status = map_residue(
                lookup, pdb_id, chain, resi_pdb, uid)

            if map_status == "mapped":
                n_mapped += 1
            else:
                n_unmapped += 1

            # 输出行: 原始列 + uniprot_idx
            original_values = "\t".join(row.get(c, "") for c in input_columns)
            output_buf.append(f"{original_values}\t{uniprot_idx}\n")

            # 映射日志
            mapping_buf.append(
                f"{uid}\t{pdb_id}\t{chain}\t{resi_pdb}\t{uniprot_idx}\t{map_status}\n")

        # 处理日志
        total_res = n_mapped + n_unmapped
        prot_status = "all_mapped" if n_unmapped == 0 else (
            "partial" if n_mapped > 0 else "all_unmapped")
        process_buf.append(
            f"{ts}\t{uid}\t{pdb_id_used}\t{total_res}\t{n_mapped}\t{n_unmapped}\t{prot_status}\n")

        stats[prot_status] += 1
        done_count += 1

        # 定期flush
        if done_count % flush_every == 0 or done_count == len(sorted_uids):
            _flush(output_buf, mapping_buf, process_buf)
            output_buf.clear()
            mapping_buf.clear()
            process_buf.clear()

            elapsed = time.time() - t0
            rate = done_count / elapsed if elapsed > 0 else 0
            print(f"  [{done_count:>6}/{len(sorted_uids)}] "
                  f"all_mapped={stats['all_mapped']} "
                  f"partial={stats['partial']} "
                  f"unmapped={stats['all_unmapped']} "
                  f"({rate:.1f} prot/s)")

    # flush残留
    if output_buf:
        _flush(output_buf, mapping_buf, process_buf)

    elapsed_total = time.time() - t0

    # ---- 5.8 摘要 ----
    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)
    print(f"处理蛋白:    {done_count}")
    print(f"耗时:        {elapsed_total:.1f}s")
    print(f"状态分布:")
    for s, c in stats.most_common():
        print(f"  {s:<20} {c:>6}")

    # 统计输出文件
    if OUTPUT_TSV.exists():
        n_out = sum(1 for _ in open(OUTPUT_TSV)) - 1
        sz = OUTPUT_TSV.stat().st_size / (1024 * 1024)
        print(f"\n输出: {OUTPUT_TSV} ({n_out} 行, {sz:.1f} MB)")

    print(f"\n文件清单:")
    print(f"  输入 (不变): {INPUT_TSV}")
    print(f"  输出 (新增): {OUTPUT_TSV}")
    print(f"  映射日志:    {MAPPING_LOG}")
    print(f"  处理日志:    {PROCESS_LOG}")


def _flush(output_buf, mapping_buf, process_buf):
    """批量写盘"""
    if output_buf:
        with open(OUTPUT_TSV, "a") as f:
            f.writelines(output_buf)
    if mapping_buf:
        with open(MAPPING_LOG, "a") as f:
            f.writelines(mapping_buf)
    if process_buf:
        with open(PROCESS_LOG, "a") as f:
            f.writelines(process_buf)


if __name__ == "__main__":
    main()