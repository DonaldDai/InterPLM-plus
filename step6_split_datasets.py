"""
Step 6: Refer集 / 验证集划分
=============================

基于 step4 的KD标注可用性和 CD-HIT 聚类结果，
将数据划分为 refer 集和验证集。

流程:
  1. 从原始FASTA中过滤掉不在step4结果中的 uniprot_id
  2. 用 CD-HIT (90% identity, word size 5) 聚类过滤后的FASTA
  3. 取每个cluster的代表蛋白, 按 85:15 比例拆分 refer/validation
  4. 根据拆分结果, 从step4的KD标注中筛选出两个子集分别保存

输入:
  cusdata/01_raw/uniprot_pdb_sequences.fasta
  cusdata/04_kd/kd_per_residue.tsv

输出:
  cusdata/06_splits/
  ├── filtered.fasta                ← 过滤后的FASTA (仅含step4中有KD标注的蛋白)
  ├── cdhit_id90                    ← CD-HIT代表序列
  ├── cdhit_id90.clstr              ← CD-HIT聚类结果
  ├── refer_ids.txt                 ← refer集 uniprot_id 列表
  ├── val_ids.txt                   ← 验证集 uniprot_id 列表
  ├── kd_refer.tsv                  ← refer集 KD标注
  ├── kd_val.tsv                    ← 验证集 KD标注
  └── split_log.tsv                 ← 全流程日志

用法:
  python step6_split_datasets.py
  python step6_split_datasets.py --val-ratio 0.15 --seed 42
  python step6_split_datasets.py --whitelist-file my_ids.txt
"""

import subprocess
import sys
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    from Bio import SeqIO
except ImportError:
    print("ERROR: biopython 未安装")
    print("安装: pip install biopython")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
RAW_FASTA = Path("cusdata/01_raw/uniprot_pdb_sequences.fasta")
KD_FILE = Path("cusdata/04_kd/kd_per_residue.tsv")
OUTPUT_DIR = Path("cusdata/06_splits")

# CD-HIT 参数
CDHIT_IDENTITY = 0.90
CDHIT_WORD_SIZE = 5
CDHIT_THREADS = 8
CDHIT_MEMORY = 16000

# 拆分比例
VAL_RATIO = 0.15
RANDOM_SEED = 42

# 白名单: 永远放入验证集的 UniProt ID
# 默认: 人类 TLR1-10 (Toll-like receptors)
HUMAN_TLR_IDS = {
    "Q15399",   # TLR1
    "O60603",   # TLR2
    "O15455",   # TLR3
    "O00206",   # TLR4
    "O60602",   # TLR5
    "Q9Y2C9",   # TLR6
    "Q9NYK1",   # TLR7
    "Q9NR97",   # TLR8
    "Q2EEY0",   # TLR9
    "Q9BXR5",   # TLR10
}


# ============================================================
# 日志
# ============================================================
class StepLogger:
    """
    流程日志: 每一步操作记录一行

    格式: timestamp  step  status  detail
    append模式, 文件不存在才写header
    """

    HEADER = "timestamp\tstep\tstatus\tdetail\n"

    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists() or log_path.stat().st_size == 0:
            with open(log_path, "w") as f:
                f.write(self.HEADER)

    def log(self, step: str, status: str, detail: str = ""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts}\t{step}\t{status}\t{detail}\n"
        with open(self.log_path, "a") as f:
            f.write(line)
        # 同时打印到终端
        print(f"  [{ts}] {step}: {status}" + (f" | {detail}" if detail else ""))


# ============================================================
# Step 6.1: 过滤FASTA — 只保留step4中有KD标注的蛋白
# ============================================================
def filter_fasta_by_kd(
    raw_fasta: Path,
    kd_file: Path,
    output_fasta: Path,
    logger: StepLogger,
) -> Set[str]:
    """
    返回过滤后保留的 uniprot_id 集合
    """
    logger.log("6.1_filter", "start", f"raw_fasta={raw_fasta}, kd_file={kd_file}")

    # 从KD文件提取有标注的 uniprot_id
    kd_uids: Set[str] = set()
    with open(kd_file) as f:
        f.readline()  # skip header
        for line in f:
            uid = line.split("\t", 1)[0]
            if uid:
                kd_uids.add(uid)
    logger.log("6.1_filter", "kd_ids_loaded", f"n={len(kd_uids)}")

    # 遍历FASTA, 只保留在kd_uids中的序列
    kept_uids: Set[str] = set()
    dropped_uids: List[str] = []
    n_total = 0

    with open(output_fasta, "w") as fout:
        for record in SeqIO.parse(str(raw_fasta), "fasta"):
            parts = record.id.split("|")
            uid = parts[1] if len(parts) >= 2 else record.id
            n_total += 1

            if uid in kd_uids:
                SeqIO.write(record, fout, "fasta")
                kept_uids.add(uid)
            else:
                dropped_uids.append(uid)

    # log被过滤掉的
    for uid in dropped_uids:
        logger.log("6.1_filter", "dropped", f"uniprot_id={uid}")

    logger.log("6.1_filter", "done",
               f"total={n_total}, kept={len(kept_uids)}, dropped={len(dropped_uids)}")
    return kept_uids


# ============================================================
# Step 6.2: CD-HIT 聚类
# ============================================================
def run_cdhit(
    input_fasta: Path,
    output_prefix: Path,
    logger: StepLogger,
) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    """
    运行CD-HIT, 解析聚类结果

    返回:
      clusters:       {cluster_id: [seq_id, ...]}
      representatives: {cluster_id: representative_seq_id}
    """
    logger.log("6.2_cdhit", "start",
               f"identity={CDHIT_IDENTITY}, word_size={CDHIT_WORD_SIZE}")

    cmd = [
        "cd-hit",
        "-i", str(input_fasta),
        "-o", str(output_prefix),
        "-c", str(CDHIT_IDENTITY),
        "-n", str(CDHIT_WORD_SIZE),
        "-M", str(CDHIT_MEMORY),
        "-T", str(CDHIT_THREADS),
        "-d", "0",
        "-g", "1",
    ]

    logger.log("6.2_cdhit", "cmd", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.log("6.2_cdhit", "error", result.stderr[:500])
        print(f"ERROR: CD-HIT failed\n{result.stderr}")
        sys.exit(1)

    # 解析 .clstr
    clstr_file = Path(f"{output_prefix}.clstr")
    clusters = defaultdict(list)
    representatives = {}
    current_cluster = None

    with open(clstr_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[-1])
            elif line and current_cluster is not None:
                parts = line.split(">")
                if len(parts) >= 2:
                    seq_id = parts[1].split("...")[0].strip()
                    clusters[current_cluster].append(seq_id)
                    if line.endswith("*"):
                        representatives[current_cluster] = seq_id

    clusters = dict(clusters)
    n_clusters = len(clusters)
    n_seqs = sum(len(v) for v in clusters.values())
    sizes = [len(v) for v in clusters.values()]

    logger.log("6.2_cdhit", "done",
               f"clusters={n_clusters}, sequences={n_seqs}, "
               f"max_size={max(sizes)}, min_size={min(sizes)}, "
               f"median_size={sorted(sizes)[len(sizes)//2]}, "
               f"singletons={sum(1 for s in sizes if s == 1)}")

    return clusters, representatives


# ============================================================
# Step 6.3: 按cluster代表蛋白拆分 refer / validation
# ============================================================
def split_by_clusters(
    clusters: Dict[int, List[str]],
    representatives: Dict[int, str],
    val_ratio: float,
    seed: int,
    whitelist: Set[str],
    logger: StepLogger,
) -> Tuple[Set[str], Set[str]]:
    """
    按cluster级别拆分:
    - 白名单蛋白所在的cluster强制进入验证集 (整个cluster, 保持无泄漏)
    - 剩余cluster中随机选 val_ratio 比例进入验证集
    - 同一cluster的所有序列只出现在一个集中

    返回 (refer_uids, val_uids) — 都是全量序列ID (非仅代表)
    """
    logger.log("6.3_split", "start",
               f"val_ratio={val_ratio}, seed={seed}, "
               f"n_clusters={len(clusters)}, whitelist_size={len(whitelist)}")

    # 提取 uniprot_id (从 "sp|P12345|PROT_HUMAN" 格式)
    def extract_uid(seq_id: str) -> str:
        parts = seq_id.split("|")
        return parts[1] if len(parts) >= 2 else seq_id

    # ---- 第一步: 找出包含白名单蛋白的clusters, 强制放入验证集 ----
    forced_val_clusters: Set[int] = set()
    whitelist_found: Set[str] = set()
    whitelist_missing: Set[str] = set(whitelist)

    for cid, members in clusters.items():
        member_uids = {extract_uid(sid) for sid in members}
        overlap = member_uids & whitelist
        if overlap:
            forced_val_clusters.add(cid)
            whitelist_found.update(overlap)
            whitelist_missing -= overlap

    for uid in sorted(whitelist_found):
        logger.log("6.3_split", "whitelist_forced",
                   f"uniprot_id={uid} → validation (cluster forced)")

    for uid in sorted(whitelist_missing):
        logger.log("6.3_split", "whitelist_missing",
                   f"uniprot_id={uid} not found in any cluster (可能被step4过滤)")

    logger.log("6.3_split", "whitelist_summary",
               f"found={len(whitelist_found)}, missing={len(whitelist_missing)}, "
               f"forced_clusters={len(forced_val_clusters)}")

    # ---- 第二步: 剩余clusters随机拆分 ----
    remaining_cluster_ids = sorted(set(clusters.keys()) - forced_val_clusters)
    rng = random.Random(seed)
    rng.shuffle(remaining_cluster_ids)

    # 目标验证集cluster数 = 总数 * val_ratio - 已强制的数量
    target_val_total = max(1, round(len(clusters) * val_ratio))
    n_random_val = max(0, target_val_total - len(forced_val_clusters))

    random_val_clusters = set(remaining_cluster_ids[:n_random_val])
    refer_cluster_ids = set(remaining_cluster_ids[n_random_val:])
    val_cluster_ids = forced_val_clusters | random_val_clusters

    # ---- 第三步: 收集序列ID ----
    refer_uids: Set[str] = set()
    val_uids: Set[str] = set()

    for cid in refer_cluster_ids:
        for sid in clusters[cid]:
            refer_uids.add(extract_uid(sid))

    for cid in val_cluster_ids:
        for sid in clusters[cid]:
            val_uids.add(extract_uid(sid))

    logger.log("6.3_split", "done",
               f"refer_clusters={len(refer_cluster_ids)}, "
               f"val_clusters={len(val_cluster_ids)} "
               f"(forced={len(forced_val_clusters)}, random={len(random_val_clusters)}), "
               f"refer_proteins={len(refer_uids)}, "
               f"val_proteins={len(val_uids)}")

    # 确认白名单蛋白全在验证集
    whitelist_in_val = whitelist_found & val_uids
    whitelist_in_refer = whitelist_found & refer_uids
    if whitelist_in_refer:
        logger.log("6.3_split", "error",
                   f"whitelist蛋白泄漏到refer集: {whitelist_in_refer} (不应发生)")
    else:
        logger.log("6.3_split", "verified",
                   f"whitelist全部在验证集 ({len(whitelist_in_val)}/{len(whitelist_found)})")

    # 验证无交集
    overlap = refer_uids & val_uids
    if overlap:
        logger.log("6.3_split", "warning",
                   f"overlap={len(overlap)}_ids (不应发生)")
    else:
        logger.log("6.3_split", "verified", "refer ∩ val = ∅ (无泄漏)")

    return refer_uids, val_uids


# ============================================================
# Step 6.4: 根据拆分结果筛选KD标注
# ============================================================
def split_kd_file(
    kd_file: Path,
    refer_uids: Set[str],
    val_uids: Set[str],
    output_dir: Path,
    logger: StepLogger,
):
    """
    读取step4的KD标注, 按refer/val集合分别写入两个文件
    """
    logger.log("6.4_split_kd", "start", f"kd_file={kd_file}")

    refer_out = output_dir / "kd_refer.tsv"
    val_out = output_dir / "kd_val.tsv"

    n_refer = 0
    n_val = 0
    n_skipped = 0
    refer_proteins: Set[str] = set()
    val_proteins: Set[str] = set()

    with open(kd_file) as fin, \
         open(refer_out, "w") as f_refer, \
         open(val_out, "w") as f_val:

        header = fin.readline()
        f_refer.write(header)
        f_val.write(header)

        for line in fin:
            uid = line.split("\t", 1)[0]

            if uid in refer_uids:
                f_refer.write(line)
                n_refer += 1
                refer_proteins.add(uid)
            elif uid in val_uids:
                f_val.write(line)
                n_val += 1
                val_proteins.add(uid)
            else:
                n_skipped += 1

    logger.log("6.4_split_kd", "refer",
               f"proteins={len(refer_proteins)}, residues={n_refer}, file={refer_out}")
    logger.log("6.4_split_kd", "val",
               f"proteins={len(val_proteins)}, residues={n_val}, file={val_out}")

    if n_skipped > 0:
        logger.log("6.4_split_kd", "skipped",
                   f"residues={n_skipped} (不在refer也不在val, 不应发生)")

    logger.log("6.4_split_kd", "done",
               f"refer={len(refer_proteins)}_proteins/{n_refer}_residues, "
               f"val={len(val_proteins)}_proteins/{n_val}_residues")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Step 6: Refer/Validation split")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO,
                        help=f"验证集比例 (默认: {VAL_RATIO})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"随机种子 (默认: {RANDOM_SEED})")
    parser.add_argument("--whitelist-file", type=str, default=None,
                        help="白名单文件 (每行一个UniProt ID, 强制放入验证集). "
                             "不指定则使用内置的人类TLR1-10列表")
    args = parser.parse_args()

    # 加载白名单
    if args.whitelist_file:
        wl_path = Path(args.whitelist_file)
        if not wl_path.exists():
            print(f"ERROR: 白名单文件不存在: {wl_path}")
            return
        whitelist = set()
        with open(wl_path) as f:
            for line in f:
                uid = line.strip()
                if uid and not uid.startswith("#"):
                    whitelist.add(uid)
        print(f"  白名单: {wl_path} ({len(whitelist)} IDs)")
    else:
        whitelist = set(HUMAN_TLR_IDS)
        print(f"  白名单: 内置人类TLR1-10 ({len(whitelist)} IDs)")

    print("=" * 60)
    print("Step 6: Refer集 / 验证集划分")
    print(f"  val_ratio={args.val_ratio}, seed={args.seed}")
    print(f"  whitelist={len(whitelist)} IDs → 强制验证集")
    print("=" * 60)

    # 检查输入
    for fp, name in [(RAW_FASTA, "原始FASTA"), (KD_FILE, "KD标注")]:
        if not fp.exists():
            print(f"ERROR: {name}不存在: {fp}")
            return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(OUTPUT_DIR / "split_log.tsv")

    logger.log("main", "start",
               f"raw_fasta={RAW_FASTA}, kd_file={KD_FILE}, "
               f"val_ratio={args.val_ratio}, seed={args.seed}, "
               f"whitelist_size={len(whitelist)}")
    t0 = time.time()

    # ---- 6.1 过滤FASTA ----
    print("\n--- 6.1 过滤FASTA ---")
    filtered_fasta = OUTPUT_DIR / "filtered.fasta"
    kept_uids = filter_fasta_by_kd(RAW_FASTA, KD_FILE, filtered_fasta, logger)

    # ---- 6.2 CD-HIT聚类 ----
    print("\n--- 6.2 CD-HIT聚类 ---")
    cdhit_prefix = OUTPUT_DIR / "cdhit_id90"
    clusters, representatives = run_cdhit(filtered_fasta, cdhit_prefix, logger)

    # ---- 6.3 拆分 ----
    print("\n--- 6.3 拆分refer/val ---")
    refer_uids, val_uids = split_by_clusters(
        clusters, representatives, args.val_ratio, args.seed, whitelist, logger)

    # 保存ID列表
    refer_ids_file = OUTPUT_DIR / "refer_ids.txt"
    val_ids_file = OUTPUT_DIR / "val_ids.txt"
    with open(refer_ids_file, "w") as f:
        for uid in sorted(refer_uids):
            f.write(f"{uid}\n")
    with open(val_ids_file, "w") as f:
        for uid in sorted(val_uids):
            f.write(f"{uid}\n")
    logger.log("6.3_split", "saved",
               f"refer_ids={refer_ids_file}, val_ids={val_ids_file}")

    # ---- 6.4 筛选KD标注 ----
    print("\n--- 6.4 筛选KD标注 ---")
    split_kd_file(KD_FILE, refer_uids, val_uids, OUTPUT_DIR, logger)

    # ---- 摘要 ----
    elapsed = time.time() - t0
    logger.log("main", "done", f"elapsed={elapsed:.1f}s")

    # 统计白名单蛋白最终在哪
    wl_in_val = whitelist & val_uids
    wl_in_refer = whitelist & refer_uids
    wl_missing = whitelist - val_uids - refer_uids

    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  过滤后蛋白:  {len(kept_uids)}")
    print(f"  Clusters:    {len(clusters)}")
    print(f"  Refer集:     {len(refer_uids)} proteins")
    print(f"  验证集:      {len(val_uids)} proteins")
    print(f"  白名单:      {len(wl_in_val)} in val, "
          f"{len(wl_in_refer)} in refer (应为0), "
          f"{len(wl_missing)} missing")
    print(f"  耗时:        {elapsed:.1f}s")
    print()
    print("输出文件:")
    for f in [filtered_fasta, cdhit_prefix, refer_ids_file, val_ids_file,
              OUTPUT_DIR / "kd_refer.tsv", OUTPUT_DIR / "kd_val.tsv",
              OUTPUT_DIR / "split_log.tsv"]:
        if f.exists():
            sz = f.stat().st_size / (1024 * 1024)
            print(f"  {f} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
