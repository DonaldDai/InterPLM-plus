"""
patch_add_plddt.py — 从 AlphaFold CIF 中提取 pLDDT, 追加到 rsa_per_residue.tsv
==================================================================================

AlphaFold CIF 中 pLDDT 存储在 _atom_site.B_iso_or_equiv 字段,
每个残基的所有原子 pLDDT 相同, 取第一个原子即可。

流程:
  1. 读取 rsa_per_residue.tsv, 收集需要处理的 (uniprot_id, position) 对
  2. 逐蛋白解析 AlphaFold CIF, 提取 per-residue pLDDT
  3. 重写 TSV, 在末尾追加 plddt 列

输入:
  cusdata/05_1_alphafold_rsa/rsa_per_residue.tsv
  cusdata/05_1_alphafold_rsa/af_structures/AF-{uid}-F1-model_v6.cif

输出:
  cusdata/05_1_alphafold_rsa/rsa_per_residue.tsv  (原地更新, 追加 plddt 列)
  cusdata/05_1_alphafold_rsa/rsa_per_residue.tsv.bak  (备份)

用法:
  python patch_add_plddt.py
"""

import sys
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple

try:
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
except ImportError:
    print("ERROR: biopython 未安装 → pip install biopython")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
RSA_FILE = Path("cusdata/05_1_alphafold_rsa/rsa_per_residue.tsv")
AF_STRUCT_DIR = Path("cusdata/05_1_alphafold_rsa/af_structures")
LOG_FILE = Path("cusdata/05_1_alphafold_rsa/patch_plddt_log.tsv")


# ============================================================
# 日志
# ============================================================
class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write("timestamp\tstep\tstatus\tdetail\n")

    def log(self, step: str, status: str, detail: str = ""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts}\t{step}\t{status}\t{detail}\n"
        with open(self.log_path, "a") as f:
            f.write(line)
        print(f"  [{ts}] {step}: {status}" + (f" | {detail}" if detail else ""))


# ============================================================
# 从 AlphaFold CIF 提取 per-residue pLDDT
# ============================================================
def extract_plddt(cif_path: Path) -> Optional[Dict[int, float]]:
    """
    返回 {position_0based: plddt} 或 None

    AlphaFold CIF 中:
      _atom_site.label_seq_id  = 残基编号 (1-based)
      _atom_site.B_iso_or_equiv = pLDDT 分数
    同一残基的所有原子 pLDDT 相同, 取首次出现即可。
    """
    try:
        mmcif = MMCIF2Dict(str(cif_path))
    except Exception:
        return None

    seq_ids = mmcif.get("_atom_site.label_seq_id", [])
    b_factors = mmcif.get("_atom_site.B_iso_or_equiv", [])
    group_pdb = mmcif.get("_atom_site.group_PDB", [])

    if not seq_ids or not b_factors:
        return None

    result: Dict[int, float] = {}
    for i in range(len(seq_ids)):
        if group_pdb and group_pdb[i] != "ATOM":
            continue
        try:
            pos = int(seq_ids[i]) - 1  # 0-based
            if pos not in result:
                result[pos] = float(b_factors[i])
        except (ValueError, IndexError):
            continue

    return result if result else None


# ============================================================
# 主流程
# ============================================================
def main():
    logger = Logger(LOG_FILE)
    logger.log("main", "start", f"rsa_file={RSA_FILE}")

    if not RSA_FILE.exists():
        logger.log("main", "error", f"文件不存在: {RSA_FILE}")
        return

    # 1. 检查是否已经有 plddt 列
    with open(RSA_FILE) as f:
        header = f.readline().strip()
    columns = header.split("\t")

    if "plddt" in columns:
        logger.log("main", "skip", "plddt 列已存在, 无需处理")
        print("plddt 列已存在, 无需处理")
        return

    # 2. 收集所有需要的 uid
    uids_needed = set()
    with open(RSA_FILE) as f:
        f.readline()
        for line in f:
            uid = line.split("\t", 1)[0]
            if uid:
                uids_needed.add(uid)
    logger.log("load", "uids", f"n={len(uids_needed)}")

    # 3. 逐蛋白提取 pLDDT
    print(f"\n提取 pLDDT ({len(uids_needed)} proteins)...")
    plddt_data: Dict[str, Dict[int, float]] = {}  # uid → {pos: plddt}
    n_ok = 0
    n_fail = 0

    for uid in sorted(uids_needed):
        cif_path = AF_STRUCT_DIR / f"AF-{uid}-F1-model_v6.cif"
        if not cif_path.exists():
            logger.log("plddt", "no_cif", f"uid={uid}")
            n_fail += 1
            continue

        plddt = extract_plddt(cif_path)
        if plddt is None:
            logger.log("plddt", "parse_fail", f"uid={uid}")
            n_fail += 1
            continue

        plddt_data[uid] = plddt
        n_ok += 1

        if n_ok % 1000 == 0:
            print(f"  提取: {n_ok}/{len(uids_needed)}")

    logger.log("plddt", "done", f"ok={n_ok}, fail={n_fail}")

    # 4. 备份原文件
    bak_path = RSA_FILE.with_suffix(".tsv.bak")
    shutil.copy2(RSA_FILE, bak_path)
    logger.log("backup", "saved", f"file={bak_path}")

    # 5. 重写 TSV, 追加 plddt 列
    print("\n重写 TSV...")
    n_matched = 0
    n_missing = 0

    with open(bak_path) as fin, open(RSA_FILE, "w") as fout:
        # 写新 header
        old_header = fin.readline().strip()
        fout.write(f"{old_header}\tplddt\n")

        for line in fin:
            stripped = line.strip()
            parts = stripped.split("\t")
            uid = parts[0]
            pos_str = parts[2] if len(parts) > 2 else ""

            plddt_val = ""
            try:
                pos = int(pos_str)
                uid_plddt = plddt_data.get(uid)
                if uid_plddt is not None and pos in uid_plddt:
                    plddt_val = f"{uid_plddt[pos]:.2f}"
                    n_matched += 1
                else:
                    n_missing += 1
            except ValueError:
                n_missing += 1

            fout.write(f"{stripped}\t{plddt_val}\n")

    logger.log("rewrite", "done",
               f"matched={n_matched}, missing={n_missing}")

    # 摘要
    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  蛋白数:       {len(uids_needed)}")
    print(f"  pLDDT提取成功: {n_ok}")
    print(f"  pLDDT提取失败: {n_fail}")
    print(f"  残基匹配:     {n_matched}")
    print(f"  残基缺失:     {n_missing}")
    print(f"  备份:         {bak_path}")
    print(f"  日志:         {LOG_FILE}")


if __name__ == "__main__":
    main()
