"""
Step 4: 逐残基标注 Kyte-Doolittle 疏水性值
==========================================

读取 step1 下载的完整 UniProt FASTA，对每条序列的每个残基
查表赋予 KD 疏水性分值，输出 per-residue 标注表。

为什么用 step1 的原始 FASTA 而不是 step3 的拆分集:
  不同特征的可提取范围不同 (如RSA依赖PDB结构, 部分蛋白无数据),
  先对全量数据提取所有特征, 后续再统一切分训练/验证集,
  避免先切分后发现某些蛋白缺失某类标注。

Kyte-Doolittle 标度来源:
  Kyte J, Doolittle RF (1982) "A simple method for displaying the
  hydropathic character of a protein." J Mol Biol 157:105-132.

输入:
  cusdata/01_raw/uniprot_pdb_sequences.fasta

输出:
  cusdata/04_kd/kd_per_residue.tsv
  列: uniprot_id  residue  position  kd_value
  (position 为 0-based UniProt 序列下标)

用法:
  python step4_annotate_kd.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# 配置
# ============================================================
INPUT_FASTA = Path("cusdata/01_raw/uniprot_pdb_sequences.fasta")
OUTPUT_DIR = Path("cusdata/04_kd")

OUTPUT_FILE = OUTPUT_DIR / "kd_per_residue.tsv"

# ============================================================
# Kyte-Doolittle 疏水性标度
# Kyte & Doolittle, J Mol Biol 157:105-132 (1982), Table II
# ============================================================
KD_SCALE = {
    "I":  4.5,   # Isoleucine
    "V":  4.2,   # Valine
    "L":  3.8,   # Leucine
    "F":  2.8,   # Phenylalanine
    "C":  2.5,   # Cysteine
    "M":  1.9,   # Methionine
    "A":  1.8,   # Alanine
    "G": -0.4,   # Glycine
    "T": -0.7,   # Threonine
    "S": -0.8,   # Serine
    "W": -0.9,   # Tryptophan
    "Y": -1.3,   # Tyrosine
    "P": -1.6,   # Proline
    "H": -3.2,   # Histidine
    "E": -3.5,   # Glutamic acid
    "Q": -3.5,   # Glutamine
    "D": -3.5,   # Aspartic acid
    "N": -3.5,   # Asparagine
    "K": -3.9,   # Lysine
    "R": -4.5,   # Arginine
}

# 非标准残基降级处理
KD_SPECIAL = {
    "U": 2.5,    # Selenocysteine → 按Cys
    "O": -3.5,   # Pyrrolysine → 按Lys近似
    "B": -3.5,   # Asx (Asp或Asn) → 取共同值
    "Z": -3.5,   # Glx (Glu或Gln) → 取共同值
    "X": 0.0,    # Unknown → 中性
}


# ============================================================
# 读取 FASTA
# ============================================================
def read_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    """
    读取FASTA, 返回 [(uniprot_id, sequence), ...]

    header格式: >sp|P12345|PROT_HUMAN ...
    提取 accession = P12345
    """
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))
                header = line[1:].split()[0]
                parts = header.split("|")
                current_id = parts[1] if len(parts) >= 2 else header
                current_seq = []
            elif line:
                current_seq.append(line)

        if current_id is not None:
            sequences.append((current_id, "".join(current_seq)))

    return sequences


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("Step 4: Kyte-Doolittle per-residue annotation")
    print("=" * 60)

    if not INPUT_FASTA.exists():
        print(f"ERROR: 输入文件不存在: {INPUT_FASTA}")
        print("请先运行 step1_download_all_data.py")
        return

    print(f"  输入: {INPUT_FASTA}")

    # 读取所有序列
    all_sequences = read_fasta(INPUT_FASTA)
    print(f"  序列数: {len(all_sequences)}")
    total_residues = sum(len(seq) for _, seq in all_sequences)
    print(f"  残基数: {total_residues}")

    # 标注并写入
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n写入: {OUTPUT_FILE}")
    t0 = time.time()
    n_residues = 0
    n_unknown = 0

    with open(OUTPUT_FILE, "w") as f:
        f.write("uniprot_id\tresidue\tposition\tkd_value\n")

        for uid, seq in all_sequences:
            for pos, aa in enumerate(seq):
                aa_upper = aa.upper()

                kd = KD_SCALE.get(aa_upper)
                if kd is None:
                    kd = KD_SPECIAL.get(aa_upper)
                if kd is None:
                    kd = 0.0
                    n_unknown += 1

                f.write(f"{uid}\t{aa_upper}\t{pos}\t{kd}\n")
                n_residues += 1

    elapsed = time.time() - t0

    # 摘要
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n完成")
    print(f"  残基数:     {n_residues}")
    print(f"  未识别残基: {n_unknown}")
    print(f"  文件大小:   {size_mb:.1f} MB")
    print(f"  耗时:       {elapsed:.1f}s")


if __name__ == "__main__":
    main()