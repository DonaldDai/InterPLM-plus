"""
Step 4: 逐残基标注 Kyte-Doolittle 疏水性值
==========================================

读取 step1 下载的完整 UniProt FASTA，对每条序列的每个残基
查表赋予 KD 疏水性分值，输出 per-residue 标注表。

为什么用 step1 的原始 FASTA 而不是 step3 的拆分集:
  不同特征的可提取范围不同 (如RSA依赖PDB结构, 部分蛋白无数据),
  先对全量数据提取所有特征, 后续再统一切分训练/验证集,
  避免先切分后发现某些蛋白缺失某类标注。

KD值说明:
  标准20种氨基酸: Kyte & Doolittle 原始分值 (-4.5 ~ +4.5)
  非标准已知残基 (U/O/B/Z/X): 降级到最接近的标准残基分值, 记入日志
  完全未识别的字符: kd_value = -99 (哨兵值, 下游可过滤), 记入日志

Kyte-Doolittle 标度来源:
  Kyte J, Doolittle RF (1982) "A simple method for displaying the
  hydropathic character of a protein." J Mol Biol 157:105-132.

输入:
  cusdata/01_raw/uniprot_pdb_sequences.fasta

输出:
  cusdata/04_kd/kd_per_residue.tsv       ← 主输出
  cusdata/04_kd/kd_residue_warnings.tsv  ← special/unknown 残基日志

用法:
  python step4_annotate_kd.py
  python step4_annotate_kd.py --flush-every 200
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

try:
    from Bio import SeqIO
except ImportError:
    print("ERROR: biopython 未安装")
    print("安装: pip install biopython")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
INPUT_FASTA = Path("cusdata/01_raw/uniprot_pdb_sequences.fasta")
OUTPUT_DIR = Path("cusdata/04_kd")

OUTPUT_FILE = OUTPUT_DIR / "kd_per_residue.tsv"
WARNING_LOG = OUTPUT_DIR / "kd_residue_warnings.tsv"

# 默认每处理多少条序列flush一次 (命令行 --flush-every 可覆盖)
FLUSH_EVERY = 500

# 未识别残基的哨兵值 (KD标度范围是 -4.5 ~ +4.5, -99 不会与真实值混淆)
KD_UNKNOWN_SENTINEL = -99

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

# 非标准但已知的残基 → 降级到最接近的标准残基
KD_SPECIAL = {
    "U": -100,    # Selenocysteine → 按Cys
    "O": -100,   # Pyrrolysine → 按Lys
    "B": -100,   # Asx (Asp或Asn) → 取共同值
    "Z": -100,   # Glx (Glu或Gln) → 取共同值
    "X": -100,    # Unknown → 中性近似
}


# ============================================================
# 残基警告日志
# ============================================================
class ResidueWarningLogger:
    """
    记录 special 和 unknown 残基

    日志格式:
      timestamp  uniprot_id  position_1based  residue  category  kd_value

    category:
      special  — 非标准但已知 (U/O/B/Z/X), 降级处理
      unknown  — 完全未识别的字符, kd=-99
    """

    HEADER = "timestamp\tuniprot_id\tposition_1based\tresidue\tcategory\tkd_value\n"

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._buf: List[str] = []
        self.n_special = 0
        self.n_unknown = 0

        log_path.parent.mkdir(parents=True, exist_ok=True)
        # append模式: 文件不存在才写header, 已存在直接追加
        if not log_path.exists() or log_path.stat().st_size == 0:
            with open(log_path, "w") as f:
                f.write(self.HEADER)

    def log_special(self, uniprot_id: str, position_0based: int, residue: str, kd: float):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._buf.append(
            f"{ts}\t{uniprot_id}\t{position_0based + 1}\t{residue}\tspecial\t{kd}\n")
        self.n_special += 1

    def log_unknown(self, uniprot_id: str, position_0based: int, residue: str, kd: float):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._buf.append(
            f"{ts}\t{uniprot_id}\t{position_0based + 1}\t{residue}\tunknown\t{kd}\n")
        self.n_unknown += 1

    def flush(self):
        if self._buf:
            with open(self.log_path, "a") as f:
                f.writelines(self._buf)
            self._buf.clear()

    def summary(self) -> str:
        return f"special={self.n_special}, unknown={self.n_unknown}"


# ============================================================
# 读取 FASTA (BioPython SeqIO)
# ============================================================
def read_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    """
    用 BioPython SeqIO 读取FASTA, 返回 [(uniprot_id, sequence), ...]
    从 record.id 中提取 UniProt accession (第二个 | 分隔字段)
    """
    sequences = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        parts = record.id.split("|")
        uid = parts[1] if len(parts) >= 2 else record.id
        sequences.append((uid, str(record.seq)))
    return sequences


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Step 4: KD per-residue annotation")
    parser.add_argument("--flush-every", type=int, default=FLUSH_EVERY,
                        help=f"每处理N条序列flush一次 (默认: {FLUSH_EVERY})")
    args = parser.parse_args()
    flush_every = args.flush_every

    print("=" * 60)
    print("Step 4: Kyte-Doolittle per-residue annotation")
    print(f"  flush-every: {flush_every}")
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

    # 初始化
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = ResidueWarningLogger(WARNING_LOG)

    print(f"\n写入: {OUTPUT_FILE}")
    print(f"日志: {WARNING_LOG}")
    t0 = time.time()
    n_residues = 0
    n_seqs_done = 0

    with open(OUTPUT_FILE, "w") as f:
        f.write("uniprot_id\tresidue\tposition\tkd_value\n")

        for uid, seq in all_sequences:
            for pos, aa in enumerate(seq):
                aa_upper = aa.upper()

                # 查表: 标准20种 → special → unknown
                kd = KD_SCALE.get(aa_upper)

                if kd is not None:
                    pass
                else:
                    kd = KD_SPECIAL.get(aa_upper)
                    if kd is not None:
                        logger.log_special(uid, pos, aa_upper, kd)
                        continue
                    else:
                        kd = KD_UNKNOWN_SENTINEL
                        logger.log_unknown(uid, pos, aa_upper, kd)
                        continue

                f.write(f"{uid}\t{aa_upper}\t{pos}\t{kd}\n")
                n_residues += 1

            n_seqs_done += 1

            # 定期flush
            if n_seqs_done % flush_every == 0:
                f.flush()
                logger.flush()
                elapsed = time.time() - t0
                rate = n_seqs_done / elapsed if elapsed > 0 else 0
                print(f"  [{n_seqs_done:>6}/{len(all_sequences)}] "
                      f"residues={n_residues} "
                      f"warnings={logger.n_special + logger.n_unknown} "
                      f"({rate:.0f} seq/s)")

    # 最终flush
    logger.flush()
    elapsed = time.time() - t0

    # 摘要
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\n完成")
    print(f"  序列数:     {n_seqs_done}")
    print(f"  残基数:     {n_residues}")
    print(f"  异常残基:   {logger.summary()}")
    print(f"  文件大小:   {size_mb:.1f} MB")
    print(f"  耗时:       {elapsed:.1f}s")

    if logger.n_special > 0 or logger.n_unknown > 0:
        log_size = WARNING_LOG.stat().st_size / 1024
        print(f"\n警告日志: {WARNING_LOG} ({log_size:.1f} KB)")
        print(f"  special (降级处理): {logger.n_special}")
        print(f"  unknown (kd={KD_UNKNOWN_SENTINEL}): {logger.n_unknown}")


if __name__ == "__main__":
    main()