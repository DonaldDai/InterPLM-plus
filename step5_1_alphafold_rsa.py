"""
Step 5.1: 从 AlphaFold DB 获取结构, 计算 per-residue RSA
==========================================================

与 step5 不同, 本脚本直接从 AlphaFold DB 按 UniProt ID 下载预测结构,
无需 PDB→UniProt 映射, 一个 UniProt ID 对应一个结构, 残基位置天然对齐。

AlphaFold DB:
  https://alphafold.ebi.ac.uk/files/AF-{UNIPROT_ID}-F1-model_v6.cif
  - 单链 (chain A), 全长序列, 残基编号 = UniProt 序列位置 (1-based)
  - CIF 格式, 无需解压

流程:
  1. 读取 step1 FASTA, 提取所有 UniProt ID
  2. 多线程下载 AlphaFold CIF (临时目录 → 正式目录, 保证完整性)
  3. 逐蛋白计算 FreeSASA → RSA
  4. 输出 per-residue TSV (格式同 kd_per_residue.tsv)

输入:
  cusdata/01_raw/uniprot_pdb_sequences.fasta

输出:
  cusdata/05_1_alphafold_rsa/
  ├── af_structures/              ← AlphaFold CIF 文件
  ├── rsa_per_residue.tsv         ← 主输出
  ├── process_log.tsv             ← 处理日志 (追加)
  └── error_log.tsv               ← 异常日志 (追加)

依赖:
  pip install freesasa biopython requests

用法:
  python step5_1_alphafold_rsa.py
  python step5_1_alphafold_rsa.py --workers 4 --download-delay 0.1
"""

import gzip
import json
import os
import sys
import time
import shutil
import traceback
import argparse
import requests
from datetime import datetime
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Set

try:
    import freesasa
except ImportError:
    print("ERROR: freesasa 未安装 → pip install freesasa")
    sys.exit(1)

try:
    from Bio.PDB import MMCIFParser, PDBIO, Select
    from Bio import SeqIO
except ImportError:
    print("ERROR: biopython 未安装 → pip install biopython")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
INPUT_FASTA = Path("cusdata/01_raw/uniprot_pdb_sequences.fasta")
OUTPUT_DIR = Path("cusdata/05_1_alphafold_rsa")
AF_STRUCT_DIR = OUTPUT_DIR / "af_structures"
AF_TMP_DIR = OUTPUT_DIR / "_tmp_download"

ALPHAFOLD_URL_TEMPLATE = (
    "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.cif"
)

# FreeSASA 参数
FREESASA_ALGORITHM = freesasa.LeeRichards
FREESASA_PROBE_RADIUS = 1.4
FREESASA_N_SLICES = 20

# 默认超参
DEFAULT_WORKERS = 4
DEFAULT_DOWNLOAD_DELAY = 0.1  # 秒, 每个下载之间的间隔

# MaxASA: Tien et al. 2013
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
    "CYS": 167.0, "GLN": 225.0, "GLU": 223.0, "GLY": 104.0,
    "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
    "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
    "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
    "MSE": 224.0, "SEC": 167.0, "UNK": 200.0,
}

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U",
}

SKIP_RESIDUES = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "DMS",
    "ACE", "NME", "NH2", "ACT", "FMT", "BME", "MPD", "TRS", "EPE",
    "CL", "NA", "MG", "ZN", "CA", "FE", "MN", "CO", "NI", "CU", "K",
}

STANDARD_AA_SET = set(MAX_ASA.keys())


# ============================================================
# PDB导出过滤器 (同step5)
# ============================================================
class ProteinSelect(Select):
    def accept_residue(self, residue):
        resname = residue.resname.strip().upper()
        hetflag = residue.id[0]
        if hetflag == "W":
            return False
        if resname in STANDARD_AA_SET:
            return True
        if resname in SKIP_RESIDUES:
            return False
        if hetflag != " ":
            return False
        return True

    def accept_atom(self, atom):
        elem = atom.element.strip().upper()
        return elem not in ("H", "D")


# ============================================================
# 日志
# ============================================================
class Logger:
    """追加模式日志, 文件不存在才写header"""

    def __init__(self, log_path: Path, header: str):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists() or log_path.stat().st_size == 0:
            with open(log_path, "w") as f:
                f.write(header)
        self._buf: List[str] = []

    def write(self, line: str):
        self._buf.append(line)

    def flush(self):
        if self._buf:
            with open(self.log_path, "a") as f:
                f.writelines(self._buf)
            self._buf.clear()

    def print_and_write(self, step: str, status: str, detail: str = ""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts}\t{step}\t{status}\t{detail}\n"
        self._buf.append(line)
        print(f"  [{ts}] {step}: {status}" + (f" | {detail}" if detail else ""))


# ============================================================
# 1. 读取 FASTA 提取 UniProt ID
# ============================================================
def load_uniprot_data(fasta_path: Path) -> Tuple[List[str], Dict[str, str]]:
    """返回 (uid列表, {uid: sequence_str})"""
    ids = []
    sequences = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        parts = record.id.split("|")
        uid = parts[1] if len(parts) >= 2 else record.id
        ids.append(uid)
        sequences[uid] = str(record.seq)
    return ids, sequences


# ============================================================
# 2. 下载 AlphaFold 结构
# ============================================================
def download_one_af(uid: str, struct_dir: Path, tmp_dir: Path,
                    delay: float) -> Tuple[str, str, str]:
    """
    下载单个 AlphaFold CIF 文件

    流程: 下载到 tmp_dir → 验证 → 移入 struct_dir
    返回 (uid, status, detail)
      status: "exists" / "downloaded" / "not_found" / "rate_limited" / "error"
    """
    final_path = struct_dir / f"AF-{uid}-F1-model_v6.cif"

    # 已存在则跳过
    if final_path.exists() and final_path.stat().st_size > 100:
        return uid, "exists", str(final_path)

    url = ALPHAFOLD_URL_TEMPLATE.format(uid=uid)
    tmp_path = tmp_dir / f"AF-{uid}-F1-model_v6.cif"

    time.sleep(delay)

    try:
        resp = requests.get(url, timeout=60)

        # 检查状态码
        if resp.status_code == 404:
            return uid, "not_found", f"HTTP 404: {url}"
        if resp.status_code == 429:
            return uid, "rate_limited", f"HTTP 429: {url}"
        if resp.status_code != 200:
            return uid, "error", f"HTTP {resp.status_code}: {url}"

        content = resp.text

        # 检查是否被风控 (返回HTML而非CIF)
        if content.strip().startswith("<!") or "<html" in content[:200].lower():
            return uid, "rate_limited", "response is HTML (rate limited or blocked)"

        # 检查是否是有效CIF (包含 _atom_site)
        if "_atom_site" not in content[:5000] and "loop_" not in content[:5000]:
            return uid, "error", "response does not look like CIF"

        # 写入临时文件
        with open(tmp_path, "w") as f:
            f.write(content)

        # 移入正式目录
        shutil.move(str(tmp_path), str(final_path))
        return uid, "downloaded", str(final_path)

    except requests.Timeout:
        return uid, "error", f"timeout: {url}"
    except Exception as e:
        return uid, "error", f"{type(e).__name__}: {e}"
    finally:
        # 清理残留临时文件
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def download_all(uids: List[str], struct_dir: Path, tmp_dir: Path,
                 workers: int, delay: float,
                 proc_log: Logger, err_log: Logger) -> Set[str]:
    """
    多线程下载, 返回成功下载 (含已存在) 的 uid 集合
    """
    struct_dir.mkdir(parents=True, exist_ok=True)

    # 清空临时目录
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    proc_log.print_and_write("download", "start",
                             f"total={len(uids)}, workers={workers}, delay={delay}s")

    success_uids: Set[str] = set()
    stats = Counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one_af, uid, struct_dir, tmp_dir, delay): uid
            for uid in uids
        }

        done_count = 0
        for future in as_completed(futures):
            uid, status, detail = future.result()
            stats[status] += 1
            done_count += 1

            if status in ("exists", "downloaded"):
                success_uids.add(uid)
            else:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                err_log.write(f"{ts}\t{uid}\tdownload_{status}\t{detail}\n")

            if done_count % 500 == 0 or done_count == len(uids):
                proc_log.print_and_write("download", "progress",
                    f"done={done_count}/{len(uids)}, "
                    + ", ".join(f"{k}={v}" for k, v in stats.most_common()))

    err_log.flush()

    proc_log.print_and_write("download", "done",
        f"success={len(success_uids)}, " +
        ", ".join(f"{k}={v}" for k, v in stats.most_common()))

    # 清理临时目录
    try:
        shutil.rmtree(tmp_dir)
    except OSError:
        pass

    return success_uids


# ============================================================
# 3. FreeSASA 计算 (简化版, 适用于 AlphaFold CIF)
# ============================================================
def _freesasa_calc_quiet(pdb_path: str, params) -> Tuple:
    """FreeSASA 计算 + 抑制C层stderr"""
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        struct = freesasa.Structure(pdb_path)
        result = freesasa.calc(struct, params)
        return struct, result
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)


def compute_rsa_for_one(uid: str, cif_path: Path, tmp_dir: Path
                        ) -> Tuple[List[dict], str, str]:
    """
    对一个 AlphaFold CIF 计算 per-residue RSA

    AlphaFold 结构特点:
      - 单链 (chain A), 无需链ID重映射
      - label_seq_id = UniProt 序列位置 (1-based)
      - 全是标准氨基酸, 无水/配体

    返回 (residue_data, cif_sequence, error_message)
      cif_sequence: 从结构中提取的氨基酸序列 (单字母), 用于和FASTA比对
      error_message 为空 = 成功
    """
    params = freesasa.Parameters()
    params.setAlgorithm(FREESASA_ALGORITHM)
    params.setProbeRadius(FREESASA_PROBE_RADIUS)
    if FREESASA_ALGORITHM == freesasa.LeeRichards:
        params.setNSlices(FREESASA_N_SLICES)

    tmp_pdb = tmp_dir / f"{uid}_conv.pdb"

    try:
        # BioPython 读 CIF
        parser = MMCIFParser(QUIET=True)
        bio_struct = parser.get_structure("prot", str(cif_path))

        if len(bio_struct) == 0:
            return [], "", "no_models_in_structure"

        # 从结构中提取序列 (按 label_seq_id 排序)
        model = bio_struct[0]
        struct_residues = []
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                if hetflag != " ":
                    continue
                resname = residue.resname.strip().upper()
                if resname in SKIP_RESIDUES:
                    continue
                seq_id = residue.id[1]  # label_seq_id (1-based)
                aa1 = AA3TO1.get(resname, "X")
                struct_residues.append((seq_id, aa1, resname))

        struct_residues.sort(key=lambda x: x[0])
        cif_sequence = "".join(aa for _, aa, _ in struct_residues)

        # 导出 PDB → FreeSASA
        io = PDBIO()
        io.set_structure(bio_struct)
        io.save(str(tmp_pdb), select=ProteinSelect())

        struct, result = _freesasa_calc_quiet(str(tmp_pdb), params)

    except Exception as e:
        _cleanup(tmp_pdb)
        return [], "", f"{type(e).__name__}: {e}"

    _cleanup(tmp_pdb)

    # 提取 per-residue RSA
    try:
        areas = result.residueAreas()
    except Exception as e:
        return [], cif_sequence, f"residueAreas_failed: {e}"

    if not areas:
        return [], cif_sequence, "residueAreas_empty"

    residue_data: List[dict] = []

    for chain_id, residues in areas.items():
        for res_key, res_area in residues.items():
            total_sasa = res_area.total
            resn = getattr(res_area, "residueType", "")
            resi = getattr(res_area, "residueNumber", str(res_key))
            if not resn:
                parts = str(res_key).split()
                resn = parts[0] if len(parts) >= 2 else "UNK"
                resi = parts[1] if len(parts) >= 2 else str(res_key)

            resn = resn.upper().strip()
            resi = str(resi).strip()

            if resn in SKIP_RESIDUES:
                continue

            max_asa = MAX_ASA.get(resn, MAX_ASA["UNK"])
            rsa = min(total_sasa / max_asa, 1.5) if max_asa > 0 else 0.0

            # AlphaFold label_seq_id 是 1-based → 0-based
            try:
                position = int(resi) - 1
            except ValueError:
                position = -1

            residue_data.append({
                "uniprot_id": uid,
                "residue": AA3TO1.get(resn, "X"),
                "position": position,
                "sasa": round(total_sasa, 2),
                "max_asa": round(max_asa, 2),
                "rsa": round(rsa, 4),
            })

    return residue_data, cif_sequence, ""


def _cleanup(path: Optional[Path]):
    try:
        if path and path.exists():
            path.unlink()
    except Exception:
        pass


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 5.1: AlphaFold DB → per-residue RSA")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"下载线程数 (默认: {DEFAULT_WORKERS})")
    parser.add_argument("--download-delay", type=float, default=DEFAULT_DOWNLOAD_DELAY,
                        help=f"下载间隔秒数 (默认: {DEFAULT_DOWNLOAD_DELAY})")
    parser.add_argument("--fasta", type=str, default=str(INPUT_FASTA),
                        help=f"输入FASTA (默认: {INPUT_FASTA})")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 5.1: AlphaFold DB → Per-Residue RSA")
    print("=" * 60)
    print(f"  FASTA:    {args.fasta}")
    print(f"  输出:     {OUTPUT_DIR}")
    print(f"  线程:     {args.workers}")
    print(f"  下载间隔: {args.download_delay}s")

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"\nERROR: FASTA 不存在: {fasta_path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    proc_log = Logger(
        OUTPUT_DIR / "process_log.tsv",
        "timestamp\tstep\tstatus\tdetail\n")
    err_log = Logger(
        OUTPUT_DIR / "error_log.tsv",
        "timestamp\tuniprot_id\terror_type\tdetail\n")

    proc_log.print_and_write("main", "start",
                             f"fasta={args.fasta}, workers={args.workers}")
    t0 = time.time()

    # ---- 1. 读取 UniProt ID ----
    print("\n--- 1. 读取 FASTA ---")
    all_uids, fasta_seqs = load_uniprot_data(fasta_path)
    print(f"  UniProt IDs: {len(all_uids)}")
    proc_log.print_and_write("load_fasta", "done", f"n={len(all_uids)}")

    # ---- 2. 下载 AlphaFold 结构 ----
    print("\n--- 2. 下载 AlphaFold 结构 ---")
    success_uids = download_all(
        all_uids, AF_STRUCT_DIR, AF_TMP_DIR,
        args.workers, args.download_delay,
        proc_log, err_log)
    proc_log.flush()
    err_log.flush()

    if not success_uids:
        print("ERROR: 没有可用的结构文件")
        return

    # ---- 3. 计算 RSA ----
    print(f"\n--- 3. 计算 RSA ({len(success_uids)} proteins) ---")
    proc_log.print_and_write("rsa", "start", f"n={len(success_uids)}")

    rsa_file = OUTPUT_DIR / "rsa_per_residue.tsv"
    rsa_header = "uniprot_id\tresidue\tposition\tsasa\tmax_asa\trsa\n"

    # 断点: 读取已完成的 uid
    done_uids: Set[str] = set()
    if rsa_file.exists():
        with open(rsa_file) as f:
            f.readline()  # skip header
            for line in f:
                uid = line.split("\t", 1)[0]
                if uid:
                    done_uids.add(uid)
        print(f"  断点恢复: {len(done_uids)} 已完成")

    todo_uids = sorted(success_uids - done_uids)
    print(f"  待处理: {len(todo_uids)}")

    if not rsa_file.exists():
        with open(rsa_file, "w") as f:
            f.write(rsa_header)

    # 临时目录 (FreeSASA CIF→PDB 转换用)
    rsa_tmp = OUTPUT_DIR / "_tmp_rsa"
    rsa_tmp.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_fail = 0
    n_seq_match = 0
    n_seq_mismatch = 0
    n_residues = 0
    buf: List[str] = []
    FLUSH_EVERY = 50

    for uid in todo_uids:
        cif_path = AF_STRUCT_DIR / f"AF-{uid}-F1-model_v6.cif"

        if not cif_path.exists():
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            err_log.write(f"{ts}\t{uid}\tno_structure\tfile not found: {cif_path}\n")
            n_fail += 1
            continue

        residues, cif_seq, error = compute_rsa_for_one(uid, cif_path, rsa_tmp)

        if error:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            err_log.write(f"{ts}\t{uid}\tfreesasa_fail\t{error}\n")
            n_fail += 1
            continue

        if not residues:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            err_log.write(f"{ts}\t{uid}\tno_residues\t0 residues extracted\n")
            n_fail += 1
            continue

        # 序列验证: CIF结构序列 vs FASTA序列
        fasta_seq = fasta_seqs.get(uid, "")
        if fasta_seq and cif_seq:
            if cif_seq == fasta_seq:
                n_seq_match += 1
            else:
                n_seq_mismatch += 1
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 找出差异位置 (最多展示前5个)
                diffs = []
                for i, (a, b) in enumerate(zip(cif_seq, fasta_seq)):
                    if a != b:
                        diffs.append(f"pos{i}:{a}→{b}")
                        if len(diffs) >= 5:
                            break
                len_info = f"cif_len={len(cif_seq)}, fasta_len={len(fasta_seq)}"
                diff_info = ",".join(diffs) if diffs else "length_only"
                if len(cif_seq) != len(fasta_seq):
                    diff_info += f",len_diff={len(cif_seq)-len(fasta_seq)}"
                err_log.write(
                    f"{ts}\t{uid}\tsequence_mismatch\t"
                    f"{len_info}, diffs=[{diff_info}]\n")
                # 跳过, 不写入输出
                continue
        elif not fasta_seq:
            # FASTA中找不到 (不应发生)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            err_log.write(f"{ts}\t{uid}\tno_fasta_seq\tuid not in fasta\n")
            n_fail += 1
            continue

        for rd in residues:
            buf.append(
                f"{rd['uniprot_id']}\t{rd['residue']}\t{rd['position']}\t"
                f"{rd['sasa']}\t{rd['max_asa']}\t{rd['rsa']}\n")
        n_residues += len(residues)
        n_done += 1

        # 定期 flush
        if n_done % FLUSH_EVERY == 0 or uid == todo_uids[-1]:
            if buf:
                with open(rsa_file, "a") as f:
                    f.writelines(buf)
                buf.clear()
            err_log.flush()

            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            proc_log.print_and_write("rsa", "progress",
                f"done={n_done}/{len(todo_uids)}, fail={n_fail}, "
                f"seq_ok={n_seq_match}, seq_bad={n_seq_mismatch}, "
                f"residues={n_residues}, ({rate:.1f} prot/s)")
            proc_log.flush()

    # flush 残留
    if buf:
        with open(rsa_file, "a") as f:
            f.writelines(buf)
    err_log.flush()
    proc_log.flush()

    # 清理
    try:
        shutil.rmtree(rsa_tmp)
    except OSError:
        pass

    elapsed_total = time.time() - t0
    proc_log.print_and_write("main", "done",
        f"elapsed={elapsed_total:.1f}s, "
        f"success={n_done}, fail={n_fail}, "
        f"seq_match={n_seq_match}, seq_mismatch={n_seq_mismatch}")
    proc_log.flush()

    # ---- 摘要 ----
    total_lines = 0
    if rsa_file.exists():
        total_lines = sum(1 for _ in open(rsa_file)) - 1

    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  UniProt IDs:      {len(all_uids)}")
    print(f"  结构下载成功:     {len(success_uids)}")
    print(f"  序列验证通过:     {n_seq_match}")
    print(f"  序列不匹配(跳过): {n_seq_mismatch}")
    print(f"  RSA写入成功:      {len(done_uids) + n_done}")
    print(f"  失败(其他):       {n_fail}")
    print(f"  总残基数:         {total_lines}")
    print(f"  耗时:             {elapsed_total:.1f}s")
    print()
    print("输出文件:")
    for fp in [rsa_file, OUTPUT_DIR / "process_log.tsv", OUTPUT_DIR / "error_log.tsv"]:
        if fp.exists():
            sz = fp.stat().st_size / 1024
            print(f"  {fp} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()