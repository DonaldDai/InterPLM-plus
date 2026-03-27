"""
Step 5.2: 基于 PDBe best_structures + 增强型 CIF 计算 per-residue RSA
=====================================================================

使用实验结构 (非预测), 通过 PDBe API 获取每个 UniProt ID 的最佳结构,
下载增强型 CIF (含 pdbx_sifts_xref_db_num → UniProt 序列下标),
计算 FreeSASA → RSA。

流程:
  1. 获取 UniProt ID → PDBe best_structures API → 选择覆盖全长的最佳结构
  2. 下载增强型 CIF: {PDB_ID}_updated.cif
  3. 从 CIF 中提取指定链/区间, 记录 pdbx_sifts_xref_db_num (UniProt position)
  4. 转换为 PDB → FreeSASA 计算 SASA → RSA

输入:
  cusdata/01_raw/uniprot_pdb_sequences.fasta

输出:
  cusdata/05_2_updated_cif_rsa/
  ├── cif/                        ← 增强型 CIF 文件
  ├── rsa_per_residue.tsv         ← 主输出
  ├── best_structures_log.tsv     ← best_structures 选择日志
  ├── process_log.tsv             ← 处理流程日志
  └── error_log.tsv               ← 异常日志

依赖:
  pip install freesasa biopython requests

用法:
  python step5_2_updated_cif_rsa.py
  python step5_2_updated_cif_rsa.py --workers 4
"""

import gzip
import json
import os
import sys
import time
import shutil
import argparse
import requests
import traceback
import multiprocessing
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set

try:
    import freesasa
except ImportError:
    print("ERROR: freesasa 未安装 → pip install freesasa")
    sys.exit(1)

try:
    from Bio.PDB import MMCIFParser, PDBIO, Select
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio import SeqIO
except ImportError:
    print("ERROR: biopython 未安装 → pip install biopython")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
INPUT_FASTA = Path("cusdata/01_raw/uniprot_pdb_sequences.fasta")
OUTPUT_DIR = Path("cusdata/05_2_updated_cif_rsa")
CIF_DIR = OUTPUT_DIR / "cif"

BEST_STRUCTURES_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uid}"
UPDATED_CIF_URL = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}_updated.cif"

# FreeSASA
FREESASA_ALGORITHM = freesasa.LeeRichards
FREESASA_PROBE_RADIUS = 1.4
FREESASA_N_SLICES = 20

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

# 默认超参
DEFAULT_WORKERS = 4
DEFAULT_REQUEST_DELAY = 0.15


# ============================================================
# ProteinSelect (同 step5)
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
    def __init__(self, log_path: Path, header: str):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists() or log_path.stat().st_size == 0:
            with open(log_path, "w") as f:
                f.write(header)

    def write(self, line: str):
        with open(self.log_path, "a") as f:
            f.write(line)

    def print_and_write(self, step: str, status: str, detail: str = ""):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts}\t{step}\t{status}\t{detail}\n"
        with open(self.log_path, "a") as f:
            f.write(line)
        print(f"  [{ts}] {step}: {status}" + (f" | {detail}" if detail else ""))


# ============================================================
# 读取 FASTA
# ============================================================
def load_fasta_data(fasta_path: Path) -> Dict[str, int]:
    """返回 {uid: seq_length}"""
    result = {}
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        parts = record.id.split("|")
        uid = parts[1] if len(parts) >= 2 else record.id
        result[uid] = len(record.seq)
    return result


# ============================================================
# Worker: 处理单个 UniProt ID (API查询 + 下载CIF + 计算RSA)
# ============================================================
def process_one_protein(args_tuple: Tuple) -> Tuple[str, List[dict], List[str]]:
    """
    处理单个 UniProt 蛋白, 返回 (uid, residue_data, log_lines)

    log_lines: 待写入各日志文件的行, 格式:
      "LOG_TYPE\tline_content" 其中 LOG_TYPE = "BEST" / "PROC" / "ERR"
    """
    uid, seq_len, cif_dir_str, request_delay = args_tuple
    cif_dir = Path(cif_dir_str)
    logs: List[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log(log_type: str, content: str):
        logs.append(f"{log_type}\t{content}")

    # ---- 1. 查询 best_structures ----
    time.sleep(request_delay)
    url = BEST_STRUCTURES_URL.format(uid=uid)
    try:
        resp = requests.get(url, timeout=30)
    except Exception as e:
        _log("ERR", f"{ts}\t{uid}\tapi_error\t{type(e).__name__}: {e}\n")
        _log("BEST", f"{ts}\t{uid}\tapi_error\t0\t\t\n")
        return uid, [], logs

    if resp.status_code == 404:
        _log("BEST", f"{ts}\t{uid}\tno_structures\t0\t\t\n")
        return uid, [], logs
    if resp.status_code != 200:
        _log("ERR", f"{ts}\t{uid}\tapi_http_{resp.status_code}\t{resp.text[:200]}\n")
        _log("BEST", f"{ts}\t{uid}\tapi_http_{resp.status_code}\t0\t\t\n")
        return uid, [], logs

    try:
        data = resp.json()
        structures = data.get(uid, [])
    except Exception as e:
        _log("ERR", f"{ts}\t{uid}\tjson_parse_error\t{e}\n")
        return uid, [], logs

    if not structures:
        _log("BEST", f"{ts}\t{uid}\tempty_response\t0\t\t\n")
        return uid, [], logs

    # ---- 2. 选择最佳结构: 找覆盖全长的 ----
    best = None
    for entry in structures:
        pdb_id = entry.get("pdb_id", "")
        chain_id = entry.get("chain_id", "")
        unp_start = entry.get("unp_start", 0)
        unp_end = entry.get("unp_end", 0)
        coverage = unp_end - unp_start + 1

        if coverage >= seq_len:
            best = entry
            break

    if best is None:
        # 无全覆盖, 取第一个 (API 已按质量排序)
        best = structures[0]
        top_cov = best.get("unp_end", 0) - best.get("unp_start", 0) + 1
        _log("BEST",
             f"{ts}\t{uid}\tpartial_coverage\t{len(structures)}\t"
             f"best_pdb={best.get('pdb_id','')},cov={top_cov}/{seq_len}\t\n")
    else:
        _log("BEST",
             f"{ts}\t{uid}\tselected\t{len(structures)}\t"
             f"pdb={best['pdb_id']},chain={best['chain_id']},"
             f"range={best['unp_start']}-{best['unp_end']}\t\n")

    pdb_id = best["pdb_id"]
    chain_id = best["chain_id"]
    unp_start = best["unp_start"]
    unp_end = best["unp_end"]

    # ---- 3. 下载增强型 CIF ----
    cif_path = cif_dir / f"{pdb_id}_updated.cif"
    if not cif_path.exists():
        time.sleep(request_delay)
        cif_url = UPDATED_CIF_URL.format(pdb_id=pdb_id)
        try:
            cif_resp = requests.get(cif_url, timeout=120)
            if cif_resp.status_code != 200:
                _log("ERR", f"{ts}\t{uid}\tcif_download_http_{cif_resp.status_code}\t{cif_url}\n")
                return uid, [], logs
            content = cif_resp.text
            if content.strip().startswith("<!") or "<html" in content[:200].lower():
                _log("ERR", f"{ts}\t{uid}\tcif_download_html\trate limited or blocked\n")
                return uid, [], logs
            with open(cif_path, "w") as f:
                f.write(content)
        except Exception as e:
            _log("ERR", f"{ts}\t{uid}\tcif_download_error\t{type(e).__name__}: {e}\n")
            return uid, [], logs

    # ---- 4. 解析 CIF, 提取 pdbx_sifts_xref_db_num ----
    try:
        mmcif_dict = MMCIF2Dict(str(cif_path))
    except Exception as e:
        _log("ERR", f"{ts}\t{uid}\tcif_parse_error\t{type(e).__name__}: {e}\n")
        return uid, [], logs

    # 构建 atom→uniprot_pos 映射
    # 增强型 CIF 中 _atom_site.pdbx_sifts_xref_db_num 包含 UniProt 序列位置
    try:
        auth_asym_ids = mmcif_dict.get("_atom_site.auth_asym_id", [])
        label_seq_ids = mmcif_dict.get("_atom_site.label_seq_id", [])
        label_comp_ids = mmcif_dict.get("_atom_site.label_comp_id", [])
        sifts_db_nums = mmcif_dict.get("_atom_site.pdbx_sifts_xref_db_num", [])
        sifts_db_names = mmcif_dict.get("_atom_site.pdbx_sifts_xref_db_name", [])
        group_pdb = mmcif_dict.get("_atom_site.group_PDB", [])

        if not sifts_db_nums:
            _log("ERR", f"{ts}\t{uid}\tno_sifts_annotation\tpdbx_sifts_xref_db_num not found in {pdb_id}\n")
            return uid, [], logs

    except Exception as e:
        _log("ERR", f"{ts}\t{uid}\tcif_field_error\t{type(e).__name__}: {e}\n")
        return uid, [], logs

    # 构建 (chain, label_seq_id) → uniprot_pos 映射
    # 只取 ATOM 行, 目标链, UniProt 标注
    residue_uniprot_map: Dict[Tuple[str, str], int] = {}  # (chain, label_seq_id) → uniprot_pos
    seen_residues: Set[Tuple[str, str]] = set()

    for i in range(len(auth_asym_ids)):
        if group_pdb[i] != "ATOM":
            continue
        chain = auth_asym_ids[i]
        if chain != chain_id:
            continue

        seq_id = label_seq_ids[i]
        key = (chain, seq_id)
        if key in seen_residues:
            continue
        seen_residues.add(key)

        # 检查 SIFTS 标注是否指向 UniProt
        db_name = sifts_db_names[i] if i < len(sifts_db_names) else ""
        db_num = sifts_db_nums[i] if i < len(sifts_db_nums) else "?"

        if db_name.upper() not in ("UNP", "UNIPROT", "UNIPROT_AC", ""):
            continue
        if db_num == "?" or db_num == "." or not db_num:
            continue

        try:
            uniprot_pos = int(db_num) - 1  # 转 0-based
            if unp_start - 1 <= uniprot_pos <= unp_end - 1:
                residue_uniprot_map[key] = uniprot_pos
        except ValueError:
            continue

    if not residue_uniprot_map:
        _log("ERR", f"{ts}\t{uid}\tno_mapped_residues\tchain={chain_id}, pdb={pdb_id}\n")
        return uid, [], logs

    # ---- 5. BioPython 解析 → PDB 导出 → FreeSASA ----
    import tempfile
    with tempfile.TemporaryDirectory(prefix=f"rsa_{uid[:8]}_") as tmp_str:
        tmp_dir = Path(tmp_str)
        tmp_pdb = tmp_dir / f"{uid}_conv.pdb"

        try:
            parser = MMCIFParser(QUIET=True)
            bio_struct = parser.get_structure("prot", str(cif_path))

            if len(bio_struct) == 0:
                _log("ERR", f"{ts}\t{uid}\tno_models\tpdb={pdb_id}\n")
                return uid, [], logs

            model = bio_struct[0]

            # 检查过滤后原子数是否超过 PDB 格式上限 (99999)
            PDB_ATOM_LIMIT = 99999
            _sel = ProteinSelect()

            atom_count = sum(
                1 for chain in model for res in chain
                if _sel.accept_residue(res)
                for atom in res if _sel.accept_atom(atom)
            )

            if atom_count > PDB_ATOM_LIMIT:
                # 只保留目标链, 移除其他链
                remove_ids = [c.id for c in model if c.id != chain_id]
                for cid in remove_ids:
                    try:
                        model.detach_child(cid)
                    except Exception:
                        pass

                # 重新计数目标链
                atom_count = sum(
                    1 for chain in model for res in chain
                    if _sel.accept_residue(res)
                    for atom in res if _sel.accept_atom(atom)
                )

                # 如果单链仍超限, 从尾部截断残基
                if atom_count > PDB_ATOM_LIMIT:
                    target_chain = model[chain_id] if chain_id in model else None
                    if target_chain:
                        residues_list = [r for r in target_chain
                                         if _sel.accept_residue(r)]
                        cumul = 0
                        cut_idx = len(residues_list)
                        for i, res in enumerate(residues_list):
                            n_atoms = sum(1 for a in res if _sel.accept_atom(a))
                            cumul += n_atoms
                            if cumul > PDB_ATOM_LIMIT:
                                cut_idx = i
                                break
                        # 移除 cut_idx 之后的残基
                        for res in residues_list[cut_idx:]:
                            try:
                                target_chain.detach_child(res.id)
                            except Exception:
                                pass

                _log("PROC",
                     f"{ts}\t{uid}\tatom_truncated\t"
                     f"original={atom_count},limit={PDB_ATOM_LIMIT},"
                     f"removed_chains={len(remove_ids)}\n")

            io = PDBIO()
            io.set_structure(bio_struct)
            io.save(str(tmp_pdb), select=ProteinSelect())

            # FreeSASA
            params = freesasa.Parameters()
            params.setAlgorithm(FREESASA_ALGORITHM)
            params.setProbeRadius(FREESASA_PROBE_RADIUS)
            if FREESASA_ALGORITHM == freesasa.LeeRichards:
                params.setNSlices(FREESASA_N_SLICES)

            # 抑制 C stderr
            stderr_fd = sys.stderr.fileno()
            saved = os.dup(stderr_fd)
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, stderr_fd)
                os.close(devnull)
                fs_struct = freesasa.Structure(str(tmp_pdb))
                fs_result = freesasa.calc(fs_struct, params)
            finally:
                os.dup2(saved, stderr_fd)
                os.close(saved)

        except Exception as e:
            _log("ERR", f"{ts}\t{uid}\tfreesasa_error\t{type(e).__name__}: {e}\n")
            return uid, [], logs

    # ---- 6. 提取 per-residue RSA, 匹配 UniProt 位置 ----
    try:
        areas = fs_result.residueAreas()
    except Exception as e:
        _log("ERR", f"{ts}\t{uid}\tresidueAreas_error\t{e}\n")
        return uid, [], logs

    residue_data: List[dict] = []

    for fs_chain_id, residues in areas.items():
        # FreeSASA 可能重映射了链ID, 尝试匹配
        for res_key, res_area in residues.items():
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

            # 查 UniProt 位置
            # FreeSASA 用的链ID可能和原始CIF不同, 尝试多种匹配
            uniprot_pos = None
            for try_chain in (chain_id, fs_chain_id):
                key = (try_chain, resi)
                if key in residue_uniprot_map:
                    uniprot_pos = residue_uniprot_map[key]
                    break

            if uniprot_pos is None:
                continue  # 无法映射到 UniProt, 跳过

            total_sasa = res_area.total
            max_asa = MAX_ASA.get(resn, MAX_ASA["UNK"])
            rsa = min(total_sasa / max_asa, 1.5) if max_asa > 0 else 0.0

            residue_data.append({
                "uniprot_id": uid,
                "residue": AA3TO1.get(resn, "X"),
                "position": uniprot_pos,
                "sasa": round(total_sasa, 2),
                "max_asa": round(max_asa, 2),
                "rsa": round(rsa, 4),
            })

    # 按 position 排序
    residue_data.sort(key=lambda x: x["position"])

    n_mapped = len(residue_data)
    n_sifts = len(residue_uniprot_map)
    _log("PROC",
         f"{ts}\t{uid}\tsuccess\tpdb={pdb_id},chain={chain_id},"
         f"sifts_residues={n_sifts},rsa_residues={n_mapped}\n")

    return uid, residue_data, logs


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 5.2: PDBe best_structures + updated CIF → per-residue RSA")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"并行进程数 (默认: {DEFAULT_WORKERS})")
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY,
                        help=f"API请求间隔秒 (默认: {DEFAULT_REQUEST_DELAY})")
    parser.add_argument("--fasta", type=str, default=str(INPUT_FASTA))
    args = parser.parse_args()

    print("=" * 60)
    print("Step 5.2: PDBe best_structures + updated CIF → RSA")
    print("=" * 60)
    print(f"  workers:  {args.workers}")
    print(f"  delay:    {args.request_delay}s")
    print(f"  fasta:    {args.fasta}")
    print(f"  output:   {OUTPUT_DIR}")

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"ERROR: {fasta_path} 不存在")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CIF_DIR.mkdir(parents=True, exist_ok=True)

    proc_log = Logger(
        OUTPUT_DIR / "process_log.tsv",
        "timestamp\tstep\tstatus\tdetail\n")
    err_log = Logger(
        OUTPUT_DIR / "error_log.tsv",
        "timestamp\tuniprot_id\terror_type\tdetail\n")
    best_log = Logger(
        OUTPUT_DIR / "best_structures_log.tsv",
        "timestamp\tuniprot_id\tstatus\tn_structures\tdetail\t\n")

    proc_log.print_and_write("main", "start",
                             f"workers={args.workers}, delay={args.request_delay}")
    t0 = time.time()

    # 1. 读 FASTA
    print("\n--- 1. 读取 FASTA ---")
    seq_lens = load_fasta_data(fasta_path)
    print(f"  UniProt IDs: {len(seq_lens)}")
    proc_log.print_and_write("fasta", "loaded", f"n={len(seq_lens)}")

    # 2. 断点恢复
    rsa_file = OUTPUT_DIR / "rsa_per_residue.tsv"
    rsa_header = "uniprot_id\tresidue\tposition\tsasa\tmax_asa\trsa\n"

    done_uids: Set[str] = set()
    if rsa_file.exists():
        with open(rsa_file) as f:
            f.readline()
            for line in f:
                uid = line.split("\t", 1)[0]
                if uid:
                    done_uids.add(uid)
        print(f"  断点恢复: {len(done_uids)} 已完成")
    else:
        with open(rsa_file, "w") as f:
            f.write(rsa_header)

    todo_uids = sorted(set(seq_lens.keys()) - done_uids)
    print(f"  待处理: {len(todo_uids)}")

    if not todo_uids:
        print("\n全部已完成!")
        return

    # 3. 构建任务
    tasks = [
        (uid, seq_lens[uid], str(CIF_DIR), args.request_delay)
        for uid in todo_uids
    ]

    # 4. 多进程执行
    print(f"\n--- 2. 处理 ({len(tasks)} proteins, {args.workers} workers) ---\n")

    n_done = 0
    n_success = 0
    n_fail = 0
    n_no_structure = 0
    rsa_buf: List[str] = []
    FLUSH_EVERY = 20

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for uid, residue_data, log_lines in pool.imap_unordered(
                process_one_protein, tasks, chunksize=1):

            n_done += 1

            # 分发日志
            for entry in log_lines:
                parts = entry.split("\t", 1)
                log_type = parts[0]
                content = parts[1] if len(parts) > 1 else ""
                if log_type == "BEST":
                    best_log.write(content)
                elif log_type == "PROC":
                    proc_log.write(content)
                elif log_type == "ERR":
                    err_log.write(content)

            if residue_data:
                for rd in residue_data:
                    rsa_buf.append(
                        f"{rd['uniprot_id']}\t{rd['residue']}\t{rd['position']}\t"
                        f"{rd['sasa']}\t{rd['max_asa']}\t{rd['rsa']}\n")
                n_success += 1
            else:
                has_no_struct = any("no_structures" in e or "empty_response" in e
                                    for e in log_lines)
                if has_no_struct:
                    n_no_structure += 1
                else:
                    n_fail += 1

            # 定期 flush
            if n_done % FLUSH_EVERY == 0 or n_done == len(tasks):
                if rsa_buf:
                    with open(rsa_file, "a") as f:
                        f.writelines(rsa_buf)
                    rsa_buf.clear()

                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"  [{n_done:>6}/{len(tasks)}] "
                      f"success={n_success} no_structure={n_no_structure} "
                      f"fail={n_fail} ({rate:.1f} prot/s)")

    # flush 残留
    if rsa_buf:
        with open(rsa_file, "a") as f:
            f.writelines(rsa_buf)

    elapsed_total = time.time() - t0
    proc_log.print_and_write("main", "done",
        f"elapsed={elapsed_total:.1f}s, success={n_success}, "
        f"no_structure={n_no_structure}, fail={n_fail}")

    # 摘要
    total_lines = 0
    if rsa_file.exists():
        total_lines = sum(1 for _ in open(rsa_file)) - 1

    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  UniProt IDs:      {len(seq_lens)}")
    print(f"  已完成(含断点):   {len(done_uids) + n_success}")
    print(f"  本次成功:         {n_success} (含部分覆盖)")
    print(f"  无PDBe结构:       {n_no_structure}")
    print(f"  失败(其他):       {n_fail}")
    print(f"  总残基数:         {total_lines}")
    print(f"  耗时:             {elapsed_total:.1f}s")
    print()
    print("输出文件:")
    for fp in [rsa_file, OUTPUT_DIR / "best_structures_log.tsv",
               OUTPUT_DIR / "process_log.tsv", OUTPUT_DIR / "error_log.tsv"]:
        if fp.exists():
            sz = fp.stat().st_size / 1024
            print(f"  {fp} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()