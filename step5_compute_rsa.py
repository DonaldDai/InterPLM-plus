"""
Step 5: 基于FreeSASA计算每个残基的RSA (Relative Solvent Accessibility)
=======================================================================

功能:
  1. 读取训练/验证集中每条蛋白质对应的PDB结构文件 (mmCIF.gz)
  2. 用FreeSASA计算每个残基的绝对SASA
  3. 除以MaxASA参考值得到RSA
  4. 输出per-residue RSA标注文件
  5. 全程记录成功/失败及具体失败原因

MaxASA参考值:
  采用 Tien et al. (2013) "Maximum allowed solvent accessibilities of residues
  in proteins", PLoS ONE 8(11):e80635 的理论最大值 (Gly-X-Gly三肽, 扩展构象)
  这是目前文献中被引用最多的MaxASA标度之一。

分辨率阈值:
  ≤ 2.5 Å (行业标准, 参考 NetSurfP-2.0, E-pRSA 等文献)

FreeSASA安装:
  pip install freesasa
  # 或
  conda install -c conda-forge freesasa

输入:
  cusdata/01_raw/pdb_files/                ← mmCIF.gz结构文件
  cusdata/01_raw/uniprot_pdb_mapping.tsv   ← UniProt→PDB映射
  cusdata/03_splits/train_id90.fasta       ← 训练集序列
  cusdata/03_splits/val_id90.fasta         ← 验证集序列

输出:
  cusdata/05_rsa/
  ├── rsa_per_residue.tsv        ← 主输出: 每个残基一行的RSA标注
  ├── rsa_process_log.tsv        ← 每个 (uniprot_id, pdb_id, chain) 的处理状态
  ├── rsa_failure_detail.tsv     ← 失败条目的详细原因
  └── rsa_summary.json           ← 汇总统计
"""

import gzip
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

try:
    import freesasa
except ImportError:
    print("ERROR: freesasa 未安装")
    print("安装方式: pip install freesasa")
    print("    或:   conda install -c conda-forge freesasa")
    sys.exit(1)

HAS_BIOPYTHON = False
try:
    from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select
    HAS_BIOPYTHON = True
except ImportError:
    print("WARNING: biopython 未安装，将仅使用 freesasa 原生解析")
    print("建议安装: pip install biopython")


# ============================================================
# 配置区
# ============================================================
PDB_DIR = Path("cusdata/01_raw/pdb_files")
MAPPING_FILE = Path("cusdata/01_raw/uniprot_pdb_mapping.tsv")
SPLIT_DIR = Path("cusdata/03_splits")
OUTPUT_DIR = Path("cusdata/05_rsa")

# FreeSASA参数
FREESASA_ALGORITHM = freesasa.LeeRichards   # LeeRichards 或 ShrakeRupley
FREESASA_PROBE_RADIUS = 1.4                 # 探针半径 (Å), 模拟水分子
FREESASA_N_SLICES = 20                      # Lee-Richards切片数

# MaxASA参考值: Tien et al. 2013, PLoS ONE 8(11):e80635
# 理论最大值, Gly-X-Gly三肽, 扩展构象 (单位: Å²)
MAX_ASA_TIEN2013 = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
    "CYS": 167.0, "GLN": 225.0, "GLU": 223.0, "GLY": 104.0,
    "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
    "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
    "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
    # 非标准残基降级处理
    "MSE": 224.0,   # selenomethionine → 按MET
    "SEC": 167.0,   # selenocysteine → 按CYS
    "UNK": 200.0,   # 未知 → 中位数近似
}

# 三字母 → 单字母
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U",
}

# 跳过的非氨基酸残基 (水, 离子, 常见缓冲液/添加剂)
SKIP_RESIDUES = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "DMS",
    "ACE", "NME", "NH2", "ACT", "FMT", "BME", "MPD", "TRS", "EPE",
    "CL", "NA", "MG", "ZN", "CA", "FE", "MN", "CO", "NI", "CU", "K",
    "IOD", "BR", "CD", "HG",
}

# 拆分集阈值
SPLIT_THRESHOLD = "id90"

# 标准氨基酸三字母码集合 (用于导出过滤)
STANDARD_AA_SET = set(MAX_ASA_TIEN2013.keys())


# ============================================================
# PDB导出过滤器: 只保留标准氨基酸残基
# ============================================================
if HAS_BIOPYTHON:
    class ProteinSelect(Select):
        """
        PDBIO导出过滤器:
        - 残基级: 只保留标准氨基酸 (20种 + MSE/SEC)
        - 原子级: 跳过氢原子 (FreeSASA不需要H, 且H占总原子数~50%)

        解决三个问题:
        1. 去掉水/核酸/配体 → 大幅减少原子数
        2. 去掉氢原子 → 再减半 (PDB中含H的结构很常见)
        3. FreeSASA的SASA计算本身就只用重原子
        """
        def accept_residue(self, residue):
            resname = residue.resname.strip().upper()
            hetflag = residue.id[0]
            if hetflag == "W":
                return False
            if resname in STANDARD_AA_SET:
                return True
            if resname in SKIP_RESIDUES:
                return False
            if resname in {"A", "U", "G", "C", "DA", "DT", "DG", "DC",
                            "ADE", "URA", "GUA", "CYT", "THY"}:
                return False
            if hetflag != " ":
                return False
            return True

        def accept_atom(self, atom):
            # 跳过氢原子 (element='H' 或 'D'=deuterium)
            elem = atom.element.strip().upper()
            if elem in ("H", "D"):
                return False
            return True



# 多进程配置
NUM_WORKERS = 4   # 根据CPU核数调整, 建议 = 物理核心数


# ============================================================
# 日志记录器 (断点续跑: 读取已有日志 + 追加写入)
# ============================================================
class ProcessLogger:
    """
    状态:
      success              计算成功 (detail中可能含chain_truncated/atom_truncated警告)
      skip_no_pdb          PDB文件不存在
      skip_no_mapping      映射表中无PDB记录
      fail_gzip            gzip解压失败
      fail_parse           FreeSASA/BioPython解析失败
      fail_freesasa        FreeSASA计算异常
      fail_no_residue      解析成功但无蛋白质残基
      fail_rsa_calc        RSA归一化阶段异常

    截断警告不作为独立状态, 而是合并到 success/fail 的 detail 字段中,
    确保有截断但计算成功的蛋白仍被标记为 success, 断点续传时正确跳过。

    断点续跑:
      __init__ 时读取已有日志, 提取 status=success 的 uniprot_id
      主流程据此跳过已完成的条目, 新记录追加到同一文件
    """

    HEADER = "timestamp\tuniprot_id\tpdb_id\tchain_id\tstatus\tdetail\n"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.log_file = output_dir / "rsa_process_log.tsv"
        self.fail_file = output_dir / "rsa_failure_detail.tsv"
        self.done_uids: Set[str] = set()
        self.counters = Counter()
        self._total_written = 0

        # 读取已有日志恢复断点
        if self.log_file.exists():
            with open(self.log_file) as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 5:
                        uid, status = parts[1], parts[4]
                        self.counters[status] += 1
                        self._total_written += 1
                        if status == "success":
                            self.done_uids.add(uid)
            print(f"  [断点恢复] 已有 {self._total_written} 条记录, "
                  f"{len(self.done_uids)} 个UniProt已完成")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w") as f:
                f.write(self.HEADER)
            with open(self.fail_file, "w") as f:
                f.write(self.HEADER)

    def is_done(self, uniprot_id: str) -> bool:
        return uniprot_id in self.done_uids

    def write_batch(self, entries: List[Tuple[str, str, str, str, str]]):
        """批量追加: [(uid, pdb_id, chain, status, detail), ...]"""
        if not entries:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_lines, fail_lines = [], []
        for uid, pdb_id, chain_id, status, detail in entries:
            dc = detail.replace("\t", " ").replace("\n", " ")[:500]
            line = f"{ts}\t{uid}\t{pdb_id}\t{chain_id}\t{status}\t{dc}\n"
            log_lines.append(line)
            self.counters[status] += 1
            self._total_written += 1
            if status == "success":
                self.done_uids.add(uid)
            else:
                fail_lines.append(line)
        with open(self.log_file, "a") as f:
            f.writelines(log_lines)
        if fail_lines:
            with open(self.fail_file, "a") as f:
                f.writelines(fail_lines)

    def save_summary(self):
        n_success = self.counters.get('success', 0)
        total = max(self._total_written, 1)
        summary = {
            "total_records": self._total_written,
            "unique_success": len(self.done_uids),
            "status_counts": dict(self.counters.most_common()),
            "success_rate": f"{n_success / total * 100:.1f}%",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        sf = self.output_dir / "rsa_summary.json"
        with open(sf, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return sf


# ============================================================
# 辅助函数: 映射表 / 拆分集 / 文件定位 / 解压
# ============================================================
def load_uniprot_pdb_mapping(mapping_path: Path) -> Dict[str, List[str]]:
    mapping = defaultdict(list)
    with open(mapping_path) as f:
        header = f.readline().strip().split("\t")
        acc_col, pdb_col = None, None
        for i, h in enumerate(header):
            hl = h.lower()
            if hl in ("entry", "accession"):
                acc_col = i
            elif "pdb" in hl:
                pdb_col = i
        if acc_col is None:
            acc_col = 0
        if pdb_col is None:
            return dict(mapping)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(acc_col, pdb_col):
                continue
            uid = parts[acc_col].strip()
            pdb_field = parts[pdb_col].strip()
            if pdb_field:
                mapping[uid] = [p.strip().lower()
                                for p in pdb_field.split(";") if p.strip()]
    return dict(mapping)


def load_split_ids(split_dir: Path, threshold: str) -> Set[str]:
    ids: Set[str] = set()
    for prefix in ("train", "val"):
        fasta = split_dir / f"{prefix}_{threshold}.fasta"
        if not fasta.exists():
            continue
        with open(fasta) as f:
            for line in f:
                if line.startswith(">"):
                    h = line[1:].split()[0]
                    parts = h.split("|")
                    ids.add(parts[1] if len(parts) >= 2 else h)
    return ids


def find_pdb_file(pdb_id: str, pdb_dir: Path) -> Optional[Path]:
    pdb_id = pdb_id.lower()
    if len(pdb_id) >= 4:
        c = pdb_dir / pdb_id[1:3] / f"{pdb_id}.cif.gz"
        if c.exists():
            return c
    for hit in pdb_dir.rglob(f"{pdb_id}.cif.gz"):
        return hit
    return None


def decompress_cif_gz(gz_path: Path, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / gz_path.name.replace(".gz", "")
    with gzip.open(gz_path, "rb") as fi, open(out, "wb") as fo:
        while True:
            chunk = fi.read(1024 * 1024)
            if not chunk:
                break
            fo.write(chunk)
    return out


def _cleanup(path: Optional[Path]):
    try:
        if path and path.exists():
            path.unlink()
    except Exception:
        pass


# ============================================================
# FreeSASA C-level stderr 抑制
# ============================================================
def _freesasa_calc_quiet(pdb_path: str, params) -> Tuple:
    """FreeSASA计算 + 抑制C层stderr"""
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


# ============================================================
# FreeSASA 计算 (mmCIF → PDB转换 + 链ID重映射)
# ============================================================
def compute_sasa(cif_path: Path) -> Tuple[Optional[freesasa.Structure],
                                          Optional[freesasa.Result], str,
                                          List[str], Dict]:
    """
    BioPython解析mmCIF → 链ID重映射 → 导出临时PDB → FreeSASA读取PDB

    返回 (structure, result, error_message, warnings, chain_id_map)
    """
    params = freesasa.Parameters()
    params.setAlgorithm(FREESASA_ALGORITHM)
    params.setProbeRadius(FREESASA_PROBE_RADIUS)
    if FREESASA_ALGORITHM == freesasa.LeeRichards:
        params.setNSlices(FREESASA_N_SLICES)

    errors_collected = []
    warnings_collected = []

    if HAS_BIOPYTHON:
        tmp_pdb = Path(str(cif_path).replace(".cif", "_conv.pdb"))
        chain_id_map = {}
        try:
            parser = MMCIFParser(QUIET=True)
            bio_struct = parser.get_structure("prot", str(cif_path))
            if len(bio_struct) == 0:
                return None, None, "mmcif_parse|no_models", [], {}

            model = bio_struct[0]
            needs_remap = any(len(c.id) > 1 for c in model)

            if needs_remap:
                single_chars = (
                    [chr(c) for c in range(ord('A'), ord('Z') + 1)] +
                    [chr(c) for c in range(ord('a'), ord('z') + 1)] +
                    [chr(c) for c in range(ord('0'), ord('9') + 1)]
                )
                chain_list = list(model.get_chains())
                used = {c.id for c in chain_list
                        if len(c.id) == 1 and c.id in single_chars}
                available = [c for c in single_chars if c not in used]
                avail_idx = 0

                remap_plan = []
                for chain in chain_list:
                    old = chain.id
                    if len(old) == 1 and old in single_chars:
                        remap_plan.append((chain, old, old))
                        chain_id_map[old] = old
                    elif avail_idx < len(available):
                        new = available[avail_idx]; avail_idx += 1
                        remap_plan.append((chain, old, new))
                        chain_id_map[old] = new
                    else:
                        remap_plan.append((chain, old, None))
                        chain_id_map[old] = None

                dropped = [old for _, old, new in remap_plan if new is None]
                if dropped:
                    tot = len(chain_list); kept = tot - len(dropped)
                    warnings_collected.append(
                        f"chain_truncated|total={tot},kept={kept},"
                        f"dropped={len(dropped)},"
                        f"dropped_ids={','.join(dropped[:10])}")

                for chain, old, new in remap_plan:
                    model.detach_child(chain.id)
                for chain, old, new in remap_plan:
                    if new is not None:
                        chain.id = new
                        model.add(chain)
            else:
                for chain in model:
                    chain_id_map[chain.id] = chain.id

            # 兜底: 过滤后原子数仍可能超过99,999 (巨型蛋白复合物)
            # 统计的原子数必须和 PDBIO.save(select=ProteinSelect()) 实际写出的一致
            PDB_ATOM_LIMIT = 99999
            _sel = ProteinSelect()

            atom_count = sum(
                1 for chain in model for res in chain
                if _sel.accept_residue(res)
                for atom in res if _sel.accept_atom(atom)
            )

            if atom_count > PDB_ATOM_LIMIT:
                # 按链的原子数从小到大排序, 优先保留小链 (累积更多链)
                chain_atoms = []
                for chain in model:
                    n = sum(1 for res in chain
                            if _sel.accept_residue(res)
                            for atom in res if _sel.accept_atom(atom))
                    chain_atoms.append((chain.id, n))
                chain_atoms.sort(key=lambda x: x[1])

                # 贪心累加, 超限后的链全部移除
                cumulative = 0
                keep_chains = set()
                for cid, n in chain_atoms:
                    if cumulative + n <= PDB_ATOM_LIMIT:
                        cumulative += n
                        keep_chains.add(cid)
                    # 超限后不再添加

                removed_chains = []
                for cid, n in chain_atoms:
                    if cid not in keep_chains:
                        removed_chains.append(cid)
                        try:
                            model.detach_child(cid)
                        except Exception:
                            pass

                warnings_collected.append(
                    f"atom_truncated|total_atoms={atom_count},"
                    f"limit={PDB_ATOM_LIMIT},"
                    f"kept_chains={len(keep_chains)},"
                    f"removed_chains={len(removed_chains)},"
                    f"removed_ids={','.join(removed_chains[:10])}")

            io = PDBIO()
            io.set_structure(bio_struct)
            io.save(str(tmp_pdb), select=ProteinSelect())

            struct, result = _freesasa_calc_quiet(str(tmp_pdb), params)

            try: tmp_pdb.unlink()
            except OSError: pass

            return struct, result, "", warnings_collected, chain_id_map

        except Exception as e:
            errors_collected.append(f"biopdb_convert=[{type(e).__name__}: {e}]")
            try: tmp_pdb.unlink()
            except OSError: pass

    if not HAS_BIOPYTHON:
        errors_collected.append("biopython_not_installed")

    return None, None, "; ".join(errors_collected), warnings_collected, {}


# ============================================================
# 提取 per-residue RSA
# ============================================================
def extract_per_residue_rsa(
    structure: freesasa.Structure,
    result: freesasa.Result,
    chain_id_map: Dict = None,
) -> Tuple[List[dict], str]:
    """从FreeSASA结果提取per-residue SASA → RSA (还原链ID)"""
    if chain_id_map is None:
        chain_id_map = {}
    reverse_map = {}
    if chain_id_map:
        for orig, remapped in chain_id_map.items():
            if remapped is not None:
                reverse_map[remapped] = orig

    try:
        areas = result.residueAreas()
    except Exception as e:
        return [], f"residueAreas_failed: {e}"
    if not areas:
        return [], "residueAreas_empty"

    residue_data: List[dict] = []
    unknown_resnames: Counter = Counter()

    for chain_id, residues in areas.items():
        original_chain_id = reverse_map.get(chain_id, chain_id)
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
            max_asa = MAX_ASA_TIEN2013.get(resn)
            if max_asa is None:
                unknown_resnames[resn] += 1
                max_asa = MAX_ASA_TIEN2013["UNK"]
            rsa = min(total_sasa / max_asa, 1.5) if max_asa > 0 else 0.0
            residue_data.append({
                "chain": original_chain_id,
                "resn": resn,
                "resi": resi,
                "resn_1letter": AA3TO1.get(resn, "X"),
                "sasa": round(total_sasa, 2),
                "max_asa": round(max_asa, 2),
                "rsa": round(rsa, 4),
            })

    warn = ""
    if unknown_resnames:
        warn = f"unknown_resnames={dict(unknown_resnames.most_common(5))}"
    return residue_data, warn


# ============================================================
# Worker函数: 处理单个UniProt (纯函数, 多进程安全)
# ============================================================
def _worker(args: Tuple) -> Tuple[str, List[dict], List[Tuple]]:
    """
    多进程worker入口

    参数 (tuple): (uniprot_id, pdb_ids, pdb_dir_str)
    返回: (uid, residue_data, log_entries)
      log_entries: [(uid, pdb_id, chain, status, detail), ...]
    """
    import tempfile as _tmpmod
    uniprot_id, pdb_ids, pdb_dir_str = args
    pdb_dir = Path(pdb_dir_str)
    log_entries: List[Tuple[str, str, str, str, str]] = []

    with _tmpmod.TemporaryDirectory(prefix=f"rsa_{uniprot_id[:8]}_") as td:
        tmp_dir = Path(td)

        for pdb_id in pdb_ids:
            pdb_id = pdb_id.lower().strip()
            if not pdb_id:
                continue

            gz_path = find_pdb_file(pdb_id, pdb_dir)
            if gz_path is None:
                log_entries.append((uniprot_id, pdb_id, "*", "skip_no_pdb",
                                    f"not_found: {pdb_id}.cif.gz"))
                continue

            # 解压
            try:
                cif_path = decompress_cif_gz(gz_path, tmp_dir)
            except gzip.BadGzipFile as e:
                log_entries.append((uniprot_id, pdb_id, "*", "fail_gzip",
                                    f"BadGzipFile: {e}"))
                continue
            except EOFError as e:
                log_entries.append((uniprot_id, pdb_id, "*", "fail_gzip",
                                    f"truncated: {e}"))
                continue
            except Exception as e:
                log_entries.append((uniprot_id, pdb_id, "*", "fail_gzip",
                                    f"{type(e).__name__}: {e}"))
                continue

            # FreeSASA
            try:
                structure, result, sasa_err, sasa_warnings, cid_map = compute_sasa(cif_path)
            except Exception as e:
                log_entries.append((uniprot_id, pdb_id, "*", "fail_freesasa",
                                    f"unexpected: {type(e).__name__}: {e}"))
                _cleanup(cif_path)
                continue

            # sasa_warnings 不再单独写日志行, 而是合并到后续的 success/fail detail 中
            warn_str = "; ".join(sasa_warnings) if sasa_warnings else ""

            if structure is None or result is None:
                detail = sasa_err
                if warn_str:
                    detail += f"; {warn_str}"
                log_entries.append((uniprot_id, pdb_id, "*", "fail_parse", detail))
                _cleanup(cif_path)
                continue

            # 提取RSA
            try:
                residue_data, rsa_warn = extract_per_residue_rsa(structure, result, cid_map)
            except Exception as e:
                tb = traceback.format_exc()[-300:]
                detail = f"{type(e).__name__}: {e} | {tb}"
                if warn_str:
                    detail += f"; {warn_str}"
                log_entries.append((uniprot_id, pdb_id, "*", "fail_rsa_calc", detail))
                _cleanup(cif_path)
                continue

            if not residue_data:
                detail = f"0_residues. {rsa_warn}"
                if warn_str:
                    detail += f"; {warn_str}"
                log_entries.append((uniprot_id, pdb_id, "*", "fail_no_residue", detail))
                _cleanup(cif_path)
                continue

            # 成功 — 把截断警告合并到 success 的 detail 中
            for rd in residue_data:
                rd["uniprot_id"] = uniprot_id
                rd["pdb_id"] = pdb_id

            for ch in sorted(set(rd["chain"] for rd in residue_data)):
                n = sum(1 for rd in residue_data if rd["chain"] == ch)
                det = f"residues={n}"
                if rsa_warn:
                    det += f"; {rsa_warn}"
                if warn_str:
                    det += f"; {warn_str}"
                log_entries.append((uniprot_id, pdb_id, ch, "success", det))

            _cleanup(cif_path)
            return uniprot_id, residue_data, log_entries

    return uniprot_id, [], log_entries


# ============================================================
# 主流程 (多进程 + 断点续跑)
# ============================================================
def main():
    import multiprocessing

    print("=" * 60)
    print(f"Step 5: FreeSASA per-residue RSA ({NUM_WORKERS} workers)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 映射表
    print(f"\n加载映射表: {MAPPING_FILE}")
    if not MAPPING_FILE.exists():
        print(f"ERROR: {MAPPING_FILE} 不存在"); return
    full_mapping = load_uniprot_pdb_mapping(MAPPING_FILE)
    print(f"  条目数: {len(full_mapping)}")

    # 2. 拆分集
    print(f"\n加载拆分集: threshold={SPLIT_THRESHOLD}")
    split_ids = load_split_ids(SPLIT_DIR, SPLIT_THRESHOLD)
    if split_ids:
        target_ids = split_ids & set(full_mapping.keys())
        print(f"  拆分集: {len(split_ids)}, 有映射: {len(target_ids)}")
    else:
        print("  WARNING: 未找到拆分文件, 处理全部")
        target_ids = set(full_mapping.keys())

    # 3. 日志 + 断点
    print(f"\n初始化日志...")
    logger = ProcessLogger(OUTPUT_DIR)
    todo_ids = sorted(target_ids - logger.done_uids)
    n_skip = len(target_ids) - len(todo_ids)
    print(f"  总目标: {len(target_ids)}, 已完成跳过: {n_skip}, 待处理: {len(todo_ids)}")

    if not todo_ids:
        print("\n全部已完成, 无需计算")
        logger.save_summary()
        return

    # 4. RSA输出 (追加)
    rsa_file = OUTPUT_DIR / "rsa_per_residue.tsv"
    rsa_header = ("uniprot_id\tpdb_id\tchain\tresi\tresn\tresn_1letter\t"
                  "sasa\tmax_asa\trsa\n")
    if not rsa_file.exists():
        with open(rsa_file, "w") as f:
            f.write(rsa_header)

    # 5. 构建任务
    tasks = []
    no_mapping = []
    for uid in todo_ids:
        pdb_ids = full_mapping.get(uid, [])
        if not pdb_ids:
            no_mapping.append((uid, "", "", "skip_no_mapping", "no_pdb_ids"))
            continue
        tasks.append((uid, pdb_ids, str(PDB_DIR)))

    if no_mapping:
        logger.write_batch(no_mapping)
        print(f"  无映射跳过: {len(no_mapping)}")

    print(f"\n开始计算 ({len(tasks)} proteins, {NUM_WORKERS} workers)")
    print(f"  算法:  {'Lee-Richards' if FREESASA_ALGORITHM == freesasa.LeeRichards else 'Shrake-Rupley'}")
    print(f"  探针:  {FREESASA_PROBE_RADIUS} Å")
    print(f"  MaxASA: Tien et al. 2013\n")

    # 6. 多进程
    t0 = time.time()
    done_count = 0
    batch_log: List[Tuple] = []
    batch_rsa: List[str] = []
    FLUSH_EVERY = 20

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=NUM_WORKERS) as pool:
        for uid, residue_data, log_entries in pool.imap_unordered(
                _worker, tasks, chunksize=4):

            done_count += 1
            batch_log.extend(log_entries)

            for rd in residue_data:
                batch_rsa.append(
                    f"{rd['uniprot_id']}\t{rd['pdb_id']}\t{rd['chain']}\t"
                    f"{rd['resi']}\t{rd['resn']}\t{rd['resn_1letter']}\t"
                    f"{rd['sasa']}\t{rd['max_asa']}\t{rd['rsa']}\n")

            if done_count % FLUSH_EVERY == 0 or done_count == len(tasks):
                if batch_rsa:
                    with open(rsa_file, "a") as f:
                        f.writelines(batch_rsa)
                    batch_rsa.clear()
                if batch_log:
                    logger.write_batch(batch_log)
                    batch_log.clear()

                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                ns = logger.counters.get("success", 0)
                nf = sum(v for k, v in logger.counters.items()
                         if k != "success")
                print(f"  [{done_count:>6}/{len(tasks)}] "
                      f"success={ns}  fail={nf}  ({rate:.1f} prot/s)")

    # flush残留
    if batch_rsa:
        with open(rsa_file, "a") as f:
            f.writelines(batch_rsa)
    if batch_log:
        logger.write_batch(batch_log)

    elapsed_total = time.time() - t0
    sf = logger.save_summary()

    # 7. 摘要
    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)
    print(f"本次处理:  {len(tasks)} proteins")
    print(f"累计完成:  {len(logger.done_uids)} (含之前断点)")
    print(f"耗时:      {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")
    print(f"吞吐:      {len(tasks)/max(elapsed_total,1):.1f} prot/s × {NUM_WORKERS} workers")
    print()
    print("状态分布 (累计):")
    for status, count in logger.counters.most_common():
        print(f"  {status:<25} {count:>6}")
    if rsa_file.exists():
        n_lines = sum(1 for _ in open(rsa_file)) - 1
        sz = rsa_file.stat().st_size / (1024 * 1024)
        print(f"\nRSA: {rsa_file} ({n_lines} residues, {sz:.1f} MB)")
    print(f"\n输出:")
    print(f"  {rsa_file}")
    print(f"  {logger.log_file}")
    print(f"  {logger.fail_file}")
    print(f"  {sf}")


if __name__ == "__main__":
    main()