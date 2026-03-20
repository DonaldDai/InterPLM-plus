"""
Step 1: 一站式数据下载 (UniProt序列 + 映射表 + PDB结构文件)
==============================================================

本脚本完成所有需要网络请求的操作，后续步骤全部离线执行。

下载内容:
  A. UniProt reviewed + PDB 的蛋白质FASTA序列
  B. UniProt → PDB 映射表 (含链信息、残基范围)
  C. PDB结构文件 (mmCIF格式，用于后续DSSP/FreeSASA计算RSA)

为什么选mmCIF而非传统PDB格式:
  - PDB格式对超过99,999个原子或62条链的结构无法表示
  - 2024年起wwPDB已将mmCIF作为主推归档格式
  - DSSP和FreeSASA均支持mmCIF输入
  - mmCIF文件包含更完整的元数据 (实验方法、分辨率等)

输出目录:
  data/01_raw/
  ├── uniprot_pdb_sequences.fasta
  ├── uniprot_pdb_mapping.tsv
  ├── pdb_files/          ← mmCIF结构文件 (按中间两位编号分目录)
  │   ├── ab/
  │   │   ├── 1abc.cif.gz
  │   │   └── 2abd.cif.gz
  │   ├── cd/
  │   │   └── 3cde.cif.gz
  │   └── ...
  ├── download_log.json   ← 下载状态记录 (支持断点续传)
  └── pdb_resolution_filter.tsv  ← PDB分辨率信息 (用于质量筛选)
"""

import requests
import time
import os
import sys
import json
import gzip
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, Dict, List, Tuple
import math


# ============================================================
# 配置区
# ============================================================
OUTPUT_DIR = Path("cusdata/01_raw")
FASTA_FILE = OUTPUT_DIR / "uniprot_pdb_sequences.fasta"
MAPPING_FILE = OUTPUT_DIR / "uniprot_pdb_mapping.tsv"
PDB_DIR = OUTPUT_DIR / "pdb_files"
DOWNLOAD_LOG = OUTPUT_DIR / "download_log.json"
RESOLUTION_FILE = OUTPUT_DIR / "pdb_resolution_filter.tsv"

# UniProt REST API
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

# PDB 下载源 (RCSB 为主, PDBe 为备用)
PDB_DOWNLOAD_SOURCES = {
    "rcsb": "https://files.rcsb.org/download/{pdb_id}.cif.gz",
    "pdbe": "https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}_updated.cif.gz",
    "pdbj": "https://pdbj.org/rest/downloadPDBfile?format=mmcif&id={pdb_id}&compression=gz",
}
PRIMARY_SOURCE = "rcsb"

# 筛选条件
QUERY = "(reviewed:true) AND (database:pdb)"
MIN_LENGTH = 50
MAX_LENGTH = 1000

# PDB下载参数
# MAX_RESOLUTION = 2.5       # 只下载分辨率 ≤ 2.5Å 的结构 (NMR结构无分辨率，保留)
# TODO： 尝试更严格的效果
MAX_RESOLUTION = math.inf
PDB_DOWNLOAD_WORKERS = 8   # 并行下载线程数
PDB_DOWNLOAD_RETRIES = 3   # 每个文件最大重试次数
PDB_DOWNLOAD_TIMEOUT = 30  # 单个文件下载超时 (秒)
RATE_LIMIT_DELAY = 0.1     # 请求间隔 (秒)，避免被服务器限流


# ============================================================
# 断点续传日志
# ============================================================
class DownloadLog:
    """
    记录下载状态，支持中断后恢复
    
    状态:
    - pending:    尚未尝试
    - downloaded: 下载成功
    - failed:     多次重试后失败
    - skipped:    被过滤条件排除 (如分辨率过低)
    """
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.data = {"uniprot_done": False, "mapping_done": False, "pdb_status": {}}
        if log_path.exists():
            with open(log_path) as f:
                self.data = json.load(f)
    
    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)
    
    def is_pdb_done(self, pdb_id):
        return self.data["pdb_status"].get(pdb_id, {}).get("status") in ("downloaded", "skipped")
    
    def mark_pdb(self, pdb_id, status, detail=""):
        self.data["pdb_status"][pdb_id] = {"status": status, "detail": detail}
    
    def get_stats(self):
        stats = defaultdict(int)
        for info in self.data["pdb_status"].values():
            stats[info["status"]] += 1
        return dict(stats)


# ============================================================
# Part A: 下载FASTA序列
# ============================================================
def download_fasta(log: DownloadLog):
    """下载UniProt FASTA序列"""
    if log.data["uniprot_done"] and FASTA_FILE.exists():
        seq_count = sum(1 for line in open(FASTA_FILE) if line.startswith(">"))
        print(f"[跳过] FASTA已存在: {seq_count} 条序列")
        return seq_count
    
    print("下载UniProt FASTA序列...")
    
    # 查询总数
    count_params = {"query": QUERY, "size": 1, "format": "json"}
    resp = requests.get(f"{UNIPROT_API}/search", params=count_params)
    resp.raise_for_status()
    total = resp.headers.get("x-total-results", "?")
    print(f"  符合条件总数: {total}")
    
    # 流式下载
    length_query = f"{QUERY} AND (length:[{MIN_LENGTH} TO {MAX_LENGTH}])"
    download_params = {"query": length_query, "format": "fasta", "compressed": "false"}
    
    with requests.get(f"{UNIPROT_API}/stream", params=download_params, stream=True) as r:
        r.raise_for_status()
        with open(FASTA_FILE, "w") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024, decode_unicode=True):
                f.write(chunk)
    
    seq_count = sum(1 for line in open(FASTA_FILE) if line.startswith(">"))
    print(f"  下载完成: {seq_count} 条序列")
    
    log.data["uniprot_done"] = True
    log.save()
    return seq_count


# ============================================================
# Part B: 下载UniProt-PDB映射表
# ============================================================
def download_mapping(log: DownloadLog):
    """
    下载映射表，字段说明:
    - accession:     UniProt ID (如 P12345)
    - id:            UniProt entry name (如 PROT_HUMAN)
    - protein_name:  蛋白质名称
    - length:        序列长度
    - xref_pdb:      PDB交叉引用 (格式: "1ABC;2DEF;3GHI")
    - ft_transmem:   跨膜区注释 (后续标注用)
    - ft_signal:     信号肽注释 (后续标注用)
    - structure_3d:  结构方法和分辨率信息
    """
    if log.data["mapping_done"] and MAPPING_FILE.exists():
        line_count = sum(1 for _ in open(MAPPING_FILE)) - 1
        print(f"[跳过] 映射表已存在: {line_count} 条记录")
        return
    
    print("下载UniProt-PDB映射表...")
    
    length_query = f"{QUERY} AND (length:[{MIN_LENGTH} TO {MAX_LENGTH}])"
    mapping_params = {
        "query": length_query,
        "format": "tsv",
        "fields": "accession,id,protein_name,length,xref_pdb,ft_transmem,ft_signal,structure_3d",
        "compressed": "false",
    }
    
    with requests.get(f"{UNIPROT_API}/stream", params=mapping_params, stream=True) as r:
        r.raise_for_status()
        with open(MAPPING_FILE, "w") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024, decode_unicode=True):
                f.write(chunk)
    
    line_count = sum(1 for _ in open(MAPPING_FILE)) - 1
    print(f"  下载完成: {line_count} 条记录")
    
    log.data["mapping_done"] = True
    log.save()


# ============================================================
# Part C-1: 从映射表中提取所有需要下载的PDB ID
# ============================================================
def extract_pdb_ids() -> Set[str]:
    """
    从映射表中提取所有唯一的PDB ID
    
    UniProt的xref_pdb字段格式: "1ABC;2DEF;3GHI"
    一个UniProt条目可能对应多个PDB结构
    """
    pdb_ids = set()
    
    with open(MAPPING_FILE) as f:
        header = f.readline().strip().split("\t")
        # 找到xref_pdb列的索引
        try:
            pdb_col = header.index("PDB")
        except ValueError:
            # 有时列名不完全一致，尝试其他可能的名称
            for i, h in enumerate(header):
                if "pdb" in h.lower():
                    pdb_col = i
                    break
            else:
                print(f"WARNING: 无法找到PDB列, header = {header}")
                return pdb_ids
        
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > pdb_col and parts[pdb_col]:
                # "1ABC;2DEF;3GHI" → ["1ABC", "2DEF", "3GHI"]
                ids = [pid.strip().lower() for pid in parts[pdb_col].split(";") if pid.strip()]
                pdb_ids.update(ids)
    
    print(f"从映射表中提取到 {len(pdb_ids)} 个唯一PDB ID")
    return pdb_ids


# ============================================================
# Part C-2: 获取PDB分辨率信息 (用于质量筛选)
# ============================================================
def fetch_pdb_resolution(pdb_ids: Set[str]) -> Dict[str, float]:
    """
    批量查询PDB结构的分辨率
    
    使用RCSB Search API批量获取，比逐个查询高效得多
    
    返回: {pdb_id: resolution_angstrom} 
          NMR结构返回 -1.0 (无分辨率概念，但保留)
    """
    if RESOLUTION_FILE.exists():
        print(f"[跳过] 分辨率信息已存在: {RESOLUTION_FILE}")
        resolutions = {}
        with open(RESOLUTION_FILE) as f:
            f.readline()  # header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    resolutions[parts[0]] = float(parts[1])
        return resolutions
    
    print(f"查询 {len(pdb_ids)} 个PDB结构的分辨率...")
    
    # RCSB GraphQL API 批量查询 (每批最多1000个)
    graphql_url = "https://data.rcsb.org/graphql"
    
    resolutions = {}
    pdb_list = sorted(pdb_ids)
    batch_size = 500
    
    for i in range(0, len(pdb_list), batch_size):
        batch = pdb_list[i:i + batch_size]
        
        # 构造GraphQL查询
        ids_str = ", ".join(f'"{pid.upper()}"' for pid in batch)
        query = f"""
        {{
          entries(entry_ids: [{ids_str}]) {{
            rcsb_id
            rcsb_entry_info {{
              resolution_combined
              experimental_method
            }}
          }}
        }}
        """
        
        try:
            resp = requests.post(graphql_url, json={"query": query}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            
            for entry in data.get("data", {}).get("entries", []) or []:
                pid = entry["rcsb_id"].lower()
                info = entry.get("rcsb_entry_info", {})
                res = info.get("resolution_combined")
                method = info.get("experimental_method", "")
                
                if res is not None:
                    resolutions[pid] = res[0] if isinstance(res, list) else res
                elif "NMR" in str(method).upper():
                    resolutions[pid] = -1.0  # NMR无分辨率，标记为-1
                else:
                    resolutions[pid] = 999.0  # 无分辨率信息
        
        except Exception as e:
            print(f"  WARNING: GraphQL查询批次 {i//batch_size} 失败: {e}")
            # 失败的批次标记为未知分辨率
            for pid in batch:
                resolutions.setdefault(pid, 999.0)
        
        if (i // batch_size) % 10 == 0:
            print(f"  已查询 {min(i + batch_size, len(pdb_list))} / {len(pdb_list)} ...")
        
        time.sleep(RATE_LIMIT_DELAY)
    
    # 保存分辨率信息
    with open(RESOLUTION_FILE, "w") as f:
        f.write("pdb_id\tresolution\n")
        for pid, res in sorted(resolutions.items()):
            f.write(f"{pid}\t{res}\n")
    
    # 统计
    valid = [r for r in resolutions.values() if 0 < r <= MAX_RESOLUTION]
    nmr = sum(1 for r in resolutions.values() if r == -1.0)
    excluded = sum(1 for r in resolutions.values() if r > MAX_RESOLUTION)
    
    print(f"  分辨率 ≤ {MAX_RESOLUTION}Å: {len(valid)} 个")
    print(f"  NMR结构 (保留):          {nmr} 个")
    print(f"  分辨率 > {MAX_RESOLUTION}Å (排除):  {excluded} 个")
    
    return resolutions


# ============================================================
# Part C-3: 下载PDB mmCIF结构文件
# ============================================================
def download_single_pdb(pdb_id: str, source: str = PRIMARY_SOURCE) -> Tuple[str, bool, str]:
    """
    下载单个PDB mmCIF文件
    
    PDB文件按中间两位字符分目录存储 (PDB标准惯例):
    1abc → pdb_files/ab/1abc.cif.gz
    2def → pdb_files/de/2def.cif.gz
    
    这种分目录策略避免单目录下文件数过多导致文件系统性能下降
    """
    pdb_id = pdb_id.lower()
    
    # 分目录: 取PDB ID的中间两位
    subdir = pdb_id[1:3]
    target_dir = PDB_DIR / subdir
    target_file = target_dir / f"{pdb_id}.cif.gz"
    
    # 已存在则跳过
    if target_file.exists() and target_file.stat().st_size > 100:
        return pdb_id, True, "already_exists"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试下载 (先主源，失败后尝试备用源)
    sources_to_try = [source] + [s for s in PDB_DOWNLOAD_SOURCES if s != source]
    
    for src in sources_to_try:
        url = PDB_DOWNLOAD_SOURCES[src].format(pdb_id=pdb_id)
        
        for attempt in range(PDB_DOWNLOAD_RETRIES):
            try:
                resp = requests.get(url, timeout=PDB_DOWNLOAD_TIMEOUT, stream=True)
                
                if resp.status_code == 200:
                    with open(target_file, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            f.write(chunk)
                    
                    # 验证文件完整性 (gzip文件至少应该有20字节)
                    if target_file.stat().st_size > 100:
                        return pdb_id, True, f"downloaded_from_{src}"
                    else:
                        target_file.unlink(missing_ok=True)
                
                elif resp.status_code == 404:
                    break  # 此源没有该文件，尝试下一个源
                
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < PDB_DOWNLOAD_RETRIES - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                continue
        
        time.sleep(RATE_LIMIT_DELAY)
    
    return pdb_id, False, "all_sources_failed"


def download_pdb_files(pdb_ids: Set[str], resolutions: Dict[str, float], log: DownloadLog):
    """
    批量并行下载PDB结构文件
    
    策略:
    1. 跳过已下载的 (断点续传)
    2. 跳过分辨率过低的
    3. 多线程并行下载
    4. 每100个文件保存一次进度日志
    """
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    
    # 过滤: 排除分辨率过低的结构
    to_download = []
    skipped_resolution = 0
    
    for pid in sorted(pdb_ids):
        if log.is_pdb_done(pid):
            continue
        
        res = resolutions.get(pid, 999.0)
        if res > MAX_RESOLUTION and res != 999.0:
            # 分辨率超限且不是未知 → 跳过
            log.mark_pdb(pid, "skipped", f"resolution={res}")
            skipped_resolution += 1
            continue
        
        to_download.append(pid)
    
    already_done = sum(1 for pid in pdb_ids if log.is_pdb_done(pid))
    
    print(f"\nPDB文件下载计划:")
    print(f"  总PDB ID数:        {len(pdb_ids)}")
    print(f"  已完成 (续传跳过): {already_done}")
    print(f"  分辨率排除:        {skipped_resolution}")
    print(f"  待下载:            {len(to_download)}")
    
    if not to_download:
        print("  所有文件已下载完毕!")
        log.save()
        return
    
    # 并行下载
    print(f"\n开始并行下载 ({PDB_DOWNLOAD_WORKERS} 线程)...")
    
    downloaded = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=PDB_DOWNLOAD_WORKERS) as executor:
        futures = {
            executor.submit(download_single_pdb, pid): pid
            for pid in to_download
        }
        
        for i, future in enumerate(as_completed(futures)):
            pid, success, detail = future.result()
            
            if success:
                log.mark_pdb(pid, "downloaded", detail)
                downloaded += 1
            else:
                log.mark_pdb(pid, "failed", detail)
                failed += 1
            
            # 进度报告
            total_processed = i + 1
            if total_processed % 100 == 0 or total_processed == len(to_download):
                pct = total_processed / len(to_download) * 100
                print(f"  [{total_processed}/{len(to_download)}] ({pct:.1f}%) "
                      f"成功: {downloaded}, 失败: {failed}")
                log.save()  # 定期保存进度
    
    log.save()
    
    # 最终统计
    stats = log.get_stats()
    print(f"\nPDB下载完成:")
    for status, count in sorted(stats.items()):
        print(f"  {status}: {count}")


# ============================================================
# Part C-4: 验证PDB文件完整性
# ============================================================
def validate_pdb_files(pdb_ids: Set[str], log: DownloadLog):
    """
    验证已下载的mmCIF文件:
    1. 文件存在且非空
    2. gzip格式有效 (能打开)
    3. 包含 _atom_site 记录 (说明有坐标数据)
    """
    print("\n验证PDB文件完整性...")
    
    valid = 0
    invalid = 0
    missing = 0
    
    downloaded_ids = [
        pid for pid, info in log.data["pdb_status"].items()
        if info.get("status") == "downloaded"
    ]
    
    for pid in downloaded_ids[:100]:  # 抽样验证前100个
        subdir = pid[1:3]
        filepath = PDB_DIR / subdir / f"{pid}.cif.gz"
        
        if not filepath.exists():
            missing += 1
            continue
        
        try:
            with gzip.open(filepath, "rt") as f:
                content_head = f.read(5000)
                if "_atom_site" in content_head or "ATOM" in content_head:
                    valid += 1
                else:
                    invalid += 1
        except (gzip.BadGzipFile, EOFError, OSError):
            invalid += 1
    
    sample_total = valid + invalid + missing
    print(f"  抽样验证 {sample_total} 个文件:")
    print(f"    有效:   {valid}")
    print(f"    无效:   {invalid}")
    print(f"    缺失:   {missing}")
    
    if invalid > 0:
        print(f"  WARNING: {invalid} 个文件可能损坏，建议重新下载")


# ============================================================
# 汇总报告
# ============================================================
def print_summary(log: DownloadLog):
    """打印最终汇总"""
    print()
    print("=" * 60)
    print("Step 1 完成: 数据下载汇总")
    print("=" * 60)
    
    # FASTA
    if FASTA_FILE.exists():
        seq_count = sum(1 for line in open(FASTA_FILE) if line.startswith(">"))
        fasta_size = FASTA_FILE.stat().st_size / (1024 * 1024)
        print(f"FASTA序列:  {seq_count} 条 ({fasta_size:.1f} MB)")
    
    # 映射表
    if MAPPING_FILE.exists():
        map_count = sum(1 for _ in open(MAPPING_FILE)) - 1
        print(f"映射表:     {map_count} 条记录")
    
    # PDB文件
    stats = log.get_stats()
    total_pdb = sum(stats.values())
    print(f"PDB结构:    {total_pdb} 个")
    for status, count in sorted(stats.items()):
        print(f"  - {status}: {count}")
    
    # 磁盘占用
    if PDB_DIR.exists():
        total_size = sum(f.stat().st_size for f in PDB_DIR.rglob("*.cif.gz"))
        print(f"PDB文件总大小: {total_size / (1024**3):.2f} GB")
    
    print()
    print("目录结构:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── uniprot_pdb_sequences.fasta")
    print(f"  ├── uniprot_pdb_mapping.tsv")
    print(f"  ├── pdb_resolution_filter.tsv")
    print(f"  ├── download_log.json")
    print(f"  └── pdb_files/")
    
    if PDB_DIR.exists():
        subdirs = sorted([d.name for d in PDB_DIR.iterdir() if d.is_dir()])
        if subdirs:
            print(f"      ├── {subdirs[0]}/ ...")
            if len(subdirs) > 1:
                print(f"      └── {subdirs[-1]}/ ...")
            print(f"      ({len(subdirs)} 个子目录)")
    
    print()
    print("下一步: python step2_cdhit_clustering.py")


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("Step 1: 一站式数据下载")
    print("  UniProt序列 + 映射表 + PDB结构文件")
    print("=" * 60)
    print(f"筛选条件:   {QUERY}")
    print(f"序列长度:   {MIN_LENGTH}-{MAX_LENGTH}")
    print(f"分辨率上限: {MAX_RESOLUTION} Å")
    print(f"输出目录:   {OUTPUT_DIR}")
    print(f"下载线程:   {PDB_DOWNLOAD_WORKERS}")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = DownloadLog(DOWNLOAD_LOG)
    
    # ----------------------------------------------------------
    # A. 下载FASTA序列
    # ----------------------------------------------------------
    print("━" * 60)
    print("Part A: UniProt FASTA 序列")
    print("━" * 60)
    download_fasta(log)
    
    # ----------------------------------------------------------
    # B. 下载映射表
    # ----------------------------------------------------------
    print()
    print("━" * 60)
    print("Part B: UniProt → PDB 映射表")
    print("━" * 60)
    download_mapping(log)
    
    # ----------------------------------------------------------
    # C. 下载PDB结构文件
    # ----------------------------------------------------------
    print()
    print("━" * 60)
    print("Part C: PDB mmCIF 结构文件")
    print("━" * 60)
    
    # C-1: 提取PDB ID
    pdb_ids = extract_pdb_ids()
    
    if not pdb_ids:
        print("WARNING: 未提取到任何PDB ID，请检查映射表")
        return
    
    # C-2: 查询分辨率 (用于质量筛选)
    resolutions = fetch_pdb_resolution(pdb_ids)
    
    # C-3: 批量下载
    download_pdb_files(pdb_ids, resolutions, log)
    
    # C-4: 验证
    validate_pdb_files(pdb_ids, log)
    
    # ----------------------------------------------------------
    # 汇总报告
    # ----------------------------------------------------------
    print_summary(log)


if __name__ == "__main__":
    main()
