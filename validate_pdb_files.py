"""
PDB文件完整性校验工具
=====================

对 data/01_raw/pdb_files/ 下所有已下载的 mmCIF.gz 文件逐一校验，
记录每个文件的具体失败原因，生成校验报告。

校验层级 (逐层递进，前一层通过才检查下一层):
  Level 1 - 文件系统:  文件存在、大小 > 0
  Level 2 - Gzip完整:  能正常解压、非截断
  Level 3 - 内容格式:  是合法的mmCIF文本 (非HTML错误页、非空文本)
  Level 4 - 结构数据:  包含 _atom_site 坐标记录
  Level 5 - 数据质量:  有足够的ATOM行、有蛋白质链

输出:
  data/01_raw/validation_report.tsv    - 每个PDB文件一行的详细报告
  data/01_raw/validation_summary.json  - 汇总统计
  (终端同时打印实时进度和最终摘要)

用法:
  python validate_pdb_files.py
  python validate_pdb_files.py --pdb-dir /path/to/pdb_files
  python validate_pdb_files.py --sample 200   # 只抽样检查200个
"""

import gzip
import json
import argparse
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple


# ============================================================
# 配置
# ============================================================
DEFAULT_PDB_DIR = Path("cusdata/01_raw/pdb_files")
DEFAULT_OUTPUT_DIR = Path("cusdata/01_raw")

# 校验阈值
MIN_FILE_SIZE_BYTES = 100        # gzip文件至少应有100字节
MIN_DECOMPRESSED_BYTES = 500     # 解压后至少500字节 (排除空/极短文件)
MIN_ATOM_LINES = 10              # 至少10行坐标记录 (排除只有header的文件)
MAX_PEEK_BYTES = 50_000          # 只读取前50KB做快速校验 (加速)
FULL_SCAN_ATOM_THRESHOLD = True  # 是否完整扫描统计ATOM行数


# ============================================================
# 校验结果数据结构
# ============================================================
@dataclass
class ValidationResult:
    pdb_id: str
    file_path: str
    status: str = "unknown"          # pass / fail
    fail_level: str = ""             # 失败所在的校验层级
    fail_reason: str = ""            # 具体失败原因
    file_size_bytes: int = 0
    decompressed_size_bytes: int = 0 # 前50KB解压后的大小 (近似)
    is_gzip_valid: bool = False
    content_type: str = ""           # mmcif / html / xml / empty / binary / unknown
    has_atom_site: bool = False
    atom_line_count: int = 0
    chain_ids: str = ""              # 检测到的链ID
    has_protein_chain: bool = False
    error_detail: str = ""           # 异常堆栈或额外信息


# ============================================================
# 核心校验逻辑
# ============================================================
def validate_single_pdb(filepath: Path) -> ValidationResult:
    """
    对单个 mmCIF.gz 文件执行五层校验
    """
    pdb_id = filepath.stem.replace(".cif", "")
    result = ValidationResult(pdb_id=pdb_id, file_path=str(filepath))
    
    # ----------------------------------------------------------
    # Level 1: 文件系统检查
    # ----------------------------------------------------------
    if not filepath.exists():
        result.status = "fail"
        result.fail_level = "L1_filesystem"
        result.fail_reason = "file_not_found"
        return result
    
    result.file_size_bytes = filepath.stat().st_size
    
    if result.file_size_bytes == 0:
        result.status = "fail"
        result.fail_level = "L1_filesystem"
        result.fail_reason = "file_empty_0_bytes"
        return result
    
    if result.file_size_bytes < MIN_FILE_SIZE_BYTES:
        result.status = "fail"
        result.fail_level = "L1_filesystem"
        result.fail_reason = f"file_too_small_{result.file_size_bytes}_bytes"
        return result
    
    # ----------------------------------------------------------
    # Level 2: Gzip完整性
    # ----------------------------------------------------------
    try:
        with gzip.open(filepath, "rb") as f:
            raw_head = f.read(MAX_PEEK_BYTES)
            result.decompressed_size_bytes = len(raw_head)
            result.is_gzip_valid = True
    except gzip.BadGzipFile as e:
        result.status = "fail"
        result.fail_level = "L2_gzip"
        result.fail_reason = "bad_gzip_format"
        result.error_detail = str(e)
        return result
    except EOFError as e:
        result.status = "fail"
        result.fail_level = "L2_gzip"
        result.fail_reason = "gzip_truncated_eof"
        result.error_detail = str(e)
        return result
    except OSError as e:
        result.status = "fail"
        result.fail_level = "L2_gzip"
        result.fail_reason = "gzip_os_error"
        result.error_detail = str(e)
        return result
    except Exception as e:
        result.status = "fail"
        result.fail_level = "L2_gzip"
        result.fail_reason = "gzip_unknown_error"
        result.error_detail = f"{type(e).__name__}: {e}"
        return result
    
    if result.decompressed_size_bytes < MIN_DECOMPRESSED_BYTES:
        result.status = "fail"
        result.fail_level = "L2_gzip"
        result.fail_reason = f"decompressed_too_small_{result.decompressed_size_bytes}_bytes"
        return result
    
    # ----------------------------------------------------------
    # Level 3: 内容格式检查
    # ----------------------------------------------------------
    try:
        text_head = raw_head.decode("utf-8", errors="replace")
    except Exception as e:
        result.status = "fail"
        result.fail_level = "L3_content"
        result.fail_reason = "decode_failed"
        result.error_detail = str(e)
        return result
    
    # 检测是否为HTML错误页 (常见于403/404被服务器返回HTML)
    text_lower = text_head[:2000].lower()
    
    if "<html" in text_lower or "<!doctype" in text_lower:
        result.content_type = "html"
        result.status = "fail"
        result.fail_level = "L3_content"
        result.fail_reason = "html_error_page"
        # 尝试提取错误信息
        for marker in ["<title>", "error", "not found", "forbidden", "403", "404", "500"]:
            if marker in text_lower:
                idx = text_lower.index(marker)
                result.error_detail = text_head[max(0,idx-20):idx+100].strip()[:200]
                break
        return result
    
    if "<?xml" in text_lower and "<html" not in text_lower:
        # XML但非HTML，可能是某种API错误响应
        if "error" in text_lower or "fault" in text_lower:
            result.content_type = "xml_error"
            result.status = "fail"
            result.fail_level = "L3_content"
            result.fail_reason = "xml_error_response"
            result.error_detail = text_head[:300].strip()
            return result
    
    # 检查是否为合法mmCIF (应包含 data_ 开头行或 loop_ 关键字)
    has_data_block = "data_" in text_head[:5000]
    has_loop = "loop_" in text_head[:10000]
    has_mmcif_keywords = "_entry.id" in text_head or "_cell." in text_head or "_entity." in text_head
    
    if has_data_block or has_loop or has_mmcif_keywords:
        result.content_type = "mmcif"
    elif text_head.strip() == "":
        result.content_type = "empty"
        result.status = "fail"
        result.fail_level = "L3_content"
        result.fail_reason = "decompressed_content_empty"
        return result
    else:
        # 检查是否为纯二进制垃圾
        printable_ratio = sum(1 for c in text_head[:1000] if c.isprintable() or c in '\n\r\t') / max(len(text_head[:1000]), 1)
        if printable_ratio < 0.8:
            result.content_type = "binary_garbage"
            result.status = "fail"
            result.fail_level = "L3_content"
            result.fail_reason = "not_text_content"
            result.error_detail = f"printable_ratio={printable_ratio:.2f}"
            return result
        else:
            result.content_type = "unknown_text"
            # 不立即判定失败，继续检查是否有atom_site
    
    # ----------------------------------------------------------
    # Level 4: 结构数据检查 (_atom_site)
    # ----------------------------------------------------------
    # 先在已读取的头部中快速检查
    has_atom_site_in_head = "_atom_site." in text_head
    
    if not has_atom_site_in_head:
        # 头部没有，需要完整扫描文件
        try:
            with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
                found = False
                for line in f:
                    if "_atom_site." in line:
                        found = True
                        break
                has_atom_site_in_head = found
        except Exception as e:
            result.status = "fail"
            result.fail_level = "L4_structure"
            result.fail_reason = "full_scan_read_error"
            result.error_detail = str(e)
            return result
    
    result.has_atom_site = has_atom_site_in_head
    
    if not result.has_atom_site:
        result.status = "fail"
        result.fail_level = "L4_structure"
        result.fail_reason = "no_atom_site_records"
        if result.content_type == "mmcif":
            result.error_detail = "valid_mmcif_but_no_coordinates"
        elif result.content_type == "unknown_text":
            # 给出文件开头帮助调试
            result.error_detail = f"first_200_chars: {text_head[:200].strip()}"
        return result
    
    # ----------------------------------------------------------
    # Level 5: 数据质量检查
    # ----------------------------------------------------------
    atom_count = 0
    chain_ids = set()
    has_protein = False
    
    try:
        with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as f:
            in_atom_site = False
            label_atom_id_col = None
            auth_asym_id_col = None
            group_pdb_col = None
            col_names = []
            
            for line in f:
                line_stripped = line.strip()
                
                # 进入 _atom_site loop
                if line_stripped.startswith("_atom_site."):
                    in_atom_site = True
                    col_name = line_stripped.split(".")[1].split()[0]
                    col_names.append(col_name)
                    
                    if col_name == "auth_asym_id":
                        auth_asym_id_col = len(col_names) - 1
                    elif col_name == "group_PDB":
                        group_pdb_col = len(col_names) - 1
                    continue
                
                if in_atom_site and not line_stripped.startswith("_") and line_stripped and not line_stripped.startswith("#"):
                    if line_stripped.startswith("loop_"):
                        # 新的loop块开始，atom_site结束
                        break
                    
                    # 这是一行坐标数据
                    atom_count += 1
                    
                    # 解析链ID和记录类型 (只在前1000行做，避免太慢)
                    if atom_count <= 1000:
                        parts = line_stripped.split()
                        
                        if auth_asym_id_col is not None and len(parts) > auth_asym_id_col:
                            chain_ids.add(parts[auth_asym_id_col])
                        
                        if group_pdb_col is not None and len(parts) > group_pdb_col:
                            if parts[group_pdb_col] == "ATOM":
                                has_protein = True
                    
                    # 如果不需要完整统计，提前退出
                    if not FULL_SCAN_ATOM_THRESHOLD and atom_count >= MIN_ATOM_LINES:
                        break
                
                elif in_atom_site and (line_stripped.startswith("#") or line_stripped == ""):
                    break
    
    except Exception as e:
        result.status = "fail"
        result.fail_level = "L5_quality"
        result.fail_reason = "atom_parsing_error"
        result.error_detail = f"{type(e).__name__}: {e}"
        return result
    
    result.atom_line_count = atom_count
    result.chain_ids = ",".join(sorted(chain_ids)) if chain_ids else ""
    result.has_protein_chain = has_protein
    
    if atom_count < MIN_ATOM_LINES:
        result.status = "fail"
        result.fail_level = "L5_quality"
        result.fail_reason = f"too_few_atom_lines_{atom_count}"
        return result
    
    if not has_protein and atom_count <= 1000:
        # 只有HETATM没有ATOM，说明可能只有配体没有蛋白质
        result.status = "fail"
        result.fail_level = "L5_quality"
        result.fail_reason = "no_protein_ATOM_records_only_HETATM"
        return result
    
    # 全部通过
    result.status = "pass"
    return result


# ============================================================
# 批量校验
# ============================================================
def find_all_pdb_files(pdb_dir: Path) -> List[Path]:
    """递归查找所有 .cif.gz 文件"""
    files = sorted(pdb_dir.rglob("*.cif.gz"))
    return files


def run_validation(pdb_dir: Path, output_dir: Path, sample_size: Optional[int] = None):
    """执行批量校验"""
    
    print("=" * 60)
    print("PDB文件完整性校验")
    print("=" * 60)
    print(f"PDB目录:   {pdb_dir}")
    print(f"输出目录:  {output_dir}")
    
    # 查找文件
    all_files = find_all_pdb_files(pdb_dir)
    print(f"发现文件:  {len(all_files)} 个 .cif.gz")
    
    if len(all_files) == 0:
        print("ERROR: 未找到任何 .cif.gz 文件")
        return
    
    # 抽样
    if sample_size and sample_size < len(all_files):
        import random
        random.seed(42)
        files_to_check = random.sample(all_files, sample_size)
        print(f"抽样校验:  {sample_size} 个文件")
    else:
        files_to_check = all_files
        print(f"全量校验:  {len(files_to_check)} 个文件")
    
    print()
    
    # 逐文件校验
    results = []
    fail_reasons = Counter()
    fail_levels = Counter()
    status_counts = Counter()
    
    start_time = time.time()
    
    for i, filepath in enumerate(files_to_check):
        result = validate_single_pdb(filepath)
        results.append(result)
        
        status_counts[result.status] += 1
        if result.status == "fail":
            fail_reasons[result.fail_reason] += 1
            fail_levels[result.fail_level] += 1
        
        # 进度报告
        if (i + 1) % 100 == 0 or (i + 1) == len(files_to_check):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            n_pass = status_counts.get("pass", 0)
            n_fail = status_counts.get("fail", 0)
            print(f"  [{i+1:>6}/{len(files_to_check)}] "
                  f"pass={n_pass} fail={n_fail} "
                  f"({rate:.0f} files/sec)")
    
    total_time = time.time() - start_time
    
    # ----------------------------------------------------------
    # 输出详细报告 (TSV)
    # ----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "validation_report.tsv"
    
    # TSV 列
    columns = [
        "pdb_id", "status", "fail_level", "fail_reason",
        "file_size_bytes", "decompressed_size_bytes",
        "is_gzip_valid", "content_type",
        "has_atom_site", "atom_line_count",
        "chain_ids", "has_protein_chain",
        "error_detail", "file_path",
    ]
    
    with open(report_file, "w") as f:
        f.write("\t".join(columns) + "\n")
        for r in results:
            row = [
                r.pdb_id, r.status, r.fail_level, r.fail_reason,
                str(r.file_size_bytes), str(r.decompressed_size_bytes),
                str(r.is_gzip_valid), r.content_type,
                str(r.has_atom_site), str(r.atom_line_count),
                r.chain_ids, str(r.has_protein_chain),
                r.error_detail.replace("\t", " ").replace("\n", " ")[:500],
                r.file_path,
            ]
            f.write("\t".join(row) + "\n")
    
    print(f"\n详细报告已保存: {report_file}")
    
    # ----------------------------------------------------------
    # 输出汇总 (JSON)
    # ----------------------------------------------------------
    summary = {
        "total_files_checked": len(files_to_check),
        "total_files_in_dir": len(all_files),
        "is_sample": sample_size is not None and sample_size < len(all_files),
        "pass": status_counts.get("pass", 0),
        "fail": status_counts.get("fail", 0),
        "pass_rate": f"{status_counts.get('pass', 0) / len(files_to_check) * 100:.1f}%",
        "elapsed_seconds": round(total_time, 1),
        "fail_by_level": dict(fail_levels.most_common()),
        "fail_by_reason": dict(fail_reasons.most_common()),
    }
    
    summary_file = output_dir / "validation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # ----------------------------------------------------------
    # 终端打印摘要
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("校验结果摘要")
    print("=" * 60)
    print(f"总文件数:    {len(files_to_check)}")
    print(f"通过:        {status_counts.get('pass', 0)} "
          f"({status_counts.get('pass', 0)/len(files_to_check)*100:.1f}%)")
    print(f"失败:        {status_counts.get('fail', 0)} "
          f"({status_counts.get('fail', 0)/len(files_to_check)*100:.1f}%)")
    print(f"耗时:        {total_time:.1f} 秒")
    
    if fail_levels:
        print(f"\n失败层级分布:")
        for level, count in fail_levels.most_common():
            pct = count / status_counts.get("fail", 1) * 100
            print(f"  {level:<25} {count:>6} ({pct:.1f}%)")
    
    if fail_reasons:
        print(f"\n失败原因 TOP 15:")
        for reason, count in fail_reasons.most_common(15):
            pct = count / status_counts.get("fail", 1) * 100
            print(f"  {reason:<45} {count:>6} ({pct:.1f}%)")
    
    # 给出失败文件的前几个示例
    failed = [r for r in results if r.status == "fail"]
    if failed:
        print(f"\n失败文件示例 (前5个):")
        for r in failed[:5]:
            print(f"  {r.pdb_id}: [{r.fail_level}] {r.fail_reason}")
            if r.error_detail:
                detail = r.error_detail[:120]
                print(f"    detail: {detail}")
    
    # ----------------------------------------------------------
    # 给出修复建议
    # ----------------------------------------------------------
    print()
    print("=" * 60)
    print("修复建议")
    print("=" * 60)
    
    if fail_reasons.get("html_error_page", 0) > 0:
        n = fail_reasons["html_error_page"]
        print(f"\n[HTML错误页: {n}个]")
        print("  原因: 下载时服务器返回了HTML错误页而非mmCIF文件")
        print("  常见触发: 403 Forbidden / 404 Not Found / 服务器限流")
        print("  修复: 删除这些文件后重新运行step1 (断点续传会自动重下)")
        print("  命令: grep 'html_error_page' validation_report.tsv | cut -f14 | xargs rm")
    
    if fail_reasons.get("bad_gzip_format", 0) + fail_reasons.get("gzip_truncated_eof", 0) > 0:
        n = fail_reasons.get("bad_gzip_format", 0) + fail_reasons.get("gzip_truncated_eof", 0)
        print(f"\n[Gzip损坏/截断: {n}个]")
        print("  原因: 下载过程中网络中断导致文件不完整")
        print("  修复: 删除后重下")
        print("  命令: grep -E 'bad_gzip|truncated' validation_report.tsv | cut -f14 | xargs rm")
    
    if fail_reasons.get("no_atom_site_records", 0) > 0:
        n = fail_reasons["no_atom_site_records"]
        print(f"\n[无坐标记录: {n}个]")
        print("  原因: mmCIF文件合法但不含原子坐标 (可能是已撤回的条目)")
        print("  修复: 从数据集中排除这些PDB ID")
    
    if fail_reasons.get("no_protein_ATOM_records_only_HETATM", 0) > 0:
        n = fail_reasons["no_protein_ATOM_records_only_HETATM"]
        print(f"\n[仅有配体无蛋白质: {n}个]")
        print("  原因: 该PDB条目可能是小分子/核酸结构，不含蛋白质链")
        print("  修复: 从数据集中排除这些PDB ID")
    
    if fail_reasons.get("file_too_small", 0) + fail_reasons.get("file_empty_0_bytes", 0) > 0:
        n = fail_reasons.get("file_too_small", 0) + fail_reasons.get("file_empty_0_bytes", 0)
        print(f"\n[空文件/极小文件: {n}个]")
        print("  原因: 下载失败但创建了空文件")
        print("  修复: 删除后重下，并检查download_log.json中这些ID的状态")
    
    # 生成自动清理脚本
    if failed:
        cleanup_script = output_dir / "cleanup_invalid_pdb.sh"
        with open(cleanup_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# 自动生成: 删除校验失败的PDB文件以便重新下载\n")
            f.write(f"# 失败文件数: {len(failed)}\n\n")
            
            # 按失败原因分组
            by_reason = defaultdict(list)
            for r in failed:
                by_reason[r.fail_reason].append(r.file_path)
            
            for reason, paths in sorted(by_reason.items()):
                f.write(f"\n# {reason} ({len(paths)} files)\n")
                for p in paths:
                    f.write(f"rm -f \"{p}\"\n")
            
            f.write(f"\necho '已删除 {len(failed)} 个无效文件，重新运行 step1 即可自动重下'\n")
        
        os.chmod(cleanup_script, 0o755)
        print(f"\n自动清理脚本: {cleanup_script}")
        print(f"  执行: bash {cleanup_script}")
        print(f"  然后: python step1_download_all_data.py  (断点续传)")
    
    print(f"\n完整报告: {report_file}")
    print(f"汇总统计: {summary_file}")


# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PDB mmCIF文件完整性校验")
    parser.add_argument("--pdb-dir", type=Path, default=DEFAULT_PDB_DIR,
                        help=f"PDB文件目录 (默认: {DEFAULT_PDB_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"报告输出目录 (默认: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--sample", type=int, default=None,
                        help="抽样数量 (默认: 全量校验)")
    args = parser.parse_args()
    
    if not args.pdb_dir.exists():
        print(f"ERROR: PDB目录不存在: {args.pdb_dir}")
        sys.exit(1)
    
    run_validation(args.pdb_dir, args.output_dir, args.sample)


if __name__ == "__main__":
    main()
