"""
patch_esm_embeddings.py — 用 InterPLM 生成 ESM-2 per-residue embedding 缓存
=============================================================================

依赖 InterPLM (https://github.com/ElanaPearl/InterPLM):
  git clone https://github.com/ElanaPearl/interPLM.git
  cd interPLM && pip install -e .

直接调用 interplm.embedders.esm.ESMEmbedder 的 extract_embeddings 方法,
逐 batch 提取指定层的 per-residue embedding, 以 uniprot_id 为 key
保存为 HDF5 缓存文件。

ESM-2 模型选项 (model_name 不含 "facebook/" 前缀):
  esm2_t6_8M_UR50D      6层,  dim=320   (~30MB)
  esm2_t12_35M_UR50D    12层, dim=480   (~135MB)
  esm2_t30_150M_UR50D   30层, dim=640   (~575MB)
  esm2_t33_650M_UR50D   33层, dim=1280  (~2.5GB)

输入:
  cusdata/01_raw/uniprot_pdb_sequences.fasta

输出:
  cusdata/esm_cache/{model_short}/layer_{N}/embeddings.h5
  cusdata/esm_cache/{model_short}/layer_{N}/meta.json

用法:
  python patch_esm_embeddings.py
  python patch_esm_embeddings.py --model esm2_t6_8M_UR50D --layer 4 --batch-size 32
  python patch_esm_embeddings.py --model esm2_t33_650M_UR50D --layer 24 --batch-size 4
"""

import sys
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

try:
    from Bio import SeqIO
except ImportError:
    print("ERROR: biopython 未安装 → pip install biopython")
    sys.exit(1)

try:
    import numpy as np
    import h5py
except ImportError as e:
    print(f"ERROR: 缺少依赖: {e}")
    print("安装: pip install h5py numpy")
    sys.exit(1)

try:
    from interplm.embedders.esm import ESM as ESMEmbedder
except ImportError:
    print("ERROR: interplm 未安装")
    print("安装: git clone https://github.com/ElanaPearl/interPLM.git")
    print("      cd interPLM && pip install -e .")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
INPUT_FASTA = Path("cusdata/01_raw/uniprot_pdb_sequences.fasta")
OUTPUT_BASE = Path("cusdata/esm_cache")

DEFAULT_MODEL = "esm2_t6_8M_UR50D"
DEFAULT_LAYER = 4
DEFAULT_BATCH_SIZE = 32

ESM2_MAX_SEQ_LEN = 1022  # ESM-2 最大残基数


# ============================================================
# 读取 FASTA
# ============================================================
def read_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    """返回 [(uniprot_id, sequence), ...]"""
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
    parser = argparse.ArgumentParser(
        description="用 InterPLM ESMEmbedder 生成 per-residue embedding 缓存")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"ESM-2模型名 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                        help=f"提取的层号 (默认: {DEFAULT_LAYER})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"batch大小 (默认: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--fasta", type=str, default=str(INPUT_FASTA),
                        help=f"输入FASTA (默认: {INPUT_FASTA})")
    args = parser.parse_args()

    # 输出路径
    model_short = args.model.replace("esm2_", "").replace("_UR50D", "")
    output_dir = OUTPUT_BASE / model_short / f"layer_{args.layer}"
    output_h5 = output_dir / "embeddings.h5"
    meta_file = output_dir / "meta.json"

    print("=" * 60)
    print("ESM-2 Per-Residue Embedding Cache (InterPLM)")
    print("=" * 60)
    print(f"  模型:      {args.model}")
    print(f"  层:        {args.layer}")
    print(f"  batch:     {args.batch_size}")
    print(f"  输入:      {args.fasta}")
    print(f"  输出:      {output_h5}")

    # 检查输入
    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"\nERROR: FASTA不存在: {fasta_path}")
        return

    # 读取序列
    print(f"\n读取FASTA...")
    all_seqs = read_fasta(fasta_path)
    print(f"  序列数: {len(all_seqs)}")

    # 断点恢复
    output_dir.mkdir(parents=True, exist_ok=True)
    done_uids = set()
    if output_h5.exists():
        with h5py.File(output_h5, "r") as f:
            done_uids = set(f.keys())
        print(f"  断点恢复: {len(done_uids)} 已完成")

    todo_seqs = [(uid, seq) for uid, seq in all_seqs if uid not in done_uids]
    print(f"  待处理: {len(todo_seqs)}")

    if not todo_seqs:
        print("\n全部已完成!")
        _print_summary(output_h5, meta_file)
        return

    # 初始化 InterPLM ESMEmbedder
    # ESMEmbedder 的接口可能接受 layer 在构造函数或在 extract_embeddings 中
    print(f"\n初始化 ESMEmbedder (model={args.model}, layer={args.layer})...")
    try:
        embedder = ESMEmbedder(model_name=args.model, layer=args.layer)
        _layer_in_init = True
    except TypeError:
        embedder = ESMEmbedder(model_name=args.model)
        _layer_in_init = False
        print(f"  (layer 将在 extract_embeddings 调用时传入)")

    def _call_extract(seqs):
        """封装 extract_embeddings 调用, 兼容不同签名"""
        if _layer_in_init:
            return embedder.extract_embeddings(seqs)
        else:
            return embedder.extract_embeddings(seqs, layer=args.layer)

    # 批量处理
    print(f"\n开始提取...\n")
    t0 = time.time()
    n_done = 0
    n_truncated = 0
    total_residues = 0

    def _to_numpy(x):
        """将 tensor 或 ndarray 统一转为 float32 numpy"""
        if hasattr(x, 'cpu'):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float32)

    def _split_by_lengths(raw, batch_seqs):
        """
        extract_embeddings 返回 (total_residues, dim) 的拼接矩阵,
        按每条序列的长度切分为 list of (seq_len_i, dim)
        """
        arr = _to_numpy(raw)
        results = []
        offset = 0
        for seq in batch_seqs:
            slen = len(seq)
            results.append(arr[offset:offset + slen, :].copy())
            offset += slen
        assert offset == arr.shape[0], \
            f"长度不匹配: sum(seq_lens)={offset} != embedding.shape[0]={arr.shape[0]}"
        return results

    with h5py.File(output_h5, "a") as h5f:
        for batch_start in range(0, len(todo_seqs), args.batch_size):
            batch = todo_seqs[batch_start:batch_start + args.batch_size]

            batch_uids = []
            batch_seqs = []
            for uid, seq in batch:
                if len(seq) > ESM2_MAX_SEQ_LEN:
                    seq = seq[:ESM2_MAX_SEQ_LEN]
                    n_truncated += 1
                batch_uids.append(uid)
                batch_seqs.append(seq)

            # 调用 InterPLM extract_embeddings
            # 返回 (total_residues, dim) 拼接矩阵, 需按序列长度切分
            try:
                raw = _call_extract(batch_seqs)
                embeddings = _split_by_lengths(raw, batch_seqs)
            except Exception as e:
                print(f"  ERROR batch {batch_start}: {e}")
                # 逐个重试
                for uid, seq in zip(batch_uids, batch_seqs):
                    try:
                        raw_single = _call_extract([seq])
                        emb = _split_by_lengths(raw_single, [seq])[0]
                        if uid not in h5f:
                            h5f.create_dataset(
                                uid, data=emb,
                                compression="gzip", compression_opts=4)
                        total_residues += emb.shape[0]
                        n_done += 1
                    except Exception as e2:
                        print(f"  SKIP {uid}: {e2}")
                continue

            # 保存
            for uid, emb in zip(batch_uids, embeddings):
                if uid not in h5f:
                    h5f.create_dataset(
                        uid, data=emb,
                        compression="gzip", compression_opts=4)
                total_residues += emb.shape[0]
                n_done += 1

            # 进度
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  [{n_done:>6}/{len(todo_seqs)}] "
                  f"residues={total_residues} "
                  f"truncated={n_truncated} "
                  f"({rate:.1f} prot/s)")

    elapsed_total = time.time() - t0

    # 元数据
    embed_dim = 0
    if output_h5.exists():
        with h5py.File(output_h5, "r") as f:
            for key in f:
                embed_dim = f[key].shape[1]  # (seq_len, dim)
                break

    meta = {
        "model": args.model,
        "layer": args.layer,
        "embed_dim": embed_dim,
        "batch_size": args.batch_size,
        "fasta": str(args.fasta),
        "max_seq_len": ESM2_MAX_SEQ_LEN,
        "n_total_sequences": len(all_seqs),
        "n_embedded": len(done_uids) + n_done,
        "n_truncated": n_truncated,
        "total_residues": total_residues,
        "elapsed_seconds": round(elapsed_total, 1),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    _print_summary(output_h5, meta_file)


def _print_summary(output_h5: Path, meta_file: Path):
    print()
    print("=" * 60)
    print("概览")
    print("=" * 60)

    meta = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"  模型:        {meta.get('model')}")
        print(f"  层:          {meta.get('layer')}")
        print(f"  维度:        {meta.get('embed_dim')}")
        print(f"  序列总数:    {meta.get('n_total_sequences')}")
        print(f"  已嵌入:      {meta.get('n_embedded')}")
        print(f"  截断:        {meta.get('n_truncated')}")

    if output_h5.exists():
        sz_gb = output_h5.stat().st_size / (1024**3)
        with h5py.File(output_h5, "r") as f:
            n_keys = len(f.keys())
        print(f"  缓存:        {output_h5} ({sz_gb:.2f} GB, {n_keys} proteins)")

    embed_dim = meta.get("embed_dim", "D")
    print(f"\n读取方式:")
    print(f"  import h5py")
    print(f'  with h5py.File("{output_h5}", "r") as f:')
    print(f'      emb = f["P12345"][:]  # (seq_len, {embed_dim})')


if __name__ == "__main__":
    main()