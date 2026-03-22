"""
patch_visualize_activation.py — 绘制特定 SAE feature 的逐氨基酸激活值
====================================================================

对指定的 UniProt ID 列表, 绘制某个 SAE feature 的 activation 曲线,
并用颜色标注生物特征为 positive 的区域, 用于人工校验 SAE 是否捕获了该特征。

输入:
  cusdata/esm_cache/...embeddings.h5   (ESM embedding 缓存)
  逐氨基酸特征文件 (如 cusdata/05_1_alphafold_rsa/rsa_per_residue.tsv)

输出:
  cusdata/eval_img/{feature_name}/AF-{uid}_f{feature_idx}_t{threshold}.png

用法:
  # 直接跑 (使用代码中的默认值, 改 DEFAULT_* 常量即可调试)
  python patch_visualize_activation.py

  # 命令行覆盖
  python patch_visualize_activation.py \\
    --uniprot-ids Q15399 O60603 \\
    --feature-idx 3842 \\
    --feature-name kd_hydrophobic \\
    --feature-file cusdata/04_kd/kd_per_residue.tsv \\
    --feature-col kd_value \\
    --feature-threshold 1.8 \\
    --positive-above

  python patch_visualize_activation.py \\
    --uniprot-ids Q15399 \\
    --feature-idx 1205 \\
    --feature-name rsa_buried \\
    --feature-file cusdata/05_1_alphafold_rsa/rsa_per_residue.tsv \\
    --feature-col rsa \\
    --feature-threshold 0.25 \\
    --no-positive-above
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib 未安装 → pip install matplotlib")
    sys.exit(1)

try:
    import torch
    import h5py
except ImportError as e:
    print(f"ERROR: {e} → pip install torch h5py")
    sys.exit(1)

try:
    from interplm.sae.inference import load_sae_from_hf
except ImportError:
    print("ERROR: interplm 未安装 → cd interPLM && pip install -e .")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
EMB_CACHE = Path("cusdata/esm_cache")
OUTPUT_BASE = Path("cusdata/eval_img")

DEFAULT_PLM_MODEL = "esm2-8m"
DEFAULT_PLM_LAYER = 4
DEFAULT_SAE_THRESHOLD = 0.5

MODEL_MAP = {
    "esm2-8m": "t6_8M",
    "esm2-650m": "t33_650M",
}


# ============================================================
# 日志
# ============================================================
class Logger:
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
        print(f"  [{ts}] {step}: {status}" + (f" | {detail}" if detail else ""))


# ============================================================
# 读取单个蛋白的逐氨基酸特征
# ============================================================
def load_feature_for_uid(
    feature_file: Path,
    uid: str,
    feature_col_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    返回 (positions, values) 或 None (找不到)
      positions: 0-based int array
      values:    float array
    """
    positions = []
    values = []

    with open(feature_file) as f:
        header = f.readline().strip().split("\t")
        if feature_col_name not in header:
            return None
        val_col = header.index(feature_col_name)
        pos_col = header.index("position") if "position" in header else 2

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(val_col, pos_col):
                continue
            if parts[0] != uid:
                continue
            positions.append(int(parts[pos_col]))
            values.append(float(parts[val_col]))

    if not positions:
        return None
    return np.array(positions, dtype=np.int64), np.array(values, dtype=np.float64)


# ============================================================
# 绘图
# ============================================================
def plot_activation(
    uid: str,
    activations: np.ndarray,
    feature_idx: int,
    sae_threshold: float,
    positions: Optional[np.ndarray],
    feature_positive: Optional[np.ndarray],
    feature_name: str,
    output_path: Path,
):
    """
    绘制单个蛋白的 SAE activation 曲线

    参数:
      activations:    shape (seq_len,) 单个 feature 的 activation 值
      feature_positive: shape (seq_len,) bool, True=该位置生物特征为positive
                        None=无特征数据, 全部用默认颜色
    """
    seq_len = len(activations)
    x = np.arange(1, seq_len + 1)  # 1-based

    fig, ax = plt.subplots(figsize=(max(14, seq_len / 50), 5))

    # 默认颜色
    default_color = "#4C72B0"
    positive_color = "#E8790C"  # 加州橙

    if feature_positive is not None and len(feature_positive) == seq_len:
        # 有特征数据: positive 区域用橙色, 其余蓝色
        # 画线: 先画全部蓝色底线, 再覆盖橙色段
        ax.plot(x, activations, color=default_color, linewidth=0.6, alpha=0.5, zorder=1)

        # positive 点
        pos_mask = feature_positive
        neg_mask = ~feature_positive
        ax.scatter(x[neg_mask], activations[neg_mask],
                   s=3, color=default_color, alpha=0.6, zorder=2, label="negative")
        ax.scatter(x[pos_mask], activations[pos_mask],
                   s=5, color=positive_color, alpha=0.8, zorder=3, label="positive")

        # 橙色线段: 连接相邻positive点
        pos_indices = np.where(pos_mask)[0]
        for i in range(len(pos_indices) - 1):
            if pos_indices[i + 1] - pos_indices[i] == 1:
                i0, i1 = pos_indices[i], pos_indices[i + 1]
                ax.plot(x[i0:i1 + 1], activations[i0:i1 + 1],
                        color=positive_color, linewidth=1.0, alpha=0.8, zorder=2)
    else:
        ax.plot(x, activations, color=default_color, linewidth=0.8)
        ax.scatter(x, activations, s=3, color=default_color, alpha=0.6)

    # activation 阈值线
    ax.axhline(y=sae_threshold, color="red", linestyle="--", linewidth=0.8,
               label=f"SAE threshold={sae_threshold}", zorder=4)

    ax.set_xlabel("Residue Position (1-based)", fontsize=11)
    ax.set_ylabel(f"SAE Feature {feature_idx} Activation", fontsize=11)
    ax.set_title(f"{uid} — feature {feature_idx} ({feature_name})", fontsize=13)
    ax.set_xlim(0.5, seq_len + 0.5)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 调试用默认值 (直接改这里, 省去命令行传参)
# ============================================================
DEFAULT_UNIPROT_IDS = ["O60603"]
# kd
# DEFAULT_FEATURE_IDX = 9740
# DEFAULT_FEATURE_NAME = "kd_hydrophobic"
# DEFAULT_FEATURE_FILE = "cusdata/04_kd/kd_per_residue.tsv"
# DEFAULT_FEATURE_COL = "kd_value"
# DEFAULT_FEATURE_THRESHOLD = 1.8
# DEFAULT_POSITIVE_ABOVE = True
# rsa
DEFAULT_FEATURE_IDX = 9740
DEFAULT_FEATURE_NAME = "rsa_hydrophobic"
DEFAULT_FEATURE_FILE = "cusdata/05_1_alphafold_rsa/rsa_per_residue.tsv"
DEFAULT_FEATURE_COL = "rsa"
DEFAULT_FEATURE_THRESHOLD = 0.25
DEFAULT_POSITIVE_ABOVE = False


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="绘制 SAE feature activation 逐残基可视化")

    parser.add_argument("--uniprot-ids", nargs="+", default=DEFAULT_UNIPROT_IDS,
                        help=f"UniProt ID 列表 (默认: {DEFAULT_UNIPROT_IDS})")
    parser.add_argument("--feature-idx", type=int, default=DEFAULT_FEATURE_IDX,
                        help=f"SAE feature 序号 (默认: {DEFAULT_FEATURE_IDX})")

    parser.add_argument("--feature-name", type=str, default=DEFAULT_FEATURE_NAME,
                        help=f"特征名/输出目录 (默认: {DEFAULT_FEATURE_NAME})")
    parser.add_argument("--feature-file", type=str, default=DEFAULT_FEATURE_FILE,
                        help=f"逐氨基酸特征 TSV (默认: {DEFAULT_FEATURE_FILE})")
    parser.add_argument("--feature-col", type=str, default=DEFAULT_FEATURE_COL,
                        help=f"特征值列名 (默认: {DEFAULT_FEATURE_COL})")
    parser.add_argument("--feature-threshold", type=float, default=DEFAULT_FEATURE_THRESHOLD,
                        help=f"特征二值化阈值 (默认: {DEFAULT_FEATURE_THRESHOLD})")
    parser.add_argument("--positive-above", action="store_true",
                        default=DEFAULT_POSITIVE_ABOVE,
                        help="value >= threshold → positive (默认)" if DEFAULT_POSITIVE_ABOVE else argparse.SUPPRESS)
    parser.add_argument("--no-positive-above", dest="positive_above",
                        action="store_false",
                        help="value < threshold → positive")

    # SAE / PLM
    parser.add_argument("--sae-threshold", type=float, default=DEFAULT_SAE_THRESHOLD,
                        help=f"SAE activation 阈值 (默认: {DEFAULT_SAE_THRESHOLD})")
    parser.add_argument("--plm-model", type=str, default=DEFAULT_PLM_MODEL,
                        help=f"PLM 模型 (默认: {DEFAULT_PLM_MODEL})")
    parser.add_argument("--plm-layer", type=int, default=DEFAULT_PLM_LAYER,
                        help=f"PLM 层号 (默认: {DEFAULT_PLM_LAYER})")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    feature_file = Path(args.feature_file)
    cache_short = MODEL_MAP.get(args.plm_model, args.plm_model)
    h5_path = EMB_CACHE / cache_short / f"layer_{args.plm_layer}" / "embeddings.h5"
    out_dir = OUTPUT_BASE / args.feature_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "visualize_log.tsv")

    print("=" * 60)
    print("SAE Feature Activation Visualization")
    print("=" * 60)
    print(f"  uniprot_ids:    {args.uniprot_ids}")
    print(f"  feature_idx:    {args.feature_idx}")
    print(f"  feature_name:   {args.feature_name}")
    print(f"  feature_file:   {feature_file}")
    print(f"  feature_col:    {args.feature_col}")
    print(f"  feature_thresh: {args.feature_threshold} "
          f"({'>=pos' if args.positive_above else '<pos'})")
    print(f"  sae_threshold:  {args.sae_threshold}")
    print(f"  plm_model:      {args.plm_model}")
    print(f"  embedding:      {h5_path}")
    print(f"  output:         {out_dir}")

    logger.log("main", "start",
               f"uids={args.uniprot_ids}, feature_idx={args.feature_idx}, "
               f"feature_name={args.feature_name}, "
               f"feature_col={args.feature_col}, "
               f"threshold={args.feature_threshold}, "
               f"positive_above={args.positive_above}, "
               f"sae_threshold={args.sae_threshold}")

    # 检查输入
    if not h5_path.exists():
        logger.log("main", "error", f"embedding cache 不存在: {h5_path}")
        return
    if not feature_file.exists():
        logger.log("main", "error", f"feature file 不存在: {feature_file}")
        return

    # 加载 SAE
    print(f"\n加载 SAE (model={args.plm_model}, layer={args.plm_layer})...")
    sae = load_sae_from_hf(
        plm_model=args.plm_model, plm_layer=args.plm_layer
    ).to(device)
    sae.eval()
    logger.log("sae", "loaded", f"model={args.plm_model}, layer={args.plm_layer}")

    # 逐蛋白处理
    n_ok = 0
    n_fail = 0

    for uid in args.uniprot_ids:
        print(f"\n--- {uid} ---")

        # 1. 获取 embedding
        try:
            with h5py.File(h5_path, "r") as h5f:
                if uid not in h5f:
                    logger.log("embed", "not_found", f"uid={uid}")
                    n_fail += 1
                    continue
                emb = h5f[uid][:]  # (seq_len, dim)
        except Exception as e:
            logger.log("embed", "error", f"uid={uid}, {type(e).__name__}: {e}")
            n_fail += 1
            continue

        # 2. SAE encode
        try:
            emb_tensor = torch.from_numpy(emb).float().to(device)
            with torch.no_grad():
                all_act = sae.encode(emb_tensor)  # (seq_len, n_features)
            all_act_np = all_act.cpu().numpy()
        except Exception as e:
            logger.log("sae_encode", "error", f"uid={uid}, {type(e).__name__}: {e}")
            n_fail += 1
            continue

        if args.feature_idx >= all_act_np.shape[1]:
            logger.log("sae_encode", "error",
                       f"uid={uid}, feature_idx={args.feature_idx} >= n_features={all_act_np.shape[1]}")
            n_fail += 1
            continue

        activations = all_act_np[:, args.feature_idx]  # (seq_len,)
        seq_len = len(activations)
        logger.log("sae_encode", "ok", f"uid={uid}, seq_len={seq_len}")

        # 3. 读取生物特征
        feature_positive = None
        feat_result = load_feature_for_uid(feature_file, uid, args.feature_col)

        if feat_result is None:
            logger.log("feature", "not_found",
                       f"uid={uid}, file={feature_file}, col={args.feature_col}")
        else:
            positions, values = feat_result
            # 构建 seq_len 长度的 bool 数组
            feature_positive = np.zeros(seq_len, dtype=bool)
            for pos, val in zip(positions, values):
                if 0 <= pos < seq_len:
                    if args.positive_above:
                        feature_positive[pos] = (val >= args.feature_threshold)
                    else:
                        feature_positive[pos] = (val < args.feature_threshold)

            n_pos = feature_positive.sum()
            logger.log("feature", "ok",
                       f"uid={uid}, residues={len(positions)}, "
                       f"positive={n_pos}/{seq_len}")

        # 4. 绘图
        fname = (f"{uid}_f{args.feature_idx}"
                 f"_st{args.sae_threshold}"
                 f"_ft{args.feature_threshold}.png")
        out_path = out_dir / fname

        try:
            plot_activation(
                uid, activations, args.feature_idx,
                args.sae_threshold,
                None if feat_result is None else positions,
                feature_positive, args.feature_name, out_path)
            logger.log("plot", "saved", f"uid={uid}, file={out_path}")
            print(f"  保存: {out_path}")
            n_ok += 1
        except Exception as e:
            logger.log("plot", "error", f"uid={uid}, {type(e).__name__}: {e}")
            n_fail += 1

    # 摘要
    logger.log("main", "done", f"ok={n_ok}, fail={n_fail}")
    print(f"\n完成: {n_ok} 成功, {n_fail} 失败")
    print(f"输出: {out_dir}")


if __name__ == "__main__":
    main()