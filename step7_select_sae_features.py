"""
Step 7: 筛选最能表征生物特征的 SAE activation 序号
===================================================

用 refer 集评估每个 SAE feature 与生物特征标签的 F1 分数,
选出 top-N 最佳 feature, 在 validation 集上验证, 并计算累计 F1。

支持的特征源 (通过 --feature-source 指定):
  kd:  KD疏水性 (value_col=3, >=1.8 → positive)
  rsa: 溶剂可及性 (value_col=5, <0.25 → positive/buried)

流程:
  1. 读取 refer 集标注, 按阈值二值化
  2. 逐蛋白 ESM embedding → SAE encode → 按阈值二值化
  3. 逐残基对比, 累积 TP/FP/FN
  4. 计算每个 SAE feature 的 F1, 选出 top-N
  5. 在 validation 集上计算 top-N features 的 F1
  6. 计算 top-k 累计 F1 (前k个feature OR联合预测)

输出:
  cusdata/07_sae_features/{feature_name}/
  ├── refer_f1_all.tsv              ← 全部 feature 的 F1 (按 feature_idx 顺序)
  ├── refer_f1_sorted.tsv           ← 全部 feature 的 F1 (按 F1 降序)
  ├── refer_f1_distribution.png     ← F1 分布直方图
  ├── top_features.tsv              ← top-N feature 序号及 F1
  ├── val_f1_top.tsv                ← top-N feature 在 val 集的单独 F1
  ├── val_cumulative_f1.tsv         ← top-k feature OR联合的累计 F1
  └── step7_log.tsv                 ← 全流程日志

用法:
  python step7_select_sae_features.py --feature-source kd --feature-name kd_hydrophobic
  python step7_select_sae_features.py --feature-source rsa --feature-name rsa_buried
  python step7_select_sae_features.py --feature-source kd --feature-threshold 2.5 --top-n 50
"""

import sys
import time
import argparse
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端
    import matplotlib.pyplot as plt
except ImportError:
    print("WARNING: matplotlib 未安装, 将跳过直方图生成")
    plt = None

try:
    import torch
    import h5py
except ImportError as e:
    print(f"ERROR: {e}")
    print("安装: pip install torch h5py")
    sys.exit(1)

try:
    from interplm.sae.inference import load_sae_from_hf
except ImportError:
    print("ERROR: interplm 未安装")
    print("安装: cd interPLM && pip install -e .")
    sys.exit(1)


# ============================================================
# 配置
# ============================================================
EMB_CACHE = Path("cusdata/esm_cache")
OUTPUT_BASE = Path("cusdata/07_sae_features")

# ---- 特征注册表 ----
# refer/val路径, 值所在列号(0-based), 默认阈值, 正例方向
FEATURE_REGISTRY = {
    "kd": {
        "refer": Path("cusdata/06_splits/kd/refer.tsv"),
        "val":   Path("cusdata/06_splits/kd/val.tsv"),
        "value_col": 3,           # kd_value
        "default_threshold": 1.8, # KD >= 1.8 → hydrophobic
        "positive_above": True,   # >= threshold → positive
    },
    "rsa": {
        "refer": Path("cusdata/06_splits/rsa/refer.tsv"),
        "val":   Path("cusdata/06_splits/rsa/val.tsv"),
        "value_col": 5,           # rsa
        "default_threshold": 0.25,# RSA < 0.25 → buried
        "positive_above": False,  # < threshold → positive
    },
}

# 默认超参
DEFAULT_TOP_N = 20
DEFAULT_SAE_THRESHOLD = 0.5      # SAE activation >= 0.5 → active
DEFAULT_PLM_MODEL = "esm2-8m"
DEFAULT_PLM_LAYER = 4
DEFAULT_BATCH_SIZE = 64
DEFAULT_FEATURE_NAME = "kd_hydrophobic"
DEFAULT_FEATURE_SOURCE = "kd"    # FEATURE_REGISTRY 中的 key


# ============================================================
# 日志
# ============================================================
class StepLogger:
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
# 1. 读取特征标注, 按 uniprot_id 分组, 二值化
# ============================================================
def load_feature_data(
    data_path: Path,
    value_col: int,
    threshold: float,
    positive_above: bool,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    返回 {uniprot_id: (positions, binary_labels)}
      positions:     np.array of int, 0-based 残基位置 (列2)
      binary_labels: np.array of int (0 or 1)
        positive_above=True:  value >= threshold → 1
        positive_above=False: value < threshold  → 1
    """
    data = defaultdict(lambda: ([], []))

    with open(data_path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= value_col:
                continue
            uid = parts[0]
            pos = int(parts[2])
            val = float(parts[value_col])
            if positive_above:
                label = 1 if val >= threshold else 0
            else:
                label = 1 if val < threshold else 0
            data[uid][0].append(pos)
            data[uid][1].append(label)

    result = {}
    for uid, (positions, labels) in data.items():
        result[uid] = (np.array(positions, dtype=np.int64),
                       np.array(labels, dtype=np.int64))
    return result


# ============================================================
# 2. 逐蛋白评估: 累积 TP/FP/FN
# ============================================================
def evaluate_on_dataset(
    kd_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    h5_path: Path,
    sae,
    sae_threshold: float,
    device: str,
    n_features: int,
    logger: StepLogger,
    dataset_name: str,
    feature_subset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    遍历 dataset 中每个蛋白:
      1. 从 h5 取 embedding
      2. SAE encode → 二值化
      3. 与 KD binary label 对比, 累积 TP/FP/FN

    参数:
      feature_subset: 如果指定, 只评估这些 feature 序号 (用于 val 集)

    返回:
      (tp, fp, fn) 各为 shape (n_eval_features,) 的 int64 数组
    """
    if feature_subset is not None:
        n_eval = len(feature_subset)
    else:
        n_eval = n_features

    tp = np.zeros(n_eval, dtype=np.int64)
    fp = np.zeros(n_eval, dtype=np.int64)
    fn = np.zeros(n_eval, dtype=np.int64)

    n_proteins = 0
    n_skipped = 0
    n_residues_total = 0

    with h5py.File(h5_path, "r") as h5f:
        sorted_uids = sorted(kd_data.keys())

        for uid in sorted_uids:
            if uid not in h5f:
                n_skipped += 1
                continue

            positions, kd_labels = kd_data[uid]
            emb = h5f[uid][:]  # (seq_len, embed_dim)

            # 检查位置范围
            valid_mask = positions < emb.shape[0]
            if not valid_mask.all():
                positions = positions[valid_mask]
                kd_labels = kd_labels[valid_mask]

            if len(positions) == 0:
                n_skipped += 1
                continue

            # SAE encode
            emb_tensor = torch.from_numpy(emb).float().to(device)
            with torch.no_grad():
                activations = sae.encode(emb_tensor)  # (seq_len, n_features)
            act_np = activations.cpu().numpy()

            # 取对应位置的 activation
            act_at_pos = act_np[positions, :]  # (n_residues, n_features)

            # 如果只评估子集
            if feature_subset is not None:
                act_at_pos = act_at_pos[:, feature_subset]

            # 二值化 SAE activation
            pred = (act_at_pos >= sae_threshold).astype(np.int64)  # (n_residues, n_eval)
            truth = kd_labels[:, np.newaxis]  # (n_residues, 1) broadcast

            # 累积
            tp += ((pred == 1) & (truth == 1)).sum(axis=0)
            fp += ((pred == 1) & (truth == 0)).sum(axis=0)
            fn += ((pred == 0) & (truth == 1)).sum(axis=0)

            n_proteins += 1
            n_residues_total += len(positions)

            if n_proteins % 500 == 0:
                logger.log(f"7_{dataset_name}", "progress",
                           f"proteins={n_proteins}, residues={n_residues_total}")

    logger.log(f"7_{dataset_name}", "done",
               f"proteins={n_proteins}, skipped={n_skipped}, "
               f"residues={n_residues_total}")

    return tp, fp, fn


# ============================================================
# 3. 从 TP/FP/FN 计算 F1
# ============================================================
def compute_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (f1, precision, recall) 各为 float64 数组
    """
    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)

    precision = np.zeros_like(tp)
    recall = np.zeros_like(tp)
    f1 = np.zeros_like(tp)

    p_denom = tp + fp
    r_denom = tp + fn
    np.divide(tp, p_denom, out=precision, where=p_denom > 0)
    np.divide(tp, r_denom, out=recall, where=r_denom > 0)

    f1_denom = precision + recall
    np.divide(2 * precision * recall, f1_denom, out=f1, where=f1_denom > 0)

    return f1, precision, recall


# ============================================================
# 4. 累计 F1: top-k features 联合评估 (OR逻辑)
# ============================================================
def evaluate_cumulative(
    feature_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    h5_path: Path,
    sae,
    sae_threshold: float,
    device: str,
    top_indices: np.ndarray,
    logger: StepLogger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 top_indices 中前 k 个 feature 联合评估 (k=1,2,...,len(top_indices)):
      只要其中任意一个 feature 激活 → 预测为 positive (OR逻辑)

    返回:
      (cum_tp, cum_fp, cum_fn) 各为 shape (len(top_indices),) 的 int64 数组
      cum_tp[k] = 使用前 k+1 个 feature 联合预测时的 TP
    """
    n_ranks = len(top_indices)
    cum_tp = np.zeros(n_ranks, dtype=np.int64)
    cum_fp = np.zeros(n_ranks, dtype=np.int64)
    cum_fn = np.zeros(n_ranks, dtype=np.int64)

    with h5py.File(h5_path, "r") as h5f:
        for uid in sorted(feature_data.keys()):
            if uid not in h5f:
                continue

            positions, labels = feature_data[uid]
            emb = h5f[uid][:]

            valid_mask = positions < emb.shape[0]
            if not valid_mask.all():
                positions = positions[valid_mask]
                labels = labels[valid_mask]
            if len(positions) == 0:
                continue

            emb_tensor = torch.from_numpy(emb).float().to(device)
            with torch.no_grad():
                activations = sae.encode(emb_tensor)
            act_np = activations.cpu().numpy()

            # 取 top features 在对应位置的激活值: (n_residues, n_ranks)
            act_top = act_np[positions, :][:, top_indices]
            pred_top = (act_top >= sae_threshold)  # bool (n_residues, n_ranks)
            truth = labels.astype(bool)             # bool (n_residues,)

            # 逐 rank 累计: rank k 使用前 k+1 个 feature 的 OR
            for k in range(n_ranks):
                pred_or = pred_top[:, :k + 1].any(axis=1)  # (n_residues,)
                cum_tp[k] += ((pred_or) & (truth)).sum()
                cum_fp[k] += ((pred_or) & (~truth)).sum()
                cum_fn[k] += ((~pred_or) & (truth)).sum()

    logger.log("7.6_cumulative", "done",
               f"n_ranks={n_ranks}, "
               f"top1_f1_tp={cum_tp[0]}, topN_tp={cum_tp[-1]}")

    return cum_tp, cum_fp, cum_fn


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step 7: 筛选最佳 SAE features for KD hydrophobicity")
    parser.add_argument("--feature-name", type=str, default=DEFAULT_FEATURE_NAME,
                        help=f"输出子目录名 (默认: {DEFAULT_FEATURE_NAME})")
    parser.add_argument("--feature-source", type=str, default=DEFAULT_FEATURE_SOURCE,
                        choices=list(FEATURE_REGISTRY.keys()),
                        help=f"特征数据源 (默认: {DEFAULT_FEATURE_SOURCE})")
    parser.add_argument("--feature-threshold", type=float, default=None,
                        help="特征二值化阈值 (默认: 从FEATURE_REGISTRY读取)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"选出的 feature 数量 (默认: {DEFAULT_TOP_N})")
    parser.add_argument("--sae-threshold", type=float, default=DEFAULT_SAE_THRESHOLD,
                        help=f"SAE activation 二值化阈值 (默认: {DEFAULT_SAE_THRESHOLD})")
    parser.add_argument("--plm-model", type=str, default=DEFAULT_PLM_MODEL,
                        help=f"InterPLM 模型名 (默认: {DEFAULT_PLM_MODEL})")
    parser.add_argument("--plm-layer", type=int, default=DEFAULT_PLM_LAYER,
                        help=f"PLM 层号 (默认: {DEFAULT_PLM_LAYER})")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (默认: 自动)")
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 从注册表解析特征配置
    feat_cfg = FEATURE_REGISTRY[args.feature_source]
    feat_refer = feat_cfg["refer"]
    feat_val = feat_cfg["val"]
    feat_value_col = feat_cfg["value_col"]
    feat_positive_above = feat_cfg["positive_above"]
    feat_threshold = (args.feature_threshold if args.feature_threshold is not None
                      else feat_cfg["default_threshold"])

    # 输出目录: cusdata/07_sae_features/{feature_name}/
    OUTPUT_DIR = OUTPUT_BASE / args.feature_name

    # 推断 embedding 缓存路径
    model_map = {
        "esm2-8m": "t6_8M",
        "esm2-650m": "t33_650M",
    }
    cache_short = model_map.get(args.plm_model, args.plm_model)
    h5_path = EMB_CACHE / cache_short / f"layer_{args.plm_layer}" / "embeddings.h5"

    print("=" * 60)
    print("Step 7: SAE Feature Selection")
    print("=" * 60)
    print(f"  feature_name:   {args.feature_name}")
    print(f"  feature_source: {args.feature_source}")
    print(f"  threshold:      {feat_threshold} ({'>=pos' if feat_positive_above else '<pos'})")
    print(f"  top_n:          {args.top_n}")
    print(f"  sae_threshold:  {args.sae_threshold}")
    print(f"  plm_model:      {args.plm_model}")
    print(f"  plm_layer:      {args.plm_layer}")
    print(f"  device:         {device}")
    print(f"  embedding:      {h5_path}")
    print(f"  refer:          {feat_refer}")
    print(f"  val:            {feat_val}")
    print(f"  output:         {OUTPUT_DIR}")

    # 检查输入
    for fp, name in [(feat_refer, "refer data"), (feat_val, "val data"),
                     (h5_path, "embedding cache")]:
        if not fp.exists():
            print(f"\nERROR: {name} 不存在: {fp}")
            return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(OUTPUT_DIR / "step7_log.tsv")

    logger.log("main", "start",
               f"feature_name={args.feature_name}, source={args.feature_source}, "
               f"threshold={feat_threshold}, positive_above={feat_positive_above}, "
               f"top_n={args.top_n}, sae_thresh={args.sae_threshold}, "
               f"model={args.plm_model}, layer={args.plm_layer}")
    t0 = time.time()

    # ---- 7.1 读取 refer 集 ----
    print("\n--- 7.1 读取 refer 集 ---")
    refer_data = load_feature_data(feat_refer, feat_value_col,
                                   feat_threshold, feat_positive_above)
    n_refer_pos = sum(labels.sum() for _, labels in refer_data.values())
    n_refer_neg = sum(len(labels) - labels.sum() for _, labels in refer_data.values())
    logger.log("7.1_load", "refer",
               f"proteins={len(refer_data)}, "
               f"positive={n_refer_pos}, negative={n_refer_neg}")

    # ---- 7.2 加载 SAE ----
    print("\n--- 7.2 加载 SAE ---")
    logger.log("7.2_sae", "loading",
               f"model={args.plm_model}, layer={args.plm_layer}")
    sae = load_sae_from_hf(
        plm_model=args.plm_model, plm_layer=args.plm_layer
    ).to(device)
    sae.eval()

    # 获取 feature 数量
    # SAE encode 一个 dummy input 来确定 n_features
    with h5py.File(h5_path, "r") as f:
        for key in f:
            dummy_emb = f[key][:1, :]  # (1, embed_dim)
            break
    dummy_tensor = torch.from_numpy(dummy_emb).float().to(device)
    with torch.no_grad():
        dummy_act = sae.encode(dummy_tensor)
    n_features = dummy_act.shape[1]
    logger.log("7.2_sae", "loaded", f"n_features={n_features}")
    print(f"  SAE features: {n_features}")

    # ---- 7.3 Refer 集评估 ----
    print("\n--- 7.3 评估 refer 集 (全部 features) ---")
    tp_r, fp_r, fn_r = evaluate_on_dataset(
        refer_data, h5_path, sae, args.sae_threshold,
        device, n_features, logger, "refer")

    f1_r, prec_r, rec_r = compute_f1(tp_r, fp_r, fn_r)

    # 保存全量 refer F1 (按 feature_idx 顺序)
    refer_f1_file = OUTPUT_DIR / "refer_f1_all.tsv"
    with open(refer_f1_file, "w") as f:
        f.write("feature_idx\tf1\tprecision\trecall\ttp\tfp\tfn\n")
        for i in range(n_features):
            f.write(f"{i}\t{f1_r[i]:.6f}\t{prec_r[i]:.6f}\t{rec_r[i]:.6f}\t"
                    f"{tp_r[i]}\t{fp_r[i]}\t{fn_r[i]}\n")
    logger.log("7.3_refer", "saved", f"file={refer_f1_file}")

    # 保存按 F1 降序排列的版本
    refer_f1_sorted_file = OUTPUT_DIR / "refer_f1_sorted.tsv"
    sorted_indices = np.argsort(f1_r)[::-1]
    with open(refer_f1_sorted_file, "w") as f:
        f.write("rank\tfeature_idx\tf1\tprecision\trecall\ttp\tfp\tfn\n")
        for rank, i in enumerate(sorted_indices):
            f.write(f"{rank + 1}\t{i}\t{f1_r[i]:.6f}\t{prec_r[i]:.6f}\t{rec_r[i]:.6f}\t"
                    f"{tp_r[i]}\t{fp_r[i]}\t{fn_r[i]}\n")
    logger.log("7.3_refer", "saved_sorted", f"file={refer_f1_sorted_file}")

    # 绘制 F1 分布直方图 (纵轴=F1, 横轴=feature下标)
    hist_file = OUTPUT_DIR / "refer_f1_distribution.png"
    if plt is not None:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(n_features), f1_r, width=1.0, color="#4C72B0", alpha=0.8)
        ax.set_xlabel("Feature Index", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(f"Refer Set F1 per SAE Feature ({args.feature_name})", fontsize=14)
        ax.set_xlim(-0.5, n_features - 0.5)
        ax.set_ylim(0, min(1.0, f1_r.max() * 1.15) if f1_r.max() > 0 else 1.0)
        ax.axhline(y=f1_r[f1_r > 0].mean() if (f1_r > 0).any() else 0,
                   color="red", linestyle="--", linewidth=0.8,
                   label=f"mean(nonzero)={f1_r[f1_r > 0].mean():.4f}" if (f1_r > 0).any() else "")
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(hist_file, dpi=150)
        plt.close(fig)
        logger.log("7.3_refer", "histogram", f"file={hist_file}")
        print(f"  直方图: {hist_file}")
    else:
        logger.log("7.3_refer", "histogram_skip", "matplotlib not installed")

    # 统计 refer F1 分布
    f1_nonzero = f1_r[f1_r > 0]
    mean_nz = f1_nonzero.mean() if len(f1_nonzero) > 0 else 0.0
    logger.log("7.3_refer", "f1_stats",
               f"max={f1_r.max():.4f}, "
               f"mean_nonzero={mean_nz:.4f}, "
               f"n_nonzero={len(f1_nonzero)}/{n_features}")

    # ---- 7.4 选出 top-N ----
    print(f"\n--- 7.4 选出 top-{args.top_n} features ---")
    top_indices = np.argsort(f1_r)[::-1][:args.top_n]

    top_file = OUTPUT_DIR / "top_features.tsv"
    with open(top_file, "w") as f:
        f.write("rank\tfeature_idx\tf1_refer\tprecision\trecall\ttp\tfp\tfn\n")
        for rank, idx in enumerate(top_indices):
            f.write(f"{rank + 1}\t{idx}\t{f1_r[idx]:.6f}\t"
                    f"{prec_r[idx]:.6f}\t{rec_r[idx]:.6f}\t"
                    f"{tp_r[idx]}\t{fp_r[idx]}\t{fn_r[idx]}\n")
            logger.log("7.4_topN", "selected",
                       f"rank={rank + 1}, feature={idx}, "
                       f"f1={f1_r[idx]:.4f}, prec={prec_r[idx]:.4f}, "
                       f"rec={rec_r[idx]:.4f}")
    logger.log("7.4_topN", "saved", f"file={top_file}")

    print(f"  Top-{args.top_n} F1 range: "
          f"{f1_r[top_indices[-1]]:.4f} ~ {f1_r[top_indices[0]]:.4f}")

    # ---- 7.5 Validation 集评估 (仅 top-N) ----
    print(f"\n--- 7.5 评估 validation 集 (top-{args.top_n} features) ---")
    val_data = load_feature_data(feat_val, feat_value_col,
                                 feat_threshold, feat_positive_above)
    n_val_pos = sum(labels.sum() for _, labels in val_data.values())
    n_val_neg = sum(len(labels) - labels.sum() for _, labels in val_data.values())
    logger.log("7.5_val", "loaded",
               f"proteins={len(val_data)}, "
               f"positive={n_val_pos}, negative={n_val_neg}")

    tp_v, fp_v, fn_v = evaluate_on_dataset(
        val_data, h5_path, sae, args.sae_threshold,
        device, n_features, logger, "val",
        feature_subset=top_indices)

    f1_v, prec_v, rec_v = compute_f1(tp_v, fp_v, fn_v)

    val_f1_file = OUTPUT_DIR / "val_f1_top.tsv"
    with open(val_f1_file, "w") as f:
        f.write("rank\tfeature_idx\tf1_refer\tf1_val\t"
                "prec_val\trecall_val\ttp_val\tfp_val\tfn_val\n")
        for rank, (idx, local_i) in enumerate(zip(top_indices, range(len(top_indices)))):
            f.write(f"{rank + 1}\t{idx}\t{f1_r[idx]:.6f}\t{f1_v[local_i]:.6f}\t"
                    f"{prec_v[local_i]:.6f}\t{rec_v[local_i]:.6f}\t"
                    f"{tp_v[local_i]}\t{fp_v[local_i]}\t{fn_v[local_i]}\n")
            logger.log("7.5_val", "feature",
                       f"rank={rank + 1}, feature={idx}, "
                       f"f1_refer={f1_r[idx]:.4f}, f1_val={f1_v[local_i]:.4f}, "
                       f"delta={f1_v[local_i] - f1_r[idx]:+.4f}")
    logger.log("7.5_val", "saved", f"file={val_f1_file}")

    # ---- 7.6 累计 F1 (top-k features OR联合) ----
    print(f"\n--- 7.6 累计 F1 (top-{args.top_n} OR联合, val集) ---")
    cum_tp, cum_fp, cum_fn = evaluate_cumulative(
        val_data, h5_path, sae, args.sae_threshold,
        device, top_indices, logger)
    cum_f1, cum_prec, cum_rec = compute_f1(cum_tp, cum_fp, cum_fn)

    cum_f1_file = OUTPUT_DIR / "val_cumulative_f1.tsv"
    with open(cum_f1_file, "w") as f:
        f.write("top_k\tfeature_indices\tf1\tprecision\trecall\ttp\tfp\tfn\n")
        for k in range(len(top_indices)):
            feat_list = ",".join(str(idx) for idx in top_indices[:k + 1])
            f.write(f"{k + 1}\t{feat_list}\t{cum_f1[k]:.6f}\t"
                    f"{cum_prec[k]:.6f}\t{cum_rec[k]:.6f}\t"
                    f"{cum_tp[k]}\t{cum_fp[k]}\t{cum_fn[k]}\n")
            logger.log("7.6_cumulative", "rank",
                       f"top_{k + 1}: f1={cum_f1[k]:.4f}, "
                       f"prec={cum_prec[k]:.4f}, rec={cum_rec[k]:.4f}")
    logger.log("7.6_cumulative", "saved", f"file={cum_f1_file}")

    elapsed_total = time.time() - t0
    logger.log("main", "done", f"elapsed={elapsed_total:.1f}s")

    # ---- 摘要 ----
    print()
    print("=" * 60)
    print("完成")
    print("=" * 60)
    print(f"  SAE features:   {n_features}")
    print(f"  Refer集:        {len(refer_data)} proteins")
    print(f"  Val集:          {len(val_data)} proteins")
    print(f"  Top-{args.top_n} 结果:")
    print(f"  {'Rank':<6}{'Feature':<10}{'F1(refer)':<12}{'F1(val)':<12}{'Delta':<10}{'F1(cum)':<10}")
    print(f"  {'-'*58}")
    for rank, (idx, local_i) in enumerate(zip(top_indices, range(len(top_indices)))):
        delta = f1_v[local_i] - f1_r[idx]
        print(f"  {rank + 1:<6}{idx:<10}{f1_r[idx]:<12.4f}"
              f"{f1_v[local_i]:<12.4f}{delta:<+10.4f}{cum_f1[rank]:<10.4f}")
    print(f"\n  耗时: {elapsed_total:.1f}s")
    print(f"\n输出文件:")
    for fp in [refer_f1_file, refer_f1_sorted_file, hist_file,
               top_file, val_f1_file, cum_f1_file, OUTPUT_DIR / "step7_log.tsv"]:
        if fp.exists():
            sz = fp.stat().st_size / 1024
            print(f"  {fp} ({sz:.1f} KB)")


if __name__ == "__main__":
    main()