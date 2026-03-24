"""
Step 3: 基于聚类的平衡训练/验证集拆分
========================================

原理说明:
- 为什么不能随机拆分？
  如果训练集和验证集中有高度相似的序列（来自同一cluster），
  SAE特征在验证集上的表现会被高估——它可能只是"记住"了相似
  序列的模式，而非学到了疏水性的物理化学规律。
  
- 基于聚类拆分的逻辑：
  同一个cluster内的序列只能出现在训练集或验证集的其中一个，
  不能跨集出现。这保证了验证集中的序列与训练集的相似度
  低于聚类阈值。

- 以最小群数量为基础的含义：
  如果你有多个fold类型的群（全α、全β、α/β等），
  为保证各群在验证集中的代表性均衡，以数量最少的群为基准
  确定每个群抽出多少cluster进入验证集。

拆分策略:
  1. 选定一个CD-HIT阈值（推荐30%用于最终评估，50%用于开发）
  2. 统计各fold类型群中的cluster数量
  3. 以最小群的cluster数量为基准，计算验证集比例
  4. 从每个群中按比例随机抽取cluster进入验证集
  5. 剩余cluster进入训练集

输入: data/02_clustered/cluster_assignments_{threshold}.tsv
输出: data/03_splits/ 目录下的训练集和验证集FASTA及标签
"""

import random
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


# ============================================================
# 配置区
# ============================================================
INPUT_FASTA = Path("data/01_raw/uniprot_pdb_sequences.fasta")
CLUSTER_DIR = Path("data/02_clustered")
OUTPUT_DIR = Path("data/03_splits")

# 选择用哪个聚类阈值做拆分
# "id30" = 最严格（验证集与训练集相似度 < 30%，最可靠的泛化评估）
# "id50" = 中等（开发阶段推荐，数据量更大）
SPLIT_THRESHOLD = "id30"

# 验证集占比
VAL_RATIO = 0.15  # 15%的clusters分配给验证集

# 随机种子（保证可复现）
RANDOM_SEED = 42


# ============================================================
# 读取FASTA序列到字典
# ============================================================
def load_fasta(fasta_path):
    """读取FASTA文件，返回 {seq_id: sequence} 字典"""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                # 提取ID: ">sp|P12345|PROT_HUMAN ..." → "sp|P12345|PROT_HUMAN"
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)
    
    return sequences


# ============================================================
# 读取聚类分配
# ============================================================
def load_cluster_assignments(assignment_file):
    """
    读取聚类分配TSV文件
    
    返回:
    - cluster_to_seqs: {cluster_id: [seq_id, ...]}
    - seq_to_cluster: {seq_id: cluster_id}
    """
    cluster_to_seqs = defaultdict(list)
    seq_to_cluster = {}
    
    with open(assignment_file) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                seq_id = parts[0]
                cluster_id = int(parts[1])
                cluster_to_seqs[cluster_id].append(seq_id)
                seq_to_cluster[seq_id] = cluster_id
    
    return dict(cluster_to_seqs), seq_to_cluster


# ============================================================
# (可选) 按fold类型分群
# ============================================================
def assign_fold_groups(cluster_to_seqs, sequences):
    """
    将clusters按蛋白质fold类型分群
    
    简化版本: 按氨基酸组成特征做粗略分群
    生产版本: 应从SCOP/CATH数据库获取fold分类
    
    这里提供两种方式:
    1. 简化版 - 按序列特征粗略分类（下面实现）
    2. 完整版 - 从CATH下载fold注释（注释中给出方法）
    """
    
    # ----------------------------------------------------------
    # 简化版: 不做fold分群，所有cluster视为一个群
    # 适用于: 先跑通流程，后续再加入fold信息
    # ----------------------------------------------------------
    all_clusters = list(cluster_to_seqs.keys())
    groups = {"all": all_clusters}
    
    print(f"分群模式: 单群 (所有 {len(all_clusters)} 个clusters)")
    print("提示: 如需按fold类型分群，请使用CATH/SCOP注释")
    
    return groups

    # ----------------------------------------------------------
    # 完整版: 从CATH获取fold分类 (取消注释以启用)
    # ----------------------------------------------------------
    """
    # 需要先下载CATH域注释:
    # wget ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt
    
    import pandas as pd
    
    # 解析CATH分类 (Class.Architecture.Topology.Homology)
    cath = pd.read_csv("cath-domain-list.txt", sep=r'\s+', comment='#',
                       names=["domain_id", "class", "arch", "topol", "homol", ...])
    
    # CATH Class:
    # 1 = Mainly Alpha
    # 2 = Mainly Beta  
    # 3 = Alpha Beta
    # 4 = Few Secondary Structures
    
    class_names = {1: "alpha", 2: "beta", 3: "alpha_beta", 4: "few_ss"}
    
    # 将PDB domain映射到UniProt，再映射到cluster
    # ... (需要PDB-UniProt映射表)
    
    groups = defaultdict(list)
    for cluster_id in cluster_to_seqs:
        fold_class = get_dominant_fold_class(cluster_id, ...)
        groups[fold_class].append(cluster_id)
    
    return dict(groups)
    """


# ============================================================
# 核心拆分逻辑
# ============================================================
def balanced_cluster_split(
    groups: Dict[str, List[int]],
    cluster_to_seqs: Dict[int, List[str]],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    基于最小群数量的平衡拆分
    
    算法:
    1. 找到序列数量最少的群 (min_group)
    2. 计算 min_group 中应分配给验证集的cluster数:
       n_val_clusters = max(1, round(len(min_group_clusters) * val_ratio))
    3. 对每个群，按同样的比例抽取clusters到验证集
    4. 确保每个群在验证集中至少有1个cluster
    
    参数:
        groups: {group_name: [cluster_ids]}
        cluster_to_seqs: {cluster_id: [seq_ids]}
        val_ratio: 验证集比例
        seed: 随机种子
    
    返回:
        (train_seq_ids, val_seq_ids)
    """
    rng = random.Random(seed)
    
    # 统计每个群的序列总数
    group_seq_counts = {}
    for group_name, cluster_ids in groups.items():
        total_seqs = sum(len(cluster_to_seqs.get(cid, [])) for cid in cluster_ids)
        group_seq_counts[group_name] = total_seqs
    
    print(f"\n各群统计:")
    print(f"{'群名':<15} {'clusters数':<12} {'序列数':<10}")
    print("-" * 37)
    for name, cids in groups.items():
        print(f"{name:<15} {len(cids):<12} {group_seq_counts[name]:<10}")
    
    # 找最小群
    min_group_name = min(group_seq_counts, key=group_seq_counts.get)
    min_group_size = group_seq_counts[min_group_name]
    min_group_clusters = len(groups[min_group_name])
    
    print(f"\n最小群: {min_group_name} ({min_group_size} 序列, {min_group_clusters} clusters)")
    
    # 计算最小群的验证集cluster数
    min_val_clusters = max(1, round(min_group_clusters * val_ratio))
    actual_val_ratio = min_val_clusters / min_group_clusters
    
    print(f"最小群验证集clusters数: {min_val_clusters} (实际比例: {actual_val_ratio:.1%})")
    
    # 对每个群按同样比例拆分
    train_seqs = set()
    val_seqs = set()
    
    print(f"\n拆分详情:")
    print(f"{'群名':<15} {'总clusters':<12} {'验证clusters':<14} {'训练seq':<10} {'验证seq':<10}")
    print("-" * 61)
    
    for group_name, cluster_ids in groups.items():
        # 按相同比例计算该群的验证集cluster数
        n_val = max(1, round(len(cluster_ids) * actual_val_ratio))
        
        # 随机选择验证集clusters
        shuffled = cluster_ids.copy()
        rng.shuffle(shuffled)
        
        val_clusters = set(shuffled[:n_val])
        train_clusters = set(shuffled[n_val:])
        
        # 分配序列
        group_train = set()
        group_val = set()
        
        for cid in train_clusters:
            for seq_id in cluster_to_seqs.get(cid, []):
                group_train.add(seq_id)
        
        for cid in val_clusters:
            for seq_id in cluster_to_seqs.get(cid, []):
                group_val.add(seq_id)
        
        train_seqs.update(group_train)
        val_seqs.update(group_val)
        
        print(f"{group_name:<15} {len(cluster_ids):<12} {n_val:<14} "
              f"{len(group_train):<10} {len(group_val):<10}")
    
    return train_seqs, val_seqs


# ============================================================
# 输出拆分结果
# ============================================================
def write_split_fasta(sequences, seq_ids, output_path):
    """将指定ID的序列写入FASTA文件"""
    count = 0
    with open(output_path, "w") as f:
        for seq_id in sorted(seq_ids):
            if seq_id in sequences:
                f.write(f">{seq_id}\n")
                # 每行80字符
                seq = sequences[seq_id]
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + "\n")
                count += 1
    return count


def write_split_ids(seq_ids, output_path):
    """保存序列ID列表"""
    with open(output_path, "w") as f:
        for sid in sorted(seq_ids):
            f.write(f"{sid}\n")


# ============================================================
# 验证拆分质量
# ============================================================
def validate_split(train_seqs, val_seqs, seq_to_cluster):
    """
    验证拆分的正确性:
    1. 训练集和验证集无交集
    2. 同一cluster的序列不会跨集出现
    """
    print(f"\n{'=' * 50}")
    print("拆分验证")
    print(f"{'=' * 50}")
    
    # 检查交集
    overlap = train_seqs & val_seqs
    if overlap:
        print(f"ERROR: 训练集和验证集有 {len(overlap)} 条序列重叠!")
        return False
    print("✓ 训练集和验证集无交集")
    
    # 检查同一cluster是否跨集
    train_clusters = set()
    val_clusters = set()
    
    for sid in train_seqs:
        if sid in seq_to_cluster:
            train_clusters.add(seq_to_cluster[sid])
    for sid in val_seqs:
        if sid in seq_to_cluster:
            val_clusters.add(seq_to_cluster[sid])
    
    cluster_overlap = train_clusters & val_clusters
    if cluster_overlap:
        print(f"ERROR: {len(cluster_overlap)} 个clusters跨训练/验证集出现!")
        return False
    print("✓ 无cluster泄漏 (同一cluster的序列只在一个集中)")
    
    # 统计
    total = len(train_seqs) + len(val_seqs)
    print(f"✓ 训练集: {len(train_seqs)} ({len(train_seqs)/total:.1%})")
    print(f"✓ 验证集: {len(val_seqs)} ({len(val_seqs)/total:.1%})")
    print(f"✓ 训练clusters: {len(train_clusters)}")
    print(f"✓ 验证clusters: {len(val_clusters)}")
    
    return True


# ============================================================
# 主流程
# ============================================================
def main():
    print("基于聚类的平衡训练/验证集拆分")
    print("=" * 60)
    
    # 1. 加载序列
    print(f"加载FASTA: {INPUT_FASTA}")
    sequences = load_fasta(INPUT_FASTA)
    print(f"  共 {len(sequences)} 条序列")
    
    # 2. 加载聚类分配
    assignment_file = CLUSTER_DIR / f"cluster_assignments_{SPLIT_THRESHOLD}.tsv"
    print(f"\n加载聚类分配: {assignment_file}")
    
    if not assignment_file.exists():
        print(f"ERROR: 聚类分配文件不存在: {assignment_file}")
        print("请先运行 step2_cdhit_clustering.py")
        return
    
    cluster_to_seqs, seq_to_cluster = load_cluster_assignments(assignment_file)
    print(f"  共 {len(cluster_to_seqs)} 个clusters, {len(seq_to_cluster)} 条序列")
    
    # 3. 分群
    groups = assign_fold_groups(cluster_to_seqs, sequences)
    
    # 4. 平衡拆分
    train_seqs, val_seqs = balanced_cluster_split(
        groups=groups,
        cluster_to_seqs=cluster_to_seqs,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
    )
    
    # 5. 验证
    is_valid = validate_split(train_seqs, val_seqs, seq_to_cluster)
    if not is_valid:
        print("拆分验证失败!")
        return
    
    # 6. 输出文件
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存拆分结果到 {OUTPUT_DIR}/")
    
    n_train = write_split_fasta(sequences, train_seqs,
                                 OUTPUT_DIR / f"train_{SPLIT_THRESHOLD}.fasta")
    n_val = write_split_fasta(sequences, val_seqs,
                               OUTPUT_DIR / f"val_{SPLIT_THRESHOLD}.fasta")
    
    write_split_ids(train_seqs, OUTPUT_DIR / f"train_{SPLIT_THRESHOLD}_ids.txt")
    write_split_ids(val_seqs, OUTPUT_DIR / f"val_{SPLIT_THRESHOLD}_ids.txt")
    
    print(f"  训练集FASTA: {n_train} 条序列")
    print(f"  验证集FASTA: {n_val} 条序列")
    
    # 7. 保存拆分元数据
    meta_file = OUTPUT_DIR / f"split_metadata_{SPLIT_THRESHOLD}.txt"
    with open(meta_file, "w") as f:
        f.write(f"split_threshold: {SPLIT_THRESHOLD}\n")
        f.write(f"val_ratio: {VAL_RATIO}\n")
        f.write(f"random_seed: {RANDOM_SEED}\n")
        f.write(f"total_clusters: {len(cluster_to_seqs)}\n")
        f.write(f"train_sequences: {len(train_seqs)}\n")
        f.write(f"val_sequences: {len(val_seqs)}\n")
        f.write(f"train_ratio: {len(train_seqs)/(len(train_seqs)+len(val_seqs)):.4f}\n")
        f.write(f"val_ratio_actual: {len(val_seqs)/(len(train_seqs)+len(val_seqs)):.4f}\n")
    
    print(f"\n完成! 元数据已保存到: {meta_file}")


if __name__ == "__main__":
    main()
