"""
Step 2: CD-HIT 多阈值聚类
===========================

原理说明:
- CD-HIT 按序列相似度对蛋白质序列做聚类，每个cluster选一个代表序列
- 不同阈值的意义:
  * 90%: 去除近同源序列（同一蛋白的不同物种直系同源物）
  * 50%: 去除远同源序列（同一超家族内的旁系同源物）
  * 30%: 极严格去冗余（不同fold之间几乎无序列相似性）
- 对SAE研究的意义：
  * 高阈值(90%)的数据集大，适合训练
  * 低阈值(30%)的数据集保证了序列多样性，适合验证SAE特征的泛化性
  * 如果SAE特征在30%去冗余集上仍与疏水性强相关，说明它学到的是
    真正的物理化学信号而非序列同源性的混淆因素

CD-HIT 安装:
    conda install -c bioconda cd-hit
    # 或
    sudo apt install cd-hit

输入: data/01_raw/uniprot_pdb_sequences.fasta
输出: data/02_clustered/ 目录下多个阈值的聚类结果
"""

import subprocess
import os
from pathlib import Path
from collections import defaultdict


# ============================================================
# 配置区
# ============================================================
INPUT_FASTA = Path("cusdata/01_raw/uniprot_pdb_sequences.fasta")
OUTPUT_DIR = Path("cusdata/02_clustered")

# CD-HIT 聚类阈值配置
# 注意: CD-HIT对不同阈值有word size要求
# identity >= 0.7  → -n 5
# identity >= 0.6  → -n 4
# identity >= 0.5  → -n 3
# identity >= 0.4  → -n 2
CLUSTER_CONFIGS = [
    {"identity": 0.90, "word_size": 5, "name": "id90"},
    {"identity": 0.50, "word_size": 3, "name": "id50"},
    {"identity": 0.30, "word_size": 2, "name": "id30"},
]

# CD-HIT 通用参数
CDHIT_PARAMS = {
    "threads": 8,        # 并行线程数，根据你的CPU调整
    "memory": 16000,     # 内存限制 (MB)
    "description": 0,    # 输出中包含完整序列描述
    "global": 1,         # 使用全局序列相似度 (而非局部)
}


# ============================================================
# 运行 CD-HIT
# ============================================================
def run_cdhit(input_fasta, output_prefix, identity, word_size):
    """
    调用 CD-HIT 执行聚类
    
    CD-HIT 关键参数说明:
    -i    输入FASTA
    -o    输出代表序列FASTA
    -c    序列相似度阈值 (0.0-1.0)
    -n    word size (影响速度和灵敏度)
    -M    内存限制 (MB, 0=无限制)
    -T    线程数
    -d    序列描述长度 (0=完整)
    -g    1=更精确的聚类 (slower但更准确)
    -aL   较长序列的alignment coverage阈值
    -aS   较短序列的alignment coverage阈值
    """
    cmd = [
        "cd-hit",
        "-i", str(input_fasta),
        "-o", str(output_prefix),
        "-c", str(identity),
        "-n", str(word_size),
        "-M", str(CDHIT_PARAMS["memory"]),
        "-T", str(CDHIT_PARAMS["threads"]),
        "-d", str(CDHIT_PARAMS["description"]),
        "-g", str(CDHIT_PARAMS["global"]),
        "-aL", "0.8",   # 要求alignment覆盖较长序列的80%
        "-aS", "0.8",   # 要求alignment覆盖较短序列的80%
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"CD-HIT 错误:\n{result.stderr}")
        raise RuntimeError(f"CD-HIT failed with return code {result.returncode}")
    
    # 打印CD-HIT摘要信息
    for line in result.stdout.split("\n"):
        if any(kw in line.lower() for kw in ["finished", "clusters", "representatives"]):
            print(f"  {line.strip()}")
    
    return output_prefix


# ============================================================
# 解析 CD-HIT 聚类结果
# ============================================================
def parse_cdhit_clusters(clstr_file):
    """
    解析 .clstr 文件，返回聚类结果
    
    .clstr 文件格式:
    >Cluster 0
    0   350aa, >sp|P12345|... at 100.00%
    1   348aa, >sp|P67890|... at 95.23%
    >Cluster 1
    0   200aa, >sp|Q11111|... at 100.00%
    ...
    
    返回:
    - clusters: dict {cluster_id: [seq_id1, seq_id2, ...]}
    - representatives: dict {cluster_id: representative_seq_id}
    - seq_to_cluster: dict {seq_id: cluster_id}
    """
    clusters = defaultdict(list)
    representatives = {}
    seq_to_cluster = {}
    
    current_cluster = None
    
    with open(clstr_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">Cluster"):
                current_cluster = int(line.split()[-1])
            elif line:
                # 提取序列ID
                # 格式: "0  350aa, >sp|P12345|PROT_HUMAN... *" (代表序列)
                # 或:   "1  348aa, >sp|P67890|PROT_MOUSE... at 95.23%"
                parts = line.split(">")
                if len(parts) >= 2:
                    seq_part = parts[1]
                    # 提取到 "..." 或 " " 为止的ID
                    seq_id = seq_part.split("...")[0].split()[0]
                    
                    clusters[current_cluster].append(seq_id)
                    seq_to_cluster[seq_id] = current_cluster
                    
                    # 代表序列标记为 "*"
                    if line.endswith("*"):
                        representatives[current_cluster] = seq_id
    
    return dict(clusters), representatives, seq_to_cluster


def print_cluster_statistics(clusters, config_name):
    """打印聚类统计信息"""
    sizes = [len(members) for members in clusters.values()]
    
    print(f"\n{'=' * 50}")
    print(f"聚类统计 - {config_name}")
    print(f"{'=' * 50}")
    print(f"总cluster数:   {len(clusters)}")
    print(f"总序列数:      {sum(sizes)}")
    print(f"最大cluster:   {max(sizes)} 条序列")
    print(f"最小cluster:   {min(sizes)} 条序列")
    print(f"平均cluster:   {sum(sizes)/len(sizes):.1f} 条序列")
    print(f"中位cluster:   {sorted(sizes)[len(sizes)//2]} 条序列")
    print(f"单序列cluster: {sum(1 for s in sizes if s == 1)} 个")
    
    # cluster大小分布
    print(f"\nCluster大小分布:")
    brackets = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, float('inf'))]
    for lo, hi in brackets:
        count = sum(1 for s in sizes if lo <= s <= hi)
        hi_str = f"{hi:.0f}" if hi != float('inf') else "∞"
        print(f"  [{lo:>4}-{hi_str:>4}]: {count:>6} clusters")


# ============================================================
# 保存聚类分配结果
# ============================================================
def save_cluster_assignments(seq_to_cluster, representatives, output_file):
    """
    保存每条序列的聚类分配，便于后续使用
    
    输出TSV格式:
    seq_id  cluster_id  is_representative
    """
    rep_set = set(representatives.values())
    
    with open(output_file, "w") as f:
        f.write("seq_id\tcluster_id\tis_representative\n")
        for seq_id, cluster_id in sorted(seq_to_cluster.items(), key=lambda x: x[1]):
            is_rep = seq_id in rep_set
            f.write(f"{seq_id}\t{cluster_id}\t{is_rep}\n")
    
    print(f"聚类分配已保存到: {output_file}")


# ============================================================
# 主流程
# ============================================================
def main():
    print("CD-HIT 多阈值聚类")
    print("=" * 60)
    
    # 检查输入文件
    if not INPUT_FASTA.exists():
        print(f"ERROR: 输入文件不存在: {INPUT_FASTA}")
        print("请先运行 step1_download_uniprot.py")
        return
    
    # 检查CD-HIT是否安装
    try:
        result = subprocess.run(["cd-hit", "-h"], capture_output=True, text=True)
    except FileNotFoundError:
        print("ERROR: cd-hit 未安装")
        print("安装方式:")
        print("  conda install -c bioconda cd-hit")
        print("  # 或")
        print("  sudo apt install cd-hit")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 按阈值从高到低依次聚类
    # 策略: 先用90%聚类得到非冗余集，再用50%和30%进一步聚类
    all_results = {}
    
    for config in CLUSTER_CONFIGS:
        name = config["name"]
        identity = config["identity"]
        word_size = config["word_size"]
        
        print(f"\n{'#' * 60}")
        print(f"# 聚类阈值: {identity*100:.0f}% ({name})")
        print(f"{'#' * 60}")
        
        output_prefix = OUTPUT_DIR / f"cdhit_{name}"
        clstr_file = Path(f"{output_prefix}.clstr")
        
        # 运行CD-HIT
        run_cdhit(INPUT_FASTA, output_prefix, identity, word_size)
        
        # 解析结果
        if clstr_file.exists():
            clusters, representatives, seq_to_cluster = parse_cdhit_clusters(clstr_file)
            print_cluster_statistics(clusters, name)
            
            # 保存分配表
            assignment_file = OUTPUT_DIR / f"cluster_assignments_{name}.tsv"
            save_cluster_assignments(seq_to_cluster, representatives, assignment_file)
            
            all_results[name] = {
                "clusters": clusters,
                "representatives": representatives,
                "seq_to_cluster": seq_to_cluster,
            }
        else:
            print(f"WARNING: 聚类结果文件不存在: {clstr_file}")
    
    # 汇总对比
    print(f"\n{'=' * 60}")
    print("各阈值聚类结果对比")
    print(f"{'=' * 60}")
    print(f"{'阈值':<10} {'clusters数':<12} {'代表序列数':<12} {'压缩比':<10}")
    print("-" * 44)
    
    total_seqs = None
    for config in CLUSTER_CONFIGS:
        name = config["name"]
        if name in all_results:
            n_clusters = len(all_results[name]["clusters"])
            n_seqs = sum(len(v) for v in all_results[name]["clusters"].values())
            if total_seqs is None:
                total_seqs = n_seqs
            ratio = n_clusters / n_seqs * 100
            print(f"{name:<10} {n_clusters:<12} {n_seqs:<12} {ratio:.1f}%")
    
    print(f"\n所有聚类结果已保存到: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
