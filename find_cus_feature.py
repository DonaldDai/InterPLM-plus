import torch
import numpy as np
from typing import List, Dict, Callable

def find_cavity_features(
    data: List[Dict], 
    get_features_fn: Callable[[str], torch.Tensor],
    top_k: int = 5,
    thresholds: List[float] = [0, 0.15, 0.5, 0.6, 0.8]
) -> Dict[float, List[int]]:
    """
    Find the SAE feature indices that highly correlate with hydrophobic cavity positions.
    
    Args:
        data: List of dicts, e.g. [{"seq_id": "1", "seq": "MV...", "cavity": [2, 5]}]
        get_features_fn: Function that takes a sequence string and returns 
                         SAE feature activations of shape (seq_length, num_features)
        top_k: Number of top features to return
        thresholds: List of activation values above which a feature is considered activated
        
    Returns:
        Dict mapping each threshold to its List of the top_k feature indices
    """
    # To accumulate counts for each threshold
    tp_counts = {t: None for t in thresholds}
    fp_counts = {t: None for t in thresholds}
    fn_counts = {t: None for t in thresholds}
    cavity_count = 0
    non_cavity_count = 0
    
    for item in data:
        seq = item['seq']
        cavity_indices = set(item['cavity'])
        
        # features shape: (seq_length, num_features)
        features = get_features_fn(seq)
        print(features.shape)
        
        # Only print once per sequence to avoid spam
        # print('features.dim(): ', features.dim())
        if features.dim() == 3:  # (1, seq_len, num_features)
            features = features.squeeze(0)
            
        seq_len, num_features = features.shape
        
        # Initialize counts if None
        for t in thresholds:
            if tp_counts[t] is None:
                tp_counts[t] = torch.zeros(num_features, device=features.device)
                fp_counts[t] = torch.zeros(num_features, device=features.device)
                fn_counts[t] = torch.zeros(num_features, device=features.device)
            
        # Create a boolean mask for cavity positions
        mask = torch.zeros(seq_len, dtype=torch.bool)
        valid_cavity_indices = [i for i in cavity_indices if i < seq_len]
        if valid_cavity_indices:
            mask[valid_cavity_indices] = True
            
        cavity_features = features[mask]
        non_cavity_features = features[~mask]
        
        cavity_len = len(cavity_features)
        non_cavity_len = len(non_cavity_features)
        
        cavity_count += cavity_len
        non_cavity_count += non_cavity_len
        
        for t in thresholds:
            if cavity_len > 0:
                activated_cavity = (cavity_features > t).sum(dim=0).float()
                tp_counts[t] += activated_cavity
                fn_counts[t] += (cavity_len - activated_cavity)
                
            if non_cavity_len > 0:
                activated_non_cavity = (non_cavity_features > t).sum(dim=0).float()
                fp_counts[t] += activated_non_cavity
            
    # Calculate Precision, Recall, and F1-score for each threshold
    results = {}
    epsilon = 1e-8
    
    for t in thresholds:
        precision = tp_counts[t] / (tp_counts[t] + fp_counts[t] + epsilon)
        recall = tp_counts[t] / (tp_counts[t] + fn_counts[t] + epsilon)
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
        
        # Get top k features
        top_indices = torch.topk(f1_score, top_k).indices.cpu().tolist()
        results[t] = top_indices
        
        print(f"\nEvaluated on {cavity_count} cavity residues and {non_cavity_count} non-cavity residues with threshold {t}.")
        for rank, idx in enumerate(top_indices):
            print(f"Rank {rank+1}: Feature {idx} (F1 Score: {f1_score[idx]:.4f}, Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f})")
        
    return results

# Example usage (you can replace the placeholder logic with your actual implementation):
if __name__ == "__main__":
    from interplm.embedders import get_embedder
    from interplm.sae.inference import load_sae_from_hf
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Embedder
    # Set the model name according to your SAE
    model_name = "facebook/esm2_t6_8M_UR50D"
    embedder = get_embedder('esm', model_name=model_name)
    plm_layer = 4
    
    # 2. Initialize SAE
    plm_model = "esm2-8m"
    sae = load_sae_from_hf(plm_model=plm_model, plm_layer=plm_layer).to(device)
    
    # 3. Define the actual feature extraction function
    def get_features(seq: str) -> torch.Tensor:
        # Get embeddings from sequence
        embeddings = embedder.embed_single_sequence(seq, layer=plm_layer)
        
        # Convert to tensor and move to device
        embeddings_tensor = torch.from_numpy(embeddings).to(device)
        
        # Expand dims if SAE expects batch dimension, SAE encode usually returns (batch, seq_len, num_features)
        if embeddings_tensor.dim() == 2:
            embeddings_tensor = embeddings_tensor.unsqueeze(0)
            
        with torch.no_grad():
            features = sae.encode(embeddings_tensor)
            
        # Return features (squeeze batch dim)
        return features.squeeze(0)

    # 示例数据结构 (请替换为您真实的疏水腔数据)
    sample_data = [
        # {
        #     "seq_id": "A6NIH7",
        #     "seq": "MSGSNPKAAAAASAAGPGGLVAGKEEKKKAGGGVLNRLKARRQAPHHAADDGVGAAVTEQELLALDTIRPEHVLRLSRVTENYLCKPEDNIYSIDFTRFKIRDLETGTVLFEIAKPCVSDQEEDEEEGGGDVDISAGRFVRYQFTPAFLRLRTVGATVEFTVGDKPVSNFRMIERHYFREHLLKNFDFDFGFCIPSSRNTCEHIYEFPQLSEDVIRLMIENPYETRSDSFYFVDNKLIMHNKADYAYNGGQ",  # Your sequence here
        #     "cavity": [95, 98, 110, 143, 147, 159, 169, 185, 187, 189, 206] # 0-indexed indices of hydrophobic cavity residues
        # },
        {
            "seq_id": "P0AFW0",
            "seq": "MQSWYLLYCKRGQLQRAQEHLERQAVNCLAPMITLEKIVRGKRTAVSEPLFPNYLFVEFDPEVIHTTTINATRGVSHFVRFGASPAIVPSAVIHQLSVYKPKDIVDPATPYPGDKVIITEGAFEGFQAIFTEPDGEARSMLLLNLINKEIKHSVKNTEFRKL",  # Your sequence here
            "cavity": [5, 32, 48, 49, 50, 55, 78, 80, 86, 88, 91, 92, 117, 121, 125, 128, 129, 132, 141, 145, 149]
        },
        {
            "seq_id": "Q14849",
            "seq": "MSKLPRELTRDLERSLPAVASLGSSLSHSQSLSSHLLPPPEKRRAISDVRRTFCLFVTFDLLFISLLWIIELNTNTGIRKNLEQEIIQYNFKTSFFDIFVLAFFRFSGLLLGYAVLRLRHWWVIAVTTLVSSAFLIVKVILSELLSKGAFGYLLPIVSFVLAWLETWFLDFKVLPQEAEEERWYLAAQVAVARGPLLFSGALSEGQFYSPPESFAGSDNESDEEVAGKKSFSAQEREYIRQGKEATAVVDQILAQEENWKFEKNNEYGDTVYTIEVPFHGKTFILKTFLPCPAELVYQEVILQPERMVLWNKTVTACQILQRVEDNTLISYDVSAGAAGGVVSPRDFVNVRRIERRRDRYLSSGIATSHSAKPPTHKYVRGENGPGGFIVLKSASNPRVCTFVWILNTDLKGRLPRYLIHQSLAATMFEFAFHLRQRISELGARA",  # Your sequence here
            "cavity": [307] # 0-indexed indices of hydrophobic cavity residues
        },
    ]
        
    test_thresholds = [0, 0.15, 0.5, 0.6, 0.8]
    top_features_dict = find_cavity_features(sample_data, get_features, top_k=5, thresholds=test_thresholds)
    print("Pre-trained SAE feature numbers found for each threshold:", top_features_dict)

    # ==========================================
    # 验证与可视化 (Validation and Plotting)
    # ==========================================
    import matplotlib.pyplot as plt
    import os
    
    # 选定一个验证数据 (这里我们选用第一个，您也可以定义新的验证序列)
    val_data = {
            "seq_id": "O60603",
            "seq": "MPHTLWMVWVLGVIISLSKEESSNQASLSCDRNGICKGSSGSLNSIPSGLTEAVKSLDLSNNRITYISNSDLQRCVNLQALVLTSNGINTIEEDSFSSLGSLEHLDLSYNYLSNLSSSWFKPLSSLTFLNLLGNPYKTLGETSLFSHLTKLQILRVGNMDTFTKIQRKDFAGLTFLEELEIDASDLQSYEPKSLKSIQNVSHLILHMKQHILLLEIFVDVTSSVECLELRDTDLDTFHFSELSTGETNSLIKKFTFRNVKITDESLFQVMKLLNQISGLLELEFDDCTLNGVGNFRASDNDRVIDPGKVETLTIRRLHIPRFYLFYDLSTLYSLTERVKRITVENSKVFLVPCLLSQHLKSLEYLDLSENLMVEEYLKNSACEDAWPSLQTLILRQNHLASLEKTGETLLTLKNLTNIDISKNSFHSMPETCQWPEKMKYLNLSSTRIHSVTGCIPKTLEILDVSNNNLNLFSLNLPQLKELYISRNKLMTLPDASLLPMLLVLKISRNAITTFSKEQLDSFHTLKTLEAGGNNFICSCEFLSFTQEQQALAKVLIDWPANYLCDSPSHVRGQQVQDVRLSVSECHRTALVSGMCCALFLLILLTGVLCHRFHGLWYMKMMWAWLQAKRKPRKAPSRNICYDAFVSYSERDAYWVENLMVQELENFNPPFKLCLHKRDFIPGKWIIDNIIDSIEKSHKTVFVLSENFVKSEWCKYELDFSHFRLFDENNDAAILILLEPIEKKAIPQRFCKLRKIMNTKTYLEWPMDEAQREGFWVNLRAAIKS",  # Your sequence here
            "cavity": list(range(250,361))
        }
    # val_data = {
    #         "seq_id": "P06454",
    #         "seq": "MSDAAVDTSSEITTKDLKEKKEVVEEAENGRDAPANGNAENEENGEQEADNEVDEEEEEGGEEEEEEEEGDGEEEDGDEDEEAESATGKRAAEDDEDDDVDTKKQKTDEDD",  # Your sequence here
    #         "cavity": []
    #     }
    
    val_seq = val_data["seq"]
    val_cavities = set(val_data["cavity"])
    
    print(f"\nEvaluating on validation sequence {val_data['seq_id']}...")
    val_features = get_features(val_seq)
    
    # 横坐标：氨基酸坐标 (1-based)
    x_coords = np.arange(1, len(val_seq) + 1)
    
    for thr, top_features in top_features_dict.items():
        if not top_features:
            continue
            
        best_feature_idx = top_features[0]
        print(f"Plotting Top-1 Feature ({best_feature_idx}) for threshold {thr}...")
        
        # 获取 Top 1 feature 在整条序列上的激活值
        activations = val_features[:, best_feature_idx].cpu().numpy()
        
        plt.figure(figsize=(15, 6))
        plt.plot(x_coords, activations, label=f'Feature {best_feature_idx} Activations', color='royalblue')
        
        # 标出实际是 Cavity 的点 (用红色散点)
        cavity_x = [idx + 1 for idx in val_cavities]
        cavity_y = [activations[idx] for idx in val_cavities]
        plt.scatter(cavity_x, cavity_y, color='red', s=40, zorder=5, label='True Cavity Positions')
        
        # 定义高激活阈值，例如大于该 feature 的阈值或者最大值的一部分，这里我们用该阈值thr
        plot_threshold = max(activations.max() * 0.25, thr)
        for i, act_val in enumerate(activations):
            if act_val > plot_threshold:
                # 如果该点恰好也是真实的疏水腔点，我们用红色字体，否则用黑色
                text_color = 'red' if i in val_cavities else 'black'
                plt.annotate(
                    f'{val_seq[i]}{i+1}', 
                    (x_coords[i], act_val), 
                    textcoords="offset points", 
                    xytext=(0, 8), 
                    ha='center', 
                    color=text_color,
                    fontsize=9,
                    fontweight='bold'
                )
                
        plt.axhline(thr, color='red', linestyle='-', alpha=0.8, label=f'Train Threshold = {thr}')
        plt.axhline(activations.max() * 0.25, color='gray', linestyle='--', alpha=0.5, label='25% Max Act Threshold')
        
        plt.xlabel('Amino Acid Coordinate (1-indexed)')
        plt.ylabel('Activation Value')
        plt.title(f'SAE Feature {best_feature_idx} Activations on {val_data["seq_id"]} (Trained with threshold {thr})')
        plt.legend()
        plt.tight_layout()
        
        plot_filename = f'cavity_{val_data["seq_id"]}_feature_{best_feature_idx}_th_{thr}_activation.png'
        plt.savefig(plot_filename, dpi=300)
        plt.close()
        print(f"Validation plot saved to '{plot_filename}'")
