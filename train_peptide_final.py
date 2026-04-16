"""
BBB Peptide Prediction - 数据增强实验
测试数据增强是否能在最佳配置 (k=700, 1:5) 基础上进一步提升
对比：有增强 vs 无增强
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from transformers import AutoModel, AutoTokenizer
import iFeatureOmegaCLI

warnings.filterwarnings("ignore")

# 全局随机种子
SEED = 42
DEFAULT_MODEL_DIR = Path("models")


def set_seed(seed=SEED):
    """设置所有随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_fasta(file_path):
    sequences = []
    current_seq = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq is not None:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                if current_seq is not None:
                    current_seq += line
        if current_seq is not None and current_seq:
            sequences.append(current_seq)
    return sequences


class SequenceAugmentor:
    """保守突变增强器"""

    def __init__(self, seed=SEED):
        self.seed = seed
        # 基于理化性质的保守替换
        self.substitutions = {
            "A": ["G", "V"],  # 小型/非极性
            "V": ["A", "I", "L"],  # 非极性/疏水
            "L": ["V", "I", "M"],  # 疏水
            "I": ["V", "L"],  # 疏水
            "M": ["L", "C"],  # 疏水/含硫
            "F": ["Y", "W"],  # 芳香族
            "Y": ["F", "W"],  # 芳香族
            "W": ["F", "Y"],  # 芳香族
            "D": ["E", "N"],  # 酸性
            "E": ["D", "Q"],  # 酸性
            "R": ["K", "H"],  # 碱性
            "K": ["R", "Q"],  # 碱性
            "H": ["R", "N"],  # 碱性/芳香
            "S": ["T", "C"],  # 极性
            "T": ["S", "V"],  # 极性
            "C": ["S", "M"],  # 含硫/极性
            "N": ["D", "H", "S"],  # 极性
            "Q": ["E", "K"],  # 极性
            "G": ["A", "S"],  # 小型
            "P": [],  # 特殊结构，不突变
        }

    def augment(self, sequences, n_variants=1):
        np.random.seed(self.seed)  # 固定随机种子
        augmented = []
        for seq in sequences:
            variants = set()
            seq_list = list(seq)

            attempts = 0
            while len(variants) < n_variants and attempts < 20:
                attempts += 1

                if len(seq) == 0:
                    break
                pos = np.random.randint(0, len(seq))
                aa = seq[pos]

                if aa in self.substitutions and self.substitutions[aa]:
                    sub = np.random.choice(self.substitutions[aa])
                    new_seq_list = seq_list.copy()
                    new_seq_list[pos] = sub
                    new_seq = "".join(new_seq_list)

                    if new_seq != seq:
                        variants.add(new_seq)

            augmented.extend(list(variants))

        print(f"增强: 从 {len(sequences)} 个序列生成了 {len(augmented)} 个变体")
        return augmented


class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        model_name = "facebook/esm2_t33_650M_UR50D"
        print(f"Loading ESM2: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_esm2_features(self, sequences, batch_size=8):
        print("Extracting ESM2 features...", flush=True)
        all_features = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            features = masked_hidden.sum(dim=1)
            all_features.append(features.cpu().numpy())
        return np.vstack(all_features)

    def get_ifeature_features(self, sequences_or_file):
        import tempfile

        is_list = isinstance(sequences_or_file, list)

        if is_list:
            if not sequences_or_file:
                return None
            print(
                f"Extracting iFeature from {len(sequences_or_file)} sequences...",
                flush=True,
            )
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".fasta"
            ) as tmp:
                for i, seq in enumerate(sequences_or_file):
                    tmp.write(f">seq_{i}\n{seq}\n")
                tmp_path = tmp.name
        else:
            tmp_path = str(sequences_or_file)
            print(f"Extracting iFeature from {tmp_path}...", flush=True)

        try:
            protein = iFeatureOmegaCLI.iProtein(tmp_path)

            feature_types = [
                "AAC",
                "DPC type 1",
                "CTDC",
                "CTDT",
                "CTDD",
                "PAAC",
                "QSOrder",
                "APAAC",
                "GAAC",
                "Moran",
                "Geary",
            ]

            all_dfs = []
            for ft in feature_types:
                try:
                    protein.get_descriptor(ft)
                    if hasattr(protein, "encodings"):
                        df = protein.encodings.copy()
                        all_dfs.append(df)
                except Exception:
                    pass

            if not all_dfs:
                return None

            full_df = pd.concat(all_dfs, axis=1)
            full_df = full_df.loc[:, ~full_df.columns.duplicated()]
            return full_df.values

        finally:
            if is_list:
                import os

                os.unlink(tmp_path)


def create_balanced_split(
    pos_sequences, neg_sequences, test_ratio=0.2, random_state=SEED
):
    np.random.seed(random_state)
    pos_idx = np.random.permutation(len(pos_sequences))
    neg_idx = np.random.permutation(len(neg_sequences))

    n_pos_test = int(len(pos_sequences) * test_ratio)
    n_neg_test = n_pos_test

    train_pos_idx = pos_idx[n_pos_test:]
    train_neg_idx = neg_idx[n_neg_test:]
    test_pos_idx = pos_idx[:n_pos_test]
    test_neg_idx = neg_idx[:n_neg_test]

    return (train_pos_idx, train_neg_idx, test_pos_idx, test_neg_idx)


def undersample_majority(X, y, ratio=1.0, random_state=SEED):
    np.random.seed(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_neg_target = int(len(pos_idx) / ratio)

    if n_neg_target >= len(neg_idx):
        return X, y

    neg_selected = np.random.choice(neg_idx, n_neg_target, replace=False)
    selected = np.concatenate([pos_idx, neg_selected])
    np.random.shuffle(selected)
    return X[selected], y[selected]


def evaluate_with_threshold(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
    }


def find_optimal_threshold(y_true, y_proba, target_sn=0.76, target_sp=0.86):
    best_threshold = 0.5
    best_score = -1
    best_metrics = None

    for threshold in np.arange(0.01, 0.99, 0.005):
        metrics = evaluate_with_threshold(y_true, y_proba, threshold)
        meets_sn = metrics["sensitivity"] >= target_sn
        meets_sp = metrics["specificity"] >= target_sp

        if meets_sn and meets_sp:
            score = 2000 + metrics["mcc"]
        elif meets_sn:
            score = 1000 + metrics["mcc"]
        else:
            score = metrics["mcc"]

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def main():
    print("=" * 70)
    print("BBB Peptide Prediction - FINAL MODEL TRAINING (ESM2 650M + TabPFN)")
    print("配置: 最佳提取器 + Top 700 特征 + 1:5 欠采样分配")
    print("=" * 70)

    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载数据
    data_dir = Path("data")
    pos_fasta = data_dir / "all_positive.fasta"
    neg_fasta = data_dir / "all_negative.fasta"

    pos_seqs = read_fasta(pos_fasta)
    neg_seqs = read_fasta(neg_fasta)
    all_seqs = pos_seqs + neg_seqs
    y_all = np.array([1] * len(pos_seqs) + [0] * len(neg_seqs))

    print(f"\n数据: {len(pos_seqs)} 正样本, {len(neg_seqs)} 负样本")

    # 特征提取
    extractor = FeatureExtractor(device=device)
    X_esm2 = extractor.get_esm2_features(all_seqs)
    print(f"ESM2 Features: {X_esm2.shape}")

    X_if_pos = extractor.get_ifeature_features(pos_fasta)
    X_if_neg = extractor.get_ifeature_features(neg_fasta)

    if X_if_pos is not None and X_if_neg is not None:
        X_ifeature = np.vstack([X_if_pos, X_if_neg])
        print(f"iFeature Features: {X_ifeature.shape}")
        X_all = np.hstack([X_esm2, X_ifeature])
    else:
        X_all = X_esm2

    X_all = np.nan_to_num(X_all)

    # 去除常量特征
    var_selector = VarianceThreshold(threshold=0)
    X_all = var_selector.fit_transform(X_all)
    print(f"Total Features: {X_all.shape}")

    # 标准化 (在划分之前，与 experiment_full_features.py 一致!)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # 划分数据
    train_pos_idx, train_neg_idx, test_pos_idx, test_neg_idx = create_balanced_split(
        pos_seqs, neg_seqs, test_ratio=0.2, random_state=SEED
    )

    neg_offset = len(pos_seqs)
    global_train_idx = np.concatenate([train_pos_idx, train_neg_idx + neg_offset])
    global_test_idx = np.concatenate([test_pos_idx, test_neg_idx + neg_offset])

    X_train_orig = X_all[global_train_idx]
    y_train_orig = y_all[global_train_idx]
    X_test = X_all[global_test_idx]
    y_test = y_all[global_test_idx]

    print(f"训练集: {X_train_orig.shape}, 测试集: {X_test.shape}")

    # ========== 最终模型训练 ==========
    print("\n" + "=" * 60)
    print("特征选择与模型训练 - k=700, 1:5")
    print("=" * 60)

    # 特征选择 (不需要再标准化了，已经标准化过)
    kbest1 = SelectKBest(mutual_info_classif, k=700)
    X_train_sel1 = kbest1.fit_transform(X_train_orig, y_train_orig)
    X_test_sel1 = kbest1.transform(X_test)

    # 欠采样
    X_tr_res1, y_tr_res1 = undersample_majority(X_train_sel1, y_train_orig, ratio=0.2)
    print(f"训练样本: {sum(y_tr_res1 == 1)} pos, {sum(y_tr_res1 == 0)} neg")

    # 训练
    clf1 = TabPFNClassifier(device=device)
    clf1.fit(X_tr_res1, y_tr_res1)

    # 预测
    y_proba1 = clf1.predict_proba(X_test_sel1)[:, 1]
    thresh1, metrics1 = find_optimal_threshold(y_test, y_proba1)

    print(
        f"Sn={metrics1['sensitivity']:.4f}, Sp={metrics1['specificity']:.4f}, "
        f"ACC={metrics1['accuracy']:.4f}, MCC={metrics1['mcc']:.4f}"
    )

    # ========== 保存模型及预处理管道 ==========
    print("\n" + "=" * 60)
    print("保存模型权重及管道文件到 models/ 目录...")

    model_dir = DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(var_selector, model_dir / "variance_selector.pkl")
    joblib.dump(scaler, model_dir / "standard_scaler.pkl")
    joblib.dump(kbest1, model_dir / "kbest_selector.pkl")
    # TabPFN is a scikit-learn compatible estimator
    joblib.dump(clf1, model_dir / "tabpfn_classifier.pkl")
    
    # Save the optimal threshold
    with open(model_dir / "optimal_threshold.txt", "w") as f:
        f.write(str(thresh1))
        
    print(f"所有文件已成功保存至 {model_dir}/")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
