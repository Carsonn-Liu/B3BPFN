"""
BBB Peptide Prediction - Inference Script
Uses the trained ESM2 650M + TabPFN model to predict BBB permeability for new peptides.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from transformers import AutoModel, AutoTokenizer
import iFeatureOmegaCLI

warnings.filterwarnings("ignore")

DEFAULT_MODEL_DIR = "models"


def read_fasta(file_path):
    sequences = []
    ids = []
    current_seq = None
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                ids.append(line[1:])
                if current_seq is not None:
                    sequences.append(current_seq)
                current_seq = ""
            else:
                if current_seq is not None:
                    current_seq += line
        if current_seq is not None and current_seq:
            sequences.append(current_seq)
    
    if len(ids) != len(sequences):
        # Fallback if names are missing
        ids = [f"Seq_{i+1}" for i in range(len(sequences))]
        
    return ids, sequences


class FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # MUST MATCH THE TRAINING SCRIPT
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
        import os

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
            if is_list and os.path.exists(tmp_path):
                os.unlink(tmp_path)


def predict(fasta_file, model_dir, output_file):
    print("=" * 60)
    print(f"Predicting BBB Permeability for peptides in: {fasta_file}")
    print("=" * 60)
    
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"Error: Model directory '{model_dir}' not found.")
        print("Please run train_peptide_final.py first to generate the models.")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Read FASTA
    ids, sequences = read_fasta(fasta_file)
    if not sequences:
        print("No valid sequences found in the input file.")
        sys.exit(1)
    
    print(f"Loaded {len(sequences)} sequences to predict.")
    
    # 2. Extract Features
    extractor = FeatureExtractor(device=device)
    X_esm2 = extractor.get_esm2_features(sequences)
    X_ifeature = extractor.get_ifeature_features(sequences)
    
    if X_ifeature is None:
        print("Error: iFeature extraction failed.")
        sys.exit(1)
        
    X_raw = np.hstack([X_esm2, X_ifeature])
    X_raw = np.nan_to_num(X_raw)
    
    # 3. Load Pipeline and Process Features
    print("\nLoading pre-trained pipeline models...")
    try:
        var_selector = joblib.load(model_path / "variance_selector.pkl")
        scaler = joblib.load(model_path / "standard_scaler.pkl")
        kbest = joblib.load(model_path / "kbest_selector.pkl")
        clf = joblib.load(model_path / "tabpfn_classifier.pkl")
        
        with open(model_path / "optimal_threshold.txt", "r") as f:
            threshold = float(f.read().strip())
    except Exception as e:
        print(f"Error loading model components: {e}")
        sys.exit(1)
        
    print("Applying feature transformations...")
    # Apply VarianceThreshold
    try:
        X_trans = var_selector.transform(X_raw)
    except Exception as e:
        print(f"Warning: Extracted features shape {X_raw.shape} mismatches expected variance selector shape. Reverting to empty array.")
        sys.exit(1)
        
    # Scale
    X_scaled = scaler.transform(X_trans)
    
    # SelectKBest
    X_final = kbest.transform(X_scaled)
    print(f"Final input feature shape: {X_final.shape}")
    
    # 4. Predict
    print(f"\nRunning TabPFN Predictions (Threshold: {threshold:.3f})...")
    # clf.to(device) # TabPFN might already be on right device, or handles it internally
    y_proba = clf.predict_proba(X_final)[:, 1]
    
    y_pred = (y_proba >= threshold).astype(int)
    
    # 5. Output Results
    results = []
    for i in range(len(ids)):
        pred_label = "BBB+" if y_pred[i] == 1 else "BBB-"
        results.append({
            "ID": ids[i],
            "Sequence": sequences[i],
            "Probability": y_proba[i],
            "Prediction": pred_label
        })
        
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 40)
    print("PREDICTION SUMMARY")
    print("=" * 40)
    print(f"Total Peptides: {len(df)}")
    print(f"BBB+ (Permeable): {sum(df['Prediction'] == 'BBB+')}")
    print(f"BBB- (Non-Permeable): {sum(df['Prediction'] == 'BBB-')}")
    
    df.to_csv(output_file, index=False)
    print(f"\nDetailed predictions saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict BBB permeability of peptides.")
    parser.add_argument("-i", "--input", required=True, help="Input FASTA file containing peptide sequences.")
    parser.add_argument("-o", "--output", default="bbb_predictions.csv", help="Output CSV file name.")
    parser.add_argument(
        "-m",
        "--model_dir",
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the trained models.",
    )
    
    args = parser.parse_args()
    predict(args.input, args.model_dir, args.output)
