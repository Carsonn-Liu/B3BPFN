# B3BPFN

B3BPFN is a reproducible repository for blood-brain barrier-penetrating peptide (BBBP) prediction. The current pipeline combines ESM2 sequence embeddings, iFeatureOmega physicochemical descriptors, mutual-information-based feature selection, and a TabPFN classifier.

The repository includes:

- the curated benchmark dataset used in the current experiments
- the training script for rebuilding the model pipeline
- the inference script for scoring new peptide sequences
- the exported preprocessing objects and trained classifier

## Overview

The released workflow follows four stages:

1. Encode peptide sequences with `facebook/esm2_t33_650M_UR50D`.
2. Extract handcrafted descriptors with `iFeatureOmegaCLI`.
3. Fuse features, remove constant dimensions, standardize them, and keep the top 700 features with mutual information.
4. Train a `TabPFNClassifier` on an undersampled training set and apply the saved threshold for final prediction.

Current benchmark scale in this repository:

- Positive peptides: `426`
- Negative peptides: `6865`
- Feature space before selection: `2121`
- Retained features: `700`

Reported performance from the current internal benchmark:

- Sensitivity: `0.9294`
- Specificity: `0.8824`
- Accuracy: `0.9059`
- MCC: `0.8127`
- AUROC: `0.9460`

## Repository Layout

```text
B3BPFN/
├── data/
│   ├── all_negative.fasta
│   └── all_positive.fasta
├── models/
│   ├── kbest_selector.pkl
│   ├── optimal_threshold.txt
│   ├── standard_scaler.pkl
│   ├── tabpfn_classifier.pkl
│   └── variance_selector.pkl
├── tests/
│   └── test_predict_cli.py
├── predict_peptide.py
├── train_peptide_final.py
└── README.md
```

## Environment

This project is intended to run on a Linux server or a compute node with a configured Python environment. At minimum, you will need:

- Python 3.10 or newer
- `numpy`
- `pandas`
- `torch`
- `joblib`
- `scikit-learn`
- `transformers`
- `tabpfn`
- `iFeatureOmegaCLI`

Notes:

- The ESM2 backbone is loaded from Hugging Face on first use.
- Training and large-batch inference are best run on a GPU-enabled server.
- `iFeatureOmegaCLI` installation can vary by environment; follow your server setup conventions if you already maintain a shared bioinformatics stack.

## Data

The benchmark data are stored in FASTA format:

- `data/all_positive.fasta`: curated BBB-penetrating peptides
- `data/all_negative.fasta`: curated non-BBB-penetrating peptides

The current training script builds a balanced test set by holding out 20% of the positive peptides and sampling the same number of negatives.

## Training

Run training from the repository root:

```bash
python3 train_peptide_final.py
```

The script will:

- read the FASTA files under `data/`
- extract ESM2 and iFeatureOmega features
- perform variance filtering, scaling, and top-700 feature selection
- undersample the negative class to a 1:5 positive-to-negative ratio
- train the TabPFN classifier
- write the resulting artifacts to `models/`

## Inference

Prepare an input FASTA file such as:

```fasta
>pep_1
RRWQWRMKKLG
>pep_2
ACDEFGHIKLMNP
```

Then run:

```bash
python3 predict_peptide.py -i input.fasta -o bbb_predictions.csv
```

Optional arguments:

- `-m, --model_dir`: path to the saved model directory, default is `models`
- `-o, --output`: output CSV path, default is `bbb_predictions.csv`

The output CSV contains:

- `ID`
- `Sequence`
- `Probability`
- `Prediction` (`BBB+` or `BBB-`)

## Reproducibility Notes

- The training script uses a fixed random seed of `42`.
- The exported objects in `models/` are the exact preprocessing and classification artifacts used by the inference script.
- `tabpfn_classifier.pkl` is tracked with Git LFS because of its size.

## Verification

This repository currently includes a lightweight regression test to ensure that the default inference model path stays aligned with the published repository layout:

```bash
python3 -m unittest tests/test_predict_cli.py
```

## Public Release Status

This repository is being prepared as a code release first. Paper metadata, citation information, and author details are intentionally omitted at this stage.
