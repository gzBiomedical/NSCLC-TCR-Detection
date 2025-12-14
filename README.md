------

# NSCLC Detection via TCR Repertoire Sequencing (Ensemble Framework)

## Project Overview

This project implements a **Multi-Modal Ensemble Learning Framework** designed to detect Non-Small Cell Lung Cancer (NSCLC) using **T-Cell Receptor (TCR) Repertoire Sequencing data**. The framework integrates three distinct computational approaches—statistical composition analysis, sequence clustering, and deep learning-based embeddings—to achieve robust classification of disease states.

## Core Architecture

The system utilizes an **Ensemble Learning** strategy comprising three base models:

1. **Model 1 (Repertoire Composition):**
   - Analyzes the usage frequency of V and J gene segments.
   - Applies PCA (Principal Component Analysis) to capture global repertoire shifts.
   - *Focus:* Macro-level statistics and V-J recombination biases.
2. **Model 2 (Convergent Clustering):**
   - Clusters CDR3 sequences based on Hamming distance to identify "public clones."
   - Uses Fisher's Exact Test to detect clusters significantly enriched in the disease cohort.
   - *Focus:* Antigen-specific sequence motifs and convergent evolution.
3. **Model 3 (Deep Language Model):**
   - Leverages the pre-trained **ESM-2 (esm2_t30_150M_UR50D)** protein language model to extract high-dimensional embeddings for CDR3 sequences.
   - Trains a sequence-level classifier and aggregates predictions to generate patient-level feature vectors.
   - *Focus:* Deep biophysical and physicochemical sequence properties.

**Ensemble Classifier:** A Logistic Regression meta-model combines the probabilistic outputs of the three base models to make the final prediction (Healthy vs. NSCLC).

------

## Directory Structure

Based on the provided scripts and screenshots, the recommended directory structure is as follows:

Plaintext

```
Project_Root/
├── Health_NSCLC_model.pkl        # [Output] The trained ensemble model
├── tcr_processed.parquet         # [Cache] Pre-processed TCR data (speeds up re-runs)
├── run_Health_NSCLC_training.py  # [Script] Main training pipeline
├── inference.py                  # [Script] Evaluation and visualization
├── TCR_files/                    # [Data] Raw input data directory
│   ├── NSCLC_T229.tsv            # Individual TCR sequencing files
│   ├── NSCLC_T230.tsv
│   ├── ...
│   └── participant_NSCLC.tsv  # Clinical metadata
└── esm/                        # [Dependency] Pre-trained ESM model weights
    └── esm2_t30_150M_UR50D/      # HuggingFace ESM-2 model directory
```

------

## Prerequisites

Ensure you have a Python environment (3.8+) with the following dependencies installed:

Bash

```
pip install pandas numpy scikit-learn scipy torch transformers h5py tqdm seaborn matplotlib umap-learn adjustText logomaker
```

- **GPU Support:** PyTorch with CUDA support is highly recommended for **Model 3**, as generating ESM-2 embeddings on a CPU is computationally intensive.
- **ESM Model:** The code expects a local copy of the ESM-2 model. Ensure the weights are located at `malid/esm2_t30_150M_UR50D` or update `ESM_MODEL_PATH` in the training script.

------

## Data Preparation

### 1. TCR Sequencing Files (`.tsv`)

Each sample must have a corresponding TSV file containing the following columns:

- `v_call`: V gene segment designation.
- `j_call`: J gene segment designation.
- `acdr3` (or `cdr3_aa`): CDR3 amino acid sequence.
- `dataset_subgroup`: Used to parse the `specimen_label`.

### 2. Metadata File (`participant_metadata_NSCLC.tsv`)

A tab-separated file linking samples to clinical labels:

- `specimen_label`: Matches the label derived from the TCR file.
- `participant_label`: Unique identifier for the patient.
- `disease`: Class label (e.g., `Health`, `NSCLC`).

------

## Usage

### 1. Training the Model

Run the training script to process data, generate embeddings, and train the ensemble.

Bash

```
python run_Health_NSCLC_training.py
```

- **Caching:** The first run will be slower as it generates `tcr_processed.parquet` and `cdr3_embeddings.h5`. Subsequent runs will load these from disk.
- **Artifacts:** The trained model object is saved as `Health_NSCLC_model.pkl`.

### 2. Inference & Evaluation

Once the model is trained, use the inference script to generate performance metrics and visualizations.

Bash

```
python inference.py
```

This script produces a comprehensive report in the `./output/` directory, including:

| **Output File**                    | **Description**                                              |
| ---------------------------------- | ------------------------------------------------------------ |
| `1_performance_metrics_table.png`  | Summary of Accuracy, F1, AUROC, Precision, and Recall.       |
| `2_roc_pr_curves.png`              | Receiver Operating Characteristic & Precision-Recall curves. |
| `3_vj_gene_usage.png`              | Heatmaps comparing V-J gene usage across cohorts.            |
| `4_clonal_grouping_dendrogram.png` | Dendrogram of the most significantly enriched sequence cluster. |
| `5_feature_importance.png`         | Feature contribution analysis for all three models.          |
| `6_repertoire_diversity.png`       | Shannon Entropy boxplots comparing diversity.                |
| `7_residue_preference_logo.png`    | SeqLogo showing amino acid preferences in top clusters.      |
| `8_prediction_scores_boxplot.png`  | Distribution of prediction probabilities vs. Ground Truth.   |
| `9_confusion_matrix_heatmap.png`   | Visual confusion matrix for the test set.                    |
| `11_patient_repertoire_umap.png`   | UMAP projection of patients based on Model 3 latent features. |

------

## Key Configurations

You can tune the following Quality Control (QC) parameters in `run_Health_NSCLC_training.py` to fit your dataset size:

- `MIN_SEQS_PER_PARTICIPANT`: Minimum number of unique CDR3s required per patient (Default: 20).
- `REQUIRED_TCRB_CLONE_COUNT`: Minimum clone count threshold (Default: 100).
- `TCR_ONLY_MODE`: Set to `True` for TCR data (disables BCR-specific features like somatic hypermutation).

------

