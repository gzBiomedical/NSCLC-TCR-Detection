import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu, entropy

import umap

import matplotlib.pyplot as plt
import seaborn as sns
import logomaker
from adjustText import adjust_text

from run_health_NSCLC_training import (TCR_Ensemble, Model1_Composition, Model2_ConvergentClustering_Optimized,
                                     Model3_LanguageModel, OptimizedEmbeddingCache)

DATA_PATH = Path('./TCR_files/')
MODEL_PATH = DATA_PATH / 'Health_NSCLC_model.pkl'
TCR_CACHE_PATH = DATA_PATH / 'tcr_processed.parquet'
META_PATH = DATA_PATH / 'participant_NSCLC.tsv'
OUTPUT_PATH = Path('./output/')

RANDOM_STATE = 42
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_plotting_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams.update({
        'figure.figsize': (10, 7),
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'savefig.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight'
    })
    OUTPUT_PATH.mkdir(exist_ok=True)

def get_predictions(model, tcr_df, meta_df):
    if tcr_df.empty or meta_df.empty:
        return None, None, None, None
    
    participants = meta_df['participant_label'].unique()
    tcr_subset = tcr_df[tcr_df['participant_label'].isin(participants)]
    
    if tcr_subset.empty:
        return None, None, None, None
        
    y_true_series = meta_df.drop_duplicates('participant_label').set_index('participant_label')['disease']
    
    predictions_proba = model.predict_proba(tcr_subset)
    results_df = predictions_proba.join(y_true_series, how='inner')
    
    y_true = results_df['disease']
    y_pred_proba = results_df.drop(columns=['disease'])
    y_pred = y_pred_proba.idxmax(axis=1)
    
    return y_true, y_pred, y_pred_proba, model.classes_

def _get_model3_patient_features(model3, tcr_df):

    if not hasattr(model3, 'seq_model') or not hasattr(model3, 'agg_model') or not model3.seq_model or not model3.agg_model:
        return pd.DataFrame()
        
    test_embeddings = model3.embedding_cache.batch_get(tcr_df['cdr3_seq_aa_q_trim'].tolist())
    if test_embeddings.size == 0:
        return pd.DataFrame()

    seq_probas = model3.seq_model.predict_proba(test_embeddings)
    proba_df = pd.DataFrame(seq_probas, columns=model3.seq_model.classes_, index=tcr_df.index)
    proba_df['participant_label'] = tcr_df['participant_label']
    
    agg_mean = proba_df.groupby('participant_label').mean()
    agg_std = proba_df.groupby('participant_label').std(ddof=0).fillna(0)
    agg_features = agg_mean.join(agg_std, lsuffix='_mean', rsuffix='_std')
    
    agg_features = agg_features.reindex(columns=model3.agg_feature_columns_, fill_value=0)
    
    return agg_features


def plot_performance_metrics_table(model, splits):
    metrics_data = []
    for name, (tcr_df, meta_df) in splits.items():
        y_true, y_pred, y_pred_proba, classes = get_predictions(model, tcr_df, meta_df)
        if y_true is None or y_true.empty: 
            continue

        pos_label = classes[1] if 'Health' in classes[0] else classes[0]
        
        y_true_binary = (y_true == pos_label).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0)
        precision = precision_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, average='binary', zero_division=0)
        
        auroc, auprc = np.nan, np.nan
        if len(y_true_binary.unique()) > 1:
            auroc = roc_auc_score(y_true_binary, y_pred_proba[pos_label])
            auprc = average_precision_score(y_true_binary, y_pred_proba[pos_label])
        
        metrics_data.append({
            'Split': name,
            'Accuracy': f"{acc:.3f}",
            'F1 Score': f"{f1:.3f}",
            'Precision': f"{precision:.3f}",
            'Recall (Sensitivity)': f"{recall:.3f}",
            'AUROC': f"{auroc:.3f}" if not np.isnan(auroc) else "N/A",
            'AUPRC': f"{auprc:.3f}" if not np.isnan(auprc) else "N/A",
        })
    
    if not metrics_data:
        return

    metrics_df = pd.DataFrame(metrics_data).set_index('Split')
    print("性能指标:")
    print(metrics_df)
    
    fig, ax = plt.subplots(figsize=(12, max(2, 0.5 * len(metrics_df))))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, rowLabels=metrics_df.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.5)
    plt.title("Model Performance Metrics", y=1.1, fontsize=20)
    plt.savefig(OUTPUT_PATH / "1_performance_metrics_table.png")
    plt.close()
    metrics_df.to_csv(OUTPUT_PATH / "1_performance_metrics.csv")


def plot_roc_and_pr_curves(model, tcr_test, meta_test):
    y_true, _, y_pred_proba, classes = get_predictions(model, tcr_test, meta_test)
    if y_true is None or y_true.empty:
        return
        
    pos_label = classes[1] if 'Health' in classes[0] else classes[0]
    y_true_binary = (y_true == pos_label).astype(int)
    
    if len(y_true_binary.unique()) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba[pos_label])
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2.5, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=16)
    ax1.set_ylabel('True Positive Rate', fontsize=16)
    ax1.set_title(f'Receiver Operating Characteristic (ROC)\nClass: {pos_label}', fontsize=18)
    ax1.legend(loc="lower right", fontsize=14)
    ax1.grid(True)

    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba[pos_label])
    avg_precision = average_precision_score(y_true_binary, y_pred_proba[pos_label])
    ax2.step(recall, precision, color='b', where='post', label=f'PR curve (AUPRC = {avg_precision:.3f})')
    ax2.set_xlabel('Recall', fontsize=16)
    ax2.set_ylabel('Precision', fontsize=16)
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_title(f'Precision-Recall Curve\nClass: {pos_label}', fontsize=18)
    ax2.legend(loc="upper right", fontsize=14)
    ax2.grid(True)
    
    plt.suptitle("Model Evaluation on Test Set", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_PATH / "2_roc_pr_curves.png")
    plt.close()

def plot_vj_usage(full_tcr_df, meta_df):
    
    meta_subset = meta_df[['participant_label', 'disease']].drop_duplicates()
    df = pd.merge(full_tcr_df, meta_subset, on='participant_label', how='left')
    
    if df.empty or 'disease' not in df.columns or df['disease'].isnull().all():
        return
        
    diseases = df['disease'].dropna().unique()
    
    fig, axes = plt.subplots(1, len(diseases), figsize=(12 * len(diseases), 10), sharey=True)
    if len(diseases) == 1: axes = [axes]

    for ax, disease in zip(axes, diseases):
        subset = df[df['disease'] == disease]
        vj_crosstab = pd.crosstab(subset['v_gene'], subset['j_gene'])
        
        top_v_count = min(20, len(vj_crosstab.index))
        top_j_count = min(15, len(vj_crosstab.columns))
        top_v = vj_crosstab.sum(axis=1).nlargest(top_v_count).index
        top_j = vj_crosstab.sum(axis=0).nlargest(top_j_count).index
        vj_crosstab_top = vj_crosstab.loc[top_v, top_j]

        vj_crosstab_norm = vj_crosstab_top.div(vj_crosstab_top.sum().sum()) * 100
        
        sns.heatmap(vj_crosstab_norm, ax=ax, cmap="viridis", annot=False)
        ax.set_title(f'V-J Gene Usage Frequency (%)\n{disease} Cohort', fontsize=18)
        ax.set_xlabel("J Gene", fontsize=16)
        ax.set_ylabel("V Gene" if ax == axes[0] else "", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "3_vj_gene_usage.png")
    plt.close()

def plot_clonal_grouping_dendrogram(model, full_tcr_df):
    model2 = model.base_models['tcr2']
    if model2.enriched_clusters_ is None or model2.enriched_clusters_.empty:
        return

    top_cluster_id = model2.enriched_clusters_.sort_values('p_value').iloc[0]['global_cluster_id']
    
    if 'global_cluster_id' not in full_tcr_df.columns or full_tcr_df['global_cluster_id'].isnull().all():
        full_tcr_df['global_cluster_id'] = full_tcr_df['cdr3_seq_aa_q_trim'].map(model2.cluster_map_)

    cluster_seqs = full_tcr_df[full_tcr_df['global_cluster_id'] == top_cluster_id]['cdr3_seq_aa_q_trim'].unique()
    
    if len(cluster_seqs) < 2:
        return
        
    if len(cluster_seqs) > 50:
        cluster_seqs = np.random.choice(cluster_seqs, 50, replace=False)

    seq_len = len(cluster_seqs[0])
    seqs_numeric = [np.frombuffer(s.encode(), dtype=np.uint8) for s in cluster_seqs]
    dist_matrix = pdist(np.array(seqs_numeric), 'hamming') * seq_len
    
    Z = linkage(dist_matrix, method='ward')
    
    plt.figure(figsize=(15, max(8, len(cluster_seqs) * 0.4)))
    dendrogram(Z, labels=cluster_seqs, orientation='left', leaf_font_size=16)
    plt.title(f"Dendrogram for Most Enriched Cluster\nID: {top_cluster_id}", fontsize=20)
    plt.xlabel("Distance (Ward)", fontsize=16)
    plt.ylabel("CDR3 Amino Acid Sequence", fontsize=16)
    plt.savefig(OUTPUT_PATH / "4_clonal_grouping_dendrogram.png")
    plt.close()

def plot_feature_importance(model):
    fig, axes = plt.subplots(3, 1, figsize=(12, 22), gridspec_kw={'hspace': 0.5})
    
    try:
        base_model_names = sorted(model.base_models.keys())
        coef_df_data = []
        for i, class_label in enumerate(model.ensemble_model.classes_):
            coefs = model.ensemble_model.coef_[0] if len(model.ensemble_model.classes_) <= 2 else model.ensemble_model.coef_[i]
            
            for bm_name in base_model_names:
                bm_indices = [j for j, col in enumerate(model.ensemble_feature_columns_) if col.startswith(bm_name)]
                if not bm_indices: continue
                importance = np.mean(np.abs(coefs[bm_indices]))
                coef_df_data.append({'Base Model': bm_name, 'Importance (Mean Abs Coef)': importance, 'Class': class_label})
        
        coef_df = pd.DataFrame(coef_df_data)
        
        sns.barplot(data=coef_df, x='Base Model', y='Importance (Mean Abs Coef)', hue='Class', ax=axes[0])
        axes[0].set_title('Ensemble: Base Model Importance', fontsize=18)
        axes[0].set_xlabel('Base Model', fontsize=16)
        axes[0].set_ylabel('Mean Absolute Coefficient', fontsize=16)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend(fontsize=14, title_fontsize=14)
    except Exception as e:
        axes[0].text(0.5, 0.5, f'Could not plot:\n{e}', ha='center')
        axes[0].set_title('Ensemble Model Importance', fontsize=18)

    try:
        model2 = model.base_models['tcr2']
        if model2.enriched_clusters_ is not None and not model2.enriched_clusters_.empty:
            top_clusters = model2.enriched_clusters_.sort_values('p_value').head(15)
            top_clusters['-log10(p_value)'] = -np.log10(top_clusters['p_value'])
            sns.barplot(x='-log10(p_value)', y='global_cluster_id', data=top_clusters, ax=axes[1], palette='viridis', hue='dominant_label', dodge=False)
            axes[1].set_title('Top 15 Enriched Clonal Groups (by p-value)', fontsize=18)
            axes[1].legend(title='Dominant in Class', title_fontsize='14', fontsize='12')
        else:
            axes[1].text(0.5, 0.5, 'No enriched clusters found.', ha='center')
            axes[1].set_title('Model 2: Enriched Clonal Groups', fontsize=18)
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Could not plot:\n{e}', ha='center')
        axes[1].set_title('Model 2: Enriched Clonal Groups', fontsize=18)

    try:
        model3 = model.base_models['tcr3']
        if hasattr(model3, 'agg_model') and hasattr(model3.agg_model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': model3.agg_feature_columns_,
                'importance': model3.agg_model.feature_importances_
            }).sort_values('importance', ascending=False)
            sns.barplot(x='importance', y='feature', data=importances, ax=axes[2], palette='mako')
            axes[2].set_title('Model 3: Patient-level Aggregation Feature Importance', fontsize=18)
        else:
             axes[2].text(0.5, 0.5, 'Model 3 not trained or has no features.', ha='center')
             axes[2].set_title('Model 3: Language Model Feature Importance', fontsize=18)
    except Exception as e:
        axes[2].text(0.5, 0.5, f'Could not plot:\n{e}', ha='center')
        axes[2].set_title('Model 3: Language Model Feature Importance', fontsize=18)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "5_feature_importance.png")
    plt.close()

def plot_repertoire_diversity(full_tcr_df, meta_df):
    
    def shannon_entropy(series):
        counts = series.value_counts()
        return entropy(counts)

    diversity_df = full_tcr_df.groupby('participant_label')['igh_or_tcrb_clone_id'].apply(shannon_entropy).reset_index()
    diversity_df.rename(columns={'igh_or_tcrb_clone_id': 'Shannon_Entropy'}, inplace=True)
    
    meta_subset = meta_df[['participant_label', 'disease']].drop_duplicates()
    plot_df = pd.merge(diversity_df, meta_subset, on='participant_label', how='inner')
    
    plt.figure(figsize=(8, 7))
    ax = sns.boxplot(x='disease', y='Shannon_Entropy', data=plot_df)
    sns.stripplot(x='disease', y='Shannon_Entropy', data=plot_df, color=".25", alpha=0.6, ax=ax)
    
    disease_groups = plot_df.groupby('disease')
    groups = [group['Shannon_Entropy'].dropna() for name, group in disease_groups]
    title_text = 'TCR Repertoire Diversity (Shannon Entropy)'
    if len(groups) == 2:
        try:
            stat, p_val = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            title_text += f'\nMann-Whitney U-test p-value: {p_val:.4f}'
        except ValueError:
            pass # Keep original title if test fails
            
    ax.set_title(title_text, fontsize=18)
    ax.set_xlabel("Disease State", fontsize=16)
    ax.set_ylabel("Shannon Entropy", fontsize=16)

    plt.savefig(OUTPUT_PATH / "6_repertoire_diversity.png")
    plt.close()

def plot_residue_preference(model, full_tcr_df):
    model2 = model.base_models['tcr2']
    if model2.enriched_clusters_ is None or model2.enriched_clusters_.empty:
        return
        
    top_cluster_id = model2.enriched_clusters_.sort_values('p_value').iloc[0]['global_cluster_id']
    if 'global_cluster_id' not in full_tcr_df.columns or full_tcr_df['global_cluster_id'].isnull().all():
         full_tcr_df['global_cluster_id'] = full_tcr_df['cdr3_seq_aa_q_trim'].map(model2.cluster_map_)

    cluster_seqs_series = full_tcr_df[full_tcr_df['global_cluster_id'] == top_cluster_id]['cdr3_seq_aa_q_trim']

    if cluster_seqs_series.empty:
        return
        
    logo_df = logomaker.alignment_to_matrix(sequences=cluster_seqs_series.tolist())
    
    fig, ax = plt.subplots(figsize=(15, 5))
    logomaker.Logo(logo_df, font_name='Arial Rounded MT Bold', ax=ax)
    ax.set_title(f'Residue Preference for Top Enriched Cluster\n{top_cluster_id}', fontsize=18)
    ax.set_xlabel("CDR3 Position", fontsize=16)
    ax.set_ylabel("Bits", fontsize=16)

    plt.savefig(OUTPUT_PATH / "7_residue_preference_logo.png")
    plt.close()

def plot_prediction_scores_boxplot(model, tcr_test, meta_test):

    y_true, _, y_pred_proba, classes = get_predictions(model, tcr_test, meta_test)
    if y_true is None or y_true.empty:

        return

    pos_label = classes[1] if 'Health' in classes[0] else classes[0]
    
    plot_df = pd.DataFrame({
        'True Label': y_true,
        f'Predicted Score for "{pos_label}"': y_pred_proba[pos_label]
    })
    
    plt.figure(figsize=(8, 7))
    ax = sns.boxplot(x='True Label', y=f'Predicted Score for "{pos_label}"', data=plot_df)
    sns.stripplot(x='True Label', y=f'Predicted Score for "{pos_label}"', data=plot_df, color=".25", alpha=0.6, ax=ax)
    
    ax.set_title('Distribution of Prediction Scores on Test Set', fontsize=18)
    ax.set_ylabel('Model Prediction Score', fontsize=16)
    ax.set_xlabel('Ground Truth', fontsize=16)
    
    plt.savefig(OUTPUT_PATH / "8_prediction_scores_boxplot.png")
    plt.close()
    

def plot_confusion_matrix_heatmap(model, tcr_test, meta_test):

    y_true, y_pred, _, classes = get_predictions(model, tcr_test, meta_test)
    if y_true is None or y_true.empty:

        return

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
    plt.title('Confusion Matrix on Test Set', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.savefig(OUTPUT_PATH / "9_confusion_matrix_heatmap.png")
    plt.close()

def plot_cdr3_length_distribution(full_tcr_df, meta_df):

    meta_subset = meta_df[['participant_label', 'disease']].drop_duplicates()
    df = pd.merge(full_tcr_df, meta_subset, on='participant_label', how='left')

    if df.empty or 'disease' not in df.columns or df['disease'].isnull().all():
        return
        
    df['cdr3_len'] = df['cdr3_seq_aa_q_trim'].str.len()
    
    plt.figure(figsize=(10, 7))
    ax = sns.violinplot(data=df, x='disease', y='cdr3_len', inner='quartile')
    ax.set_title('CDR3 Length Distribution by Cohort', fontsize=18)
    ax.set_xlabel('Disease State', fontsize=16)
    ax.set_ylabel('Trimmed CDR3 Length (amino acids)', fontsize=16)
    plt.savefig(OUTPUT_PATH / "10_cdr3_length_distribution.png")
    plt.close()


def plot_patient_umap(model, full_tcr_df, meta_df):
    
    model3 = model.base_models.get('tcr3')
    if model3 is None:
        return

    patient_features = _get_model3_patient_features(model3, full_tcr_df)
    
    if patient_features.empty:
        return
        
    y_true_series = meta_df.drop_duplicates('participant_label').set_index('participant_label')['disease']
    
    n_samples = patient_features.shape[0]
    n_neighbors = min(15, n_samples - 1)
    
    if n_neighbors < 2:
        return

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        random_state=RANDOM_STATE
    )
    embedding = reducer.fit_transform(patient_features)
    
    umap_df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'], index=patient_features.index)
    umap_df = umap_df.join(y_true_series)

    plt.figure(figsize=(10, 8))
    ax = sns.scatterplot(
        x='UMAP 1', 
        y='UMAP 2', 
        hue='disease', 
        data=umap_df, 
        palette='viridis', 
        s=100, 
        alpha=0.8
    )
    
    ax.set_title('UMAP of Patient Repertoires\n(based on Language Model Features)', fontsize=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=16)
    ax.set_ylabel('UMAP Dimension 2', fontsize=16)
    plt.legend(title='Disease State', fontsize=14, title_fontsize=14)
    plt.grid(True)
    
    plt.savefig(OUTPUT_PATH / "11_patient_repertoire_umap.png")
    plt.close()

if __name__ == '__main__':
    setup_plotting_style()
    
        
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise SystemExit(f"-----")
    
    full_tcr_df = pd.read_parquet(TCR_CACHE_PATH)
    meta_df = pd.read_csv(META_PATH, sep='\t')
    
    participants = meta_df.drop_duplicates('participant_label')['participant_label'].unique()
    labels = meta_df.drop_duplicates('participant_label').set_index('participant_label').loc[participants]['disease']
    
    try:
        train_val_participants, test_participants, y_train_val, _ = train_test_split(
            participants, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
        train_participants, val_participants, _, _ = train_test_split(
            train_val_participants, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val)
    except ValueError as e:
         raise SystemExit(f"-------")

    meta_train = meta_df[meta_df['participant_label'].isin(train_participants)]
    meta_val = meta_df[meta_df['participant_label'].isin(val_participants)]
    meta_test = meta_df[meta_df['participant_label'].isin(test_participants)]
    
    tcr_train_df = full_tcr_df[full_tcr_df['participant_label'].isin(train_participants)]
    tcr_val_df = full_tcr_df[full_tcr_df['participant_label'].isin(val_participants)]
    tcr_test_df = full_tcr_df[full_tcr_df['participant_label'].isin(test_participants)]
    
    data_splits = {
        'Train': (tcr_train_df, meta_train),
        'Validation': (tcr_val_df, meta_val),
        'Test': (tcr_test_df, meta_test)
    }

    
    try:
        plot_performance_metrics_table(model, data_splits)
        plot_roc_and_pr_curves(model, tcr_test_df, meta_test)
        plot_vj_usage(full_tcr_df, meta_df)
        plot_clonal_grouping_dendrogram(model, full_tcr_df.copy())
        plot_feature_importance(model)
        plot_repertoire_diversity(full_tcr_df, meta_df)
        plot_residue_preference(model, full_tcr_df.copy())
        plot_prediction_scores_boxplot(model, tcr_test_df, meta_test)
        
        plot_confusion_matrix_heatmap(model, tcr_test_df, meta_test)
        plot_cdr3_length_distribution(full_tcr_df, meta_df)
        
        plot_patient_umap(model, full_tcr_df, meta_df)

    except Exception as e:
        logger.error(f"======={e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(model, 'close_resources'):
            model.close_resources()

