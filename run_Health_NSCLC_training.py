import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from glob import glob
from pathlib import Path
import sqlite3
import pickle
from collections import defaultdict, OrderedDict
import logging
import argparse


import h5py
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import fisher_exact
import torch
from transformers import AutoTokenizer, EsmModel

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_PATH = Path('./TCR_files/')
MODEL_SAVE_PATH = Path('./Health_NSCLC_model.pkl')
ESM_MODEL_PATH = Path('./esm/esm2_t30_150M_UR50D') 


TCR_ONLY_MODE = True

# --- Global settings tailored for TCR analysis ---
RANDOM_STATE = 42
MIN_SEQS_PER_PARTICIPANT = 20 # 每个参与者在QC后需要保留的最小序列数
REQUIRED_TCRB_CLONE_COUNT = 100 #【MODIFIED】降低了TCRB克隆数要求以适应新数据，您可以根据需要调整

# Cache paths
TCR_CACHE_PATH = DATA_PATH / 'tcr_processed.parquet'
EMBEDDINGS_PATH = DATA_PATH / 'cdr3_embeddings.h5'
EMBEDDINGS_INDEX_PATH = DATA_PATH / 'cdr3_embeddings.db'

# --- Helper functions ---
def mutation_rate(series):
    return pd.Series(0.0, index=series.index)


# --- 【MODIFIED】 Core data processing functions ---

def prepare_raw_tcr_data(df: pd.DataFrame) -> pd.DataFrame:

    # 1. 检查并处理关键输入列
    required_input_cols = ['v_call', 'j_call', 'acdr3', 'dataset_subgroup']
    for col in required_input_cols:
        if col not in df.columns:
            raise KeyError(f"输入TSV文件缺少必需的列 '{col}'。请检查您的数据。")

    if 'clone_id' not in df.columns:
        # logger.info("列 'clone_id' 未找到。正在从 v_call, j_call, 和 acdr3 生成克隆ID。")
        # 使用astype(str)确保拼接时不会因数据类型问题而出错
        df['clone_id'] = (df['v_call'].astype(str) + '_' + 
                          df['j_call'].astype(str) + '_' + 
                          df['acdr3'].astype(str))


    if 'productive' not in df.columns:
        df['productive'] = 't'
    if 'v_score' not in df.columns:
        df['v_score'] = 100.0 


    if 'cdr3_aa' not in df.columns:
        df.rename(columns={'acdr3': 'cdr3_aa'}, inplace=True)
    
    df['specimen_label'] = df['dataset_subgroup'].str.split('_').str[-1]

    required_output_cols = ['v_call', 'j_call', 'clone_id', 'sequence_id', 'cdr3_aa', 'specimen_label', 'productive', 'v_score']
    if 'sequence_id' not in df.columns:
        df['sequence_id'] = df.index.astype(str)

    for col in required_output_cols:
        if col not in df.columns:
            raise ValueError(f"预处理后，必需的输出列 '{col}' 丢失了。检查函数逻辑。")
            
    return df


def apply_final_tcr_qc(df: pd.DataFrame) -> pd.DataFrame:

    df.rename(columns={'v_call': 'v_segment', 'j_call': 'j_segment', 'clone_id': 'igh_or_tcrb_clone_id', 'cdr3_aa': 'cdr3_seq_aa_q'}, inplace=True)
    
    df = df[df['productive'] == 't'].copy()
    df['v_score'] = pd.to_numeric(df['v_score'], errors='coerce').fillna(0)
    df = df[df['v_score'] > 80]
    
    df['isotype_supergroup'] = "TCRB"
    df['v_gene'] = df['v_segment'].str.split('*').str[0].astype('category')
    df['j_gene'] = df['j_segment'].str.split('*').str[0].astype('category')
    
    df['cdr3_seq_aa_q'] = df['cdr3_seq_aa_q'].str.slice(start=1, stop=-1).replace(r"^\s*$", np.nan, regex=True)
    df.dropna(subset=['cdr3_seq_aa_q'], inplace=True)
    df.rename(columns={'cdr3_seq_aa_q': 'cdr3_seq_aa_q_trim'}, inplace=True)

    df['v_mut'] = 0.0

    processed_participants = []
    for participant_label, participant_df in tqdm(df.groupby('participant_label'), desc="  Applying participant-level QC", leave=False):
        participant_df = participant_df[participant_df["cdr3_seq_aa_q_trim"].str.len() >= 8]
        if participant_df.empty: continue
        
        clone_counts = participant_df['igh_or_tcrb_clone_id'].nunique()
        if clone_counts < REQUIRED_TCRB_CLONE_COUNT:
            logger.warning(f"Participant {participant_label} dropped. Has {clone_counts} TCRB clones, requires {REQUIRED_TCRB_CLONE_COUNT}.")
            continue
            
        if participant_df.shape[0] < MIN_SEQS_PER_PARTICIPANT:
             logger.warning(f"Participant {participant_label} dropped. Has {participant_df.shape[0]} sequences after filtering, requires {MIN_SEQS_PER_PARTICIPANT}.")
             continue
        
        # 每个克隆只保留一条序列以避免偏见
        participant_df = participant_df.sort_values('sequence_id').groupby("igh_or_tcrb_clone_id", as_index=False, observed=True).first()
        processed_participants.append(participant_df)
        
    return pd.concat(processed_participants, ignore_index=True) if processed_participants else pd.DataFrame()


# --- 【MODIFIED】 DataLoader ---
class DataLoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.tcr_cache_path = self.base_path / 'tcr_processed.parquet'
        meta_path = self.base_path / 'participant_NSCLC.tsv'
        if not meta_path.exists():
            raise FileNotFoundError(f"元数据文件未找到: {meta_path}. 请遵循说明创建此文件。")
        self.meta_df = pd.read_csv(meta_path, sep='\t')
        if not all(col in self.meta_df.columns for col in ['specimen_label', 'participant_label', 'disease']):
            raise ValueError("元数据文件必须包含 'specimen_label', 'participant_label', 'disease' 列。")

    def _process_and_cache_data(self):
        print("缓存未找到。正在从原始TSV文件处理数据...")
        
        tcr_files = glob(str(self.base_path / 'TCR_files_NSCLC' / '*.tsv'))
        if not tcr_files:
            raise FileNotFoundError(f"在 {self.base_path / 'TCR_files_NSCLC'} 目录下未找到任何 .tsv 文件。")
        
        all_tcr_data = []
        pbar = tqdm(tcr_files, desc="Loading and Preparing Raw TSV files")
        for file_path in pbar:
            # 1. 读取原始数据
            raw_df = pd.read_csv(file_path, sep='\t', low_memory=False)
            
            # 2. 【关键步骤】准备数据格式 (添加/生成列，重命名)
            prepared_df = prepare_raw_tcr_data(raw_df)
            
            # 3. 与元数据合并，添加 participant_label 和 disease
            merged_df = pd.merge(prepared_df, self.meta_df, on='specimen_label', how='inner')
            if not merged_df.empty:
                all_tcr_data.append(merged_df)

        if not all_tcr_data:
            raise ValueError("所有文件处理后数据为空。请检查文件内容和元数据中的 'specimen_label' 是否匹配。")

        # 4. 合并所有文件的数据
        full_raw_df = pd.concat(all_tcr_data, ignore_index=True)
        
        # 5. 应用最终的QC流程
        tcr_final_df = apply_final_tcr_qc(full_raw_df)
        
        if tcr_final_df.empty:
            raise ValueError("数据在最终QC后为空。可能是因为没有参与者满足克隆数或序列数要求。请检查 'REQUIRED_TCRB_CLONE_COUNT' 和 'MIN_SEQS_PER_PARTICIPANT' 的设置。")

        # 6. 缓存处理好的数据
        tcr_final_df.to_parquet(self.tcr_cache_path)
        
        # 更新元数据，只保留通过QC的参与者
        final_participants = tcr_final_df['participant_label'].unique()
        self.meta_df = self.meta_df[self.meta_df['participant_label'].isin(final_participants)].reset_index(drop=True)
        
        print(f"数据处理和缓存完成。最终有 {len(final_participants)} 名参与者通过了QC。")
        return self.meta_df

    def ensure_cache_exists(self):
        if self.tcr_cache_path.exists():
            print(f"找到缓存文件 {self.tcr_cache_path}。正在加载已处理数据。")
            try:
                # 只加载 participant_label 列来快速获取通过QC的参与者列表
                cached_participants = pd.read_parquet(self.tcr_cache_path, columns=['participant_label'])['participant_label'].unique()
                self.meta_df = self.meta_df[self.meta_df['participant_label'].isin(cached_participants)].reset_index(drop=True)
                return self.meta_df
            except Exception as e:
                print(f"缓存文件无效或已损坏: {e}。正在重新处理...")
                return self._process_and_cache_data()
        else:
            return self._process_and_cache_data()


# --- Model 2: Parallelized Convergent Clustering ---
from joblib import Parallel, delayed
from multiprocessing import cpu_count

def _cluster_group_worker(group_key: tuple, group_df: pd.DataFrame, sequence_identity_threshold: float) -> List[Tuple[str, str]]:
    v_gene, j_gene, cdr3_len = group_key
    if len(group_df) <= 1:
        seq = group_df['cdr3_seq_aa_q_trim'].iloc[0]
        global_cluster_id = f"{v_gene}-{j_gene}-{cdr3_len}-0"
        return [(seq, global_cluster_id)]
    sequences = group_df['cdr3_seq_aa_q_trim'].tolist()
    sequences_numeric = [np.frombuffer(s.encode(), dtype=np.uint8) for s in sequences]
    sequences_padded = np.array([np.pad(s, (0, cdr3_len - len(s))) for s in sequences_numeric])
    dist_matrix = pdist(sequences_padded, 'hamming')
    Z = linkage(dist_matrix, method='single')
    cluster_labels = fcluster(Z, t=(1 - sequence_identity_threshold), criterion='distance')
    results = []
    for seq, label in zip(sequences, cluster_labels):
        global_cluster_id = f"{v_gene}-{j_gene}-{cdr3_len}-{label}"
        results.append((seq, global_cluster_id))
    return results

class Model2_ConvergentClustering_Optimized:
    def __init__(self, is_bcr=True, p_value_threshold=0.05, sequence_identity_threshold=0.85, n_jobs=-1):
        self.is_bcr = is_bcr
        self.p_value_threshold = p_value_threshold
        self.sequence_identity_threshold = sequence_identity_threshold
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.model = LogisticRegression(multi_class='ovr', class_weight='balanced', random_state=RANDOM_STATE)
        self.cluster_map_ = None
        self.enriched_clusters_ = None
        self.classes_ = None
        self.feature_columns_ = None

    def _precompute_cluster_mapping(self, df: pd.DataFrame):
        logger.info(f"      - [OPT] Pre-computing cluster map for {df['cdr3_seq_aa_q_trim'].nunique()} unique sequences using {self.n_jobs} cores...")
        unique_seq_df = df[['cdr3_seq_aa_q_trim', 'v_gene', 'j_gene']].drop_duplicates('cdr3_seq_aa_q_trim').copy()
        unique_seq_df['cdr3_len'] = unique_seq_df['cdr3_seq_aa_q_trim'].str.len()
        groups = list(unique_seq_df.groupby(['v_gene', 'j_gene', 'cdr3_len'], observed=True))
        pbar = tqdm(groups, desc="        - Performing parallel single-linkage clustering", leave=False, dynamic_ncols=True)
        results_list = Parallel(n_jobs=self.n_jobs)(
            delayed(_cluster_group_worker)(key, group, self.sequence_identity_threshold) for key, group in pbar
        )
        flat_results = [item for sublist in results_list for item in sublist]
        self.cluster_map_ = dict(flat_results)
        logger.info(f"      - [OPT] Cluster map created for {len(self.cluster_map_)} sequences.")

    def _find_enriched_clusters(self, df_with_clusters, y_df):
        logger.info("      - Finding enriched clusters using Fisher's Exact Test...")
        train_df = pd.merge(df_with_clusters, y_df, on='participant_label')
        all_participants = y_df['participant_label'].unique()
        disease_counts = y_df.groupby('disease')['participant_label'].nunique()
        self.classes_ = sorted(y_df['disease'].unique())
        participants_by_disease = {
            disease: set(y_df[y_df['disease'] == disease]['participant_label'])
            for disease in self.classes_
        }
        enriched_clusters = []
        cluster_pbar = tqdm(train_df.groupby('global_cluster_id'), desc="        - Performing Fisher's test on clusters", leave=False, dynamic_ncols=True)
        for cluster_id, cluster_group in cluster_pbar:
            participants_in_cluster_set = set(cluster_group['participant_label'].unique())
            if not participants_in_cluster_set: continue
            min_p_value, dominant_label = 1.0, None
            for disease in self.classes_:
                participants_with_this_disease = participants_by_disease[disease]
                total_with_disease = disease_counts.get(disease, 0)
                total_without_disease = len(all_participants) - total_with_disease
                a = len(participants_in_cluster_set.intersection(participants_with_this_disease))
                b = total_with_disease - a
                c = len(participants_in_cluster_set) - a
                d = total_without_disease - c
                _, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
                if p_value < min_p_value:
                    min_p_value = p_value
                    dominant_label = disease
            if min_p_value <= self.p_value_threshold:
                enriched_clusters.append({
                    'global_cluster_id': cluster_id,
                    'dominant_label': dominant_label,
                    'p_value': min_p_value
                })
        if not enriched_clusters:
            logger.warning("    No enriched clusters found for Model 2. The model will not be effective.")
            return pd.DataFrame(columns=['global_cluster_id', 'dominant_label', 'p_value'])
        return pd.DataFrame(enriched_clusters)

    def _featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.enriched_clusters_ is None or self.enriched_clusters_.empty:
            logger.warning("Model 2 was not trained or found no enriched clusters. Returning zero-feature matrix.")
            if self.classes_ is None: self.classes_ = []
            return pd.DataFrame(0, index=df['participant_label'].unique(), columns=self.classes_)
        logger.info("    (Featurizing) Assigning sequences to enriched clusters via pre-computed map...")
        df_with_clusters = df.copy()
        df_with_clusters['global_cluster_id'] = df_with_clusters['cdr3_seq_aa_q_trim'].map(self.cluster_map_)
        df_with_clusters.dropna(subset=['global_cluster_id'], inplace=True)
        enriched_cluster_ids = set(self.enriched_clusters_['global_cluster_id'])
        relevant_hits = df_with_clusters[df_with_clusters['global_cluster_id'].isin(enriched_cluster_ids)]
        hits_with_labels = pd.merge(relevant_hits, self.enriched_clusters_[['global_cluster_id', 'dominant_label']], on='global_cluster_id')
        feature_df = hits_with_labels.groupby(['participant_label', 'dominant_label']).size().unstack(fill_value=0)
        feature_df = feature_df.reindex(df['participant_label'].unique(), fill_value=0)
        feature_df = feature_df.reindex(columns=self.classes_, fill_value=0)
        return feature_df

    def fit(self, X_df, y_df):
        print(f"  Fitting Model 2 ({'BCR' if self.is_bcr else 'TCR'})...")
        if X_df.empty:
            logger.warning("  Model 2: Training data is empty, skipping fit.")
            self.classes_ = sorted(y_df['disease'].unique()) if not y_df.empty else []
            self.enriched_clusters_ = pd.DataFrame()
            return self
        self._precompute_cluster_mapping(X_df)
        X_df_with_clusters = X_df.copy()
        X_df_with_clusters['global_cluster_id'] = X_df_with_clusters['cdr3_seq_aa_q_trim'].map(self.cluster_map_)
        X_df_with_clusters.dropna(subset=['global_cluster_id'], inplace=True)
        self.enriched_clusters_ = self._find_enriched_clusters(X_df_with_clusters, y_df)
        if self.enriched_clusters_.empty:
            self.classes_ = sorted(y_df['disease'].unique())
            self.feature_columns_ = self.classes_
            return self
        print(f"    Found {len(self.enriched_clusters_)} enriched clusters. Now training classifier on hit counts...")
        feature_df = self._featurize(X_df)
        merged_df = feature_df.join(y_df.set_index('participant_label'), how='inner')
        y_train, X_train = merged_df['disease'], merged_df.drop(columns=['disease'])
        self.feature_columns_ = X_train.columns
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X_df):
        if self.enriched_clusters_ is None or self.enriched_clusters_.empty:
            num_classes = len(self.classes_) if self.classes_ is not None else 0
            participants = X_df['participant_label'].unique()
            if num_classes == 0 or not participants.any(): return pd.DataFrame(index=participants)
            return pd.DataFrame(np.ones((len(participants), num_classes)) / num_classes, index=participants, columns=self.classes_)
        feature_df = self._featurize(X_df)
        X = feature_df.reindex(columns=self.feature_columns_, fill_value=0)
        probas = self.model.predict_proba(X)
        return pd.DataFrame(probas, columns=self.model.classes_, index=X.index)

# --- Model 1 ---
class Model1_Composition:
    def __init__(self, is_bcr=True):
        self.is_bcr = is_bcr
        self.pca = PCA(n_components=15, random_state=RANDOM_STATE)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(multi_class='ovr', class_weight='balanced', random_state=RANDOM_STATE, C=0.1)

    def _featurize(self, df_group, v_genes, j_genes):
        features = []
        desc = f"Featurizing Model 1 ({'BCR' if self.is_bcr else 'TCR'})"
        for participant, group in tqdm(df_group, desc=desc, leave=False, dynamic_ncols=True):
            participant_features = {'participant_label': participant}
            vj_counts = group.groupby(['v_gene', 'j_gene'], observed=True).size().unstack(fill_value=0)
            vj_counts = vj_counts.reindex(index=v_genes, columns=j_genes, fill_value=0)
            participant_features.update({f'vj_{i}': c for i, c in enumerate((vj_counts.values / (vj_counts.values.sum() + 1e-9)).flatten())})
            if self.is_bcr:
                participant_features['shm_median'] = group['v_mut'].median()
                participant_features['shm_mean'] = group['v_mut'].mean()
                participant_features['shm_std'] = group['v_mut'].std(ddof=0)
                isotype_props = group['isotype_supergroup'].value_counts(normalize=True)
                for iso in ['IGHA', 'IGHG', 'IGHD-M']: participant_features[f'isotype_{iso}'] = isotype_props.get(iso, 0)
            features.append(participant_features)
        return pd.DataFrame(features).set_index('participant_label').fillna(0)

    def fit(self, X_df, y_df):
        print(f"  Fitting Model 1 ({'BCR' if self.is_bcr else 'TCR'})...")
        if X_df.empty:
            logger.warning("  Model 1: Training data is empty, skipping fit.")
            self.v_genes_, self.j_genes_, self.feature_columns_ = [], [], []
            self.pca = None
            return self

        self.v_genes_ = sorted(X_df['v_gene'].cat.categories)
        self.j_genes_ = sorted(X_df['j_gene'].cat.categories)
        
        feature_df = self._featurize(X_df.groupby('participant_label'), self.v_genes_, self.j_genes_)
        merged_df = feature_df.join(y_df.set_index('participant_label'), how='inner')
        y, X = merged_df['disease'], merged_df.drop(columns=['disease'])
        self.feature_columns_ = X.columns
        X_scaled = self.scaler.fit_transform(X)
        n_components_pca = min(15, X_scaled.shape[1], X_scaled.shape[0]-1)
        if n_components_pca > 0:
            self.pca = PCA(n_components=n_components_pca, random_state=RANDOM_STATE)
            X_final = self.pca.fit_transform(X_scaled)
        else:
            self.pca = None
            X_final = X_scaled
        self.model.fit(X_final, y)
        return self

    def predict_proba(self, X_df):
        if not hasattr(self, 'v_genes_') or not self.v_genes_ or not self.v_genes_:
            num_classes = len(getattr(self.model, 'classes_', [0,1]))
            participants = X_df['participant_label'].unique()
            return pd.DataFrame(np.ones((len(participants), num_classes)) / num_classes, index=participants, columns=getattr(self.model, 'classes_', ['class_0', 'class_1']))

        feature_df = self._featurize(X_df.groupby('participant_label'), self.v_genes_, self.j_genes_)
        X = feature_df.reindex(columns=self.feature_columns_, fill_value=0)
        X_scaled = self.scaler.transform(X)
        X_final = self.pca.transform(X_scaled) if self.pca else X_scaled
        return pd.DataFrame(self.model.predict_proba(X_final), columns=self.model.classes_, index=X.index)

class OptimizedEmbeddingCache:
    def __init__(self, h5_path, db_path, cache_size_mb=2048):
        self.h5_path = h5_path
        self.db_path = db_path
        self.cache_size_mb = cache_size_mb
        self.memory_cache = OrderedDict()
        self.memory_usage = 0
        self.max_memory_bytes = cache_size_mb * 1024 * 1024
        self.h5_file = None
        self.db_conn = None
        self._init_database()
        self._load_metadata()

    def _init_database(self):
        self.db_conn = sqlite3.connect(self.db_path)
        self.db_conn.execute('CREATE TABLE IF NOT EXISTS embeddings (cdr3_seq TEXT PRIMARY KEY, chunk_id INTEGER, index_in_chunk INTEGER)')
        self.db_conn.execute('CREATE INDEX IF NOT EXISTS idx_cdr3 ON embeddings(cdr3_seq)')

    def _load_metadata(self):
        if not self.h5_path.exists():
            self.embedding_dim, self.zero_embedding = 0, None
            return
        with h5py.File(self.h5_path, 'r') as f:
            if 'metadata' in f:
                self.embedding_dim = f['metadata'].attrs['embedding_dim']
            else:
                chunk_keys = [k for k in f.keys() if k.startswith('chunk_')]
                self.embedding_dim = f[chunk_keys[0]].shape[1] if chunk_keys else 0
            self.zero_embedding = np.zeros(self.embedding_dim, dtype=np.float32) if self.embedding_dim > 0 else None

    def _get_h5_file(self):
        if self.h5_file is None: self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file

    def _manage_memory_cache(self, new_size):
        while self.memory_usage + new_size > self.max_memory_bytes and self.memory_cache:
            oldest_key, old_embedding = self.memory_cache.popitem(last=False)
            self.memory_usage -= old_embedding.nbytes

    def get(self, cdr3_seq, default=None):
        if self.zero_embedding is None: return default
        if cdr3_seq in self.memory_cache: return self.memory_cache[cdr3_seq]
        cursor = self.db_conn.execute('SELECT chunk_id, index_in_chunk FROM embeddings WHERE cdr3_seq = ?', (cdr3_seq,))
        result = cursor.fetchone()
        if result is None: return default if default is not None else self.zero_embedding
        chunk_id, index_in_chunk = result
        try:
            h5_file = self._get_h5_file()
            embedding = h5_file[f'chunk_{chunk_id}'][index_in_chunk].astype(np.float32)
            self._manage_memory_cache(embedding.nbytes)
            self.memory_cache[cdr3_seq] = embedding
            self.memory_usage += embedding.nbytes
            return embedding
        except Exception as e:
            logger.error(f"Error loading embedding for {cdr3_seq}: {e}")
            return default if default is not None else self.zero_embedding

    def batch_get(self, cdr3_list):
        if self.zero_embedding is None or not cdr3_list: return np.array([])
        results = [None] * len(cdr3_list)
        missing_seqs_map = defaultdict(list)
        for i, cdr3_seq in enumerate(cdr3_list):
            if cdr3_seq in self.memory_cache:
                results[i] = self.memory_cache[cdr3_seq]
            else:
                missing_seqs_map[cdr3_seq].append(i)
        missing_seqs = list(missing_seqs_map.keys())
        if not missing_seqs: return np.array(results)
        db_results, batch_size = {}, 900
        db_query_pbar = tqdm(range(0, len(missing_seqs), batch_size), desc="  Querying DB for embeddings", leave=False, dynamic_ncols=True)
        for i in db_query_pbar:
            batch_of_seqs = missing_seqs[i:i + batch_size]
            cursor = self.db_conn.execute(f"SELECT cdr3_seq, chunk_id, index_in_chunk FROM embeddings WHERE cdr3_seq IN ({','.join('?'*len(batch_of_seqs))})", batch_of_seqs)
            for row in cursor.fetchall(): db_results[row[0]] = (row[1], row[2])
        chunk_loads = defaultdict(list)
        for seq in missing_seqs:
            if seq in db_results: chunk_loads[db_results[seq][0]].append((seq, db_results[seq][1]))
        h5_file, loaded_embeddings = self._get_h5_file(), {}
        h5_load_pbar = tqdm(chunk_loads.items(), desc="  Loading embeddings from HDF5", leave=False, total=len(chunk_loads), dynamic_ncols=True)
        for chunk_id, seq_indices in h5_load_pbar:
            try:
                indices_in_chunk = [idx for _, idx in seq_indices]
                sorted_indices, original_order_map = zip(*sorted(enumerate(indices_in_chunk), key=lambda x: x[1]))
                embeddings_from_chunk = h5_file[f'chunk_{chunk_id}'][list(original_order_map)].astype(np.float32)
                for new_pos, (seq, _) in enumerate(seq_indices):
                    embedding = embeddings_from_chunk[sorted_indices.index(new_pos)]
                    loaded_embeddings[seq] = embedding
                    self._manage_memory_cache(embedding.nbytes)
                    self.memory_cache[seq] = embedding
                    self.memory_usage += embedding.nbytes
            except Exception as e: logger.error(f"Error loading chunk {chunk_id}: {e}")
        for seq, original_indices in missing_seqs_map.items():
            for i in original_indices: results[i] = loaded_embeddings.get(seq, self.zero_embedding)
        return np.array(results)

    def close(self):
        if self.h5_file: self.h5_file.close(); self.h5_file = None
        if self.db_conn: self.db_conn.close(); self.db_conn = None

def precompute_and_save_embeddings(tcr_cache_path, model_path, h5_output_path, db_output_path, device='cpu', chunk_size=10000):
    print(f"\n--- Starting Optimized Pre-computation ---")
    tcr_cdr3s = pd.read_parquet(tcr_cache_path, columns=['cdr3_seq_aa_q_trim'])['cdr3_seq_aa_q_trim']
    all_cdr3s = tcr_cdr3s.dropna().unique().tolist()

    all_cdr3s = [s for s in all_cdr3s if isinstance(s, str) and 4 < len(s) < 50]
    if not all_cdr3s:
        print("No valid CDR3 sequences. Skipping embedding generation.")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path).to(device)
    model.eval()
    db_conn = sqlite3.connect(db_output_path)
    # 将一行拆分为两行
    db_conn.execute('PRAGMA journal_mode=WAL;')
    db_conn.execute('PRAGMA synchronous=NORMAL;')

    db_conn.execute('CREATE TABLE IF NOT EXISTS embeddings (cdr3_seq TEXT PRIMARY KEY, chunk_id INTEGER, index_in_chunk INTEGER)')
    db_conn.execute('CREATE INDEX IF NOT EXISTS idx_cdr3 ON embeddings(cdr3_seq)')
    batch_size, embedding_dim, current_chunk_embeddings, current_chunk_seqs, chunk_id = 64, None, [], [], 0
    with h5py.File(h5_output_path, 'w') as hf, torch.no_grad():
        for i in tqdm(range(0, len(all_cdr3s), batch_size), desc="Generating Embeddings", dynamic_ncols=True):
            batch_seqs = all_cdr3s[i:i+batch_size]
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=50).to(device)
            batch_embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
            if embedding_dim is None: embedding_dim = batch_embeddings.shape[1]
            current_chunk_embeddings.append(batch_embeddings)
            current_chunk_seqs.extend(batch_seqs)
            if len(current_chunk_seqs) >= chunk_size:
                chunk_data = np.vstack(current_chunk_embeddings)
                hf.create_dataset(f'chunk_{chunk_id}', data=chunk_data, compression='gzip', compression_opts=4)
                index_data = [(seq, chunk_id, idx) for idx, seq in enumerate(current_chunk_seqs)]
                db_conn.executemany('INSERT OR REPLACE INTO embeddings (cdr3_seq, chunk_id, index_in_chunk) VALUES (?, ?, ?)', index_data)
                db_conn.commit()
                current_chunk_embeddings, current_chunk_seqs = [], []
                chunk_id += 1
        if current_chunk_seqs:
            chunk_data = np.vstack(current_chunk_embeddings)
            hf.create_dataset(f'chunk_{chunk_id}', data=chunk_data, compression='gzip', compression_opts=4)
            index_data = [(seq, chunk_id, idx) for idx, seq in enumerate(current_chunk_seqs)]
            db_conn.executemany('INSERT OR REPLACE INTO embeddings (cdr3_seq, chunk_id, index_in_chunk) VALUES (?, ?, ?)', index_data)
            db_conn.commit()
        metadata_group = hf.create_group('metadata')
        metadata_group.attrs['embedding_dim'] = embedding_dim if embedding_dim is not None else 0
    db_conn.close()
    print("--- Pre-computation Complete ---")

# --- Model 3 ---
class Model3_LanguageModel:
    def __init__(self, embeddings_h5_path, embeddings_db_path, is_bcr=True):
        self.is_bcr = is_bcr
        print(f"  Initializing optimized Model 3 ({'BCR' if is_bcr else 'TCR'})...")
        self.embedding_cache = OptimizedEmbeddingCache(embeddings_h5_path, embeddings_db_path)
        self.seq_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1, max_depth=10, min_samples_leaf=5)
        self.agg_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)

    def fit(self, X_df, y_df):
        print(f"  Fitting Model 3 ({'BCR' if self.is_bcr else 'TCR'})...")
        train_df = X_df
        if train_df.empty:
            logger.warning("  Model 3: Training data is empty, skipping fit."); self.seq_model, self.agg_model = None, None
            return self
        
        print("    Stage 1: Training sequence-level model...")
        unique_cdr3s_df = train_df[['cdr3_seq_aa_q_trim', 'disease']].drop_duplicates('cdr3_seq_aa_q_trim')
        if len(unique_cdr3s_df) > 100000: unique_cdr3s_df = unique_cdr3s_df.sample(n=100000, random_state=RANDOM_STATE)
        print(f"      Training on {len(unique_cdr3s_df)} unique CDR3s...")
        X_seq_embed = self.embedding_cache.batch_get(unique_cdr3s_df['cdr3_seq_aa_q_trim'].tolist())
        if X_seq_embed.size == 0 or len(np.unique(unique_cdr3s_df['disease'])) < 2:
            logger.warning("    Skipping Stage 1 training due to no embeddings or <2 classes."); self.seq_model = None
        else:
            self.seq_model.fit(X_seq_embed, unique_cdr3s_df['disease'])
            
        print("    Stage 2: Training patient-level aggregation model...")
        if self.seq_model is None:
            logger.warning("    Skipping Stage 2 training as sequence model is not available."); self.agg_model = None; return self
        print(f"      Generating sequence predictions for {len(train_df)} sequences...")
        train_embeddings = self.embedding_cache.batch_get(train_df['cdr3_seq_aa_q_trim'].tolist())
        seq_probas = self.seq_model.predict_proba(train_embeddings)
        proba_df = pd.DataFrame(seq_probas, columns=self.seq_model.classes_, index=train_df.index)
        proba_df['participant_label'] = train_df['participant_label']
        agg_mean = proba_df.groupby('participant_label').mean()
        agg_std = proba_df.groupby('participant_label').std(ddof=0).fillna(0)
        agg_features = agg_mean.join(agg_std, lsuffix='_mean', rsuffix='_std')
        merged_df = agg_features.join(y_df.set_index('participant_label'), how='inner')
        y_agg, X_agg = merged_df['disease'], merged_df.drop(columns=['disease'])
        self.agg_model_classes_ = self.seq_model.classes_
        self.agg_feature_columns_ = X_agg.columns
        self.agg_model.fit(X_agg, y_agg)
        return self

    def predict_proba(self, X_df):
        if self.agg_model is None or self.seq_model is None:
            num_classes = len(getattr(self, 'agg_model_classes_', [0,1]))
            participants = X_df['participant_label'].unique()
            return pd.DataFrame(np.ones((len(participants), num_classes)) / num_classes, index=participants, columns=getattr(self, 'agg_model_classes_', ['class_0', 'class_1']))

        test_embeddings = self.embedding_cache.batch_get(X_df['cdr3_seq_aa_q_trim'].tolist())
        seq_probas = self.seq_model.predict_proba(test_embeddings)
        proba_df = pd.DataFrame(seq_probas, columns=self.seq_model.classes_, index=X_df.index)
        proba_df['participant_label'] = X_df['participant_label']
        agg_mean = proba_df.groupby('participant_label').mean()
        agg_std = proba_df.groupby('participant_label').std(ddof=0).fillna(0)
        agg_features = agg_mean.join(agg_std, lsuffix='_mean', rsuffix='_std')
        X_agg = agg_features.reindex(columns=self.agg_feature_columns_, fill_value=0)
        probas = self.agg_model.predict_proba(X_agg)
        return pd.DataFrame(probas, columns=self.agg_model.classes_, index=X_agg.index)

    def close(self):
        if self.embedding_cache:
            self.embedding_cache.close()
# --- Ensemble Class ---
class TCR_Ensemble:
    def __init__(self, embeddings_h5_path, embeddings_db_path):
        self.embeddings_h5_path, self.embeddings_db_path = embeddings_h5_path, embeddings_db_path
        self.base_models = {
            'tcr1': Model1_Composition(is_bcr=False),
            'tcr2': Model2_ConvergentClustering_Optimized(is_bcr=False),
            'tcr3': Model3_LanguageModel(self.embeddings_h5_path, self.embeddings_db_path, is_bcr=False)
        }
        self.ensemble_model = LogisticRegression(multi_class='ovr', class_weight='balanced', random_state=RANDOM_STATE)
        self.label_encoder = LabelEncoder()

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'tcr3' in state.get('base_models', {}):
            model3_instance = state['base_models']['tcr3']
            if hasattr(model3_instance, 'embedding_cache'):
                model3_instance.embedding_cache = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        logger.info("Re-initializing embedding caches after loading from pickle...")
        if 'tcr3' in self.base_models:
             self.base_models['tcr3'].embedding_cache = OptimizedEmbeddingCache(self.embeddings_h5_path, self.embeddings_db_path)

    def fit(self, tcr_train, meta_train, tcr_val, meta_val):
        print("\n--- Starting TCR Ensemble Training ---")
        y_df = pd.concat([meta_train, meta_val])[['participant_label', 'disease']].copy()
        self.label_encoder.fit(y_df['disease'])
        self.classes_ = self.label_encoder.classes_
        print(f"  Base model training on {meta_train['participant_label'].nunique()} participants, ensemble validation on {meta_val['participant_label'].nunique()} participants.")
        
        y_base_train, y_ensemble_train = meta_train[['participant_label', 'disease']], meta_val[['participant_label', 'disease']]
        
        print("\n--- Training Base Models ---")
        for name, model in self.base_models.items():
            model.fit(tcr_train, y_base_train)
            
        print("\n--- Generating Base Model Predictions for Ensemble Training ---")
        ensemble_train_participants = meta_val['participant_label'].unique()
        val_preds = []
        for name, model in self.base_models.items():
            preds = model.predict_proba(tcr_val) if not tcr_val.empty else pd.DataFrame(np.ones((len(ensemble_train_participants), len(self.classes_))) / len(self.classes_), index=ensemble_train_participants, columns=self.classes_)
            preds = preds.reindex(ensemble_train_participants).fillna(1.0/len(self.classes_))
            preds.columns = [f'{name}_{c}' for c in preds.columns]
            val_preds.append(preds)
            
        ensemble_X = pd.concat(val_preds, axis=1).fillna(0)
        ensemble_train_df = ensemble_X.join(y_ensemble_train.set_index('participant_label'), how='inner')
        if ensemble_train_df.empty:
            logger.error("Ensemble training dataframe is empty. Cannot fit ensemble model."); return self
            
        y_ensemble = self.label_encoder.transform(ensemble_train_df['disease'])
        X_ensemble = ensemble_train_df.drop(columns=['disease'])
        self.ensemble_feature_columns_ = X_ensemble.columns
        self.ensemble_model.fit(X_ensemble, y_ensemble)
        print("--- Ensemble Training Complete ---")
        return self

    def predict_proba(self, tcr_df):
        all_participants = tcr_df['participant_label'].unique()
        if len(all_participants) == 0: return pd.DataFrame(columns=self.classes_, dtype=float)
        
        test_preds = []
        for name, model in self.base_models.items():
            preds_present = model.predict_proba(tcr_df)
            preds_reindexed = preds_present.reindex(all_participants).fillna(1.0 / len(self.classes_))
            preds_reindexed.columns = [f'{name}_{c}' for c in preds_reindexed.columns]
            test_preds.append(preds_reindexed)
            
        ensemble_X = pd.concat(test_preds, axis=1).fillna(0)
        ensemble_X = ensemble_X.reindex(columns=self.ensemble_feature_columns_, fill_value=0)
        final_probas = self.ensemble_model.predict_proba(ensemble_X)
        return pd.DataFrame(final_probas, columns=self.classes_, index=ensemble_X.index)

    def close_resources(self):
        for model in self.base_models.values():
            if model and hasattr(model, 'close'): model.close()


def enforce_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure that key columns have 'category' dtype."""
    if df.empty: return df
    df = df.copy()
    if 'v_gene' in df.columns: df['v_gene'] = df['v_gene'].astype('category')
    if 'j_gene' in df.columns: df['j_gene'] = df['j_gene'].astype('category')
    return df

# --- Evaluation Helper Function ---
def evaluate_and_print_results(model, tcr_df, meta_df, description=""):
    print("\n" + "="*20 + f" {description} " + "="*20)
    if meta_df.empty or meta_df['participant_label'].nunique() == 0:
        print("No data to evaluate."); return
        
    predictions_proba = model.predict_proba(tcr_df)
    y_true_df = meta_df[['participant_label', 'disease']].drop_duplicates().set_index('participant_label')
    results_df = predictions_proba.join(y_true_df, how='inner')
    
    if results_df.empty:
        print("Could not join predictions with true labels. Evaluation skipped."); return
        
    y_true, y_pred_proba = results_df['disease'], results_df.drop(columns=['disease'])
    if y_pred_proba.empty or y_true.empty:
        print("No valid predictions or labels to evaluate."); return
        
    y_pred = y_pred_proba.idxmax(axis=1)
    print(f"Evaluation on {len(y_true)} participants.")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    
    y_true_encoded = model.label_encoder.transform(y_true)
    y_pred_proba_ordered = y_pred_proba[model.label_encoder.classes_]
    
    if len(np.unique(y_true_encoded)) > 1:
        try:
            auroc = roc_auc_score(y_true_encoded, y_pred_proba_ordered, multi_class='ovr', average='weighted')
            print(f"Weighted OvR AUROC: {auroc:.4f}")
        except ValueError as e:
            print(f"Could not compute AUROC: {e}")
            
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=model.classes_, target_names=model.classes_, zero_division=0))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    print(pd.DataFrame(cm, index=model.classes_, columns=model.classes_))
    print("="* (42 + len(description)))


# --- Main Execution Block ---
if __name__ == '__main__':
    # 清理旧缓存，确保从新数据开始
    for p in [TCR_CACHE_PATH, EMBEDDINGS_PATH, EMBEDDINGS_INDEX_PATH, MODEL_SAVE_PATH]:
        if p.exists(): 
            print(f"Removing old cache/model file: {p}")
            os.remove(p)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\nStep 1: Loading data and ensuring cache exists...")
    data_loader = DataLoader(base_path=DATA_PATH)
    meta_df = data_loader.ensure_cache_exists()
    
    if meta_df.empty or meta_df['participant_label'].nunique() < 3: # 需要足够的数据进行分割
        raise SystemExit(f"Error: Not enough participants ({meta_df['participant_label'].nunique()}) passed QC to proceed with training. Please check your data and QC parameters.")

    print(f"Loaded metadata for {meta_df['participant_label'].nunique()} participants passing QC.")
    print("Disease distribution:\n", meta_df.drop_duplicates('participant_label')['disease'].value_counts())

    if not EMBEDDINGS_PATH.exists() or not EMBEDDINGS_INDEX_PATH.exists():
        print("\nStep 2: Pre-computing embeddings (one-time setup)...")
        precompute_and_save_embeddings(
            tcr_cache_path=TCR_CACHE_PATH, model_path=ESM_MODEL_PATH, 
            h5_output_path=EMBEDDINGS_PATH, db_output_path=EMBEDDINGS_INDEX_PATH, 
            device=device
        )
    else:
        print("\nStep 2: Found existing embeddings.")

    print("\nStep 3: Splitting data into train/validation/test sets...")
    participants = meta_df['participant_label'].unique()
    labels = meta_df.drop_duplicates('participant_label').set_index('participant_label').loc[participants]['disease']
    
    # 60% train, 20% validation, 20% test
    # First split: 80% for train/val, 20% for test
    try:
        train_val_participants, test_participants, y_train_val, y_test = train_test_split(
            participants, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
        
        # Second split: From the 80%, take 75% for train and 25% for validation (0.75*0.8=0.6, 0.25*0.8=0.2)
        train_participants, val_participants, _, _ = train_test_split(
            train_val_participants, y_train_val, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_val)
    except ValueError as e:
         raise SystemExit(f"Error during data splitting: {e}. There might not be enough samples in each class to perform a stratified split. Please check the disease distribution.")


    meta_train = meta_df[meta_df['participant_label'].isin(train_participants)]
    meta_val = meta_df[meta_df['participant_label'].isin(val_participants)]
    meta_test = meta_df[meta_df['participant_label'].isin(test_participants)]
    
    print("Loading data splits from Parquet cache...")
    full_tcr_df = pd.read_parquet(TCR_CACHE_PATH)
    
    tcr_train_df = full_tcr_df[full_tcr_df['participant_label'].isin(train_participants)].pipe(enforce_categorical_dtypes)
    tcr_val_df = full_tcr_df[full_tcr_df['participant_label'].isin(val_participants)].pipe(enforce_categorical_dtypes)
    tcr_test_df = full_tcr_df[full_tcr_df['participant_label'].isin(test_participants)].pipe(enforce_categorical_dtypes)
    
    print(f"Split sizes: Train={len(train_participants)}, Validation={len(val_participants)}, Test={len(test_participants)}")

    model, loaded_model = None, None
    try:
        print("\nStep 4: Training the TCR Ensemble Model...")
        model = TCR_Ensemble(embeddings_h5_path=EMBEDDINGS_PATH, embeddings_db_path=EMBEDDINGS_INDEX_PATH)
        model.fit(
            tcr_train=tcr_train_df, meta_train=meta_train,
            tcr_val=tcr_val_df, meta_val=meta_val
        )
        
        print(f"\nStep 5: Saving the trained model to {MODEL_SAVE_PATH}...")
        model.close_resources() # 重要：在序列化前关闭文件句柄
        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(model, f)
        print("Model saved successfully.")

        print("\nStep 6: Loading model from disk to demonstrate usage...")
        with open(MODEL_SAVE_PATH, 'rb') as f:
            loaded_model = pickle.load(f)
        print("Model loaded successfully.")

        print("\nStep 7: Evaluating the model on all data splits...")
        evaluate_and_print_results(loaded_model, tcr_train_df, meta_train, "Training Set Performance")
        evaluate_and_print_results(loaded_model, tcr_val_df, meta_val, "Validation Set Performance")
        evaluate_and_print_results(loaded_model, tcr_test_df, meta_test, "Test Set Performance")

    finally:
        print("\nCleaning up model resources...")
        if model is not None:
            model.close_resources()
        if loaded_model is not None:
            loaded_model.close_resources()
        print("Cleanup complete.")
