"""Shared helpers for the CCS2 clustering / engagement-prediction analysis.

This module consolidates the code that was duplicated across the three original
notebooks:

  * ``clustering.ipynb``                  (target-only features)
  * ``clustering_alignment.ipynb``        (+ target<->reference alignment features)
  * ``clustering_alignment_extend.ipynb`` (+ speaker features + cross-validated comparison)

``analysis.ipynb`` drives the whole pipeline through these functions with a single
``CONFIG`` switch.  Functions are intentionally kept faithful to the original notebook
cells (same PCA dims, same KMeans/GMM/Agglomerative settings, same RandomForest/CV setup)
so results reproduce.

Only pandas / numpy / scikit-learn / scipy / matplotlib / seaborn are required.
"""

import os
import pickle
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, accuracy_score,
    classification_report, confusion_matrix, adjusted_rand_score, f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import f_classif
from scipy.cluster.hierarchy import dendrogram, linkage

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
DNN_PCA_DIM = 100
AU_PCA_DIM = 20

# Feature-column groups (see clustering_alignment_extend.ipynb cell 23).
TARGET_ONLY_COLS = ['au_features_pca', 'dnn_feature_pca', 'manual_feature']
ALIGN_COLS = ['au_align_ref1', 'au_align_ref2', 'au_align_ref1_ref2',
              'dnn_align_ref1', 'dnn_align_ref2', 'dnn_align_ref1_ref2',
              'manual_diff_ref1', 'manual_diff_ref2', 'manual_diff_ref1_ref2']
COSINE_ALIGN_COLS = ['au_align_ref1', 'au_align_ref2', 'au_align_ref1_ref2',
                     'dnn_align_ref1', 'dnn_align_ref2', 'dnn_align_ref1_ref2']

# The 9-column alignment block bundled with the targets in the original
# ``target_feature_columns`` for clustering (alignment / extended configs).
ALIGNMENT_FEATURE_COLS = TARGET_ONLY_COLS + ALIGN_COLS

# Semantic groups by index into the 46-dim manual_feature vector
# (meta indices 0,7,8 = recording quality -> excluded). See extend cell 25.
HG_GROUPS = {'pose':     list(range(1, 7)),    # avg_pose(3)+pose_std(3)
             'motion':   list(range(9, 26)),   # velocities(12)+intensity(1)+move_class onehot(4)
             'nodshake': list(range(26, 38)),  # nods(7)+shakes(5)
             'contact':  list(range(38, 46))}  # chin/face/forehead(6)+hand_raise(2)


# Human-readable names for the 46-dim manual_feature vector, in the exact order
# preprocess_manual_features (feature_gathering.ipynb) builds it: json-expanded pose
# triplets, 4-way movement_classification one-hot, 3-way nod_type/shake_type one-hots,
# booleans -> 0/1. Pose/velocity triplets follow the yaw/pitch/roll convention used
# throughout the feature set. Index boundaries match HG_GROUPS.
MANUAL_FEATURE_NAMES = [
    "valid_frames",                                                       # 0  (meta)
    "avg_pose_yaw", "avg_pose_pitch", "avg_pose_roll",                    # 1-3  pose
    "pose_std_yaw", "pose_std_pitch", "pose_std_roll",                   # 4-6  pose
    "num_valid_transitions",                                              # 7  (meta)
    "avg_time_between_detections",                                        # 8  (meta)
    "avg_yaw_velocity", "avg_pitch_velocity", "avg_roll_velocity",        # 9-11  motion
    "max_yaw_velocity", "max_pitch_velocity", "max_roll_velocity",        # 12-14 motion
    "total_yaw_movement", "total_pitch_movement", "total_roll_movement",  # 15-17 motion
    "yaw_velocity_std", "pitch_velocity_std", "roll_velocity_std",        # 18-20 motion
    "movement_intensity",                                                 # 21    motion
    "movement_class=High", "movement_class=Moderate",                     # 22-23 motion (one-hot)
    "movement_class=Low", "movement_class=Static",                        # 24-25 motion (one-hot)
    "nod_count", "nod_frequency",                                         # 26-27 nod/shake
    "nod_type=none", "nod_type=occasional", "nod_type=frequent",          # 28-30 (one-hot)
    "fast_nod_count", "slow_nod_count",                                   # 31-32
    "shake_count", "shake_frequency",                                     # 33-34
    "shake_type=none", "shake_type=occasional", "shake_type=frequent",    # 35-37 (one-hot)
    "chin_rest_detected", "chin_rest_frame_ratio",                        # 38-39 contact
    "touching_face_detected", "touching_face_frame_ratio",               # 40-41 contact
    "support_forehead_detected", "support_forehead_frame_ratio",          # 42-43 contact
    "hand_raise_detected", "hand_raise_frame_ratio",                      # 44-45 contact
]
assert len(MANUAL_FEATURE_NAMES) == 46, len(MANUAL_FEATURE_NAMES)


def prettify_feature_names(names):
    """Map expanded column names to readable ones: manual_feature_<i> -> behaviour name,
    au_features_pca_<i> -> AU_pca_<i>, dnn_feature_pca_<i> -> DNN_pca_<i>; others unchanged."""
    out = []
    for n in names:
        body = n[len('manual_feature_'):] if n.startswith('manual_feature_') else None
        if body is not None and body.isdigit() and int(body) < len(MANUAL_FEATURE_NAMES):
            out.append(MANUAL_FEATURE_NAMES[int(body)])
        elif n.startswith('au_features_pca_') and n[len('au_features_pca_'):].isdigit():
            out.append('AU_pca_' + n[len('au_features_pca_'):])
        elif n.startswith('dnn_feature_pca_') and n[len('dnn_feature_pca_'):].isdigit():
            out.append('DNN_pca_' + n[len('dnn_feature_pca_'):])
        else:
            out.append(n)
    return out


def feature_family(col):
    """Map an expanded feature column to its semantic family (for importance aggregation)."""
    if col.startswith('au_features_pca'):                       return 'listener:AU'
    if col.startswith('dnn_feature_pca'):                       return 'listener:CLIP'
    if col.startswith('manual_feature_') and col.split('_')[-1].isdigit():
        i = int(col.split('_')[-1])
        for gn, ix in HG_GROUPS.items():
            if i in ix:                                         return f'listener:manual_{gn}'
        return 'listener:manual_meta'
    if col.startswith('au_align') or col.startswith('dnn_align'): return 'align:others'
    if col.startswith('ls_'):                                   return 'align:listener-speaker'
    if col.startswith('spk_ac_'):                               return 'speaker:acoustic'
    if col.startswith('spk_txt_'):                              return 'speaker:text'
    if col.startswith('spk_ctx_'):                              return 'speaker:context'
    if col in ('speaker_overlap_sec', 'spk_has_speaker'):       return 'speaker:meta'
    return 'other'


def family_importance(X_lab, y, plot=True):
    """RF impurity importance aggregated by feature family.

    NOTE: impurity importance is biased toward high-cardinality blocks (CLIP has ~100 dims) and does
    NOT reflect predictive value — the CV ablation is the authoritative "what helps". Returned for
    description only.
    """
    model = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, class_weight='balanced')
    model.fit(StandardScaler().fit_transform(X_lab), y)
    fams = [feature_family(c) for c in X_lab.columns]
    s = pd.Series(model.feature_importances_, index=X_lab.columns).groupby(fams).sum().sort_values(ascending=False)
    if plot:
        plt.figure(figsize=(8, max(4, len(s) * 0.3)))
        sns.barplot(x=s.values, y=s.index)
        plt.title('RF impurity importance by feature family (descriptive; dim-count biased)')
        plt.xlabel('summed impurity importance'); plt.tight_layout(); plt.show()
    return s


def set_seed(seed=RANDOM_STATE):
    import random
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_features(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def load_labels(label_path='./label_df_3.pickle'):
    with open(label_path, 'rb') as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Engagement label schemes  (3-class -> binary regroupings)
# --------------------------------------------------------------------------- #
# Raw `majority_label_3` values.
RAW_ACTIVE = 'Active engagement'
RAW_INTERMITTENT = 'Intermittent engagement'
RAW_DISENGAGED = 'Disengagement'

# The raw middle value 'Intermittent engagement' (an annotator hedge for "unsure") is renamed
# to 'Low engagement'. Three reductions of the 3-class annotation, by self-documenting name:
#   active_vs_rest        : Active vs (Low + Disengaged) — "is the listener actively/fully engaged?"
#                           balanced (~44/45) and learnable on ~89 labels.
#   engaged_vs_disengaged : (Active + Low) vs Disengaged — "any engagement vs none"; 81/19, the
#                           17 disengaged are unlearnable (RF collapses to the majority; disengaged F1=0).
#   confident_only        : Active vs Disengaged with the uncertain middle DROPPED (mapped to NaN);
#                           the cases annotators were confident about. Rows with NaN are excluded downstream.
LABEL_SCHEMES = {
    '3class':                {RAW_ACTIVE: 'Active engagement', RAW_INTERMITTENT: 'Low engagement',       RAW_DISENGAGED: 'Disengagement'},
    'active_vs_rest':        {RAW_ACTIVE: 'Active engagement', RAW_INTERMITTENT: 'Not actively engaged', RAW_DISENGAGED: 'Not actively engaged'},
    'engaged_vs_disengaged': {RAW_ACTIVE: 'Engaged',          RAW_INTERMITTENT: 'Engaged',              RAW_DISENGAGED: 'Disengaged'},
    'confident_only':        {RAW_ACTIVE: 'Active engagement', RAW_INTERMITTENT: np.nan,                 RAW_DISENGAGED: 'Disengagement'},
}


def apply_label_scheme(y, scheme='3class'):
    """Map raw 3-class engagement labels to the chosen scheme (see ``LABEL_SCHEMES``).

    ``confident_only`` maps the uncertain middle to NaN — callers should drop NaN rows
    (and align their feature matrix) before fitting.
    """
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {list(LABEL_SCHEMES)}")
    return pd.Series(y).map(LABEL_SCHEMES[scheme])


def speaker_column_groups(df):
    """Derive speaker column groups by prefix (extend cell 2).

    Returns a dict with keys ``ac``, ``txt``, ``ctx`` and ``speaker`` (the full
    list used as the speaker block, excluding the categorical ``speaker_id``).
    """
    ac = [c for c in df.columns if c.startswith('spk_ac_')]
    txt = [c for c in df.columns if c.startswith('spk_txt_')]
    ctx = [c for c in df.columns if c.startswith('spk_ctx_')]
    speaker = ['speaker_overlap_sec', 'spk_has_speaker'] + ac + txt + ctx
    return {'ac': ac, 'txt': txt, 'ctx': ctx, 'speaker': speaker}


# --------------------------------------------------------------------------- #
# PCA of DNN / AU embeddings  (clustering cells 3-4)
# --------------------------------------------------------------------------- #
def add_dnn_pca(df, n_components=DNN_PCA_DIM, plot=False):
    """PCA the CLIP/DNN embeddings of target + ref1 + ref2 into ``*_pca`` columns."""
    n = len(df)
    dnn = (df['dnn_feature'].tolist()
           + df['dnn_feature_ref1'].tolist()
           + df['dnn_feature_ref2'].tolist())
    dnn = np.array(dnn)
    pca = PCA(n_components)
    dnn_pca = pca.fit_transform(dnn)
    if plot:
        _plot_scree(pca, 'DNN')
    df['dnn_feature_pca'] = dnn_pca.tolist()[:n]
    df['dnn_feature_pca_ref1'] = dnn_pca.tolist()[n:n * 2]
    df['dnn_feature_pca_ref2'] = dnn_pca.tolist()[n * 2:]
    return df


def add_au_pca(df, n_components=AU_PCA_DIM, plot=False):
    """PCA the (standardized) Action-Unit features of target + ref1 + ref2."""
    n = len(df)
    cols = ['AU_feature', 'AU_feature_ref1', 'AU_feature_ref2']
    au = []
    for col in cols:
        if col in df.columns:
            au.extend(df[col].tolist())
    au = np.array(au)
    au_scaled = StandardScaler().fit_transform(au)
    pca = PCA(n_components=n_components)
    au_pca = pca.fit_transform(au_scaled)
    if plot:
        _plot_scree(pca, 'AU')
    df['au_features_pca'] = au_pca[:n].tolist()
    df['au_features_pca_ref1'] = au_pca[n:n * 2].tolist()
    df['au_features_pca_ref2'] = au_pca[n * 2:].tolist()
    return df


def _plot_scree(pca, name):
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel(f'Number of Principal Components ({name})')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'Cumulative Explained Variance ({name} features)')
    plt.grid(True)
    plt.show()


# --------------------------------------------------------------------------- #
# Alignment features  (clustering_alignment cell 5)
# --------------------------------------------------------------------------- #
def calculate_cosine_similarity(feature1, feature2):
    """Cosine similarity between two feature vectors (NaN if either is empty)."""
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    if feature1.ndim == 1:
        feature1 = feature1.reshape(1, -1)
    if feature2.ndim == 1:
        feature2 = feature2.reshape(1, -1)
    if feature1.size == 0 or feature2.size == 0:
        return np.nan
    return cosine_similarity(feature1, feature2)[0][0]


def add_alignment_features(df):
    """Add the 9 target<->reference alignment columns (6 cosine sims + 3 abs-diffs)."""
    align = pd.DataFrame(index=df.index)

    if all(c in df.columns for c in ['au_features_pca', 'au_features_pca_ref1', 'au_features_pca_ref2']):
        align['au_align_ref1'] = df.apply(
            lambda r: calculate_cosine_similarity(r['au_features_pca'], r['au_features_pca_ref1']), axis=1)
        align['au_align_ref2'] = df.apply(
            lambda r: calculate_cosine_similarity(r['au_features_pca'], r['au_features_pca_ref2']), axis=1)
        align['au_align_ref1_ref2'] = df.apply(
            lambda r: calculate_cosine_similarity(r['au_features_pca_ref1'], r['au_features_pca_ref2']), axis=1)

    if all(c in df.columns for c in ['dnn_feature_pca', 'dnn_feature_pca_ref1', 'dnn_feature_pca_ref2']):
        align['dnn_align_ref1'] = df.apply(
            lambda r: calculate_cosine_similarity(r['dnn_feature_pca'], r['dnn_feature_pca_ref1']), axis=1)
        align['dnn_align_ref2'] = df.apply(
            lambda r: calculate_cosine_similarity(r['dnn_feature_pca'], r['dnn_feature_pca_ref2']), axis=1)
        align['dnn_align_ref1_ref2'] = df.apply(
            lambda r: calculate_cosine_similarity(r['dnn_feature_pca_ref1'], r['dnn_feature_pca_ref2']), axis=1)

    manual_cols = ['manual_feature', 'manual_feature_ref1', 'manual_feature_ref2']
    if all(c in df.columns for c in manual_cols):
        align['manual_diff_ref1'] = df.apply(
            lambda r: np.abs(np.array(r['manual_feature']) - np.array(r['manual_feature_ref1'])), axis=1)
        align['manual_diff_ref2'] = df.apply(
            lambda r: np.abs(np.array(r['manual_feature']) - np.array(r['manual_feature_ref2'])), axis=1)
        align['manual_diff_ref1_ref2'] = df.apply(
            lambda r: np.abs(np.array(r['manual_feature_ref1']) - np.array(r['manual_feature_ref2'])), axis=1)

    return pd.concat([df, align], axis=1)


# --------------------------------------------------------------------------- #
# Train / test split + numeric feature matrices
# --------------------------------------------------------------------------- #
def split_train_test(df, label_df):
    """Train = unlabeled (for clustering); test = the human-labeled segments."""
    test_ids = label_df['segment_id'].unique()
    train_df = df[~df['segment_id'].isin(test_ids)].copy().reset_index(drop=True)
    test_df = df[df['segment_id'].isin(test_ids)].copy().reset_index(drop=True)
    return train_df, test_df


def extract_and_combine_features(df, feature_cols):
    """Expand list/array feature columns into one numeric matrix (clustering cell 5/6)."""
    feature_dfs = []
    for col in feature_cols:
        expanded = pd.DataFrame(df[col].tolist(), index=df.index).add_prefix(f'{col}_')
        feature_dfs.append(expanded)
    combined = pd.concat(feature_dfs, axis=1)
    return combined.apply(pd.to_numeric, errors='coerce').fillna(0)


def fit_speaker_imputer(train_df, ac_cols):
    """Train-set medians used to impute silent-segment acoustic NaNs (no leakage)."""
    return train_df[ac_cols].median(numeric_only=True)


def speaker_block(df, speaker_cols, ac_cols, spk_medians):
    """Numeric speaker feature block, acoustic NaNs imputed with train medians (extend cell 7)."""
    blk = df[speaker_cols].copy()
    blk['spk_has_speaker'] = blk['spk_has_speaker'].astype(float)
    blk[ac_cols] = blk[ac_cols].fillna(spk_medians)
    return blk.apply(pd.to_numeric, errors='coerce').fillna(0.0)


# --------------------------------------------------------------------------- #
# Listener <-> speaker behavioral alignment
# --------------------------------------------------------------------------- #
# Similarity between the silent listener (target) and the *identified active speaker*
# in the SAME window, on the visual (CLIP) and AU channels. The speaker's features are
# looked up from the already-extracted reference assets (no new downloads):
#   - speaker CLIP embedding: dnn_feature_reference.pickle, key '<meeting>-<spk>/clip_<s>_<e>.mp4'
#   - speaker AU vector:      reference_behavior_results.zip, 'reference_behavior_results/<meeting>-<spk>/clip_<s>_<e>.csv'
# Requires the EXTENDED table (has 'speaker_id'). NaN where the speaker is silent/unmatched;
# impute with TRAIN medians (fit_ls_imputer / ls_align_block) — no leakage.
LS_ALIGN_COLS = ['ls_dnn_cos', 'ls_dnn_pear', 'ls_au_cos', 'ls_au_pear']


def _ls_cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def _ls_pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def add_listener_speaker_alignment(df, dnn_ref_path='./dnn_feature_reference.pickle',
                                   behavior_zip_path='./reference_behavior_results.zip'):
    """Add LS_ALIGN_COLS (visual + AU cosine & correlation between listener and active speaker).

    Computed for every row that has an identified speaker with available features; NaN otherwise.
    """
    import zipfile
    with open(dnn_ref_path, 'rb') as f:
        dref = pickle.load(f)
    z = zipfile.ZipFile(behavior_zip_path)
    au_cols = {}

    def speaker_au(meeting, spk, tail):
        try:
            with z.open(f"reference_behavior_results/{meeting}-{spk}/clip_{tail}.csv") as fh:
                c = pd.read_csv(fh)
        except KeyError:
            return None
        if 'cols' not in au_cols:
            au_cols['cols'] = [col for col in c.columns if col.startswith('AU')]
        return c[au_cols['cols']].iloc[0].to_numpy(float)

    rows = []
    for _, r in df.iterrows():
        sid = r['segment_id']
        meeting = sid.split('-')[0]
        tail = sid.split('_clip_')[1]
        spk = r.get('speaker_id')
        o = {c: np.nan for c in LS_ALIGN_COLS}
        if not pd.isna(spk):
            key = f"{meeting}-{spk}/clip_{tail}.mp4"
            if key in dref:
                o['ls_dnn_cos'] = _ls_cos(r['dnn_feature'], dref[key])
                o['ls_dnn_pear'] = _ls_pear(r['dnn_feature'], dref[key])
            sau = speaker_au(meeting, spk, tail)
            if sau is not None and len(sau) == len(np.asarray(r['AU_feature'])):
                o['ls_au_cos'] = _ls_cos(r['AU_feature'], sau)
                o['ls_au_pear'] = _ls_pear(r['AU_feature'], sau)
        rows.append(o)
    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)


def fit_ls_imputer(train_df, cols=LS_ALIGN_COLS):
    """Train-set medians for listener-speaker alignment columns (impute silent/unmatched windows)."""
    return train_df[cols].median(numeric_only=True)


def ls_align_block(df, ls_medians, cols=LS_ALIGN_COLS):
    """Numeric listener-speaker alignment block, NaNs imputed with train medians (no leakage)."""
    return df[cols].fillna(ls_medians).apply(pd.to_numeric, errors='coerce').fillna(0.0)


# --------------------------------------------------------------------------- #
# Segment-id -> clip path helpers
# --------------------------------------------------------------------------- #
def segment_id_to_filepath(segment_id, root='/content/drive/MyDrive/filtered_clips'):
    """e.g. 20211007-SP07F_clip_2652_2657 -> <root>/20211007-SP07F/clip_2652_2657.mp4."""
    parts = segment_id.split('_')
    date_subject = parts[0]
    clip_name = '_'.join(parts[1:])
    return f"{root}/{date_subject}/{clip_name}.mp4"


def clip_path(segment_id, clips_base='./filtered_clips'):
    """Local variant: split on ``_clip_`` (handles person ids that contain '_')."""
    mp, tail = segment_id.split('_clip_')
    return os.path.join(clips_base, mp, f"clip_{tail}.mp4")


def get_segments_for_manual_labeling(df_with_segment_ids, cluster_labels, n_segments_per_cluster=3):
    """Sample N segment ids per cluster for manual inspection (clustering cell 5)."""
    df = df_with_segment_ids.copy()
    df['cluster'] = cluster_labels
    out = {}
    for cid in sorted(df['cluster'].unique()):
        seg = df[df['cluster'] == cid]['segment_id']
        out[cid] = seg.sample(min(n_segments_per_cluster, len(seg)), random_state=RANDOM_STATE).tolist()
    return out


def export_representative_clips(train_df, algo_to_col, out_root, clips_base='./filtered_clips',
                                n_per_cluster=3):
    """Copy N sample clips per cluster into ``<out_root>/<algo>/<cluster>/`` (clustering cell 16)."""
    sample_segments = {}
    for algo, col in algo_to_col.items():
        samples = get_segments_for_manual_labeling(
            train_df[['segment_id', col]].copy(), train_df[col], n_per_cluster)
        sample_segments[algo] = samples
        print(f"\n=== {algo}  ({col}) ===")
        n_copied = n_missing = 0
        for cid, segs in samples.items():
            print(f"  Cluster {cid}: {segs}")
            dest = os.path.join(out_root, algo, str(cid))
            os.makedirs(dest, exist_ok=True)
            for sid in segs:
                src = clip_path(sid, clips_base)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(dest, os.path.basename(src)))
                    n_copied += 1
                else:
                    n_missing += 1
        print(f"  -> copied {n_copied} clips to {out_root}/{algo}/ ({n_missing} missing in {clips_base})")
    return sample_segments


# --------------------------------------------------------------------------- #
# Clustering scans  (clustering cells 9-14)
# --------------------------------------------------------------------------- #
def _cluster_labels(X_scaled, algo, k):
    """Hard cluster assignment for a given algorithm at k clusters."""
    if algo == 'kmeans':
        return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto').fit_predict(X_scaled)
    if algo == 'gmm':
        return GaussianMixture(n_components=k, random_state=RANDOM_STATE,
                               covariance_type='full').fit_predict(X_scaled)
    if algo == 'agg':
        return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_scaled)
    raise ValueError(f"unknown algo {algo!r}")


def silhouette_by_k(X_scaled, algo, k_range=range(2, 12)):
    """Silhouette score for ``algo`` at each k in ``k_range`` (label-agnostic; same metric for all 3)."""
    return {k: silhouette_score(X_scaled, _cluster_labels(X_scaled, algo, k)) for k in k_range}


def gmm_bic_aic(X_scaled, n_range=range(2, 12)):
    """BIC / AIC per n_components for GMM — a complexity-penalised cross-check on the silhouette choice."""
    bic, aic = {}, {}
    for n in n_range:
        g = GaussianMixture(n_components=n, random_state=RANDOM_STATE, covariance_type='full').fit(X_scaled)
        bic[n], aic[n] = g.bic(X_scaled), g.aic(X_scaled)
    return bic, aic


def select_cluster_counts(X_scaled, k_range=range(2, 12), plot=True, verbose=True):
    """Pick k for KMeans / GMM / Agglomerative by **maximising silhouette** over ``k_range``.

    One consistent, data-driven, label-agnostic criterion for all three algorithms (replaces the
    three hardcoded ``final_k_*`` values). Returns ``(best, scores)`` where ``best`` is a dict
    ``{'kmeans': k, 'gmm': k, 'agg': k}`` and ``scores`` maps each algo to its ``{k: silhouette}`` curve.
    """
    algos = {'kmeans': 'KMeans', 'gmm': 'GMM', 'agg': 'Agglomerative'}
    scores = {a: silhouette_by_k(X_scaled, a, k_range) for a in algos}
    best = {a: max(s, key=s.get) for a, s in scores.items()}
    if verbose:
        for a, label in algos.items():
            row = "  ".join(f"k={k}:{v:.3f}" for k, v in scores[a].items())
            print(f"{label:14s} silhouette  {row}")
        print(f"Silhouette-optimal -> KMeans k={best['kmeans']}, GMM n={best['gmm']}, "
              f"Agglomerative k={best['agg']}")
    if plot:
        plt.figure(figsize=(8, 5))
        for a, label in algos.items():
            ks = list(scores[a]); plt.plot(ks, [scores[a][k] for k in ks], marker='o', label=label)
            plt.scatter([best[a]], [scores[a][best[a]]], s=140, edgecolor='k', zorder=5)
        plt.xlabel('Number of clusters (k)'); plt.ylabel('Silhouette score')
        plt.title('Cluster-count selection by silhouette'); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.show()
    return best, scores


def plot_dendrogram(X_scaled, sample_size=300, p=20):
    """Plot a (sampled) Ward-linkage dendrogram (clustering cell 13)."""
    sample = min(len(X_scaled), sample_size)
    if len(X_scaled) > sample:
        idx = np.random.choice(X_scaled.shape[0], sample, replace=False)
        X = X_scaled[idx]
    else:
        X = X_scaled
    linked = linkage(X, method='ward')
    plt.figure(figsize=(12, 7))
    dendrogram(linked, orientation='top', distance_sort='descending',
               show_leaf_counts=True, truncate_mode='lastp', p=p)
    plt.title('Hierarchical Clustering Dendrogram (Sampled)')
    plt.xlabel('Sample index or (Cluster size)')
    plt.ylabel('Distance (Ward linkage)')
    plt.show()


def fit_clusterers(X_scaled, k_kmeans, n_gmm, k_hier):
    """Fit KMeans / GMM / Agglomerative and return (kmeans, gmm, agg, labels_dict)."""
    kmeans = KMeans(n_clusters=k_kmeans, random_state=RANDOM_STATE, n_init='auto')
    km_labels = kmeans.fit_predict(X_scaled)
    gmm = GaussianMixture(n_components=n_gmm, random_state=RANDOM_STATE, covariance_type='full')
    gmm_labels = gmm.fit_predict(X_scaled)
    agg = AgglomerativeClustering(n_clusters=k_hier, linkage='ward')
    agg_labels = agg.fit_predict(X_scaled)
    labels = {'kmeans_cluster': km_labels, 'gmm_cluster': gmm_labels, 'hierarchical_cluster': agg_labels}
    return kmeans, gmm, agg, labels


# --------------------------------------------------------------------------- #
# Cross-validated supervised comparison  (extend cells 23-26)
#
# NOTE: the old in-sample "auto majority-vote" cluster->engagement evaluation was
# removed. It named each cluster with the same test labels it was then scored
# against (optimistic) and degenerated to predicting the majority class whenever a
# clustering put all labeled points in one cluster (e.g. GMM here). The honest
# cluster/engagement signal is `kmeans_ari` (below); the rigorous predictive
# answer is the cross-validated comparison.
# --------------------------------------------------------------------------- #
def make_cv(n_splits=5):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def rf():
    return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced')


def lr():
    return LogisticRegression(max_iter=2000, class_weight='balanced')


def cv_report(name, clf_factory, configs, y, cv):
    """Print accuracy / macro-F1 for each (label, X) config under a Pipeline (extend cell 23)."""
    print(f"\n=== {name} ===")
    for label, X in configs:
        pipe = make_pipeline(StandardScaler(), clf_factory())
        acc = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
        f1 = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro')
        print(f"  {label:34s}  acc={acc.mean():.3f}±{acc.std():.3f}   "
              f"macroF1={f1.mean():.3f}±{f1.std():.3f}   (dim={X.shape[1]})")


def kmeans_ari(configs, y, k=3):
    """KMeans(k) Adjusted Rand Index vs true engagement per config (extend cell 23)."""
    print(f"\n=== Adjusted Rand Index (KMeans k={k} vs true engagement) ===")
    for label, X in configs:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto').fit_predict(
            StandardScaler().fit_transform(X))
        print(f"  {label:34s} ARI = {adjusted_rand_score(y, km):.4f}")


# --------------------------------------------------------------------------- #
# Head-and-Gesture compact alignment encodings  (extend cell 25)
# --------------------------------------------------------------------------- #
def make_hg_blocks(train_df, test_df):
    """Build compact HG alignment blocks (cosine 3d / distance 3d / grouped-cosine 12d).

    Scaler fitted ONLY on the unlabeled train manual_feature pool (leakage-free);
    blocks computed for ``test_df`` rows. Returns ``(hg_cos, hg_dist, hg_grp)`` DataFrames.
    """
    mtrain = np.vstack(train_df['manual_feature'].tolist()
                       + train_df['manual_feature_ref1'].tolist()
                       + train_df['manual_feature_ref2'].tolist()).astype(float)
    hg_scaler = StandardScaler().fit(mtrain)

    pairs = [('manual_feature', 'manual_feature_ref1'),
             ('manual_feature', 'manual_feature_ref2'),
             ('manual_feature_ref1', 'manual_feature_ref2')]
    pair_tags = ['ref1', 'ref2', 'ref1_ref2']

    def _z(v):
        return hg_scaler.transform(np.asarray(v, float).reshape(1, -1))[0]

    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    def _whole_cos(u, v):
        return _cos(_z(u), _z(v))

    def _whole_dist(u, v):
        return float(np.linalg.norm(_z(u) - _z(v)))

    def _group_cos(u, v):
        zu, zv = _z(u), _z(v)
        return [_cos(zu[ix], zv[ix]) for ix in HG_GROUPS.values()]

    def build_block(fn, names):
        rows = []
        for _, r in test_df.iterrows():
            vals = []
            for a, b in pairs:
                o = fn(r[a], r[b])
                vals += (o if isinstance(o, list) else [o])
            rows.append(vals)
        return pd.DataFrame(rows, columns=names).reset_index(drop=True)

    hg_cos = build_block(_whole_cos, [f'hg_cos_{t}' for t in pair_tags])
    hg_dist = build_block(_whole_dist, [f'hg_dist_{t}' for t in pair_tags])
    hg_grp = build_block(_group_cos, [f'hg_{g}_{t}' for t in pair_tags for g in HG_GROUPS])
    return hg_cos, hg_dist, hg_grp


# --------------------------------------------------------------------------- #
# Feature importance  (extend cell 28)
# --------------------------------------------------------------------------- #
def plot_feature_importances(importances, names, title, top_n=20):
    s = pd.Series(importances, index=names).sort_values(ascending=False)
    plt.figure(figsize=(9, max(5, top_n * 0.3)))
    sns.barplot(x=s.head(top_n).values, y=s.head(top_n).index)
    plt.title(title); plt.tight_layout(); plt.show()
    return s


def engagement_feature_importance(X_lab, y, top_n=20):
    """RandomForest importance + ANOVA F-stat for engagement on the labeled set (extend cell 28)."""
    model = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, class_weight='balanced')
    model.fit(StandardScaler().fit_transform(X_lab), y)
    names = prettify_feature_names(X_lab.columns)
    imp = plot_feature_importances(model.feature_importances_, names,
                                   "RF importance for engagement", top_n=top_n)
    F, _ = f_classif(X_lab.fillna(0), y)
    anova = plot_feature_importances(np.nan_to_num(F), names,
                                     "ANOVA F-statistic vs engagement", top_n=top_n)
    return imp, anova
