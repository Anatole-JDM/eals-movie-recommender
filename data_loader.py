"""
data_loader.py
==============
Download, filter, re-encode, and split the Amazon Movies & TV dataset
used in the eALS paper (He et al., SIGIR 2016).

Pipeline
--------
1. Download raw CSV (user_id_str, item_id_str, rating, timestamp).
2. Binarise: every observed rating becomes an implicit interaction (r=1).
3. Iteratively filter users and items with < min_interactions entries.
4. Map string IDs to consecutive integers (user_idx, item_idx).
5. Leave-one-out split: hold out each user's chronologically *last*
   interaction as the test instance (same protocol as the paper).
6. Compute per-item confidence scores c_i for the eALS weighting.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
from pyspark import SparkContext

# --------------------------------------------------------------------------- #
#  1.  Raw data download                                                       #
# --------------------------------------------------------------------------- #


def download_data(url: str, local_path: str) -> None:
    """Download the dataset if it is not already present."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        print(f"[data_loader] Dataset already on disk: {local_path}")
        return
    print(f"[data_loader] Downloading {url} …")
    urllib.request.urlretrieve(url, local_path)
    print(f"[data_loader] Saved to {local_path}")


# --------------------------------------------------------------------------- #
#  2.  Load & iterative filter                                                 #
# --------------------------------------------------------------------------- #


def load_and_filter(
    local_path: str,
    min_interactions: int = 10,
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the raw CSV and return a filtered DataFrame with columns
    [user_id, item_id, timestamp].

    The filtering loop (standard in CF literature) removes users and
    items with fewer than min_interactions entries until the dataset is
    stable.  If sample_fraction < 1, we sample *before* filtering so
    the resulting dataset is still internally consistent.
    """
    print("[data_loader] Reading raw CSV …")
    df = pd.read_csv(
        local_path,
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": str, "item_id": str, "rating": float, "timestamp": int},
    )
    df = df[["user_id", "item_id", "timestamp"]].dropna()

    if sample_fraction < 1.0:
        df = df.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)
        print(f"[data_loader] Sampled {sample_fraction:.0%} → {len(df):,} rows")

    print(f"[data_loader] Raw interactions : {len(df):,}")

    # Iterative core filtering
    iteration = 0
    while True:
        u_counts = df["user_id"].value_counts()
        i_counts = df["item_id"].value_counts()
        valid_users = u_counts[u_counts >= min_interactions].index
        valid_items = i_counts[i_counts >= min_interactions].index
        df_new = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        iteration += 1
        if len(df_new) == len(df):
            break
        df = df_new

    df = df.reset_index(drop=True)
    print(
        f"[data_loader] After {iteration}-pass filter (≥{min_interactions} interactions): "
        f"{len(df):,} rows | {df['user_id'].nunique():,} users | {df['item_id'].nunique():,} items"
    )
    return df


# --------------------------------------------------------------------------- #
#  3.  ID encoding                                                             #
# --------------------------------------------------------------------------- #


def encode_ids(df: pd.DataFrame):
    """
    Map string user/item IDs to consecutive integers.

    Returns
    -------
    df        : DataFrame with added columns user_idx, item_idx
    user2id   : dict str → int
    item2id   : dict str → int
    n_users   : int
    n_items   : int
    """
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())

    user2id = {u: i for i, u in enumerate(users)}
    item2id = {v: i for i, v in enumerate(items)}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2id).astype(int)
    df["item_idx"] = df["item_id"].map(item2id).astype(int)

    n_users = len(users)
    n_items = len(items)

    print(f"[data_loader] n_users={n_users:,}  n_items={n_items:,}")
    return df, user2id, item2id, n_users, n_items


# --------------------------------------------------------------------------- #
#  4.  Train / test split                                                      #
# --------------------------------------------------------------------------- #


def leave_one_out_split(df: pd.DataFrame):
    """
    Hold out each user's chronologically *latest* interaction as the test
    instance.  The model trains on all remaining data.

    This matches the offline evaluation protocol in the paper
    (Section 5.1, "Offline Protocol").
    """
    df_sorted = df.sort_values(["user_idx", "timestamp"])
    test = df_sorted.groupby("user_idx", group_keys=False).tail(1)
    train = df_sorted.drop(index=test.index)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"[data_loader] Leave-one-out split → train: {len(train):,} | test: {len(test):,}")
    return train, test


# --------------------------------------------------------------------------- #
#  5.  Item confidence                                                         #
# --------------------------------------------------------------------------- #


def compute_item_confidence(
    train_df: pd.DataFrame,
    n_items: int,
    c0: float,
    alpha: float,
) -> np.ndarray:
    """
    Compute per-item confidence values c_i (Eq. 8 in the paper):

        c_i = c0 * f_i^α / Σ_j f_j^α

    where f_i = |R_i| / Σ_j |R_j|  is the normalised item popularity.

    Special case: α = 0 reduces to the uniform weight w0 = c0 / N,
    which corresponds to the baseline used in ALS [Hu et al., ICDM 2008].

    Parameters
    ----------
    train_df  : training interactions (must contain column 'item_idx')
    n_items   : total number of distinct items
    c0        : overall weight scale
    alpha     : popularity exponent (paper finds α ≈ 0.4–0.5 optimal)

    Returns
    -------
    c : np.ndarray of shape (n_items,)
    """
    item_counts = train_df["item_idx"].value_counts()
    total_counts = item_counts.sum()

    freq = np.zeros(n_items, dtype=np.float64)
    for idx, cnt in item_counts.items():
        freq[int(idx)] = cnt / total_counts

    if alpha == 0:
        # Uniform: every item gets the same weight
        c = np.full(n_items, c0 / n_items, dtype=np.float64)
    else:
        freq_alpha = np.power(np.maximum(freq, 1e-12), alpha)
        c = c0 * freq_alpha / freq_alpha.sum()

    print(
        f"[data_loader] Item confidence  c0={c0}  α={alpha}  "
        f"min={c.min():.2e}  max={c.max():.2e}  mean={c.mean():.2e}"
    )
    return c


# --------------------------------------------------------------------------- #
#  6.  Build Spark RDD                                                         #
# --------------------------------------------------------------------------- #


def build_training_rdd(sc: SparkContext, train_df: pd.DataFrame, n_partitions: int = 8):
    """
    Convert the training DataFrame to a Spark RDD of tuples:
        (user_idx: int, item_idx: int, r_ui: float, w_ui: float)

    r_ui is always 1.0 (implicit, positive interaction).
    w_ui is always 1.0 (weight of observed entries – paper default).

    The RDD is cached so downstream transformations reuse it cheaply.
    """
    records = [
        (int(row.user_idx), int(row.item_idx), 1.0, 1.0) for row in train_df.itertuples(index=False)
    ]
    rdd = sc.parallelize(records, numSlices=n_partitions).cache()
    print(f"[data_loader] Training RDD: {rdd.count():,} interactions, {n_partitions} partitions")
    return rdd


# --------------------------------------------------------------------------- #
#  7.  Convenience summary                                                     #
# --------------------------------------------------------------------------- #


def dataset_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    n_users = train_df["user_idx"].nunique()
    n_items = train_df["item_idx"].nunique()
    n_train = len(train_df)
    n_test = len(test_df)
    sparsity = 1.0 - n_train / (n_users * n_items)
    print("\n" + "=" * 55)
    print("  Dataset statistics")
    print("=" * 55)
    print(f"  Users        : {n_users:>10,}")
    print(f"  Items        : {n_items:>10,}")
    print(f"  Train inter. : {n_train:>10,}")
    print(f"  Test  inter. : {n_test:>10,}")
    print(f"  Sparsity     : {sparsity:>10.4%}")
    print("=" * 55 + "\n")
