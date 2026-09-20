"""
app.py
======
Streamlit front-end for the Spark eALS implicit-feedback recommender
(He et al., SIGIR 2016). Wraps the existing data_loader / eals_spark /
evaluation modules in an interactive UI: configure hyperparameters,
load & filter the Amazon Movies & TV dataset, train eALS with a live
per-epoch HR@K / NDCG@K chart, compare the uniform vs. popularity
weighting variants, and generate top-N recommendations for a user.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from data_loader import (
    download_data,
    load_and_filter,
    encode_ids,
    leave_one_out_split,
    compute_item_confidence,
    build_training_rdd,
)
from eals_spark import eALS
from evaluation import Evaluator

st.set_page_config(page_title="eALS Recommender", page_icon="🎬", layout="wide")


# --------------------------------------------------------------------------- #
#  Cached resources                                                            #
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_spark_context(n_partitions: int):
    from pyspark import SparkContext, SparkConf

    conf = (
        SparkConf()
        .setAppName("eALS_Streamlit")
        .setMaster("local[*]")
        .set("spark.executor.memory", "2g")
        .set("spark.driver.memory", "3g")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.ui.showConsoleProgress", "false")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    return sc


@st.cache_data(show_spinner=False)
def load_dataset(data_url: str, raw_path: str, min_interactions: int,
                  sample_fraction: float, seed: int):
    download_data(data_url, raw_path)
    df_raw = load_and_filter(
        raw_path,
        min_interactions=min_interactions,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    df, user2id, item2id, n_users, n_items = encode_ids(df_raw)
    train_df, test_df = leave_one_out_split(df)
    return train_df, test_df, user2id, item2id, n_users, n_items


# --------------------------------------------------------------------------- #
#  Sidebar — configuration                                                     #
# --------------------------------------------------------------------------- #

st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Data")
data_url = st.sidebar.text_input(
    "Dataset URL",
    value="https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv",
)
raw_data_path = st.sidebar.text_input("Local cache path", value="./data/ratings_Movies_and_TV.csv")
min_interactions = st.sidebar.slider("Min interactions (k-core filter)", 2, 50, 10)
sample_fraction = st.sidebar.slider(
    "Sample fraction", 0.05, 1.0, 0.2, step=0.05,
    help=(
        "Fraction of raw rows to sample before filtering. Because filtering "
        "requires each user/item to have at least 'Min interactions' entries, "
        "very small fractions combined with a high threshold can filter the "
        "dataset down to zero rows — lower the threshold or raise this if that happens."
    ),
)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.subheader("Model")
K = st.sidebar.slider("Latent factors (K)", 8, 128, 32, step=8)
lambda_reg = st.sidebar.number_input("L2 regularization (λ)", value=0.01, format="%.4f")
c0 = st.sidebar.number_input("Missing-data weight scale (c0)", value=64.0)
w_obs = st.sidebar.number_input("Observed-interaction weight (w_obs)", value=1.0)
n_epochs = st.sidebar.slider("Epochs", 1, 30, 5)

st.sidebar.subheader("Evaluation")
top_k = st.sidebar.slider("Top-K (HR@K / NDCG@K)", 5, 200, 20)
max_eval_users = st.sidebar.slider("Max eval users", 50, 5000, 500, step=50)

n_partitions = st.sidebar.slider("Spark partitions", 2, 16, 4)


# --------------------------------------------------------------------------- #
#  Header                                                                      #
# --------------------------------------------------------------------------- #

st.title("🎬 eALS Implicit-Feedback Recommender")
st.caption(
    "Spark-parallelised element-wise ALS — He et al., "
    "\"Fast Matrix Factorization for Online Recommendation with Implicit Feedback\", SIGIR 2016."
)

tab_data, tab_train, tab_compare, tab_recommend = st.tabs(
    ["📦 Data", "🚀 Train", "📊 Compare weighting", "🔮 Recommend"]
)


# --------------------------------------------------------------------------- #
#  Tab 1 — Data                                                                #
# --------------------------------------------------------------------------- #

with tab_data:
    st.subheader("Load & filter the dataset")
    st.write(
        "Downloads the raw Amazon ratings CSV (if not already cached), binarises "
        "ratings into implicit interactions, applies iterative k-core filtering, "
        "re-encodes IDs, and performs a leave-one-out train/test split."
    )

    if st.button("Load / refresh data", type="primary"):
        with st.spinner("Loading and filtering data…"):
            t0 = time.time()
            (train_df, test_df, user2id, item2id,
             n_users, n_items) = load_dataset(
                data_url, raw_data_path, min_interactions, sample_fraction, int(seed)
            )
            elapsed = time.time() - t0

        if n_users == 0 or n_items == 0:
            st.error(
                "Filtering removed every interaction — no users/items satisfy "
                "the 'Min interactions' threshold at this sample fraction. "
                "Lower 'Min interactions' or raise 'Sample fraction' and try again."
            )
            st.stop()

        st.session_state["train_df"] = train_df
        st.session_state["test_df"] = test_df
        st.session_state["user2id"] = user2id
        st.session_state["item2id"] = item2id
        st.session_state["n_users"] = n_users
        st.session_state["n_items"] = n_items
        st.success(f"Data ready in {elapsed:.1f}s")

    if "train_df" in st.session_state:
        train_df = st.session_state["train_df"]
        test_df = st.session_state["test_df"]
        n_users = st.session_state["n_users"]
        n_items = st.session_state["n_items"]
        sparsity = 1.0 - len(train_df) / (n_users * n_items)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Users", f"{n_users:,}")
        c2.metric("Items", f"{n_items:,}")
        c3.metric("Train interactions", f"{len(train_df):,}")
        c4.metric("Test interactions", f"{len(test_df):,}")
        c5.metric("Sparsity", f"{sparsity:.4%}")

        st.dataframe(train_df.head(20), use_container_width=True)
    else:
        st.info("Click **Load / refresh data** to get started.")


# --------------------------------------------------------------------------- #
#  Tab 2 — Train                                                               #
# --------------------------------------------------------------------------- #

with tab_train:
    st.subheader("Train a single eALS model")

    alpha = st.slider(
        "Popularity exponent α (0 = uniform, 0.5 = paper's default)",
        0.0, 1.0, 0.5, step=0.1, key="train_alpha",
    )

    if "train_df" not in st.session_state:
        st.warning("Load the data first in the **Data** tab.")
    elif st.button("Train model", type="primary"):
        train_df = st.session_state["train_df"]
        test_df = st.session_state["test_df"]
        n_users = st.session_state["n_users"]
        n_items = st.session_state["n_items"]

        progress = st.progress(0.0, text="Starting Spark…")
        sc = get_spark_context(n_partitions)
        train_rdd = build_training_rdd(sc, train_df, n_partitions=n_partitions)

        evaluator = Evaluator(top_k=top_k, max_eval_users=max_eval_users, seed=int(seed))
        evaluator.build_train_lookup(train_df)

        item_conf = compute_item_confidence(train_df, n_items, c0=c0, alpha=alpha)
        model = eALS(
            n_users=n_users, n_items=n_items, K=K,
            lambda_reg=lambda_reg, c0=c0, alpha=alpha, w_obs=w_obs, seed=int(seed),
        )

        chart_placeholder = st.empty()
        history = []
        for epoch in range(1, n_epochs + 1):
            epoch_hist = model.fit(
                sc=sc, train_rdd=train_rdd, item_conf=item_conf,
                n_epochs=1, evaluator=evaluator, test_df=test_df,
                train_df=train_df, verbose=False,
            )
            row = epoch_hist[0]
            row["epoch"] = epoch
            history.append(row)

            hist_df = pd.DataFrame(history).set_index("epoch")
            chart_placeholder.line_chart(hist_df[[f"hr@{top_k}", f"ndcg@{top_k}"]])
            progress.progress(epoch / n_epochs, text=f"Epoch {epoch}/{n_epochs}")

        progress.empty()
        st.session_state["model"] = model
        st.session_state["history"] = history
        st.session_state["evaluator"] = evaluator
        st.success("Training complete.")
        st.dataframe(pd.DataFrame(history), use_container_width=True)


# --------------------------------------------------------------------------- #
#  Tab 3 — Compare uniform vs. popularity weighting                            #
# --------------------------------------------------------------------------- #

with tab_compare:
    st.subheader("Uniform (α=0) vs. Popularity-weighted (α=0.5) missing-data weighting")
    st.write(
        "Runs both variants from `main.py`'s experiment and compares best "
        f"HR@{top_k} / NDCG@{top_k} plus average epoch time."
    )

    if "train_df" not in st.session_state:
        st.warning("Load the data first in the **Data** tab.")
    elif st.button("Run comparison", type="primary"):
        train_df = st.session_state["train_df"]
        test_df = st.session_state["test_df"]
        n_users = st.session_state["n_users"]
        n_items = st.session_state["n_items"]

        sc = get_spark_context(n_partitions)
        train_rdd = build_training_rdd(sc, train_df, n_partitions=n_partitions)

        evaluator = Evaluator(top_k=top_k, max_eval_users=max_eval_users, seed=int(seed))
        evaluator.build_train_lookup(train_df)

        results = {}
        for label, a in [("Uniform (α=0)", 0.0), ("Popularity (α=0.5)", 0.5)]:
            with st.spinner(f"Training {label}…"):
                item_conf = compute_item_confidence(train_df, n_items, c0=c0, alpha=a)
                model = eALS(
                    n_users=n_users, n_items=n_items, K=K,
                    lambda_reg=lambda_reg, c0=c0, alpha=a, w_obs=w_obs, seed=int(seed),
                )
                history = model.fit(
                    sc=sc, train_rdd=train_rdd, item_conf=item_conf,
                    n_epochs=n_epochs, evaluator=evaluator, test_df=test_df,
                    train_df=train_df, verbose=False,
                )
                results[label] = history

        rows = []
        for label, history in results.items():
            best = max(history, key=lambda r: r.get(f"hr@{top_k}", 0.0))
            rows.append({
                "Variant": label,
                f"Best HR@{top_k}": best.get(f"hr@{top_k}"),
                f"Best NDCG@{top_k}": best.get(f"ndcg@{top_k}"),
                "Avg epoch time (s)": round(np.mean([r["time_s"] for r in history]), 2),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Variant"), use_container_width=True)

        combined = pd.DataFrame({
            label: [r[f"hr@{top_k}"] for r in history]
            for label, history in results.items()
        })
        combined.index = range(1, n_epochs + 1)
        combined.index.name = "epoch"
        st.line_chart(combined)


# --------------------------------------------------------------------------- #
#  Tab 4 — Recommend                                                           #
# --------------------------------------------------------------------------- #

with tab_recommend:
    st.subheader("Top-N recommendations for a user")

    if "model" not in st.session_state:
        st.warning("Train a model first in the **Train** tab.")
    else:
        model: eALS = st.session_state["model"]
        train_df = st.session_state["train_df"]
        n_users = st.session_state["n_users"]

        user_idx = st.number_input("User index", min_value=0, max_value=n_users - 1, value=0)
        n_recs = st.slider("Number of recommendations", 5, 50, 10)

        if st.button("Recommend"):
            scores = model.predict_user(int(user_idx))
            seen_items = set(train_df.loc[train_df["user_idx"] == user_idx, "item_idx"])
            ranked = np.argsort(-scores)
            recs = [i for i in ranked if i not in seen_items][:n_recs]
            st.dataframe(
                pd.DataFrame({"item_idx": recs, "score": scores[recs]}),
                use_container_width=True,
            )
