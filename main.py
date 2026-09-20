"""
main.py
=======
End-to-end experiment runner for the Spark eALS implementation.

Experiments performed
---------------------
1. eALS-Uniform   : α = 0   (missing data weighted uniformly, like Hu et al. 2008)
2. eALS-Popularity: α = 0.5 (missing data weighted by item popularity — paper's method)

For each variant we report:
  * HR@100 and NDCG@100 per epoch  (convergence analysis)
  * Wall-clock time per epoch      (efficiency analysis)
  * Best achieved metric values    (summary comparison)

Dataset: Amazon Movies & TV  (20% sample by default for quick runs;
         set CFG.sample_fraction = 1.0 to match paper scale).

Usage
-----
    python main.py

Requirements: pyspark, numpy, pandas  (see requirements.txt)
"""

import time
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ---- project imports ---- #
from config import CFG
from data_loader import (
    download_data,
    load_and_filter,
    encode_ids,
    leave_one_out_split,
    compute_item_confidence,
    build_training_rdd,
    dataset_summary,
)
from eals_spark import eALS
from evaluation import Evaluator

import os
import sys  

# Force Spark workers to use the same interpreter as this process.
# This avoids mismatches when VS Code uses a virtual environment.
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


# =========================================================================== #
#   Spark session                                                               #
# =========================================================================== #

def build_spark_context():
    """Create and return a SparkContext configured for local execution."""
    from pyspark import SparkContext, SparkConf

    conf = (
        SparkConf()
        .setAppName(CFG.spark_app_name)
        .setMaster(CFG.spark_master)
        .set("spark.executor.memory",      CFG.spark_executor_memory)
        .set("spark.driver.memory",        CFG.spark_driver_memory)
        .set("spark.serializer",           "org.apache.spark.serializer.KryoSerializer")
        .set("spark.kryoserializer.buffer.max", "512m")
        .set("spark.driver.maxResultSize", "2g")
        # Silence verbose Spark / Hadoop logs
        .set("spark.ui.showConsoleProgress", "false")
    )

    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    print(f"[Spark] {sc.version}  master={sc.master}")
    return sc


# =========================================================================== #
#   Helpers                                                                    #
# =========================================================================== #

def print_section(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_results_table(history: list, label: str, top_k: int) -> None:
    """Pretty-print per-epoch metrics."""
    print(f"\n  [{label}]")
    print(f"  {'Epoch':>6}  {'Time(s)':>8}  {'HR@'+str(top_k):>9}  {'NDCG@'+str(top_k):>10}")
    print("  " + "-" * 40)
    for r in history:
        hr_key   = f"hr@{top_k}"
        ndcg_key = f"ndcg@{top_k}"
        hr   = r.get(hr_key,   float("nan"))
        ndcg = r.get(ndcg_key, float("nan"))
        print(
            f"  {r['epoch']:>6}  {r['time_s']:>8.1f}  "
            f"{hr:>9.4f}  {ndcg:>10.4f}"
        )


def best_row(history: list, top_k: int) -> dict:
    """Return the epoch row with the highest HR@K."""
    key = f"hr@{top_k}"
    valid = [r for r in history if key in r]
    return max(valid, key=lambda r: r[key]) if valid else {}


# =========================================================================== #
#   Main                                                                       #
# =========================================================================== #

def main():
    t_total = time.time()

    # ------------------------------------------------------------------ #
    #  0.  Data                                                           #
    # ------------------------------------------------------------------ #
    print_section("0. Data preparation")

    download_data(CFG.data_url, CFG.raw_data_path)

    df_raw = load_and_filter(
        CFG.raw_data_path,
        min_interactions=CFG.min_interactions,
        sample_fraction=CFG.sample_fraction,
        seed=CFG.seed,
    )

    df, user2id, item2id, n_users, n_items = encode_ids(df_raw)
    train_df, test_df = leave_one_out_split(df)
    dataset_summary(train_df, test_df)

    # ------------------------------------------------------------------ #
    #  1.  Spark context                                                  #
    # ------------------------------------------------------------------ #
    print_section("1. Starting Spark")
    sc = build_spark_context()

    train_rdd = build_training_rdd(sc, train_df, n_partitions=CFG.n_partitions)

    # ------------------------------------------------------------------ #
    #  2.  Evaluator (shared between experiments)                         #
    # ------------------------------------------------------------------ #
    evaluator = Evaluator(
        top_k=CFG.top_k,
        max_eval_users=CFG.max_eval_users,
        seed=CFG.seed,
    )
    evaluator.build_train_lookup(train_df)

    # ------------------------------------------------------------------ #
    #  3.  Experiment A — eALS with UNIFORM missing-data weights (α = 0) #
    # ------------------------------------------------------------------ #
    print_section("3. Experiment A  |  eALS-Uniform (α = 0)")
    print(f"  K={CFG.K}  λ={CFG.lambda_reg}  c0={CFG.c0}  α=0  epochs={CFG.n_epochs}\n")

    item_conf_uniform = compute_item_confidence(
        train_df, n_items, c0=CFG.c0, alpha=0.0
    )

    model_uniform = eALS(
        n_users=n_users,
        n_items=n_items,
        K=CFG.K,
        lambda_reg=CFG.lambda_reg,
        c0=CFG.c0,
        alpha=0.0,
        w_obs=CFG.w_obs,
        seed=CFG.seed,
    )

    history_uniform = model_uniform.fit(
        sc=sc,
        train_rdd=train_rdd,
        item_conf=item_conf_uniform,
        n_epochs=CFG.n_epochs,
        evaluator=evaluator,
        test_df=test_df,
        train_df=train_df,
        verbose=True,
    )

    print_results_table(history_uniform, "eALS-Uniform", CFG.top_k)

    # ------------------------------------------------------------------ #
    #  4.  Experiment B — eALS with POPULARITY weights (α = 0.5)         #
    # ------------------------------------------------------------------ #
    print_section("4. Experiment B  |  eALS-Popularity (α = 0.5)")
    print(f"  K={CFG.K}  λ={CFG.lambda_reg}  c0={CFG.c0}  α=0.5  epochs={CFG.n_epochs}\n")

    item_conf_pop = compute_item_confidence(
        train_df, n_items, c0=CFG.c0, alpha=CFG.alpha
    )

    model_pop = eALS(
        n_users=n_users,
        n_items=n_items,
        K=CFG.K,
        lambda_reg=CFG.lambda_reg,
        c0=CFG.c0,
        alpha=CFG.alpha,
        w_obs=CFG.w_obs,
        seed=CFG.seed,
    )

    history_pop = model_pop.fit(
        sc=sc,
        train_rdd=train_rdd,
        item_conf=item_conf_pop,
        n_epochs=CFG.n_epochs,
        evaluator=evaluator,
        test_df=test_df,
        train_df=train_df,
        verbose=True,
    )

    print_results_table(history_pop, "eALS-Popularity", CFG.top_k)

    # ------------------------------------------------------------------ #
    #  5.  Summary comparison                                             #
    # ------------------------------------------------------------------ #
    print_section("5. Summary — best epoch results")

    top_k   = CFG.top_k
    hr_key  = f"hr@{top_k}"
    nk_key  = f"ndcg@{top_k}"

    best_u = best_row(history_uniform, top_k)
    best_p = best_row(history_pop,     top_k)

    rows = [
        {
            "Model":             "eALS-Uniform (α=0)",
            "Best Epoch":        best_u.get("epoch",  "-"),
            f"HR@{top_k}":       best_u.get(hr_key,   float("nan")),
            f"NDCG@{top_k}":     best_u.get(nk_key,   float("nan")),
            "Avg time/epoch(s)": round(
                np.mean([r["time_s"] for r in history_uniform]), 2
            ),
        },
        {
            "Model":             f"eALS-Popularity (α={CFG.alpha})",
            "Best Epoch":        best_p.get("epoch",  "-"),
            f"HR@{top_k}":       best_p.get(hr_key,   float("nan")),
            f"NDCG@{top_k}":     best_p.get(nk_key,   float("nan")),
            "Avg time/epoch(s)": round(
                np.mean([r["time_s"] for r in history_pop]), 2
            ),
        },
    ]
    summary_df = pd.DataFrame(rows).set_index("Model")
    print("\n" + summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------ #
    #  6.  Relative gain of popularity weighting                          #
    # ------------------------------------------------------------------ #
    if best_u and best_p:
        hr_gain   = (best_p[hr_key]  - best_u[hr_key])  / (best_u[hr_key]  + 1e-12) * 100
        ndcg_gain = (best_p[nk_key]  - best_u[nk_key])  / (best_u[nk_key]  + 1e-12) * 100
        print(f"\n  Relative improvement (Popularity vs Uniform):")
        print(f"    HR@{top_k}  : {hr_gain:+.2f}%")
        print(f"    NDCG@{top_k}: {ndcg_gain:+.2f}%")

    # ------------------------------------------------------------------ #
    #  7.  Efficiency: time per epoch                                     #
    # ------------------------------------------------------------------ #
    print_section("6. Efficiency — per-epoch training times")
    print(f"\n  {'Epoch':>6}  {'Uniform (s)':>12}  {'Popularity (s)':>15}")
    print("  " + "-" * 38)
    for u_row, p_row in zip(history_uniform, history_pop):
        print(
            f"  {u_row['epoch']:>6}  "
            f"{u_row['time_s']:>12.1f}  "
            f"{p_row['time_s']:>15.1f}"
        )

    print(
        f"\n  Mean epoch time — Uniform:    "
        f"{np.mean([r['time_s'] for r in history_uniform]):.1f}s"
    )
    print(
        f"  Mean epoch time — Popularity: "
        f"{np.mean([r['time_s'] for r in history_pop]):.1f}s"
    )

    # ------------------------------------------------------------------ #
    #  8.  Save results to CSV                                            #
    # ------------------------------------------------------------------ #
    out_path_u = "./results_uniform.csv"
    out_path_p = "./results_popularity.csv"
    pd.DataFrame(history_uniform).to_csv(out_path_u, index=False)
    pd.DataFrame(history_pop).to_csv(out_path_p, index=False)
    print(f"\n  Results saved → {out_path_u}  and  {out_path_p}")

    # ------------------------------------------------------------------ #
    #  9.  Stop Spark                                                     #
    # ------------------------------------------------------------------ #
    sc.stop()
    print(f"\n  Total wall-clock time: {(time.time() - t_total)/60:.1f} min")
    print_section("Done")


if __name__ == "__main__":
    main()
