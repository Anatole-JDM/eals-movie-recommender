"""
config.py
=========
Central configuration for the eALS-Spark recommendation experiment.
All hyperparameters match or are derived from:
  He et al., "Fast Matrix Factorization for Online Recommendation
  with Implicit Feedback", SIGIR 2016.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters and settings for one eALS-Spark experiment run."""

    # ------------------------------------------------------------------ #
    #  Dataset                                                             #
    # ------------------------------------------------------------------ #
    dataset_name: str = "amazon_movies"

    # Direct CSV download from SNAP (user_id, item_id, rating, timestamp)
    data_url: str = (
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Movies_and_TV.csv"
    )
    raw_data_path: str = "./data/ratings_Movies_and_TV.csv"

    # Keep only users AND items with >= min_interactions interactions
    # (paper uses 10 for both Yelp and Amazon)
    min_interactions: int = 10

    # Optional: subsample the raw data before filtering (0 < frac <= 1.0).
    # Set to 1.0 to reproduce paper results; use 0.2 for a quick smoke-test.
    sample_fraction: float = 0.2

    # Random seed for sampling & factor initialisation
    seed: int = 42

    # ------------------------------------------------------------------ #
    #  Model hyper-parameters                                              #
    # ------------------------------------------------------------------ #
    # Number of latent factors
    K: int = 64  # paper uses 128; 64 keeps runtime reasonable

    # L2 regularisation strength  (paper: λ = 0.01)
    lambda_reg: float = 0.01

    # Weight assigned to every OBSERVED interaction  (paper: w_ui = 1)
    w_obs: float = 1.0

    # --- Missing-data weighting (Section 4.1 of the paper) ---
    # c0  : overall scale of the missing-data weight  (paper searches 8–2048)
    c0: float = 64.0

    # alpha: popularity exponent in c_i = c0 * f_i^α / Σ f_j^α
    #   α = 0  → uniform weighting (baseline)
    #   α = 0.5 → popularity-aware (paper's best setting)
    alpha: float = 0.5

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    n_epochs: int = 20

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #
    # Truncation point for HR and NDCG  (paper: top-100)
    top_k: int = 100

    # Max test users to score per evaluation pass.
    # Full ranking over all N items is O(n_eval_users * N * K).
    # Set to None to evaluate every test user (slower but exact).
    max_eval_users: int | None = 2000

    # ------------------------------------------------------------------ #
    #  Spark                                                               #
    # ------------------------------------------------------------------ #
    spark_app_name: str = "eALS_Spark_Recommendation"
    spark_master: str = "local[*]"  # use all local cores
    spark_executor_memory: str = "4g"
    spark_driver_memory: str = "6g"

    # Number of RDD partitions. Rule of thumb: 2-4x the number of cores.
    n_partitions: int = 8


# Singleton – import this throughout the project
CFG = Config()
