"""
eals_spark.py
=============
PySpark implementation of eALS
  He et al., "Fast Matrix Factorization for Online Recommendation
  with Implicit Feedback", SIGIR 2016.

Parallelisation strategy (Section 4.2.1 of the paper)
------------------------------------------------------
The paper proves that user-factor updates are *embarrassingly parallel*:
given a fixed Q (item factors) and a fixed S^q cache, every user's p_u
can be updated independently.  The same holds for item updates.

Spark mapping
~~~~~~~~~~~~~
1. Pre-group interactions by user  → user_items_rdd  (cached)
   Pre-group interactions by item  → item_users_rdd  (cached)
2. Per epoch:
   a. Compute S^q = Σ_i c_i · q_i q_i^T  on the driver via NumPy BLAS.
   b. Broadcast (P, Q, S^q) to all workers.
   c. Map update_user() over user_items_rdd  → collect new P.
   d. Compute S^p = P^T P  on the driver.
   e. Broadcast (P, Q, S^p) to all workers.
   f. Map update_item() over item_users_rdd → collect new Q.

Equation references use the paper's numbering:
  Eq. 7  : objective function with item-oriented missing-data weighting
  Eq. 12 : closed-form update for p_uf
  Eq. 13 : closed-form update for q_if
"""

import time

import numpy as np
from pyspark import Broadcast, SparkContext

# =========================================================================== #
#   Module-level worker functions (must be picklable → not inside a class)    #
# =========================================================================== #


def _make_user_updater(
    Q_bc: Broadcast, P_bc: Broadcast, Sq_bc: Broadcast, K: int, lambda_reg: float
):
    """
    Return a function that maps  (user_id, items_list) → (user_id, new_p_u).

    items_list : list of (item_id, r_ui, w_ui, c_i)

    Uses a factory closure so that the broadcast variables are captured at
    *definition time* (avoiding late-binding Python closure bugs in loops).
    """

    def update_user(record: tuple[int, list]) -> tuple[int, list]:
        user_id, items = record
        if not items:
            return user_id, P_bc.value[user_id].tolist()

        # ---------- pull broadcast payloads ----------
        Q_local: np.ndarray = Q_bc.value  # shape (N, K)
        p_u: np.ndarray = P_bc.value[user_id].copy()  # shape (K,)
        Sq: np.ndarray = Sq_bc.value  # shape (K, K)

        # ---------- gather observed-item data ----------
        item_ids = [x[0] for x in items]
        Q_obs = np.array([Q_local[iid] for iid in item_ids], dtype=np.float64)  # (|R_u|, K)
        r_uis = np.array([x[1] for x in items], dtype=np.float64)  # (|R_u|,)
        w_uis = np.array([x[2] for x in items], dtype=np.float64)  # (|R_u|,)
        c_is = np.array([x[3] for x in items], dtype=np.float64)  # (|R_u|,)

        # ---------- initialise prediction cache ----------
        # r_hat_ui = p_u^T q_i  for each observed item i  (Eq. 1)
        r_hats: np.ndarray = Q_obs @ p_u  # (|R_u|,)

        # ---------- element-wise ALS loop (Algorithm 1, lines 6–9) ----------
        # Eq. 12:
        #   p_uf = [ Σ_{i∈R_u}(w_ui·r_ui − (w_ui−c_i)·r̂^f_ui)·q_if
        #            − Σ_{k≠f} p_uk · s^q_kf ]
        #          ÷ [ Σ_{i∈R_u}(w_ui−c_i)·q_if² + s^q_ff + λ ]
        #
        # where r̂^f_ui = r̂_ui − p_uf·q_if   (prediction without factor f)
        for f in range(K):
            old_puf = p_u[f]

            # Remove contribution of factor f from all predictions
            r_hats_f: np.ndarray = r_hats - old_puf * Q_obs[:, f]  # (|R_u|,)

            # Numerator — observed data term
            num = float(np.dot(w_uis * r_uis - (w_uis - c_is) * r_hats_f, Q_obs[:, f]))
            # Subtract missing-data cache term: Σ_{k≠f} p_uk · Sq[k,f]
            #   = p_u @ Sq[:,f] − p_uf · Sq[f,f]
            num -= float(p_u @ Sq[:, f]) - old_puf * float(Sq[f, f])

            # Denominator
            denom = float(np.dot(w_uis - c_is, Q_obs[:, f] ** 2)) + float(Sq[f, f]) + lambda_reg

            # Update factor f
            p_u[f] = num / max(denom, 1e-10)

            # Refresh prediction cache (line 9 of Algorithm 1)
            r_hats = r_hats_f + p_u[f] * Q_obs[:, f]

        return user_id, p_u.tolist()

    return update_user


def _make_item_updater(
    P_bc: Broadcast,
    Q_bc: Broadcast,
    Sp_bc: Broadcast,
    item_conf_bc: Broadcast,
    K: int,
    lambda_reg: float,
):
    """
    Return a function that maps  (item_id, users_list) → (item_id, new_q_i).

    users_list : list of (user_id, r_ui, w_ui)

    Eq. 13:
      q_if = [ Σ_{u∈R_i}(w_ui·r_ui − (w_ui−c_i)·r̂^f_ui)·p_uf
               − c_i · Σ_{k≠f} q_ik · s^p_kf ]
             ÷ [ Σ_{u∈R_i}(w_ui−c_i)·p_uf² + c_i·s^p_ff + λ ]

    Note: c_i is a *scalar* for each item (item-level popularity weight).
          s^p_kf  is the (k,f) element of S^p = P^T P.
    """

    def update_item(record: tuple[int, list]) -> tuple[int, list]:
        item_id, users = record
        if not users:
            return item_id, Q_bc.value[item_id].tolist()

        # ---------- pull broadcast payloads ----------
        P_local: np.ndarray = P_bc.value  # shape (M, K)
        q_i: np.ndarray = Q_bc.value[item_id].copy()  # shape (K,)
        Sp: np.ndarray = Sp_bc.value  # shape (K, K)
        c_i = float(item_conf_bc.value[item_id])

        # ---------- gather observed-user data ----------
        user_ids = [x[0] for x in users]
        P_obs = np.array([P_local[uid] for uid in user_ids], dtype=np.float64)  # (|R_i|, K)
        r_uis = np.array([x[1] for x in users], dtype=np.float64)
        w_uis = np.array([x[2] for x in users], dtype=np.float64)

        # ---------- initialise prediction cache ----------
        r_hats: np.ndarray = P_obs @ q_i  # (|R_i|,)

        # ---------- element-wise ALS loop ----------
        for f in range(K):
            old_qif = q_i[f]

            r_hats_f: np.ndarray = r_hats - old_qif * P_obs[:, f]

            # Numerator — observed users
            num = float(np.dot(w_uis * r_uis - (w_uis - c_i) * r_hats_f, P_obs[:, f]))
            # Subtract missing-data cache term: c_i · Σ_{k≠f} q_ik · Sp[k,f]
            #   = c_i · (q_i @ Sp[:,f] − q_if · Sp[f,f])
            num -= c_i * (float(q_i @ Sp[:, f]) - old_qif * float(Sp[f, f]))

            # Denominator
            denom = (
                float(np.dot(w_uis - c_i, P_obs[:, f] ** 2)) + c_i * float(Sp[f, f]) + lambda_reg
            )

            q_i[f] = num / max(denom, 1e-10)

            r_hats = r_hats_f + q_i[f] * P_obs[:, f]

        return item_id, q_i.tolist()

    return update_item


# =========================================================================== #
#   eALS model                                                                 #
# =========================================================================== #


class eALS:
    """
    Element-wise Alternating Least Squares for implicit feedback.

    Trains a matrix-factorisation model where missing data is weighted
    by item popularity (Eq. 7–8).  Spark is used to parallelise the
    user-factor and item-factor update passes (Algorithm 1).

    Parameters
    ----------
    n_users, n_items : dataset dimensions
    K                : number of latent factors
    lambda_reg       : L2 regularisation (paper: 0.01)
    c0               : overall missing-data weight scale
    alpha            : popularity exponent (0 → uniform, 0.5 → paper default)
    w_obs            : weight on every observed interaction (paper: 1.0)
    seed             : random seed for factor initialisation
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        K: int = 64,
        lambda_reg: float = 0.01,
        c0: float = 64.0,
        alpha: float = 0.5,
        w_obs: float = 1.0,
        seed: int = 42,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.K = K
        self.lambda_reg = lambda_reg
        self.c0 = c0
        self.alpha = alpha
        self.w_obs = w_obs
        self.seed = seed

        # Initialise factor matrices with small Gaussian noise
        rng = np.random.default_rng(seed)
        self.P: np.ndarray = rng.normal(0, 0.01, (n_users, K))  # (M, K)
        self.Q: np.ndarray = rng.normal(0, 0.01, (n_items, K))  # (N, K)

    # ----------------------------------------------------------------------- #
    #   Cache computations (run on driver; fast via NumPy BLAS)               #
    # ----------------------------------------------------------------------- #

    def _compute_sq(self, item_conf: np.ndarray) -> np.ndarray:
        """
        S^q = Σ_i c_i · q_i q_i^T  ∈  R^{K×K}

        Vectorised form: S^q = (diag(c) Q)^T Q = (c ⊙ Q)^T Q
        Complexity: O(N·K²) via BLAS dgemm — fast even for N=75K, K=128.
        """
        weighted_Q = self.Q * item_conf[:, np.newaxis]  # (N, K)
        return weighted_Q.T @ self.Q  # (K, K)

    def _compute_sp(self) -> np.ndarray:
        """
        S^p = P^T P  ∈  R^{K×K}

        Complexity: O(M·K²).
        """
        return self.P.T @ self.P  # (K, K)

    # ----------------------------------------------------------------------- #
    #   Main training loop                                                     #
    # ----------------------------------------------------------------------- #

    def fit(
        self,
        sc: SparkContext,
        train_rdd,  # RDD of (user_idx, item_idx, r_ui, w_ui)
        item_conf: np.ndarray,  # shape (n_items,)
        n_epochs: int = 20,
        evaluator=None,  # optional Evaluator instance
        test_df=None,  # optional pandas test DataFrame
        train_df=None,  # optional pandas train DataFrame (for eval)
        verbose: bool = True,
    ) -> list[dict]:
        """
        Train for n_epochs using Spark-parallelised factor updates.

        Returns a list of per-epoch result dicts with keys:
            epoch, time_s, hr@{K}, ndcg@{K}
        """
        # ---- build and cache grouped RDDs (created once) ---- #
        #
        # user_items_rdd : (user_id, [(item_id, r_ui, w_ui, c_i)])
        # item_users_rdd : (item_id, [(user_id, r_ui, w_ui)])

        item_conf_init_bc = sc.broadcast(item_conf)

        def add_item_conf(record):
            uid, iid, r, w = record
            c = float(item_conf_init_bc.value[iid])
            return (int(uid), (int(iid), float(r), float(w), c))

        user_items_rdd = train_rdd.map(add_item_conf).groupByKey().mapValues(list).cache()

        item_users_rdd = (
            train_rdd.map(lambda x: (int(x[1]), (int(x[0]), float(x[2]), float(x[3]))))
            .groupByKey()
            .mapValues(list)
            .cache()
        )

        # Force materialisation so timings are fair
        n_users_rdd = user_items_rdd.count()
        n_items_rdd = item_users_rdd.count()
        if verbose:
            print(f"[eALS] Cached RDDs — {n_users_rdd:,} user groups, {n_items_rdd:,} item groups")

        history: list[dict] = []

        for epoch in range(n_epochs):
            t0 = time.time()

            # ============================================================== #
            #   STEP A: Update user factors (Eq. 12, Algorithm 1 lines 4–11) #
            # ============================================================== #

            Sq = self._compute_sq(item_conf)

            # Broadcast current model state to all workers
            Q_bc = sc.broadcast(self.Q)
            P_bc = sc.broadcast(self.P)
            Sq_bc = sc.broadcast(Sq)

            user_updater = _make_user_updater(Q_bc, P_bc, Sq_bc, self.K, self.lambda_reg)
            new_user_factors = user_items_rdd.map(user_updater).collect()

            # Write results back into driver-side P matrix
            for user_id, p_u in new_user_factors:
                self.P[user_id] = np.array(p_u, dtype=np.float64)

            Q_bc.unpersist()
            P_bc.unpersist()
            Sq_bc.unpersist()

            # ============================================================== #
            #   STEP B: Update item factors (Eq. 13, Algorithm 1 lines 12–19)#
            # ============================================================== #

            Sp = self._compute_sp()

            P_bc = sc.broadcast(self.P)
            Q_bc = sc.broadcast(self.Q)
            Sp_bc = sc.broadcast(Sp)
            item_conf_bc = sc.broadcast(item_conf)

            item_updater = _make_item_updater(
                P_bc, Q_bc, Sp_bc, item_conf_bc, self.K, self.lambda_reg
            )
            new_item_factors = item_users_rdd.map(item_updater).collect()

            for item_id, q_i in new_item_factors:
                self.Q[item_id] = np.array(q_i, dtype=np.float64)

            P_bc.unpersist()
            Q_bc.unpersist()
            Sp_bc.unpersist()
            item_conf_bc.unpersist()

            elapsed = time.time() - t0

            # ============================================================== #
            #   Evaluation                                                     #
            # ============================================================== #
            result: dict = {"epoch": epoch + 1, "time_s": round(elapsed, 2)}

            if evaluator is not None and test_df is not None:
                hr, ndcg = evaluator.evaluate(self.P, self.Q, test_df, train_df)
                result[f"hr@{evaluator.top_k}"] = round(hr, 4)
                result[f"ndcg@{evaluator.top_k}"] = round(ndcg, 4)
                if verbose:
                    print(
                        f"  Epoch {epoch + 1:>3}/{n_epochs} | "
                        f"time: {elapsed:>6.1f}s | "
                        f"HR@{evaluator.top_k}: {hr:.4f} | "
                        f"NDCG@{evaluator.top_k}: {ndcg:.4f}"
                    )
            else:
                if verbose:
                    print(f"  Epoch {epoch + 1:>3}/{n_epochs} | time: {elapsed:>6.1f}s")

            history.append(result)

        # Clean up
        user_items_rdd.unpersist()
        item_users_rdd.unpersist()
        item_conf_init_bc.unpersist()

        return history

    # ----------------------------------------------------------------------- #
    #   Inference                                                              #
    # ----------------------------------------------------------------------- #

    def predict_user(self, user_id: int) -> np.ndarray:
        """Return scores for ALL items for the given user (shape: (N,))."""
        return self.Q @ self.P[user_id]

    def predict_all(self) -> np.ndarray:
        """
        Return the full (M × N) score matrix.
        Caution: O(M · N) memory — use only for small datasets.
        """
        return self.P @ self.Q.T

    def __repr__(self) -> str:
        return (
            f"eALS(n_users={self.n_users}, n_items={self.n_items}, "
            f"K={self.K}, λ={self.lambda_reg}, c0={self.c0}, α={self.alpha})"
        )
