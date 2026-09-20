"""
evaluation.py
=============
Hit Ratio (HR@K) and Normalised Discounted Cumulative Gain (NDCG@K)
for top-K recommendation, following the leave-one-out offline evaluation
protocol of He et al. (SIGIR 2016), Section 5.1.

Offline protocol recap
-----------------------
* The test set contains exactly ONE held-out interaction per user
  (their chronologically last interaction).
* Items already consumed in training are EXCLUDED from the candidate list
  before ranking — this avoids trivially recommending known items.
* Every remaining item is a candidate (full ranking, not sampled).
* HR@K  = fraction of test users for whom the test item lands in the
  top-K ranked list.
* NDCG@K = average normalised discounted gain, where a hit at rank r
  contributes 1 / log2(r+1).

Efficiency note
---------------
Computing scores for every (user, item) pair takes O(M·N·K) time.
For large datasets we sub-sample up to `max_eval_users` test rows at
random — enough to get stable estimates while keeping evaluation fast.
"""

import numpy as np
import pandas as pd


class Evaluator:
    """
    Evaluate a (P, Q) factorisation under the leave-one-out protocol.

    Parameters
    ----------
    top_k         : cutoff position (paper uses 100)
    max_eval_users: cap on the number of test users scored per evaluation
                    call.  None = evaluate all.  2 000 is typically enough
                    for stable estimates.
    seed          : random seed for user sub-sampling
    """

    def __init__(
        self,
        top_k: int = 100,
        max_eval_users: int | None = 2000,
        seed: int = 42,
    ):
        self.top_k = top_k
        self.max_eval_users = max_eval_users
        self.rng = np.random.default_rng(seed)
        self._train_user_items: dict | None = None  # cached lookup

    # ----------------------------------------------------------------------- #
    #   Build training-item lookup (call once before the epoch loop)          #
    # ----------------------------------------------------------------------- #

    def build_train_lookup(self, train_df: pd.DataFrame) -> None:
        """
        Pre-compute the set of training items per user so they can be
        efficiently excluded during ranking.

        Must be called before the first evaluate() call.  Calling it again
        rebuilds the lookup (useful if train_df changes).
        """
        self._train_user_items: dict = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # ----------------------------------------------------------------------- #
    #   Core evaluation                                                        #
    # ----------------------------------------------------------------------- #

    def evaluate(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame | None = None,
    ) -> tuple[float, float]:
        """
        Compute HR@K and NDCG@K.

        Parameters
        ----------
        P        : user factor matrix  (M, K)
        Q        : item factor matrix  (N, K)
        test_df  : test interactions with columns [user_idx, item_idx]
        train_df : training interactions; used to build the exclusion set.
                   If None, previously built lookup is used.

        Returns
        -------
        hr   : float  ∈ [0, 1]
        ndcg : float  ∈ [0, 1]
        """
        # Build or reuse training-item exclusion sets
        if train_df is not None:
            self.build_train_lookup(train_df)
        if self._train_user_items is None:
            raise RuntimeError(
                "Call build_train_lookup(train_df) before evaluate(), or pass train_df directly."
            )

        # Sub-sample test users if needed
        eval_df = test_df
        if self.max_eval_users is not None and len(test_df) > self.max_eval_users:
            eval_df = test_df.sample(
                n=self.max_eval_users, random_state=int(self.rng.integers(1 << 31))
            )

        hr_list = []
        ndcg_list = []

        for row in eval_df.itertuples(index=False):
            u = int(row.user_idx)
            i = int(row.item_idx)

            # Score all items for this user: shape (N,)
            scores = Q @ P[u]

            # Mask out training items (set to -inf so they can't outrank test item)
            for train_item in self._train_user_items.get(u, set()):
                scores[train_item] = -np.inf

            # Rank of the test item (1-indexed)
            # rank = 1 + number of items with score strictly greater than test score
            test_score = scores[i]
            rank = int((scores > test_score).sum()) + 1

            hr = 1.0 if rank <= self.top_k else 0.0
            ndcg = (1.0 / np.log2(rank + 1)) if rank <= self.top_k else 0.0

            hr_list.append(hr)
            ndcg_list.append(ndcg)

        hr_mean = float(np.mean(hr_list))
        ndcg_mean = float(np.mean(ndcg_list))
        return hr_mean, ndcg_mean

    # ----------------------------------------------------------------------- #
    #   Convenience: evaluate a list of models for a comparison table         #
    # ----------------------------------------------------------------------- #

    @staticmethod
    def format_history(history: list, top_k: int) -> pd.DataFrame:
        """Convert the list of epoch-result dicts to a pretty DataFrame."""
        df = pd.DataFrame(history)
        rename = {f"hr@{top_k}": f"HR@{top_k}", f"ndcg@{top_k}": f"NDCG@{top_k}"}
        df = df.rename(columns=rename)
        return df
