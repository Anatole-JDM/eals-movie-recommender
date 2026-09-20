"""
tests/test_data_loader.py
==========================
Unit tests for the pandas-only data import / filtering functions in
data_loader.py: load_and_filter, encode_ids, leave_one_out_split and
compute_item_confidence.

These tests use small synthetic CSVs / DataFrames only — no network
access and no Spark session are required, so they run fast in CI.
"""

import numpy as np
import pandas as pd

from data_loader import (
    compute_item_confidence,
    encode_ids,
    leave_one_out_split,
    load_and_filter,
)

# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _write_raw_csv(path, rows):
    """Write rows of (user_id, item_id, rating, timestamp) with no header,
    matching the format load_and_filter expects."""
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    df.to_csv(path, header=False, index=False)
    return path


# --------------------------------------------------------------------------- #
#  load_and_filter                                                             #
# --------------------------------------------------------------------------- #


class TestLoadAndFilter:
    def test_keeps_users_and_items_at_or_above_threshold(self, tmp_path):
        # u1 has 4 interactions spread over 2 items (each item seen twice,
        # so both u1 and its items clear the threshold on their own).
        # u2 has a single interaction and must be dropped, which does not
        # push either shared item below the threshold.
        rows = [
            ("u1", "i1", 5.0, 100),
            ("u1", "i1", 4.0, 101),
            ("u1", "i2", 3.0, 102),
            ("u1", "i2", 2.0, 103),
            ("u2", "i1", 5.0, 104),
        ]
        csv_path = _write_raw_csv(tmp_path / "raw.csv", rows)

        df = load_and_filter(str(csv_path), min_interactions=2, sample_fraction=1.0)

        assert set(df["user_id"]) == {"u1"}
        assert len(df) == 4

    def test_iterative_filtering_cascades(self, tmp_path):
        # After removing item "i4" (1 interaction), user "u3" drops below
        # threshold too, so the *second* filtering pass must remove u3.
        rows = [
            ("u1", "i1", 5.0, 100),
            ("u1", "i2", 5.0, 101),
            ("u2", "i1", 5.0, 102),
            ("u2", "i2", 5.0, 103),
            ("u3", "i1", 5.0, 104),
            ("u3", "i4", 5.0, 105),  # i4 only appears once -> gets dropped
        ]
        csv_path = _write_raw_csv(tmp_path / "raw.csv", rows)

        df = load_and_filter(str(csv_path), min_interactions=2, sample_fraction=1.0)

        # u3 only has 1 remaining interaction (with i1) after i4 is dropped
        assert "u3" not in set(df["user_id"])
        assert set(df["user_id"]) == {"u1", "u2"}

    def test_empty_result_when_threshold_too_high(self, tmp_path):
        rows = [("u1", "i1", 5.0, 100), ("u2", "i2", 5.0, 101)]
        csv_path = _write_raw_csv(tmp_path / "raw.csv", rows)

        df = load_and_filter(str(csv_path), min_interactions=10, sample_fraction=1.0)

        assert len(df) == 0

    def test_sample_fraction_reduces_row_count_deterministically(self, tmp_path):
        rows = [(f"u{i}", f"i{i % 3}", 5.0, 100 + i) for i in range(50)]
        csv_path = _write_raw_csv(tmp_path / "raw.csv", rows)

        df_a = load_and_filter(str(csv_path), min_interactions=1, sample_fraction=0.5, seed=42)
        df_b = load_and_filter(str(csv_path), min_interactions=1, sample_fraction=0.5, seed=42)

        # Same seed -> identical sample
        pd.testing.assert_frame_equal(df_a.reset_index(drop=True), df_b.reset_index(drop=True))
        assert len(df_a) < 50

    def test_output_columns(self, tmp_path):
        rows = [("u1", "i1", 5.0, 100), ("u1", "i2", 4.0, 101)]
        csv_path = _write_raw_csv(tmp_path / "raw.csv", rows)

        df = load_and_filter(str(csv_path), min_interactions=1, sample_fraction=1.0)

        assert list(df.columns) == ["user_id", "item_id", "timestamp"]


# --------------------------------------------------------------------------- #
#  encode_ids                                                                  #
# --------------------------------------------------------------------------- #


class TestEncodeIds:
    def test_dense_zero_based_indices(self):
        df = pd.DataFrame(
            {
                "user_id": ["u2", "u1", "u2"],
                "item_id": ["iB", "iA", "iA"],
            }
        )

        encoded, user2id, item2id, n_users, n_items = encode_ids(df)

        assert n_users == 2
        assert n_items == 2
        assert set(encoded["user_idx"]) == {0, 1}
        assert set(encoded["item_idx"]) == {0, 1}

    def test_mapping_is_bijective_and_sorted(self):
        df = pd.DataFrame(
            {
                "user_id": ["u3", "u1", "u2"],
                "item_id": ["i1", "i1", "i1"],
            }
        )

        _, user2id, item2id, n_users, _ = encode_ids(df)

        # sorted() order -> u1=0, u2=1, u3=2
        assert user2id == {"u1": 0, "u2": 1, "u3": 2}
        assert len(set(user2id.values())) == n_users

    def test_consistent_mapping_across_rows(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "item_id": ["iA", "iB", "iA"],
            }
        )

        encoded, user2id, item2id, _, _ = encode_ids(df)

        u1_rows = encoded[encoded["user_id"] == "u1"]
        assert (u1_rows["user_idx"] == user2id["u1"]).all()


# --------------------------------------------------------------------------- #
#  leave_one_out_split                                                         #
# --------------------------------------------------------------------------- #


class TestLeaveOneOutSplit:
    def test_exactly_one_test_row_per_user(self):
        df = pd.DataFrame(
            {
                "user_idx": [0, 0, 0, 1, 1],
                "item_idx": [0, 1, 2, 0, 1],
                "timestamp": [10, 20, 30, 5, 15],
            }
        )

        train, test = leave_one_out_split(df)

        assert len(test) == df["user_idx"].nunique()
        assert set(test["user_idx"]) == set(df["user_idx"].unique())
        assert test["user_idx"].value_counts().max() == 1

    def test_held_out_row_is_chronologically_last(self):
        df = pd.DataFrame(
            {
                "user_idx": [0, 0, 0],
                "item_idx": [10, 20, 30],
                "timestamp": [100, 300, 200],  # last by time is item_idx=20
            }
        )

        train, test = leave_one_out_split(df)

        assert test.iloc[0]["item_idx"] == 20
        assert set(train["item_idx"]) == {10, 30}

    def test_train_and_test_are_disjoint_and_complete(self):
        df = pd.DataFrame(
            {
                "user_idx": [0, 0, 1, 1, 1],
                "item_idx": [1, 2, 3, 4, 5],
                "timestamp": [1, 2, 3, 4, 5],
            }
        )

        train, test = leave_one_out_split(df)

        assert len(train) + len(test) == len(df)
        assert set(train["item_idx"]).isdisjoint(set(test["item_idx"]))


# --------------------------------------------------------------------------- #
#  compute_item_confidence                                                     #
# --------------------------------------------------------------------------- #


class TestComputeItemConfidence:
    def test_alpha_zero_is_uniform(self):
        train_df = pd.DataFrame({"item_idx": [0, 0, 1, 2, 2, 2]})
        n_items = 3

        c = compute_item_confidence(train_df, n_items, c0=30.0, alpha=0.0)

        assert np.allclose(c, 10.0)  # c0 / n_items for every item

    def test_confidence_sums_to_c0_for_nonzero_alpha(self):
        train_df = pd.DataFrame({"item_idx": [0, 0, 1, 2, 2, 2]})
        n_items = 3

        c = compute_item_confidence(train_df, n_items, c0=64.0, alpha=0.5)

        assert c.shape == (n_items,)
        assert np.isclose(c.sum(), 64.0)

    def test_more_popular_items_get_higher_confidence_when_alpha_positive(self):
        # item 2 is the most frequent -> should receive the highest weight
        train_df = pd.DataFrame({"item_idx": [0, 1, 2, 2, 2, 2]})
        n_items = 3

        c = compute_item_confidence(train_df, n_items, c0=64.0, alpha=0.5)

        assert c[2] > c[0]
        assert c[2] > c[1]

    def test_unseen_items_get_zero_or_minimal_confidence(self):
        # item_idx 2 never appears in train_df, n_items covers it anyway
        train_df = pd.DataFrame({"item_idx": [0, 0, 1, 1]})
        n_items = 3

        c = compute_item_confidence(train_df, n_items, c0=64.0, alpha=0.5)

        assert c[2] < c[0]
        assert c[2] < c[1]
