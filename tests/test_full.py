"""Synthetic tests for compact full-scale execution and frozen subset design."""

from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from marketplace_ml.data import USER_COLUMNS
from marketplace_ml.full_data import DTYPE, UNKNOWN_CODE, pack_policy
from marketplace_ml.full_encoding import CompactSparseEncoder, CONTEXTUAL
from marketplace_ml.prepare_full import minute_nested_order
from marketplace_ml.tune_full import log_loss_binary, rates_item_position, score_item_position


class FullProtocolTests(unittest.TestCase):
    def test_sparse_unknowns_and_sparse_density(self):
        rows = np.zeros(4, dtype=DTYPE)
        rows["item_id"] = [1, 2, 1, 2]
        rows["position"] = [1, 2, 1, 2]
        rows["user_feature_0"] = [2, 3, 2, 3]
        train = rows[:3]
        unseen = rows[3:].copy()
        unseen["user_feature_0"] = UNKNOWN_CODE
        encoder = CompactSparseEncoder(CONTEXTUAL).fit(train)
        a, known = encoder.transform(train)
        b, missing = encoder.transform(unseen)
        self.assertEqual(a.nnz, len(train) * len(CONTEXTUAL))
        self.assertEqual(b.nnz, len(unseen) * len(CONTEXTUAL))
        self.assertEqual(known["user_feature_0"], 0)
        self.assertEqual(missing["user_feature_0"], 1)
        self.assertEqual(float(b[0, encoder.offsets["user_feature_0"]]), 1.0)

    def test_nested_minute_order_is_block_contiguous(self):
        rows = np.zeros(20, dtype=DTYPE)
        rows["minute_id"] = np.repeat(np.arange(5), 4)
        original = np.arange(20, dtype=np.int32)
        order = minute_nested_order(rows, original, 20261114)
        self.assertTrue(np.array_equal(np.sort(order), original))
        minutes = rows["minute_id"][order]
        for minute in range(5):
            positions = np.flatnonzero(minutes == minute)
            self.assertEqual(int(positions.max() - positions.min() + 1), len(positions))
        np.testing.assert_array_equal(order[:4], order[:12][:4])
        np.testing.assert_array_equal(order, minute_nested_order(rows, original, 20261114))

    def test_b1_aggregate_predictor(self):
        rows = np.zeros(5, dtype=DTYPE)
        rows["item_id"] = [1, 1, 2, 2, 3]
        rows["position"] = [1, 1, 2, 2, 1]
        rows["click"] = [1, 0, 0, 0, 0]
        rates, prevalence = rates_item_position(rows[:4], prior=2)
        self.assertEqual(prevalence, .25)
        np.testing.assert_allclose(score_item_position(rows, rates), [.375, .375, .125, .125, .25])
        self.assertLess(log_loss_binary(np.array([0, 1]), np.array([.1, .9])),
                        log_loss_binary(np.array([0, 1]), np.array([.5, .5])))
        self.assertAlmostEqual(log_loss_binary(np.array([0, 1], dtype=np.uint8),
                                               np.array([.1, .9])),
                               log_loss_binary(np.array([0, 1], dtype=np.int8),
                                               np.array([.1, .9])))

    def test_zip_stream_preserves_duplicates_and_unknown_policy_codes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "tiny.zip"
            raw = pd.DataFrame({
                "Unnamed: 0": [0, 1, 2],
                "timestamp": ["2019-11-24 00:00:00.100+00:00"] * 2
                             + ["2019-11-24 00:01:00.100+00:00"],
                "item_id": [1, 1, 2], "position": [1, 1, 2],
                "click": [0, 0, 1],
                **{feature: ["A", "A", "B"] for feature in USER_COLUMNS},
            })
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("all/All bts.csv", raw.to_csv(index=False))
                external = raw.copy()
                external["user_feature_0"] = ["C", "A", "C"]
                z.writestr("all/All random.csv", external.to_csv(index=False))
            minutes = pd.DataFrame({"minute_utc": ["2019-11-24 00:00", "2019-11-24 00:01"],
                                    "n_bts": [2, 1]})
            codebooks = {feature: {} for feature in USER_COLUMNS}
            pack_policy(archive, "BTS", minutes, root / "bts.npy", codebooks=codebooks,
                        n_expected=3, allow_new_codes=True, chunk_size=2)
            bts = np.load(root / "bts.npy")
            self.assertEqual(len(bts), 3)
            self.assertTrue(np.array_equal(bts[0], bts[1]))
            pack_policy(archive, "Random", minutes, root / "rand.npy", codebooks=codebooks,
                        n_expected=3, allow_new_codes=False, chunk_size=2)
            external = np.load(root / "rand.npy")
            self.assertEqual(int(external["user_feature_0"][0]), UNKNOWN_CODE)
            self.assertEqual(int(external["user_feature_0"][1]), int(bts[0]["user_feature_0"]))
            self.assertEqual(len(codebooks["user_feature_0"]), 2)


if __name__ == "__main__":
    unittest.main()
