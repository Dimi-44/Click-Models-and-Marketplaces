"""Fast synthetic tests: no access to the real dataset or model training at scale."""

import unittest

import numpy as np
import pandas as pd

from marketplace_ml.data import choose_pilot_minute
from marketplace_ml.encoding import SparseCategoricalEncoder
from marketplace_ml.metrics import probability_metrics
from marketplace_ml.models import ConstantCTR, SmoothedItemPositionCTR
from marketplace_ml.splits import SPLITS, partition_pilot, split_minute_counts


class ProtocolTests(unittest.TestCase):
    def test_deterministic_minute_selection(self):
        minutes = [f"2019-11-24T00:{minute:02d}" for minute in range(60)]
        first = [choose_pilot_minute(m, .2, 20261114) for m in minutes]
        self.assertEqual(first, [choose_pilot_minute(m, .2, 20261114) for m in minutes])
        self.assertTrue(any(first))
        self.assertFalse(all(first))

    def test_group_split_uses_whole_minutes_and_daily_balance(self):
        counts = {f"2019-11-{day:02d}T{hour:02d}:{minute:02d}": 100
                  for day in (24, 25, 26) for hour in (0, 1)
                  for minute in range(60)}
        a = split_minute_counts(counts, seed=42)
        self.assertEqual(a, split_minute_counts(dict(reversed(list(counts.items()))), seed=42))
        self.assertEqual(len(a), len(counts))
        for day in (24, 25, 26):
            daily = [a[k] for k in counts if k[:10] == f"2019-11-{day:02d}"]
            for key, target in (("train", .7), ("validation", .15), ("internal_test", .15)):
                self.assertLess(abs(daily.count(key) / len(daily) - target), .02)
        self.assertEqual(set(a.values()), set(SPLITS))

    def test_pilot_membership_is_disjoint_by_minute_and_index(self):
        counts = {f"2019-11-24T00:{i:02d}": 2 for i in range(30)}
        assignment = split_minute_counts(counts, seed=123)
        frame = pd.DataFrame({
            "minute_utc": [minute for minute in counts for _ in range(2)],
            "Unnamed: 0": np.arange(60, dtype=np.int64),
            "click": [0, 1] * 30,
        })
        part = partition_pilot(frame, assignment)
        self.assertEqual(sum(len(p) for p in part.values()), 60)
        self.assertTrue(all(len(p) > 0 for p in part.values()))
        self.assertEqual(sum(p["Unnamed: 0"].nunique() for p in part.values()), 60)

    def test_explicit_unknown_indicator_and_only_training_levels(self):
        train = pd.DataFrame({"item_id": [1, 2, 1], "user_feature_0": ["A", "B", "A"]})
        external = pd.DataFrame({"item_id": [1, 200], "user_feature_0": ["X", "B"]})
        encoder = SparseCategoricalEncoder(("item_id", "user_feature_0")).fit(train)
        x_train, train_unknown = encoder.transform(train)
        x_external, counts = encoder.transform(external)
        self.assertEqual(train_unknown, {"item_id": 0, "user_feature_0": 0})
        self.assertEqual(counts, {"item_id": 1, "user_feature_0": 1})
        self.assertEqual(x_train.shape[1], x_external.shape[1])
        self.assertEqual(float(x_external[0, encoder.offset_["user_feature_0"]]), 1.0)
        self.assertEqual(float(x_external[1, encoder.offset_["item_id"]]), 1.0)
        self.assertNotIn("X", encoder.levels_["user_feature_0"])
        self.assertTrue(np.all(np.diff(x_external.indptr) == 2))

    def test_baselines_preserve_rate_and_smoothing(self):
        train = pd.DataFrame({"item_id": [1, 1, 2, 2], "position": [1, 1, 2, 2],
                              "click": [1, 0, 0, 0]})
        probe = pd.DataFrame({"item_id": [1, 2, 100], "position": [1, 2, 3]})
        constant = ConstantCTR().fit(train).predict_proba(probe)
        smooth = SmoothedItemPositionCTR(prior_strength=2).fit(train).predict_proba(probe)
        np.testing.assert_allclose(constant, [.25, .25, .25])
        np.testing.assert_allclose(smooth, [.375, .125, .25])

    def test_metrics_for_rare_binary_outcome(self):
        y = np.array([0, 0, 0, 0, 1])
        good = probability_metrics(y, np.array([.01, .01, .01, .01, .95]))
        poor = probability_metrics(y, np.full(5, .20))
        self.assertLess(good["log_loss"], poor["log_loss"])
        self.assertEqual(good["roc_auc"], 1.0)
        self.assertEqual(good["average_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
