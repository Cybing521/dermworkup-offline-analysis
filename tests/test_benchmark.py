import tempfile
import unittest

from rda.benchmark import protocol_consistency
from rda.io import protocol
from rda.metrics import auprc, auroc, average_precision_grouped, average_precision_rank, per_class, trapezoidal_pr_auc
from rda.split import patient_split
from rda.trajectories import ACQUIRE, EXCISE, WATCH
from rda.verify import split_integrity

from tests.test_trajectories import load_fixture


class MetricTests(unittest.TestCase):
    def test_auroc_separates_scores(self):
        y = [0, 0, 1, 1]
        self.assertAlmostEqual(auroc(y, [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(auroc(y, [0.0, 0.0, 0.0, 0.0]), 0.5)
        self.assertGreater(auprc(y, [0.1, 0.2, 0.8, 0.9]), auprc(y, [0.0, 0.0, 0.0, 0.0]))

    def test_grouped_ap_obeys_visit_rate_bound(self):
        visits = [1] * 8 + [2] * 8
        y = [1, 0, 0, 0, 1, 0, 0, 0] + [0, 0, 1, 0, 0, 0, 0, 0]
        scores = [-float(v) for v in visits]
        grouped = average_precision_grouped(y, scores)
        rank_ap = average_precision_rank(y, scores)
        max_rate = 2 / 8
        self.assertLessEqual(grouped, max_rate + 1e-12)
        self.assertGreaterEqual(rank_ap, grouped)
        self.assertGreater(trapezoidal_pr_auc(y, scores), grouped)

    def test_grouped_ap_matches_rank_ap_without_ties(self):
        y = [0, 1, 0, 1]
        scores = [0.1, 0.4, 0.2, 0.9]
        self.assertAlmostEqual(average_precision_grouped(y, scores), average_precision_rank(y, scores))

    def test_standard_ap_invariant_to_row_permutation(self):
        y = [1, 0, 0, 0, 1, 0, 0, 0] + [0, 0, 1, 0]
        scores = [-1.0] * 8 + [-2.0] * 4
        base = average_precision_grouped(y, scores)
        perm = list(range(len(y)))
        perm = perm[::-1]
        y2 = [y[i] for i in perm]
        s2 = [scores[i] for i in perm]
        self.assertAlmostEqual(average_precision_grouped(y2, s2), base)
        self.assertNotEqual(average_precision_rank(y, scores), average_precision_rank(y2, s2))

    def test_per_class_perfect(self):
        y = [0, 1, 2]
        out = per_class(y, y)
        self.assertEqual(out["ReferExcise"]["recall"], 1.0)
        self.assertEqual(out["AcquireDerm"]["support"], 1)


class BenchmarkTests(unittest.TestCase):
    def test_fixture_acquire_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps, _ = load_fixture(tmp)
        report = protocol_consistency(steps)
        self.assertEqual(report["acquire_rule_mismatch"], 0)
        self.assertGreater(report["n_transitions"], 0)
        self.assertEqual({s["action"] for s in steps}, {WATCH, ACQUIRE, EXCISE})

    def test_patient_split_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps, _ = load_fixture(tmp)
        grouped, _ = patient_split(steps, protocol())
        report = split_integrity(grouped)
        self.assertTrue(report["ok"])
        self.assertEqual(report["patient_leak_n"], 0)
        self.assertEqual(report["lesion_leak_n"], 0)


if __name__ == "__main__":
    unittest.main()
