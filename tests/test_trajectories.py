import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rda.fixtures import write_fixture
from rda.io import protocol, rename_columns
from rda.split import patient_split
from rda.models import behavior_rates, protocol_rates
from rda.trajectories import ACQUIRE, EXCISE, WATCH, build_transitions, outcome_under_action


def load_fixture(dirpath):
    tiles, parts = write_fixture(dirpath)
    tdf, _ = rename_columns(pd.read_csv(tiles))
    pdf, _ = rename_columns(pd.read_csv(parts))
    return build_transitions(tdf, pdf, protocol())


class TrajectoryTests(unittest.TestCase):
    def test_fixture_actions_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps, stats = load_fixture(tmp)
        self.assertEqual(stats["reasons"]["hops_control"], 1)
        self.assertEqual(stats["reasons"]["short"], 1)
        g1 = [s for s in steps if "General_1" in s["participant"]]
        self.assertEqual([s["action"] for s in g1], [WATCH, ACQUIRE, WATCH])
        self.assertEqual(g1[1]["derm_visible"], 1)
        h2 = [s for s in steps if "HighRisk_2" in s["participant"]]
        self.assertEqual(h2[0]["action"], WATCH)
        self.assertEqual(h2[0]["cost_miss"], 1.0)
        self.assertEqual(h2[-1]["action"], EXCISE)
        self.assertEqual(h2[-1]["alignment"], "visit")
        g4 = [s for s in steps if "General_4" in s["participant"]]
        self.assertEqual(g4[-1]["action"], EXCISE)
        self.assertEqual(g4[-1]["alignment"], "lesion_level")
        self.assertEqual(g4[0]["cost_miss"], 0.0)
        self.assertEqual(g4[-1]["cost_miss"], 0.0)
        self.assertFalse(any("HighRisk_3" in s["participant"] for s in steps))

    def test_v2_gate_ignores_lesion_level_share(self):
        from rda.trajectories import dataset_gate

        proto = protocol()
        stats = {
            "n_transitions": 10,
            "reasons": {
                "action_WatchTBP": 8,
                "action_AcquireDerm": 1,
                "action_ReferExcise": 1,
            },
        }
        self.assertEqual(dataset_gate(stats, proto), "pass")

    def test_patient_split_no_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            steps, _ = load_fixture(tmp)
        grouped, counts = patient_split(steps, protocol())
        ids = {split: {r["participant"] for r in rows} for split, rows in grouped.items()}
        self.assertFalse(ids.get("train", set()) & ids.get("test", set()))
        self.assertEqual(sum(counts.values()), len({s["participant"] for s in steps}))

    def test_outcome_matches_logged_action(self):
        proto = protocol()
        with tempfile.TemporaryDirectory() as tmp:
            steps, _ = load_fixture(tmp)
        for step in steps:
            out = outcome_under_action(step, step["action"], proto)
            self.assertEqual(out["reward"], step["reward"])
            self.assertEqual(out["cost_miss"], step["cost_miss"])
            self.assertEqual(out["cost_over"], step["cost_over"])
        logged = behavior_rates(steps)
        rates = protocol_rates(steps, [s["action"] for s in steps], proto)
        self.assertAlmostEqual(rates["return"], logged["return"])
        self.assertAlmostEqual(rates["cost_miss"], logged["cost_miss"])
        self.assertAlmostEqual(rates["cost_over"], logged["cost_over"])

    def test_counterfactual_costs(self):
        proto = protocol()
        with tempfile.TemporaryDirectory() as tmp:
            steps, _ = load_fixture(tmp)
        g1 = [s for s in steps if "General_1" in s["participant"]]
        over = outcome_under_action(g1[0], EXCISE, proto)
        self.assertEqual(over["cost_over"], 1.0)
        self.assertEqual(over["cost_miss"], 0.0)
        h2 = [s for s in steps if "HighRisk_2" in s["participant"]]
        catch = outcome_under_action(h2[0], EXCISE, proto)
        self.assertEqual(catch["cost_miss"], 0.0)
        self.assertEqual(catch["reward"], proto["rewards"]["catch_cancer"])
        g4 = [s for s in steps if "General_4" in s["participant"]]
        self.assertEqual(outcome_under_action(g4[0], WATCH, proto)["cost_miss"], 0.0)
        self.assertEqual(outcome_under_action(g4[-1], WATCH, proto)["cost_miss"], 1.0)


if __name__ == "__main__":
    unittest.main()
