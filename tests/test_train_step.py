import tempfile
import unittest

import pandas as pd
import torch

from rda.dataset import encode_batch
from rda.features import feature_dim
from rda.fixtures import write_fixture
from rda.io import protocol, rename_columns
from rda.models import Agent, train_step
from rda.trajectories import build_transitions


class TrainStepTests(unittest.TestCase):
    def test_one_train_step_on_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            tiles, parts = write_fixture(tmp)
            tdf, _ = rename_columns(pd.read_csv(tiles))
            pdf, _ = rename_columns(pd.read_csv(parts))
            proto = protocol()
            steps, _ = build_transitions(tdf, pdf, proto)
        batch = encode_batch(steps, proto)
        agent = Agent(feature_dim(), 32, "rda")
        opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
        dual = {"lam_miss": 0.1, "lam_over": 0.1}
        cfg = {
            "gamma": 0.95,
            "expectile": 0.7,
            "awr_temp": 0.5,
            "cql_alpha": 1.0,
            "kappa_miss": 0.2,
            "kappa_over": 0.1,
        }
        out = train_step(agent, opt, batch, cfg, dual)
        self.assertTrue(out["loss"] == out["loss"])
        q = agent.q(torch.tensor(batch[0], dtype=torch.float32))
        self.assertEqual(q.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
