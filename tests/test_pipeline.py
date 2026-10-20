"""Training loop, checkpoint round trip and evaluation on the CPU, without gradient updates."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from slot_state import agents, checkpoint, config
from slot_state.environment import ACTORS, STATES, make_environment, public_capacities
from slot_state.evaluate import load_policy, rollout
from slot_state.train import train

N_MAX = 4  # about 250 transitions per episode, below every warm-up threshold, so no update runs


class PipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.folder = tempfile.TemporaryDirectory()
        self.saved = {k: os.environ.get(k) for k in ('SLOT_STATE_CHECKPOINT_DIR', 'SLOT_STATE_OUTPUT_DIR')}
        os.environ['SLOT_STATE_CHECKPOINT_DIR'] = str(Path(self.folder.name) / 'checkpoints')
        os.environ['SLOT_STATE_OUTPUT_DIR'] = str(Path(self.folder.name) / 'outputs')

    def tearDown(self):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.folder.cleanup()

    def test_train_save_load_evaluate(self):
        device = torch.device('cpu')
        for actor in ACTORS:
            for state in STATES:
                train(actor, state, N_MAX, seed=1, episodes=1, device=device)
                policy, ckpt, _ = load_policy(config.checkpoint_path(actor, N_MAX, state, 1), device)
                self.assertEqual((ckpt['actor'], ckpt['state'], ckpt['n_max'], ckpt['episode']),
                                 (actor, state, N_MAX, 1))
                with torch.inference_mode():
                    summary, slot_means = rollout(policy, ckpt, config.EVAL_SEEDS[0], device)
                self.assertEqual(len(slot_means), config.EVAL_SLOTS)
                self.assertTrue(0 < summary['post_mean'] <= summary['post_p99'])

    def test_checkpoint_keeps_the_policy(self):
        device = torch.device('cpu')
        for actor in ACTORS:
            torch.manual_seed(0)
            agent, _ = agents.make_agent(actor, device)
            network = getattr(agent, agents.POLICY_ATTRIBUTE[actor])
            env = make_environment(actor, 'slot', N_MAX, public_capacities())
            path = Path(self.folder.name) / actor / 'best.pt'
            bank = np.random.default_rng(0).random((config.TIME_SLOTS, N_MAX, config.NUM_SERVERS))
            checkpoint.save(path, actor=actor, state='slot', n_max=N_MAX, seed=1, policy=network,
                            average_delays=[2.0, 1.5],
                            environment=checkpoint.environment_spec(env, env.ES_capacities, N_MAX),
                            hyperparameters=agents.HYPERPARAMETERS[actor],
                            feedback_history=bank if actor == 'fdedge' else None)
            policy, ckpt, _ = load_policy(path, device)
            probe = torch.rand(4, config.STATE_DIM) * 50
            if actor == 'fdedge':
                stored = torch.softmax(torch.rand(4, config.NUM_SERVERS), -1)
                torch.manual_seed(1)
                expected = network(probe, stored)
                torch.manual_seed(1)
                self.assertTrue(torch.equal(policy(probe, stored), expected))
                self.assertTrue(np.array_equal(ckpt['feedback_history'].numpy(), bank))
            else:
                self.assertTrue(torch.equal(policy(probe), network(probe)))


if __name__ == '__main__':
    unittest.main()
