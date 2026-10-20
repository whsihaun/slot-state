"""The state option changes only what the scheduler observes (CPU, no training)."""
import hashlib
import unittest

import numpy as np

from slot_state import config
from slot_state.environment import (ACTORS, STATES, TrainingTasks, evaluation_tasks, make_environment,
                                    public_capacities)

UPSTREAM_SHA256 = {
    'README.md': 'a50dd2714968df64a1ad3f62b0c8610f65880dc63473f0c6a811ec1202ba5f70',
    'edge_environment.py': 'a312dfa69abdf7eb181f735e99c720b8cad88c22638586054ecf5323632b8d6e',
    'fdedge_main.py': '40597c0c94d1a5679c6b94b5674a929207b34997777558305ddb56a35dac0b55',
    'fdsac_model.py': 'e47f4ed71134462ff3c91f475abdddbfed856e1a833a455696b0434214493d8d',
    'feedback_diffusion.py': '50626d6a1f3a52a4b9193b95c5655ee364dda220b9ae192811620f7569038e3f',
    'helpers.py': '408716182e58ee3dde07714967a9f5977ec02b691eafd72fce8e6ccab9c585f1',
    'baselines/DQN/dqn_environment.py': '69cc6d97310a8727220892b5f7fbc17531680c934849371c650656b93de6d6dd',
    'baselines/DQN/dqn_main.py': '8a7224090c57f5751085e4b1ca6965f36b830f04a8cc462489d17967ed971163',
    'baselines/DQN/dqn_model.py': '6c8655f8736f48ea226dbebd7580cc0d58177c73a365c8755b87ba9e942e96fc',
    'baselines/Opt/opt_environment.py': '3b44a51d16c2b7a47007a98178a2ebf9f505cf32ae5a5ae930eaf60a072b3a6c',
    'baselines/Opt/opt_main.py': 'cef8ab0f355113ee0d19b01d7dd73ef4a239a63c2f5247b4b02faa953cd55544',
    'baselines/SAC/sac_environment.py': 'adff96b4b1709ed843bf96dd843f7ef36c23344a03cafd85ea6ebe10cb3a0308',
    'baselines/SAC/sac_main.py': '469b4d63fdc067e644df2b65d6b75325ebe243b84e31c631b93ed8bacab0e96e',
    'baselines/SAC/sac_model.py': '774c27e2f41e3a2d60db40fb96d60ddcae6c258415186aa2f3e5c4e8f33b439d',
}


def run_episode(env, tasks, actions):
    """Drive one episode with fixed actions -> (observations, next states, delays)."""
    env.reset_env(tasks)
    observations, next_states, delays = [], [], []
    for t in range(config.TIME_SLOTS - 1):
        for n in range(len(tasks[t])):
            observations.append(env.observe(t, n))
            next_state, _, delay, _ = env.transition(t, n, next(actions))
            next_states.append(next_state)
            delays.append(delay)
        env.update_proc_queues(t)
    return observations, next_states, delays


class EnvironmentTest(unittest.TestCase):
    def test_upstream_copy_is_unmodified(self):
        for name, digest in UPSTREAM_SHA256.items():
            self.assertEqual(hashlib.sha256((config.UPSTREAM / name).read_bytes()).hexdigest(), digest, name)

    def test_public_capacities(self):
        self.assertEqual(public_capacities().tolist(), [50, 25, 18, 32, 28, 21, 50, 17, 44, 41])

    def test_training_tasks_match_the_fdedge_episode_loop(self):
        make_environment('fdedge', 'start', 60, public_capacities())  # seeds NumPy's global generator like FDEdge
        expected = [np.random.uniform(10, 40, size=[np.random.randint(1, 61)]) for _ in range(config.TIME_SLOTS)]
        actual = TrainingTasks(60).episode()
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(expected, actual)))

    def test_state_options(self):
        tasks = evaluation_tasks(config.EVAL_SEEDS[0], 40)
        count = sum(len(tasks[t]) for t in range(config.TIME_SLOTS - 1))
        for actor in ACTORS:
            runs = {}
            for state in STATES:
                env = make_environment(actor, state, 40, public_capacities().astype(float))
                actions = iter(np.random.default_rng(3).integers(0, config.NUM_SERVERS, size=count).tolist())
                runs[state] = (env, *run_episode(env, tasks, actions))
            # The state option never changes delays or queues.
            self.assertEqual(runs['slot'][3], runs['start'][3], actor)

            env, obs_slot = runs['slot'][0], runs['slot'][1]
            obs_start = runs['start'][1]
            i = 0
            for t in range(config.TIME_SLOTS - 1):
                for n in range(len(tasks[t])):
                    if n == 0:  # nothing assigned yet in this slot
                        self.assertTrue(np.array_equal(obs_slot[i], obs_start[i]))
                    else:
                        self.assertTrue((obs_slot[i][2:] >= obs_start[i][2:]).all())
                    i += 1
            self.assertGreater(sum((s[2:] > o[2:]).any() for s, o in zip(obs_slot, obs_start)), count // 2)

            # After the last task of a slot, the slot-aware next state holds the queues the next slot starts with.
            last = len(tasks[0]) - 1
            self.assertTrue(np.array_equal(runs['start'][2][last][2:], np.zeros(config.NUM_SERVERS)))
            self.assertTrue(np.array_equal(runs['slot'][2][last][2:], env.proc_queue_len[1]))

if __name__ == '__main__':
    unittest.main()
