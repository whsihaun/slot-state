"""Task-scheduling environment with two state options.

Delays, rewards and queue updates always come from each actor's public FDEdge environment. The state option changes
only what the scheduler observes: the state of the current task and the next state stored for training.

  start  Public code. Queues at the start of the slot; after the last task of a slot, the next state carries the
         next slot's queues before they are updated (zeros).
  slot   Slot-aware state. Queues plus the workload already assigned in this slot (q_{t-1} + q^bef); after the last
         task of a slot, the next state carries the queues the next slot starts with.

The slot option leaves the last executed slot (t = TIME_SLOTS - 2) as in the public code, whose bootstrap target refers
to a slot that is never run.
"""
import numpy as np

from . import config, upstream

ACTORS = ('fdedge', 'sac', 'dqn')
STATES = ('start', 'slot')


def public_environment(actor):
    if actor == 'fdedge':
        return upstream.fdedge('edge_environment').OffloadEnvironment
    return upstream.baseline(actor, 'environment').OffloadEnvironment


def make_environment(actor, state, n_max, capacities):
    if actor not in ACTORS or state not in STATES:
        raise ValueError(f'Unknown actor/state: {actor}/{state}')

    class Environment(public_environment(actor)):
        state_option = state

        def observe(self, t, n):
            queue = self.proc_queue_len[t].copy()
            if self.state_option == 'slot':
                queue = queue + self.proc_queue_bef[t]
            d = self.tasks_bit[t][n]
            return np.hstack([d, self.comp_density[n] * d, queue])

        def transition(self, t, n, action):
            """Public step for task n of slot t -> (next_state, reward, delay, next_feedback).

            next_feedback is the stored action probability of the next task (FDEdge only, otherwise None).
            """
            if actor == 'fdedge':
                public_next, next_feedback, reward, delay = self.step(t, n, action)
            else:
                (public_next, reward, delay), next_feedback = self.step_(t, n, action), None
            return self._next_state(t, n, public_next), reward, delay, next_feedback

        def _next_state(self, t, n, public_next):
            if self.state_option == 'start':
                return public_next
            if n < len(self.tasks_bit[t]) - 1:
                return self.observe(t, n + 1)
            if t < self.time_slots - 2:
                next_state = public_next.copy()
                next_state[2:] = np.maximum(self.proc_queue_len[t] + self.proc_queue_bef[t]
                                            - self.ES_capacities * self.duration, 0)
                return next_state
            return public_next

    return Environment(n_max, list(config.BIT_RANGE), config.NUM_SERVERS, config.TIME_SLOTS, capacities)


def public_capacities():
    """Server capacities of the public code: seed 2, randint(10, 51) GHz."""
    np.random.seed(2)
    return np.random.randint(10, config.CAPACITY_MAX + 1, size=[config.NUM_SERVERS])


class TrainingTasks:
    """Training task stream of the public FDEdge code for one N_max, identical for every actor and training seed.

    The FDEdge environment seeds NumPy's global generator with 1 and draws the computation densities and the initial
    feedback bank before its episode loop, which then draws the tasks. This stream repeats those draws on its own
    generator, so SAC and DQN, whose environments skip the bank, train on the same tasks as FDEdge.
    """

    def __init__(self, n_max):
        self.n_max = n_max
        self.rng = np.random.RandomState(1)
        self.rng.uniform(0.1024, 0.3072, size=[n_max])
        self.rng.normal(size=[config.TIME_SLOTS, n_max, config.NUM_SERVERS])

    def episode(self):
        tasks = []
        for _ in range(config.TIME_SLOTS):
            count = self.rng.randint(1, self.n_max + 1)
            tasks.append(self.rng.uniform(config.BIT_RANGE[0], config.BIT_RANGE[1], size=[count]))
        return tasks


def evaluation_tasks(seed, n_max):
    """One frozen-evaluation trace: 1 to N_max tasks of 10-40 Mbit in each of 100 slots."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(*config.BIT_RANGE, size=int(rng.integers(1, n_max + 1))) for _ in range(config.TIME_SLOTS)]
