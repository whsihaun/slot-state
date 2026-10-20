"""Train one actor with one state option on one GPU.

    python -m slot_state.train --actor fdedge --state slot --n-max 100 --seed 1

Select the GPU with CUDA_VISIBLE_DEVICES. The checkpoint goes to $SLOT_STATE_CHECKPOINT_DIR/<actor>/<config>/best.pt
and the per-episode delays to $SLOT_STATE_OUTPUT_DIR/train/<actor>/<config>/. A new run replaces that best.pt.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import random
import time

import numpy as np
import torch

from . import agents, checkpoint, config
from .environment import ACTORS, STATES, TrainingTasks, make_environment, public_capacities


def train(actor, state, n_max, seed, episodes, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    capacities = public_capacities()
    env = make_environment(actor, state, n_max, capacities)
    agent, buffer = agents.make_agent(actor, device)
    tasks = TrainingTasks(n_max)
    spec = checkpoint.environment_spec(env, capacities, n_max)
    warmup, batch_size = agents.WARMUP[actor], agents.HYPERPARAMETERS[actor]['batch_size']
    feedback = actor == 'fdedge'

    cid = config.config_id(n_max, state, seed)
    best_path = config.checkpoint_path(actor, n_max, state, seed)
    out = config.output_dir() / 'train' / actor / cid
    out.mkdir(parents=True, exist_ok=True)
    run = dict(actor=actor, state=state, n_max=n_max, train_seed=seed, episodes=episodes, status='running',
               checkpoint=config.display_path(best_path), started_utc=datetime.now(timezone.utc).isoformat(),
               torch=torch.__version__, numpy=np.__version__)
    (out / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    print(f'Training {actor}/{cid}; checkpoint {config.display_path(best_path)}', flush=True)

    delays_per_episode, best, started = [], math.inf, time.monotonic()
    try:
        for episode in range(1, episodes + 1):
            env.reset_env(tasks.episode())
            delays = []
            for t in range(env.time_slots - 1):
                for n in range(len(env.tasks_bit[t])):
                    observation = env.observe(t, n)
                    if feedback:
                        stored = env.latent_action_prob_space[t][n].copy()
                        action, probs = agent.take_action(observation, stored)
                        env.latent_action_prob_space[t][n] = probs
                        next_state, reward, delay, next_stored = env.transition(t, n, action)
                        buffer.add(observation, action, stored, reward, next_state, next_stored)
                    else:
                        action = agent.take_action(observation)
                        next_state, reward, delay, _ = env.transition(t, n, action)
                        buffer.add(observation, action, reward, next_state)
                    delays.append(delay)
                env.update_proc_queues(t)
                if buffer.size() > warmup:
                    batch = buffer.sample(batch_size)
                    if feedback:
                        keys = ('states', 'actions', 'latent_action_probs', 'rewards', 'next_states',
                                'next_latent_action_probs')
                    else:
                        keys = ('states', 'actions', 'rewards', 'next_states')
                    agent.update(dict(zip(keys, batch)))
            mean = float(np.mean(delays))
            if not math.isfinite(mean):
                raise RuntimeError(f'Non-finite mean delay in episode {episode}')
            delays_per_episode.append(mean)
            with (out / 'train_delays.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['episode', 'average_delay'])
                writer.writerows(enumerate(delays_per_episode, start=1))
            print(f'episode {episode}: average delay {mean:.4f} s', flush=True)
            if mean < best:
                best = mean
                checkpoint.save(best_path, actor=actor, state=state, n_max=n_max, seed=seed,
                                policy=getattr(agent, agents.POLICY_ATTRIBUTE[actor]),
                                average_delays=delays_per_episode, environment=spec,
                                hyperparameters=agents.HYPERPARAMETERS[actor],
                                feedback_history=env.latent_action_prob_space if feedback else None)
        run.update(status='complete', best_episode=int(np.argmin(delays_per_episode)) + 1, best_delay=best)
    except BaseException as exc:
        run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        run['elapsed_seconds'] = time.monotonic() - started
        (out / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    print(f'Done: best episode {run["best_episode"]} ({best:.4f} s) -> {config.display_path(best_path)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--actor', choices=ACTORS, required=True)
    parser.add_argument('--state', choices=STATES, required=True)
    parser.add_argument('--n-max', type=int, default=100, help='largest number of tasks in a slot (default 100)')
    parser.add_argument('--seed', type=int, default=1, help='training seed (default 1)')
    parser.add_argument('--episodes', type=int, default=config.EPISODES)
    args = parser.parse_args()
    if args.n_max < 1 or args.episodes < 1 or not 0 <= args.seed < 2 ** 31:
        parser.error('n-max and episodes must be positive and the seed in [0, 2**31)')
    if not torch.cuda.is_available():
        raise SystemExit('Training needs a CUDA GPU; choose one with CUDA_VISIBLE_DEVICES.')
    torch.set_num_threads(1)
    train(args.actor, args.state, args.n_max, args.seed, args.episodes, torch.device('cuda:0'))


if __name__ == '__main__':
    main()
