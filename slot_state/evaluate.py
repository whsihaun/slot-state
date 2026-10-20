"""Frozen evaluation of trained checkpoints on the five evaluation traces.

    python -m slot_state.evaluate --actor fdedge --state slot --n-max 100 --seed 1
    python -m slot_state.evaluate --all

Writes summary.csv (one row per trace) and evaluation.json to $SLOT_STATE_OUTPUT_DIR/eval/<actor>/<config>/.
Metrics use the tasks of slots 50-98. FDEdge samples noise in its reverse process, so its delays depend on the
device's random number generator; the README numbers come from GPU evaluation, and CPU means differ from them by up to
5% (DQN and SAC do not).
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import agents, checkpoint, config
from .environment import ACTORS, STATES, evaluation_tasks, make_environment


def load_policy(path, device):
    """-> (policy network in eval mode, checkpoint dict, file sha256)."""
    ckpt, digest = checkpoint.load(path)
    policy = agents.policy_network(ckpt['actor'], device)
    policy.load_state_dict(ckpt['policy'], strict=True)
    policy.eval().requires_grad_(False)
    return policy, ckpt, digest


def rollout(policy, ckpt, seed, device):
    """Run 99 slots of one evaluation trace with argmax actions -> (summary, mean delay of each slot)."""
    actor, n_max, spec = ckpt['actor'], ckpt['n_max'], ckpt['environment']
    tasks = evaluation_tasks(seed, n_max)
    capacity = np.asarray(spec['capacities'], dtype=float)
    env = make_environment(actor, ckpt['state'], n_max, capacity.copy())
    env.tran_rate_BSs = np.asarray(spec['transmission_rates'], dtype=float)
    env.comp_density = np.asarray(spec['computation_density'], dtype=float)
    env.ES_capacities = capacity.copy()
    env.reset_env(tasks)
    feedback = actor == 'fdedge'
    if feedback:
        env.latent_action_prob_space = ckpt['feedback_history'].numpy().astype(np.float64, copy=True)
    torch.manual_seed(seed + 100000)  # FDEdge reverse-process noise; the other actors draw no random numbers

    post, slot_means = [], []
    for t in range(config.EVAL_SLOTS):
        slot_delays = []
        for n in range(len(tasks[t])):
            observation = torch.as_tensor(env.observe(t, n), dtype=torch.float32, device=device)[None]
            if feedback:
                stored = torch.as_tensor(env.latent_action_prob_space[t, n], dtype=torch.float32, device=device)
                output = policy(observation, stored[None])
            else:
                output = policy(observation)
            if not torch.isfinite(output).all():
                raise RuntimeError('Non-finite policy output')
            action = int(output.argmax(dim=-1).item())
            if feedback:
                env.latent_action_prob_space[t, n] = output[0].cpu().numpy()
            delay = float(env.transition(t, n, action)[2])
            if not math.isfinite(delay):
                raise RuntimeError('Non-finite delay')
            slot_delays.append(delay)
        env.update_proc_queues(t)
        slot_means.append(float(np.mean(slot_delays)))
        if t >= config.POST_START:
            post.extend(slot_delays)
    summary = dict(seed=seed, post_mean=float(np.mean(post)), post_p99=float(np.percentile(post, 99)))
    return summary, slot_means


def evaluate(path, device):
    policy, ckpt, digest = load_policy(path, device)
    cid = config.config_id(ckpt['n_max'], ckpt['state'], ckpt['train_seed'])
    out = config.output_dir() / 'eval' / ckpt['actor'] / cid
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        summaries = [rollout(policy, ckpt, seed, device)[0] for seed in config.EVAL_SEEDS]
    with (out / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    meta = dict(actor=ckpt['actor'], state=ckpt['state'], n_max=ckpt['n_max'], train_seed=ckpt['train_seed'],
                checkpoint=config.display_path(path), checkpoint_sha256=digest, checkpoint_episode=ckpt['episode'],
                device=device.type, evaluation_seeds=list(config.EVAL_SEEDS),
                post_mean=float(np.mean([s['post_mean'] for s in summaries])),
                post_p99=float(np.mean([s['post_p99'] for s in summaries])))
    (out / 'evaluation.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'{ckpt["actor"]}/{cid}: mean delay {meta["post_mean"]:.4f} s, P99 {meta["post_p99"]:.4f} s', flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--actor', choices=ACTORS)
    parser.add_argument('--state', choices=STATES)
    parser.add_argument('--n-max', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1, help='training seed of the checkpoint')
    parser.add_argument('--checkpoint', type=Path, help='checkpoint file (default: from actor, state, N_max, seed)')
    parser.add_argument('--all', action='store_true', help='evaluate every checkpoint in $SLOT_STATE_CHECKPOINT_DIR')
    parser.add_argument('--device', default='cuda', help="'cuda' (default, as in the paper) or 'cpu'")
    args = parser.parse_args()
    if args.all:
        paths = sorted(config.checkpoint_dir().glob('*/*/best.pt'))
        if not paths:
            parser.error(f'no checkpoints under {config.display_path(config.checkpoint_dir())}')
    elif args.checkpoint is not None:
        paths = [args.checkpoint]
    elif args.actor is not None and args.state is not None:
        paths = [config.checkpoint_path(args.actor, args.n_max, args.state, args.seed)]
    else:
        parser.error('give --actor and --state, --checkpoint, or --all')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU found; pass --device cpu to evaluate on the CPU.')
    torch.set_num_threads(1)
    for path in paths:
        evaluate(path, device)


if __name__ == '__main__':
    main()
