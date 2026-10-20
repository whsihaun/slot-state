"""Experiment settings and output locations.

The environment follows the public FDEdge code. Output folders default to the repository and can be moved with
environment variables:

    SLOT_STATE_CHECKPOINT_DIR   trained weights            (default: <repo>/checkpoints)
    SLOT_STATE_OUTPUT_DIR       training and evaluation logs (default: <repo>/outputs)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / 'third_party' / 'fdedge'

# Environment of the public FDEdge code: 10 edge servers, 1 s slots, 10-40 Mbit tasks.
NUM_SERVERS = 10
STATE_DIM = 2 + NUM_SERVERS  # task size, its workload, one queue per server
TIME_SLOTS = 100
BIT_RANGE = (10, 40)
CAPACITY_MAX = 50  # capacities are drawn from [10, 50] GHz with seed 2 -> [50, 25, 18, 32, 28, 21, 50, 17, 44, 41]

# Training: 100 episodes of 100 slots; the checkpoint keeps the episode with the lowest mean delay.
EPISODES = 100

# Frozen evaluation: 99 slots on five unseen task traces; metrics use the tasks of slots 50-98.
EVAL_SEEDS = (3101, 3102, 3103, 3104, 3105)
EVAL_SLOTS = 99
POST_START = 50


def checkpoint_dir():
    return Path(os.environ.get('SLOT_STATE_CHECKPOINT_DIR', REPO / 'checkpoints')).expanduser()


def output_dir():
    return Path(os.environ.get('SLOT_STATE_OUTPUT_DIR', REPO / 'outputs')).expanduser()


def config_id(n_max, state, seed):
    return f'n{n_max}_{state}_seed{seed}'


def checkpoint_path(actor, n_max, state, seed):
    return checkpoint_dir() / actor / config_id(n_max, state, seed) / 'best.pt'


def display_path(path):
    """Path relative to the repository when possible, so logs and summaries carry no user-specific prefix."""
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
