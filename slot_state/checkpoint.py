"""Checkpoint files: the evaluation policy, the FDEdge feedback bank and the environment it was trained in."""
import hashlib
import io
import os
from pathlib import Path

import numpy as np
import torch

from . import config
from .environment import ACTORS, STATES

FORMAT = 'slot-state/1'


def bank_shape(n_max):
    return (config.TIME_SLOTS, n_max, config.NUM_SERVERS)


def save(path, *, actor, state, n_max, seed, policy, average_delays, environment, hyperparameters,
         feedback_history=None):
    payload = dict(format=FORMAT, actor=actor, state=state, n_max=int(n_max), train_seed=int(seed),
                   episode=len(average_delays), average_delay=float(average_delays[-1]),
                   average_delays=[float(v) for v in average_delays], policy=policy.state_dict(),
                   environment=environment, hyperparameters=hyperparameters,
                   selection='episode with the lowest mean training delay')
    if actor == 'fdedge':
        bank = np.array(feedback_history, dtype=np.float64, copy=True)
        if bank.shape != bank_shape(n_max) or not np.isfinite(bank).all():
            raise ValueError('Invalid feedback bank')
        payload['feedback_history'] = torch.from_numpy(bank)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path):
    """-> (checkpoint dict, sha256 of the file). Loads tensors only (weights_only=True)."""
    data = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if checkpoint.get('format') != FORMAT:
        raise ValueError(f'{path}: not a {FORMAT} checkpoint')
    if checkpoint['actor'] not in ACTORS or checkpoint['state'] not in STATES:
        raise ValueError(f'{path}: unknown actor/state')
    spec = checkpoint['environment']
    for key, length in (('capacities', config.NUM_SERVERS), ('transmission_rates', config.NUM_SERVERS),
                        ('computation_density', checkpoint['n_max'])):
        values = np.asarray(spec[key], dtype=float)
        if values.shape != (length,) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f'{path}: invalid environment {key}')
    if checkpoint['actor'] == 'fdedge':
        bank = checkpoint['feedback_history']
        if tuple(bank.shape) != bank_shape(checkpoint['n_max']) or not torch.isfinite(bank).all():
            raise ValueError(f'{path}: invalid feedback bank')
    return checkpoint, hashlib.sha256(data).hexdigest()


def environment_spec(env, capacities, n_max):
    return dict(num_servers=config.NUM_SERVERS, max_tasks=int(n_max), bit_range=list(config.BIT_RANGE),
                time_slots=config.TIME_SLOTS, capacities=np.asarray(capacities).tolist(),
                transmission_rates=np.asarray(env.tran_rate_BSs).tolist(),
                computation_density=np.asarray(env.comp_density).tolist())
