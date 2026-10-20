"""Tables of the evaluated runs (reads $SLOT_STATE_OUTPUT_DIR/eval).

    python -m slot_state.summarize

Each cell averages a run's five evaluation traces and then the training seeds. The ratio to the original state is
taken per training seed and then averaged. Prints Markdown and writes summary.md.
"""
import csv
import json
from collections import defaultdict

import numpy as np

from . import config

NAMES = {'dqn': 'DQN', 'sac': 'SAC', 'fdedge': 'FDEdge'}
STATE_NAMES = {'start': 'original', 'slot': 'slot-aware'}


def load_runs():
    """-> {(actor, n_max, state): {train_seed: (mean delay, P99 delay)}}"""
    runs = defaultdict(dict)
    for path in sorted((config.output_dir() / 'eval').glob('*/*/evaluation.json')):
        meta = json.loads(path.read_text())
        with (path.parent / 'summary.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        runs[meta['actor'], meta['n_max'], meta['state']][meta['train_seed']] = (
            float(np.mean([float(r['post_mean']) for r in rows])), float(np.mean([float(r['post_p99']) for r in rows])))
    return runs


def tables(runs):
    mean = lambda cell, i=0: float(np.mean([v[i] for v in cell.values()]))
    lines = []
    for n_max in sorted({key[1] for key in runs}):
        lines += [f'## N_max = {n_max}', '',
                  '| Actor | State | Seeds | Mean delay (s) | P99 delay (s) | Slot-aware / original |',
                  '|---|---|---:|---:|---:|---|']
        for actor in NAMES:
            for state in STATE_NAMES:
                cell = runs.get((actor, n_max, state))
                if not cell:
                    continue
                ratio = '—'
                original = runs.get((actor, n_max, 'start'))
                if state == 'slot' and original:
                    seeds = sorted(set(cell) & set(original))
                    values = [cell[s][0] / original[s][0] for s in seeds]
                    ratio = f'{np.mean(values):.3f} ({sum(v < 1 for v in values)}/{len(values)} seeds lower)'
                lines.append(f'| {NAMES[actor]} | {STATE_NAMES[state]} | {len(cell)} | {mean(cell):.3f} | '
                             f'{mean(cell, 1):.3f} | {ratio} |')
        lines.append('')
        for state in STATE_NAMES:
            fdedge = runs.get(('fdedge', n_max, state))
            others = [runs[a, n_max, state] for a in ('dqn', 'sac') if (a, n_max, state) in runs]
            if fdedge and others:
                best = min(mean(cell) for cell in others)
                lines.append(f'Best of DQN/SAC over FDEdge, {STATE_NAMES[state]} state: {best / mean(fdedge):.3f}')
        lines.append('')
    return '\n'.join(lines)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit(f'No evaluations found under {config.display_path(config.output_dir() / "eval")}')
    text = tables(runs)
    print(text)
    (config.output_dir() / 'summary.md').write_text(text + '\n')


if __name__ == '__main__':
    main()
