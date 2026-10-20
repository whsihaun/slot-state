<div align="center">

# Impact of State Design on Actor Comparisons in DRL-Based Task Scheduling for Collaborative Edge Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](requirements.txt)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-v1.0-2ea44f)](https://github.com/whsihaun/slot-state/releases/tag/v1.0)

</div>

This repository contains the code of the paper above. A link to the paper will be added after publication.

DRL schedulers for collaborative edge computing assign the tasks of each time slot to edge servers one at a time.
A task's delay includes waiting behind the work assigned to the same server earlier in that slot, yet the state of
[FDEdge](https://github.com/ChangfuXu/FDEdge) (IEEE TMC 2025) and of the SAC and DQN baselines released with it holds
only the queues at the start of the slot. This repository trains these three actors with that original state and with
a **slot-aware state** that adds the work already assigned in the slot.

## Results

Mean task delay in seconds over slots 50–98 of five unseen task traces, averaged over training seeds 1–3, from GPU
evaluation of the released checkpoints.

| N_max (average load) | 60 (0.45) | 80 (0.60) | 100 (0.76) | 120 (0.92) |
|---|---:|---:|---:|---:|
| DQN, original state | 1.983 | 3.094 | 9.691 | 7.860 |
| SAC, original state | 2.483 | 3.392 | 9.518 | 18.669 |
| FDEdge, original state | 0.534 | 0.701 | 1.167 | 1.987 |
| DQN, slot-aware state | 0.435 | 0.551 | 0.943 | 1.752 |
| SAC, slot-aware state | 0.438 | 0.583 | 2.016 | 5.463 |
| FDEdge, slot-aware state | 0.455 | 0.564 | 1.041 | 1.724 |
| Best of DQN/SAC ÷ FDEdge, original state | 3.71 | 4.41 | 8.16 | 3.96 |
| Best of DQN/SAC ÷ FDEdge, slot-aware state | 0.96 | 0.98 | 0.91 | 1.02 |

The slot-aware state lowers the delay of every actor at every load and training seed. With it, the 3.7–8.2× lead of
FDEdge over the better of DQN and SAC shrinks to within 2% or reverses.

## How the state changes

Following FDEdge, the delay of the n-th task of slot t on server b is

$$D_{t,n,b} = \frac{d_n}{v_b} + \frac{\rho_n d_n}{f_b} + \frac{q_{t-1,b} + q^{\mathrm{bef}}_{t,n,b}}{f_b}$$

where $d_n$ is the task size, $\rho_n$ its computation density, $v_b$ and $f_b$ the transmission rate and capacity of
server $b$, $q_{t-1,b}$ the work left from earlier slots and $q^{\mathrm{bef}}_{t,n,b}$ the work assigned to $b$
earlier in slot $t$.

| `--state` | Queues in the state | Next state after the last task of a slot |
|---|---|---|
| `start` (original, public code) | $q_{t-1}$ | queues not yet updated (zeros) |
| `slot` (slot-aware) | $q_{t-1} + q^{\mathrm{bef}}_{t,n}$ | queues at the start of the next slot |

The slot-aware state keeps the size of the original state and uses only the scheduler's own assignments. Delays,
rewards and queue updates are unchanged; see `observe` and `_next_state` in
[slot_state/environment.py](slot_state/environment.py).

## Repository layout

```text
slot_state/
  environment.py        state options on the public environments; training and evaluation tasks
  agents.py             FDEdge with the feedback input on; the public SAC and DQN baselines
  train.py              python -m slot_state.train
  evaluate.py           python -m slot_state.evaluate
  summarize.py          python -m slot_state.summarize
  checkpoint.py, config.py, upstream.py
scripts/                shell scripts that run the commands below
third_party/fdedge/     unmodified FDEdge code (see NOTICE)
tests/
```

## Setup

```bash
git clone https://github.com/whsihaun/slot-state.git
cd slot-state
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest      # CPU only, a few seconds
```

Run all commands from the repository root. The code was tested with Python 3.13 and PyTorch 2.11 (CUDA 12.8) on
RTX A6000 GPUs. Training needs a CUDA GPU; evaluation also runs on the CPU.

## Evaluate the released checkpoints

```bash
bash scripts/download_checkpoints.sh    # 72 checkpoints (seeds 1-3), 16 MB, into checkpoints/
bash scripts/evaluate_checkpoints.sh    # evaluates them on GPU 0 (about 30 minutes) and prints the table
```

To evaluate one checkpoint:

```bash
python -m slot_state.evaluate --actor fdedge --state slot --n-max 100 --seed 1    # add --device cpu without a GPU
```

On the CPU, DQN and SAC match the GPU values to four decimal places, but FDEdge does not, because its reverse diffusion
draws noise from the device's random number generator; its CPU means differ from the GPU means above by up to 5%.

## Train

Each script trains and evaluates the three actors with both states at one load, using training seed 1 on GPU 0, and
then prints the table. Change `CUDA_VISIBLE_DEVICES` in a script to use another GPU, or `--seed` to train seeds 2 and 3.

| Script | N_max | Time on one GPU |
|---|---:|---|
| `scripts/train_n60.sh` | 60 | about 2 hours |
| `scripts/train_n80.sh` | 80 | about 2.5 hours |
| `scripts/train_n100.sh` | 100 | about 2 hours |
| `scripts/train_n120.sh` | 120 | about 3.5 hours |

A run trains for 100 episodes of 100 slots and keeps the episode with the lowest mean delay as
`checkpoints/<actor>/n<N_max>_<state>_seed<seed>/best.pt`, replacing a downloaded checkpoint of the same name. At a
given N_max, every actor and seed trains on the same task sequence, the one the public FDEdge code generates, so a seed
changes only the initial weights and the replay sampling. GPU kernels are not guaranteed to be deterministic, so
retrained delays may differ slightly from the released checkpoints.

## Configuration

| Environment variable | Default | Purpose |
|---|---|---|
| `SLOT_STATE_CHECKPOINT_DIR` | `checkpoints/` | where training writes and evaluation reads `best.pt` |
| `SLOT_STATE_OUTPUT_DIR` | `outputs/` | training curves, evaluation results and the summary table |

The environment, the evaluation traces and the training budget are fixed in [slot_state/config.py](slot_state/config.py).

## Relation to FDEdge

- [third_party/fdedge](third_party/fdedge) is the public FDEdge code at commit `5519888`, unchanged; the tests check its
  file hashes.
- The released FDEdge code builds its feedback input, the action probabilities stored for the same slot and position
  in the previous episode, but starts the reverse diffusion from noise instead. We switch this input on and keep the
  rest of the public code, including its environment, hyperparameters, loss, update schedule and its storage of
  probabilities after the softmax. This model follows the public code rather than reproducing every detail of the
  paper; we use it as the FDEdge baseline because its mean delay with the original state, 1.167 s at N_max = 100, lies
  within the 1.12 ± 0.14 s reported in the FDEdge paper.
- SAC and DQN are the released baselines with their published hyperparameters, trained on the same task sequence as
  FDEdge.

## License

The code of this repository is released under the [MIT License](LICENSE). The FDEdge code in `third_party/fdedge` has
no stated license and remains with its authors; see [NOTICE](NOTICE).
