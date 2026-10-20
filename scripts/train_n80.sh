#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python -m slot_state.train --actor fdedge --state start --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.evaluate --actor fdedge --state start --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.train --actor fdedge --state slot --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.evaluate --actor fdedge --state slot --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.train --actor sac --state start --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.evaluate --actor sac --state start --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.train --actor sac --state slot --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.evaluate --actor sac --state slot --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.train --actor dqn --state start --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.evaluate --actor dqn --state start --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.train --actor dqn --state slot --n-max 80 --seed 1
CUDA_VISIBLE_DEVICES=0 python -m slot_state.evaluate --actor dqn --state slot --n-max 80 --seed 1
python -m slot_state.summarize
