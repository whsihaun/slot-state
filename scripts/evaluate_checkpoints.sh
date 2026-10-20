#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python -m slot_state.evaluate --all
python -m slot_state.summarize
