#!/bin/bash

# Assign CLI inputs to variables
SCENE_NAME=$1
EXP_NAME=$2

# Run the Python script with the provided inputs
CUDA_LAUNCH_BLOCKING=1 python gui.py \
    -s "/data/Condense_v2/scenes/$SCENE_NAME/" \
    --expname "$EXP_NAME" \
    --configs arguments/condense/bench.py \
    --test_iterations 1000
