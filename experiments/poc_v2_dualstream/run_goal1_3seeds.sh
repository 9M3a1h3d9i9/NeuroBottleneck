#!/bin/bash
cd ~/NeuroBottleneck/experiments/poc_v2_dualstream

for seed in 42 123 456; do
    echo "=========================================="
    echo "Starting Goal 1 - Seed $seed - $(date)"
    echo "=========================================="
    
    NB_TOPOLOGY=abilene \
    NB_SEED=$seed \
    NB_TIMESTEPS=50000 \
    NB_VERSION=v4_fixed_reward \
    bash scripts/run_goal1_dualstream.sh abilene $seed 50000 v4_fixed_reward
    
    echo "Seed $seed finished at $(date)"
done
echo "ALL SEEDS DONE!"
