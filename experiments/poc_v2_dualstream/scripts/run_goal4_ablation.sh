#!/bin/bash
set -e
cd "$(dirname "$0")/.."
TOPOLOGY=${1:-abilene}
TIMESTEPS=${2:-5000}
SEEDS=${3:-"42 123 456"}
python -c "
import sys; sys.path.insert(0, '.')
from configs import GOAL4_ABLATION, TrainConfig
from src.training.train import run_experiments
from src.evaluation.plot_utils import plot_all
import os
seeds = [int(s) for s in '$SEEDS'.split()]
train_cfg = TrainConfig(total_timesteps=$TIMESTEPS)
output_dir = f'./outputs/goal4_{'$TOPOLOGY'}'
os.makedirs(output_dir, exist_ok=True)
df = run_experiments(GOAL4_ABLATION, train_cfg, topology='$TOPOLOGY',
                     seeds=seeds, output_dir=output_dir)
plot_all(df, output_dir, GOAL4_ABLATION, ['$TOPOLOGY'], seeds)
print('DONE. Results in', output_dir)
"