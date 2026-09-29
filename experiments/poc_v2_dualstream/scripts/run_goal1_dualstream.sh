#!/bin/bash
# هدف ۱: اثبات نوآوری Dual-Stream در برابر Baseline

set -e
cd "$(dirname "$0")/.."

TOPOLOGY=${1:-abilene}
SEED=${2:-42}
TIMESTEPS=${3:-5000}

python -c "
import sys
sys.path.insert(0, '.')
from configs import GOAL1_EXPERIMENTS, TrainConfig
from src.training.train import run_experiments
from src.evaluation.plot_utils import plot_all
import os

train_cfg = TrainConfig(total_timesteps=$TIMESTEPS, seed=$SEED)
output_dir = f'./outputs/goal1_{'$TOPOLOGY'}_seed{'$SEED'}'
os.makedirs(output_dir, exist_ok=True)

df = run_experiments(
    GOAL1_EXPERIMENTS,
    train_cfg,
    topology='$TOPOLOGY',
    seeds=[$SEED],
    output_dir=output_dir,
)

plot_all(df, output_dir, GOAL1_EXPERIMENTS, ['$TOPOLOGY'], [$SEED])
print('DONE. Results in', output_dir)
"