#!/bin/bash
set -e
cd "$(dirname "$0")/.."
TOPOLOGY=${1:-abilene}
SEED=${2:-42}
TIMESTEPS=${3:-5000}
python -c "
import sys; sys.path.insert(0, '.')
from configs import ExperimentConfig, TrainConfig, GOAL3_GNN_TYPES
from src.training.train import run_experiments
from src.evaluation.plot_utils import plot_all
import os
experiments = []
for gnn in GOAL3_GNN_TYPES:
    exp = ExperimentConfig(
        name=f'GNN_{gnn.upper()}',
        use_gnn=True, use_gh_feature=True, use_gh_reward=True,
        use_soft_penalty=True, use_dual_stream=True,
    )
    exp.gnn_type = gnn
    experiments.append(exp)
train_cfg = TrainConfig(total_timesteps=$TIMESTEPS, seed=$SEED)
output_dir = f'./outputs/goal3_{'$TOPOLOGY'}_seed{'$SEED'}'
os.makedirs(output_dir, exist_ok=True)
df = run_experiments(experiments, train_cfg, topology='$TOPOLOGY',
                     seeds=[$SEED], output_dir=output_dir)
plot_all(df, output_dir, experiments, ['$TOPOLOGY'], [$SEED])
print('DONE. Results in', output_dir)
"