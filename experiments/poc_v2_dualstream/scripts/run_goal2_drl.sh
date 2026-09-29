#!/bin/bash
set -e
cd "$(dirname "$0")/.."
TOPOLOGY=${1:-abilene}
SEED=${2:-42}
TIMESTEPS=${3:-5000}
python -c "
import sys; sys.path.insert(0, '.')
from configs import ExperimentConfig, TrainConfig, GOAL2_DRL_ALGOS
from src.training.train import run_experiments
from src.evaluation.plot_utils import plot_all
import os
experiments = []
for algo in GOAL2_DRL_ALGOS:
    exp = ExperimentConfig(
        name=f'DRL_{algo.upper()}',
        use_gnn=True, use_gh_feature=True, use_gh_reward=True,
        use_soft_penalty=True, use_dual_stream=True,
    )
    exp.drl_algo = algo
    experiments.append(exp)
train_cfg = TrainConfig(total_timesteps=$TIMESTEPS, seed=$SEED)
output_dir = f'./outputs/goal2_{'$TOPOLOGY'}_seed{'$SEED'}'
os.makedirs(output_dir, exist_ok=True)
df = run_experiments(experiments, train_cfg, topology='$TOPOLOGY',
                     seeds=[$SEED], output_dir=output_dir)
plot_all(df, output_dir, experiments, ['$TOPOLOGY'], [$SEED])
print('DONE. Results in', output_dir)
"
