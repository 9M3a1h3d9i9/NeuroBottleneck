import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs import ExperimentConfig, TrainConfig, GOAL2_DRL_ALGOS
from src.training.train import run_experiments
from src.evaluation.plot_utils import plot_all

TOPOLOGY = os.environ.get("NB_TOPOLOGY", "abilene")
SEED = int(os.environ.get("NB_SEED", "42"))
TIMESTEPS = int(os.environ.get("NB_TIMESTEPS", "10000"))

experiments = []
for algo in GOAL2_DRL_ALGOS:
    exp = ExperimentConfig(
        name=f"DRL_{algo.upper()}",
        use_gnn=True, use_gh_feature=True, use_gh_reward=True,
        use_soft_penalty=True, use_hard_mask=False,
    )
    exp.drl_algo = algo
    experiments.append(exp)

train_cfg = TrainConfig(total_timesteps=TIMESTEPS, seed=SEED)
output_dir = os.path.join(ROOT, "outputs", f"goal2_{TOPOLOGY}_seed{SEED}")
os.makedirs(output_dir, exist_ok=True)
df = run_experiments(experiments, train_cfg, topology=TOPOLOGY,
                     seeds=[SEED], output_dir=output_dir)
plot_all(df, output_dir, experiments, TOPOLOGY, [SEED])
print(f"DONE. Results in {output_dir}")
