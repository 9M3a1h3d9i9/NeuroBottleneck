import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs import ExperimentConfig, TrainConfig, GOAL3_GNN_TYPES
from src.training.train import run_experiments
from src.evaluation.plot_utils import plot_all

TOPOLOGY = os.environ.get("NB_TOPOLOGY", "abilene")
SEED = int(os.environ.get("NB_SEED", "42"))
TIMESTEPS = int(os.environ.get("NB_TIMESTEPS", "10000"))

experiments = []
for gnn in GOAL3_GNN_TYPES:
    exp = ExperimentConfig(
        name=f"GNN_{gnn.upper()}",
        use_gnn=True, use_gh_feature=True, use_gh_reward=True,
        use_soft_penalty=True, use_hard_mask=False,
    )
    exp.gnn_type = gnn
    experiments.append(exp)

train_cfg = TrainConfig(total_timesteps=TIMESTEPS, seed=SEED)
output_dir = os.path.join(ROOT, "outputs", f"goal3_{TOPOLOGY}_seed{SEED}")
os.makedirs(output_dir, exist_ok=True)
df = run_experiments(experiments, train_cfg, topology=TOPOLOGY,
                     seeds=[SEED], output_dir=output_dir)
plot_all(df, output_dir, experiments, TOPOLOGY, [SEED])
print(f"DONE. Results in {output_dir}")
