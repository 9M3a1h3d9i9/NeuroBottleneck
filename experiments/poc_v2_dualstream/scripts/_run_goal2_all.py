import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs import ExperimentConfig, TrainConfig
from src.training.train import run_experiments
from src.evaluation.plot_utils import plot_all

TOPOLOGY = os.environ.get("NB_TOPOLOGY", "abilene")
SEED = int(os.environ.get("NB_SEED", "42"))
TIMESTEPS = int(os.environ.get("NB_TIMESTEPS", "50000"))

experiments = []
# for algo in ["ppo", "a2c", "dqn", "sac"]:
for algo in ["ppo", "a2c", "dqn"]:  # ← SAC حذف شده
    exp = ExperimentConfig(
        name=f"DRL_{algo.upper()}",
        use_gnn=True,
        use_gh_feature=False,   # ← GH در Observation نیست
        use_gh_reward=True,     # ← GH در Reward هست
        use_soft_penalty=True,
        use_hard_mask=False,
        use_gh_in_obs=False,    # ← کلید موفقیت v3
    )
    exp.drl_algo = algo
    experiments.append(exp)

train_cfg = TrainConfig(total_timesteps=TIMESTEPS, seed=SEED)
output_dir = os.path.join(ROOT, "outputs", f"goal2_{TOPOLOGY}_seed{SEED}_{TIMESTEPS//1000}K")
os.makedirs(output_dir, exist_ok=True)

df = run_experiments(experiments, train_cfg, topology=TOPOLOGY,
                     seeds=[SEED], output_dir=output_dir)
plot_all(df, output_dir, experiments, TOPOLOGY, [SEED])
print(f"DONE. Results in {output_dir}")
