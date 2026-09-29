import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from configs import GOAL4_ABLATION, TrainConfig
from src.training.train import run_experiments
from src.evaluation.plot_utils import plot_all

TOPOLOGY = os.environ.get("NB_TOPOLOGY", "abilene")
TIMESTEPS = int(os.environ.get("NB_TIMESTEPS", "10000"))
SEEDS_STR = os.environ.get("NB_SEEDS", "42 123")
seeds = [int(s) for s in SEEDS_STR.split()]

train_cfg = TrainConfig(total_timesteps=TIMESTEPS)
output_dir = os.path.join(ROOT, "outputs", f"goal4_{TOPOLOGY}")
os.makedirs(output_dir, exist_ok=True)
df = run_experiments(GOAL4_ABLATION, train_cfg, topology=TOPOLOGY,
                     seeds=seeds, output_dir=output_dir)
plot_all(df, output_dir, GOAL4_ABLATION, TOPOLOGY, seeds)
print(f"DONE. Results in {output_dir}")
