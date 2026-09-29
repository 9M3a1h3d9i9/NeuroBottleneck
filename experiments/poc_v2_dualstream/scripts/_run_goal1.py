"""اجرای هدف ۱ (Dual-Stream vs Baseline)"""
import os
import sys

# افزودن ریشه پروژه به sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs import GOAL1_EXPERIMENTS, TrainConfig
from src.training.train import run_experiments
from src.evaluation.plot_utils import plot_all

# خواندن از environment variables
TOPOLOGY = os.environ.get("NB_TOPOLOGY", "abilene")
SEED = int(os.environ.get("NB_SEED", "42"))
TIMESTEPS = int(os.environ.get("NB_TIMESTEPS", "10000"))

print(f"[MAIN] Topology={TOPOLOGY}, Seed={SEED}, Timesteps={TIMESTEPS}")

train_cfg = TrainConfig(total_timesteps=TIMESTEPS, seed=SEED)
output_dir = os.path.join(ROOT, "outputs", f"goal1_{TOPOLOGY}_seed{SEED}")
os.makedirs(output_dir, exist_ok=True)

df = run_experiments(
    GOAL1_EXPERIMENTS,
    train_cfg,
    topology=TOPOLOGY,
    seeds=[SEED],
    output_dir=output_dir,
)

plot_all(df, output_dir, GOAL1_EXPERIMENTS, TOPOLOGY, [SEED])
print(f"DONE. Results in {output_dir}")
