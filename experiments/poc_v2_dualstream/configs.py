"""تنظیمات مرکزی پروژه NeuroBottleneck v2.0 — Dual-Stream"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EnvConfig:
    topology: str = "abilene"
    c_min: float = 300.0
    c_max: float = 5000.0
    c_init: float = 1000.0
    delta_c: float = 200.0
    d_min: float = 100.0
    d_max: float = 1200.0
    d_init: float = 500.0
    sigma_d: float = 30.0
    alpha_revert: float = 0.1
    u_th: float = 0.8
    episode_len: int = 50
    lambda_threshold_mult: float = 1.5


@dataclass
class GHConfig:
    update_interval: int = 10
    feature_dim: int = 16
    cache_enabled: bool = True


@dataclass
class RewardConfig:
    alpha_res: float = 0.45
    beta_thr: float = 0.25
    gamma_vio: float = 0.15
    delta_cost: float = 0.05
    epsilon_sinr: float = 0.10
    zeta_soft: float = 0.50
    use_soft_penalty: bool = True
    use_hard_mask: bool = False


@dataclass
class ModelConfig:
    gnn_type: str = "graphsage"
    gnn_hidden: int = 64
    gnn_out: int = 64
    gh_out: int = 16
    drl_algo: str = "ppo"
    use_dual_stream: bool = True


@dataclass
class TrainConfig:
    total_timesteps: int = 5000
    seed: int = 42
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000


@dataclass
class ExperimentConfig:
    name: str = "Neuro_Full"
    use_gnn: bool = True
    use_gh_feature: bool = True
    use_gh_reward: bool = True
    use_soft_penalty: bool = True
    use_hard_mask: bool = False
    use_dual_stream: bool = True
    gnn_type: str = "graphsage"
    drl_algo: str = "ppo"


# ============================================================
# پیکربندی‌های آماده برای اهداف چهارگانه
# ============================================================

GOAL1_EXPERIMENTS = [
    ExperimentConfig(name="Baseline", use_gnn=False, use_gh_feature=False,
                     use_gh_reward=False, use_soft_penalty=False,
                     use_hard_mask=False, use_dual_stream=False),
    ExperimentConfig(name="GNN_only", use_gnn=True, use_gh_feature=False,
                     use_gh_reward=False, use_soft_penalty=False,
                     use_hard_mask=False, use_dual_stream=False),
    ExperimentConfig(name="Neuro_Full", use_gnn=True, use_gh_feature=True,
                     use_gh_reward=True, use_soft_penalty=True,
                     use_hard_mask=False, use_dual_stream=True),
]

GOAL2_DRL_ALGOS = ["ppo", "a2c", "dqn", "sac"]

GOAL3_GNN_TYPES = ["graphsage", "gat", "gin", "mlp"]

GOAL4_ABLATION = [
    ExperimentConfig(name="Baseline", use_gnn=False, use_gh_feature=False,
                     use_gh_reward=False, use_soft_penalty=False,
                     use_hard_mask=False, use_dual_stream=False),
    ExperimentConfig(name="GNN_only", use_gnn=True, use_gh_feature=False,
                     use_gh_reward=False, use_soft_penalty=False,
                     use_hard_mask=False, use_dual_stream=False),
    ExperimentConfig(name="GH_Feature_only", use_gnn=True, use_gh_feature=True,
                     use_gh_reward=False, use_soft_penalty=False,
                     use_hard_mask=False, use_dual_stream=True),
    ExperimentConfig(name="GH_Reward_only", use_gnn=True, use_gh_feature=False,
                     use_gh_reward=True, use_soft_penalty=False,
                     use_hard_mask=False, use_dual_stream=False),
    ExperimentConfig(name="Neuro_Full_Soft", use_gnn=True, use_gh_feature=True,
                     use_gh_reward=True, use_soft_penalty=True,
                     use_hard_mask=False, use_dual_stream=True),
    ExperimentConfig(name="Neuro_Full_Hard", use_gnn=True, use_gh_feature=True,
                     use_gh_reward=True, use_soft_penalty=False,
                     use_hard_mask=True, use_dual_stream=True),
]