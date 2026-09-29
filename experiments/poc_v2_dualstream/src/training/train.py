"""حلقه آموزش با SB3 — Dual-Stream"""

import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, A2C, DQN, SAC
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from ..envs.network_env import NetworkEnv
from ..models.dual_stream_extractor import DualStreamExtractor
from configs import (EnvConfig, GHConfig, RewardConfig, ModelConfig,
                     TrainConfig, ExperimentConfig)


def make_env(env_cfg, gh_cfg, reward_cfg, topology, seed):
    return NetworkEnv(env_cfg, gh_cfg, reward_cfg,
                      topology_name=topology, seed=seed)


def make_extractor_class(exp_cfg, env):
    N = env.N
    M = env.M
    node_dim = env.node_dim
    edge_dim = env.edge_dim

    class Extractor(DualStreamExtractor):
        def __init__(self, observation_space, features_dim=128):
            super().__init__(
                observation_space,
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_nodes=N,
                num_edges=M,
                gnn_type=exp_cfg.get('gnn_type', 'graphsage'),
                gnn_hidden=64,
                gnn_out=64,
                gh_out=16,
                use_gnn=exp_cfg['use_gnn'],
                use_gh_feature=exp_cfg['use_gh_feature'],
                features_dim=features_dim,
            )
    return Extractor


class MetricCallback(BaseCallback):
    def __init__(self, log_interval=10):
        super().__init__()
        self.log_interval = log_interval
        self.rewards = []
        self.violations = []
        self.lambda_mins = []
        self.step_counter = 0

    def _on_step(self):
        self.step_counter += 1
        if len(self.locals.get('rewards', [])) > 0:
            self.rewards.append(float(self.locals['rewards'][0]))
        infos = self.locals.get('infos', [])
        if infos and 'violations' in infos[0]:
            self.violations.append(infos[0]['violations'])
            self.lambda_mins.append(infos[0].get('lambda_min', 0.0))
        return True


def train_single(exp_cfg, env_cfg, gh_cfg, reward_cfg, train_cfg,
                 topology, seed, output_dir):
    name = exp_cfg.name
    run_dir = os.path.join(output_dir, f"{name}_{topology}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

    env = make_env(env_cfg, gh_cfg, reward_cfg, topology, seed)

    extractor_cls = make_extractor_class({
        'use_gnn': exp_cfg.use_gnn,
        'use_gh_feature': exp_cfg.use_gh_feature,
        'gnn_type': exp_cfg.gnn_type,
    }, env)

    policy_kwargs = dict(
        features_extractor_class=extractor_cls,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    algo = exp_cfg.drl_algo

    if exp_cfg.use_hard_mask:
        def mask_fn(env):
            return env.get_action_mask()
        env = ActionMasker(env, mask_fn)
        model = MaskablePPO(
            MaskableActorCriticPolicy, env,
            learning_rate=train_cfg.learning_rate,
            n_steps=train_cfg.n_steps,
            batch_size=train_cfg.batch_size,
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            clip_range=train_cfg.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=seed,
            tensorboard_log=os.path.join(run_dir, "logs"),
        )
    elif algo == 'ppo':
        model = PPO(
            "MlpPolicy", env,
            learning_rate=train_cfg.learning_rate,
            n_steps=train_cfg.n_steps,
            batch_size=train_cfg.batch_size,
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            clip_range=train_cfg.clip_range,
            policy_kwargs=policy_kwargs,
            verbose=0, seed=seed,
            tensorboard_log=os.path.join(run_dir, "logs"),
        )
    elif algo == 'a2c':
        model = A2C("MlpPolicy", env, policy_kwargs=policy_kwargs,
                    verbose=0, seed=seed)
    elif algo == 'dqn':
        model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs,
                    verbose=0, seed=seed)
    elif algo == 'sac':
        model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs,
                    verbose=0, seed=seed)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    cb = MetricCallback()

    print(f"[TRAIN] {name} | {topology} | seed={seed} | steps={train_cfg.total_timesteps}")
    model.learn(total_timesteps=train_cfg.total_timesteps, callback=cb)

    model.save(os.path.join(run_dir, "models", "final_model"))

    metrics = pd.DataFrame({
        'step': np.arange(len(cb.rewards)),
        'reward': cb.rewards,
        'violations': cb.violations if len(cb.violations) == len(cb.rewards) else [None]*len(cb.rewards),
        'lambda_min': cb.lambda_mins if len(cb.lambda_mins) == len(cb.rewards) else [None]*len(cb.rewards),
    })
    metrics.to_csv(os.path.join(run_dir, "logs", "training_metrics.csv"), index=False)

    from ..evaluation.evaluate import evaluate_model
    eval_results = evaluate_model(model, env, n_episodes=5,
                                  use_mask=exp_cfg.use_hard_mask)

    with open(os.path.join(run_dir, "logs", "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    return run_dir, eval_results


def run_experiments(experiments, train_cfg, topology, seeds, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    env_cfg = EnvConfig(topology=topology)
    gh_cfg = GHConfig()

    for exp_cfg in experiments:
        for seed in seeds:
            rc = RewardConfig(
                use_soft_penalty=exp_cfg.use_soft_penalty,
                use_hard_mask=exp_cfg.use_hard_mask,
            )
            try:
                run_dir, eval_res = train_single(
                    exp_cfg, env_cfg, gh_cfg, rc, train_cfg,
                    topology, seed, output_dir
                )
                all_results.append({
                    'experiment': exp_cfg.name,
                    'topology': topology,
                    'seed': seed,
                    'mean_reward': eval_res['mean_reward'],
                    'std_reward': eval_res['std_reward'],
                    'mean_violations': eval_res['mean_violations'],
                    'mean_lambda_min': eval_res['mean_lambda_min'],
                    'run_dir': run_dir,
                })
            except Exception as e:
                print(f"[ERROR] {exp_cfg.name} seed={seed}: {e}")
                import traceback; traceback.print_exc()
                all_results.append({
                    'experiment': exp_cfg.name,
                    'topology': topology,
                    'seed': seed,
                    'mean_reward': np.nan,
                    'std_reward': np.nan,
                    'mean_violations': np.nan,
                    'mean_lambda_min': np.nan,
                    'run_dir': '',
                })

    df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n[SUMMARY] Saved to {summary_path}")
    print(df.to_string(index=False))
    return df