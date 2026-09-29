"""اسکریپت اصلی آموزش PoC v1.1"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules.sndlib_loader import load_network
from modules.network_env import NetworkEnvV11
from modules.gnn_encoder import GNNFeaturesExtractor

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker


SEEDS = [42, 123]

EXPERIMENTS = [
    {'name': 'Baseline',      'use_gnn': False, 'use_mask': False},
    {'name': 'Telecom_Mask',  'use_gnn': False, 'use_mask': True},
    {'name': 'GNN_PPO',       'use_gnn': True,  'use_mask': False},
    {'name': 'Neuro_Full',    'use_gnn': True,  'use_mask': True},
]


def mask_fn(env):
    return env.get_action_mask()


# New "run_experiment"
def run_experiment(config, seed, graph, timesteps=30000):
    print(f"\n[RUN] {config['name']} | Seed {seed} | GNN: {config['use_gnn']} | Mask: {config['use_mask']}")

    env = NetworkEnvV11(graph, seed=seed)
    n_edges = env.n_edges

    if config['use_mask']:
        env = ActionMasker(env, mask_fn)

    # === انتخاب Policy ===
    if config['use_gnn']:
        # پیاده‌سازی GNN واقعی
        from modules.gnn_encoder import GNNFeaturesExtractor
        
        policy_kwargs = {
            "features_extractor_class": GNNFeaturesExtractor,
            "features_extractor_kwargs": {
                "features_dim": 65,
                "n_edges": n_edges,
                "gnn_hidden": 64,
                "gnn_out": 64,
            },
        }
        policy = "MlpPolicy"
    else:
        policy_kwargs = {}
        policy = "MlpPolicy"

    if config['use_mask']:
        model = MaskablePPO(
            policy, env, seed=seed, verbose=0,
            learning_rate=3e-4, n_steps=128, batch_size=64,
            n_epochs=4, gamma=0.99,
            policy_kwargs=policy_kwargs,
        )
    else:
        model = PPO(
            policy, env, seed=seed, verbose=0,
            learning_rate=3e-4, n_steps=128, batch_size=64,
            n_epochs=4, gamma=0.99,
            policy_kwargs=policy_kwargs,
        )

    model.learn(total_timesteps=timesteps)

    # === ارزیابی ===
    obs, _ = env.reset(seed=seed)
    rewards, violations, lambdas = [], [], []

    for _ in range(50):
        if config['use_mask']:
            mask = env.unwrapped.get_action_mask()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, r, term, trunc, info = env.step(action)
        rewards.append(r)
        violations.append(info['violations'])
        lambdas.append(info['lambda_min'])
        if term or trunc:
            break

    avg_r = float(np.mean(rewards))
    avg_v = float(np.mean(violations))
    avg_l = float(np.mean(lambdas))

    print(f"[DONE] {config['name']} | Seed {seed} | "
          f"R: {avg_r:.3f} | V: {avg_v:.2f} | λ: {avg_l:.1f}")

    return avg_r, avg_v, avg_l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=30000)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--network', type=str, default='abilene')
    args = parser.parse_args()

    os.makedirs('./outputs/logs', exist_ok=True)
    os.makedirs('./outputs/plots', exist_ok=True)
    os.makedirs('./outputs/tables', exist_ok=True)

    base = os.path.dirname(os.path.dirname(PROJECT_ROOT))
    filepath = os.path.join(base, f'data/raw/sndlib/static/{args.network}.txt')

    print(f"\n[MAIN] Loading network: {args.network}")

    graph = load_network(filepath)
    print(f"[MAIN] Timesteps: {args.timesteps}, Seeds: {args.seeds}")
    print("=" * 60)

    results = []
    for config in EXPERIMENTS:
        rs, vs, ls = [], [], []
        for seed in args.seeds:
            r, v, l = run_experiment(config, seed, graph, args.timesteps)
            rs.append(r); vs.append(v); ls.append(l)

        results.append({
            'Model': config['name'],
            'Mean_Reward': np.mean(rs),
            'Std_Reward': np.std(rs),
            'Mean_Violations': np.mean(vs),
            'Std_Violations': np.std(vs),
            'Mean_Lambda_Min': np.mean(ls),
            'Std_Lambda_Min': np.std(ls),
        })
        print(f"\n[SUMMARY] {config['name']}: R={np.mean(rs):.3f}±{np.std(rs):.3f}")

    df = pd.DataFrame(results)
    df.to_csv('./outputs/tables/results.csv', index=False)

    print("\n" + "=" * 60)
    print("[FINAL TABLE]")
    print(df.to_string(index=False))
    print("=" * 60)

    # نمودار
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['gray', 'orange', 'blue', 'green']

    axes[0].bar(df['Model'], df['Mean_Reward'], yerr=df['Std_Reward'],
                capsize=5, color=colors)
    axes[0].set_ylabel('Mean Reward')
    axes[0].set_title('Reward')
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(df['Model'], df['Mean_Violations'], yerr=df['Std_Violations'],
                capsize=5, color=colors)
    axes[1].set_ylabel('Mean Violations')
    axes[1].set_title('Violations')
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(df['Model'], df['Mean_Lambda_Min'], yerr=df['Std_Lambda_Min'],
                capsize=5, color=colors)
    axes[2].set_ylabel('Mean λ_min')
    axes[2].set_title('Resilience')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./outputs/plots/comparison.png', dpi=150)
    print("\n[INFO] Plot saved")

    neuro = df[df['Model'] == 'Neuro_Full'].iloc[0]
    baseline = df[df['Model'] == 'Baseline'].iloc[0]
    print("\n[CONCLUSION]")
    if neuro['Mean_Reward'] > baseline['Mean_Reward']:
        print(f"✅ Neuro_Full > Baseline (Reward)")
    if neuro['Mean_Lambda_Min'] > baseline['Mean_Lambda_Min']:
        print(f"✅ Neuro_Full > Baseline (λ_min)")
    if neuro['Mean_Violations'] < baseline['Mean_Violations']:
        print(f"✅ Neuro_Full < Baseline (Violations)")


if __name__ == "__main__":
    main()
