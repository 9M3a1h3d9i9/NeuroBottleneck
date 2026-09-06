# اسکریپت اصلی برای اجرای ۴ حالت Ablation و ذخیره نتایج

import os
import sys
import argparse
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
import torch
import matplotlib.pyplot as plt

from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker

# ایمپورت محیط خودمان
from env_poc import NeuroBottleneckEnv

# تنظیمات اولیه
SEEDS = [42, 123, 456, 789, 1012]
EXPERIMENTS = [
    {'name': 'Baseline', 'use_gnn': False, 'use_mask': False},
    {'name': 'Telecom_Mask', 'use_gnn': False, 'use_mask': True},
    {'name': 'GNN_PPO', 'use_gnn': True, 'use_mask': False},
    {'name': 'Neuro_Full', 'use_gnn': True, 'use_mask': True},  # نوآوری ما
]

def run_single_experiment(exp_config, seed, total_timesteps=30000):
    """اجرای یک آزمایش برای یک Seed مشخص"""
    
    # ساخت محیط
    env = NeuroBottleneckEnv(use_gnn=exp_config['use_gnn'])
    env.reset(seed=seed)

    # اگر ماسک فعال است، محیط را با ActionMasker اعمال کن
    if exp_config['use_mask']:
        def mask_fn(env):
            # return env._get_action_mask()
            return env.get_action_mask()
        env = ActionMasker(env, mask_fn)
    
    # انتخاب مدل
    model_name = f"{exp_config['name']}_seed{seed}"
    log_dir = f"./outputs/logs/{exp_config['name']}/"
    os.makedirs(log_dir, exist_ok=True)
    
    if exp_config['use_mask']:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            tensorboard_log=log_dir
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            tensorboard_log=log_dir
        )
    
    # آموزش
    print(f"[RUN] {exp_config['name']} | Seed {seed} | Training...")
    model.learn(total_timesteps=total_timesteps)
    
    # ارزیابی (اجرای ۲۰ گام با مدل آموزش‌دیده)
    obs, _ = env.reset()
    rewards = []
    violations = []
    
    for _ in range(20):
        if exp_config['use_mask']:
            # mask = env._get_action_mask()
            mask = env.get_action_mask()

            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        
        obs, r, terminated, truncated, info = env.step(action)
        rewards.append(r)
        violations.append(info['violations'])
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    avg_reward = np.mean(rewards)
    avg_violations = np.mean(violations)
    
    print(f"[DONE] {exp_config['name']} | Seed {seed} | Reward: {avg_reward:.3f} | Violations: {avg_violations:.2f}")
    
    return avg_reward, avg_violations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=30000, help='Number of training steps')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS, help='List of seeds')
    args = parser.parse_args()
    
    # ساخت پوشه‌های خروجی
    os.makedirs("./outputs/tables", exist_ok=True)
    os.makedirs("./outputs/plots", exist_ok=True)
    os.makedirs("./outputs/logs", exist_ok=True)
    
    results = []
    
    print("="*60)
    print("[MAIN] Starting Ablation Study for NeuroBottleneck")
    print(f"[MAIN] Timesteps per run: {args.timesteps}")
    print(f"[MAIN] Seeds: {args.seeds}")
    print("="*60)
    
    for exp in EXPERIMENTS:
        exp_name = exp['name']
        print(f"\n[EXPERIMENT] Running: {exp_name}")
        
        rewards_list = []
        viol_list = []
        
        for seed in args.seeds:
            avg_r, avg_v = run_single_experiment(exp, seed, args.timesteps)
            rewards_list.append(avg_r)
            viol_list.append(avg_v)
        
        # محاسبه آمار
        mean_r = np.mean(rewards_list)
        std_r = np.std(rewards_list)
        mean_v = np.mean(viol_list)
        std_v = np.std(viol_list)
        
        results.append({
            'Model': exp_name,
            'Mean_Reward': mean_r,
            'Std_Reward': std_r,
            'Mean_Violations': mean_v,
            'Std_Violations': std_v
        })
        
        print(f"[SUMMARY] {exp_name} -> Reward: {mean_r:.3f}±{std_r:.3f} | Violations: {mean_v:.2f}±{std_v:.2f}")
    
    # -------------------------------------------------
    # ذخیره جدول نتایج
    # -------------------------------------------------
    df = pd.DataFrame(results)
    csv_path = "./outputs/tables/ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Results saved to {csv_path}")
    
    # نمایش جدول نهایی
    print("\n" + "="*60)
    print("[FINAL TABLE]")
    print(df.to_string(index=False))
    print("="*60)
    
    # -------------------------------------------------
    # رسم نمودار مقایسه‌ای
    # -------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # نمودار پاداش
    ax1.bar(df['Model'], df['Mean_Reward'], yerr=df['Std_Reward'], capsize=5, color=['gray', 'orange', 'blue', 'green'])
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Comparison of Models (Reward)')
    ax1.grid(True, alpha=0.3)
    
    # نمودار تخلفات
    ax2.bar(df['Model'], df['Mean_Violations'], yerr=df['Std_Violations'], capsize=5, color=['gray', 'orange', 'blue', 'green'])
    ax2.set_ylabel('Average Violations (Saturated Links)')
    ax2.set_title('Comparison of Models (Violations)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "./outputs/plots/ablation_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] Plot saved to {plot_path}")
    
    # -------------------------------------------------
    # نتیجه‌گیری نهایی
    # -------------------------------------------------
    neuro_row = df[df['Model'] == 'Neuro_Full']
    baseline_row = df[df['Model'] == 'Baseline']
    
    if not neuro_row.empty and not baseline_row.empty:
        if neuro_row.iloc[0]['Mean_Reward'] > baseline_row.iloc[0]['Mean_Reward']:
            print("\n[CONCLUSION] ✅ Neuro_Full outperforms Baseline! GH Prior injection is effective.")
        else:
            print("\n[CONCLUSION] ⚠️ Neuro_Full did NOT outperform Baseline. Consider Soft Penalty strategy.")
    else:
        print("\n[CONCLUSION] Data not sufficient for comparison.")

if __name__ == "__main__":
    main()