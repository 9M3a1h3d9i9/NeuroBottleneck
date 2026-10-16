"""رسم نمودارهای مقایسه‌ای — نسخه مقاوم"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison_bar(summary_df, output_dir, metric='mean_reward'):
    """نمودار میله‌ای مقایسه میانگین‌ها"""
    if metric not in summary_df.columns:
        print(f"[WARN] Metric {metric} not found in summary")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = summary_df.groupby(['experiment'])[metric].agg(['mean', 'std']).reset_index()
    x = range(len(groups))
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    ax.bar(x, groups['mean'].fillna(0), yerr=groups['std'].fillna(0),
           capsize=5, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(groups['experiment'], rotation=30, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison: {metric}")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"comparison_{metric}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


def plot_training_curves(output_dir, experiments, topologies, seeds,
                          metric='reward', window=50):
    """
    رسم منحنی‌های آموزش.
    
    ✅ FIX: if topologies is a string, wrap in list.
    """
    # ✅ اگر رشته بود، به لیست تبدیل کن
    if isinstance(topologies, str):
        topologies = [topologies]
    
    fig, axes = plt.subplots(1, len(topologies), figsize=(6*len(topologies), 5),
                              squeeze=False)
    
    for ti, topo in enumerate(topologies):
        ax = axes[0][ti]
        has_data = False  # ✅ این خط حیاتی است
        
        for exp in experiments:
            all_curves = []
            for seed in seeds:
                # ✅ مسیر: output_dir/EXP_NAME_TOPO_seedSEED/logs/training_metrics.csv
                run_dir = os.path.join(output_dir, f"{exp.name}_{topo}_seed{seed}")
                csv_path = os.path.join(run_dir, "logs", "training_metrics.csv")
                
                # ✅ Debug print برای شناسایی مسیر
                if not os.path.exists(csv_path):
                    print(f"[DEBUG] CSV not found: {csv_path}")
                    continue
                
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"[DEBUG] Failed to read {csv_path}: {e}")
                    continue
                
                if metric not in df.columns:
                    print(f"[DEBUG] Column '{metric}' not in {csv_path}. Columns: {list(df.columns)}")
                    continue
                
                vals = df[metric].dropna().values
                if len(vals) == 0:
                    continue
                
                # پنجره داینامیک
                w = min(window, max(1, len(vals) // 5))
                if w > 1:
                    vals = np.convolve(vals, np.ones(w)/w, mode='valid')
                all_curves.append(vals)
            
            if all_curves:
                has_data = True
                min_len = min(len(c) for c in all_curves)
                curves = np.array([c[:min_len] for c in all_curves])
                mean = curves.mean(axis=0)
                std = curves.std(axis=0)
                x = np.arange(len(mean))
                ax.plot(x, mean, label=exp.name, linewidth=2)
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        
        if has_data:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No training data\n(check CSV paths in logs)",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.set_xlabel("Training Step")
        ax.set_ylabel(metric)
        ax.set_title(f"Training Curves ({metric}) - {topo}")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"training_curves_{metric}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


def plot_overfitting_check(output_dir, experiments, topology, seeds, plots_dir):
    """بررسی Overfitting: مقایسه reward آموزش و ارزیابی"""
    if isinstance(topology, str):
        topology = [topology]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_means = []
    eval_means = []
    labels = []
    
    for exp in experiments:
        for seed in seeds:
            run_dir = os.path.join(output_dir, f"{exp.name}_{topology[0]}_seed{seed}")
            csv_path = os.path.join(run_dir, "logs", "training_metrics.csv")
            json_path = os.path.join(run_dir, "logs", "eval_results.json")
            
            if not os.path.exists(csv_path) or not os.path.exists(json_path):
                print(f"[DEBUG] Missing files for {exp.name} seed {seed}")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                with open(json_path) as f:
                    eval_res = json.load(f)
                
                # میانگین ۱۰۰ گام آخر آموزش
                if len(df) > 0:
                    train_mean = df['reward'].rolling(50, min_periods=1).mean().iloc[-1]
                    train_means.append(train_mean)
                    eval_means.append(eval_res['mean_reward'])
                    labels.append(f"{exp.name}\nseed={seed}")
            except Exception as e:
                print(f"[DEBUG] Failed to process {run_dir}: {e}")
                continue
    
    if not train_means:
        ax.text(0.5, 0.5, "No overfitting data available",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    else:
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, train_means, width, label='Train (last 50 avg)', alpha=0.7)
        ax.bar(x + width/2, eval_means, width, label='Eval (deterministic)', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel("Reward")
        ax.legend()
    
    ax.set_title(f"Overfitting Check ({topology[0]})")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "overfitting_check.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


def plot_all(summary_df, output_dir, experiments, topology, seeds):
    """
    فراخوانی همه نمودارها.
    
    ✅ FIX: همه توابع با لیست topology صدا زده می‌شوند.
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # نمودارهای میله‌ای
    plot_comparison_bar(summary_df, plots_dir, 'mean_reward')
    plot_comparison_bar(summary_df, plots_dir, 'mean_violations')
    plot_comparison_bar(summary_df, plots_dir, 'mean_lambda_min')
    
    # ✅ نمودارهای منحنی — با [topology] که لیست است
    topologies_list = [topology] if isinstance(topology, str) else topology
    for metric in ['reward', 'violations']:
        plot_training_curves(output_dir, experiments, topologies_list, seeds, metric=metric)
    
    # Overfitting
    plot_overfitting_check(output_dir, experiments, topology, seeds, plots_dir)
