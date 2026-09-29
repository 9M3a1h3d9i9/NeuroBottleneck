"""رسم نمودارهای مقایسه‌ای + Overfitting Check"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(output_dir, experiments, topologies, seeds,
                         metric='reward', window=50):
    fig, axes = plt.subplots(1, len(topologies), figsize=(6*len(topologies), 5),
                              squeeze=False)

    for ti, topo in enumerate(topologies):
        ax = axes[0][ti]
        for exp in experiments:
            all_curves = []
            for seed in seeds:
                run_dir = os.path.join(output_dir, f"{exp.name}_{topo}_seed{seed}")
                csv_path = os.path.join(run_dir, "logs", "training_metrics.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if metric in df.columns:
                        vals = df[metric].dropna().values
                        if window > 1 and len(vals) >= window:
                            vals = np.convolve(vals, np.ones(window)/window,
                                               mode='valid')
                        all_curves.append(vals)

            if all_curves:
                min_len = min(len(c) for c in all_curves)
                curves = np.array([c[:min_len] for c in all_curves])
                mean = curves.mean(axis=0)
                std = curves.std(axis=0)
                x = np.arange(len(mean))
                ax.plot(x, mean, label=exp.name, linewidth=2)
                ax.fill_between(x, mean-std, mean+std, alpha=0.2)

        ax.set_xlabel("Training Step")
        ax.set_ylabel(metric)
        ax.set_title(f"Topology: {topo}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"training_curves_{metric}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


def plot_comparison_bar(summary_df, output_dir, metric='mean_reward'):
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = summary_df.groupby(['experiment', 'topology'])[metric].agg(
        ['mean', 'std']).reset_index()

    x_labels = []
    means = []
    stds = []
    for _, row in groups.iterrows():
        x_labels.append(f"{row['experiment']}\n({row['topology']})")
        means.append(row['mean'])
        stds.append(row['std'])

    colors = sns.color_palette("husl", len(x_labels))
    ax.bar(range(len(x_labels)), means, yerr=stds, capsize=5, color=colors)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison: {metric}")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"comparison_{metric}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


def plot_overfitting_check(output_dir, experiments, topology, seed, plots_dir):
    """بررسی Overfitting: مقایسه reward آموزش و ارزیابی"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp in experiments:
        run_dir = os.path.join(output_dir, f"{exp.name}_{topology}_seed{seed}")
        csv_path = os.path.join(run_dir, "logs", "training_metrics.csv")
        json_path = os.path.join(run_dir, "logs", "eval_results.json")
        if os.path.exists(csv_path) and os.path.exists(json_path):
            df = pd.read_csv(csv_path)
            with open(json_path) as f:
                eval_res = json.load(f)

            train_mean = df['reward'].rolling(50, min_periods=1).mean().iloc[-1]
            eval_mean = eval_res['mean_reward']
            eval_std = eval_res['std_reward']

            ax.bar(exp.name, train_mean, alpha=0.5, label='Train (last 50 avg)')
            ax.errorbar(exp.name, eval_mean, yerr=eval_std, fmt='o',
                        color='red',
                        label='Eval' if exp.name == experiments[0].name else None)

    ax.set_ylabel("Reward")
    ax.set_title(f"Overfitting Check ({topology})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "overfitting_check.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


def plot_all(summary_df, output_dir, experiments, topologies, seeds):
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")

    plot_comparison_bar(summary_df, plots_dir, 'mean_reward')
    plot_comparison_bar(summary_df, plots_dir, 'mean_violations')
    plot_comparison_bar(summary_df, plots_dir, 'mean_lambda_min')

    for topo in topologies:
        plot_training_curves(output_dir, experiments, [topo], seeds,
                             metric='reward')
        plot_training_curves(output_dir, experiments, [topo], seeds,
                             metric='violations')

    plot_overfitting_check(output_dir, experiments, topologies[0], seeds[0],
                           plots_dir)