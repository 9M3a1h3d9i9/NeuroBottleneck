"""رسم نمودارهای مقایسه‌ای — نسخه ۲ (خوانا و تحلیلی)"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# ابزارهای کمکی
# ============================================================

def _rolling_mean_std(vals, window):
    """میانگین و انحراف معیار متحرک با پنجره"""
    if len(vals) < window:
        window = max(1, len(vals) // 5)
    mean = pd.Series(vals).rolling(window, min_periods=1).mean().values
    std = pd.Series(vals).rolling(window, min_periods=1).std().fillna(0).values
    return mean, std


def _format_k(x, pos):
    """فرمت محور X به K"""
    return f"{int(x/1000)}K" if x >= 1000 else f"{int(x)}"


# ============================================================
# نمودار ۱: مقایسه میله‌ای (بدون تغییر)
# ============================================================

def plot_comparison_bar(summary_df, output_dir, metric='mean_reward'):
    if metric not in summary_df.columns:
        print(f"[WARN] Metric {metric} not found")
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


# ============================================================
# نمودار ۲: منحنی آموزش (نسخه بهبود‌یافته)
# ============================================================

def plot_training_curves(output_dir, experiments, topologies, seeds,
                          metric='reward', window=500):
    """
    رسم منحنی‌های آموزش با:
    - میانگین متحرک (پنجره ۵۰۰ گام)
    - نوار انحراف معیار
    - نشانگر نقطه همگرایی (۹۰٪ پاداش نهایی)
    """
    if isinstance(topologies, str):
        topologies = [topologies]
    
    fig, axes = plt.subplots(1, len(topologies), figsize=(8*len(topologies), 6),
                              squeeze=False)
    
    for ti, topo in enumerate(topologies):
        ax = axes[0][ti]
        has_data = False
        
        for exp in experiments:
            all_curves = []
            for seed in seeds:
                run_dir = os.path.join(output_dir, f"{exp.name}_{topo}_seed{seed}")
                csv_path = os.path.join(run_dir, "logs", "training_metrics.csv")
                if not os.path.exists(csv_path):
                    continue
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                if metric not in df.columns:
                    continue
                vals = df[metric].dropna().values
                if len(vals) == 0:
                    continue
                all_curves.append(vals)
            
            if all_curves:
                has_data = True
                min_len = min(len(c) for c in all_curves)
                curves = np.array([c[:min_len] for c in all_curves])
                
                # میانگین و std روی Seedها
                mean_across_seeds = curves.mean(axis=0)
                std_across_seeds = curves.std(axis=0)
                
                # میانگین متحرک برای خوانایی
                smoothed_mean, smoothed_std = _rolling_mean_std(mean_across_seeds, window)
                x = np.arange(len(smoothed_mean))
                
                # نمودار اصلی
                ax.plot(x, smoothed_mean, label=exp.name, linewidth=2.5)
                ax.fill_between(x, 
                                smoothed_mean - smoothed_std,
                                smoothed_mean + smoothed_std,
                                alpha=0.2)
                
                # نشانگر نقطه‌ی همگرایی (۹۰٪ پاداش نهایی)
                final_val = smoothed_mean[-1]
                target = 0.9 * final_val
                if metric == 'violations':
                    target = 1.1 * final_val  # برای violations، کاهش بهتر است
                idx = np.where(smoothed_mean <= target)[0] if metric == 'violations' \
                      else np.where(smoothed_mean >= target)[0]
                if len(idx) > 0:
                    conv_step = idx[0]
                    ax.axvline(conv_step, linestyle=':', alpha=0.4, 
                               color=ax.lines[-1].get_color())
        
        if has_data:
            ax.legend(loc='best', fontsize=10)
        else:
            ax.text(0.5, 0.5, "No training data",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f"{metric.capitalize()} Curves — {topo}", fontsize=13)
        ax.xaxis.set_major_formatter(FuncFormatter(_format_k))
        ax.grid(True, alpha=0.3)
        
        # توضیح اضافه
        if has_data:
            ax.text(0.02, 0.98, 
                    f"Window: {window}\nShading: ±1σ",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"training_curves_{metric}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


# ============================================================
# نمودار ۳: Overfitting Check (نسخه کاملاً بازنویسی‌شده)
# ============================================================

def plot_overfitting_check(output_dir, experiments, topology, seeds, plots_dir):
    """
    بررسی Overfitting با مقیاس یکسان:
    - Train: میانگین مجموع پاداش ۵۰ گام (rolling 50)
    - Eval: میانگین مجموع پاداش ۵ اپیزود
    """
    if isinstance(topology, str):
        topology = [topology]
    topo = topology[0]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    exp_labels = []
    train_means = []
    train_stds = []
    eval_means = []
    eval_stds = []
    
    for exp in experiments:
        train_per_seed = []
        eval_per_seed = []
        
        for seed in seeds:
            run_dir = os.path.join(output_dir, f"{exp.name}_{topo}_seed{seed}")
            csv_path = os.path.join(run_dir, "logs", "training_metrics.csv")
            json_path = os.path.join(run_dir, "logs", "eval_results.json")
            
            if not os.path.exists(csv_path):
                continue
            
            try:
                df = pd.read_csv(csv_path)
                # ✅ تبدیل پاداش هر گام به "مجموع ۵۰ گام" (یک اپیزود)
                if len(df) >= 50:
                    # rolling sum با پنجره ۵۰ (طول اپیزود)
                    rolling_sum = df['reward'].rolling(50, min_periods=1).sum()
                    train_mean_last = rolling_sum.iloc[-1]  # آخرین اپیزود
                    train_per_seed.append(train_mean_last)
                
                # Eval از json
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        eval_res = json.load(f)
                    eval_per_seed.append(eval_res['mean_reward'])
            except Exception as e:
                print(f"[DEBUG] {run_dir}: {e}")
                continue
        
        if train_per_seed and eval_per_seed:
            exp_labels.append(exp.name)
            train_means.append(np.mean(train_per_seed))
            train_stds.append(np.std(train_per_seed))
            eval_means.append(np.mean(eval_per_seed))
            eval_stds.append(np.std(eval_per_seed))
    
    if not exp_labels:
        ax.text(0.5, 0.5, "No overfitting data",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    else:
        x = np.arange(len(exp_labels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_means, width, yerr=train_stds,
                       capsize=5, label='Train (last episode avg)',
                       color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, eval_means, width, yerr=eval_stds,
                       capsize=5, label='Eval (5 episodes avg)',
                       color='#e67e22', alpha=0.8)
        
        # نمایش اعداد روی میله‌ها
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)
        
        # تحلیل فاصله (Gap)
        gaps = [abs(t - e) / max(abs(e), 1) * 100 for t, e in zip(train_means, eval_means)]
        gap_str = ", ".join([f"{exp}: {gap:.1f}%" 
                             for exp, gap in zip(exp_labels, gaps)])
        
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels, fontsize=11)
        ax.set_ylabel("Episode Reward (sum over 50 steps)", fontsize=11)
        ax.set_title(f"Train vs Eval — Overfitting Check ({topo})\n"
                     f"Gap (|Train−Eval|/Eval): {gap_str}", fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(plots_dir, "overfitting_check.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {save_path}")


# ============================================================
# تابع اصلی
# ============================================================

def plot_all(summary_df, output_dir, experiments, topology, seeds):
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # ۱. میله‌ای‌ها
    plot_comparison_bar(summary_df, plots_dir, 'mean_reward')
    plot_comparison_bar(summary_df, plots_dir, 'mean_violations')
    plot_comparison_bar(summary_df, plots_dir, 'mean_lambda_min')
    
    # ۲. منحنی‌های آموزش
    topologies_list = [topology] if isinstance(topology, str) else topology
    for metric in ['reward', 'violations']:
        plot_training_curves(output_dir, experiments, topologies_list, seeds, metric=metric)
    
    # ۳. Overfitting
    plot_overfitting_check(output_dir, experiments, topology, seeds, plots_dir)
