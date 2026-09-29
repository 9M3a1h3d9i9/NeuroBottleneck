"""نمایش زنده تصمیم‌گیری مدل روی گراف"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules.sndlib_loader import load_network
from modules.network_env import NetworkEnvV11
from stable_baselines3 import PPO


def visualize_episode(model, env, max_steps=50):
    """اجرای یک اپیزود و ذخیره فریم‌ها"""
    
    obs, _ = env.reset(seed=42)
    
    # موقعیت گره‌ها (ثابت)
    pos = nx.spring_layout(env.graph, seed=42)
    
    # ذخیره تاریخچه
    history = []
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        
        # ذخیره وضعیت
        edge_colors = []
        edge_widths = []
        for u, v in env.edges:
            util = env.graph[u][v]['utilization']
            if util > 0.8:
                edge_colors.append('red')
                edge_widths.append(4)
            elif util > 0.6:
                edge_colors.append('orange')
                edge_widths.append(3)
            else:
                edge_colors.append('green')
                edge_widths.append(2)
        
        history.append({
            'step': step,
            'action': action,
            'reward': reward,
            'violations': info['violations'],
            'lambda_min': info['lambda_min'],
            'edge_colors': edge_colors.copy(),
            'edge_widths': edge_widths.copy(),
            'edge_labels': {
                (u, v): f"{env.graph[u][v]['utilization']:.2f}"
                for u, v in env.edges
            },
        })
        
        if term or trunc:
            break
    
    return history, pos


def create_animation(history, pos, env, output_file='episode.gif'):
    """ساخت انیمیشن از تاریخچه"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def update(frame):
        ax.clear()
        h = history[frame]
        
        # رسم گراف
        nx.draw_networkx_nodes(env.graph, pos, ax=ax,
                               node_color='lightblue', node_size=600)
        nx.draw_networkx_labels(env.graph, pos, ax=ax, font_size=10)
        nx.draw_networkx_edges(env.graph, pos, ax=ax,
                               edge_color=h['edge_colors'],
                               width=h['edge_widths'])
        nx.draw_networkx_edge_labels(env.graph, pos, ax=ax,
                                     edge_labels=h['edge_labels'],
                                     font_size=8)
        
        # اطلاعات
        title = (f"Step {h['step']} | Reward: {h['reward']:.3f} | "
                 f"Violations: {h['violations']} | λ_min: {h['lambda_min']:.1f}")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    anim = animation.FuncAnimation(fig, update, frames=len(history),
                                    interval=500, repeat=True)
    anim.save(output_file, writer='pillow', fps=2)
    print(f"[INFO] Animation saved to {output_file}")
    plt.close()


def main():
    # بارگذاری
    base = os.path.dirname(os.path.dirname(PROJECT_ROOT))
    filepath = os.path.join(base, 'data/raw/sndlib/static/abilene.txt')
    graph = load_network(filepath)
    
    # محیط
    env = NetworkEnvV11(graph, seed=42)
    
    # مدل آموزش‌دیده (اگر موجود است)
    model_path = "./outputs/models/best_model"
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
    else:
        print("[WARN] No trained model found. Training a quick model...")
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=5000)
        os.makedirs("./outputs/models", exist_ok=True)
        model.save(model_path)
    
    # اجرای اپیزود و ذخیره فریم‌ها
    print("[INFO] Running episode...")
    history, pos = visualize_episode(model, env, max_steps=30)
    
    # ساخت انیمیشن
    create_animation(history, pos, env, output_file='./outputs/plots/episode.gif')


if __name__ == "__main__":
    main()