import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules.sndlib_loader import load_network
from modules.network_env import NetworkEnvV11
from stable_baselines3 import PPO

st.title("🎯 NeuroBottleneck Live Visualization")

# بارگذاری
base = os.path.dirname(os.path.dirname(PROJECT_ROOT))
filepath = os.path.join(base, 'data/raw/sndlib/static/abilene.txt')
graph = load_network(filepath)
env = NetworkEnvV11(graph, seed=42)

# دکمه شروع
if st.button("Run Episode"):
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=5000)
    
    obs, _ = env.reset(seed=42)
    pos = nx.spring_layout(env.graph, seed=42)
    
    progress = st.progress(0)
    status = st.empty()
    plot_area = st.empty()
    
    for step in range(30):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        
        # رسم
        fig, ax = plt.subplots(figsize=(10, 7))
        nx.draw_networkx_nodes(env.graph, pos, ax=ax, node_color='lightblue', node_size=600)
        nx.draw_networkx_labels(env.graph, pos, ax=ax)
        
        edge_colors = ['red' if env.graph[u][v]['utilization'] > 0.8
                       else 'orange' if env.graph[u][v]['utilization'] > 0.6
                       else 'green' for u, v in env.edges]
        nx.draw_networkx_edges(env.graph, pos, ax=ax, edge_color=edge_colors, width=2)
        
        ax.set_title(f"Step {step} | Reward: {reward:.3f} | Violations: {info['violations']}")
        ax.axis('off')
        
        plot_area.pyplot(fig)
        status.metric("Reward", f"{reward:.3f}", f"λ_min: {info['lambda_min']:.1f}")
        progress.progress((step + 1) / 30)
        plt.close()