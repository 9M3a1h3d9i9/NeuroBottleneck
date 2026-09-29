"""تشخیص مشکلات Neuro_Full"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.sndlib_loader import load_network
from modules.network_env import NetworkEnvV11
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
filepath = os.path.join(base, 'data/raw/sndlib/static/abilene.txt')
graph = load_network(filepath)

env = NetworkEnvV11(graph, seed=42)
obs, _ = env.reset(seed=42)

print("=" * 60)
print("DIAGNOSTIC REPORT")
print("=" * 60)

# λ_min
gh = env.gh_analyzer.compute(env.graph, step=0, force=True)
print(f"\n[Gomory-Hu]")
print(f"  λ_min = {gh['lambda_min']:.2f}")
print(f"  λ_max = {gh['lambda_max']:.2f}")
print(f"  Critical edges: {gh['critical_edges']}")
print(f"  Cache interval: {env.gh_analyzer.cache_interval}")

# ماسک
mask = env.get_action_mask()
print(f"\n[Mask Stats]")
print(f"  Total: {len(mask)}, Valid: {mask.sum()}, Invalid: {len(mask) - mask.sum()}")
print(f"  Valid ratio: {mask.sum()/len(mask):.2%}")

# پارامترها
print(f"\n[Env Params]")
print(f"  CAPACITY_INIT = {env.CAPACITY_INIT}")
print(f"  LAMBDA_THRESHOLD = {env.masker.__dict__.get('lambda_threshold', 'N/A')}")

# ۱۰ گام تصادفی
print(f"\n[10 Random Steps]")
print(f"  Step | Action | Reward  | Viol | λ_min")
print(f"  {'-'*50}")
for step in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(f"  {step+1:>4} | {action:>6} | {reward:>7.3f} | {info['violations']:>4} | {info['lambda_min']:>7.1f}")

print("=" * 60)
