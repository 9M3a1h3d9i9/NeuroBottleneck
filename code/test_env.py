from network_env import NetworkEnv

env = NetworkEnv()
obs, info = env.reset(seed=42)
print("Observation shape:", obs.shape)
print("Initial obs:", obs)

action = env.action_space.sample()
print("Random action:", action)

obs, reward, terminated, truncated, info = env.step(action)
print("Reward:", reward)
print("Info:", info)
print("New obs:", obs)