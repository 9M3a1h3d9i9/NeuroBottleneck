"""ارزیابی مدل آموزش‌دیده"""

import numpy as np


def evaluate_model(model, env, n_episodes=5, use_mask=False):
    rewards = []
    violations = []
    lambda_mins = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_violations = []
        ep_lambda = []

        done = False
        while not done:
            if use_mask and hasattr(env, 'get_action_mask'):
                mask = env.get_action_mask()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            ep_violations.append(info['violations'])
            ep_lambda.append(info['lambda_min'])
            done = terminated or truncated

        rewards.append(ep_reward)
        violations.append(np.mean(ep_violations))
        lambda_mins.append(np.mean(ep_lambda))

    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_violations': float(np.mean(violations)),
        'mean_lambda_min': float(np.mean(lambda_mins)),
        'raw_rewards': [float(x) for x in rewards],
    }