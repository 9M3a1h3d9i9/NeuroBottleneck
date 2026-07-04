# This file is the beating heart of the settings
# So we don't have to manually change hyperparameters in all files

from dataclasses import dataclass, field

@dataclass
class NetworkConfig:
    # Setting releted to O-RAN Network Topology
    num_nodes: int = 20
    edge_probability: float = 0.3
    min_capacity: float = 10.0 # Mbps
    max_capacity: float = 100.0 # Mbps

@dataclass
class PPOConfig:
    # Hyperparameters related to DRL Algorithm
    learning_rate: float = 3e-4
    gamma: float = 0.99          # ضریب تخفیف پاداش
    eps_clip: float = 0.2        # Clipped Surrogate Objective PPO
    k_epochs: int = 4            # تعداد دفعات به روزرسانی وزن ها
                                    # در هر اپیزود
    batch_size: int = 64

@dataclass
class MainConfig:
    # network: NetworkConfig = NetworkConfig()                          .* Mini Eror
    # ppo: PPOConfig = PPOConfig()                                      .* Mini Eror

    # استفاده از default_factory برای شیءهای تغییرپذیر جهت رفع خطای پایتون
    # Using default_factory for mutable objects to avoid Python error    .* Mini Eror : Solved
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    seed: int = 42
    max_episodes: int = 1000