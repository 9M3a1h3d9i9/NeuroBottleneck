import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces

class NetworkEnv(gym.Env):
    def __init__(self, num_nodes=15, num_edges=30, augment_amount=20.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.augment_amount = augment_amount
        self.graph = None
        self.edge_list = None

        # فضای عمل: شماره یال انتخاب‌شده
        self.action_space = spaces.Discrete(num_edges)
        # فضای مشاهده: ظرفیت تمام یال‌ها
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(num_edges,), dtype=np.float32
        )

    # def _create_graph(self):
    #     G = nx.gnm_random_graph(self.num_nodes, self.action_space.n,
    #                             seed=np.random.randint(0, 10000))
    #     for u, v in G.edges():
    #         G[u][v]['capacity'] = np.random.randint(10, 100)
    #     return G

    # بایستی بعد از ساخت گراف، بررسی کنیم که همبند باشد و اگر نبود،
    # بایستی یال اضافه کنیم تا همبند شود.
    def _create_graph(self):
        G = nx.gnm_random_graph(self.num_nodes, self.action_space.n,
                                seed=np.random.randint(0, 10000))
        # Ensure the graph is connected
        while not nx.is_connected(G):
            # Pick two random nodes from different components and connect them
            components = list(nx.connected_components(G))
            if len(components) < 2:
                break
            comp1 = list(components[0])
            comp2 = list(components[1])
            u = np.random.choice(comp1)
            v = np.random.choice(comp2)
            G.add_edge(u, v)
        for u, v in G.edges():
            G[u][v]['capacity'] = np.random.randint(10, 100)
        return G

    def _get_obs(self):
        return np.array([self.graph[u][v]['capacity'] for u, v in self.edge_list],
                        dtype=np.float32)

    def _global_mincut(self):
        cut_value, _ = nx.stoer_wagner(self.graph, weight='capacity')
        return cut_value

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.graph = self._create_graph()
        self.edge_list = list(self.graph.edges())
        # در صورتی که تعداد یال‌ها دقیقاً ۳۰ نبود، فضای عمل و مشاهده را اصلاح کن
        n_edges = len(self.edge_list)
        self.action_space = spaces.Discrete(n_edges)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(n_edges,), dtype=np.float32
        )
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        u, v = self.edge_list[action]
        old_mincut = self._global_mincut()

        # تقویت یال انتخاب‌شده
        self.graph[u][v]['capacity'] += self.augment_amount

        new_mincut = self._global_mincut()
        reward = (new_mincut - old_mincut) - 0.1

        terminated = False
        truncated = False
        info = {"old_mincut": old_mincut, "new_mincut": new_mincut}

        return self._get_obs(), reward, terminated, truncated, info