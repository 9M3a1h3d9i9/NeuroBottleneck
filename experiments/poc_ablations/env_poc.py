# محیط شبیه‌ساز NeuroBottleneck با قابلیت تزریق دانش گوموری-هو

import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
import torch

# ایمپورت ماژول GNN که در فایل مجزا نوشته‌ایم
from gnn_encoder_poc import SimpleGraphSAGE, nx_to_pyg_data

class NeuroBottleneckEnv(gym.Env):
    def __init__(self, graph=None, use_gnn=False, gnn_out_dim=16):
        super(NeuroBottleneckEnv, self).__init__()
        
        # -------------------------------------------------
        # اگر گرافی داده نشد، گراف Abilene را می‌سازیم
        # -------------------------------------------------
        if graph is None:
            graph = self._build_abilene()
        self.graph = graph
        self.edges = list(self.graph.edges())
        self.n_edges = len(self.edges)
        self.use_gnn = use_gnn
        
        # -------------------------------------------------
        # فضای اقدام: برای هر لینک دو حرکت (افزایش/کاهش ظرفیت)
        # -------------------------------------------------
        self.action_space = spaces.Discrete(self.n_edges * 2)
        
        # -------------------------------------------------
        # فضای مشاهده:
        # اگر GNN فعال باشد => بردار خروجی GNN (ابعاد ثابت)
        # اگر GNN غیرفعال باشد => ویژگی‌های خام لینک‌ها (ظرفیت، اشغال، تأخیر)
        # -------------------------------------------------
        if use_gnn:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(gnn_out_dim,), dtype=np.float32
            )
            self.gnn = SimpleGraphSAGE(in_channels=3, hidden_channels=8, out_channels=gnn_out_dim)
            self.gnn.eval()  # برای تست سریع، وزن‌ها را فریز می‌کنیم
        else:
            self.observation_space = spaces.Box(
                low=0, high=10000, shape=(self.n_edges * 3,), dtype=np.float32
            )
            self.gnn = None
        
        # -------------------------------------------------
        # متغیرهای ذخیره‌سازی تاریخچه
        # -------------------------------------------------
        self.history = {'reward': [], 'violations': [], 'lambda_min': []}
        self.step_count = 0
        self._gh_lambda_threshold = 1200  # آستانه بحرانی برای ماسک گوموری-هو
        # از 1500 به 1200
        print(f"[ENV] Init complete. Use GNN: {use_gnn}, Obs Dim: {self.observation_space.shape[0]}")

    # --------------------------------------------------------------------
    # تابع کمکی: ساخت گراف Abilene (۱۱ گره، ۱۵ لینک)
    # --------------------------------------------------------------------
    def _build_abilene(self):
        G = nx.Graph()
        nodes = ['NY', 'WA', 'CHI', 'DEN', 'KC', 'LA', 'ATL', 'HO', 'SEA', 'SV', 'IN']
        G.add_nodes_from(range(len(nodes)))
        edges = [
            (0, 1, 2000), (0, 2, 2000), (0, 8, 2000),
            (1, 2, 2000), (1, 9, 2000),
            (2, 3, 2000), (2, 4, 2000), (2, 5, 2000),
            (3, 5, 2000), (3, 10, 2000),
            (4, 6, 2000), (4, 7, 2000),
            (5, 6, 2000), (5, 7, 2000),
            (6, 0, 2000)
        ]
        G.add_weighted_edges_from(edges, weight='capacity')
        for u, v in G.edges():
            G[u][v]['utilization'] = np.random.uniform(0.2, 0.5)
            G[u][v]['delay'] = np.random.randint(5, 25)
        return G

    # --------------------------------------------------------------------
    # تابع دریافت وضعیت (Observation)
    # --------------------------------------------------------------------
    def _get_obs(self):
        if self.use_gnn and self.gnn is not None:
            with torch.no_grad():
                pyg_data = nx_to_pyg_data(self.graph)
                emb = self.gnn(pyg_data.x, pyg_data.edge_index)
                return emb.numpy().flatten()
        else:
            obs = []
            for u, v in self.edges:
                obs.append(self.graph[u][v]['capacity'])
                obs.append(self.graph[u][v]['utilization'])
                obs.append(self.graph[u][v]['delay'])
            return np.array(obs, dtype=np.float32)

    # --------------------------------------------------------------------
    # تابع محاسبه درخت گوموری-هو و استخراج λ_min
    # --------------------------------------------------------------------
    def _compute_gh_lambda(self):
        try:
            gh_tree = nx.gomory_hu_tree(self.graph, capacity='capacity')
            min_cut = np.inf
            for u, v in gh_tree.edges():
                w = gh_tree[u][v]['weight']
                if w < min_cut:
                    min_cut = w
            return min_cut if min_cut != np.inf else 0.0
        except Exception as e:
            # اگر گراف جهت‌دار بود یا خطای دیگر، مقدار پیش‌فرض برمی‌گردانیم
            return 1000.0

    # --------------------------------------------------------------------
    # هسته نوآوری: تولید ماسک اقدامات (Telecom + Gomory-Hu)
    # --------------------------------------------------------------------
    def _get_action_mask(self):
        mask = np.ones(self.action_space.n, dtype=np.int8)
        
        # 1. ماسک مخابراتی (محدودیت فیزیکی ظرفیت)
        for idx, (u, v) in enumerate(self.edges):
            cap = self.graph[u][v]['capacity']
            if cap <= 100:
                mask[2*idx] = 0      # کاهش بیشتر ممنوع
            if cap >= 5000:
                mask[2*idx + 1] = 0  # افزایش بیشتر ممنوع
        
        # 2. ماسک گوموری-هو (نوآوری اصلی)
        # اگر شبکه در آستانه فروپاشی است، کاهش لینک‌های حیاتی را ممنوع کن
        current_lambda = self._compute_gh_lambda()
        if current_lambda < self._gh_lambda_threshold:
            for idx, (u, v) in enumerate(self.edges):
                # شبیه‌سازی کاهش ۱۰۰ واحدی
                temp_cap = self.graph[u][v]['capacity'] - 100
                if temp_cap >= 100:
                    original_cap = self.graph[u][v]['capacity']
                    self.graph[u][v]['capacity'] = temp_cap
                    new_lambda = self._compute_gh_lambda()
                    self.graph[u][v]['capacity'] = original_cap  # بازگردانی
                    
                    if new_lambda < self._gh_lambda_threshold:
                        mask[2*idx] = 0  # کاهش این لینک ممنوع!
        
        return mask

    def get_action_mask(self):
        """  متد عمومی برای دریافت ماسک اقدامات """
        """  استفاده توسط ActionMasker و ارزیابی """
        return self._get_action_mask()

    # --------------------------------------------------------------------
    # توابع استاندارد Gymnasium
    # --------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # بازنشانی گراف به حالت اولیه
        for u, v in self.edges:
            self.graph[u][v]['capacity'] = 2000
            self.graph[u][v]['utilization'] = np.random.uniform(0.2, 0.5)
            self.graph[u][v]['delay'] = np.random.randint(5, 25)

        # تنش‌زایی: کاهش تصادفی ظرفیت برخی لینک‌ها
        # 🆕 تنش‌زایی: کاهش تصادفی ۲-۳ لینک به ۵۰۰-۸۰۰
        import random
        num_critical = min(3, len(self.edges))
        critical_edges = random.sample(self.edges, num_critical)
        for u, v in critical_edges:
            self.graph[u][v]['capacity'] = random.randint(500, 800)
        
        self.step_count = 0
        self.history = {'reward': [], 'violations': [], 'lambda_min': []}
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        
        # دیکد کردن اقدام
        edge_idx = action // 2
        direction = action % 2  # 0: کاهش, 1: افزایش
        u, v = self.edges[edge_idx]
        
        delta = 100 if direction == 1 else -100
        new_cap = max(100, min(5000, self.graph[u][v]['capacity'] + delta))
        self.graph[u][v]['capacity'] = new_cap
        
        # شبیه‌سازی پویای ترافیک (تصادفی با نویز)
        # self.graph[u][v]['utilization'] = max(0.1, min(1.0, np.random.normal(0.4, 0.1)))
        
        # شبیه‌سازی پویای ترافیک (بار بالا: میانگین ۷۵٪)
        self.graph[u][v]['utilization'] = max(0.3, min(1.0, np.random.normal(0.75, 0.15)))


        # -------------------------------------------------
        # محاسبه پاداش (3 مؤلفه)
        # -------------------------------------------------
        total_cap = sum(self.graph[u][v]['capacity'] for u, v in self.edges)
        reward_tp = total_cap / 10000.0  # نرمال‌سازی
        
        violations = sum(1 for u, v in self.edges if self.graph[u][v]['utilization'] > 0.8)
        reward_vio = -violations * 0.5
        
        lambda_min = self._compute_gh_lambda()
        reward_res = lambda_min / 5000.0
        
        # پاداش نهایی (ضرایب قابل تنظیم)
        reward = (0.4 * reward_tp) + (0.3 * reward_res) + (0.3 * reward_vio)
        
        # ذخیره تاریخچه
        self.history['reward'].append(reward)
        self.history['violations'].append(violations)
        self.history['lambda_min'].append(lambda_min)
        
        # پایان دوره بعد از ۵۰ گام (برای تست سریع)
        terminated = self.step_count >= 50
        truncated = False
        
        info = {
            'mask': self._get_action_mask(),
            'gh_lambda': lambda_min,
            'violations': violations
        }
        
        return self._get_obs(), reward, terminated, truncated, info