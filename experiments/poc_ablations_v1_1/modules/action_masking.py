# ============================================================================
# ماژول Action Masking دوگانه (مخابراتی + ساختاری)
# ============================================================================
# فرمول‌های کلیدی (از سند v1.1):
# - M_telecom: محدودیت ظرفیت فیزیکی
# - M_GH: محدودیت ساختاری بر اساس CriticalEdges
# - M_total = M_telecom * M_GH
# ============================================================================

import numpy as np
from .gomory_hu_module import GomoryHuAnalyzer


class DualActionMasker:
    """
    تولید ماسک دوگانه برای اقدامات PPO
    
    اقدام‌ها به‌صورت (edge_idx, direction) کدگذاری می‌شوند:
      - action = 2 * edge_idx + direction
      - direction = 0: کاهش ظرفیت
      - direction = 1: افزایش ظرفیت
    """
    
    def __init__(self, capacity_min=300.0, capacity_max=5000.0, 
                 capacity_step=200.0):
        self.cap_min = capacity_min
        self.cap_max = capacity_max
        self.cap_step = capacity_step
        # self.gh_analyzer = GomoryHuAnalyzer(cache_interval=5)
        self.gh_analyzer = GomoryHuAnalyzer(cache_interval=100)  # هر ۱۰۰ گام

    
    def get_mask(self, graph, edges, step=0):
        """
        تولید ماسک نهایی
        
        ورودی:
            graph: NetworkX graph
            edges: لیست یال‌ها (ترتیب مهم است)
            step: شماره گام (برای caching GH)
        
        خروجی:
            mask: آرایه numpy با ابعاد (2 * len(edges),)
                  مقادیر ۰ = نامعتبر، ۱ = معتبر
        """
        n_edges = len(edges)
        mask = np.ones(2 * n_edges, dtype=np.int8)
        
        # === ۱. ماسک مخابراتی ===
        mask = self._apply_telecom_mask(mask, graph, edges)
        
        # === ۲. ماسک ساختاری (Gomory-Hu) ===
        mask = self._apply_gh_mask(mask, graph, edges, step)
        
        return mask
    
    def _apply_telecom_mask(self, mask, graph, edges):
        """
        M_telecom: جلوگیری از اقدامات غیرفیزیکی
        
        طبق سند:
          - اگر c(e) <= c_min + Δc: کاهش ممنوع
          - اگر c(e) >= c_max - Δc: افزایش ممنوع
        """
        for idx, (u, v) in enumerate(edges):
            cap = graph[u][v].get('capacity', 2000.0)
            
            # کاهش (direction=0 → index=2*idx)
            if cap <= self.cap_min + self.cap_step:
                mask[2 * idx] = 0
            
            # افزایش (direction=1 → index=2*idx+1)
            if cap >= self.cap_max - self.cap_step:
                mask[2 * idx + 1] = 0
        
        return mask
    
    def _apply_gh_mask(self, mask, graph, edges, step):
        """
        M_GH: جلوگیری از کاهش ظرفیت یال‌های بحرانی
        
        طبق سند:
          - اگر e ∈ CriticalEdges و action = کاهش: ممنوع
        """
        # محاسبه (یا استفاده از cache) درخت GH
        gh_result = self.gh_analyzer.compute(graph, step=step)
        critical_indices = self.gh_analyzer.get_critical_edge_indices(edges)
        
        for idx in critical_indices:
            # کاهش ظرفیت یال بحرانی ممنوع
            mask[2 * idx] = 0
        
        return mask
    
    def get_stats(self, mask, edges):
        """آمار ماسک برای گزارش‌دهی"""
        n_edges = len(edges)
        n_actions = 2 * n_edges
        n_valid = int(mask.sum())
        n_invalid = n_actions - n_valid
        
        # تفکیک بر اساس نوع
        n_telecom_blocked = 0
        n_gh_blocked = 0
        
        return {
            'total_actions': n_actions,
            'valid_actions': n_valid,
            'invalid_actions': n_invalid,
            'valid_ratio': n_valid / n_actions if n_actions > 0 else 0.0,
        }