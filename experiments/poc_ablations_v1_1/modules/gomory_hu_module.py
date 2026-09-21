# ============================================================================
# ماژول Gomory-Hu و استخراج یال‌های بحرانی
# ============================================================================
# این ماژول، درخت Gomory-Hu را محاسبه کرده و یال‌های بحرانی (CriticalEdges)
# را استخراج می‌کند. برای کارایی، از caching استفاده می‌کند.
# ============================================================================

import networkx as nx
import numpy as np


class GomoryHuAnalyzer:
    """
    تحلیل‌گر درخت Gomory-Hu با caching برای کارایی
    
    فرمول‌های کلیدی (از سند v1.1):
    - ظرفیت مؤثر: c_e^eff = c_e * (1 - U_e)
    - λ_min = min(w_GH) روی درخت Gomory-Hu
    - λ_max = Σ c_e^eff
    - CriticalEdges = یال‌های T_GH با وزن = λ_min
    """
    
    def __init__(self, cache_interval=5):
        """
        cache_interval: هر چند گام، درخت GH بازمحاسبه شود
        (برای کاهش بار محاسباتی)
        """
        self.cache_interval = cache_interval
        self._cached_gh_tree = None
        self._cached_lambda_min = None
        self._cached_lambda_max = None
        self._cached_critical_edges = None
        self._last_update_step = -1
    
    def compute(self, graph, step=0, force=False):
        """
        محاسبه درخت Gomory-Hu و استخراج پارامترها
        
        ورودی:
            graph: NetworkX graph با ویژگی capacity و utilization
            step: شماره گام فعلی (برای caching)
            force: اگر True، cache نادیده گرفته شود
        """
        # اگر در فاصله cache هستیم و force نیست، از مقادیر قبلی استفاده کن
        if (not force and self._cached_gh_tree is not None 
            and step - self._last_update_step < self.cache_interval):
            return self._get_cached_result()
        
        # === ۱. محاسبه ظرفیت مؤثر ===
        # c_e^eff = c_e * (1 - U_e)
        G_eff = graph.copy()
        for u, v in G_eff.edges():
            cap = G_eff[u][v].get('capacity', 1000.0)
            util = G_eff[u][v].get('utilization', 0.5)
            util = min(max(util, 0.0), 0.99)  # جلوگیری از صفر شدن
            G_eff[u][v]['effective_capacity'] = cap * (1.0 - util)
        
        # === ۲. محاسبه درخت Gomory-Hu ===
        try:
            gh_tree = nx.gomory_hu_tree(G_eff, capacity='effective_capacity')
        except Exception as e:
            print(f"[WARN] Gomory-Hu failed: {e}. Using fallback.")
            # Fallback: یک درخت پوشای کمینه به عنوان تقریب
            gh_tree = nx.minimum_spanning_tree(G_eff, weight='effective_capacity')
            for u, v in gh_tree.edges():
                gh_tree[u][v]['weight'] = G_eff[u][v]['effective_capacity']
        
        # === ۳. محاسبه λ_min و λ_max ===
        if gh_tree.number_of_edges() > 0:
            lambda_min = min(
                gh_tree[u][v].get('weight', 0.0) 
                for u, v in gh_tree.edges()
            )
        else:
            lambda_min = 0.0
        
        lambda_max = sum(
            G_eff[u][v]['effective_capacity'] 
            for u, v in G_eff.edges()
        )
        
        # === ۴. استخراج یال‌های بحرانی ===
        # طبق الگوریتم سند: یال‌هایی که وزن آن‌ها = λ_min است
        critical_edges = []
        for u, v in gh_tree.edges():
            if abs(gh_tree[u][v].get('weight', 0.0) - lambda_min) < 1e-6:
                # این یال، یک برش کمینه است
                critical_edges.append((u, v))
        
        # === ۵. ذخیره در cache ===
        self._cached_gh_tree = gh_tree
        self._cached_lambda_min = lambda_min
        self._cached_lambda_max = lambda_max
        self._cached_critical_edges = critical_edges
        self._last_update_step = step
        
        return self._get_cached_result()
    
    def _get_cached_result(self):
        """بازگرداندن نتیجه cache شده"""
        return {
            'gh_tree': self._cached_gh_tree,
            'lambda_min': self._cached_lambda_min,
            'lambda_max': self._cached_lambda_max,
            'critical_edges': self._cached_critical_edges,
        }
    
    def get_critical_edge_indices(self, edge_list):
        """
        تبدیل یال‌های بحرانی به ایندکس در لیست یال‌های محیط
        
        ورودی:
            edge_list: لیست یال‌های محیط (مثل self.edges در env)
        خروجی:
            set از ایندکس‌ها
        """
        critical_indices = set()
        if self._cached_critical_edges is None:
            return critical_indices
        
        for u, v in self._cached_critical_edges:
            # جستجو در هر دو جهت
            for idx, (e_u, e_v) in enumerate(edge_list):
                if (e_u == u and e_v == v) or (e_u == v and e_v == u):
                    critical_indices.add(idx)
                    break
        
        return critical_indices


# ============================================================================
# تست خودکار
# ============================================================================
if __name__ == "__main__":
    from sndlib_loader import load_network
    import os
    
    base = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    filepath = os.path.join(base, 'data/raw/sndlib/static/abilene.txt')
    
    G = load_network(filepath)
    analyzer = GomoryHuAnalyzer()
    result = analyzer.compute(G, step=0, force=True)
    
    print(f"\n[RESULT]")
    print(f"  λ_min = {result['lambda_min']:.2f}")
    print(f"  λ_max = {result['lambda_max']:.2f}")
    print(f"  Critical Edges: {result['critical_edges']}")