# تحلیل گلوگاه‌های شبکه India35 با استفاده از درخت گوموری-هو

import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------
# تابع بارگذاری فایل SNDlib (فرمت استاندارد)
# --------------------------------------------------------------------
def load_sndlib_topology(filepath):
    """
    پارس کردن فایل‌های SNDlib با فرمت:
    خط اول: تعداد گره‌ها و تعداد یال‌ها
    سپس به ازای هر یال: u v capacity
    سپس به ازای هر تقاضا: u v demand
    """
    G = nx.Graph()
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # حذف خطوط خالی و کامنت‌ها (خطوطی که با # شروع می‌شوند)
    lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
    
    # خط اول: تعداد گره‌ها و یال‌ها
    parts = lines[0].split()
    if len(parts) != 2:
        # برخی فرمت‌ها ممکن است با "nodes" و "edges" شروع شوند
        for l in lines:
            if 'nodes' in l.lower() and 'edges' in l.lower():
                # پیدا کردن اعداد در خط
                import re
                nums = re.findall(r'\d+', l)
                if len(nums) >= 2:
                    n_nodes = int(nums[0])
                    n_edges = int(nums[1])
                    break
        else:
            raise ValueError("Could not parse SNDlib file format.")
    else:
        n_nodes = int(parts[0])
        n_edges = int(parts[1])
    
    # افزودن گره‌ها
    G.add_nodes_from(range(n_nodes))
    
    # پارس کردن یال‌ها
    edge_lines = lines[1:1+n_edges]
    for line in edge_lines:
        parts = line.split()
        if len(parts) >= 3:
            u = int(parts[0])
            v = int(parts[1])
            cap = float(parts[2])
            G.add_edge(u, v, capacity=cap)
    
    print(f"[LOAD] India35 loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

# --------------------------------------------------------------------
# تابع اصلی
# --------------------------------------------------------------------
def main():
    # مسیر فایل India35 در ریپازیتوری شما
    base_dir = os.path.join(os.path.dirname(__file__), '../../')
    filepath = os.path.join(base_dir, 'data/raw/sndlib/india35.txt')
    
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        print("[INFO] Please check the path or download India35 from SNDlib.")
        return
    
    # بارگذاری گراف
    G = load_sndlib_topology(filepath)
    
    # -------------------------------------------------
    # ۱. محاسبه درخت گوموری-هو
    # -------------------------------------------------
    print("[INFO] Computing Gomory-Hu tree...")
    try:
        gh_tree = nx.gomory_hu_tree(G, capacity='capacity')
    except Exception as e:
        print(f"[ERROR] Gomory-Hu computation failed: {e}")
        print("[INFO] Trying with undirected graph...")
        # بعضی گراف‌ها جهت‌دار هستند. اگر خطا داد، جهت‌دار را به بدون‌جهت تبدیل می‌کنیم.
        G_und = G.to_undirected()
        gh_tree = nx.gomory_hu_tree(G_und, capacity='capacity')
    
    # استخراج ظرفیت برش کمینه برای هر یال درخت
    edge_cuts = []
    for u, v in gh_tree.edges():
        cap = gh_tree[u][v]['weight']
        edge_cuts.append((u, v, cap))
    
    # مرتب‌سازی بر اساس ظرفیت (صعودی)
    edge_cuts.sort(key=lambda x: x[2])
    
    # -------------------------------------------------
    # ۲. شناسایی گلوگاه‌های بحرانی (۲۰٪ پایین‌ترین ظرفیت‌ها)
    # -------------------------------------------------
    threshold_idx = max(1, int(len(edge_cuts) * 0.2))
    critical_edges = edge_cuts[:threshold_idx]
    
    print("\n" + "="*50)
    print("[RESULTS] Critical Bottlenecks in India35 (Lowest Min-Cut Capacities)")
    print("="*50)
    print(f"Total edges in GH tree: {len(edge_cuts)}")
    print(f"Showing top {threshold_idx} critical edges (lowest capacity):\n")
    
    for u, v, cap in critical_edges:
        print(f"Edge ({u}, {v}) -> Min-Cut Capacity: {cap:.2f} Mbps")
    
    # -------------------------------------------------
    # ۳. رسم گراف و مشخص کردن گلوگاه‌ها
    # -------------------------------------------------
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    
    # رسم کل گراف
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=500, font_size=10)
    
    # رسم یال‌های بحرانی به رنگ قرمز
    critical_edge_list = [(u, v) for u, v, _ in critical_edges]
    nx.draw_networkx_edges(G, pos, edgelist=critical_edge_list, 
                          edge_color='red', width=3, label='Critical Bottlenecks')
    
    plt.title("India35 Topology with Critical Bottlenecks (Gomory-Hu Cuts)")
    plt.legend()
    plt.tight_layout()
    
    # ذخیره نمودار
    output_dir = "./outputs/plots/"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "india35_bottlenecks.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n[INFO] Plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()