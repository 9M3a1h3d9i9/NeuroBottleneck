#!/bin/bash
# ============================================================
# Master overnight script - Goal 1 (Ablation) + Goal 2 (DRL)
# 3 seeds × 50K steps × (4 architectures + 3 algorithms)
# Estimated: ~5.5 hours
# ============================================================

set -e
cd "$(dirname "$0")"

LOG_DIR="outputs/overnight_$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

echo "===========================================" | tee -a "$MASTER_LOG"
echo "OVERNIGHT RUN STARTED: $(date)" | tee -a "$MASTER_LOG"
echo "Log directory: $LOG_DIR" | tee -a "$MASTER_LOG"
echo "===========================================" | tee -a "$MASTER_LOG"

# -------------------------------
# PHASE 1: Goal 1 - Ablation (4 architectures × 3 seeds)
# -------------------------------
echo -e "\n[PHASE 1] Goal 1 Ablation - $(date)" | tee -a "$MASTER_LOG"

for seed in 42 123 456; do
    echo ">>> Goal 1 | Seed $seed | $(date)" | tee -a "$MASTER_LOG"
    
    NB_TOPOLOGY=abilene \
    NB_SEED=$seed \
    NB_TIMESTEPS=50000 \
    NB_VERSION=v5_bugfixed \
    python scripts/_run_goal1.py 2>&1 | tee -a "$LOG_DIR/goal1_seed${seed}.log" \
        || echo "⚠️ Goal1 Seed $seed FAILED" | tee -a "$MASTER_LOG"
    
    echo "<<< Goal 1 Seed $seed finished at $(date)" | tee -a "$MASTER_LOG"
done

# -------------------------------
# PHASE 2: Goal 2 - DRL Comparison (3 algorithms × 3 seeds)
# -------------------------------
echo -e "\n[PHASE 2] Goal 2 DRL Comparison - $(date)" | tee -a "$MASTER_LOG"

for seed in 42 123 456; do
    echo ">>> Goal 2 | Seed $seed | $(date)" | tee -a "$MASTER_LOG"
    
    NB_TOPOLOGY=abilene \
    NB_SEED=$seed \
    NB_TIMESTEPS=50000 \
    python scripts/_run_goal2_all.py 2>&1 | tee -a "$LOG_DIR/goal2_seed${seed}.log" \
        || echo "⚠️ Goal2 Seed $seed FAILED" | tee -a "$MASTER_LOG"
    
    echo "<<< Goal 2 Seed $seed finished at $(date)" | tee -a "$MASTER_LOG"
done

# -------------------------------
# Summary
# -------------------------------
echo -e "\n===========================================" | tee -a "$MASTER_LOG"
echo "ALL RUNS COMPLETED: $(date)" | tee -a "$MASTER_LOG"
echo "===========================================" | tee -a "$MASTER_LOG"

# Aggregate all summaries
echo -e "\n[AGGREGATING RESULTS]" | tee -a "$MASTER_LOG"
python - <<'PY' 2>&1 | tee -a "$MASTER_LOG"
import glob, pandas as pd, os
print("\n=== Goal 1 Summaries (3 seeds) ===")
g1 = glob.glob("outputs/goal1_abilene_seed*_v5_bugfixed/summary.csv")
if g1:
    df = pd.concat([pd.read_csv(f) for f in g1], ignore_index=True)
    print(df.groupby("experiment")[["mean_reward","mean_violations","mean_lambda_min"]].agg(["mean","std"]).round(3))
print("\n=== Goal 2 Summaries (3 seeds) ===")
g2 = glob.glob("outputs/goal2_abilene_seed*_50K/summary.csv")
if g2:
    df = pd.concat([pd.read_csv(f) for f in g2], ignore_index=True)
    print(df.groupby("experiment")[["mean_reward","mean_violations","mean_lambda_min"]].agg(["mean","std"]).round(3))
PY

echo "DONE. All logs in: $LOG_DIR" | tee -a "$MASTER_LOG"