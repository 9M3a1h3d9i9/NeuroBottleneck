#!/bin/bash
# هدف ۱: اثبات نوآوری Dual-Stream در برابر Baseline
set -e
cd "$(dirname "$0")/.."

export NB_TOPOLOGY=${1:-abilene}
export NB_SEED=${2:-42}
export NB_TIMESTEPS=${3:-10000}
export NB_VERSION=${4:-v1}   # ← نسخه جدید


python scripts/_run_goal1.py
