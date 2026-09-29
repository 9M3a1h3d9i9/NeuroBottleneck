#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export NB_TOPOLOGY=${1:-abilene}
export NB_TIMESTEPS=${2:-10000}
export NB_SEEDS=${3:-"42 123"}
python scripts/_run_goal4.py
