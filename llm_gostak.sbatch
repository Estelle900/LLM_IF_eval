#!/bin/bash
#SBATCH --job-name=gostak-qwen32b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gostak-%j.out
#SBATCH --error=logsgostak-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .llm_eval/bin/activate

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Runs both the local LLM and TextWorld.
# Memory toggle remains disabled.
python -u agent.py

