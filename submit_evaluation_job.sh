#!/bin/bash
#SBATCH --job-name=llm_eval_tigrinya
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --partition=ampere             # ✅ Use the ampere GPU partition
#SBATCH --gres=gpu:4                   # Request 4 GPUs for parallel model loading
#SBATCH --cpus-per-task=32             # Use 32 CPU cores for faster preprocessing
#SBATCH --mem=128G                     # Allocate 128 GB RAM for large model support
#SBATCH --time=72:00:00                # Set time limit to 72 hours
#SBATCH --hint=nomultithread           # Optional: disable hyperthreading

# ✅ Prevent CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True  # Updated from deprecated PYTORCH_CUDA_ALLOC_CONF

# Activate your virtual environment
# NOTE: the project moved to Project/LLM-Probe; the old venv path
# (.../TigrinyaTokenizer/MPETokenization/Paralleldata/MoVoC/LLM_Probe/env)
# no longer exists on this filesystem. Point this at a real venv built from
# requirements.txt in the current project root before submitting.
source /homes/neumann/teklehaymanot/Project/LLM-Probe/env/bin/activate

# Navigate to project root
cd /homes/neumann/teklehaymanot/Project/LLM-Probe

# Optional: check if required files exist before running
if [ ! -f "scripts/run_all_evaluations.sh" ]; then
  echo "❌ Missing run_all_evaluations.sh script. Aborting."
  exit 1
fi

# Run all evaluation scripts
./scripts/run_all_evaluations.sh
