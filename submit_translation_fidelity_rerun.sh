#!/bin/bash
#SBATCH --job-name=translation_fidelity_rerun
#SBATCH --output=logs/eval_tf_rerun_%j.out
#SBATCH --error=logs/eval_tf_rerun_%j.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --hint=nomultithread

# Scoped rerun of just translation_fidelity after the per-word match fix in
# tasks/translation_fidelity.py (old results backed up to
# results/evaluation_reports/translation_fidelity_pre_wordmatch_fix_backup_2026-09-21/).
# Reuses submit_evaluation_job.sh's env/resource setup but skips the other
# three tasks, which already have current results from job 75167.

export PYTORCH_ALLOC_CONF=expandable_segments:True

source /homes/neumann/teklehaymanot/Project/LLM-Probe/env/bin/activate

cd /homes/neumann/teklehaymanot/Project/LLM-Probe

python tasks/translation_fidelity.py

echo "📊 Merging all evaluation reports into one JSON..."
python << 'EOF'
import json, glob, os

base_dir = "results/evaluation_reports"
tasks = ["lexical_alignment", "pos_tagging", "morphosyntax_probe", "translation_fidelity"]

combined = {}

for task in tasks:
    task_dir = os.path.join(base_dir, task)
    task_files = glob.glob(os.path.join(task_dir, "*.json"))
    task_results = []
    for f in task_files:
        try:
            with open(f, encoding="utf-8") as infile:
                task_results.extend(json.load(infile))
        except Exception:
            continue
    combined[task] = task_results

out_path = os.path.join(base_dir, "all_results.json")
with open(out_path, "w", encoding="utf-8") as out:
    json.dump(combined, out, ensure_ascii=False, indent=2)

print(f"✅ Consolidated results written to {out_path}")
EOF
