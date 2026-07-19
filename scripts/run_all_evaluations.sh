#!/bin/bash

# ================================
# Run All Evaluations for Tigrinya Lexicon
# ================================

echo "🔁 Starting LLM Evaluation for Tigrinya Lexicon..."

# --- Lexical Alignment ---
echo "🔍 Lexical Alignment"
python tasks/lexical_alignment.py

# --- POS Tagging ---
echo "🧠 POS Tagging"
python tasks/pos_tagging.py

# --- Morphosyntactic Probing ---
echo "🧬 Morphosyntactic Probing"
python tasks/morphosyntax_probe.py

# --- Translation Fidelity ---
echo "🌍 Translation Fidelity"
python tasks/translation_fidelity.py

echo "✅ All evaluations completed."

# --- Merge Results ---
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
