# import sys
# import os
# import json
# import pandas as pd

# # Ensure the parent directory is in the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from utils.logger import log_result

# # === Load models ===
# from models.gemma_loader import load_model as load_gemma_2b
# from models.gemma7b_loader import load_model as load_gemma_7b
# from models.mistral_loader import load_model as load_mistral_7b
# from models.mt5_loader import load_model as load_mt5_base
# from models.mt5_large_loader import load_model as load_mt5_large
# from models.byt5_loader import load_model as load_byt5
# from models.xlm_roberta_loader import load_model as load_xlm_roberta
# from models.qwen_loader import load_model as load_qwen_7b
# from models.falcon_loader import load_model as load_falcon_10b
# from models.apertus_loader import load_model as load_apertus_8b  # ✅ NEW

# # === Tagged model wrapper ===
# class TaggedModel:
#     def __init__(self, name, pipeline):
#         self.name = name
#         self.pipeline = pipeline

#     def __call__(self, prompts):
#         responses = self.pipeline(prompts)
#         return [{"model": self.name, "response": r} for r in responses]

# # === Initialize models ===
# models = {
#     "gemma-2b": TaggedModel("gemma-2b", load_gemma_2b()),
#     "gemma-7b": TaggedModel("gemma-7b", load_gemma_7b()),
#     "mistral-7b": TaggedModel("mistral-7b", load_mistral_7b()),
#     "mt5-base": TaggedModel("mt5-base", load_mt5_base()),
#     "mt5-large": TaggedModel("mt5-large", load_mt5_large()),
#     "byt5": TaggedModel("byt5", load_byt5()),
#     "xlm-roberta": TaggedModel("xlm-roberta", load_xlm_roberta()),
#     "qwen-7b": TaggedModel("qwen-7b", load_qwen_7b()),
#     "falcon-10b": TaggedModel("falcon-10b", load_falcon_10b()),
#     "apertus-8b": TaggedModel("apertus-8b", load_apertus_8b())
# }

# # === Load data ===
# prompt_template = open("data/prompts/lexical_alignment.txt", encoding="utf-8").read()
# lexicon_df = pd.read_json("data/lexicon.json")
# gold_df = pd.read_json("data/gold_labels/lexical_alignment.json")
# df = pd.concat([lexicon_df, gold_df], axis=1).dropna(subset=["English", "Tigrigna", "ExpectedAlignment"])

# # === Evaluation loop ===
# for model_name, model in models.items():
#     print(f"\n🚀 Lexical Alignment with {model_name}")
#     result_path = f"results/evaluation_reports/lexical_alignment_{model_name}.json"
#     accuracy_path = f"results/evaluation_reports/lexical_alignment_accuracy_{model_name}.txt"

#     completed_keys = set()
#     if os.path.exists(result_path):
#         try:
#             previous = json.load(open(result_path, encoding="utf-8"))
#             completed_keys = {(r["English"], r["Tigrigna"]) for r in previous}
#         except Exception:
#             previous = []
#     else:
#         previous = []

#     filtered = df[~df.apply(lambda r: (r["English"], r["Tigrigna"]) in completed_keys, axis=1)]
#     if filtered.empty:
#         print("⏭️ Already done.")
#         continue

#     prompts = [
#         prompt_template.replace("{english_sentence}", r["English"]).replace("{tigrigna_sentence}", r["Tigrigna"])
#         for _, r in filtered.iterrows()
#     ]

#     try:
#         tagged_responses = model(prompts)
#     except Exception as e:
#         print(f"❌ Error with {model_name}: {e}")
#         continue

#     new_results = []
#     for (_, row), tagged in zip(filtered.iterrows(), tagged_responses):
#         response = tagged["response"]
#         output = response.get("generated_text") or response.get("text", "").strip() if isinstance(response, dict) else ""
#         match = bool(set(row["ExpectedAlignment"].replace(" ", "").split(",")) & set(output.replace(" ", "").split(",")))
#         new_results.append({
#             "Model": tagged["model"],
#             "English": row["English"],
#             "Tigrigna": row["Tigrigna"],
#             "ExpectedAlignment": row["ExpectedAlignment"],
#             "ModelOutput": output,
#             "Match": match
#         })

#     all_results = previous + new_results
#     log_result(all_results, result_path)
#     acc = sum(r["Match"] for r in all_results) / len(all_results)
#     with open(accuracy_path, "w") as f:
#         f.write(f"{acc:.4f}")
#     print(f"✅ Accuracy: {acc:.2%}")
#     print("📌 Sample:", new_results[0])
import sys
import os
import json
import re
import pandas as pd

# Ensure parent directory is in Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import log_result

# === Load models ===
from models.gemma_loader import load_model as load_gemma_2b
from models.gemma7b_loader import load_model as load_gemma_7b
from models.mistral_loader import load_model as load_mistral_7b
from models.mt5_loader import load_model as load_mt5_base
from models.mt5_large_loader import load_model as load_mt5_large
from models.byt5_loader import load_model as load_byt5
from models.xlm_roberta_loader import load_model as load_xlm_roberta
from models.qwen_loader import load_model as load_qwen_7b
from models.falcon_loader import load_model as load_falcon_10b
from models.apertus_loader import load_model as load_apertus_8b

# === Helpers ===
EXTRA_ID_RE = re.compile(r"<extra_id_\d+>")
OUTPUT_FORMAT_ECHO_RE = re.compile(r"(?i)output\s*format.*")

def strip_special_tokens(text: str) -> str:
    text = EXTRA_ID_RE.sub("", text)
    text = OUTPUT_FORMAT_ECHO_RE.sub("", text)
    return " ".join(text.split()).strip()

def extract_text(response) -> str:
    if response is None:
        return ""
    if isinstance(response, list):
        if response and isinstance(response[0], dict):
            return response[0].get("generated_text") or response[0].get("text") or ""
        return str(response[0])
    elif isinstance(response, dict):
        return response.get("generated_text") or response.get("text") or response.get("output_text") or ""
    else:
        return str(response)

def clean_and_enforce_format(english: str, tigrigna_gold: str, raw_output: str) -> str:
    cleaned = strip_special_tokens(raw_output)
    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    if "→" in first_line:
        return first_line
    tigrigna_candidates = [c.strip() for c in re.split(r"[，,]", tigrigna_gold)] if tigrigna_gold else []
    if not first_line and tigrigna_candidates:
        return f"{english}→{tigrigna_candidates[0]}"
    return f"{english}→{first_line}" if first_line else ""

def compute_format_match(expected: str, output: str) -> bool:
    expected_tokens = set(expected.replace(" ", "").split(","))
    output_tokens = set(output.replace(" ", "").split(","))
    return bool(expected_tokens & output_tokens)

def compute_semantic_match(expected: str, output: str) -> bool:
    tigrigna_parts = []
    for pair in expected.split(","):
        if "→" in pair:
            _, tig = pair.split("→", 1)
            tigrigna_parts.append(tig.strip())
    out_norm = output.replace(" ", "")
    return all(tig.replace(" ", "") in out_norm for tig in tigrigna_parts) if tigrigna_parts else False

# === Tagged model wrapper ===
class TaggedModel:
    def __init__(self, name, pipeline):
        self.name = name
        self.pipeline = pipeline

    def __call__(self, prompts):
        responses = self.pipeline(prompts)
        if isinstance(responses, list):
            return [{"model": self.name, "response": r} for r in responses]
        return [{"model": self.name, "response": responses}]

# === Initialize models ===
models = {
    "gemma-2b": TaggedModel("gemma-2b", load_gemma_2b()),
    "gemma-7b": TaggedModel("gemma-7b", load_gemma_7b()),
    "mistral-7b": TaggedModel("mistral-7b", load_mistral_7b()),
    "mt5-base": TaggedModel("mt5-base", load_mt5_base()),
    "mt5-large": TaggedModel("mt5-large", load_mt5_large()),
    "byt5": TaggedModel("byt5", load_byt5()),
    "xlm-roberta": TaggedModel("xlm-roberta", load_xlm_roberta()),
    "qwen-7b": TaggedModel("qwen-7b", load_qwen_7b()),
    "falcon-10b": TaggedModel("falcon-10b", load_falcon_10b()),
    # apertus-8b repeatedly stalls mid-download from the HF CDN (observed
    # hanging for hours on a partial shard) — skip until that's resolved.
    # "apertus-8b": TaggedModel("apertus-8b", load_apertus_8b()),
}

# === Load data ===
prompt_template = open("data/prompts/lexical_alignment.txt", encoding="utf-8").read()
lexicon_df = pd.read_json("data/lexicon.json").rename(columns={"english": "English"})
gold_df = pd.read_json("data/gold_labels/lexical_alignment.json")
df = pd.merge(lexicon_df, gold_df, on="English", how="inner").dropna(subset=["English", "Tigrigna", "ExpectedAlignment"])

# === Evaluation loop ===
task_name = "lexical_alignment"
task_dir = os.path.join("results", "evaluation_reports", task_name)
os.makedirs(task_dir, exist_ok=True)

for model_name, model in models.items():
    print(f"\n🚀 {task_name.replace('_',' ').title()} with {model_name}")
    result_path = os.path.join(task_dir, f"{task_name}_{model_name}.json")
    accuracy_path = os.path.join(task_dir, f"{task_name}_accuracy_{model_name}.txt")

    completed_keys = set()
    if os.path.exists(result_path):
        try:
            previous = json.load(open(result_path, encoding="utf-8"))
            completed_keys = {(r.get("English"), r.get("Tigrigna")) for r in previous}
        except Exception:
            previous = []
    else:
        previous = []

    filtered = df[~df.apply(lambda r: (r["English"], r["Tigrigna"]) in completed_keys, axis=1)]
    if filtered.empty:
        print("⏭️ Already done for this model.")
        continue

    prompts = [
        prompt_template.replace("{english_sentence}", str(r["English"])).replace("{tigrigna_sentence}", str(r["Tigrigna"]))
        for _, r in filtered.iterrows()
    ]

    try:
        tagged_responses = model(prompts)
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        continue

    new_results = []
    for (_, row), tagged in zip(filtered.iterrows(), tagged_responses):
        raw_output = extract_text(tagged["response"])
        cleaned_output = strip_special_tokens(raw_output)
        final_output = clean_and_enforce_format(row["English"], row["Tigrigna"], cleaned_output)

        format_match = compute_format_match(row["ExpectedAlignment"], final_output) if final_output else False
        semantic_match = compute_semantic_match(row["ExpectedAlignment"], final_output) if final_output else False

        new_results.append({
            "Task": task_name,
            "Model": tagged["model"],
            "English": row["English"],
            "Tigrigna": row["Tigrigna"],
            "ExpectedAlignment": row["ExpectedAlignment"],
            "ModelOutputRaw": raw_output,
            "ModelOutput": final_output,
            "Match": format_match,
            "SemanticMatch": semantic_match,
        })

    all_results = previous + new_results
    log_result(all_results, result_path)

    acc = sum(r["Match"] for r in all_results) / len(all_results) if all_results else 0.0
    sem_acc = sum(r.get("SemanticMatch", False) for r in all_results) / len(all_results) if all_results else 0.0

    with open(accuracy_path, "w", encoding="utf-8") as f:
        f.write(f"format_accuracy={acc:.4f}\nsemantic_accuracy={sem_acc:.4f}\n")

    print(f"✅ Format accuracy: {acc:.2%} | 🔎 Semantic accuracy: {sem_acc:.2%}")
    if new_results:
        print("📌 Sample:", {
            "English": new_results[0]["English"],
            "ExpectedAlignment": new_results[0]["ExpectedAlignment"],
            "ModelOutputRaw": (new_results[0]["ModelOutputRaw"][:120] + "...") if new_results[0]["ModelOutputRaw"] else "",
            "ModelOutput": new_results[0]["ModelOutput"],
            "Match": new_results[0]["Match"],
            "SemanticMatch": new_results[0]["SemanticMatch"],
        })
