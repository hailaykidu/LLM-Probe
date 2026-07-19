import sys
import os
import json
import re
import pandas as pd

# Add project root to Python path
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
from models.apertus_loader import load_model as load_apertus_8b  # ✅ NEW

# === Helpers ===
EXTRA_ID_RE = re.compile(r"<extra_id_\d+>")
OUTPUT_FORMAT_ECHO_RE = re.compile(r"(?i)output\s*format.*")

def strip_special_tokens(text: str) -> str:
    """Remove placeholder tokens and echoed instructions."""
    text = EXTRA_ID_RE.sub("", text)
    text = OUTPUT_FORMAT_ECHO_RE.sub("", text)
    return " ".join(text.split()).strip()

def extract_text(response) -> str:
    """Robustly extract generated text from pipeline responses."""
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

def normalize_features(tag_string: str) -> set:
    """Normalize comma-separated morphosyntactic features into a set of lowercase tokens."""
    return set(t.strip().lower() for t in tag_string.replace(" ", "").split(",") if t)

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
prompt_template = open("data/prompts/morphosyntax_probe.txt", encoding="utf-8").read()
probe_df = pd.read_json("data/lexicon.json").rename(columns={"english": "English"})
gold_df = pd.read_json("data/gold_labels/morpho_features_fixed.json")
df = pd.merge(probe_df, gold_df, on="English", how="inner").dropna(subset=["English", "Tigrigna", "Expected"])

# === Evaluation loop ===
task_name = "morphosyntax_probe"
task_dir = os.path.join("results", "evaluation_reports", task_name)
os.makedirs(task_dir, exist_ok=True)

for model_name, model in models.items():
    print(f"\n🔍 {task_name.replace('_',' ').title()} with {model_name}")
    result_path = os.path.join(task_dir, f"{task_name}_{model_name}.json")
    accuracy_path = os.path.join(task_dir, f"{task_name}_accuracy_{model_name}.txt")

    completed_keys = set()
    if os.path.exists(result_path):
        try:
            previous = json.load(open(result_path, encoding="utf-8"))
            completed_keys = {(r["English"], r["Tigrigna"]) for r in previous}
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

        expected_features = normalize_features(row["Expected"])
        output_features = normalize_features(cleaned_output)

        match = bool(expected_features & output_features)

        new_results.append({
            "Task": task_name,
            "Model": tagged["model"],
            "English": row["English"],
            "Tigrigna": row["Tigrigna"],
            "ExpectedTags": row["Expected"],
            "ModelOutputRaw": raw_output,
            "ModelOutput": cleaned_output,
            "Match": match
        })

    all_results = previous + new_results
    log_result(all_results, result_path)

    acc = sum(r["Match"] for r in all_results) / len(all_results) if all_results else 0.0
    with open(accuracy_path, "w", encoding="utf-8") as f:
        f.write(f"{acc:.4f}")

    print(f"✅ Accuracy: {acc:.2%}")
    if new_results:
        print("📌 Sample:", {
            "English": new_results[0]["English"],
            "ExpectedTags": new_results[0]["ExpectedTags"],
            "ModelOutputRaw": (new_results[0]["ModelOutputRaw"][:120] + "...") if new_results[0]["ModelOutputRaw"] else "",
            "ModelOutput": new_results[0]["ModelOutput"],
            "Match": new_results[0]["Match"],
        })
