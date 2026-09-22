import sys
import os
import json
import re
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import log_result
from models.span_infilling import to_span_infilling_prompts

# === Load models ===
from models.gemma_loader import load_model as load_gemma_2b
from models.gemma7b_loader import load_model as load_gemma_7b
from models.mistral_loader import load_model as load_mistral_7b
from models.mt5_loader import load_model as load_mt5_small
from models.mt5_large_loader import load_model as load_mt5_large
from models.byt5_loader import load_model as load_byt5
from models.xlm_roberta_loader import load_model as load_xlm_roberta
from models.qwen_loader import load_model as load_qwen_7b
from models.falcon_loader import load_model as load_falcon_7b
from models.apertus_loader import load_model as load_apertus_8b  # ✅ NEW

# === Helpers ===
EXTRA_ID_RE = re.compile(r"<extra_id_\d+>")
OUTPUT_FORMAT_ECHO_RE = re.compile(r"(?i)output\s*format.*")
# Models routinely echo the prompt's few-shot "Output:" label back as a
# prefix to their own answer (e.g. "Output: verb", "**Output:** noun").
# Left unstripped, this polluted every downstream comparison against gold
# labels that never contain that prefix.
OUTPUT_PREFIX_RE = re.compile(r"(?i)^\s*[\*_]*\s*(output|answer)\s*[\*_]*\s*:\s*")

def strip_special_tokens(text: str) -> str:
    """Remove placeholder tokens and echoed instructions."""
    text = EXTRA_ID_RE.sub("", text)
    text = OUTPUT_FORMAT_ECHO_RE.sub("", text)
    text = OUTPUT_PREFIX_RE.sub("", text)
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

PAREN_ABBREV_RE = re.compile(r"\([^)]*\)")

def normalize_features(tag_string: str) -> set:
    """Normalize gold/model feature strings into a set of lowercase word tokens.

    Gold labels are formatted like "(adv) adverb" or "(nm) noun masculine".
    The previous implementation stripped spaces *before* splitting on commas,
    which glued the parenthesized abbreviation onto the following word(s)
    into a single token (e.g. "(adv)adverb") that no plausible model output
    could ever match -- silently forcing every row to Match=False regardless
    of whether the model's answer was correct. This strips the parenthetical
    abbreviation and splits on both commas and whitespace so tokens like
    "adverb" or "noun" from either side can be compared directly.
    """
    tag_string = PAREN_ABBREV_RE.sub(" ", tag_string)
    tokens = re.split(r"[,\s]+", tag_string.strip().lower())
    return set(t for t in tokens if t)

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
# Each model is loaded independently, with failures caught and logged rather
# than fatal -- previously one model's load failure (e.g. xlm-roberta's
# known issues) raised an exception that killed the entire module before
# the evaluation loop ever ran, silently discarding every other model's
# results for the whole task.
_MODEL_LOADERS = [
    ("gemma-2b", load_gemma_2b),
    ("gemma-7b", load_gemma_7b),
    ("mistral-7b", load_mistral_7b),
    # google/mt5-small is what this loader actually loads, despite the
    # earlier "mt5-base" naming; relabeled to match the checkpoint used.
    ("mt5-small", load_mt5_small),
    ("mt5-large", load_mt5_large),
    ("byt5", load_byt5),
    ("xlm-roberta", load_xlm_roberta),
    ("qwen-7b", load_qwen_7b),
    # tiiuae/falcon-7b-instruct -- the Falcon model this project evaluates.
    # Published as "Falcon-10B"; that label is incorrect.
    ("falcon-7b", load_falcon_7b),
    # apertus-8b repeatedly stalls mid-download from the HF CDN (observed
    # hanging for hours on a partial shard) — skip until that's resolved.
    # ("apertus-8b", load_apertus_8b),
]

# Optional scoping for follow-up/re-runs of just one or a few models (e.g.
# after fixing a loader bug for a model that was skipped in the main run)
# without re-running every other model's already-completed work. Unset by
# default, so normal full-sweep invocations are unaffected.
_only = os.environ.get("EVAL_ONLY_MODELS")
if _only:
    _wanted = {m.strip() for m in _only.split(",") if m.strip()}
    _MODEL_LOADERS = [(n, l) for n, l in _MODEL_LOADERS if n in _wanted]

models = {}
for _name, _loader in _MODEL_LOADERS:
    try:
        models[_name] = TaggedModel(_name, _loader())
    except Exception as e:
        print(f"❌ Skipping {_name}: failed to load ({e})")

# === Load data ===
prompt_template = open("data/prompts/morphosyntax_probe.txt", encoding="utf-8").read()
# gold_labels/morpho_features_fixed.json is now the authoritative,
# already-corrected source (regenerated directly from the fixed
# Combined_POS_Lexicon.csv, with the English/Tigrigna column swap and
# duplicate rows already resolved, and "Expected" now genuinely encoding
# gender/number features rather than duplicating the POS tag) -- load it
# directly instead of re-merging against lexicon.json, which avoids
# reintroducing a fragile English-only join.
gold_df = pd.read_json("data/gold_labels/morpho_features_fixed.json")
# Morphosyntactic features (gender, number, agreement) are a property of a
# single word/phrase, not a list. Some rows still give multiple
# comma-separated Tigrinya forms for one English headword, which turns the
# prompt's "Phrase:" into several unrelated words at once and asks the model
# for one feature set covering all of them. Use only the first form.
gold_df["Tigrigna"] = gold_df["Tigrigna"].apply(lambda t: str(t).split(",")[0].strip())
# Dedup on (English, Tigrigna) rather than English alone -- the lexicon
# contains real distinct senses of the same headword, and deduping on
# English alone would silently discard those senses instead of preserving
# them as separate evaluation items.
df = gold_df.drop_duplicates(subset=["English", "Tigrigna"], keep="first").dropna(subset=["English", "Tigrigna", "Expected"])

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
    prompts = to_span_infilling_prompts(prompts, model_name)

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
