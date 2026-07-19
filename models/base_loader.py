from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    pipeline
)
from huggingface_hub import login
import torch  # ✅ Correct import for torch.float16

MODEL_CONFIGS = {
    "apertus": {"type": "causal", "task": "text-generation"},
    "falcon": {"type": "causal", "task": "text-generation"},
    "gemma": {"type": "causal", "task": "text-generation"},
    "mistral": {"type": "causal", "task": "text-generation"},
    "qwen": {"type": "causal", "task": "text-generation"},
    "mt5": {"type": "seq2seq", "task": "text2text-generation"},
    "byt5": {"type": "seq2seq", "task": "text2text-generation"},
    "xlm-roberta": {"type": "masked", "task": "fill-mask"}
}

def detect_model_config(model_name: str):
    name = model_name.lower()
    for key, config in MODEL_CONFIGS.items():
        if key in name:
            return config
    return {"type": "causal", "task": "text-generation"}

def load_model(model_name: str, task: str = None, hf_token: str = None):
    if hf_token:
        login(token=hf_token)

    config = detect_model_config(model_name)
    model_type = config["type"]
    task = task or config["task"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=False)

    try:
        if model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.float16
            )
        elif model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.float16
            )
        elif model_type == "masked":
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model '{model_name}': {e}")

    if task == "text-generation":
        # Causal LM pipelines echo the input prompt back in `generated_text`
        # by default, which pollutes downstream parsing/matching — only the
        # newly generated continuation is wanted.
        return pipeline(task=task, model=model, tokenizer=tokenizer, return_full_text=False)

    return pipeline(task=task, model=model, tokenizer=tokenizer)
