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
    # "text2text-generation" was removed from the transformers pipeline
    # registry in the version installed for this project (5.17.0) -- the
    # pipeline() factory now only supports "text-generation" and that impl
    # is hardcoded to AutoModelForCausalLM, so it cannot serve seq2seq
    # models at all. Seq2seqPipeline below reimplements the same call
    # contract (list-of-prompts in, list of [{"generated_text": ...}] out)
    # directly on top of model.generate(), bypassing pipeline() entirely for
    # these two model families.
    "mt5": {"type": "seq2seq", "task": "text2text-generation"},
    "byt5": {"type": "seq2seq", "task": "text2text-generation"},
    "xlm-roberta": {"type": "masked", "task": "fill-mask"}
}


class Seq2SeqPipeline:
    """Minimal drop-in replacement for the removed "text2text-generation"
    pipeline task. Matches the same call contract every task script relies
    on: __call__(list[str]) -> list[{"generated_text": str}], one entry per
    input prompt, in order."""

    def __init__(self, model, tokenizer, max_new_tokens=128, min_new_tokens=1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens

    def __call__(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        results = []
        device = next(self.model.parameters()).device
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=self.min_new_tokens,
                )
            text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results.append({"generated_text": text})
        return results

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

    # use_fast=True (the default): the slow, pure-Python tokenizers were
    # observed producing mojibake (mixed Ethiopic/Latin garbage such as
    # "ገSitz", "ገΑνα") when decoding Tigrinya (Ge'ez script) output from
    # causal models. The fast (Rust/tokenizers-backed) tokenizers handle
    # multi-byte Unicode decoding far more reliably.
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=True)

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
            # device_map="auto" is avoided here: transformers 5.17.0's
            # infer_auto_device_map() crashes with "IndexError: list index
            # out of range" in get_module_size_with_ties() when it tries to
            # split a model with tied embedding/decoder weights (e.g.
            # xlm-roberta's word_embeddings <-> MLM head) across multiple
            # GPUs -- it can't find which remaining module "owns" the tied
            # param once its sibling has already been placed. Masked-LM
            # models used here (~125M-355M params) fit on a single GPU
            # anyway, so there's no need to shard them; load directly onto
            # one CUDA device (falling back to CPU) to sidestep the bug.
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                dtype=torch.float16
            ).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model '{model_name}': {e}")

    if task == "text-generation":
        # Causal LM pipelines echo the input prompt back in `generated_text`
        # by default, which pollutes downstream parsing/matching — only the
        # newly generated continuation is wanted.
        #
        # Without an explicit max_new_tokens/min_new_tokens, an instruct-tuned
        # model whose very first generated token is EOS produces an empty
        # continuation once return_full_text=False strips the prompt back
        # out -- this silently zeroed out every output for several models
        # (falcon-7b, gemma-2b/7b, mistral-7b, qwen-7b) in an earlier run.
        # Force at least a few tokens and cap generation length explicitly.
        return pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            max_new_tokens=128,
            min_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

    if model_type == "seq2seq":
        # "text2text-generation" no longer exists as a pipeline() task in
        # this transformers version -- use the direct model.generate()
        # wrapper instead (see Seq2SeqPipeline above).
        return Seq2SeqPipeline(model, tokenizer, max_new_tokens=128, min_new_tokens=1)

    return pipeline(task=task, model=model, tokenizer=tokenizer)
