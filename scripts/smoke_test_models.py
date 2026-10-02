"""Smoke-test every model in the roster: load, tokenize, generate.

Reports tokens-per-word on one Tigrinya and one Amharic sentence, and a short
generation, so tokenizer regressions and loading failures surface before a
full evaluation run is submitted.
"""
import importlib
import sys
import traceback

sys.path.append(".")

TI = "ኣብ ገዛ ዝነብር ቆልዓ ንእሽቶይ እዩ"
AM = "በቤት ውስጥ የሚኖር ልጅ ትንሽ ነው"

ROSTER = [
    ("gemma-4-e2b", "models.gemma_loader"),
    ("gemma-4-12b", "models.gemma7b_loader"),
    ("ministral-3-8b", "models.mistral_loader"),
    ("falcon-h1r-7b", "models.falcon_loader"),
    ("qwen3.6-27b", "models.qwen_loader"),
    ("t5gemma-2-270m", "models.mt5_loader"),
    ("t5gemma-2-1b", "models.mt5_large_loader"),
    ("byt5", "models.byt5_loader"),
    ("falcon3-10b", "models.falcon3_loader"),
]


def extract_text(response) -> str:
    """Extract the first generated string from a pipeline response.

    The roster mixes two return shapes for a list of prompts:

      * the HF "text-generation" pipeline batches via run_multi, returning
        one entry *per prompt* that is itself a list of generated sequences
        -- list[list[{"generated_text": str}]];
      * Seq2SeqPipeline (models/base_loader.py) and the endpoint loaders
        return a flat list[{"generated_text": str}].

    Unwrapping only the flat shape raises "list indices must be integers or
    slices, not str" on every causal model, which is what made a previous
    smoke run report the entire roster as FAILED. Peel nested lists until a
    dict is reached so both contracts are handled.
    """
    while isinstance(response, list):
        if not response:
            return ""
        response = response[0]
    if isinstance(response, dict):
        return response.get("generated_text") or response.get("text") or ""
    return str(response)


def main():
    from transformers import AutoTokenizer
    from models.span_infilling import to_span_infilling_prompts

    print(f"{'model':<18}{'ti tok/wd':>11}{'am tok/wd':>11}  generation")
    print("-" * 78)
    for alias, module in ROSTER:
        try:
            mod = importlib.import_module(module)
            pipe = mod.load_model()
            tok = getattr(pipe, "tokenizer", None)
            if tok is None:
                src = mod.load_model.__doc__ or ""
                tok = None
            ti = am = float("nan")
            if tok is not None:
                ti = len(tok(TI)["input_ids"]) / len(TI.split())
                am = len(tok(AM)["input_ids"]) / len(AM.split())
            prompts = to_span_infilling_prompts(
                [f"Identify the part of speech.\nPhrase: {TI} (child)\nOutput:"], alias
            )
            out = pipe(prompts)
            text = extract_text(out)
            print(f"{alias:<18}{ti:>11.2f}{am:>11.2f}  {text.strip()[:40]!r}")
            del pipe
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{alias:<18}{'--':>11}{'--':>11}  FAILED {type(e).__name__}: {str(e)[:60]}")
            traceback.print_exc(limit=1)


if __name__ == "__main__":
    main()
