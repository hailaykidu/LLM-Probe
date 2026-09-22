"""Span-infilling prompt adaptation for base (non-instruction-tuned) seq2seq models.

google/mt5-small, mt5-large and byt5 are pretrained only on masked-span
denoising (predict the content behind an <extra_id_N> sentinel), never on
instruction-following. Feeding them the same plain natural-language
instruction prompt used for causal/masked models produces degenerate output
(observed: mt5-small echoing "." for 98.7% of POS-tagging items, scoring
0.0000 accuracy) because there is no sentinel in the input for the model to
infill.

This module is opt-in per call site: task scripts pass a model name and only
substitute the prompt when that name is a known base seq2seq model, leaving
every other model's prompt untouched.
"""

SPAN_INFILLING_MODELS = {"mt5-small", "mt5-large", "byt5"}


def to_span_infilling_prompts(prompts: list[str], model_name: str) -> list[str]:
    """Return prompts reformatted for span-infilling if model_name needs it.

    Appends a trailing "<extra_id_0>" sentinel so the model has an explicit
    span to infill instead of being asked to follow an instruction it was
    never trained to follow. Prompts for every other model are returned
    unchanged.
    """
    if model_name not in SPAN_INFILLING_MODELS:
        return prompts
    return [p.rstrip() + " <extra_id_0>" for p in prompts]
