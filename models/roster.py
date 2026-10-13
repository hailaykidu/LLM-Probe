"""Which model labels a task run uses.

The eight models the paper reports were later relabelled in the task scripts
to a 2026 roster, against the same loaders and so the same checkpoints. A
full run therefore wrote results under the new names and the published
tables could no longer be reproduced by the documented command.

`EVAL_ROSTER` selects the label set:

  paper    (default) the eight labels of the paper's Table 2
  2026     the newer roster, including the endpoint-served models
  all      every entry the task script defines

Checkpoints are unchanged either way; only the label a result file is
written under differs. `EVAL_ONLY_MODELS` still narrows whatever this
returns, so a scoped rerun of one model works as before.
"""
import os

# Paper label -> the 2026 label attached to the same loader. The mapping is
# what makes the default reproducible: a task script lists the 2026 labels,
# and this renames them back.
PAPER_LABELS = {
    "gemma-4-e2b": "gemma-2b",
    "gemma-4-12b": "gemma-7b",
    "ministral-3-8b": "mistral-7b",
    "t5gemma-2-270m": "mt5-small",
    "t5gemma-2-1b": "mt5-large",
    "byt5": "byt5",
    "qwen3.6-27b": "qwen-7b",
    "falcon-h1r-7b": "falcon-7b",
}

# The paper's roster in table order, so a run's console output matches it.
PAPER_ORDER = [
    "gemma-2b", "gemma-7b", "mistral-7b", "qwen-7b", "falcon-7b",
    "mt5-small", "mt5-large", "byt5",
]


def select(model_loaders):
    """Filter and relabel a task script's (label, loader) list.

    Returns the pairs the current EVAL_ROSTER and EVAL_ONLY_MODELS select,
    with the paper's labels substituted under the default roster.
    """
    roster = os.environ.get("EVAL_ROSTER", "paper").strip().lower()

    if roster == "paper":
        renamed = [
            (PAPER_LABELS[name], loader)
            for name, loader in model_loaders
            if name in PAPER_LABELS
        ]
        order = {label: i for i, label in enumerate(PAPER_ORDER)}
        pairs = sorted(renamed, key=lambda p: order.get(p[0], len(order)))
    elif roster in ("2026", "all"):
        pairs = list(model_loaders)
    else:
        raise SystemExit(
            f"EVAL_ROSTER={roster!r} is not one of: paper, 2026, all"
        )

    only = os.environ.get("EVAL_ONLY_MODELS")
    if only:
        wanted = {m.strip() for m in only.split(",") if m.strip()}
        pairs = [(n, l) for n, l in pairs if n in wanted]

    return pairs
