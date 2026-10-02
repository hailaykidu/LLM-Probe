"""The KBS endpoint model roster, as configured in .claude/Test_f/config.py.

That file lists six models with "vllm/" or "ollama/" routing prefixes. Only
three of them are actually served by the endpoint; the other three return a
404. Verified against GET /v1/models on 2026-09-29:

    vllm/qwen3.8:27b-fp8                        -> served
    vllm/deepseek-v4-flash-0731:284b-a13b-nvfp4 -> served
    vllm/gpt-oss:120b-mxfp4                     -> served
    ollama/gemma3:27b                           -> NOT served
    ollama/qwen3:8b                             -> NOT served
    ollama/embeddinggemma:300m                  -> NOT served (and is an
                                                   embedding model, which
                                                   has no chat endpoint and
                                                   cannot do generation)

The unavailable three are listed in UNAVAILABLE rather than deleted, so the
discrepancy between the configured roster and the served roster stays
visible instead of looking like an omission.

Note also that config.py's list is missing a comma between the gpt-oss and
qwen3 entries, so Python implicitly concatenates them into the single string
"vllm/gpt-oss:120b-mxfp4ollama/qwen3:8b" -- the literal there is a 5-element
list, not 6. The roster is restated explicitly here rather than imported, so
that typo does not silently drop a model from the evaluation.
"""

# Short evaluation labels -> model IDs as served by the endpoint.
# Labels follow the existing results naming (used in result filenames and
# the Model column), so they are kept filesystem-safe.
ENDPOINT_MODELS = {
    "qwen3.8-27b": "vllm/qwen3.8:27b-fp8",
    "deepseek-v4-flash-284b": "vllm/deepseek-v4-flash-0731:284b-a13b-nvfp4",
    "gpt-oss-120b": "vllm/gpt-oss:120b-mxfp4",
}

# Configured but not served; kept for traceability.
UNAVAILABLE = {
    "gemma3-27b": "ollama/gemma3:27b",
    "qwen3-8b": "ollama/qwen3:8b",
    "embeddinggemma-300m": "ollama/embeddinggemma:300m",
}
