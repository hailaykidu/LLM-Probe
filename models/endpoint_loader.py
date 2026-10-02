"""Loader for models served over the KBS OpenAI-compatible HTTP endpoint.

The other loaders in this package pull a checkpoint from the Hugging Face
Hub and run it locally on the cluster GPUs. These models are instead served
remotely at inference.kbs.uni-hannover.de, so there is no checkpoint to
download and no GPU to allocate: generation is an HTTP POST.

`EndpointPipeline` reimplements the same call contract every task script
already relies on -- __call__(list[str]) -> list[{"generated_text": str}],
one entry per input prompt, in order -- so these models drop into the
existing `TaggedModel` wrapper and evaluation loops unchanged.

Three endpoint behaviours are handled here because they silently produce
empty or wrong results otherwise:

1. Model IDs are served WITHOUT the "vllm/" or "ollama/" routing prefix
   that appears in the OpenCode client config. Posting the prefixed name
   returns a 404, so prefixes are stripped (see `normalize_model_id`).

2. These are reasoning models. Tokens are spent on hidden chain-of-thought
   first and are billed to `completion_tokens_details.reasoning_tokens`;
   `message.content` stays None until that budget is exhausted. With a
   small max_tokens (the 128 used by the local loaders) every reply comes
   back None. Observed: max_tokens=8 -> content None, 8 reasoning tokens;
   max_tokens=512 -> content "ቤት", 81 reasoning tokens. MAX_TOKENS is
   therefore set well above the local limit, and `reasoning_content` is
   used as a fallback when content is empty.

3. A request can fail transiently (read timeout, 5xx) partway through a
   5775-row sweep. Each call is retried with backoff and degrades to an
   empty string rather than aborting the task, matching the task scripts'
   existing "skip a failing model, keep the rest" behaviour.
"""
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE_URL = os.environ.get(
    "KBS_INFERENCE_BASE_URL", "https://inference.kbs.uni-hannover.de/v1"
).rstrip("/")

# The endpoint requires a key. It is read from the environment rather than
# hardcoded so the token is not committed to the repository; see
# .claude/Test_f/config.json for the value used during development.
API_KEY_ENV = "KBS_INFERENCE_API_KEY"

# Routing prefixes used by the OpenCode client config but not by the
# served model IDs themselves.
_ROUTING_PREFIXES = ("vllm/", "ollama/")

# Far above the local loaders' max_new_tokens=128: that budget is consumed
# entirely by hidden reasoning tokens before any visible content is emitted.
#
# The value matters for correctness, not just cost. When the budget runs out
# mid-reasoning the reply comes back with finish_reason="length" and
# content=None, which scores as a wrong answer even though the model was on
# its way to the right one. Measured on translation_fidelity prompts:
# max_tokens=512 returned None for "abscess" and "abbot", while 1024
# returned "ሳጹር" and "ኣቦ ምሕረት". At 3072, 3/40 sampled rows still truncated;
# those resolved at 8192 (worst case seen: 3612 completion tokens).
MAX_TOKENS = 8192

# Greedy decoding, to match the deterministic local evaluation setup.
TEMPERATURE = 0.0

RETRIES = 3
BACKOFF_SECONDS = 2.0
TIMEOUT_SECONDS = 180

# Requests are independent and the endpoint serves them concurrently, so
# prompts are issued in parallel -- at ~1-40s per call, a 5775-row task is
# impractical serially. Order is preserved by the executor's map().
CONCURRENCY = int(os.environ.get("KBS_INFERENCE_CONCURRENCY", "8"))


def normalize_model_id(model_id: str) -> str:
    """Strip the client-side routing prefix from a configured model name."""
    for prefix in _ROUTING_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def _api_key():
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set; export it before running an endpoint model."
        )
    return key


class EndpointPipeline:
    """Drop-in replacement for a transformers pipeline, backed by HTTP.

    Matches the contract the task scripts expect:
    __call__(list[str]) -> list[{"generated_text": str}].
    """

    def __init__(self, model_id, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                 concurrency=CONCURRENCY):
        self.model_id = normalize_model_id(model_id)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        # Rows whose reasoning budget ran out before any answer was emitted.
        # Reported per batch so a silently-truncated run is visible rather
        # than being mistaken for genuine model failure.
        self.truncated = 0

    def _post(self, prompt):
        body = json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.load(response)
        choice = payload["choices"][0]
        message = choice["message"]
        if choice.get("finish_reason") == "length" and not message.get("content"):
            self.truncated += 1
        # content is None while the model is still spending its budget on
        # hidden reasoning; fall back to the reasoning channel so a
        # truncated-but-present answer is not silently dropped.
        text = message.get("content") or message.get("reasoning_content") or ""
        return text.strip()

    def _generate(self, prompt):
        for attempt in range(RETRIES):
            try:
                return self._post(prompt)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                    KeyError, ValueError) as exc:
                if attempt == RETRIES - 1:
                    # Degrade to an empty output rather than killing the
                    # whole sweep; the row is recorded as a non-match and
                    # stays resumable on a later run.
                    print(f"⚠️  {self.model_id}: giving up on a prompt ({exc})")
                    return ""
                time.sleep(BACKOFF_SECONDS * (attempt + 1))
        return ""

    def __call__(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        self.truncated = 0
        if self.concurrency > 1 and len(prompts) > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                # map() yields results in input order, which the task
                # scripts rely on when zipping outputs back onto gold rows.
                texts = list(pool.map(self._generate, prompts))
        else:
            texts = [self._generate(p) for p in prompts]
        if self.truncated:
            print(
                f"⚠️  {self.model_id}: {self.truncated}/{len(prompts)} prompts hit the "
                f"{self.max_tokens}-token budget with no answer emitted."
            )
        return [{"generated_text": t} for t in texts]


def load_model(model_id):
    """Return an EndpointPipeline for a model served by the KBS endpoint."""
    return EndpointPipeline(model_id)
