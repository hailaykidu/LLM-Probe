"""NLLB-200 loader for English->Tigrinya translation.

Unlike the general-purpose LLMs in this project, NLLB is a dedicated
translation model with Tigrinya (tir_Ethi) as an explicitly supported
target language. It needs handling the shared base_loader does not provide:
the target language is selected by forcing the decoder's first generated
token to that language's id, not by prompt text. Prompt templates are
therefore not used -- the raw English phrase is the input.

Exposed with the same call contract as the other pipelines in this project
(list[str] in, list[{"generated_text": str}] out) so task scripts can use it
without special-casing.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_CHECKPOINT = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "tir_Ethi"


class NllbTranslationPipeline:
    def __init__(self, model, tokenizer, tgt_lang=TGT_LANG, max_new_tokens=64,
                 batch_size=16, num_beams=5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.num_beams = num_beams
        tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        if tgt_id is None or tgt_id == tokenizer.unk_token_id:
            raise ValueError(f"target language {tgt_lang!r} not in this NLLB tokenizer")
        self.tgt_id = tgt_id

    def __call__(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        device = next(self.model.parameters()).device
        results = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            inputs = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tgt_id,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    # The gold entries are single lexicon terms, so a repeated
                    # n-gram is always degenerate output ("ጎደና፣ ጎደና"), never a
                    # legitimate translation.
                    no_repeat_ngram_size=3,
                )
            for text in self.tokenizer.batch_decode(out, skip_special_tokens=True):
                results.append({"generated_text": text})
        return results


def load_model(checkpoint=DEFAULT_CHECKPOINT):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint, dtype=torch.float16, device_map="auto"
    )
    return NllbTranslationPipeline(model, tokenizer)
