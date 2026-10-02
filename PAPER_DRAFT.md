# How Far Can General-Purpose LLMs Go in Tigrinya? A Lexicon-Grounded Evaluation with Trivial Baselines

**Draft — replaces the retracted LLMs4SSH @ LREC 2026 paper. Not submitted.**

Every number below is reproducible from artifacts in this repository. The
command and the file behind each table are given in §7.

---

## Abstract

We evaluate eight general-purpose large language models on three
lexicon-grounded Tigrinya tasks — part-of-speech tagging, morphosyntactic
labelling, and English→Tigrinya translation — using a 5,775-pair annotated
bilingual lexicon. Our central methodological finding is that such
evaluations are easy to over-read: with a permissive matching rule, models
appear to reach 80–89% accuracy, but the same outputs score 0–82% under a
first-token rule, and a constant-answer baseline reaches 53.7% while an
English-only tagger that never sees the Tigrinya reaches 60.4%. Measured
against those baselines, three of eight models demonstrate above-trivial POS
performance and none exceeds the English-only baseline on morphosyntactic
labelling. On translation all eight models score at or below chrF 0.81,
while NLLB-200-3.3B, which supports Tigrinya, reaches chrF 16.05 on identical
items with an identical scorer. We argue that low-resource evaluations
without trivial baselines are uninterpretable, and release the lexicon,
code, model outputs, and job logs.

---

## 1. Introduction

Tigrinya is a Semitic language of Eritrea and northern Ethiopia with
approximately 9–10 million speakers and very little machine-readable
linguistic infrastructure. This scarcity makes it a natural test of whether
general-purpose LLMs generalise beyond high-resource languages, and a
difficult one to measure: without established benchmarks it is easy to
produce numbers that look like competence.

This paper makes three contributions.

1. **A lexicon-grounded benchmark** of 5,775 unique English–Tigrinya pairs
   (4,331 English headwords, 3,767 Tigrinya forms), each carrying a
   part-of-speech label, derived from a digitised bilingual dictionary.
2. **An evaluation of eight LLMs** across POS tagging, morphosyntactic
   labelling and translation, reported under three matching rules and
   against two trivial baselines.
3. **A methodological result**: on a lexicon-scale task, the choice of
   matching rule moves reported accuracy by up to 26 points, and two
   baselines that use no Tigrinya at all reach 53.7% and 60.4%. Papers in
   this setting that report neither are not interpretable.

The third contribution is the reason for this paper. An earlier version of
this work reported results computed with a permissive matching rule and no
baselines, and drew an architectural conclusion that does not survive either
correction. That paper has been retracted; §6 states what changed.

## 2. Dataset

### 2.1 Source and construction

The lexicon was digitised from the Swansea Tigrinya–English dictionary, with
additional entries contributed by native-speaker linguists, and carries the
source dictionary's own part-of-speech abbreviations. The digitised file,
`Combined_POS_Lexicon.csv`, holds 7,234 rows in two blocks: 3,587
English→Tigrinya entries followed by 3,647 Tigrinya→English entries. The two
blocks share column headers, so in the second block the column named
`English` holds Ge'ez script and the column named `Tigrigna` holds English.

**Provenance.** That CSV is a flattening of
`data/POS_english_to_tigrigna_Annotated.xlsx`, the working annotation
workbook, whose eleven part-of-speech sheets hold the same 7,234 rows with
the same per-category counts (3,929 noun, 1,784 verb, 993 adjective, 194
adverb, …). The flattening concatenates those sheets in workbook order and
lifts the inline gender/number markers that the workbook keeps inside the
Tigrinya field — `(nf) ወካሲት, ከሳሲት` — out into the `POS_CATEGORIES` column,
which is where the finer labels `(nm) noun masculine`, `(nf) noun feminine`
and `(npl) noun plural` come from. The two files differ on exactly the 247
rows carrying such a marker. The column swap described above is present in
the workbook as well: 3,635 of its 7,234 rows (50.2%) carry Ge'ez script in
the `English` column, so the defect originates in digitisation, not in the
CSV export. Both files are released; `scripts/compute_iaa.py` reports the
sheet-level counts.

`scripts/build_merged_eval_set.py` detects the second block by script,
reorients it, and merges both directions, yielding **5,775 unique
(English, Tigrinya) pairs**. Expanding comma-separated senses gives 5,864
atomic pairs across 4,331 English headwords and 3,767 Tigrinya forms.

**7,234 counts rows, not independent annotations.** Of the 3,646 unique
reverse-direction pairs, 1,448 (39.7%) also occur in the forward block.

### 2.2 What the annotations do and do not contain

The POS labels are the source dictionary's, not independently re-annotated
by us. They are coarse: `noun` accounts for 3,109 of 5,783 labelled entries
(53.7%), `verb` 1,542, `adjective` 879.

Morphosyntactic features are **sparse**. Only 185 entries carry gender,
number or plurality; the remaining 5,598 (96.8%) carry a value identical to
the POS label. We therefore do not treat morphosyntactic labelling as a
distinct linguistic dimension, and report it in §4.2 as what it measures:
a second POS task with a different prompt.

### 2.3 Inter-annotator agreement

We do not report inter-annotator agreement, and the reason is worth stating
precisely, because the annotation records do exist. They are released here as
`data/POS_english_to_tigrigna_Annotated.xlsx`, which holds a 500-item sample,
two annotator sheets, an adjudicated sheet and an annotation guide.
`scripts/compute_iaa.py` recomputes everything in this section from it.

Agreement cannot be measured from these records, for two independent reasons.

**The second annotation is not independent of the adjudication.** On the 500
shared items, the Annotator 2 sheet is identical to the adjudicated sheet on
part of speech, agreement and alignment, and differs on one gender and two
number cells; Annotator 1 differs from the adjudication on 19–52 cells per
dimension. Cohen's κ between the two sheets is therefore 1.0000 for gender,
number, agreement and alignment — not strong agreement, but a column compared
against its own adjudication. Part of speech is the only dimension on which
the two sheets genuinely differ, giving κ = 0.9076 on 500 items.

**The sample is not random.** Tracing each sampled item back to its source
sheet shows the 500 items are the alphabetical head of each category: all 9
prepositions, all 10 pronouns and all 6 interjections, then 305 adjectives
and 150 nouns from the tops of those sheets. The corpus is 54% nouns and 25%
verbs; the sample is 61% adjectives and contains **no verbs at all**. A κ
computed here would not describe the corpus even if the two columns were
independent.

The retracted paper reported κ of 0.89 (POS), 0.86 (gender), 0.88 (number),
0.84 (agreement) and 0.91 (lexical alignment). None of these is recoverable.
We release the workbook so that this is checkable rather than asserted.

The annotator's free-text notes, on 201 of the 500 items, are a more useful
record than the κ values ever were: they document mislabelled parts of speech
in the source dictionary, inflected forms given as citation forms, and
outright mistranslations (`at least` → ብብዚሒ, which means *at most*). They are
direct evidence for the caveat in §2.2 that the POS labels are the source
dictionary's and were not independently verified.

## 3. Experimental setup

Eight models, all loaded through the Hugging Face `transformers` interface
on an Ampere-class GPU node (4×A100, 32 CPU cores, 128 GB RAM) under SLURM.

| Label | Checkpoint | Architecture | Instruction-tuned |
|---|---|---|---|
| gemma-2b | `google/gemma-2b-it` | causal | yes |
| gemma-7b | `google/gemma-7b-it` | causal | yes |
| mistral-7b | `mistralai/Mistral-7B-Instruct-v0.2` | causal | yes |
| qwen-7b | `Qwen/Qwen1.5-7B-Chat` | causal | yes |
| falcon-7b | `tiiuae/falcon-7b-instruct` | causal | yes |
| mt5-small | `google/mt5-small` | seq2seq | no |
| mt5-large | `google/mt5-large` | seq2seq | no |
| byt5 | `google/byt5-small` | seq2seq | no |

**We do not compare architectures.** All five causal models are
instruction-tuned and all three sequence-to-sequence models are raw
pretrained checkpoints; the two factors are perfectly confounded in this
model set and no result here can separate them. The three base seq2seq
models receive span-infilling prompts matched to their pretraining objective
(`models/span_infilling.py`); they nonetheless produce almost no usable
output, which we report as a property of those checkpoints on this task
rather than of their architecture.

Generation is greedy with `max_new_tokens=128`, `min_new_tokens=1`.
`xlm-roberta-base` was included initially but fails on all tasks — its
prompts contain no `<mask>` token — and is excluded.

### 3.1 Scoring

Model answers are frequently verbose: gemma-7b's median POS answer is 14
words. We report three rules.

- **exact** — the cleaned answer equals the gold label.
- **first-token** — the model's first word is a gold label word. This is
  the answer the model commits to, and is our headline rule.
- **set-intersection** — any word of the answer is a gold label word. This
  is permissive: an answer naming several parts of speech is correct as soon
  as one fits. We report it only to show the gap.

### 3.2 Baselines

Two reference points, both of which a model must beat to demonstrate
Tigrinya knowledge:

- **Majority** — always answer the most frequent gold label (`noun`).
- **English-only** — an NLTK English POS tagger applied to the English
  gloss alone. The prompt format is `Phrase: {Tigrinya} ({English})`, and
  a dictionary headword's part of speech is largely recoverable from the
  English word, so this baseline uses no Tigrinya whatsoever.

## 4. Results

### 4.1 POS tagging (n = 5,701)

| Model | exact | **first-token** | set-intersection |
|---|---|---|---|
| mistral-7b | 33.71 | **81.56** | 85.91 |
| gemma-7b | 0.00 | **79.39** | 82.44 |
| gemma-2b | 71.71 | **72.86** | 73.29 |
| falcon-7b | 26.94 | **40.64** | 44.45 |
| qwen-7b | 17.00 | **17.00** | 17.07 |
| mt5-large | 0.00 | 0.09 | 1.75 |
| byt5 | 0.00 | 0.00 | 0.05 |
| mt5-small | 0.00 | 0.00 | 0.00 |
| **Majority ("noun")** | | **53.73** | |
| **English-only tagger** | | **60.38** | |

Three models exceed both baselines: mistral-7b, gemma-7b, gemma-2b.
Falcon-7b and qwen-7b fall below the majority baseline; all three seq2seq
checkpoints are at or near zero.

The exact column shows how brittle these numbers are. Gemma-7B scores 0.00%
exact and 79.39% first-token: it always answers in a sentence, never a bare
label. A reader given only one of these columns would draw opposite
conclusions.

### 4.2 Morphosyntactic labelling (n = 5,701)

Because the gold is 96.8% identical to the POS label (§2.2), this is
effectively a second POS task.

| Model | **first-token** | set-intersection | echoes prompt example |
|---|---|---|---|
| gemma-2b | **64.59** | 78.02 | 1.2% |
| gemma-7b | **63.32** | 89.42 | **97.4%** |
| mistral-7b | **54.92** | 81.09 | 3.7% |
| falcon-7b | **39.17** | 76.28 | 19.5% |
| qwen-7b | **34.33** | 54.45 | 0.4% |
| mt5-small | 0.86 | 2.49 | 0.0% |
| mt5-large | 0.11 | 0.74 | 0.0% |
| byt5 | 0.00 | 38.55 | 69.7% |
| **Majority** | **52.38** | | |
| **English-only tagger** | **61.57** | | |

**No model exceeds the English-only baseline of 61.57.** Two approach it.

The final column measures how often an answer contains the prompt's own
example, `preposition, noun, singular`. Gemma-7b echoes it in 97.4% of
answers; because the example contains `noun`, and `noun` is the majority
gold label, set-intersection scores those echoes as correct. Its apparent
89.42 is largely an artefact of copying the prompt. ByT5's 38.55 is the same
artefact at 69.7% echo, and drops to 0.00 under first-token.

### 4.3 Translation, English→Tigrinya (n = 5,775)

Tigrinya references average 1.37 whitespace tokens, so word-level BLEU-4 is
near-undefined; we report chrF and character-level BLEU.

| Model | chrF | BLEU-4 (char) | BLEU-4 (word) | exact |
|---|---|---|---|---|
| **NLLB-200-3.3B** | **16.05** | **14.23** | 0.24 | **16.24** |
| NLLB-200-1.3B | 15.01 | 13.29 | 0.23 | 15.00 |
| NLLB-200-600M | 14.11 | 12.59 | 0.25 | 14.42 |
| byt5 | 0.81 | 0.01 | 0.00 | 0.05 |
| gemma-2b | 0.39 | 0.01 | 0.01 | 0.07 |
| mistral-7b | 0.26 | 0.00 | 0.00 | 0.03 |
| mt5-small | 0.20 | 0.04 | 0.02 | 0.05 |
| gemma-7b | 0.16 | 0.00 | 0.00 | 0.03 |
| mt5-large | 0.13 | 0.04 | 0.01 | 0.02 |
| falcon-7b | 0.13 | 0.00 | 0.00 | 0.03 |
| qwen-7b | 0.04 | 0.00 | 0.00 | 0.28 |

All eight general-purpose models are at the floor. NLLB-200, for which
Tigrinya (`tir_Ethi`) is a supported target language, reaches chrF 16.05 on
identical items with an identical scorer — roughly twenty times the best LLM.

Note that word-level BLEU-4 is 0.24 for NLLB-3.3B, which translates 16.24%
of items exactly. **Word-level BLEU is not usable at lexicon scale**, and a
paper reporting it would show a working system as scoring zero.

We verified the LLM failure is not one of output format. Few-shot prompting
with eight exemplars and an explicit format constraint moved gemma-7b from
chrF 0.16 to 0.66 with exact match unchanged at 0.00; extracting the Ge'ez
span from verbose answers recovered the reference in 2 of 5,775 items. The
median character-level similarity between model output and reference is
0.000: the models emit Ge'ez-shaped strings that are not the target words
(`ባህል` for ብዙሕ, `ልልል` as repetition).

### 4.4 Lexical alignment

We report this task for completeness but do not draw conclusions from it:
2,208 of 5,783 alignment labels (38%) are positional fallbacks
(`AlignmentSource: auto`) rather than annotation. Format accuracy ranges
from 0.0383 (mt5-large) to 0.5484 (gemma-2b); semantic accuracy from 0.0012
(mt5-small) to 0.8876 (mistral-7b). The divergence between the two for
individual models is large enough that we regard the task as not yet
well-posed.

## 5. Discussion

**Trivial baselines are not optional.** Two baselines using no Tigrinya
reach 53.7% and 60.4% on our POS task. Of eight models, five score below
60.4% and two below 53.7%. A paper reporting only model accuracies would
present several of these as successes.

**The matching rule is a result, not a detail.** Between exact and
set-intersection, gemma-7b moves from 0.00 to 82.44 on the same outputs.
Any lexicon-scale evaluation should state its rule precisely and report at
least two.

**Prompt echo inflates scores.** Where the gold label distribution is
skewed and the prompt contains an example, a model that copies the example
scores well by construction. This is measurable (our final column) and
should be measured.

**Purpose-built beats general-purpose by a wide margin.** A 600M translation
model outperforms every 7B general-purpose model by more than an order of
magnitude on chrF. For low-resource languages, model selection dominates
model size.

**What we can claim about Tigrinya competence.** Three of eight models show
above-baseline POS performance. None shows above-baseline morphosyntactic
performance. None can translate into Tigrinya. Even NLLB reaches only 16.24%
exact on a dictionary task, which indicates the difficulty is real rather
than an artefact of our setup.

## 6. Relation to the retracted paper

An earlier version of this work was published at LLMs4SSH @ LREC 2026 and
has been retracted. The following were wrong and are corrected here.

| Issue | Then | Now |
|---|---|---|
| Matching rule | set-intersection only | three rules reported, first-token headline |
| Baselines | none | majority 53.7, English-only 60.4 |
| Architecture claim | seq2seq superior | withdrawn — confounded with instruction-tuning |
| Morphosyntax | distinct task | reported as a second POS task (gold 96.8% identical) |
| Gold POS labels | `(adv) adverb`, unmatched by any output | prefix stripped |
| Direction blocks | reverse block not reoriented | reoriented before merge |
| Dataset statistics | computed on the swapped file | recomputed after reorientation |
| Model identifiers | "Falcon-10B", "mT5-base" | `falcon-7b-instruct`, `mt5-small` |
| BLEU | reported, never computed | chrF and char-BLEU, implemented |
| Inter-annotator agreement | κ 0.84–0.91 over five dimensions | not reported; records released and shown unusable (§2.3) |

The published Table 5 values could not be reproduced from the codebase
before or after these fixes, and their origin has not been determined.

## 7. Reproducibility

Every number in this paper comes from a committed artifact.

| Table | Command | Artifact | Job log |
|---|---|---|---|
| §2.1 counts | `python scripts/build_merged_eval_set.py --dry-run` | `Combined_POS_Lexicon.csv` | — |
| §2.3 agreement | `python scripts/compute_iaa.py` | `POS_english_to_tigrigna_Annotated.xlsx` | — |
| §4.1, §4.2 | `python scripts/rescore_tasks.py` | `rescored_2026-09-28.json` | `logs/eval_75167.out` |
| §4.3 LLMs | `python scripts/compute_bleu.py` | `translation_fidelity_bleu.json` | `logs/eval_tf_rerun_76683.out` |
| §4.3 NLLB | `python scripts/probe_nllb_translation.py --checkpoint facebook/nllb-200-3.3B --n 0` | `nllb_probe_*.json` | `logs/nllb_sweep_77617.out` |
| §4.3 few-shot | `python scripts/probe_fewshot_translation.py --model gemma-7b` | `fewshot_probe_gemma-7b.json` | `logs/fewshot_probe_77601.out` |

Raw model outputs for every item are in
`results/evaluation_reports/<task>/<task>_<model>.json`. Pre-correction
results are preserved unmodified in
`results/evaluation_reports_pre_fix_backup/`. `CORRECTIONS.md` documents the
audit.

## 8. Limitations

- POS labels are the source dictionary's, not independently annotated. The
  annotation workbook is released, but its second annotator sheet is not
  independent of the adjudication and its 500-item sample is alphabetical and
  verb-free, so no usable agreement statistic can be derived from it (§2.3).
- Morphosyntactic features cover 185 of 5,775 entries.
- Lexical alignment gold is 38% positional fallback; §4.4 is not a result.
- Entries are dictionary headwords, not running text; performance here does
  not predict sentence-level performance.
- The model set cannot separate architecture from instruction-tuning. An
  instruction-tuned seq2seq model (`flan-t5`, `mt0`) would be needed.
- The English-only baseline shows the POS task is substantially solvable
  without Tigrinya. A harder task would withhold the English gloss.
