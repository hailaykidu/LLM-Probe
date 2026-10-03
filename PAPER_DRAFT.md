# How Far Can General-Purpose LLMs Go in Tigrinya? A Lexicon-Grounded Evaluation with Trivial Baselines

**Draft — not submitted.** Retraction of the earlier LLMs4SSH @ LREC 2026
version has been requested by the authors and is not yet in effect (§7).

Every number below is reproducible from artifacts in this repository. The
command and the file behind each table are given in §8.

---

## Abstract

We evaluate eight general-purpose large language models on three
lexicon-grounded Tigrinya tasks — part-of-speech tagging, morphosyntactic
labelling, and English→Tigrinya translation — using a 5,783-row annotated
bilingual lexicon. Our central methodological finding is that such
evaluations are easy to over-read: with a permissive matching rule, models
appear to reach 80–89% accuracy, but the same outputs score 0–82% under a
first-token rule, and a constant-answer baseline reaches 53.7% while an
English-only tagger that never sees the Tigrinya reaches 60.4%. Measured
against those baselines, three of eight models demonstrate above-trivial POS
performance and two exceed the English-only baseline on morphosyntactic
labelling, by under three points. On translation all eight models score at or below chrF 0.81,
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

1. **A lexicon-grounded benchmark** of 5,783 English–Tigrinya rows
   (4,331 English headwords, 3,767 Tigrinya forms), each carrying a
   part-of-speech label, derived from a digitised bilingual dictionary [43].
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
correction. The authors have requested its retraction; §7 states the
relationship between the two.

## 2. Related work

### 2.1 Tigrinya and Ge'ez-script NLP

Tigrinya remains severely underrepresented in NLP: a recent survey of over 50
studies from 2011 to 2025 finds a field whose progress is driven almost
entirely by discrete resource-creation milestones rather than by modelling
advances [2], a pattern echoed in broader surveys of Ethiopian-language NLP
[3]. The resources that exist are small and task-specific. Öktem et al. built
a domain-specific Tigrinya→English NMT system using transfer learning from
other Ge'ez-script languages, reporting a 1.3 BLEU improvement over a neural
baseline [4]; Tela et al. showed that an English XLNet fine-tuned on 10k
examples reaches 78.88 F1 on Tigrinya sentiment, outperforming mBERT by 7
points [5]; TiGQA contributed 2,685 expert-annotated question–answer pairs
from 537 context paragraphs [6]; and VEXMLM demonstrated that vocabulary
extension with 30k Ge'ez-script subwords substantially improves XLM-R on
Amharic and Tigrinya, attributing the baseline weakness to high
out-of-vocabulary rates under Latin-centric tokenizers [7]. Tigrinya also
appears as one slice of multilingual benchmarks — MasakhaPOS, the largest POS
dataset for African languages, covering 20 typologically diverse languages
under UD guidelines [8], and Belebele, a parallel multiple-choice
reading-comprehension set in 122 language variants [9]. Our lexicon
complements these: it is not running text, and it tests the
dictionary-headword level that sits beneath all of them. The structural
reason such resources are scarce at all is the one Joshi et al. documented —
representation in NLP venues tracks resource availability, calling into
question the "language agnostic" status of current systems [10].

### 2.2 LLM evaluation for low-resource languages

NLLB-200 added Tigrinya (`tir_Ethi`, Ge'ez script, Afro-Asiatic/Semitic) to a
200-language translation model [11], evaluated on FLORES, whose
professionally translated, multilingually aligned design set the standard for
low-resource MT benchmarking [12]. Against that backdrop, several studies have
established that general-purpose LLMs lag purpose-built MT on exactly the
languages we study. Robinson et al. evaluated ChatGPT on 204 FLORES-200
varieties and found NLLB superior on 169 of 201 languages (84.1%), by an
average of 11.9 chrF, with the gap widest for African languages and the
Ge'ez script specifically [13]; they further report that five-shot prompting
improved on zero-shot by only 0.88 chrF on average across 203 directions,
which is consistent with our own few-shot probe moving gemma-7b from chrF
0.16 to 0.66 with exact match unchanged. Hendy et al. reach the same
conclusion for GPT models across eighteen directions — competitive for
high-resource, limited for low-resource [14] — and Zhu et al. find GPT-4
beating NLLB in 40.91% of directions overall while still trailing badly on
the long tail [15]. Adebara et al. extend this to a continent-scale African
benchmark and attribute the residual disparities to data inequities rather
than model scale [16]. Our §5.3 result — three NLLB checkpoints at chrF
14.1–16.1 against eight LLMs at ≤ 0.81 — is this finding at lexicon
granularity, where the LLM floor is not merely lower but absolute.

A separate strand asks whether LLM evaluations in these settings are
conducted carefully enough. LAraBench benchmarked 33 Arabic tasks across 61
datasets and explicitly computed a random baseline "to determine if the LLMs
predictions are not merely the result of chance" [17]. We regard that step as
mandatory rather than optional, and §5 shows why.

### 2.3 Trivial baselines and spurious cues

The methodological core of this paper has a direct precedent in natural
language inference. Poliak et al. showed that a hypothesis-only model — one
that never sees the premise, a degenerate solution by construction —
significantly outperforms a majority-class baseline on a number of NLI
datasets [19]; Gururangan et al. showed the same artefacts let a simple text
classifier label the hypothesis alone correctly in about 67% of SNLI and 53%
of MultiNLI, concluding that model success "has been overestimated" [20].
McCoy et al. demonstrated that strong test-set scores can rest on fallible
syntactic heuristics that collapse on controlled sets [21], and Niven and Kao
found BERT's 77% on argument reasoning comprehension to be entirely accounted
for by spurious statistical cues [22]. Our English-only tagger is a
hypothesis-only baseline in this sense: the prompt format
`Phrase: {Tigrinya} ({English})` leaves the part of speech largely
recoverable from the gloss, and the baseline reaching 60.38 is the
lexicon-task analogue of the 67% SNLI result.

For POS tagging specifically, Kann et al. make the closest argument to ours:
weakly supervised taggers reported to perform "almost as well as supervised
ones" were evaluated on languages unlike truly low-resource ones, and on 15
genuinely low-resource languages the best model gets fewer than half the
words right, with a strongest baseline macro-average of 39.11% [18]. They
also note that such methods presuppose "high-coverage and almost error-free
dictionaries" [18] — precisely the assumption our §3.3 annotator notes
undermine, where the source dictionary contains mislabelled parts of speech
and outright mistranslations. At the level of benchmark design, Reuel et
al.'s assessment of 46 best practices across 24 benchmarks identifies stating
random performance as a scored criterion, since without it one cannot
separate capability from chance or metric design [23].

### 2.4 Scoring rules, answer extraction and prompt artefacts

Our finding that the matching rule moves reported accuracy by up to 26 points
on fixed outputs belongs to a growing literature on evaluation mechanics. Tam
et al. show that constraining LLMs to structured output formats significantly
degrades measured reasoning, and disentangle format errors from content by
using an LLM as a "perfect parser" rather than a regex [24] — the mirror
image of our problem, where permissive parsing manufactures competence
instead of destroying it. Kirouane and Petrocheilos document the same failure
mode concretely in a low-resource setting: a scorer that scanned the whole
response instead of the requested answer line produced a +29.8 pp artefact,
one-directional against verbose outputs, and they conclude that "any
benchmark that requests an answer format must score that format first and
report how often it was absent" [26]. Our first-token rule is that
recommendation applied; our exact / first-token / set-intersection triple
reports the spread it would otherwise hide.

The prompt-echo column in §5.2 relates to known label-copying behaviour. Zhao
et al. show that few-shot LLM accuracy is destabilised by bias towards
answers appearing in the prompt or common in pretraining, varying "from near
chance to near state-of-the-art" with prompt format alone [25]. When the gold
distribution is 53.7% `noun` and the prompt example contains the word `noun`,
a model that copies the example is scored correct by construction under
set-intersection — gemma-7b does this on 97.4% of answers. We therefore treat
echo rate as a reportable diagnostic rather than a footnote.

### 2.5 Metrics and agreement statistics

We report chrF [27] because character n-gram F-score is
tokenisation-independent and therefore appropriate for Ge'ez-script targets
averaging 1.37 whitespace tokens, where word-level BLEU [28] is
near-undefined; we follow Post's argument that BLEU is a parameterised family
whose configurations are not comparable across papers unless the scheme is
pinned down [29], and report character-level BLEU alongside chrF for this
reason. Agreement is reported as Cohen's κ [30]. We note the standard caveat
that κ is hard to interpret under class imbalance — the "kappa paradox" — and
that absolute thresholds are not meaningful without a contextual reference
[31]; this is why §3.3 reports κ beside raw observed agreement and beside the
filled-label-only variant, rather than reporting a single coefficient.

### 2.6 Dictionaries as evaluation resources

Digitised bilingual dictionaries are a standard bottom-up resource for
languages lacking parallel corpora: Wickramasinghe and de Silva argue that
for low-resource pairs it is "more feasible to move in the bottom-up
direction where finer granular pairs such as dictionary datasets are
developed first" [33], and Alnajjar et al.'s Ve'rdd addresses the
complementary problem of grassroots paper dictionaries edited by many hands,
which require re-evaluation rather than direct ingestion [32]. Our source
exhibits exactly the defects that motivate that work — a direction-swapped
second block, inline gender markers embedded in the target field, and
annotator-flagged mistranslations — and §3.1–§3.3 document them rather than
silently cleaning them.

### 2.7 Relation to the earlier version

An earlier version of this work, *LLM Probe: Evaluating LLMs for Low-Resource
Languages* [1], introduced the lexicon and the four-task framework used here.
It reported POS accuracies of 73.0–80.0 and morphosyntactic accuracies of
70.5–77.0 under a permissive matching rule with no baselines, assigned
different model subsets to different tasks, and concluded that
sequence-to-sequence models "excel in morphosyntactic analysis and
translation quality" [1]. §7 states which of those claims survive re-scoring
and which do not.

## 3. Dataset

### 3.1 Source and construction

The lexicon was digitised from the Swansea Tigrinya–English dictionary [43], with
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
reorients it, and merges both directions, writing **5,783 rows** to
`data/gold_labels/`. Expanding comma-separated senses gives 5,864 atomic
pairs across 4,331 English headwords and 3,767 Tigrinya forms.

**7,234 counts rows, not independent annotations.** Of the 3,646 unique
reverse-direction pairs, 1,448 (39.7%) also occur in the forward block.

**Three counts appear in this paper, and they are not interchangeable.** The
gold files hold **5,783 rows**; `--verify` confirms the script reproduces
them exactly. Deduplication there keys on (English, Tigrinya, features)
rather than the pair alone, so distinct senses of one form survive as
separate items: `estimate → ግምት` is kept as both noun and verb, and
`grocer → በዓል ድኳን` as both `noun` and `noun, masculine`.

Each task script then applies its own pair-level deduplication at load time,
dropping 7 duplicate (English, Tigrinya) pairs and 1 row with no English
side, leaving **5,775 evaluated items** for translation and lexical
alignment. POS tagging and morphosyntactic labelling were run against a
further reduction in which a multi-sense entry is represented by its first
sense — `("a little", "ንእሽቶይ, ቁሩብ,ውሕድ")` becomes `("a little", "ንእሽቶይ")` —
giving **5,701 items**.

That sense-split file was an intermediate and was not originally committed,
so the task scripts as they stand would evaluate 5,775 rather than the 5,701
behind §5.1 and §5.2. `scripts/build_sense_split_eval_set.py` reconstructs it
from the committed gold and verifies the result row-for-row against the saved
model outputs: 5,701 items, no row missing, none extra, no label mismatch. It
is released as `data/gold_labels/{pos_tagging,morphosyntax_probe}_sense_split.json`
so that every table in §5 rests on a tracked artifact.

### 3.2 What the annotations do and do not contain

The POS labels are the source dictionary's, not independently re-annotated
by us. They are coarse: `noun` accounts for 3,109 of 5,783 labelled entries
(53.7%), `verb` 1,542, `adjective` 879.

Morphosyntactic features are **sparse**. Only 185 entries carry gender,
number or plurality; the remaining 5,598 (96.8%) carry a value identical to
the POS label. We therefore do not treat morphosyntactic labelling as a
distinct linguistic dimension, and report it in §5.2 as what it measures:
a second POS task with a different prompt.

### 3.3 Inter-annotator agreement

A 500-item subset of the lexicon was double-annotated for part of speech,
gender, number, category agreement and lexical alignment. The records are
released as `data/POS_english_to_tigrigna_Annotated.xlsx` — two annotator
sheets, an adjudicated sheet and the annotation guide that defines each
value — and `scripts/compute_iaa.py` recomputes every figure below from them.

An annotator sometimes left a cell empty where the other assigned a label.
Those are not matching judgements, so we count them as disagreements and
report Cohen's κ over all 500 items, with a blank treated as its own
category:

| Dimension | κ | Observed agreement | Items |
|---|---|---|---|
| Part of speech | 0.9076 | 0.9500 | 500 |
| Gender | 0.8079 | 0.9240 | 500 |
| Number | 0.6762 | 0.9280 | 500 |
| Category agreement | 0.5912 | 0.9000 | 500 |
| Lexical alignment | 0.8487 | 0.9480 | 500 |

Restricting instead to the items both annotators labelled raises every
dimension — 0.9076, 1.0000, 1.0000, 0.9815, 0.9850 on 500, 420, 428, 445 and
474 items. We report the stricter figures because the difference between the
two columns is largely a difference in what each annotator left blank:
Annotator 1 omits 32 gender, 32 number and 46 agreement labels that
Annotator 2 supplies, against 6, 4 and 3 in the other direction. Discarding
those rows would hide the main source of divergence.

Three properties of this subset bound what the coefficients support, and we
state them rather than leave them to be discovered.

**The two columns are not independent of the adjudication.** On the 25
part-of-speech items where the annotators differ, the adjudicated sheet
reproduces Annotator 2 on all 25 and Annotator 1 on none; on the three
gender, agreement and alignment disagreements it reproduces Annotator 1. The
adjudication is assembled from the two columns rather than decided
separately, so it cannot serve as an independent third judgement.

**Agreement on filled labels is near-total.** Where both annotators assigned
a gender or number value, they never differ — 0 disagreements across 420 and
428 items. The divergence between them is entirely in coverage, not in
choice of label, which is why κ falls as far as it does once blanks count.

**The subset is not a random sample.** Tracing each item to its source sheet
shows the 500 are the alphabetical head of each category: all 9 prepositions,
all 10 pronouns and all 6 interjections, then 305 adjectives and 150 nouns
from the tops of those sheets. The corpus is 54% noun and 25% verb; this
subset is 61% adjective and contains **no verbs**. The coefficients therefore
describe agreement on this subset and are not an estimate for the corpus. A
stratified re-annotation covering verbs would be required for that, and we
report it as future work rather than extrapolating (§9).

The workbook also carries the annotators' free-text notes on 201 of the 500
items, documenting mislabelled parts of speech in the source dictionary,
inflected forms given as citation forms, and outright mistranslations
(`at least` → ብብዚሒ, which means *at most*). These support the caveat in §3.2
that the POS labels are the source dictionary's and were not independently
verified.

## 4. Experimental setup

Eight models, all loaded through the Hugging Face `transformers` interface
[41] on an Ampere-class GPU node (4×A100, 32 CPU cores, 128 GB RAM) under SLURM.

| Label | Checkpoint | Architecture | Instruction-tuned |
|---|---|---|---|
| gemma-2b | `google/gemma-2b-it` [34] | causal | yes |
| gemma-7b | `google/gemma-7b-it` [34] | causal | yes |
| mistral-7b | `mistralai/Mistral-7B-Instruct-v0.2` [35] | causal | yes |
| qwen-7b | `Qwen/Qwen1.5-7B-Chat` [36] | causal | yes |
| falcon-7b | `tiiuae/falcon-7b-instruct` [37] | causal | yes |
| mt5-small | `google/mt5-small` [38] | seq2seq | no |
| mt5-large | `google/mt5-large` [38] | seq2seq | no |
| byt5 | `google/byt5-small` [39] | seq2seq | no |

**We do not compare architectures.** All five causal models are
instruction-tuned and all three sequence-to-sequence models are raw
pretrained checkpoints; the two factors are perfectly confounded in this
model set and no result here can separate them. The three base seq2seq
models receive span-infilling prompts matched to their pretraining objective
(`models/span_infilling.py`); they nonetheless produce almost no usable
output, which we report as a property of those checkpoints on this task
rather than of their architecture.

Generation is greedy with `max_new_tokens=128`, `min_new_tokens=1`.
`xlm-roberta-base` [40] was included initially but fails on all tasks — its
prompts contain no `<mask>` token — and is excluded.

### 4.1 Scoring

Model answers are frequently verbose: gemma-7b's median POS answer is 14
words. We report three rules.

- **exact** — the cleaned answer equals the gold label.
- **first-token** — the model's first word is a gold label word. This is
  the answer the model commits to, and is our headline rule.
- **set-intersection** — any word of the answer is a gold label word. This
  is permissive: an answer naming several parts of speech is correct as soon
  as one fits. We report it only to show the gap.

### 4.2 Baselines

Two reference points, both of which a model must beat to demonstrate
Tigrinya knowledge:

- **Majority** — always answer the most frequent gold label (`noun`).
- **English-only** — an NLTK [42] English POS tagger applied to the English
  gloss alone. The prompt format is `Phrase: {Tigrinya} ({English})`, and
  a dictionary headword's part of speech is largely recoverable from the
  English word, so this baseline uses no Tigrinya whatsoever.

## 5. Results

### 5.1 POS tagging (n = 5,701)

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

### 5.2 Morphosyntactic labelling (n = 5,701)

Because the gold is 96.8% identical to the POS label (§3.2), this is
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

**Only two models exceed the English-only baseline of 61.57**, by 3.02 and
1.75 points; the remaining six fall below it, and five fall below the
majority baseline of 52.38.

The final column measures how often an answer contains the prompt's own
example, `preposition, noun, singular`. Gemma-7b echoes it in 97.4% of
answers; because the example contains `noun`, and `noun` is the majority
gold label, set-intersection scores those echoes as correct. Its apparent
89.42 is largely an artefact of copying the prompt. ByT5's 38.55 is the same
artefact at 69.7% echo, and drops to 0.00 under first-token.

### 5.3 Translation, English→Tigrinya (n = 5,775)

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

### 5.4 Lexical alignment

We report this task for completeness but do not draw conclusions from it:
2,208 of 5,783 alignment labels (38%) are positional fallbacks
(`AlignmentSource: auto`) rather than annotation. Format accuracy ranges
from 0.0383 (mt5-large) to 0.5484 (gemma-2b); semantic accuracy from 0.0012
(mt5-small) to 0.8876 (mistral-7b). The divergence between the two for
individual models is large enough that we regard the task as not yet
well-posed.

## 6. Discussion

**Trivial baselines are not optional.** Two baselines using no Tigrinya
reach 53.7% and 60.4% on our POS task. Of eight models, five score below
both. A paper reporting only model accuracies would present several of these
as successes.

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
above-baseline POS performance. Two exceed the English-only baseline on
morphosyntactic labelling, by 3.02 and 1.75 points — a margin small enough
that we would not rest a claim on it without a held-out re-run. None can
translate into Tigrinya. Even NLLB reaches only 16.24% exact on a dictionary
task, which indicates the difficulty is real rather than an artefact of our
setup.

## 7. Relation to an earlier version of this work

An earlier version of this work was published as *LLM Probe: Evaluating LLMs
for Low-Resource Languages* at LLMs4SSH @ LREC 2026, pp. 224–234
(2026.llms4ssh-1.24). Its
POS tagging and morphosyntactic results were computed with a permissive
matching rule, under which an answer counts as correct if any of its words
matches a gold label word, and were reported without baselines. Re-scored
under a first-token rule and measured against a constant-answer baseline and
an English-only baseline, the central claims of that paper do not hold: only
three of the eight systems exceed both baselines on part-of-speech tagging
and two on morphosyntactic labelling, by a few points in each case, where the
published version reported accuracies of 80–89% against no reference point at
all. The architectural conclusion it drew is confounded with
instruction-tuning, since the sequence-to-sequence checkpoints evaluated were
never instruction-tuned. The authors have requested that it be retracted.

This paper reports the same model outputs, re-scored. It shares the lexicon
and the saved generations with that version and supersedes it; no figure here
is taken from it, and the dataset counts, matching rules and baselines are
all regenerable from the artifacts listed in §8. Readers comparing the two
should treat the earlier numbers as withdrawn rather than as an alternative
measurement.

## 8. Reproducibility

Every number in this paper comes from a committed artifact. `$ROSTER` below
is the evaluated roster, `gemma-2b,gemma-7b,mistral-7b,falcon-7b,qwen-7b,`
`mt5-small,mt5-large,byt5`; passing it pins the tables to these eight models,
since later runs add result files to the same directories.

| Table | Command | Artifact | Job log |
|---|---|---|---|
| §3.1 counts | `python scripts/build_merged_eval_set.py --dry-run --verify` | `Combined_POS_Lexicon.csv` → `data/gold_labels/` | — |
| §3.1 5,701 item set | `python scripts/build_sense_split_eval_set.py --verify` | `{pos_tagging,morphosyntax_probe}_sense_split.json` | — |
| §3.3 agreement | `python scripts/compute_iaa.py` | `POS_english_to_tigrigna_Annotated.xlsx` | — |
| §5.1, §5.2 | `python scripts/rescore_tasks.py --models $ROSTER` | `rescored_2026-09-28.json` | `logs/eval_75167.out` |
| §5.3 LLMs | `python scripts/compute_bleu.py` | `translation_fidelity_bleu.json` | `logs/eval_tf_rerun_76683.out` |
| §5.3 NLLB | `python scripts/probe_nllb_translation.py --checkpoint facebook/nllb-200-3.3B --n 0` | `nllb_probe_*.json` | `logs/nllb_sweep_77617.out` |
| §5.3 few-shot | `python scripts/probe_fewshot_translation.py --model gemma-7b` | `fewshot_probe_gemma-7b.json` | `logs/fewshot_probe_77601.out` |

Raw model outputs for every item are in
`results/evaluation_reports/<task>/<task>_<model>.json`.

## 9. Limitations

- POS labels are the source dictionary's, not independently re-annotated. The
  agreement coefficients in §3.3 describe a 500-item subset that is
  alphabetical and contains no verbs, and whose adjudication is assembled
  from the two annotator columns rather than decided separately. They are not
  an estimate for the corpus; a stratified re-annotation covering verbs would
  be needed for that.
- Morphosyntactic features cover 185 of 5,783 gold rows.
- Lexical alignment gold is 38% positional fallback; §5.4 is not a result.
- Entries are dictionary headwords, not running text; performance here does
  not predict sentence-level performance.
- The model set cannot separate architecture from instruction-tuning. An
  instruction-tuned seq2seq model (`flan-t5`, `mt0`) would be needed.
- The English-only baseline shows the POS task is substantially solvable
  without Tigrinya. A harder task would withhold the English gloss.


## References

[1] Teklehaymanot, H. K., Gebremariam, G. & Nejdl, W. "LLM Probe: Evaluating
LLMs for Low-Resource Languages." *Proc. LLMs4SSH @ LREC 2026*, 224–234
(2026). ACL Anthology 2026.llms4ssh-1.24.

[2] Gaim, F. & Park, J. C. "Natural Language Processing for Tigrinya: Current
State and Future Directions." arXiv (2026). doi:10.48550/arXiv.2507.17974

[3] Tonja, A. L. et al. "Natural Language Processing in Ethiopian Languages:
Current State, Challenges, and Opportunities." arXiv (2023).
doi:10.48550/arXiv.2303.14406

[4] Öktem, A., Plitt, M. & Tang, G. "Tigrinya Neural Machine Translation with
Transfer Learning for Humanitarian Response." *AfricaNLP Workshop* (2020).
doi:10.48550/arXiv.2003.11523

[5] Tela, A., Woubie, A. & Hautamäki, V. "Transferring Monolingual Model to
Low-Resource Language: The Case of Tigrinya." arXiv (2020).
doi:10.48550/arXiv.2006.07698

[6] Teklehaymanot, H., Fazlija, D., Ganguly, N., Patro, G. K. & Nejdl, W.
"TiGQA: An Expert-Annotated Question-Answering Dataset in Tigrinya." arXiv
(2024). doi:10.48550/arXiv.2404.17194

[7] Teklehaymanot, H. K., Yadeta, D. D. & Nejdl, W. "Expanding the Lexicon of
Ge'ez Based African Languages: A Comparative Study of Amharic and Tigrinya."
arXiv (2026). doi:10.48550/arXiv.2607.15209

[8] Dione, C. M. B. et al. "MasakhaPOS: Part-of-Speech Tagging for
Typologically Diverse African Languages." *Proc. ACL* (2023).
doi:10.48550/arXiv.2305.13989

[9] Bandarkar, L. et al. "The Belebele Benchmark: a Parallel Reading
Comprehension Dataset in 122 Language Variants." *Proc. ACL* (2024).
doi:10.18653/v1/2024.acl-long.44

[10] Joshi, P., Santy, S., Budhiraja, A., Bali, K. & Choudhury, M. "The State
and Fate of Linguistic Diversity and Inclusion in the NLP World." *Proc.
ACL*, 6282–6293 (2020). doi:10.48550/arXiv.2004.09095

[11] NLLB Team et al. "No Language Left Behind: Scaling Human-Centered
Machine Translation." arXiv (2022). doi:10.48550/arXiv.2207.04672

[12] Goyal, N. et al. "The FLORES-101 Evaluation Benchmark for Low-Resource
and Multilingual Machine Translation." *Trans. Assoc. Comput. Linguist.* 10,
522–538 (2022). doi:10.48550/arXiv.2106.03193

[13] Robinson, N. R., Ogayo, P., Mortensen, D. R. & Neubig, G. "ChatGPT MT:
Competitive for High- (but not Low-) Resource Languages." arXiv (2023).
doi:10.48550/arXiv.2309.07423

[14] Hendy, A. et al. "How Good Are GPT Models at Machine Translation? A
Comprehensive Evaluation." arXiv (2023). doi:10.48550/arXiv.2302.09210

[15] Zhu, W. et al. "Multilingual Machine Translation with Large Language
Models: Empirical Results and Analysis." arXiv (2023).
doi:10.48550/arXiv.2304.04675

[16] Adebara, I., Toyin, H. O., Ghebremichael, N. T., Elmadany, A. &
Abdul-Mageed, M. "Where Are We? Evaluating LLM Performance on African
Languages." arXiv (2025). doi:10.48550/arXiv.2502.19582

[17] Abdelali, A. et al. "LAraBench: Benchmarking Arabic AI with Large
Language Models." arXiv (2023). doi:10.48550/arXiv.2305.14982

[18] Kann, K., Lacroix, O. & Søgaard, A. "Weakly Supervised POS Taggers
Perform Poorly on Truly Low-Resource Languages." *Proc. AAAI* (2020).
doi:10.48550/arXiv.2004.13305

[19] Poliak, A., Naradowsky, J., Haldar, A., Rudinger, R. & Van Durme, B.
"Hypothesis Only Baselines in Natural Language Inference." *Proc. \*SEM*
(2018). doi:10.48550/arXiv.1805.01042

[20] Gururangan, S. et al. "Annotation Artifacts in Natural Language
Inference Data." *Proc. NAACL-HLT* (2018). doi:10.48550/arXiv.1803.02324

[21] McCoy, R. T., Pavlick, E. & Linzen, T. "Right for the Wrong Reasons:
Diagnosing Syntactic Heuristics in Natural Language Inference." *Proc. ACL*
(2019). doi:10.48550/arXiv.1902.01007

[22] Niven, T. & Kao, H.-Y. "Probing Neural Network Comprehension of Natural
Language Arguments." *Proc. ACL* (2019). doi:10.48550/arXiv.1907.07355

[23] Reuel, A. et al. "BetterBench: Assessing AI Benchmarks, Uncovering
Issues, and Establishing Best Practices." arXiv (2024).
doi:10.48550/arXiv.2411.12990

[24] Tam, Z. R., Wu, C.-K., Tsai, Y.-L., Lin, C.-Y., Lee, H.-y. & Chen, Y.-N.
"Let Me Speak Freely? A Study on the Impact of Format Restrictions on
Performance of Large Language Models." arXiv (2024).
doi:10.48550/arXiv.2408.02442

[25] Zhao, T. Z., Wallace, E., Feng, S., Klein, D. & Singh, S. "Calibrate
Before Use: Improving Few-Shot Performance of Language Models." *Proc.
ICML*, 12697–12706 (2021). doi:10.48550/arXiv.2102.09690

[26] Kirouane, A. & Petrocheilos, C. "Thinking in a Low-Resource Language:
What SFT Builds, What RL Fixes, What Accuracy Cannot See." arXiv (2026).
doi:10.48550/arXiv.2608.17744

[27] Popović, M. "chrF: character n-gram F-score for automatic MT
evaluation." *Proc. Tenth Workshop on Statistical Machine Translation*,
392–395 (2015). doi:10.18653/v1/W15-3049

[28] Papineni, K., Roukos, S., Ward, T. & Zhu, W.-J. "BLEU: a Method for
Automatic Evaluation of Machine Translation." *Proc. ACL*, 311–318 (2002).
doi:10.3115/1073083.1073135

[29] Post, M. "A Call for Clarity in Reporting BLEU Scores." *Proc. Third
Conf. on Machine Translation*, 186–191 (2018). doi:10.48550/arXiv.1804.08771

[30] Cohen, J. "A Coefficient of Agreement for Nominal Scales." *Educational
and Psychological Measurement* 20, 37–46 (1960).
doi:10.1177/001316446002000104

[31] Wong, K., Paritosh, P. & Aroyo, L. "Cross-replication Reliability — An
Empirical Approach to Interpreting Inter-rater Reliability." *Proc. ACL*
(2021). doi:10.48550/arXiv.2106.07393

[32] Alnajjar, K., Hämäläinen, M., Rueter, J. & Partanen, N. "Ve'rdd.
Narrowing the Gap between Paper Dictionaries, Low-Resource NLP and Community
Involvement." arXiv (2020). doi:10.48550/arXiv.2012.02578

[33] Wickramasinghe, K. & de Silva, N. "Sinhala-English Parallel Word
Dictionary Dataset." *Proc. ICIIS* (2023). doi:10.1109/ICIIS58898.2023.10253560

[34] Gemma Team, Google DeepMind. "Gemma: Open Models Based on Gemini
Research and Technology." arXiv (2024). doi:10.48550/arXiv.2403.08295

[35] Jiang, A. Q. et al. "Mistral 7B." arXiv (2023).
doi:10.48550/arXiv.2310.06825

[36] Bai, J. et al. "Qwen Technical Report." arXiv (2023).
doi:10.48550/arXiv.2309.16609

[37] Almazrouei, E. et al. "The Falcon Series of Open Language Models." arXiv
(2023). doi:10.48550/arXiv.2311.16867

[38] Xue, L. et al. "mT5: A Massively Multilingual Pre-trained Text-to-Text
Transformer." *Proc. NAACL-HLT* (2021). doi:10.48550/arXiv.2010.11934

[39] Xue, L. et al. "ByT5: Towards a Token-Free Future with Pre-trained
Byte-to-Byte Models." *Trans. Assoc. Comput. Linguist.* 10, 291–306 (2022).
doi:10.48550/arXiv.2105.13626

[40] Conneau, A. et al. "Unsupervised Cross-lingual Representation Learning
at Scale." *Proc. ACL* (2020). doi:10.48550/arXiv.1911.02116

[41] Wolf, T. et al. "Transformers: State-of-the-Art Natural Language
Processing." *Proc. EMNLP: System Demonstrations*, 38–45 (2020).
doi:10.48550/arXiv.1910.03771

[42] Loper, E. & Bird, S. "NLTK: The Natural Language Toolkit." *Proc. ACL
Workshop on Effective Tools and Methodologies for Teaching NLP* (2002).
doi:10.48550/arXiv.cs/0205028

[43] *Tigrinya–English Dictionary.* Union of International Democrats /
Swansea (digitised edition).
https://uidswansea.com/wp-content/uploads/2015/03/tigrinya-english-dictionary.pdf
