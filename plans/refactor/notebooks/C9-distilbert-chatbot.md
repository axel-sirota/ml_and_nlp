# C9 - DistilBERT Chatbot - Cell-by-Cell Plan

## Status

RENEWED. Source notebook: `3-Text-Classification/9-BERT_Text_Classification.ipynb` (56 cells).
New file: `C-Transfer-Learning/9-DistilBERT_Chatbot.ipynb`. Slug: `distilbert-chatbot`.

Keep the BERT fine-tune core (attention intuition, WordPiece tokenizer, load model, Trainer,
evaluate, save/load). Swap the dataset from CNN headlines (4-class) to GLUE SST-2 (binary
sentiment) so the notebook continues the B8 IMDB-sentiment thread and directly breaks the
B7/B8 MLP ceiling. Drop the manual PyTorch training loop and the linear-warmup scheduler
(they overlap B6/B7 and blow the cell budget). ADD a guarded Gradio chatbot that loads the
fine-tuned model. This is Part C, the transfer-learning payoff and the first time the course
ships a real Gradio chatbot.

## Context

Students arrive from B7 (the STOPPER) and B8 (Capstone B). They already have, by exact name:
- `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` and `SEED = 42`
  (`torch.manual_seed(SEED)` + `np.random.seed(SEED)`), the course-wide convention.
- The mental model "text -> word2vec vectors -> mean document vector -> MLP -> label" and the
  honest limitation that averaging word vectors throws away word order and context.
- A trained MLP sentiment classifier (B8, IMDB) and a stated accuracy baseline that this
  notebook must beat. B8's bridge line was "how close did your hand-built model get to a
  pretrained transformer?" - C9 answers it.
- The `pipeline()` mental model from A2 (zero-shot inference, no training).

Key insight they leave with: a transformer reads a sentence with **attention** so each token's
representation depends on context, and **subword (WordPiece) tokenization** shares roots across
inflections. Fine-tuning a pretrained DistilBERT is about 40 lines of HuggingFace config and
beats the static-embedding MLP. They also leave with a working Gradio chatbot wrapping their
fine-tuned model - the course's end deliverable shape.

## Chatbot Through-Line

This is the notebook where the course's promised artifact first appears. Parts A and B built
the skill stack (tools-first NLP -> embeddings as geometry -> MLP on word2vec features). C9
opens the black box one final time: attention replaces averaging, and a fine-tuned
`distilbert-base-uncased` becomes a sentiment classifier that the student wraps in a
`gradio` `Interface`. The one-line bridge to C10: "Classification gives a label; a real
assistant generates an answer. Next we fine-tune an encoder-decoder (T5/BART) and put it in
the same Gradio chatbot shell so it can write answers, not just pick a class." The Gradio
guard pattern (try/except ImportError, `gr.Interface.from_pipeline`) and the save/reload
pattern are reused verbatim in C10 and the C11 capstone.

## STAR Story

- **Situation.** You are still the ML practitioner on the support-platform team. In B8 you
  hand-built an MLP on averaged word2vec vectors and it classified review sentiment, but it
  plateaued: it reads "the update is not good" the same as "the update is good" because mean
  pooling destroys word order. The product team wants higher accuracy before this auto-tags
  incoming support messages as positive or negative.
- **Task.** Fine-tune a pretrained DistilBERT on the SST-2 sentiment benchmark, beat the MLP
  baseline on the same kind of binary task, then load the fine-tuned model into a simple
  Gradio chatbot a non-technical teammate can actually use.
- **Action.** Install HuggingFace `transformers`, `datasets`, `evaluate`, `accelerate`, and
  `gradio`. Tokenize with WordPiece via `AutoTokenizer`. Load
  `distilbert-base-uncased` with a 2-class head. Fine-tune with the high-level `Trainer`.
  Evaluate on the held-out SST-2 validation split and compare to the MLP. Save the model,
  reload it, and wrap it in a guarded Gradio `Interface`.
- **Result.** The fine-tuned DistilBERT lands around 90 percent accuracy on SST-2 validation,
  clearly above the averaged-embedding MLP, and it correctly separates "not good" from "good".
  You ship a chatbot: a teammate types a message, the model returns POSITIVE or NEGATIVE with
  a confidence score. You also have a reusable recipe for fine-tuning any HuggingFace encoder.

## Deliverables

- Exercise: `C-Transfer-Learning/9-DistilBERT_Chatbot.ipynb`
- Solution: `Solutions/C-Transfer-Learning/9-DistilBERT_Chatbot.ipynb`

## Session Timing (~60-90 min)

| Segment | Cells | Minutes |
|---------|-------|---------|
| Setup + hyperparameters | 0-4 | 8 |
| Why transformers: attention + BERT at a glance | 5-6 | 10 |
| Tokenization demo + Lab 1 | 7-11 | 15 |
| Load SST-2, tokenize, load model | 12-16 | 12 |
| Fine-tune with Trainer + evaluate vs MLP + Lab 2 stretch | 17-19 | 22 |
| Inference + confusion matrix | 20 | 6 |
| Save/reload + Gradio chatbot | 21-22 | 10 |
| Wrap-up + homework | 23 | 5 |

## Cell Migration Map

| Old cell id | Old purpose | New cell N | Action |
|-------------|-------------|-----------:|--------|
| 3e4aa94f | STAR header (CNN headlines) | 0 | edit (retheme to SST-2 + chatbot) |
| 389a1849 | setup md | 1 | edit (add gradio + numpy<2 + restart note) |
| 7f92fc42 | install | 2 | edit (pin versions, add gradio, numpy<2) |
| 4cccaffa | imports/seed/device | 3 | edit (drop scheduler/DataLoader imports) |
| 206c8597 | hyperparameters | 4 | edit (NUM_LABELS=2, model SST-2 framing) |
| c724185d | MLP ceiling md | 5 | edit (retheme to sentiment/negation) |
| a2a48040 + f8de8803 | attention + BERT at a glance md | 6 | edit (merge two md cells into one) |
| f3bfb1c6 | "the plan" md | - | drop (folded into cell 6) |
| e86c11e1 | WordPiece tokenization md | 8 | edit (move after first demo) |
| 38cfaa39 | load tokenizer + demo | 7 | edit (sentiment demo text) |
| 982c4c4e | fields md | 8 | edit (merge with WordPiece md) |
| 1df5b47b | subword split demo | 7 | edit (fold into tokenizer demo) |
| 93cca266 | special token ids | 9 | edit (fold into batch demo) |
| 36a35e92 | batch tokenization | 9 | keep (light edit) |
| aa4fbd63 | Lab 1 instructions md | 10 | keep (light edit) |
| 06ba0a7a | Lab 1 starter | 11 | edit (stronger no-leak hints) |
| 3e47301e | max_length tradeoff md | - | drop (one line folded into cell 8) |
| e7eced9a | dataset section md | 12 | edit (CNN -> SST-2) |
| 3c9284d5 | download CNN csv | 13 | edit (replace with load_dataset glue sst2) |
| 747ed632 | to HF Dataset | 13 | edit (folded into load cell) |
| b948b2a8 | train/val/test split | 13 | edit (use SST-2 native splits + subset) |
| 3a5def2d | tokenize_function + map | 14 | edit (rename label->labels gotcha) |
| fb1a6424 | padding=False md | 14 | edit (folded as comment + short md) |
| 91c87503 | load model md | 15 | keep (light edit) |
| 4dd58e04 | load model code | 16 | keep (NUM_LABELS=2) |
| 94cb6b79 | pre-train forward pass | 16 | edit (folded into load cell) |
| 1d17d0df | fine-tune vs scratch md | 15 | edit (folded into load-model md) |
| 76d08bbb | Trainer section md | 17 | keep (light edit) |
| f7f6081f | compute_metrics | 18 | keep (folded into train cell) |
| 4161fc65 | TrainingArguments | 18 | edit (eval_strategy, fp16 guard) |
| 88a1f2d4 | Trainer instantiate | 18 | edit (processing_class) |
| a59dee36 | trainer.train() | 18 | edit (folded into train cell) |
| c0136109 | evaluate test | 19 | edit (SST-2 validation as test) |
| 746fd697 | compare to MLP md | 19 | edit (folded into eval md, retheme) |
| 34d1014b | Lab 2 epochs md | 19 | edit (becomes stretch: freeze backbone) |
| 5715837d | Lab 2 starter | 19 | edit (becomes freeze-backbone stretch) |
| 066e9a73 | manual loop section md | - | drop (overlaps B6/B7) |
| 04b1d563 | fresh model for manual | - | drop |
| 2cdf8b7a | DataLoader | - | drop |
| fc64129f | optimizer + scheduler | - | drop |
| 8cb932ee | manual training loop | - | drop |
| d96b42e9 | Lab 3 early stopping md | - | drop |
| 8caa73f2 | Lab 3 starter | - | drop |
| ef8e70a7 | inference section md | 20 | edit (folded into inference cell md) |
| 9f3fe599 | pipeline inference | 20 | edit (sentiment headlines, device int) |
| 44545d67 | confusion matrix loop | 20 | edit (use trainer.predict, fold in) |
| 6f01251b | confusions md | 20 | edit (negation framing, fold in) |
| 0c4b79ec | error analysis | 20 | edit (folded into inference cell) |
| 4c13d449 | when is BERT worth it md | 23 | edit (folded into wrap-up table) |
| 5d57693a | save section md | 21 | keep (light edit) |
| 3d8febb1 | save_pretrained | 21 | keep |
| c1637e30 | reload + verify | 21 | keep (light edit) |
| 8554444f | push to hub md | - | drop (out of scope, no keys) |
| 1549f2a5 | deployment tips md | 23 | edit (folded into homework/take-homes) |
| 0f172483 | wrap-up md | 23 | edit (retheme + bridge to C10) |
| (none) | Gradio chatbot | 22 | FROM-NEW |

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~24 cells)

## Cell 0 - Markdown: Title, objectives, STAR story

Provenance: FROM-OLD 3-Text-Classification/9-BERT_Text_Classification.ipynb cell 3e4aa94f
Change: retheme from CNN headlines (4-class) to SST-2 binary sentiment; add the Gradio chatbot
deliverable; renumber to Notebook 9, Part C.

```
# Notebook 9 - DistilBERT Chatbot: Fine-tune a Transformer and Ship It

**Part C - Transfer Learning**
*Author: Axel Sirota, Data Trainers LLC*

## Story (STAR)

- **Situation.** In Notebook 8 you hand-built an MLP on averaged word2vec vectors. It classified
  review sentiment, but it plateaued. It reads "the update is not good" the same as "the update
  is good", because averaging word vectors throws away word order. The product team wants higher
  accuracy before this auto-tags incoming support messages.
- **Task.** Fine-tune a pretrained DistilBERT on the SST-2 sentiment benchmark, beat your MLP
  baseline on the same kind of binary task, then load the fine-tuned model into a simple Gradio
  chatbot a non-technical teammate can use.
- **Action.** Install HuggingFace `transformers`, `datasets`, `evaluate`, `accelerate`, and
  `gradio`. Tokenize with WordPiece via `AutoTokenizer`. Load `distilbert-base-uncased` with a
  2-class head. Fine-tune with the high-level `Trainer`. Evaluate on the held-out SST-2
  validation split. Save the model, reload it, and wrap it in a guarded Gradio interface.
- **Result.** Fine-tuned DistilBERT lands around 90 percent accuracy on SST-2 validation, above
  the averaged-embedding MLP, and it separates "not good" from "good". You ship a chatbot and a
  reusable recipe for fine-tuning any HuggingFace encoder.

## Learning objectives

By the end of this notebook you will be able to:

1. Explain in plain English what attention does and why it beats averaging word vectors.
2. Describe DistilBERT in one breath: encoder-only, 6 layers, 66M params, bidirectional context.
3. Tokenize text with WordPiece subwords and read `input_ids`, `attention_mask`, special tokens.
4. Fine-tune `distilbert-base-uncased` for sequence classification with the HuggingFace `Trainer`.
5. Evaluate against your MLP baseline and decide when a transformer is worth its cost.
6. Save, reload, and wrap the model in a Gradio chatbot.

## Prerequisites

Notebooks B6-B8 (PyTorch nn, MLP on word2vec, Capstone B). You have `device`, `SEED = 42`, and a
working mental model of "embeddings as features". GPU recommended (Runtime -> Change runtime type
-> T4 GPU on Colab); the notebook still runs on CPU, just slower.
```

## Cell 1 - Markdown: Section 0 Environment Setup

Provenance: FROM-OLD cell 389a1849
Change: add gradio, numpy<2, and the Colab restart-runtime warning.

```
## Section 0: Environment Setup

Install dependencies. This pulls `transformers`, `datasets`, `evaluate`, `accelerate`, `gradio`,
and pins `numpy<2`. On a fresh Colab session budget 2-4 minutes the first time.

> **Colab restart note.** Colab now ships NumPy 2.x preinstalled. Because parts of the stack are
> compiled against NumPy 1.x, we pin `numpy<2`. After the install cell finishes you may see
> "You must restart the runtime to use newly installed versions." If so, click **Runtime ->
> Restart runtime**, then run the cells again from the top (skip nothing).

> **Heads-up:** The first call to `from_pretrained('distilbert-base-uncased')` downloads about
> 270 MB of weights from the HuggingFace Hub and caches them under `~/.cache/huggingface/`.
```

## Cell 2 - Code: Install

Provenance: FROM-OLD cell 7f92fc42
Change: pin transformers==4.57.1 (never 5.x), datasets<3, add gradio, add numpy<2.

```python
# Install the fine-tuning + chatbot stack, pinned to versions verified for this course.
# - transformers 4.57.1: AutoTokenizer, AutoModelForSequenceClassification, Trainer.
#   Pinned below 5.x on purpose; 5.x is a breaking release.
# - datasets <3: load_dataset('glue', 'sst2') and .map() tokenization.
# - evaluate: accuracy / F1 metrics.
# - accelerate: required by Trainer for device placement and fp16.
# - gradio: the chatbot UI at the end.
# - scikit-learn: confusion matrix and classification report.
# - numpy<2: avoid the "ndarray size changed" binary-incompatibility error.
!pip install -q "transformers==4.57.1" "datasets>=2.19,<3" "evaluate>=0.4" \
    "accelerate>=0.28" "gradio>=4.0" "scikit-learn>=1.3" "numpy<2"

# If Colab warns about restarting the runtime, do it now (Runtime -> Restart runtime),
# then re-run from the top.
```

## Cell 3 - Code: Imports, seed, device

Provenance: FROM-OLD cell 4cccaffa
Change: drop DataLoader / get_linear_schedule_with_warmup imports (manual loop is cut); keep the
course device/seed convention identical to B6-B8.

```python
# Imports in the course order: data -> model -> training -> evaluation.
import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    pipeline,
)
from datasets import load_dataset
import evaluate

from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# Reproducibility: same SEED convention as every notebook in this course.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device auto-select, same object you have used since B4.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"transformers ready | torch {torch.__version__} | device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Cell 4 - Code: Hyperparameters

Provenance: FROM-OLD cell 206c8597
Change: NUM_LABELS=2 (binary SST-2), retheme comments to sentiment, drop scheduler-only knobs.

```python
# All hyperparameters up front (course convention), so you can tune in one place.
MODEL_NAME   = "distilbert-base-uncased"  # encoder-only, 6 layers, ~66M params, Colab-friendly
NUM_LABELS   = 2        # SST-2 is binary: 0 = negative, 1 = positive
MAX_LENGTH   = 64       # SST-2 sentences are short; 64 subword tokens is plenty
LR           = 2e-5     # canonical transformer fine-tune LR (about 100x smaller than an MLP LR)
NUM_EPOCHS   = 2        # 2-3 epochs is enough for GLUE; more overfits
BATCH_SIZE   = 32       # fits a T4 with fp16; halve if you hit out-of-memory
WEIGHT_DECAY = 0.01     # AdamW decoupled weight decay
TRAIN_SUBSET = 6000     # subsample train for class-time speed (full SST-2 is 67k rows)
FP16         = torch.cuda.is_available()  # half precision only makes sense on GPU

# Label names so predictions read as words, not integers.
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

print(f"Will fine-tune '{MODEL_NAME}' for {NUM_EPOCHS} epochs at LR={LR} on {NUM_LABELS} classes.")
```

## Cell 5 - Markdown: Why transformers - the MLP ceiling

Provenance: FROM-OLD cell c724185d
Change: retheme from headline topics to sentiment + negation, tie directly to B8's MLP.

```
## Section 1: Why Transformers? The MLP Ceiling

In Notebook 8 you averaged the word2vec vectors of a review into one document vector, then fed
that to an MLP. It works, but it has a hard ceiling. Consider two reviews:

1. *"the update is good"*
2. *"the update is not good"*

To a human these are opposites. To your averaged-embedding MLP they are almost the same vector:
mean pooling adds up the same word vectors, and the tiny vector for "not" barely moves the
average. The MLP literally cannot see that "not" flips the meaning, because **averaging destroys
word order**.

A transformer fixes this two ways:

1. **Subword tokenization.** "disappointing" becomes `['disappoint', '##ing']`, so the model
   shares the root "disappoint" across "disappointed", "disappoints", "disappointing".
2. **Contextual embeddings via attention.** The representation of "good" in sentence 2 is
   different from "good" in sentence 1, because the model lets every token look at every other
   token and re-weight them. "good" attends to "not" and learns the phrase is negative.

That is why a fine-tuned BERT-family model beats a static-embedding MLP on sentiment, where
word order and negation decide the label.
```

## Cell 6 - Markdown: Attention intuition + DistilBERT at a glance (merged)

Provenance: FROM-OLD cells a2a48040 + f8de8803 (merged; f3bfb1c6 "the plan" folded in)
Change: merge the two old theory cells into one and append a one-line roadmap, so we never
chain more than the allowed theory before the first code cell (cell 7 is code).

```
### Attention in plain English (no math)

Attention lets every token in a sentence look at every other token and decide how much each one
should contribute to its own updated meaning. In a plain feed-forward net each token's
representation is fixed; in a transformer it is recomputed, layer by layer, as a learned weighted
mix of all positions.

Example: in *"The update shipped late. **It** broke login."*, the token "it" should attend
strongly to "update" to know what "it" refers to. The model learns those weights from data.

The takeaway: every token's hidden state is a learned weighted blend of all the other tokens.
That is how the model "reads context". For classification we add a special `[CLS]` token at
position 0; after the forward pass its hidden state has attended to the whole sentence, and we
feed that one vector into a linear classifier head.

### DistilBERT at a glance

**DistilBERT** is a distilled (compressed) version of BERT:

- **Encoder-only.** It sees the whole sentence at once (bidirectional), unlike a decoder-only
  GPT model that reads strictly left to right.
- **Pretrained** by self-supervision (masked-language modeling: hide 15 percent of tokens and
  predict them). That is where its language knowledge comes from, for free, before you ever
  fine-tune.
- **6 transformer layers** (BERT-base has 12), about **66M parameters**, 40 percent smaller and
  60 percent faster than BERT-base while keeping roughly 97 percent of its quality. That is why
  it fits class time.

Roadmap for this notebook: tokenize -> load the pretrained model with a 2-class head ->
fine-tune with `Trainer` -> evaluate vs the MLP -> save, reload, and ship a Gradio chatbot.
```

## Cell 7 - Code: Load tokenizer + subword demo

Provenance: FROM-OLD cells 38cfaa39 + 1df5b47b (merged)
Change: sentiment demo sentence; fold the rare-word subword split into the same cell so the first
theory block is immediately followed by code in action.

```python
# Load the tokenizer that matches our model. AutoTokenizer picks WordPiece for DistilBERT.
# First call downloads ~470 KB of vocab files.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer class: {tokenizer.__class__.__name__} | vocab size: {len(tokenizer)}")

# Tokenize one sentiment sentence and look at what BERT actually sees.
demo_text = "this update is unbelievably disappointing"
demo = tokenizer(demo_text)
print(f"\nText  : {demo_text}")
print(f"IDs   : {demo['input_ids']}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(demo['input_ids'])}")

# Subword splitting in action: rare words break into pieces with a '##' continuation prefix.
for word in ["unbelievably", "disappointing", "soaring"]:
    print(f"  '{word}' -> {tokenizer.tokenize(word)}")
# '##' means "glue this onto the previous piece". This is how WordPiece shares roots across
# inflections and still represents words it never saw whole.
```

## Cell 8 - Markdown: The fields, special tokens, max_length

Provenance: FROM-OLD cells e86c11e1 + 982c4c4e (merged; 3e47301e max_length folded in)
Change: merge WordPiece explanation and the field dictionary; reduce to one cell.

```
### What the tokenizer returns

The tokenizer hands back a dictionary. The two fields that matter for us:

- **`input_ids`**: the subword IDs. They always start with `[CLS]` (ID 101 for DistilBERT) and
  end with `[SEP]` (ID 102). You saw both in the cell above.
- **`attention_mask`**: a 1/0 mask, 1 for real tokens and 0 for padding. The model ignores every
  position where the mask is 0. This is how a batch of different-length sentences becomes one
  rectangular tensor: pad the short ones, then mask the padding out.

Why subwords are worth it:

1. They handle words the model never saw whole (split into known pieces).
2. They share roots across inflections ("disappoint" inside "disappointing").
3. They keep the vocabulary small (about 30K subwords instead of 100K+ full words).

About `max_length`: longer sequences cost more memory and time. SST-2 sentences are short, so 64
tokens is plenty. Rule of thumb: tweets/headlines 32-64, paragraphs 128-256, long documents 512
(DistilBERT's hard limit).
```

## Cell 9 - Code: Batch tokenization + special token IDs

Provenance: FROM-OLD cells 36a35e92 + 93cca266 (merged)
Change: fold the special-token-id printout into the batch demo; sentiment example texts.

```python
# Special token IDs, confirmed: [CLS]=101, [SEP]=102, [PAD]=0.
print("Special tokens:")
for name, tok, tid in [
    ("CLS", tokenizer.cls_token, tokenizer.cls_token_id),
    ("SEP", tokenizer.sep_token, tokenizer.sep_token_id),
    ("PAD", tokenizer.pad_token, tokenizer.pad_token_id),
]:
    print(f"  [{name}] = {tok} -> id {tid}")

# Tokenize a small batch with padding + truncation, returning PyTorch tensors.
# padding=True   -> pad every sequence to the longest one in this batch.
# truncation=True-> cut anything longer than max_length.
batch_texts = [
    "great product",
    "the support team was helpful and quick",
    "honestly the worst experience i have had with any service this year",
]
enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
print(f"\ninput_ids shape: {tuple(enc['input_ids'].shape)}  (rows=sentences, cols=padded length)")
print(f"first row ids : {enc['input_ids'][0].tolist()}")
print(f"first row mask: {enc['attention_mask'][0].tolist()}  (1=real token, 0=padding)")
```

## Cell 10 - Markdown: Lab 1 instructions (tokenization analysis)

Provenance: FROM-OLD cell aa4fbd63
Change: light edit to sentiment framing; keep the analysis task.

```
### Lab 1: Read what the tokenizer does (15 min)

Tokenize five sentences of varying length and report two numbers, so you build intuition for
padding and truncation before you tokenize a whole dataset.

**Your task:**

1. Write a list of 5 sentences of clearly different lengths (a 2-word one, a medium one, a long
   one, and so on). Sentiment-flavoured is fine.
2. Tokenize them together with padding, truncation, `max_length=32`, and PyTorch tensors.
3. From the result, find which sentence has the **most real (non-padding) tokens**, and compute
   the **padding ratio** of the shortest sentence: number of padding positions divided by 32.

**Verification** (provided) prints the shapes and your two numbers so you can sanity-check.
```

## Cell 11 - Code: Lab 1 starter

Provenance: FROM-OLD cell 06ba0a7a
Change: rewrite hints so they reference role/shape only and never name the field, method, or the
pad id that solves the line (cover-the-solution rule).

```python
# LAB 1: Tokenization analysis.

# 1. Five sentences of clearly different lengths.
lab_sentences = None  # YOUR CODE (a list of five strings, short to long)

# 2. Tokenize them together so they come back as one padded PyTorch tensor batch.
#    Hint: pass the list to the tokenizer and ask it to pad, truncate, cap length at 32,
#    and return tensors. The call returns a dict with the ids and the mask.
lab_enc = None  # YOUR CODE

# 3. Using lab_enc, compute:
#    real_counts      -> for each sentence, how many positions are real (not padding).
#                        Hint: the mask field is 1 on real tokens; a row-wise total gives the count.
#    longest_idx      -> index of the sentence with the most real tokens.
#    shortest_pad_ratio -> for the sentence with the fewest real tokens, the fraction of its 32
#                        positions that are padding.
real_counts = None         # YOUR CODE
longest_idx = None         # YOUR CODE
shortest_pad_ratio = None  # YOUR CODE

# ---- Verification (provided) ----
if lab_enc is not None and real_counts is not None:
    print(f"input_ids shape    : {tuple(lab_enc['input_ids'].shape)}")
    print(f"real-token counts  : {list(map(int, real_counts))}")
    print(f"longest sentence   : index {int(longest_idx)} -> '{lab_sentences[int(longest_idx)]}'")
    print(f"shortest pad ratio : {float(shortest_pad_ratio):.2f}")
```

## Cell 12 - Markdown: Load the SST-2 dataset

Provenance: FROM-OLD cell e7eced9a
Change: replace CNN-headlines framing with GLUE SST-2; explain the hidden-test-label gotcha.

```
## Section 2: Load the SST-2 Sentiment Dataset

We use **SST-2** (Stanford Sentiment Treebank, the GLUE binary version): single sentences
labelled 0 = negative, 1 = positive. It is the standard sentiment benchmark, and it is the same
kind of binary sentiment task your MLP did in Notebook 8, so the comparison is fair.

One gotcha to know up front: GLUE's official **test split has hidden labels** (every label is
`-1`) because the real answers live on a private leaderboard. So we do the standard thing:
**train on the train split and evaluate on the validation split** (872 labelled sentences). We
also subsample the training set to a few thousand rows so fine-tuning finishes inside class time;
the full set is about 67K rows.
```

## Cell 13 - Code: Load SST-2, subset, rename label column

Provenance: FROM-OLD cells 3c9284d5 + 747ed632 + b948b2a8 (replaced)
Change: drop the Dropbox CSV download and sklearn split; load_dataset('glue','sst2'); subset
train; rename "label" -> "labels" (the Trainer gotcha); validation becomes our test set.

```python
# Load SST-2 from the HuggingFace Hub. Fields: 'sentence', 'label', 'idx'.
raw = load_dataset("glue", "sst2")
print(raw)  # DatasetDict with train (67349), validation (872), test (1821, labels = -1)

# Subsample the train split for class-time speed; shuffle with our SEED for reproducibility.
train_ds = raw["train"].shuffle(seed=SEED).select(range(TRAIN_SUBSET))
eval_ds  = raw["validation"]   # 872 labelled rows -> this is our held-out test set

# IMPORTANT gotcha: the model's forward() looks for a column literally named 'labels' to compute
# the loss. SST-2 calls it 'label'. If you skip this rename the Trainer trains on no labels and
# silently learns nothing. Rename it now.
train_ds = train_ds.rename_column("label", "labels")
eval_ds  = eval_ds.rename_column("label", "labels")

print(f"\nTrain rows: {len(train_ds)} | Eval (validation) rows: {len(eval_ds)}")
print(f"Example: {train_ds[0]}")
```

## Cell 14 - Code: Tokenize with .map + dynamic-padding collator

Provenance: FROM-OLD cells 3a5def2d + fb1a6424 (merged)
Change: tokenize SST-2 'sentence'; keep padding=False with DataCollatorWithPadding; explain
dynamic padding inline.

```python
# Tokenize the whole dataset in one pass with .map(batched=True).
# padding=False on purpose: we do NOT pad here. Different batches have different longest
# sentences, so padding everything to MAX_LENGTH now would waste memory. Instead we pad
# dynamically per batch at training time with a data collator (next line).
def tokenize_fn(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=MAX_LENGTH)

train_ds = train_ds.map(tokenize_fn, batched=True)
eval_ds  = eval_ds.map(tokenize_fn, batched=True)

# The collator pads each batch to that batch's longest sequence, not the global max. Faster.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print(f"Tokenized columns: {train_ds.column_names}")
print("Added: input_ids, attention_mask. Kept: labels.")
```

## Cell 15 - Markdown: Load the pretrained model (fine-tune vs scratch)

Provenance: FROM-OLD cells 91c87503 + 1d17d0df (merged)
Change: merge the architecture explanation with the fine-tune-vs-scratch explanation.

```
## Section 3: Load the Pretrained Model

`AutoModelForSequenceClassification` loads the pretrained DistilBERT backbone and bolts a fresh
classification head on top:

```
input -> DistilBERT (6 layers) -> [CLS] hidden state -> Dropout -> Linear(768 -> 2)
```

The **backbone** already knows English from pretraining. The **head** (the final Linear layer)
is **randomly initialized** and gets trained from scratch on sentiment.

Fine-tuning vs training from scratch:

- **From scratch**: every weight random, needs millions of labelled examples. Only big labs do
  this.
- **Fine-tuning**: start from pretrained weights, nudge them on your task. Works with thousands
  of examples. The head trains fully; the backbone updates gently at LR 2e-5 so we do not wipe
  out the pretrained knowledge (catastrophic forgetting).

You will see a warning that some weights are "newly initialized". That is expected: it is the new
classification head.
```

## Cell 16 - Code: Load model + sanity forward pass

Provenance: FROM-OLD cells 4dd58e04 + 94cb6b79 (merged)
Change: NUM_LABELS=2 with id2label/label2id; fold the pre-training forward pass into the same
cell so a beginner sees the untrained head guesses near 50/50.

```python
# Load DistilBERT with a 2-class head. First call downloads ~270 MB of weights.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,   # so predictions come back as POSITIVE / NEGATIVE
    label2id=label2id,
).to(device)

print(f"Loaded {model.__class__.__name__} | params: {sum(p.numel() for p in model.parameters()):,}")

# Sanity check: BEFORE training, the head is random, so it should guess near 50/50.
model.eval()
probe = tokenizer(["this is wonderful", "this is terrible"],
                  padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
with torch.no_grad():
    probs = torch.softmax(model(**probe).logits, dim=-1)
for text, p in zip(["this is wonderful", "this is terrible"], probs):
    print(f"  '{text}' -> {id2label[int(p.argmax())]} (probs {p.cpu().numpy().round(3)})  # untrained = near random")
```

## Cell 17 - Markdown: Fine-tune with the Trainer API

Provenance: FROM-OLD cell 76d08bbb
Change: light edit; note we skip the manual loop on purpose (it was B6/B7's job).

```
## Section 4: Fine-tune with the Trainer

HuggingFace `Trainer` is the high-level wrapper that runs the loop you built by hand in B6 and
B7: forward, loss, `backward()`, optimizer step, evaluation, checkpointing, fp16. You already
know what it does under the hood, so here we just configure it and call `train()`.

You configure three things: a `compute_metrics` function (so it reports accuracy and F1), a
`TrainingArguments` object (epochs, batch size, learning rate, when to evaluate), and the
`Trainer` itself (model + data + tokenizer + collator + metrics). Then `trainer.train()`.
```

## Cell 18 - Code: metrics + TrainingArguments + Trainer + train

Provenance: FROM-OLD cells f7f6081f + 4161fc65 + 88a1f2d4 + a59dee36 (merged)
Change: use `eval_strategy` (not evaluation_strategy), `processing_class=tokenizer` (not
tokenizer=), fp16 guard; combine metrics + args + trainer + train into one runnable cell.

```python
# 1. Metrics the Trainer will compute on the eval split after each epoch.
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# 2. Training configuration.
#    eval_strategy='epoch'   -> evaluate after every epoch (4.46+ name; 'evaluation_strategy' is gone).
#    load_best_model_at_end  -> after training, reload the checkpoint with the best F1.
#    fp16=FP16               -> half precision, GPU only (it errors on CPU, hence the guard).
training_args = TrainingArguments(
    output_dir="./distilbert_sst2",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    fp16=FP16,
    seed=SEED,
    report_to="none",
)

# 3. The Trainer. processing_class=tokenizer is the current name (the old 'tokenizer=' is deprecated).
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 4. Fine-tune. ~3-6 min on a Colab T4; much slower on CPU. Watch loss fall and F1 rise.
trainer.train()
print("\nFine-tuning done. Best checkpoint reloaded automatically.")
```

## Cell 19 - Code: Evaluate vs MLP + Lab 2 stretch (freeze backbone)

Provenance: FROM-OLD cells c0136109 + 746fd697 + 34d1014b + 5715837d (merged)
Change: SST-2 validation as the test set; reframe comparison against B8's MLP; convert the old
"more epochs" lab into the labelled freeze-the-backbone stretch (validated in Cycle 4).

```python
# Evaluate the fine-tuned model on the held-out validation split (our test set).
results = trainer.evaluate(eval_ds)
print("=== DistilBERT on SST-2 validation ===")
print(f"Accuracy: {results['eval_accuracy']:.4f}")
print(f"Macro-F1: {results['eval_f1']:.4f}")

# Compare to your MLP from Notebook 8. The averaged-embedding MLP sat in the low-to-mid 80s on
# binary sentiment; fine-tuned DistilBERT should land around 0.90. The gain comes from attention
# (it reads "not good" as negative) plus pretraining. That is the MLP ceiling, broken.
#
# Cost side of the trade: DistilBERT is ~270 MB and ~10-20 ms per sentence on GPU, versus a
# ~2 MB MLP at well under 1 ms. You buy accuracy with size and latency.

# ---------------------------------------------------------------------------------------------
# STRETCH (fast finishers, ~10 min): freeze the backbone, train only the head.
# Production teams sometimes freeze the pretrained layers and train just the classifier: far
# faster and lighter, usually a few points less accurate. Try it and measure the gap.
#
#   frozen = AutoModelForSequenceClassification.from_pretrained(
#       MODEL_NAME, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id).to(device)
#   for p in frozen.distilbert.parameters():   # freeze the 6-layer backbone
#       p.requires_grad = False                # keep frozen.classifier trainable
#   # Build a second Trainer with `model=frozen` and the SAME args/data, then train + evaluate.
#   # Question to answer: how many accuracy points did you give up to gain the speed?
# ---------------------------------------------------------------------------------------------
```

## Cell 20 - Code: Inference pipeline + confusion matrix + error look

Provenance: FROM-OLD cells 9f3fe599 + 44545d67 + 6f01251b + 0c4b79ec (merged)
Change: use trainer.predict() for the confusion matrix (drop the hand-rolled eval loop); sentiment
examples incl. a negation case; device int for pipeline().

```python
# Quick human-readable inference with a pipeline. Note: pipeline's device wants an INT
# (0 = first GPU, -1 = CPU), not a torch.device object, so convert.
clf = pipeline("text-classification", model=model, tokenizer=tokenizer,
               device=0 if torch.cuda.is_available() else -1)

probe_texts = [
    "this is the best support i have ever received",
    "the app keeps crashing and nobody replies",
    "the update is good",
    "the update is not good",   # negation: the MLP missed this; the transformer should not
]
for t in probe_texts:
    out = clf(t, top_k=1)[0]
    print(f"{t:<52} -> {out['label']} ({out['score']:.3f})")

# Confusion matrix on the validation set, straight from trainer.predict (no manual loop needed).
pred = trainer.predict(eval_ds)
y_pred = np.argmax(pred.predictions, axis=-1)
y_true = pred.label_ids

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NEGATIVE", "POSITIVE"], yticklabels=["NEGATIVE", "POSITIVE"], square=True)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("DistilBERT SST-2 Confusion Matrix")
plt.tight_layout(); plt.show()

print(classification_report(y_true, y_pred, target_names=["NEGATIVE", "POSITIVE"]))
# Off-diagonal cells are the mistakes. Many will be genuinely ambiguous one-liners; a systematic
# block (e.g. all positives called negative) would mean a bug in labels or the rename step.
```

## Cell 21 - Code: Save and reload

Provenance: FROM-OLD cells 3d8febb1 + c1637e30 (merged)
Change: light edit; note safetensors + that id2label travels in config.json so reload returns
human-readable labels.

```python
# Save the fine-tuned model and tokenizer. save_pretrained writes config.json (which carries our
# id2label map) plus model.safetensors (the weights). The chatbot will reload from this folder.
SAVE_DIR = "./distilbert_sst2_final"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved to {SAVE_DIR}/  ->  {sorted(os.listdir(SAVE_DIR))}")

# Reload to prove the round trip works and that labels come back as words (from config.json).
reloaded = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
reloaded_tok = AutoTokenizer.from_pretrained(SAVE_DIR)
check = tokenizer("support was fantastic today", return_tensors="pt").to(device)
reloaded.eval()
with torch.no_grad():
    pid = int(reloaded(**check).logits.argmax(-1))
print(f"Reloaded model predicts: {reloaded.config.id2label[pid]}")
```

## Cell 22 - Markdown + Code: The Gradio chatbot (guarded)

Provenance: FROM-NEW

```
## Section 5: Ship It - A Gradio Chatbot

Time to deliver the artifact this whole course builds toward. We wrap the reloaded model in a
`pipeline`, then hand that pipeline to Gradio, which auto-builds a textbox-in / label-out web UI.
A teammate types a support message; the model returns POSITIVE or NEGATIVE with a confidence.

We guard the UI in `try/except ImportError` so the notebook still runs end to end in an
environment without Gradio. On Colab, `.launch()` prints a public share link automatically.
```

```python
# Build the inference pipeline from the reloaded, saved model (same recipe a real service uses).
chat_pipe = pipeline("text-classification", model=SAVE_DIR,
                     device=0 if torch.cuda.is_available() else -1)

def classify_message(text):
    """Return a human-readable verdict for one message."""
    if not text or not text.strip():
        return "Type a message to classify."
    out = chat_pipe(text, top_k=1)[0]
    return f"{out['label']}  (confidence {out['score']:.2f})"

# Quick sanity check that works with or without Gradio installed.
print(classify_message("the new release fixed my problem instantly"))
print(classify_message("i have been waiting three days for a reply"))

# Launch the chatbot UI, guarded so a no-Gradio environment still runs everything above.
try:
    import gradio as gr
    demo = gr.Interface(
        fn=classify_message,
        inputs=gr.Textbox(lines=2, placeholder="Type a support message..."),
        outputs=gr.Textbox(label="Sentiment"),
        title="DistilBERT Sentiment Chatbot",
        description="Fine-tuned distilbert-base-uncased on SST-2. POSITIVE or NEGATIVE with confidence.",
    )
    # On Colab a public share link is created automatically. share=True forces it elsewhere.
    demo.launch(share=True)
except ImportError:
    print("\n[Gradio not installed] Skipping the UI. classify_message() still works above.")
```

## Cell 23 - Markdown: Wrap-up, when-to-use table, homework, bridge to C10

Provenance: FROM-OLD cells 0f172483 + 4c13d449 + 1549f2a5 (merged)
Change: retheme to sentiment + chatbot; fold the "when is BERT worth it" table and deployment
tips into take-homes; add the validated abstention homework; bridge to C10.

```
## Wrap-up

### What you can now ship

- A fine-tuned transformer sentiment classifier that beats your averaged-embedding MLP.
- A reusable recipe: swap `MODEL_NAME` and `num_labels` to fine-tune any HuggingFace encoder
  (RoBERTa, DeBERTa, ELECTRA) on any classification task.
- A working Gradio chatbot wrapping your model, the course's end-deliverable shape.

### Key takeaways

| Concept | Remember |
|---------|----------|
| Attention | Tokens re-weight each other's features, so "good" and "not good" differ. |
| WordPiece | Splits rare words into `##` subwords and shares roots across inflections. |
| Fine-tune LR | 2e-5 is standard, about 100x smaller than an MLP LR. Too high -> NaN / no convergence. |
| Epochs | 2-3 for GLUE; 6+ overfits. |
| labels column | The model wants a column named `labels`; rename SST-2's `label` or it learns nothing. |
| Trainer | High-level loop; `eval_strategy`, `processing_class` are the current arg names. |
| When BERT | >1K labels, context/negation matters, GPU budget. Else keep the MLP (sub-ms, 2 MB). |

### Homework (async, deeper)

Add a **confidence gate** to the chatbot for the support-platform. Compute the softmax
confidence of each prediction; if it is below a threshold (say 0.65), do not auto-tag. Return
"uncertain - route to a human agent" instead. This is the standard selective-prediction / reject
-option pattern: the model answers only when confident and defers the rest to a person. Measure
how coverage (fraction auto-handled) trades off against accuracy as you move the threshold.

Production take-homes worth reading: quantize the model to int8 (about 4x smaller, 2-3x faster,
negligible accuracy loss on classification); batch your inputs instead of one at a time;
host the Gradio app on a HuggingFace Space; monitor average confidence over time to catch drift.

### Next: C10

Classification gives a label. A real assistant *writes* an answer. In C10 you fine-tune an
encoder-decoder model (T5 / BART) on a Q&A dataset and drop it into this same Gradio shell, so
the chatbot generates answers instead of picking a class.
```

# VERIFICATION CHECKLIST

- [ ] Cell 2 install pins `transformers==4.57.1` (not 5.x), `datasets<3`, adds `gradio`, `numpy<2`.
- [ ] Colab restart-runtime note present (cell 1) because numpy is downgraded.
- [ ] `SEED = 42`, `device`, and the seed calls match the B6-B8 convention exactly (cell 3).
- [ ] `NUM_LABELS = 2`, `id2label`/`label2id` defined once (cell 4).
- [ ] Tokenizer demo shows `##` subword splitting and special token ids (cells 7, 9).
- [ ] `load_dataset('glue','sst2')`; train subset; `rename_column('label','labels')` (cell 13).
- [ ] `tokenize_fn` uses `truncation`, no upfront padding; `DataCollatorWithPadding` (cell 14).
- [ ] `AutoModelForSequenceClassification` with `num_labels=2`, moved to device (cell 16).
- [ ] `TrainingArguments` uses `eval_strategy` (not evaluation_strategy); `fp16=FP16` guard (cell 18).
- [ ] `Trainer` uses `processing_class=tokenizer` (not deprecated `tokenizer=`) (cell 18).
- [ ] Evaluate on the validation split (SST-2 test labels are -1) (cells 13, 19).
- [ ] Confusion matrix uses `trainer.predict` (no hand-rolled eval loop) (cell 20).
- [ ] `pipeline(..., device=0 if cuda else -1)` uses an int, not a torch.device (cells 20, 22).
- [ ] `save_pretrained` + reload; id2label survives via config.json (cell 21).
- [ ] Gradio cell guarded with try/except ImportError; runs inference without the UI (cell 22).
- [ ] Lab 1 starter hints never name the field/method/pad id that solves the line (cell 11).
- [ ] Exactly one in-class stretch (freeze backbone, cell 19) and one async homework (abstention, cell 23).
- [ ] No more than 3 theory cells chained without code (cells 5-6 -> code 7; cell 8 -> code 9).
- [ ] Plain ASCII throughout: no em/en dashes, no Unicode multiplication, no emoji.
- [ ] Every cell tagged FROM-OLD (with cell id) or FROM-NEW; Cell Migration Map present.

# RESEARCH VALIDATED (June 2026)

- DistilBERT fine-tuned on SST-2 reaches ~91.3 percent dev accuracy; load with
  AutoModelForSequenceClassification, num_labels=2. Source:
  https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- SST-2 fields are `sentence`, `label`, `idx`; 67349 train / 872 validation / 1821 test rows.
  Source: https://huggingface.co/docs/datasets/package_reference/loading_methods
- GLUE SST-2 test labels are hidden (`-1`); standard practice is to evaluate on the validation
  split. Source: https://github.com/huggingface/datasets/issues/245 and
  https://discuss.huggingface.co/t/sst2-dataset-labels-look-worng/10895/2
- `evaluation_strategy` was deprecated and removed at transformers 4.46; use `eval_strategy`.
  Source: https://discuss.huggingface.co/t/solved-difference-between-eval-strategy-and-evaluation-strategy/96657
- Trainer `tokenizer=` is deprecated in favor of `processing_class=` (since ~4.46). Source:
  https://github.com/huggingface/transformers/issues/37734
- Pinning `numpy<2` on Colab (which now ships NumPy 2.x) requires a runtime restart to avoid the
  "ndarray size changed" binary-incompatibility error. Source:
  https://github.com/googlecolab/colabtools/issues/5205 and
  https://numpy.org/doc/stable/user/troubleshooting-importerror.html
- `gr.Interface.from_pipeline(pipe).launch()` builds a UI from a transformers pipeline; Colab
  auto-creates a share link. Source: https://www.gradio.app/guides/using-hugging-face-integrations
- BERT fine-tune learning rates are 2e-5 / 3e-5 / 5e-5; high LR (>=4e-4) fails to converge and
  can NaN. Source: https://arxiv.org/pdf/1905.05583 (Sun et al., How to Fine-Tune BERT)
- Devlin et al. recommend 2-3 epochs for GLUE fine-tuning; 6+ epochs overfit. Source:
  https://machinelearningmastery.com/fine-tuning-a-bert-model/
- distilbert-base-uncased tokenizer: WordPiece, 30K vocab, [CLS]=101, [SEP]=102, `##` continuation
  prefix. Source: https://huggingface.co/distilbert/distilbert-base-uncased and
  https://huggingface.co/transformers/v4.2.2/model_doc/distilbert.html
- Freeze the backbone via `model.base_model`/`model.distilbert` parameters `requires_grad=False`,
  keeping `model.classifier` trainable (feature-extraction mode). Source:
  https://github.com/huggingface/transformers/issues/400
- Confidence-threshold abstention (MaxProb selective prediction / reject option) is a standard
  pattern; softmax probability is the canonical confidence signal. Source:
  https://medium.com/@nvarshney97/selective-prediction-in-natural-language-processing-838244f1e8e1
- Pipeline `device` is an int: 0 for GPU, -1 for CPU. save_pretrained writes config.json (carries
  id2label) + model.safetensors; from_pretrained reloads the label map. Source:
  https://huggingface.co/docs/transformers/pipeline_tutorial
- Production: int8 quantization is ~4x smaller / 2-3x faster with negligible classification
  accuracy loss; batch inputs for throughput. Source:
  https://medium.com/@bhagyarana80/optimizing-transformer-inference-with-onnx-runtime-and-quantization-098f8149a15c
```
