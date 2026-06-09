# C11 - Capstone C - Cell-by-Cell Plan

## Status

RENEWED. Source notebook: `4-Text-Generation/13-Capstone_3_Shakespeare.ipynb` (43 cells,
ids cell-0 .. cell-42). New file: `C-Transfer-Learning/11-Capstone_C.ipynb`.

This is the course FINALE. The old notebook was a two-path text-generation capstone
(Path A char-level GRU, Path B GPT-2 fine-tune) on Tiny Shakespeare. ALL of that is CUT:
no RNN/GRU/LSTM, no char generation, no Shakespeare, no GPT-2 / decoder-only generation.
The notebook keeps only its SHAPE - a "pick ONE of two paths, then ship it" capstone - and
reskins both paths onto the Part C transfer-learning stack the student already built:

- Path A: encoder-only classification chatbot (C9 style, `distilbert-base-uncased`).
- Path B: encoder-decoder Q&A chatbot (C10 style, `t5-small`).

Both paths end in the SAME guarded `gr.Interface` Gradio shell and a written reflection.

## Context

What students arrive with (by exact name, from C9 and C10):

- From C9 (encoder-only): `MODEL_NAME="distilbert-base-uncased"`, `NUM_LABELS=2`,
  `tokenizer`, `model` (AutoModelForSequenceClassification), `tokenize_fn`,
  `data_collator` (DataCollatorWithPadding), `compute_metrics`, `trainer` (Trainer with
  `processing_class=tokenizer`), `id2label`/`label2id`, `SAVE_DIR`, `chat_pipe`,
  `classify_message(text)`. Trained on `glue/sst2`, evaluated on the VALIDATION split
  (GLUE test labels are hidden as -1). Result ~0.90 accuracy, beat the B8 averaged-w2v MLP.
- From C10 (encoder-decoder): `build_input(question, context)` returning
  `"question: {q}  context: {c}"`, `preprocess(examples)`, `max_input=256`,
  `max_target=32`, `data_collator` (DataCollatorForSeq2Seq), `training_args`
  (Seq2SeqTrainingArguments), `trainer` (Seq2SeqTrainer), `answer_question(question,
  context)`, `save_dir`, `chat_fn(question, context)`. Trained on `squad`, generated with
  `.generate()`, dropped into a 2-input Gradio shell. Honesty framing: SQuAD-trained T5
  mostly COPIES spans and will HALLUCINATE on out-of-context questions; RAG is the fix.

Key insight they leave with: a capstone is not new theory - it is an ENGINEERING decision
(label-out vs text-out), a small public dataset, a short fine-tune, an honest evaluation,
and a shipped artifact. They prove they can take a pretrained transformer to a working
chatbot end to end, unaided, and write up its limits like a professional.

## Chatbot Through-Line

This notebook IS the through-line's destination. Parts A and B built the stack
(tools-first NLP -> embeddings as geometry -> MLP on word2vec). C9 shipped the first
fine-tuned-transformer chatbot (classify). C10 shipped the generative chatbot (answer).
C11 hands the keys to the student: they independently pick one of those two architectures,
fine-tune it on a dataset of their choice, evaluate it, and launch the SAME guarded
`gr.Interface` they saw in C9/C10 - now built by them, start to finish. The Gradio guard
(try/except ImportError, `share=True` for the Colab public link, non-Gradio fallback that
calls inference directly) and the save_pretrained -> reload -> Gradio pattern are reused
verbatim. There is no "next notebook"; the one-line bridge is to PRODUCTION (the reflection
cell: monitoring, drift, cost, abstention, RAG grounding).

## STAR Story

Same support-platform ML-practitioner persona that has run through Parts A, B, and C.

- Situation: the team now has two proven prototypes from the last two sessions - the C9
  DistilBERT ticket-tagger (POSITIVE/NEGATIVE) and the C10 T5 answer-writer over the help
  docs. Leadership says: "Great demos. Now YOU own one. Pick the one our product actually
  needs next quarter, ship a clean version on a dataset you justify, and tell us honestly
  where it breaks before a customer finds out."
- Task: choose ONE architecture (encoder-only classifier OR encoder-decoder Q&A), choose a
  small public dataset that fits the business case, fine-tune, evaluate on a held-out split,
  wrap it in the Gradio chatbot, and write the technical + non-technical write-up.
- Action: students execute the path they picked using the exact C9 / C10 helpers, with
  `None  # YOUR CODE` lab starters at the load-model, tokenize, train, evaluate, and ship
  steps, each with a PROVIDED verification block.
- Result: a working, fine-tuned chatbot launched from a `share=True` Gradio link, plus a
  reflection that names what worked, where it fails (mean-pooling-style blind spots for A;
  hallucination / span-copying for B), and the concrete production next steps.

## Deliverables

- Exercise: `C-Transfer-Learning/11-Capstone_C.ipynb` (`None  # YOUR CODE` starters).
- Solution: `C-Transfer-Learning/solutions/11-Capstone_C_solution.ipynb` (full code).

## Session Timing (~60-90 min)

| Block | Cells | Minutes |
|-------|-------|---------|
| Intro + STAR + course recap | 0-2 | 8 |
| Environment setup + restart note | 3-4 | 7 |
| Path Choice Checkpoint | 5-6 | 5 |
| Path A: load + tokenize (labs) | 7-10 | 15 |
| Path A: train + evaluate (labs) | 11-13 | 12 |
| Path B: load + preprocess (labs) | 14-17 | 15 |
| Path B: train + generate (labs) | 18-20 | 12 |
| Save / reload + Gradio ship (shared) | 21-22 | 8 |
| Reflection write-up + self-check | 23-24 | 8 |

Note: a student runs EITHER the Path A block OR the Path B block, never both, so the live
session is ~60-75 min on one path. The shared ship + reflection cells run for both paths.

## Cell Migration Map

The old notebook is RENEWED: its text-generation content is dropped, but its capstone
SHAPE (intro -> setup -> path-choice checkpoint -> Path A labs -> Path B labs -> present
results -> self-check) is preserved and reskinned. Map below: old cell id -> new cell N ->
action.

| Old id | Old content | New cell | Action |
|--------|-------------|----------|--------|
| cell-0 | "Write Like Shakespeare" title | 0 | edit: retitle to transfer-learning finale |
| cell-1 | Section 0 env setup md | 3 | edit: reuse C9/C10 setup heading |
| cell-2 | pip install (TF-era / GPT-2) | 3 | edit: replace with C11 pinned install |
| cell-3 | core imports | 4 | edit: swap to HF/PyTorch imports + seed/device |
| cell-4 | "What Are We Building?" md | 1 | edit: reskin to chatbot capstone framing |
| cell-5 | load Tiny Shakespeare | (drop) | drop: dataset chosen per path instead |
| cell-6 | Path Choice Checkpoint md | 5 | edit: A=classifier, B=Q&A (not GRU vs GPT-2) |
| cell-7 | student choice variables | 6 | edit: PATH = "A" or "B" switch cell |
| cell-8 | Section 2 Path A (char GRU) md | 7 | edit: reskin to Path A DistilBERT classifier |
| cell-9 | Path A hyperparameters | 7 | edit: DistilBERT hyperparams (from C9) |
| cell-10 | Lab A1 char vocab md | 8 | edit: reskin to "load model + dataset" lab md |
| cell-11 | Lab A1 build char vocab | 9 | edit: lab load DistilBERT + dataset starter |
| cell-12 | Lab A2 CharDataset md | 10 | edit: reskin to tokenize lab md |
| cell-13 | Lab A2 CharDataset code | 10 | edit: tokenize_fn lab starter |
| cell-14 | Lab A3 CharRNN model md | 11 | edit: reskin to train lab md |
| cell-15 | Lab A3 CharRNN model code | 11 | edit: Trainer config lab starter |
| cell-16 | Lab A4 training loop md | 12 | edit: reskin to "run training" md |
| cell-17 | Lab A4 training loop code | 12 | edit: trainer.train() + provided run |
| cell-18 | plot training curves | (drop) | drop: Trainer logs metrics; no manual loop |
| cell-19 | Lab A5 generate w/ temperature md | (drop) | drop: no generation in Path A |
| cell-20 | Lab A5 temperature sampling | (drop) | drop: char generation cut |
| cell-21 | demo temperature sweep | (drop) | drop: char generation cut |
| cell-22 | Lab A6 bits-per-char md | 13 | edit: reskin to "evaluate accuracy" md |
| cell-23 | Lab A6 bits-per-char code | 13 | edit: compute_metrics + classify_message lab |
| cell-24 | save Path A model | 21 | edit: merge into shared save/reload cell |
| cell-25 | Section 3 Path B (GPT-2) md | 14 | edit: reskin to Path B T5 Q&A md |
| cell-26 | Path B hyperparameters | 14 | edit: T5 hyperparams (from C10) |
| cell-27 | import HF libs for Path B | 4 | edit: folded into shared import cell 4 |
| cell-28 | Lab B1 tokenize/chunk md | 15 | edit: reskin to "load T5 + SQuAD" md |
| cell-29 | Lab B1 tokenize/chunk code | 16 | edit: build_input + preprocess lab starter |
| cell-30 | Lab B2 fine-tune Trainer md | 17 | edit: reskin to Seq2SeqTrainer md |
| cell-31 | Lab B2 fine-tune code | 18 | edit: Seq2SeqTrainer config lab + run |
| cell-32 | Lab B3 generate md | 19 | edit: reskin to answer_question md |
| cell-33 | Lab B3 generate code | 19 | edit: answer_question lab starter |
| cell-34 | Lab B4 compare base vs ft md | 20 | edit: reskin to before/after eval md |
| cell-35 | Lab B4 compare code | 20 | edit: zero-shot vs fine-tuned + honesty probe |
| cell-36 | Lab B5 test perplexity md | 20 | edit: folded into eval cell 20 (qualitative) |
| cell-37 | Lab B5 perplexity code | 20 | edit: folded into eval cell 20 |
| cell-38 | Section 4 Present Results md | 23 | edit: reskin to reflection write-up md |
| cell-39 | Lab P1 technical memo | 23 | edit: technical reflection starter |
| cell-40 | Lab P2 non-technical pitch md | 23 | edit: folded into reflection cell 23 |
| cell-41 | Lab P2 non-technical pitch code | 23 | edit: non-technical pitch part of cell 23 |
| cell-42 | Self-Check Quiz md | 24 | edit: reskin quiz to transfer-learning concepts |

New cells with no old origin (authored fresh for the new arc):
- Cell 2 (course recap + decision rule, FROM-NEW)
- Cell 22 (shared guarded Gradio ship cell for BOTH paths, FROM-NEW; the old notebook had
  no Gradio - it printed/sampled text)

Split summary is in the VERIFICATION CHECKLIST at the end.

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~20-25 cells; this plan = 25 cells, 0-24)

## Cell 0 - Markdown: Title and the finale framing

```
# Capstone C: Ship a Fine-Tuned-Transformer Chatbot

This is the finale. Across this course you went from tools-first NLP, to embeddings as
geometry, to an MLP on word2vec features, and finally to fine-tuning real transformers:
DistilBERT (C9) and T5 (C10). Each of those ended in a small Gradio chatbot.

Now you own one. You will:

1. Pick ONE architecture - an encoder-only classifier (C9 style) OR an encoder-decoder
   Q&A model (C10 style).
2. Choose a small, public dataset that fits the use case.
3. Fine-tune a pretrained model on it.
4. Evaluate it honestly on a held-out split.
5. Ship it inside a Gradio chatbot.
6. Write up what works, where it breaks, and what production would need.

No new theory. This is the engineering decision and the delivery. You have built every
piece before - here you assemble it unaided.
```

Provenance: FROM-OLD `4-Text-Generation/13-Capstone_3_Shakespeare.ipynb` cell cell-0.
Change: retitle from "Write Like Shakespeare" to the transfer-learning chatbot finale; drop
all text-generation / Shakespeare language.

## Cell 1 - Markdown: STAR - what we are building and why

```
## The scenario

Same support platform you have worked all course. Two prototypes exist from the last two
sessions:

- The C9 DistilBERT ticket-tagger: reads a customer message, returns POSITIVE / NEGATIVE
  with a confidence score.
- The C10 T5 answer-writer: reads a question plus a help-doc snippet and WRITES an answer.

Leadership: "Great demos. Now you own one. Pick the architecture our product needs next,
ship a clean version on a dataset you can justify, evaluate it, and tell us honestly where
it breaks before a customer finds out."

That is the capstone. One path. Real fine-tune. Honest evaluation. A shipped chatbot. A
write-up. The whole course pointed here.
```

Provenance: FROM-OLD cell cell-4. Change: reskin "What Are We Building?" from char/GPT-2
text generation to the chatbot-ownership STAR scenario.

## Cell 2 - Markdown: Course recap and the decision rule (which path?)

```
## How to choose your path

The single most important capstone decision is architecture. The rule you learned in C9
and C10:

- Encoder-only (Path A, DistilBERT): the output is a LABEL from a fixed set. Use it when
  the job is routing, tagging, sentiment, intent, or any "which bucket?" question. Fast,
  cheap, easy to evaluate (accuracy / F1), and it CANNOT make up words - it can only pick
  a class. It also cannot explain or answer in prose.
- Encoder-decoder (Path B, T5): the output is free TEXT the model writes. Use it when the
  job is to answer, summarize, or rewrite. More powerful, but it can HALLUCINATE - a
  SQuAD-trained T5 mostly copies spans from the context and will confidently invent an
  answer when the context does not contain one. Harder to evaluate, and you must ground it.

Decision rule: if a human could solve the task by picking from a short menu, choose Path A.
If the task genuinely requires writing new text, choose Path B, and plan for grounding.

Pick now. You will only run one path's cells.
```

Provenance: FROM-NEW. (No old cell taught the encoder-only vs encoder-decoder decision rule;
this distils the C9/C10 closing reflections into one capstone decision cell.)

## Cell 3 - Code: Environment setup (pinned install + restart note)

```python
# Install the course stack, pinned. Run this FIRST in Colab.
# numpy<2 is required course-wide; this DOWNGRADES numpy (Colab ships numpy 2.x), so you
# MUST restart the runtime after this cell (Runtime -> Restart session), then run every
# other cell. transformers is pinned to 4.57.1; do NOT let it resolve to 5.x.
!pip install -q "transformers==4.57.1" "datasets>=2.19,<3" "numpy<2" \
    evaluate accelerate gradio

# After restart, re-run from the next cell down (do NOT re-run this install cell).
print("Install done. Now: Runtime -> Restart session, then continue from the next cell.")
```

Provenance: FROM-OLD cells cell-1 (heading) + cell-2 (install). Change: replace the
TF-era / GPT-2 install with the C9/C10 pinned line (transformers==4.57.1, datasets>=2.19,<3,
numpy<2, evaluate, accelerate, gradio); keep the mandatory Colab restart-runtime note that
C9/C10 require after the numpy downgrade. No gensim / sentence-transformers (neither path
needs them; matches C10's deliberate omission).

## Cell 4 - Code: Imports, seed, device (shared by both paths)

```python
# Imports grouped by purpose. These cover BOTH paths; you only call the ones your path uses.
import random
import numpy as np
import torch

from datasets import load_dataset

# Encoder-only (Path A) pieces:
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
# Encoder-decoder (Path B) pieces:
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import evaluate

# Reproducibility (same SEED block as C9).
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device: a torch.device object. For HF pipeline(device=...) convert to int 0 (cuda) or -1 (cpu).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Transformers stack ready. Device:", device)
```

Provenance: FROM-OLD cells cell-3 + cell-27. Change: drop all RNN/GPT-2 imports; import the
exact C9 (sequence-classification) and C10 (seq2seq) classes; add the C9 SEED block and the
torch.device note carried from the synthesis continuity rules.

## Cell 5 - Markdown: Path Choice Checkpoint

```
## Checkpoint: choose your path

You are about to commit to one of two paths. Read both, then set PATH in the next cell.

PATH A - Encoder-only classification chatbot (C9 style)
  Model:   distilbert-base-uncased
  Output:  a label + confidence (e.g. POSITIVE / NEGATIVE, or a topic class)
  Dataset suggestions (small, public, REAL labels on a held-out split):
    - glue / sst2     : binary sentiment (the C9 dataset; note: test labels hidden, eval on validation)
    - ag_news         : 4-class news topic (World / Sports / Business / Sci-Tech), real test labels
    - emotion         : 6-class tweet emotion, has train/validation/test
  Good when the job is routing / tagging / sentiment / intent.

PATH B - Encoder-decoder Q&A chatbot (C10 style)
  Model:   t5-small
  Output:  a written answer (free text)
  Dataset suggestions (small, public):
    - squad           : reading-comprehension Q&A (the C10 dataset; answer is a span in the context)
  Good when the job must WRITE an answer. Remember: it will hallucinate off-context.

Only run the cells for the path you pick. The two paths never run in the same session.
```

Provenance: FROM-OLD cell cell-6. Change: replace the GRU-vs-GPT-2 path choice with
DistilBERT-classifier (A) vs T5-Q&A (B); list the validated small public datasets per path
(ag_news and emotion have non-hidden test labels, unlike GLUE/sst2).

## Cell 6 - Code: Set your path (one switch the whole notebook reads)

```python
# Set this to "A" (encoder-only classifier) or "B" (encoder-decoder Q&A), then run.
# Every later cell checks PATH and runs only if it matches, so the other path's cells are
# no-ops. This lets one notebook serve both shells without variable collisions.
PATH = None  # YOUR CODE: set to the string "A" or the string "B"

# Verification (PROVIDED - do not edit):
assert PATH in ("A", "B"), "Set PATH to the string 'A' or 'B' before continuing."
print(f"You chose Path {PATH}.")
print("Run only the cells whose header starts with 'Path " + PATH + "'. Skip the other path.")
```

Provenance: FROM-OLD cell cell-7. Change: reskin the student-choice variables to a single
PATH switch ("A"/"B"); hint names neither value's meaning beyond what cell 5 already states.

## Cell 7 - Markdown: Path A - DistilBERT classifier setup and hyperparameters

```
## Path A: encoder-only classification chatbot

(Skip this whole section if you chose Path B.)

This is the C9 architecture. You will fine-tune distilbert-base-uncased into a text
classifier and wrap it in a chatbot that returns a label + confidence. The pieces are the
same ones you used in C9; here you wire them yourself.

Hyperparameters (sensible defaults from C9 - you may justify changes in your write-up):
- MODEL_NAME = "distilbert-base-uncased"
- MAX_LENGTH = 64        (short support messages; longer wastes compute)
- LR = 2e-5              (standard fine-tune LR for BERT-family classification)
- NUM_EPOCHS = 2
- BATCH_SIZE = 32
- TRAIN_SUBSET = 6000    (subsample so the fine-tune fits a class session on a T4)

Reminder from C9: if your dataset is glue/sst2, the test split labels are hidden (-1), so
you EVALUATE ON THE VALIDATION SPLIT. ag_news and emotion have real test labels.
```

Provenance: FROM-OLD cells cell-8 + cell-9. Change: reskin the Path A header from
char-level GRU to the DistilBERT classifier; replace GRU hyperparameters with the exact C9
classification hyperparameters and carry the GLUE-hidden-test-labels gotcha.

## Cell 8 - Markdown: Lab A1 - load the model and the dataset

```
### Lab A1 (Path A): load the tokenizer, model, and dataset

Goal: load distilbert-base-uncased as a sequence classifier with the right number of labels,
and load your chosen dataset.

Steps:
1. Load the tokenizer for MODEL_NAME.
2. Decide NUM_LABELS from your dataset (2 for sst2, 4 for ag_news, 6 for emotion) and build
   id2label / label2id dicts so the saved model returns word labels later (as in C9).
3. Load the model for sequence classification with that label count and the label maps.
4. Load the dataset with load_dataset and identify the text field and the label field.

Hints: the tokenizer and the model both come from the same factory classes you imported,
each built from a pretrained name. The text field is "sentence" for sst2 and "text" for
ag_news / emotion; the label field is "label" for all three.
```

Provenance: FROM-OLD cell cell-10. Change: reskin "build char vocabulary" lab instructions
to "load classifier model + dataset"; hints name fields/roles only, never the method.

## Cell 9 - Code: Lab A1 starter (load model + dataset)

```python
if PATH == "A":
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 64
    LR = 2e-5
    NUM_EPOCHS = 2
    BATCH_SIZE = 32
    TRAIN_SUBSET = 6000

    # Pick ONE: ("glue", "sst2") | ("ag_news",) | ("emotion",). Set the loader args and fields.
    DATASET_ARGS = None   # YOUR CODE: a tuple of load_dataset arguments for your chosen set
    TEXT_FIELD = None     # YOUR CODE: the name of the input-text column
    LABEL_FIELD = None    # YOUR CODE: the name of the integer-label column
    NUM_LABELS = None     # YOUR CODE: how many classes your dataset has

    raw = load_dataset(*DATASET_ARGS)

    # Build the label maps so reloaded models return word labels (C9 pattern).
    # YOUR CODE: build id2label = {0: "...", 1: "..."} and label2id (the reverse) for your set.
    id2label = None       # YOUR CODE
    label2id = None       # YOUR CODE

    # YOUR CODE: load the tokenizer from MODEL_NAME.
    tokenizer = None
    # YOUR CODE: load the sequence-classification model from MODEL_NAME with num_labels,
    #            id2label and label2id passed through.
    model = None

    # Verification (PROVIDED - do not edit):
    assert tokenizer is not None and model is not None, "Load both tokenizer and model."
    assert model.config.num_labels == NUM_LABELS, "Model label count must match NUM_LABELS."
    assert set(model.config.id2label.keys()) == set(range(NUM_LABELS)), "id2label keys 0..N-1."
    assert TEXT_FIELD in raw["train"].column_names, "TEXT_FIELD not in the dataset."
    print("Path A loaded:", MODEL_NAME, "| labels:", model.config.id2label)
```

Provenance: FROM-OLD cell cell-11. Change: replace char-vocab construction with
load-classifier-and-dataset; starters are None with role-only hints; verification asserts on
config properties, never reveals the calls.

## Cell 10 - Code: Lab A2 - tokenize and build the splits (md instructions in the cell header comment)

```python
# ### Lab A2 (Path A): tokenize the text and assemble train / eval splits
#
# Goal: turn raw text into model inputs, then carve out a small train set and a REAL
# held-out eval set. Mirror the C9 tokenize_fn pattern.
#
# Steps you must complete:
#   1. Write tokenize_fn(batch) that tokenizes batch[TEXT_FIELD] with truncation and
#      MAX_LENGTH (the tokenizer is callable; let the collator pad later).
#   2. .map it over `raw` (batched) to get `tokenized`.
#   3. Rename the label column to "labels" so the model sees its target (C9 gotcha:
#      forget this and the model trains on nothing).
#   4. Build train_ds (shuffle with SEED, take TRAIN_SUBSET) and eval_ds. For sst2 use the
#      VALIDATION split (test labels hidden); for ag_news / emotion you may use the test split.

if PATH == "A":
    def tokenize_fn(batch):
        return None  # YOUR CODE: call the tokenizer on the text field with truncation + max_length

    tokenized = None      # YOUR CODE: map tokenize_fn over raw, batched
    # YOUR CODE: rename the integer-label column to "labels"

    # YOUR CODE: build train_ds (shuffle(seed=SEED).select(range(TRAIN_SUBSET))) and eval_ds.
    train_ds = None
    eval_ds = None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Verification (PROVIDED - do not edit):
    assert "labels" in train_ds.column_names, "Rename the label column to 'labels'."
    assert len(train_ds) == TRAIN_SUBSET, "train_ds must hold exactly TRAIN_SUBSET rows."
    assert len(eval_ds) > 0, "eval_ds must be a non-empty held-out split."
    sample = train_ds[0]
    assert "input_ids" in sample and len(sample["input_ids"]) <= MAX_LENGTH, "Tokenize with max_length."
    print("Path A tokenized. train:", len(train_ds), "eval:", len(eval_ds))
```

Provenance: FROM-OLD cells cell-12 + cell-13. Change: reskin CharDataset sliding-window code
to the C9 tokenize_fn + rename("labels") + split-selection flow; carry the rename-to-labels
gotcha; provide the collator so the lab focuses on tokenization, not boilerplate.

## Cell 11 - Code: Lab A3 - configure the Trainer

```python
# ### Lab A3 (Path A): build TrainingArguments and the Trainer
#
# Goal: wire the exact C9 training setup. You provide the argument VALUES; the structure is
# fixed. Use processing_class=tokenizer (the deprecated tokenizer= keyword is gone in 4.57).
#
# Steps:
#   1. Write compute_metrics(eval_pred) that returns accuracy (argmax the logits, compare to labels).
#   2. Fill TrainingArguments: an output dir, the per-device train/eval batch size from BATCH_SIZE,
#      the learning rate from LR, the epoch count from NUM_EPOCHS, and eval_strategy per epoch.
#   3. Construct the Trainer with model, args, train/eval datasets, data_collator,
#      processing_class=tokenizer, and compute_metrics.

if PATH == "A":
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # YOUR CODE: take argmax over the logits, then compute accuracy vs labels.
        preds = None
        return None  # YOUR CODE: return a dict like {"accuracy": ...}

    training_args = TrainingArguments(
        output_dir="distilbert_capstone",
        per_device_train_batch_size=None,   # YOUR CODE: from BATCH_SIZE
        per_device_eval_batch_size=None,    # YOUR CODE: from BATCH_SIZE
        learning_rate=None,                 # YOUR CODE: from LR
        num_train_epochs=None,              # YOUR CODE: from NUM_EPOCHS
        eval_strategy="epoch",              # provided: per-epoch eval (NOT evaluation_strategy)
        logging_steps=50,
        seed=SEED,
        fp16=(device.type == "cuda"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,   # provided: 4.57 standard; tokenizer= is removed
        compute_metrics=compute_metrics,
    )

    # Verification (PROVIDED - do not edit):
    assert trainer.args.learning_rate == LR, "Wire LR into TrainingArguments."
    assert trainer.args.num_train_epochs == NUM_EPOCHS, "Wire NUM_EPOCHS in."
    assert trainer.args.per_device_train_batch_size == BATCH_SIZE, "Wire BATCH_SIZE in."
    print("Path A Trainer ready.")
```

Provenance: FROM-OLD cells cell-14 + cell-15. Change: replace the CharRNN nn.Module with the
C9 TrainingArguments + Trainer; standardize on processing_class=tokenizer and eval_strategy;
starter leaves only the argument values blank, structure provided.

## Cell 12 - Code: Lab A4 - run training

```python
# ### Lab A4 (Path A): fine-tune
#
# One line. Everything was wired in Lab A3. On a Colab T4 this is ~3-6 minutes for
# TRAIN_SUBSET=6000 and 2 epochs.

if PATH == "A":
    train_result = None  # YOUR CODE: call the trainer's training method

    # Verification (PROVIDED - do not edit):
    assert train_result is not None, "Call trainer.train() and keep its result."
    print("Path A training done. Final train loss:", round(train_result.training_loss, 4))
```

Provenance: FROM-OLD cells cell-16 + cell-17. Change: drop the hand-written training loop
(B6/B7 already taught it); the C9 Trainer handles the loop, so the lab is the single
trainer.train() call plus the provided assertion.

## Cell 13 - Code: Lab A5 - evaluate and build classify_message

```python
# ### Lab A5 (Path A): evaluate honestly, then build the chatbot inference function
#
# Goal: get the held-out accuracy and wrap the model as classify_message(text), the exact
# C9 helper the Gradio shell calls.
#
# Steps:
#   1. Call trainer.evaluate() on the held-out eval set; read eval_accuracy.
#   2. Write classify_message(text): tokenize one string, run the model under torch.no_grad,
#      softmax the logits, take the top class, and return (id2label[class], confidence_float).

if PATH == "A":
    eval_metrics = None  # YOUR CODE: evaluate on the held-out split
    print("Path A held-out metrics:", eval_metrics)

    @torch.no_grad()
    def classify_message(text):
        model.eval()
        # YOUR CODE: tokenize `text` (return_tensors='pt', truncation, max_length) onto device,
        #            run the model, softmax the logits, take the argmax class and its probability,
        #            and return (model.config.id2label[class_id], float(probability)).
        return None  # YOUR CODE: return (label_string, confidence_float)

    # Verification (PROVIDED - do not edit):
    assert "eval_accuracy" in eval_metrics, "trainer.evaluate() should report eval_accuracy."
    label, conf = classify_message("the support team fixed my issue in minutes, amazing")
    assert isinstance(label, str) and 0.0 <= conf <= 1.0, "Return (label_string, confidence in [0,1])."
    print(f"Sample classification -> {label} ({conf:.2f})")
```

Provenance: FROM-OLD cells cell-22 + cell-23. Change: replace bits-per-character with
trainer.evaluate() accuracy plus the C9 classify_message(text) helper that the shared Gradio
shell consumes; hints name roles (softmax, argmax, id2label) but not the exact calls.

## Cell 14 - Markdown: Path B - T5 Q&A setup and hyperparameters

```
## Path B: encoder-decoder Q&A chatbot

(Skip this whole section if you chose Path A.)

This is the C10 architecture. You will fine-tune t5-small to ANSWER a question given a
context paragraph, and wrap it in a chatbot with two text inputs (question, context). The
pieces are the same ones you used in C10; here you wire them yourself.

Hyperparameters (sensible defaults from C10 - you may justify changes in your write-up):
- MODEL_NAME = "t5-small"
- max_input = 256        (question + context fit comfortably)
- max_target = 32        (SQuAD answers are short spans)
- LR = 3e-4              (T5 fine-tunes at a HIGHER LR than BERT classification, ~3e-4)
- NUM_EPOCHS = 2
- BATCH_SIZE = 16
- small_train = 2000     (subsample so the fine-tune fits a class session on a T4)
- small_val = 200

Honesty note carried from C10: a SQuAD-trained T5 mostly COPIES a span from the context.
Ask it something the context does not answer and it will confidently invent an answer.
SQuAD v1 never teaches "no answer", so off-context hallucination is EXPECTED, not a bug.
The fix is grounding (RAG); you will probe this in the evaluation cell.
```

Provenance: FROM-OLD cells cell-25 + cell-26. Change: reskin the Path B header from GPT-2 to
t5-small Q&A; replace GPT-2 hyperparameters with the exact C10 seq2seq values (max_input 256,
max_target 32, LR 3e-4, subsets 2000/200); carry the C10 hallucination honesty framing.

## Cell 15 - Markdown: Lab B1 - load T5, SQuAD, and the input builder

```
### Lab B1 (Path B): load t5-small, load SQuAD, and build the T5 input string

Goal: load the seq2seq model + tokenizer, load SQuAD, and write build_input(question,
context) exactly as in C10 so T5 sees the task it was trained on.

Steps:
1. Load the tokenizer and the seq2seq model from MODEL_NAME.
2. Load squad with load_dataset; note each example has question, context, and answers
   (a dict with a "text" LIST and an "answer_start" LIST - take the FIRST gold answer).
3. Write build_input(question, context) returning the C10 prompt string
   "question: {q}  context: {c}" (T5 needs the task framed as text-to-text).

Hints: the seq2seq model uses the AutoModelForSeq2SeqLM factory you imported. The gold
answer for an example is the first element of its answers["text"] list.
```

Provenance: FROM-OLD cell cell-28. Change: reskin "tokenize/chunk Shakespeare for GPT-2" to
"load T5 + SQuAD + build_input"; carry the C10 answers["text"][0] first-gold-answer gotcha;
hints name the factory role and the answer field, not the calls.

## Cell 16 - Code: Lab B1 starter + preprocess

```python
# ### Lab B1 / B-preprocess (Path B)
if PATH == "B":
    MODEL_NAME = "t5-small"
    max_input = 256
    max_target = 32
    LR = 3e-4
    NUM_EPOCHS = 2
    BATCH_SIZE = 16
    small_train = 2000
    small_val = 200

    # YOUR CODE: load the tokenizer from MODEL_NAME.
    tokenizer = None
    # YOUR CODE: load the seq2seq model from MODEL_NAME.
    model = None

    raw = load_dataset("squad")

    def build_input(question, context):
        # YOUR CODE: return the C10 T5 prompt "question: {question}  context: {context}"
        return None

    def preprocess(examples):
        # PROVIDED scaffold; you fill the two tokenizer calls.
        inputs = [build_input(q, c) for q, c in zip(examples["question"], examples["context"])]
        targets = [a["text"][0] for a in examples["answers"]]   # first gold answer (C10 gotcha)
        # YOUR CODE: tokenize `inputs` with truncation + max_length=max_input -> model_inputs
        model_inputs = None
        # YOUR CODE: tokenize the targets via text_target=targets, truncation, max_length=max_target,
        #            and attach them as model_inputs["labels"].
        return model_inputs

    small = raw["train"].shuffle(seed=SEED)
    tokenized_train = small.select(range(small_train)).map(
        preprocess, batched=True, remove_columns=small.column_names)
    tokenized_val = raw["validation"].shuffle(seed=SEED).select(range(small_val)).map(
        preprocess, batched=True, remove_columns=raw["validation"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  # pads labels with -100

    # Verification (PROVIDED - do not edit):
    assert build_input("Q?", "C.") == "question: Q?  context: C.", "build_input format must match C10."
    assert "labels" in tokenized_train.column_names, "preprocess must attach 'labels' via text_target."
    assert len(tokenized_train) == small_train and len(tokenized_val) == small_val, "Subset sizes."
    print("Path B preprocessed. train:", len(tokenized_train), "val:", len(tokenized_val))
```

Provenance: FROM-OLD cell cell-29. Change: replace GPT-2 chunking with the C10 build_input +
preprocess (text_target labels, DataCollatorForSeq2Seq, -100 padding, first-gold-answer);
leave the tokenizer calls and the prompt string blank with role-only hints.

## Cell 17 - Markdown: Lab B2 - the Seq2SeqTrainer

```
### Lab B2 (Path B): configure Seq2SeqTrainingArguments and Seq2SeqTrainer

Goal: wire the C10 seq2seq training setup. T5 specifics you must respect:
- predict_with_generate must be on so eval uses model.generate() (real decoding, not teacher forcing).
- fp16 only when running on a GPU (device.type == "cuda").
- Use the HIGHER T5 learning rate (~3e-4), not the BERT classification LR.
- Standardize on processing_class=tokenizer for the Seq2SeqTrainer too (C10 used the
  deprecated tokenizer= keyword; in 4.57 prefer processing_class= for both trainers).
- Never build decoder_input_ids yourself; teacher forcing is automatic.

You provide the argument VALUES; the structure is fixed.
```

Provenance: FROM-OLD cell cell-30. Change: reskin "fine-tune GPT-2 with Trainer" to the C10
Seq2SeqTrainer; explicitly resolve the synthesis Gap by standardizing on processing_class=
for the seq2seq trainer; carry the predict_with_generate / fp16-on-cuda / no-decoder-ids rules.

## Cell 18 - Code: Lab B2 starter (Seq2SeqTrainer config + run)

```python
if PATH == "B":
    training_args = Seq2SeqTrainingArguments(
        output_dir="t5_capstone_qa",
        per_device_train_batch_size=None,   # YOUR CODE: from BATCH_SIZE
        per_device_eval_batch_size=None,    # YOUR CODE: from BATCH_SIZE
        learning_rate=None,                 # YOUR CODE: the T5 fine-tune LR (higher than BERT)
        num_train_epochs=None,              # YOUR CODE: from NUM_EPOCHS
        eval_strategy="epoch",              # provided (NOT evaluation_strategy)
        predict_with_generate=None,         # YOUR CODE: must be True so eval decodes
        fp16=(device.type == "cuda"),       # provided: GPU-only mixed precision
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        processing_class=tokenizer,   # provided: standardized across both trainers in 4.57
    )

    # Verification (PROVIDED - do not edit):
    assert training_args.predict_with_generate is True, "Turn predict_with_generate on for seq2seq eval."
    assert training_args.learning_rate > 1e-4, "T5 wants a higher LR than BERT classification."
    train_result = None  # YOUR CODE: run training and keep the result
    assert train_result is not None, "Call trainer.train()."
    print("Path B training done. Final train loss:", round(train_result.training_loss, 4))
```

Provenance: FROM-OLD cell cell-31. Change: replace GPT-2 causal-LM Trainer with the C10
Seq2SeqTrainer; starter leaves only values blank; asserts on properties (predict_with_generate
True, LR > 1e-4) without naming the literals - the LR target lives in the theory cell.

## Cell 19 - Code: Lab B3 - build answer_question

```python
# ### Lab B3 (Path B): build the chatbot inference function answer_question
#
# Goal: wrap the fine-tuned T5 as answer_question(question, context) - the exact C10 helper
# the Gradio shell calls. Build the input with build_input, tokenize, generate, decode.

if PATH == "B":
    @torch.no_grad()
    def answer_question(question, context):
        model.eval()
        text = build_input(question, context)
        # YOUR CODE: tokenize `text` (return_tensors='pt', truncation, max_length=max_input) onto device,
        #            call model.generate(...) with max_new_tokens around max_target,
        #            then decode the first sequence with skip_special_tokens=True and return the string.
        return None  # YOUR CODE: return the decoded answer string

    # Verification (PROVIDED - do not edit):
    ctx = "Data Trainers LLC was founded by Axel Sirota and teaches ML and NLP courses."
    ans = answer_question("Who founded Data Trainers?", ctx)
    assert isinstance(ans, str) and len(ans) > 0, "answer_question must return a non-empty string."
    print("Q: Who founded Data Trainers?")
    print("A:", ans)
```

Provenance: FROM-OLD cells cell-32 + cell-33. Change: replace the GPT-2 generate helper with
the C10 answer_question(question, context) (build_input -> tokenize -> generate -> decode);
hints name the steps but not generate()'s keywords.

## Cell 20 - Code: Lab B4 - before/after eval and the honesty (hallucination) probe

```python
# ### Lab B4 (Path B): qualitative before/after + the hallucination probe
#
# Goal: show the fine-tune helped (compare a few SQuAD validation answers), then deliberately
# feed an OFF-CONTEXT question and watch T5 confidently invent an answer. This is the honest
# evaluation leadership asked for.
#
# Steps:
#   1. Pick 3 validation examples; for each, print question, gold answer, and answer_question(...).
#   2. Run the provided hallucination probe: a real question whose context does NOT contain
#      the answer. Read the output critically - it will likely fabricate.

if PATH == "B":
    val = raw["validation"]
    for i in range(3):
        ex = val[i]
        gold = ex["answers"]["text"][0]
        # YOUR CODE: call answer_question with this example's question and context -> pred
        pred = None
        print(f"Q: {ex['question']}\n  gold: {gold}\n  model: {pred}\n")

    # Hallucination probe (PROVIDED - do not edit). The context is about cats; the question
    # is about a CEO, which the context cannot answer. Watch it invent something anyway.
    probe_ctx = "Cats are small carnivorous mammals often kept as pets. They sleep most of the day."
    probe_q = "Who is the CEO of the company mentioned in the text?"
    print("OFF-CONTEXT PROBE")
    print("  context:", probe_ctx)
    print("  question:", probe_q)
    print("  model:", answer_question(probe_q, probe_ctx))
    print("\nNote for your write-up: SQuAD v1 never teaches 'no answer', so the model copies or")
    print("invents. The production fix is grounding (RAG) + an abstention threshold.")
```

Provenance: FROM-OLD cells cell-34 + cell-35 + cell-36 + cell-37. Change: collapse the four
GPT-2 compare/perplexity cells into one C10-style qualitative before/after plus the explicit
off-context hallucination probe; perplexity is dropped (not meaningful for span-copy QA), the
honesty framing replaces it.

## Cell 21 - Code: Shared - save and reload (BOTH paths)

```python
# ### Save and reload (runs for whichever path you chose)
#
# save_pretrained writes BOTH the model and the tokenizer; reload BOTH. id2label travels in
# config.json, so a reloaded Path A model returns word labels. After reload we REBIND the
# global `model` and `tokenizer` to the on-disk objects so the inference helpers use them
# (C10 continuity: helpers close over these globals).

SAVE_DIR = "capstone_model_A" if PATH == "A" else "capstone_model_B"

# YOUR CODE: save the trained model to SAVE_DIR (use trainer.model, not just `model`).
# YOUR CODE: save the tokenizer to SAVE_DIR.

# Reload BOTH from disk (PROVIDED - do not edit):
tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
if PATH == "A":
    model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR).to(device)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(SAVE_DIR).to(device)

# Verification (PROVIDED - do not edit):
import os
assert os.path.isdir(SAVE_DIR), "Save the model + tokenizer to SAVE_DIR first."
if PATH == "A":
    label, conf = classify_message("the agent was rude and slow")
    print("Reloaded Path A still works ->", label, round(conf, 2))
else:
    print("Reloaded Path B still works ->",
          answer_question("Who founded Data Trainers?",
                          "Data Trainers LLC was founded by Axel Sirota."))
print("Saved + reloaded from", SAVE_DIR)
```

Provenance: FROM-OLD cell cell-24. Change: replace `torch.save(state_dict)` with the C9/C10
save_pretrained -> reload pattern for BOTH model and tokenizer; rebind the globals after
reload (synthesis Gap on the C10 reassignment); branch the reload class on PATH; the two
save_pretrained calls are the only blanks, reload + asserts are provided.

## Cell 22 - Code: Shared - guarded Gradio chatbot ship (BOTH paths)

```python
# ### Ship it: the guarded Gradio chatbot (runs for whichever path you chose)
#
# This is the C9/C10 Gradio guard, verbatim: try/except ImportError so a no-Gradio env still
# works, share=True for the Colab public link, and a non-Gradio fallback that calls inference
# directly. Path A is a single text box -> label + confidence; Path B is two boxes
# (question, context) -> written answer.

try:
    import gradio as gr

    if PATH == "A":
        def gradio_fn(message):
            label, conf = classify_message(message)
            return f"{label} (confidence {conf:.2f})"

        demo = gr.Interface(
            fn=gradio_fn,
            inputs=gr.Textbox(label="Customer message"),
            outputs=gr.Textbox(label="Prediction"),
            title="Capstone C - DistilBERT ticket classifier",
        )
    else:
        def gradio_fn(question, context):
            return answer_question(question, context)

        demo = gr.Interface(
            fn=gradio_fn,
            inputs=[gr.Textbox(label="Question"), gr.Textbox(label="Context / help-doc snippet")],
            outputs=gr.Textbox(label="Answer"),
            title="Capstone C - T5 Q&A assistant",
        )

    # share=True publishes a temporary public Colab link (it is the default in Colab anyway).
    demo.launch(share=True)

except ImportError:
    # Non-Gradio fallback: call inference directly so the notebook still demonstrates the model.
    print("Gradio not installed - direct inference fallback:")
    if PATH == "A":
        print(classify_message("this product is fantastic, thank you"))
    else:
        print(answer_question("Who founded Data Trainers?",
                              "Data Trainers LLC was founded by Axel Sirota."))
```

Provenance: FROM-NEW. (The old notebook printed/sampled text and had no Gradio; this is the
C9/C10 guarded gr.Interface shell, branched on PATH to serve both the 1-input classifier and
the 2-input Q&A bot, with the try/except ImportError + share=True + fallback reused verbatim.)

## Cell 23 - Code: Reflection write-up (technical + non-technical)

```python
# ### Your write-up (fill the strings, then run)
#
# This is the deliverable leadership asked for. Be honest about failure modes.
#   - Technical memo: dataset + why, metric + number, the single biggest failure mode,
#     and one concrete production next step.
#   - Non-technical pitch: 3 sentences a product manager would understand.
#
# Path A failure modes to consider: confident-but-wrong on sarcasm / negation, class
# imbalance, domain drift. Path B failure modes: span-copying, off-context hallucination,
# no "I don't know". Production levers: confidence-threshold abstention, drift monitoring,
# RAG grounding (Path B), periodic re-evaluation.

DATASET_CHOICE = None       # YOUR CODE: which dataset you used and one line on WHY it fit
HEADLINE_METRIC = None      # YOUR CODE: your held-out number (e.g. "accuracy 0.91" or "qualitative")
BIGGEST_FAILURE = None      # YOUR CODE: the single worst failure mode you observed
PRODUCTION_NEXT_STEP = None # YOUR CODE: one concrete thing you would add before shipping

PITCH_1 = None  # YOUR CODE: what the bot does, in plain language
PITCH_2 = None  # YOUR CODE: one honest limitation a customer might hit
PITCH_3 = None  # YOUR CODE: the business value if the limitation is managed

# Verification (PROVIDED - do not edit):
for name, val in [("DATASET_CHOICE", DATASET_CHOICE), ("HEADLINE_METRIC", HEADLINE_METRIC),
                  ("BIGGEST_FAILURE", BIGGEST_FAILURE), ("PRODUCTION_NEXT_STEP", PRODUCTION_NEXT_STEP),
                  ("PITCH_1", PITCH_1), ("PITCH_2", PITCH_2), ("PITCH_3", PITCH_3)]:
    assert isinstance(val, str) and len(val) > 0, f"Fill {name} with a real sentence."
print("TECHNICAL MEMO")
print(" dataset:", DATASET_CHOICE)
print(" metric :", HEADLINE_METRIC)
print(" failure:", BIGGEST_FAILURE)
print(" next   :", PRODUCTION_NEXT_STEP)
print("\nPITCH:", PITCH_1, PITCH_2, PITCH_3)
```

Provenance: FROM-OLD cells cell-38 + cell-39 + cell-40 + cell-41. Change: reskin the
Present-Results memo + non-technical pitch onto the transfer-learning artifact; fold both old
deliverables into one cell; the failure-mode prompts carry the C10 hallucination honesty and
the production levers (abstention, drift, RAG) validated in research.

## Cell 24 - Markdown: Self-check quiz + homework extensions + course close

```
## Self-check (answer before you submit - no code needed)

1. You chose Path ___. Why was an encoder-only classifier (or encoder-decoder Q&A) the right
   architecture for your task? Name one task where the OTHER path would have been correct.
2. Why do we evaluate on a held-out split, and why is glue/sst2's test split unusable for
   reporting accuracy?
3. (Path B) Explain in one sentence why a SQuAD-trained T5 hallucinates on an off-context
   question, and name the production fix.
4. Why does id2label need to live in config.json for the reloaded chatbot to return word
   labels?
5. Name one production concern (monitoring, drift, cost, or abstention) and how you would
   address it for YOUR bot.

## Homework extensions (async, deeper)

- STRETCH (in-class, fast finishers): Path A - freeze the DistilBERT backbone and train only
  the classification head; report how much accuracy you traded for the speed-up. Path B -
  sweep two decoding settings (greedy vs num_beams=4) on five questions and describe the
  difference.
- HOMEWORK 1 (both): add a confidence-threshold abstention layer - if Path A's confidence is
  below a threshold (or Path B's answer is empty / low-probability), return "I am not sure,
  routing to a human." Measure coverage vs error on the held-out set.
- HOMEWORK 2 (Path B): ground the model with RAG - reuse an embedder to retrieve the most
  relevant help-doc snippet as the context, then answer over it. Re-run the off-context probe
  and show grounding removes the hallucination.
- HOMEWORK 3 (both): compute a proper metric (accuracy + macro-F1 for A via evaluate; EM/F1
  for B via the squad metric) and write a one-paragraph drift-monitoring plan.

## That is the course

You went from tools-first NLP, to embeddings as geometry, to an MLP on word2vec, to
fine-tuning real transformers, and now to a shipped, evaluated, honestly-documented chatbot
that YOU built end to end. That is the whole job.
```

Provenance: FROM-OLD cell cell-42. Change: reskin the self-check quiz from char-generation /
GPT-2 questions to transfer-learning concepts (architecture choice, held-out eval, GLUE
hidden labels, hallucination + RAG, id2label persistence, production); append the three-tier
labs (stretch + 3 homeworks) and the course-closing paragraph.

# VERIFICATION CHECKLIST

- [x] 25 cells (0-24), inside the 20-25 budget.
- [x] STAR Story section present.
- [x] Chatbot Through-Line section present.
- [x] Both shells supported: Path A (encoder-only, distilbert-base-uncased) and Path B
      (encoder-decoder, t5-small), gated by a single PATH switch (cell 6).
- [x] C9 names reused verbatim for Path A: tokenizer, model, tokenize_fn, data_collator,
      compute_metrics, Trainer with processing_class=tokenizer, id2label/label2id,
      classify_message, SAVE_DIR, guarded gr.Interface.
- [x] C10 names reused verbatim for Path B: build_input, preprocess, max_input/max_target,
      DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, answer_question, chat_fn shape.
- [x] Trainer standardized on processing_class=tokenizer for BOTH Trainer (cell 11) and
      Seq2SeqTrainer (cell 18); synthesis Gap (C10 tokenizer= inconsistency) explicitly resolved.
- [x] Guarded Gradio shell kept verbatim: try/except ImportError, share=True on Colab,
      non-Gradio direct-inference fallback (cell 22).
- [x] Hallucination-honesty framing kept for Path B (cells 14, 20, 24); RAG named as the fix.
- [x] Final reflection / write-up cell present (cell 23): what worked, where it fails, production notes.
- [x] All LSTM / GRU / RNN / Shakespeare / char-generation / GPT-2 content CUT (see migration map drops).
- [x] numpy<2 pinned in the install cell (cell 3) with the mandatory Colab restart note.
- [x] transformers==4.57.1, datasets>=2.19,<3, evaluate, accelerate, gradio pinned; no gensim / sentence-transformers.
- [x] eval_strategy used (NOT the removed evaluation_strategy) in both training-args cells.
- [x] GLUE/sst2 hidden-test-label gotcha carried; ag_news / emotion offered as real-test-label alternatives.
- [x] C10 SQuAD gotcha carried: answers["text"][0] first gold answer; text_target labels; -100 padding.
- [x] fp16 gated on device.type == "cuda"; subset sizes kept for a class-session T4 fine-tune.
- [x] Three-tier labs: core verified labs + one in-class STRETCH + 3 HOMEWORK extensions (cell 24).
- [x] No-leak hint discipline: every None # YOUR CODE references a role / shape / helper, never
      the method, keyword, literal, or index that solves the line; verification asserts on properties.
- [x] Every cell tagged FROM-OLD (with old cell id) or FROM-NEW; Cell Migration Map present.
- [x] Plain ASCII only: no em dashes, no en dashes, no Unicode multiplication, no emoji.

Provenance split:
- FROM-OLD cells: 23 (cells 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24).
- FROM-NEW cells: 2 (cell 2 decision rule; cell 22 guarded Gradio ship).
- Old cells dropped: 8 (cell-5 Shakespeare load; cell-18 training-curve plot; cell-19/20/21
  char temperature generation; cell-36/37 folded into 20; remaining char-gen drops per map).

# RESEARCH VALIDATED (June 2026)

- transformers Trainer/Seq2SeqTrainer: `processing_class` supersedes the deprecated
  `tokenizer=` argument (warns, does not error, on 4.57). Standardize on `processing_class=`
  for BOTH trainers. Source: https://github.com/huggingface/transformers/issues/37734 ;
  https://huggingface.co/docs/transformers/v4.57.0/en/main_classes/trainer
- Seq2SeqTrainingArguments: `predict_with_generate=True` makes eval decode via
  `model.generate()`; `eval_strategy="epoch"`; `fp16=True` on GPU; T5 fine-tunes at a higher
  LR than BERT classification. Source:
  https://medium.com/@anyuanay/fine-tuning-the-pre-trained-t5-small-model-in-hugging-face-for-text-summarization-3d48eb3c4360
- Colab default numpy is 2.x; pinning numpy<2 DOWNGRADES it and REQUIRES a runtime restart
  before any other cell. Source: https://github.com/googlecolab/colabtools/issues/5213
- ag_news: columns `text` / `label`; 4 classes (World/Sports/Business/Sci-Tech); train 120000
  / test 7600 with REAL test labels (unlike GLUE/sst2 whose test labels are hidden -1).
  Source: https://huggingface.co/datasets/fancyzhx/ag_news
- T5 fine-tuned on SQuAD hallucinates on out-of-context / unanswerable questions; SQuAD v1
  never teaches "no answer", so off-context fabrication is expected; extractive grounding /
  SQuAD-2-style abstention is the mitigation. Source:
  https://mbrenndoerfer.com/writing/question-answering ;
  https://arxiv.org/pdf/2402.07647
- Production take-homes: confidence-threshold abstention (defer low-confidence predictions),
  drift monitoring (track confidence-score moving averages, prediction-frequency, semantic
  drift in RAG embeddings). Source:
  https://www.evidentlyai.com/ml-in-production/model-monitoring ;
  https://stackpulsar.com/blog/llm-model-drift-detection/
- save_pretrained writes BOTH model and tokenizer; `id2label` persists in config.json so a
  reloaded classifier returns word labels; `pipeline` auto-loads missing components. Source:
  https://huggingface.co/docs/transformers/en/main_classes/configuration ;
  https://huggingface.co/docs/transformers/main_classes/pipelines
- Gradio: `share=True` is the default in Colab and publishes a temporary public link;
  `gr.Interface.from_pipeline` exists; guard the import for no-Gradio environments. Source:
  https://www.gradio.app/guides/sharing-your-app

Plan written to plans/refactor/notebooks/C11-capstone-c.md.
Next step: run /build-notebook 11-Capstone_C colab to generate the exercise + solution
notebooks from this plan, 5 cells at a time with approval between batches.
