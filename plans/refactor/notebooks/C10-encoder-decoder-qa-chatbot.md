# C10 - Encoder-Decoder Q&A Chatbot - Cell-by-Cell Plan

## Status

NEW. No old source notebook. Build from scratch. This is the finale of Part C and the
course: it completes the architecture arc encoder-only (C9 DistilBERT classifier) ->
encoder-decoder (C10 T5 seq2seq Q&A), and ships a fine-tuned generative model inside a
Gradio chatbot. Every cell is authored fresh; every cell is tagged `Provenance: FROM-NEW`.

## Context

What students arrive with (by exact name, from C9 and earlier):

- From C9: they fine-tuned `distilbert-base-uncased` with the HuggingFace `Trainer`, used
  `AutoTokenizer`, evaluated with `evaluate`, saved with `save_pretrained`, and shipped a
  guarded `gr.Interface` chatbot. They know `device = torch.device('cuda' if
  torch.cuda.is_available() else 'cpu')` and `torch.manual_seed(42)`.
- From B5: they have `embedder` (a `SentenceTransformer`, `all-MiniLM-L6-v2`, 384-d) and a
  `semantic_search(query, top_k=3)` helper over a `corpus`. That semantic search is the
  retrieval half of a RAG pipeline, which C10's take-home calls back to explicitly.
- From B7/C9: an encoder turns text into vectors for classification; the limit of a classifier
  is that it can only pick a label, never write a sentence.

The key insight they leave with: an encoder-decoder (seq2seq) model READS a question plus a
context (encoder) and WRITES an answer token by token (decoder). Fine-tuning T5 on SQuAD
teaches it to extract and phrase answers. They can now choose, per task, between encoder-only
classification (a label) and encoder-decoder generation (free text), and they have shipped
both kinds of model inside a chatbot.

## Chatbot Through-Line

C9 shipped an encoder-only chatbot that returns a LABEL. C10 ships the generative finale: a
chatbot that returns a written ANSWER. This is the payoff the whole course pointed at - a
fine-tuned transformer loaded into a simple Gradio app. The closing reflection draws the line
between the two chatbot shapes (classify vs generate) and shows that feeding C10's QA model a
context retrieved by B5's `semantic_search` is exactly a RAG assistant, naming the next step
(C11 capstone: pick one chatbot path and ship it on your own dataset).

## STAR Story

- Situation: The support-platform team from C9 shipped a DistilBERT chatbot that TAGS each
  incoming message (billing, bug, praise). Leadership loved it, then asked the obvious next
  thing: "Great, it sorts tickets. Can it actually ANSWER the customer from our help docs?"
- Task: A classifier cannot answer; it only labels. You need a model that reads a question
  plus a relevant help-doc passage and writes back a short, correct answer in plain language.
- Action: You reach for an encoder-decoder model. You load pretrained `t5-small`, see it
  answer poorly zero-shot, then fine-tune it on SQuAD (questions + context -> answer span)
  with `Seq2SeqTrainer`. You generate answers with `model.generate`, save the model, and wrap
  it in a Gradio Q&A chatbot.
- Result: A fine-tuned seq2seq assistant that takes a question and a context and returns a
  written answer, live in a Gradio UI. You can now choose encoder-only or encoder-decoder per
  task, and you understand why grounding the context (RAG, reusing B5 semantic search) keeps
  the answers honest.

## Deliverables

- Exercise notebook: `C-Transfer-Learning/10-EncoderDecoder_QA_Chatbot.ipynb`
- Solution notebook: `Solutions/C-Transfer-Learning/10-EncoderDecoder_QA_Chatbot.ipynb`

## Session Timing (~60-90 min)

| Block | Cells | Minutes |
|-------|-------|---------|
| Setup + story + dataset preview | 0-4 | 8 |
| Concept 1: three transformer shapes + load t5-small | 5-8 | 12 |
| Concept 2: seq2seq tokenization (LAB: preprocess fn) | 9-12 | 18 |
| Concept 3: fine-tune with Seq2SeqTrainer (LAB: training args) | 13-16 | 18 |
| Concept 4: generate answers + before/after (LAB: answer fn) | 17-19 | 14 |
| Chatbot: save/reload + guarded Gradio Q&A app | 20-21 | 12 |
| Stretch + homework + wrap-up | 22-23 | 8 |

Total target: 24 cells (0-23), within the 20-25 band.

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~20-25 cells)

## Cell 0 - Markdown: Title, objectives, prerequisites, environment intro

Provenance: FROM-NEW

```markdown
# C10 - Encoder-Decoder Q&A Chatbot (the finale)

In C9 you fine-tuned an encoder-only model (DistilBERT) and shipped a chatbot that
returns a **label**. That is the whole power of a classifier: it can sort, but it can
never write a sentence back to the user.

Today we cross the last bridge. We load an **encoder-decoder** (sequence-to-sequence)
model, `t5-small`, fine-tune it to answer questions from a context passage, and load it
into a Gradio chatbot that returns a **written answer**. This is the deliverable the whole
course pointed at: a fine-tuned transformer in a simple chatbot.

## Learning objectives

By the end you will be able to:

1. Explain the difference between encoder-only, decoder-only, and encoder-decoder models,
   and say which shape fits which task.
2. Load a pretrained seq2seq model with `AutoModelForSeq2SeqLM` and run it with `.generate()`.
3. Tokenize inputs and targets for seq2seq (the T5 task prefix, label padding with -100).
4. Fine-tune `t5-small` on a SQuAD subset with `Seq2SeqTrainer`.
5. Save the fine-tuned model and load it into a guarded Gradio Q&A chatbot.

## Prerequisites

- C9 (DistilBERT fine-tuning, the HuggingFace `Trainer`, `save_pretrained`, Gradio guard).
- Comfort with `device`, `torch.manual_seed(42)`, and HuggingFace `datasets`.

## Session format

Theory -> Demo -> Lab, four concepts. One guided core lab per concept, plus a labelled
in-notebook stretch and an async homework at the end.

## Runtime

Google Colab. A GPU runtime (Runtime -> Change runtime type -> T4 GPU) makes the fine-tune
finish in a few minutes. The notebook still runs on CPU, just slower.

## Environment setup (next cell)

The next cell installs pinned versions so the notebook behaves the same for everyone. We pin
`numpy<2` because Colab now ships numpy 2.x, and several of our libraries were built against
numpy 1.x. After the install runs, Colab may ask you to **restart the runtime**
(Runtime -> Restart runtime) so the numpy downgrade takes effect. Restart, then run every
cell from the top. We only install what C10 actually uses: `transformers` (the model +
`Seq2SeqTrainer`), `datasets` (SQuAD), `evaluate` (the SQuAD metric, used in the homework),
`accelerate` (the `Trainer` runs on top of it), and `gradio` (the chatbot UI). No gensim or
sentence-transformers here - C10 is pure seq2seq.
```

## Cell 1 - Code: Install dependencies

Provenance: FROM-NEW

```python
# Install pinned dependencies for Colab.
# numpy<2 is required because Colab ships numpy 2.x and our stack expects numpy 1.x.
# After this runs, if Colab prints a "restart runtime" prompt, do it, then run all cells.
!pip install -q \
    "transformers==4.57.1" \
    "datasets>=2.19,<3" \
    "evaluate" \
    "accelerate" \
    "gradio" \
    "numpy<2"

# Why each package:
# transformers   -> AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, .generate()
# datasets       -> load_dataset("squad")
# evaluate       -> the SQuAD exact-match / F1 metric (used in the homework)
# accelerate     -> the HuggingFace Trainer is powered by accelerate under the hood
# gradio         -> the final Q&A chatbot UI
# numpy<2        -> compatibility pin for the whole course

print("Install step done. If Colab asked to restart the runtime, restart then re-run.")
```

## Cell 2 - Code: Imports, device, seed

Provenance: FROM-NEW

```python
# Standard imports for a seq2seq fine-tuning notebook.
import numpy as np
import torch
import transformers
import datasets

from transformers import (
    AutoTokenizer,             # turns text into token ids (and back)
    AutoModelForSeq2SeqLM,     # an encoder-decoder model with a generation head
    DataCollatorForSeq2Seq,    # pads inputs AND labels (labels padded with -100)
    Seq2SeqTrainer,            # the Trainer subclass that knows how to .generate() while evaluating
    Seq2SeqTrainingArguments,  # the training config object
)
from datasets import load_dataset

# Confirm versions so a broken install is caught immediately.
print(f"transformers: {transformers.__version__}")   # expect 4.57.1
print(f"datasets:     {datasets.__version__}")
print(f"numpy:        {np.__version__}")              # expect a 1.x version (<2)

# Device: use the GPU if Colab gave us one, else CPU. Same pattern as C9.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Reproducibility: same seed as the rest of the course.
torch.manual_seed(42)
np.random.seed(42)

print("Environment ready.")
```

## Cell 3 - Markdown: What are we building today?

Provenance: FROM-NEW

```markdown
## What are we building today?

Back to the support-platform team. Your C9 DistilBERT chatbot TAGS each incoming message:
billing, bug report, praise. Leadership saw it and immediately asked the next question:

> "Great, it sorts the tickets. Can it actually ANSWER the customer from our help docs?"

A classifier cannot answer. It only picks a label. To answer, you need a model that can
**read** a question plus a relevant help-doc passage and **write** a short answer back.

That is a sequence-to-sequence job. We will:

1. Load `t5-small`, a small encoder-decoder model, and watch it answer poorly out of the box.
2. Fine-tune it on SQuAD, a dataset of (question, context, answer) triples, so it learns to
   pull the answer out of the context.
3. Generate answers with `.generate()` and compare before vs after.
4. Save the model and drop it into a Gradio Q&A chatbot.

We will use SQuAD as a stand-in for "your help docs": each example gives a question, a
paragraph of context, and the correct answer span inside that context. That is exactly the
shape of the support-answering task.
```

## Cell 4 - Code: Load SQuAD and preview one example

Provenance: FROM-NEW

```python
# SQuAD = Stanford Question Answering Dataset. Each example has:
#   question : a natural-language question
#   context  : a paragraph that contains the answer
#   answers  : a dict with 'text' (list of answer strings) and 'answer_start' (list of ints)
# We load the standard "squad" config from the HuggingFace Hub.
raw = load_dataset("squad")
print(raw)  # shows train / validation splits and their sizes

# Look at one example so the data shape is concrete.
example = raw["train"][0]
print("\n--- one SQuAD example ---")
print("QUESTION:", example["question"])
print("CONTEXT :", example["context"][:300], "...")
# The gold answer lives at answers['text'][0]. SQuAD stores it as a list because some
# validation examples have several acceptable answers; for training we take the first.
print("ANSWER  :", example["answers"]["text"][0])
```

## Cell 5 - Markdown: Concept 1 - three shapes of a transformer

Provenance: FROM-NEW

```markdown
## Concept 1: encoder, decoder, encoder-decoder

Every model in this course has been built from the same transformer block. The difference
between models is which HALF of the architecture they keep.

Think of a human translator working from English to French:

- First she **reads** the whole English sentence and builds an understanding of it in her
  head. That reading-and-understanding stage is the **encoder**. It looks at the entire
  input at once and turns it into a rich set of vectors.
- Then she **writes** the French sentence one word at a time, each new word depending on the
  words she has already written. That writing stage is the **decoder**. It generates output
  token by token (this is called autoregressive generation).

Now the three model shapes:

- **Encoder-only** (BERT, DistilBERT - your C9 model). Keeps only the reading half. Great for
  understanding a whole input and producing a fixed output: a class label, an embedding, a
  span. It cannot write free text.
- **Decoder-only** (the GPT family). Keeps only the writing half. Great at continuing text.
  Out of scope for this course.
- **Encoder-decoder** (T5, BART - today). Keeps both. It READS the input with the encoder and
  WRITES a new sequence with the decoder. This is the right shape when the output is free text
  that depends on the input: translation, summarization, and abstractive question answering.

Rule of thumb:

```python
# Output is a label or a score?         -> encoder-only (classification head)   (C9)
# Output is a written sequence of text? -> encoder-decoder (generation head)     (C10)
```

For our task - read a question + a context, write an answer - we need the encoder-decoder
shape. We will use `t5-small`: small enough to fine-tune in a few minutes, and built around
a clean "text in, text out" idea that makes the encoder-decoder story easy to see.
```

## Cell 6 - Code: Load t5-small and inspect it

Provenance: FROM-NEW

```python
# Load the pretrained T5-small tokenizer and seq2seq model.
# AutoModelForSeq2SeqLM gives us a model that already has a generation (LM) head on the
# decoder, so we can call .generate() right away.
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# T5 is genuinely two stacks: an encoder and a decoder. We can see both.
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model_name}")
print(f"Total parameters: {n_params:,}")          # ~60M, small by transformer standards
print(f"Has encoder: {model.get_encoder() is not None}")
print(f"Has decoder: {model.get_decoder() is not None}")

# T5's "text in, text out" design means even tasks like translation are phrased as text.
# We tell T5 which task we want by prepending a TASK PREFIX to the input. We will use a
# "question: ... context: ..." prefix for QA. T5 was pretrained with these prefixes, so the
# format is not arbitrary - it matches how the model expects work to be described.
print("\nTokenizer is ready. Prefixes like 'question:' / 'context:' tell T5 what to do.")
```

## Cell 7 - Markdown: Demo - T5 before any fine-tuning (the "before")

Provenance: FROM-NEW

```markdown
### Demo: ask pretrained T5 a question (before fine-tuning)

Before we train anything, let us see how well stock `t5-small` answers a question from a
context. We build the input string with the QA prefix, tokenize it, call `.generate()`, and
decode the output tokens back to text.

`t5-small` was pretrained on a mix of tasks but never specifically taught the SQuAD
question-answering format, so expect a weak, vague, or off-target answer. That weak baseline
is the point: it is the "before" picture that fine-tuning will fix.
```

## Cell 8 - Code: Demo - generate an answer with the un-tuned model

Provenance: FROM-NEW

```python
# A helper that turns a (question, context) pair into a T5 input string.
# This is the SAME prefix format we will train on later, so the before/after is a fair test.
def build_input(question, context):
    # The "question: ... context: ..." prefix tells T5 this is a QA task.
    return f"question: {question}  context: {context}"

# Pick one SQuAD validation example to probe.
probe = raw["validation"][0]
q, c = probe["question"], probe["context"]
gold = probe["answers"]["text"][0]

# Tokenize the input string into ids the model can read.
inputs = tokenizer(build_input(q, c), return_tensors="pt", truncation=True, max_length=256).to(device)

# Generate an answer. .generate() runs the decoder autoregressively (token by token)
# until it emits the end-of-sequence token or hits max_new_tokens.
with torch.no_grad():
    out_ids = model.generate(**inputs, max_new_tokens=32)

# Decode ids back to text, skipping the special tokens (pad, eos).
pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)

print("QUESTION    :", q)
print("GOLD ANSWER :", gold)
print("T5 (un-tuned):", pred)   # likely vague or wrong - that is expected before fine-tuning
```

## Cell 9 - Markdown: Concept 2 - tokenizing for seq2seq

Provenance: FROM-NEW

```markdown
## Concept 2: tokenizing inputs AND targets

For classification (C9) you tokenized one piece of text and attached an integer label. For
seq2seq there are TWO pieces of text per example:

- the **input**: `"question: {q}  context: {c}"` (what the encoder reads)
- the **target**: the answer string (what the decoder must learn to write)

So we tokenize twice. The input tokens become `input_ids`; the target tokens become
`labels`. Two details matter and are the classic beginner mistakes:

1. **The task prefix.** Always prepend `"question: ... context: ..."`. Drop it and T5 does not
   know what job to do.
2. **Label padding with -100.** Examples in a batch have different lengths, so labels get
   padded. But padding tokens must NOT count toward the loss, or the model wastes capacity
   learning to predict padding. The convention is to set padded label positions to `-100`,
   which PyTorch's cross-entropy ignores. We do NOT do this by hand: `DataCollatorForSeq2Seq`
   sets it for us at batch time.

One more thing you do NOT need to do: you never build `decoder_input_ids` yourself. Give the
model `labels` and T5 automatically shifts them right and prepends its start token to form the
decoder input. This is teacher forcing - during training the decoder is fed the correct
previous answer tokens so it learns the next token at every position.

```python
# The shape of one preprocessed example:
#   input_ids : ids of "question: ... context: ..."
#   labels    : ids of the answer string (padding -> -100 later, done by the collator)
```
```

## Cell 10 - Code: Demo - tokenize a single example end to end

Provenance: FROM-NEW

```python
# Demo the two-part tokenization on ONE example so the shapes are concrete.
ex = raw["train"][0]
ex_input = build_input(ex["question"], ex["context"])   # the prefixed input string
ex_target = ex["answers"]["text"][0]                     # the answer string

# Tokenize the input (what the encoder reads).
enc = tokenizer(ex_input, max_length=256, truncation=True)

# Tokenize the target (what the decoder must produce). The text_target argument tells the
# tokenizer to treat this string as a generation target rather than as model input.
lab = tokenizer(text_target=ex_target, max_length=32, truncation=True)

print("INPUT string :", ex_input[:90], "...")
print("input_ids len:", len(enc["input_ids"]))
print("TARGET string:", ex_target)
print("label ids    :", lab["input_ids"])
# Decode the label ids back to prove they round-trip to the answer text.
print("labels decode:", tokenizer.decode(lab["input_ids"], skip_special_tokens=True))
```

## Cell 11 - Markdown: Lab 1 - write the preprocess function

Provenance: FROM-NEW

```markdown
### Lab 1: write the seq2seq preprocess function (core, ~12 min)

You will write the function that turns a BATCH of SQuAD examples into the
`input_ids` / `labels` that `Seq2SeqTrainer` consumes. We then `.map(batched=True)` it over
the dataset.

The function receives `examples`, a dict of LISTS (because `batched=True`). For the batch:

1. Build the list of input strings. For each (question, context) pair, produce the prefixed
   string in the same `"question: ... context: ..."` format used in the demo. The helper that
   builds one such string already exists from the demo above; here you build one per pair.
2. Tokenize that list of input strings with truncation and `max_length=max_input` to get the
   model inputs.
3. Build the list of target strings: for each example, the FIRST gold answer (SQuAD stores
   answers as a list; you want the first element of each).
4. Tokenize the targets. Pass them through the tokenizer's TARGET path (the keyword argument
   that marks text as a generation target, shown in the demo cell above) with
   `max_length=max_target`.
5. Attach the tokenized target ids onto the model inputs under the key the Trainer reads for
   supervision (the same key name the demo printed when it showed "label ids").

Return the model-inputs dict. Do not pad here and do not set -100 here - the data collator
does both at batch time.
```

## Cell 12 - Code: Lab 1 starter + verification

Provenance: FROM-NEW

```python
# Lengths: inputs can be long (context paragraphs); answers are short.
max_input = 256
max_target = 32

def preprocess(examples):
    # 1. Build one prefixed input string per (question, context) pair in the batch.
    #    Hint: you have a helper that formats a single (question, context) into the prefixed
    #    string; apply it across the parallel lists examples["question"] and examples["context"].
    model_inputs = None  # YOUR CODE  (tokenize the list of prefixed input strings;
                         #             pass truncation=True and max_length=max_input)

    # 2. Build the list of target answer strings (the first gold answer of each example).
    targets = None  # YOUR CODE

    # 3. Tokenize the targets through the tokenizer's TARGET path with max_length=max_target
    #    and truncation=True.
    labels = None  # YOUR CODE

    # 4. Attach the tokenized target ids onto model_inputs under the supervision key the
    #    Trainer reads (the key the demo printed as "label ids").
    None  # YOUR CODE

    return model_inputs

# --- Verification (provided) ---
_sample = raw["train"].select(range(4))
_out = preprocess(_sample[:])  # pass the batch dict
assert _out is not None, "preprocess returned None - did you build and return model_inputs?"
assert "input_ids" in _out, "missing input_ids - did you tokenize the prefixed inputs?"
assert "labels" in _out, "missing labels - did you attach the tokenized targets?"
assert len(_out["input_ids"]) == 4, "expected 4 examples back from a batch of 4"
_decoded = tokenizer.decode([t for t in _out["labels"][0] if t >= 0], skip_special_tokens=True)
print("First decoded target:", _decoded)
print("Lab 1 verification passed.")
```

## Cell 13 - Code: Build the tokenized datasets and the collator

Provenance: FROM-NEW

```python
# Keep it class-sized so the fine-tune finishes in a few minutes.
# We take a small training slice and a small validation slice. The whole point is to SEE
# fine-tuning move the needle, not to win a leaderboard.
small_train = raw["train"].shuffle(seed=42).select(range(2000))
small_val = raw["validation"].shuffle(seed=42).select(range(200))

# Apply the preprocess function from Lab 1 to both splits.
# remove_columns drops the original text columns so only model tensors remain.
tokenized_train = small_train.map(preprocess, batched=True, remove_columns=small_train.column_names)
tokenized_val = small_val.map(preprocess, batched=True, remove_columns=small_val.column_names)

# The data collator pads each batch and - crucially - sets padded label positions to -100
# so they are ignored by the loss. We never had to write that by hand.
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print("Tokenized train:", tokenized_train)
print("Tokenized val:  ", tokenized_val)
```

## Cell 14 - Markdown: Concept 3 - fine-tuning with Seq2SeqTrainer

Provenance: FROM-NEW

```markdown
## Concept 3: fine-tuning with Seq2SeqTrainer

In C9 you used the plain `Trainer` for classification. For generation we use its sibling,
`Seq2SeqTrainer`, which knows how to call `.generate()` when it evaluates (so eval metrics
reflect real generated text, not just next-token loss).

The config object is `Seq2SeqTrainingArguments`. A few choices matter for seq2seq:

- **learning_rate**: T5 likes a higher learning rate than the classification default. Values
  around 3e-4 work well; the default 5e-5 trains painfully slowly for T5.
- **predict_with_generate=True**: tells the trainer to actually generate during evaluation.
- **fp16**: half-precision speeds training up a lot on a GPU, but it is only valid on CUDA, so
  we switch it on only when a GPU is present.
- **eval_strategy** (this is the current argument name; the old `evaluation_strategy` is
  deprecated): when to run evaluation. We evaluate once per epoch.

We keep `num_train_epochs` small (1) and the dataset small so the run fits class time.
```

## Cell 15 - Markdown: Lab 2 - configure and run the trainer

Provenance: FROM-NEW

```markdown
### Lab 2: configure Seq2SeqTrainingArguments and train (core, ~12 min)

Fill in four training-argument values, then build the trainer and call train. The theory cell
above named every value you need; translate each idea into a number or flag here.

1. Set the per-device TRAIN batch size to a small power of two that fits a T4 (drop it if you
   hit out-of-memory).
2. Set the LEARNING RATE to the T5-friendly value from the theory cell, clearly higher than
   the classification default.
3. Set the number of TRAIN EPOCHS so the model makes the smallest number of passes that still
   shows the jump.
4. Turn on generation during evaluation by setting the generate-during-eval flag.

The rest of the arguments (output dir, eval cadence once per epoch, the GPU-only half
precision flag) are provided. Then build the `Seq2SeqTrainer` with the model, the args, the
two tokenized splits, the tokenizer, and the data collator, and call `.train()`.

Expect roughly 3 to 6 minutes on a T4 GPU. When it finishes, the next cells will show the
model answering far better than the un-tuned baseline.
```

## Cell 16 - Code: Lab 2 starter + verification

Provenance: FROM-NEW

```python
training_args = Seq2SeqTrainingArguments(
    output_dir="t5-squad-qa",            # where checkpoints land
    eval_strategy="epoch",               # evaluate once per epoch (current arg name)
    save_strategy="no",                  # skip checkpoint saving to save disk/time in class
    per_device_train_batch_size=None,    # YOUR CODE  (small power of two that fits a T4)
    per_device_eval_batch_size=16,       # provided
    learning_rate=None,                  # YOUR CODE  (the T5-friendly rate from the theory cell)
    num_train_epochs=None,               # YOUR CODE  (fewest passes that still shows the jump)
    predict_with_generate=None,          # YOUR CODE  (the generate-during-eval flag)
    fp16=(device.type == "cuda"),        # half precision only on GPU (provided)
    logging_steps=50,                    # provided
    report_to="none",                    # no external loggers (provided)
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- Verification (provided) ---
assert training_args.per_device_train_batch_size is not None, "set the train batch size"
assert training_args.learning_rate and training_args.learning_rate > 1e-4, \
    "T5 wants a higher learning rate than the classification default"
assert training_args.num_train_epochs is not None, "set num_train_epochs"
assert training_args.predict_with_generate is True, "set predict_with_generate=True"
print("Config looks good. Starting training...")

trainer.train()
print("Training complete.")
```

## Cell 17 - Markdown: Concept 4 - generating answers + the before/after

Provenance: FROM-NEW

```markdown
## Concept 4: generating answers and seeing the jump

The model is fine-tuned. Now we read answers out of it the same way we did in the "before"
demo: build the prefixed input, tokenize, call `.generate()`, decode. The only difference is
the weights have learned the SQuAD QA pattern.

A couple of `.generate()` knobs worth knowing now (you will experiment with them in the
stretch):

- **max_new_tokens**: a cap on how long the answer can be. SQuAD answers are short, so 32 is
  plenty.
- **num_beams**: 1 is plain greedy decoding (take the single most likely next token every
  step). A value like 4 turns on beam search, which keeps several candidate answers alive and
  usually returns a cleaner result, at a little extra compute.

Reusing the same probe question from the "before" demo, we will see the un-tuned answer next
to the fine-tuned answer. That side-by-side is the whole story of transfer learning in one
print.
```

## Cell 18 - Markdown: Lab 3 - write the answer_question function

Provenance: FROM-NEW

```markdown
### Lab 3: write the answer_question function (core, ~8 min)

Write the single function the chatbot will call. It takes a `question` and a `context` string
and returns the model's answer string. Steps:

1. Build the prefixed input string from the question and context (same helper as before).
2. Tokenize it into tensors on the right device, with truncation and the input max length.
3. Generate output ids. Pass a sensible answer-length cap, and turn on beam search with a few
   beams for cleaner answers.
4. Decode the FIRST generated sequence back to text, skipping special tokens, and return it.

The verification block calls your function on the probe example and prints the fine-tuned
answer next to the un-tuned answer captured earlier, plus the gold answer, so you can see the
jump immediately.
```

## Cell 19 - Code: Lab 3 starter + verification (before/after)

Provenance: FROM-NEW

```python
def answer_question(question, context):
    # 1. Build the prefixed input string.
    text = None  # YOUR CODE

    # 2. Tokenize to tensors on the model's device (truncation=True, max_length=max_input).
    enc = None  # YOUR CODE

    # 3. Generate ids. Use a short answer cap and turn on beam search with a few beams.
    with torch.no_grad():
        out_ids = None  # YOUR CODE

    # 4. Decode the first sequence to text, skipping special tokens, and return it.
    return None  # YOUR CODE

# --- Verification (provided): before vs after ---
tuned_pred = answer_question(q, c)   # q, c, gold, pred were set in the "before" demo cell
assert isinstance(tuned_pred, str) and len(tuned_pred) > 0, "answer_question must return a non-empty string"
print("QUESTION      :", q)
print("GOLD ANSWER   :", gold)
print("T5 (un-tuned) :", pred)         # weak baseline from before training
print("T5 (fine-tuned):", tuned_pred)  # should now match or closely match the gold span
print("\nLab 3 verification passed.")
```

## Cell 20 - Code: Save then reload the fine-tuned model, and sanity-check

Provenance: FROM-NEW

```python
# Ship it: save, reload, and serve.
# A chatbot is a separate process from this training notebook, so it must LOAD the model from
# disk. Same pattern as DistilBERT in C9: save BOTH the model and the tokenizer to a folder,
# then reload BOTH. The model folder holds the weights and config; the tokenizer folder holds
# the vocabulary and the rules for turning text into ids. Reload one without the other and the
# model cannot read text.
save_dir = "t5-squad-qa-final"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Saved model + tokenizer to: {save_dir}")

# Reload from disk into fresh objects, exactly as the chatbot process would.
reloaded_model = AutoModelForSeq2SeqLM.from_pretrained(save_dir).to(device)
reloaded_tokenizer = AutoTokenizer.from_pretrained(save_dir)

# Point the live objects at the reloaded ones so answer_question uses the on-disk model.
model = reloaded_model
tokenizer = reloaded_tokenizer

# Sanity-check: the reloaded model answers the probe question.
print("Reloaded model answer:", answer_question(q, c))
```

## Cell 21 - Code: Guarded Gradio Q&A chatbot

Provenance: FROM-NEW

```python
# The finale: a Gradio Q&A chatbot. The user types a QUESTION and a CONTEXT passage, and the
# fine-tuned model writes an answer. We guard the import so a no-Gradio environment still runs
# every earlier cell and just prints a fallback instead of launching a UI.
try:
    import gradio as gr

    def chat_fn(question, context):
        # Reuse the exact function from Lab 3, so the UI and the notebook agree.
        if not question or not context:
            return "Please provide both a question and a context passage."
        return answer_question(question, context)

    # Two text inputs (question, context) -> one text output (the generated answer).
    demo = gr.Interface(
        fn=chat_fn,
        inputs=[
            gr.Textbox(label="Question", placeholder="What does the passage say about ...?"),
            gr.Textbox(label="Context", lines=6, placeholder="Paste a help-doc paragraph here"),
        ],
        outputs=gr.Textbox(label="Answer"),
        title="Seq2Seq Q&A Chatbot (fine-tuned t5-small)",
        description="Reads your question + context and writes an answer. The finale of the course.",
    )
    # In Colab, launch creates a public share link automatically.
    demo.launch(share=True)

except ImportError:
    # No Gradio here - fall back to a plain inference call so the cell still works.
    print("Gradio not installed; running a direct inference instead.\n")
    demo_q = "What architecture writes output token by token?"
    demo_c = ("Encoder-decoder models read the input with an encoder and generate the output "
              "with a decoder, one token at a time, which is called autoregressive generation.")
    print("Q:", demo_q)
    print("A:", answer_question(demo_q, demo_c))
```

## Cell 22 - Markdown: Stretch + Homework

Provenance: FROM-NEW

```markdown
## Stretch (fast finishers) and Homework (async)

### Stretch A: decoding-parameter sweep (in notebook, ~10 min)

`.generate()` has knobs that visibly change the answer. On the probe example, generate the
answer several times while varying ONE knob at a time, and read what each does:

- `num_beams=1` (greedy) vs `num_beams=4` (beam search): beam search usually returns a cleaner,
  steadier answer because it keeps several candidates alive instead of committing greedily.
- `max_new_tokens`: a tighter cap forces a shorter answer; too tight and the answer is cut off.
- `no_repeat_ngram_size=2`: forbids any 2-gram from repeating, which kills stutter like
  "the the the".
- `length_penalty`: above 1.0 nudges toward longer answers, below 1.0 toward shorter ones.

```python
# Sketch:
# for nb in [1, 4]:
#     ids = model.generate(**enc, max_new_tokens=32, num_beams=nb)
#     print(nb, tokenizer.decode(ids[0], skip_special_tokens=True))
```

### Stretch B: one prefix, a different task (in notebook, ~5 min)

T5 is "text in, text out". Change ONLY the task prefix from `"question: ..."` to
`"summarize: ..."` and feed it a paragraph (no fine-tuning needed - summarization was in T5's
pretraining). Watch the SAME model do a completely different job. That is the text-to-text
idea made visible.

### Homework (async, production-oriented)

1. **Measure it properly.** Use `evaluate.load("squad")` to compute exact-match and F1 on a
   larger held-out slice. Build the predictions list (each item needs the predicted text and
   the example id) and the references list (each needs the gold answers and the id), then call
   `metric.compute(...)`. Report your numbers.
2. **Probe hallucination.** Feed the model a question whose answer is NOT in the supplied
   context (or a context about a different topic). Observe that the model still confidently
   writes an answer. This is the central risk of generative QA: it will invent. Note when it
   does and how badly.
3. **Ground it (RAG).** Recall B5's `semantic_search(query, top_k=3)` over a `corpus`. A real
   assistant does not get the context handed to it - it RETRIEVES the most relevant passage
   first, then answers from that. Sketch a pipeline that uses B5 semantic search to fetch the
   context, then feeds it to `answer_question`. That is a retrieval-augmented generation (RAG)
   QA system, and grounding the answer in retrieved text is the main defense against the
   hallucination you saw in step 2.
```

## Cell 23 - Markdown: Wrap-up and the course finish line

Provenance: FROM-NEW

```markdown
## What you built, and where it leaves you

You fine-tuned an **encoder-decoder** model and shipped it as a Q&A chatbot. That completes
the architecture arc of the whole course:

- **Encoder-only** (B5/B7 embeddings, C9 DistilBERT): read an input, produce a label or a
  vector. One fast forward pass. Use it when the output is a class or a score.
- **Encoder-decoder** (C10 T5): read an input, WRITE a new sequence token by token. Use it
  when the output is free text whose content depends on the input - translation, summarization,
  abstractive question answering.

The trade-off you now understand: generation is autoregressive, so it is slower and costlier
than a single classification pass, and it can hallucinate. The production answer is to ground
it - retrieve a real context (B5 semantic search) and make the model answer from that. Question
+ retrieved context -> generated answer is exactly a RAG assistant.

Two honest notes to carry forward:

- Trained on SQuAD, whose answers are spans copied from the context, T5 mostly learns to COPY
  the right span rather than paraphrase. Its generative muscle really shows when the answer has
  to be synthesized (that is what the `"summarize:"` stretch demonstrates).
- A fine-tuned model is only as good and as fair as its data. The same care about data you
  practiced all course applies here.

### Next: C11 Capstone - ship your own chatbot

In the capstone you choose ONE path - an encoder-only classification chatbot (C9 shape) or an
encoder-decoder Q&A chatbot (C10 shape) - pick a public dataset, fine-tune, evaluate, and ship
it in Gradio. Everything you need is now in your hands.

### Resources

- T5 docs: https://huggingface.co/docs/transformers/model_doc/t5
- Seq2Seq QA example: https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering
- Generation strategies: https://huggingface.co/docs/transformers/generation_strategies
- Gradio Interface: https://www.gradio.app/guides/the-interface-class
```

# VERIFICATION CHECKLIST

- [ ] Install cell pins `numpy<2`, `transformers==4.57.1`, `datasets>=2.19,<3`; includes
      evaluate, accelerate, gradio; no gensim or sentence-transformers (unused in C10) (Cell 1).
- [ ] Colab "restart runtime after numpy downgrade" note present (Cell 0 intro + Cell 1 comment).
- [ ] Device select + `torch.manual_seed(42)` + `np.random.seed(42)` present (Cell 2).
- [ ] SQuAD loaded via `load_dataset("squad")`; answer read as `answers["text"][0]` (Cells 4, 10, 12).
- [ ] Concept 1 explains encoder vs decoder vs encoder-decoder with the translator analogy
      and the label-vs-text rule of thumb (Cell 5).
- [ ] `AutoModelForSeq2SeqLM.from_pretrained("t5-small")` loaded; encoder + decoder shown (Cell 6).
- [ ] Before-fine-tuning demo runs `.generate()` on un-tuned t5-small for the contrast (Cell 8).
- [ ] Concept 2 covers the task prefix, label padding with -100 via the collator, and no
      manual `decoder_input_ids` (teacher forcing) (Cell 9).
- [ ] Lab 1 (preprocess fn) starter never names the methods/keys that solve each line; hints
      reference role and the demo output (Cells 11-12). Verification block provided.
- [ ] `DataCollatorForSeq2Seq` used; small_train=2000, small_val=200 for class time (Cell 13).
- [ ] Concept 3 names `Seq2SeqTrainer`, `Seq2SeqTrainingArguments`, `eval_strategy` (not the
      deprecated `evaluation_strategy`), `predict_with_generate`, GPU-only fp16, T5 LR (Cell 14).
- [ ] Lab 2 (training args) starter leaves batch size, LR, epochs, predict_with_generate blank
      with role-only hints that point back to the theory cell (no literal target values in the
      starter comments); provides fp16/eval/output args; verification asserts LR>1e-4 (Cells 15-16).
- [ ] Concept 4 + Lab 3 (answer_question) produce the before/after side-by-side (Cells 17-19).
- [ ] Save with `save_pretrained` (model AND tokenizer); reload with `from_pretrained` (Cell 20).
- [ ] Gradio chatbot guarded with try/except ImportError; two text inputs (question, context),
      one text output; non-Gradio fallback prints a direct inference (Cell 21).
- [ ] Stretch A (decoding sweep), Stretch B (one-prefix-to-summarize), and the 3-part homework
      (EM/F1 via evaluate, hallucination probe, RAG via B5 semantic_search) present (Cell 22).
- [ ] Wrap-up states encoder-only vs encoder-decoder choice, autoregressive cost, RAG grounding,
      the copy-vs-paraphrase honesty note, and the bridge to C11 (Cell 23).
- [ ] Every cell tagged `Provenance: FROM-NEW` (NEW notebook; no migration map needed).
- [ ] Plain ASCII only: no em/en dashes, no Unicode multiplication, no emoji.
- [ ] Cell budget: 24 cells (0-23), within the 20-25 band.

# RESEARCH VALIDATED (June 2026)

- HuggingFace seq2seq QA example confirms T5/BART are generative QA models fine-tuned with the
  Seq2Seq pipeline (generate the answer, not predict start/end spans).
  https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/README.md
- T5 task prefixes ("summarize:", "question:") are required and were used in pretraining; T5
  auto-creates decoder_input_ids from labels (teacher forcing); T5 wants a higher LR (~1e-4 to
  3e-4) than the classification default.
  https://discuss.huggingface.co/t/t5-finetuning-tips/684/2 and
  https://huggingface.co/docs/transformers/model_doc/t5
- DataCollatorForSeq2Seq pads labels with -100 (label_pad_token_id default -100), ignored by
  the PyTorch loss; targets tokenized via the tokenizer text_target path.
  https://huggingface.co/docs/datasets/en/about_map_batch
- `evaluation_strategy` is deprecated; the current TrainingArguments name is `eval_strategy`.
  https://discuss.huggingface.co/t/solved-difference-between-eval-strategy-and-evaluation-strategy/96657
- SQuAD schema: features id/title/context/question/answers, answers has text (list) and
  answer_start (list); first answer = answers["text"][0].
  https://huggingface.co/datasets/rajpurkar/squad
- `predict_with_generate=True` makes Seq2SeqTrainer generate during eval; fp16 speeds GPU
  training; valid only on CUDA.
  https://medium.com/@anyuanay/fine-tuning-the-pre-trained-t5-small-model-in-hugging-face-for-text-summarization-3d48eb3c4360
- t5-small zero-shot QA is weak (not instruction-tuned), giving a clean before/after contrast;
  Flan-T5 would be too strong zero-shot and erase the contrast.
  https://huggingface.co/google/flan-t5-base
- t5-small fine-tune time scales down to a few minutes on a single T4 for a ~2000-example
  subset (29,718 examples took ~2h on 2x T4).
  https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb
- transformers 4.57 requires Python >=3.10 and is tested with accelerate 1.x; accelerate is
  required for Trainer. evaluate 0.4.6 is installable in 2026.
  https://github.com/huggingface/transformers/issues/41339 and https://pypi.org/project/evaluate/
- Colab ships numpy 2.0.2; pinning numpy<2 forces a downgrade that needs a runtime restart.
  https://github.com/googlecolab/colabtools/issues/5205
- gradio is numpy<2 compatible; `gr.Interface` takes a list of inputs; launch(share=True)
  auto-creates a public link in Colab.
  https://www.gradio.app/guides/the-interface-class and https://www.gradio.app/guides/sharing-your-app
- save_pretrained / from_pretrained reload a fine-tuned seq2seq model + tokenizer from a folder.
  https://huggingface.co/docs/transformers/model_doc/t5
- Generation params: num_beams>1 = beam search (cleaner output), length_penalty shapes length,
  no_repeat_ngram_size prevents repeated n-grams, early_stopping controls beam stop.
  https://huggingface.co/docs/transformers/generation_strategies
- Generative QA can hallucinate when the answer is not in the context (SQuAD 2.0 unanswerable
  framing); grounding answers in retrieved context (RAG) is the main mitigation.
  https://arxiv.org/pdf/1806.03822 and
  https://medium.com/@hrk84ya/rag-in-2025-from-quick-fix-to-core-architecture-9a9eb0a42493
- Encoder-only = classification/embeddings; encoder-decoder = generation where output depends
  on input. Generation is autoregressive hence slower than one classification pass.
  https://sebastianraschka.com/books/ml-q-and-ai-chapters/ch17/
- T5 fine-tuned on SQuAD tends to COPY spans (near-extractive) because SQuAD answers are spans;
  the generative power shows on synthesis tasks like summarization.
  https://huggingface.co/docs/transformers/model_doc/t5

---

Plan complete. Build with `/build-notebook encoder-decoder-qa-chatbot colab`.
