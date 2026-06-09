# A2 - Pipeline Tour - Cell-by-Cell Plan

## Status

NEW. New file: `A-What-NLP-Can-Do/2-Pipeline_Tour.ipynb`. Built from scratch (no old
source notebook). Every cell is authored fresh: `Provenance: FROM-NEW`.

## Context

- **What students arrive with (from A1):** they just did NLP by hand with classical tools.
  Concretely they have a spaCy `nlp` object, read entities off `doc.ents`, tokenized with
  `TextBlob(x).words`, and ran a small LDA / BERTopic topic model. They felt the effort:
  every task needed its own tool, its own preprocessing, its own glue. No PyTorch yet.
- **The key insight they leave with:** one function, `pipeline("<task>")`, downloads a
  pretrained transformer and solves a real NLP task in three lines, with zero training.
  They will have solved six tasks (sentiment, zero-shot classification, NER, question
  answering, summarization, fill-mask) and learned to swap models from the HuggingFace Hub
  with a single `model=` keyword. They leave wanting to know HOW the black box works,
  which is exactly what Part B opens.
- **Names introduced (no collision with A1's `nlp` / `doc`):** `classifier`, `router`,
  `candidate_labels`, `ner`, `qa`, `summarizer`, `unmasker`, `device`.

## Chatbot Through-Line

The course end goal is a fine-tuned transformer in a simple Gradio chatbot. A2 advances
that goal in two concrete ways. First, the default `sentiment-analysis` pipeline loads
`distilbert-base-uncased-finetuned-sst-2-english`, which is literally
`distilbert-base-uncased` (the model C9 fine-tunes) already fine-tuned on SST-2 (a C9
dataset). Students meet the finished product before they build it. Second, the closing
cell previews `gr.Interface.from_pipeline(...)`, showing that a pipeline already IS a
chatbot backend: wrap it in one line and you have a UI. The one-line bridge to the next
notebook: "this pipeline is a sealed black box; in Part B we build every piece inside it,
and in Part C we fine-tune and ship our own."

## STAR Story

- **Situation:** You are an NLP engineer dropped onto a customer-support platform. Tickets
  arrive as raw text and the product team wants sentiment, routing, entity extraction, and
  short summaries shipped this sprint. There is no labelled training data and no time to
  build models from scratch.
- **Task:** Stand up a working baseline for five or more NLP tasks today, with no training,
  so the team can see real output and decide where to invest.
- **Action:** Use the HuggingFace `pipeline()` API. For each task, pick the task string, let
  the default pretrained transformer download, and call it on a real support message. Use
  zero-shot classification to route tickets to custom categories with no training data, then
  swap in a better model from the Hub with one keyword to compare.
- **Result:** Six NLP tasks solved in a handful of lines each, a zero-shot ticket router that
  works on day one, and a clear-eyed view of where pretrained baselines are enough and where
  fine-tuning (Part C) will be needed. The team has a demo by end of day.

## Deliverables

- Exercise: `A-What-NLP-Can-Do/2-Pipeline_Tour.ipynb`
- Solution: `Solutions/A-What-NLP-Can-Do/2-Pipeline_Tour.ipynb`

## Session Timing (~60-90 min)

| Segment | Cells | Minutes |
|---------|-------|---------|
| Setup and context | 0-5 | 12 |
| The pipeline mental model + sentiment | 6-7 | 12 |
| Sentiment demo | 8 | 5 |
| Zero-shot classification + Lab 1 | 9-12 | 18 |
| Named entity recognition (vs A1 spaCy) | 13-14 | 8 |
| Question answering | 15-16 | 8 |
| Summarization | 17-18 | 7 |
| Fill-mask (breadth) | 19 | 6 |
| Swap models from the Hub + Lab 2 stretch | 20-22 | 12 |
| Gradio preview, bridge, wrap-up | 23-24 | 8 |

Total: 25 cells, about 90 minutes including both labs.

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~20-25 cells; this plan: 25)

## Cell 0 - Markdown: Title, objectives, prerequisites

```markdown
# A2 - The Pipeline Tour: Real NLP in Three Lines

**Part A - What NLP Can Do | Notebook 2 of 3**

In A1 you did NLP by hand. You tokenized with TextBlob, pulled entities out of a spaCy
`doc`, and ran a topic model. It worked, but every task needed its own tool and its own
glue code. This notebook is the payoff. You will watch a pretrained transformer solve each
of those tasks (and several more) in about three lines, with no training at all.

By the end of this notebook you will be able to:

- Explain the `pipeline()` mental model: pick a task, get a default pretrained model, call it.
- Run sentiment analysis, zero-shot classification, named entity recognition, question
  answering, summarization, and fill-mask, each in a few lines.
- Use zero-shot classification to route text into your own custom categories with no training data.
- Swap models from the HuggingFace Hub with a single `model=` keyword and compare them.

**Prerequisites:** A1 (you have seen tokenization and NER done by hand). No PyTorch knowledge
needed yet. We treat the transformer as a black box on purpose. Part B opens that box.

**Runtime:** Google Colab. A GPU is optional and makes the larger models faster, but every
cell runs on CPU. Set the runtime with `Runtime -> Change runtime type -> GPU` if you have one.
```

Provenance: FROM-NEW

## Cell 1 - Markdown: Section 0 - Environment Setup

```markdown
## Section 0: Environment Setup

We pin the HuggingFace stack to the `transformers` 4.x line on purpose. The 4.x line keeps
the beginner-friendly `summarization` and `question-answering` pipelines that this tour
relies on, and it works cleanly with `numpy<2` (the version the rest of the course uses).
Run the install cell once. On Colab you may be prompted to restart the runtime after the
install so the pinned versions are the ones actually imported.
```

Provenance: FROM-NEW

## Cell 2 - Code: Install pinned dependencies

```python
# Install the HuggingFace stack, pinned for this course.
# transformers 4.57.x is the last 4.x line: it keeps the summarization and
# question-answering pipelines that transformers 5.x removed, and it is happy with numpy<2.
# If Colab asks you to restart the runtime after this cell, do it, then continue.
!pip install -q "transformers==4.57.*" "datasets>=2.19,<4" "evaluate>=0.4.2" "accelerate>=0.30" "numpy<2"
```

Provenance: FROM-NEW

## Cell 3 - Code: Imports, version check, device, seed

```python
# Standard imports for the whole notebook.
import numpy as np
import torch
from transformers import pipeline
import transformers

# Confirm we got the versions we pinned (transformers 4.57.x, numpy < 2).
print(f"transformers version: {transformers.__version__}")
print(f"numpy version:        {np.__version__}")

# Device selection. The pipeline() API uses an integer device argument:
#   device = 0   -> first CUDA GPU
#   device = -1  -> CPU
# We compute it once and pass it to every pipeline so GPU is used when available.
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

# Reproducibility. Pipelines are mostly deterministic for these tasks, but we set the seed
# so any sampling (for example in generation) behaves the same on every run.
torch.manual_seed(42)

print("\nEnvironment setup complete.")
```

Provenance: FROM-NEW

## Cell 4 - Markdown: What are we building today (story)

```markdown
## What Are We Building Today?

You are an NLP engineer on a customer-support platform. Tickets land as raw text all day:

> "I have been charged twice for my subscription and support has not replied in 3 days.
> My account email is jordan@example.com and I am in Berlin."

The product team wants four things shipped this sprint: read the sentiment of each ticket,
route it to the right team, pull out the useful entities (people, places, emails), and
produce a one-line summary for the dashboard. There is no labelled training data and no
time to train models.

This is exactly the situation pretrained transformers were built for. In the rest of this
notebook we solve each of these tasks with the `pipeline()` API, no training required, then
swap in better models to see how easy it is to upgrade. By the end you will have a working
baseline for five or more NLP tasks and a clear sense of where these baselines are good
enough and where you would invest in fine-tuning later (Part C).
```

Provenance: FROM-NEW

## Cell 5 - Code: A sample support ticket we will reuse

```python
# One realistic support ticket we will reuse across several tasks, so you can see how the
# same input flows through different pipelines. Keep it short so every cell runs fast.
ticket = (
    "I have been charged twice for my subscription and support has not replied in 3 days. "
    "My account email is jordan@example.com and I am in Berlin."
)

print("Sample support ticket:\n")
print(ticket)
```

Provenance: FROM-NEW

## Cell 6 - Markdown: The pipeline() mental model

```markdown
## The `pipeline()` Mental Model

Everything in this notebook is one idea repeated. The `pipeline()` function from
`transformers` takes a task name and hands you back a ready-to-call object:

```python
from transformers import pipeline
solver = pipeline("<task-name>")   # downloads a default pretrained model the first time
result = solver("some input text") # call it like a function
```

Three things happen behind that one line:

1. **Task to model.** HuggingFace looks up a sensible default pretrained model for that task
   and downloads it from the Hub (the first time only; after that it is cached and instant).
2. **Preprocess and run.** The text is tokenized into the numbers the model expects, run
   through the transformer, and the raw output is collected.
3. **Postprocess.** The numbers are turned back into something human: a label, a score, a
   span of text, a summary.

You do not have to know any of that yet. That is the whole point: the pipeline is a black
box. In Part B we build every piece inside it, and in Part C we fine-tune our own. For now,
learn the rhythm: **pick a task, get a model, call it.**

### Task 1: Sentiment Analysis (our first task)

We start with sentiment analysis: label a piece of text as positive or negative (with a
confidence score). The default model behind `pipeline("sentiment-analysis")` is
`distilbert-base-uncased-finetuned-sst-2-english`. Hold onto that name: it is
`distilbert-base-uncased` (a model we will fine-tune ourselves in Part C) already fine-tuned
on the SST-2 movie-review dataset. You are about to use the exact thing you will later build.

The output is a list of dicts, one per input:

```python
[{'label': 'POSITIVE', 'score': 0.9991}]
```

You can pass a single string or a list of strings. Passing a list runs them as a batch and
returns one result per input, which is how you would score a whole queue of tickets.
```

Provenance: FROM-NEW

## Cell 7 - Code: Minimal pipeline() demo (one line of wow)

```python
# The smallest possible demo: one task, one call, one result.
# "sentiment-analysis" is a built-in task name. With no model= argument, HuggingFace picks
# a strong default and downloads it the first time (this first call may take a few seconds).
quick = pipeline("sentiment-analysis", device=device)

# Call it like a function. The result is a list with one dict per input.
print(quick("This three-line demo is honestly kind of magical."))
# Expect something like: [{'label': 'POSITIVE', 'score': 0.9998...}]
```

Provenance: FROM-NEW

## Cell 8 - Code: Sentiment demo on real reviews

```python
# Build the sentiment pipeline once and reuse it.
classifier = pipeline("sentiment-analysis", device=device)

# A small batch of customer messages. Passing a list runs them together and returns
# one result dict per input, in order.
reviews = [
    "The new dashboard is fast and the support team fixed my issue in minutes.",
    "Billed me twice and nobody answered for three days. Cancelling.",
    "It works, I guess. Nothing special, nothing terrible.",
]

results = classifier(reviews)

# Each result is a dict with a 'label' (POSITIVE / NEGATIVE) and a 'score' (confidence 0..1).
for review, r in zip(reviews, results):
    print(f"[{r['label']}  {r['score']:.3f}]  {review}")
```

Provenance: FROM-NEW

## Cell 9 - Markdown: Task 2 - Zero-shot classification (the superpower)

```markdown
## Task 2: Zero-Shot Classification (No Training Data)

This is the one to remember. Sentiment analysis only knows the labels it was trained on
(positive / negative). But your support tickets need custom categories that no model was
trained on: `billing`, `technical`, `account`, `shipping`. Normally you would label
thousands of examples and train a classifier. Zero-shot classification skips all of that.

You give the pipeline your candidate labels at call time, in plain English, and it scores
how well each one fits, with no training data at all:

```python
router = pipeline("zero-shot-classification")
router("My card was charged twice", candidate_labels=["billing", "technical", "account"])
```

The default model is `facebook/bart-large-mnli`. The output is a single dict:

```python
{'sequence': 'My card was charged twice',
 'labels':  ['billing', 'account', 'technical'],   # sorted best-first
 'scores':  [0.95, 0.03, 0.02]}                     # aligned with labels
```

`labels[0]` is the prediction and `scores[0]` is its confidence. By default the scores sum
to 1 (the model picks one best label). Pass `multi_label=True` if more than one label can
apply at once. This is why zero-shot is a practitioner favourite: you can ship a router on
day one and change the categories by editing a Python list.
```

Provenance: FROM-NEW

## Cell 10 - Code: Zero-shot demo (routing the ticket)

```python
# Build the zero-shot router once. The default model (facebook/bart-large-mnli) is larger,
# so the first download takes a bit longer; after that it is cached.
router = pipeline("zero-shot-classification", device=device)

# Our custom routing categories. These were never seen during the model's training:
# we are inventing them right here, in plain English.
candidate_labels = ["billing", "technical support", "account", "shipping"]

# Route our sample ticket. We pass the text plus the candidate labels.
result = router(ticket, candidate_labels=candidate_labels)

# labels and scores come back aligned and sorted best-first.
print(f"Ticket: {result['sequence']}\n")
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label:20s} {score:.3f}")

print(f"\nRoute this ticket to: {result['labels'][0].upper()}")
```

Provenance: FROM-NEW

## Cell 11 - Markdown: Lab 1 instructions - build a ticket router

```markdown
## Lab 1: Build Your Own Zero-Shot Ticket Router (15 min)

Your turn. You will route a small queue of support messages into your own categories using
zero-shot classification, with no training data.

**Steps:**

1. You already have a `router` pipeline from the demo above. Reuse it.
2. Define `my_labels`: a Python list of at least four routing categories that make sense for
   a support desk (think about the kinds of problems customers actually report).
3. For each message in `support_queue` (provided), call the router with your labels and keep
   the top predicted label.
4. Store the chosen labels in a list called `predicted_routes`, in the same order as the queue.

**Hints:**

- The router is called as `router(text, candidate_labels=my_labels)`.
- The best label for one call is at `result["labels"][0]`.
- The verification cell below checks that you produced one route per message.

**Stretch (if you finish early):** add a confidence guardrail. If the top score is below
0.5, set the route to `"needs human review"` instead of the predicted label. This is exactly
how real routers avoid acting on low-confidence guesses.
```

Provenance: FROM-NEW

## Cell 12 - Code: Lab 1 starter (exercise) with verification

```python
# Lab 1: Zero-shot ticket router.

# A small queue of incoming support messages (provided).
support_queue = [
    "My payment failed but I was still charged. Please refund me.",
    "The app crashes every time I open the reports page.",
    "I need to change the email address on my account.",
    "Where is my order? It has been two weeks and nothing arrived.",
]

# 1. Define your own routing categories (at least four labels).
my_labels = None  # YOUR CODE

# 2. Route each message and collect the top label for each one.
predicted_routes = []  # fill this in, one route per message, same order as support_queue

for message in support_queue:
    # Call the router on `message` with your `my_labels`, then append the top label
    # to predicted_routes.
    top_label = None  # YOUR CODE
    predicted_routes.append(top_label)

# --- Verification (provided) ---
if my_labels is not None and all(r is not None for r in predicted_routes):
    assert len(predicted_routes) == len(support_queue), "One route per message expected."
    print("Routing results:\n")
    for msg, route in zip(support_queue, predicted_routes):
        print(f"  -> {route:20s} | {msg}")
    print("\nNice. You built a working router with zero training data.")
else:
    print("Fill in my_labels and the routing loop, then re-run.")
```

Provenance: FROM-NEW

## Cell 13 - Markdown: Task 3 - Named entity recognition (vs A1's spaCy)

```markdown
## Task 3: Named Entity Recognition (Remember A1?)

In A1 you extracted entities by hand with spaCy: load `nlp`, run the text, read `doc.ents`.
Here is the same task as a one-line pipeline. The task name is `"ner"` (an alias of
`"token-classification"`), and we pass `aggregation_strategy="simple"`.

That aggregation argument matters. Transformers split words into subword tokens, so a name
like "jordan" might come back as `jo` + `##rdan`. With `aggregation_strategy="simple"`, the
pipeline stitches those pieces back into whole entities for you, returning a clean list:

```python
[{'entity_group': 'PER', 'score': 0.99, 'word': 'Jordan', 'start': 10, 'end': 16}, ...]
```

`entity_group` is the type (PER = person, LOC = location, ORG = organization), `word` is the
text, and `start`/`end` are character offsets. Same job spaCy did in A1, now from a
pretrained transformer in a single call.
```

Provenance: FROM-NEW

## Cell 14 - Code: NER demo on the ticket

```python
# Build the NER pipeline. aggregation_strategy="simple" merges subword pieces back into
# whole entities so we get clean, readable output.
ner = pipeline("ner", aggregation_strategy="simple", device=device)

# Run it on our support ticket (which mentions a person, a place, and an email).
entities = ner(ticket)

print(f"Ticket: {ticket}\n")
print("Entities found:")
for ent in entities:
    # entity_group: PER / LOC / ORG / MISC ; word: the matched text ; score: confidence
    print(f"  {ent['entity_group']:6s} {ent['word']:20s} (score {ent['score']:.3f})")
```

Provenance: FROM-NEW

## Cell 15 - Markdown: Task 4 - Question answering

```markdown
## Task 4: Question Answering

Extractive question answering takes a `context` (a passage) and a `question`, and returns
the span of the context that answers it. The default model is
`distilbert-base-cased-distilled-squad`, fine-tuned on the SQuAD dataset (the same dataset
the Part C Q&A chatbot uses).

You call it with two named arguments:

```python
qa(question="How long has support been silent?", context=ticket)
```

The output is a single dict:

```python
{'answer': '3 days', 'score': 0.87, 'start': 71, 'end': 77}
```

`answer` is the extracted text, `score` is the confidence, and `start`/`end` are the
character offsets inside the context. Note what this is NOT: it does not invent an answer,
it points at the part of the context that contains one.
```

Provenance: FROM-NEW

## Cell 16 - Code: Question answering demo

```python
# Build the question-answering pipeline (default: distilbert fine-tuned on SQuAD).
qa = pipeline("question-answering", device=device)

# A slightly longer context so there is something to extract answers from.
context = (
    "Jordan opened a support ticket on Monday after being charged twice for the Pro plan. "
    "Support did not reply for 3 days. The refund of 49 dollars was finally issued on Friday."
)

# Ask a few questions against the same context.
questions = [
    "Who opened the support ticket?",
    "How long did support take to reply?",
    "How much was the refund?",
]

for q in questions:
    answer = qa(question=q, context=context)
    # answer['answer'] is the extracted span; answer['score'] is the confidence.
    print(f"Q: {q}\nA: {answer['answer']}  (score {answer['score']:.3f})\n")
```

Provenance: FROM-NEW

## Cell 17 - Markdown: Task 5 - Summarization

```markdown
## Task 5: Summarization

Summarization condenses a longer passage into a few sentences. Unlike the tasks above, this
one generates new text rather than picking a label or a span. The default model is
`sshleifer/distilbart-cnn-12-6`, a distilled BART trained for news summarization.

```python
summarizer = pipeline("summarization")
summarizer(long_text, max_length=60, min_length=20)
```

The output is a list with one dict:

```python
[{'summary_text': 'A customer was charged twice and waited three days for a reply...'}]
```

Two gotchas worth knowing:

- `max_length` and `min_length` are measured in tokens and control the summary length.
- If your input is shorter than `max_length`, the model warns you (it cannot summarize a
  sentence into a longer summary). Feed it a real paragraph, not a single line.
```

Provenance: FROM-NEW

## Cell 18 - Code: Summarization demo

```python
# Build the summarization pipeline (default: distilbart fine-tuned on CNN news).
summarizer = pipeline("summarization", device=device)

# A paragraph long enough to actually summarize. Too-short inputs trigger a length warning.
long_text = (
    "Over the past quarter our support platform handled a record number of tickets. "
    "Customers most often reported duplicate billing charges, slow response times, and "
    "trouble updating account details. The engineering team shipped a faster dashboard and "
    "an automated refund flow, which cut average resolution time from four days to under one. "
    "However, billing complaints remain the single largest category and are the focus for "
    "next quarter, along with proactive notifications when a payment is retried."
)

# Control the summary length in tokens. min_length avoids one-word summaries; max_length caps it.
summary = summarizer(long_text, max_length=60, min_length=20)

# Output is a list with one dict; the text lives under 'summary_text'.
print("Summary:\n")
print(summary[0]["summary_text"])
```

Provenance: FROM-NEW

## Cell 19 - Code: Task 6 - Fill-mask demo (markdown intro folded in)

Lead-in markdown to render directly above this code cell in the notebook (keep it short,
two sentences, then run the code):

> **Task 6: Fill-Mask (one more, for breadth).** Fill-mask is the actual pretraining task
> behind models like BERT: write a sentence with a special mask token and the model predicts
> the most likely words for the blank, ranked by probability. The default model is
> `distilroberta-base` (mask token `<mask>`); BERT-family models use `[MASK]`, so read the
> token off `unmasker.tokenizer.mask_token` rather than guessing. The output is a ranked list
> of dicts like `[{'sequence': '... very frustrated ...', 'score': 0.21, 'token_str': ' frustrated'}, ...]`.

```python
# Task 6: Fill-mask. This is the actual pretraining task behind BERT-family models:
# predict the masked-out word. It is a nice peek at where these models come from.
# Default model: distilroberta-base, whose mask token is <mask>.
unmasker = pipeline("fill-mask", device=device)

# Read the correct mask token off the tokenizer instead of hard-coding it. The token differs
# by model family (<mask> for RoBERTa-family, [MASK] for BERT), so never guess it.
mask = unmasker.tokenizer.mask_token
print(f"This model's mask token is: {mask}\n")

# Predict the blank. Use the mask variable so this works regardless of the model.
sentence = f"The customer was very {mask} with the slow billing response."
predictions = unmasker(sentence)

# Output is a ranked list of dicts; show the top few candidate words and their scores.
for p in predictions[:5]:
    print(f"  {p['token_str']!r:15s} (score {p['score']:.3f})")
```

Provenance: FROM-NEW

## Cell 20 - Markdown: Swapping models from the Hub

```markdown
## Swapping Models From the HuggingFace Hub

Every pipeline so far used a default model. The real power of the Hub is that you can swap in
a different model for the same task with one keyword: `model=`.

```python
# Default sentiment model (POSITIVE / NEGATIVE):
pipeline("sentiment-analysis")

# A Twitter-tuned model that also knows NEUTRAL:
pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
```

Why swap?

- **Domain fit.** A model trained on tweets handles informal text better than one trained on
  movie reviews.
- **More classes.** Some sentiment models add a NEUTRAL label, which matters for support text.
- **Size and speed.** Distilled models (names often starting with `distil`) are roughly half
  the size and noticeably faster, trading a small amount of accuracy for latency.

How to choose? Open the model's page on huggingface.co and read its model card: what data it
was trained on, what labels it outputs, how big it is. Different models return different
label sets, so always check the output format after a swap. This "read the card, then swap"
habit is the practitioner's superpower for the rest of the course.
```

Provenance: FROM-NEW

## Cell 21 - Markdown: Lab 2 instructions - swap and compare (stretch)

```markdown
## Lab 2: Swap a Model and Compare (Stretch, 12 min)

Now compare two models on the same task and see how the choice changes the answer.

**Steps:**

1. Build a second sentiment pipeline that uses
   `model="cardiffnlp/twitter-roberta-base-sentiment-latest"`. Call it `twitter_sentiment`.
2. You already have the default `classifier` from Task 1. Run BOTH models on every message
   in `tricky_messages` (provided), which includes sarcasm and neutral statements.
3. For each message, print the label and score from each model side by side.
4. Write a one-line observation (as a comment or print) about where the two models disagree.

**Hints:**

- Build the second pipeline exactly like the first, but add the `model=` argument.
- The Twitter model returns labels like `negative` / `neutral` / `positive`, while the
  default returns `POSITIVE` / `NEGATIVE`. That difference is the point: read each output.
- Each result is still a list with one dict, so the top result is `result[0]`.

**Homework extension (async, deeper):** turn this into a confidence-thresholded router. Run
the default sentiment model over a batch of messages, accept a label only when its score is
above a threshold you choose (say 0.9), and otherwise mark the message `"low confidence:
review"`. Then write two sentences on when you would stop using zero-shot or a generic
pretrained model and fine-tune your own instead. (Hint: think about narrow, high-stakes
categories like billing disputes. This is exactly the decision that leads into Part C.)
```

Provenance: FROM-NEW

## Cell 22 - Code: Lab 2 starter (exercise) with verification

```python
# Lab 2: Swap a model and compare on the same task.

# Messages that are genuinely hard: sarcasm, neutral tone, mixed feelings.
tricky_messages = [
    "Oh great, charged twice again. Exactly what I wanted today.",
    "The update installed without any problems.",
    "It is fine. Does the job. Would not rave about it.",
]

# 1. Build a second sentiment pipeline that uses the Twitter-tuned model.
twitter_sentiment = None  # YOUR CODE

# 2. Run BOTH models on each message and print their labels side by side.
#    You already have `classifier` (the default model) from Task 1.
if twitter_sentiment is not None:
    print(f"{'message':50s} | {'default':18s} | twitter-roberta")
    print("-" * 95)
    for msg in tricky_messages:
        default_pred = None   # YOUR CODE: run `classifier` on msg, take result[0]
        twitter_pred = None   # YOUR CODE: run `twitter_sentiment` on msg, take result[0]
        if default_pred is not None and twitter_pred is not None:
            d = f"{default_pred['label']} {default_pred['score']:.2f}"
            t = f"{twitter_pred['label']} {twitter_pred['score']:.2f}"
            print(f"{msg[:48]:50s} | {d:18s} | {t}")
else:
    print("Build twitter_sentiment first, then fill in the comparison loop.")

# 3. As a comment or print, note where the two models disagree and why that might be.
```

Provenance: FROM-NEW

## Cell 23 - Code: Gradio preview - a pipeline is already a chatbot backend

```python
# A glimpse of where this course is heading. Gradio can wrap any pipeline in a UI in one
# line with gr.Interface.from_pipeline. We guard the import so this notebook still runs
# end-to-end in an environment without gradio installed.
try:
    import gradio as gr

    # Turn our sentiment classifier into an interactive web UI. In Colab this launches an
    # inline app you can type into. The pipeline IS the backend; Gradio just wraps it.
    demo = gr.Interface.from_pipeline(classifier)
    # demo.launch()  # uncomment in Colab to open the interactive app
    print("Gradio is available. Uncomment demo.launch() to try the interactive app.")
    print("This is the shape of the final deliverable: a model behind a simple chat UI.")
except ImportError:
    # No gradio in this environment. The point still stands: a pipeline is a ready backend.
    print("gradio is not installed here (that is fine).")
    print("Key idea: gr.Interface.from_pipeline(classifier) would wrap this pipeline in a")
    print("web UI in one line. In Part C you will load YOUR fine-tuned model the same way.")
```

Provenance: FROM-NEW

## Cell 24 - Markdown: Wrap-up, key takeaways, bridge to Part B

```markdown
## What You Learned, and Where This Goes Next

In one notebook, with no training, you solved six NLP tasks:

| Task | Pipeline | Default model |
|------|----------|---------------|
| Sentiment | `pipeline("sentiment-analysis")` | distilbert-base-uncased-finetuned-sst-2 |
| Zero-shot routing | `pipeline("zero-shot-classification")` | facebook/bart-large-mnli |
| Named entities | `pipeline("ner", aggregation_strategy="simple")` | bert-large finetuned conll03 |
| Question answering | `pipeline("question-answering")` | distilbert-base-cased-distilled-squad |
| Summarization | `pipeline("summarization")` | sshleifer/distilbart-cnn-12-6 |
| Fill-mask | `pipeline("fill-mask")` | distilroberta-base |

**Key takeaways:**

- The pattern never changed: pick a task, get a pretrained model, call it.
- Zero-shot classification gives you a custom classifier with no labelled data: edit a list,
  re-route. It is your fastest path to a working baseline.
- Swap models from the Hub with one `model=` keyword; always read the model card and check
  the output format, because labels differ between models.
- Pretrained baselines are excellent starting points, but for narrow, high-stakes categories
  you will eventually fine-tune your own model. That is Part C.

**Production take-homes (worth remembering):**

- The first call downloads and caches the model; later calls are instant.
- Pin a model `revision="<commit-hash>"` when you need reproducible results across runs.
- Pass a list and set `batch_size=` to score many inputs efficiently on a GPU.

**The bridge.** Every pipeline here was a sealed black box. Inside it, three things happened:
a tokenizer turned text into numbers, a transformer (built from PyTorch tensors and neural
network layers) processed them, and a postprocessor turned the output back into a label or a
span. In **Part B** you build every one of those pieces yourself: tensors, word embeddings,
neural networks, and an MLP classifier. In **Part C** you fine-tune `distilbert-base-uncased`
(the very model behind today's sentiment pipeline) on your own data and load it into a Gradio
chatbot like the preview above. You have seen the destination. Now we learn how to get there.

**Next notebook:** A3 - Capstone A, where you ship a real customer-support router on the
Twitter support dataset using the zero-shot skill you just learned.
```

Provenance: FROM-NEW

# VERIFICATION CHECKLIST

- [ ] Install cell pins `transformers==4.57.*` and `numpy<2`; runs on Colab CPU and GPU.
- [ ] transformers 4.x confirmed (5.x removed summarization + question-answering pipelines).
- [ ] `device` computed once as `0 if torch.cuda.is_available() else -1` and passed to every pipeline.
- [ ] `torch.manual_seed(42)` set in the setup cell.
- [ ] All six pipeline tasks run: sentiment-analysis, zero-shot-classification, ner,
      question-answering, summarization, fill-mask.
- [ ] NER uses `aggregation_strategy="simple"` and reads `entity_group` / `word` / `score`.
- [ ] Zero-shot uses `candidate_labels=` and reads `labels[0]` / `scores[0]`.
- [ ] QA called with `question=` and `context=`; reads `answer` / `score`.
- [ ] Summarization uses `max_length` / `min_length`; input is a real paragraph (no length warning).
- [ ] Fill-mask reads the mask token from `unmasker.tokenizer.mask_token` (no hard-coded token).
- [ ] Model swap demo uses `model="cardiffnlp/twitter-roberta-base-sentiment-latest"`.
- [ ] Lab 1 starter `None # YOUR CODE` does not leak the labels or the router call.
- [ ] Lab 2 starter does not leak the second-pipeline construction or the comparison calls.
- [ ] Each lab has a stretch and a homework extension.
- [ ] Gradio cell is guarded with try/except ImportError so a no-gradio env still runs.
- [ ] No PyTorch nn / training code (that is Part B); pipeline treated as a black box.
- [ ] No TF/Keras, no RNN/LSTM, no LLM API keys.
- [ ] Plain ASCII only: no em/en dashes, no Unicode multiplication, no emoji.
- [ ] Continuity with A1: NER framed as "the spaCy task from A1, now one line"; no `nlp`/`doc` name reuse.
- [ ] Bridge to B (build the black box) and C (fine-tune distilbert + Gradio) stated explicitly.

# RESEARCH VALIDATED (June 2026)

- transformers 5.0.0 (released Jan 26, 2026) is PyTorch-first and REMOVED the summarization,
  question-answering, and translation pipelines; it requires numpy 2. The 4.x line keeps
  those pipelines and supports numpy<2, so we pin `transformers==4.57.*` (4.57.6 is the last
  stable 4.x, Jan 2026). Sources:
  https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md ;
  https://howaiworks.ai/blog/transformers-v5-release-announcement-2025 ;
  https://pypi.org/project/transformers/
- numpy<2 vs transformers: numpy 2.0 is supported by recent transformers, but the course
  pins numpy<2; the 4.57.x line is compatible with numpy<2. Source:
  https://github.com/huggingface/transformers/issues/31740
- Pipeline task names and default models. sentiment-analysis (alias text-classification) ->
  distilbert-base-uncased-finetuned-sst-2-english; zero-shot-classification ->
  facebook/bart-large-mnli; ner (alias token-classification) with aggregation_strategy=simple
  -> dbmdz/bert-large-cased-finetuned-conll03-english; question-answering ->
  distilbert-base-cased-distilled-squad; summarization -> sshleifer/distilbart-cnn-12-6;
  fill-mask -> distilroberta-base. Sources:
  https://huggingface.co/docs/transformers/main_classes/pipelines ;
  https://huggingface.co/facebook/bart-large-mnli ;
  https://huggingface.co/tasks/fill-mask
- pipeline() device argument: device=0 for first GPU, device=-1 for CPU. Source:
  https://huggingface.co/docs/transformers/en/pipeline_tutorial
- Output formats: zero-shot returns {sequence, labels (sorted), scores}, multi_label toggles
  independent scoring; NER with aggregation_strategy=simple returns
  {entity_group, score, word, start, end}; QA returns {answer, score, start, end};
  summarization returns [{summary_text}] with max_length/min_length controls; fill-mask
  returns ranked [{sequence, score, token, token_str}]. Sources:
  https://huggingface.co/tasks/zero-shot-classification ;
  https://github.com/huggingface/transformers/issues/27876 ;
  https://huggingface.co/docs/transformers/tasks/question_answering
- Zero-shot vs fine-tuned: zero-shot reaches roughly 60-85% of fine-tuned accuracy; weaker on
  narrow/critical intents; use a confidence threshold with human fallback in production.
  Sources: https://www.parloa.com/knowledge-hub/zero-shot-classification/ ;
  https://ar5iv.labs.arxiv.org/html/2305.07157
- Model swap and size/latency: pass model= to swap; distilled models are roughly 51% smaller
  and ~43% faster with small accuracy loss; cardiffnlp twitter sentiment adds a NEUTRAL class.
  Sources: https://www.philschmid.de/optimizing-transformers-with-optimum-gpu ;
  https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
- pipeline hides AutoTokenizer preprocess -> AutoModelForSequenceClassification forward ->
  postprocess; the default sentiment model is distilbert-base-uncased fine-tuned on SST-2,
  previewing C9. Source: https://huggingface.co/learn/llm-course/en/chapter2/2
- Production take-homes: pin revision= for reproducibility; cache via HF cache; batch with
  batch_size=. Sources: https://huggingface.co/docs/transformers/main_classes/pipelines ;
  https://huggingface.co/docs/huggingface_hub/guides/manage-cache
- Gradio bridge: gr.Interface.from_pipeline(pipe) wraps any pipeline in a UI in one line;
  demo.launch() opens it. Source: https://www.gradio.app/guides/using-hugging-face-integrations


