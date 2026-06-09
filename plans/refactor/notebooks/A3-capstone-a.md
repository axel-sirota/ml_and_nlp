# A3 - Capstone A: Customer Support Routing - Cell-by-Cell Plan

## Status

RENEWED. Source notebook: `1-Pre-NLP/4-Capstone_1.ipynb` (Twitter Customer Support Routing).
Keep the scenario and the label-derivation twist; retarget the SOLUTION from classical-ML +
PyTorch-MLP training to Part A tools only: `pipeline("zero-shot-classification")` plus optional
`pipeline("ner")`. No training of any kind. New file: `A-What-NLP-Can-Do/3-Capstone_A.ipynb`.

## Context

Students arrive from A1 (spaCy NER, gensim topic modelling) and A2 (the `pipeline()` mental
model, having already called `pipeline("zero-shot-classification")` with `candidate_labels`,
`pipeline("ner", aggregation_strategy="simple")`, plus sentiment, QA, summarization). They know
a pipeline maps a task string to a default model, downloads it, and is called on text. They have
NOT trained anything yet (that is Part B). The key insight they leave with: you can ship a real,
useful classifier on a brand new problem with ZERO labelled training data, by writing good
`candidate_labels` and letting an NLI model reason about them. They also leave knowing the honest
ceiling: zero-shot tops out around 70-85 percent accuracy, which is exactly why Part B and Part C
exist (fine-tuning reaches 95 percent plus).

Carried-over API names students already have: `pipeline` (from `transformers`), the result dict
shape `{'sequence', 'labels', 'scores'}` (labels auto-sorted descending), `candidate_labels`,
`aggregation_strategy="simple"`, `device`. New names introduced here: `clf` (the zero-shot pipe),
`ner` (the NER pipe), `reply_map`, `route_tweet`, `CANDIDATE_LABELS`, `CONF_THRESHOLD`.

## Chatbot Through-Line

The course end goal is a fine-tuned transformer in a simple Gradio chatbot. A3 is the last
Part A node and it plants two seeds for that goal. First, it ends with a guarded
`gr.Interface` mini-app: students type a tweet and watch it get routed live, which is the exact
UX skeleton the C9/C10 chatbots will reuse (same Gradio shape, just a different model behind it).
Second, it makes the case for fine-tuning visceral: students measure that zero-shot routing is
good but not great, so the natural next question is "how do I push past this ceiling?" The
one-line bridge to B4: "Zero-shot got us 70-something percent with no training. To beat it we have
to understand tensors, embeddings, and training loops, then fine-tune a real transformer. That is
Part B."

## STAR Story

- **Situation**: You are a data scientist on a customer-support team. Customers tweet complaints
  at the company all day. Each tweet is routed to a queue (billing, technical, account, shipping)
  by a human, which is slow and expensive. There is no labelled training set for routing.
- **Task**: Ship a router that reads a tweet and predicts the right queue, this week, with no
  time to label thousands of examples and no time to train a model.
- **Action**: Load the TWCS corpus, recover weak company labels from reply metadata (the twist:
  the labels are not in the file), define human-readable routing categories as `candidate_labels`,
  and classify every tweet with `pipeline("zero-shot-classification")`. Add `pipeline("ner")` to
  pull product and account mentions as routing metadata, and add a confidence threshold so
  low-confidence tweets fall back to a human.
- **Result**: An end-to-end routing function that turns raw noisy tweets into routed tickets with
  metadata, built entirely from pretrained pipelines, plus an honest read on where zero-shot
  stops and fine-tuning (Part B and C) must begin.

## Deliverables

- Exercise: `A-What-NLP-Can-Do/3-Capstone_A.ipynb`
- Solution: `Solutions/A-What-NLP-Can-Do/3-Capstone_A.ipynb`

## Session Timing (~60-90 min)

| Segment | Cells | Minutes |
|---------|-------|---------|
| Setup + scenario + dataset load + sanity check | 0-5 | 12 |
| Section 2: derive weak labels from reply metadata (Lab 1) | 6-9 | 15 |
| Section 3: routing categories + clean tweets (Lab 2) | 10-13 | 12 |
| Section 4: zero-shot routing demo + Lab 3 | 14-17 | 18 |
| Section 5: NER metadata (Lab 4) | 18-19 | 8 |
| Section 6: evaluate + confidence fallback (Lab 5) | 20-21 | 10 |
| Packaging + Gradio demo + wrap-up + homework | 22-23 | 10 |

## Cell Migration Map

| Old cell id (index) | New cell | Action |
|---------------------|----------|--------|
| 0 (title md) | 0 | edit (retheme: tools-first, no training) |
| 1 (setup md) | 1 | edit (HF stack, not bare torch) |
| 2 (pip) | 2 | edit (transformers/datasets stack + numpy<2) |
| 3 (imports/seed) | 3 | edit (drop sklearn-model/torch-nn imports, add pipeline) |
| 4 (TWCS schema md) | 4 | keep (light edit) |
| 5,6 (wget shell) | 5 | edit (single pandas read_csv from Dropbox dl=1) |
| 7 (read_csv nrows) | 5 | merge into 5 |
| 8 (inbound counts) | 5 | merge into 5 (sanity check folded into load) |
| 9 (extract-labels md) | 6 | keep (retheme to weak labels for eval; absorb old "plan" recap) |
| 10,11,12 (reply_map build) | 7 (demo) | edit (collapse into one demo) |
| 13 (attach labels) | 8-9 (Lab 1) | edit -> becomes Lab 1 |
| 14 (class dist plot) | 9 | keep (verification plot) |
| 15,16 (Lab cap per class) | 12-13 (Lab 2) | edit (sample for spot-check, no training cap) |
| 17 (cleaning md) | 10 | edit (lighter cleaning, NLI-aware) |
| 18 (clean_tweet demo) | 11 | keep (light edit) |
| 19,20 (Lab apply+split) | 12-13 (Lab 2) | edit (apply clean; NO train/val/test split) |
| 21-24 (TF-IDF + LogReg + CM) | drop | drop (classical training moved to Basics/) |
| 25-39 (PyTorch model, vocab, Dataset, training loop) | drop | drop (training moved to Part B) |
| 40 (wrap-up md) | 23 | edit (retheme to ship-with-zero-training + bridge; absorb homework) |
| (none) | 14,15,16,17,18,19,20,21,22 | FROM-NEW (zero-shot, NER, eval, fallback, packaging, Gradio) |

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~20-25 cells)

## Cell 0 - Markdown: Title and learning objectives

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 0
Change: retheme from "classical toolkit + first PyTorch classifier" to "Part A tools only, zero
training"; drop the 4-class company-classifier task framing.

```
# Capstone A - Customer Support Routing with Zero Training

*Part A capstone - ML and NLP by Data Trainers LLC.*

This capstone is the payoff for Part A. In A1 you did NLP by hand with spaCy and gensim. In A2
you watched pretrained transformers solve task after task in three lines with `pipeline()`. Now
you put it all together on a real, noisy, production-shaped problem and ship a working solution
with **zero training and zero labelled data**.

## Learning objectives

By the end you will be able to:

1. Load and sanity-check a real customer-support corpus (TWCS).
2. Recover weak labels from implicit reply metadata (the labels are not a column in the file).
3. Define human-readable routing categories and classify tweets with
   `pipeline("zero-shot-classification")` - no training.
4. Extract product and account mentions with `pipeline("ner")` as routing metadata.
5. Evaluate routing quality qualitatively and against the weak labels.
6. Add a confidence threshold so low-confidence tweets fall back to a human.
7. Wrap the router in a tiny Gradio demo and reason about when zero-shot is not enough.

## Prerequisites

- A1 (spaCy NER, gensim topics) and A2 (the `pipeline()` tour). You should already have called
  `pipeline("zero-shot-classification")` and `pipeline("ner", aggregation_strategy="simple")`.

## Session format

Roughly 60-90 minutes: short theory, a runnable demo, then a hands-on lab for each section.
Runs on Google Colab (GPU optional; everything here also works on CPU, just slower).
```

## Cell 1 - Markdown: Section 0 - Environment Setup

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 1
Change: replace "PyTorch not Keras/TF" framing with HuggingFace pipelines framing.

```
## Section 0: Environment Setup

This notebook uses **HuggingFace `transformers` pipelines** on top of PyTorch. We do not train
anything; every model is pretrained and downloaded from the HuggingFace Hub. Run the next two
cells once per Colab session.
```

## Cell 2 - Code: Install packages

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 2
Change: swap the classical/torch-only stack for the HF stack; pin `numpy<2`.

```python
# Install required packages (run this first in Google Colab).
# numpy<2 is pinned because several pretrained-model wheels are not yet built against numpy 2.x.
!pip install -q "numpy<2" "transformers>=4.40" "datasets>=2.19" "accelerate>=0.30" \
    pandas matplotlib seaborn

# spaCy is already used in A1; we only need transformers' own NER here, so spaCy is optional.
print("Install step done. If Colab asks you to restart the runtime, do it, then re-run from here.")
```

## Cell 3 - Code: Imports, seed, device

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 3
Change: drop sklearn LogisticRegression / TfidfVectorizer / torch.nn imports; add `pipeline`.
Keep numpy/pandas/plotting, seed, and device select.

```python
# Core imports
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import pipeline

# We keep sklearn ONLY for an optional metrics report (no training, just metrics helpers).
from sklearn.metrics import classification_report

# Reproducibility (sampling is the only randomness here; the models are deterministic at inference)
SEED = 42
np.random.seed(SEED)
torch.manual_seed(42)

# Device: pipelines take an int device id. 0 = first GPU, -1 = CPU.
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"transformers will run on: {'GPU (cuda:0)' if DEVICE == 0 else 'CPU'}")
print(f"PyTorch version: {torch.__version__}")
print("Environment ready.")
```

## Cell 4 - Markdown: Section 1 - Load the TWCS dataset

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 4
Change: light edit; keep the schema table (verified against the Kaggle TWCS schema).

```
## Section 1: Load the TWCS Dataset

The **Twitter Customer Support (TWCS)** corpus is millions of tweets between customers and company
support accounts. We will use a subsample for speed; the methodology scales up unchanged.

Each row is a single tweet with these columns:

| Column | Meaning |
|--------|---------|
| `tweet_id` | Unique id for the tweet |
| `author_id` | Username (company, e.g. `AppleSupport`) or a numeric string (a customer) |
| `inbound` | `True` if the tweet is from a customer to a company, `False` if from the company |
| `created_at` | Timestamp |
| `text` | The tweet text |
| `response_tweet_id` | Comma-separated reply `tweet_id`s, or empty |
| `in_response_to_tweet_id` | If this tweet is a reply, the `tweet_id` it replies to |

Notice there is **no routing column**. We will recover weak labels from the reply structure in a
moment. But first: the whole point of Part A is that we can route tweets even without those labels.
```

## Cell 5 - Code: Download, load a subsample, and sanity-check inbound vs outbound

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cells 5,6,7,8
Change: replace the `%%writefile` + bash + separate read with one pandas `read_csv` straight from
the Dropbox URL with `dl=1`; cap rows for speed; fold the old inbound/outbound sanity-check
(old cell 8) into this same load cell so loading and verifying the data is one step.

```python
# Course-hosted copy of TWCS (real company handles preserved). dl=1 streams the raw CSV.
TWCS_URL = "https://www.dropbox.com/scl/fi/6w2dchvsthwdol934nprt/twcs.csv?rlkey=u5205iehhrog88qt8iwm4udxs&dl=1"

# Read only the first chunk of rows: enough to see every major company, fast to download.
SUBSET_SIZE = 80_000
twcs = pd.read_csv(TWCS_URL, nrows=SUBSET_SIZE)

print(f"Loaded {len(twcs):,} tweets")
print(f"Columns: {list(twcs.columns)}")
display(twcs.head(3))

# Sanity check: how many tweets are customer-to-company (inbound) vs company replies (outbound)?
print("\ninbound value counts (True = customer tweet, the thing we will route):")
print(twcs['inbound'].value_counts())

print("\nTop 10 author_ids (company support accounts dominate the outbound side):")
print(twcs['author_id'].value_counts().head(10))
```

## Cell 6 - Markdown: Section 2 - Recover weak labels from reply metadata (and the plan)

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 9
Change: retheme. We no longer build a training label; we build a weak label to spot-check
zero-shot against later. Keep the ASCII reply-arrow diagram. Also fold in the former standalone
"Our plan" recap (old FROM-NEW Cell 7) as a short numbered roadmap at the top of this section, so
students get the five-move overview without a separate markdown cell.

```
## Section 2: Recover Weak Labels From Reply Metadata

Here is the whole capstone in five moves, each a short lab:

1. **Recover weak labels** from reply metadata, so we have something to spot-check against later.
   These are *weak* labels (the company that happened to reply), not a routing ground truth.
2. **Define routing categories** as plain-English `candidate_labels` (billing, technical, etc.).
3. **Route every tweet** with `pipeline("zero-shot-classification")` - the no-training superpower.
4. **Extract metadata** (products, account mentions) with `pipeline("ner")`.
5. **Evaluate and add a safety net**: a confidence threshold that sends unsure tweets to a human.

Not one line of model training in the whole notebook. That is the Part A promise. Let us start
with move 1.

The dataset has no routing column, but it does encode *which company handled each tweet*: the
reply to a customer tweet was authored by that company's support account. We can walk the reply
links backwards to recover the handling company:

```
Customer tweet (inbound=True)   <-- we want to route this
        ^
        | in_response_to_tweet_id
        |
Company reply (inbound=False)   <-- its author_id tells us who handled it
```

We will use this as a **weak label**: useful to spot-check our zero-shot router, but not a true
routing label (the company that replied is not the same thing as the right queue). This is also a
real, reusable skill: recovering labels from implicit structure when nobody handed you a clean
`label` column.
```

## Cell 7 - Code: Demo - build the reply_map and attach weak company labels

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cells 10,11,12
Change: collapse the three steps (top authors, inbound/outbound split, reply_map build) into one
heavily commented demo cell.

```python
# Demo: recover the handling company for each customer tweet, end to end.

# The companies we will spot-check on (the busiest support accounts in this subsample).
TOP_AUTHORS = ['AppleSupport', 'AmazonHelp', 'SpotifyCares', 'Uber_Support']

# 1. Split customer tweets (inbound) from company replies (outbound).
inbound = twcs[twcs['inbound'] == True].copy()
outbound = twcs[twcs['inbound'] == False].copy()

# 2. Keep only replies authored by one of our target companies, with a valid parent tweet id.
outbound_targeted = outbound[outbound['author_id'].isin(TOP_AUTHORS)].copy()
outbound_targeted = outbound_targeted.dropna(subset=['in_response_to_tweet_id'])
outbound_targeted['in_response_to_tweet_id'] = outbound_targeted['in_response_to_tweet_id'].astype('int64')

# 3. Build {parent_tweet_id -> company} and attach it to the inbound tweets it points back to.
reply_map = dict(zip(outbound_targeted['in_response_to_tweet_id'], outbound_targeted['author_id']))
inbound['company'] = inbound['tweet_id'].map(reply_map)

print(f"Built reply_map with {len(reply_map):,} entries.")
print(f"Inbound tweets total: {len(inbound):,}; with a recovered company: {inbound['company'].notna().sum():,}")
```

## Cell 8 - Markdown: Lab 1 - keep only the labelled tweets

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 13
Change: convert the old "attach labels" step into a short guided lab; no leak.

```
### Lab 1: Keep the tweets we recovered a company for

The demo above attached a `company` to every inbound tweet it could. Most inbound tweets have no
recovered company (their reply was not in our subsample, or came from a different account). For the
spot-check later we only want the rows that DID get a company.

**Your task** (about 5 minutes):

1. Filter `inbound` down to rows where `company` is not null. Call it `labeled`.
2. Keep only the columns `tweet_id`, `text`, `company` and reset the index.

Hints:

- `notna()` gives a boolean mask you can index a DataFrame with.
- `df[['col_a', 'col_b']]` selects columns; `.reset_index(drop=True)` renumbers the rows.
```

## Cell 9 - Code: Lab 1 starter + verification (plot class distribution)

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cells 13,14
Change: starter placeholders for the filter; keep the class-distribution plot as verification.

```python
# 1. Keep only inbound tweets that received a recovered company label.
labeled = None  # YOUR CODE (filter inbound to rows where 'company' is not null)

# 2. Keep just the columns we need and renumber the rows.
labeled = None  # YOUR CODE (select ['tweet_id', 'text', 'company'] and reset_index(drop=True))

# Verification: count and plot the weak-label distribution.
if labeled is not None:
    print(f"Labeled tweets: {len(labeled):,}")
    counts = labeled['company'].value_counts()
    print(counts)
    plt.figure(figsize=(7, 3.5))
    counts.plot(kind='bar', color=['#5B8DEF', '#FF9F40', '#4ECDC4', '#FFD166'])
    plt.title('Recovered weak labels per company')
    plt.ylabel('Count'); plt.xticks(rotation=0); plt.tight_layout(); plt.show()
```

## Cell 10 - Markdown: Section 3 - Define routing categories and clean tweets

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 17
Change: reframe cleaning as NLI-aware (light cleaning only) and introduce the routing categories
as the real classification target. State the @handle leakage reason explicitly.

```
## Section 3: Routing Categories and Light Cleaning

Here is the reframe that makes this a Part A capstone. The company that replied is *not* the
routing decision. A real support inbox routes by **what the customer needs**: billing, technical
help, account access, shipping, or general feedback. Those categories are our classification
target, and we have zero labelled examples for them. That is exactly what zero-shot is for.

We define the categories as plain-English `candidate_labels`. Research on zero-shot is blunt about
this: **label wording is the single biggest lever** on accuracy. Vague labels like "tech" or two
labels that mean almost the same thing will hurt you; clear, distinct descriptions help.

Before routing we do **light** cleaning only. Transformer tokenizers tolerate moderate noise, so we
do not strip everything. We do two things:

1. Remove URLs (they carry no routing signal and waste tokens).
2. Remove `@handles`. This matters: if `@AppleSupport` stays in the text, the model can cheat off
   the handle instead of reading the complaint, and in production customers often do not tag anyone.
```

## Cell 11 - Code: Demo - clean_tweet and the candidate labels

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 18
Change: lighter cleaning (only URLs + handles + whitespace, keep punctuation/case for NLI);
add the CANDIDATE_LABELS definition and a hypothesis template.

```python
# Light cleaning: drop URLs and @handles, collapse whitespace. Keep case and punctuation;
# the NLI model reads natural English better than aggressively stripped text.
URL_RE = re.compile(r'http\S+|www\.\S+|t\.co/\S+')
HANDLE_RE = re.compile(r'@\w+')
WS_RE = re.compile(r'\s+')

def clean_tweet(text):
    if not isinstance(text, str):
        return ''
    text = URL_RE.sub(' ', text)
    text = HANDLE_RE.sub(' ', text)
    return WS_RE.sub(' ', text).strip()

# The routing categories: clear, distinct, human-readable phrases (wording matters a lot).
CANDIDATE_LABELS = [
    "billing or payment problem",
    "technical issue or bug",
    "account access or login",
    "shipping or delivery",
    "general feedback or praise",
]
# A hypothesis template tailored to the domain reads better to the NLI model than the default.
HYPOTHESIS_TEMPLATE = "This customer support message is about {}."

# Quick look at cleaning on 3 real tweets.
for ex in labeled['text'].iloc[:3] if labeled is not None else []:
    print('RAW  :', ex)
    print('CLEAN:', clean_tweet(ex))
    print()
```

## Cell 12 - Markdown: Lab 2 - clean all tweets and sample a routing batch

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cells 19,20
Change: apply cleaning, drop empties, then SAMPLE a small set to route (no train/val/test split,
since we never train). Keep it cheap so zero-shot finishes fast in class.

```
### Lab 2: Clean every tweet and sample a routing batch

Zero-shot is slower per tweet than a trained model (each tweet runs once per candidate label), so
in class we route a sample. The method is identical at full scale.

**Your task** (about 8 minutes):

1. Apply `clean_tweet` to `labeled['text']` and store the result in `labeled['clean']`.
2. Drop rows whose `clean` text is an empty string (some tweets were nothing but a link/handle).
3. Sample `SAMPLE_SIZE` rows (use `random_state=SEED`) into `routing_set` and reset the index.

Hints:

- `Series.apply(clean_tweet)` runs the function on every row.
- An empty-string mask is `labeled['clean'].str.len() > 0`.
- `DataFrame.sample(n=..., random_state=SEED)` then `.reset_index(drop=True)`.
```

## Cell 13 - Code: Lab 2 starter + verification

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 20
Change: starter placeholders for clean/drop/sample; verification prints sizes and a sample row.

```python
SAMPLE_SIZE = 200  # small enough to route in a minute or two in class

# 1. Clean every tweet.
labeled['clean'] = None  # YOUR CODE (apply clean_tweet to labeled['text'])

# 2. Drop rows whose cleaned text is empty.
labeled = None  # YOUR CODE (keep rows where clean length > 0, then reset_index(drop=True))

# 3. Sample a routing batch.
routing_set = None  # YOUR CODE (sample SAMPLE_SIZE rows with random_state=SEED, reset_index)

# Verification
if routing_set is not None:
    print(f"Routing set size: {len(routing_set):,}")
    print("Example cleaned tweet:", routing_set['clean'].iloc[0])
```

## Cell 14 - Markdown: Section 4 - Zero-shot routing

Provenance: FROM-NEW

```
## Section 4: Route Tweets With Zero-Shot Classification

This is the heart of the capstone. We load one pipeline and route any tweet to any of our
categories, with no training and no labelled data:

```python
clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=DEVICE)
result = clf("my payment was charged twice this month",
             candidate_labels=CANDIDATE_LABELS,
             hypothesis_template=HYPOTHESIS_TEMPLATE)
```

The pipeline returns a dict whose `labels` and `scores` are **already sorted best-first**, so
`result['labels'][0]` is the predicted route and `result['scores'][0]` is its confidence. No
manual `argmax` needed. The model behind the scenes is an NLI model: it treats your tweet as a
premise and each candidate label, plugged into the hypothesis template, as a hypothesis, then
scores entailment.
```

## Cell 15 - Code: Demo - load the zero-shot pipeline and route one tweet

Provenance: FROM-NEW

```python
# Load the zero-shot pipeline once (downloads facebook/bart-large-mnli on first run).
clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=DEVICE)

# Route a single, obviously-billing tweet to see the output shape.
demo_tweet = "I was charged twice for my subscription this month and want a refund"
result = clf(demo_tweet, candidate_labels=CANDIDATE_LABELS, hypothesis_template=HYPOTHESIS_TEMPLATE)

print("Tweet:", demo_tweet)
print("Predicted route:", result['labels'][0], f"(confidence {result['scores'][0]:.2f})")
print("\nFull ranking:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  {score:.3f}  {label}")
```

## Cell 16 - Markdown: Lab 3 - route the whole sample

Provenance: FROM-NEW

```
### Lab 3: Route the entire routing batch

Now route every tweet in `routing_set`. Passing a list of texts lets the pipeline batch them.

**Your task** (about 12 minutes):

1. Call `clf` on the list `routing_set['clean'].tolist()`, passing `candidate_labels`,
   `hypothesis_template`, and `batch_size=16`. Store the list of result dicts in `results`.
2. For each result dict, take `r['labels'][0]` as the predicted route and `r['scores'][0]` as the
   confidence. Store them in two new columns `routing_set['route']` and `routing_set['confidence']`.

Hints:

- `clf(list_of_texts, candidate_labels=..., hypothesis_template=..., batch_size=16)` returns a
  list of dicts, one per text, in the same order.
- A list comprehension over `results` is the clean way to pull out `labels[0]` / `scores[0]`.
```

## Cell 17 - Code: Lab 3 starter + verification

Provenance: FROM-NEW

```python
# 1. Route every tweet in the sample (this is the slow cell; a minute or two on GPU).
results = None  # YOUR CODE (call clf on routing_set['clean'].tolist() with candidate_labels,
                #            hypothesis_template, batch_size=16)

# 2. Pull the top label and its confidence into new columns.
routing_set['route'] = None        # YOUR CODE (r['labels'][0] for each r in results)
routing_set['confidence'] = None   # YOUR CODE (r['scores'][0] for each r in results)

# Verification: distribution of routes + a few routed examples.
if routing_set['route'].notna().any():
    print("Routes assigned:")
    print(routing_set['route'].value_counts())
    print("\nSpot-check:")
    for _, row in routing_set.head(5).iterrows():
        print(f"[{row['route']}  {row['confidence']:.2f}]  {row['clean'][:90]}")
```

## Cell 18 - Markdown: Section 5 - NER metadata + Lab 4

Provenance: FROM-NEW (concept reused from A2's `pipeline("ner")`)

```
## Section 5: Add Routing Metadata With NER

Routing decides the *queue*; metadata makes the ticket actionable. The agent who picks up a tweet
wants to see at a glance which product or organization it mentions. We get that for free with the
NER pipeline you already met in A2:

```python
ner = pipeline("ner", aggregation_strategy="simple", device=DEVICE)
ner("my iPhone from Amazon stopped charging")
# -> [{'entity_group': 'MISC', 'word': 'iPhone', ...},
#     {'entity_group': 'ORG',  'word': 'Amazon', ...}]
```

`aggregation_strategy="simple"` merges sub-word pieces into clean spans, so you get `iPhone`, not
`i ##Phone`. We will pull the ORG and product mentions out as a metadata tag.

### Lab 4: Attach entity metadata to each routed tweet

**Your task** (about 8 minutes):

1. Build the NER pipeline `ner` with `aggregation_strategy="simple"` and `device=DEVICE`.
2. Write `extract_entities(text)` that runs `ner(text)` and returns the list of `word` values whose
   `entity_group` is in `{"ORG", "MISC", "PRODUCT"}` (different models use different tags).
3. Apply it to the first 20 rows of `routing_set` and print tweet -> route -> entities.
```

## Cell 19 - Code: Lab 4 starter + verification

Provenance: FROM-NEW

```python
# 1. Build the NER pipeline.
ner = None  # YOUR CODE (pipeline("ner", aggregation_strategy="simple", device=DEVICE))

# 2. Extract the org/product-style entity words from a tweet.
KEEP_GROUPS = {"ORG", "MISC", "PRODUCT"}
def extract_entities(text):
    ents = None  # YOUR CODE (call ner(text))
    # YOUR CODE: return the list of e['word'] for entities whose e['entity_group'] is in KEEP_GROUPS
    return None

# Verification: route + entities for the first 20 tweets.
if ner is not None:
    for _, row in routing_set.head(20).iterrows():
        ents = extract_entities(row['clean'])
        print(f"[{row['route']}] entities={ents}  ::  {row['clean'][:80]}")
```

## Cell 20 - Markdown: Section 6 - evaluate and add a confidence fallback

Provenance: FROM-NEW

```
## Section 6: Evaluate and Add a Safety Net

Two questions remain: how good is the routing, and what do we do when the model is unsure?

**Evaluation.** We have no true routing labels, but we have the weak `company` labels. They do not
map one-to-one to our categories, so a raw accuracy number would be misleading. Instead we do two
honest things: a qualitative spot-check (read routed examples), and a confidence histogram to see
how decisive the model is.

**Confidence fallback.** Production routers do not auto-route everything. The standard pattern is:
accept high-confidence predictions, and **abstain** below a threshold, sending those tweets to a
human queue. The confidence score is the knob that trades human effort for accuracy.

### Lab 5: Confidence threshold and route-or-escalate

**Your task** (about 10 minutes):

1. Plot a histogram of `routing_set['confidence']`.
2. Set `CONF_THRESHOLD = 0.5`. Create a `final_route` column equal to the predicted `route` when
   `confidence >= CONF_THRESHOLD`, otherwise the string `"ESCALATE_TO_HUMAN"`.
3. Print how many tweets were auto-routed vs escalated.
```

## Cell 21 - Code: Lab 5 starter + verification

Provenance: FROM-NEW

```python
# 1. Confidence histogram.
plt.figure(figsize=(6, 3))
plt.hist(routing_set['confidence'], bins=20, color='#5B8DEF')
plt.title('Zero-shot routing confidence'); plt.xlabel('top-label score'); plt.ylabel('count')
plt.tight_layout(); plt.show()

# 2. Apply a confidence threshold: auto-route or escalate.
CONF_THRESHOLD = 0.5
routing_set['final_route'] = None  # YOUR CODE (route if confidence >= CONF_THRESHOLD else
                                   #            "ESCALATE_TO_HUMAN"; np.where is one clean way)

# 3. Report the auto vs escalate split.
if routing_set['final_route'].notna().any():
    escalated = (routing_set['final_route'] == "ESCALATE_TO_HUMAN").sum()
    print(f"Auto-routed: {len(routing_set) - escalated:,}")
    print(f"Escalated to human: {escalated:,}")
    print("\nFinal route distribution:")
    print(routing_set['final_route'].value_counts())
```

## Cell 22 - Code: Demo - the route_tweet function and a Gradio mini-app

Provenance: FROM-NEW

Note (defect-fix): this packaging demo reuses the `extract_entities` helper the student already
finished in Lab 4 (Cell 19), instead of re-inlining the org/product comprehension. That keeps the
Lab 4 answer from being re-revealed here as a literal one-liner; the helper is simply called. By
this point Lab 4 is closed out, so calling the verified helper is reuse, not a solution leak.

```python
# Package everything into one function: clean -> route -> entities -> confidence gate.
# extract_entities was written and verified in Lab 4 above; we reuse it here unchanged.
def route_tweet(text, threshold=CONF_THRESHOLD):
    clean = clean_tweet(text)
    r = clf(clean, candidate_labels=CANDIDATE_LABELS, hypothesis_template=HYPOTHESIS_TEMPLATE)
    route = r['labels'][0]
    conf = r['scores'][0]
    final = route if conf >= threshold else "ESCALATE_TO_HUMAN"
    entities = extract_entities(clean)  # reuse the Lab 4 helper, do not re-implement it
    return {"final_route": final, "confidence": round(conf, 3), "entities": entities}

# Try it on a fresh, untagged complaint (no @handle, like real production traffic).
print(route_tweet("my package never arrived and the tracking link is dead"))

# Optional live demo: a tiny Gradio app. Guarded so the notebook still runs without gradio.
# This is the SAME UX skeleton the Part C chatbot will use - just a different model behind it.
try:
    import gradio as gr
    def _ui(text):
        out = route_tweet(text)
        return f"Route: {out['final_route']} (conf {out['confidence']})\nEntities: {out['entities']}"
    demo = gr.Interface(fn=_ui, inputs=gr.Textbox(label="Customer tweet"),
                        outputs=gr.Textbox(label="Routing decision"), title="Support Router (zero-shot)")
    # demo.launch(share=True)  # uncomment in Colab to get a public link
    print("Gradio app defined. Uncomment demo.launch(share=True) to open it.")
except ImportError:
    print("gradio not installed - skipping the live demo (the route_tweet function still works).")
```

## Cell 23 - Markdown: Wrap-up, the bridge to Part B, and homework

Provenance: FROM-OLD 1-Pre-NLP/4-Capstone_1.ipynb cell 40
Change: retheme from "classical vs neural you trained" to "zero-shot you shipped vs fine-tuned you
will build"; add the explicit Part B bridge and the accuracy-gap numbers; fold the former
standalone homework cell (old FROM-NEW Cell 26) in as a "Homework Extension" subsection so the
notebook closes on one wrap-up-plus-homework cell.

```
## Wrap-up: You Shipped a Router With Zero Training

Look back at what you just did. With no labelled training set and no training loop, you:

- recovered weak labels from implicit reply metadata,
- defined routing categories as plain-English `candidate_labels`,
- routed real, noisy tweets with `pipeline("zero-shot-classification")`,
- enriched each ticket with NER metadata,
- added a confidence threshold that escalates unsure tweets to a human,
- and wrapped it all in a tiny Gradio app.

**Reflect:**

1. Would you ship this today? For a brand-new problem with no data, very possibly yes - zero-shot
   is a real, deployable baseline.
2. Where does it stop? Zero-shot typically tops out around **70-85 percent** accuracy. A fine-tuned
   transformer on the same task routinely reaches **95 percent or more**. For high-volume routing,
   that gap is millions of correctly handled tickets.
3. What would push past the ceiling? A stronger zero-shot backbone helps a little
   (`MoritzLaurer/deberta-v3-base-mnli-fever-anli` beats `bart-large-mnli`). But the real jump is
   **fine-tuning your own classifier** on labelled data.

**The bridge to Part B.** To fine-tune a transformer you first have to understand what is inside
one: tensors, embeddings, neural-network layers, and the training loop. That is exactly Part B,
starting with PyTorch tensors in B4. By Part C you will fine-tune DistilBERT and serve it in this
same Gradio shell - your zero-shot router, upgraded into a trained one.

## Homework Extension (async, deeper)

Pick one or more. These are genuinely harder than the in-class labs, not just more volume.

1. **Multi-label routing.** Some tweets need two queues ("I was double-charged AND the app
   crashed"). Re-run the router with `multi_label=True`, then surface every label whose score is
   above a secondary threshold (e.g. 0.4) as additional routes. Decide how an agent should see a
   primary plus secondary route.
2. **Threshold tuning against weak labels.** Build a crude mapping from your `company` weak labels
   to plausible categories, then sweep `CONF_THRESHOLD` from 0.3 to 0.8 and plot escalation rate vs
   apparent agreement. Where is the sensible operating point?
3. **Model upgrade.** Swap `facebook/bart-large-mnli` for
   `MoritzLaurer/deberta-v3-base-mnli-fever-anli` and compare routes and confidences on the same
   sample. Did the harder cases improve? At what speed cost?
4. **Ship it.** Push the `route_tweet` Gradio app to a free HuggingFace Space (app.py +
   requirements.txt with `transformers` and `torch`). This is the deployment muscle you will reuse
   for the Part C chatbot.
```

# VERIFICATION CHECKLIST

- [ ] All install cells pin `numpy<2`.
- [ ] Imports use `from transformers import pipeline`; no TensorFlow/Keras, no RNN/LSTM, no torch.nn.
- [ ] No model training anywhere (no `.fit()`, no optimizer, no training loop).
- [ ] TWCS loaded directly from the course Dropbox URL with `dl=1`; schema table matches the file.
- [ ] Load cell also prints the inbound/outbound sanity check (merged, no separate cell).
- [ ] Zero-shot uses `candidate_labels` + `hypothesis_template`; `result['labels'][0]` is the route.
- [ ] NER uses `aggregation_strategy="simple"`.
- [ ] `extract_entities` is defined once in Lab 4 and reused by `route_tweet`; not re-inlined.
- [ ] Every `None # YOUR CODE` placeholder hides the answer; verification cells reveal correctness only.
- [ ] Each section has theory -> demo -> lab; no more than 3 theory cells without a code cell.
- [ ] Confidence-threshold fallback (`ESCALATE_TO_HUMAN`) present.
- [ ] Gradio cell is guarded with try/except ImportError so the notebook runs without gradio.
- [ ] Wrap-up states the 70-85 vs 95 percent accuracy gap and bridges to B4; homework folded in.
- [ ] STAR story and chatbot through-line are reflected in the title, scenario, and wrap-up.
- [ ] Solution notebook fills every placeholder with commented code and an explanation note.
- [ ] Total cell count is 24 (Cell 0 through Cell 23), within the 20-25 ceiling.
- [ ] Plain ASCII only; no em/en dashes, no emoji.

# RESEARCH VALIDATED (June 2026)

- https://huggingface.co/tasks/zero-shot-classification - zero-shot result dict has
  `sequence`/`labels`/`scores`, labels auto-sorted descending, so `labels[0]` is the prediction.
- https://huggingface.co/facebook/bart-large-mnli - default zero-shot model; `multi_label=True`
  scores each label independently; NLI premise/hypothesis mechanism.
- https://www.statworx.com/en/content-hub/blog/zero-shot-text-classification - `hypothesis_template`
  customization (e.g. "This text is about {}.") measurably affects accuracy.
- https://github.com/huggingface/transformers/issues/19063 - zero-shot pipeline accepts a list of
  texts and a `batch_size` argument; batching helps modestly on short text.
- https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py
  - `aggregation_strategy="simple"` merges sub-word tokens into clean `entity_group` spans.
- https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter - TWCS schema:
  `tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id`.
- https://qpriori.substack.com/p/understanding-zero-shot-classification and
  https://machinelearningmastery.com/getting-started-with-zero-shot-text-classification - label
  wording is the biggest accuracy lever; zero-shot ~70-85 percent vs fine-tuned ~95-99 percent.
- https://www.parloa.com/knowledge-hub/zero-shot-classification/ - production pattern: accept
  high-confidence predictions, abstain below a confidence threshold and route to a human.
- https://huggingface.co/MoritzLaurer/deberta-v3-base-mnli-fever-anli - stronger zero-shot backbone
  than bart-large-mnli (stretch/homework upgrade).
- https://huggingface.co/docs/transformers/pipeline_gradio - `gr.Interface` wraps a pipeline into a
  demo; the chatbot-shaped deployment skeleton.
- https://github.com/pandas-dev/pandas/issues/60087 - `groupby.apply` grouping-column deprecation;
  we use a simple `sample`/mask approach instead to avoid it.
```
