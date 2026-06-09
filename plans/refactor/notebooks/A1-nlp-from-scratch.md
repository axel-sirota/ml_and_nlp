# A1 - NLP From Scratch - Cell-by-Cell Plan

## Status

RENEWED. Heavy rebuild on old bones.
New file: `A-What-NLP-Can-Do/1-NLP_From_Scratch.ipynb`
Slug: `nlp-from-scratch`
Old source notebook: `1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb` (52 cells).

This is the on-ramp of the whole course. Tools-first: show what NLP IS with classical
and embedding tools before any model training. NO PyTorch training here, NO LDA fit,
NO BERTopic (both were time-sinks in the old notebook). Topic discovery is done with the
lighter, more modern sentence-transformers + KMeans cluster-and-name pattern, which also
plants the "sentence embedding" seed that B5, B7, and C9 grow.

DATASET NOTE (important). The tour corpus is the HuggingFace dataset `SetFit/bbc-news`
(`load_dataset("SetFit/bbc-news", split="train")`), which ships FULL raw article bodies with
natural capitalization in a `text` column and the category name in a `label_text` column
(values: business, entertainment, politics, sport, tech). We deliberately do NOT use the
Dropbox `bbc.csv` whose `Description` column is already lowercased, stemmed, and
stopword-stripped (row 0 reads "india attend g7 meet seven lead industrialis nation friday
unlik cow newcom status london ..."). That pre-processed text would break every
casing-dependent step in this notebook: NER returns almost nothing on lowercased+stemmed
input, displacy highlights nothing, the PERSON lab comes back empty, and lemmatizing
already-stemmed tokens ("industrialis", "competit") yields garbage. Raw article text is
required for tokenization/lemmatization, NER, the casing experiment, and embedding to all
teach what they claim. The HF dataset gives us that raw text and a clean label column.

## Context

- What students arrive with: nothing course-specific. This is notebook 1. They are ML
  practitioners (know train/test, sklearn, basic Python) but are weak on practical NLP.
- Exact tools/objects they leave with by name:
  - `nlp = spacy.load("en_core_web_sm")` - the reusable spaCy pipeline.
  - `doc.ents`, `ent.text`, `ent.label_` - NER access pattern.
  - `displacy.render(doc, style="ent", jupyter=True)` - inline entity visual.
  - `nlp.pipe(texts, batch_size=..., disable=[...])` - fast batched processing.
  - `clean_text(text)` and `preprocess(text)` - reusable text-cleaning helpers.
  - `embedder = SentenceTransformer("all-MiniLM-L6-v2")` and `embedder.encode(texts)` -
    sentence embeddings (the course through-line artifact, reused in B5).
  - `KMeans(n_clusters=k, n_init=10, random_state=42)` + per-cluster TF-IDF top words.
  - `load_dataset("SetFit/bbc-news", split="train")` - the raw BBC news corpus loaded from
    the HuggingFace Hub (full article bodies with natural capitalization).
- Key insight they leave with: text is messy and ambiguous; classical and embedding tools
  already extract real structure (entities, topics) with zero training, BUT it takes setup,
  casing care, and plumbing. That effort is exactly what `pipeline()` collapses in A2.

## Chatbot Through-Line

The course ends with a fine-tuned transformer served in a simple Gradio chatbot. A1 lays
the foundation in two ways. (1) It establishes the vocabulary of NLP - tokens, lemmas,
entities, topics - so later notebooks can say "the transformer does this, but learned".
(2) It introduces sentence embeddings via `all-MiniLM-L6-v2` (a distilled 6-layer BERT),
the exact "text -> vector" idea that becomes features for the MLP in B7 and the backbone of
DistilBERT in C9. The one-line bridge to A2: "All of this took installs, custom stopwords,
casing care, and clustering plumbing. Watch one line of `pipeline()` do the same tasks next
- and now you will understand WHY it works."

## STAR Story

- Situation: You just joined a media-analytics team as the ML person. On day one a manager
  drops a folder of unlabeled text on your desk: thousands of customer reviews and news
  articles. No schema, no labels, no model.
- Task: Answer two business questions without training anything: "Who and what do these
  documents mention?" (entities) and "What are they about?" (topics).
- Action: Reach for the classical NLP toolbox - spaCy for tokenization, lemmatization, and
  named entity recognition, and modern sentence embeddings plus KMeans for topic discovery.
  Inspect every output instead of trusting it blindly.
- Result: You surface the most-mentioned people and organizations and a handful of coherent
  topics in a few minutes on CPU, with zero modelling. You also feel the friction - and that
  friction is the motivation for transformers and `pipeline()` in the next notebook.

## Deliverables

- Exercise: `A-What-NLP-Can-Do/1-NLP_From_Scratch.ipynb`
- Solution: `Solutions/A-What-NLP-Can-Do/1-NLP_From_Scratch.ipynb` (gitignored)

## Session Timing (~60-90 min)

| Block | Cells | Minutes |
|-------|-------|---------|
| Intro + environment + dataset preview | 0-5 | 10 |
| Concept 1: Why NLP is hard | 6-7 | 8 |
| Concept 2: Tokenization, lemmatization, cleanup + Lab | 8-12 | 18 |
| Concept 3: Named Entity Recognition + Lab | 13-17 | 22 |
| Concept 4: Topic discovery with embeddings + Lab | 18-21 | 22 |
| Wrap-up + bridge to A2 | 22 | 5 |

## Cell Migration Map

RENEWED notebook. The old notebook was a deep classical dive (preprocessing + NER + LDA +
BERTopic, ~52 cells, several multi-minute cells). The new arc keeps the spaCy sanity check,
"why NLP is hard", tokenization/lemmatization, and NER, but REPLACES LDA + BERTopic + the
heavy full-corpus preprocessing with a single fast sentence-embeddings + KMeans topic demo,
and adds the "what you will NOT do" expectation-setter and the A2 bridge.

| Old cell id | Old content | New cell | Action |
|-------------|-------------|----------|--------|
| cell-0 | Title + story + objectives | Cell 0 | edit (retheme to on-ramp, drop LDA/BERTopic promises) |
| cell-1 | Section 0 env setup md | Cell 1 | edit |
| cell-2 | pip install + spacy download | Cell 2 | edit (drop bertopic, add sentence-transformers + datasets) |
| cell-3 | imports + seeds + hyperparams | Cell 3 | edit (drop bertopic/LDA imports, add SentenceTransformer + load_dataset) |
| cell-0 (story) | story recap | Cell 4 | edit (What are we building today) |
| cell-9 | load Yelp | Cell 5 | edit (load raw BBC from HF SetFit/bbc-news; preview) |
| cell-6 | Why NLP is hard md | Cell 6 | keep (light edit) |
| cell-7 | ambiguity demo | Cell 7 | keep |
| cell-4/cell-5 | spaCy sanity check md+code | Cell 8 | edit (merge sanity check into tokenization theory) |
| cell-13 | tokenize spaCy vs TextBlob | Cell 9 | edit (demo tokenization + lemmatization together) |
| cell-18 | Lab build preprocess md | Cell 10 | edit |
| cell-19 | Lab preprocess starter | Cell 11 | edit (trim scope) |
| cell-21 | apply preprocess md/code | Cell 12 | edit (verification cell) |
| cell-22 | NER theory md | Cell 13 | keep (light edit) |
| cell-23 | NER on reviews demo | Cell 14 | edit (run on BBC raw text; casing teaching point) |
| cell-25 | displacy render | Cell 15 | keep |
| cell-31 | Lab top PERSON md | Cell 16 | keep (light edit) |
| cell-32 | Lab top PERSON starter | Cell 17 | keep (light edit) |
| (new) | embeddings topic theory | Cell 18 | FROM-NEW |
| (new) | embed + KMeans + top words demo | Cell 19 | FROM-NEW |
| cell-41 | Lab recover categories md | Cell 20 | edit (retarget to embedding clusters vs labels) |
| cell-42 | Lab crosstab starter | Cell 21 | edit (crosstab on KMeans labels not LDA) |
| cell-51 | Wrap-up md | Cell 22 | edit (add "what you will NOT do" + A2 bridge) |
| cell-8,10,11,12,14,15,16,17,20 | preprocessing sub-steps | dropped | drop (folded into Cell 9/11) |
| cell-26..30 | BBC ORG counting + bar chart | dropped | drop (folded into NER demo Cell 14) |
| cell-33..40 | LDA section | dropped | drop (replaced by embeddings) |
| cell-43..50 | BERTopic + analyze | dropped | drop (replaced by embeddings) |

---

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~23 cells)

## Cell 0 - Markdown: Title, story, objectives

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-0
Change: retheme to the course on-ramp; drop the LDA and BERTopic promises; switch the
topic-modelling line to "sentence embeddings + clustering"; add the A2 bridge line.

```
# A1. NLP From Scratch: Tokens, Entities, and Topics

**Part A. What NLP Can Do**

## The Story

You just joined a media-analytics team as the ML person. On day one your manager drops a
folder of raw text on your desk: thousands of customer reviews and a pile of news articles.
No labels, no schema, no model. Two questions land on you:

1. **"Who and what do these documents mention?"** Which people, companies, and places show up?
2. **"What are they about?"** Can we surface the themes without reading every document?

This notebook is the classical toolbox that answers both - with zero model training. We will
tokenize and lemmatize text with spaCy, pull out named entities, and discover topics with
modern sentence embeddings plus clustering. Everything runs on CPU in a few minutes.

## Why start here?

The rest of this course builds toward a fine-tuned transformer running in a chatbot. Before
we get there, you need to feel the raw material: text is messy and ambiguous, and the tools
that tame it take real effort. In the very next notebook you will watch one line of
HuggingFace `pipeline()` do these same tasks - and because you saw the plumbing here, you
will understand WHY it works.

## Learning Objectives

By the end of this notebook you will be able to:

1. Explain **why natural language is hard** for computers (ambiguity, casing, morphology).
2. Use **spaCy** to tokenize, lemmatize, and tag part-of-speech.
3. Run **Named Entity Recognition** and extract people, organizations, and places.
4. Discover **topics** in an unlabeled corpus with **sentence embeddings + KMeans**, then
   name each cluster from its top words.
5. Articulate when classical or embedding tools are enough and when to reach for transformers.

> **No deep learning training in this notebook.** We only USE pretrained tools. We start
> building models in Part B.
```

## Cell 1 - Markdown: Section 0 environment setup

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-1
Change: light edit; mention sentence-transformers; note the CPU-only, few-minutes runtime.

```
## Section 0. Environment Setup

We need a handful of packages. In Google Colab, run the next cell to install them. If you
are running locally in a virtualenv, you can skip the `!pip install` (but make sure these
packages are in your `requirements.txt`).

Everything in this notebook runs on **CPU** in a few minutes. A GPU is not required.
```

## Cell 2 - Code: pip install + spaCy model download

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-2
Change: drop `bertopic`; add `sentence-transformers` and `datasets` (we load the raw BBC
corpus from the HuggingFace Hub); keep `numpy<2`; keep the spaCy model download as a separate
command.

```python
# Install required packages (run this first on Google Colab).
# If running locally in a venv, you can skip this cell.
# We pin numpy<2 because several NLP libraries in this stack are not yet numpy 2 ready.
# `datasets` is HuggingFace's loader - we use it to pull the raw BBC news corpus from the Hub.

!pip install -q "spacy>=3.7,<3.9" "sentence-transformers>=2.7" "scikit-learn>=1.4" \
    "datasets>=2.19" textblob pandas matplotlib seaborn wordcloud "numpy<2"

# Download the small English spaCy pipeline. This download is SEPARATE from `pip install
# spacy` - it fetches the trained model weights we use for tokenization, POS tagging,
# lemmatization, and NER. If a later cell raises "Can't find model 'en_core_web_sm'",
# rerun this line.
!python -m spacy download en_core_web_sm
```

## Cell 3 - Code: imports, seeds, hyperparameters

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-3
Change: remove BERTopic and LDA imports; add `SentenceTransformer`, `KMeans`,
`TfidfVectorizer`, and `load_dataset`; keep seed and a small hyperparameter block.

```python
# ---- Standard library ----
import re
import random
import warnings
from collections import Counter

# ---- Scientific stack ----
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Classical NLP ----
import spacy
from spacy import displacy
from textblob import TextBlob

# ---- Embeddings + clustering (modern, but still "tools" - no training by us) ----
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# ---- Dataset loading from the HuggingFace Hub ----
from datasets import load_dataset

# ---- Word cloud for visual inspection ----
from wordcloud import WordCloud

# ---- Housekeeping ----
warnings.filterwarnings("ignore")

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---- Global knobs (edit these here, not deeper in the notebook) ----
TOUR_SIZE = 400          # how many documents we tour for speed (CPU friendly)
N_CLUSTERS = 5           # BBC news has 5 real categories; a good target for KMeans
TOP_WORDS_PER_TOPIC = 8  # how many words to print per discovered topic

print("Imports OK. sentence-transformers, spaCy, and datasets ready.")
```

## Cell 4 - Markdown: What are we building today

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-0
Change: extract the story into the standard "What Are We Building Today" framing; preview
the four concrete artifacts and reinforce "no training".

```
## What Are We Building Today?

Think of this notebook as a guided tour of the classical NLP toolbox, framed by the day-one
scenario above. By the end you will have produced four concrete artifacts, all without
training a single model:

1. A **tokenized, lemmatized** view of raw text (so a computer can count words sensibly).
2. A **named-entity extractor** that lists the people and organizations in a corpus.
3. A coloured **entity visualization** you could screenshot for a stakeholder.
4. A set of **discovered topics**, each summarized by its top words.

Our tour corpus is **BBC news** - real, full-length news articles across five categories
(business, entertainment, politics, sport, tech), loaded straight from the HuggingFace Hub
(`SetFit/bbc-news`). These are the original article bodies with **natural capitalization**
("David Beckham", "Microsoft", "Tony Blair") - exactly the raw text NER and our casing
experiment need. We will pretend we do NOT know the labels (that is the realistic unlabeled
setting), then peek at them at the very end to grade ourselves. We subsample to `TOUR_SIZE`
documents so every cell runs in seconds.
```

## Cell 5 - Code: load BBC corpus + preview

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-9
Change: switch the primary tour corpus from Yelp to the raw BBC news corpus, loaded from the
HuggingFace Hub (`SetFit/bbc-news`) instead of the pre-stemmed Dropbox CSV. The HF `text`
column is full raw article text with natural casing - mandatory for NER and the casing
experiment. Use the explicit, correct column names (`text`, `label_text`); subsample to
TOUR_SIZE for speed.

```python
# Load the raw BBC news corpus from the HuggingFace Hub.
# IMPORTANT: we use the HF dataset, NOT a pre-cleaned CSV. The `text` column here is the
# ORIGINAL article body with natural capitalization and punctuation ("David Beckham",
# "Microsoft", "London") - which is exactly what Named Entity Recognition needs. A
# lowercased/stemmed copy would silently break NER, displacy, and the PERSON lab later on.
raw = load_dataset("SetFit/bbc-news", split="train")
bbc = raw.to_pandas()

# Column names in SetFit/bbc-news: 'text' (raw article) and 'label_text' (category name).
text_col = "text"        # raw, naturally-cased article body
cat_col = "label_text"   # one of: business, entertainment, politics, sport, tech

# Subsample with a fixed seed for a fast, reproducible tour.
bbc = bbc.sample(n=TOUR_SIZE, random_state=SEED).reset_index(drop=True)

print(f"BBC shape after subsample: {bbc.shape}")
print(f"Text column: '{text_col}'   Category column: '{cat_col}'")

# Quick proof that the text is RAW (capital letters and punctuation are present).
sample0 = bbc[text_col].iloc[0]
print(f"\nFirst article (first 200 chars, note the natural casing):\n{sample0[:200]}")

print("\nHidden category counts (we IGNORE these until the final self-check):")
print(bbc[cat_col].value_counts())
bbc[[cat_col, text_col]].head(3)
```

## Cell 6 - Markdown: Why is NLP hard

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-6
Change: light edit; add the casing point (sets up the NER teaching moment later); keep the
ambiguity examples.

```
## Section 1. Why is NLP hard?

Computers eat numbers. Human language is messy, ambiguous, and context-dependent. A few
canonical headaches:

1. **Lexical ambiguity.** *"I saw her duck."* Did she crouch (verb), or do I see her pet
   duck (noun)? The same word, two meanings.
2. **Parsing ambiguity.** *"Time flies like an arrow; fruit flies like a banana."* Identical
   surface structure, radically different parse.
3. **Pronoun resolution.** *"The trophy doesn't fit in the suitcase because it is too big."*
   What is *it*?
4. **Negation and sarcasm.** *"This place was so great, I'll never come back."*
5. **Casing and spelling.** To a model trained on clean news text, *apple* (the fruit) and
   *Apple* (the company) are different things. Lowercase your text carelessly and entity
   recognition quietly falls apart - a trap we will hit on purpose later.

The tools in this notebook do not magically solve these. They handle them *probabilistically*:
right most of the time, wrong on edge cases. That failure mode is exactly why we INSPECT the
output instead of trusting it blindly. "Read your data first" is the oldest rule in NLP.

Here is ambiguity in action.
```

## Cell 7 - Code: ambiguity demo

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-7
Change: keep; load `nlp` here so the very first code touch of spaCy is in this cell.

```python
# Load the small English pipeline once. This object is reusable: load once, use many times.
nlp = spacy.load("en_core_web_sm")

# Classic headline ambiguity: is "flies" a verb or a noun? It depends on context.
sentences = [
    "Time flies like an arrow",
    "Fruit flies like a banana",
]
for s in sentences:
    doc = nlp(s)
    tags = [(t.text, t.pos_) for t in doc]
    print(s, "->", tags)

# Notice that "flies" gets a DIFFERENT part-of-speech tag in each sentence. That is spaCy's
# statistical model resolving the ambiguity from the surrounding words - something a naive
# dictionary lookup could never do.
```

## Cell 8 - Markdown: Tokenization, lemmatization, normalization theory

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-4 + cell-5
Change: merge the spaCy sanity check into the tokenization theory; add lemmatization and the
clean-vs-raw distinction; preview the inline code so the next cell is a real demo.

```
## Section 2. Tokenization, lemmatization, and normalization

Before any counting or clustering, we have to turn a wall of text into clean units. Three
ideas do most of the work:

- **Tokenization**: split text into words/punctuation. spaCy respects contractions and
  punctuation via its statistical pipeline; TextBlob is a simpler regex tokenizer.
- **Lemmatization**: map word variants to a dictionary base form. *running -> run*,
  *mice -> mouse*, *better -> well*. This matters for topic work: if *good* and *better* are
  two tokens, they split their weight and neither topic reads as "positive".
- **Normalization / cleanup**: lowercase, strip URLs and punctuation, collapse whitespace.
  Great for word counting and clustering - but DANGEROUS before entity recognition, because
  NER leans on capitalization. We will keep a clean version AND the raw version around.

A spaCy `Doc` exposes all of this per token:

```python
doc = nlp("Apple is looking at buying a U.K. startup for $1 billion")
for t in doc:
    print(t.text, t.lemma_, t.pos_, t.is_stop)
```

Let us see tokenization and lemmatization side by side.
```

## Cell 9 - Code: tokenization + lemmatization demo

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-13
Change: extend the spaCy-vs-TextBlob demo to also show lemmas and a small DataFrame, and
define the reusable `clean_text` helper here so the lab can use it.

```python
sample = "It's the best coffee shops I've ever visited - 10/10 would go again!"

# spaCy tokenization keeps punctuation as its own tokens; TextBlob drops it.
spacy_tokens = [t.text for t in nlp(sample)]
textblob_tokens = list(TextBlob(sample).words)
print("spaCy    :", spacy_tokens)
print("TextBlob :", textblob_tokens)

# Lemmatization: notice "shops" -> "shop" and "visited" -> "visit".
print("\nToken -> lemma:")
for t in nlp(sample):
    if t.is_alpha:
        print(f"  {t.text:10s} -> {t.lemma_}")

# A reusable, predictable cleanup helper we will lean on in the lab.
def clean_text(text: str) -> str:
    """Lowercase, drop URLs, keep only letters and spaces, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)            # keep letters and whitespace only
    text = re.sub(r"\s+", " ", text).strip()         # collapse repeated whitespace
    return text

print("\nclean_text demo:")
print("  RAW  :", sample)
print("  CLEAN:", clean_text(sample))
```

## Cell 10 - Markdown: Lab 2.1 build the preprocess function

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-18
Change: trim scope to the essentials; keep the "lemmas, not surface text" guidance; do not
leak the one-liner.

```
### Lab 2.1. Build a reusable `preprocess` function

Combine the pieces above into one `preprocess(text)` function that returns a clean,
space-joined string of lemmas. Concretely it should:

1. Run `clean_text` on the input.
2. Parse the cleaned string with `nlp`.
3. Keep only tokens that are **alphabetic**, **not stopwords**, and whose **lemma** is longer
   than two characters.
4. Return those lemmas joined by single spaces.

**Why these rules?**
- Use `token.lemma_`, not `token.text`, so *shops* and *shop* collapse to one token.
- `token.is_stop` removes high-frequency filler (*the, is, and*) that drowns out topics.
- The length filter drops stray single letters left over from cleanup.

You have everything you need: `clean_text`, `nlp`, and the token attributes `is_alpha`,
`is_stop`, and `lemma_`. Build it, then run the verification cell.
```

## Cell 11 - Code: Lab 2.1 starter

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-19
Change: trim to a single function; placeholders never reveal the comprehension; add a
verification block.

```python
def preprocess(text: str) -> str:
    """Clean -> parse -> keep good lemmas -> join into one string."""

    # 1. Basic regex cleanup. Reuse the helper you already have.
    cleaned = None  # YOUR CODE

    # 2. Parse the cleaned text with the spaCy pipeline.
    doc = None  # YOUR CODE

    # 3. Build a list of lemmas. Keep a token only if it is alphabetic, is NOT a stopword,
    #    and its lemma is longer than two characters. Iterate over `doc` and test each token
    #    with its attributes, then collect token.lemma_ for the survivors.
    lemmas = None  # YOUR CODE

    # 4. Join the surviving lemmas into a single space-separated string.
    return None  # YOUR CODE


# ---- Verification (provided) ----
example = bbc[text_col].iloc[0]
print("RAW   :", example[:160])
if preprocess(example) is not None:
    out = preprocess(example)
    print("CLEAN :", out[:160])
    assert isinstance(out, str), "preprocess should return a string"
    assert " the " not in f" {out} ", "stopword 'the' should be gone"
    print("\nLooks good - preprocess returns a clean lemma string.")
```

## Cell 12 - Code: apply preprocess to the corpus

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-21
Change: apply to the subsampled BBC tour corpus (fast); print a sanity sample and average
length. Safety-net: produces `bbc["clean"]`, consumed by the topic lab.

```python
# Apply the pipeline to the whole tour corpus. On TOUR_SIZE=400 articles this runs in seconds.
# (For big corpora you would use nlp.pipe(texts, batch_size=64) for a 2-5x speedup; for a
#  few hundred docs the simple .apply is fine and easier to read.)
bbc["clean"] = bbc[text_col].apply(preprocess)

# Sanity check: first three cleaned articles and the average cleaned length in words.
for i in range(3):
    print(f"[{i}] {bbc['clean'].iloc[i][:150]}\n")

avg_len = bbc["clean"].str.split().str.len().mean()
print(f"Average cleaned-article length (words): {avg_len:.1f}")
```

## Cell 13 - Markdown: NER theory

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-22
Change: light edit; keep the label table; foreground the casing lesson and the
"feed RAW text" rule that the demo then proves.

```
## Section 3. Named Entity Recognition (NER)

**NER** spots spans of text that refer to real-world *things* and labels them: people,
companies, places, dates, money.

Why care? Three concrete reasons:

1. **Information extraction**: populate a search index or knowledge graph.
2. **Anonymization**: redact PII (PERSON, GPE) before sharing data.
3. **Faceted exploration**: "which organizations dominate this news feed?" - exactly the
   day-one question from our story.

spaCy's `en_core_web_sm` ships with an NER model. Calling `nlp(text)` already runs it, and
entities land at `doc.ents`. Each entity has `.text` and `.label_`.

| Label | Meaning |
|-------|---------|
| `PERSON` | People, real or fictional |
| `ORG` | Companies, agencies, institutions |
| `GPE` | Countries, cities, states |
| `LOC` | Non-GPE locations (rivers, mountain ranges) |
| `MONEY` | Monetary values |
| `DATE` | Dates and periods |

**Critical gotcha.** This model was trained on properly capitalized news text. Feed it the
lowercased `clean` column and entity recall collapses, because casing is a major signal -
*apple* is not *Apple*. **Always run NER on the original, raw text.** We prove this next.
```

## Cell 14 - Code: NER demo on raw text + casing experiment

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-23
Change: run on BBC RAW article text (the HF `text` column, naturally cased); ADD the explicit
raw-vs-lowercased experiment so the casing lesson is demonstrated on data where it actually
holds (the old Dropbox `Description` was pre-lowercased, which would have made this lesson
false). Sum the gap across several articles so the effect is unmistakable.

```python
# Run NER on the first few RAW articles. Note: use bbc[text_col] (original casing/punctuation),
# NOT bbc["clean"] - the NER model needs proper capitalization to find entities at all.
ent_rows = []
for idx, article in enumerate(bbc[text_col].head(5)):
    doc = nlp(article)
    for ent in doc.ents:
        ent_rows.append({"article_id": idx, "entity": ent.text, "label": ent.label_})

ent_df = pd.DataFrame(ent_rows)
print("Entities from the first 5 RAW articles:")
print(ent_df.head(12))

# ---- Casing experiment: prove that lowercasing breaks NER ----
# We compare entity counts on RAW vs lowercased text across several real articles. Because
# these are full naturally-cased news articles, the RAW count is much higher: capitalization
# is one of the strongest signals spaCy's NER model uses.
raw_total, lower_total = 0, 0
for article in bbc[text_col].head(10):
    raw_total += len(nlp(article).ents)
    lower_total += len(nlp(article.lower()).ents)

print(f"\nAcross 10 articles: {raw_total} entities on RAW text "
      f"vs {lower_total} on lowercased text.")
print("Lowercasing throws most entities away. This is exactly why we run NER on the RAW")
print("`text` column and never on the lowercased `clean` column.")
```

## Cell 15 - Code: displacy inline visualization

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-25
Change: keep; render one RAW BBC article (the naturally-cased HF `text` column) instead of one
review; keep the one-document safety note. Because the article is full raw text, the render is
densely highlighted - a real stakeholder-ready artifact (it would be near-empty on lowercased
text).

```python
# spaCy ships an inline visualizer. style="ent" highlights each entity with its label.
# We render the RAW article text (bbc[text_col]) so capitalization is intact and entities
# light up. SAFETY: render ONE document only. Rendering a whole corpus produces one HTML
# block per document and will freeze the notebook.
doc = nlp(bbc[text_col].iloc[0])
displacy.render(doc, style="ent", jupyter=True)
```

## Cell 16 - Markdown: Lab 3.1 top PERSON entities

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-31
Change: light edit; point at the BBC tour corpus; introduce `nlp.pipe` + `disable` for speed;
add a clearly labelled stretch and a homework extension.

```
### Lab 3.1. Who shows up most? Top PERSON entities

Your turn. Using the fast batched API `nlp.pipe`, find the people mentioned most often in the
tour corpus.

Steps:

1. Iterate over `bbc[text_col]` using `nlp.pipe(...)`. Pass `batch_size=32` and
   `disable=["parser", "tagger", "lemmatizer"]` so spaCy only runs the NER component (much
   faster - calling `nlp()` in a plain loop is the classic beginner mistake).
2. For every entity in each doc, keep only those whose label is `"PERSON"`.
3. Count occurrences in a `Counter`, storing `ent.text.strip()` (the `.strip()` keeps
   "Tony Blair" and "Tony Blair " from counting as two different people).
4. Print the **top 10** with `.most_common(10)`.

**Stretch (entity normalization).** Real names appear in variants: *Tony Blair*, *Blair*,
*Mr Blair*. Counting them separately understates the true frequency. Write a tiny normalizer
that maps each PERSON entity to its **last token** (a crude surname key), recount, and compare
the top 10 to your raw count. This is a baby version of *entity linking* / *grounding* - the
real production task of mapping surface mentions to canonical identities.

**Homework extension (co-occurrence).** Build an ORG-and-GPE co-occurrence view: for each
article, collect the set of `ORG` and `GPE` entities, then count which (ORG, GPE) pairs appear
together most often. This is the seed of a knowledge graph. Bonus: think about how news-trained
NER would degrade on noisy product reviews or tweets (domain shift) and how you would detect it.
```

## Cell 17 - Code: Lab 3.1 starter

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-32
Change: keep the structure; placeholders never reveal the `nlp.pipe` call or the label test;
add a verification block.

```python
person_counter = Counter()

# 1. Loop over the corpus with nlp.pipe for speed. Replace None with the nlp.pipe(...) call
#    over bbc[text_col]; remember batch_size and disable=[...] for the components you do not
#    need (you only want NER here).
for doc in None:  # YOUR CODE
    # 2. Look at every entity in this doc.
    for ent in doc.ents:
        # 3. Keep only PERSON entities, then count the stripped text.
        if None:  # YOUR CODE  -- test ent.label_
            person_counter[None] += 1  # YOUR CODE  -- what string do we store?

# 4. Top 10 most mentioned people.
top_people = None  # YOUR CODE
print(top_people)

# ---- Verification (provided) ----
if top_people is not None:
    assert len(top_people) <= 10, "should be at most the top 10"
    assert all(isinstance(name, str) for name, _ in top_people), "keys should be names"
    print("\nNice - you extracted the most-mentioned people with batched NER.")
```

## Cell 18 - Markdown: Topic discovery with sentence embeddings theory

Provenance: FROM-NEW

```
## Section 4. Discovering topics with sentence embeddings

We never trained a topic model the old way (LDA, which treats text as a bag of independent
words and ignores word order). Instead we use the modern, simpler recipe that also previews
the most important idea in this course: **turn each document into a vector, then cluster the
vectors.**

The recipe:

1. **Embed.** A pretrained `sentence-transformers` model maps each article to a fixed-size
   vector (384 numbers) that captures meaning. We use `all-MiniLM-L6-v2` - a distilled,
   6-layer BERT, only ~22 MB, fast on CPU. Documents about the same theme land near each
   other in this space.
2. **Cluster.** `KMeans` groups the vectors into `k` clusters. Each cluster is a discovered
   topic. We set `n_init=10` and `random_state=42` so the result is reproducible (KMeans
   starts from random centroids; without these it can drift run to run).
3. **Name.** A cluster of vectors is not human-readable, so we run a per-cluster TF-IDF over
   the cleaned text and read off each cluster's most distinctive words. Those top words ARE
   the topic label.

> This `all-MiniLM-L6-v2` embedder is the through-line of the course. In Part B you use the
> same "text -> vector" idea as features for a small neural net; in Part C the model that
> produces these vectors becomes the thing you fine-tune. Meet it now.

Here it is end to end.
```

## Cell 19 - Code: embed + KMeans + per-cluster top words demo

Provenance: FROM-NEW

```python
# 1. EMBED: load the sentence-transformer and encode the RAW articles (bbc[text_col]).
#    We pass raw text, NOT the cleaned column, because this model was trained on natural
#    sentences: it uses word order, casing, stopwords, and punctuation to build meaning.
#    Stripping all that (the `clean` column) would throw away signal the embedder relies on.
#    ~22 MB download, runs on CPU.
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(bbc[text_col].tolist(), show_progress_bar=False)
print(f"Embeddings shape: {embeddings.shape}  (documents x 384-dim vectors)")

# 2. CLUSTER: KMeans on the embeddings. n_init=10 + fixed seed = reproducible clusters.
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
bbc["cluster"] = kmeans.fit_predict(embeddings)
print("Documents per cluster:")
print(bbc["cluster"].value_counts().sort_index())

# 3. NAME: run TF-IDF over the CLEANED text, then for each cluster read off the words with
#    the highest average TF-IDF weight inside that cluster. Those words name the topic.
tfidf = TfidfVectorizer(min_df=3, max_df=0.5)
tfidf_matrix = tfidf.fit_transform(bbc["clean"])
terms = np.array(tfidf.get_feature_names_out())

print("\nDiscovered topics (top words per cluster):")
for c in range(N_CLUSTERS):
    rows = (bbc["cluster"] == c).values
    mean_weights = np.asarray(tfidf_matrix[rows].mean(axis=0)).ravel()
    top_idx = mean_weights.argsort()[::-1][:TOP_WORDS_PER_TOPIC]
    print(f"  Cluster {c}: {', '.join(terms[top_idx])}")
```

## Cell 20 - Markdown: Lab 4.1 did the clusters recover the categories

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-41
Change: retarget from "did LDA recover categories" to "did the embedding clusters recover
categories"; introduce the silhouette stretch and a BERTopic-comparison homework.

```
### Lab 4.1. Did the clusters rediscover the real categories?

This is the main self-check of the notebook. BBC has 5 hidden categories (business,
entertainment, politics, sport, tech) and we asked KMeans for 5 clusters. Did clustering on
sentence embeddings rediscover the human labels - without ever seeing them?

Steps:

1. Build a **cross-tabulation** of cluster id (`bbc["cluster"]`) vs the true category
   (`bbc[cat_col]`) with `pd.crosstab`.
2. Draw it as a heatmap with `sns.heatmap(..., annot=True, fmt="d", cmap="Blues")`.
3. In one sentence, say which clusters cleanly map to which categories. A clean recovery has
   one dominant cell per row.

Expect strong recovery: sentence embeddings usually separate sport, tech, and entertainment
cleanly; business and politics sometimes bleed together.

**Stretch (choosing k).** We assumed k=5 because we secretly knew the answer. In the real
unlabeled setting you must choose k. Loop k over `range(2, 9)`, fit KMeans for each, compute
`silhouette_score(embeddings, labels)` from `sklearn.metrics`, and plot score vs k. Does the
best silhouette agree with the true 5? Combine with an elbow (inertia) plot for a second
opinion.

**Homework extension (compare to BERTopic).** Install `bertopic`, fit it on the same raw
articles, and compare its discovered topics to your KMeans-plus-TF-IDF topics. BERTopic adds
UMAP dimensionality reduction and HDBSCAN (which picks the number of topics for you and flags
outliers as cluster -1). Which gives more coherent, more specific topics on this corpus? When
would the extra machinery be worth it in production?
```

## Cell 21 - Code: Lab 4.1 starter

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-42
Change: crosstab on KMeans `cluster` labels instead of LDA dominant topic; placeholders do
not reveal the crosstab or heatmap call; add a verification block.

```python
# 1. Cross-tabulate discovered cluster vs true category.
#    Hint: pd.crosstab takes two Series - the cluster column and the category column.
crosstab = None  # YOUR CODE

print(crosstab)

# 2. Heatmap of the crosstab.
fig, ax = plt.subplots(figsize=(8, 5))
# YOUR CODE: call sns.heatmap on `crosstab` with annot=True, fmt="d", cmap="Blues", ax=ax
None
ax.set_title("KMeans cluster vs true BBC category (one big cell per row = clean recovery)")
ax.set_xlabel("True category")
ax.set_ylabel("Discovered cluster")
plt.tight_layout()
plt.show()

# ---- Verification (provided) ----
if crosstab is not None:
    assert crosstab.values.sum() == len(bbc), "crosstab should cover every document"
    print("\nGreat - you compared unsupervised clusters against the hidden labels.")
```

## Cell 22 - Markdown: Wrap-up, what you will NOT do, bridge to A2

Provenance: FROM-OLD 1-Pre-NLP/1-Topic_Modelling_and_NER.ipynb cell-51
Change: trim to the new arc; add the explicit "what you will NOT do here" expectation-setter
and the one-line bridge to A2 (pipeline tour); keep a short when-to-use-what table.

```
## Wrap-up

### What you built (zero training)

- A reusable **clean -> tokenize -> lemmatize** pipeline (`clean_text`, `preprocess`).
- A **named-entity extractor** that surfaces the people and organizations in a corpus, plus
  an inline visualization - and a hard-won lesson that casing matters for NER.
- A **topic discovery** pipeline using sentence embeddings + KMeans, with each topic named by
  its top TF-IDF words, that rediscovered the hidden BBC categories.

### What you did NOT do here (on purpose)

- No model **training** - we only USED pretrained tools. Training starts in Part B.
- No deep dive into LDA or BERTopic internals - we picked the simplest recipe that works.
- No transformers yet - but you just met the sentence embedder that becomes their foundation.

### When is this toolbox enough?

| Situation | Reach for |
|-----------|-----------|
| Extract names/places, redact PII | spaCy NER (rules + small model, fast, cheap) |
| Group unlabeled docs by theme | sentence embeddings + KMeans |
| Quick, interpretable baseline before any DL | clean + count + cluster, every time |
| Nuanced meaning, sarcasm, negation, real accuracy | transformers (next notebooks) |

Start simple. A clean baseline is fast, interpretable, and surprisingly hard to beat - so it
is always worth building before you reach for a heavier model.

### Bridge to A2

All of this took installs, custom stopwords, casing care, and clustering plumbing. In the
**next notebook (A2: Pipeline Tour)** you will solve these same tasks - sentiment, NER,
zero-shot classification, question answering - in essentially **one line each** with
HuggingFace `pipeline()`. The payoff: because you saw the raw material here, you will
understand exactly what that one line is doing under the hood.
```

---

# VERIFICATION CHECKLIST

- [ ] Notebook runs top-to-bottom on CPU in Colab in a few minutes (no GPU needed).
- [ ] Install cell pins `numpy<2`; includes `datasets`; no `bertopic`, no TensorFlow/Keras,
      no PyTorch training.
- [ ] `spacy>=3.7,<3.9` and `python -m spacy download en_core_web_sm` both present.
- [ ] `SentenceTransformer("all-MiniLM-L6-v2")` loads and `.encode` returns a (N, 384) array.
- [ ] Corpus is the RAW `SetFit/bbc-news` HF dataset via `load_dataset(..., split="train")`;
      `text_col="text"` (naturally-cased article body), `cat_col="label_text"`. NOT the
      pre-stemmed Dropbox `bbc.csv` `Description` column.
- [ ] Cell 5 prints a raw-text sample showing capital letters and punctuation are present.
- [ ] NER demo proves the casing point on RAW text: raw entity count >> lowercased count
      (summed across multiple articles, so the gap is large and unambiguous).
- [ ] `displacy.render(..., style="ent", jupyter=True)` renders exactly ONE RAW article and is
      densely highlighted (a real stakeholder artifact, not empty).
- [ ] PERSON lab returns real, named people (e.g. politicians/athletes) from raw articles, not
      an empty Counter.
- [ ] `clean_text`/`preprocess` operate on genuinely raw text, so lemmatization produces real
      lemmas (not garbage from already-stemmed tokens).
- [ ] KMeans uses `n_init=10` and `random_state=42` for reproducibility.
- [ ] Per-cluster TF-IDF top words read as coherent topic names.
- [ ] Every `None # YOUR CODE` placeholder hides the answer (cover-the-solution test passes).
- [ ] Each lab has a standard task, a labelled stretch, and a homework extension.
- [ ] Verification blocks present in all three lab starter cells.
- [ ] Plain ASCII only: no em/en dashes, no Unicode multiplication, no emoji.
- [ ] Bridge to A2 and "what you will NOT do" present in the wrap-up.

---

# RESEARCH VALIDATED (June 2026)

- Tour corpus is `SetFit/bbc-news` on the HuggingFace Hub: 2,225 BBC news articles, train
  split 1,230 rows, columns `text` (full RAW article body with natural capitalization),
  `label` (0-4), and `label_text` (business, entertainment, politics, sport, tech). Verified
  the `text` field contains naturally-cased entities (David Beckham, Microsoft, Tony Blair,
  George W. Bush). This RAW text is required for NER, displacy, the PERSON lab, lemmatization,
  and the casing experiment to teach what they claim. Loaded with
  `load_dataset("SetFit/bbc-news", split="train").to_pandas()`.
  https://huggingface.co/datasets/SetFit/bbc-news ,
  https://huggingface.co/docs/datasets/loading
- REJECTED the Dropbox `bbc.csv`
  (https://www.dropbox.com/scl/fi/lfa2ryv86uqd3y988irfw/bbc.csv?rlkey=vtwdf6g8sejhkf75p7o36ev00&dl=1):
  inspected directly, its `Description` column is already lowercased, stemmed, and
  stopword-stripped (row 0 = "india attend g7 meet seven lead industrialis nation friday
  unlik cow newcom status london ..."); only `Title` keeps natural casing but is a single
  short headline. spaCy NER, displacy, and PERSON extraction collapse on that pre-processed
  text and lemmatizing already-stemmed tokens yields garbage, so it cannot be the NER/topic
  corpus. (Local fetch + column inspection, June 2026.)
- spaCy 3.8 is current/stable; `en_core_web_sm` pipeline (tok2vec, tagger, parser, ner,
  lemmatizer) and `doc.ents` / `displacy.render(style="ent", jupyter=True)` are unchanged.
  https://spacy.io/usage/models , https://pypi.org/project/spacy/3.8.0/ ,
  https://spacy.io/usage/visualizers
- spaCy NER depends heavily on capitalization: lowercasing entities prevents recognition
  ("apple" not tagged ORG, "Apple" is). Justifies feeding RAW text to NER.
  https://github.com/explosion/spaCy/issues/701 ,
  https://github.com/explosion/spaCy/discussions/11931
- spaCy speedups: use `nlp.pipe(texts, batch_size=...)` and `disable=[...]` instead of calling
  `nlp()` in a loop. https://spacy.io/usage/processing-pipelines ,
  https://github.com/explosion/spaCy/discussions/8402
- sentence-transformers latest 5.5.1 (May 2026); `all-MiniLM-L6-v2` maps text to 384-dim
  vectors, ~22 MB, distilled 6-layer BERT, ideal for clustering/semantic search on CPU.
  https://pypi.org/project/sentence-transformers/ ,
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- gensim 4.3.3 requires `numpy<2` AND `scipy<1.13` (the `scipy.linalg.triu` removal breaks
  import). Decision: keep gensim OUT of A1 core; topic discovery uses sentence-transformers,
  which avoids the conflict. Confirms the `numpy<2` pin.
  https://github.com/piskvorky/gensim/issues/3525 ,
  https://github.com/googlecolab/colabtools/issues/5199
- KMeans needs explicit `random_state` and `n_init` for reproducibility; default `n_init`
  changed to "auto" in recent sklearn, so set `n_init=10` deliberately.
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html ,
  https://github.com/scikit-learn/scikit-learn/discussions/25016
- Name clusters with per-cluster (class-based) TF-IDF top words - the c-TF-IDF idea.
  https://medium.com/@shashankag14/understanding-tf-idf-and-c-tf-idf-in-topic-modeling-071eb82fa858
- Choosing k: hybrid elbow (inertia) + `silhouette_score(X, labels)`; best value 1, worst -1.
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html ,
  https://www.geeksforgeeks.org/machine-learning/elbow-method-vs-silhouette-score-which-is-better/
- BERTopic vs LDA vs KMeans: BERTopic (UMAP+HDBSCAN+embeddings) is heavier and slower; for a
  fast small-corpus CPU teaching demo, embeddings+KMeans is simpler and reliable. Good
  homework comparison. https://bertopic.com/how-is-bertopic-different-from-lda/ ,
  https://towardsdatascience.com/a-practical-guide-to-bertopic-for-transformer-based-topic-modeling/
- Entity normalization / grounding: surface mentions map to canonical entities; name variants
  (capitalization, reordering, typos) justify the dedup stretch.
  https://en.wikipedia.org/wiki/Entity_linking ,
  https://arxiv.org/pdf/1811.07514
- NER domain shift: news-trained models degrade on tweets/reviews ("training data should look
  like your inputs") - basis for the homework caution.
  https://github.com/explosion/spaCy/discussions/7949
- Start-simple guidance: benchmark classical/TF-IDF baselines before transformers; classical
  methods stay in production for cost/interpretability; "read your data first" (EDA).
  https://buildml.substack.com/p/before-transformers-the-nlp-fundamentals ,
  https://spotintelligence.com/2023/09/15/exploratory-data-analysis-nlp/
- HuggingFace `pipeline()` solves NER/sentiment/QA/zero-shot in ~3 lines with no training -
  the A2 payoff this notebook sets up. https://huggingface.co/docs/transformers/main_classes/pipelines ,
  https://huggingface.co/learn/llm-course/chapter2/2

---

> Plan written to `plans/refactor/notebooks/A1-nlp-from-scratch.md`.
>
> Next step: run `/build-notebook nlp-from-scratch colab` to generate the exercise + solution
> notebooks from this plan, 5 cells at a time with approval between batches.
