# B5 - Word2Vec and Sentence Embeddings - Cell-by-Cell Plan

## Status

RENEWED. Merge of two old source notebooks:

- `2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb` (concepts only, NOT the from-scratch
  CBOW training loop, which is cut per the manifest).
- `2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb` (sentence-transformers
  similarity + semantic search, condensed).

New file: `B-How-It-Works/5-Word2Vec_and_Sentence_Embeddings.ipynb`, slug
`word2vec-sentence-embeddings`. Environment: Colab.

The whole point of the renew: we do NOT train embeddings from scratch. We LOAD pretrained
word vectors (gensim) and a pretrained sentence encoder (sentence-transformers), and explore
the geometry. That is what you do in real life, and it keeps the class moving toward the
B7 stopper.

## Context

Students arrive from B4 (PyTorch Tensors) holding: tensor creation, dtype/shape, matrix
ops, broadcasting, and autograd (`requires_grad`, `.backward()`, `.grad`, `torch.no_grad()`).
They have a `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` idiom and
`torch.manual_seed(42)`. From Part A they already met the text-to-vector idea: `all-MiniLM-L6-v2`
was the A1 embedder. This notebook turns "a vector" into "a geometry you can do arithmetic in."

They leave with the key insight: a word (or a sentence) is a point in a dense space where
direction and distance carry meaning. Cosine similarity reads that geometry. A pretrained
model gives you that space for free. The mean-pool-of-word-vectors document vector they build
in the homework is the exact feature matrix the B7 MLP will consume.

Exact names introduced here and reused later:

- `wv` - gensim `KeyedVectors`, the pretrained word vectors (glove-wiki-gigaword-100).
- `embedder` - `SentenceTransformer('all-MiniLM-L6-v2')` (same model id as A1, reused).
- `cos_sim`, `most_similar`, `analogy()` helper, `semantic_search()` helper.
- `doc_vector(text)` (homework) - mean-pool of word2vec vectors, the B7 feature.

No variable-name collision with B4 (B4 uses `x`, `w`, `y`, tensor scratch names; B5 uses
`wv`, `embedder`, `corpus`, `corpus_emb`). `device` and `SEED=42` carry over unchanged.

## Chatbot Through-Line

The course ends on a fine-tuned DistilBERT loaded into a Gradio chatbot. This notebook is the
"text becomes geometry" rung of the ladder:

- A1 showed text-to-vector with `all-MiniLM-L6-v2` as a black box. B5 opens that box one
  level: vectors live in a space, cosine similarity reads it, and you can search and cluster
  with nothing but geometry. Semantic search here is literally the retrieval primitive that
  a RAG chatbot uses to pull context before answering.
- The homework `doc_vector()` (mean-pool of word2vec) is the exact feature the B7 STOPPER
  feeds into an MLP ("embeddings as features"). B5 hands B7 its inputs.
- Static word2vec is fixed per word: "bank" (river) and "bank" (money) collapse to one vector.
  That polysemy gap is the explicit motivation for the contextual transformer (DistilBERT) in
  C9. B5 sets up the limitation that C9 resolves.

One-line bridge to B6: "Now that text is a fixed-size vector, we need a model that maps those
vectors to a label. B6 builds the PyTorch neural-net machinery; B7 points it at these
embeddings."

## STAR Story

- Situation: You are the ML/NLP person on a real team (the customer-support platform from
  Part A). Part A let you call `pipeline()` and an embedder as black boxes. Now leadership
  wants a semantic search over the support knowledge base so agents stop missing relevant
  articles, and they want to know whether you can build classifier features without paying a
  GPU bill on every request.
- Task: Show that text turns into vectors you can reason about geometrically, build a working
  semantic search, and produce reusable document features, all from PRETRAINED models with no
  training from scratch.
- Action: Load pretrained word vectors with gensim, probe similarity and analogies, project a
  word set to 2D, demonstrate why averaging word vectors is a weak sentence baseline, switch to
  a pretrained sentence encoder, and build a brute-force semantic search over a real corpus.
- Result: A semantic search engine in under 50 lines that beats keyword matching, an honest map
  of where static vectors break (polysemy, order), and a `doc_vector()` feature builder that
  feeds directly into the B7 MLP. The team gets retrieval today and a clear path to a fine-tuned
  classifier next.

## Deliverables

- Exercise: `B-How-It-Works/5-Word2Vec_and_Sentence_Embeddings.ipynb`
- Solution: `Solutions/B-How-It-Works/5-Word2Vec_and_Sentence_Embeddings.ipynb` (gitignored)

## Session Timing (~60-90 min)

| Block | Cells | Time |
|-------|-------|------|
| Setup + context | 0-5 | 10 min |
| Part 1: word vectors (similarity, analogy, PCA) + Lab 1 | 6-14 | 25 min |
| Part 2: sentence embeddings (weak baseline -> SBERT) + Lab 2 | 15-20 | 25 min |
| Part 3: semantic search + Lab 3 | 21-22 | 15 min |
| Homework + wrap-up | 23 | async + 5 min |

## Cell Migration Map

Old source cells -> new cell N -> action. Old A = `5-CBOW_Word_Embeddings.ipynb`,
Old B = `7-Sentence_Transformers_Similarity.ipynb`. CBOW training cells (A 7-28, 30) are
DROPPED per the manifest (no from-scratch training).

| Old cell | New cell | Action |
|----------|----------|--------|
| A cell-0 (story) | 0 | edit - retheme to chatbot/support, drop logreg framing |
| A cell-1 (env md) | 1 | edit - merge into setup header |
| A cell-2 (pip) + B idx2 (pip) | 2 | edit - new pinned install (gensim, sentence-transformers, scipy<1.13, numpy<2) |
| A cell-3 + B idx3/idx4 (imports/seed) | 3 | edit - merge imports + seed + device |
| A cell-0 / B idx0 (scenario) | 4 | edit - "what we build today" chatbot framing |
| (new) | 5 | FROM-NEW - load `wv` pretrained vectors preview |
| A cell-4 (one-hot md) | 6 | edit - condense one-hot problem |
| A cell-5 (one-hot demo) | 7 | keep - one-hot orthogonality demo |
| A cell-29 (KeyedVectors md) + A cell-31 (probe md) | 8 | edit - "querying the space" |
| A cell-32 (most_similar lab) | 9 | edit - now a DEMO on pretrained wv |
| A cell-33 (analogy md) | 10 | edit - drop "fails on Yelp", now works on GloVe |
| A cell-34 (analogy demo) | 11 | edit - real working analogies + bias caveat |
| A cell-31/32 (probe) | 12 | edit - Lab 1 starter: similarity + analogy on new words |
| (new verification) | 13 | FROM-NEW - Lab 1 verification block |
| A cell-35 (PCA md) + A cell-36 (PCA demo) | 14 | edit - PCA 2D of a chosen word set (demo) |
| B idx5 (why sentence emb md) | 15 | edit - condense, drop GloVe-file download |
| B idx7/idx8 (mean baseline + failures) | 16 | edit - mean-of-wv baseline using `wv`, failure pairs |
| B idx10 (SBERT md) | 17 | edit - condense SBERT theory |
| B idx11/idx12/idx13 (load + encode + compare) | 18 | edit - load `embedder`, encode, compare to baseline |
| B idx16 (lab md) | 19 | edit - Lab 2 instructions: STS-B similarity |
| B idx18 (lab starter) | 20 | edit - Lab 2 starter on STS-B pairs + verification |
| B idx19/idx20/idx21/idx22 (search engine) | 21 | edit - semantic_search over corpus (demo) |
| B idx23/idx24 (filtered search lab) | 22 | edit - Lab 3 starter: build search() + verification |
| B idx35/idx36 (homework + wrap md) | 23 | edit - homework doc_vector() bridge to B7 + wrap-up |

Dropped old cells: A 6-28, A 30 (CBOW dataset/model/training/save - the entire from-scratch
pipeline); B idx2 GloVe-txt-file path (replaced by gensim downloader), B idx6 (GloVe file load),
B idx9/idx14/idx15/idx17/idx25-34 (Doc2Vec legacy, batching micro-benchmark, TF-IDF aside,
KMeans/t-SNE clustering) - condensed away to hold the 24-cell budget; clustering becomes a
homework option, not core.

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~24 cells)

## Cell 0 - Markdown: Title, objectives, prerequisites

```markdown
# B5. Word2Vec and Sentence Embeddings

*ML and NLP course - Data Trainers LLC - Axel Sirota*

## The story

You are the ML/NLP person on the customer-support platform from Part A. So far you have
called `pipeline()` and a sentence embedder as black boxes. Now the team wants two things:
a semantic search over the support knowledge base (so agents stop missing relevant articles),
and classifier features that do not require a GPU on every single request.

Both come from the same idea: text becomes a **vector**, and vectors live in a space where
**direction and distance carry meaning**. In B4 you learned tensors and autograd. Here you
learn the geometry that the rest of the course classifies, searches, and fine-tunes.

We will NOT train embeddings from scratch. We load **pretrained** ones, because that is what
you do in real life, and explore the geometry.

## What you will be able to do

- Load pretrained word vectors and query them: nearest neighbors, cosine similarity, analogies.
- Project word vectors to 2D and read the clusters.
- Explain why averaging word vectors is a weak sentence representation.
- Encode sentences with a pretrained transformer and build a working semantic search.
- Build a document vector (mean-pooled word vectors): the exact feature the B7 MLP will use.

## Prerequisites

- B4: tensors, dtype/shape, matrix ops, broadcasting, autograd.
- Comfort with numpy arrays and cosine similarity as "dot product over norms".

## Session format

Theory -> Demo -> Lab, three labs. Runs on Colab CPU; a GPU only speeds up the sentence
encoder. About 60-90 minutes.
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-0
Change: retheme the story from the logistic-regression/one-hot framing to the chatbot/support
scenario; replace "learn dense vectors from Yelp" with "load pretrained vectors"; rewrite the
objectives list to match the renewed (no-training) scope.

## Cell 1 - Markdown: Section 0 environment setup header

```markdown
## Section 0. Environment Setup

We install a small, pinned stack, import libraries, fix the random seed, and detect the GPU.

Two pins matter and are easy to get wrong:

- `numpy<2`. Colab now ships numpy 2.x, but `gensim` 4.3 requires numpy < 2. If numpy 2 is
  already imported, you may need to restart the runtime after the install cell (Runtime ->
  Restart session) and run again.
- `scipy<1.13`. gensim 4.3 imports `scipy.linalg.triu`, which scipy removed in 1.13. Without
  this pin you get `ImportError: cannot import name 'triu' from 'scipy.linalg'`.
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-1
Change: replace the reproducibility blurb with the two real install gotchas (numpy<2 restart,
scipy<1.13) surfaced by research.

## Cell 2 - Code: pinned package install

```python
# Install required packages (run this first on Colab).
# Pins explained in the markdown above:
#   numpy<2          gensim 4.3 needs it; Colab ships numpy 2.x by default.
#   scipy<1.13       gensim 4.3 imports scipy.linalg.triu, removed in scipy 1.13.
#   gensim==4.3.3    stable; brings gensim.downloader for pretrained vectors.
#   sentence-transformers pinned to a 3.x line that works with transformers 4.57.x.
#   transformers==4.57.* (NOT 5.x: 5.x forces numpy 2 and drops pipelines we use later).
!pip install -q "numpy<2" "scipy<1.13" "gensim==4.3.3" \
    "sentence-transformers==3.4.1" "transformers==4.57.1" \
    "datasets>=2.19,<3" "scikit-learn>=1.3" "matplotlib>=3.7" "pandas>=2.0"

# If you see a numpy-2 ImportError after this cell, restart the runtime and re-run from here.
print("Install done. If numpy was already 2.x, restart the runtime now and re-run.")
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-2
Change: replace the loose `!pip install torch gensim textblob ...` with a fully pinned line
(numpy<2, scipy<1.13, gensim==4.3.3, sentence-transformers==3.4.1, transformers==4.57.1).
Drop textblob corpora download (no tokenizer training here). Merge in the install intent of
old B idx2.

## Cell 3 - Code: imports, seed, device

```python
# Core imports - grouped: data -> embeddings -> similarity -> viz.
import warnings
import numpy as np
import pandas as pd

import torch
import gensim.downloader as api               # pretrained word vectors, no training
from sentence_transformers import SentenceTransformer, util

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Versions - useful when a student reports a bug.
print("numpy   :", np.__version__)            # expect 1.26.x (NOT 2.x)
import gensim, transformers
print("gensim  :", gensim.__version__)        # expect 4.3.3
print("transformers:", transformers.__version__)

# Reproducibility: one seed for every RNG we touch.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device: GPU if present (only the sentence encoder benefits). Same idiom as B4.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-3
Change: drop the CBOW hyperparameters (CORPUS_SIZE, VOCAB_MIN_FREQ, etc.) and TextBlob/torch.nn
imports; add `gensim.downloader as api` and `sentence_transformers`; keep SEED=42 and the B4
`device` idiom. Merge the seeding block from old B idx3/idx4.

## Cell 4 - Markdown: What are we building today

```markdown
## What we are building today

Three concrete artifacts, all from pretrained models:

1. **A word-vector explorer.** Load 400,000 pretrained word vectors and ask: what is near
   "king"? what is "paris - france + italy"? Plot a handful in 2D.
2. **A sentence encoder.** Turn whole sentences into vectors that respect meaning (not just
   word overlap), and measure their similarity against the STS-B benchmark.
3. **A semantic search engine.** Encode a corpus once, then answer free-text queries by
   nearest-neighbor in vector space. This is the retrieval step a RAG chatbot runs before it
   answers.

Everything is geometry. By the end, "embedding" should feel like "a point in space," and
"similar" should feel like "a small angle between two points."
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 0)
Change: rewrite the scenario into a three-artifact roadmap themed on the chatbot/support arc;
drop the Notebook 5/6 word-embedding backreference (those notebooks no longer exist).

## Cell 5 - Code: load pretrained word vectors (preview)

```python
# Load pretrained GloVe vectors via gensim's downloader. First call downloads ~130MB and
# caches it under ~/gensim-data, so re-runs in the same session are instant.
# glove-wiki-gigaword-100: 400,000 words, 100 dimensions, trained on Wikipedia + Gigaword.
print("Loading pretrained word vectors (first run downloads ~130MB)...")
wv = api.load("glove-wiki-gigaword-100")      # returns a gensim KeyedVectors object

print("Vocabulary size :", len(wv))                       # ~400,000
print("Vector size     :", wv.vector_size)                # 100
print("Vector for 'coffee' (first 5 dims):", wv["coffee"][:5])
```

Provenance: FROM-NEW

## Cell 6 - Markdown: Section 1, the one-hot problem

```markdown
## Section 1. Why Embeddings?

### The one-hot problem

The simplest way to represent a word is a one-hot vector: a column index in a vocabulary.

```python
king  = [1, 0, 0, 0, ...]
queen = [0, 1, 0, 0, ...]
dog   = [0, 0, 1, 0, ...]
```

The cosine similarity between any two distinct one-hot vectors is exactly 0. The
representation claims `king` is as unrelated to `queen` as it is to `dog`. That is a hard
ceiling on what any downstream model can recover. Dense embeddings lift the ceiling: similar
words get similar vectors. The next cell shows the failure numerically.
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-4
Change: condense; drop the "Module 1 logistic regression" backreference. Keep the one-hot
illustration and the cosine-is-zero claim.

## Cell 7 - Code: one-hot orthogonality demo

```python
# Tiny demo: one-hot vectors are mutually orthogonal, so every pair has cosine 0.
vocab_toy = ["king", "queen", "dog", "car"]
one_hot = np.eye(len(vocab_toy))              # 4x4 identity = 4 one-hot rows

sims = cosine_similarity(one_hot)             # pairwise cosine over the rows
print("Cosine similarity of one-hot vectors:")
print(pd.DataFrame(sims, index=vocab_toy, columns=vocab_toy))
print("\nEvery off-diagonal entry is 0: king = queen = dog = car, as far as the model knows.")
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-5
Change: keep essentially verbatim; inline the DataFrame construction for brevity.

## Cell 8 - Markdown: querying the pretrained space

```markdown
### Querying a pretrained space

We already loaded `wv` in Section 0: 400,000 words, each a 100-dimensional dense vector,
learned by GloVe on Wikipedia. Unlike one-hot, these vectors put related words near each
other. gensim's `KeyedVectors` gives us the query API:

```python
wv.most_similar("coffee", topn=5)     # 5 nearest neighbors by cosine
wv.similarity("coffee", "tea")        # cosine between two words
"coffee" in wv.key_to_index           # membership test, avoids KeyError
```

Two things to remember:

- The vocabulary is lowercase. Query `"paris"`, not `"Paris"`.
- A word not in the vocabulary raises `KeyError`. Always guard with `in wv.key_to_index`
  (equivalently `in wv`) before you index.
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-29
Change: replace "we extract embedding.weight from the trained model and wrap it in
KeyedVectors" with "we already loaded pretrained `wv`"; fold in the probe framing from old
A cell-31; add the lowercase + KeyError guard surfaced by research.

## Cell 9 - Code: most_similar and similarity demo

```python
# Demo: nearest neighbors and pairwise cosine on the pretrained vectors.
for anchor in ["coffee", "king", "python"]:
    print(f"\nTop-5 neighbors of '{anchor}':")
    for word, score in wv.most_similar(anchor, topn=5):    # returns (word, cosine) pairs
        print(f"  {word:15s}  cos={score:.3f}")

# Direct pairwise cosine between two words.
print("\nsimilarity(coffee, tea)  =", round(wv.similarity("coffee", "tea"), 3))
print("similarity(coffee, car)  =", round(wv.similarity("coffee", "car"), 3))
# Note: 'python' pulls BOTH the language and the snake. Static vectors have ONE vector per
# word, so the two senses are blended. That polysemy limit is why Part C reaches for
# contextual transformers.
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-32
Change: convert the old Lab-4 starter into a worked DEMO on pretrained `wv` (the lab moves to
cell 12); add the `python` polysemy observation that seeds the C9 motivation.

## Cell 10 - Markdown: the analogy

```markdown
### Vector arithmetic: king - man + woman ~ queen

The poster-child result of word embeddings: semantic relationships show up as directions in
the space. "Man -> woman" is roughly the same direction as "king -> queen", so

```python
wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
```

returns `queen` near the top. gensim computes `king - man + woman` and finds the nearest
vector, automatically EXCLUDING the three input words (otherwise it would just return `king`).

### Honesty check

This works on Wikipedia-scale GloVe, but it is more fragile than the hype suggests. Many
"analogies" only work because the input words are excluded. And the same geometry encodes
social bias: `doctor - man + woman` can return `nurse`, because the training text carried that
stereotype. Embeddings learn the corpus, biases and all. We will look at one such case below.
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-33
Change: remove the "this will fail on our 5K Yelp CBOW" disclaimer (we now use full GloVe, so
it works); add the research-backed honesty notes (input-word exclusion, encoded bias) and the
forward pointer to bias in the demo.

## Cell 11 - Code: analogy demo with bias caveat

```python
# Demo: vector arithmetic. most_similar(positive=..., negative=...) does king - man + woman.
def analogy(a, b, c, topn=3):
    """a is to b as c is to ?  ->  b - a + c.  Returns top matches, input words excluded."""
    return wv.most_similar(positive=[b, c], negative=[a], topn=topn)

print("man : king  ::  woman : ?")
print(analogy("man", "king", "woman"))          # expect queen near the top

print("\nfrance : paris  ::  italy : ?")
print(analogy("france", "paris", "italy"))      # expect rome

print("\nwalk : walked  ::  run : ?")
print(analogy("walk", "walked", "run"))         # expect ran (morphology direction)

# The honesty check: the geometry also encodes bias from the training corpus.
print("\nman : doctor  ::  woman : ?  (watch for stereotype)")
print(analogy("man", "doctor", "woman"))
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-34
Change: rewrite as a reusable `analogy()` helper; swap the "expect it to fail on Yelp" pairs
for real working GloVe analogies (capital, morphology) plus an explicit bias probe.

## Cell 12 - Code: Lab 1 starter (word vector probing)

```python
# ============================================================
# Lab 1. Probe the word-vector space (15 min)
# ============================================================
# You have `wv` (pretrained GloVe, 100d) and the `analogy()` helper from the demo.
#
# Task A: For each anchor word, print its top-5 nearest neighbors. GUARD against words that
#         are not in the vocabulary so you never hit a KeyError.
# Task B: Complete one analogy of your own choosing using the analogy() helper.
#
# Reminder: the vocabulary is lowercase; membership test is `word in wv.key_to_index`.

anchors = ["doctor", "guitar", "pizza", "rocket", "winter"]

# Task A: nearest neighbors with an OOV guard.
for anchor in anchors:
    # 1. Skip the anchor if it is not in the vocabulary (print a note and continue).
    in_vocab = None  # YOUR CODE: a boolean, True if anchor is queryable
    if not in_vocab:
        print(f"'{anchor}' not in vocab, skipping")
        continue
    # 2. Get the top-5 neighbors for this anchor.
    neighbors = None  # YOUR CODE: list of (word, score) pairs
    print(f"\nTop-5 neighbors of '{anchor}':")
    for word, score in neighbors:
        print(f"  {word:15s}  cos={score:.3f}")

# Task B: one analogy of your choice. Pick a, b, c so that "a is to b as c is to ?".
my_analogy = None  # YOUR CODE: call analogy(a, b, c)
print("\nMy analogy result:", my_analogy)
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-32
Change: turn the old probe into a real exercise on pretrained `wv` with two tasks; placeholders
hide the membership test, the most_similar call, and the analogy call without naming the method
that solves them in the same line.

## Cell 13 - Code: Lab 1 verification

```python
# ---- Lab 1 verification (provided) ----
assert in_vocab is not None, "Task A step 1: set in_vocab to a membership test."
assert isinstance(neighbors, list) and len(neighbors) == 5, \
    "Task A step 2: neighbors should be a list of 5 (word, score) pairs."
assert all(isinstance(w, str) and isinstance(s, float) for w, s in neighbors), \
    "neighbors entries should be (str, float)."
assert my_analogy is not None and len(my_analogy) >= 1, \
    "Task B: my_analogy should be the result of analogy(a, b, c)."
print("Lab 1 passed. You can query a 400k-word space by meaning, with an OOV guard.")
```

Provenance: FROM-NEW

## Cell 14 - Code: PCA 2D projection demo

```python
# Demo: project a chosen set of words to 2D with PCA so we can eyeball the geometry.
# PCA (not t-SNE) on purpose: it is deterministic, fast, and needs no perplexity tuning,
# which makes it the right tool for a small, hand-picked word set.
words_2d = [
    "king", "queen", "man", "woman",          # gender / royalty
    "paris", "london", "rome", "berlin",      # capitals
    "coffee", "tea", "water", "juice",        # drinks
]
vectors_2d = np.stack([wv[w] for w in words_2d])          # (12, 100)

coords = PCA(n_components=2, random_state=SEED).fit_transform(vectors_2d)   # (12, 2)

plt.figure(figsize=(9, 6))
plt.scatter(coords[:, 0], coords[:, 1], s=40)
for (x, y), w in zip(coords, words_2d):
    plt.annotate(w, (x, y), fontsize=11, xytext=(4, 4), textcoords="offset points")
plt.title("PCA of GloVe vectors (100d -> 2d). Watch the drinks / capitals / royalty groups.")
plt.grid(True, alpha=0.3)
plt.show()
# 2D throws away 98 dimensions, so do not over-read it. But related words should still cluster.
```

Provenance: FROM-OLD 2-Text-Similarity/5-CBOW_Word_Embeddings.ipynb cell cell-36
Change: project a curated, interpretable word set (royalty/capitals/drinks) instead of the 50
most frequent Yelp tokens; add the explicit PCA-over-t-SNE rationale from research; fold the
old PCA markdown (A cell-35) caveat into the trailing comment.

## Cell 15 - Markdown: Section 2, why sentence embeddings

```markdown
## Section 2. From Words to Sentences

Almost every real product works at the sentence or document level: FAQ matching, duplicate
detection, semantic search, chatbot intent, RAG. So we need one vector per SENTENCE, not per
word.

### The naive baseline: average the word vectors

The simplest sentence vector is the mean of its word vectors:

```python
def mean_vector(sentence):
    vecs = [wv[w] for w in sentence.lower().split() if w in wv]
    return np.mean(vecs, axis=0)
```

It is a real baseline, and sometimes a strong one. But it has two fatal flaws we can see in
one experiment: it ignores word ORDER ("dog bites man" == "man bites dog"), and it barely
distinguishes ANTONYMS ("I love it" vs "I hate it" share every word but one). The next cell
makes both failures concrete.
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 5)
Change: condense; reuse the already-loaded `wv` instead of downloading a separate GloVe .txt
file; keep the help-center / order / antonym motivation.

## Cell 16 - Code: mean-vector baseline and its failures

```python
# Demo: build the mean-of-word-vectors baseline using wv, then expose its two failure modes.
def mean_vector(sentence, dim=100):
    """Mean of in-vocab word vectors. Returns zeros if no token is known."""
    vecs = [wv[w] for w in sentence.lower().split() if w in wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32)

def cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

pairs = [
    ("dog bites man",          "man bites dog"),           # word order
    ("i love it",              "i hate it"),               # antonym
    ("the food was excellent", "the meal was outstanding"),# true paraphrase
    ("the food was excellent", "my car broke down"),       # unrelated
]
print(f"{'Sentence A':<26}{'Sentence B':<26}{'mean-vec cos':>14}")
print("-" * 66)
for a, b in pairs:
    print(f"{a:<26}{b:<26}{cos(mean_vector(a), mean_vector(b)):>14.3f}")
# Expect: order pair ~1.00 (identical), antonym pair ~0.95 (no discrimination). Broken.
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 8)
Change: reuse `wv` and a local `mean_vector` (drop the separate GloVe dict from old idx7);
trim the pair list to the four most instructive cases; keep the `cos` helper.

## Cell 17 - Markdown: SBERT theory

```markdown
## Sentence-BERT (SBERT)

[Reimers and Gurevych, EMNLP 2019](https://arxiv.org/abs/1908.10084) introduced Sentence-BERT:
take a pretrained BERT, mean-pool its token vectors, then fine-tune the whole stack on labeled
sentence pairs (NLI, STS) with a siamese setup. The result is an encoder where ONE forward
pass turns a sentence into a fixed-size vector, and cosine similarity in that space tracks
SEMANTIC similarity, with word order and composition respected.

We use `all-MiniLM-L6-v2`: 6 layers, 384-dimensional output, ~22M parameters, around 14,000
sentences/second on a CPU. This is the SAME model you met in Part A as the embedder, so the
text-to-vector tool you called as a black box is now something you understand.

> Caveat: SBERT is paraphrase-aware, not polarity-aware. It captures topical similarity very
> well; for sentiment you still want a sentiment-tuned classifier (that is Part C).
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 10)
Change: condense the architecture diagram to prose; add the explicit "same model as A1" tie-in
and the model-size/speed numbers from research; keep the polarity caveat.

## Cell 18 - Code: load embedder, encode, compare to baseline

```python
# Load the SAME sentence encoder used in Part A. First call downloads ~80MB and caches it.
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
print("Embedding dimension:", embedder.get_sentence_embedding_dimension())   # 384

# Encode all sentences in one batched call: list in, ndarray out (one row per sentence).
sents_A = [a for a, _ in pairs]
sents_B = [b for _, b in pairs]
emb_A = embedder.encode(sents_A, convert_to_numpy=True)   # (4, 384)
emb_B = embedder.encode(sents_B, convert_to_numpy=True)

# Re-run the four pairs: mean-vec baseline vs SBERT, side by side.
print(f"{'Sentence A':<26}{'Sentence B':<26}{'mean-vec':>10}{'SBERT':>10}")
print("-" * 72)
for i, (a, b) in enumerate(pairs):
    base = cos(mean_vector(a), mean_vector(b))
    sbert = float(util.cos_sim(emb_A[i], emb_B[i]))       # util.cos_sim handles L2 norm
    print(f"{a:<26}{b:<26}{base:>10.3f}{sbert:>10.3f}")
# Expect the antonym pair to drop sharply under SBERT, and the true paraphrase to stay high.
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 11)
Change: merge load (old idx11) + first-encode (idx12) + comparison (idx13) into one cell to
hold the budget; name the model `embedder` (matches A1); use `convert_to_numpy=True` and
`util.cos_sim` per research.

## Cell 19 - Markdown: Lab 2 instructions (STS-B)

```markdown
### Lab 2. Score real sentence similarity against STS-B (15 min)

STS-B (Semantic Textual Similarity Benchmark) is a standard set of sentence pairs, each with a
human similarity score from 0 (unrelated) to 5 (equivalent). We will encode the pairs, compute
cosine similarity, and check whether our cosines RANK the pairs the way humans did.

The right metric is the SPEARMAN rank correlation between SBERT cosine and the gold score:
Spearman cares about ordering, not absolute values, which is exactly what a similarity model
should get right.

Steps:

1. Load STS-B: `load_dataset("glue", "stsb", split="validation")`. Take the first 200 rows.
   Fields are `sentence1`, `sentence2`, `label` (the 0-5 gold score).
2. Encode `sentence1` and `sentence2` with `embedder`.
3. For each pair, compute SBERT cosine into a list `preds`.
4. Compute the Spearman correlation between `preds` and the gold `labels`.

A well-behaved encoder lands around 0.8 Spearman on STS-B. Higher is better.
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 16)
Change: replace the Yelp polarity-heatmap lab with a quantitative STS-B Spearman lab (the
manifest's stated dataset for B5); spell out the four steps and the target metric from research.

## Cell 20 - Code: Lab 2 starter + verification (STS-B Spearman)

```python
# ============================================================
# Lab 2. Sentence similarity vs STS-B gold scores (15 min)
# ============================================================
from datasets import load_dataset
from scipy.stats import spearmanr

# Provided: load 200 validation pairs from STS-B.
stsb = load_dataset("glue", "stsb", split="validation").select(range(200))
s1 = stsb["sentence1"]
s2 = stsb["sentence2"]
gold = stsb["label"]            # human scores in [0, 5]

# 1. Encode each sentence column into a numpy array. The embedder's encode method
#    takes a list of strings and accepts convert_to_numpy=True.
emb1 = None  # YOUR CODE  -> array of shape (200, 384)
emb2 = None  # YOUR CODE  -> array of shape (200, 384)

# 2. Turn the two arrays into a list `preds` of 200 cosine similarities, one per row pair.
#    You already used a cosine helper from `util` in the previous cell; it returns a small
#    tensor, so pull the scalar out of it before appending. Decide the loop yourself.
preds = None  # YOUR CODE  -> a Python list of length 200

# 3. Measure how well `preds` RANKS the pairs versus the human `gold` scores. The Spearman
#    function you imported returns an object; read its docs to find the field that holds
#    the correlation coefficient.
rho = None  # YOUR CODE  -> a single float

# ---- verification (provided) ----
assert emb1 is not None and emb1.shape == (200, 384), "Step 1: emb1 should be (200, 384)."
assert isinstance(preds, list) and len(preds) == 200, "Step 2: preds should be length 200."
assert rho is not None and 0.6 < rho < 1.0, \
    f"Step 3: Spearman should be ~0.8 for a good encoder; got {rho}."
print(f"Lab 2 passed. SBERT cosine ranks sentence pairs like humans (Spearman={rho:.3f}).")
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 18)
Change: rewrite the heatmap starter into an STS-B encode + Spearman starter; placeholders hide
the encode call, the cosine loop, and the correlation call. Hints describe the goal and point
back to helpers seen earlier (the util cosine helper, the imported Spearman function) WITHOUT
naming the method or attribute on the blank line, so no step is copy-pasteable; verification
asserts shape and a plausible Spearman band only.

## Cell 21 - Code: semantic search engine demo

```python
# Demo: a working semantic search engine. Encode a corpus ONCE (offline), then answer
# free-text queries by nearest-neighbor in vector space (online). This is the retrieval
# step a RAG chatbot runs before it answers.
corpus = [
    "Tracking a missing shipment",
    "How to reset your account password",
    "Refund policy for damaged items",
    "Updating your billing address",
    "Cancelling a subscription before renewal",
    "Why was my payment declined",
    "Changing the email on your account",
    "Estimated delivery times by region",
]
# Offline step: precompute corpus embeddings once. In production you cache these.
corpus_emb = embedder.encode(corpus, convert_to_tensor=True)     # (8, 384) on `device`

def semantic_search(query, top_k=3):
    """Encode the query, score against the cached corpus, return top-k (score, text)."""
    q_emb = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, corpus_emb, top_k=top_k)[0]  # list of {corpus_id, score}
    return [(h["score"], corpus[h["corpus_id"]]) for h in hits]

# Online step: a query with NO literal word overlap with the right answer.
for score, text in semantic_search("my package never arrived"):
    print(f"  [{score:.3f}] {text}")
# Top hit should be 'Tracking a missing shipment' though they share zero words. Keyword
# search would miss it; geometry finds it.
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 22)
Change: merge corpus build (idx20) + encode (idx21) + search function (idx22) into one compact
demo; swap the 2000-review Yelp corpus for a small support-KB corpus that matches the chatbot
story and runs instantly on CPU; use `util.semantic_search` (returns corpus_id/score) per
research; add the offline/online + caching framing.

## Cell 22 - Code: Lab 3 starter + verification (semantic search)

```python
# ============================================================
# Lab 3. Build your own semantic search (15 min)
# ============================================================
# You have `embedder` and the `corpus` / `corpus_emb` from the demo.
# Build a search that returns the single best match and its score for any query.
#
# 1. Encode the query into a tensor.
# 2. Score it against corpus_emb with util.semantic_search, asking for the top 1 hit.
# 3. Return (score, text) for that best hit.

def best_match(query):
    # 1. Encode the query the same way the demo encoded a single query (a tensor).
    q_emb = None    # YOUR CODE  -> a query embedding tensor
    # 2. Search the cached corpus for the single closest article. The demo used a `util`
    #    search helper that returns a list-of-lists of hit dicts; ask it for just the top hit.
    hits = None     # YOUR CODE  -> the search result for this query
    # 3. From `hits`, pull out the one best hit dict (keys: 'corpus_id', 'score').
    top = None      # YOUR CODE  -> the single best hit dict
    return (top["score"], corpus[top["corpus_id"]])

# ---- verification (provided) ----
score, text = best_match("I forgot my login details")
assert isinstance(score, float), "best_match should return a float score first."
assert text == "How to reset your account password", \
    f"Expected the password-reset article, got: {text!r}"
print(f"Lab 3 passed. Best match: [{score:.3f}] {text}")
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 24)
Change: replace the stars-filtered Yelp search starter with a simpler best-match starter on the
support corpus; placeholders hide the encode + search calls. Hints reference "the util search
helper from the demo" by role, never naming the method, its top_k argument, or the [0] index on
the blank line, so no step is copy-pasteable; verification asserts the exact expected article.

## Cell 23 - Markdown: homework + wrap-up (bridge to B7)

```markdown
## Homework and Wrap-up

### What you can now do

- Query a 400,000-word space by meaning: neighbors, cosine, analogies (with an honest view of
  fragility and bias).
- Encode sentences with a pretrained transformer and measure quality against STS-B (Spearman).
- Ship a semantic search engine in under 50 lines that beats keyword matching.

### Homework Extension (async, deeper, builds B7's input)

This is the bridge to the B7 STOPPER. There, you train an MLP whose INPUT is a document vector.
Here you build that exact feature.

1. **`doc_vector(text)`**: mean-pool the word2vec vectors of a text into one fixed-size 100-d
   vector, skipping out-of-vocab tokens (reuse the `mean_vector` idea). This is the standard
   "average word2vec" document feature.
2. Load ~500 short labeled texts (for example `load_dataset("glue", "sst2", split="train")`,
   fields `sentence` and `label`). Build a feature matrix `X` of shape `(500, 100)` by applying
   `doc_vector` to each, and a label array `y`.
3. Sanity-check: fit a plain `LogisticRegression` on `X, y` and report accuracy. That number is
   the BASELINE the B7 MLP must beat. Save `X` and `y` (`np.save`) so B7 can load them.

### Stretch options (pick one)

- **Asymmetric search**: re-run Lab 3 with `multi-qa-MiniLM-L6-cos-v1` (built for short query /
  long answer) and compare hits to `all-MiniLM-L6-v2` on a query like "what is the refund
  window". Symmetric vs asymmetric models retrieve differently.
- **Scaling note**: brute-force cosine is fine up to ~1M vectors. Read the FAISS index docs and
  describe (in a markdown cell) when you would switch to an IVF or HNSW index and why.
- **Clustering**: run `KMeans(n_clusters=4)` on `corpus_emb` and print each cluster's members.
  Does unsupervised structure match the article topics?

### One-line bridge to B6

Text is now a fixed-size vector. Next we need a model that maps those vectors to a label.
B6 builds the PyTorch neural-net machinery (`nn.Module`, loss, optimizer, DataLoader); B7
points that machinery at these embeddings and beats the baseline you just measured.

### Resources

- gensim KeyedVectors: https://radimrehurek.com/gensim/models/keyedvectors.html
- Sentence Transformers semantic search: https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
- SBERT paper (Reimers and Gurevych, 2019): https://arxiv.org/abs/1908.10084
```

Provenance: FROM-OLD 2-Text-Similarity/7-Sentence_Transformers_Similarity.ipynb cell (idx 36)
Change: replace the "pick another dataset and redo search" homework with the doc_vector ->
LogisticRegression baseline that becomes B7's explicit input and bar-to-beat; fold the dropped
clustering and asymmetric-model material into clearly-labelled stretch options; add the B6/B7
bridge and keep the takeaways table as prose.

# VERIFICATION CHECKLIST

- [ ] 24 cells total (0-23), within the 20-25 budget.
- [ ] numpy<2 AND scipy<1.13 AND gensim==4.3.3 pinned in the install cell (cell 2).
- [ ] transformers pinned to 4.57.x (not 5.x); sentence-transformers pinned to a compatible 3.x.
- [ ] No from-scratch CBOW training anywhere (manifest requirement); only pretrained `wv`.
- [ ] `device` and `SEED=42` carried from B4 unchanged; no variable collision with B4.
- [ ] `embedder = SentenceTransformer('all-MiniLM-L6-v2')` matches the A1 embedder id verbatim.
- [ ] Every concept has a demo cell AND a lab cell; never more than 3 markdown cells in a row.
- [ ] Lab 1, Lab 2, Lab 3 each have a `None # YOUR CODE` starter plus a provided verification.
- [ ] No `# YOUR CODE` comment names the exact method that solves it in the same line.
- [ ] OOV guard (`in wv.key_to_index`) taught before any `wv[...]` indexing in a lab.
- [ ] STS-B loaded via `load_dataset("glue", "stsb")`; fields sentence1/sentence2/label verified.
- [ ] `util.cos_sim` and `util.semantic_search` used (not hand-rolled) per current sbert API.
- [ ] Homework builds `doc_vector()` + saves `X, y` as the explicit B7 input.
- [ ] Bias caveat present in the analogy section; PCA-over-t-SNE rationale present.
- [ ] Both notebooks run top-to-bottom on Colab CPU; file size reasonable (< 500KB).
- [ ] Plain ASCII only: no em/en dashes, no Unicode multiply sign, no emoji.

# RESEARCH VALIDATED (June 2026)

- gensim downloader is the modern way to fetch pretrained vectors; `api.load("glove-wiki-gigaword-100")`
  returns a KeyedVectors of 400k words, 100d.
  https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html
- gensim 4.x KeyedVectors uses `key_to_index` / `index_to_key` (the `vocab` attribute was
  removed in 4.0); membership test `word in wv.key_to_index` avoids KeyError.
  https://github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4
- `most_similar(positive=[b, c], negative=[a])` computes b - a + c and EXCLUDES the input words;
  the king-man+woman analogy is fragile and encodes corpus bias.
  https://p.migdal.pl/blog/2017/01/king-man-woman-queen-why/
  https://arxiv.org/pdf/1607.06520
- gensim 4.3 breaks on scipy 1.13 (removed `scipy.linalg.triu`); pin `scipy<1.13`.
  https://github.com/piskvorky/gensim/issues/3525
- gensim 4.3.3 requires numpy<2, consistent with the course-wide `numpy<2` pin; Colab ships
  numpy 2.0.2 so a runtime restart may be needed after the install.
  https://pypi.org/project/gensim/
  https://github.com/googlecolab/colabtools/issues/5115
- all-MiniLM-L6-v2: 6 layers, 384-dim, ~22M params, ~14k sentences/sec on CPU; symmetric model.
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  https://sbert.net/docs/sentence_transformer/usage/efficiency.html
- `encode(convert_to_numpy=True)` default; `convert_to_tensor=True` for GPU; `util.cos_sim`
  L2-normalizes; `util.semantic_search` returns list-of-{corpus_id, score}.
  https://sbert.net/docs/package_reference/util.html
- Mean-of-word-vectors is a weak sentence baseline (order + antonym failures); SBERT fixes this
  via siamese fine-tuning on NLI/STS.
  https://arxiv.org/pdf/1908.10084
  https://machinelearningmastery.com/why-and-when-to-use-sentence-embeddings-over-word-embeddings/
- STS-B loads via `load_dataset("glue", "stsb")` with fields sentence1, sentence2, label (0-5
  float); evaluate sentence similarity with SPEARMAN correlation against gold scores.
  https://huggingface.co/datasets/nyu-mll/glue
  https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html
- PCA is the right 2D projection for a small word set: deterministic, fast, no perplexity tuning
  (t-SNE is stochastic and perplexity-sensitive on small sets).
  https://machinelearningmastery.com/choosing-between-pca-and-t-sne-for-visualization/
- Mean-pooled word2vec is a standard fixed-size document feature for a downstream classifier;
  this is the exact B7 MLP input.
  https://github.com/sdimi/average-word2vec
- Brute-force cosine is fine up to ~1M vectors; FAISS IVF/HNSW beyond that.
  https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- Symmetric (all-MiniLM-L6-v2) vs asymmetric (multi-qa-MiniLM-L6-cos-v1) models retrieve
  differently; choose by query/document length.
  https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
- Embedding-based semantic search is the retrieval foundation for RAG chatbots; precompute and
  cache corpus embeddings offline, encode only the query online.
  https://www.appetenza.com/understanding-embeddings-in-depth-the-foundation-of-semantic-search-in-rag-systems
  https://www.comet.com/site/blog/retrieval-part-2-text-embeddings/
- Static word2vec gives one vector per word (polysemy gap); contextual transformers
  (DistilBERT, C9) resolve it; fine-tuning beats frozen feature-extraction by ~8%.
  https://medium.com/@alexbuzunov/how-the-representation-era-connected-word2vec-to-transformers-768665eccf9d
  https://codefinity.com/blog/Fine-Tuning-vs-Feature-Extraction-in-Transfer-Learning

---

Plan written to `plans/refactor/notebooks/B5-word2vec-sentence-embeddings.md`.

Next step: run `/build-notebook word2vec-sentence-embeddings colab` to generate the exercise +
solution notebooks from this plan, 5 cells at a time with approval between batches.
