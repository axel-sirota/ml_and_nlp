# B8 - Capstone B: Own the Pipeline (IMDB Sentiment) - Cell-by-Cell Plan

## Status

RENEWED. Source notebook: `3-Text-Classification/10-Capstone_2.ipynb` (IMDB sentiment, old MLP-vs-BERT choose-your-own capstone). This plan keeps the IMDB scenario and the "do it solo, beat a baseline" capstone spirit, but RETARGETS the whole pipeline onto the Part B skills the student actually built: gensim PRETRAINED word vectors (B5), mean-pool document vectors (B5/B7 `doc_vector`), and a hand-written PyTorch `nn.Module` MLP trained with the B6 loop (B6/B7). The old notebook trained Word2Vec from scratch, built an embedding matrix with padding, and offered a parallel "fine-tune DistilBERT" Path B. Both are dropped here: from-scratch Word2Vec is replaced by reusing the B5 pretrained `wv`, and the BERT path moves to C9 where fine-tuning is taught properly. B8 stays MLP-focused so it is a clean, self-contained proof of Part B.

## Context

What students arrive with (exact names / APIs they already used):
- From B5: `wv` (a gensim `KeyedVectors`, 100-dim, from `gensim.downloader.api.load("glove-wiki-gigaword-100")`), the OOV guard `word in wv.key_to_index`, and the mean-pool idea behind `mean_vector(sentence)` / the homework `doc_vector(text)` that produced a 100-dim feature `X` and label `y`. Also `embedder` (a `SentenceTransformer("all-MiniLM-L6-v2")`, 384-dim) for the homework.
- From B6: `nn.Module` subclassing, `nn.Linear`, `nn.ReLU`, `nn.CrossEntropyLoss`, `torch.optim.Adam`, `TensorDataset`, `DataLoader`, and the train-one-epoch / eval-one-epoch loop shape.
- From B7 (THE STOPPER): the full "embeddings as features -> average into a document vector -> feed an MLP -> classify, beat a baseline" pattern, applied with the instructor driving.
- From B4: tensors, dtypes, and autograd (`.backward()` is the engine).
- The standing setup idiom: `SEED = 42`, `random.seed/np.random.seed/torch.manual_seed`, `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.

Key insight they leave with: they can now OWN the entire "text -> averaged-embedding features -> MLP -> evaluate -> beat the baseline" pipeline solo on a brand-new dataset, AND they can name exactly why this bag-of-words pipeline tops out (it loses word order and negation: "not good" looks positive), which is the precise reason Part C reaches for a fine-tuned contextual transformer.

## Chatbot Through-Line

The course end goal is a fine-tuned transformer (DistilBERT) in a simple Gradio chatbot. B8 is the last rung of Part B and the launch ramp into Part C. It does three through-line jobs:
1. It proves the student can build a trained text classifier end to end without the instructor - the muscle they will reuse to fine-tune DistilBERT in C9 (the same `.backward()` engine from B4, now driving a real model).
2. The in-notebook stretch runs the pretrained sentiment `pipeline` (the A2 zero-shot tool) on the SAME IMDB reviews and measures the gap between the hand-built averaged-embedding MLP (~78-82%) and the pretrained transformer (~90%+ on IMDB). That measured gap is the motivation to "earn fine-tuning" in Part C.
3. The closing bridge is explicit: averaging word vectors throws away order and negation; Part C swaps the bag-of-words feature for a contextual transformer and FINE-TUNES it (not just zero-shot) into the chatbot. One-line bridge to C9: "you measured the ceiling of averaged features; C9 breaks it by fine-tuning DistilBERT, then drops it into a Gradio chatbot."

## STAR Story

Situation: The customer-support-platform team from Part A has been leaning on your zero-shot router, but leadership wants proof the team can ship a TRAINED model, not just call pretrained ones. They hand you a new, unfamiliar corpus - IMDB movie-review sentiment - and say: "Show us you can build a sentiment classifier from raw text, end to end, and that it beats a sensible baseline. No hand-holding."

Task: Independently take raw review strings to a trained classifier: turn each review into a single feature vector using the pretrained word vectors you already loaded, fit a quick baseline, build your own MLP, train it, and prove on a held-out test set that the MLP beats the baseline.

Action: You reuse the exact Part B toolkit - `wv` for word vectors, a mean-pool `doc_vector` to collapse each review into one 100-dim vector, a `LogisticRegression` baseline to set the bar, a small `nn.Module` MLP trained with the B6 loop, and a clean train/val/test split with everything seeded.

Result: Your MLP beats the logistic-regression baseline on a held-out IMDB test set. Then you run the pretrained transformer on the same reviews and SEE the remaining gap - which tells the team, honestly, when a cheap CPU model is enough and when to spend on fine-tuning. That gap is your pitch for Part C.

## Deliverables

- Exercise notebook: `B-How-It-Works/8-Capstone_B.ipynb`
- Solution notebook: `Solutions/B-How-It-Works/8-Capstone_B.ipynb`

## Session Timing (~60-90 min)

| Segment | Cells | Time |
|---------|-------|------|
| Header, setup, install, imports | 0-4 | 10 min |
| Scenario + IMDB load/preview + subsample | 5-7 | 10 min |
| Concept recap + `doc_vector` demo (toy) | 8-9 | 8 min |
| Lab 1: build features X/y for train/val/test | 10-11 | 12 min |
| Baseline: LogisticRegression bar to beat | 12-13 | 10 min |
| MLP demo (toy) + Lab 2: define MLPClassifier | 14-16 | 12 min |
| Train loop demo + Lab 3: train the MLP | 17-19 | 15 min |
| Lab 4: evaluate on test, beat baseline, confusion matrix | 20-21 | 10 min |
| Stretch: zero-shot transformer on same reviews (bridge to C) | 22 | 8 min |
| Homework: SBERT 384-d features, re-fit MLP | 23 | async |
| Wrap-up + bridge to C9 | 24 | 5 min |

## Cell Migration Map

Old source: `3-Text-Classification/10-Capstone_2.ipynb` (35 cells, indices 0-34; cells have no ids, referenced by index). Action key: keep / edit / drop / new.

| Old cell (index) | Old content | New cell | Action |
|------------------|-------------|----------|--------|
| 0 | Title "Capstone 2: IMDB (MLP vs BERT)" | 0 | edit (retitle Capstone B, Part B framing, drop BERT-vs framing) |
| 1 | Section 0 setup header | 1 | edit (keep header, reword) |
| 2 | pip install (torch/transformers>=4.40/datasets/gensim loose) | 2 | edit (hard-pin numpy<2, scipy<1.13, gensim==4.3.3, transformers==4.57.1, add Colab restart note) |
| 3 | shared imports + seed + device | 3 | edit (keep seed/device idiom, add gensim.downloader, sklearn LogisticRegression; drop Word2Vec-from-scratch import) |
| 4 | load_dataset('imdb') + preview | 5 | edit (keep load + preview, move under scenario) |
| 0 (story) | - | 4 | new (STAR scenario markdown) |
| 4 (extend) | - | 6 | new (subsample to balanced ~2000/2000 + held-out test, stratified) |
| 5,6 | "pick a path A/B" + my choice | - | drop (no path choice; B8 is MLP-only) |
| 7 | Section 1 Path A header | 8 | edit (becomes "recap: embeddings as features") |
| 10,11 | train Word2Vec from scratch | - | drop (reuse pretrained `wv` from B5 instead) |
| 12,13,14,15 | vocab/pad/embedding-matrix | - | drop (replaced by mean-pool `doc_vector`) |
| 8,9 | split train/val | 9,10,11 | edit (toy doc_vector demo + Lab 1 features) |
| 16,17 | define MLP (embedding+pool model) | 14,15,16 | edit (toy MLP demo + Lab 2 define MLPClassifier on 100-d input) |
| 18,19 | training loop + early stopping | 17,18,19 | edit (train loop demo + Lab 3 train) |
| 20,21 | evaluate + latency | 12,13,20,21 | edit (baseline section + Lab 4 evaluate + confusion matrix) |
| 22-30 | Path B fine-tune DistilBERT | 22 | drop the Trainer path; replace with a small zero-shot pipeline comparison (bridge to C9) |
| 31,32,33 | engineering memo + confusion plot | 21,24 | edit (fold memo prompts into wrap-up; confusion matrix in Lab 4) |
| 34 | wrap-up | 24 | edit (retheme to Part B capstone + bridge to C9) |
| - | - | 23 | new (homework: SBERT 384-d features re-fit MLP) |

---

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~25 cells)

## Cell 0 - Markdown: Title and learning objectives

```markdown
# Capstone B: Own the Pipeline - IMDB Sentiment

**Part B - How It Works. Capstone.**
*Author: Axel Sirota, Data Trainers LLC.*

You have spent Part B opening the black box: tensors and autograd (B4), word and
sentence embeddings (B5), the PyTorch neural-net machinery (B6), and the MLP-on-
word2vec classifier we built together (B7). This capstone is where you prove it.
You will take a brand-new dataset - IMDB movie-review sentiment - and build a
trained sentiment classifier end to end, by yourself.

## What you will do
1. Load IMDB and carve out a clean train / validation / test split.
2. Turn each review into a single feature vector by AVERAGING its pretrained word
   vectors (the `doc_vector` pattern from B5/B7).
3. Fit a quick `LogisticRegression` baseline - the bar your model must beat.
4. Define your own `MLPClassifier(nn.Module)` and train it with the B6 loop.
5. Evaluate on the held-out test set and BEAT the baseline.
6. (Stretch) Run a pretrained transformer on the same reviews and measure the gap.

## Prerequisites
- B4 tensors + autograd, B5 embeddings (`wv`), B6 `nn.Module` + training loop,
  B7 the embeddings-as-features pattern.

## Session format
- ~60-90 min. Mostly hands-on: this is YOUR pipeline. Short demos on toy data,
  then you rebuild on IMDB.

## Runtime
- Colab CPU is fine (the whole pipeline runs on CPU in a few minutes). A GPU only
  helps the optional stretch transformer cell.
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 0
Change: retitle to "Capstone B: Own the Pipeline", reframe under Part B (B4-B7) instead of "Module 3 MLP vs BERT", drop the path-choice framing, add explicit objective list ending in the stretch/bridge.

## Cell 1 - Markdown: Section 0 environment setup header

```markdown
## Section 0. Environment Setup

Install the pinned stack, import everything, set the seed, and pick the device.
This is the same environment you used in B5 and B7, so nothing here should surprise
you. The pins matter: gensim needs `numpy<2` and `scipy<1.13`, and we hold
`transformers` at 4.57.1 (never 5.x).
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 1
Change: reword to reference B5/B7 continuity and call out the pin rationale.

## Cell 2 - Code: Package install (Colab, pinned)

```python
# Install the pinned course stack (run this first in Google Colab).
# Pins explained:
#   numpy<2      -> gensim 4.3 is not numpy-2 compatible
#   scipy<1.13   -> gensim 4.3 imports scipy.linalg.triu, removed in scipy 1.13
#   gensim==4.3.3-> brings gensim.downloader and the KeyedVectors API from B5
#   transformers==4.57.1 -> stable; do NOT let it resolve to 5.x (forces numpy 2)
!pip install -q "numpy<2" "scipy<1.13" "gensim==4.3.3" \
    "transformers==4.57.1" "datasets>=2.19,<3" "sentence-transformers==3.4.1" \
    "scikit-learn>=1.3" "matplotlib" "seaborn"

# IMPORTANT (Colab): Colab preinstalls numpy 2.x. After this cell finishes, use
# Runtime > Restart session ONCE, then run from the top. If you skip the restart
# you may hit "ImportError: cannot import name 'triu' from 'scipy.linalg'".
print("Install complete. If on Colab, restart the runtime now, then re-run from the top.")
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 2
Change: replace loose pins (torch>=2.1, transformers>=4.40, gensim>=4.3) with the hard course pins, add sentence-transformers (for homework), add the Colab restart note.

## Cell 3 - Code: Imports, seed, device

```python
# Shared imports.
import random
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from datasets import load_dataset
import gensim.downloader as api

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")

# Reproducibility: seed everything (the same idiom as B4-B7).
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device: a torch.device object (used for tensors and models).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch : {torch.__version__}")
print(f"Device  : {device}")
print("Environment ready.")
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 3
Change: drop `from gensim.models import Word2Vec` (no from-scratch training) and `import evaluate`; add `import gensim.downloader as api` and `from sklearn.linear_model import LogisticRegression`; keep the seed-everything + device idiom verbatim.

## Cell 4 - Markdown: STAR scenario - what are we building?

```markdown
## The Scenario

The customer-support-platform team you have worked with since Part A has been
leaning on your zero-shot router. Now leadership wants proof the team can ship a
TRAINED model, not just call pretrained ones. They hand you a new, unfamiliar
corpus - IMDB movie-review sentiment - and the brief is blunt:

> "Build a sentiment classifier from raw text, end to end, and beat a sensible
> baseline. No hand-holding."

Here is your plan, and it is exactly the Part B toolkit:

1. **Features.** Turn each review into ONE vector by averaging its pretrained word
   vectors (`wv` from B5). A review becomes a single 100-dim point in space.
2. **Baseline.** Fit a `LogisticRegression` on those vectors. That accuracy is the
   bar you must beat - if your neural net cannot beat a linear model, it is not
   earning its complexity.
3. **Model.** Build your own `MLPClassifier(nn.Module)` and train it with the loop
   from B6.
4. **Proof.** Evaluate on a held-out test set the model never saw.

Then, honestly, you will run a pretrained transformer on the same reviews and
measure how much head-room is left. That gap is the whole argument for Part C.
```

Provenance: FROM-NEW
(The old notebook had a "pick Path A or Path B" choice cell here; B8 has a single
linear narrative, so this STAR scenario replaces it.)

## Cell 5 - Code: Load IMDB and preview

```python
# Load IMDB from the HuggingFace Hub.
#   - train: 25,000 reviews, test: 25,000 reviews, both perfectly balanced 50/50.
#   - fields: 'text' (the review string), 'label' (0 = negative, 1 = positive).
imdb = load_dataset("imdb")
print(imdb)

# Peek at one positive review.
sample = imdb["train"][0]
print("\nLabel:", sample["label"], "(0=negative, 1=positive)")
print("Review (first 300 chars):")
print(sample["text"][:300], "...")
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 4
Change: keep the load + preview; trim the comment, surface that fields are text/label and splits are 25k/25k.

## Cell 6 - Code: Subsample to a balanced, CPU-friendly split

```python
# Full IMDB is 50k reviews. Averaging word vectors over all of it on CPU is doable
# but slow for a class. We carve a balanced, reproducible subsample:
#   - TRAIN_N positives + TRAIN_N negatives from imdb['train']
#   - TEST_N  positives + TEST_N  negatives from imdb['test'] (the held-out set)
# Everything is seeded, so every student gets the same rows.
TRAIN_N = 2000   # per class -> 4000 train+val reviews total
TEST_N  = 1000   # per class -> 2000 test reviews total

def balanced_sample(split, n_per_class, seed=SEED):
    df = split.to_pandas()
    pos = df[df["label"] == 1].sample(n_per_class, random_state=seed)
    neg = df[df["label"] == 0].sample(n_per_class, random_state=seed)
    out = pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)  # shuffle
    return out.reset_index(drop=True)

train_full = balanced_sample(imdb["train"], TRAIN_N)
test_df    = balanced_sample(imdb["test"],  TEST_N)

# Split train_full -> train / validation (80/20), stratified by label.
train_df, val_df = train_test_split(
    train_full, test_size=0.2, random_state=SEED, stratify=train_full["label"]
)

print(f"train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
print("train label balance:\n", train_df["label"].value_counts())
```

Provenance: FROM-NEW
(The old notebook split train/val inside the Path-A lab; here we add an explicit,
seeded, balanced subsample so the whole pipeline runs fast on CPU and every student
sees identical data. Stratified split keeps classes balanced.)

## Cell 7 - Markdown: Load the pretrained word vectors

```markdown
## Step 1. The Features: Pretrained Word Vectors

You will NOT train word vectors from scratch. You already have them. In B5 you
loaded GloVe vectors with gensim:

```python
import gensim.downloader as api
wv = api.load("glove-wiki-gigaword-100")   # 400,000 words, 100 dims each
```

`wv` is a `KeyedVectors`: a lookup from a word string to a 100-dim numpy vector.
Two facts you will use:
- `wv[word]` (or `wv.get_vector(word)`) returns the 100-dim vector for a word.
- `word in wv.key_to_index` is the cheap, correct OOV check - GloVe does not know
  every token, so you must skip words it has never seen.

The first `api.load` downloads ~130 MB the first time and caches it. After that it
is instant.
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 7
Change: was "Section 1 Path A header"; rewritten into a recap that points students
back to B5's exact `wv` load and the `key_to_index` OOV guard (no from-scratch W2V).

## Cell 8 - Code: Load wv

```python
# Load the same pretrained vectors you used in B5 (100-dim GloVe, 400k words).
# First call downloads ~130 MB and caches it; later calls are instant.
wv = api.load("glove-wiki-gigaword-100")
print("Vocab size:", len(wv.key_to_index))
print("Vector dim:", wv.vector_size)          # 100
print("Vector for 'movie' (first 5 dims):", wv["movie"][:5])
print("Is 'zzqx' known?", "zzqx" in wv.key_to_index)   # False -> OOV guard works
```

Provenance: FROM-NEW
(Replaces the old "train Word2Vec on IMDB" cells; we load the B5 pretrained `wv`
and demonstrate the OOV guard inline so the demo is runnable.)

## Cell 9 - Code: doc_vector demo on toy data

```python
# DEMO (toy): a document vector is just the average of its known word vectors.
# This is the same mean-pool idea as B5's doc_vector / mean_vector helper.
def doc_vector(text, kv=wv, dim=100):
    # Lowercase + naive whitespace tokenization is enough for a bag-of-words feature.
    tokens = text.lower().split()
    # Keep only words the vectors know about (the OOV guard).
    vecs = [kv[t] for t in tokens if t in kv.key_to_index]
    if len(vecs) == 0:
        return np.zeros(dim, dtype=np.float32)   # no known words -> zero vector
    # Mean-pool: average across all kept word vectors -> one dim-D vector.
    return np.mean(vecs, axis=0).astype(np.float32)

# Show it on two tiny "reviews":
v_pos = doc_vector("an absolutely wonderful and moving film")
v_neg = doc_vector("a boring and terrible waste of time")
print("doc_vector shape:", v_pos.shape)        # (100,)
print("pos[:4]:", v_pos[:4])
print("neg[:4]:", v_neg[:4])
# They differ -> the average already carries sentiment signal.
print("vectors differ:", not np.allclose(v_pos, v_neg))
```

Provenance: FROM-NEW
(Toy mean-pool demo so students SEE doc_vector before they apply it; mirrors the
B5 helper. Note the float32 cast - gensim returns float32 already, and we keep it
so the torch tensors downstream are float32, not float64.)

## Cell 10 - Markdown: Lab 1 instructions - build the feature matrices

```markdown
## Lab 1: Build the Feature Matrices (your turn)

You have `doc_vector` and three pandas frames (`train_df`, `val_df`, `test_df`),
each with a `text` column and a `label` column. Turn each split into a feature
matrix and a label vector.

Steps:
1. Write a helper `build_xy(df)` that returns `(X, y)` where:
   - `X` is a 2-D numpy array of shape `(len(df), 100)` - one `doc_vector` per row.
   - `y` is a 1-D numpy array of the labels.
   Hint: apply the demo's document-vector helper to every value in `df['text']`,
   then stack the results into one array. The label array comes straight from
   `df['label']`.
2. Call it for all three splits to get
   `X_train, y_train`, `X_val, y_val`, `X_test, y_test`.

You are done when the verification block prints the right shapes and dtypes
(float32 features, integer labels).
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 8
Change: was "split train/val"; now the student's first solo task is vectorizing all
three splits with the mean-pool helper. Hints name the ROLE (apply the helper, stack,
take the label column) without writing `np.vstack`/`.values`.

## Cell 11 - Code: Lab 1 starter (exercise)

```python
# Build feature matrices for all three splits.

# 1. Helper: turn a dataframe into (X, y).
def build_xy(df):
    # Apply the document-vector helper to every review string, then stack the
    # resulting vectors into one 2-D array.
    X = None  # YOUR CODE: stack doc_vector over df['text'] -> shape (len(df), 100)
    # The labels come straight from the dataframe's label column, as a numpy array.
    y = None  # YOUR CODE: df['label'] as a numpy array
    return X, y

# 2. Build all three splits.
X_train, y_train = None, None  # YOUR CODE: build_xy on the training frame
X_val,   y_val   = None, None  # YOUR CODE: build_xy on the validation frame
X_test,  y_test  = None, None  # YOUR CODE: build_xy on the test frame

# Verification (provided).
for name, X, y in [("train", X_train, y_train),
                   ("val", X_val, y_val),
                   ("test", X_test, y_test)]:
    if X is not None and y is not None:
        print(f"{name:5s}  X={X.shape} ({X.dtype})  y={y.shape} ({y.dtype})")
# Expect X=(N, 100) float32 and y=(N,) integer for each split.
```

Solution (for the solution notebook):
```python
def build_xy(df):
    X = np.vstack([doc_vector(t) for t in df["text"]]).astype(np.float32)
    y = df["label"].to_numpy()
    return X, y

X_train, y_train = build_xy(train_df)
X_val,   y_val   = build_xy(val_df)
X_test,  y_test  = build_xy(test_df)
# Explanation: doc_vector returns a (100,) float32 array per review; np.vstack
# stacks N of them into (N, 100). Labels are pulled straight from the frame.
# Common mistake: forgetting astype(float32) -> torch tensors become float64 and
# mismatch the model weights later.
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 9
Change: old cell did a train/val split; this is the feature-build lab. Starter uses
`None  # YOUR CODE` and a shape/dtype verification block; hints never name vstack.

## Cell 12 - Markdown: The baseline - bar to beat

```markdown
## Step 2. The Baseline: Logistic Regression

Before training a neural net, set the bar. A `LogisticRegression` on the SAME
features is the honest baseline: if your MLP cannot beat a linear model on the same
inputs, the extra complexity is not earning its keep. (This is also exactly the
LogisticRegression baseline you measured in the B5 homework - same idea, new data.)

We fit it on `X_train, y_train` and score it on the validation set. That number is
`baseline_acc` - the line in the sand your MLP has to clear.
```

Provenance: FROM-NEW
(The old notebook never set a logistic baseline on Path A; this makes the "beat the
baseline" goal concrete and ties back to the B5 homework baseline.)

## Cell 13 - Code: Fit the baseline

```python
# Fit a logistic-regression baseline on the averaged-embedding features.
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)

# Score on validation (held out from fitting).
baseline_val_pred = baseline.predict(X_val)
baseline_acc = accuracy_score(y_val, baseline_val_pred)
print(f"Baseline (LogReg) validation accuracy: {baseline_acc:.4f}")
print("This is the bar your MLP must beat.")
```

Provenance: FROM-NEW
(Concrete baseline fit; `baseline_acc` is consumed by Lab 4's comparison.)

## Cell 14 - Markdown: The model - a tiny MLP

```markdown
## Step 3. The Model: Your Own MLP

Now the part you built in B6 and B7: a small `nn.Module`. The architecture is
deliberately simple, because the features are already dense and informative:

```python
input (100) -> Linear(100, 64) -> ReLU -> Linear(64, 2) -> logits (2)
```

Two things to keep straight (you saw both in B6):
- The final layer outputs 2 RAW logits, one per class. Do NOT add a softmax in the
  model: `nn.CrossEntropyLoss` applies log-softmax internally. Adding your own
  double-counts it.
- This is a Deep Averaging Network: average the word vectors, then push the average
  through a feed-forward net. Simple, fast, and a real baseline architecture.

Below is a TOY demo of the module on 3 random "documents" so you can see the output
shape before you write your own.
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 16
Change: old cell defined an Embedding+pool model with a frozen matrix; here the model
takes precomputed 100-d features directly (Linear->ReLU->Linear). Add the no-softmax
warning and the DAN framing.

## Cell 15 - Code: MLP demo on toy tensors

```python
# DEMO (toy): define and run a tiny MLP on 3 fake 100-dim "documents".
class _DemoMLP(nn.Module):
    def __init__(self, in_dim=100, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),   # raw logits, no softmax
        )

    def forward(self, x):
        return self.net(x)

demo_model = _DemoMLP()
fake_docs = torch.randn(3, 100)             # 3 documents, 100-dim each
logits = demo_model(fake_docs)
print("input shape :", fake_docs.shape)     # (3, 100)
print("logits shape:", logits.shape)        # (3, 2) -> one row per doc, 2 classes
print("logits:\n", logits)
# These are raw scores. argmax over dim=1 would give the predicted class.
```

Provenance: FROM-NEW
(Runnable toy demo of the exact module shape students will write, so the forward
pass and (N, 2) logit output are concrete before Lab 2.)

## Cell 16 - Code: Lab 2 starter - define MLPClassifier (exercise)

```markdown
### Lab 2: Define Your MLPClassifier

Write your own `MLPClassifier(nn.Module)` with the architecture from the demo:
`Linear(100 -> hidden) -> ReLU -> Linear(hidden -> 2)`. Then instantiate it, move
it to `device`, and confirm a forward pass on a small batch returns `(batch, 2)`
logits.
```

```python
# Lab 2: define your own MLP.
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=100, hidden=64, n_classes=2):
        super().__init__()
        # YOUR CODE: build the layers. You need a linear in_dim->hidden, a ReLU,
        # and a linear hidden->n_classes that outputs raw logits (no softmax).
        self.net = None  # YOUR CODE

    def forward(self, x):
        # YOUR CODE: pass x through your layers and return the logits.
        return None  # YOUR CODE

# Instantiate and move to device.
model = None  # YOUR CODE: create an MLPClassifier and send it to `device`

# Verification (provided).
if model is not None:
    probe = torch.randn(4, 100).to(device)
    out = model(probe)
    print("Output shape:", out.shape)   # expect (4, 2)
    assert out.shape == (4, 2), "Forward pass should return (batch, 2) logits."
    print("MLPClassifier looks correct.")
```

Solution:
```python
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=100, hidden=64, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)

model = MLPClassifier().to(device)
# Explanation: identical shape to the demo. Common mistakes: adding nn.Softmax at
# the end (breaks CrossEntropyLoss), or forgetting .to(device) (device mismatch at
# train time).
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 17
Change: old lab defined an embedding-matrix model; here it is the precomputed-feature
MLP. Starter `None  # YOUR CODE` for the layers and forward; hints name the layer
roles, not the exact nn.Sequential call. Verification asserts (4, 2).

## Cell 17 - Markdown: The training loop

```markdown
## Step 4. Train It

This is the B6 loop, unchanged. Wrap the features and labels in tensors, batch them
with a `DataLoader`, and for each epoch:

1. `model.train()`, then for each batch: `optimizer.zero_grad()` -> forward ->
   `loss = criterion(logits, labels)` -> `loss.backward()` -> `optimizer.step()`.
2. `model.eval()` under `torch.no_grad()` to measure validation accuracy.

Two dtype rules that bite everyone (you saw them in B6):
- Features must be `float32` tensors (the model weights are float32).
- Labels for `CrossEntropyLoss` must be `int64` / `long`, shape `(N,)` - class
  indices, NOT one-hot.

The demo below builds the tensors and DataLoaders for you and shows ONE training
step so the moving parts are visible. In Lab 3 you write the epoch loop.
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 18
Change: was "training loop with early stopping" header; reworded to the B6 loop with
the float32 features / int64 labels dtype rules called out explicitly.

## Cell 18 - Code: Build tensors + DataLoaders + one-step demo

```python
# Wrap numpy features/labels into tensors with the CORRECT dtypes.
def make_loader(X, y, batch_size=64, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)   # features: float32
    y_t = torch.tensor(y, dtype=torch.long)      # labels: int64 for CrossEntropy
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, batch_size=64, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   batch_size=256, shuffle=False)

# DEMO: one training step, so you can see the five moves before writing the loop.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

xb, yb = next(iter(train_loader))
xb, yb = xb.to(device), yb.to(device)
model.train()
optimizer.zero_grad()            # 1. clear old gradients
logits = model(xb)              # 2. forward
loss = criterion(logits, yb)    # 3. loss (logits vs class indices)
loss.backward()                 # 4. backprop (the B4 engine)
optimizer.step()                # 5. update weights
print(f"One step done. batch loss = {loss.item():.4f}")
```

Provenance: FROM-NEW
(Explicit tensor-dtype helper + a single annotated training step. Replaces the old
padded-sequence DataLoader; here inputs are precomputed dense vectors.)

## Cell 19 - Code: Lab 3 starter - write the epoch loop (exercise)

```markdown
### Lab 3: Train for Several Epochs

Write the loop that trains `model` for `EPOCHS` epochs and prints validation
accuracy after each one. Reuse `train_loader`, `val_loader`, `criterion`,
`optimizer` from the demo.
```

```python
EPOCHS = 15

def evaluate(model, loader):
    # Provided: returns accuracy over a loader (no gradients).
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        # YOUR CODE: the five training moves on this batch
        #   clear gradients -> forward -> loss(logits, labels) -> backward -> step
        None  # YOUR CODE

    val_acc = evaluate(model, val_loader)
    print(f"epoch {epoch+1:2d}  val_acc = {val_acc:.4f}")

# Verification: did we beat the logistic baseline on validation?
final_val = evaluate(model, val_loader)
print(f"\nMLP val acc {final_val:.4f}  vs baseline {baseline_acc:.4f}  "
      f"-> {'BEAT IT' if final_val > baseline_acc else 'not yet, tune EPOCHS/hidden'}")
```

Solution (the loop body):
```python
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
    val_acc = evaluate(model, val_loader)
    print(f"epoch {epoch+1:2d}  val_acc = {val_acc:.4f}")
# Explanation: identical five moves as the demo step, now inside the batch loop.
# Common mistakes: forgetting optimizer.zero_grad() (gradients accumulate across
# batches and training diverges), or calling .backward() outside the batch loop.
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 19
Change: drop the old early-stopping machinery for a clean fixed-epoch loop; provide
`evaluate` so the only `None  # YOUR CODE` is the five training moves; hint lists the
moves by name without writing them. Verification compares to `baseline_acc`.

## Cell 20 - Markdown: Lab 4 - final evaluation on the test set

```markdown
## Step 5. Prove It on the Held-Out Test Set

The validation set guided training, so the honest score is the TEST set the model
has never touched. In this lab you:

1. Get the MLP's predictions on `X_test`.
2. Report test accuracy and macro-F1.
3. Also score the SAME logistic baseline on the test set, so the comparison is
   apples-to-apples.
4. Plot a confusion matrix to see WHERE the errors land (false positives vs false
   negatives).

You are done when you can state one sentence: "My MLP scored X on test, the baseline
scored Y, so the MLP beat the baseline by Z points."
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 20
Change: was "evaluate on test set + latency"; refocused on test accuracy + macro-F1 +
baseline-on-test + confusion matrix. Latency moves to the wrap-up take-home.

## Cell 21 - Code: Lab 4 starter - evaluate + confusion matrix (exercise)

```python
# Lab 4: final evaluation on the held-out test set.

# 1. MLP predictions on the test features.
test_loader = make_loader(X_test, y_test, batch_size=256, shuffle=False)
model.eval()
mlp_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        # YOUR CODE: append this batch's predicted classes (argmax over logits)
        None  # YOUR CODE
mlp_preds = np.array(mlp_preds)

# 2. Metrics for the MLP.
mlp_acc = None  # YOUR CODE: accuracy of mlp_preds vs y_test
mlp_f1  = None  # YOUR CODE: macro-F1 of mlp_preds vs y_test

# 3. Same baseline, scored on the test set.
base_test_acc = accuracy_score(y_test, baseline.predict(X_test))

# Verification (provided).
if mlp_acc is not None:
    print(f"MLP   test acc={mlp_acc:.4f}  macro-F1={mlp_f1:.4f}")
    print(f"LogReg test acc={base_test_acc:.4f}")
    print("MLP beat baseline:" , mlp_acc > base_test_acc)
    cm = confusion_matrix(y_test, mlp_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title("MLP confusion matrix")
    plt.show()
    print(classification_report(y_test, mlp_preds, target_names=["neg", "pos"]))
```

Solution:
```python
mlp_preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        mlp_preds.extend(model(xb).argmax(dim=1).cpu().numpy())
mlp_preds = np.array(mlp_preds)

mlp_acc = accuracy_score(y_test, mlp_preds)
mlp_f1  = f1_score(y_test, mlp_preds, average="macro")
# Explanation: argmax(dim=1) over the (batch, 2) logits gives the class; move to CPU
# before numpy. macro-F1 averages the two classes equally. Common mistake: comparing
# the MLP test score to the baseline's VALIDATION score - score both on test.
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cell 21
Change: keep accuracy/macro-F1/confusion-matrix/classification_report; drop the CPU
latency timing (moved to wrap-up). Only `None  # YOUR CODE` are the batch-argmax and
the two metric calls; the plotting/report is provided.

## Cell 22 - Code: STRETCH - zero-shot transformer on the same reviews (bridge to C)

```markdown
### Stretch: How Close Did You Get to a Pretrained Transformer?

Your MLP learns sentiment from AVERAGED static word vectors. A pretrained transformer
reads each review WITH context (word order, negation). Run the A2 sentiment pipeline
on a handful of the SAME test reviews and compare. Honest caveat: this transformer was
fine-tuned on SST-2 (short single sentences), and IMDB reviews are long documents, so
this is an indicative comparison, not a like-for-like benchmark - which is itself a
useful lesson about transfer.
```

```python
# Stretch (optional, needs the transformers install from Cell 2).
from transformers import pipeline

# The pretrained sentiment model from A2 (DistilBERT fine-tuned on SST-2, ~91% on
# SST-2 dev). device=0 uses GPU if present, else -1 for CPU (pipeline int convention).
clf = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if device.type == "cuda" else -1,
)

# Score the first 200 test reviews (truncate to 512 chars to keep it quick).
sample_texts = list(test_df["text"].iloc[:200].str.slice(0, 512))
sample_labels = test_df["label"].iloc[:200].to_numpy()
preds = clf(sample_texts, truncation=True)
trans_preds = np.array([1 if p["label"] == "POSITIVE" else 0 for p in preds])
trans_acc = accuracy_score(sample_labels, trans_preds)

print(f"Pretrained transformer acc on 200 IMDB reviews: {trans_acc:.4f}")
print(f"Your MLP test acc                              : {mlp_acc:.4f}")
print(f"Gap (transformer - MLP)                        : {trans_acc - mlp_acc:+.4f}")
print("\nThat gap is what Part C 'earns' by FINE-TUNING a transformer for the task.")
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cells 22-30
Change: the old Path B fine-tuned DistilBERT with Trainer (now C9's job). Replace the
whole Trainer path with a short ZERO-SHOT pipeline comparison on the same reviews, with
the honest SST-2-vs-IMDB caveat, framed as the bridge to C9.

## Cell 23 - Code: HOMEWORK - better features (SBERT 384-d) re-fit the same MLP

```markdown
### Homework (async, deeper): Features Matter More Than the Model

You proved an MLP beats a linear baseline on averaged GloVe. Now test a stronger
HYPOTHESIS: better FEATURES beat a fancier model. Swap the 100-d averaged-GloVe
features for 384-d sentence embeddings from `all-MiniLM-L6-v2` (the `embedder` from
B5), keep the SAME MLP architecture (just widen the input to 384), retrain, and
measure the lift. SBERT reads the whole sentence with context, so even an averaged-
word baseline should be left behind.

Tasks:
1. Build a `SentenceTransformer("all-MiniLM-L6-v2")` (this is B5's `embedder`).
2. Encode `train_df`, `val_df`, `test_df` texts to 384-d feature matrices.
3. Instantiate `MLPClassifier(in_dim=384)`, retrain with the SAME loop, evaluate.
4. Compare 384-d-SBERT test accuracy to your 100-d-GloVe `mlp_acc`. How big is the
   lift from features alone, with the model held constant?
```

```python
# Homework starter.
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

# 1. Encode each split to 384-d features (truncate long reviews for speed).
def sbert_features(df):
    texts = list(df["text"].str.slice(0, 512))
    # YOUR CODE: encode texts with the embedder -> (len(df), 384) float32 array
    return None  # YOUR CODE

Xs_train = None  # YOUR CODE: sbert_features(train_df)
Xs_val   = None  # YOUR CODE
Xs_test  = None  # YOUR CODE

# 2. Same MLP, wider input.
sbert_model = None  # YOUR CODE: MLPClassifier(in_dim=384).to(device)

# 3. Retrain with the same loop pattern (reuse make_loader / evaluate), then:
#    sbert_acc = accuracy on Xs_test, y_test
# 4. print the lift: sbert_acc - mlp_acc

# Reference solution lives in the solution notebook.
```

Solution (sketch, full code in solution notebook):
```python
def sbert_features(df):
    texts = list(df["text"].str.slice(0, 512))
    return embedder.encode(texts, batch_size=64, show_progress_bar=False).astype(np.float32)

Xs_train, Xs_val, Xs_test = sbert_features(train_df), sbert_features(val_df), sbert_features(test_df)
sbert_model = MLPClassifier(in_dim=384).to(device)
# ... reuse make_loader + the Lab 3 loop + evaluate ...
# Expected: SBERT features lift test accuracy a few points over averaged GloVe with
# the SAME model, proving the "features matter" point and motivating Part C.
```

Provenance: FROM-NEW
(New production-oriented homework that reuses B5's `embedder`, holds the model
constant, and isolates the effect of richer features. Resolves the synthesis "100-d
vs 384-d" note by making BOTH widths explicit and comparable.)

## Cell 24 - Markdown: Wrap-up and bridge to C9

```markdown
## Wrap-Up: You Own the Pipeline

What you just did, solo:
- Turned raw reviews into averaged-embedding features with `doc_vector` (B5).
- Set an honest baseline with `LogisticRegression`.
- Built and trained your own `MLPClassifier` with the B6 loop (the B4 `.backward()`
  engine under the hood).
- Beat the baseline on a held-out test set and read the confusion matrix.
- Measured, honestly, the gap to a pretrained transformer.

### When is this the RIGHT model in production?
The averaged-embedding MLP is cheap, fast, runs on CPU, and is easy to ship and
explain - a great default when latency and cost matter and the accuracy is "good
enough." A fine-tuned transformer buys you accuracy (it reads word order and
negation) at the cost of GPU, latency, and serving complexity. Knowing which to
reach for is the job.

### The ceiling you just hit
Averaging word vectors is bag-of-words: it throws away order and negation, so
"not good" looks positive. That is the hard ceiling on this whole approach.

### Bridge to Part C
Part C breaks that ceiling. In C9 you stop averaging static vectors and instead
FINE-TUNE a contextual transformer (`distilbert-base-uncased`) - the SAME
`.backward()` engine from B4, now updating a real language model - then drop it into
a Gradio chatbot. You measured the gap; C9 closes it.

**One-line bridge:** "You measured the ceiling of averaged features; C9 fine-tunes
DistilBERT to break it, then serves it as the chatbot."
```

Provenance: FROM-OLD 3-Text-Classification/10-Capstone_2.ipynb cells 31-34
Change: fold the old engineering-memo prompts and wrap-up into a Part B capstone
recap; add the production "which model" take-home and the explicit C9 bridge.

---

# VERIFICATION CHECKLIST

- [ ] 25 cells total (0-24); every cell has a Provenance line (FROM-OLD with source
      file + cell index, or FROM-NEW); RENEWED Cell Migration Map present near the top.
- [ ] Theory -> Demo -> Lab cadence holds; never more than 3 markdown cells without a
      code cell (each step has a demo or starter code cell).
- [ ] Install cell pins `numpy<2`, `scipy<1.13`, `gensim==4.3.3`, `transformers==4.57.1`,
      `datasets>=2.19,<3`, `sentence-transformers==3.4.1`; Colab restart note present.
- [ ] Setup uses `SEED=42`, seeds random/numpy/torch, `device = torch.device(...)`.
- [ ] Dataset id verified: `load_dataset("imdb")` -> fields text/label, 25k train, 25k
      test. Subsample is balanced + stratified + seeded.
- [ ] Reuses B5 `wv` (gensim `glove-wiki-gigaword-100`, 100-d) with the
      `word in wv.key_to_index` OOV guard; reuses B5 `embedder` (all-MiniLM-L6-v2, 384-d)
      in the homework. No from-scratch Word2Vec, no embedding matrix, no padding.
- [ ] Model outputs raw logits (no softmax in the module); `CrossEntropyLoss` fed
      logits + int64 labels shape (N,); features cast to float32 tensors.
- [ ] Training loop demo shows zero_grad -> forward -> loss -> backward -> step; eval
      under `model.eval()` + `torch.no_grad()`.
- [ ] Every lab starter uses `None  # YOUR CODE` and a provided verification block; no
      hint names the exact method/call that solves the line (cover-the-solution test).
- [ ] Lab tiers present: core (Labs 1-4) + in-notebook stretch (Cell 22 zero-shot) +
      async homework (Cell 23 SBERT 384-d).
- [ ] Baseline is concrete (`baseline_acc` from LogisticRegression) and the MLP is
      compared to it on the TEST set (apples-to-apples).
- [ ] No BERT fine-tuning / Trainer path (moved to C9); no TF/Keras; no RNN/LSTM; no
      LLM API keys.
- [ ] STAR story (Cell 4) and Chatbot Through-Line (header sections + Cell 22/24 bridge)
      both present.
- [ ] Honest caveat that SST-2 transformer score is not a like-for-like IMDB benchmark.
- [ ] Plain ASCII only: no em/en dashes, no Unicode multiply sign, no emoji.

---

# RESEARCH VALIDATED (June 2026)

All five research cycles used the `/research` skill with live web search. Sources and
the specific fact extracted from each:

- gensim KeyedVectors / downloader API:
  https://radimrehurek.com/gensim/models/keyedvectors.html and
  https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html -
  `api.load("glove-wiki-gigaword-100")` returns a KeyedVectors (400k words, 100 dims);
  `.key_to_index` maps word->index; OOV guard is `word in kv.key_to_index`,
  `kv.get_vector(word)` raises KeyError on OOV.
- HuggingFace IMDB dataset:
  https://huggingface.co/docs/datasets/process and the stanfordnlp/imdb card -
  `load_dataset('imdb')` -> features ['text','label'], train 25,000 / test 25,000
  (plus 50,000 unsupervised), label 0=neg / 1=pos, balanced.
- Averaged-embedding document classification (DAN):
  https://github.com/miyyer/dan and
  https://arxiv.org/html/2204.03954v6 (Are We Really Making Progress in Text
  Classification) - mean-pool word embeddings then a feed-forward net (Deep Averaging
  Network); logistic regression on the same averaged embeddings is an architecturally
  comparable baseline; SWEM/averaged-embedding models are strong simple baselines.
  https://medium.com/@roysid456/... reports a simple DAN around 79% on a sentiment dev set.
- gensim / scipy / numpy Colab pins:
  https://github.com/piskvorky/gensim/issues/3525 and
  https://github.com/googlecolab/colabtools/issues/5199 - gensim 4.3 breaks on
  scipy>=1.13 (removed `scipy.linalg.triu`) and needs numpy<2; pin
  `gensim==4.3.3 scipy<1.13 numpy<2` and restart the Colab runtime (Colab ships numpy 2.x).
- transformers version:
  https://pypi.org/project/transformers/ and
  https://github.com/huggingface/transformers/releases - 4.57.x is current stable, not
  5.x; declares numpy>=1.17 (no hard numpy>=2), so it coexists with the numpy<2 pin.
- CrossEntropyLoss contract:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html and
  https://github.com/mrdbourke/pytorch-deep-learning/discussions/252 - expects raw
  logits (applies log-softmax internally; do NOT add softmax), input (N, C), target
  class indices shape (N,) dtype torch.long; float targets error.
- Training-loop idioms:
  https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html and
  https://sebastianraschka.com/faq/docs/training-loop-in-pytorch.html - gradients
  accumulate, so `optimizer.zero_grad()` each step; `model.train()` / `model.eval()`
  toggle dropout/BN; evaluate under `torch.no_grad()`.
- DistilBERT SST-2 accuracy bar:
  https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english -
  91.3% on the SST-2 dev set (BERT-base reaches 92.7%).
- all-MiniLM-L6-v2:
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 - maps sentences to a
  384-dim dense vector; contextual encoder, stronger than averaged static word vectors.
- Averaged embeddings lose order/negation (Part C motivation):
  https://jina.ai/news/text-embeddings-fail-to-capture-word-order-and-how-to-fix-it/ and
  https://orca.cardiff.ac.uk/130910 - averaging is order-invariant bag-of-words; "not
  good" looks positive; transformers/self-attention resolve this by contextualizing.
- Production model-choice tradeoff:
  https://arxiv.org/pdf/2602.06370 (Cost-Aware Model Selection) - averaged-embedding +
  linear/LR: cheap, fast, CPU, interpretable (~161ms, ~$2.20/1M); fine-tuned DistilBERT:
  higher accuracy at higher cost/latency (~100ms GPU, ~$5/1M).
- SST-2 vs IMDB are different tasks:
  https://arxiv.org/html/2601.07235v2 and the SST/IMDB dataset descriptions - IMDB is
  document-level (avg ~221 words); SST-2 is sentence/phrase-level and short, so SST-2
  accuracy is not a like-for-like IMDB benchmark (honest caveat in Cell 22).

---

Plan written to `plans/refactor/notebooks/B8-capstone-b.md`.

Next step: run `/build-notebook capstone-b colab` to generate the exercise + solution
notebooks from this plan, 5 cells at a time with approval between batches.






