# B7 - MLP on Word2Vec (THE STOPPER) - Cell-by-Cell Plan

## Status

TOUCHED. Source notebook: `3-Text-Classification/8-MLP_Text_Classification.ipynb`.
New file: `B-How-It-Works/7-MLP_Word2Vec.ipynb`, slug `mlp-word2vec`. Environment: Colab.

This is THE STOPPER: the single most important notebook in the course, the convergence
point of Parts A and B. Everything earned so far pays off here. Pretrained word vectors
(B5) become FEATURES, averaged into a fixed-size document vector, fed to the MLP machinery
(B6) to classify sentiment, and the trained MLP must BEAT the LogisticRegression baseline
that B5's homework measured.

The source NB8 trained Word2Vec from scratch on CNN Headlines and used an `nn.Embedding`
plus masked-mean-pool plus sequence-of-token-ids architecture (a 4-class news classifier).
The refactor changes the input representation, the dataset, and the architecture: the
document vector is ALREADY pooled (100-d averaged GloVe from B5), so the model is a plain
dense MLP `Linear(100, 128) -> ReLU -> Dropout -> Linear(128, 2)` over precomputed features,
trained on SST-2 (the single sentiment dataset carried through to the Part C fine-tune).
Most old cells are therefore re-themed or dropped; the training-loop, early-stopping,
evaluation, confusion-matrix, ablation, and save/wrap-up STRUCTURE is reused.

## Resolved input-representation decision (the synthesis MUST-RESOLVE)

The Topic-B synthesis flags that B5 ships TWO document representations: a 100-d averaged
word2vec `doc_vector` (homework, saved as `X, y`) and a 384-d SBERT path (`all-MiniLM-L6-v2`).
B7 picks ONE explicitly and states it:

- B7's canonical input is the 100-d averaged word2vec feature `X` (shape `(N, 100)`) and `y`,
  exactly what B5's homework builds and saves. The notebook is titled "MLP on Word2Vec",
  so the word2vec path IS the input. Input width is 100, not 384.
- B7 first tries to LOAD the saved `X.npy` / `y.npy` from B5. If they are absent (fresh
  Colab session), B7 REBUILDS them deterministically: load `wv` via
  `gensim.downloader.api.load('glove-wiki-gigaword-100')`, define `doc_vector(text)` (mean-pool
  of in-vocab word vectors, OOV-guarded by `word in wv.key_to_index`), and apply it to a
  subsample of SST-2. Same names, same 100-d width, same `doc_vector` helper as B5.
- The 384-d SBERT representation is the HOMEWORK lift (re-encode the same SST-2 sentences with
  `embedder`, retrain the identical MLP, measure the accuracy gain). This reconciles the
  "384-dim" through-line language: averaged static word2vec (100-d) is the core; contextual
  SBERT (384-d) is the documented next rung, and fine-tuning the encoder itself is C9.

Names reused verbatim from B5/B6 (no redefinition, no collision): `wv`, `embedder`,
`doc_vector`, `X`, `y`, `device` (a `torch.device` object), `SEED = 42`. The old NB8 names
(`NewsClassifier`, `HeadlineDataset`, `embedding_matrix`, `word_to_idx`, masked-mean-pool
`forward(input_ids, mask)`) are DROPPED because the doc vector is already pooled.

## Context

Students arrive from B6 holding the PyTorch neural-net machinery: how to subclass `nn.Module`
with `__init__` and `forward`, `nn.Linear`, `nn.ReLU`, `nn.Sequential`, `nn.CrossEntropyLoss`
(takes raw logits and integer class labels, no softmax, no one-hot), `torch.optim.Adam`, the
`zero_grad -> backward -> step` cycle, `TensorDataset` + `DataLoader` for batching/shuffling,
and the full train/eval loop on toy data. They also carry from B4/B5: `SEED = 42`,
`torch.manual_seed(SEED)`, `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`,
`wv` (gensim KeyedVectors, glove-wiki-gigaword-100, 400k words, 100-d), `embedder`
(SentenceTransformer all-MiniLM-L6-v2, 384-d), and the `doc_vector` idea (mean-pool word
vectors) plus the LogisticRegression baseline number from the B5 homework.

They leave with the key insight: the "embeddings as features" pattern. A frozen pretrained
encoder turns text into a fixed-size vector; a small trainable head learns the task on top.
A nonlinear MLP head beats a linear LogisticRegression head on the SAME averaged features,
but BOTH are capped because averaging is a bag-of-words operation that throws away word order
and context ("not good" averages to nearly the same vector as "good"). That ceiling is the
exact reason Part C reaches for a contextual transformer with attention.

## Chatbot Through-Line

The course ends on a fine-tuned DistilBERT loaded into a Gradio chatbot. B7 is the rung where
the student first TRAINS a text classifier end to end and sees the embeddings-as-features
pattern that the whole transfer-learning story is built on:

- B5 handed B7 its inputs: the 100-d averaged word2vec `X, y` and the LogisticRegression
  baseline. B7 cashes that in by training an MLP head that beats the baseline.
- The pattern (freeze a pretrained encoder, train a small head) is the literal precursor to
  C9. In B7 the encoder is frozen word vectors and only the MLP head trains. In C9 the encoder
  (DistilBERT) ITSELF becomes trainable: same loss, same `.backward()` loop from B4/B6, just
  unfrozen. B7 is "frozen features + head"; C9 is "fine-tune the whole encoder".
- The averaging ceiling is the motivation for attention. B7 ends by showing a misclassified
  negation example ("not good") where the averaged vector cannot encode order. The one-line
  bridge: a contextual transformer reads the whole sentence with attention, so C9's fine-tuned
  DistilBERT beats even SBERT, and that fine-tuned model is what the Gradio chatbot loads.

One-line bridge to B8 (Capstone B): "You now own the full classical-to-neural pipeline -
pretrained features in, trained classifier out, baseline beaten. Capstone B has you run it end
to end on a fresh dataset; then Part C swaps the frozen encoder for a fine-tuned transformer."

## STAR Story

- Situation: You are the ML/NLP person on the customer-support platform from Parts A and B.
  In Part A you topped out around 70-85 percent with zero-shot `pipeline()` calls. In B5 you
  built reusable document features and measured a LogisticRegression baseline on them. Leadership
  now wants a sentiment classifier that actually beats that baseline, runs cheaply (no GPU per
  request), and is yours to own and deploy.
- Task: Train a model that turns the B5 document features into a sentiment prediction and beats
  the LogisticRegression bar, using only pretrained embeddings (no GPU bill, no labeled data
  beyond a small set), and understand exactly where the approach hits a wall.
- Action: Load (or rebuild) the 100-d averaged word2vec feature matrix `X, y` from SST-2,
  standardize it, fit the LogisticRegression baseline, then build and train an MLP head with the
  B6 machinery (CrossEntropyLoss, Adam, DataLoader, early stopping), evaluate against the
  baseline, and inspect the errors.
- Result: An MLP that beats the LogisticRegression baseline on the same features, an honest
  confusion matrix and error analysis showing where averaging fails (negation, order), and a
  clear, measured path forward: contextual sentence embeddings (homework, +8 to +9 points) and
  then a fine-tuned transformer (Part C). The team gets a cheap, owned classifier today and a
  roadmap to a better one.

## Deliverables

- Exercise: `B-How-It-Works/7-MLP_Word2Vec.ipynb`
- Solution: `Solutions/B-How-It-Works/7-MLP_Word2Vec.ipynb` (gitignored)

## Session Timing (~60-90 min)

| Block | Cells | Time |
|-------|-------|------|
| Setup + context + load/rebuild features | 0-5 | 12 min |
| Part 1: features + LogisticRegression baseline + Lab 1 | 6-9 | 20 min |
| Part 2: define MLP + train loop (Labs 2-3) | 10-16 | 30 min |
| Part 3: evaluate vs baseline + error analysis + Lab 4 | 17-21 | 15 min |
| Wrap-up + homework (SBERT lift) | 22-23 | async + 8 min |

## Cell Migration Map

Old source = `3-Text-Classification/8-MLP_Text_Classification.ipynb` (44 cells, indices 0-43,
no stable ids; cited by index). New target ~24 cells. Action = keep / edit / drop / new.

| Old cell (index) | New cell | Action |
|------------------|----------|--------|
| 0 (title/objectives md) | 0 | edit - retheme to STOPPER / support-team / SST-2 sentiment |
| 1 (env setup md) | 1 | edit - fold into setup header, add Colab restart note |
| 2 (pip install) | 2 | edit - new pinned line (numpy<2, scipy<1.13, gensim==4.3.3, datasets<3, scikit-learn) |
| 3 (imports) | 3 | edit - drop nn.Embedding/sequence imports; add sklearn, gensim, datasets |
| 4 (hyperparameters) | 3 | merge - SEED/device/hyperparams into the imports+config cell |
| 5 (load CNN headlines) | 4 | edit - replace with load-or-rebuild SST-2 averaged-word2vec X, y |
| 6 (preprocess/baseline md) | 5 | edit - "embeddings as features" + the baseline bar md |
| 7 (TextBlob tokenize) | 4 | merge - tokenization folded into doc_vector rebuild fallback |
| 8 (train Word2Vec) | - | DROP - no from-scratch training; wv is pretrained (B5) |
| 9 (w2v sanity check) | - | DROP - covered in B5 |
| 10 (vocab word->id) | - | DROP - doc vector is pooled, no id sequences |
| 11 (embedding matrix) | - | DROP - no nn.Embedding |
| 12 (text_to_ids) | - | DROP - no id sequences |
| 13 (Lab 1: Dataset md) | 11 | edit - re-theme to TensorDataset/DataLoader on X, y (Lab 2 here) |
| 14 (Lab 1: HeadlineDataset) | 12 | edit - becomes the DataLoader-build lab on dense features |
| 15 (MLP architecture md) | 6 / 10 | edit - split: baseline md (6) + MLP architecture md (10) |
| 16 (NewsClassifier def) | 13 | edit - SentimentMLP: Linear(100,128)->ReLU->Dropout->Linear(128,2), no embedding/pool |
| 17 (count_parameters) | 14 | keep - count_parameters helper + dummy forward |
| 18 (dummy forward test) | 14 | merge - into the demo cell |
| 19 (Lab 2: instantiate md) | 15 | edit - instantiate model + loss + optimizer lab md |
| 20 (Lab 2: instantiate code) | 15 | edit - model/loss/optimizer starter (no pretrained weights load) |
| 21 (training strategy md) | 10 | merge - training-loop strategy folded into Part 2 md |
| 22 (train/val/test split) | 7 | edit - stratified split on X, y with random_state=SEED (in Lab 1 / baseline) |
| 23 (loss + optimizer) | 15 | merge - into the instantiate lab |
| 24 (train/eval helpers) | 14 | edit - train_one_epoch/evaluate on (features, labels) batches |
| 25 (Lab 3: training loop md) | 16 | keep - training loop + early stopping lab md |
| 26 (Lab 3: training loop) | 16 | edit - training loop starter on feature DataLoaders |
| 27 (plot training curves) | 17 | keep - loss/acc curves (demo, provided) |
| 28 (reload best checkpoint) | 17 | merge - reload best state_dict before test |
| 29 (why test once md) | 18 | edit - condense into evaluation md |
| 30 (eval section md) | 18 | merge - into evaluation md |
| 31 (test metrics) | 18 | edit - test accuracy + macro-F1 vs baseline (demo) |
| 32 (confusion matrix) | 19 | edit - binary pos/neg confusion matrix |
| 33 (predict_headline) | 20 | edit - predict_sentiment(text) inference helper on doc_vector |
| 34 (misclassification analysis) | 21 | edit - Lab 4: find a negation/order failure |
| 35 (Lab 4: adversarial md) | 21 | merge - adversarial idea folded into Lab 4 md |
| 36 (Lab 4: adversarial code) | 21 | merge - into Lab 4 starter |
| 37 (compare with baseline) | 18 | merge - baseline comparison into eval cell |
| 38 (ablation md) | 23 | edit - freeze-vs-scale ablation becomes stretch in homework cell |
| 39 (ablation demo) | 23 | merge - into stretch description |
| 40 (Lab 5: hyperparam md) | 23 | merge - hyperparam sweep becomes a stretch option |
| 41 (save model md) | 22 | edit - save state_dict + scaler for inference (wrap-up md) |
| 42 (save model code) | 22 | edit - torch.save(state_dict) + pickle scaler demo |
| 43 (wrap-up md) | 23 | edit - wrap-up + homework (SBERT lift) + bridge to B8/C9 |

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~24 cells)

## Cell 0 - Markdown: Title, the convergence, objectives, prerequisites

```markdown
# B7. MLP on Word2Vec: The Stopper

*ML and NLP course - Data Trainers LLC - Axel Sirota*

## Everything converges here

This is the notebook the whole course has been climbing toward. Look at what you now hold:

- From B5: pretrained word vectors (`wv`) and a way to turn a sentence into one fixed-size
  vector by averaging them (`doc_vector`). You also measured a LogisticRegression baseline
  on those vectors.
- From B6: how to build a model with `nn.Module`, pick a loss, pick an optimizer, batch data
  with a `DataLoader`, and run a training loop.

We snap those two halves together. Text becomes a 100-dimensional vector (pretrained word2vec,
averaged), and that vector feeds a small neural network you train to classify sentiment. This
is the **embeddings as features** pattern: freeze a pretrained encoder, train a small head on
top of its output. It is the foundation under every modern transfer-learning system, and it is
the direct precursor to fine-tuning a transformer in Part C.

The bar is concrete. Your MLP must BEAT the LogisticRegression baseline on the exact same
features. If a linear model already does well, a nonlinear one should do at least a little
better, and we will see exactly why.

## What you will be able to do

- Load (or rebuild) a 100-d averaged-word2vec feature matrix `X, y` for SST-2 sentiment.
- Fit and measure the LogisticRegression baseline: the bar to beat.
- Define and train an MLP head on those features with the B6 machinery.
- Evaluate against the baseline, read a confusion matrix, and find where the model fails.
- Explain why averaging word vectors caps accuracy, and what fixes it (attention, Part C).

## Prerequisites

- B5: `wv`, `doc_vector`, cosine geometry, the LogisticRegression baseline idea.
- B6: `nn.Module`, `nn.Linear`, `nn.ReLU`, `CrossEntropyLoss`, `Adam`, `TensorDataset`,
  `DataLoader`, the training loop.

## Session format

Theory -> Demo -> Lab, four labs. Runs on Colab CPU in a few minutes; a GPU is optional and
barely matters because the model is tiny.
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cell 0
Change: retheme from "Neural News Classifier" (4-class CNN Headlines) to THE STOPPER framing
on SST-2 sentiment; restate prerequisites in terms of B5/B6 carry-forward names; drop the
from-scratch-Word2Vec objective.

## Cell 1 - Markdown: Section 0, setup and the Colab restart note

```markdown
## Section 0. Setup

We install pinned versions so the gensim word-vector loader works. Two pins matter:

- `numpy<2` and `scipy<1.13`: gensim 4.3 imports `scipy.linalg.triu`, which was removed in
  scipy 1.13. Without these pins you get `ImportError: cannot import name 'triu'`.
- `gensim==4.3.3`: brings the `gensim.downloader` API and the `key_to_index` vocabulary
  interface we use to guard against out-of-vocabulary words.

Colab ships numpy 2.x preinstalled, so after the install cell you may need to restart the
runtime (Runtime -> Restart runtime) once, then run the notebook from the top. This is the same
restart you did in B5.
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cell 1
Change: replace the generic "environment setup" prose with the exact gensim/scipy/numpy pin
rationale and the Colab restart note carried from B5.

## Cell 2 - Code: pinned install

```python
# Pinned install. Restart the Colab runtime once after this cell if prompted, then run from top.
!pip install -q "numpy<2" "scipy<1.13" "gensim==4.3.3" "datasets>=2.19,<3" scikit-learn

# Notes:
# - numpy<2 and scipy<1.13 keep gensim 4.3 importable (scipy.linalg.triu was removed in 1.13).
# - datasets<3 keeps the GLUE/SST-2 loader stable (datasets 4.x dropped script-based loaders).
# - scikit-learn gives us StandardScaler, LogisticRegression, and the metrics.
# - torch, pandas, numpy, matplotlib, seaborn are preinstalled on Colab.
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cell 2
Change: swap the old install set for the B5-consistent pinned line; add datasets<3 and
scikit-learn; drop any from-scratch-training or textblob requirement (doc_vector fallback uses
a simple regex split, no textblob).

## Cell 3 - Code: imports, seed, device, hyperparameters

```python
# Imports
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility: same SEED as B4/B5/B6
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device: a torch.device object (the B4/B5/B6 idiom)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters, all in one place
N_SAMPLES   = 2000   # SST-2 rows to subsample (fast on CPU); raise for a stronger model
INPUT_DIM   = 100    # averaged word2vec dimension (glove-wiki-gigaword-100). NOT 384 (SBERT).
HIDDEN_DIM  = 128
N_CLASSES   = 2      # SST-2: negative / positive
DROPOUT     = 0.3
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS_MAX  = 50
PATIENCE    = 6      # early-stopping patience (epochs without val improvement)
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 3 + 4
Change: drop `nn.Embedding`, `Dataset`, sequence/tokenizer imports; add sklearn and the
reproducibility triple (random/np/torch); keep `device` as a torch.device object; rename the
hyperparameter block to the dense-feature MLP (INPUT_DIM=100, N_CLASSES=2, add PATIENCE).

## Cell 4 - Code: load or rebuild the 100-d averaged-word2vec features X, y

```python
# Get the B7 input: a 100-d averaged-word2vec feature matrix X and labels y for SST-2.
# Path 1: load the X.npy / y.npy you saved in the B5 homework (fast, no re-download).
# Path 2 (fallback): rebuild them here from pretrained word vectors + a SST-2 subsample.

import os

def load_saved_features():
    """Try to load the B5-homework features. Returns (X, y) or None."""
    if os.path.exists('X.npy') and os.path.exists('y.npy'):
        X = np.load('X.npy')
        y = np.load('y.npy')
        if X.shape[1] == INPUT_DIM:
            print(f"Loaded saved features from B5: X={X.shape}, y={y.shape}")
            return X, y
    return None

loaded = load_saved_features()

if loaded is not None:
    X, y = loaded
else:
    print("No saved features found. Rebuilding from pretrained word vectors (this is the B5 recipe).")
    import re
    import gensim.downloader as api
    from datasets import load_dataset

    # Pretrained word vectors, exactly as in B5: 400k words, 100 dimensions.
    wv = api.load('glove-wiki-gigaword-100')

    def doc_vector(text):
        """Mean-pool the in-vocab word vectors of a text into one 100-d vector (B5 recipe)."""
        tokens = re.findall(r"[a-z']+", text.lower())
        vecs = [wv[t] for t in tokens if t in wv.key_to_index]  # OOV guard
        if not vecs:
            return np.zeros(INPUT_DIM, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    # SST-2 sentiment: fields 'sentence' and 'label' (0=negative, 1=positive).
    ds = load_dataset('nyu-mll/glue', 'sst2', split='train')
    ds = ds.shuffle(seed=SEED).select(range(N_SAMPLES))

    X = np.vstack([doc_vector(s) for s in ds['sentence']]).astype(np.float32)
    y = np.array(ds['label'], dtype=np.int64)
    np.save('X.npy', X)  # cache so a re-run is instant
    np.save('y.npy', y)
    print(f"Rebuilt features: X={X.shape}, y={y.shape}")

print(f"Class balance (mean of y, 1=positive): {y.mean():.3f}")
print(f"Feature matrix dtype: {X.dtype}, label dtype: {y.dtype}")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 5 + 7
Change: replace the CNN-Headlines Dropbox download and TextBlob tokenizer with a load-or-rebuild
of the B5 averaged-word2vec features on SST-2; reuse the exact B5 names `wv`, `doc_vector`, `X`,
`y` and the 100-d width; OOV-guard with `key_to_index`; cache to X.npy/y.npy.

## Cell 5 - Markdown: Section 1, embeddings as features and the baseline bar

```markdown
## Section 1. Embeddings as Features, and the Bar to Beat

You now have `X`, an `(N, 100)` matrix where each row is one sentence turned into a vector by
averaging its word2vec vectors, and `y`, the sentiment label (0 = negative, 1 = positive).

This is the **embeddings as features** pattern. The pretrained word vectors are a frozen feature
extractor: we never train them. We only train a small classifier head that reads the 100-d
vector and predicts a label. That is cheap (no GPU bill per request), it works with very little
labeled data, and it is exactly what you do before deciding whether a full fine-tune is worth it.

Before we build a neural network, we need to know what "good" means. The honest way to justify a
neural network is to first fit the simplest reasonable model on the same features and treat its
score as the bar to beat. Here that model is **LogisticRegression**: a single linear layer with a
sigmoid, the same idea as one `nn.Linear` with no hidden layer. If our MLP cannot beat it, the
extra complexity is not earning its keep.

One detail that matters for both models: the 100 features live on different scales, and both
logistic regression and neural nets train better when features are standardized to zero mean and
unit variance. We fit the scaler on the TRAINING split only, then apply it to validation and test,
so no test information leaks into training.
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 6 + 15
Change: rewrite the "baseline bar" prose around the embeddings-as-features pattern and the
LogisticRegression-as-one-linear-layer framing; add the standardize-fit-on-train-only rule;
drop the TF-IDF baseline reference (we compare on the SAME averaged features).

## Cell 6 - Code: Demo, stratified split + standardize + LogisticRegression baseline

```python
# Demo: split, standardize, and fit the LogisticRegression baseline.
# We split ONCE into train / val / test (70 / 15 / 15), stratified, reproducible with SEED.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)
print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

# Standardize: fit on TRAIN only, then apply the same transform to val and test (no leakage).
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# LogisticRegression baseline on the standardized features.
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train_s, y_train)
baseline_val_acc = accuracy_score(y_val, baseline.predict(X_val_s))
print(f"\nBASELINE (LogisticRegression) validation accuracy: {baseline_val_acc:.4f}")
print("This is the bar your MLP must beat.")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cell 22
Change: keep the stratified train/val/test split idea but operate on the dense `X, y` (not a
DataFrame of headlines); add StandardScaler fit-on-train; add the LogisticRegression baseline
that NB8 only referenced as a hardcoded "~90%"; report the val accuracy as the explicit bar.

## Cell 7 - Markdown: Lab 1 instructions, measure the baseline on the test split

```markdown
### Lab 1: Lock in the baseline (guided, ~10 min)

You just saw the baseline on the VALIDATION split. To report it fairly later, also compute the
baseline accuracy and macro-F1 on the held-out TEST split, using the already-fitted `scaler` and
`baseline` model from the demo.

Steps:

1. Transform `X_test` with the SAME fitted `scaler` (do not fit again).
2. Use the fitted `baseline` to predict labels for the transformed test features.
3. Compute accuracy and macro-F1 against `y_test`.
4. Store the accuracy in `baseline_test_acc`; the verification block checks it.

Reminder: the scaler was fit on train. Re-fitting on test would leak test statistics into your
numbers. Use the transform-only step.
```

Provenance: FROM-NEW
Change: new guided lab that turns the NB8 "compare with baseline" idea into a hands-on
transform-only / predict / score exercise; sets up `baseline_test_acc` for the final comparison.

## Cell 8 - Code: Lab 1 starter

```python
# Lab 1: compute the baseline on the TEST split using the already-fitted scaler and baseline.

# Step 1: apply the fitted scaler to the test features (transform only, do NOT fit here).
X_test_baseline = None  # YOUR CODE: transform X_test with the scaler fitted on train

# Step 2: predict test labels with the fitted baseline model.
baseline_test_preds = None  # YOUR CODE: use the baseline to predict on the transformed test features

# Step 3: score against y_test.
baseline_test_acc = None  # YOUR CODE: accuracy of baseline_test_preds vs y_test
baseline_test_f1  = None  # YOUR CODE: macro-F1 of baseline_test_preds vs y_test

# --- verification (provided) ---
assert X_test_baseline is not None and baseline_test_preds is not None, "Fill in steps 1 and 2."
assert baseline_test_acc is not None, "Compute baseline_test_acc."
assert X_test_baseline.shape == X_test.shape, "Use transform (not fit_transform); shape must match X_test."
assert 0.0 <= baseline_test_acc <= 1.0, "Accuracy must be a fraction in [0, 1]."
print(f"Baseline TEST accuracy: {baseline_test_acc:.4f} | macro-F1: {baseline_test_f1:.4f}")
print("Good. Now let's see if a neural net can beat this on the SAME features.")
```

Provenance: FROM-NEW
Change: new starter. `None # YOUR CODE` hints name the ROLE ("transform with the fitted scaler",
"predict", "accuracy", "macro-F1") but never the exact call, so the line is not solvable from the
comment alone. The verification asserts shape (catches fit_transform misuse) and range.

## Cell 9 - Code: Lab 1 SOLUTION-only reference (for the Solutions notebook)

```python
# Solution reference (lives only in the Solutions notebook, never in the exercise).
X_test_baseline = scaler.transform(X_test)
baseline_test_preds = baseline.predict(X_test_baseline)
baseline_test_acc = accuracy_score(y_test, baseline_test_preds)
baseline_test_f1  = f1_score(y_test, baseline_test_preds, average='macro')
```

Provenance: FROM-NEW
Change: solution body for Lab 1; the builder places this in the Solutions notebook only. Listed
here so the builder does not have to invent it.

## Cell 10 - Markdown: Section 2, the MLP architecture (theory)

```markdown
## Section 2. The MLP: a Nonlinear Head on Frozen Features

Logistic regression draws ONE straight decision boundary in the 100-d feature space. That is all
a linear model can do. Many real boundaries are not straight, and that is exactly the gap a
neural network fills.

Our model adds one hidden layer with a nonlinear activation:

```
input (100-d averaged word2vec)
  -> Linear(100, 128)     # learn 128 combinations of the input features
  -> ReLU                 # the nonlinearity: this is what beats a linear model
  -> Dropout(0.3)         # randomly zero some hidden units during training (regularization)
  -> Linear(128, 2)       # logits for [negative, positive]
```

The hidden layer plus ReLU lets the network bend the decision boundary: it composes the 100
inputs into 128 new features, and the output layer draws its line in that bent space. That is why
an MLP can beat logistic regression on the SAME inputs.

Two rules we carry from B6 and must not break:

- The model outputs RAW LOGITS. We do NOT put a softmax in `forward`. `nn.CrossEntropyLoss`
  applies log-softmax internally, and adding our own would double-apply it and hurt training.
- The labels are INTEGER class indices (0 or 1) with dtype `long`, never one-hot vectors.

Dropout is on only during training (`model.train()`) and off during evaluation (`model.eval()`).
Forgetting `model.eval()` at test time is a classic bug that makes results noisy.
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 15 + 21
Change: rewrite the architecture diagram for the pooled-feature MLP (no Embedding, no
masked-mean-pool layer); foreground WHY nonlinearity beats logistic regression; restate the
raw-logits and integer-label rules from B6; keep the train/eval dropout caveat.

## Cell 11 - Markdown: Lab 2 instructions, wrap features in DataLoaders

```markdown
### Lab 2: Wrap the features in DataLoaders (guided, ~8 min)

A PyTorch training loop pulls mini-batches from a `DataLoader`. In B6 you built `DataLoader`s
from a `TensorDataset`. Do the same here with the standardized feature splits.

You need three things per split: float32 feature tensors, int64 (long) label tensors, and a
`TensorDataset` wrapping them. Then a `DataLoader` over each.

Steps:

1. Convert `X_train_s`, `X_val_s`, `X_test_s` to float tensors, and `y_train`, `y_val`, `y_test`
   to long tensors.
2. Build a `TensorDataset` for each split.
3. Build a `DataLoader` for each. Shuffle the TRAINING loader (use `BATCH_SIZE`); the val and
   test loaders do not need shuffling.

Why long labels? `CrossEntropyLoss` expects integer class indices, and PyTorch represents those
as the `long` dtype. Float labels raise an error here.
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cell 13
Change: re-theme the old "implement a custom Dataset for headlines" lab into a TensorDataset +
DataLoader build over dense feature tensors (the B6 pattern), dropping the sequence/padding logic
that the pooled doc vector makes unnecessary.

## Cell 12 - Code: Lab 2 starter

```python
# Lab 2: build TensorDatasets and DataLoaders from the standardized features.

# Step 1: tensors. Features must be float; labels must be long (integer class indices).
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_s,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_s,  dtype=torch.float32)
y_train_t = None  # YOUR CODE: y_train as a torch long tensor
y_val_t   = None  # YOUR CODE: y_val as a torch long tensor
y_test_t  = None  # YOUR CODE: y_test as a torch long tensor

# Step 2: datasets (pair each feature tensor with its label tensor).
train_ds = None  # YOUR CODE: TensorDataset of train features and labels
val_ds   = None  # YOUR CODE: TensorDataset of val features and labels
test_ds  = None  # YOUR CODE: TensorDataset of test features and labels

# Step 3: loaders. Shuffle only the training loader.
train_loader = None  # YOUR CODE: DataLoader over train_ds, batch_size=BATCH_SIZE, shuffled
val_loader   = None  # YOUR CODE: DataLoader over val_ds, batch_size=BATCH_SIZE
test_loader  = None  # YOUR CODE: DataLoader over test_ds, batch_size=BATCH_SIZE

# --- verification (provided) ---
assert y_train_t is not None and y_train_t.dtype == torch.long, "Labels must be torch.long."
assert train_ds is not None and len(train_ds) == X_train_s.shape[0], "train_ds size mismatch."
xb, yb = next(iter(train_loader))
assert xb.shape[1] == INPUT_DIM, f"Each batch row must be {INPUT_DIM}-d."
assert yb.dtype == torch.long, "Batched labels must be long."
print(f"Loaders ready. One train batch: features {tuple(xb.shape)}, labels {tuple(yb.shape)}.")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cell 14
Change: starter for the DataLoader build. Hints name ROLE ("as a torch long tensor",
"DataLoader over train_ds ... shuffled") not the exact constructor args order; verification
asserts dtype long (catches the classic float-label bug) and feature width 100.

## Cell 13 - Code: Demo, define the SentimentMLP model

```python
# Demo: the model. A plain dense MLP over the 100-d averaged-word2vec features.
class SentimentMLP(nn.Module):
    """MLP head on frozen averaged-word2vec features: Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 n_classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: [batch, input_dim] -> logits: [batch, n_classes]. No softmax here on purpose.
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)

# Quick shape sanity check with a dummy batch (no training yet).
demo_model = SentimentMLP().to(device)
dummy = torch.randn(4, INPUT_DIM, device=device)
out = demo_model(dummy)
print(f"Dummy forward: input {tuple(dummy.shape)} -> logits {tuple(out.shape)} (expect (4, 2)).")
print("Logits are raw scores, not probabilities. CrossEntropyLoss handles the softmax.")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 16 + 18
Change: replace `NewsClassifier` (embedding + masked-mean-pool, `forward(input_ids, mask)`) with
`SentimentMLP` operating on already-pooled dense vectors, `forward(x)`; keep the dummy-forward
shape check; emphasize raw logits.

## Cell 14 - Code: Demo, count parameters + train/eval helpers

```python
# Demo: parameter count and the train/eval helpers (the B6 training loop, packaged).
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Trainable parameters: {count_parameters(demo_model):,}")
print("Tiny model: every parameter is in the two Linear layers (the word vectors are frozen).")

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One training pass; returns average loss."""
    model.train()  # dropout ON
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader, loss_fn, device):
    """Evaluate; returns (average loss, accuracy)."""
    model.eval()  # dropout OFF
    total = 0.0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total += loss_fn(logits, yb).item()
            preds_all.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_all.extend(yb.cpu().numpy())
    return total / len(loader), accuracy_score(labels_all, preds_all)

print("Helpers ready: train_one_epoch, evaluate.")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 17 + 24
Change: simplify `count_parameters` (no embedding-vs-other breakdown); adapt `train_one_epoch`
and `evaluate` to `(features, labels)` batches with `forward(x)` (drop the mask argument and the
`.squeeze()` on labels); pass `loss_fn` explicitly.

## Cell 15 - Code: Lab 3 part A, instantiate model + loss + optimizer

```python
# Lab 3a: create the real model, the loss function, and the optimizer.
torch.manual_seed(SEED)  # re-seed so weight init is reproducible

model = None      # YOUR CODE: a SentimentMLP, moved to device
loss_fn = None    # YOUR CODE: the loss for multi-class logits + integer labels
optimizer = None  # YOUR CODE: Adam over the model parameters with learning rate LR

# --- verification (provided) ---
assert isinstance(model, SentimentMLP), "model must be a SentimentMLP."
assert next(model.parameters()).device.type == device.type, "model must be on `device`."
assert loss_fn is not None and optimizer is not None, "Set loss_fn and optimizer."
_xb, _yb = next(iter(train_loader))
_loss = loss_fn(model(_xb.to(device)), _yb.to(device))
print(f"Setup OK. One untrained batch loss: {_loss.item():.4f} (around ln(2)=0.69 before training).")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 19 + 20 + 23
Change: drop the "load pretrained embedding weights" step (no embedding layer); reduce to the
B6 triple model/loss/optimizer. Hints name ROLE ("the loss for multi-class logits + integer
labels", "Adam over the model parameters with learning rate LR") not `nn.CrossEntropyLoss` or the
exact `torch.optim.Adam(...)` call; verification checks type, device, and a sane initial loss.

## Cell 16 - Code: Lab 3 part B, training loop with early stopping

```python
# Lab 3b: the training loop with early stopping. Save the best checkpoint by val accuracy.
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0
patience_counter = 0
best_path = 'best_b7_mlp.pt'

for epoch in range(EPOCHS_MAX):
    train_loss = None          # YOUR CODE: run one training epoch (use the helper)
    val_loss, val_acc = None, None  # YOUR CODE: evaluate on the validation loader

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"Epoch {epoch+1:02d} | train {train_loss:.4f} | val {val_loss:.4f} | val acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), best_path)  # save best weights
        print("  improved -> checkpoint saved")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  no improvement for {PATIENCE} epochs -> early stop")
            break

print(f"\nBest validation accuracy: {best_val_acc:.4f}")
print(f"Baseline to beat (val): {baseline_val_acc:.4f}")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 25 + 26
Change: keep the early-stopping loop structure but only the two `None # YOUR CODE` lines are the
calls to the provided helpers (train_one_epoch and evaluate); the early-stopping bookkeeping is
PROVIDED (it is mechanical and not the lesson), so the lab focuses on wiring the helpers and
reading val accuracy against the baseline. Hints name ROLE only.

## Cell 17 - Code: Demo, training curves + reload best checkpoint

```python
# Demo: plot the learning curves, then reload the best checkpoint for a fair test.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(history['train_loss'], label='train')
ax1.plot(history['val_loss'], label='val')
ax1.set_title('Loss'); ax1.set_xlabel('epoch'); ax1.legend(); ax1.grid(alpha=0.3)
ax2.plot(history['val_acc'], label='val acc', color='green')
ax2.axhline(baseline_val_acc, color='red', linestyle='--', label='LogReg baseline')
ax2.set_title('Validation accuracy vs baseline'); ax2.set_xlabel('epoch')
ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# Reload the best weights (the last epoch is not always the best). weights_only=True is the
# safe way to load a state_dict.
model.load_state_dict(torch.load(best_path, weights_only=True))
print("Reloaded best checkpoint. If val accuracy crossed the red line, the MLP beat the baseline.")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 27 + 28
Change: add the baseline accuracy as a dashed reference line on the accuracy plot (the whole
point of the notebook); add `weights_only=True` to the checkpoint reload per current PyTorch
guidance.

## Cell 18 - Markdown + Code: Section 3, test evaluation vs baseline

```markdown
## Section 3. Did We Beat the Baseline?

The test set is the final exam: we touch it once, after all training and model selection. We
report accuracy (SST-2 is roughly balanced, so accuracy is meaningful) and macro-F1 (the average
of the per-class F1 scores), and we put the MLP and the baseline side by side.
```

```python
# Test the reloaded best model and compare to the LogisticRegression baseline.
test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

# Gather predictions for the report and confusion matrix.
model.eval()
mlp_preds, mlp_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        mlp_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        mlp_labels.extend(yb.numpy())

mlp_test_f1 = f1_score(mlp_labels, mlp_preds, average='macro')

print("=== Test results ===")
print(f"MLP        accuracy: {test_acc:.4f} | macro-F1: {mlp_test_f1:.4f}")
print(f"Baseline   accuracy: {baseline_test_acc:.4f} | macro-F1: {baseline_test_f1:.4f}")
print(f"Lift (MLP - baseline): {test_acc - baseline_test_acc:+.4f}")
print()
print(classification_report(mlp_labels, mlp_preds, target_names=['negative', 'positive']))
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 29 + 30 + 31 + 37
Change: condense the "why test once" md and the eval section into one short md + one code cell;
switch target names to negative/positive (binary SST-2); print the explicit MLP-vs-baseline lift
instead of comparing to a hardcoded TF-IDF number.

## Cell 19 - Code: Demo, confusion matrix

```python
# Demo: confusion matrix on the test set.
cm = confusion_matrix(mlp_labels, mlp_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['neg', 'pos'], yticklabels=['neg', 'pos'])
plt.xlabel('predicted'); plt.ylabel('true'); plt.title('Confusion Matrix (test)')
plt.tight_layout(); plt.show()

# Which direction does the model err more? That hints at what the averaged features miss.
fn = cm[1, 0]  # true positive predicted negative
fp = cm[0, 1]  # true negative predicted positive
print(f"Positives called negative: {fn} | Negatives called positive: {fp}")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cell 32
Change: shrink to a 2x2 binary confusion matrix; replace the 4-class news labels and the
"most confused pair" logic with a neg/pos error-direction readout.

## Cell 20 - Code: Demo, predict_sentiment inference helper

```python
# Demo: a tiny inference function. Text -> doc_vector -> scale -> model -> label.
# This is the embeddings-as-features pipeline end to end, the shape a deployed service has.
import torch.nn.functional as Fnn

def predict_sentiment(text):
    """Return (label_str, confidence) for a raw sentence using the trained MLP."""
    vec = doc_vector(text).reshape(1, -1)          # 100-d averaged word2vec (B5 recipe)
    vec = scaler.transform(vec)                    # SAME scaler fit on train
    t = torch.tensor(vec, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        probs = Fnn.softmax(model(t), dim=1)[0]    # softmax here, for a readable confidence
    label = 'positive' if probs[1] > probs[0] else 'negative'
    return label, float(probs.max())

for s in ["a delightful, warm-hearted film", "a boring, lifeless mess"]:
    label, conf = predict_sentiment(s)
    print(f"{label:8s} ({conf:.2f})  <-  {s}")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cell 33
Change: rewrite `predict_headline` (which embedded token-id sequences) as `predict_sentiment`
that reuses `doc_vector` + `scaler` + the trained MLP; note the softmax is applied HERE only to
report a human-readable confidence (not inside the model).

## Cell 21 - Markdown + Code: Lab 4, find where averaging fails

```markdown
### Lab 4: Break the model on purpose (guided, ~10 min)

Averaging word vectors throws away word ORDER. "not good" and "good not" average to the same
vector, and "not good" shares almost every word with "good". So the model should struggle with
negation and word order.

Your task: write three short sentences where word order or negation flips the true sentiment, run
them through `predict_sentiment`, and find at least one the model gets WRONG. Then write one
sentence in the `note` string explaining why averaged features cannot capture it.

You are hunting for a failure, not avoiding one. A clean negation like "this is not good at all"
is a great candidate.
```

```python
# Lab 4: adversarial sentences that stress word order / negation.
adversarial = [
    None,  # YOUR CODE: a sentence whose sentiment depends on a negation word
    None,  # YOUR CODE: a sentence whose sentiment depends on word order
    None,  # YOUR CODE: your own tricky case
]
note = None  # YOUR CODE: one sentence on why averaging loses this information

# --- verification (provided) ---
assert all(isinstance(s, str) and s.strip() for s in adversarial), "Provide three real sentences."
assert isinstance(note, str) and len(note) > 20, "Write a one-sentence explanation in `note`."
for s in adversarial:
    label, conf = predict_sentiment(s)
    print(f"{label:8s} ({conf:.2f})  <-  {s}")
print(f"\nYour explanation: {note}")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 34 + 35 + 36
Change: fold the misclassification-analysis demo and the adversarial-headlines lab into one
sentiment-flavored Lab 4 that targets the negation/order weakness; the `None # YOUR CODE` lines
ask for student-authored sentences and an explanation, never naming a method.

## Cell 22 - Markdown + Code: Section 4, save the model for inference

```markdown
## Section 4. Save It Like You Mean to Deploy It

To reuse this classifier later you need TWO artifacts, not one:

1. The model weights, saved as a `state_dict` (portable, the recommended way; saving the whole
   pickled model object is brittle across versions).
2. The fitted `StandardScaler`. Inference must apply the exact same scaling that training used,
   so the scaler is part of the model, not an afterthought.

In production you would also precompute and cache the document vectors for any static corpus, so
serving a request is just `doc_vector -> scale -> tiny MLP`, which is fast and cheap precisely
because the encoder is frozen.
```

```python
# Save both artifacts.
import pickle
torch.save(model.state_dict(), 'b7_sentiment_mlp.pt')
with open('b7_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved b7_sentiment_mlp.pt (weights) and b7_scaler.pkl (scaler).")
print("To load: rebuild SentimentMLP, load_state_dict(..., weights_only=True), call model.eval().")
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 41 + 42
Change: keep the save-for-reuse idea but save the `state_dict` plus the `scaler` (the NB8 version
pickled a vocab and hyperparameters for the embedding model); add the precompute/cache production
note and the load recipe with `weights_only=True` and `model.eval()`.

## Cell 23 - Markdown: Wrap-up, homework (SBERT lift), stretch, bridges

```markdown
## Wrap-up and Homework

### What you did

- Turned text into 100-d averaged word2vec features (the B5 recipe), standardized them, and fit
  a LogisticRegression baseline: the bar to beat.
- Built and trained an MLP head with the B6 machinery and BEAT that baseline on the same features.
- Saw, in a confusion matrix and a negation example, exactly where averaging hits its ceiling.
- Saved the model and scaler the way a deployable service needs them.

The big idea: **embeddings as features**. A frozen pretrained encoder plus a small trained head
is the cheapest useful classifier you can ship, and it is the exact pattern Part C upgrades by
making the encoder trainable.

### Homework Extension (async, the real lift): swap in contextual sentence embeddings

The MLP's ceiling is the REPRESENTATION, not the model. Averaged static word vectors cannot see
order or context. Contextual sentence embeddings can. Prove it:

1. Encode the SAME SST-2 sentences with the B5 sentence encoder
   `embedder = SentenceTransformer('all-MiniLM-L6-v2')` to get a 384-d feature matrix
   `X_sbert` (shape `(N, 384)`). The labels `y` are unchanged.
2. Re-run the EXACT pipeline: stratified split, `StandardScaler`, the same `SentimentMLP` but
   with `input_dim=384`, the same training loop and early stopping.
3. Compare test accuracy. On SST, averaged GloVe lands around 80 percent; an SBERT-style sentence
   embedding lands closer to 89 percent. Report YOUR lift and write two sentences on why the
   contextual model wins.

This is the bridge to Part C: SBERT is a frozen contextual encoder; C9 goes one step further and
FINE-TUNES the encoder (DistilBERT) on SST-2, which beats even SBERT, and that fine-tuned model is
what the Gradio chatbot loads.

### Stretch options (pick one, in-notebook)

- **More capacity, more regularization**: add a second hidden layer and try `Adam(..., weight_decay=1e-4)`.
  Does accuracy improve, or does the averaged-feature ceiling cap it regardless? (Usually the
  representation, not the model size, is the limit here.)
- **Freeze vs scale ablation**: retrain WITHOUT the StandardScaler and compare convergence and
  final accuracy. Quantify how much scaling bought you.
- **Hyperparameter sweep**: vary `HIDDEN_DIM` over {32, 128, 256} and plot val accuracy. Report
  the point of diminishing returns.

### Production take-homes

- Precompute and cache embeddings for static corpora; serve only the small head online.
- Use frozen features for low-data / quick-prototype settings; fine-tune when you have the data
  and compute and need task-specific accuracy.
- A softmax confidence is often miscalibrated. For a deployed classifier, calibrate the scores
  and tune the decision threshold, and monitor the confidence distribution for drift.

### Bridge to B8 (Capstone B)

You own the full pipeline now: pretrained features in, trained classifier out, baseline beaten,
model saved. Capstone B runs it end to end on a fresh dataset. Then Part C swaps the frozen
encoder for a fine-tuned transformer.

### Resources

- gensim KeyedVectors: https://radimrehurek.com/gensim/models/keyedvectors.html
- PyTorch CrossEntropyLoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Saving for inference: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- Sentence-BERT (Reimers and Gurevych, 2019): https://arxiv.org/abs/1908.10084
```

Provenance: FROM-OLD 3-Text-Classification/8-MLP_Text_Classification.ipynb cells 38 + 39 + 40 + 43
Change: replace the embedding freeze/fine-tune ablation and the hyperparameter mini-experiment
with a homework that swaps the 100-d averaged word2vec input for 384-d SBERT and measures the
lift (the load-bearing bridge to C9); fold the old ablation and hyperparameter ideas into clearly
labelled in-notebook stretch options; add the production take-homes and the B8/C9 bridges.

# VERIFICATION CHECKLIST

- [ ] 24 cells total (0-23), within the 20-25 budget.
- [ ] Input representation resolved EXPLICITLY: 100-d averaged word2vec X, y (NOT 384-d SBERT);
      SBERT is the homework lift. INPUT_DIM = 100 is asserted by the feature load/rebuild cell.
- [ ] numpy<2 AND scipy<1.13 AND gensim==4.3.3 AND datasets<3 pinned in the install cell (cell 2).
- [ ] Colab restart-runtime note present (cell 1) for the gensim/scipy/numpy combination.
- [ ] Names reused verbatim from B5/B6: wv, embedder, doc_vector, X, y, device (torch.device), SEED=42.
      No redefinition that collides; old NB8 names (NewsClassifier, embedding_matrix, word_to_idx) dropped.
- [ ] Feature load-or-rebuild path works both ways: loads X.npy/y.npy if present, else rebuilds via
      gensim glove-wiki-gigaword-100 + doc_vector on SST-2 (nyu-mll/glue, config sst2).
- [ ] StandardScaler fit on TRAIN only, transform on val/test (Lab 1 verification asserts shape, catching fit_transform misuse).
- [ ] LogisticRegression baseline computed; baseline_val_acc and baseline_test_acc are the explicit bar.
- [ ] MLP beats the baseline (val accuracy plotted against the baseline reference line).
- [ ] SentimentMLP = Linear(100,128) -> ReLU -> Dropout(0.3) -> Linear(128,2); forward returns RAW logits (no softmax).
- [ ] CrossEntropyLoss with long (int64) labels; Lab 2 verification asserts label dtype long.
- [ ] Training loop with early stopping saves best state_dict; reload uses weights_only=True; model.eval() before test.
- [ ] Every concept has a Demo cell AND a Lab cell; never more than 3 markdown cells in a row.
- [ ] Lab 1, Lab 2, Lab 3 (a+b), Lab 4 each have None # YOUR CODE starters plus a provided verification.
- [ ] No # YOUR CODE comment names the exact method/constructor that solves it in the same line.
- [ ] Lab 4 surfaces the negation/order failure (the averaging ceiling) honestly.
- [ ] Save section saves BOTH the state_dict and the fitted scaler (scaler is part of the model).
- [ ] Homework swaps 100-d word2vec for 384-d SBERT and measures the lift (the bridge to C9).
- [ ] STAR story and chatbot through-line sections present; B8 and C9 bridges stated in-text.
- [ ] Runs top-to-bottom on Colab CPU in a few minutes (N_SAMPLES=2000, tiny MLP).
- [ ] Plain ASCII only: no em/en dashes, no Unicode multiply sign, no emoji.

# RESEARCH VALIDATED (June 2026)

- Averaged word2vec/GloVe document vectors fed to a classifier is the standard "average word2vec"
  text-classification baseline; an MLP head adds nonlinearity over the same features.
  https://towardsdatascience.com/the-word2vec-classifier-5656b04143da/
  https://github.com/sdimi/average-word2vec
- Standardize features (StandardScaler) before BOTH logistic regression and an MLP; gradient
  descent is scale-sensitive. Fit on train, transform val/test (fit_transform on test leaks).
  https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
  https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
- glove-wiki-gigaword-100 is 100-d static word vectors (averaged for a doc vector); all-MiniLM-L6-v2
  is 384-d contextual sentence embeddings. The "MLP on Word2Vec" input is the 100-d averaged path.
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html
- An MLP with a hidden layer plus a nonlinear activation can form decision boundaries logistic
  regression cannot (the XOR argument), so it can beat LR on identical features.
  https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7/
- Averaging word embeddings discards word order (bag of words); contextual transformers with
  attention and positional information fix this. This is the explicit motivation for Part C.
  https://www.tandfonline.com/doi/full/10.1080/19312458.2018.1455817
  https://arxiv.org/pdf/1908.10084
- CrossEntropyLoss expects RAW logits (no softmax in the model) and INTEGER class indices of dtype
  long (not one-hot); applying softmax before it double-applies and hurts stability.
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html
  https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html
- Early stopping with patience plus dropout prevents overfitting on small datasets; save the best
  checkpoint by validation metric and reload it before testing.
  https://www.geeksforgeeks.org/deep-learning/using-early-stopping-to-reduce-overfitting-in-neural-networks/
- Reproducible stratified split: train_test_split(..., stratify=y, random_state=SEED), applied twice
  for train/val/test; preserves class balance.
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- SST-2 (GLUE) has 67,349 train rows, roughly balanced (~55.8% positive / 44.2% negative), 872 val;
  load via nyu-mll/glue config sst2 (Parquet, no trust_remote_code on datasets<3); fields sentence, label.
  https://huggingface.co/datasets/nyu-mll/glue
  https://docs.pytorch.org/text/stable/_modules/torchtext/datasets/sst2.html
- SBERT vs averaged static embeddings on SST: averaged GloVe ~80%, SBERT-NLI ~89%, a real +8-9 point
  lift that motivates contextual encoders (the homework payoff and the C9 bridge).
  https://medium.com/@kashyapkathrani/all-about-embeddings-829c8ff0bf5b
- "Embeddings as features" (frozen encoder + small head) is the canonical low-data / low-compute
  transfer-learning step; fine-tune the encoder when you have data/compute (the frozen-then-finetune
  curriculum order). Frozen encoder + linear/MLP head is standard for sentiment.
  https://medium.com/@utsavsharma1990/frozen-encoders-foundations-applications-and-advanced-techniques-5b17688dd09a
  https://wandb.ai/wandb_fc/genai-research/reports/Transfer-learning-versus-fine-tuning--VmlldzoxNDQxOTM3OQ
- Production: precompute/cache embeddings offline so online inference is just the small head (sub-100ms,
  big cost win); softmax confidence is often miscalibrated, so calibrate and monitor for drift.
  https://discuss.huggingface.co/t/how-can-i-reduce-latency-when-running-a-large-transformer-model-for-sentence-embeddings-in-production/172103
  https://link.springer.com/article/10.1007/s10994-023-06336-7
- Save a state_dict (not the full pickled model) for inference; load with weights_only=True and call
  model.eval() before predicting; persist the StandardScaler alongside the weights.
  https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- datasets 4.0 dropped script-based loaders; pin datasets<3 to keep the GLUE/SST-2 loader stable,
  consistent with B5.
  https://discuss.huggingface.co/t/dataset-scripts-are-no-longer-supported/163891
- gensim 4.3 breaks on scipy>=1.13 (scipy.linalg.triu removed); pin scipy<1.13. Colab ships numpy 2.x
  so a runtime restart after install may be required (carry the B5 note).
  https://github.com/piskvorky/gensim/issues/3525
