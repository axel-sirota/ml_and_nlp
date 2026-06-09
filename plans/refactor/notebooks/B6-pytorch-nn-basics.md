# B6 - PyTorch NN Basics - Cell-by-Cell Plan

## Status

TOUCHED. Source notebook: `Frameworks/PyTorch_Exercises.ipynb` (33 cells, 6 sections plus an
optional end-to-end lab). We lift the four sections B7 actually needs (Layers, Sequential models,
custom `nn.Module`, the training loop) and the freezing section (folded into homework), retheme the
MNIST/sine story to the chatbot arc, swap MNIST images for a toy feature matrix that mimics B5's
word2vec document vectors, and fold the dual-branch / custom-loss / freezing material into a single
stretch + homework so the notebook stays inside the cell budget.

New file: `B-How-It-Works/6-PyTorch_NN_Basics.ipynb`. Slug: `pytorch-nn-basics`.

## Context

Students arrive from B4 (PyTorch tensors and autograd) and B5 (word2vec + sentence embeddings).
They already have: `SEED = 42`, `torch.manual_seed(SEED)`, `np.random.seed(SEED)`, the
`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` idiom (a `torch.device`
object, not a pipeline int), `torch.tensor`, `.reshape/.view`, broadcasting, `requires_grad`,
`.backward()`, `.grad`, and `torch.no_grad()` from B4. From B5 they have `wv` (a 100-d
`KeyedVectors`), `embedder` (a 384-d `SentenceTransformer`), the `doc_vector(text)` helper, and the
saved `X` (shape `(N, 100)`) / `y` feature matrix plus a LogisticRegression baseline accuracy.

B4 deliberately DROPPED `nn.Module`, loss, optimizer, `TensorDataset`, and `DataLoader` on the
explicit promise that they "move to B6 where they are first used." B6 pays off that promise. The
B4 import cell dropped the `from torch.utils.data import TensorDataset, DataLoader` line for the
same reason; B6 reintroduces it.

Key insight they leave with: a neural network is a stack of `nn.Linear` layers wired in a class,
and the same six-line loop (`zero_grad -> forward -> loss -> backward -> step`) trains every model
in the rest of the course, from this toy MLP to the DistilBERT fine-tune in C9. They build and
train a small MLP end to end on a toy feature matrix, so B7 is "just swap in real word2vec
features."

New names introduced here (reused downstream, not redefined): `nn.Module`, `nn.Linear`, `nn.ReLU`,
`nn.Sequential`, `nn.Dropout`, `nn.CrossEntropyLoss`, `torch.optim.Adam`, `torch.optim.AdamW`,
`TensorDataset`, `DataLoader`, `model.train()`, `model.eval()`, the toy `X_toy` / `y_toy` matrix,
and `MLPClassifier` (the `nn.Module` subclass B7 reuses verbatim).

## Chatbot Through-Line

The course end goal is a fine-tuned DistilBERT in a simple Gradio chatbot. B6 contributes the
assembly manual: how to turn tensors into a trained model. The one-line story is "the six-line
training loop you write here is the literal engine that fine-tunes the chatbot's model." This is
exact, not a metaphor. `DistilBertForSequenceClassification` is an encoder plus a `pre_classifier`
(`nn.Linear`) plus ReLU plus dropout plus a `classifier` (`nn.Linear`) head trained with
`nn.CrossEntropyLoss` on the pooled output. That head IS the toy MLP students build in this
notebook. The HuggingFace `Trainer` that C9 uses runs the same `zero_grad -> forward -> loss ->
backward -> step` loop internally. So when a student trains the toy classifier here, they are
running the same machine that will fine-tune DistilBERT in C9.

Bridge to B7: "Keep this exact `MLPClassifier` and this exact loop. In B7 you delete the toy
`make_classification` matrix and feed in the word2vec document vectors from B5 instead, then beat
the LogisticRegression baseline you measured." Bridge to C9: "the same loop, with DistilBERT's
encoder in front of this head, becomes the fine-tune behind the chatbot."

## STAR Story

- Situation: You are the practitioner from Part A and B. You turned text into tensors (B4) and into
  geometry (B5): every support ticket is now a fixed-size float vector, and you have a
  LogisticRegression baseline on those vectors. The team wants to beat that baseline with a neural
  network, but you have never assembled or trained one in PyTorch.
- Task: Learn just enough of `torch.nn` to build a small classifier and train it end to end: define
  layers, snap them into a model class, pick a loss and optimizer, batch the data, and write the
  training loop. No deep dive, only the bricks B7 and C9 snap together.
- Action: Work four short theory-demo-lab cycles. Create `nn.Linear` and activation layers and read
  their parameter shapes; subclass `nn.Module` to define an MLP with `__init__` and `forward`; pick
  `nn.CrossEntropyLoss` and `torch.optim.Adam`; wrap a toy feature matrix in `TensorDataset` /
  `DataLoader`; then write the six-line loop and watch loss fall and accuracy rise. Finish by
  training the whole thing end to end on a toy stand-in for B5's word2vec vectors.
- Result: You can define an `nn.Module` classifier and train it with the canonical PyTorch loop.
  You have an `MLPClassifier` class and a loop that B7 reuses verbatim on real word2vec features,
  and you understand that the same loop fine-tunes DistilBERT for the chatbot in C9.

## Deliverables

- Exercise: `B-How-It-Works/6-PyTorch_NN_Basics.ipynb`
- Solution: `Solutions/B-How-It-Works/6-PyTorch_NN_Basics.ipynb`

## Session Timing (~60-90 min)

| Block | Cells | Minutes |
|-------|-------|---------|
| Intro + STAR + setup (install, imports, toy data) | 0-4 | 10 |
| Cycle 1: Layers (theory, demo, Lab 1) | 5-7 | 12 |
| Cycle 2: nn.Module model (theory, demo, Lab 2) | 8-10 | 14 |
| Cycle 3: Loss + optimizer + DataLoader (theory, demo, Lab 3) | 11-13 | 12 |
| Cycle 4: Training loop (theory, demo, Lab 4 + verification) | 14-16 | 16 |
| End-to-end toy MLP + stretch (validation split) | 17-19 | 12 |
| Homework (transfer learning freeze/unfreeze) + recap + bridge | 20-21 | async + 4 |

Total in-class: about 76 minutes, inside the 60-90 window. The homework cell is async.

## Cell Migration Map

Old source: `Frameworks/PyTorch_Exercises.ipynb`. Action is keep / edit / drop / merge.

| Old cell id (idx) | New cell | Action |
|-------------------|----------|--------|
| ba37ca17 (0) | Cell 0 | edit - retheme intro from CV-startup/MNIST to the support-team chatbot arc |
| b168f35b (1) | Cell 1 | edit - keep setup heading, add B4/B5 carry-forward + restart note |
| 4450bef2 (2) | Cell 2 | edit - drop torch/torchvision install (Colab has torch), pin `numpy<2` |
| a7f4f1ed (3) | Cell 3 | edit - keep seeds + device; drop F/torchvision; ADD TensorDataset/DataLoader import |
| 180ca89c (4) | Cell 4 | edit - retheme "what are we building" to toy feature vectors -> B7 word2vec |
| 34c4581a (5) | Cell 4 | merge - replace MNIST preview with make_classification toy matrix preview |
| 64ec4402 (6) | Cell 5 | edit - keep layers theory, drop Conv2d row (no images), keep Linear/ReLU/Dropout |
| 0b5c67c0 (7) | Cell 6 | edit - demo Linear/ReLU/Dropout on a feature vector; drop Conv2d block |
| 0909e3c7 (8) | Cell 7 | edit - Lab 1 retargeted to feature-dim layers (drop conv task) |
| 3225d9cc (9) | Cell 7 | merge - Lab 1 starter retargeted, conv removed, verification kept |
| d726877f (10) | Cell 8 | edit - Sequential theory, shapes traced on (B, D) not (B,1,28,28) |
| aacd3462 (11) | Cell 9 | edit - demo Sequential MLP on toy feature dim |
| 92c96a2c (12) | Cell 8 | merge - fold Sequential lab intent into the nn.Module section intro |
| a36fe9fe (13) | drop | Sequential-only lab dropped; nn.Module subclass lab subsumes it |
| 344dddd9 (14) | Cell 8 | edit - nn.Module subclassing theory (super().__init__, forward) |
| bcc94f9d (15) | Cell 9 | edit - demo: MLPClassifier (single clean class, drop dual-branch complexity) |
| 13a6e0dd (16) | Cell 10 | edit - Lab 2 instructions: implement MLPClassifier (drop residual framing) |
| dc1bc2c3 (17) | Cell 10 | merge - Lab 2 starter is the MLPClassifier skeleton + verification |
| a7115e5d (18) | Cell 11 | edit - retheme custom-loss theory to "loss + optimizer choice" |
| ce46960e (19) | Cell 12 | merge - demo loss+optimizer on toy logits (drop sine dataset) |
| ab34aeeb (20) | drop | custom Huber/regularized-loss lab dropped (too deep for B6 budget) |
| a39764b9 (21) | drop | custom-loss starter dropped |
| 3f1f07e8 (22) | Cell 11 | merge - training-loop theory (the six lines) folded with loss/optimizer theory |
| e191b15f (23) | Cell 15 | edit - demo training loop on the toy DataLoader (drop sine, use classifier) |
| e99ac07b (24) | Cell 16 | edit - Lab 4 instructions: write the six-line loop + eval |
| 13bd65df (25) | Cell 16 | merge - Lab 4 starter is the loop skeleton + accuracy verification |
| 4c1e03a8 (26) | Cell 20 | edit - freezing theory moved into the homework cell |
| d87a3d58 (27) | Cell 20 | merge - freeze demo folded into homework guidance |
| 8e1adeac (28) | Cell 20 | merge - Lab 6 freeze tasks become the homework extension |
| 7be26845 (29) | Cell 20 | merge - freeze/unfreeze starter folded into homework |
| 137d60f0 (30) | Cell 17 | edit - "end-to-end" framing kept; toy MLP end-to-end demo |
| b3859256 (31) | Cell 17 | merge - end-to-end transfer lab simplified to toy end-to-end train |
| 10a2e758 (32) | Cell 21 | edit - recap table retargeted to the four kept sections + bridge to B7/C9 |
| (none) | Cell 13 | FROM-NEW - Lab 3 (loss + optimizer + DataLoader) lab, new for the toy arc |
| (none) | Cell 14 | FROM-NEW - training-loop theory ("six lines") as its own cell |
| (none) | Cell 18 | FROM-NEW - end-to-end toy training Lab instructions |
| (none) | Cell 19 | FROM-NEW - stretch: validation split + best-val tracking |

---

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~22 cells)

## Cell 0 - Markdown: Title and welcome

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell ba37ca17
Change: retheme the intro away from the CV-startup/MNIST framing to the support-team chatbot arc;
drop the Conv2d bullet; point forward to B7 and C9.

```markdown
# PyTorch NN Basics - Build and Train a Model

*ML & NLP course - Data Trainers LLC - Axel Sirota*

## From tensors to a trained model

In B4 you turned text into tensors and met autograd: `.backward()` and `.grad`. In B5 you turned
text into geometry: every support ticket is now a fixed-size float vector, and you measured a
LogisticRegression baseline on those vectors. This notebook is the assembly manual. You will snap
the tensor bricks into a small neural network and train it end to end.

You will not need a GPU or a big dataset. We train on a small toy feature matrix that stands in for
the word2vec document vectors from B5. In B7 you will keep this exact model and this exact training
loop and just swap the toy matrix for the real word2vec features.

## What you will practice

1. **Layers** - `nn.Linear`, `nn.ReLU`, `nn.Dropout`, and reading their parameter shapes.
2. **Models** - subclassing `nn.Module` (`__init__` + `forward`) and `nn.Sequential`.
3. **Loss and optimizer** - `nn.CrossEntropyLoss` (raw logits, integer labels) and `Adam`.
4. **Batching** - `TensorDataset` and `DataLoader`.
5. **The training loop** - the six lines that train every model in this course.

## Why this matters for the chatbot

The end goal of the course is a fine-tuned transformer in a Gradio chatbot. The classification head
that DistilBERT fine-tunes in C9 is a `nn.Linear` layer trained with `nn.CrossEntropyLoss`: it is
exactly the toy model you build here, and the HuggingFace trainer runs exactly the loop you write
here. Learn it once on a toy problem; reuse it on real features in B7 and on DistilBERT in C9.

Runtime: about 60 to 90 minutes for all exercises.
```

## Cell 1 - Markdown: Section 0 - Environment setup

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell b168f35b
Change: keep the setup heading; add the B4/B5 carry-forward note and the Colab restart-runtime note
that the sister notebooks use.

```markdown
## Section 0 - Environment setup

Run the next two cells first. Colab already ships PyTorch, so we only pin `numpy<2` to stay
consistent with the gensim and scipy versions used in B5. Because Colab preinstalls numpy 2.x, you
must restart the runtime after the install cell (Runtime -> Restart runtime), then run the import
cell. We reuse the same `SEED = 42` and `device` idiom you set up in B4, and we bring back the
`TensorDataset` and `DataLoader` imports that B4 deferred to this notebook.
```

## Cell 2 - Code: Install

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell 4450bef2
Change: drop the `torch torchvision` install (Colab already has a numpy-2-compatible torch 2.x);
pin `numpy<2`; add the restart comment.

```python
# Colab already has torch; we only pin numpy<2 for consistency with B5 (gensim/scipy).
# After this cell runs, restart the runtime (Runtime -> Restart runtime), then run the imports.
!pip install -q "numpy<2"
```

## Cell 3 - Code: Imports, seeds, device

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell a7f4f1ed
Change: keep seeds and device detection; drop the `torch.nn.functional as F` and torchvision usage;
ADD `from torch.utils.data import TensorDataset, DataLoader` (the import B4 deferred to B6).

```python
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---- reproducibility (same SEED as B4/B5) ----
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- device: a torch.device object, same idiom as B4 ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch version : {torch.__version__}")
print(f"Using device    : {device}")
print("Setup complete. nn.Module, loss, optimizer, and DataLoader were promised in B4 - here they are.")
```

## Cell 4 - Code: Toy feature matrix (stand-in for B5 word2vec vectors)

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell 34c4581a
Change: replace the MNIST image preview with a `make_classification` toy matrix preview; use 100
features so the shape matches B5's 100-d `doc_vector` output, making the B7 swap literal. Keep a
small visualization.

```python
# Toy dataset: a float feature matrix that mimics B5's word2vec document vectors.
# In B5 each text became a 100-dim vector (doc_vector). Here we fake 100-dim vectors with
# make_classification so you can focus on the model and the training loop. In B7 you delete
# this cell and load the real word2vec X, y instead.
from sklearn.datasets import make_classification

N_FEATURES = 100   # same width as B5's doc_vector output (glove-wiki-gigaword-100)
N_CLASSES  = 2     # binary, like the SST-2 sentiment task in Part C

X_np, y_np = make_classification(
    n_samples=2000,
    n_features=N_FEATURES,
    n_informative=20,
    n_redundant=10,
    n_classes=N_CLASSES,
    random_state=SEED,
)
X_np = X_np.astype(np.float32)   # features must be float32 for nn.Linear
y_np = y_np.astype(np.int64)     # labels must be int64 (long) for CrossEntropyLoss

print(f"X shape : {X_np.shape}   (samples, features) - features stand in for word2vec dims")
print(f"y shape : {y_np.shape}   class labels in {{0, 1}}")
print(f"X dtype : {X_np.dtype} | y dtype : {y_np.dtype}")

# Visualize the first two feature columns, colored by class (just to see structure exists)
plt.figure(figsize=(5, 4))
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='coolwarm', s=8, alpha=0.5)
plt.xlabel('feature 0'); plt.ylabel('feature 1')
plt.title('Toy feature matrix (2 of 100 dims)')
plt.tight_layout(); plt.show()
```

## Cell 5 - Markdown: Section 1 - Creating layers (theory)

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell 64ec4402
Change: keep the Linear/ReLU/Dropout rows; drop the Conv2d row (no images in this notebook); reframe
shapes around a feature vector instead of a 28x28 image.

```markdown
## Section 1 - Creating layers

Every layer in PyTorch lives in `torch.nn`. For a model that classifies feature vectors you need
three building blocks:

| Layer | What it does | Key arguments |
|-------|--------------|---------------|
| `nn.Linear(in, out)` | Fully connected: `y = x @ W.T + b` | `in_features`, `out_features` |
| `nn.ReLU()` | Element-wise `max(0, x)` | none |
| `nn.Dropout(p)` | Randomly zeroes activations during training | `p` = drop probability |

You recognize `nn.Linear` from B4: it is the matrix multiply plus broadcasted bias add you wrote by
hand. Layers are objects: create one, then call it like a function.

```python
layer = nn.Linear(100, 64)   # create: 100 inputs -> 64 outputs
y = layer(x)                 # call: runs the forward pass
```

A layer tracks its own parameters (weights and biases). You never touch those arrays directly; the
optimizer updates them through `optimizer.step()`. `nn.Linear(in, out)` stores `weight` with shape
`(out, in)` and `bias` with shape `(out,)`.
```

## Cell 6 - Code: Demo - instantiate and inspect layers

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell 0b5c67c0
Change: drop the Conv2d block and the fake-image batch; demo Linear, ReLU, Dropout on a feature
vector of width `N_FEATURES`.

```python
# Demo: create the three layers and inspect their behavior on a feature vector

# 1. Linear: maps 100 features -> 64 hidden units
fc = nn.Linear(in_features=N_FEATURES, out_features=64)
print("nn.Linear(100, 64)")
print(f"  weight shape : {fc.weight.shape}   # (out_features, in_features) = (64, 100)")
print(f"  bias shape   : {fc.bias.shape}     # (64,)")

# Run one fake feature vector through it (batch of 1)
one_vec = torch.randn(1, N_FEATURES)
print(f"  input  shape : {one_vec.shape}  -> output shape : {fc(one_vec).shape}")

# 2. ReLU: no parameters, just max(0, x)
relu = nn.ReLU()
demo = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"\nnn.ReLU() on {demo.tolist()} -> {relu(demo).tolist()}")

# 3. Dropout: zeroes a fraction of activations in train mode, no-op in eval mode
drop = nn.Dropout(p=0.2)
ones = torch.ones(10)
drop.train()
print(f"\nnn.Dropout(0.2) train mode: {drop(ones).tolist()}")
drop.eval()
print(f"nn.Dropout(0.2) eval mode : {drop(ones).tolist()}  (no-op)")
```

## Cell 7 - Code: Lab 1 - create and configure layers

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell 3225d9cc
Change: retarget tasks to feature-dim layers; drop the Conv2d task; keep the verification block.
Starters point to the ROLE of each layer, never name the constructor or its arguments inline.

```python
# Lab 1: create and configure layers
# Build the three layers a feature-vector classifier needs.

# 1. A fully connected layer that maps the 100 input features to 64 hidden units.
fc_layer = None  # YOUR CODE

# 2. A ReLU activation (no arguments).
relu_layer = None  # YOUR CODE

# 3. A dropout layer that drops 30 percent of activations during training.
dropout_layer = None  # YOUR CODE

# --- Verification (do not modify below this line) ---
if fc_layer is not None:
    print(f"fc_layer weight shape : {fc_layer.weight.shape}   (expected torch.Size([64, 100]))")
if relu_layer is not None:
    test = torch.tensor([-1.0, 0.0, 1.0])
    print(f"relu_layer(-1,0,1)    : {relu_layer(test).tolist()}   (expected [0.0, 0.0, 1.0])")
if dropout_layer is not None:
    print(f"dropout_layer.p       : {dropout_layer.p}   (expected 0.3)")
```

## Cell 8 - Markdown: Section 2 - Building a model with nn.Module (theory)

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell 344dddd9
Change: merge the Sequential theory (cell d726877f) and the nn.Module theory; trace shapes on a
`(batch, features)` tensor, not an image; drop the dual-branch motivation in favor of a plain MLP.

```markdown
## Section 2 - Building a model with `nn.Module`

A model is a stack of layers wired together. There are two ways to express one.

For a straight chain, `nn.Sequential` is the quickest:

```python
model = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 2),   # 2 raw logits, one per class
)
```

For anything you want to name, inspect, or extend, subclass `nn.Module`. This is the form the rest
of the course uses, because it is exactly how HuggingFace models are written.

```python
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()                 # MANDATORY: registers the layers as parameters
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):                   # define the computation
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)                  # raw logits, NO softmax
```

Two rules that beginners get wrong:

1. Always call `super().__init__()`. Skip it and your layers are not registered, so the optimizer
   never sees them and the model never learns.
2. Return raw logits from `forward`. Do NOT apply softmax. The loss function in Section 3 applies
   log-softmax internally; doing it twice silently breaks training.

Trace the shapes: input `(batch, 100)` -> `fc1` -> `(batch, 64)` -> `fc2` -> `(batch, 2)`.
```

## Cell 9 - Code: Demo - define and inspect an MLPClassifier

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell bcc94f9d
Change: replace the dual-branch model with a single clean `MLPClassifier` (the class B7 reuses);
run a shape check and count parameters; show `model` prints its architecture.

```python
# Demo: define an nn.Module classifier and inspect it

class DemoMLP(nn.Module):
    """A 2-layer MLP: in_dim -> hidden -> n_classes. Returns raw logits."""

    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()                          # registers child layers as parameters
        self.fc1  = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2  = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))                  # (B, in_dim) -> (B, hidden)
        x = self.drop(x)
        return self.fc2(x)                          # (B, hidden) -> (B, n_classes), raw logits

demo_model = DemoMLP(in_dim=N_FEATURES, hidden=64, n_classes=N_CLASSES)

# Shape check on a fake batch of 8 feature vectors
probe = torch.randn(8, N_FEATURES)
logits = demo_model(probe)
print(f"Input  : {probe.shape}")
print(f"Output : {logits.shape}   (expected (8, 2) raw logits)")

print("\nArchitecture:")
print(demo_model)

n_params = sum(p.numel() for p in demo_model.parameters() if p.requires_grad)
print(f"\nTrainable parameters: {n_params:,}")
```

## Cell 10 - Code: Lab 2 - implement the MLPClassifier

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell dc1bc2c3
Change: replace the residual-block skeleton with the `MLPClassifier` skeleton (the class B7 reuses
verbatim); keep a verification block. Starters describe each layer by role and the forward order in
words, never spelling out the constructor calls.

```python
# Lab 2: implement the model class B7 will reuse on real word2vec features.

class MLPClassifier(nn.Module):
    """A 2-layer MLP classifier. forward(x) returns raw logits of shape (B, n_classes)."""

    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()  # keep this line - it registers your layers
        # First fully connected layer: maps in_dim features to hidden units.
        self.fc1  = None  # YOUR CODE
        # A ReLU activation.
        self.relu = None  # YOUR CODE
        # A dropout layer that drops 20 percent of activations.
        self.drop = None  # YOUR CODE
        # Second fully connected layer: maps hidden units to n_classes outputs.
        self.fc2  = None  # YOUR CODE

    def forward(self, x):
        # Apply fc1, then the activation, then dropout, then fc2.
        # Return raw logits (do not apply softmax).
        x = None  # YOUR CODE - first layer then activation
        x = None  # YOUR CODE - dropout
        return None  # YOUR CODE - second layer, raw logits

# --- Verification (do not modify) ---
try:
    m = MLPClassifier(in_dim=N_FEATURES, hidden=64, n_classes=N_CLASSES)
    out = m(torch.randn(4, N_FEATURES))
    print(f"Output shape : {out.shape}   (expected torch.Size([4, 2]))")
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Parameters   : {n:,}   (expected {N_FEATURES*64 + 64 + 64*N_CLASSES + N_CLASSES:,})")
    print("MLPClassifier looks good - this is the exact class you reuse in B7.")
except Exception as e:
    print(f"Not ready yet: {e}")
```

## Cell 11 - Markdown: Section 3 - Loss, optimizer, and the training loop (theory)

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell a7115e5d
Change: drop the custom-loss material; keep only loss choice (`CrossEntropyLoss`), optimizer choice
(`Adam`/`AdamW`), and merge in the six-line-loop framing from cell 3f1f07e8.

```markdown
## Section 3 - Loss, optimizer, and batching

Three pieces turn a model into a trained model: a loss, an optimizer, and a loop.

**Loss.** For classification, use `nn.CrossEntropyLoss`. Two things to memorize, because both are
common silent bugs:

- It takes RAW LOGITS, not softmax outputs. It applies log-softmax internally.
- Its targets are INTEGER CLASS INDICES (`torch.long`), shape `(N,)`, not one-hot vectors. Pass
  floats or one-hot and you get a dtype error or wrong gradients.

**Optimizer.** `torch.optim.Adam(model.parameters(), lr=1e-3)` is the default for most tasks.
`AdamW` adds proper weight decay and is the default for transformers (you will meet it again in C9).

**Batching.** Wrap your tensors in a `TensorDataset`, then a `DataLoader` shuffles and batches them:

```python
ds = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(ds, batch_size=64, shuffle=True)
```

**The loop.** Keras gives you `model.fit`. PyTorch makes you write the loop, which is the same six
lines every time:

```python
for xb, yb in loader:
    optimizer.zero_grad()        # 1. clear stale gradients
    logits = model(xb)           # 2. forward pass
    loss = loss_fn(logits, yb)   # 3. compute loss
    loss.backward()              # 4. backpropagate
    optimizer.step()             # 5. update weights
# (6. the for-loop itself is the iteration over batches)
```

The number one beginner bug is forgetting `optimizer.zero_grad()`. PyTorch accumulates gradients by
default, so skipping it sums this batch's gradient onto every previous batch's, and the loss slowly
explodes. Make `zero_grad` the first line of the loop, always.
```

## Cell 12 - Code: Demo - loss and optimizer on a fresh model

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell ce46960e
Change: drop the sine dataset and custom-MSE comparison; demo `CrossEntropyLoss` on raw logits with
integer targets, and show one optimizer step shrinking the loss. Convert the toy numpy arrays to
tensors here (float32 features, long labels) so the dtype rules are visible.

```python
# Demo: loss + optimizer, one step at a time

# Convert the toy numpy arrays to tensors with the REQUIRED dtypes
X_tensor = torch.tensor(X_np, dtype=torch.float32)   # features: float32
y_tensor = torch.tensor(y_np, dtype=torch.long)      # labels: long (int64) class indices
print(f"X_tensor dtype : {X_tensor.dtype} | y_tensor dtype : {y_tensor.dtype}")

model     = DemoMLP(in_dim=N_FEATURES, hidden=64, n_classes=N_CLASSES).to(device)
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# One forward pass on a small batch
xb = X_tensor[:128].to(device)
yb = y_tensor[:128].to(device)

logits = model(xb)
print(f"logits shape   : {logits.shape}   (batch, n_classes) - RAW, no softmax")
loss_before = loss_fn(logits, yb)
print(f"loss before    : {loss_before.item():.4f}")

# One optimization step
optimizer.zero_grad()
loss_before.backward()
optimizer.step()

loss_after = loss_fn(model(xb), yb)
print(f"loss after 1 step: {loss_after.item():.4f}   (should be lower)")
```

## Cell 13 - Code: Lab 3 - wire up data, loss, and optimizer

Provenance: FROM-NEW
Change: new lab for the toy arc. Student builds the `TensorDataset` / `DataLoader` and creates the
loss and optimizer for their `MLPClassifier`. Hints name the roles only.

```python
# Lab 3: prepare the data pipeline and the training ingredients.

BATCH_SIZE = 64
LR = 1e-3

# 1. Wrap X_tensor and y_tensor in a dataset, then build a loader that
#    shuffles and serves batches of BATCH_SIZE.
train_ds     = None  # YOUR CODE - a dataset from the two tensors
train_loader = None  # YOUR CODE - a loader over that dataset, shuffled, batch_size=BATCH_SIZE

# 2. Instantiate your MLPClassifier (100 -> 64 -> 2) and move it to device.
model = None  # YOUR CODE

# 3. The loss function for classification (raw logits + integer labels).
loss_fn = None  # YOUR CODE

# 4. An Adam optimizer over the model's parameters at learning rate LR.
optimizer = None  # YOUR CODE

# --- Verification (do not modify) ---
if train_loader is not None:
    xb, yb = next(iter(train_loader))
    print(f"batch X shape : {xb.shape}   (expected (64, 100))")
    print(f"batch y shape : {yb.shape}   (expected (64,))")
    print(f"batch y dtype : {yb.dtype}   (expected torch.int64)")
if model is not None and loss_fn is not None and optimizer is not None:
    sample = loss_fn(model(xb.to(device)), yb.to(device))
    print(f"sample loss   : {sample.item():.4f}   (a single float, ready to train)")
```

## Cell 14 - Markdown: Section 4 - The training loop and evaluation (theory)

Provenance: FROM-NEW
Change: new short theory cell isolating the full epoch loop and the eval pattern (`model.eval()`,
`torch.no_grad()`, accuracy idiom), so the lab that follows is pure recall.

```markdown
## Section 4 - Train for several epochs and evaluate

One pass over all the batches is one epoch. You repeat for several epochs and watch the loss fall.
After training, you measure accuracy in evaluation mode:

- `model.train()` before the training batches: turns dropout ON.
- `model.eval()` before evaluating: turns dropout OFF so predictions are deterministic.
- Wrap evaluation in `with torch.no_grad():` so PyTorch does not build the autograd graph. This is
  faster and uses less memory.

Accuracy for a classification model is the fraction of correct predictions. Take the argmax over the
class dimension and compare to the labels:

```python
preds = logits.argmax(dim=1)
acc = (preds == y).float().mean().item()
```

That is the whole evaluation. In B7 you run this exact code on real word2vec features and compare
the accuracy to the LogisticRegression baseline from B5.
```

## Cell 15 - Code: Demo - train the toy model for several epochs

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell e191b15f
Change: drop the sine regression; train the classifier on the toy `DataLoader`; track and plot the
loss curve; print accuracy each epoch using the argmax idiom.

```python
# Demo: train the classifier end to end on the toy data

EPOCHS = 15

demo_model = DemoMLP(in_dim=N_FEATURES, hidden=64, n_classes=N_CLASSES).to(device)
demo_loss_fn   = nn.CrossEntropyLoss()
demo_optimizer = torch.optim.Adam(demo_model.parameters(), lr=1e-3)

demo_ds     = TensorDataset(X_tensor, y_tensor)
demo_loader = DataLoader(demo_ds, batch_size=64, shuffle=True)

losses = []
for epoch in range(EPOCHS):
    demo_model.train()                      # dropout ON
    epoch_loss = 0.0
    for xb, yb in demo_loader:
        xb, yb = xb.to(device), yb.to(device)
        demo_optimizer.zero_grad()          # 1. clear grads
        logits = demo_model(xb)             # 2. forward
        loss = demo_loss_fn(logits, yb)     # 3. loss
        loss.backward()                     # 4. backward
        demo_optimizer.step()               # 5. update
        epoch_loss += loss.item()
    avg = epoch_loss / len(demo_loader)
    losses.append(avg)

    # evaluation
    demo_model.eval()                       # dropout OFF
    with torch.no_grad():
        all_logits = demo_model(X_tensor.to(device))
        preds = all_logits.argmax(dim=1).cpu()
        acc = (preds == y_tensor).float().mean().item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{EPOCHS} - loss: {avg:.4f} | train acc: {acc:.4f}")

plt.figure(figsize=(6, 3))
plt.plot(range(1, EPOCHS+1), losses, marker='o')
plt.xlabel('epoch'); plt.ylabel('avg loss'); plt.title('Training loss')
plt.tight_layout(); plt.show()
```

## Cell 16 - Code: Lab 4 - write the training loop

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell 13bd65df
Change: retarget from MNIST to the toy classifier; the student fills the six loop lines and the eval
forward/argmax; verification asserts accuracy crosses a threshold. Loop-step hints name the role,
never the method.

```python
# Lab 4: write the training loop for YOUR model from Lab 3.
# Reuse train_loader, model, loss_fn, optimizer from Lab 3.

LAB_EPOCHS = 15

for epoch in range(LAB_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # Step 1: clear the gradients left over from the previous batch.
        None  # YOUR CODE

        # Step 2: run the forward pass to get logits.
        logits = None  # YOUR CODE

        # Step 3: compute the loss between logits and labels.
        loss = None  # YOUR CODE

        # Step 4: backpropagate.
        None  # YOUR CODE

        # Step 5: take one optimizer step.
        None  # YOUR CODE

        epoch_loss += loss.item()

    # --- evaluation (provided) ---
    model.eval()
    with torch.no_grad():
        # Forward pass over the full dataset, then take the predicted class per row.
        eval_logits = None  # YOUR CODE - forward pass on X_tensor (moved to device)
        preds = None        # YOUR CODE - predicted class index per row (argmax over dim 1)
        acc = (preds.cpu() == y_tensor).float().mean().item()

    avg = epoch_loss / len(train_loader)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:2d}/{LAB_EPOCHS} - loss: {avg:.4f} | train acc: {acc:.4f}")

# --- Verification ---
print(f"\nFinal training accuracy: {acc:.4f}   (expected above 0.85)")
assert acc > 0.85, "Train accuracy should exceed 0.85 - check your loop wiring."
print("Training loop works. This is the loop that fine-tunes DistilBERT in C9.")
```

## Cell 17 - Code: Demo - the full pattern in one place (end-to-end)

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell b3859256
Change: simplify the transfer end-to-end lab into a clean end-to-end demo function `train_model`
that wraps the whole pattern, so B7 can call it on real features. Print final accuracy.

```python
# The whole pattern as one reusable function. B7 calls this on real word2vec features.

def train_model(X, y, in_dim, hidden=64, n_classes=2, epochs=15, lr=1e-3, batch_size=64):
    """Train an MLPClassifier on (X, y) tensors. Returns the trained model and final accuracy."""
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    net = MLPClassifier(in_dim, hidden, n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()

    net.eval()
    with torch.no_grad():
        preds = net(X.to(device)).argmax(dim=1).cpu()
        acc = (preds == y).float().mean().item()
    return net, acc

trained_net, final_acc = train_model(X_tensor, y_tensor, in_dim=N_FEATURES)
print(f"End-to-end toy accuracy: {final_acc:.4f}")
print("In B7: call train_model on the word2vec X, y and beat the B5 LogisticRegression baseline.")
```

## Cell 18 - Markdown: Lab 5 instructions and Stretch

Provenance: FROM-NEW
Change: new instructions cell for the end-to-end lab plus the labelled stretch (validation split),
keeping the three-tier lab structure.

```markdown
### Lab 5 - Run it end to end, then make it honest (stretch)

**Core task.** You already have `train_model`. Call it on the toy tensors and confirm you get a
trained model and an accuracy above 0.85. Then change `hidden` to 128 and re-run: note whether more
capacity helps on this toy data.

**Stretch (for fast finishers).** Training accuracy alone is misleading: a model can memorize the
training set. Split the data into a train part and a held-out validation part, train only on the
train part, and report accuracy on BOTH each epoch. Track the best validation accuracy you see.
This is the single most important habit before C9, where you fine-tune on a train split and report
on a validation split. The starter is in the next cell.
```

## Cell 19 - Code: Lab 5 core + Stretch (validation split)

Provenance: FROM-NEW
Change: new lab. Core line calls `train_model`. Stretch builds a train/val split and tracks best val
accuracy. Hints name roles only (a slicing point, a held-out part), never the exact split index.

```python
# Lab 5 core: train end to end on the toy tensors.
core_net, core_acc = None, None  # YOUR CODE - call train_model on X_tensor, y_tensor (in_dim=N_FEATURES)
if core_acc is not None:
    print(f"Core accuracy: {core_acc:.4f}   (expected above 0.85)")

# --- Stretch: train/validation split with best-val tracking ---
# 1. Choose a split point that holds out the last 20 percent of the rows for validation.
n_total = X_tensor.shape[0]
n_val   = None  # YOUR CODE - number of rows to hold out (20 percent of n_total, as an int)

# 2. Slice tensors into train and validation parts (first part trains, last n_val rows validate).
X_train = None  # YOUR CODE
y_train = None  # YOUR CODE
X_val   = None  # YOUR CODE
y_val   = None  # YOUR CODE

if n_val is not None and X_val is not None:
    train_loader_s = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    net_s = MLPClassifier(N_FEATURES, 64, N_CLASSES).to(device)
    loss_fn_s = nn.CrossEntropyLoss()
    opt_s = torch.optim.Adam(net_s.parameters(), lr=1e-3)

    best_val = 0.0
    for epoch in range(15):
        net_s.train()
        for xb, yb in train_loader_s:
            xb, yb = xb.to(device), yb.to(device)
            opt_s.zero_grad()
            loss_fn_s(net_s(xb), yb).backward()
            opt_s.step()
        net_s.eval()
        with torch.no_grad():
            tr_acc  = (net_s(X_train.to(device)).argmax(1).cpu() == y_train).float().mean().item()
            val_acc = (net_s(X_val.to(device)).argmax(1).cpu()   == y_val).float().mean().item()
        best_val = max(best_val, val_acc)
    print(f"train acc: {tr_acc:.4f} | val acc: {val_acc:.4f} | best val: {best_val:.4f}")
    print("If train acc is much higher than val acc, the model is overfitting - see the homework.")
```

## Cell 20 - Markdown: Homework extension (transfer learning: freeze and unfreeze)

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cells 4c1e03a8 + d87a3d58 + 8e1adeac +
7be26845
Change: fold the entire freezing section (theory, demo, lab) into one async homework cell, retheme
to the chatbot transfer-learning story (frozen backbone + fresh head + unfreeze), and add a second
bonus on overfitting with dropout + weight decay. This is the pattern C9 uses on DistilBERT.

```markdown
## Homework extension (async, deeper)

These are async, production-oriented, and point straight at C9. Do them in this notebook after class.

**Homework 1 - Freeze a backbone, train a fresh head, then fine-tune (the C9 pattern).**

Real transfer learning loads a pretrained model, freezes it, snaps a fresh head on top, and trains
only the head. Then it optionally unfreezes and fine-tunes at a low learning rate. You will simulate
this with two MLPs.

1. Build a model with a `backbone` (`nn.Sequential` of a couple of `Linear` + `ReLU` layers) and a
   separate `head` (`nn.Linear`) as two attributes in one `nn.Module`.
2. Freeze the backbone: loop over `model.backbone.parameters()` and set `requires_grad = False`.
3. Build the optimizer over trainable params only:
   `torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)`.
4. Count trainable vs frozen params with `sum(p.numel() for p in model.parameters() if p.requires_grad)`
   and confirm only the head is trainable.
5. Train a few epochs, record accuracy. Then unfreeze everything (`requires_grad = True` for all
   params), rebuild the optimizer at `lr=1e-4`, fine-tune one more epoch, and compare.

This is exactly what C9 does: DistilBERT is the frozen-then-fine-tuned backbone, and a `nn.Linear`
classifier is the fresh head trained with `nn.CrossEntropyLoss`.

**Homework 2 - See overfitting, then fix it.**

1. Take a tiny slice (say 200 rows) of the toy data and a train/val split.
2. Train a wide MLP for many epochs. Watch train accuracy approach 1.0 while validation accuracy
   stalls or drops: that gap is overfitting.
3. Now regularize: raise `nn.Dropout` p, and swap `Adam` for `torch.optim.AdamW(..., weight_decay=0.01)`.
   AdamW applies decoupled weight decay and is the default optimizer for transformers in C9.
4. Re-run and confirm the train/val gap shrinks.

Write two or three sentences on what changed and why this matters before fine-tuning a large model.
```

## Cell 21 - Markdown: Recap and bridge to B7

Provenance: FROM-OLD Frameworks/PyTorch_Exercises.ipynb cell 10a2e758
Change: retarget the recap table to the four kept sections; drop the Conv2d/dual-branch/custom-loss
rows; add the explicit bridge to B7 (swap toy matrix for word2vec features) and to C9 (same loop
fine-tunes DistilBERT).

```markdown
## Recap and what is next

You built and trained a neural network end to end. The pieces:

| Section | PyTorch concept |
|---------|-----------------|
| 1 | Layers: `nn.Linear`, `nn.ReLU`, `nn.Dropout` and their parameter shapes |
| 2 | Models: `nn.Module` subclassing (`__init__` + `forward`), `nn.Sequential` |
| 3 | Loss + optimizer + batching: `nn.CrossEntropyLoss`, `Adam`, `TensorDataset`, `DataLoader` |
| 4 | The six-line training loop: `zero_grad -> forward -> loss -> backward -> step`, plus eval |

Three rules to keep:

1. `optimizer.zero_grad()` is the first line of the loop. Forgetting it is the most common bug.
2. `nn.CrossEntropyLoss` takes raw logits and integer (long) labels. Never apply softmax first.
3. `model.train()` for training, `model.eval()` plus `torch.no_grad()` for evaluation.

### Bridge to B7 (the stopper)

You trained on a toy `make_classification` matrix. In B7 you delete that matrix and load the real
word2vec document vectors from B5 (the saved `X`, `y`). You keep this exact `MLPClassifier` and this
exact loop, and your job is to beat the LogisticRegression baseline you measured in B5. That is the
"embeddings as features" pattern: words become vectors (B5), vectors feed an MLP (B6), the MLP
classifies (B7).

### Bridge to C9 (the chatbot)

The classifier `DistilBertForSequenceClassification` fine-tunes in C9 is an encoder plus a
`nn.Linear` head trained with `nn.CrossEntropyLoss` - the same head you built here. The HuggingFace
trainer runs the same `zero_grad -> forward -> loss -> backward -> step` loop you just wrote. You now
know the engine that trains the chatbot.
```

---

# VERIFICATION CHECKLIST

- [ ] Cell count is 22 (Cells 0-21), inside the 20-25 target.
- [ ] Install cell pins `numpy<2` and drops the torch install (Colab has torch 2.x); restart-runtime
      note present in Cell 1 markdown.
- [ ] Import cell reintroduces `from torch.utils.data import TensorDataset, DataLoader` (the import
      B4 deferred to B6) and reuses `SEED = 42` + the `device = torch.device(...)` idiom from B4.
- [ ] B6 actually introduces nn.Module, loss, optimizer, TensorDataset, and DataLoader (pays off the
      B4 deferral promise made in B4 Cells 0, 3, 9, 17, 24).
- [ ] Toy feature width is 100 (`N_FEATURES`), matching B5's `doc_vector` output, so the B7 swap is
      literal. Features are float32; labels are int64 (long).
- [ ] Theory->demo->lab cadence holds across four cycles; never more than 3 markdown cells in a row
      without code (max run is Cells 8 alone, then code at 9).
- [ ] Every `None  # YOUR CODE` describes the role/shape only and never names the constructor,
      method, or index that solves it (cover-the-solution test passes for Labs 1-5).
- [ ] Each lab has a verification block; Lab 4 asserts train accuracy > 0.85.
- [ ] Three-tier labs present: core (Lab 5 call), stretch (validation split), homework (freeze/
      unfreeze transfer learning + overfit-then-regularize).
- [ ] `MLPClassifier` and `train_model` are reusable by B7 verbatim on word2vec features.
- [ ] `nn.CrossEntropyLoss` is described and used with raw logits + long labels; no softmax in
      `forward`.
- [ ] Bridge to B7 (swap toy matrix for word2vec X, y; beat the B5 baseline) and bridge to C9 (same
      head + same loop fine-tunes DistilBERT) both stated.
- [ ] Every cell carries a `Provenance:` line; FROM-OLD cells cite the source cell id and a `Change:`
      line; Cell Migration Map present near the top.
- [ ] STAR Story and Chatbot Through-Line sections both present.
- [ ] Plain ASCII only: no em/en dashes, no Unicode multiply, no emoji.

# RESEARCH VALIDATED (June 2026)

- nn.CrossEntropyLoss combines LogSoftmax + NLLLoss, expects RAW LOGITS of shape (N, C) and INTEGER
  CLASS-INDEX targets of shape (N,); never apply softmax before it (silent training bug). Source:
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html and
  https://github.com/mrdbourke/pytorch-deep-learning/discussions/252
- The canonical PyTorch training loop is zero_grad -> forward -> loss -> backward -> step; gradients
  accumulate by default so zero_grad is mandatory; validation runs under torch.no_grad() with
  model.eval(). Source:
  https://sebastianraschka.com/faq/docs/training-loop-in-pytorch.html and
  https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Colab (June 2026) ships torch 2.10.0, numpy 2.0.2, Python 3.12.13 preinstalled; pin numpy<2 and
  restart the runtime after install. Source:
  https://developers.google.com/colab/release-notes and
  https://github.com/googlecolab/colabtools/issues/5115
- scikit-learn is preinstalled in Colab; make_classification generates an (n_samples, n_features)
  float matrix with n_classes labels (binary by default) - a faithful stand-in for word2vec doc
  vectors. Source:
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
- CrossEntropyLoss targets must be torch.long/int64 (class indices); float or one-hot targets raise
  a dtype error; model inputs must be float. Source:
  https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542
- Classification accuracy idiom: preds = logits.argmax(dim=1); acc = (preds == y).float().mean().
  Source: https://saturncloud.io/blog/calculating-the-accuracy-of-pytorch-models-every-epoch/
- Transfer learning: set requires_grad=False to freeze a backbone, train a fresh randomly-init head,
  build the optimizer with filter(lambda p: p.requires_grad, model.parameters()), then unfreeze and
  fine-tune at a lower lr; HF freezes via model.base_model.parameters(). Source:
  https://www.learnpytorch.io/06_pytorch_transfer_learning/ and
  https://medium.com/@prabhatzade/freezing-layers-and-fine-tuning-transformer-models-in-pytorch-a-simple-guide-119cad0980c6
- AdamW applies decoupled weight decay and is the modern default optimizer for transformers; combine
  with dropout to fight overfitting. Source:
  https://mbrenndoerfer.com/writing/adamw-optimizer-decoupled-weight-decay and
  https://huggingface.co/docs/transformers/main_classes/trainer
- DistilBertForSequenceClassification = encoder + pre_classifier (nn.Linear dim->dim) + ReLU +
  dropout + classifier (nn.Linear dim->num_labels) on the pooled output, trained with
  CrossEntropyLoss; the HuggingFace Trainer runs the same zero_grad/forward/backward/step loop
  internally. This is the exact head + loop B6 builds. Source:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
  and https://huggingface.co/learn/llm-course/chapter3/4

Plan written to `plans/refactor/notebooks/B6-pytorch-nn-basics.md`.

Next step: run `/build-notebook pytorch-nn-basics colab` to generate the exercise + solution
notebooks from this plan, 5 cells at a time with approval between batches.
