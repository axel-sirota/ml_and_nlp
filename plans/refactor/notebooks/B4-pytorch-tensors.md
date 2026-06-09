# B4 - PyTorch Tensors - Cell-by-Cell Plan

## Status

TOUCHED. Source notebook: `Frameworks/PyTorch_Tensor_Exercises.ipynb` (39 cells, 8 sections plus
optional). We lift the five sections that B5/B6/B7 actually need (Basics, Operations, Reshaping,
Indexing+Broadcasting, Autograd), drop the DataLoader and nn.Module sections (they move to B6 where
they are first used), retheme the story to the chatbot arc, and convert the autograd optional lab
into a labelled in-notebook stretch plus a new gradient-checking homework.

New file: `B-How-It-Works/4-PyTorch_Tensors.ipynb`. Slug: `pytorch-tensors`.

## Context

Students arrive from Part A (A1 tools, A2 `pipeline()`, A3 zero-shot router). They have used
`pipeline`, the result dict `{'sequence','labels','scores'}`, `candidate_labels`, `device`, and the
`all-MiniLM-L6-v2` sentence embedder from A1. They have NOT written a single line of model-internal
PyTorch yet. The promise held back through all of Part A ("we will open the black box and beat the
zero-shot ceiling") starts being paid here.

Key insight they leave with: a PyTorch tensor is a NumPy array with two superpowers - optional GPU
placement and autograd. `.backward()` plus `.grad` is the entire engine that trains every model in
the rest of the course. They can build a small tensor expression and differentiate it by hand-driven
gradient descent, which is exactly what `optimizer.step()` will automate in B6.

API names they already have (reused, not redefined): `device`, `SEED`/`torch.manual_seed(42)`,
`np`, `torch`. New names introduced here: `torch.tensor`, `torch.zeros/ones/rand/randn`,
`torch.from_numpy`, `.reshape/.view/.transpose/.permute/.squeeze/.unsqueeze`, boolean masks,
broadcasting, `requires_grad`, `.backward()`, `.grad`, `torch.no_grad()`.

## Chatbot Through-Line

The course end goal is a fine-tuned transformer in a simple Gradio chatbot. B4 contributes the
single most load-bearing brick under that goal: autograd. The one-line story is "every text becomes
a tensor, and `.backward()` is how a model learns from it." Concretely: in B5 each text turns into a
384-dimensional float vector from `all-MiniLM-L6-v2`; in B6 those vectors flow through `nn.Linear`
layers; in B7 (the STOPPER) we train an MLP head on those 384-dim tensors; in C9 the same
`.backward()` mechanic fine-tunes DistilBERT itself. So the bridge to B5 is exact: "You can now
build and differentiate a tensor expression. Next we will turn real sentences into the 384-dim
tensors that this machinery trains on."

## STAR Story

- Situation: You are the practitioner from Part A. You shipped a zero-shot tweet router that tops
  out around 70-85 percent accuracy. The team wants better, which means training a model, which
  means PyTorch - the framework everything here runs on. You have ML background but have never
  written tensor-level PyTorch.
- Task: Get fluent in just enough PyTorch to build and train a small network. No deep dive - only
  the bricks B5, B6, and B7 actually snap together: tensors, shapes, broadcasting, and autograd.
- Action: Work five short theory-demo-lab cycles. Create tensors, do arithmetic and matrix
  multiply, reshape for layer wiring, index and broadcast like a bias add, then meet autograd: mark
  a tensor `requires_grad=True`, call `.backward()`, read `.grad`. Finish by running gradient
  descent by hand so you see exactly what an optimizer automates.
- Result: You can build a small differentiable tensor expression and update parameters from its
  gradients. You are ready for `nn` in B6 and the word2vec MLP in B7, and you understand that the
  same `.backward()` will later fine-tune a transformer for the chatbot.

## Deliverables

- Exercise: `B-How-It-Works/4-PyTorch_Tensors.ipynb`
- Solution: `Solutions/B-How-It-Works/4-PyTorch_Tensors.ipynb`

## Session Timing (~60-90 min)

| Block | Cells | Minutes |
|-------|-------|---------|
| Header + environment + through-line | 0-4 | 8 |
| Section 1 - Tensor Basics | 5-8 | 12 |
| Section 2 - Tensor Operations | 9-12 | 12 |
| Section 3 - Reshaping | 13-16 | 13 |
| Section 4 - Indexing + Broadcasting | 17-20 | 14 |
| Section 5 - Autograd (+ stretch + homework) | 21-23 | 18 |
| Wrap-up | 24 | 3 |

## Cell Migration Map

| Old cell id (idx) | New cell | Action |
|-------------------|----------|--------|
| b8bd61cd (0) | Cell 0 | edit - retheme story to Part A handoff, trim objectives to 5 sections |
| 4cc87c8a (1) | Cell 1 | keep |
| f1a54e83 (2) | Cell 2 | edit - pin `numpy<2`, drop `torch` install (Colab has it), add restart note |
| 41ee36e7 (3) | Cell 3 | edit - drop `TensorDataset/DataLoader` import |
| - | Cell 4 | FROM-NEW - "what we are building" through-line markdown |
| eebee412 (4) | Cell 5 | keep |
| a6ba2f17 (5) | Cell 6 | keep |
| 0a1f90ae (6) | Cell 7 | keep |
| cee08ecc (7) | Cell 8 | keep |
| 407bc194 (8) | Cell 9 | keep |
| e752d12f (9) | Cell 10 | keep |
| e0a09010 (10) | Cell 11 | keep |
| 72593e14 (11) | Cell 12 | keep |
| b50a00f1 (12) | Cell 13 | edit - add the view-vs-reshape contiguous gotcha note |
| 7a37a57b (13) | Cell 14 | edit - add a view-after-transpose failure line |
| 24e9cc63 (14) | Cell 15 | keep |
| 3a6c0004 (15) | Cell 16 | keep |
| 04877aab (16) + a80e4ba2 (20) | Cell 17 | edit - merge indexing theory and broadcasting theory |
| df4760d1 (17) + 1371f1a6 (21) | Cell 18 | edit - merge indexing demo and broadcasting demo |
| e8566889 (18) + a563d635 (22) | Cell 19 | edit - merge indexing lab and broadcasting lab instructions, add stretch |
| 78d6cded (19) + 501aabfd (23) | Cell 20 | edit - merge lab starters, add pairwise-distance stretch |
| 5c32f0fd (24) | Cell 21 | edit - keep autograd theory, keep the one tf.GradientTape comparison, add grad-accumulation note |
| 12531c79 (25) | Cell 22 | keep |
| 04a7e2b0 (26) | Cell 23a (lab md) | edit - fold stretch + homework instructions in |
| 9ecaf823 (27) | Cell 23b (lab code) | edit - de-leak `# YOUR CODE` comments; append stretch GD starter |
| 353a23eb (36) | into Cell 23 stretch md | edit - becomes the labelled in-notebook stretch |
| 96dde56b (37) | into Cell 23 stretch code | edit - manual gradient descent starter |
| 67cb5e23 (38) | Cell 24 | edit - retheme takeaways to 5 sections, bridge to B5 |
| 6ea82449..c9b9e1be (28-35) | DROP | DataLoader + nn.Module sections move to B6 |

Note on cell numbering: Section 4 collapses old Sections 4 and 5 into four cells (17-20). Section 5
(Autograd) uses Cells 21, 22, and one composite Cell 23 that contains the lab markdown, the lab
starter, the labelled stretch (manual gradient descent), and the homework instructions. Final count
is 25 cells (Cell 0 through Cell 24).

---

# MAIN NOTEBOOK - Cell-by-Cell Content (Target: ~20-25 cells)

## Cell 0 - Markdown: Title, objectives, prerequisites, how to run

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell b8bd61cd
Change: retheme the story from "TF veteran joining a PyTorch team" to the Part A handoff persona;
trim the eight learning objectives down to the five sections we keep.

```
# PyTorch Tensors - the bricks under every model

*ML & NLP course - Data Trainers LLC - Axel Sirota*

## Where we are

In Part A you shipped real NLP with zero training: a `pipeline()` for every task and a zero-shot
router that hit roughly 70-85 percent accuracy. We kept promising one thing: to push past that
ceiling you have to train a model, and to train a model you have to speak PyTorch. This is where we
open the black box.

We are not doing a deep PyTorch course. We are picking up exactly the bricks the next three
notebooks snap together - tensors, shapes, broadcasting, and autograd - and nothing more.

## Learning objectives

By the end of this notebook you will be able to:

1. Create tensors of any shape and dtype with `torch.tensor`, `torch.zeros/ones/rand/randn`, and
   convert to and from NumPy.
2. Run arithmetic, reductions, ReLU, and matrix multiply on tensors.
3. Reshape tensors with `reshape/view/transpose/permute/squeeze/unsqueeze`, and know when `view`
   fails.
4. Index, slice, boolean-mask, and apply broadcasting confidently.
5. Compute gradients with `requires_grad=True` and `.backward()` - the engine that trains every
   model in Part B and Part C, and the PyTorch replacement for `tf.GradientTape`.

## Prerequisites

- Comfortable with NumPy arrays and basic ML.
- No prior PyTorch needed - that is the point.

## How to run

- Colab (recommended): Runtime -> Change runtime type -> GPU is nice but not required.
- Everything here is tiny and runs fine on CPU.

Let's pick up the bricks.
```

## Cell 1 - Markdown: Section 0 - Environment Setup

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 4cc87c8a

```
## Section 0 - Environment Setup

Colab already ships PyTorch and NumPy, so the install cell is mostly a safety net. We pin
`numpy<2` to stay consistent with the rest of the course (gensim and friends in B5 need it). Then
we import, set seeds, and detect the device.
```

## Cell 2 - Code: Install (numpy<2 pin)

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell f1a54e83
Change: drop the `torch` install (Colab already has a NumPy-2-compatible torch 2.x); pin `numpy<2`;
add the Colab restart note because Colab preinstalls numpy 2.0.x.

```
# Install required packages (run this first on Google Colab).
# Colab already ships a recent PyTorch, so we only pin NumPy to <2 for course-wide consistency.
# NOTE: Colab preinstalls numpy 2.x. After this cell runs, if Colab shows a
# "RESTART RUNTIME" button, click it, then run the notebook again from the top.
!pip install -q "numpy<2"
```

## Cell 3 - Code: Imports, seeds, device

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 41ee36e7
Change: remove the `from torch.utils.data import TensorDataset, DataLoader` import (those move to
B6); keep seeds and device detection.

```
import numpy as np
import torch
import torch.nn as nn  # we only use nn for relu/Linear demos later; full nn comes in B6

# ---- Reproducibility (same SEED=42 we use course-wide) ----
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---- Device detection: GPU if available, else CPU ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch version : {torch.__version__}")
print(f"NumPy version   : {np.__version__}  (expected 1.x because we pinned numpy<2)")
print(f"Using device    : {device}")
if device.type == 'cuda':
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")

print("\nEnvironment setup complete.")
```

## Cell 4 - Markdown: What are we building today (through-line)

Provenance: FROM-NEW

```
## What are we building toward?

Nothing fancy in this notebook - on purpose. The payoff is later, and it helps to see the chain now:

- Today (B4): a tensor is a NumPy array plus GPU and autograd. You will differentiate a small
  expression by hand.
- B5: every sentence becomes a fixed 384-dimensional float vector (from the `all-MiniLM-L6-v2`
  embedder you met in A1). A vector is just a 1-D tensor.
- B6: those vectors flow through `nn.Linear` layers - which are matrix multiply plus a broadcasted
  bias add, both of which you will write today.
- B7 (the stopper): you train a small MLP on those 384-dim tensors. Training is just calling
  `.backward()` in a loop - the exact mechanic from Section 5 below.
- C9: the same `.backward()` fine-tunes a whole transformer for the chatbot.

So when you call `loss.backward()` near the end of this notebook, you are running the literal engine
that trains everything that follows. Let's earn it.
```

## Cell 5 - Markdown: Section 1 theory - Tensor Basics

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell eebee412

```
## Section 1 - Tensor Basics

Mental model: a PyTorch tensor is a NumPy array, plus optional GPU placement, plus optional autograd
tracking. The factory functions are almost a 1-to-1 map from NumPy:

| NumPy | PyTorch |
|-------|---------|
| `np.array([1, 2, 3])` | `torch.tensor([1, 2, 3])` |
| `np.zeros((3, 3))` | `torch.zeros(3, 3)` |
| `np.ones((2, 4))` | `torch.ones(2, 4)` |
| `np.random.rand(2, 3)` | `torch.rand(2, 3)` |
| `np.random.randn(2, 3)` | `torch.randn(2, 3)` |

Two attributes you will check constantly:

```python
t = torch.tensor([1.0, 2.0, 3.0])
t.shape   # torch.Size([3])
t.dtype   # torch.float32
```

NumPy and PyTorch convert both ways:

```python
arr  = np.array([1.0, 2.0])
t    = torch.from_numpy(arr)   # SHARES memory: mutating arr also changes t
t2   = torch.tensor(arr)       # COPIES: independent of arr
back = t.numpy()               # back to NumPy (CPU tensors only)
```

The shared-memory detail of `from_numpy` is a classic source of surprise bugs - know it exists.

### Demo - factory functions in action
```

## Cell 6 - Code: Section 1 demo

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell a6ba2f17

```
# Demo: tensor creation and inspection
scalar  = torch.tensor(3.14)                       # 0-D tensor
vector  = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # 1-D
matrix  = torch.zeros(3, 3)                         # 2-D all zeros
rand3d  = torch.randn(2, 3, 4)                      # 3-D standard normal

for name, t in [('scalar', scalar), ('vector', vector),
                ('matrix', matrix), ('rand3d', rand3d)]:
    print(f"{name:8s}  shape={str(t.shape):20s}  dtype={t.dtype}")

# NumPy round-trip
arr  = np.array([[1.0, 2.0], [3.0, 4.0]])
t_np = torch.from_numpy(arr)     # zero-copy view: shares memory with arr
back = t_np.numpy()              # back to numpy
print(f"\nNumPy -> torch -> NumPy: arr type={type(arr).__name__}, "
      f"t_np type={type(t_np).__name__}, back type={type(back).__name__}")
print(f"All values identical: {np.allclose(arr, back)}")
```

## Cell 7 - Markdown: Lab 1 instructions

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 0a1f90ae

```
### Lab 1 - Tensor Basics

Complete the tasks below. Verification code is provided so you can check yourself immediately.

Tasks:
1. Create a scalar tensor with the value `7.0` and name it `s`.
2. Create a 1-D tensor with the five elements `[10, 20, 30, 40, 50]` as `torch.float32`, named `v`.
3. Create a 2-D tensor of shape `(3, 3)` filled with ones, named `m`.
4. Create a 3-D tensor of shape `(2, 4, 4)` filled with random normal values, named `t3d`.
5. Convert the NumPy array `np_arr` (provided in the starter) to a PyTorch tensor named `t_from_np`.
6. Convert `t_from_np` back to a NumPy array named `arr_back`.
```

## Cell 8 - Code: Lab 1 starter

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell cee08ecc

```
np_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 1. Scalar tensor holding the value 7.0
s = None  # YOUR CODE

# 2. 1-D tensor [10, 20, 30, 40, 50] as float32
v = None  # YOUR CODE

# 3. 2-D tensor of ones, shape (3, 3)
m = None  # YOUR CODE

# 4. 3-D tensor of random normal values, shape (2, 4, 4)
t3d = None  # YOUR CODE

# 5. Convert np_arr to a PyTorch tensor
t_from_np = None  # YOUR CODE

# 6. Convert t_from_np back to a NumPy array
arr_back = None  # YOUR CODE

# ---- Verification ----
if s is not None:
    print(f"s          : value={s.item():.1f}  shape={s.shape}  dtype={s.dtype}")
if v is not None:
    print(f"v          : shape={v.shape}  dtype={v.dtype}  values={v.tolist()}")
if m is not None:
    print(f"m          : shape={m.shape}  all_ones={m.all().item()}")
if t3d is not None:
    print(f"t3d        : shape={t3d.shape}  dtype={t3d.dtype}")
if t_from_np is not None:
    print(f"t_from_np  : shape={t_from_np.shape}  dtype={t_from_np.dtype}")
if arr_back is not None:
    print(f"arr_back   : type={type(arr_back).__name__}  shape={arr_back.shape}")
```

## Cell 9 - Markdown: Section 2 theory - Tensor Operations

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 407bc194

```
## Section 2 - Tensor Operations

PyTorch operators map almost identically to NumPy - the difference is they run on the GPU and can
track gradients.

Arithmetic (`+ - * /`) is element-wise. There are named equivalents `torch.add/sub/mul/div`.

Statistics:
```python
t.mean()        # scalar mean over all elements
t.mean(dim=0)   # mean along dim 0 (collapse rows -> one value per column)
t.std()         # standard deviation
```

Activation:
```python
torch.relu(t)   # max(0, x) element-wise, no import needed
```

Matrix multiply:
```python
A @ B           # same as torch.matmul(A, B)
torch.mm(A, B)  # strict 2-D only
```

Matrix multiply (`@`) is the heartbeat of a neural network layer: an `nn.Linear` is literally
`x @ W.T + b`. You will recognize it again in B6.

### Demo - operations
```

## Cell 10 - Code: Section 2 demo

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell e752d12f

```
torch.manual_seed(SEED)

a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([10.0, 20.0, 30.0, 40.0])

print("a + b  =", (a + b).tolist())
print("a - b  =", (a - b).tolist())
print("a * b  =", (a * b).tolist())
print("a / b  =", (a / b).tolist())

print(f"\na.mean() = {a.mean().item():.4f}")
print(f"a.std()  = {a.std().item():.4f}")

# ReLU on a tensor with negatives: clamps everything below 0 to 0
c = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"\ntorch.relu({c.tolist()}) = {torch.relu(c).tolist()}")

# Matrix multiply: (2,3) @ (3,2) -> (2,2)
A = torch.ones(2, 3)
B = torch.ones(3, 2) * 2.0
C = A @ B
print(f"\n(2x3) @ (3x2) = {C.shape}, values:\n{C}")
```

## Cell 11 - Markdown: Lab 2 instructions

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell e0a09010

```
### Lab 2 - Tensor Operations

Tasks:
1. With `x = [1.0, 4.0, 9.0, 16.0]` and `y = [2.0, 2.0, 3.0, 4.0]`, compute element-wise add,
   subtract, multiply, and divide into `add_xy`, `sub_xy`, `mul_xy`, `div_xy`.
2. Build a `(4, 4)` random normal tensor `mat`; compute its global mean (`mat_mean`) and std
   (`mat_std`).
3. Apply ReLU to `torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])` into `relu_out`.
4. With `P = torch.ones(3, 4)` and `Q = torch.ones(4, 5) * 3.0`, compute the matrix product
   `PQ`.
```

## Cell 12 - Code: Lab 2 starter

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 72593e14

```
torch.manual_seed(SEED)

# 1. Element-wise arithmetic
x = torch.tensor([1.0, 4.0, 9.0, 16.0])
y = torch.tensor([2.0, 2.0, 3.0,  4.0])

add_xy = None  # YOUR CODE
sub_xy = None  # YOUR CODE
mul_xy = None  # YOUR CODE
div_xy = None  # YOUR CODE

# 2. Mean and std of a (4,4) random matrix
mat = torch.randn(4, 4)
mat_mean = None  # YOUR CODE
mat_std  = None  # YOUR CODE

# 3. ReLU applied to torch.tensor([-2., -1., 0., 1., 2.])
relu_out = None  # YOUR CODE

# 4. Matrix product of P and Q
P  = torch.ones(3, 4)
Q  = torch.ones(4, 5) * 3.0
PQ = None  # YOUR CODE

# ---- Verification ----
if add_xy is not None:
    print(f"x + y  = {add_xy.tolist()}")
    print(f"x - y  = {sub_xy.tolist()}")
    print(f"x * y  = {mul_xy.tolist()}")
    print(f"x / y  = {div_xy.tolist()}")
if mat_mean is not None:
    print(f"\nmat mean = {mat_mean.item():.4f}  std = {mat_std.item():.4f}")
if relu_out is not None:
    print(f"relu_out = {relu_out.tolist()}")
if PQ is not None:
    print(f"PQ shape = {PQ.shape}  (expected (3, 5))  all entries 12.0: {(PQ == 12.0).all().item()}")
```

## Cell 13 - Markdown: Section 3 theory - Reshaping

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell b50a00f1
Change: add the view-vs-reshape contiguous gotcha (research-validated) to the theory.

```
## Section 3 - Manipulating Shapes

Reshaping is the plumbing of model wiring: one layer outputs `(B, T, D)` and the next wants
`(B, T*D)`. You will do this constantly.

| Method | Notes |
|--------|-------|
| `.reshape(new_shape)` | Returns a view if possible, else a copy. Safe general purpose. |
| `.view(new_shape)` | Always a view, but the tensor MUST be contiguous in memory. Faster, stricter. |
| `.transpose(d0, d1)` | Swaps two dims. Returns a view (and makes the tensor non-contiguous). |
| `.permute(*dims)` | Reorders all dims at once. Returns a view. |
| `.flatten(start, end)` | Collapses a range of dims into one. |
| `.squeeze(dim)` | Removes size-1 dims. No `dim` removes all of them. |
| `.unsqueeze(dim)` | Inserts a new size-1 dim at `dim`. |

The number-one reshape gotcha: `.view()` fails after `.transpose()` or `.permute()`, because those
return a non-contiguous view and `.view()` needs contiguous memory. PyTorch raises a RuntimeError
that literally tells you to use `.reshape()`. Two fixes:

```python
t2 = t.transpose(0, 1)
t2.view(-1)                 # RuntimeError: not compatible with size/stride
t2.reshape(-1)              # works: reshape copies if it has to
t2.contiguous().view(-1)    # also works: force a contiguous copy first
```

Rule of thumb: reach for `.reshape()` unless you have a reason to want `.view()`.

Common pattern - adding a batch dimension:
```python
x = torch.randn(28, 28)   # one image
x = x.unsqueeze(0)        # (1, 28, 28) - batch of 1
```

### Demo - shape manipulation
```

## Cell 14 - Code: Section 3 demo

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 7a37a57b
Change: append a try/except that triggers the real view-after-transpose RuntimeError, then shows the
reshape fix.

```
t = torch.arange(16, dtype=torch.float32)  # [0, 1, ..., 15]
print("Original shape:", t.shape)

r1 = t.reshape(4, 4)
print("reshape(4,4)   :", r1.shape)

r2 = t.reshape(2, 8)
print("reshape(2,8)   :", r2.shape)

# transpose swaps two dims (and yields a non-contiguous view)
r3 = r1.transpose(0, 1)
print("transpose(0,1) :", r3.shape, "  (rows become columns)")

# permute on a 3-D tensor
t3 = torch.randn(2, 3, 5)
r4 = t3.permute(2, 0, 1)   # (2,3,5) -> (5,2,3)
print(f"permute(2,0,1) on {t3.shape}: {r4.shape}")

# flatten and squeeze/unsqueeze
t4 = torch.zeros(2, 1, 3, 1)
print(f"\nsqueeze()      : {t4.shape} -> {t4.squeeze().shape}")
print(f"unsqueeze(0)   : {t.shape} -> {t.unsqueeze(0).shape}")
print(f"flatten()      : {r1.shape} -> {r1.flatten().shape}")

# The classic gotcha: view() on a transposed (non-contiguous) tensor fails.
print("\n--- view vs reshape after transpose ---")
try:
    bad = r3.view(-1)
except RuntimeError as e:
    print(f"r3.view(-1) raised: {type(e).__name__} (non-contiguous)")
print("r3.reshape(-1) works:", r3.reshape(-1).shape)
```

## Cell 15 - Markdown: Lab 3 instructions

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 24e9cc63

```
### Lab 3 - Manipulating Shapes

Tasks:
1. Create `base = torch.arange(24, dtype=torch.float32)`. Reshape it to `(3, 8)` -> `reshaped_3x8`.
2. Reshape `base` to `(2, 3, 4)` -> `reshaped_3d`.
3. Transpose `reshaped_3x8` (swap dim 0 and dim 1) -> `transposed`.
4. Permute `reshaped_3d` from `(2, 3, 4)` to `(4, 2, 3)` -> `permuted`.
5. Flatten `reshaped_3d` entirely -> `flat`.
6. Start with `img = torch.randn(28, 28)`. Add a batch dimension at position 0 -> `img_batched`
   (shape should be `(1, 28, 28)`).
7. Remove that batch dimension -> `img_back` (shape should be `(28, 28)`).
```

## Cell 16 - Code: Lab 3 starter

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 3a6c0004

```
base = torch.arange(24, dtype=torch.float32)
img  = torch.randn(28, 28)

# 1. Reshape to (3, 8)
reshaped_3x8 = None  # YOUR CODE

# 2. Reshape to (2, 3, 4)
reshaped_3d  = None  # YOUR CODE

# 3. Transpose reshaped_3x8 (swap dim 0 and dim 1)
transposed   = None  # YOUR CODE

# 4. Permute reshaped_3d to (4, 2, 3)
permuted     = None  # YOUR CODE

# 5. Flatten reshaped_3d entirely
flat         = None  # YOUR CODE

# 6. Add a batch dim to img -> (1, 28, 28)
img_batched  = None  # YOUR CODE

# 7. Remove that batch dim -> (28, 28)
img_back     = None  # YOUR CODE

# ---- Verification ----
checks = [
    ('reshaped_3x8', reshaped_3x8, (3, 8)),
    ('reshaped_3d',  reshaped_3d,  (2, 3, 4)),
    ('transposed',   transposed,   (8, 3)),
    ('permuted',     permuted,     (4, 2, 3)),
    ('flat',         flat,         (24,)),
    ('img_batched',  img_batched,  (1, 28, 28)),
    ('img_back',     img_back,     (28, 28)),
]
for name, t, expected in checks:
    if t is not None:
        status = "OK" if tuple(t.shape) == expected else f"WRONG (got {tuple(t.shape)})"
        print(f"{name:15s} shape={tuple(t.shape)}  expected={expected}  [{status}]")
```

## Cell 17 - Markdown: Section 4 theory - Indexing and Broadcasting

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 04877aab + cell a80e4ba2
Change: merge the indexing theory and the broadcasting theory into one section to fit the cell
budget.

```
## Section 4 - Indexing and Broadcasting

These two go together: you slice a tensor to pull out the rows you want, and you broadcast to
combine tensors of different shapes without copying.

### Indexing - identical to NumPy

```python
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

t[0]        # first row -> tensor([1, 2, 3])
t[0, 2]     # row 0, col 2 -> tensor(3)
t[1:3, :]   # rows 1 and 2, all columns
t[:, 1]     # all rows, column 1
t[t > 4]    # boolean mask: every element greater than 4
```

Boolean masking is everywhere in NLP: dropping padding tokens, keeping high-confidence predictions,
filtering embeddings.

### Broadcasting - the rules

Broadcasting lets PyTorch combine different shapes without copying data. Same rules as NumPy:

1. If the tensors have different numbers of dims, prepend 1s to the smaller shape.
2. Dimensions of size 1 are stretched to match the other tensor.
3. If sizes still disagree after rules 1 and 2, you get an error.

```
(3, 4) + (4,)   -> prepend 1 -> (1, 4) -> stretch -> (3, 4)   OK
(3, 1) + (1, 4) -> stretch both -> (3, 4)                     OK
(3, 4) + (3, 3) -> 4 vs 3 cannot align                        ERROR
```

This is exactly how a bias add works inside `nn.Linear`: the bias is `(out_features,)` and gets
broadcast across every row of the `(batch, out_features)` output. You are previewing B6 plumbing.

### Demo - indexing and broadcasting
```

## Cell 18 - Code: Section 4 demo

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell df4760d1 + cell 1371f1a6
Change: merge the indexing demo and the broadcasting demo (including the intentional-error case)
into one cell.

```
# ---- Indexing ----
grid = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
print("grid:\n", grid)
print("\ngrid[0]        =", grid[0].tolist(),   "  (first row)")
print("grid[0, 2]     =", grid[0, 2].item(),    "  (row 0, col 2)")
print("grid[1:3, :]   =\n", grid[1:3, :])
print("grid[:, 1]     =", grid[:, 1].tolist(),  "  (middle column)")

mask = grid > 4
print("\ngrid > 4 mask:\n", mask)
print("grid[grid > 4] =", grid[grid > 4].tolist())  # boolean masking

# ---- Broadcasting ----
print("\n--- broadcasting ---")
# Case 1: (3, 4) + (4,) -> (3, 4). This is the bias-add pattern.
M = torch.ones(3, 4)
row = torch.tensor([0.0, 1.0, 2.0, 3.0])    # shape (4,)
print("(3,4) + (4,) ->", (M + row).shape)
print(M + row)

# Case 2: (3, 1) + (1, 4) -> (3, 4). Both dims stretch (outer-product shape).
col  = torch.tensor([[10.0], [20.0], [30.0]])   # (3, 1)
row2 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])     # (1, 4)
print("\n(3,1) + (1,4) ->", (col + row2).shape)
print(col + row2)

# Case 3: intentional failure to read the error message
try:
    bad = torch.ones(3, 4) + torch.ones(3, 3)
except RuntimeError as e:
    print(f"\nExpected error (4 vs 3 cannot broadcast): {type(e).__name__}")
```

## Cell 19 - Markdown: Lab 4 instructions (with stretch)

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell e8566889 + cell a563d635
Change: merge the indexing lab and broadcasting lab; add a labelled in-notebook stretch
(pairwise-distance matrix via unsqueeze broadcasting) that previews B5 similarity.

```
### Lab 4 - Indexing and Broadcasting

Use `data` and the broadcasting tensors created in the starter.

Core tasks:
1. Extract the element at row 1, column 2 -> `elem`.
2. Extract rows 1 and 2 (all columns) -> `rows_1_2`.
3. Extract the last column (all rows) -> `last_col`.
4. Boolean-mask all elements less than 0 -> `negatives`.
5. Add the bias vector `bias` (shape `(5,)`) to every row of `features` (shape `(6, 5)`) ->
   `biased`. The result should be `(6, 5)`.
6. Multiply the column vector `col_scale` (shape `(3, 1)`) by the row vector `row_weights`
   (shape `(1, 3)`) -> `scaled`. The result should be `(3, 3)`.

Stretch (fast finishers): build a pairwise distance matrix without any loop. Given `pts` of shape
`(N, D)`, compute `dists` of shape `(N, N)` where `dists[i, j]` is the Euclidean distance between
row i and row j. Hint about the SHAPES only (not the call): you want to subtract a
`(N, 1, D)`-shaped view from a `(1, N, D)`-shaped view so broadcasting produces an `(N, N, D)`
difference, then reduce the last dimension. This is the exact trick behind the sentence-similarity
work in B5.
```

## Cell 20 - Code: Lab 4 starter (with stretch)

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 78d6cded + cell 501aabfd
Change: merge the two lab starters; de-leak comments; add the pairwise-distance stretch block whose
hint never names `unsqueeze`, `norm`, or the exact ops.

```
torch.manual_seed(SEED)

# ---- Indexing data ----
data = torch.randn(4, 5) * 5   # (4, 5) values roughly in [-15, 15]
print("data:\n", data.round(decimals=2))

# 1. Element at row 1, column 2
elem      = None  # YOUR CODE

# 2. Rows 1 and 2, all columns
rows_1_2  = None  # YOUR CODE

# 3. Last column (all rows)
last_col  = None  # YOUR CODE

# 4. All elements less than 0 (boolean mask)
negatives = None  # YOUR CODE

# ---- Broadcasting data ----
features    = torch.randn(6, 5)
bias        = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])     # (5,)
col_scale   = torch.tensor([[2.0], [3.0], [4.0]])         # (3, 1)
row_weights = torch.tensor([[1.0, 0.5, 0.25]])            # (1, 3)

# 5. Add bias to every row of features -> (6, 5)
biased = None  # YOUR CODE

# 6. Combine col_scale and row_weights so the result is (3, 3)
scaled = None  # YOUR CODE

# ---- Stretch: pairwise distance matrix (no loops) ----
# Goal: dists[i, j] = Euclidean distance between pts[i] and pts[j]. Final shape (N, N).
pts   = torch.randn(5, 3)   # N=5 points in D=3 dimensions
dists = None  # YOUR CODE  (shape (5, 5); the diagonal should be ~0)

# ---- Verification ----
if elem is not None:
    print(f"\nelem      = {elem.item():.4f}")
if rows_1_2 is not None:
    print(f"rows_1_2  shape={rows_1_2.shape}  (expected (2, 5))")
if last_col is not None:
    print(f"last_col  shape={last_col.shape}  (expected (4,))")
if negatives is not None:
    print(f"negatives = {negatives.round(decimals=2).tolist()}")
if biased is not None:
    print(f"biased    shape={biased.shape}  (expected (6, 5))")
if scaled is not None:
    print(f"scaled    shape={scaled.shape}  (expected (3, 3))\n{scaled}")
if dists is not None:
    print(f"dists     shape={dists.shape}  (expected (5, 5))")
    print(f"diagonal ~0: {torch.allclose(dists.diagonal(), torch.zeros(5), atol=1e-4)}")
```

## Cell 21 - Markdown: Section 5 theory - Autograd

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 5c32f0fd
Change: keep the single `tf.GradientTape` comparison (the audience migrates from TF, per the intent
stub); add the research-validated notes that `.backward()` needs a scalar, gradients accumulate,
and only float tensors can require grad.

```
## Section 5 - Autograd and Differentiation

This is where PyTorch leaves NumPy behind. Autograd is the engine that makes backpropagation
automatic - and it is the entire reason the rest of this course works.

The idea: mark a tensor with `requires_grad=True` and PyTorch records every operation on it in a
computation graph. Call `.backward()` on a scalar loss and gradients flow backward through that
graph by the chain rule, landing in each tensor's `.grad`.

```python
w = torch.tensor(3.0, requires_grad=True)
loss = w ** 2          # builds the graph loss = w^2
loss.backward()        # d(loss)/d(w) = 2w = 6
print(w.grad)          # tensor(6.)
```

Three rules worth burning in now, because every one of them bites beginners:

1. `.backward()` needs a SCALAR. If your output is a vector or matrix, reduce it first with
   `.mean()` or `.sum()`, otherwise PyTorch raises "grad can be implicitly created only for scalar
   outputs."
2. Gradients ACCUMULATE. Each `.backward()` ADDS into `.grad`. In a training loop you must zero them
   between steps (an optimizer does this for you via `zero_grad()`; doing it by hand you call
   `.grad.zero_()`).
3. Only floating-point tensors can require gradients. `torch.tensor([1, 2, 3], requires_grad=True)`
   errors because it is integer - use `dtype=torch.float32`.

Coming from TensorFlow, this replaces `tf.GradientTape`:

```python
# TF2
with tf.GradientTape() as tape:
    loss = w ** 2
grad = tape.gradient(loss, w)

# PyTorch - no context manager needed
loss = w ** 2
loss.backward()
grad = w.grad
```

And `torch.no_grad()` switches tracking off for inference, to save memory and time:

```python
with torch.no_grad():
    predictions = model(x)   # no graph built
```

### Demo - autograd
```

## Cell 22 - Code: Section 5 demo

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 12531c79

```
# Demo 1: simple scalar gradient
w = torch.tensor(3.0, requires_grad=True)
loss = w ** 2          # loss = w^2, so d/dw = 2w
loss.backward()
print(f"w = {w.item()},  loss = {loss.item()},  grad = {w.grad.item()}")
print(f"Expected grad: 2 * w = {2 * w.item()}")

# Demo 2: gradient of a two-variable function
# f(x, y) = x^2 + 3*x*y   at x=2, y=5
# df/dx = 2x + 3y = 19,  df/dy = 3x = 6
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(5.0, requires_grad=True)
f = x**2 + 3 * x * y
f.backward()
print(f"\nf(2,5) = {f.item()},  df/dx = {x.grad.item()},  df/dy = {y.grad.item()}")
print(f"Expected: df/dx = {2*x.item() + 3*y.item()},  df/dy = {3*x.item()}")

# Demo 3: torch.no_grad() disables tracking (inference mode)
with torch.no_grad():
    z = w ** 3
    print(f"\nz.requires_grad = {z.requires_grad}  (should be False)")
```

## Cell 23 - Markdown + Code: Lab 5 (autograd), stretch (manual gradient descent), homework

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 04a7e2b0 + cell 9ecaf823 +
cell 353a23eb + cell 96dde56b
Change: this is a composite cell group. The lab markdown folds in the stretch and homework; the lab
starter de-leaks the old `# YOUR CODE: a ** 3` style comments (which gave the answer away) and
appends the manual gradient-descent stretch starter. /build-notebook should render this as one
markdown lab cell followed by one starter code cell (the stretch lives inside the same starter).

Markdown (lab cell):

```
### Lab 5 - Autograd (core + stretch + homework)

Core tasks:
1. Create a scalar tensor `a` holding `4.0` that tracks gradients. Compute `loss_a = a ** 3`,
   call backward, and store the gradient in `grad_a`. It should equal `3 * a^2 = 48`.
2. Create scalar tensors `p = 2.0` and `q = -3.0` that both track gradients. Compute
   `loss_pq = p * q + p ** 2`, call backward, and store the gradients in `grad_p` and `grad_q`.
   Check them against the calculus: d/dp = q + 2p, d/dq = p.
3. Recompute `p * q + p ** 2` inside a no-grad context into `loss_no_grad`, and confirm that
   `loss_no_grad.requires_grad` is `False`.

Stretch (in-notebook, fast finishers) - gradient descent by hand:
Fit `y = 2x + 1` from noisy data WITHOUT an optimizer, so you see what `optimizer.step()` will
automate in B6. Initialize `w` and `b` to track gradients, then loop: compute the prediction,
compute MSE loss, call backward, and update the parameters inside a no-grad block. Remember rule 2
from the theory - you must zero the gradients each step or they pile up. After ~200 steps `w` should
be near 2.0 and `b` near 1.0.

Homework extension (async, production-grade) - gradient checking:
Real teams verify a hand-written backward pass against a numerical gradient before trusting it. Take
`f(x) = x**3 + 2*x` at `x = 1.5`. Compute the autograd gradient with `.backward()`. Then compute the
numerical gradient with the central finite-difference formula `(f(x+eps) - f(x-eps)) / (2*eps)` for
a small `eps` (use `dtype=torch.float64` for precision). Confirm the two agree to within `1e-4`.
This is exactly the idea behind `torch.autograd.gradcheck`, the tool you would use when writing a
custom autograd Function. Write it up in your own notebook cell.
```

Starter (code cell):

```
# ---- Core ----

# 1. Gradient of a^3 at a = 4.0
a      = None  # YOUR CODE: a scalar tensor 4.0 that tracks gradients
loss_a = None  # YOUR CODE: a cubed
# YOUR CODE: trigger backpropagation on loss_a
grad_a = None  # YOUR CODE: read the gradient of a

# 2. Two-variable gradient at p = 2.0, q = -3.0
p       = None  # YOUR CODE: scalar tensor 2.0 that tracks gradients
q       = None  # YOUR CODE: scalar tensor -3.0 that tracks gradients
loss_pq = None  # YOUR CODE: p*q + p squared
# YOUR CODE: trigger backpropagation on loss_pq
grad_p  = None  # YOUR CODE: gradient of p
grad_q  = None  # YOUR CODE: gradient of q

# 3. Same expression as task 2, but with gradient tracking switched off
loss_no_grad = None  # YOUR CODE: recompute the task-2 expression inside a no-grad context

# ---- Verification (core) ----
if grad_a is not None:
    expected_a = 3 * (4.0 ** 2)
    print(f"grad_a = {grad_a.item():.1f}  expected = {expected_a:.1f}  "
          f"OK={abs(grad_a.item()-expected_a)<1e-4}")
if grad_p is not None and grad_q is not None:
    exp_p = -3.0 + 2 * 2.0   # q + 2p = 1
    exp_q = 2.0              # p = 2
    print(f"grad_p = {grad_p.item():.1f}  expected = {exp_p:.1f}  OK={abs(grad_p.item()-exp_p)<1e-4}")
    print(f"grad_q = {grad_q.item():.1f}  expected = {exp_q:.1f}  OK={abs(grad_q.item()-exp_q)<1e-4}")
if loss_no_grad is not None:
    print(f"loss_no_grad.requires_grad = {loss_no_grad.requires_grad}  (should be False)")

# ---- Stretch: gradient descent by hand ----
torch.manual_seed(SEED)
lr, STEPS = 0.1, 200
x_gd = torch.linspace(-1, 1, 50)
y_gd = 2 * x_gd + 1 + 0.1 * torch.randn(50)   # ground truth y = 2x + 1 + noise

w_gd = None  # YOUR CODE: scalar 0.0 that tracks gradients
b_gd = None  # YOUR CODE: scalar 0.0 that tracks gradients

for step in range(STEPS):
    if w_gd is None or b_gd is None:
        break
    y_pred_gd = None  # YOUR CODE: predict w_gd * x_gd + b_gd
    loss_gd   = None  # YOUR CODE: mean squared error between y_pred_gd and y_gd
    # YOUR CODE: backpropagate loss_gd
    with torch.no_grad():
        pass  # YOUR CODE: step w_gd and b_gd down their gradients by lr
    # YOUR CODE: zero the gradients of w_gd and b_gd for the next step

if w_gd is not None and w_gd.grad is not None:
    print(f"\nLearned w = {w_gd.item():.4f}  (expected ~2.0)")
    print(f"Learned b = {b_gd.item():.4f}  (expected ~1.0)")
```

## Cell 24 - Markdown: Wrap-up, key takeaways, bridge to B5

Provenance: FROM-OLD Frameworks/PyTorch_Tensor_Exercises.ipynb cell 67cb5e23
Change: retheme the takeaways table to the five sections we kept (drop DataLoader/nn rows), repoint
"next steps" to B5, and add the production hygiene take-homes from research.

```
## You built the bricks

Here is what you now have in your hands:

| Section | Core skill |
|---------|-----------|
| 1 - Basics | `torch.tensor`, factory functions, `.shape`, `.dtype`, numpy round-trip |
| 2 - Operations | arithmetic, `mean`/`std`, `relu`, `@` matrix multiply |
| 3 - Reshaping | `reshape`/`view` (and when `view` fails), `transpose`/`permute`, `flatten`, `squeeze` |
| 4 - Indexing + Broadcasting | slicing, boolean masks, the bias-add broadcast pattern |
| 5 - Autograd | `requires_grad`, `.backward()`, `.grad`, `torch.no_grad()`, hand-rolled gradient descent |

### Production hygiene (carry these forward)

- At inference, wrap the forward pass in `with torch.no_grad():` (or the newer, faster
  `torch.inference_mode()`) so you do not build a graph you will throw away.
- Use `.detach()` or `.item()` when you log a loss, so you do not accidentally keep the whole graph
  alive.
- Keep your model and your data on the SAME device, or you get a device-mismatch error.

### Self-check

1. Why must the argument to `.backward()` be a scalar, and how do you make it one?
2. Why does `.view()` fail right after `.transpose()`, and what are two fixes?
3. What happens if you forget to zero gradients inside a training loop?

### Next: B5 - Word2Vec and Sentence Embeddings

You can now build and differentiate a tensor expression. Next we turn real sentences into the
384-dimensional float tensors (from `all-MiniLM-L6-v2`) that this exact machinery will train on. The
gradient descent you just ran by hand is what `optimizer.step()` will automate when we build the
network in B6 and the word2vec MLP in B7.
```

---

# VERIFICATION CHECKLIST

- [ ] Install cell pins `numpy<2` and carries the Colab restart note; no `pip install torch`.
- [ ] Import cell drops `TensorDataset`/`DataLoader` (those move to B6) and sets `SEED=42`, `device`.
- [ ] Exactly five concept sections: Basics, Operations, Reshaping, Indexing+Broadcasting, Autograd.
- [ ] No more than three theory cells run without a code cell between them.
- [ ] Every section has a Theory markdown, a runnable Demo, and a Lab with verification.
- [ ] Reshape theory + demo include the view-after-transpose contiguity gotcha and the reshape fix.
- [ ] Autograd theory states the three rules: scalar `.backward()`, gradient accumulation, float-only
      `requires_grad`; keeps exactly one `tf.GradientTape` comparison.
- [ ] Lab 4 stretch (pairwise-distance matrix) hints at SHAPES only and never names `unsqueeze`/`norm`.
- [ ] Lab 5 starter comments are de-leaked (no `# YOUR CODE: a ** 3`-style answers); the stretch
      (manual gradient descent) and the homework (finite-difference gradient checking) are present.
- [ ] Every `None # YOUR CODE` placeholder hides the answer; verification cells only reveal
      correctness, not the solution.
- [ ] Through-line is explicit: 384-dim embedding -> tensor -> nn.Linear -> MLP -> DistilBERT chatbot.
- [ ] Wrap-up bridges to B5 and folds in the `no_grad`/`inference_mode`/`detach`/device take-homes.
- [ ] STAR story and chatbot through-line both present.
- [ ] Solution notebook fills every placeholder with commented code and a short explanation note.
- [ ] Total cell count is 25 (Cell 0 through Cell 24), within the 20-25 ceiling.
- [ ] Plain ASCII only; no em/en dashes, no Unicode multiply sign, no emoji.

# RESEARCH VALIDATED (June 2026)

- https://docs.pytorch.org/docs/stable/autograd.html - autograd is define-by-run; `.backward()`
  populates `.grad` on leaf tensors with `requires_grad=True`; `torch.no_grad()` disables tracking.
- https://docs.pytorch.org/docs/2.12/tensors.html - tensor factory functions
  (`tensor`/`zeros`/`ones`/`rand`/`randn`) and the `dtype`/`device` constructor arguments are
  current in PyTorch 2.x.
- https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html - `torch.from_numpy` shares
  memory with the source array; `torch.tensor(arr)` copies.
- https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152
  - `.backward()` requires a scalar; reduce a non-scalar output with `.mean()`/`.sum()` first.
- https://github.com/pytorch/pytorch/issues/764 and
  https://clay-atlas.com/us/blog/2024/08/21/en-runtimeerror-view-size-not-compatible-with-input/ -
  `.view()` raises a RuntimeError on a non-contiguous (e.g. transposed) tensor; fix with `.reshape()`
  or `.contiguous().view()`.
- https://github.com/pytorch/pytorch/issues/40264 - only floating-point (and complex) tensors can
  set `requires_grad=True`; integer tensors raise an error, so use `dtype=torch.float32`.
- https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html - gradients
  accumulate into `.grad` across `.backward()` calls; you must zero them each training step.
- https://developers.google.com/colab/release-notes and
  https://github.com/googlecolab/colabtools/issues/5390 - Colab ships PyTorch 2.6 and numpy 2.0.2
  preinstalled; pinning `numpy<2` requires a runtime restart.
- https://github.com/pytorch/pytorch/issues/107302 - PyTorch >= 2.2 builds are NumPy-2 ABI
  compatible, but the course standardizes on `numpy<2` for gensim/scipy consistency in B5.
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 - `all-MiniLM-L6-v2` maps text to a
  384-dimensional dense vector; this is the exact input width for the B7 MLP.
- https://www.python-engineer.com/courses/pytorchbeginner/05-gradient-descent/ and
  https://www.deep-teaching.org/notebooks/differentiable-programming/pytorch/exercise-pytorch-univariate-linear-regression
  - manual gradient descent (update under `torch.no_grad()`, then `.grad.zero_()`) is the canonical
  exercise that shows what `optimizer.step()` automates.
- https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html - gradient
  checking compares autograd's analytical gradient against a finite-difference numerical gradient and
  needs float64 precision; the basis for the homework extension.
- https://docs.pytorch.org/docs/stable/notes/randomness.html - set seeds early; `torch.manual_seed`
  plus `np.random.seed` is the standard reproducibility idiom used course-wide (`SEED=42`).
- https://www.geeksforgeeks.org/deep-learning/difference-between-detach-and-with-torchnograd-in-pytorch/
  - at inference use `model.eval()` with `torch.no_grad()` (or `torch.inference_mode()`); use
  `.detach()`/`.item()` when logging; keep model and data on the same device.
```
