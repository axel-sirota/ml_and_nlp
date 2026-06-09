Notebook audited:  
- exercises/2-Build-It/6-PyTorch_NN_Basics.ipynb (“exercise”)  
- Solutions/2-Build-It/6-PyTorch_NN_Basics.ipynb (“solution”)

--------------------------------------------------------------------
1. Correct execution
- Setup/import cell (exercise cell 3, starts with “import numpy as np”) runs.  
- First lab cell (cell 6, “# Lab 1: create and configure layers”) executes because prints are gated by `if fc_layer is not None`, but later code depends on the variables being real layers—so the notebook **breaks on the very next section** when `SentimentMLP` tries to read `nn.Linear` objects that were never built (undefined attribute error).  
- Training loop section (cell 18, `train_ds = None  # YOUR CODE`) will raise at `TensorDataset` construction if still `None`.  
- No circular-dependency or out-of-order imports were found; all helpers (`train_model`) are defined before use.  
⇒ Each logical section after a “# YOUR CODE” block will error until the student fills it in.

2. Pedagogical gaps
- `TensorDataset`/`DataLoader` are used in the first lab **before** any narrative that explains what they are; explanation only appears two markdown cells later.  
- The forward diagram for `SentimentMLP` is shown, but no worked example of calling a small `nn.Module` is provided before students must implement `forward`.  
- Dropout is required in Lab 1 yet the term “inference disables dropout” is not defined anywhere in the preceding text.

3. Lab quality & leakage
- Labs are genuine: every task is an executable stub (`None  # YOUR CODE`).  
- No code answers are leaked, but the verification lines reveal exact target shapes/values, e.g.  
  Cell 6 line 16: `print(f"fc_layer weight shape : … (expected torch.Size([64, 100]))")`  
  That signals the required dimensions, slightly lowering discovery value.

4. Solution notebook hygiene
- All answers are fully filled in (as expected).  
- Safety-net prints are still present and **not collapsed** (no `metadata: {“tags”: [“solution”]}` or `"collapsed": true`). Anyone who opens the solution sees the answer immediately.

5. Plain-script run
- Linear execution would work **only if** every `# YOUR CODE` is completed; there are no hidden cell-order dependencies or Jupyter-only magics besides the single `!pip install ...` line.

6. numpy pin
- Both notebooks contain the install cell pinning exactly `numpy<2`:  
  `!pip install -q "numpy<2"`

--------------------------------------------------------------------
Ranked concrete fixes
1. Wrap every verification/safety-net block in an **initially-collapsed** code cell (`"tags": ["hidden"]`) so answers aren’t visible at first glance.  
2. Insert a short demo cell that builds a tiny `nn.Module` and calls `DataLoader` **before** Lab 1 to close the conceptual gap.  
3. Explain dropout’s train-vs-eval behaviour in the markdown that precedes its first use.  
4. Replace “expected …” strings with generic placeholders (“shape should be (64, 100)”) or move them into the hidden safety-net so dimension hints aren’t exposed.  
5. Add a final “Run-all test” cell that raises an informative error if any `None` placeholders survive, instead of failing deep inside the training loop.  
6. In the solution notebook, tag answer cells with `"solution"` and collapse them by default to slow accidental spoilers.