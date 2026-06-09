Adversarial audit of  
- `exercises/2-Build-It/4-PyTorch_Tensors.ipynb` (student copy)  
- `Solutions/2-Build-It/4-PyTorch_Tensors.ipynb` (key)  

==================================================================
1. Execution correctness
- All demo cells import `torch`, `numpy`, `nn` only after the setup cell, so in‐order execution in Colab works.  
- No undefined names detected; every later cell uses variables it defines locally (`np_arr`, `base`, `data`, …).  
- Potential break: the install cell (code cell 2 in both notebooks) uses shell syntax `!pip install -q "numpy<2"`. Outside Jupyter this is a syntax error.  
- No cross-GPU/CPU mixing; `device` is printed only, never used.  
- Autograd demos accumulate into `.grad` but never re-use gradients in a way that would raise an error.

2. Pedagogical gaps
- Section 3 demos `.view()` vs `.reshape()` **before** explaining “contiguous” memory; the term appears without definition (markdown cell 134).  
- In Lab 5 the task asks for “gradient descent by hand” before any loop/optimizer pattern has been shown (students have not yet seen `optimizer.step()`; reference is forward-looking).  
- The Euclidean-distance “stretch” in Lab 4 relies on broadcasting with three-tensor subtraction that has not been demonstrated; only a hint is given.

3. Lab reality & leakage
REAL labs: five genuine fill-in-the-blank code cells (`s = None`, etc.).  
Leakage:  
- Lab 5 prompt literally states the numeric target: “It should equal `3 * a^2 = 48`.” (markdown cell 259) — gives the answer away.  
- Verification code prints “all entries 12.0” (Lab 2) and “diagonal ~0” (Lab 4) — minor hints but acceptable. No other starter comment exposes code.

4. Solution-notebook hygiene
- All answer cells are fully filled; **none of them are hidden or tagged**. Metadata for the answer code cell `id="6e55899d"` is empty, so students opening the solutions file see answers immediately.  
- The solution notebook keeps a duplicate *empty* version of the Lab 5 cell (has `# YOUR CODE`) after the solved one — confusing but harmless.

5. Plain-script viability
- Fails as a script because of `!pip …` shell magic (cell 2) and markdown cells containing back-tick mermaid fences (ignored but fine). Replace with `subprocess` or `%pip` if scriptability is desired.

6. Environment pin
- Install cell correctly pins `numpy<2` in **both** notebooks: `!pip install -q "numpy<2"`.

==================================================================
Recommended fixes (priority order)

1. Tag every filled answer cell in the *solution* notebook with `{"tags": ["solution"], "jupyter": {"source_hidden": true}}` or use Colab’s “Hide code” to collapse by default.  
2. Remove numeric answer hints from prompts, e.g. change “should equal 48” → “should equal `3 * a**2`”.  
3. Replace `!pip` with `%pip` magic (works in notebooks, ignored by python scripts) or move installation into a guarded Python block for script compatibility.  
4. Add one-sentence definition of “contiguous tensor” before using it in Section 3 to avoid concept jump.  
5. Show a short broadcasting demo that subtracts `(N,1,D)` from `(1,N,D)` **before** assigning the Lab 4 stretch goal.  
6. Delete the leftover empty Lab 5 cell in the solution notebook to prevent confusion.

Ranking reflects impact on learners: visibility of answers > leaking numeric target > runtime portability > conceptual clarity.