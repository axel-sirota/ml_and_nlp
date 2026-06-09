Audit of `exercises/2-Build-It/7-MLP_Word2Vec.ipynb` (“EX”) and `solutions/…` (“SOL”)  
Cell numbers refer to each notebook individually (they line up one-for-one).

1. Runtime integrity  
   - EX cell 16 creates a stub `SentimentMLP` whose layers are `None`; the forward pass in the same cell crashes (`AttributeError: 'NoneType' object is not callable`). Safety-net cell 17 is never executed because it only runs when *no* `SentimentMLP` is in `globals()`.  
   - EX cell 22 assumes `history`, `best_path` and a trained `model` exist. If the student skipped the loop, `history` is still undefined, so the first line of safety-net cell 23 (`if not history['val_acc'] …`) raises `NameError`.  
   - All later demo cells (18, 24-28) depend on a working `model`, `train_loader`, `best_path`, etc.; they will cascade-fail if the above labs are not finished.  
   - Import / name errors: none (Matplotlib & Seaborn are correctly imported in cell 3; `sns` is used later).  
   - SOL runs end-to-end, but it inherits the same fragile safety-net logic: if an earlier cell is skipped out-of-order the same NameErrors will happen.  
   - Plain-script execution (`jupyter nbconvert --to py && python`): will fail because of the same guard-logic bugs **and** because magic line `!pip install …` (cell 2) is a notebook-only shell escape.

2. Pedagogical gaps  
   - Cross-entropy vs. accuracy/F1 is *used* (cells 19, 26) before it is ever defined or linked to previous lessons.  
   - Early stopping hyper-parameters (`PATIENCE`, `best_path`) appear in cell 3, but the notion of early stopping is explained only in the graphic caption of cell 21; no textual walkthrough precedes the lab.  
   - The “averaged word2vec ceiling” (Section 3, cells 29-30) claims that order is lost, yet order was never *demonstrated* with a worked example before asking students to break the model.

3. Lab quality & leakage  
   - Labs are genuine coding tasks (`None  # YOUR CODE` placeholders, asserts).  
   - Every safety-net immediately **exposes the full reference code**. Example: EX cell 15:  
     ```
     y_train_t = torch.tensor(y_train, dtype=torch.long)
     … train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
     ```  
     A student can simply copy-paste instead of thinking.  
   - Guiding comments themselves leak answers, e.g. EX cell 9:  
     “`# transform (NOT fit_transform) reuses the train statistics, so no test info leaks in.`”

4. Solution-notebook disclosure controls  
   - SOL contains all filled answers (expected) **but** no cell metadata marks them as collapsed / hidden. Opening the file shows the answers first thing, defeating “peek-only-if-stuck” intent.  
   - Safety-nets in SOL still print the code openly; nothing is collapsed.

5. Scriptability  
   - Notebook relies on IPython magics (`!pip install`) and Colab GPU detection; will not run in a vanilla `python` script.  
   - Guard logic assumes interactive execution order (see §1); without it, converting to a script fails.

6. Dependency pin  
   - Cell 2 correctly pins `numpy<2` (“numpy<2” is in the `pip install` line).

-----------------------------------------------------------------
Concrete ranked fixes (highest impact first)

1. Rewrite safety-nets:  
   - Guard on *success flags* (`completed_lab1 = False`) instead of testing variable existence; run fallback only if the flag is `False`.  
   - Move fallback implementations into *collapsed* Colab-code-folds (`"metadata": {"colab":{"collapsed":true}}}`) or separate, instructor-only notebooks.

2. Pedagogy:  
   - Add a short text cell *before* Lab 1 explaining accuracy vs. macro-F1 and why both matter.  
   - Insert a mini-demo that shows “good” vs. “not good” vectors being similar to motivate Lab 4.

3. Execution robustness:  
   - Wrap `history` access in `if 'history' in globals()` or initialise `history = {}` at the top.  
   - In safety-nets, *overwrite* broken stubs: `if not callable(getattr(SentimentMLP,'forward',None)):` then redefine.

4. Scriptability:  
   - Replace `!pip install` with a standard `subprocess.call` guard or move installs to a README/setup.  
   - Gate Colab-only code (`from google.colab import files`) behind `if 'google.colab' in sys.modules`.

5. Disclosure hygiene in SOL:  
   - Mark every answer cell with `"jupyter":{"source_hidden":true}` or Colab fold-tags so students must expand manually.

6. Minor: clarify in Section 2 text what “early stopping” means before showing its hyper-parameters.