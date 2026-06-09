Notebook audited:  
- student version `exercises/3-Transfer-Chatbot/9-DistilBERT_Chatbot.ipynb`  
- instructor solution `solutions/3-Transfer-Chatbot/9-DistilBERT_Chatbot.ipynb`  

================================================================  
1. Correctness / run-ability (top-to-bottom)  
- Code cell 3 (`!pip install … numpy<2`) is a shell magic; fine inside Colab but will raise a `SyntaxError` if the file is run as a Python script.  
- Code cell 5 defines `device`, seeds RNGs, imports etc. OK.  
- Hyper-parameter cell 7 defines `MODEL_NAME` etc. before later use. OK.  
- Data-loading cell 11 correctly renames `"label"`→`"labels"` – prevents the silent-no-loss bug.  
- LAB 1 cell 13 can be run even when left blank because the verification block is guarded by `if lab_enc is not None`; no crash.  
- LAB 2 cell 18 is critical: `tokenize_fn` is `pass` and `train_ds`/`eval_ds` are **not** tokenised. When cell 23 starts `Trainer(train_dataset=train_ds, …)` it will raise a `KeyError: 'input_ids'` (missing fields) or a shape error.  
- All subsequent evaluation / Gradio cells depend on a trained model, so they cascade-fail after the above.  
Solution notebook runs start-to-finish – no undefined names or ordering problems.

2. Pedagogical gaps  
- `datasets.map(batched=True)` is used in LAB 2 before students have ever seen an example of `map`; there is only a paragraph saying *“There is no None # YOUR CODE scaffold inside the function body”* but no working demo.  
- `DataCollatorWithPadding` is referenced but never demonstrated; students must infer why padding isn’t done in `tokenize_fn`.  
- The final Gradio block is shown after the `Trainer` but the notebook never demonstrates generating a prediction outside the trainer loop before wrapping it – a small logical jump.

3. Lab quality / leakage  
Labs are genuine implementation tasks:  
- LAB 1 asks for token-count statistics.  
- LAB 2 wires tokenisation.  
No finished code is present; only high-level hints such as *“Hint: pass the list to the tokenizer…”*. No answer leakage detected.

4. Solution notebook privacy  
- All LAB cells are fully filled in (e.g., LAB 1 solution cell shows the exact sentences and tensor ops).  
- None of those cells are marked `{"collapsed": true}` or `{"jupyter": {"source_hidden": true}}`; the answers are visible immediately on open.  
=> Safety-net not hidden; students will see the answer first.

5. Script compatibility  
Both notebooks rely on `!pip`, IPython magics and interactive `gradio` launching; running `jupyter nbconvert --to python && python …` will crash on the first `!pip` line. Not script-safe.

6. NumPy pin  
Install cell explicitly includes `numpy<2` – ✔ present.

================================================================  
Recommended fixes (priority order)

1. Add a guard cell *before* the training block that checks `{'input_ids','attention_mask'}⊂train_ds.column_names` and raises a friendly message like “⚠️ Run LAB 2 first”.  
2. Provide a short explicit demo of `datasets.map()` **before** LAB 2 (e.g., map a dummy lambda that upper-cases a column) so students have seen the pattern.  
3. Mark every filled answer in the SOLUTION notebook with `\"jupyter\": {\"source_hidden\": true}` or `collapsed`: true` so the code is folded by default.  
4. Replace `!pip …` with a Python-safe helper (e.g., `import subprocess, sys; subprocess.check_call([sys.executable,'-m','pip', …])`) or add “This cell uses shell magics, skip when running as .py”.  
5. Insert a minimal forward-pass example (**before** Gradio) to show how to get predictions outside the Trainer.  
6. Briefly explain `DataCollatorWithPadding` right after it is instantiated (one-line comment).