Notebook audited:  
- exercises/1-Pre-NLP-Tools/3-Capstone_A.ipynb (33 cells)  
- solutions/1-Pre-NLP-Tools/3-Capstone_A.ipynb (33 cells, filled answers)

----------------------------------------------------------------------  
1) Correctness / execution

- Import cell (code-2) installs `numpy<2`, hf ≥4.40 etc. OK.  
- All later cells execute in linear order **only because every “lab” is followed by a _safety-net_ cell that back-fills the objects when they are still `None`.**  
  * e.g. cell-12 sets `labeled=None`; cell-13 (safety-net) rebuilds it so later plotting (cell-14) does not crash.  
  * Same pattern for `routing_set` (cells-20/21) and `extract_entities` (cells-28/29).  
- No other undefined names or out-of-order references found. Would error only if a student half-fills a lab so the sentinel is **not** `None` but still wrong (e.g. wrong column names); the safety-net silently trusts the student output and the next cell may raise.  
⇒ Runs end-to-end, but the correctness guarantee is provided by the instructor code, not by the student.

2) Pedagogical gaps

- `pipeline("zero-shot-classification")` is first **used** in cell-19 before showing any minimal API tour; newcomers must infer arguments themselves.  
- `URL_RE`, `HANDLE_RE` regexes appear without earlier explanation of basic regex syntax.  
- Confidence gating (cell-31) jumps from histogram straight to production threshold without demonstrating precision/recall trade-off.  
- Gradio UI code is shown but Gradio has never been imported/introduced earlier in the track.

3) “Real lab” quality & leakage

- Each lab cell contains explicit inline hints that border on a walkthrough:  
  * cell-12: `# YOUR CODE … (filter inbound to rows where 'company' is not null)`  
  * cell-20: `… np.where is one clean way`  
  * cell-28 docstring lists exact logic for accepted tags (`ORG`, `MISC`, `PRODUCT`).  
  These comments leak almost the full solution; only the literal one-line call is missing.

4) Solution notebook hygiene

- The solution file is fully filled; no hiding metadata (`"collapsed":true`, `"tags":["solution"]`, etc.).  
- Exercise notebook’s safety-net cells **show the answer up front** (they compute the same objects the student is asked to build). Nothing is collapsed; running the notebook immediately reveals counts, plots and entity extractor behaviour.

5) Script-mode viability

- Because the safety-nets redefine everything, `jupyter nbconvert --to python` followed by `python` will run cleanly. Without them it would crash on `NoneType` operations.

6) Dependency pin

- Install cell correctly pins `numpy<2`.

----------------------------------------------------------------------  
Recommended fixes (priority order)

1. Remove or gate safety-net cells behind `if DEBUG: …` or move them to hidden `%load` snippets so students cannot accidentally reveal answers.  
2. Add cell metadata (`tags=["solution"]`, `collapsed":true`) in the *solution* notebook and collapse hints in the exercise file.  
3. Soften inline hints: replace code-shaped comments with higher-level guidance; keep API references in markdown, not code comments.  
4. Insert a short demo cell **before** first zero-shot pipeline call explaining inputs/outputs.  
5. Add a brief regex primer or link when introducing `URL_RE`.  
6. Show a small precision-recall table before applying the 0.5 threshold to justify the choice.