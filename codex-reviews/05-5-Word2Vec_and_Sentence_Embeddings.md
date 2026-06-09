exercises/2-Build-It/5-Word2Vec_and_Sentence_Embeddings.ipynb  
solutions/2-Build-It/5-Word2Vec_and_Sentence_Embeddings.ipynb  

================ audit ================

1. Runtime soundness  
- Cell 3 imports `torch`/`sentence_transformers`; install pin (cell 2) guarantees the right versions, so import errors unlikely.  
- All later cells reference only names defined earlier in a strictly increasing order (e.g. `pairs` set in cell 20, used in cell 23). No dangling variables found.  
- Only red flag: every `!pip …` (cell 2) and `%matplotlib inline`-style magic (none here now, good) would crash in a non-IPython runner.  
- Lab assert cells (16, 32, 37) depend on variables created inside the student cells; if the student skips a task the assert itself raises, but that is intended.

2. Pedagogical gaps  
- Cosine similarity is used in cell 8 (“Tiny demo”) before the notebook ever *defines* or reminds students of the formula—could add a 1-line markdown recap.  
- The mean-pool “failure modes” discussion (cell 20 markdown inside code comments) refers to “polysemy” and “length bias” without a prior definition/example; learners may be lost.  
- PCA is run in cell 17 without first showing why 2-D projection is useful or the limitations (the markdown comment inside the code mentions them but no earlier narrative).  
- Lab 2 abruptly imports `datasets` & `scipy.stats` (cell 32) before the library install cell lists them; students running section-by-section might hit missing-package errors unless they happened to execute cell 2.

3. Are the labs real & leak-free?  
Real: three genuine TODO blocks (cells 15, 32, 37) marked `# YOUR CODE`.  
No leakage: the exercise notebook never shows the finished code.  
No misleading hints; the comments are instructional, not answer-revealing.

4. Solution notebook hygiene  
- The solution notebook contains fully filled answers (as expected).  
- NONE of its answer cells are tagged `solution`, `hide-input`, or `collapsed`; opening the file shows the answers immediately (e.g., cell 15 “Lab 1 … SOLUTION”). This defeats the “only reveal if stuck” intent.  
- Verification/safety-net cells are fine (identical to exercise), but again they are not hidden.

5. Scriptability  
Running `jupyter nbconvert --to script` then `python …py` would fail:  
  a. `!pip install …` (cell 2) is illegal syntax in Python.  
  b. The matplotlib plotting cells rely on an implicit GUI backend; as a script they silently hang or save nothing.  
Hence **not** script-safe.

6. numpy pin present?  
Yes: cell 2 installs `"numpy<2"` explicitly in both notebooks.

=============== fixes (priority order) ===============

1. Tag every answer cell in the **solution** notebook with `{"tags":["solution"], "jupyter":{"source_hidden":true}}` or at least `collapsed:true` so students do not see solutions by default.  
2. Replace `!pip …` with a standard Python fallback (`subprocess.call([...])` guarded by `if __name__ == "__main__":`) or move the install to the Colab header note so the exported script runs.  
3. Add a short markdown cell before cell 8 that restates the cosine-similarity formula and why it matters for vectors.  
4. Insert a concrete example (two near-duplicate sentences vs. two dissimilar ones) *before* the mean-pool baseline to illustrate “polysemy” and “length bias”.  
5. Pre-install `datasets`, `scipy` in the same pip cell or import-guard Lab 2 with a try/except + install to avoid section-order package errors.  
6. Wrap `plt.show()` calls with `if __name__ == "__main__":` or `plt.savefig()` so a script run terminates cleanly.