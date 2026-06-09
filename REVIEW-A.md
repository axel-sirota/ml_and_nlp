# Adversarial Review - Part A (1-Pre-NLP-Tools)

Six criteria checked per notebook:

1. Smoke-run: does the converted script execute end-to-end on the pinned .venv?
2. Static trace: import order, cell order, undefined names, forward refs.
3. Labs-are-labs: are the labs genuine implementation tasks (not run-this cells)?
4. Hint/solution leaks: do starter comments or markdown dictate the answer?
5. Solution completeness: are all `None # YOUR CODE` placeholders resolved in solutions/?
6. Safety-nets, pedagogy, and the numpy<2 pin.

## Smoke-run results

| Notebook | Result | First error |
|----------|--------|-------------|
| NB1 1-NLP_From_Scratch | FAIL | OSError [E050] Can't find model 'en_core_web_sm' at cell 7 (`nlp = spacy.load("en_core_web_sm")`). Model not provisioned; smoke_run.py strips the `!python -m spacy download` line and venv has no pip. |
| NB2 2-Pipeline_Tour | FAIL | ValueError (CVE-2025-32434) at Task 5 summarization cell `1446c200`. `sshleifer/distilbart-cnn-12-6` ships only a `.bin`; torch 2.2.2 < 2.6 + transformers 4.57.1 refuse the load. Tasks 1-4 ran fine. |
| NB3 3-Capstone_A | FAIL | NameError: name 'display' is not defined at cell 5 (`display(twcs.head(3))`). `display` is a Jupyter builtin never imported; dies before any model loads. Same bug present in the solution. |

All three FAIL. In every case the static trace of the SOURCE (assuming a live Jupyter kernel and the model present) is otherwise clean: imports precede uses, no undefined names beyond the noted ones, no forward refs, cell order linear.

## Generic findings (whole part)

These recur across multiple NBs in Part A. Stated once.

- **G1 - Visible, non-collapsible safety-net cells leak the lab answer.** The course pattern puts a plain, always-rendered "SAFETY-NET (provided, do not edit)" code cell directly below the lab, reproducing the verbatim solution, with no `<details>` wrapper and no collapse/skip metadata (verified `metadata == {}`). A student scrolling normally hits the answer before attempting. Present in NB1 (cell 14, the only safety-net it has) and ALL FOUR labs of NB3 (cells 11, 16, 22, 26 - cell 26 reprints the entire Lab 4 solution). Fix: wrap safety-nets in `<details>` or a clearly skip-marked cell.

- **G2 - Lab starter comments + lab markdown dictate the solution token-for-token.** Across NB1, NB2, NB3 the inline `# YOUR CODE` comments name the exact call, kwargs, and indexing, reducing labs to transcription. Examples: NB3 `# YOUR CODE (call clf on routing_set['clean'].tolist() ... batch_size=16)`, `np.where is one clean way`; NB2 `router(text, candidate_labels=my_labels)`, `result["labels"][0]`, the literal model string; NB1 names `clean_text`, `nlp`, `lemma_`, `batch_size=32`, `disable=[parser,tagger,lemmatizer]`, `most_common(10)`. Fix: reduce hints to intent, not the literal code.

- **G3 - Script-conversion safety.** The smoke harness converts notebooks to bare scripts, so any Jupyter-only builtin or stripped install line hard-fails. Two distinct manifestations in Part A: (a) un-imported `display()` (NB3); (b) install lines stripped by smoke_run.py - the `spacy download` (NB1) and the implicit need for torch>=2.6 / safetensors models (NB2). Harness must pre-provision spaCy models and a torch>=2.6 (or safetensors-only models) for spaCy/HF notebooks to smoke-pass; notebooks must use `from IPython.display import display` or `print()`.

- **G4 - numpy<2 is correctly pinned** in the pip cell of all three notebooks. No action.

- **G5 (good) - Labs are genuine and solutions are complete** in all three. Every lab leaves real logic blank, and the sibling `solutions/1-Pre-NLP-Tools/*.ipynb` resolves every placeholder (often with teaching / "Common mistake" comments). The only solution defect is NB3's solution carrying the same `display()` bug.

## Per-notebook specific findings

### NB1 - 1-NLP_From_Scratch
- Only blocker is environmental (en_core_web_sm). Section 1 (cell 7) is the named breaking section; every cell from 7 onward depends on `nlp`.
- TF-IDF (cell 23, `TfidfVectorizer(min_df=3, max_df=0.5)`) is used to NAME clusters but never explained, in a notebook that otherwise teaches tokenization/lemmatization from first principles. `min_df=3` also risks an empty vocabulary on the 5-doc smoke subset (path not reached due to the model failure).
- Cell 8 markdown claims irregular lemmatization (`better -> well`, `mice -> mouse`) that the demo (cell 9) never shows - only regular `-s`/`-ed` cases (shops->shop, visited->visit) are demonstrated. The surprising claim is unverified.
- Inconsistent safety-net coverage: Lab 2.1 has one (and it leaks, see G1); Labs 3.1 and 4.1 have NONE. If a student skips them, the assert-guarded cells just print `None` and nothing rebuilds `top_people`/`crosstab`.
- Minor: Section 4 silently downloads a second model (all-MiniLM-L6-v2, ~22MB) with no prior mention; surprise offline.

### NB2 - 2-Pipeline_Tour
- BLOCKER detail: summarization model `sshleifer/distilbart-cnn-12-6` is `.bin`-only. Fix options: bump torch>=2.6, or switch to a safetensors model (`facebook/bart-large-cnn`). Tasks 1-4 (sentiment, zero-shot, NER, QA) all ran correctly; Task 6 fill-mask and everything after never execute as a script.
- Contradiction: intro (cell 626782b8) promises "every cell runs on CPU" and Section 0 (cell 9d69866e) says the 4.x pin "works cleanly" - directly falsified by the summarization failure on the course's own .venv.
- Unverified claims: wrap-up table (cell 973b3697) lists an NER default model the notebook never prints; summarization markdown (cell 34232329) promises a length-warning the provided long input never triggers; the NEUTRAL-class claim (Lab 2) is asserted but the run never reaches it.
- No in-notebook safety-nets at all and no `<details>` anywhere; the only answers live in the sibling solution file, shown up front. (Note: this is the opposite failure mode from G1 - here the absence, not the leak.)

### NB3 - 3-Capstone_A
- Smoke FAIL is the `display(twcs.head(3))` builtin (cell 5), present in BOTH exercise and solution. Fix: `from IPython.display import display` or `print()`.
- Worst instance of G1: all four safety-nets (cells 11, 16, 22, 26) are visible plain cells; cell 26 reprints the entire `extract_entities` Lab 4 solution verbatim one cell below the lab.
- Unsupported numbers: wrap-up (cell 32) asserts "zero-shot tops out 70-85%" vs "fine-tuned 95%+", but the notebook computes NO accuracy - the weak labels are company names, never compared against the 5 routing categories.
- Section 6 is titled "Evaluate and Add a Safety Net" and cell 3 imports sklearn "for an optional metrics report," but no metric is ever produced - only a confidence histogram and a magic `CONF_THRESHOLD = 0.5` (cell 29) with no justification or grounding in the plotted distribution.
- Pedagogical leap: Lab 4 asks the student to filter NER `entity_group` into an ORG/MISC/PRODUCT family, but the NER demo (cell 24) prints raw output for only ONE sentence and never shows the full label set the model emits (ORG/MISC/PER/LOC).

## Verdict + ranked fixes

Verdict: **Part A FAILS smoke on all three notebooks.** Two failures are environmental/harness (NB1 spaCy model, NB2 torch/safetensors) and one is a real notebook bug (NB3 `display`). Beyond smoke, the dominant pedagogical defect is the leaking safety-net pattern (G1) and over-specified hints (G2), which together reduce most labs to transcription. Labs and solutions themselves are well-built and complete (G5).

Ranked fixes:

1. **Unblock smoke (harness + 1 code fix).** Pre-provision `en_core_web_sm` and a torch>=2.6 (or safetensors-only HF models) in the pinned .venv / smoke_run.py; OR pin notebooks to safetensors models. Fix NB3 `display` -> `from IPython.display import display` (exercise + solution).
2. **Hide all safety-nets (G1).** Wrap every "SAFETY-NET" cell in `<details>` or a skip-marked cell. Highest-leak first: NB3 cell 26, NB1 cell 14, NB3 cells 11/16/22.
3. **Add missing safety-nets / make coverage consistent.** NB1 Labs 3.1 & 4.1 and NB2's two labs have none; either all labs get a hidden net or none do.
4. **De-leak hints (G2).** Rewrite `# YOUR CODE` comments to describe intent, not the literal call/kwargs, across all three NBs.
5. **Fix unverified/contradicted claims.** NB2 "runs on CPU"/"works cleanly" (fix model or claim); NB3 70-85% vs 95% accuracy (compute a real metric or drop the figures); NB1 `better->well`/`mice->mouse` (demo it or remove).
6. **Explain or remove unexplained machinery.** NB1 TF-IDF (and `min_df=3` empty-vocab risk on small subsets); NB3 `CONF_THRESHOLD=0.5` justification + the promised sklearn metrics report; NB3 Lab 4 NER label-set demo.
