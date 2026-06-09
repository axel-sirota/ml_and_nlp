# Adversarial Review - Part B (2-Build-It)

Six criteria checked per notebook:

1. Smoke-run: notebook converts to a script and runs to completion in the pinned .venv (numpy<2, transformers 4.57.1) under the size-shrinking harness.
2. Runs top-to-bottom: static trace of import order, name definition, cell-order dependencies; no forward references or undefined names.
3. Labs are labs: every target is a genuine `None # YOUR CODE` implementation task, not a "run this" cell.
4. Hint leaks: starter comments / markdown do not dictate the solution token-for-token.
5. Solution completeness: the sibling solutions/ notebook resolves every placeholder and re-executes correctly.
6. Safety-nets + numpy pin: safety-net cells do not leak the verbatim answer in plain sight; numpy<2 is pinned.

## Smoke-run results

| Notebook | PASS/FAIL | First error |
|----------|-----------|-------------|
| NB4 - PyTorch_Tensors | PASS | - |
| NB5 - Word2Vec_and_Sentence_Embeddings | FAIL (solution) | `AssertionError: Step 1: emb1 should be (200, 384).` - provided Lab 2 assert hardcodes (200,384)/len==200; harness shrinks select(range(200))->range(5). Exercise also fails earlier at Lab 1 (`assert in_vocab is not None`) - that is the expected placeholder guard. |
| NB6 - MLP_Sentiment | FAIL (exercise) | `AttributeError: 'NoneType' object has no attribute 'item'` at Lab 4 training loop (cell 21) - `epoch_loss += loss.item()` where `loss=None` is unguarded and has no safety-net. Solution PASSES. |
| NB7 - MLP_Word2Vec | FAIL (exercise) | `AssertionError: Fill in steps 1 and 2.` at Lab 1 (cell 9) - inline assert fires on the placeholder before the sibling safety-net cell 10 can run. Solution PASSES. |
| NB8 - Capstone_B | PASS | - |

## Generic findings (whole part)

- G1 (recurrence, see Part A) - Visible, non-collapsible safety-net cells leak the verbatim lab answer (metadata == {}, no <details>, no skip/collapse). Recurs in NB6 (cells 12, 17), NB7 (cells 10, 15, 17, 20, 23), and NB8 (all four labs: cells 14, 22, 28, 31 - the leak now spans an entire capstone). Fix: wrap each safety-net body in <details> or set jupyter skip/collapsed metadata. NEW sub-pattern for this part: in NB6/NB7/NB8 the safety-nets are `if X is None`-guarded so they are runtime no-ops on a completed run (softer than Part A's unconditional reprints), but the answer source is still statically visible to anyone scrolling.

- G2 (recurrence, see Part A) - Lab starter comments dictate the solution token-for-token, reducing labs to transcription. Recurs in every notebook of this part (NB4 cells 23/29, NB5 cells 15/25/28, NB6 cells 11/16/21/24, NB7 cells 16/19/22, NB8 cells 13/21/30). Fix: reduce hints to intent, not mechanics. Often compounded by a demo cell printing the identical code one cell above the lab (NB6 cell 20 before Lab 4; NB8 cell 25 before Lab 3).

- G3 (recurrence, see Part A) - Script-conversion safety. NEW twist for this part: the harness shrinks loop/range sizes, which silently breaks any PROVIDED verification cell that hardcodes the full-size magic number. Manifested in NB5 (cell 25 asserts `emb1.shape == (200, 384)` / `len(preds) == 200`). Fix: derive expected shapes from the data (`(len(s1), 384)`, `len(s1)`), never hardcode the pre-shrink size. Related under-shrink note in NB7: `EPOCHS_MAX=50` is not matched by the harness regex, so a "tiny" run relies on early-stopping to stay fast.

- G4 (recurrence, good, see Part A) - numpy<2 is correctly pinned in the pip cell of all five notebooks; several also pin scipy<1.13 / gensim==4.3.3 / transformers 4.57.1 and explain the Colab restart-runtime caveat. No action.

- NEW G6 - Inline `assert ... is not None` verification placed in the SAME lab cell (or a cell BEFORE the sibling safety-net) makes the safety-net architecture self-defeating: on a straight exercise run the assert raises (or an f-string formats None) before control ever reaches the guarded safety-net, so the safety-net is unreachable dead code and the exercise crashes instead of degrading gracefully. This is the root cause of the NB6 and NB7 exercise smoke FAILs. Fix: move verification asserts AFTER the safety-net cell, or strip the hard asserts from placeholder cells and let the `if X is None` rescue fire first. (NB4/NB5/NB8 avoid this by guarding verification with `if x is not None` and never crashing on a placeholder.)

- NEW G7 - Broken diagram references. NB5 cites `../diagrams/...` which resolves out of the notebook's own folder, and 3 of 4 cited filenames do not exist under any path; the solutions/2-Build-It/diagrams/ directory does not exist at all, so every diagram link in that solution is dead. Spot-check diagram links across the part for path (`../` prefix) and filename drift, and create the missing solutions diagrams dir.

## Per-notebook specific findings

### NB4 - PyTorch_Tensors (cleanest in the part)
- GOOD: completely avoids G1 - zero solution-leaking cells in the exercise; each lab code cell is immediately followed by the next section's markdown; answers live only in solutions/. No action.
- GOOD: solutions/ is complete including both stretch tasks (Lab 4 pairwise-distance matrix, Lab 5 hand-rolled gradient descent) and the async finite-difference gradient-check homework that the exercise only describes in prose. Re-executed: w=1.998, b=1.010, autograd grad 8.75 matches 3x^2+2.
- Minor pedagogy: wrap-up (cell 30) name-drops `torch.inference_mode()` and `.detach()` and Section 2 mentions `torch.mm` / `torch.add` named ops, none of which are demonstrated anywhere - assertions without a demo. The nn.Linear "literally x @ W.T + b" claim is also never shown here (deferred to B6). Add a one-line demo or downgrade to "see B6".

### NB5 - Word2Vec_and_Sentence_Embeddings
- SPECIFIC (smoke FAIL, high): Lab 2 verification (cell 25) hardcodes `emb1.shape == (200, 384)` and `len(preds) == 200`; the harness shrinks range(200)->range(5) so the SOLUTION smoke-run fails. Derive from data. (This is the part-level G3 twist, but the brittle cell lives here.)
- SPECIFIC (G7 instance): all `[View diagram](../diagrams/word2vec-sentence-embeddings/<name>.mmd)` links are broken - the `../` escapes the notebook folder; real files live in exercises/2-Build-It/diagrams/...; solutions side has no diagrams dir at all.
- SPECIFIC: even after the path fix, 3 of 4 cited filenames are wrong - notebook cites text-to-geometry-overview / analogy-parallelogram / sbert-siamese-encoder; actual files are query-word-space / analogy-arithmetic / sbert-encode. Only one-hot-vs-dense.mmd matches.
- Pedagogy: Spearman-vs-Pearson rationale (why `.correlation` not `.pvalue`) lives only in the solution comment, not the exercise markdown. Homework tells the student to `np.save` a (500,100) feature matrix "for B7" but no worked example of the mean-pool-to-100d build is shown in this notebook.

### NB6 - MLP_Sentiment
- SPECIFIC (smoke FAIL, highest severity / G6 instance): Lab 4 (cell 21) puts `logits=None`/`loss=None` inside a LIVE training loop with no guard and no trailing safety-net, so the exercise crashes at `epoch_loss += loss.item()`. Unlike Labs 1/2/3/5 (guarded or fallback-protected), Lab 4 has neither. Fix: guard the loop body, add a Lab-4 safety-net, or make the student loop a function the verification calls only if defined.
- SPECIFIC: Lab 5 core (cell 24) `core_net, core_acc = None, None` is guarded so it does not crash, but the exercise then never trains end-to-end when unfilled - silently no-ops. Add a fallback so the downstream stretch has a trained reference.
- This notebook both over-exposes (Labs 2/3 G1 safety-nets visible) AND under-protects (Labs 4/5 have no safety-net) - opposite failure modes in one notebook.

### NB7 - MLP_Word2Vec
- SPECIFIC (smoke FAIL, high / G6 instance): inline `assert X_test_baseline is not None ...` (cell 9) raises before the sibling safety-net cell 10 runs; same fatal ordering in cells 14, 16, 19, 22 (cell 22 even formats `train_loss=None` in an f-string -> TypeError before its assert). Every safety-net is unreachable dead code on a straight exercise run. Fix per G6.
- SPECIFIC (low): cross-notebook dependency - cell 4 loads "B5 homework" X.npy/y.npy and calls it "the B5 recipe". A student starting at B7 has no such file and silently hits the rebuild path (downloads glove + SST-2). Unexplained jump for a standalone run; document or guard it.
- Pedagogy: no demo of the SentimentMLP forward pass before cell 16 asks the student to build the class (the dummy-batch check is the first time forward runs) - borderline "concept used as introduced".

### NB8 - Capstone_B
- SPECIFIC (G1, worst instance): the leak spans the ENTIRE 4-lab capstone - safety-nets at cells 14, 22, 28, 31 print the verbatim build_xy, MLPClassifier, training loop, and test-eval respectively, all always-rendered and non-collapsible. A student reading top-to-bottom sees every answer one cell after each prompt.
- SPECIFIC (contradiction): Lab 3 (cell 26) is framed "Tier 3 - no step list, no scaffolding... no hand-holding", but the Step-4 markdown (cell 23) enumerates the five-move loop verbatim and the demo (cell 25) executes one full training step one cell above. The "no scaffolding" claim is not supported by surrounding content.
- Minor: `balanced_sample` (cell 7) draws pos and neg with the same `random_state=seed`; harmless (disjoint frames) but undocumented and may confuse a careful reader.
- GOOD: solution is fully complete (including the SBERT homework) and uses plt.show()/print - no display() script-conversion bug. Smoke PASS.

## Verdict + ranked fixes

Verdict: NEEDS WORK. 3 of 5 notebooks fail the smoke-run (NB5 solution, NB6 exercise, NB7 exercise). NB4 is the model for the part (no G1, complete solution, clean trace) and NB8 is functionally clean but is the worst G1 offender. The recurring Part-A leaks (G1, G2) persist, and two genuinely new generic defects appear: self-defeating inline asserts that crash the exercise (G6) and harness-shrink-incompatible hardcoded asserts (G3 twist) plus broken diagram links (G7).

Ranked fixes (highest impact first):

1. Fix the exercise smoke FAILs (G6): in NB6 guard/relocate the Lab 4 loop placeholder; in NB7 move all inline `assert ... is not None` after their sibling safety-net cells (10/15/17/20/23). These are blocking - the exercises cannot run top-to-bottom as shipped.
2. Fix the NB5 solution smoke FAIL (G3 twist): replace the hardcoded `(200, 384)` / `len==200` asserts in cell 25 with data-derived shapes; audit all provided verification cells in the part for other pre-shrink magic numbers.
3. Add missing safety-nets in NB6 (Labs 4/5) so unfilled labs degrade gracefully instead of crashing/no-op.
4. Wrap all G1 safety-nets in <details> / skip-mark metadata: NB6 (12,17), NB7 (10,15,17,20,23), NB8 (14,22,28,31). Highest cell count is NB8 (whole capstone).
5. Reduce G2 hints to intent across all five notebooks; in NB6/NB8 also stop printing the exact loop in the demo one cell above the lab that asks for it.
6. Fix NB5 diagram links (G7): drop the `../` prefix, correct the 3 wrong filenames, and create solutions/2-Build-It/diagrams/.
7. Minor pedagogy: NB4 demo the name-dropped ops or downgrade to "see B6"; NB5 surface the Spearman rationale and a mean-pool worked example; NB7 document the B5 X.npy/y.npy dependency; NB8 reconcile the Lab 3 "no scaffolding" claim and document the balanced_sample seed.
