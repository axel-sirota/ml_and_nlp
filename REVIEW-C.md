# Adversarial Review - Part C (3-Transfer-Chatbot)

Scope: NB9 (9-DistilBERT_Chatbot), NB10 (10-EncoderDecoder_QA_Chatbot), NB11 (11-Capstone_C).

Six criteria checked per notebook:
1. Smoke run - does a real .venv run of the exercise (harness-shrunk) complete without error?
2. Cell-order / static trace - imports and dependencies defined before use; no undefined names.
3. Labs are real labs - genuine implementation tasks, not run-this cells.
4. Answer leaks - do starter comments, docstrings, or safety-net cells reveal the solution?
5. Solution completeness - does the sibling solution fill every `None # YOUR CODE` / `pass`?
6. Pedagogical soundness - concept-before-lab ordering, prompt/solution consistency, honest claims.

## Smoke-run results

| Notebook | Exercise | Solution | First error |
|----------|----------|----------|-------------|
| NB9  9-DistilBERT_Chatbot       | PASS | n/a    | - (clean run; harness shrank to subset 5 / 1 epoch, neutralized Gradio + pip) |
| NB10 10-EncoderDecoder_QA       | FAIL | FAIL*  | EX: cell 14 `AssertionError: preprocess returned None` (G6). SOL: MPS-only `Placeholder storage has not been allocated on MPS device!` in post-train `.generate()` |
| NB11 11-Capstone_C              | FAIL | FAIL*  | EX: cell 7 `AssertionError: Set PATH to 'A' or 'B'` on `PATH = None` (top-level gate). SOL: MPS-only `Placeholder storage...` in `trainer.train()` |

\* The SOLUTION failures for NB10 and NB11 are environment-specific Apple-Silicon artifacts, not logic bugs (see Generic finding G8). Both solutions are logically complete and would pass on Colab GPU or a CPU-only box.

## Generic findings (whole part)

- G1 (see Part A/B) - Visible, non-collapsible SAFETY-NET cells leak the verbatim lab answer (metadata == {}, no `<details>`/skip). Recurs in NB9 (cell 19 - leaks the full Lab 2 tokenize_fn + both `.map(..., batched=True)` calls), NB10 (cells 15/22/27 - cell 22 reprints the full working Seq2SeqTrainingArguments AND a full Seq2SeqTrainer), NB11 (cells 16/24 - cell 24 reprints answer_question + build_input). Fix: wrap each net in `<details>` or skip-mark.

- G2 (see Part A/B) - Lab starter comments + docstrings dictate the solution token-for-token, reducing labs to transcription. Recurs in every notebook: NB9 (cells 14, 18), NB10 (cells 14, 21, 26), NB11 (cells 11, 18/19). New flavour: in NB10/NB11 the only numeric value a lab asks the student to "derive" (T5 learning rate 3e-4) is already printed in the theory cell above the lab, so the "derivation" is pure copy. Fix: reduce hints to intent.

- G6 (see Part B) - Hard `assert ... is not None` in a provided verification block runs against the untouched `None # YOUR CODE` placeholder BEFORE the sibling safety-net fires, so the exercise CRASHES instead of degrading and the net becomes unreachable dead code. Root cause of both Part C exercise smoke FAILs. NB10: all three labs (cells 14, 21, 26) crash this way; nets 15/22/27 are dead. NB11: escalated to a NEW variant (see G9). NB9 is the lone notebook that avoids this - its Lab verifications are guarded (`if lab_enc is not None`) and run before no hard pre-net assert. Fix: move asserts AFTER the net, or strip hard asserts from placeholder cells.

- G8 (NEW) - HF Trainer / accelerate auto-dispatches the model to the Apple-Silicon MPS backend regardless of the notebook's explicit `device = "cuda" if ... else "cpu"` (which resolves to CPU on a Mac). This produces `RuntimeError: Placeholder storage has not been allocated on MPS device!` during `trainer.train()` (NB11) or post-train `.generate()` (NB10), causing both SOLUTION smoke runs to FAIL on Apple Silicon while passing on Colab GPU / pure CPU. This is a reproducibility hazard, not a logic bug. Fix: force CPU when not cuda (e.g. `TrainingArguments(use_cpu=True)` / disable MPS) or document the limitation. (NB9 sidesteps it because its harness run shrinks to 1 epoch and the failure mode happens in the longer train/generate path.)

- G9 (NEW) - Top-level selector gate defeats the documented graceful-degradation design. NB11 cell 7 is `PATH = None # YOUR CODE` followed by `assert PATH in ("A","B")`; every downstream cell - including both safety-nets - lives inside `if PATH == "A"/"B"`. On a straight exercise run the assert crashes before any branch executes, so NO safety-net ever fires and the cell-0 promise that "the chatbot finale is always reachable" is FALSE. This is a worse variant of G6: not one lab cell crashing, but the whole notebook gated at a single top-level selector. Fix: default PATH to a valid value (or make the gate a soft warning) so the degradation design actually holds.

- G4 (see Part A/B, good) - numpy<2 correctly pinned in the pip cell of all three notebooks (NB9 cell 2, NB10 cell 1, NB11 cell 4), alongside transformers==4.57.1 and datasets bounds. No action.

- G7 (NB5, NOT recurring here, good) - Part C diagrams are all inline mermaid code blocks; no `../diagrams/*.png` references exist, so the NB5 dead-link problem does not recur (even though solutions/3-Transfer-Chatbot/diagrams/ is absent, nothing points to it).

- G3 / G5 (see Part A) - script-conversion safety and solution-completeness hold across Part C; no new manifestations beyond G8 above.

## Per-notebook specific findings

### NB9 - 9-DistilBERT_Chatbot (exercise smoke PASS)
- SPECIFIC (high): Lab 1 prompt vs solution mismatch. Exercise cell 13/14 says "tokenize ... with padding, truncation, max_length=32" and defines `shortest_pad_ratio` over a denominator of 32. A prompt-faithful student uses `padding=True` (pad-to-longest, ~17 wide), so `input_ids` is NOT (5,32) and the `/32` ratio is ill-defined. The SOLUTION silently switches to `padding="max_length"` and only THEN adds a "Common mistake: using padding=True ... the pad-ratio denominator no longer matches" note. The trap is disclosed only in the answer key. Fix: tell the student to pad to max_length in the exercise, or define the ratio against the actual padded width.
- GOOD: Verification cells derive sizes dynamically (`len(train_ds)`, column-name checks), so the harness subset-5/1-epoch shrink breaks no hardcoded magic number (avoids the NB5 G3 shrink trap). No inline pre-net assert, so the NB6/NB7 G6 crash-before-net failure does not occur - this is the only Part C exercise that smokes green.
- MINOR: The "beat your MLP" headline comparison is narrative only - the MLP baseline number lives back in NB8 and is never shown in this notebook; the f1 macro claim (~0.90) is asserted in prose, not demonstrated against a baseline here.

### NB10 - 10-EncoderDecoder_QA_Chatbot (exercise smoke FAIL - G6)
- SPECIFIC (medium): device selection (cell 2) is `cuda` or `cpu` only and never MPS, triggering the G8 MPS crash on Apple Silicon in the solution's post-train `.generate()`. (Logic is correct; environment hazard.)
- SPECIFIC (minor): Stretch B (cell 31) tells students to reuse the SAME now-SQuAD-fine-tuned model with a `summarize:` prefix and expect good summaries; fine-tuning has drifted the weights so summaries may degrade. Add a caveat to load a fresh t5-small.
- MINOR: cell 7 prints `get_encoder()/get_decoder() is not None` and the narrative slightly oversells T5 as an inspectable split; harmless.

### NB11 - 11-Capstone_C (exercise smoke FAIL - G9 top-level gate)
- SPECIFIC (high): the G9 PATH gate also nullifies the cell-0 documented promise ("safety-nets exist so the chatbot finale is always reachable") - false, because the cell-7 assert crashes before any `if PATH == ...` block runs. Worst case: cell 24 sits directly under the TIER-3 "no scaffolding, no hints" Lab B3 and prints the full answer_question solution one cell below it, fully nullifying that lab's design.
- SPECIFIC (medium): G8 MPS crash in `trainer.train()` (cell 14, Path A) on Apple Silicon - solution cannot green locally on a Mac.
- SPECIFIC (medium): SQuAD truncation quality caveat. Cell 17 sets `max_input = 256` claiming "question + context fit comfortably", but SQuAD contexts routinely exceed 256 tokens; cell 19 truncates silently and can drop the gold answer span - an unexplained quality hit in a capstone that elsewhere preaches "honest evaluation".
- SPECIFIC (low): pedagogical claim/code mismatch. Lab B4 (cell 25) header promises a "before/after" comparison ("show the fine-tune helped") but the code only ever calls answer_question on the fine-tuned model - no baseline/before call, so the advertised contrast is not implemented.

## Verdict + ranked fixes

Verdict: NEEDS FIXES. NB9 is in good shape (smoke PASS, dynamic verifications) and only needs the Lab 1 padding fix plus the standard G1/G2 hygiene. NB10 and NB11 both FAIL the exercise smoke on a straight run and ship the same answer-leak + crash-before-net structure as Part B; NB11 additionally gates its entire graceful-degradation design behind a single top-level assert (G9) and breaks its own cell-0 promise.

Ranked fixes:
1. (G9, NB11) Remove or soften the cell-7 PATH gate so the safety-nets/finale are actually reachable on a straight run; default PATH to a valid branch or make the gate a warning. Highest impact - it defeats the notebook's own stated design and is the exercise smoke blocker.
2. (G6, NB10) Move the hard `assert ... is not None` blocks in cells 14/21/26 to AFTER the safety-net (or strip them from placeholder cells) so the exercise degrades instead of crashing. Unblocks NB10 exercise smoke.
3. (G1, all three) Wrap safety-net cells (NB9 c19; NB10 c15/22/27; NB11 c16/24) in `<details>` or skip-mark so the verbatim answer is not visible above/below the lab - especially NB11 c24 under the TIER-3 lab.
4. (SPECIFIC, NB9) Fix the Lab 1 padding prompt/solution mismatch: instruct `padding="max_length"` in the exercise, or define the pad ratio against the real padded width.
5. (G8, NB10+NB11) Force CPU when not cuda (`use_cpu=True` / disable MPS) so the solution smoke is reproducible on Apple Silicon, or document the Colab-only assumption.
6. (G2, all three) Cut starter comments/docstrings down to intent; stop pre-printing the 3e-4 learning rate in the theory cell directly above the lab that "derives" it.
7. (SPECIFIC, NB11) Fix the Lab B4 before/after claim (add a baseline call or relabel) and add a SQuAD >256-token truncation caveat at cell 17.
8. (SPECIFIC, NB10) Add a "load a fresh t5-small" caveat to Stretch B (cell 31).
